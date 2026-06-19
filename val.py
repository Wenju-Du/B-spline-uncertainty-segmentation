#!/usr/bin/env python
import argparse
import os
from glob import glob
import random
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import OrderedDict

import archs
import archs_base
from dataset import Dataset
from metrics import iou_score, compute_ece
from utils import AverageMeter
from albumentations import RandomRotate90, Resize
import time

from PIL import Image

# EDL-related imports
from archs_edl import UKAN_edl
from losses import edl_inference

try:
    from UNet_UKH import UNet_UKH
except ImportError:
    UNet_UKH = None
try:
    from UNet import UNet as UNet_base
except ImportError:
    UNet_base = None
try:
    from UNet_edl import UNet_edl
except ImportError:
    UNet_edl = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='outputs', help='ouput dir')
    args = parser.parse_args()
    return args


def seed_torch(seed=1187):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main():
    args = parse_args()

    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    seed_num = config.get('dataseed', 1187)
    seed_torch(seed_num)

    # Detect whether the model is an EDL variant.
    is_edl = config.get('edl', False) or config.get('arch', '').endswith('_edl')

    print('-' * 20)
    print(f"Model: {config['name']} {'[EDL]' if is_edl else '[Base]'}")
    print('-' * 20)

    cudnn.benchmark = False

    # ============================================================
    # Model initialization with EDL branches.
    # ============================================================
    arch_name = config['arch']

    if arch_name == 'UKAN_edl':
        # EDL branch
        model = UKAN_edl(
            num_classes=2,
            input_channels=config['input_channels'],
            embed_dims=config['input_list'],
            no_kan=config.get('no_kan', False),
        )
    elif arch_name == 'UNet_edl' and UNet_edl is not None:
        # EDL branch
        model = UNet_edl(
            n_channels=config['input_channels'],
            n_classes=2,
            base_c=config.get('base_c', 64),
        )
    elif arch_name == 'UNet_UKH' and UNet_UKH is not None:
        model = UNet_UKH(
            n_channels=config['input_channels'],
            n_classes=config['num_classes'],
            base_c=config.get('base_c', 64),
        )
    elif arch_name == 'UNet' and UNet_base is not None:
        model = UNet_base(
            n_channels=config['input_channels'],
            n_classes=config['num_classes'],
            base_c=config.get('base_c', 64),
        )
    elif arch_name == 'UKAN_base':
        model = archs_base.UKAN_base(
            config['num_classes'],
            config['input_channels'],
            config['deep_supervision'],
            embed_dims=config['input_list'],
            no_kan=config['no_kan']
        )
    else:
        model = archs.__dict__[arch_name](
            config['num_classes'], config['input_channels'],
            config['deep_supervision'], embed_dims=config['input_list']
        )
    model = model.cuda()

    dataset_name = config['dataset']
    img_ext = '.png'
    mask_ext = '.png'
    if dataset_name == 'busi':
        mask_ext = '_mask.png'

    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=seed_num)

    ckpt_path = f'{args.output_dir}/{args.name}/model.pth'
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"=> Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    print("=> Model loaded successfully.")
    model.eval()

    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
        ToTensorV2(),
    ])

    # EDL models still use single-channel binary masks in the dataset.
    ds_num_classes = 1 if is_edl else config['num_classes']

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=ds_num_classes,
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()
    ece_avg_meter = AverageMeter()

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input)

            if is_edl:
                # EDL branch: convert two-channel logits to single-channel logits.
                # Keep iou_score and compute_ece consistent with base models.
                result = edl_inference(output)
                prob = result['prob']  # (B, 1, H, W), foreground probability [0,1]

                # Convert probabilities back to equivalent single-channel logits: logit = log(p / (1-p)).
                # Then sigmoid(logit) inside iou_score equals prob.
                eps = 1e-6
                prob_clamped = prob.clamp(eps, 1.0 - eps)
                equiv_logits = torch.log(prob_clamped / (1.0 - prob_clamped))

                iou, dice, hd95_ = iou_score(equiv_logits, target)
                batch_ece = compute_ece(equiv_logits, target)

                # Threshold probabilities directly when saving predictions.
                output_save = prob.cpu().numpy()
            else:
                # Base branch: keep the original logic.
                iou, dice, hd95_ = iou_score(output, target)
                batch_ece = compute_ece(output, target)

                output_save = torch.sigmoid(output).cpu().numpy()

            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            hd95_avg_meter.update(hd95_, input.size(0))
            ece_avg_meter.update(batch_ece, input.size(0))

            # Save image.
            output_save[output_save >= 0.5] = 1
            output_save[output_save < 0.5] = 0

            os.makedirs(os.path.join(args.output_dir, config['name'], 'out_val'), exist_ok=True)
            for pred, img_id in zip(output_save, meta['img_id']):
                pred_np = pred[0].astype(np.uint8) * 255
                img = Image.fromarray(pred_np, 'L')
                img.save(os.path.join(args.output_dir, config['name'], 'out_val/{}.jpg'.format(img_id)))

    print('=' * 40)
    print(f"Model: {config['name']} {'[EDL]' if is_edl else '[Base]'}")
    print('IoU:   %.4f' % iou_avg_meter.avg)
    print('Dice:  %.4f' % dice_avg_meter.avg)
    print('HD95:  %.4f' % hd95_avg_meter.avg)
    print('ECE:   %.4f' % ece_avg_meter.avg)
    print('=' * 40)


if __name__ == '__main__':
    main()


