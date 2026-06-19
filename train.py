import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
import shutil
import matplotlib.pyplot as plt
import matplotlib
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
matplotlib.use('Agg')

import albumentations as A
from albumentations.pytorch import ToTensorV2
import archs_base
import archs
import losses
from dataset import Dataset
from metrics import iou_score, indicators
from utils import AverageMeter, str2bool

ARCH_NAMES = ['UKAN','UKAN_base','UKAN_base_mc']
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', default=42, type=int,
                        help='')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UKAN')

    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='busi', help='dataset name')
    parser.add_argument('--data_dir', default='inputs', help='dataset dir')

    parser.add_argument('--output_dir', default='outputs', help='ouput dir')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=12, type=int)

    parser.add_argument('--no_kan', action='store_true')

    # Visualization control
    parser.add_argument('--vis_interval', default=50, type=int,
                        help='visualize every N epochs (default: 50)')

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou, dice, _ = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion, epoch=0):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.eval()

    vis_dir = os.path.join(config['output_dir'], config['name'], 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # Save visualizations every N epochs or at the final epoch.
    should_visualize = (epoch % config['vis_interval'] == 0) or (epoch == config['epochs'] - 1)

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for i, (input, target, _) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
                output = outputs[-1]
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            # Visualization: save the first three samples only at selected epochs.
            if should_visualize and i < 3:
                img_vis = input[0].cpu()
                mask_vis = target[0].cpu()
                pred_vis = torch.sigmoid(output[0]).cpu()

                # Prepare image for display by de-normalizing it.
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_display = img_vis * std + mean
                img_display = img_display.permute(1, 2, 0).numpy()
                img_display = np.clip(img_display, 0, 1)

                mask_display = mask_vis.squeeze().numpy()
                pred_display = pred_vis.squeeze().numpy()

                # Create a 3-column layout: image | GT | prediction.
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(img_display)
                axes[0].set_title('Input Image', fontsize=12, fontweight='bold')
                axes[0].axis('off')

                axes[1].imshow(mask_display, cmap='gray')
                axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
                axes[1].axis('off')

                axes[2].imshow(pred_display, cmap='gray')
                axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
                axes[2].axis('off')

                plt.tight_layout()
                save_path = os.path.join(vis_dir, f'epoch_{epoch}_sample_{i}_vis.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    result = OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg)
    ])

    return result


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    config = vars(parse_args())

    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])

    exp_name = config['name']

    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)
    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # Define the loss function.
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = False

    # create model
    if config['arch'] == 'UKAN':
        model = archs.UKAN(config['num_classes'], 
                           config['input_channels'], 
                           config['deep_supervision'],
                           embed_dims=config['input_list'], 
                           no_kan=config['no_kan'])

    elif config['arch'] == 'UKAN_base':
        model = archs_base.UKAN_base(config['num_classes'], 
                                     config['input_channels'], 
                                     config['deep_supervision'],
                                     embed_dims=config['input_list'], 
                                     no_kan=config['no_kan'])
    elif config['arch'] == 'UKAN_base_mc':
        model = archs_base.UKAN_base_mc(config['num_classes'], 
                                     config['input_channels'], 
                                     config['deep_supervision'],
                                     embed_dims=config['input_list'], 
                                     no_kan=config['no_kan'])
    else:
        raise NotImplementedError(f"Architecture {config['arch']} not supported")
        
    model = model.cuda()

    # Use parameter groups to assign a separate learning rate to KAN layers.
    param_groups = []
    for name, param in model.named_parameters():
        if 'layer' in name.lower() and 'fc' in name.lower():
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']})
        else:
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})

    # Optimizer
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'],
                              weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    # Scheduler
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Back up source files.
    try:
        shutil.copy2('train.py', f'{output_dir}/{exp_name}/')
        shutil.copy2('archs.py', f'{output_dir}/{exp_name}/')
    except Exception as e:
        print(f"Backup warning: {e}")

    dataset_name = config['dataset']
    img_ext = '.png'
    mask_ext = '_mask.png' if dataset_name == 'busi' else '.png'

    # Data loading code
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))

    if len(img_ids) == 0:
        print(f"\n❌ Error: No images found in {os.path.join(config['data_dir'], config['dataset'], 'images')}")
        print("Please check your inputs folder structure.")
        return

    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

    train_transform = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
        ToTensorV2(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    best_dice = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        train_log = train(config, train_loader, model, criterion, optimizer)
        val_log = validate(config, val_loader, model, criterion, epoch=epoch)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)
        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
        my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print("=> saved best model")
            print('IoU: %.4f' % best_iou)
            print('Dice: %.4f' % best_dice)
            trigger = 0

        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
