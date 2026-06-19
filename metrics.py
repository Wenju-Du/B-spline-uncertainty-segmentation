import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision


def iou_score(output, target, compute_hd95=False):
    """
    Compute IoU and Dice; supports multi-class masks by averaging over channels.
    """
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5

    # ---------- Multi-class: compute per channel and average ----------
    if output_.ndim >= 3 and output_.shape[-3] > 1:
        num_classes = output_.shape[-3]
        ious, dices, hd95s = [], [], []

        for c in range(num_classes):
            if output_.ndim == 4:
                o_c = output_[:, c]
                t_c = target_[:, c]
            else:
                o_c = output_[c]
                t_c = target_[c]

            # Skip classes absent from this batch.
            if t_c.sum() == 0 and o_c.sum() == 0:
                continue

            intersection = (o_c & t_c).sum()
            union = (o_c | t_c).sum()

            iou_c = (intersection + smooth) / (union + smooth)
            dice_c = (2 * intersection + smooth) / (o_c.sum() + t_c.sum() + smooth)
            ious.append(iou_c)
            dices.append(dice_c)

            if compute_hd95:
                try:
                    hd95s.append(hd95(o_c, t_c))
                except:
                    hd95s.append(0)

        iou = np.mean(ious) if ious else 0.0
        dice_val = np.mean(dices) if dices else 0.0
        hd95_val = np.mean(hd95s) if hd95s else 0.0

        return iou, dice_val, hd95_val

    # ---------- Single-class: original logic ----------
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice_val = (2 * iou) / (iou + 1)

    hd95_val = 0
    if compute_hd95:
        try:
            hd95_val = hd95(output_, target_)
        except:
            hd95_val = 0

    return iou, dice_val, hd95_val


def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    if output_.ndim >= 3 and output_.shape[-3] > 1:
        num_classes = output_.shape[-3]
        dices = []
        for c in range(num_classes):
            if output_.ndim == 4:
                o_c = output_[:, c]
                t_c = target_[:, c]
            else:
                o_c = output_[c]
                t_c = target_[c]
            if t_c.sum() == 0 and o_c.sum() == 0:
                continue
            intersection = (o_c & t_c).sum()
            dices.append((2. * intersection + smooth) / (o_c.sum() + t_c.sum() + smooth))
        return np.mean(dices) if dices else 0.0

    output_flat = output.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (output_flat * target_flat).sum()
    return (2. * intersection + smooth) / (output_flat.sum() + target_flat.sum() + smooth)


def indicators(output, target):
    """
    Full evaluation metrics; supports multi-class masks by averaging over channels.
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    output_ = output > 0.5
    target_ = target > 0.5

    if output_.ndim >= 3 and output_.shape[-3] > 1:
        num_classes = output_.shape[-3]
        all_metrics = {k: [] for k in ['iou', 'dice', 'hd', 'hd95', 'recall', 'spec', 'prec']}

        for c in range(num_classes):
            if output_.ndim == 4:
                o_c = output_[:, c]
                t_c = target_[:, c]
            else:
                o_c = output_[c]
                t_c = target_[c]

            if t_c.sum() == 0 and o_c.sum() == 0:
                continue

            try:
                all_metrics['iou'].append(jc(o_c, t_c))
                all_metrics['dice'].append(dc(o_c, t_c))
                all_metrics['hd'].append(hd(o_c, t_c))
                all_metrics['hd95'].append(hd95(o_c, t_c))
                all_metrics['recall'].append(recall(o_c, t_c))
                all_metrics['spec'].append(specificity(o_c, t_c))
                all_metrics['prec'].append(precision(o_c, t_c))
            except Exception:
                continue

        mean = lambda lst: np.mean(lst) if lst else 0.0
        return (mean(all_metrics['iou']), mean(all_metrics['dice']),
                mean(all_metrics['hd']), mean(all_metrics['hd95']),
                mean(all_metrics['recall']), mean(all_metrics['spec']),
                mean(all_metrics['prec']))

    # Single-class
    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    hd_ = hd(output_, target_)
    hd95_ = hd95(output_, target_)
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)
    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_


def compute_ece(output, target, n_bins=10):
    if not torch.is_tensor(output):
        output = torch.tensor(output)
    if not torch.is_tensor(target):
        target = torch.tensor(target)

    preds = torch.sigmoid(output)
    preds = preds.view(-1)
    labels = target.view(-1).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=preds.device)
    ece = torch.tensor(0.0, device=preds.device)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (preds > bin_lower) & (preds <= bin_upper)
        else:
            in_bin = (preds > bin_lower) & (preds < bin_upper)

        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = preds[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()
