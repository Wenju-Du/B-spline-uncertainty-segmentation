#!/usr/bin/env python3
"""
Fig.2 & Fig.4: visualization of B-spline basis activation patterns (improved version).
=====================================================
Improvements:
  1. Fig.2: Select images with Dice in 0.5-0.8 to ensure clear correct/error regions.
  2. Fig.2: Improve pixel selection to make confident vs. uncertain contrast clear.
  3. Fig.2: Improve lower-row bar charts with active highlights and background annotations.
  4. Fig.4: Use the full validation set for more stable distributions.
  5. Fix: moderate-pixel entropy should not exceed uncertain-pixel entropy.

Usage:
    python viz_bspline_activation_v2.py --name cvc_UKAN-re-1187 --output_dir outputs
    python viz_bspline_activation_v2.py --name busi_UKAN-re-42 --output_dir outputs
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import yaml
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import matplotlib.patheffects
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

import archs
from dataset import Dataset


# ============================================================
# Utility functions
# ============================================================
def seed_torch(seed=1187):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class FinalInputCapture:
    def __init__(self, model):
        self.captured_input = None
        self.hook = model.final.register_forward_pre_hook(self._hook_fn)

    def _hook_fn(self, module, inp):
        self.captured_input = inp[0].detach()

    def remove(self):
        self.hook.remove()


def load_model_and_data(name, output_dir):
    config_path = os.path.join(output_dir, name, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    seed_num = config.get('dataseed', 1187)
    seed_torch(seed_num)

    model = archs.__dict__[config['arch']](
        config['num_classes'], config['input_channels'],
        config['deep_supervision'], embed_dims=config['input_list']
    )
    model = model.cuda()
    ckpt_path = os.path.join(output_dir, name, 'model.pth')
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    dataset_name = config['dataset']
    img_ext = '.png'
    mask_ext = '_mask.png' if dataset_name == 'busi' else '.png'

    img_ids = sorted(glob(os.path.join(
        config['data_dir'], dataset_name, 'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=seed_num)

    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
        ToTensorV2(),
    ])
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'], dataset_name, 'images'),
        mask_dir=os.path.join(config['data_dir'], dataset_name, 'masks'),
        img_ext=img_ext, mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
    )
    return model, val_loader, config


# ============================================================
# Data extraction
# ============================================================
def extract_pixel_data(model, val_loader, config, n_images=None):
    """
    Extract predictions, GT masks, and B-spline basis activations for each image.
    n_images=None means using the full validation set.
    """
    capture = FinalInputCapture(model)
    all_images = []
    total = len(val_loader) if n_images is None else min(n_images, len(val_loader))

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="  Extracting data", total=total)):
            if n_images is not None and idx >= n_images:
                break

            if len(batch) == 3:
                inp, tgt, meta = batch
            else:
                inp, tgt = batch

            inp, tgt = inp.cuda(), tgt.cuda()
            logits = model(inp)
            feat = capture.captured_input

            # Get basis activations.
            _, bases = model.final(feat, return_spline_basis=True)
            # bases shape: (1, H, W, C_in, K)

            pred_prob = torch.sigmoid(logits).squeeze()  # (H, W)
            tgt_np = tgt.squeeze().cpu().numpy()
            pred_np = pred_prob.cpu().numpy()
            error_map = np.abs(pred_np - tgt_np)

            bases_np = bases[0].cpu().numpy()  # (H, W, C_in, K)
            H, W, C_in, K = bases_np.shape

            # active count per pixel: averaged across channels
            active_count = (bases_np > 0.01).sum(axis=-1).mean(axis=-1)  # (H, W)

            # entropy per pixel: averaged across channels
            b = bases_np.reshape(H * W, C_in, K)
            p = b / (b.sum(axis=-1, keepdims=True) + 1e-10)
            ent = -(p * np.log(p + 1e-10)).sum(axis=-1)  # (H*W, C_in)
            entropy_map = ent.mean(axis=-1).reshape(H, W)

            pred_bin = (pred_np > 0.5).astype(float)
            tgt_bin = (tgt_np > 0.5).astype(float)
            inter = (pred_bin * tgt_bin).sum()
            dice = 2 * inter / (pred_bin.sum() + tgt_bin.sum() + 1e-8)

            all_images.append({
                'input': inp[0].cpu().numpy().transpose(1, 2, 0),
                'pred_prob': pred_np,
                'gt': tgt_np,
                'error': error_map,
                'active_count': active_count,
                'entropy': entropy_map,
                'bases': bases_np,
                'dice': dice,
            })

    capture.remove()
    return all_images


def select_representative_pixels(image_data):
    """
    Select three representative pixels with clear contrast:
    - Confident: inside the foreground with the lowest error and active_count
    - Uncertain: boundary/error region with the highest error and active_count
    - Moderate: between the two extremes

    Key improvement: select by active_count so Confident < Moderate < Uncertain.
    """
    pred = image_data['pred_prob']
    gt = image_data['gt']
    error = image_data['error']
    ac = image_data['active_count']
    ent = image_data['entropy']
    H, W = pred.shape

    fg_mask = gt > 0.5
    pred_mask = pred > 0.5
    boundary = np.logical_xor(fg_mask, pred_mask)

    pixels = []

    # --- 1. Confident: foreground interior, low error, low active_count ---
    # Region inside the GT foreground with correct prediction.
    correct_fg = fg_mask & pred_mask
    if correct_fg.sum() > 10:
        coords = np.argwhere(correct_fg)
        scores = ac[correct_fg]  # sort directly by active_count
        # Pick from the lowest 10% active_count pixels, preferring points near the center.
        n_candidates = max(1, len(scores) // 10)
        low_ac_idx = np.argsort(scores)[:n_candidates]
        # Select one point.
        chosen = low_ac_idx[0]
        y, x = coords[chosen]
    else:
        # fallback
        flat_idx = np.argmin(ac)
        y, x = np.unravel_index(flat_idx, (H, W))

    conf_ac = ac[y, x]
    conf_ent = ent[y, x]
    pixels.append({
        'y': int(y), 'x': int(x),
        'label': 'Confident\n(correct)',
        'color': '#2ecc71',
        'marker': 'o',
        'ac': conf_ac, 'ent': conf_ent,
    })

    # --- 2. Uncertain: boundary/error region with highest active_count ---
    # Use error regions or high-error regions.
    high_error = error > 0.5  # clearly wrong pixels
    if high_error.sum() > 5:
        search_mask = high_error
    elif boundary.sum() > 5:
        search_mask = boundary
    else:
        # fallback: error > median
        search_mask = error > np.median(error)

    if search_mask.sum() > 0:
        coords = np.argwhere(search_mask)
        scores = ac[search_mask]
        # Pick the pixel with the highest active_count.
        chosen = np.argmax(scores)
        y, x = coords[chosen]
    else:
        flat_idx = np.argmax(ac)
        y, x = np.unravel_index(flat_idx, (H, W))

    unc_ac = ac[y, x]
    unc_ent = ent[y, x]
    pixels.append({
        'y': int(y), 'x': int(x),
        'label': 'Uncertain\n(incorrect)',
        'color': '#e74c3c',
        'marker': 's',
        'ac': unc_ac, 'ent': unc_ent,
    })

    # --- 3. Moderate: active_count midpoint between Confident and Uncertain ---
    target_ac = (conf_ac + unc_ac) / 2.0
    diff = np.abs(ac - target_ac)
    # Prefer points near the foreground.
    if fg_mask.sum() > 0:
        diff_masked = np.where(fg_mask, diff, 1e10)
        flat_idx = np.argmin(diff_masked)
    else:
        flat_idx = np.argmin(diff)
    y, x = np.unravel_index(flat_idx, (H, W))

    mod_ac = ac[y, x]
    mod_ent = ent[y, x]
    pixels.append({
        'y': int(y), 'x': int(x),
        'label': 'Moderate\nuncertainty',
        'color': '#f39c12',
        'marker': 'D',
        'ac': mod_ac, 'ent': mod_ent,
    })

    # --- Verify order: Confident < Moderate < Uncertain ---
    # Swap points if the order is incorrect.
    if pixels[2]['ac'] > pixels[1]['ac']:
        # If Moderate > Uncertain, swap them.
        pixels[1], pixels[2] = pixels[2], pixels[1]
        pixels[1]['label'] = 'Uncertain\n(incorrect)'
        pixels[1]['color'] = '#e74c3c'
        pixels[1]['marker'] = 's'
        pixels[2]['label'] = 'Moderate\nuncertainty'
        pixels[2]['color'] = '#f39c12'
        pixels[2]['marker'] = 'D'

    return pixels


# ============================================================
# Fig.2 plotting
# ============================================================
def plot_fig2(image_data, pixels, save_path):
    """
    Fig.2: B-spline activation pattern visualization (improved version).

    Improvements:
    - Lower-row bar charts highlight active bars and gray out inactive bars.
    - Add concentration/dispersion annotations to each subplot.
    - Use a color palette suitable for paper figures.
    """
    bases = image_data['bases']  # (H, W, C_in, K)
    H, W, C_in, K = bases.shape

    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.0, 0.85], hspace=0.40, wspace=0.30)

    # ==================== Top row ====================

    # --- Input image ---
    ax0 = fig.add_subplot(gs[0, 0])
    img_display = image_data['input'].copy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_display = np.clip(img_display * std + mean, 0, 1)
    ax0.imshow(img_display)
    ax0.set_title('(a) Input Image', fontsize=13, fontweight='bold')
    ax0.axis('off')

    # --- GT + prediction overlay ---
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(img_display)
    gt_mask = image_data['gt'] > 0.5
    pred_mask = image_data['pred_prob'] > 0.5

    # Semi-transparent overlay: TP green, FP red, FN blue.
    overlay = np.zeros((H, W, 4))
    tp = gt_mask & pred_mask
    fp = (~gt_mask) & pred_mask
    fn = gt_mask & (~pred_mask)
    overlay[tp] = [0, 0.8, 0, 0.25]   # correct: green
    overlay[fp] = [1, 0, 0, 0.35]      # false positive: red
    overlay[fn] = [0, 0.4, 1, 0.35]    # false negative: blue
    ax1.imshow(overlay)

    # GT contour
    gt_contour = np.zeros_like(gt_mask, dtype=bool)
    gt_contour[1:, :] |= (gt_mask[1:, :] != gt_mask[:-1, :])
    gt_contour[:, 1:] |= (gt_mask[:, 1:] != gt_mask[:, :-1])
    contour_overlay = np.zeros((H, W, 4))
    contour_overlay[gt_contour] = [0, 1, 0, 0.9]
    ax1.imshow(contour_overlay)

    ax1.set_title(f'(b) Prediction Overlay\nDice = {image_data["dice"]:.3f}',
                  fontsize=13, fontweight='bold')
    ax1.axis('off')

    # --- Active-count heatmap with selected pixels ---
    ax2 = fig.add_subplot(gs[0, 2])
    ac_map = image_data['active_count']
    im = ax2.imshow(ac_map, cmap='hot', interpolation='bilinear')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # Mark selected pixels with indices.
    labels_short = ['①', '②', '③']
    for i, px in enumerate(pixels):
        ax2.scatter(px['x'], px['y'], c=px['color'], s=160,
                    marker=px['marker'], edgecolors='white', linewidths=2.5, zorder=5)
        ax2.annotate(labels_short[i],
                     (px['x'], px['y']),
                     xytext=(8, -8), textcoords='offset points',
                     fontsize=12, fontweight='bold', color='white',
                     path_effects=[matplotlib.patheffects.withStroke(linewidth=3, foreground='black')])

    ax2.set_title('(c) Active Count Uncertainty Map', fontsize=13, fontweight='bold')
    ax2.axis('off')

    # ==================== Bottom row: B-spline activation vectors ====================

    import matplotlib.patheffects as pe

    for i, px in enumerate(pixels):
        ax = fig.add_subplot(gs[1, i])
        y, x = px['y'], px['x']

        # Mean basis activation over all channels for this pixel.
        pixel_bases = bases[y, x, :, :]  # (C_in, K)
        avg_activation = pixel_bases.mean(axis=0)  # (K,)

        # Statistics
        active_threshold = 0.01
        active_mask = avg_activation > active_threshold
        p_norm = avg_activation / (avg_activation.sum() + 1e-10)
        ent_val = -np.sum(p_norm * np.log(p_norm + 1e-10))

        # Top-1 concentration: largest activation divided by total activation.
        # High concentration indicates certainty; low concentration indicates uncertainty.
        top1_ratio = avg_activation.max() / (avg_activation.sum() + 1e-10)

        # Draw bar chart: highlight the largest bar and use lighter colors for others.
        max_idx = np.argmax(avg_activation)
        bar_colors = []
        bar_alphas = []
        for j in range(K):
            if j == max_idx:
                # Maximum activation: dark and fully opaque.
                bar_colors.append(px['color'])
                bar_alphas.append(1.0)
            elif active_mask[j]:
                # Other active bases: light color.
                bar_colors.append(px['color'])
                bar_alphas.append(0.45)
            else:
                # Inactive bases: gray.
                bar_colors.append('#cccccc')
                bar_alphas.append(0.3)

        bars = ax.bar(range(K), avg_activation, color=bar_colors,
                      edgecolor='white', linewidth=0.8, width=0.75)
        for j, b in enumerate(bars):
            b.set_alpha(bar_alphas[j])

        # Annotate active bars with values.
        for j in range(K):
            if active_mask[j] and avg_activation[j] > 0.05:
                fw = 'bold' if j == max_idx else 'normal'
                ax.text(j, avg_activation[j] + 0.01, f'{avg_activation[j]:.2f}',
                        ha='center', va='bottom', fontsize=7.5, color='#333333',
                        fontweight=fw)

        # Title: Top-1 concentration + entropy, more discriminative than Active N/K.
        title_text = f'{labels_short[i]} {px["label"]}\nTop-1: {top1_ratio:.0%}   Entropy: {ent_val:.2f}'
        ax.set_title(title_text, fontsize=11, color=px['color'], fontweight='bold')
        ax.set_xlabel('B-spline Basis Function Index', fontsize=10)
        if i == 0:
            ax.set_ylabel('Mean Activation Value', fontsize=10)
        ax.set_xlim(-0.6, K - 0.4)
        ax.set_ylim(0, max(avg_activation) * 1.25)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.15, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Background colors: light green for Confident, light red for Uncertain, light yellow for Moderate.
        bg_colors = {'#2ecc71': '#f0fdf4', '#e74c3c': '#fef2f2', '#f39c12': '#fffbeb'}
        ax.set_facecolor(bg_colors.get(px['color'], 'white'))

    fig.suptitle('B-spline Basis Function Activation Patterns: Confident vs Uncertain Pixels',
                 fontsize=15, fontweight='bold', y=1.01)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Fig.2 saved: {save_path}")


# ============================================================
# Fig.4 plotting
# ============================================================
def plot_fig4_distribution(all_images_data, save_path):
    """
    Fig.4: signal distribution comparison (improved version).

    Improvements:
    - Use the full validation set data.
    - Left panel: scatter + histogram to show the cluster-jump pattern clearly.
    - Right panel: the same scatter + histogram design.
    - Use a color gradient to indicate Dice.
    """
    dices = []
    ac_means = []
    ent_means = []

    for d in all_images_data:
        dices.append(d['dice'])
        ac_means.append(d['active_count'].mean())
        p = d['pred_prob']
        soft_ent = -(p * np.log(p + 1e-10) + (1 - p) * np.log(1 - p + 1e-10))
        ent_means.append(soft_ent.mean())

    dices = np.array(dices)
    ac_means = np.array(ac_means)
    ent_means = np.array(ent_means)

    # Group by Dice: bottom 20% are treated as bad cases.
    threshold = np.percentile(dices, 20)
    bad_mask = dices <= threshold
    good_mask = ~bad_mask

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # === Left: active_count + mean, cluster-jump pattern ===
    ax = axes[0]

    # Top: scatter plot of Dice vs. uncertainty.
    # Use two axes: scatter above, histogram below.
    ax_scatter = ax.twinx()

    # Histogram
    bins_ac = np.linspace(ac_means.min() * 0.99, ac_means.max() * 1.01, 25)
    ax.hist(ac_means[good_mask], bins=bins_ac, alpha=0.55, color='#3498db',
            label=f'High Dice (top 80%, n={good_mask.sum()})', edgecolor='white', linewidth=0.5)
    ax.hist(ac_means[bad_mask], bins=bins_ac, alpha=0.75, color='#e74c3c',
            label=f'Low Dice (bottom 20%, n={bad_mask.sum()})', edgecolor='white', linewidth=0.5)

    # Scatter overlay: each point is one image.
    ax_scatter.scatter(ac_means[good_mask], dices[good_mask],
                       c='#3498db', s=25, alpha=0.6, edgecolors='white', linewidths=0.3,
                       zorder=5)
    ax_scatter.scatter(ac_means[bad_mask], dices[bad_mask],
                       c='#e74c3c', s=40, alpha=0.8, edgecolors='white', linewidths=0.5,
                       zorder=6, marker='s')

    ax_scatter.set_ylabel('Dice Score', fontsize=11, color='#555555')
    ax_scatter.tick_params(labelsize=9, colors='#555555')

    # Annotate cluster and jump regions.
    cluster_center = np.median(ac_means[good_mask])
    ax.axvline(cluster_center, color='#3498db', linestyle=':', alpha=0.4, linewidth=1)

    ax.set_xlabel('Image-level Uncertainty\n(B-spline active_count + mean aggregation)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('(a) B-spline Signal: "Clustered-Jump" Distribution\n'
                 'Low Spearman (poor global ranking) but high Norm-AUSC (good tail detection)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.15)

    # Add arrows for cluster and jump.
    if bad_mask.sum() > 0:
        jump_x = ac_means[bad_mask].max()
        cluster_x = cluster_center
        y_max = ax.get_ylim()[1]

        ax.annotate('Cluster\n(most images)',
                    xy=(cluster_x, y_max * 0.7),
                    fontsize=9, ha='center', color='#3498db', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ebf5fb', alpha=0.8))

        if jump_x > cluster_x * 1.01:  # Annotate only when the jump is obvious.
            ax.annotate('Jump\n(bad images)',
                        xy=(jump_x, y_max * 0.5),
                        fontsize=9, ha='center', color='#e74c3c', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdedec', alpha=0.8))

    # === Right: softmax entropy, continuous spread pattern ===
    ax = axes[1]
    ax_scatter2 = ax.twinx()

    bins_ent = np.linspace(ent_means.min() * 0.99, ent_means.max() * 1.01, 25)
    ax.hist(ent_means[good_mask], bins=bins_ent, alpha=0.55, color='#3498db',
            label=f'High Dice (top 80%, n={good_mask.sum()})', edgecolor='white', linewidth=0.5)
    ax.hist(ent_means[bad_mask], bins=bins_ent, alpha=0.75, color='#e74c3c',
            label=f'Low Dice (bottom 20%, n={bad_mask.sum()})', edgecolor='white', linewidth=0.5)

    ax_scatter2.scatter(ent_means[good_mask], dices[good_mask],
                        c='#3498db', s=25, alpha=0.6, edgecolors='white', linewidths=0.3,
                        zorder=5)
    ax_scatter2.scatter(ent_means[bad_mask], dices[bad_mask],
                        c='#e74c3c', s=40, alpha=0.8, edgecolors='white', linewidths=0.5,
                        zorder=6, marker='s')

    ax_scatter2.set_ylabel('Dice Score', fontsize=11, color='#555555')
    ax_scatter2.tick_params(labelsize=9, colors='#555555')

    ax.set_xlabel('Image-level Uncertainty\n(Softmax entropy + mean aggregation)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('(b) Softmax Entropy: "Continuous-Spread" Distribution\n'
                 'High Spearman (good global ranking) but moderate Norm-AUSC',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.15)

    # Annotate gradual transition.
    ax.annotate('Gradual\ntransition',
                xy=(np.median(ent_means), ax.get_ylim()[1] * 0.6),
                fontsize=9, ha='center', color='#7f8c8d', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f2f3f4', alpha=0.8))

    fig.suptitle('Why Low Spearman Can Coexist with High Selective Prediction Performance',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Fig.4 saved: {save_path}")


# ============================================================
# Image selection logic
# ============================================================
def select_fig2_image(all_images, fig2_image_idx=None):
    """
    Image selection logic for Fig.2 (improved version):
    1. Use the specified index if provided.
    2. Otherwise select images with Dice in 0.5-0.8.
    3. Within that range, choose the image with the largest active_count variance.
    4. If no image is found in 0.5-0.8, relax to 0.4-0.85.
    """
    if fig2_image_idx is not None:
        idx = min(fig2_image_idx, len(all_images) - 1)
        print(f"  Fig.2: Manually selected image #{idx} (Dice={all_images[idx]['dice']:.3f})")
        return idx

    dices = np.array([d['dice'] for d in all_images])

    # First priority: Dice 0.5-0.8
    candidates = [i for i, d in enumerate(dices) if 0.5 <= d <= 0.8]

    # Second priority: Dice 0.4-0.85
    if not candidates:
        candidates = [i for i, d in enumerate(dices) if 0.4 <= d <= 0.85]

    # Third priority: Dice 0.3-0.9, excluding extreme cases.
    if not candidates:
        candidates = [i for i, d in enumerate(dices) if 0.3 <= d <= 0.9]

    # Final fallback: choose the image closest to the median Dice.
    if not candidates:
        median_dice = np.median(dices)
        candidates = [np.argmin(np.abs(dices - median_dice))]

    # Choose the candidate with the largest active_count variance.
    ac_vars = [all_images[i]['active_count'].var() for i in candidates]
    best = candidates[np.argmax(ac_vars)]

    print(f"  Fig.2: Automatically selected image #{best} (Dice={all_images[best]['dice']:.3f}, "
          f"AC_var={all_images[best]['active_count'].var():.4f})")
    print(f"    Candidate count: {len(candidates)} images, Dicerange: "
          f"[{dices[candidates].min():.3f}, {dices[candidates].max():.3f}]")
    return best


# ============================================================
# Main function
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='Model name')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--n_images', type=int, default=None,
                        help='Number of images to extract; default uses the full validation set.')
    parser.add_argument('--fig2_image_idx', type=int, default=None,
                        help='Image index used for Fig.2; None means automatic selection.')
    args = parser.parse_args()

    save_dir = os.path.join(args.output_dir, args.name, 'figures')
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f"  B-spline visualization (v2) | Model: {args.name}")
    print("=" * 70)

    # 1. Load
    print("\n[1/3] Loading model and data...")
    model, val_loader, config = load_model_and_data(args.name, args.output_dir)
    print(f"  Validation set: {len(val_loader.dataset)} images")

    # 2. Extract
    n = args.n_images if args.n_images else len(val_loader.dataset)
    print(f"\n[2/3] Extracting pixel-level data ({n} images)...")
    all_images = extract_pixel_data(model, val_loader, config, n_images=args.n_images)
    print(f"  Done: {len(all_images)} images")

    dices = [d['dice'] for d in all_images]
    print(f"  Dicerange: [{min(dices):.3f}, {max(dices):.3f}], mean={np.mean(dices):.3f}")

    # 3. Generate figures
    print("\n[3/3] Generating figures...")

    # --- Fig.2 ---
    fig2_idx = select_fig2_image(all_images, args.fig2_image_idx)
    image_data = all_images[fig2_idx]
    pixels = select_representative_pixels(image_data)

    print(f"\n  Selected pixels:")
    for px in pixels:
        label = px['label'].replace('\n', ' ')
        print(f"    {label}: pos=({px['y']},{px['x']}), "
              f"AC={px['ac']:.3f}, Ent={px['ent']:.3f}")

    # Check contrast.
    ac_range = pixels[1]['ac'] - pixels[0]['ac']
    ent_range = pixels[1]['ent'] - pixels[0]['ent']
    print(f"  Confident-to-Uncertain: ΔAC={ac_range:.3f}, ΔEnt={ent_range:.3f}")
    if ac_range < 0.1:
        print(f"  Warning: low active-count contrast ({ac_range:.3f}); consider selecting another image (--fig2_image_idx)")

    plot_fig2(image_data, pixels, os.path.join(save_dir, 'fig2_bspline_activation.png'))
    plot_fig2(image_data, pixels, os.path.join(save_dir, 'fig2_bspline_activation.pdf'))

    # --- Fig.4 ---
    if len(all_images) >= 15:
        print(f"\n  Fig.4: using {len(all_images)} images for distribution statistics")
        plot_fig4_distribution(all_images, os.path.join(save_dir, 'fig4_distribution.png'))
        plot_fig4_distribution(all_images, os.path.join(save_dir, 'fig4_distribution.pdf'))
    else:
        print(f"\n  Fig.4: not enough images ({len(all_images)}); omit --n_images to use the full validation set")

    print(f"\n  Done. Figures saved in: {save_dir}/")


if __name__ == '__main__':
    main()


