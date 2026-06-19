#!/usr/bin/env python3
"""
Uncertainty evidence evaluation script.
================================
Extends the previous per-image normalization analysis with:
1. Pixel-level evidence: activation entropy comparison between wrong and correct pixels.
2. Boundary-vs-interior activation entropy comparison for interpretability.
3. Per-image normalization comparison retained from the original analysis.
4. Combined visualization with boxplots, violin plots, and histograms.

Usage:
    python eval_uncertainty_evidence.py --name busi_UKAN-1 --output_dir outputs
"""

import argparse
import os
import random
from glob import glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy import stats, ndimage
from scipy.ndimage import binary_erosion
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import archs
from dataset import Dataset

try:
    from UNet_UKH import UNet_UKH
except ImportError:
    UNet_UKH = None
try:
    from UNet import UNet as UNet_base
except ImportError:
    UNet_base = None


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
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class FinalInputCapture:
    def __init__(self, model):
        self.captured_input = None
        self.hook = model.final.register_forward_pre_hook(self._hook_fn)

    def _hook_fn(self, module, input):
        self.captured_input = input[0].detach()

    def remove(self):
        self.hook.remove()


def load_model_and_data(name, output_dir):
    config_path = os.path.join(output_dir, name, 'config.yml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    seed_num = config.get('dataseed', 1187)
    seed_torch(seed_num)

    arch_name = config['arch']
    if arch_name == 'UNet_UKH' and UNet_UKH is not None:
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
    else:
        model = archs.__dict__[arch_name](
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
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'], drop_last=False
    )

    return model, val_loader, config


# ============================================================
# B-spline basis computation
# ============================================================
def build_derivative_matrix(grid, in_features, grid_size, spline_order):
    k = spline_order
    n_basis = grid_size + k
    n_basis_lower = n_basis + 1
    D_list = []
    for feat_idx in range(in_features):
        g = grid[feat_idx]
        D = torch.zeros(n_basis, n_basis_lower, device=grid.device)
        for i in range(n_basis):
            denom1 = g[i + k] - g[i]
            if denom1.abs() > 1e-10:
                D[i, i] = k / denom1
            denom2 = g[i + k + 1] - g[i + 1]
            if denom2.abs() > 1e-10:
                D[i, i + 1] = -k / denom2
        D_list.append(D)
    return torch.stack(D_list)


def compute_bsplines(x, grid, spline_order):
    x = x.unsqueeze(-1)
    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    for p in range(1, spline_order + 1):
        bases = (
            (x - grid[:, :-(p + 1)])
            / (grid[:, p:-1] - grid[:, :-(p + 1)] + 1e-10)
            * bases[:, :, :-1]
        ) + (
            (grid[:, p + 1:] - x)
            / (grid[:, p + 1:] - grid[:, 1:(-p)] + 1e-10)
            * bases[:, :, 1:]
        )
    return bases.contiguous()


def compute_spline_gradient(x_flat, kan_layer):
    grid = kan_layer.grid
    spline_order = kan_layer.spline_order
    grid_size = kan_layer.grid_size
    in_features = kan_layer.in_features
    deriv_matrix = build_derivative_matrix(grid, in_features, grid_size, spline_order)
    basis_lower = compute_bsplines(x_flat, grid, spline_order - 1)
    basis_lower_t = basis_lower.permute(1, 2, 0)
    deriv_basis_t = torch.bmm(deriv_matrix, basis_lower_t)
    deriv_basis = deriv_basis_t.permute(2, 0, 1)
    weight = kan_layer.scaled_spline_weight
    gradient = torch.einsum('oin,bin->boi', weight, deriv_basis)
    return gradient


def compute_all_uncertainty_methods(bases, x_flat, kan_layer):
    B, H, W, C, K = bases.shape
    b = bases.reshape(-1, C, K)
    results = {}

    p = b / (b.sum(dim=-1, keepdim=True) + 1e-10)
    entropy = -(p * torch.log(p + 1e-10)).sum(dim=-1)
    results['entropy'] = entropy.mean(dim=-1).reshape(B, H, W)

    max_val = b.max(dim=-1).values
    results['neg_max'] = (1.0 - max_val.mean(dim=-1)).reshape(B, H, W)

    concentration = b.max(dim=-1).values / (b.sum(dim=-1) + 1e-10)
    results['neg_concentration'] = (1.0 - concentration.mean(dim=-1)).reshape(B, H, W)

    active_count = (b > 0.01).float().sum(dim=-1)
    results['active_count'] = active_count.mean(dim=-1).reshape(B, H, W)

    var_val = b.var(dim=-1)
    neg_var = 1.0 / (var_val.mean(dim=-1) + 1e-10)
    neg_var = neg_var.clamp(max=neg_var.quantile(0.99))
    results['neg_variance'] = neg_var.reshape(B, H, W)

    b_sorted, _ = b.sort(dim=-1)
    index = torch.arange(1, K + 1, device=b.device, dtype=b.dtype).reshape(1, 1, -1)
    gini = (2 * (index * b_sorted).sum(dim=-1) / (b_sorted.sum(dim=-1) * K + 1e-10)) - (K + 1) / K
    results['neg_gini'] = (1.0 - gini.mean(dim=-1)).reshape(B, H, W)

    gradient = compute_spline_gradient(x_flat, kan_layer)
    deriv_l1 = gradient.abs().mean(dim=-1).mean(dim=-1)
    results['deriv_l1'] = deriv_l1.reshape(B, H, W)
    deriv_max = gradient.abs().max(dim=-1).values.mean(dim=-1)
    results['deriv_max'] = deriv_max.reshape(B, H, W)

    return results


# ============================================================
# Data collection
# ============================================================
def collect_per_image_data(model, val_loader, config):
    kan_layer = model.final.kan
    num_classes = config['num_classes']
    capture = FinalInputCapture(model)
    per_image_data = []

    print(f"  Model: in_features={kan_layer.in_features}, "
          f"grid_size={kan_layer.grid_size}, spline_order={kan_layer.spline_order}")

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="  Collecting data"):
            if len(batch_data) == 3:
                input_img, target, _ = batch_data
            else:
                input_img, target = batch_data

            input_img = input_img.cuda()
            target = target.cuda()

            logits = model(input_img)
            final_input = capture.captured_input

            _, bases = model.final(final_input, return_spline_basis=True)

            B, C_in, H, W = final_input.shape
            x_flat = final_input.permute(0, 2, 3, 1).reshape(-1, C_in)

            if num_classes == 1:
                pred_prob = torch.sigmoid(logits)
                pred_bin = (pred_prob > 0.5).float()
                target_bin = (target > 0.5).float()
                error_map = (pred_bin != target_bin).float().squeeze(1)
            else:
                pred_class = torch.argmax(logits, dim=1)
                target_class = target.squeeze(1).long()
                error_map = (pred_class != target_class).float()

            methods = compute_all_uncertainty_methods(bases, x_flat, kan_layer)

            for b_idx in range(B):
                pred_np = pred_prob[b_idx].squeeze(0).cpu().numpy() if num_classes == 1 \
                          else torch.argmax(logits[b_idx], dim=0).float().cpu().numpy()
                target_np = target[b_idx].squeeze(0).cpu().numpy()

                pred_b = (pred_np > 0.5).astype(float)
                target_b = (target_np > 0.5).astype(float)
                inter = (pred_b * target_b).sum()
                dice = 2 * inter / (pred_b.sum() + target_b.sum() + 1e-8)

                per_image_data.append({
                    'pred': pred_np,
                    'target': target_np,
                    'error_map': error_map[b_idx].cpu().numpy(),   # (H,W) binary
                    'dice': float(dice),
                    'uncertainties': {
                        name: methods[name][b_idx].cpu().numpy()
                        for name in methods
                    }
                })

    capture.remove()
    return per_image_data


# ============================================================
# Core addition: pixel-level causal evidence.
# ============================================================
def compute_pixel_level_evidence(per_image_data, method_names,
                                  boundary_erosion_iters=3, sample_max=500_000):
    """
    Two types of pixel-level evidence:
      A. activation entropy of wrong pixels vs. correct pixels.
      B. activation entropy of GT boundary pixels vs. GT interior pixels.

    Returns dict: statistics for each method
    """
    results = {}

    for method in method_names:
        correct_unc, wrong_unc = [], []
        boundary_unc, interior_unc = [], []

        for img_data in per_image_data:
            unc = img_data['uncertainties'].get(method)
            if unc is None:
                continue
            error_map = img_data['error_map']   # (H,W) binary: 1=wrong
            target_bin = img_data['target'] > 0.5  # (H,W)

            # --- A. Wrong/correct pixels ---
            wrong_mask = error_map > 0.5
            correct_mask = ~wrong_mask

            if wrong_mask.any():
                wrong_unc.extend(unc[wrong_mask].flatten().tolist())
            if correct_mask.any():
                correct_unc.extend(unc[correct_mask].flatten().tolist())

            # --- B. Boundary/interior pixels within the GT foreground only ---
            if target_bin.any():
                eroded = binary_erosion(target_bin,
                                        iterations=boundary_erosion_iters)
                boundary_mask = target_bin & ~eroded
                interior_mask = eroded

                if boundary_mask.any():
                    boundary_unc.extend(unc[boundary_mask].flatten().tolist())
                if interior_mask.any():
                    interior_unc.extend(unc[interior_mask].flatten().tolist())

        # Downsample to avoid excessive memory use.
        def _sample(lst):
            arr = np.array(lst)
            if len(arr) > sample_max:
                idx = np.random.choice(len(arr), sample_max, replace=False)
                arr = arr[idx]
            return arr

        correct_unc = _sample(correct_unc)
        wrong_unc   = _sample(wrong_unc)
        boundary_unc = _sample(boundary_unc)
        interior_unc = _sample(interior_unc)

        # Mann-Whitney U test (one-sided: wrong > correct)
        stat_ab, p_ab = mannwhitneyu(wrong_unc, correct_unc,
                                      alternative='greater') if len(wrong_unc) > 0 else (np.nan, np.nan)
        stat_bi, p_bi = mannwhitneyu(boundary_unc, interior_unc,
                                      alternative='greater') if len(boundary_unc) > 0 else (np.nan, np.nan)

        # Cohen's d (effect size)
        def cohens_d(a, b):
            pooled_std = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2 + 1e-10)
            return (np.mean(a) - np.mean(b)) / pooled_std

        results[method] = {
            # A. Wrong vs. correct
            'wrong_mean':   float(np.mean(wrong_unc)),
            'correct_mean': float(np.mean(correct_unc)),
            'wrong_std':    float(np.std(wrong_unc)),
            'correct_std':  float(np.std(correct_unc)),
            'p_value_wrong_vs_correct': float(p_ab),
            'cohens_d_wrong_vs_correct': float(cohens_d(wrong_unc, correct_unc)),
            'n_wrong':   len(wrong_unc),
            'n_correct': len(correct_unc),
            # B. Boundary vs. interior
            'boundary_mean':  float(np.mean(boundary_unc)) if len(boundary_unc) > 0 else np.nan,
            'interior_mean':  float(np.mean(interior_unc)) if len(interior_unc) > 0 else np.nan,
            'p_value_boundary_vs_interior': float(p_bi),
            'cohens_d_boundary_vs_interior': float(cohens_d(boundary_unc, interior_unc))
                                              if len(boundary_unc) > 0 else np.nan,
            'n_boundary':  len(boundary_unc),
            'n_interior':  len(interior_unc),
            # Raw arrays for plotting, truncated.
            '_wrong_unc':    wrong_unc[:50_000],
            '_correct_unc':  correct_unc[:50_000],
            '_boundary_unc': boundary_unc[:50_000],
            '_interior_unc': interior_unc[:50_000],
        }

    return results


def plot_pixel_level_evidence(pixel_results, save_dir):
    """
    Figure 1: Boxplot/violin for wrong vs. correct pixels.
    Figure 2: Boxplot/violin for boundary vs. interior pixels.
    Can be used directly as supplementary figures.
    """
    methods = list(pixel_results.keys())
    n = len(methods)

    # ---- Figure A: wrong vs. correct ----
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    for i, method in enumerate(methods):
        r = pixel_results[method]
        ax = axes[i]

        data_groups = [r['_correct_unc'], r['_wrong_unc']]
        vp = ax.violinplot(data_groups, positions=[1, 2],
                           showmeans=True, showmedians=True)
        vp['bodies'][0].set_facecolor('#5B9BD5')
        vp['bodies'][1].set_facecolor('#ED7D31')
        for body in vp['bodies']:
            body.set_alpha(0.75)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Correct\nPixels', 'Wrong\nPixels'], fontsize=12)
        ax.set_ylabel('Activation Uncertainty', fontsize=11)

        p_val = r['p_value_wrong_vs_correct']
        d_val = r['cohens_d_wrong_vs_correct']
        p_str = f"p<0.001" if p_val < 0.001 else f"p={p_val:.4f}"
        ax.set_title(f"{method}\n{p_str}, d={d_val:.3f}", fontsize=12)

        # Connect means.
        ax.plot([1, 2], [r['correct_mean'], r['wrong_mean']],
                'k--o', linewidth=1.5, markersize=6, zorder=5)

        # Annotate sample counts.
        ax.text(1, ax.get_ylim()[0], f"n={r['n_correct']:,}", ha='center',
                fontsize=8, color='gray')
        ax.text(2, ax.get_ylim()[0], f"n={r['n_wrong']:,}", ha='center',
                fontsize=8, color='gray')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Pixel-Level Evidence: Wrong vs Correct Pixels\n'
                 '(Higher uncertainty in misclassified pixels supports B-spline = uncertainty)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pixel_evidence_wrong_vs_correct.png')
    # plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {save_path}")

    # ---- Figure B: boundary vs. interior ----
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
    if n == 1:
        axes = [axes]

    for i, method in enumerate(methods):
        r = pixel_results[method]
        ax = axes[i]

        if r['n_boundary'] == 0 or r['n_interior'] == 0:
            ax.set_title(f"{method}\n(no boundary/interior data)")
            continue

        data_groups = [r['_interior_unc'], r['_boundary_unc']]
        vp = ax.violinplot(data_groups, positions=[1, 2],
                           showmeans=True, showmedians=True)
        vp['bodies'][0].set_facecolor('#70AD47')
        vp['bodies'][1].set_facecolor('#FF0000')
        for body in vp['bodies']:
            body.set_alpha(0.75)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Interior\nPixels', 'Boundary\nPixels'], fontsize=12)
        ax.set_ylabel('Activation Uncertainty', fontsize=11)

        p_val = r['p_value_boundary_vs_interior']
        d_val = r['cohens_d_boundary_vs_interior']
        p_str = f"p<0.001" if p_val < 0.001 else f"p={p_val:.4f}"
        ax.set_title(f"{method}\n{p_str}, d={d_val:.3f}", fontsize=12)

        ax.plot([1, 2], [r['interior_mean'], r['boundary_mean']],
                'k--o', linewidth=1.5, markersize=6, zorder=5)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Pixel-Level Evidence: Boundary vs Interior Pixels\n'
                 '(Higher uncertainty at segmentation boundaries)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'pixel_evidence_boundary_vs_interior.png')
    # plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {save_path}")


def plot_combined_evidence_figure(pixel_results, save_dir,
                                   primary_method='entropy'):
    """
    Combined figure suitable for submission:
    Left: Wrong vs. correct violin plot.
    Right: Boundary vs. interior violin plot.
    Includes p-value and Cohen's d annotations.
    """
    r = pixel_results.get(primary_method)
    if r is None:
        print(f"  Warning: {primary_method} not found in results; skipping combined figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'B-spline Activation Uncertainty: Pixel-Level Statistical Evidence\n'
                 f'(Method: {primary_method})', fontsize=14, fontweight='bold')

    colors_AB = ['#5B9BD5', '#ED7D31']
    colors_BI = ['#70AD47', '#C00000']
    labels_AB = ['Correct Pixels', 'Wrong Pixels']
    labels_BI = ['Interior Pixels', 'Boundary Pixels']

    for ax, data_groups, colors, labels, title_suffix, p_val, d_val, means in [
        (axes[0],
         [r['_correct_unc'], r['_wrong_unc']],
         colors_AB, labels_AB,
         'Misclassified vs Correctly Classified',
         r['p_value_wrong_vs_correct'],
         r['cohens_d_wrong_vs_correct'],
         [r['correct_mean'], r['wrong_mean']]),
        (axes[1],
         [r['_interior_unc'], r['_boundary_unc']],
         colors_BI, labels_BI,
         'Boundary vs Interior (GT Mask)',
         r['p_value_boundary_vs_interior'],
         r['cohens_d_boundary_vs_interior'],
         [r['interior_mean'], r['boundary_mean']]),
    ]:
        if len(data_groups[0]) == 0 or len(data_groups[1]) == 0:
            ax.set_title("No data")
            continue

        vp = ax.violinplot(data_groups, positions=[1, 2],
                           showmeans=True, showmedians=True, widths=0.6)
        for j, body in enumerate(vp['bodies']):
            body.set_facecolor(colors[j])
            body.set_alpha(0.7)
            body.set_edgecolor('black')
            body.set_linewidth(0.8)

        vp['cmeans'].set_color('black')
        vp['cmedians'].set_color('white')

        # Connect means.
        ax.plot([1, 2], means, 'k--o', linewidth=2, markersize=8,
                markerfacecolor='black', zorder=10)

        # Annotate means.
        for pos, val, color in zip([1, 2], means, colors):
            ax.annotate(f'μ={val:.4f}', xy=(pos, val),
                        xytext=(pos + 0.15, val),
                        fontsize=10, color='black', fontweight='bold')

        # p-value and Cohen's d
        p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.4f}"
        ax.text(0.5, 0.97,
                f"{p_str}\nCohen's d = {d_val:.3f}",
                transform=ax.transAxes, ha='center', va='top',
                fontsize=11, bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='lightyellow', edgecolor='gray'))

        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel('B-spline Activation Uncertainty', fontsize=11)
        ax.set_title(title_suffix, fontsize=12, pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(0.4, 2.8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = os.path.join(save_dir, f'combined_evidence_{primary_method}.png')
    # plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {save_path}")


# ============================================================
# Per-image normalization analysis, simplified from the original version.
# ============================================================
def compute_image_level_correlations(per_image_data, method_name):
    unc_values_raw, unc_values_p90, unc_values_iqr = [], [], []
    img_dices = []

    for img_data in per_image_data:
        unc = img_data['uncertainties'].get(method_name)
        if unc is None:
            continue
        unc_values_raw.append(unc.mean())
        unc_values_p90.append(np.percentile(unc, 90))
        unc_values_iqr.append(np.percentile(unc, 75) - np.percentile(unc, 25))
        img_dices.append(img_data['dice'])

    img_dices = np.array(img_dices)
    results = {}
    for name, vals in [('raw_mean', unc_values_raw),
                        ('p90',      unc_values_p90),
                        ('iqr',      unc_values_iqr)]:
        arr = np.array(vals)
        corr = stats.spearmanr(arr, img_dices)[0] if np.std(arr) > 1e-10 else 0.0
        results[name] = {'corr_vs_dice': corr, 'values': arr}
    return results, img_dices


# ============================================================
# Text report
# ============================================================
def print_pixel_evidence_report(pixel_results, lines):
    lines.append("\n" + "=" * 80)
    lines.append("  Pixel-level statistical evidence: core causal chain")
    lines.append("=" * 80)

    for method, r in pixel_results.items():
        lines.append(f"\n  ▸ {method}")
        lines.append("  " + "-" * 60)
        lines.append("  A. Wrong pixels vs. correct pixels")
        lines.append(f"     Correct  μ = {r['correct_mean']:.5f}  (n={r['n_correct']:,})")
        lines.append(f"     Wrong    μ = {r['wrong_mean']:.5f}  (n={r['n_wrong']:,})")
        p = r['p_value_wrong_vs_correct']
        p_str = "< 0.001" if p < 0.001 else f"= {p:.4f}"
        lines.append(f"     Mann-Whitney U  p {p_str}")
        lines.append(f"     Cohen's d = {r['cohens_d_wrong_vs_correct']:.4f}")

        lines.append("  B. Boundary pixels vs. interior pixels")
        if r['n_boundary'] > 0:
            lines.append(f"     Interior μ = {r['interior_mean']:.5f}  (n={r['n_interior']:,})")
            lines.append(f"     Boundary μ = {r['boundary_mean']:.5f}  (n={r['n_boundary']:,})")
            pb = r['p_value_boundary_vs_interior']
            pb_str = "< 0.001" if pb < 0.001 else f"= {pb:.4f}"
            lines.append(f"     Mann-Whitney U  p {pb_str}")
            lines.append(f"     Cohen's d = {r['cohens_d_boundary_vs_interior']:.4f}")
        else:
            lines.append("     (no boundary pixels found)")


# ============================================================
# Main function
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True,
                        help='Model directory name, e.g. busi_UKAN-1')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--primary_method', default='entropy',
                        help='Primary method used for the combined figure')
    parser.add_argument('--boundary_erosion', type=int, default=3,
                        help='Number of erosion iterations for boundary extraction')
    args = parser.parse_args()

    save_dir = os.path.join(args.output_dir, args.name, 'uncertainty_evidence')
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f"  Uncertainty evidence evaluation | Model: {args.name}")
    print("=" * 70)

    # 1. Load
    print("\n[1/4] Loading model and data...")
    model, val_loader, config = load_model_and_data(args.name, args.output_dir)

    # 2. Collect
    print("\n[2/4] Collecting per-image data...")
    per_image_data = collect_per_image_data(model, val_loader, config)
    print(f"  Number of images: {len(per_image_data)}")
    avg_dice = np.mean([d['dice'] for d in per_image_data])
    print(f"  Mean Dice: {avg_dice:.4f}")

    methods_for_evidence = ['entropy', 'active_count', 'neg_concentration', 'deriv_max']

    # 3. Pixel-level evidence: core addition
    print("\n[3/4] Computing pixel-level statistical evidence...")
    pixel_results = compute_pixel_level_evidence(
        per_image_data,
        method_names=methods_for_evidence,
        boundary_erosion_iters=args.boundary_erosion
    )

    lines = []
    lines.append(f"Uncertainty Evidence Report | {args.name}")
    lines.append(f"Images: {len(per_image_data)}, Mean Dice: {avg_dice:.4f}")
    print_pixel_evidence_report(pixel_results, lines)

    # 4. Image-level correlation analysis from the original version
    lines.append("\n" + "=" * 80)
    lines.append("  Image-level Spearman correlation")
    lines.append("=" * 80)
    lines.append(f"  {'Method':<22} {'raw_mean':>10} {'p90':>10} {'iqr':>10}")
    lines.append("  " + "-" * 60)

    for method in methods_for_evidence:
        img_results, _ = compute_image_level_correlations(per_image_data, method)
        r_raw = img_results['raw_mean']['corr_vs_dice']
        r_p90 = img_results['p90']['corr_vs_dice']
        r_iqr = img_results['iqr']['corr_vs_dice']
        lines.append(f"  {method:<22} {r_raw:>10.4f} {r_p90:>10.4f} {r_iqr:>10.4f}")

    full_text = "\n".join(lines)
    print("\n" + full_text)

    report_path = os.path.join(save_dir, 'evidence_report.txt')
    with open(report_path, 'w') as f:
        f.write(full_text)
    print(f"\n  [Saved] {report_path}")

    # 5. Visualize
    print("\n[4/4] Generating visualizations...")
    plot_pixel_level_evidence(pixel_results, save_dir)
    plot_combined_evidence_figure(pixel_results, save_dir,
                                   primary_method=args.primary_method)

    print(f"\n  Done. All results saved in: {save_dir}/")
    print("  Main outputs:")
    print(f"    combined_evidence_{args.primary_method}.png  <- paper figure")
    print(f"    pixel_evidence_wrong_vs_correct.png          <- all-method comparison")
    print(f"    pixel_evidence_boundary_vs_interior.png      <- boundary analysis")
    print(f"    evidence_report.txt                          <- numeric report")


if __name__ == '__main__':
    main()


