#!/usr/bin/env python3
"""
Selective Prediction evaluation script v2 (standard metrics).
==============================================
Key changes:
- Add AURC (Area Under Risk-Coverage curve, Geifman et al. 2019); lower is better.
- Add E-AURC (Excess-AURC = AURC_method - AURC_oracle); lower is better and 0 is perfect.
- Keep 20% rejection Delta Dice and Spearman as auxiliary metrics.
- Per-image IoU / HD95
- Remove the custom Norm-AUSC metric.

Usage:
    python eval_selective_prediction_v2.py --name cvc_UNet-re-2981 --output_dir outputs
    python eval_selective_prediction_v2.py --name busi_TransUNet-UKH-42 --output_dir outputs
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
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import archs
from dataset import Dataset

from medpy.metric.binary import hd95 as compute_hd95

try:
    from UNet_UKH import UNet_UKH
except ImportError:
    UNet_UKH = None
try:
    from UNet import UNet as UNet_base
except ImportError:
    UNet_base = None
try:
    from TransUNet_UKH import TransUNet_UKH
except ImportError:
    TransUNet_UKH = None

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


def compute_per_image_metrics(pred_bin, tgt_bin):
    """
    Compute Dice, IoU, and HD95 for a single image.
    pred_bin, tgt_bin: numpy array, binary {0, 1}
    """
    inter = (pred_bin * tgt_bin).sum()
    union = ((pred_bin + tgt_bin) > 0).astype(float).sum()

    dice = 2 * inter / (pred_bin.sum() + tgt_bin.sum() + 1e-8)
    iou = inter / (union + 1e-8)

    try:
        if pred_bin.sum() > 0 and tgt_bin.sum() > 0:
            hd95_val = compute_hd95(pred_bin.astype(bool), tgt_bin.astype(bool))
        else:
            hd95_val = 0.0
    except Exception:
        hd95_val = 0.0

    return dice, iou, hd95_val


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
    elif arch_name == 'TransUNet_UKH' and TransUNet_UKH is not None:
        model = TransUNet_UKH(
            in_ch=config.get('input_channels', 3),
            num_classes=config['num_classes'],
            base_ch=config.get('trans_base_ch', 64),
            embed_dim=config.get('trans_embed_dim', 512),
            num_heads=config.get('trans_num_heads', 8),
            depth=config.get('trans_depth', 8),
            mlp_ratio=config.get('trans_mlp_ratio', 3.0),
            dropout=config.get('drop_rate', 0.1),
            img_size=config.get('input_h', 256),
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
    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name in ['glas', 'cvc']:
        mask_ext = '.png'
    else:
        mask_ext = '.png'

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


def compute_basis_uncertainty(bases, method='entropy'):
    """Compute pixel-level uncertainty from basis activations."""
    B, H, W, C, K = bases.shape
    b = bases.reshape(-1, C, K)

    if method == 'entropy':
        p = b / (b.sum(dim=-1, keepdim=True) + 1e-10)
        unc = -(p * torch.log(p + 1e-10)).sum(dim=-1)
        return unc.mean(dim=-1).reshape(B, H, W)
    elif method == 'active_count':
        active = (b > 0.01).float().sum(dim=-1)
        return active.mean(dim=-1).reshape(B, H, W)
    elif method == 'neg_concentration':
        conc = b.max(dim=-1).values / (b.sum(dim=-1) + 1e-10)
        return (1.0 - conc.mean(dim=-1)).reshape(B, H, W)
    else:
        raise ValueError(f"Unknown method: {method}")


def aggregate_image_level(unc_map_np, strategy='p90'):
    """Aggregate pixel-level uncertainty into an image-level scalar."""
    if strategy == 'p90':
        return np.percentile(unc_map_np, 90)
    elif strategy == 'mean':
        return unc_map_np.mean()
    elif strategy == 'std':
        return unc_map_np.std()
    elif strategy == 'iqr':
        return np.percentile(unc_map_np, 75) - np.percentile(unc_map_np, 25)
    elif strategy == 'max':
        return np.percentile(unc_map_np, 99)
    elif strategy == 'p95':
        return np.percentile(unc_map_np, 95)
    elif strategy == 'p75':
        return np.percentile(unc_map_np, 75)
    # Combined strategies
    elif strategy == 'mean_std_0604':
        return 0.6 * unc_map_np.mean() + 0.4 * unc_map_np.std()
    elif strategy == 'mean_std_0505':
        return 0.5 * unc_map_np.mean() + 0.5 * unc_map_np.std()
    elif strategy == 'mean_std_0703':
        return 0.7 * unc_map_np.mean() + 0.3 * unc_map_np.std()
    elif strategy == 'mean_std_0406':
        return 0.4 * unc_map_np.mean() + 0.6 * unc_map_np.std()
    elif strategy == 'mean_p90_0505':
        return 0.5 * unc_map_np.mean() + 0.5 * np.percentile(unc_map_np, 90)
    elif strategy == 'mean_p90_0604':
        return 0.6 * unc_map_np.mean() + 0.4 * np.percentile(unc_map_np, 90)
    elif strategy == 'mean_p90_0406':
        return 0.4 * unc_map_np.mean() + 0.6 * np.percentile(unc_map_np, 90)
    elif strategy == 'weighted_stats':
        return 0.4 * unc_map_np.mean() + 0.3 * unc_map_np.std() + 0.3 * np.percentile(unc_map_np, 90)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================
# Data collection
# ============================================================
def collect_all_data(model, val_loader, config):
    """Collect predictions, errors, and uncertainty scores for each image."""
    kan_layer = model.final.kan
    num_classes = config['num_classes']
    capture = FinalInputCapture(model)
    all_data = []

    print(f"  Model: in_features={kan_layer.in_features}, "
          f"grid_size={kan_layer.grid_size}, spline_order={kan_layer.spline_order}")

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Collecting data"):
            if len(batch) == 3:
                inp, tgt, _ = batch
            else:
                inp, tgt = batch

            inp, tgt = inp.cuda(), tgt.cuda()
            logits = model(inp)
            feat = capture.captured_input

            _, bases = model.final(feat, return_spline_basis=True)
            B, C_in, H, W = feat.shape

            if num_classes == 1:
                pred_prob = torch.sigmoid(logits).squeeze(1)
                error_map = (pred_prob - tgt.squeeze(1)).abs()
            else:
                pred_prob = torch.softmax(logits, dim=1)
                pred_cls = torch.argmax(logits, dim=1)
                tgt_cls = tgt.squeeze(1).long()
                error_map = (pred_cls != tgt_cls).float()

            if num_classes == 1:
                p = pred_prob
                softmax_ent = -(p * torch.log(p + 1e-10) +
                                (1 - p) * torch.log(1 - p + 1e-10))
            else:
                softmax_ent = -(pred_prob * torch.log(pred_prob + 1e-10)).sum(dim=1)

            kan_entropy = compute_basis_uncertainty(bases, 'entropy')
            kan_active = compute_basis_uncertainty(bases, 'active_count')
            kan_negconc = compute_basis_uncertainty(bases, 'neg_concentration')

            for b in range(B):
                pred_np = pred_prob[b].cpu().numpy()
                tgt_np = tgt[b].squeeze(0).cpu().numpy()
                pred_bin = (pred_np > 0.5).astype(float) if num_classes == 1 else pred_np
                tgt_bin = (tgt_np > 0.5).astype(float)

                # Per-image metrics
                dice, iou, hd95_val = compute_per_image_metrics(pred_bin, tgt_bin)

                all_data.append({
                    'dice': float(dice),
                    'iou': float(iou),
                    'hd95': float(hd95_val),
                    'mean_error': float(error_map[b].cpu().numpy().mean()),
                    'softmax_ent': softmax_ent[b].cpu().numpy(),
                    'kan_entropy': kan_entropy[b].cpu().numpy(),
                    'kan_active': kan_active[b].cpu().numpy(),
                    'kan_negconc': kan_negconc[b].cpu().numpy(),
                })

    capture.remove()
    return all_data


# ============================================================
# Core selective prediction computation
# ============================================================
def compute_selective_curve(dices, uncertainties):
    n = len(dices)
    order = np.argsort(uncertainties)
    sorted_dices = np.array(dices)[order]

    coverages = []
    performances = []

    for k in range(1, n + 1):
        cov = k / n
        perf = sorted_dices[:k].mean()
        coverages.append(cov)
        performances.append(perf)

    return np.array(coverages), np.array(performances)


def _trapz(y, x):
    """Trapezoidal integration compatible with old and new NumPy versions."""
    try:
        return np.trapezoid(y, x)
    except AttributeError:
        return np.trapz(y, x)


def compute_aurc(dices, uncertainties):
    n = len(dices)
    risks = 1.0 - np.array(dices)
    order = np.argsort(uncertainties)
    sorted_risks = risks[order]

    coverages = []
    selective_risks = []

    for k in range(1, n + 1):
        cov = k / n
        sel_risk = sorted_risks[:k].mean()
        coverages.append(cov)
        selective_risks.append(sel_risk)

    return _trapz(selective_risks, coverages)


def compute_metrics(dices, uncertainties, oracle_aurc):
    n = len(dices)
    aurc = compute_aurc(dices, uncertainties)
    e_aurc = aurc - oracle_aurc

    # 20% rejection ΔDice
    order = np.argsort(uncertainties)
    sorted_dices = np.array(dices)[order]
    k80 = int(n * 0.8)
    if k80 > 0:
        dice_at_80 = sorted_dices[:k80].mean()
    else:
        dice_at_80 = sorted_dices.mean()
    delta_dice_20 = dice_at_80 - np.mean(dices)

    # Spearman
    errors = 1.0 - np.array(dices)
    sp_corr = stats.spearmanr(uncertainties, errors)[0]

    return {
        'aurc': aurc,
        'e_aurc': e_aurc,
        'delta_dice_20': delta_dice_20,
        'spearman': sp_corr,
    }


# ============================================================
# Visualization
# ============================================================
def plot_risk_coverage_curves(all_results, dices, save_path):
    """Standard risk-coverage curve."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
    })

    colors = {
        'Oracle':          ('#27ae60', '--', 1.8, 5),
        'Ours_best':       ('#c0392b', '-',  2.8, 10),
        'TTA':             ('#2980b9', '-',  2.0, 8),
        'MC Dropout':      ('#8e44ad', '-',  2.0, 7),
        'Softmax Entropy': ('#d35400', '-',  2.0, 6),
        'Random':          ('#7f8c8d', ':',  1.3, 2),
    }

    n = len(dices)
    risks = 1.0 - np.array(dices)

    for name, data in all_results.items():
        if name not in colors:
            continue
        c, ls, lw, zo = colors[name]
        unc = data.get('uncertainties', None)
        if unc is not None:
            order = np.argsort(unc)
        elif name == 'Oracle':
            order = np.argsort(risks)
        else:
            continue

        sorted_risks = risks[order]
        coverages = np.arange(1, n + 1) / n
        sel_risks = np.cumsum(sorted_risks) / np.arange(1, n + 1)

        label = f"{name} (E-AURC={data['e_aurc']:.4f})" if name != 'Oracle' else 'Oracle'
        ax.plot(coverages, sel_risks, color=c, linestyle=ls, linewidth=lw,
                zorder=zo, label=label)

    # Random baseline: flat line
    mean_risk = risks.mean()
    ax.axhline(y=mean_risk, color='#7f8c8d', linestyle=':', linewidth=1.3,
               zorder=2, label=f'Random (risk={mean_risk:.3f})', alpha=0.7)

    ax.set_xlabel('Coverage (fraction of images retained)')
    ax.set_ylabel('Selective Risk (1 - Dice)')
    ax.set_title('Risk-Coverage Curve', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Risk-Coverage curve: {save_path}")


# ============================================================
# Main function
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='Model name')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--n_random_trials', type=int, default=100,
                        help='Number of random baseline trials')
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test: evaluate only key strategies')
    args = parser.parse_args()

    save_dir = os.path.join(args.output_dir, args.name, 'selective_prediction_v2')
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 70)
    print(f"  Selective Prediction evaluation v2 (standard metrics) | Model: {args.name}")
    print("=" * 70)

    # 1. Load
    print("\n[1/4] Loading model and data...")
    model, val_loader, config = load_model_and_data(args.name, args.output_dir)

    # 2. Collect
    print("\n[2/4] Collecting per-image data...")
    all_data = collect_all_data(model, val_loader, config)
    n = len(all_data)

    dices = np.array([d['dice'] for d in all_data])
    ious = np.array([d['iou'] for d in all_data])
    hd95s = np.array([d['hd95'] for d in all_data])
    errors = np.array([d['mean_error'] for d in all_data])

    print(f"  Number of images: {n}")
    print(f"  Mean Dice: {dices.mean():.4f} ± {dices.std():.4f}")
    print(f"  Mean IoU:  {ious.mean():.4f} ± {ious.std():.4f}")
    print(f"  Mean HD95: {hd95s.mean():.4f} ± {hd95s.std():.4f}")

    # 3. Compute metrics
    print("\n[3/4] Computing standard metrics...")

    # --- Oracle AURC ---
    oracle_aurc = compute_aurc(dices, -dices)

    # --- Uncertainty method configuration ---
    if args.quick_test:
        uncertainty_configs = {
            'active_count+mean': ('kan_active', 'mean'),
            'active_count+p90': ('kan_active', 'p90'),
            'entropy+mean': ('kan_entropy', 'mean'),
            'entropy+p90': ('kan_entropy', 'p90'),
            'softmax_ent+p90': ('softmax_ent', 'p90'),
            'softmax_ent+mean': ('softmax_ent', 'mean'),
        }
    else:
        uncertainty_configs = {
            # B-spline metrics
            'entropy+p90': ('kan_entropy', 'p90'),
            'entropy+mean': ('kan_entropy', 'mean'),
            'active_count+p90': ('kan_active', 'p90'),
            'active_count+mean': ('kan_active', 'mean'),
            'active_count+std': ('kan_active', 'std'),
            'neg_conc+p90': ('kan_negconc', 'p90'),
            # Baseline
            'softmax_ent+p90': ('softmax_ent', 'p90'),
            'softmax_ent+mean': ('softmax_ent', 'mean'),
            # Combined strategies
            'active_count+mean_std_0604': ('kan_active', 'mean_std_0604'),
            'active_count+mean_std_0505': ('kan_active', 'mean_std_0505'),
            'active_count+mean_std_0703': ('kan_active', 'mean_std_0703'),
            'active_count+mean_p90_0505': ('kan_active', 'mean_p90_0505'),
            'active_count+mean_p90_0604': ('kan_active', 'mean_p90_0604'),
            'active_count+weighted': ('kan_active', 'weighted_stats'),
        }

    results = {}

    for unc_name, (map_key, agg_strategy) in uncertainty_configs.items():
        img_uncs = np.array([
            aggregate_image_level(d[map_key], agg_strategy) for d in all_data
        ])

        metrics = compute_metrics(dices, img_uncs, oracle_aurc)
        metrics['uncertainties'] = img_uncs
        results[unc_name] = metrics

    # --- Random baseline ---
    rand_aurcs = []
    for _ in range(args.n_random_trials):
        rand_unc = np.random.randn(n)
        rand_aurcs.append(compute_aurc(dices, rand_unc))
    random_aurc = np.mean(rand_aurcs)
    random_e_aurc = random_aurc - oracle_aurc

    # 4. Output
    print("\n[4/4] Generating results...")

    lines = []
    lines.append("=" * 90)
    lines.append(f"  Selective Prediction v2 | Model: {args.name}")
    lines.append(f"  Number of images: {n} | Mean Dice: {dices.mean():.4f} ± {dices.std():.4f}")
    lines.append(f"  Mean IoU: {ious.mean():.4f} ± {ious.std():.4f} | Mean HD95: {hd95s.mean():.4f} ± {hd95s.std():.4f}")
    lines.append(f"  Oracle AURC: {oracle_aurc:.6f} | Random AURC: {random_aurc:.6f}")
    lines.append("=" * 90)

    # Sort by E-AURC
    ranked = sorted(results.items(), key=lambda x: x[1]['e_aurc'])

    lines.append(f"\n  {'Method':<35} {'AURC↓':>10} {'E-AURC↓':>10} {'20%ΔDice↑':>10} {'Spearman':>10}")
    lines.append("  " + "-" * 80)
    lines.append(f"  {'Oracle':<35} {oracle_aurc:>10.6f} {'0.000000':>10} {'—':>10} {'—':>10}")

    for name, m in ranked:
        lines.append(f"  {name:<35} {m['aurc']:>10.6f} {m['e_aurc']:>10.6f} "
                     f"{m['delta_dice_20']:>+10.4f} {m['spearman']:>10.4f}")

    lines.append(f"  {'Random':<35} {random_aurc:>10.6f} {random_e_aurc:>10.6f} {'—':>10} {'—':>10}")

    # Dice at different rejection rates
    lines.append(f"\n  Dice at different rejection rates:")
    lines.append(f"  {'Method':<35} {'0%':>8} {'5%':>8} {'10%':>8} {'20%':>8} {'30%':>8} {'50%':>8}")
    lines.append("  " + "-" * 90)

    reject_fracs = [0, 0.05, 0.1, 0.2, 0.3, 0.5]

    # Oracle
    order_oracle = np.argsort(-dices)
    sorted_dices_oracle = dices[order_oracle]
    vals = []
    for rf in reject_fracs:
        k = max(1, int(n * (1.0 - rf)))
        vals.append(f"{sorted_dices_oracle[:k].mean():.4f}")
    lines.append(f"  {'Oracle':<35} {'  '.join(vals)}")

    for name, m in ranked[:10]:
        unc = m['uncertainties']
        order = np.argsort(unc)
        sorted_d = dices[order]
        vals = []
        for rf in reject_fracs:
            k = max(1, int(n * (1.0 - rf)))
            vals.append(f"{sorted_d[:k].mean():.4f}")
        lines.append(f"  {name:<35} {'  '.join(vals)}")

    # Random
    vals = [f"{dices.mean():.4f}"] * len(reject_fracs)
    lines.append(f"  {'Random':<35} {'  '.join(vals)}")

    # Segmentation performance summary table
    lines.append(f"\n  === Segmentation performance summary (Per-image) ===")
    lines.append(f"  Dice:  {dices.mean():.4f} ± {dices.std():.4f}")
    lines.append(f"  IoU:   {ious.mean():.4f} ± {ious.std():.4f}")
    lines.append(f"  HD95:  {hd95s.mean():.4f} ± {hd95s.std():.4f}")

    lines.append("\n" + "=" * 90)

    # ============================================================
    # Comparison with the old Norm-AUSC metric for sanity checking.
    # ============================================================
    lines.append("\n  === Metric comparison for validation only; Norm-AUSC is not used in the paper. ===")
    lines.append(f"  {'Method':<35} {'E-AURC↓':>10} {'Norm-AUSC↑':>12} {'Relation check':>15}")
    lines.append("  " + "-" * 75)

    for name, m in ranked:
        cov, perf = compute_selective_curve(dices, m['uncertainties'])
        ausc = _trapz(perf, cov)
        cov_o, perf_o = compute_selective_curve(dices, -dices)
        ausc_oracle = _trapz(perf_o, cov_o)
        ausc_random = dices.mean() * 1.0
        norm_ausc = (ausc - ausc_random) / (ausc_oracle - ausc_random + 1e-10)

        e_aurc_from_ausc = ausc_oracle - ausc
        lines.append(f"  {name:<35} {m['e_aurc']:>10.6f} {norm_ausc:>12.4f} "
                     f"{'✓ match' if abs(m['e_aurc'] - e_aurc_from_ausc) < 0.001 else '✗ mismatch':>15}")

    lines.append("\n" + "=" * 90)

    report = "\n".join(lines)
    print(report)

    with open(os.path.join(save_dir, 'selective_prediction_v2_report.txt'), 'w') as f:
        f.write(report)

    # --- Save raw data ---
    save_data = {'dices': dices, 'ious': ious, 'hd95s': hd95s,
                 'errors': errors, 'oracle_aurc': oracle_aurc}
    for name, m in results.items():
        save_data[f'unc_{name}'] = m['uncertainties']
        save_data[f'aurc_{name}'] = m['aurc']
        save_data[f'e_aurc_{name}'] = m['e_aurc']
    np.savez(os.path.join(save_dir, 'selective_data_v2.npz'), **save_data)

    print(f"\n  Done. Results saved in: {save_dir}/")

    # Best method
    best_name, best_m = ranked[0]
    print(f"\n  Best method: {best_name}")
    print(f"     AURC: {best_m['aurc']:.6f}, E-AURC: {best_m['e_aurc']:.6f}")
    print(f"     20% Rej ΔDice: {best_m['delta_dice_20']:+.4f}, Spearman: {best_m['spearman']:.4f}")

    # Key numbers
    print("\n" + "=" * 70)
    print("  Key numbers for summary table")
    print("=" * 70)
    print(f"  Dice={dices.mean():.4f}  IoU={ious.mean():.4f}  HD95={hd95s.mean():.4f}")
    for name, m in ranked:
        print(f"  {name:<35} AURC={m['aurc']:.6f}  E-AURC={m['e_aurc']:.6f}  "
              f"Spearman={m['spearman']:>7.4f}  "
              f"Dice@80%={dices.mean() + m['delta_dice_20']:.4f} ({m['delta_dice_20']:+.4f})")


if __name__ == '__main__':
    main()


