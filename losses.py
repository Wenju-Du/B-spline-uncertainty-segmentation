import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = [
    'BCEDiceLoss', 'UncertaintyBCEDiceLoss', 'LovaszHingeLoss',
    'EDLDiceLoss', 'edl_inference',
]


# ============================================================
# Shared EDL utilities used by training, evaluation, and model code.
# ============================================================

def _kl_dirichlet_uniform(alpha):
    """KL( Dir(alpha) || Dir(1,...,1) )"""
    ones = torch.ones_like(alpha)
    S_a = alpha.sum(dim=1, keepdim=True)
    S_1 = ones.sum(dim=1, keepdim=True)

    kl = (torch.lgamma(S_a) - torch.lgamma(S_1)
          - (torch.lgamma(alpha) - torch.lgamma(ones)).sum(dim=1, keepdim=True)
          + ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S_a))).sum(dim=1, keepdim=True))
    return kl.squeeze(1)


def edl_inference(logits):
    """
    Extract prediction and uncertainty from EDL model outputs.

    Args:
        logits: (B, 2, H, W)
    Returns:
        dict with keys:
            'pred':       (B, 1, H, W) binary prediction
            'prob':       (B, 1, H, W) foreground probability
            'vacuity':    (B, 1, H, W) epistemic uncertainty = K/S
            'entropy':    (B, 1, H, W) predictive entropy
            'evidence':   (B, 2, H, W) raw evidence
    """
    evidence = F.relu(logits)
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)

    p_fg = alpha[:, 1:2] / S
    pred = (p_fg > 0.5).float()

    vacuity = 2.0 / S
    p = alpha / S
    entropy = -(p * torch.log(p + 1e-8)).sum(dim=1, keepdim=True)

    return {
        'pred': pred,
        'prob': p_fg,
        'vacuity': vacuity,
        'entropy': entropy,
        'evidence': evidence,
    }


# ============================================================
# Loss Functions
# ============================================================

class BCEDiceLoss(nn.Module):
    """Original loss."""
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class UncertaintyBCEDiceLoss(nn.Module):
    """
    BCEDiceLoss with uncertainty feedback.
    """
    def __init__(
            self,
            uncertainty_weight=0.1,
            calibration_weight=0.05,
            bce_weight=0.5,
    ):
        super().__init__()
        self.uncertainty_weight = uncertainty_weight
        self.calibration_weight = calibration_weight
        self.bce_weight = bce_weight
        self.smooth = 1e-5

    def forward(self, input, target, uncertainty=None):
        num = target.size(0)
        if uncertainty is not None:
            bce_unreduced = F.binary_cross_entropy_with_logits(input, target, reduction='none')
            unc_norm = torch.sigmoid(uncertainty)
            with torch.no_grad():
                pred_prob = torch.sigmoid(input)
                error_map = torch.abs(pred_prob - target)
            unc_norm = torch.sigmoid(uncertainty)
            weights = 1.0 + self.uncertainty_weight * unc_norm * error_map
            bce = (bce_unreduced * weights).mean()
        else:
            bce = F.binary_cross_entropy_with_logits(input, target)
        input_sigmoid = torch.sigmoid(input)
        input_flat = input_sigmoid.view(num, -1)
        target_flat = target.view(num, -1)
        intersection = (input_flat * target_flat)
        dice = (2. * intersection.sum(1) + self.smooth) / (input_flat.sum(1) + target_flat.sum(1) + self.smooth)
        dice_loss = 1 - dice.sum() / num
        base_loss = self.bce_weight * bce + dice_loss
        if uncertainty is not None:
            with torch.no_grad():
                pred = (input_sigmoid > 0.5).float()
                is_wrong = (pred != target).float()
            unc_norm = torch.sigmoid(uncertainty)
            calibration_loss = F.mse_loss(unc_norm, is_wrong)
            total_loss = base_loss + self.calibration_weight * calibration_loss
            return {
                'total': total_loss,
                'bce': bce,
                'dice': dice_loss,
                'calibration': calibration_loss,
                'base': base_loss
            }
        return base_loss


class EDLDiceLoss(nn.Module):
    """
    Evidential Deep Learning + Dice Loss
    Reference: Sensoy et al., "Evidential Deep Learning to Quantify Classification
          Uncertainty", NeurIPS 2018
    Code reference: https://github.com/dougbrion/pytorch-classification-uncertainty

    Input: logits (B, 2, H, W) — two-channel raw model output
    Target: target (B, 1, H, W) — binary mask, {0, 1}

    All hyperparameters use standard community defaults and do not require tuning.
    """

    def __init__(
            self,
            edl_weight=1.0,
            dice_weight=1.0,
            kl_weight_max=0.01,
            annealing_epochs=50,
            total_epochs=200,
    ):
        super().__init__()
        self.edl_weight = edl_weight
        self.dice_weight = dice_weight
        self.kl_weight_max = kl_weight_max
        self.annealing_epochs = annealing_epochs
        self.total_epochs = total_epochs

    def forward(self, logits, target, epoch=0):
        """
        Args:
            logits: (B, 2, H, W) raw model output without activation
            target: (B, 1, H, W) binary GT mask
            epoch:  current epoch (0-indexed), for KL annealing
        """
        loss_edl = self._edl_digamma_loss(logits, target, epoch)
        loss_dice = self._dice_loss(logits, target)
        return self.edl_weight * loss_edl + self.dice_weight * loss_dice

    def _edl_digamma_loss(self, logits, target, epoch):
        """Expected CE (digamma form) + KL regularization with annealing."""
        # one-hot: (B, 2, H, W)
        target_oh = torch.zeros_like(logits)
        target_oh.scatter_(1, target.long(), 1)

        # evidence -> Dirichlet parameters
        evidence = F.relu(logits)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)

        # Expected Cross-Entropy
        ece = (target_oh * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)

        # KL regularization: penalize evidence assigned to incorrect classes only
        alpha_tilde = target_oh + (1 - target_oh) * (alpha - 1) + 1
        kl = _kl_dirichlet_uniform(alpha_tilde)

        # KL annealing
        annealing_step = min(self.annealing_epochs, self.total_epochs // 4)
        annealing = min(1.0, epoch / annealing_step) if annealing_step > 0 else 1.0
        kl_weight = self.kl_weight_max * annealing

        return (ece + kl_weight * kl).mean()

    def _dice_loss(self, logits, target):
        """Dice loss on expected foreground probability from Dirichlet."""
        evidence = F.relu(logits)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        p_fg = alpha[:, 1:2] / S  # foreground probability (B, 1, H, W)

        target_f = target.float()
        smooth = 1.0
        num = target.size(0)
        p_flat = p_fg.view(num, -1)
        t_flat = target_f.view(num, -1)
        intersection = (p_flat * t_flat).sum(1)
        union = p_flat.sum(1) + t_flat.sum(1)
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)
        return loss


