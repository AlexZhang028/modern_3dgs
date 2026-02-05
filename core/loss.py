"""
Loss Functions for Gaussian Splatting

Supported Losses:
- L1 Loss
- L2 Loss
- SSIM Loss (structural similarity)
- Combined Loss (L1 + λ * (1 - SSIM))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict
from math import exp


try:
    from fused_ssim import fusedssim, fusedssim_backward
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False
    pass


# SSIM Constants
C1 = 0.01 ** 2
C2 = 0.03 ** 2


class FusedSSIMMap(torch.autograd.Function):
    """
    Fused SSIM Implementation (if available).
    
    Uses C++/CUDA implementation for faster SSIM compared to pure PyTorch.
    """
    
    @staticmethod
    def forward(ctx, C1: float, C2: float, img1: torch.Tensor, img2: torch.Tensor):
        ctx.C1 = C1
        ctx.C2 = C2
        
        try:
            # Try original signature (4 args)
            ssim_map = fusedssim(C1, C2, img1, img2)
            ctx.save_for_backward(img1.detach(), img2)
            return ssim_map
            
        except TypeError:
            # Standalone version (needs 4D inputs, returns tuple, needs auxiliaries)
            is_3d = (img1.dim() == 3)
            if is_3d:
                img1_in = img1.unsqueeze(0)
                img2_in = img2.unsqueeze(0)
            else:
                img1_in = img1
                img2_in = img2
            
            # Forward pass with train=True
            result_tuple = fusedssim(C1, C2, img1_in, img2_in, True)
            ssim_map = result_tuple[0]
            dm_dmu1 = result_tuple[1]
            dm_dsigma1_sq = result_tuple[2]
            dm_dsigma12 = result_tuple[3]
            
            if is_3d:
                ssim_map = ssim_map.squeeze(0)
            
            # Save tensors (len=5 indicates standalone mode)
            ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
            return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        saved = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        
        if len(saved) == 2:
            # Standard path
            img1, img2 = saved
            grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        else:
            # Standalone path (8 args)
            img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = saved
            
            # Ensure 4D inputs
            is_3d = (img1.dim() == 3)
            if is_3d:
                img1_in = img1.unsqueeze(0)
                img2_in = img2.unsqueeze(0)
                opt_grad_in = opt_grad.unsqueeze(0)
            else:
                img1_in = img1
                img2_in = img2
                opt_grad_in = opt_grad
            
            grad = fusedssim_backward(
                C1, C2, 
                img1_in, img2_in, 
                opt_grad_in, 
                dm_dmu1, dm_dsigma1_sq, dm_dsigma12
            )
            
            if is_3d:
                grad = grad.squeeze(0)
                
        return None, None, grad, None


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L1 Loss (Mean Absolute Error).
    
    Args:
        pred: Predicted image [B, C, H, W] or [C, H, W].
        target: Target image, same shape as pred.
        
    Returns:
        Scalar loss.
    """
    return torch.abs(pred - target).mean()


def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L2 Loss (Mean Squared Error).
    
    Args:
        pred: Predicted image [B, C, H, W] or [C, H, W].
        target: Target image, same shape as pred.
        
    Returns:
        Scalar loss.
    """
    return ((pred - target) ** 2).mean()


def gaussian_kernel(window_size: int, sigma: float = 1.5) -> torch.Tensor:
    """
    Create 1D Gaussian kernel.
    
    Args:
        window_size: Window size.
        sigma: Standard deviation.
        
    Returns:
        Normalized Gaussian kernel [window_size].
    """
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) 
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_ssim_window(window_size: int, channel: int) -> torch.Tensor:
    """
    Create 2D Gaussian window for SSIM.
    
    Args:
        window_size: Window size (usually 11).
        channel: Number of channels (usually 3).
        
    Returns:
        2D Gaussian window [channel, 1, window_size, window_size].
    """
    _1D_window = gaussian_kernel(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(
    img1: torch.Tensor, 
    img2: torch.Tensor, 
    window_size: int = 11,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM).
    
    Args:
        img1: First image [B, C, H, W] or [C, H, W].
        img2: Second image, same shape as img1.
        window_size: SSIM window size (default 11).
        reduction: 'mean' or 'none'.
        
    Returns:
        SSIM value, range [0, 1], 1 means identical.
    """
    channel = img1.size(-3)
    window = create_ssim_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _compute_ssim(img1, img2, window, window_size, channel, reduction)


def _compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Core function for SSIM computation.
    """
    # Calculate local means
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Calculate local variances and covariance
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if reduction == 'mean':
        return ssim_map.mean()
    else:
        return ssim_map


def fast_ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Fast SSIM (using Fused CUDA implementation).
    
    If Fused SSIM is not available, fall back to standard implementation.
    
    Args:
        img1: First image [C, H, W].
        img2: Second image [C, H, W].
        
    Returns:
        SSIM value.
    """
    if FUSED_SSIM_AVAILABLE:
        ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
        return ssim_map.mean()
    else:
        return ssim(img1, img2)


class GaussianLoss(nn.Module):
    """
    Combined Loss for Gaussian Splatting.
    
    Loss = (1 - lambda) * L1 + lambda * (1 - SSIM)
    
    Args:
        lambda_dssim: Weight for SSIM loss, default 0.2.
        use_fused_ssim: Whether to use Fused SSIM (if available).
    """
    
    def __init__(self, lambda_dssim: float = 0.2, use_fused_ssim: bool = True):
        super().__init__()
        self.lambda_dssim = lambda_dssim
        self.use_fused_ssim = use_fused_ssim and FUSED_SSIM_AVAILABLE
        
        if not FUSED_SSIM_AVAILABLE and use_fused_ssim:
            print("Warning: Fused SSIM not available, using standard SSIM implementation.")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted image [C, H, W].
            target: Target image [C, H, W].
            
        Returns:
            Combined loss.
        """
        # L1 Loss
        l1 = l1_loss(pred, target)
        
        # SSIM Loss
        if self.use_fused_ssim:
            ssim_val = fast_ssim(pred, target)
        else:
            ssim_val = ssim(pred, target)
        
        # Combine
        loss = (1.0 - self.lambda_dssim) * l1 + self.lambda_dssim * (1.0 - ssim_val)
        
        return loss
    
    def get_components(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Return individual loss components (for logging).
        
        Args:
            pred: Predicted image [C, H, W].
            target: Target image [C, H, W].
            
        Returns:
            Dictionary containing loss components.
        """
        l1 = l1_loss(pred, target)
        
        if self.use_fused_ssim:
            ssim_val = fast_ssim(pred, target)
        else:
            ssim_val = ssim(pred, target)
        
        total_loss = (1.0 - self.lambda_dssim) * l1 + self.lambda_dssim * (1.0 - ssim_val)
        
        return {
            'total': total_loss,
            'l1': l1,
            'ssim': ssim_val,
            'dssim': 1.0 - ssim_val
        }


def compute_loss(
    pred: torch.Tensor, 
    target: torch.Tensor,
    lambda_dssim: float = 0.2,
    use_fused: bool = True
) -> torch.Tensor:
    """
    Convenience function to compute Gaussian Splatting loss.
    
    Args:
        pred: Predicted image.
        target: Target image.
        lambda_dssim: SSIM weight.
        use_fused: Whether to use Fused SSIM.
        
    Returns:
        Total loss.
    """
    loss_fn = GaussianLoss(lambda_dssim, use_fused)
    return loss_fn(pred, target)
