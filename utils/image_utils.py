"""
Image Utility Functions
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple


def PILtoTorch(pil_image: Image.Image, resolution: Tuple[int, int]) -> torch.Tensor:
    """
    Convert PIL Image to PyTorch Tensor.
    
    Args:
        pil_image: PIL Image
        resolution: Target resolution (width, height)
    
    Returns:
        Tensor of shape [C, H, W], value range [0, 1]
    """
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        img1: First image, shape [C, H, W]
        img2: Second image, shape [C, H, W]
        max_val: Maximum pixel value (usually 1.0 or 255.0)
    
    Returns:
        PSNR value (float)
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()
