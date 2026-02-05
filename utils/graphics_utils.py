"""
Graphics Utility Functions
"""

import torch
import math
import numpy as np
from typing import NamedTuple, Optional


class BasicPointCloud(NamedTuple):
    """Basic Point Cloud Data Structure"""
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    # Optional extensions for Temporal/FreeTimeGS
    t: Optional[np.ndarray] = None
    t_scale: Optional[np.ndarray] = None
    motion: Optional[np.ndarray] = None


def geom_transform_points(points, transf_matrix):
    """
    Transform points using a transformation matrix.
    
    Args:
        points: Point coordinates [P, 3]
        transf_matrix: Transformation matrix [4, 4]
    
    Returns:
        Transformed points [P, 3]
    """
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View(R, t):
    """
    Compute World-to-View transformation matrix (Simple Version).
    
    Args:
        R: Rotation matrix (3x3)
        t: Translation vector (3,)
    
    Returns:
        4x4 Transformation Matrix
    """
    Rt = np.zeros((4, 4))
    if isinstance(R, torch.Tensor):
        Rt[:3, :3] = R.T.cpu().numpy()
    else:
        Rt[:3, :3] = R.T
    if isinstance(t, torch.Tensor):
        Rt[:3, 3] = t.cpu().numpy()
    else:
        Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """
    Compute World-to-View transformation matrix with optional translation and scaling.
    
    Args:
        R: Rotation matrix (3x3)
        t: Translation vector (3,)
        translate: Additional translation
        scale: Scaling factor
    
    Returns:
        4x4 Transformation Matrix
    """
    Rt = np.zeros((4, 4))
    if isinstance(R, torch.Tensor):
        Rt[:3, :3] = R.T.cpu().numpy()
    else:
        Rt[:3, :3] = R.T
    if isinstance(t, torch.Tensor):
        Rt[:3, 3] = t.cpu().numpy()
    else:
        Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    """
    Compute Projection Matrix.
    
    Args:
        znear: Near plane distance
        zfar: Far plane distance
        fovX: Field of view X
        fovY: Field of view Y
    
    Returns:
        4x4 Projection Matrix
    """
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    """Convert FOV to Focal Length."""
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    """Convert Focal Length to FOV."""
    return 2 * math.atan(pixels / (2 * focal))
