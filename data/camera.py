"""
Camera Class - Stores camera parameters and image data.
"""

import torch
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional
from utils import getWorld2View2, getProjectionMatrix


def compute_fov_from_aspect(fov: float, aspect_ratio: float, is_horizontal: bool = True) -> float:
    """
    Compute the FOV in one direction given the FOV in the other direction and the aspect ratio.
    
    Args:
        fov: Known FOV (radians)
        aspect_ratio: Width / Height
        is_horizontal: True if fov is FovX, False if FovY
        
    Returns:
        Calculated FOV (radians)
        
    Theory:
        tan(FovY/2) / tan(FovX/2) = height / width = 1 / aspect_ratio
    """
    tan_half_fov = math.tan(fov * 0.5)
    
    if is_horizontal:
        # Known FovX, computing FovY
        tan_half_fov_y = tan_half_fov / aspect_ratio
        return 2.0 * math.atan(tan_half_fov_y)
    else:
        # Known FovY, computing FovX
        tan_half_fov_x = tan_half_fov * aspect_ratio
        return 2.0 * math.atan(tan_half_fov_x)


@dataclass
class Camera:
    """
    Camera Data Class.
    
    Modernized version with dataclass, lazy loading support, clearer naming, and automatic FOV calculation.
    
    FOV Calculation:
        You can provide either FovX or FovY, and the other will be automatically calculated based on the aspect ratio.
        If both are provided, the provided values are used.
    """
    
    # Basic Info
    uid: int
    image_name: str
    
    # Camera Parameters
    R: np.ndarray  # Rotation Matrix (3x3)
    T: np.ndarray  # Translation Vector (3,)
    width: int
    height: int
    
    # FOV (At least one must be provided)
    FovX: Optional[float] = None  # Horizontal FOV (radians)
    FovY: Optional[float] = None  # Vertical FOV (radians)
    
    # Image Data
    image: Optional[torch.Tensor] = None  # [3, H, W], range [0, 1]
    alpha_mask: Optional[torch.Tensor] = None  # [1, H, W]
    
    # Depth Info (Optional)
    depth_map: Optional[torch.Tensor] = None  # [1, H, W]
    depth_mask: Optional[torch.Tensor] = None  # [1, H, W]
    depth_reliable: bool = False
    
    # Time Info (FreeTimeGS)
    timestamp: Optional[float] = None
    timestamp_seconds: Optional[float] = None # Real time in seconds
    
    @property
    def time(self) -> Optional[float]:
        """Alias for timestamp to match FreeTimeGS paper/doc semantics."""
        return self.timestamp

    # Scene Normalization
    trans: np.ndarray = None
    scale: float = 1.0
    
    # Camera Transforms (Lazily Computed)
    world_view_transform: Optional[torch.Tensor] = None
    projection_matrix: Optional[torch.Tensor] = None
    full_proj_transform: Optional[torch.Tensor] = None
    camera_center: Optional[torch.Tensor] = None
    
    # Others
    znear: float = 0.01
    zfar: float = 100.0
    colmap_id: int = -1
    
    def __post_init__(self):
        """Compute FOV and transforms after initialization."""
        if self.trans is None:
            self.trans = np.array([0.0, 0.0, 0.0])
        
        # Auto-compute missing FOV
        if self.FovX is None and self.FovY is None:
            raise ValueError("Must provide at least one of FovX or FovY")
        
        aspect_ratio = self.width / self.height
        
        if self.FovX is None:
            # Compute FovX from FovY
            self.FovX = compute_fov_from_aspect(self.FovY, aspect_ratio, is_horizontal=False)
        elif self.FovY is None:
            # Compute FovY from FovX
            self.FovY = compute_fov_from_aspect(self.FovX, aspect_ratio, is_horizontal=True)
        # If both are provided, keep them as is
        
        # Compute camera transforms
        self.compute_transforms()
    
    def compute_transforms(self):
        """Compute camera transformation matrices."""
        # World to View Transform
        w2v = getWorld2View2(self.R, self.T, self.trans, self.scale)
        self.world_view_transform = torch.tensor(w2v).transpose(0, 1)
        
        # Projection Matrix
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, 
            zfar=self.zfar, 
            fovX=self.FovX, 
            fovY=self.FovY
        ).transpose(0, 1)
        
        # Full Projection Transform
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        
        # Camera Center (World Coordinates)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def to(self, device: torch.device):
        """Move camera data to specified device."""
        # Move image data
        if self.image is not None:
            self.image = self.image.to(device)
        if self.alpha_mask is not None:
            self.alpha_mask = self.alpha_mask.to(device)
        if self.depth_map is not None:
            self.depth_map = self.depth_map.to(device)
        if self.depth_mask is not None:
            self.depth_mask = self.depth_mask.to(device)
        
        # Move transforms
        if self.world_view_transform is not None:
            self.world_view_transform = self.world_view_transform.to(device)
        if self.projection_matrix is not None:
            self.projection_matrix = self.projection_matrix.to(device)
        if self.full_proj_transform is not None:
            self.full_proj_transform = self.full_proj_transform.to(device)
        if self.camera_center is not None:
            self.camera_center = self.camera_center.to(device)
        
        self.world_view_transform = self.world_view_transform.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        
        return self
    
    @property
    def image_width(self):
        """Compatibility property."""
        return self.width
    
    @property
    def image_height(self):
        """Compatibility property."""
        return self.height
    
    @property
    def original_image(self):
        """Compatibility property."""
        return self.image
