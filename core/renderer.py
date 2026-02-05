"""
Gaussian Splatting Renderer

Unified rendering interface, supporting both static and freetime modes.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Union, Any, Tuple

try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except ImportError:
    print("Warning: diff_gaussian_rasterization not found. Rendering will not work.")
    GaussianRasterizationSettings = None
    GaussianRasterizer = None

from core.gaussian_model import GaussianModel
from data.camera import Camera
from utils.sh_utils import eval_sh
from config.config import PipelineConfig


class GaussianRenderer(nn.Module):
    """
    Unified Gaussian Splatting Renderer.
    
    Supports two rendering modes:
    1. Static: Standard 3D Gaussian Splatting.
    2. FreeTimeGS: Temporal dynamic scene rendering.
    
    Usage:
        >>> renderer = GaussianRenderer(config)
        >>> 
        >>> # Static Rendering
        >>> output = renderer(model, camera, bg_color)
        >>> 
        >>> # Dynamic Rendering
        >>> output = renderer(model, camera, bg_color, timestamp=0.5)
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        super().__init__()
        
        self.config = config or PipelineConfig()
        
        if GaussianRasterizer is None:
            raise ImportError(
                "diff_gaussian_rasterization not found. "
                "Please install it from submodules/diff-gaussian-rasterization"
            )
    
    def _setup_rasterizer(
        self,
        camera: Camera,
        bg_color: torch.Tensor,
        scaling_modifier: float = 1.0,
        sh_degree: Optional[int] = None
    ) -> Any:
        """
        Setup rasterizer configuration.
        
        Common part for static and dynamic rendering.
        """
        # Calculate tan of FOV
        tanfovx = math.tan(camera.FovX * 0.5)
        tanfovy = math.tan(camera.FovY * 0.5)
        
        # Create rasterization settings
        raster_settings = GaussianRasterizationSettings(
            image_height=int(camera.height),
            image_width=int(camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=sh_degree or 0,  # Use 0 if not specified
            campos=camera.camera_center,
            prefiltered=False,
            debug=self.config.debug,
            antialiasing=self.config.antialiasing
        )
        
        return GaussianRasterizer(raster_settings=raster_settings)
    
    def _prepare_colors(
        self,
        gaussians: GaussianModel,
        camera: Camera,
        xyz: torch.Tensor,
        colors_override: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Prepare color data (SH or precomputed RGB).
        
        Returns:
            (shs, colors_precomp): One of them will be None.
        """
        shs = None
        colors_precomp = None
        
        if colors_override is not None:
            # Use external colors
            colors_precomp = colors_override
        elif self.config.convert_SHs_python:
            # Convert SH to RGB in Python
            features = gaussians.get_features  # [N, K, 3]
            shs_view = features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
            
            # Calculate view direction
            dir_pp = xyz - camera.camera_center.repeat(xyz.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            
            # SH to RGB
            sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # Let rasterizer handle SH
            shs = gaussians.get_features
        
        return shs, colors_precomp
    
    def _rasterize(
        self,
        rasterizer: Any,
        xyz: torch.Tensor,
        screenspace_points: torch.Tensor,
        opacity: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        shs: Optional[torch.Tensor],
        colors_precomp: Optional[torch.Tensor],
        cov3D_precomp: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute rasterization (common part).
        
        Args:
            screenspace_points: Must be created externally and passed in to maintain gradient links.
        
        Returns:
            (rendered_image, radii, depth_image)
        """
        # Handle empty scene (e.g., all points culled)
        if xyz.shape[0] == 0:
            # Avoid calling CUDA rasterizer with empty tensors to prevent backward pass errors
            # (e.g. invalid gradient shapes for SHs when num_points=0 but sh_degree>0)
            settings = rasterizer.raster_settings
            
            # Create dummy background output
            # Ensure it's on the same device and has compatible shape
            rendered_image = settings.bg.view(3, 1, 1).repeat(1, settings.image_height, settings.image_width)
            
            # Radii is 1D tensor [N] -> [0]
            radii = torch.empty(0, device=xyz.device, dtype=xyz.dtype)
            
            # Depth image is [1, H, W] (assuming rasterizer returns depth)
            depth_image = torch.zeros(1, settings.image_height, settings.image_width, device=xyz.device, dtype=xyz.dtype)
            
            return rendered_image, radii, depth_image

        # Call CUDA rasterizer
        rendered_image, radii, depth_image = rasterizer(
            means3D=xyz,
            means2D=screenspace_points,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp
        )
        
        return rendered_image, radii, depth_image
    
    def render_static(
        self,
        gaussians: GaussianModel,
        camera: Camera,
        bg_color: torch.Tensor,
        scaling_modifier: float = 1.0,
        colors_override: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Static Scene Rendering (Standard 3DGS).
        
        Args:
            gaussians: GaussianModel (static mode).
            camera: Camera.
            bg_color: Background color [3] (must be on GPU).
            scaling_modifier: Scale modifier.
            colors_override: Optional color override [N, 3].
            
        Returns:
            Dict containing:
                - render: Rendered image [3, H, W].
                - depth: Depth map [H, W].
                - viewspace_points: Points in view space.
                - visibility_filter: Visible Gaussians indices.
                - radii: Projected radii.
        """
        # Get Gaussian parameters
        xyz = gaussians.get_xyz
        opacity = gaussians.get_opacity
        scales = gaussians.get_scaling
        
        # Create screenspace points (for gradient tracking) - must be created here and reference kept!
        screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        # Setup rasterizer
        rasterizer = self._setup_rasterizer(
            camera, bg_color, scaling_modifier, gaussians.active_sh_degree
        )
        
        # Prepare scaling and rotation
        if self.config.compute_cov3D_python:
            scales = None
            rotations = None
            cov3D_precomp = gaussians.get_covariance(scaling_modifier)
        else:
            scales = gaussians.get_scaling
            rotations = gaussians.get_rotation
            cov3D_precomp = None
        
        # Prepare colors
        shs, colors_precomp = self._prepare_colors(gaussians, camera, xyz, colors_override)
        
        # Rasterize
        rendered_image, radii, depth_image = self._rasterize(
            rasterizer, xyz, screenspace_points, opacity, scales, rotations, 
            shs, colors_precomp, cov3D_precomp
        )
        
        return {
            'render': rendered_image,
            'depth': depth_image,
            'viewspace_points': screenspace_points,
            'visibility_filter': radii > 0,
            'radii': radii
        }
    
    def render_temporal(
        self,
        gaussians: GaussianModel,
        camera: Camera,
        bg_color: torch.Tensor,
        timestamp: float,
        scaling_modifier: float = 1.0,
        opacity_threshold: float = 0.001,
        enable_culling: bool = True,
        colors_override: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Temporal Scene Rendering (FreeTimeGS).
        
        Args:
            gaussians: GaussianModel (freetime mode).
            camera: Camera.
            bg_color: Background color [3] (must be on GPU).
            timestamp: Current rendering time (0.0-1.0).
            scaling_modifier: Scale modifier.
            opacity_threshold: Opacity threshold for culling.
            enable_culling: Whether to enable culling.
            colors_override: Optional color override [N, 3].
            
        Returns:
            Dict containing:
                - render: Rendered image [3, H, W].
                - depth: Depth map [H, W].
                - viewspace_points: Points in view space.
                - visibility_filter: Visible Gaussians indices.
                - radii: Projected radii.
                - temporal_info: Temporal transformation info.
        """
        if gaussians.mode != "freetime":
            raise ValueError(f"render_temporal requires freetime mode, current mode: {gaussians.mode}")
        
        # 1. Time transformation
        transform_result = gaussians.get_at_time(timestamp, opacity_threshold)
        
        xyz_at_t = transform_result['xyz_at_t']
        opacity_at_t = transform_result['opacity_at_t']
        mask = transform_result['mask']
        
        # 2. Apply culling
        if enable_culling:
            # Handle empty scene (all culled)
            if mask.sum() == 0:
                # Create dummy output with gradient connection to prevent "no grad" error
                print("   All Gaussians culled at timestamp:", timestamp)
                dummy_connector = (
                    gaussians.get_xyz.sum() * 0 + 
                    gaussians.get_opacity.sum() * 0 + 
                    gaussians.get_scaling.sum() * 0 +
                    gaussians.get_rotation.sum() * 0
                )
                
                rendered_image = bg_color.view(3, 1, 1).repeat(1, int(camera.height), int(camera.width)) + dummy_connector
                depth_image = torch.zeros(1, int(camera.height), int(camera.width), device="cuda") + dummy_connector.mean()
                
                return {
                    'render': rendered_image,
                    'depth': depth_image,
                    'viewspace_points': torch.zeros((0, 3), device="cuda", requires_grad=True),
                    'visibility_filter': torch.zeros(0, dtype=torch.bool, device="cuda"),
                    'radii': torch.zeros(0, device="cuda"),
                    'opacity_at_t': torch.zeros(0, device="cuda"),
                    'temporal_weight': torch.zeros(0, device="cuda"),
                    'base_opacity': torch.zeros(0, device="cuda"),
                    'culled_mask': mask,
                    'temporal_info': {
                        'timestamp': timestamp,
                        'total_gaussians': gaussians.num_points,
                        'visible_gaussians': 0,
                        'temporal_weight_range': [0.0, 0.0]
                    }
                }

            xyz_at_t = xyz_at_t[mask]
            opacity_at_t = opacity_at_t[mask]
            
            # Cull other parameters
            scales = gaussians.get_scaling[mask]
            rotations = gaussians.get_rotation[mask]
            features = gaussians.get_features[mask]
        else:
            scales = gaussians.get_scaling
            rotations = gaussians.get_rotation
            features = gaussians.get_features
        
        # 3. Create screenspace points (for gradient tracking) - Must be created here!
        screenspace_points = torch.zeros_like(xyz_at_t, dtype=xyz_at_t.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        # 4. Setup rasterizer
        rasterizer = self._setup_rasterizer(
            camera, bg_color, scaling_modifier, gaussians.active_sh_degree
        )
        
        # 5. Prepare colors (using culled features)
        if colors_override is not None and enable_culling:
            colors_override = colors_override[mask]
        
        shs = None
        colors_precomp = None
        
        if colors_override is not None:
            colors_precomp = colors_override
        elif self.config.convert_SHs_python:
            shs_view = features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
            dir_pp = xyz_at_t - camera.camera_center.repeat(xyz_at_t.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = features
        
        # 6. Rasterize (using transformed parameters)
        cov3D_precomp = None
        if self.config.compute_cov3D_python:
            # Need to implement culled covariance calculation
            raise NotImplementedError("FreeTimeGS does not support Python covariance calculation yet")
        
        rendered_image, radii, depth_image = self._rasterize(
            rasterizer, xyz_at_t, screenspace_points, opacity_at_t, scales, rotations,
            shs, colors_precomp, cov3D_precomp
        )
        
        # Clamp to [0, 1] (Match Original 3DGS behavior)
        rendered_image = rendered_image.clamp(0, 1)
        
        return {
            'render': rendered_image,
            'depth': depth_image,
            'viewspace_points': screenspace_points,
            'visibility_filter': radii > 0,
            'radii': radii,
            'opacity_at_t': transform_result['opacity_at_t'] if not enable_culling else opacity_at_t, 
            'temporal_weight': transform_result['temporal_weight'] if not enable_culling else (transform_result['temporal_weight'][mask]),
            'base_opacity': gaussians.get_opacity if not enable_culling else (gaussians.get_opacity[mask]),
            'culled_mask': mask if enable_culling else None, 
            'temporal_info': {
                'timestamp': timestamp,
                'total_gaussians': gaussians.num_points,
                'visible_gaussians': mask.sum().item() if enable_culling else gaussians.num_points,
                'temporal_weight_range': [
                    transform_result['temporal_weight'].min().item(),
                    transform_result['temporal_weight'].max().item()
                ]
            }
        }
    
    def forward(
        self,
        gaussians: GaussianModel,
        camera: Camera,
        bg_color: torch.Tensor,
        timestamp: Optional[float] = None,
        scaling_modifier: float = 1.0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Unified Rendering Interface (Auto-selects rendering mode).
        
        Args:
            gaussians: GaussianModel.
            camera: Camera.
            bg_color: Background color [3].
            timestamp: Optional timestamp (triggers FreeTimeGS rendering).
            scaling_modifier: Scale modifier.
            **kwargs: Arguments passed to specific render methods.
            
        Returns:
            Rendering result dictionary.
            
        Auto-selection logic:
            - If gaussians.mode == "freetime" and timestamp is not None -> render_temporal()
            - Else -> render_static()
        """
        # Ensure bg_color is on GPU
        if not bg_color.is_cuda:
            bg_color = bg_color.cuda()
        
        # Auto-select rendering mode
        if gaussians.mode == "freetime" and timestamp is not None:
            return self.render_temporal(
                gaussians, camera, bg_color, timestamp, scaling_modifier, **kwargs
            )
        else:
            return self.render_static(
                gaussians, camera, bg_color, scaling_modifier, **kwargs
            )
