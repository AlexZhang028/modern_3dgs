"""
Adaptive Density Control for Gaussian Splatting

Implements adaptive density control algorithms:
- Clone: Duplicate small Gaussians.
- Split: Split large Gaussians.
- Prune: Remove Gaussians with low opacity or excessive size.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from config.config import DensificationConfig


class GaussianDensifier:
    """
    Manager for Gaussian Densification.
    
    Responsibilities:
    - Track gradient statistics.
    - Clone small Gaussians.
    - Split large Gaussians.
    - Prune low-quality Gaussians.
    - Update optimizer state.
    """
    
    def __init__(
        self,
        config: DensificationConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer
    ):
        """
        Initialize the GaussianDensifier.

        Args:
            config: Densification configuration.
            model: GaussianModel instance.
            optimizer: Optimizer (must support adding/removing parameters).
        """
        self.config = config
        self.model = model
        self.optimizer = optimizer
        
        # Gradient statistics (on model.device)
        device = model.device
        self.xyz_gradient_accum = torch.zeros((model.num_points, 1), device=device)
        self.denom = torch.zeros((model.num_points, 1), device=device)
        
        # Statistics
        self.max_radii2D = torch.zeros((model.num_points), device=device)
    
    def update_stats(self, viewspace_point_tensor: torch.Tensor, visibility_filter: torch.Tensor, radii: torch.Tensor):
        """
        Update statistics called at each iteration.

        Args:
            viewspace_point_tensor: Projected points in view space.
            visibility_filter: Visibility mask.
            radii: Radii of the projected points.
        """
        # Update gradient statistics
        self.add_densification_stats(viewspace_point_tensor, visibility_filter)
        
        # Update max radii statistics
        self.update_max_radii2D(radii, visibility_filter)

    def add_densification_stats(self, viewspace_point_tensor: torch.Tensor, update_filter: torch.Tensor):
        """
        Accumulate gradient statistics.
        
        Args:
            viewspace_point_tensor: Projected points in view space [N, 3].
            update_filter: Boolean mask for valid points [N].
        """
        if viewspace_point_tensor.grad is None:
            return
        
        # Ensure update_filter matches gradient shape
        grad_norm = torch.norm(viewspace_point_tensor.grad[:, :2], dim=-1, keepdim=True)
        
        # Note: update_filter is a boolean mask, used for direct indexing update
        # Ensure self.xyz_gradient_accum length matches update_filter
        if self.xyz_gradient_accum.shape[0] != update_filter.shape[0]:
             # Protection against shape mismatch due to async operations after densify
             return

        self.xyz_gradient_accum[update_filter] += grad_norm[update_filter]
        self.denom[update_filter] += 1.0
    
    def update_max_radii2D(self, radii: torch.Tensor, visibility_filter: torch.Tensor):
        """
        Update maximum 2D radii statistics.
        
        Args:
            radii: Current rendered radii [N].
            visibility_filter: Visibility filter [N].
        """
        self.max_radii2D[visibility_filter] = torch.max(
            self.max_radii2D[visibility_filter],
            radii[visibility_filter]
        )
    
    def densify_and_clone(self, grads: torch.Tensor, grad_threshold: float, scene_extent: float) -> int:
        """
        Clone small Gaussians with high gradients.
        
        Args:
            grads: Average gradients [N, 1].
            grad_threshold: Gradient threshold.
            scene_extent: Scene extent.

        Returns:
            Number of cloned Gaussians.
        """
        # Find Gaussians to clone
        # Condition: High gradient AND small size
        device = self.model.device
        
        # --- Motion-Guided Densification ---
        if hasattr(self.model, 'get_motion') and self.model.get_motion is not None:
            velocities = self.model.get_motion
            speed = torch.norm(velocities, dim=-1)
            # Shield against NaN velocities which would poison the threshold and block all cloning
            speed = torch.nan_to_num(speed, nan=0.0)
            speed_factor = torch.clamp(speed / 2.0, 0.0, 1.0)
            dynamic_threshold = grad_threshold * (1.0 - 0.75 * speed_factor)
            selected_pts_mask = (torch.norm(grads, dim=-1) >= dynamic_threshold).to(device)
        else:
            selected_pts_mask = (torch.norm(grads, dim=-1) >= grad_threshold).to(device)
        
        # Get Gaussian sizes (max scale in 3D space)
        scales = self.model.get_scaling
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            (torch.max(scales, dim=1).values <= self.config.percent_dense * scene_extent).to(device)
        )
        
        # Clone these Gaussians
        new_params = self._extract_params(selected_pts_mask)
        self._add_gaussians(new_params)
        
        return selected_pts_mask.sum().item()
    
    def densify_and_split(
        self, 
        grads: torch.Tensor, 
        grad_threshold: float, 
        scene_extent: float,
        N: int = 2
    ) -> int:
        """
        Split large Gaussians with high gradients.
        
        Args:
            grads: Average gradients [N, 1].
            grad_threshold: Gradient threshold.
            scene_extent: Scene extent.
            N: Number of samples for splitting.

        Returns:
            Number of split Gaussians.
        """
        # Find Gaussians to split
        # Condition: High gradient AND large size
        device = self.model.device
        
        # Pad grads to match current model size (after potential cloning)
        n_init_points = self.model.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        # --- Motion-Guided Densification ---
        if hasattr(self.model, 'get_motion') and self.model.get_motion is not None:
            velocities = self.model.get_motion
            speed = torch.norm(velocities, dim=-1)
            # Shield against NaN velocities which would poison the threshold and block all splitting
            speed = torch.nan_to_num(speed, nan=0.0)
            speed_factor = torch.clamp(speed / 2.0, 0.0, 1.0)
            dynamic_threshold = grad_threshold * (1.0 - 0.75 * speed_factor)
            selected_pts_mask = (padded_grad >= dynamic_threshold).to(device)
        else:
            selected_pts_mask = (padded_grad >= grad_threshold).to(device)
        
        scales = self.model.get_scaling
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            (torch.max(scales, dim=1).values > self.config.percent_dense * scene_extent).to(device)
        )
        
        # Sample from selected Gaussians
        stds = scales[selected_pts_mask].repeat(N, 1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        
        # Create rotation matrices
        rots = self.model.get_rotation[selected_pts_mask].repeat(N, 1)
        
        # Rotate samples using quaternions
        new_xyz = self._quaternion_multiply_vec(rots, samples) + \
                  self.model.get_xyz[selected_pts_mask].repeat(N, 1)
        
        # Shrink scale (split Gaussians should be smaller)
        split_scaling = self.model.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        new_scaling = torch.log(split_scaling)
        
        # Copy other attributes
        new_params = {
            '_xyz': new_xyz,
            '_features_dc': self.model._features_dc[selected_pts_mask].repeat(N, 1, 1),
            '_features_rest': self.model._features_rest[selected_pts_mask].repeat(N, 1, 1),
            '_scaling': new_scaling,
            '_rotation': rots,
            '_opacity': self.model._opacity[selected_pts_mask].repeat(N, 1),
        }
        
        # Copy FreeTimeGS parameters if present
        if hasattr(self.model, '_t') and self.model._t is not None:
             new_params['_t'] = self.model._t[selected_pts_mask].repeat(N, 1)
        if hasattr(self.model, '_t_scale') and self.model._t_scale is not None:
             new_params['_t_scale'] = self.model._t_scale[selected_pts_mask].repeat(N, 1)
        if hasattr(self.model, '_motion') and self.model._motion is not None:
             new_params['_motion'] = self.model._motion[selected_pts_mask].repeat(N, 1)
        
        # Add new Gaussians
        self._add_gaussians(new_params)
        
        # Remove original Gaussians (that have been split)
        prune_filter = torch.cat([
            selected_pts_mask,
            torch.zeros(N * int(selected_pts_mask.sum().item()), dtype=torch.bool, device=device)
        ])
        self._prune_points(prune_filter)
        
        return selected_pts_mask.sum().item()
    
    def prune_low_opacity(self, min_opacity: float) -> int:
        """
        Prune Gaussians with low opacity.
        
        Args:
            min_opacity: Minimum opacity threshold.

        Returns:
            Number of pruned Gaussians.
        """
        prune_mask = (self.model.get_opacity < min_opacity).squeeze()
        self._prune_points(prune_mask)
        return prune_mask.sum().item()
    
    def prune_big_points(self, max_screen_size: float, max_world_size: float) -> int:
        """
        Prune Gaussians that are too large.
        
        Args:
            max_screen_size: Maximum screen space size.
            max_world_size: Maximum world space size.

        Returns:
            Number of pruned Gaussians.
        """
        # Note: In Original 3DGS, max_radii2D is improperly reset to zero during densification_postfix (called by densify_and_clone/split).
        # This effectively disables screen-space pruning in the same iteration as densification.
        # To match the original behavior and prevent "holes" (dropping valid points that are close to camera),
        # we disable screen-space pruning here.
        # big_points_vs = self.max_radii2D > max_screen_size
        big_points_vs = torch.zeros_like(self.max_radii2D, dtype=torch.bool)
        
        # ModernGS FIX: Match Original 3DGS pruning threshold (0.1 * extent)
        big_points_ws = torch.max(self.model.get_scaling, dim=1).values > 0.1 * max_world_size
        prune_mask = torch.logical_or(big_points_vs, big_points_ws)
        
        self._prune_points(prune_mask)
        return prune_mask.sum().item()


    def densify_and_prune(
        self,
        iteration: int,
        max_grad: float,
        min_opacity: float,
        extent: float,
        max_screen_size: Optional[float]
    ):
        """
        Main Densify and Prune process.
        
        Args:
            iteration: Current iteration number.
            max_grad: Gradient threshold.
            min_opacity: Minimum opacity.
            extent: Scene extent.
            max_screen_size: Maximum screen size (None for no size pruning).
        """
        # Calculate average gradients
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        # Densify
        if iteration >= self.config.densify_from_iter and \
           iteration <= self.config.densify_until_iter:
            # Clone small Gaussians
            self.densify_and_clone(grads, max_grad, extent)
            
            # Split large Gaussians
            self.densify_and_split(grads, max_grad, extent, self.config.N_split)
        
        # Prune
        if iteration >= self.config.prune_from_iter and iteration <= self.config.densify_until_iter:
            # Prune low opacity Gaussians
            self.prune_low_opacity(min_opacity)
            
            # Prune large Gaussians
            if max_screen_size is not None:
                self.prune_big_points(max_screen_size, extent)
        
        # Reset stats
        self.reset_stats()
    
    def reset_stats(self):
        """Reset gradient statistics."""
        self.xyz_gradient_accum.zero_()
        self.denom.zero_()
        self.max_radii2D.zero_()
    
    def _extract_params(self, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract parameters for selected Gaussians.
        
        Args:
            mask: Boolean mask [N].
            
        Returns:
            Dictionary of parameters.
        """
        params = {
            '_xyz': self.model._xyz[mask],
            '_features_dc': self.model._features_dc[mask],
            '_features_rest': self.model._features_rest[mask],
            '_scaling': self.model._scaling[mask],
            '_rotation': self.model._rotation[mask],
            '_opacity': self.model._opacity[mask],
        }
        
        # FreeTimeGS parameters
        if hasattr(self.model, '_t') and self.model._t is not None:
             params['_t'] = self.model._t[mask]
        if hasattr(self.model, '_t_scale') and self.model._t_scale is not None:
             params['_t_scale'] = self.model._t_scale[mask]
        if hasattr(self.model, '_motion') and self.model._motion is not None:
             params['_motion'] = self.model._motion[mask]
        
        return params
    
    def _add_gaussians(self, new_params: Dict[str, torch.Tensor]):
        """
        Add new Gaussians.
        
        Args:
            new_params: Dictionary of new Gaussian parameters.
        """
        # Map internal model names to optimizer group names
        name_map = {
            '_xyz': 'xyz',
            '_features_dc': 'f_dc',
            '_features_rest': 'f_rest',
            '_opacity': 'opacity',
            '_scaling': 'scaling',
            '_rotation': 'rotation',
            '_t': 't',
            '_t_scale': 't_scale',
            '_motion': 'velocity'
        }

        # Prepare dictionary for optimizer (group_name -> tensor)
        optimizer_tensors = {}
        for param_name, tensor in new_params.items():
            if param_name in name_map:
                optimizer_tensors[name_map[param_name]] = tensor

        # Update Optimizer (adds parameters and returns new nn.Parameters)
        # Note: GaussianOptimizer.cat_tensors_to_optimizer handles the creation of new parameters
        new_model_params = self.optimizer.cat_tensors_to_optimizer(optimizer_tensors)
        
        # Update Model parameters with the new parameters from optimizer
        for param_name, group_name in name_map.items():
            if group_name in new_model_params:
                # Set attribute on model (handles property setters)
                setattr(self.model, param_name, new_model_params[group_name])

        # Update statistics buffers (manually managed buffers)
        n_new = new_params['_xyz'].shape[0]
        device = self.model.device
        self.xyz_gradient_accum = torch.cat([
            self.xyz_gradient_accum,
            torch.zeros((n_new, 1), device=device)
        ], dim=0)
        self.denom = torch.cat([
            self.denom,
            torch.zeros((n_new, 1), device=device)
        ], dim=0)
        self.max_radii2D = torch.cat([
            self.max_radii2D,
            torch.zeros(n_new, device=device)
        ], dim=0)
    
    def _prune_points(self, mask: torch.Tensor):
        """
        Remove Gaussians.
        
        Args:
            mask: Boolean mask of Gaussians to remove [N].
        """
        valid_points_mask = ~mask
        
        # Update Optimizer (removes parameters and returns new nn.Parameters)
        # Note: GaussianOptimizer.prune_optimizer takes a mask of KEPT points (valid_points_mask)
        new_model_params = self.optimizer.prune_optimizer(valid_points_mask)
        
        # Update Model parameters
        name_map = {
            'xyz': '_xyz',
            'f_dc': '_features_dc',
            'f_rest': '_features_rest',
            'opacity': '_opacity',
            'scaling': '_scaling',
            'rotation': '_rotation',
            't': '_t',
            't_scale': '_t_scale',
            'velocity': '_motion'
        }
        
        # Map back from group name to model attribute name
        for group_name, new_param in new_model_params.items():
            if group_name in name_map:
                setattr(self.model, name_map[group_name], new_param)
        
        # Update statistics buffers
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def _quaternion_multiply_vec(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Rotate vector using quaternion.
        
        Args:
            q: Quaternion [N, 4] (w, x, y, z).
            v: Vector [N, 3].
            
        Returns:
            Rotated vector [N, 3].
        """
        q_w = q[:, 0:1]
        q_vec = q[:, 1:]
        
        t = 2.0 * torch.linalg.cross(q_vec, v)
        return v + q_w * t + torch.linalg.cross(q_vec, t)

