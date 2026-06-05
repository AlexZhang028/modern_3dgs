"""
Adaptive Density Control for Gaussian Splatting

Implements adaptive density control algorithms:
- Clone: Duplicate small Gaussians.
- Split: Split large Gaussians.
- Prune: Remove Gaussians with low opacity or excessive size.
"""

import time
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple

from config.config import DensificationConfig
from utils.general_utils import inverse_sigmoid


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
            # Selection is a control policy, not a differentiable training target.
            # Detach explicitly to prevent accidental gradient coupling in future refactors.
            velocities = self.model.get_motion.detach()
            speed = torch.norm(velocities, dim=-1)
            # Shield against NaN velocities which would poison the threshold and block all cloning
            speed = torch.nan_to_num(speed, nan=0.0)
            velocity_normalizer = getattr(self.config, 'velocity_normalizer', 2.0)
            speed_factor = torch.clamp(speed / velocity_normalizer, 0.0, 1.0)
            clone_coeff = getattr(self.config, 'clone_speed_shrink_coeff', 0.75)
            dynamic_threshold = grad_threshold * (1.0 - clone_coeff * speed_factor)
            selected_pts_mask = (torch.norm(grads.detach(), dim=-1) >= dynamic_threshold).to(device)
        else:
            selected_pts_mask = (torch.norm(grads.detach(), dim=-1) >= grad_threshold).to(device)
        
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
            # Selection is a control policy, not a differentiable training target.
            # Detach explicitly to prevent accidental gradient coupling in future refactors.
            velocities = self.model.get_motion.detach()
            speed = torch.norm(velocities, dim=-1)
            # Shield against NaN velocities which would poison the threshold and block all splitting
            speed = torch.nan_to_num(speed, nan=0.0)
            velocity_normalizer = getattr(self.config, 'velocity_normalizer', 2.0)
            speed_factor = torch.clamp(speed / velocity_normalizer, 0.0, 1.0)
            clone_coeff = getattr(self.config, 'clone_speed_shrink_coeff', 0.75)
            dynamic_threshold = grad_threshold * (1.0 - clone_coeff * speed_factor)
            selected_pts_mask = (padded_grad.detach() >= dynamic_threshold).to(device)
        else:
            selected_pts_mask = (padded_grad.detach() >= grad_threshold).to(device)
        
        scales = self.model.get_scaling
        base_size_threshold = self.config.percent_dense * scene_extent
        if hasattr(self.model, '_motion') and self.model._motion is not None:
            velocities = self.model._motion.detach()
            speed = torch.nan_to_num(torch.norm(velocities, dim=-1), nan=0.0)
            velocity_normalizer = getattr(self.config, 'velocity_normalizer', 2.0)
            scale_shrink_multiplier = getattr(self.config, 'scale_shrink_multiplier', 2.0)
            speed_factor = torch.clamp(speed / velocity_normalizer, 0.0, 1.0)
            dynamic_size_threshold = base_size_threshold / (1.0 + scale_shrink_multiplier * speed_factor)
        else:
            dynamic_size_threshold = base_size_threshold

        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            (torch.max(scales, dim=1).values > dynamic_size_threshold).to(device)
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
        split_shrink_base = getattr(self.config, 'split_scale_shrink_base', 0.8)
        split_scaling = self.model.get_scaling[selected_pts_mask].repeat(N, 1) / (split_shrink_base * N)
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
        if 'gate' in self.model._gaussian_params:
             new_params['_gate'] = self.model._gaussian_params['gate'][selected_pts_mask].repeat(N, 1)
        
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
        max_screen_size: Optional[float],
        n_split_override: Optional[int] = None,
    ):
        """
        Main Densify and Prune process.

        Args:
            iteration: Current iteration number.
            max_grad: Gradient threshold.
            min_opacity: Minimum opacity.
            extent: Scene extent.
            max_screen_size: Maximum screen size (None for no size pruning).
            n_split_override: If set, overrides config.N_split for this call.
        """
        # Calculate average gradients
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        n_split = n_split_override if n_split_override is not None else self.config.N_split

        # Densify
        if iteration >= self.config.densify_from_iter and \
           iteration <= self.config.densify_until_iter:
            # Clone small Gaussians
            self.densify_and_clone(grads, max_grad, extent)

            # Split large Gaussians
            self.densify_and_split(grads, max_grad, extent, n_split)
        
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
        if 'gate' in self.model._gaussian_params:
             params['_gate'] = self.model._gaussian_params['gate'][mask]

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
            '_motion': 'velocity',
            '_gate': 'gate',
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
            'velocity': '_motion',
            'gate': '_gate',
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


# ==================== Densification Scheduler ====================

class DensificationScheduler:
    """
    Owns the per-iteration densification policy (when / with what parameters).

    Wraps GaussianDensifier (mechanics) so the Trainer only needs two calls
    per iteration inside `with torch.no_grad()`:

        scheduler.record_stats(rendered, iteration)
        timing.update(scheduler.step(iteration))

    Override points for subclasses:
        _compute_grad_threshold(iteration) -> float
        _compute_n_split(iteration)        -> Optional[int]
    """

    def __init__(
        self,
        densifier: GaussianDensifier,
        model: nn.Module,
        optimizer: Any,
        config: Any,
        scene_extent: float,
        static_mask_fn: Optional[Callable] = None,
    ):
        self.densifier    = densifier
        self.model        = model
        self.optimizer    = optimizer
        self.config       = config
        self.scene_extent = scene_extent
        # Callable () -> BoolTensor[N]: True = static/persistent.
        # Used for phase-based stat masking and opacity soft-decay.
        self.static_mask_fn = static_mask_fn

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def record_stats(self, rendered: Dict, iteration: int) -> None:
        """Accumulate gradient stats from one render output.

        Handles culled-mask mapping (temporal renderer) and phase-based
        stat masking (decouple training).  No-op past densify_until_iter.
        """
        if iteration >= self.config.densify_until_iter:
            return

        if 'culled_mask' in rendered and rendered['culled_mask'] is not None:
            self._record_stats_culled(rendered)
        else:
            self.densifier.update_stats(
                viewspace_point_tensor=rendered['viewspace_points'],
                visibility_filter=rendered['visibility_filter'],
                radii=rendered['radii'],
            )

        self._apply_phase_stat_masking(iteration)

    def step(self, iteration: int) -> Dict[str, float]:
        """Run densify+prune and/or opacity reset at the appropriate iterations.

        Returns a timing dict with keys 'densify_prune_ms' and 'opacity_reset_ms'.
        """
        timing: Dict[str, float] = {}
        if iteration >= self.config.densify_until_iter:
            return timing

        opacity_reset_interval = getattr(self.config, 'opacity_reset_interval', 3000)
        just_decayed = (iteration % opacity_reset_interval == 0)

        densify_from     = getattr(self.config, 'densify_from_iter', 500)
        densify_interval = getattr(self.config, 'densify_interval', 100)
        if iteration > densify_from and iteration % densify_interval == 0:
            size_threshold = 20 if iteration > opacity_reset_interval else None
            t0 = time.perf_counter()
            self.densifier.densify_and_prune(
                iteration       = iteration,
                max_grad        = self._compute_grad_threshold(iteration),
                min_opacity     = getattr(self.config, 'prune_opacity_threshold', 0.005),
                extent          = self.scene_extent,
                max_screen_size = size_threshold,
                n_split_override= self._compute_n_split(iteration),
            )
            timing['densify_prune_ms'] = (time.perf_counter() - t0) * 1000.0

        white_bg_first = (
            getattr(self.config, 'white_background', False)
            and iteration == densify_from
        )
        if just_decayed or white_bg_first:
            t0 = time.perf_counter()
            self._reset_opacity(iteration)
            self.densifier.reset_stats()
            timing['opacity_reset_ms'] = (time.perf_counter() - t0) * 1000.0

        return timing

    # ------------------------------------------------------------------ #
    #  Override points                                                     #
    # ------------------------------------------------------------------ #

    def _compute_grad_threshold(self, iteration: int) -> float:
        opacity_reset_until = getattr(self.config, 'opacity_reset_until_iter', 15000)
        densify_until       = self.config.densify_until_iter
        base_grad           = self.config.densify_grad_threshold
        final_grad          = getattr(self.config, 'densify_grad_threshold_final', 0.0003)

        threshold = base_grad
        if iteration > opacity_reset_until and densify_until > opacity_reset_until:
            progress  = min(
                (iteration - opacity_reset_until) / float(densify_until - opacity_reset_until),
                1.0,
            )
            threshold = base_grad + progress * (final_grad - base_grad)
        return threshold

    def _compute_n_split(self, iteration: int) -> Optional[int]:
        return None  # base: use densifier config default

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _record_stats_culled(self, rendered: Dict) -> None:
        """Stats update path for temporal renderers that cull Gaussians."""
        culled_mask    = rendered['culled_mask']
        vis_filter     = rendered['visibility_filter']
        viewspace      = rendered['viewspace_points']
        radii          = rendered['radii']

        active_indices  = torch.nonzero(culled_mask).view(-1)
        visible_indices = active_indices[vis_filter]

        if viewspace.grad is not None:
            grad_norm = torch.norm(viewspace.grad[:, :2], dim=-1, keepdim=True)
            self.densifier.xyz_gradient_accum[visible_indices] += grad_norm[vis_filter]
            self.densifier.denom[visible_indices] += 1.0

        self.densifier.max_radii2D[visible_indices] = torch.max(
            self.densifier.max_radii2D[visible_indices],
            radii[vis_filter],
        )

    def _apply_phase_stat_masking(self, iteration: int) -> None:
        """Zero gradient stats for frozen Gaussians (phase-decoupled training)."""
        if not getattr(self.config, 'decouple_training_enabled', False):
            return
        if not (hasattr(self.model, '_t_scale') and self.model._t_scale is not None):
            return
        if self.static_mask_fn is None:
            return

        joint_end   = getattr(self.config, 'decouple_joint_end_iter',   15000)
        dynamic_end = getattr(self.config, 'decouple_dynamic_end_iter', 30000)
        if iteration <= joint_end:
            return

        is_static   = self.static_mask_fn()
        freeze_mask = is_static if iteration <= dynamic_end else ~is_static
        self.densifier.xyz_gradient_accum[freeze_mask] = 0.0
        self.densifier.denom[freeze_mask]              = 0.0

    def _reset_opacity(self, iteration: int) -> None:
        current_opacity = self.model.get_opacity

        if iteration <= getattr(self.config, 'opacity_reset_until_iter', 15000):
            target_opacity = torch.min(
                current_opacity, torch.ones_like(current_opacity) * 0.05
            )
            opacities_new = inverse_sigmoid(target_opacity)
            opt_tensors   = self.optimizer.replace_tensor_to_optimizer(opacities_new, "opacity")
            self.model._opacity = opt_tensors["opacity"]
        else:
            is_semi_transparent = current_opacity < 0.5
            above_prune_safe    = current_opacity > 0.015

            if (hasattr(self.model, '_t_scale') and self.model._t_scale is not None
                    and self.static_mask_fn is not None):
                is_static_mask = self.static_mask_fn().unsqueeze(-1)
                target_mask    = is_static_mask & is_semi_transparent & above_prune_safe
            else:
                target_mask = is_semi_transparent & above_prune_safe

            decay_mask = target_mask.squeeze(-1)
            if decay_mask.any():
                with torch.no_grad():
                    decayed = current_opacity[decay_mask] * 0.95
                    self.model._opacity.data[decay_mask] = inverse_sigmoid(decayed)


class FreeTimeDensificationScheduler(DensificationScheduler):
    """Adds FreeTimeGS-specific policy: dynamic densification boost and MCMC relocation."""

    def step(self, iteration: int) -> Dict[str, float]:
        timing: Dict[str, float] = {}

        # Relocation runs before densify so recycled Gaussians participate immediately
        densify_from   = getattr(self.config, 'densify_from_iter', 500)
        densify_until  = getattr(self.config, 'densify_until_iter', 15000)
        reloc_interval = getattr(self.config, 'relocation_interval', 500)
        if (iteration > densify_from and iteration <= densify_until
                and iteration % reloc_interval == 0):
            t0 = time.perf_counter()
            self._relocate_gaussians(iteration)
            timing['relocate_ms'] = (time.perf_counter() - t0) * 1000.0

        timing.update(super().step(iteration))
        return timing

    def _compute_grad_threshold(self, iteration: int) -> float:
        threshold = super()._compute_grad_threshold(iteration)
        if getattr(self.config, 'decouple_training_enabled', False):
            joint_end   = getattr(self.config, 'decouple_joint_end_iter',   15000)
            dynamic_end = getattr(self.config, 'decouple_dynamic_end_iter', 30000)
            if joint_end < iteration <= dynamic_end:
                mult      = getattr(self.config, 'decouple_dynamic_densify_mult', 0.5)
                threshold *= mult
        return threshold

    def _compute_n_split(self, iteration: int) -> Optional[int]:
        if getattr(self.config, 'decouple_training_enabled', False):
            joint_end   = getattr(self.config, 'decouple_joint_end_iter',   15000)
            dynamic_end = getattr(self.config, 'decouple_dynamic_end_iter', 30000)
            if joint_end < iteration <= dynamic_end:
                return getattr(self.config, 'decouple_dynamic_n_split', 3)
        return None

    def _relocate_gaussians(self, iteration: int) -> None:
        """MCMC-style relocation (FreeTimeGS++, Secret 4)."""
        opac       = self.model.get_opacity.squeeze()
        prune_mask = (opac < 0.01) & (opac >= 0.005)
        num_prune  = prune_mask.sum().item()
        if num_prune == 0 or self.densifier.denom.sum() == 0:
            return

        grads     = self.densifier.xyz_gradient_accum / torch.clamp(
            self.densifier.denom, min=1e-6
        )
        grads[grads.isnan()] = 0.0
        grad_norm = torch.norm(grads, dim=-1)
        g_max     = grad_norm.max()
        if g_max > 0:
            grad_norm = grad_norm / g_max

        score             = 0.5 * grad_norm + 0.5 * opac
        score[prune_mask] = -1.0

        n_relocate = min(num_prune, (~prune_mask).sum().item())
        if n_relocate == 0:
            return

        _, receptor_indices = torch.topk(score, n_relocate)
        donor_all = torch.nonzero(prune_mask).squeeze()
        if donor_all.ndim == 0:
            donor_all = donor_all.unsqueeze(0)
        donor_indices = donor_all[:n_relocate]

        final_mask = torch.zeros_like(prune_mask)
        final_mask[donor_indices] = True

        counts = torch.zeros(self.model.num_points, device='cuda', dtype=torch.float32)
        counts.scatter_add_(0, receptor_indices, torch.ones(n_relocate, device='cuda'))
        log_n_total = torch.log((counts[receptor_indices] + 1.0).clamp(min=1.0))

        with torch.no_grad():
            xyz_noise = (torch.rand(n_relocate, 3, device='cuda') - 0.5) * 0.01
            self.model._xyz.data[donor_indices] = (
                self.model._xyz.data[receptor_indices] + xyz_noise
            )
            self.model._features_dc.data[donor_indices]   = self.model._features_dc.data[receptor_indices]
            self.model._features_rest.data[donor_indices] = self.model._features_rest.data[receptor_indices]
            self.model._rotation.data[donor_indices]      = self.model._rotation.data[receptor_indices]
            self.model._scaling.data[donor_indices] = (
                self.model._scaling.data[receptor_indices] - (log_n_total / 3.0).unsqueeze(-1)
            )
            self.model._opacity.data[donor_indices] = (
                self.model._opacity.data[receptor_indices] - log_n_total.unsqueeze(-1)
            )
            if hasattr(self.model, '_t') and self.model._t is not None:
                self.model._t.data[donor_indices]     = self.model._t.data[receptor_indices]
            if hasattr(self.model, '_t_scale') and self.model._t_scale is not None:
                self.model._t_scale.data[donor_indices] = self.model._t_scale.data[receptor_indices]
            if hasattr(self.model, '_motion') and self.model._motion is not None:
                vel_noise = torch.randn(n_relocate, 3, device='cuda') * 0.05
                self.model._motion.data[donor_indices] = (
                    self.model._motion.data[receptor_indices] + vel_noise
                )
            if 'gate' in self.model._gaussian_params:
                self.model._gate.data[donor_indices] = self.model._gate.data[receptor_indices]

        self.optimizer.reset_optimizer_state(final_mask)

