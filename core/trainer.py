"""
Gaussian Splatting Trainer

Unified training framework supporting both Static and FreeTime modes.
"""

import os
import time
import math
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from tqdm import tqdm
from torch.utils.data import DataLoader

from config.config import TrainerConfig, DataConfig
from utils.image_utils import psnr
import utils.general_utils as utils
from core.loss import l1_loss, ssim, GaussianLoss
from core.densify import GaussianDensifier, DensificationConfig
from data.samplers import DataSampler, StaticSampler, TemporalSampler
from utils.general_utils import inverse_sigmoid

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available.")


class Trainer:
    """
    Unified Gaussian Splatting Trainer.
    
    Automatically selects training mode based on model.config.mode:
    - "static": Static scene (Original 3DGS).
    - "freetime": Temporal scene (FreeTimeGS).
    
    Args:
        model: GaussianModel instance.
        optimizer: GaussianOptimizer instance.
        renderer: GaussianRenderer instance.
        dataset: GaussianDataset instance.
        config: TrainerConfig configuration.
        test_dataset: Test dataset (optional).
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Any,
        renderer: nn.Module,
        dataset: Any,
        config: TrainerConfig,
        data_config: Optional[DataConfig] = None,
        test_dataset: Optional[Any] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.renderer = renderer
        self.dataset = dataset
        self.test_dataset = test_dataset
        self.config = config
        self.data_config = data_config
        
        # Auto-detect training mode
        self.mode = model.config.mode
        print(f"Training Mode: {self.mode}")
        
        # Create Data Sampler
        self.sampler = self._create_sampler()

        
        # Create Loss Function
        self.loss_fn = GaussianLoss(
            lambda_dssim=config.lambda_dssim,
            lambda_lpips=config.lambda_lpips
        )
        # Move loss to device (important for LPIPS weights)
        if config.lambda_lpips > 0:
            self.loss_fn = self.loss_fn.cuda()
        
        # Create Densifier
        densify_config = DensificationConfig(
            densify_grad_threshold=config.densify_grad_threshold,
            densify_from_iter=config.densify_from_iter,
            densify_until_iter=config.densify_until_iter,
            densify_interval=config.densify_interval,
            prune_opacity_threshold=config.prune_opacity_threshold,
            prune_size_threshold=config.prune_size_threshold,
        )
        # Fix: Pass the GaussianOptimizer wrapper, not the raw Adam optimizer
        # This allows Densifier to use wrapper methods like cat_tensors_to_optimizer
        self.densifier = GaussianDensifier(densify_config, model, optimizer)
        
        # Create Output Directories
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create TensorBoard writer
        self.writer = None
        if config.enable_tensorboard and TENSORBOARD_AVAILABLE:
            log_dir = self.output_dir / config.log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_dir))
            print(f"TensorBoard Log: {log_dir}")
        
        # Background Color
        if config.white_background:
            self.bg_color = torch.tensor([1.0, 1.0, 1.0], device="cuda")
        else:
            self.bg_color = torch.tensor([0.0, 0.0, 0.0], device="cuda")
        
        # Training State
        self.current_iteration = 0
        self.current_sh_degree = 0

        # Depth Loss Scheduler (for whiteroom/monocular depth datasets)
        from utils.general_utils import get_expon_lr_func
        self.depth_l1_weight = get_expon_lr_func(
            config.depth_l1_weight_init, 
            config.depth_l1_weight_final, 
            max_steps=config.iterations
        )
        
        # Depth Loss Scheduler (for whiteroom/monocular depth datasets)
        from utils.general_utils import get_expon_lr_func
        self.depth_l1_weight = get_expon_lr_func(
            config.depth_l1_weight_init, 
            config.depth_l1_weight_final, 
            max_steps=config.iterations
        )

        # Statistics
        self.stats = {
            'loss_history': [],
            'gaussian_count': []
        }
        
        # Compute Scene Extent (Once)
        self.scene_extent = self._compute_scene_extent()

        # Select fixed views for testing
        self.train_view_indices = self._select_fixed_views(len(dataset))
        self.test_view_indices = []
        if self.test_dataset:
            self.test_view_indices = self._select_fixed_views(len(test_dataset))
        
        self.logged_gt = False

    def _select_fixed_views(self, num_total_views: int) -> List[int]:
        """Select fixed indices for test views."""
        num_views = self.config.num_test_views
        if num_views == -1 or num_views >= num_total_views:
            return list(range(num_total_views))
        else:
            return torch.linspace(0, num_total_views - 1, num_views).round().long().tolist()
    
    def _create_sampler(self) -> DataSampler:
        """Create data sampler."""
        num_workers = self.data_config.num_workers if self.data_config else 0
        if self.mode == "static":
            return StaticSampler(self.dataset, num_workers=num_workers)
        elif self.mode == "freetime":
            return TemporalSampler(self.dataset, num_workers=num_workers)
        else:
            raise ValueError(f"Unknown training mode: {self.mode}")
    
    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print(f"Start Training ({self.mode} Mode)")
        print(f"   Iterations: {self.config.iterations}")
        print(f"   L_DSSIM: {self.config.lambda_dssim}")
        if hasattr(self.config, 'lambda_lpips'):
            print(f"   L_LPIPS: {self.config.lambda_lpips}")
        print(f"   Initial Gaussians: {self.model.num_points}")
        print("=" * 60 + "\n")
        
        # Training Loop
        # ModernGS FIX: Match Original 3DGS iteration range (1 to iterations)
        # Original: range(first_iter, opt.iterations + 1) -> 1 to 30000
        progress_bar = tqdm(range(1, self.config.iterations + 1), desc="Training")
        
        for iteration in progress_bar:
            self.current_iteration = iteration
            
            # Training Step
            torch.cuda.synchronize()
            iter_start = time.time()
            metrics = self.train_step(iteration)
            torch.cuda.synchronize()
            iter_end = time.time()
            metrics['iteration_time'] = (iter_end - iter_start) * 1000.0
            
            # Update Progress Bar
            postfix = {
                'loss': f"{metrics['loss']:.4f}",
                'gaussians': self.model.num_points
            }
            if 'lpips' in metrics:
                postfix['lpips'] = f"{metrics['lpips']:.4f}"
            progress_bar.set_postfix(postfix)
            
            # Logging
            if iteration % self.config.log_interval == 0:
                self._log_metrics(iteration, metrics)
            
            # Testing
            if self.test_dataset:
                is_test_interval = (self.config.test_interval > 0) and (iteration % self.config.test_interval == 0)
                is_test_iter = iteration in self.config.test_iterations
                if is_test_interval or is_test_iter:
                    self._test(iteration)
            
            # Save Checkpoint
            is_save_interval = (self.config.save_interval > 0) and (iteration % self.config.save_interval == 0) and (iteration > 0)
            is_save_iter = iteration in self.config.save_iterations or iteration in self.config.checkpoint_iterations
            if is_save_interval or is_save_iter:
                self.save_checkpoint(iteration)
        
        # Save Final Model
        print("\nTraining Complete! Saving final model...")
        self.save_checkpoint(self.config.iterations, final=True)

        # Final Test
        if self.test_dataset:
             self._test(self.config.iterations)
        
        if self.writer:
            self.writer.close()
    
    def train_step(self, iteration: int) -> Dict[str, float]:
        """
        Execute one training step.
        
        Args:
            iteration: Current iteration number.
            
        Returns:
            metrics: Dictionary containing loss and other metrics.
        """
        # 3. Sample Data
        camera, timestamp = self.sampler.sample()

        # 1. Update Learning Rate
        # Must be done AFTER sampling time - Fixed: Velocity LR uses iteration now
        self.optimizer.update_learning_rate(iteration)
        
        # 2. Update SH Degree
        self._update_sh_degree(iteration)
        
        # Ensure camera data is on GPU (safe for multiprocessing)
        # This handles both images and transformation matrices
        # IMPORTANT: Use copy to avoid modifying persistent dataset objects when using 0 workers
        import copy
        camera = copy.copy(camera)
        camera.to("cuda")
        
        # 4. Random Background (Optional)
        if self.config.random_background:
            bg_color = torch.rand(3, device="cuda")
        else:
            bg_color = self.bg_color
        
        # 5. Render
        rendered = self.renderer(
            gaussians=self.model,
            camera=camera,
            bg_color=bg_color,
            timestamp=timestamp,
            enable_culling=False  # Disable culling for stable training and proper densification stats
        )
        
        # 6. Compute Loss
        target = camera.image.cuda()

        # Handle Alpha Mask (Mask out background if necessary)
        if hasattr(camera, 'alpha_mask') and camera.alpha_mask is not None:
             alpha_mask = camera.alpha_mask.cuda()
             rendered['render'] = rendered['render'] * alpha_mask

        loss_components = self.loss_fn.get_components(rendered['render'], target)
        loss = loss_components['total']
        
        # Loss Hook
        loss = self._compute_loss_hook(loss, rendered, iteration)

        # Depth regularization (if available)
        Ll1depth_pure = 0.0
        if hasattr(camera, 'depth_map') and camera.depth_map is not None and camera.depth_reliable:
            weight = self.depth_l1_weight(iteration)
            if weight > 0:
                invDepth = rendered["depth"]
                mono_invdepth = camera.depth_map.cuda()
                depth_mask = camera.depth_mask.cuda()
                
                Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
                loss += weight * Ll1depth_pure
        
        # 7. Backward Pass
        loss.backward()

        # 8. Adaptive Control (Densification, Pruning, Relocation)
        with torch.no_grad():
            if iteration < self.config.densify_until_iter:
                # Accumulate stats
                # Handle FreeTimeGS Culled rendering:
                # If renderer returns 'culled_mask', we need to map results back to full indices
                # or tell densifier that we only have partial update.
                # However, Densifier logic assumes it tracks stats for ALL gaussians globally.
                # Standard 3DGS doesn't cull gaussians from the Model list during render, it just culls them from rasterizer.
                # But FreeTimeGS `render_temporal` actively slices the tensors if `enable_culling=True`.
                # This breaks the 1:1 mapping needed for update_stats.
                
                # Check if we have culled mask
                if 'culled_mask' in rendered and rendered['culled_mask'] is not None:
                     culled_mask = rendered['culled_mask']
                     
                     # Map partial filters to full size
                     # rendered indices are relative to the SUBSET of selected gaussians
                     # We need to construct full-size filter/radii updates.
                     
                     # Only update stats for the visible ones in the culled set
                     vis_filter_subset = rendered['visibility_filter'] # [M]
                     viewspace_subset = rendered['viewspace_points'] # [M, 3]
                     radii_subset = rendered['radii'] # [M]
                     
                     # Construct global update masks
                     # We need to update self.densifier.xyz_gradient_accum[culled_mask][vis_filter_subset]
                     # BUT densifier.update_stats takes simple flat arguments and assumes full alignment.
                     
                     # We need to call a specialized update or manually handle it.
                     # Let's call a modified method if available, or manually inject.
                     # Actually, the densifier class needs robustness.
                     
                     # Simpler approach: Map `visibility_filter` to global indices
                     global_vis_filter = torch.zeros(self.model.num_points, dtype=torch.bool, device="cuda")
                     
                     # The indices within `culled_mask` that are visible
                     # culled_mask is boolean [N]. indices = which are true.
                     active_indices = torch.nonzero(culled_mask).view(-1) # [M]
                     
                     # Visible ones within the active set
                     visible_active_indices = active_indices[vis_filter_subset]
                     
                     global_vis_filter[visible_active_indices] = True
                     
                     # For viewspace points and radii, update_stats normally expects full-size tensors?
                     # No, update_stats(viewspace, filter, radii)
                     # self.xyz_gradient_accum[update_filter] += grad_norm[update_filter]
                     # Here update_filter acts as the indexer.
                     
                     # HOWEVER, viewspace_point_tensor must have gradients.
                     # In culled mode, only viewspace_subset has gradients.
                     # If we pass a sparse viewspace tensor, it won't work.
                     
                     # Solution: Pass the SUBSET to a specific method in Densifier, 
                     # OR map everything to global.
                     # Since gradients are on viewspace_subset, we must use viewspace_subset.
                     
                     # We will use a custom call to `densifier.add_densification_stats` with indices.
                     
                     # 1. Update Gradient Stats
                     # Extract gradients from subset
                     if viewspace_subset.grad is not None:
                         grad_norm_subset = torch.norm(viewspace_subset.grad[:, :2], dim=-1, keepdim=True) # [M, 1]
                         
                         # Apply visibility on subset
                         visible_grad_norm = grad_norm_subset[vis_filter_subset]
                         
                         # Accumulate to global
                         self.densifier.xyz_gradient_accum[visible_active_indices] += visible_grad_norm
                         self.densifier.denom[visible_active_indices] += 1.0

                         del visible_grad_norm
                     
                     radii_visible = radii_subset[vis_filter_subset]
                     
                     self.densifier.max_radii2D[visible_active_indices] = torch.max(
                         self.densifier.max_radii2D[visible_active_indices],
                         radii_visible
                     )

                     # Explicitly delete intermediate indexing tensors to prevent accumulation
                     del visible_active_indices
                     del active_indices
                     del culled_mask
                     
                else:
                    # Standard execution
                    self.densifier.update_stats(
                        viewspace_point_tensor=rendered['viewspace_points'],
                        visibility_filter=rendered['visibility_filter'],
                        radii=rendered['radii']
                    )
                
                # Relocation Hook
                self._post_backward_hook(iteration, rendered, target, camera, timestamp)

                # Standard Densify and Prune
                if iteration > self.config.densify_from_iter and iteration % self.config.densify_interval == 0:
                    size_threshold = 20 if iteration > self.config.opacity_reset_interval else None
                    
                    self.densifier.densify_and_prune(
                        iteration=iteration,
                        max_grad=self.config.densify_grad_threshold,
                        min_opacity=self.config.prune_opacity_threshold,
                        extent=self.scene_extent,
                        max_screen_size=size_threshold
                    )
                    
                    # Periodic garbage collection for VRAM fragmentation
                    if iteration % (self.config.densify_interval * 5) == 0:
                        torch.cuda.empty_cache()
                
                # Opacity Reset
                if iteration % self.config.opacity_reset_interval == 0 or \
                   (self.config.white_background and iteration == self.config.densify_from_iter):
                    self._reset_opacity()
        
        # 9. Optimizer Step
        if iteration < self.config.iterations:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        
        # Metrics
        metrics = {
            'loss': loss.item(),
            'l1': loss_components['l1'].item(),
            'ssim': loss_components['ssim'].item(),
            'num_gaussians': self.model.num_points
        }
        if 'lpips' in loss_components:
             val = loss_components['lpips']
             metrics['lpips'] = val.item() if hasattr(val, 'item') else val

        # Explicit cleanup to allow GC to reclaim graph immediately
        del rendered
        del loss
        del loss_components
        
        return metrics
    
    def _compute_loss_hook(self, loss: torch.Tensor, rendered: Dict, iteration: int) -> torch.Tensor:
        """Hook for additional loss computation."""
        return loss

    def _post_backward_hook(self, iteration: int, rendered: Dict, target: torch.Tensor, camera: Any, timestamp: float = 0.0):
        """Hook for post-backward operations (e.g. relocation)."""
        pass

    def _update_sh_degree(self, iteration: int):
        """Progressive SH degree activation."""
        max_sh_degree = self.model.max_sh_degree
        new_degree = min(
            iteration // self.config.sh_degree_interval,
            max_sh_degree
        )
        
        if new_degree > self.current_sh_degree:
            self.current_sh_degree = new_degree
            self.model.active_sh_degree = new_degree
            print(f"\nSH degree increased to: {new_degree}/{max_sh_degree}")


    
    def _reset_opacity(self):
        """
        Reset opacity of Gaussians.
        """
        # Calculate new values (Logit space)
        opacities_new = utils.inverse_sigmoid(torch.min(self.model.get_opacity, torch.ones_like(self.model.get_opacity)*0.01))
        
        # Update Optimizer and Get New Parameter
        optimizable_tensors = self.optimizer.replace_tensor_to_optimizer(opacities_new, "opacity")
        
        # Update Model
        self.model._opacity = optimizable_tensors["opacity"]
        
        # print(f"Opacity Reset Done.")

    
    def _compute_scene_extent(self) -> float:
        """
        Compute scene extent for densification.
        
        Returns:
            Scene radius.
        """
        # Use all camera positions
        cameras_positions = []
        for camera in self.dataset.cameras:
            cameras_positions.append(camera.camera_center)
        
        cameras_positions = torch.stack(cameras_positions)
        scene_center = cameras_positions.mean(dim=0)
        scene_extent = (cameras_positions - scene_center).norm(dim=1).max().item() * 1.1
        
        return scene_extent
    
    def _log_metrics(self, iteration: int, metrics: Dict[str, float]):
        """Log training metrics."""
        # TensorBoard
        if self.writer:
            self.writer.add_scalar('Loss/total', metrics['loss'], iteration)
            self.writer.add_scalar('Loss/l1', metrics['l1'], iteration)
            self.writer.add_scalar('Loss/ssim', metrics['ssim'], iteration)
            if 'lpips' in metrics:
                self.writer.add_scalar('Loss/lpips', metrics['lpips'], iteration)
            if 'iteration_time' in metrics:
                self.writer.add_scalar('Stats/iteration_time', metrics['iteration_time'], iteration)
            self.writer.add_scalar('Stats/num_gaussians', metrics['num_gaussians'], iteration)
            self.writer.add_scalar('Stats/sh_degree', self.current_sh_degree, iteration)
            self.writer.add_scalar('stats/gaussian_t_min', self.model.get_t.min().item(), iteration)
            self.writer.add_scalar('stats/gaussian_t_max', self.model.get_t.max().item(), iteration)
        
        # History
        self.stats['loss_history'].append(metrics['loss'])
        self.stats['gaussian_count'].append(metrics['num_gaussians'])
    
    def _test(self, iteration: int):
        """Evaluate on test set."""
        print(f"\nTesting (iter {iteration})...")
        self.model.eval()
        
        # 1. Evaluate subset of training views
        self._evaluate_set(self.dataset, self.train_view_indices, "Train", iteration)

        # 2. Evaluate subset of test views
        if self.test_dataset:
            self._evaluate_set(self.test_dataset, self.test_view_indices, "Test", iteration)
        
        # Mark GT as logged
        self.logged_gt = True
        self.model.train()

    def _evaluate_set(self, dataset, indices: List[int], prefix: str, iteration: int):
        """Evaluate a specific dataset subset."""
        psnr_list = []
        l1_list = []
        ssim_list = []
        
        print(f"   Evaluating {prefix} set ({len(indices)} images)...")
        
        with torch.no_grad():
            for idx in tqdm(indices, desc=f"Evaluating {prefix}", leave=False):
                sample = dataset[idx]
                camera = sample["camera"]
                camera.to("cuda")
                
                # Render
                rendered = self.renderer(
                    gaussians=self.model,
                    camera=camera,
                    bg_color=self.bg_color,
                    timestamp=camera.timestamp if hasattr(camera, 'timestamp') else 0.0,
                    enable_culling=False
                )
                
                prediction = rendered['render']
                target = camera.image.cuda()
                
                # Compute Metrics
                l1 = l1_loss(prediction, target).item()
                # Use clamped prediction for PSNR to match Original 3DGS behavior
                psnr_val = psnr(prediction.clamp(0.0, 1.0), target)
                ssim_val = ssim(prediction, target).item()
                
                l1_list.append(l1)
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                
                # TensorBoard Images
                if self.writer:
                    # GT only on first log
                    if not self.logged_gt:
                        self.writer.add_image(f'{prefix}_GT/{camera.image_name}', target, iteration)
                    
                    # Render result on every log
                    self.writer.add_image(f'{prefix}_Render/{camera.image_name}', prediction, iteration)
        
        # Compute Averages
        avg_l1 = torch.tensor(l1_list).mean().item()
        avg_psnr = torch.tensor(psnr_list).mean().item()
        avg_ssim = torch.tensor(ssim_list).mean().item()
        
        print(f"   {prefix} Results - L1: {avg_l1:.4f} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")
        
        if self.writer:
            self.writer.add_scalar(f'{prefix}/l1', avg_l1, iteration)
            self.writer.add_scalar(f'{prefix}/psnr', avg_psnr, iteration)
            self.writer.add_scalar(f'{prefix}/ssim', avg_ssim, iteration)

    
    def save_checkpoint(self, iteration: int, final: bool = False):
        """
        Save training results (Checkpoints and/or PLY).
        
        Args:
            iteration: Current iteration number.
            final: Whether this is the final model.
        """
        # 1. Save .pth Checkpoint
        if self.config.save_checkpoint or final:
            if final:
                checkpoint_path = self.checkpoint_dir / "final.pth"
            else:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_{iteration}.pth"
            
            checkpoint = {
                'iteration': iteration,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'sh_degree': self.current_sh_degree,
                'config': self.config,
                'stats': self.stats
            }
            
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint Saved: {checkpoint_path}")
            
            # Cleanup old checkpoints
            if not final:
                self._cleanup_old_checkpoints()

        # 2. Save PLY Model
        if self.config.save_ply or final:
            # Standard 3DGS output structure: point_cloud/iteration_X/point_cloud.ply
            ply_dir = self.output_dir / "point_cloud" / f"iteration_{iteration}"
            ply_dir.mkdir(parents=True, exist_ok=True)
            ply_path = ply_dir / "point_cloud.ply"
            
            self.model.save_ply(str(ply_path))
            print(f"Point Cloud Saved: {ply_path}")

    
    def _cleanup_old_checkpoints(self):
        """Cleanup old checkpoint files."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pth"),
            key=lambda p: int(p.stem.split('_')[1])
        )
        
        # Keep last N
        if len(checkpoints) > self.config.keep_checkpoints:
            for old_checkpoint in checkpoints[:-self.config.keep_checkpoints]:
                old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Checkpoint file path.
        """
        print(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_iteration = checkpoint['iteration']
        self.current_sh_degree = checkpoint['sh_degree']
        self.stats = checkpoint.get('stats', self.stats)
        
        print(f"Checkpoint Loaded (iter {self.current_iteration})")
    
    def export_model(self, output_path: str):
        """
        Export trained model (PLY format).
        
        Args:
            output_path: Output path.
        """
        print(f"Exporting model to: {output_path}")
        self.model.save_ply(output_path)
        print("Model Exported")


class FreeTimeTrainer(Trainer):
    """
    Trainer specific for FreeTimeGS (Temporal).
    Adds 4D Regularization and Relocation logic.
    """
    def _compute_loss_hook(self, loss: torch.Tensor, rendered: Dict, iteration: int) -> torch.Tensor:
        # 4D Regularization Loss
        # Prevents "walls" of opacity in early training
        # Formula: L_reg = 1/N * sum(sigma * sg[sigma(t)])
        # Based on FreeTimeGS reference. 
        # sigma: base_opacity
        # sigma(t): temporal_weight (which is 0.0-1.0)
        
        base_opacity = rendered['base_opacity']
        temporal_weight = rendered['temporal_weight']
        
        # Weight from config
        reg_weight = self.config.lambda_reg
        
        # Only detach the temporal instance component
        # We penalize base_opacity if the gaussian is active at this time (high temporal_weight)
        # if iteration < 1000:
        #     l_reg = 0
        # else:
        l_reg = (base_opacity * temporal_weight.detach()).mean()
        loss += reg_weight * l_reg
            
        return loss

    def _post_backward_hook(self, iteration: int, rendered: Dict, target: torch.Tensor, camera: Any, timestamp: float = 0.0):
        # Periodic Relocation
        if iteration > self.config.densify_from_iter and \
           iteration % self.config.relocation_interval == 0:
            self._relocate_gaussians(iteration, rendered, target, camera, timestamp)

    def _relocate_gaussians(self, iteration: int, rendered: Dict, target: torch.Tensor, camera: Any, timestamp: float = 0.0):
        """
        FreeTimeGS Relocation Strategy:
        Teleport "dead" gaussians (donors) to high-score regions (receptors).
        Score s = 0.5 * grad + 0.5 * opacity
        """
        # 1. Identify "Donors" (Low Base Opacity)
        # Using 0.01 threshold as per paper
        # And ensure we don't prune everything
        opac = self.model.get_opacity.squeeze()
        prune_mask = (opac < 0.01) & (opac >= 0.005) # Only recycle standard victims
        # If we just pick < 0.01, standard pruning (0.005) might have killed them? 
        # But this runs usually before pruning or around same time.
        
        num_prune = prune_mask.sum().item()
        
        if num_prune == 0:
            return

        # 2. Identify "Receptors" (High Sampling Score)
        # Use Densifier's stats! Model's buffers are often empty if not updated.
        if self.densifier.denom.sum() == 0: # Safety check
             return

        # Calculate average gradients
        # Note: densifier uses self.xyz_gradient_accum
        grads = self.densifier.xyz_gradient_accum / torch.clamp(self.densifier.denom, min=1e-6)
        grads[grads.isnan()] = 0.0
        grad_norm = torch.norm(grads, dim=-1)
        
        # Normalize to combine with opacity
        g_max = grad_norm.max()
        if g_max > 0: grad_norm /= g_max
        
        score = 0.5 * grad_norm + 0.5 * opac
        
        # Mask out donors so we don't relocate to dead zones
        score[prune_mask] = -1.0
        
        # Select Top-K receptors
        n_receptors = min(num_prune, (~prune_mask).sum().item())
        if n_receptors == 0: return

        top_scores, receptor_indices = torch.topk(score, n_receptors)
        
        # 3. Relocate
        # Move donors to receptors + perturbation
        receptor_pos = self.model.get_xyz[receptor_indices]
        receptor_scales = self.model.get_scaling[receptor_indices]
        
        # Random perturbation within receptor scale (or small constant)
        # Using small constant is safer for stability
        noise = (torch.rand_like(receptor_pos) - 0.5) * 0.01 # +/- 0.005 range
        new_xyz = receptor_pos + noise
        
        # Get receptor velocity for inheritance
        receptor_motion = self.model.get_motion[receptor_indices]
        # Add small velocity perturbation (e.g. 10% of velocity magnitude or fixed small value)
        motion_noise = (torch.rand_like(receptor_motion) - 0.5) * 0.2 * torch.norm(receptor_motion, dim=1, keepdim=True).clamp(min=1e-3)
        new_motion = receptor_motion + motion_noise
        
        # Apply strict mask limit if n_receptors < num_prune
        active_indices = torch.nonzero(prune_mask).squeeze()
        if active_indices.ndim == 0 and num_prune>0: active_indices=active_indices.unsqueeze(0)
        target_indices = active_indices[:n_receptors]
        
        final_mask = torch.zeros_like(prune_mask)
        final_mask[target_indices] = True

        # Apply Relocation (XYZ + Time + Motion)
        self.model.relocate(final_mask, new_xyz, timestamp, new_motion)
        
        # Reset Opacity to 0.01
        new_opacity = utils.inverse_sigmoid(torch.ones(n_receptors, device="cuda") * 0.01)
        with torch.no_grad():
            self.model._opacity[final_mask] = new_opacity.unsqueeze(1)
        
        # Reset optimizer state for relocated points
        # Essential to prevent momentum from moving them back or erratically
        self.optimizer.reset_optimizer_state(final_mask)
        
        # print(f"[Relocation] Relocated {n_receptors} gaussians at iter {iteration}")

    def _log_metrics(self, iteration: int, metrics: Dict[str, float]):
        super()._log_metrics(iteration, metrics)
        if self.writer:
            self.writer.add_histogram('params/t_scale_log', self.model._t_scale, iteration)


