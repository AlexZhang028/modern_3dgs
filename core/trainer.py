"""
Gaussian Splatting Trainer

Unified training framework supporting Static, FreeTime, and SharpTime modes.
"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm

from config.config import TrainerConfig, DataConfig
from utils.image_utils import psnr
from utils.general_utils import build_rotation
from core.loss import l1_loss, ssim, fast_ssim, GaussianLoss
from core.densify import (
    GaussianDensifier, DensificationConfig,
    DensificationScheduler, FreeTimeDensificationScheduler,
)
from data.samplers import DataSampler, StaticSampler, TemporalSampler

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
        self.lpips_enabled = config.lambda_lpips > 0 and self.loss_fn.lpips_model is not None
        # Move loss to device (important for LPIPS weights)
        if self.lpips_enabled:
            self.loss_fn = self.loss_fn.cuda()
        
        
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
        self.densifier = self._create_densifier()

        # Select fixed views for testing
        self.train_view_indices = self._select_fixed_views(len(dataset))
        self.test_view_indices = []
        if self.test_dataset:
            self.test_view_indices = self._select_fixed_views(len(test_dataset))
        
        self.logged_gt = False
        # Cache for per-view dynamic weight maps: {view_id: {'map': Tensor, 'iter': int}}
        self.dynamic_weight_cache = {}

        # Affine Color Correction (FreeTimeGS++): per-camera scale+bias to absorb
        # illumination inconsistencies, reducing run-to-run variance (Secret 5).
        self.cc_enabled = getattr(config, 'color_correction_enabled', False)
        if self.cc_enabled:
            self.color_correction = nn.ModuleDict({
                str(cam.uid): nn.ParameterList([
                    nn.Parameter(torch.ones(3, device='cuda')),   # scale, init=1
                    nn.Parameter(torch.zeros(3, device='cuda')),  # bias,  init=0
                ]) for cam in dataset.cameras
            })
            self.cc_optimizer = torch.optim.Adam(
                self.color_correction.parameters(),
                lr=getattr(config, 'cc_lr', 0.001)
            )
            print(f"Color Correction: enabled for {len(dataset.cameras)} cameras")
        else:
            self.color_correction = None
            self.cc_optimizer = None

    def _select_fixed_views(self, num_total_views: int) -> List[int]:
        """Select fixed indices for test views."""
        num_views = self.config.num_test_views
        if num_views == -1 or num_views >= num_total_views:
            return list(range(num_total_views))
        else:
            return torch.linspace(0, num_total_views - 1, num_views).round().long().tolist()
    
    def _create_densifier(self) -> DensificationScheduler:
        """Factory: build the densification scheduler for this trainer.
        Subclasses override to return a FreeTimeDensificationScheduler.
        """
        densify_config = DensificationConfig(
            densify_grad_threshold=self.config.densify_grad_threshold,
            densify_from_iter=self.config.densify_from_iter,
            densify_until_iter=self.config.densify_until_iter,
            densify_interval=self.config.densify_interval,
            prune_opacity_threshold=self.config.prune_opacity_threshold,
            prune_size_threshold=self.config.prune_size_threshold,
        )
        raw_densifier = GaussianDensifier(densify_config, self.model, self.optimizer)
        return DensificationScheduler(
            densifier      = raw_densifier,
            model          = self.model,
            optimizer      = self.optimizer,
            config         = self.config,
            scene_extent   = self.scene_extent,
            static_mask_fn = None,
        )

    def _apply_gradient_routing(self, iteration: int) -> None:
        """Hook: phase-based gradient zeroing. Override in FreeTimeTrainer."""
        pass

    def _create_sampler(self) -> DataSampler:
        num_workers = self.data_config.num_workers if self.data_config else 0
        return StaticSampler(self.dataset, num_workers=num_workers)
    
    # ----------------------------------------------------------------

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
        timing = {
            'sample_ms': 0.0,
            'lr_sh_ms': 0.0,
            'camera_to_cuda_ms': 0.0,
            'render_ms': 0.0,
            'dynamic_weight_render_ms': 0.0,
            'loss_ms': 0.0,
            'loss_hook_ms': 0.0,
            'depth_reg_ms': 0.0,
            'backward_ms': 0.0,
            'stats_update_ms': 0.0,
            'relocate_ms': 0.0,
            'densify_prune_ms': 0.0,
            'opacity_reset_ms': 0.0,
            'adaptive_control_ms': 0.0,
            'empty_cache_ms': 0.0,
            'optimizer_step_ms': 0.0,
        }

        step_start = time.perf_counter()
        # 3. Sample Data
        t0 = time.perf_counter()
        camera, timestamp = self.sampler.sample()
        timing['sample_ms'] = (time.perf_counter() - t0) * 1000.0

        # 1. Update Learning Rate
        # Must be done AFTER sampling time - Fixed: Velocity LR uses iteration now
        t0 = time.perf_counter()
        self.optimizer.update_learning_rate(iteration)
        
        # 2. Update SH Degree
        self._update_sh_degree(iteration)
        timing['lr_sh_ms'] = (time.perf_counter() - t0) * 1000.0
        
        # Ensure camera data is on GPU (safe for multiprocessing)
        # This handles both images and transformation matrices
        # IMPORTANT: Use copy to avoid modifying persistent dataset objects when using 0 workers
        import copy
        t0 = time.perf_counter()
        camera = copy.copy(camera)
        camera.to("cuda")
        timing['camera_to_cuda_ms'] = (time.perf_counter() - t0) * 1000.0
        
        # 4. Random Background (Optional)
        if self.config.random_background:
            bg_color = torch.rand(3, device="cuda")
        else:
            bg_color = self.bg_color
        
        # 5. Render
        t0 = time.perf_counter()
        rendered = self.renderer(
            gaussians=self.model,
            camera=camera,
            bg_color=bg_color,
            timestamp=timestamp,
            enable_culling=False  # Disable culling for stable training and proper densification stats
        )
        timing['render_ms'] = (time.perf_counter() - t0) * 1000.0
        
        # 6. Compute Loss
        t0 = time.perf_counter()
        target = camera.image.cuda()
        pred_img = rendered['render']

        # Affine Color Correction: pred = scale * pred + bias (per-camera)
        if self.cc_enabled and self.color_correction is not None:
            uid_key = str(camera.uid)
            if uid_key in self.color_correction:
                scale, bias = self.color_correction[uid_key]
                pred_img = pred_img * scale[:, None, None] + bias[:, None, None]

        pixel_loss_multiplier, dw_timing_ms, dw_metrics = self._compute_dynamic_weighting(
            iteration, camera, timestamp
        )
        timing['dynamic_weight_render_ms'] = dw_timing_ms

        # Handle Alpha Mask (Mask out background if necessary)
        if hasattr(camera, 'alpha_mask') and camera.alpha_mask is not None:
             alpha_mask = camera.alpha_mask.cuda()
             pred_img = pred_img * alpha_mask
             if isinstance(pixel_loss_multiplier, torch.Tensor):
                 pixel_loss_multiplier = pixel_loss_multiplier * alpha_mask

        # Base loss components
        if self.loss_fn.use_fused_ssim:
            ssim_val = fast_ssim(pred_img, target)
        else:
            ssim_val = ssim(pred_img, target)

        base_l1_val = l1_loss(pred_img, target)
        if isinstance(pixel_loss_multiplier, torch.Tensor):
            l1_diff_per_pixel = torch.abs(pred_img - target)
            weighted_l1_diff = l1_diff_per_pixel * pixel_loss_multiplier
            l1_loss_val = weighted_l1_diff.mean()
        else:
            l1_loss_val = base_l1_val

        loss = (1.0 - self.config.lambda_dssim) * l1_loss_val + self.config.lambda_dssim * (1.0 - ssim_val)

        lpips_metric = 0.0
        if self.lpips_enabled:
            pred_norm = pred_img * 2.0 - 1.0
            target_norm = target * 2.0 - 1.0
            lpips_tensor = self.loss_fn.lpips_model(pred_norm.unsqueeze(0), target_norm.unsqueeze(0)).mean()
            loss = loss + self.config.lambda_lpips * lpips_tensor
            lpips_metric = lpips_tensor.item()

        # CC regularization: penalize scale≠1 and bias≠0 to prevent collapse
        if self.cc_enabled and self.color_correction is not None:
            uid_key = str(camera.uid)
            if uid_key in self.color_correction:
                scale, bias = self.color_correction[uid_key]
                lambda_cc = getattr(self.config, 'lambda_cc', 0.001)
                loss = loss + lambda_cc * (((scale - 1.0) ** 2).mean() + (bias ** 2).mean())

        loss_components = {
            'total': loss,
            'l1_base': base_l1_val,
            'l1': l1_loss_val,
            'ssim': ssim_val,
            'lpips': lpips_metric,
        }
        timing['loss_ms'] = (time.perf_counter() - t0) * 1000.0
        
        # Loss Hook — expose camera/timestamp to subclass hooks via self
        self._current_camera = camera
        self._current_timestamp = timestamp
        t0 = time.perf_counter()
        loss = self._compute_loss_hook(loss, rendered, iteration)
        timing['loss_hook_ms'] = (time.perf_counter() - t0) * 1000.0

        # Depth regularization (if available)
        Ll1depth_pure = 0.0
        t0 = time.perf_counter()
        if hasattr(camera, 'depth_map') and camera.depth_map is not None and camera.depth_reliable:
            weight = self.depth_l1_weight(iteration)
            if weight > 0:
                invDepth = rendered["depth"]
                mono_invdepth = camera.depth_map.cuda()
                depth_mask = camera.depth_mask.cuda()
                
                Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
                loss += weight * Ll1depth_pure
        timing['depth_reg_ms'] = (time.perf_counter() - t0) * 1000.0
        
        # 7. Backward Pass (fine-grained timing)
        t_bw_total_start = time.perf_counter()
        # sync before backward to capture any pending CUDA work
        torch.cuda.synchronize()
        t_sync_before = time.perf_counter()
        timing['backward_sync_before_ms'] = (t_sync_before - t_bw_total_start) * 1000.0

        # actual autograd compute
        t_bw_compute_start = time.perf_counter()
        loss.backward()
        t_bw_compute_end = time.perf_counter()
        timing['backward_compute_ms'] = (t_bw_compute_end - t_bw_compute_start) * 1000.0

        # sync after backward to ensure kernels finished
        torch.cuda.synchronize()
        t_sync_after = time.perf_counter()
        timing['backward_sync_after_ms'] = (t_sync_after - t_bw_compute_end) * 1000.0

        # total backward time (preserves existing metric)
        timing['backward_ms'] = (t_sync_after - t_bw_total_start) * 1000.0

        # 7.5. Phase-based gradient routing (must stay in trainer: directly touches grad tensors)
        self._apply_gradient_routing(iteration)

        # 8. Adaptive Control (Densification, Pruning, Relocation)
        with torch.no_grad():
            t_block = time.perf_counter()
            t_stats = time.perf_counter()
            self.densifier.record_stats(rendered, iteration)
            timing['stats_update_ms'] = (time.perf_counter() - t_stats) * 1000.0
            timing.update(self.densifier.step(iteration))
            timing['adaptive_control_ms'] = (time.perf_counter() - t_block) * 1000.0
        
        # Periodic garbage collection for VRAM fragmentation 
        # (Moved outside densify block to prevent OOM/slowdowns from dynamic weighting double-renders after 30k)
        if iteration % (getattr(self.config, 'densify_interval', 100) * 5) == 0:
            t_cache = time.perf_counter()
            torch.cuda.empty_cache()
            timing['empty_cache_ms'] = (time.perf_counter() - t_cache) * 1000.0
        
        # 9. Optimizer Step
        if iteration < self.config.iterations:
            t_step = time.perf_counter()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            # Color Correction parameters use a separate Adam optimizer
            if self.cc_optimizer is not None:
                self.cc_optimizer.step()
                self.cc_optimizer.zero_grad(set_to_none=True)
            timing['optimizer_step_ms'] = (time.perf_counter() - t_step) * 1000.0
        
        # Metrics
        metrics = {
            'loss': loss.item(),
            'l1': loss_components['l1'].item(),
            'ssim': loss_components['ssim'].item(),
            'num_gaussians': self.model.num_points,
        }
        if 'lpips' in loss_components:
             val = loss_components['lpips']
             metrics['lpips'] = val.item() if hasattr(val, 'item') else val

        metrics.update(dw_metrics)
        metrics.update(timing)
        metrics['iteration_time'] = (time.perf_counter() - step_start) * 1000.0

        # Explicit cleanup to allow GC to reclaim graph immediately
        del rendered
        del loss
        del loss_components
        
        return metrics
    
    def _compute_loss_hook(self, loss: torch.Tensor, rendered: Dict, iteration: int) -> torch.Tensor:
        """Hook for additional loss computation."""
        return loss

    def _compute_dynamic_weighting(
        self, iteration: int, camera: Any, timestamp: Any
    ):
        """Hook: returns (pixel_loss_multiplier, timing_ms, extra_metrics).
        Override in FreeTimeTrainer for dynamic region focus weighting.
        """
        return 1.0, 0.0, {}

    def _get_eval_dynamic_map(self, camera: Any, iteration: int) -> Optional[torch.Tensor]:
        """Hook: return single-channel dynamic weight map [1,H,W] for eval, or None."""
        return None

    def _log_eval_tensorboard_extras(
        self, camera: Any, prediction: torch.Tensor, target: torch.Tensor,
        prefix: str, iteration: int
    ) -> None:
        """Hook: log FreeTime-specific TensorBoard images during eval."""
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

    def _compute_masked_psnr(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
        """Compute PSNR over a masked region."""
        if mask is None:
            return None

        mask = mask.float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape != prediction.shape:
            mask = mask.expand_as(prediction)

        if mask.sum().item() <= 0:
            return None

        mse = ((prediction - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
        if mse <= 0:
            return float('inf')

        return 20.0 * torch.log10(torch.tensor(1.0, device=prediction.device) / torch.sqrt(mse)).item()


    
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
            if 'l1_base' in metrics:
                self.writer.add_scalar('Loss/l1_base', metrics['l1_base'], iteration)
            self.writer.add_scalar('Loss/l1', metrics['l1'], iteration)
            self.writer.add_scalar('Loss/ssim', metrics['ssim'], iteration)
            if 'lpips' in metrics:
                self.writer.add_scalar('Loss/lpips', metrics['lpips'], iteration)
            if 'iteration_time' in metrics:
                self.writer.add_scalar('Stats/iteration_time', metrics['iteration_time'], iteration)
            self.writer.add_scalar('Stats/num_gaussians', metrics['num_gaussians'], iteration)
            self.writer.add_scalar('Stats/sh_degree', self.current_sh_degree, iteration)
            timing_keys = [
                'sample_ms',
                'lr_sh_ms',
                'camera_to_cuda_ms',
                'render_ms',
                'dynamic_weight_render_ms',
                'loss_ms',
                'loss_hook_ms',
                'depth_reg_ms',
                'backward_ms',
                'backward_sync_before_ms',
                'backward_compute_ms',
                'backward_sync_after_ms',
                'stats_update_ms',
                'relocate_ms',
                'densify_prune_ms',
                'opacity_reset_ms',
                'adaptive_control_ms',
                'empty_cache_ms',
                'optimizer_step_ms',
            ]
            for key in timing_keys:
                if key in metrics:
                    self.writer.add_scalar(f'Timing/{key}', metrics[key], iteration)
            if 'dynamic_boost' in metrics:
                self.writer.add_scalar('DynamicWeight/boost', metrics['dynamic_boost'], iteration)
            if 'dynamic_map_mean' in metrics:
                self.writer.add_scalar('DynamicWeight/map_mean', metrics['dynamic_map_mean'], iteration)
            if 'dynamic_map_max' in metrics:
                self.writer.add_scalar('DynamicWeight/map_max', metrics['dynamic_map_max'], iteration)
            if 'dynamic_multiplier_mean' in metrics:
                self.writer.add_scalar('DynamicWeight/multiplier_mean', metrics['dynamic_multiplier_mean'], iteration)
            if 'dynamic_multiplier_max' in metrics:
                self.writer.add_scalar('DynamicWeight/multiplier_max', metrics['dynamic_multiplier_max'], iteration)

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

    def _render_for_eval(self, camera, bg_color: torch.Tensor, **kwargs):
        return self.renderer(gaussians=self.model, camera=camera, bg_color=bg_color, **kwargs)

    def _evaluate_set(self, dataset, indices: List[int], prefix: str, iteration: int):
        """Evaluate a specific dataset subset."""
        psnr_list = []
        l1_list = []
        ssim_list = []
        masked_psnr_dynamic_list = []
        masked_psnr_static_list = []
        
        print(f"   Evaluating {prefix} set ({len(indices)} images)...")
        
        with torch.no_grad():
            for idx in tqdm(indices, desc=f"Evaluating {prefix}", leave=False):
                sample = dataset[idx]
                camera = sample["camera"]
                camera.to("cuda")
                
                # Render
                rendered = self._render_for_eval(
                    camera=camera,
                    bg_color=self.bg_color,
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

                dynamic_weight_map = self._get_eval_dynamic_map(camera, iteration)
                if dynamic_weight_map is not None:
                    dynamic_mask_threshold = getattr(self.config, 'dynamic_mask_threshold', 0.5)
                    dynamic_mask = (dynamic_weight_map >= dynamic_mask_threshold).float()
                    if hasattr(camera, 'alpha_mask') and camera.alpha_mask is not None:
                        valid_mask = camera.alpha_mask.cuda()[0:1]
                        dynamic_mask = dynamic_mask * valid_mask
                        static_mask = (1.0 - dynamic_mask) * valid_mask
                    else:
                        static_mask = 1.0 - dynamic_mask

                    masked_psnr_dynamic = self._compute_masked_psnr(prediction, target, dynamic_mask)
                    masked_psnr_static = self._compute_masked_psnr(prediction, target, static_mask)
                    if masked_psnr_dynamic is not None:
                        masked_psnr_dynamic_list.append(masked_psnr_dynamic)
                    if masked_psnr_static is not None:
                        masked_psnr_static_list.append(masked_psnr_static)
                
                # TensorBoard Images
                if self.writer:
                    # GT only on first log
                    if not self.logged_gt:
                        self.writer.add_image(f'{prefix}_GT/{camera.image_name}', target, iteration)
                    
                    # Render result on every log
                    self.writer.add_image(f'{prefix}_Render/{camera.image_name}', prediction, iteration)

                    self._log_eval_tensorboard_extras(camera, prediction, target, prefix, iteration)

                    # 5. Gradient Contribution Map
                    with torch.enable_grad():
                        rendered_grad = self._render_for_eval(
                            camera=camera,
                            bg_color=self.bg_color,
                            enable_culling=False
                        )
                        pred_grad = rendered_grad['render']
                        
                        if hasattr(camera, 'alpha_mask') and camera.alpha_mask is not None:
                            alpha_mask = camera.alpha_mask.cuda()
                            pred_grad = pred_grad * alpha_mask

                        loss_comp = self.loss_fn.get_components(pred_grad, target)
                        loss_grad = loss_comp['total']
                        
                        if hasattr(camera, 'depth_map') and camera.depth_map is not None and camera.depth_reliable:
                            weight = self.depth_l1_weight(iteration)
                            if weight > 0:
                                invDepth = rendered_grad["depth"]
                                mono_invdepth = camera.depth_map.cuda()
                                depth_mask = camera.depth_mask.cuda()
                                Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
                                loss_grad += weight * Ll1depth_pure
                        
                        loss_grad.backward()
                        
                        viewspace_points = rendered_grad['viewspace_points']
                        grad_color = None
                        if viewspace_points.grad is not None:
                            import matplotlib.cm as cm
                            import numpy as np
                            
                            grad_norm = torch.norm(viewspace_points.grad[:, :2], dim=-1, keepdim=True)
                            max_grad = self.config.densify_grad_threshold
                            grad_norm_normalized = torch.clamp(grad_norm / (max_grad + 1e-5), 0.0, 1.0).squeeze(-1)
                            
                            # Apply plasma colormap
                            plasma_map = cm.get_cmap('plasma')
                            grad_color_np = plasma_map(grad_norm_normalized.detach().cpu().numpy())[:, :3]
                            grad_color = torch.from_numpy(grad_color_np).to(dtype=torch.float32, device="cuda")
                        
                        self.optimizer.zero_grad(set_to_none=True)
                        self.model.zero_grad(set_to_none=True)
                    
                    if grad_color is not None:
                        rendered_vis = self._render_for_eval(
                            camera=camera,
                            bg_color=torch.tensor([0.0, 0.0, 0.0], device="cuda"),
                            enable_culling=False,
                            colors_override=grad_color
                        )
                        self.writer.add_image(f'{prefix}_GradMap/{camera.image_name}', rendered_vis['render'].clamp(0.0, 1.0), iteration)

                    if dynamic_weight_map is not None:
                        try:
                            import matplotlib.cm as cm

                            w_np = dynamic_weight_map.squeeze(0).cpu().numpy()
                            cmap = cm.get_cmap('plasma')
                            rgba = cmap(w_np)[:, :, :3]
                            rgb = torch.from_numpy(rgba).permute(2, 0, 1).to(dtype=torch.float32)
                            self.writer.add_image(f'{prefix}_DynamicWeight/{camera.image_name}', rgb, iteration)
                        except Exception:
                            self.writer.add_image(f'{prefix}_DynamicWeight/{camera.image_name}', dynamic_weight_map, iteration)


        
        # Compute Averages
        avg_l1 = torch.tensor(l1_list).mean().item()
        avg_psnr = torch.tensor(psnr_list).mean().item()
        avg_ssim = torch.tensor(ssim_list).mean().item()

        avg_masked_psnr_dynamic = torch.tensor(masked_psnr_dynamic_list).mean().item() if masked_psnr_dynamic_list else None
        avg_masked_psnr_static = torch.tensor(masked_psnr_static_list).mean().item() if masked_psnr_static_list else None
        
        if avg_masked_psnr_dynamic is not None and avg_masked_psnr_static is not None:
            print(
                f"   {prefix} Results - L1: {avg_l1:.4f} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | "
                f"MaskedDynPSNR: {avg_masked_psnr_dynamic:.4f} | MaskedStaPSNR: {avg_masked_psnr_static:.4f}"
            )
        else:
            print(f"   {prefix} Results - L1: {avg_l1:.4f} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f}")
        
        if self.writer:
            self.writer.add_scalar(f'{prefix}/l1', avg_l1, iteration)
            self.writer.add_scalar(f'{prefix}/psnr', avg_psnr, iteration)
            self.writer.add_scalar(f'{prefix}/ssim', avg_ssim, iteration)
            if avg_masked_psnr_dynamic is not None:
                self.writer.add_scalar(f'{prefix}/masked_psnr_dynamic', avg_masked_psnr_dynamic, iteration)
            if avg_masked_psnr_static is not None:
                self.writer.add_scalar(f'{prefix}/masked_psnr_static', avg_masked_psnr_static, iteration)

    
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
            if self.color_correction is not None:
                checkpoint['color_correction_state_dict'] = self.color_correction.state_dict()
            if self.cc_optimizer is not None:
                checkpoint['cc_optimizer_state_dict'] = self.cc_optimizer.state_dict()
            
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
        if self.color_correction is not None and 'color_correction_state_dict' in checkpoint:
            self.color_correction.load_state_dict(checkpoint['color_correction_state_dict'])
        if self.cc_optimizer is not None and 'cc_optimizer_state_dict' in checkpoint:
            self.cc_optimizer.load_state_dict(checkpoint['cc_optimizer_state_dict'])
        
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
    """FreeTimeGS trainer. Extends Trainer with all temporal-specific logic."""

    # ------------------------------------------------------------------ #
    #  Sampler                                                             #
    # ------------------------------------------------------------------ #

    def _create_sampler(self):
        num_workers = self.data_config.num_workers if self.data_config else 0
        return TemporalSampler(self.dataset, num_workers=num_workers)

    # ------------------------------------------------------------------ #
    #  Static / dynamic classification                                     #
    # ------------------------------------------------------------------ #

    def _get_static_mask(self) -> torch.Tensor:
        """Boolean mask [N]: True = static/persistent, False = dynamic/transient."""
        dur_thresh = getattr(self.config, 'decouple_static_duration_threshold', 0.5)
        return self.model.get_t_scale.squeeze() > dur_thresh

    def _get_dynamic_score(self) -> torch.Tensor:
        """Per-Gaussian dynamism score [N,1] in [0,1]: 0=static, 1=dynamic."""
        duration    = self.model.get_t_scale.detach()
        static_dur  = getattr(self.config, 'dynamic_duration_static',  0.5)
        dynamic_dur = getattr(self.config, 'dynamic_duration_dynamic', 0.1)
        denom = max(static_dur - dynamic_dur, 1e-6)
        return torch.clamp((static_dur - duration) / denom, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    #  Densifier                                                           #
    # ------------------------------------------------------------------ #

    def _create_densifier(self) -> FreeTimeDensificationScheduler:
        densify_config = DensificationConfig(
            densify_grad_threshold=self.config.densify_grad_threshold,
            densify_from_iter=self.config.densify_from_iter,
            densify_until_iter=self.config.densify_until_iter,
            densify_interval=self.config.densify_interval,
            prune_opacity_threshold=self.config.prune_opacity_threshold,
            prune_size_threshold=self.config.prune_size_threshold,
        )
        raw_densifier = GaussianDensifier(densify_config, self.model, self.optimizer)
        return FreeTimeDensificationScheduler(
            densifier      = raw_densifier,
            model          = self.model,
            optimizer      = self.optimizer,
            config         = self.config,
            scene_extent   = self.scene_extent,
            static_mask_fn = self._get_static_mask,
        )

    # ------------------------------------------------------------------ #
    #  Gradient routing (phase-based decoupled training)                  #
    # ------------------------------------------------------------------ #

    def _apply_gradient_routing(self, iteration: int) -> None:
        if not getattr(self.config, 'decouple_training_enabled', False):
            return

        joint_end   = getattr(self.config, 'decouple_joint_end_iter',   15000)
        dynamic_end = getattr(self.config, 'decouple_dynamic_end_iter', 30000)

        # Phase 1 (joint) or past Phase 2 (joint refinement): no routing
        if iteration <= joint_end or iteration > dynamic_end:
            return

        # Phase 2 only: zero static Gaussian gradients so densification
        # and optimization focus on dynamic Gaussians.
        with torch.no_grad():
            is_static = self._get_static_mask()
            if not is_static.any():
                return
            for key in ['xyz', 'features_dc', 'features_rest', 'opacity', 'scaling', 'rotation']:
                p = self.model._gaussian_params.get(key)
                if p is not None and p.grad is not None:
                    p.grad[is_static] = 0.0

    # ------------------------------------------------------------------ #
    #  Dynamic region focus weighting                                      #
    # ------------------------------------------------------------------ #

    def _compute_dynamic_weighting(self, iteration: int, camera: Any, timestamp: Any):
        use_dynamic_weighting = getattr(self.config, 'dynamic_weighting_enabled', False)
        if not use_dynamic_weighting:
            return 1.0, 0.0, {}

        boost_start_iter  = getattr(self.config, 'dynamic_boost_start_iter', 15000)
        boost_end_iter    = getattr(self.config, 'dynamic_boost_end_iter',   20000)
        max_dynamic_boost = getattr(self.config, 'max_dynamic_boost', 5.0)
        curve_power       = max(getattr(self.config, 'dynamic_boost_curve_power', 3.0), 1e-6)

        if iteration < boost_start_iter:
            return 1.0, 0.0, {'dynamic_boost': 0.0}

        if iteration < boost_end_iter and boost_end_iter > boost_start_iter:
            progress       = (iteration - boost_start_iter) / float(boost_end_iter - boost_start_iter)
            eased_progress = (math.exp(curve_power * progress) - 1.0) / (math.exp(curve_power) - 1.0)
            current_boost  = max_dynamic_boost * eased_progress
        else:
            current_boost = max_dynamic_boost

        if current_boost <= 0.0:
            return 1.0, 0.0, {'dynamic_boost': 0.0}

        t0   = time.perf_counter()
        mask = self._load_sam2_mask(camera)
        if mask is None:
            # No SAM2 masks configured or available: skip boost for this view
            return 1.0, (time.perf_counter() - t0) * 1000.0, {
                'dynamic_boost': current_boost, 'dynamic_map_mean': 0.0,
            }

        # mask: [1, H, W] float32 in [0, 1], 1 = dynamic pixel
        # multiplier: 1.0 in static regions, (1 + boost) in dynamic regions
        multiplier = 1.0 + current_boost * mask
        timing_ms  = (time.perf_counter() - t0) * 1000.0
        return multiplier, timing_ms, {
            'dynamic_boost':           current_boost,
            'dynamic_map_mean':        mask.mean().item(),
            'dynamic_multiplier_mean': multiplier.mean().item(),
            'dynamic_multiplier_max':  multiplier.max().item(),
        }

    def _load_sam2_mask(self, camera: Any) -> Optional[torch.Tensor]:
        """Load SAM2 dynamic mask for a camera, resized to training resolution.

        Mask files follow the sharptime_init convention:
            {sam2_masks_dir}/{cam_name}/{eff_idx:05d}.png
        where cam_name and eff_idx are encoded in camera.image_name as
            "{cam_name}_frame_{eff_idx:05d}".

        Returns a [1, H, W] float32 tensor on the model device (0=static,
        1=dynamic), or None when masks are unavailable or not configured.
        Results are cached as uint8 numpy arrays to bound memory usage.
        """
        import cv2 as _cv2

        # Lazy init of cache and configured directory
        if not hasattr(self, '_sam2_mask_cache'):
            self._sam2_mask_cache: Dict[Tuple, Optional[np.ndarray]] = {}
            self._sam2_masks_dir: Optional[str] = (
                getattr(self.config, 'sam2_masks_dir', '') or None
            )
            if self._sam2_masks_dir:
                print(f"[trainer] SAM2 mask boost enabled: {self._sam2_masks_dir}")

        if not self._sam2_masks_dir:
            return None

        img_name = getattr(camera, 'image_name', '') or ''
        if '_frame_' not in img_name:
            return None
        cam_name, eff_str = img_name.rsplit('_frame_', 1)
        try:
            eff_idx = int(eff_str)
        except ValueError:
            return None

        target_h, target_w = camera.height, camera.width
        cache_key: Tuple = (cam_name, eff_idx, target_h, target_w)

        if cache_key in self._sam2_mask_cache:
            cached = self._sam2_mask_cache[cache_key]
            if cached is None:
                return None
            return torch.from_numpy(cached.astype(np.float32) / 255.0).unsqueeze(0).to(self.model.device)

        mask_path = Path(self._sam2_masks_dir) / cam_name / f"{eff_idx:05d}.png"
        if not mask_path.exists():
            self._sam2_mask_cache[cache_key] = None
            return None

        mask_u8 = _cv2.imread(str(mask_path), _cv2.IMREAD_GRAYSCALE)
        if mask_u8 is None:
            self._sam2_mask_cache[cache_key] = None
            return None

        if mask_u8.shape != (target_h, target_w):
            mask_u8 = _cv2.resize(mask_u8, (target_w, target_h), interpolation=_cv2.INTER_NEAREST)

        self._sam2_mask_cache[cache_key] = mask_u8
        return torch.from_numpy(mask_u8.astype(np.float32) / 255.0).unsqueeze(0).to(self.model.device)

    # ------------------------------------------------------------------ #
    #  Loss hook                                                           #
    # ------------------------------------------------------------------ #

    def _compute_opacity_reg(self, loss: torch.Tensor, rendered: Dict) -> torch.Tensor:
        """L_opacity = λ_reg · mean(O · stopGrad(l(t))). Shared by FreeTime and SharpTime."""
        l_reg = (rendered['base_opacity'] * rendered['temporal_weight'].detach()).mean()
        return loss + self.config.lambda_reg * l_reg

    def _compute_loss_hook(self, loss: torch.Tensor, rendered: Dict, iteration: int) -> torch.Tensor:
        loss = self._compute_opacity_reg(loss, rendered)

        if iteration > getattr(self.config, 'motion_blur_start_iter', 15000) \
                and hasattr(self.model, '_motion') and self.model._motion is not None:
            speed = torch.norm(self.model._motion, dim=-1).detach()
            duration = torch.exp(self.model._t_scale).squeeze(-1)
            fast_mask = speed > getattr(self.config, 'motion_blur_speed_threshold', 0.5)
            if fast_mask.any():
                loss += getattr(self.config, 'lambda_motion_blur', 0.05) * \
                        (speed[fast_mask] * duration[fast_mask]).mean()
        return loss

    # ------------------------------------------------------------------ #
    #  Evaluation helpers                                                  #
    # ------------------------------------------------------------------ #

    def _render_for_eval(self, camera, bg_color: torch.Tensor, **kwargs):
        ts = getattr(camera, 'timestamp', None)
        kwargs.setdefault('timestamp', 0.0 if ts is None else ts)
        return super()._render_for_eval(camera, bg_color, **kwargs)

    def _get_eval_dynamic_map(self, camera: Any, iteration: int) -> Optional[torch.Tensor]:
        if not (
            getattr(self.config, 'dynamic_weighting_enabled', False)
            and iteration >= getattr(self.config, 'dynamic_boost_start_iter', 15000)
        ):
            return None
        with torch.no_grad():
            override_colors = self._get_dynamic_score().repeat(1, 3)
            out = self._render_for_eval(
                camera=camera, bg_color=torch.zeros(3, device="cuda"),
                enable_culling=False, colors_override=override_colors,
            )
            result = out['render'][0:1].detach().clamp(0.0, 1.0)
            del out
        return result

    def _log_eval_tensorboard_extras(
        self, camera: Any, prediction: torch.Tensor, target: torch.Tensor,
        prefix: str, iteration: int
    ) -> None:
        if not (self.writer and getattr(self.config, 'dynamic_weighting_enabled', False)):
            return
        with torch.no_grad():
            override_colors = self._get_dynamic_score().repeat(1, 3)
            out = self._render_for_eval(
                camera=camera, bg_color=torch.zeros(3, device="cuda"),
                enable_culling=False, colors_override=override_colors,
            )
            dmap = out['render'][0:1].detach().clamp(0.0, 1.0)
            del out
        try:
            import matplotlib.cm as cm
            import numpy as np
            rgba = cm.get_cmap('plasma')(dmap.squeeze(0).cpu().numpy())[:, :, :3]
            rgb = torch.from_numpy(rgba).permute(2, 0, 1).to(dtype=torch.float32)
            self.writer.add_image(f'{prefix}_DynamicWeight/{camera.image_name}', rgb, iteration)
        except Exception:
            self.writer.add_image(f'{prefix}_DynamicWeight/{camera.image_name}', dmap, iteration)

    # ------------------------------------------------------------------ #
    #  Metrics logging                                                     #
    # ------------------------------------------------------------------ #

    def _log_metrics(self, iteration: int, metrics: Dict[str, float]):
        super()._log_metrics(iteration, metrics)
        if not self.writer:
            return
        # Temporal parameter distributions
        self.writer.add_scalar('stats/gaussian_t_min', self.model.get_t.min().item(), iteration)
        self.writer.add_scalar('stats/gaussian_t_max', self.model.get_t.max().item(), iteration)
        self.writer.add_histogram('params/t_scale_log', self.model._t_scale, iteration)
        if hasattr(self.model, '_motion') and self.model._motion is not None:
            speed = torch.norm(self.model._motion, dim=-1)
            self.writer.add_histogram('params/velocity_norm', speed, iteration)
            self.writer.add_scalar('stats/speed_max',  speed.max().item(),  iteration)
            self.writer.add_scalar('stats/speed_mean', speed.mean().item(), iteration)
        # Phase-based decoupled training stats
        if getattr(self.config, 'decouple_training_enabled', False):
            with torch.no_grad():
                n_static  = int(self._get_static_mask().sum().item())
                n_dynamic = self.model.num_points - n_static
            self.writer.add_scalar('Decouple/n_static',  n_static,  iteration)
            self.writer.add_scalar('Decouple/n_dynamic', n_dynamic, iteration)
            joint_end   = getattr(self.config, 'decouple_joint_end_iter',   15000)
            dynamic_end = getattr(self.config, 'decouple_dynamic_end_iter', 30000)
            phase = 1 if iteration <= joint_end else (2 if iteration <= dynamic_end else 3)
            self.writer.add_scalar('Decouple/phase', phase, iteration)


class SharpTimeTrainer(FreeTimeTrainer):
    """
    Trainer for SharpTimeGS (arXiv 2602.02989).

    Inherits all FreeTimeGS training infrastructure (temporal sampler,
    gradient routing, dynamic weighting, MCMC relocation, etc.) and adds:

      - L_t: lifespan regularization that encourages Gaussians to extend
             their active duration, reducing fragmentation into many
             short-lived primitives.
    """

    # ------------------------------------------------------------------ #
    #  Loss hook                                                           #
    # ------------------------------------------------------------------ #

    def _compute_loss_hook(
        self, loss: torch.Tensor, rendered: Dict, iteration: int
    ) -> torch.Tensor:
        # L_opacity only — skip FreeTimeGS motion-blur penalty (not part of SharpTimeGS)
        loss = self._compute_opacity_reg(loss, rendered)

        # L_scale: PGSR flatness regularization
        # L_scale = 1/N Σ min(s_i)  — penalizes the thinnest axis being non-zero,
        # pushing Gaussians toward flat disc/plane shapes on the surface.
        lambda_scale = getattr(self.config, 'lambda_scale', 0.0)
        if lambda_scale > 0.0:
            scales  = self.model.get_scaling           # [N, 3], exp-activated
            L_scale = scales.min(dim=-1).values.mean()
            loss   += lambda_scale * L_scale

        # L_t: lifespan regularization (SharpTimeGS eq. 7-8)
        # L_t = 1/N Σ 1/√(-2·log(o_th)·σ_t² + r)
        # Minimising this pushes σ_t and r to grow → longer-lived primitives.
        lambda_t = getattr(self.config, 'lambda_lifespan', 0.0)
        if lambda_t > 0.0:
            o_th    = getattr(self.config, 'prune_opacity_threshold', 0.005)
            sigma_t = self.model.get_t_scale  # [N,1], already exp-activated
            r       = self.model.get_r        # [N,1], already softplus-activated
            a       = -2.0 * math.log(max(float(o_th), 1e-8))
            denom   = (a * sigma_t ** 2 + r).clamp(min=1e-8)
            L_t     = (1.0 / torch.sqrt(denom)).mean()
            loss    += lambda_t * L_t

        # L_n: single-view normal–depth consistency
        lambda_n = getattr(self.config, 'lambda_n', 0.0)
        if lambda_n > 0.0:
            camera    = getattr(self, '_current_camera', None)
            timestamp = getattr(self, '_current_timestamp', 0.0)
            if camera is not None:
                L_n   = self._compute_ln(rendered, camera, timestamp)
                loss += lambda_n * L_n

        return loss

    # ------------------------------------------------------------------ #
    #  L_n helpers                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _depth_to_normal(depth: torch.Tensor, camera) -> torch.Tensor:
        """Finite-difference surface normals from depth map (camera space, detached)."""
        H, W = depth.shape[1:]
        fx = W / (2.0 * math.tan(camera.FovX / 2.0))
        fy = H / (2.0 * math.tan(camera.FovY / 2.0))
        cx, cy = W * 0.5, H * 0.5

        u = torch.arange(W, device=depth.device, dtype=torch.float32)
        v = torch.arange(H, device=depth.device, dtype=torch.float32)
        vv, uu = torch.meshgrid(v, u, indexing='ij')   # [H, W] each

        z   = depth[0]                                  # [H, W]
        pts = torch.stack([
            (uu - cx) / fx * z,
            (vv - cy) / fy * z,
            z,
        ], dim=0)                                       # [3, H, W]

        # Central differences → cross product → unit normal
        du = F.pad(pts[:, :, 2:] - pts[:, :, :-2], (1, 1))         # [3, H, W]
        dv = F.pad(pts[:, 2:, :] - pts[:, :-2, :], (0, 0, 1, 1))   # [3, H, W]
        n  = torch.cross(du, dv, dim=0)
        return F.normalize(n, dim=0).detach()           # target → no grad

    def _get_gaussian_normals_cam(self, camera) -> torch.Tensor:
        """Per-Gaussian normals (smallest-scale axis) transformed to camera space."""
        R_mat   = build_rotation(self.model.get_rotation)  # [N, 3, 3]
        scales  = self.model.get_scaling                   # [N, 3]
        min_idx = scales.argmin(dim=-1)                    # [N]
        N       = R_mat.shape[0]
        normals_w = R_mat[torch.arange(N, device=R_mat.device), :, min_idx]  # [N, 3]

        # world_view_transform is W2V^T; .T gives actual W2V [4,4]
        R_w2c     = camera.world_view_transform.T[:3, :3].float()
        normals_c = normals_w @ R_w2c.T                    # [N, 3]
        return F.normalize(normals_c, dim=-1)

    def _compute_ln(
        self, rendered: Dict, camera, timestamp: float
    ) -> torch.Tensor:
        """L_n = mean(1 - dot(rendered_normal, depth_normal))."""
        depth_normal = self._depth_to_normal(rendered['depth'], camera)  # [3,H,W]

        # Map Gaussian normals [-1,1] → [0,1] for rasterizer, then back
        normals_color = (self._get_gaussian_normals_cam(camera) + 1.0) * 0.5  # [N,3]

        normal_map = self.renderer(
            gaussians=self.model,
            camera=camera,
            bg_color=torch.zeros(3, device='cuda'),
            timestamp=timestamp,
            colors_override=normals_color,
            enable_culling=False,
        )['render']                                      # [3, H, W]
        normal_map = F.normalize(normal_map * 2.0 - 1.0, dim=0)  # back to [-1,1]

        return (1.0 - (normal_map * depth_normal).sum(dim=0)).mean()

    # ------------------------------------------------------------------ #
    #  Metrics logging                                                     #
    # ------------------------------------------------------------------ #

    def _log_metrics(self, iteration: int, metrics: Dict[str, float]):
        super()._log_metrics(iteration, metrics)
        if self.writer:
            r = self.model.get_r  # softplus-activated
            self.writer.add_scalar('params/r_mean', r.mean().item(), iteration)
            self.writer.add_scalar('params/r_max',  r.max().item(),  iteration)
            self.writer.add_histogram('params/r', r, iteration)

