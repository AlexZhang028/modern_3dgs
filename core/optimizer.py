"""
Gaussian Splatting Optimizer

Handles dynamic optimization of Gaussian parameters, including state updates during densification and pruning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .gaussian_model import GaussianModel
from config.config import OptimConfig
from utils.general_utils import get_expon_lr_func


class GaussianOptimizer:
    """
    Optimizer for GaussianModel.
    
    Features:
    - Different learning rates for different parameters.
    - Support for dynamic addition/removal of parameters (densification/pruning).
    - Automatic updates of Adam optimizer state.
    """
    
    def __init__(
        self,
        model: GaussianModel,
        config: OptimConfig
    ):
        self.model = model
        self.config = config
        
        # Create parameter groups
        self.param_groups = self._create_param_groups()
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.param_groups,
            lr=0.0,  # LR is set via param_groups
            betas=config.betas,
            eps=config.eps
        )
        
        # Learning Rate Scheduler
        self.xyz_scheduler = get_expon_lr_func(
            lr_init=config.position_lr_init * model.spatial_lr_scale,
            lr_final=config.position_lr_final * model.spatial_lr_scale,
            lr_delay_mult=config.position_lr_delay_mult,
            max_steps=config.position_lr_max_steps
        )

        # Velocity Learning Rate Scheduler (FreeTimeGS specific)
        # Formula: Lr = Init^(1-t) * Final^t, where t = timestamp
        # Note: Strictly exponential, no warmup/delay.
        def get_velocity_scheduler(initial_lr, final_lr):
            def scheduler(progress):
                alpha = progress # Progress should be 0.0 to 1.0
                # Exponential decay: lr = init * (factor ^ alpha) 
                # where factor = final / init
                # or lr = init^(1-alpha) * final^alpha
                return (initial_lr ** (1.0 - alpha)) * (final_lr ** alpha)
            return scheduler
        
        # Initial: 1.6e-4, Final: 1.6e-6 (factor 0.01)
        # Scaled by spatial_lr_scale as it is a motion (spatial) parameter
        vel_lr_init = config.velocity_lr * model.spatial_lr_scale
        vel_lr_final = (config.velocity_lr * 0.5) * model.spatial_lr_scale
        
        self.velocity_scheduler = get_velocity_scheduler(
            initial_lr=vel_lr_init,
            final_lr=vel_lr_final
        )
        
        print(f"GaussianOptimizer Initialized")
        print(f"   Param Groups: {len(self.param_groups)}")
        print(f"   Initial Pos LR: {config.position_lr_init * model.spatial_lr_scale:.6f}")

    def reset_optimizer_state(self, mask: torch.Tensor):
        """
        Reset Adam optimizer state (momentum) for specific Gaussian indices.
        Used after relocation or pruning.
        
        Args:
            mask: Boolean mask of Gaussians to reset [N]
        """
        for group in self.optimizer.param_groups:
            # Assumes the first parameter in the group is the one determining the size (N)
            # Typically group['params'] is a list of Tensors.
            # In GaussianModel, params are often [N, 1] or [N, 3] etc.
            if len(group["params"]) == 0:
                continue
                
            param = group["params"][0]
            if param.shape[0] != mask.shape[0]:
                continue
                
            state = self.optimizer.state.get(param, None)
            if state is not None:
                # Zero out momentum (exp_avg) and variance (exp_avg_sq)
                if "exp_avg" in state:
                    state["exp_avg"][mask] = 0.0
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"][mask] = 0.0
    
    def _create_param_groups(self) -> List[Dict]:
        """Create parameter groups (different LR for each parameter)."""
        return self.model.get_param_groups(self.config)
    
    def step(self):
        """Perform a single optimization step."""
        self.optimizer.step()
    
    def zero_grad(self, set_to_none: bool = True):
        """Zero the gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def update_learning_rate(self, iteration: int):
        """
        Update learning rates.
        
        Args:
            iteration: Current iteration number.
        """
        # Update position learning rate (exponential decay)
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "velocity":
                # FreeTimeGS Fix: Anneal based on training progress, NOT video timestamp.
                # Max steps is typically config.iterations (e.g. 30,000)
                max_steps = self.config.iterations
                # Calculate normalized training time t (0.0 -> 1.0)
                train_progress_t = min(iteration / max_steps, 1.0)
                
                lr = self.velocity_scheduler(train_progress_t)
                param_group['lr'] = lr
    
    def reset_optimizer_state(self, mask: torch.Tensor):
        """
        Reset Adam optimizer state (momentum) for specific Gaussian indices.
        Used after relocation or pruning.
        
        Args:
            mask: Boolean mask of Gaussians to reset [N]
        """
        for group in self.optimizer.param_groups:
            # group['params'][0] shape dim 0 should match gaussians count
            if len(group["params"]) != 1:
                continue
                
            param = group["params"][0]
            if param.shape[0] != mask.shape[0]:
                continue
                
            state = self.optimizer.state.get(param, None)
            if state is not None:
                # Reset momentum (exp_avg) and variance (exp_avg_sq)
                # This prevents relocated points from being "flung" away by old momentum
                if "exp_avg" in state:
                    state["exp_avg"][mask] = 0.0
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"][mask] = 0.0

    def state_dict(self) -> Dict:
        """Get optimizer state dict."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict: Dict):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)
    
    # ==================== Densification Support ====================
    
    def replace_tensor_to_optimizer(self, tensor: torch.Tensor, name: str):
        """
        Replace tensor in optimizer (for densification/pruning).
        
        Args:
            tensor: New tensor.
            name: Parameter name.
        """
        optimizable_tensors = {}
        
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                
                # If history exists, reset it for the new tensor
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.zeros_like(tensor)
                    stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors
    
    def prune_optimizer(self, mask: torch.Tensor):
        """
        Remove pruned Gaussians from optimizer.
        
        Args:
            mask: Boolean mask, True indicates keep.
        """
        optimizable_tensors = {}
        
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if stored_state is not None:
                # Apply mask to history state
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                
                del self.optimizer.state[group['params'][0]]
                
                # Update parameter
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                self.optimizer.state[group['params'][0]] = stored_state
                
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors
    
    def cat_tensors_to_optimizer(self, tensors_dict: Dict[str, torch.Tensor]):
        """
        Add new Gaussians to optimizer (densification).
        
        Args:
            tensors_dict: {param_name: new_tensor}
        """
        optimizable_tensors = {}
        
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            
            if group["name"] not in tensors_dict:
                 continue

            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if stored_state is not None:
                # Extend history state
                stored_state["exp_avg"] = torch.cat([
                    stored_state["exp_avg"],
                    torch.zeros_like(extension_tensor)
                ], dim=0)
                
                stored_state["exp_avg_sq"] = torch.cat([
                    stored_state["exp_avg_sq"],
                    torch.zeros_like(extension_tensor)
                ], dim=0)
                
                del self.optimizer.state[group['params'][0]]
                
                # Concatenate parameter
                group["params"][0] = nn.Parameter(
                    torch.cat([group["params"][0], extension_tensor], dim=0).requires_grad_(True)
                )
                self.optimizer.state[group['params'][0]] = stored_state
                
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat([group["params"][0], extension_tensor], dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors


def setup_optimizer(
    model: GaussianModel,
    config: Optional[OptimConfig] = None
) -> GaussianOptimizer:
    """
    Convenience function: Create optimizer.
    
    Args:
        model: GaussianModel
        config: Optimizer config (if None, use default)
        
    Returns:
        GaussianOptimizer
    """
    if config is None:
        config = OptimConfig()
    
    return GaussianOptimizer(model, config)
