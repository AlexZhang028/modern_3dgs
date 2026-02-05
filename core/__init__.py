"""
Core module - Gaussian Models, Optimizers, Renderers, Losses, Densifiers, and Trainers
"""

from .gaussian_model import GaussianModel, FreeTimeGaussianModel, create_model_from_config
from .optimizer import GaussianOptimizer, setup_optimizer
from .renderer import GaussianRenderer
from .loss import (
    GaussianLoss, 
    l1_loss, 
    l2_loss, 
    ssim, 
    fast_ssim,
    compute_loss
)
from .densify import GaussianDensifier
from .trainer import Trainer, DataSampler, StaticSampler, TemporalSampler

# Import all config classes from config
from config.config import (
    ModelConfig,
    OptimConfig,
    DensificationConfig,
    PipelineConfig,
    TrainerConfig,
    DataConfig,
    TrainingConfig
)

__all__ = [
    # Core modules
    'GaussianModel',
    'FreeTimeGaussianModel',
    'create_model_from_config',
    'GaussianOptimizer',
    'setup_optimizer',
    'GaussianRenderer',
    'GaussianLoss',
    'l1_loss',
    'l2_loss',
    'ssim',
    'fast_ssim',
    'compute_loss',
    'GaussianDensifier',
    'Trainer',
    'DataSampler',
    'StaticSampler',
    'TemporalSampler',
    # Config classes
    'ModelConfig',
    'OptimConfig',
    'DensificationConfig',
    'PipelineConfig',
    'TrainerConfig',
    'DataConfig',
    'TrainingConfig',
]

