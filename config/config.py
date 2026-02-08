"""
Unified Configuration System

Manages configuration via dataclasses, supporting loading from and saving to YAML files.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple
from pathlib import Path
import yaml


# ============================================================================
# Data Configuration
# ============================================================================

@dataclass
class DataConfig:
    """Dataset configuration."""
    # Paths
    source_path: str = ""  # Dataset root directory
    model_path: str = "output"  # Model output path
    images: str = "images"  # Images folder name
    depths: str = ""  # Depths folder name (optional)
    
    # Image Processing
    resolution: int = 8  # Image resolution (1, 2, 4, 8 for downscale factor)
    white_background: bool = False  # Use white background
    
    # Initial Point Cloud
    init_point_cloud_path: str = ""  # Custom initial point cloud path (optional)
    
    # Data Loading
    data_device: str = "cuda"  # Device to load data to
    eval: bool = False  # Evaluation mode
    train_test_exp: bool = False  # Use train/test exposure compensation
    cache_images: bool = True  # Cache images to memory/VRAM
    num_workers: int = 4  # DataLoader workers
    pin_memory: bool = True  # Use pin_memory
    lazy_loading: bool = False  # Lazy loading (False = preload all images)
    normalized_t: bool = False # Use normalized time (0-1) or seconds for dataset loading
    
    # SelfCap / Video Dataset Options
    start_frame: int = 0  # Frame start index
    end_frame: int = -1  # Frame end index (-1 = all)
    test_cameras: List[str] = field(default_factory=list)  # List of camera names to use for testing
    train_cameras: List[str] = field(default_factory=list)  # List of camera names to use for training
    fps: float = -1.0 # Override FPS for video datasets (-1 = auto)
    use_tmp: bool = True # Use temporary directory for frame extraction (reduces IO bottleneck, cleans up after)



# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """
    Gaussian Model Configuration.
    """
    # Basic Parameters
    sh_degree: int = 3  # SH degree (0-4)
    mode: str = "static"  # Model type: "static" or "freetime"
    
    # Densification Parameters
    percent_dense: float = 0.01  # Percentage of scene extent for densification
    
    # FreeTimeGS Specific Parameters
    time_dim: int = 0  # Time dimension
    motion_dim: int = 0  # Motion dimension
    normalized_t: bool = False # Use normalized time (0-1) or seconds
    
    def __post_init__(self):
        assert self.mode in ["static", "freetime"], \
            f"Mode must be 'static' or 'freetime', got '{self.mode}'"
        assert 0 <= self.sh_degree <= 4, \
            f"SH degree must be between 0-4, got {self.sh_degree}"


# ============================================================================
# Optimizer Configuration
# ============================================================================

@dataclass
class OptimConfig:
    """
    Optimizer Configuration.
    """
    # Training Iterations
    iterations: int = 30_000
    
    # Position Learning Rate (Exponential Decay)
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    
    # Other Learning Rates
    feature_lr: float = 0.0025
    opacity_lr: float = 0.025
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    
    # FreeTimeGS Specific Learning Rates
    t_lr: float = 0.0001
    t_scale_lr: float = 0.005
    velocity_lr: float = 0.00016
    
    # Exposure Compensation Learning Rates
    exposure_lr_init: float = 0.01
    exposure_lr_final: float = 0.001
    exposure_lr_delay_steps: int = 0
    exposure_lr_delay_mult: float = 0.0
    
    # Optimizer Hyperparameters
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-15
    
    # Loss Weights
    lambda_dssim: float = 0.2
    depth_l1_weight_init: float = 1.0
    depth_l1_weight_final: float = 0.01
    
    # Background
    random_background: bool = False
    
    # Densification Parameters (Placed here for compatibility)
    percent_dense: float = 0.01
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_interval: int = 100
    densification_interval: int = 100  # Alias
    densify_grad_threshold: float = 0.0002
    opacity_reset_interval: int = 3000
    
    # Optimizer Type
    optimizer_type: str = "default"  # "default" or "sparse_adam"


# ============================================================================
# Densification Configuration
# ============================================================================

@dataclass
class DensificationConfig:
    """
    Densification and Pruning Configuration.
    """
    # Densification Timing
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_interval: int = 100
    
    # Densification Thresholds
    densify_grad_threshold: float = 0.0002
    percent_dense: float = 0.01
    N_split: int = 2
    
    # Pruning Timing
    prune_from_iter: int = 500
    prune_interval: int = 100
    prune_max_screen_size_iter: int = 3000
    
    # Pruning Thresholds
    prune_opacity_threshold: float = 0.005
    prune_size_threshold: float = 20.0
    
    # Opacity Reset
    opacity_reset_interval: int = 3000
    
    # Scene Extent
    cameras_extent: float = 1.0


# ============================================================================
# Renderer Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """
    Renderer Pipeline Configuration.
    """
    # Computation Mode
    compute_cov3D_python: bool = False  # Debug
    convert_SHs_python: bool = False  # Debug
    
    # Render Options
    debug: bool = False
    antialiasing: bool = False


# ============================================================================
# Trainer Configuration
# ============================================================================

@dataclass
class TrainerConfig:
    """
    Trainer Configuration.
    """
    # Training Iterations
    iterations: int = 30_000
    
    # Intervals
    save_interval: int = 5000
    test_interval: int = 1000
    log_interval: int = 10
    
    # Specific Iterations
    save_iterations: List[int] = field(default_factory=list)
    test_iterations: List[int] = field(default_factory=list)
    checkpoint_iterations: List[int] = field(default_factory=list)
    
    # SH Progressive Activation
    sh_degree_interval: int = 1000
    
    # Checkpointing
    output_dir: str = "output"
    checkpoint_dir: str = "checkpoints"
    keep_checkpoints: int = 3
    save_ply: bool = True
    save_checkpoint: bool = False
    
    # Logging
    enable_tensorboard: bool = True
    log_dir: str = "logs"
    num_test_views: int = 5
    
    # Background
    random_background: bool = False
    white_background: bool = False
    
    # Densification Parameters (For Trainer context)
    densify_grad_threshold: float = 0.0002
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_interval: int = 100
    opacity_reset_interval: int = 3000
    prune_opacity_threshold: float = 0.005
    prune_size_threshold: float = 20.0
    
    # Loss Weights
    lambda_dssim: float = 0.2
    depth_l1_weight_init: float = 1.0 
    depth_l1_weight_final: float = 0.01

    # FreeTimeGS Specific Training Params
    lambda_reg: float = 1.0        # 4D Regularization weight
    reg_end_iter: int = 3000       # Iteration to stop regularization
    relocation_interval: int = 500 # Iteration interval for relocation

# ============================================================================
# Full Training Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """
    Full Training Configuration.
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    densify: DensificationConfig = field(default_factory=DensificationConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    
    # Global Settings
    seed: int = 42
    use_tensorboard: bool = True
    
    # Logging and Saving (Shortcuts)
    log_interval: int = 10
    save_iterations: List[int] = field(default_factory=lambda: [7_000, 30_000])
    checkpoint_iterations: List[int] = field(default_factory=lambda: [7_000, 30_000])
    test_iterations: List[int] = field(default_factory=lambda: [7_000, 30_000])
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            optim=OptimConfig(**config_dict.get('optim', {})),
            densify=DensificationConfig(**config_dict.get('densify', {})),
            pipeline=PipelineConfig(**config_dict.get('pipeline', {})),
            trainer=TrainerConfig(**config_dict.get('trainer', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['data', 'model', 'optim', 'densify', 'pipeline', 'trainer']}
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'optim': asdict(self.optim),
            'densify': asdict(self.densify),
            'pipeline': asdict(self.pipeline),
            'trainer': asdict(self.trainer),
            'seed': self.seed,
            'use_tensorboard': self.use_tensorboard,
            'log_interval': self.log_interval,
            'save_iterations': self.save_iterations,
            'checkpoint_iterations': self.checkpoint_iterations,
            'test_iterations': self.test_iterations,
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def __str__(self) -> str:
        """Pretty print configuration."""
        lines = ["=" * 60, "Training Configuration", "=" * 60]
        for key, value in asdict(self).items():
            if isinstance(value, dict):
                lines.append(f"\n[{key.upper()}]")
                for sub_key, sub_value in value.items():
                    lines.append(f"  {sub_key}: {sub_value}")
            else:
                lines.append(f"{key}: {value}")
        lines.append("=" * 60)
        return "\n".join(lines)



