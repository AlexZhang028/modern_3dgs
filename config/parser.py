"""
Config Parser & Utils
Responsible for parsing command line arguments, loading YAML, merging configurations, and generating config objects.
"""

import yaml
import argparse
from pathlib import Path
from typing import Tuple, Dict

from .config import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    PipelineConfig,
    TrainerConfig
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Gaussian Splatting Model")
    
    # Configuration File
    parser.add_argument("--config", type=str, default="", help="Path to YAML config file")
    
    # Basic Arguments (Override config)
    parser.add_argument("--source_path", type=str, default="", help="Dataset root directory")
    parser.add_argument("--model_path", type=str, default=None, help="Model output path")
    parser.add_argument("--model_type", type=str, default=None, 
                        choices=["static", "freetime"], help="Model type")
    
    # Training Arguments
    parser.add_argument("--iterations", type=int, default=None, help="Training iterations")
    parser.add_argument("--sh_degree", type=int, default=None, help="SH degree")
    parser.add_argument("--resolution", type=int, default=None, help="Image resolution (-1 for original)")
    
    # Optional Arguments
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    parser.add_argument("--random_background", action="store_true", help="Use random background")
    parser.add_argument("--init_point_cloud", type=str, default="", help="Initial point cloud path")
    parser.add_argument("--cache_images", action="store_true", help="Cache images to GPU")
    
    # FreeTimeGS Override Arguments
    parser.add_argument("--start_frame", type=int, default=None, help="Start frame for temporal dataset")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame for temporal dataset")
    parser.add_argument("--train_views", nargs='+', type=str, default=None, help="Specific cameras for training (overrides config)")
    parser.add_argument("--test_views", nargs='+', type=str, default=None, help="Specific cameras for testing (overrides config)")
    parser.add_argument("--normalized_t", type=int, default=None, help="Use normalized time (0/1) or seconds. 1=True, 0=False")
    parser.add_argument("--fps", type=float, default=None, help="Override video FPS")
    parser.add_argument("--use_tmp", action="store_true", help="Use temporary directory for frames")
    
    # Debug Arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--disable_tensorboard", action="store_true", help="Disable TensorBoard")
    
    # Checkpoint Arguments
    parser.add_argument("--resume_from", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--test_only", action="store_true", help="Test mode only")
    parser.add_argument("--save_checkpoint", action="store_true", help="Save .pth checkpoints periodically")
    parser.add_argument("--no_save_ply", action="store_true", help="Disable periodic PLY saving")
    
    return parser.parse_args()


def load_config_from_yaml(yaml_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def merge_configs(yaml_config: dict, args: argparse.Namespace) -> dict:
    """Merge YAML config and command line arguments (CLI takes precedence)."""
    config = yaml_config.copy()
    
    if args.source_path:
        config['data']['source_path'] = args.source_path
    if args.model_path:
        config['data']['model_path'] = args.model_path
    if args.model_type:
        config['model']['mode'] = args.model_type
    if args.iterations is not None:
        config['optim']['iterations'] = args.iterations
    if args.sh_degree is not None:
        config['model']['sh_degree'] = args.sh_degree
    if args.resolution is not None:
        config['data']['resolution'] = args.resolution
    if args.white_background:
        config['data']['white_background'] = True
    if args.random_background:
        config['optim']['random_background'] = True
    if args.init_point_cloud:
        config['data']['init_point_cloud_path'] = args.init_point_cloud
    if args.cache_images:
        config['data']['cache_images'] = True
    
    # FreeTimeGS Overrides
    if args.start_frame is not None:
        config['data']['start_frame'] = args.start_frame
    if args.end_frame is not None:
        config['data']['end_frame'] = args.end_frame
    if args.normalized_t is not None:
        val = bool(args.normalized_t)
        # Ensure 'model' and 'data' dicts exist
        if 'model' not in config: config['model'] = {}
        if 'data' not in config: config['data'] = {}
        config['model']['normalized_t'] = val
        config['data']['normalized_t'] = val
        
    if args.fps is not None:
        if 'data' not in config: config['data'] = {}
        config['data']['fps'] = args.fps
        
    if args.use_tmp:
        if 'data' not in config: config['data'] = {}
        config['data']['use_tmp'] = True
        
    # View Selection Overrides
    if args.train_views:
        # Clean input: remove brackets and commas if present
        views = []
        for v in args.train_views:
             # Handle cases like "[01," "02]" or "01,02"
             cleaned = v.replace('[', ' ').replace(']', ' ').replace(',', ' ')
             views.extend(cleaned.split())
        config['data']['train_cameras'] = views

    if args.test_views:
        views = []
        for v in args.test_views:
             cleaned = v.replace('[', ' ').replace(']', ' ').replace(',', ' ')
             views.extend(cleaned.split())
        config['data']['test_cameras'] = views

    # Trainer config
    if 'trainer' not in config:
        config['trainer'] = {}
    
    if args.save_checkpoint:
        config['trainer']['save_checkpoint'] = True
    if args.no_save_ply:
        config['trainer']['save_ply'] = False
        
    return config


def create_configs(config_dict: dict, args: argparse.Namespace) -> Tuple[
    DataConfig, ModelConfig, OptimConfig, PipelineConfig, TrainerConfig, Dict
]:
    """
    Create configuration objects.
    
    Returns:
        (data_config, model_config, optim_config, pipeline_config, trainer_config, full_config_dict)
    """
    # Data Config
    data_config = DataConfig(**config_dict.get('data', {}))
    
    # Optim Config
    # Merge 'densify' section into 'optim' for compatibility with OptimConfig shared fields
    densify_dict = config_dict.get('densify', {})
    optim_input = config_dict.get('optim', {}).copy()
    
    # Keys shared between DensificationConfig and OptimConfig
    shared_densify_keys = [
        'percent_dense', 'densify_from_iter', 'densify_until_iter', 
        'densify_interval', 'densify_grad_threshold', 'opacity_reset_interval'
    ]
    for k in shared_densify_keys:
        if k in densify_dict:
            optim_input[k] = densify_dict[k]

    optim_config = OptimConfig(**optim_input)
    
    # Model Config
    model_dict = config_dict.get('model', {}).copy()
    if 'model_type' in model_dict:
        model_dict['mode'] = model_dict.pop('model_type')
    
    # Sync percent_dense
    if 'percent_dense' not in model_dict and hasattr(optim_config, 'percent_dense'):
        model_dict['percent_dense'] = optim_config.percent_dense
    
    model_config = ModelConfig(**model_dict)
    
    # Pipeline Config
    pipeline_config = PipelineConfig(**config_dict.get('pipeline', {}))
    
    # Trainer Config
    trainer_dict = config_dict.get('trainer', {})
    
    # Helper to get value from densify_dict -> optim_config -> default
    def get_densify_param(key, fallback):
        return densify_dict.get(key, fallback)

    trainer_config = TrainerConfig(
        iterations=optim_config.iterations,
        output_dir=data_config.model_path,
        white_background=data_config.white_background,
        random_background=optim_config.random_background,
        lambda_dssim=optim_config.lambda_dssim,
        # Pass depth weights from optim_config to trainer_config
        depth_l1_weight_init=optim_config.depth_l1_weight_init,
        depth_l1_weight_final=optim_config.depth_l1_weight_final,
        densify_grad_threshold=optim_config.densify_grad_threshold,
        densify_from_iter=optim_config.densify_from_iter,
        densify_until_iter=optim_config.densify_until_iter,
        densify_interval=getattr(optim_config, 'densify_interval', getattr(optim_config, 'densification_interval', 100)),
        opacity_reset_interval=optim_config.opacity_reset_interval,
        prune_opacity_threshold=densify_dict.get('prune_opacity_threshold', 0.005),
        prune_size_threshold=densify_dict.get('prune_size_threshold', 20.0),
        enable_tensorboard=not args.disable_tensorboard,
        log_interval=trainer_dict.get('log_interval', config_dict.get('log_interval', 10)),
        test_interval=trainer_dict.get('test_interval', 1000),
        save_interval=trainer_dict.get('save_interval', 5000),
        save_iterations=trainer_dict.get('save_iterations', config_dict.get('save_iterations', [])),
        test_iterations=trainer_dict.get('test_iterations', config_dict.get('test_iterations', [])),
        checkpoint_iterations=trainer_dict.get('checkpoint_iterations', config_dict.get('checkpoint_iterations', [])),
        num_test_views=trainer_dict.get('num_test_views', 5),
        save_ply=trainer_dict.get('save_ply', True),
        save_checkpoint=trainer_dict.get('save_checkpoint', False),
    )
    
    return data_config, model_config, optim_config, pipeline_config, trainer_config, config_dict


def get_combined_configs(args: argparse.Namespace) -> Tuple:
    """
    High-level flow: Load YAML -> Merge with CLI -> Create Config Objects.
    """
    if args.config and Path(args.config).exists():
        print(f"   Loading from file: {args.config}")
        config_dict = load_config_from_yaml(args.config)
    else:
        print("   Using default configuration")
        # Assuming default config is relative to project root or handles absolute logic elsewhere
        # Ideally, we look for a default
        default_config_path = Path("config/default_config.yaml") 
        if default_config_path.exists():
            config_dict = load_config_from_yaml(str(default_config_path))
        else:
            print("   Warning: Default config not found, using empty structure")
            config_dict = {
                'data': {},
                'model': {},
                'optim': {},
                'pipeline': {}
            }
    
    config_dict = merge_configs(config_dict, args)
    return create_configs(config_dict, args)

def save_config(config_dict: dict, output_dir: Path):
    """Save config to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\nConfig saved: {config_path}")
