"""
Modern Gaussian Splatting - Training Script

Supports:
- Static Scenes (Original 3D Gaussian Splatting)
- Temporal Scenes (FreeTimeGS)
- Automatic Point Cloud Initialization (COLMAP / Custom PLY)
- Complete Densification Strategy
- TensorBoard Logging
- Checkpoint Saving and Restoration

Usage:
    python train.py --config config/default_config.yaml
    python train.py --source_path /path/to/dataset --model_path ./output
"""

import sys
import torch
from pathlib import Path
import traceback

from config.parser import parse_args, get_combined_configs, save_config
from core.builder import setup_dataset, setup_model, setup_optimizer, setup_renderer
from core.trainer import Trainer, FreeTimeTrainer
from utils.general_utils import seed_everything


def print_banner():
    banner = """
╔════════════════════════════════════════════════════════════╗
║       Modern Gaussian Splatting - Training                 ║
║       Supports: Static & Temporal (FreeTimeGS)             ║
╚════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    print_banner()
    
    # 1. Parse Arguments & Configuration
    args = parse_args()
    
    seed_everything(args.seed)
    print(f"Random Seed: {args.seed}")
    
    # Load and merge configurations
    # Returns config objects and the raw dictionary
    try:
        data_config, model_config, optim_config, pipeline_config, trainer_config, config_dict = \
            get_combined_configs(args)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Check required arguments
    if not data_config.source_path:
        print("\nError: --source_path required")
        print("   Example: python train.py --source_path /path/to/dataset")
        return
    
    # Save effective configuration
    save_config(config_dict, Path(data_config.model_path))
    
    # 2. System Setup
    
    # Dataset
    train_dataset = setup_dataset(data_config, split="train")
    if train_dataset is None:
        print("Error: Could not load training dataset. Exiting.")
        return
        
    # Test Dataset (Optional)
    test_dataset = setup_dataset(data_config, split="test")
    if test_dataset is None:
        print("   Test set not available (optional/skipped)")
    
    # Model (Gaussian Splatting)
    # Note: setup_model handles point cloud initialization (auto-search or random)
    model = setup_model(model_config, data_config, train_dataset)
    
    # Optimizer
    optimizer = setup_optimizer(model, optim_config)
    
    # Renderer
    renderer = setup_renderer(pipeline_config).cuda()
    
    # 3. Resume (Optional)
    start_iteration = 0
    if args.resume_from and Path(args.resume_from).exists():
        print(f"\nResuming from checkpoint: {args.resume_from}")
        try:
            checkpoint = torch.load(args.resume_from)
            model.load_state_dict(checkpoint['model'])
            optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
            start_iteration = checkpoint.get('iteration', 0)
            print(f"   Resuming at iteration: {start_iteration}")
        except Exception as e:
            print(f"   Error loading checkpoint: {e}")
            print("   Starting from scratch.")
    
    if args.test_only:
        print("\nTest Mode Only")
        # Placeholder for test-only logic
        if test_dataset:
             # Manually trigger test?
             pass
        return
    
    # 4. Training
    print("\nInitializing Trainer")
    
    # Factory Logic
    if model.config.mode == "freetime":
        TrainerClass = FreeTimeTrainer
    else:
        TrainerClass = Trainer

    trainer = TrainerClass(
        model=model,
        optimizer=optimizer,
        renderer=renderer,
        dataset=train_dataset,
        config=trainer_config,
        data_config=data_config,
        test_dataset=test_dataset
    )
    
    # Fix for resume functionality: explicitly update trainer's iteration
    if start_iteration > 0:
        trainer.current_iteration = start_iteration
        print(f"   Trainer iteration updated to: {trainer.current_iteration}")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("   Saving current model...")
        trainer.save_checkpoint(trainer.current_iteration, final=True)
        print("   Model saved")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        traceback.print_exc()
        raise
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"   Model saved to {data_config.model_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
