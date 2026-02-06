"""
System Builder
Responsible for initializing and assembling core components like Dataset, Model, Optimizer, Renderer, etc.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional

from config.config import DataConfig, ModelConfig, OptimConfig, PipelineConfig
from data.dataset import GaussianDataset, SelfCapVideoDataset, StaticGaussianDataset
from core.gaussian_model import create_model_from_config, GaussianModel
import os

from core.optimizer import GaussianOptimizer
from core.renderer import GaussianRenderer


def setup_dataset(data_config: DataConfig, split: str = "train") -> Optional[GaussianDataset]:
    """Setup dataset."""
    print(f"\nLoading dataset ({split}): {data_config.source_path}")
    
    cache_device = "cuda" if data_config.cache_images else None
    
    # If using multiprocessing DataLoader, cache device for dataset must be CPU
    dataset_cache_device = cache_device
    if data_config.num_workers > 0:
        dataset_cache_device = "cpu"
        # Only print for train split to avoid spam
        if split == "train":
            print("   Multiprocessing enabled: Forcing dataset cache to CPU")

    # Detect dataset type
    is_selfcap = (os.path.exists(os.path.join(data_config.source_path, "extri.yml")) or 
                  os.path.exists(os.path.join(data_config.source_path, "optimized", "extri.yml")))
    
    DatasetClass = SelfCapVideoDataset if is_selfcap else StaticGaussianDataset

    try:
        dataset = DatasetClass(
            source_path=data_config.source_path,
            split=split,
            resolution=data_config.resolution,
            white_background=data_config.white_background,
            cache_device=dataset_cache_device,
            images_folder=data_config.images,
            depths_folder=data_config.depths,
            eval_mode=data_config.eval,
            train_test_exp=data_config.train_test_exp,
            init_point_cloud_path=data_config.init_point_cloud_path,
            start_frame=data_config.start_frame,
            end_frame=data_config.end_frame,
            test_camera_names=data_config.test_cameras,
            train_camera_names=data_config.train_cameras,
            normalized_t=data_config.normalized_t,
        )
        
        # Inject fps override if available
        if hasattr(data_config, 'fps') and data_config.fps > 0:
            dataset.fps = data_config.fps
    except Exception as e:
        print(f"   Failed to load {split} dataset: {e}")
        return None
    
    print(f"   Number of cameras: {len(dataset)}")
    if len(dataset) > 0 and split == "train":
        sample = dataset[0]
        cam = sample["camera"]
        print(f"   Image Size: {cam.width} x {cam.height}")
    
    return dataset


def setup_model(
    model_config: ModelConfig,
    data_config: DataConfig,
    dataset: GaussianDataset
) -> GaussianModel:
    """Setup model with initialization strategy (PointCloud / Random)."""
    print(f"\nInitializing Model (Mode: {model_config.mode})")
    
    gaussian_config = ModelConfig(
        mode=model_config.mode,
        sh_degree=model_config.sh_degree,
        time_dim=model_config.time_dim if model_config.mode == "freetime" else 0,
        motion_dim=model_config.motion_dim if model_config.mode == "freetime" else 0,
        normalized_t=model_config.normalized_t,
    )
    
    # Create model
    model = create_model_from_config(gaussian_config)
    
    # Important: Move to CUDA before create_from_pcd
    model = model.cuda()
    
    # Calculate Time Bounds
    t_start = 0.0
    t_extent = 1.0
    
    # Always calculate time stats from dataset if available
    if dataset and hasattr(dataset, "cameras") and len(dataset.cameras) > 0:
        # Check if cameras have timestamp_seconds (SelfCap datasets)
        if hasattr(dataset.cameras[0], 'timestamp_seconds'):
            times = [c.timestamp_seconds for c in dataset.cameras]
            if times:
                t_min = min(times)
                t_max_val = max(times)
                # t_start is the offset of the video in world time (often 0)
                t_start = t_min
                t_extent = t_max_val - t_min
                
                # If single frame or very fast, prevent zero division
                if t_extent < 1e-6:
                     t_extent = 1.0
                
                print(f"   Time Extent: {t_extent:.4f}s (Start: {t_start:.4f}s)")
    
    # If not using normalized time, we still pass the extent so random initialization covers the full range
    if not model_config.normalized_t:
        print("   Using Raw Time (Seconds). Metrics will be in seconds.")
    
    time_info = {
        't_start': t_start,
        't_extent': t_extent
    }
    
    # Initialize point cloud
    init_path = data_config.init_point_cloud_path
    source_path = Path(data_config.source_path)
    
    if not init_path:
        # Auto search
        candidates = [
            source_path / "points3d.ply",
            source_path / "sparse" / "0" / "points3D.ply",
            source_path / "init_pointcloud.ply",
        ]
        for candidate in candidates:
            if candidate.exists():
                init_path = str(candidate)
                print(f"   Found initial point cloud: {init_path}")
                break
    
    if init_path and Path(init_path).exists():
        print(f"   Initializing from: {init_path}")
        from data.ply_utils import fetchPly
        pcd = fetchPly(init_path)
        
        # Calculate scene extent
        cameras_extent = dataset.get_cameras_extent()
        print(f"   Scene extent (cameras_extent): {cameras_extent:.6f}")
        
        # Fallback if dataset returns default 1.0 likely incorrect
        if cameras_extent == 1.0 and len(dataset) > 0:
            all_camera_centers = []
            for i in range(min(len(dataset), 100)):
                sample = dataset[i]
                cam = sample["camera"]
                all_camera_centers.append(cam.camera_center.cpu())
            
            if all_camera_centers:
                camera_centers = torch.stack(all_camera_centers)
                cameras_extent = camera_centers.std().item() * 3
                print(f"   Fallback to std*3 extent: {cameras_extent:.6f}")
        
        model.create_from_pcd(pcd, spatial_lr_scale=cameras_extent, time_info=time_info)
    else:
        print("   No initial point cloud found, using random initialization")
        
        num_points = 10000
        cameras_extent = 1.0
        
        all_camera_centers = []
        if dataset:
            for i in range(min(len(dataset), 100)):
                sample = dataset[i]
                cam = sample["camera"]
                all_camera_centers.append(cam.camera_center)
        
        if all_camera_centers:
            camera_centers = torch.stack(all_camera_centers)
            center = camera_centers.mean(dim=0)
            scale = camera_centers.std(dim=0).max()
            cameras_extent = scale * 3
        else:
            center = torch.zeros(3)
            cameras_extent = 5.0
        
        points = torch.randn(num_points, 3) * cameras_extent + center
        colors = torch.rand(num_points, 3)
        normals = np.zeros_like(points.cpu().numpy())
        
        from utils.graphics_utils import BasicPointCloud
        pcd = BasicPointCloud(
            points=points.cpu().numpy(),
            colors=colors.cpu().numpy(),
            normals=normals
        )
        
        model.create_from_pcd(pcd, spatial_lr_scale=cameras_extent, time_info=time_info)
    
    print(f"   Initial points: {model.num_points}")
    
    return model


def setup_optimizer(model: GaussianModel, optim_config: OptimConfig) -> GaussianOptimizer:
    """Setup optimizer."""
    print("\nConfiguring Optimizer")
    
    optimizer_config = OptimConfig(
        position_lr_init=optim_config.position_lr_init,
        position_lr_final=optim_config.position_lr_final,
        position_lr_delay_mult=optim_config.position_lr_delay_mult,
        position_lr_max_steps=optim_config.position_lr_max_steps,
        feature_lr=optim_config.feature_lr,
        opacity_lr=optim_config.opacity_lr,
        scaling_lr=optim_config.scaling_lr,
        rotation_lr=optim_config.rotation_lr,
    )
    
    optimizer = GaussianOptimizer(model, optimizer_config)
    
    print("   Optimizer: Adam")
    return optimizer


def setup_renderer(pipeline_config: PipelineConfig) -> GaussianRenderer:
    """Setup renderer."""
    print("\nConfiguring Renderer")
    
    render_config = PipelineConfig(
        convert_SHs_python=pipeline_config.convert_SHs_python,
        compute_cov3D_python=pipeline_config.compute_cov3D_python,
        debug=getattr(pipeline_config, 'debug', False),
    )
    
    renderer = GaussianRenderer(render_config)
    print("   Render Mode: CUDA Accelerated")
    
    return renderer
