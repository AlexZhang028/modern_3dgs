import os
import sys
import argparse
import torch
import numpy as np
import cv2
import math
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.gaussian_model import create_model_from_config, detect_mode_from_ply
from core.renderer import GaussianRenderer
from core.builder import setup_dataset
from config.config import DataConfig, ModelConfig

def mkdir_p(path):
    os.makedirs(path, exist_ok=True)

def render_visualization(
    model,
    renderer,
    dataset,
    output_path: str,
    modes: list,
    selected_cameras: list = None,
    save_format: str = "mp4"
):
    """
    Render visualization videos for selected modes.
    """
    mkdir_p(output_path)
    
    # 1. Prepare Cameras
    if selected_cameras is None:
        cameras = dataset
    else:
        cameras = [dataset[i] for i in selected_cameras]

    print(f"Rendering {len(cameras)} cameras for modes: {modes}")

    # Prepare Video Writers
    video_writers = {}
    fps = 30 # Default FPS
    
    first_cam = cameras[0]
    height, width = int(first_cam.height), int(first_cam.width)
    
    for mode in modes:
        if save_format == "mp4":
            mode_path = os.path.join(output_path, f"{mode}.mp4")
            video_writers[mode] = cv2.VideoWriter(
                mode_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                fps, 
                (width, height)
            )
            print(f"initialized video writer for {mode} at {mode_path}")
        else:
            mode_dir = os.path.join(output_path, mode)
            mkdir_p(mode_dir)
            print(f"initialized image directory for {mode} at {mode_dir}")

    # Render Loop
    for cam_idx, camera in enumerate(tqdm(cameras, desc="Rendering Frames")):
        
        # Determine timestamp
        # For FreeTimeGS, camera usually has 'time' attribute.
        # If not, use normalized time from camera index if appropriate, or 0.
        timestamp = getattr(camera, 'time', 0.0)
        
        # Put camera and bg on GPU
        bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        
        # Pre-compute colors for each mode and Render
        for mode in modes:
            colors_override = None
            
            with torch.no_grad():
                if mode == 'velocity':
                    # Visualization 1: Velocity Map
                    if hasattr(model, 'get_motion'):
                        velocity = model.get_motion
                        speed = torch.norm(velocity, dim=1, keepdim=True)
                        # Normalize speed (using 5.0 as max_val as per doc)
                        velocity_norm = torch.clamp(speed / 5.0, 0.0, 1.0)
                        # Direction color [-1, 1] -> [0, 1]
                        direction_color = (torch.nn.functional.normalize(velocity, dim=1) + 1.0) / 2.0
                        colors_override = direction_color * velocity_norm
                    else:
                        # Fallback for static models
                        colors_override = torch.zeros((model.num_points, 3), device="cuda")

                elif mode == 'duration':
                    # Visualization 2: Time Duration / Scale-t
                    try:
                        if hasattr(model, '_t_scale'):
                             t_scale_log = model._t_scale
                        elif hasattr(model, 'get_scaling_t'):
                             t_scale_log = model.get_scaling_t
                        else:
                             t_scale_log = torch.zeros((model.num_points, 1), device="cuda") - 10.0 # Small duration
                        
                        duration = torch.exp(t_scale_log)
                        # Normalize: red=long (>0.5), blue=short
                        heatmap = torch.clamp(duration / 0.5, 0.0, 1.0)
                        # Red=Long, Blue=Short.
                        colors_override = torch.cat([heatmap, torch.zeros_like(heatmap), 1.0 - heatmap], dim=1)
                    except:
                        colors_override = torch.zeros((model.num_points, 3), device="cuda")

                elif mode == 'time_center':
                    # Visualization 3: Time Center
                    if hasattr(model, 'get_t'):
                        mu_t = model.get_t # [N, 1]
                        colors_override = torch.cat([mu_t, 1.0 - mu_t, torch.zeros_like(mu_t)], dim=1)
                        colors_override = torch.clamp(colors_override, 0.0, 1.0)
                    else:
                        colors_override = torch.zeros((model.num_points, 3), device="cuda")

                elif mode == 'spatial_scale':
                    # Visualization 4: Spatial Scale Heatmap
                    scales = model.get_scaling # [N, 3]
                    avg_scale = torch.mean(scales, dim=1, keepdim=True)
                    # Normalize: Red=Large (>0.01), Blue=Small
                    norm_scale = torch.clamp(avg_scale / 0.01, 0.0, 1.0)
                    colors_override = norm_scale.repeat(1, 3) 

                elif mode == 'rgb':
                    colors_override = None 
                
                else:
                    continue

                if colors_override is not None:
                    colors_override = colors_override.to("cuda")

                # Render
                output = renderer(
                    model, 
                    camera, 
                    bg_color=bg_color,
                    timestamp=timestamp,
                    colors_override=colors_override
                )
                
                image = output['render']
                
                # Convert to numpy for video/image
                image_np = image.permute(1, 2, 0).detach().cpu().numpy()
                image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                if save_format == "mp4":
                    video_writers[mode].write(image_np)
                else:
                    image_name = getattr(camera, 'image_name', f"{cam_idx:05d}")
                    cv2.imwrite(os.path.join(output_path, mode, f"{image_name}.png"), image_np)

    # Clean up
    if save_format == "mp4":
        for mode, writer in video_writers.items():
            writer.release()
        
    print(f"Rendering complete. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization Script")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model base directory") # e.g. output/exp_name
    parser.add_argument("--source_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load (e.g. 30000)")
    parser.add_argument("--output_path", type=str, default="output/visualization", help="Root output path")
    parser.add_argument("--modes", type=str, default="rgb,velocity,duration,time_center,spatial_scale", 
                        help="Visualize modes")
    parser.add_argument("--camera_indices", type=str, default="all", help="all or 0,1,2")
    parser.add_argument("--save_format", type=str, default="mp4", choices=["mp4", "images"], help="Output format: mp4 or images")
    
    args, _ = parser.parse_known_args()
    
    # 1. Setup Data Config
    data_config = DataConfig(
        source_path=args.source_path,
        model_path=args.model_path,
        eval=True
    )
    
    print("Loading Dataset...")
    # Try loading test split, else train
    # Note: Modern 3DGS dataset loading might require specific splits
    try:
        dataset = setup_dataset(data_config, split="test")
        if dataset is None or len(dataset) == 0:
            raise ValueError("Empty test set")
    except:
        print("Falling back to train dataset...")
        dataset = setup_dataset(data_config, split="train")

    if dataset is None:
        print("Error: Could not load dataset")
        sys.exit(1)
        
    print(f"Loaded {len(dataset)} images")

    # 2. Load Model
    # Determine iteration
    loaded_iter = args.iteration
    if loaded_iter == -1:
        pc_path = os.path.join(args.model_path, "point_cloud")
        if os.path.exists(pc_path):
            iters = [int(folder.split("_")[-1]) for folder in os.listdir(pc_path) if folder.startswith("iteration_")]
            if iters:
                loaded_iter = max(iters)
        else:
            print("Point cloud directory not found.")
            sys.exit(1)
            
    print(f"Loading iteration {loaded_iter}")
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{loaded_iter}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        print(f"Error: PLY file not found at {ply_path}")
        sys.exit(1)

    # Detect mode
    mode_str = detect_mode_from_ply(ply_path)
    print(f"Detected model mode: {mode_str}")
    
    # Create Model
    model_config = ModelConfig(mode=mode_str, sh_degree=3) 
    model = create_model_from_config(model_config)
    model.load_ply(ply_path)
    model.active_sh_degree = model.max_sh_degree
    
    # 3. Setup Renderer
    renderer = GaussianRenderer()
    
    # 4. Run Visualization
    modes_list = [m.strip() for m in args.modes.split(",")]
    
    if args.camera_indices == "all":
        selected_cameras = None
    else:
        selected_cameras = [int(i) for i in args.camera_indices.split(",")]
        
    render_visualization(
        model, 
        renderer, 
        dataset, 
        args.output_path, 
        modes_list,
        selected_cameras,
        args.save_format
    )
