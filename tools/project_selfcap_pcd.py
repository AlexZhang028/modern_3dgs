
import os
import sys
import torch
import numpy as np
import cv2
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.selfcap_loader import read_selfcap_cameras
from data.camera import Camera
from data.ply_utils import fetchPly
from utils.graphics_utils import focal2fov

def main():
    parser = argparse.ArgumentParser(description="Project SelfCap Point Cloud to Image")
    parser.add_argument("--source_path", type=str, required=True, help="Path to selfcap dataset root")
    parser.add_argument("--camera_name", type=str, default=None, help="Camera name to use (default: first found)")
    parser.add_argument("--output_path", type=str, default="projection_verification.png", help="Output image path")
    args = parser.parse_args()
    
    source_path = args.source_path
    
    # 1. Load Cameras
    print(f"Loading cameras from {source_path}...")
    try:
        cam_names, extrinsics, intrinsics = read_selfcap_cameras(source_path)
    except Exception as e:
        print(f"Error loading cameras: {e}")
        return

    if not cam_names:
        print("No camera names found.")
        return

    if args.camera_name:
        cam_name = args.camera_name
        if cam_name not in cam_names:
            print(f"Error: Camera {cam_name} not found in available cameras.")
            return
    else:
        cam_name = cam_names[0]
        print(f"Using first camera: {cam_name}")

    if cam_name not in extrinsics:
        print(f"Error: No extrinsics for camera {cam_name}")
        return
        
    extrinsic = extrinsics[cam_name]
    R = extrinsic['R']
    T = extrinsic['T']
    
    # 3DGS convention: The Camera class expects R such that R.transpose() is the World-to-Camera rotation.
    # Since the yaml provides OpenCV W2C rotation directly, we must transpose it before passing to Camera.
    R = R.transpose()
    
    # 2. Setup Camera Intrinsics
    width = 0
    height = 0
    
    if cam_name in intrinsics:
        intrinsic = intrinsics[cam_name]
        K = intrinsic['K']
        
        # Try to find video for resolution
        videos_root = os.path.join(source_path, "videos")
        if not os.path.exists(videos_root):
             videos_root = source_path
             
        video_path = None
        # Try exact match or match with extension
        possible_files = [f for f in os.listdir(videos_root) if f.startswith(cam_name)]
        for f in possible_files:
            if f.lower().endswith('.mp4'):
                # Heuristic: check if filename starts with camera name
                # standard selfcap: '01.mp4' -> cam '01'
                if os.path.splitext(f)[0] == cam_name:
                    video_path = os.path.join(videos_root, f)
                    break
        
        # Fallback loop if exact match failed
        if not video_path:
             for f in possible_files:
                 if f.lower().endswith('.mp4'):
                     video_path = os.path.join(videos_root, f)
                     break

        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f"Video resolution from {video_path}: {width}x{height}")
        else:
            print("Warning: Could not find video file to determine resolution. Using default 1920x1080.")
            width = 1920
            height = 1080

        fx = K[0,0]
        fy = K[1,1]
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)
    else:
        print(f"Error: No intrinsics for camera {cam_name}")
        return

    # 3. Create Camera Object
    print(f"Creating Camera object for {cam_name}...")
    # Initialize camera with raw validation parameters
    cam = Camera(
        uid=0,
        image_name=cam_name,
        R=R,
        T=T,
        width=width,
        height=height,
        FovX=FovX,
        FovY=FovY,
        image=torch.zeros((3, height, width)),
        timestamp=0.0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cam.to(device)

    # 4. Load Point Cloud
    ply_path = os.path.join(source_path, "dense_pcds", "000000.ply")
    if not os.path.exists(ply_path):
        print(f"Error: Point cloud not found at {ply_path}")
        return
        
    print(f"Loading point cloud from {ply_path}...")
    try:
        pcd = fetchPly(ply_path)
        points = torch.tensor(pcd.points, dtype=torch.float32).to(device)
    except Exception as e:
        print(f"Error loading PLY: {e}")
        return
    
    # 5. Project Points
    print("Projecting points...")
    
    full_proj_transform = cam.full_proj_transform
    
    ones = torch.ones((points.shape[0], 1), dtype=torch.float32, device=device)
    points_hom = torch.cat([points, ones], dim=1)
    
    # Project: The full_proj_transform is (P @ V)^T. 
    # Points are row vectors (N,4).
    # Result = Points @ T = Points @ (P @ V)^T = (P @ V @ Points^T)^T
    p_hom = points_hom @ full_proj_transform
    p_w = 1.0 / (p_hom[:, 3:4] + 1e-7)
    p_proj = p_hom[:, :3] * p_w
    
    x_ndc = p_proj[:, 0]
    y_ndc = p_proj[:, 1]
    
    # Filter points slightly outside to handle edge cases but mainly clip to frame
    mask = (x_ndc >= -2.0) & (x_ndc <= 2.0) & (y_ndc >= -2.0) & (y_ndc <= 2.0)
    
    # Convert To Pixel
    # Screen space conversion
    x_pix = ((x_ndc + 1.0) * width * 0.5)
    y_pix = ((y_ndc + 1.0) * height * 0.5)
    
    valid_points = mask.sum().item()
    print(f"Points loosely in frustum: {valid_points} / {points.shape[0]}")
    
    x_pix = x_pix[mask].cpu().numpy().astype(np.int32)
    y_pix = y_pix[mask].cpu().numpy().astype(np.int32)
    
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Strictly valid indices for array access
    valid_indices = (x_pix >= 0) & (x_pix < width) & (y_pix >= 0) & (y_pix < height)
    x_pix = x_pix[valid_indices]
    y_pix = y_pix[valid_indices]
    
    print(f"Points drawn: {len(x_pix)}")
    canvas[y_pix, x_pix] = [255, 255, 255]
    
    cv2.imwrite(args.output_path, canvas)
    print(f"Saved projection verification to {args.output_path}")

if __name__ == "__main__":
    main()
