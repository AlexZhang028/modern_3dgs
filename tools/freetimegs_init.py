#!/usr/bin/env python3
"""
FreeTimeGS Initialization Script

This script initializes the point cloud for FreeTimeGS training using a multi-view spatial reconstruction 
method combined with temporal motion estimation, instead of the traditional static COLMAP initialization.

Pipeline:
1. Spatial Reconstruction: Use RoMA to match features between camera pairs in each frame and triangulate 3D points.
2. Temporal Motion Estimation: Use KNN to estimate motion vectors between frames.
3. Aggregation: Combine all points into a single PLY file with time, motion, and scale attributes.

Usage:
    python tools/freetimegs_init.py --source_path <path_to_dataset> --output_dir <path_to_output>
"""

import os
import sys
import argparse
import re
import math
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import json
from sklearn.neighbors import NearestNeighbors
from plyfile import PlyData, PlyElement

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import GaussianDataset
from data.camera import Camera
from utils.sh_utils import RGB2SH
from utils.general_utils import inverse_sigmoid

# Try importing RoMA
try:
    from romatch import roma_outdoor
    HAS_ROMA = True
except ImportError:
    HAS_ROMA = False
    print("Warning: RoMA (romatch) not found. Please install it to use this script.")
    print("pip install git+https://github.com/Parskatt/RoMa")

def parse_args():
    parser = argparse.ArgumentParser(description="FreeTimeGS Initialization Pipeline")
    parser.add_argument("--source_path", type=str, required=True, help="Path to the source dataset (COLMAP/Blender)")
    parser.add_argument("--output_path", type=str, default="init_point_cloud.ply", help="Path to save the output PLY file")
    parser.add_argument("--resolution", type=int, default=-1, help="Image resolution scale")
    parser.add_argument("--knn_k", type=int, default=1, help="KNN k for motion estimation")
    parser.add_argument("--points_per_frame", type=int, default=20000, help="Target number of points to keep per frame")
    return parser.parse_args()

def get_projection_matrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def get_full_projection_matrix(w2c, proj):
    return torch.matmul(proj, w2c)

def extract_frame_index(image_name):
    # Try to find the last sequence of digits in the filename
    matches = re.findall(r"(\d+)", image_name)
    if matches:
        return int(matches[-1])
    return -1

def group_cameras_by_time(cameras):
    """
    Group cameras by extracted time frame index.
    Returns a dictionary {frame_idx: [Camera, ...]} and sorted frame indices.
    """
    grouped = {}
    
    # Check if we can extract valid indices
    indices = [extract_frame_index(cam.image_name) for cam in cameras]
    if all(idx == -1 for idx in indices):
        print("Warning: Could not extract frame indices from image names. Assuming single frame or sequential.")
        # Fallback: Assume simple sequential if small count, or fail?
        # Let's assume the order in list implies some structure if parsing failed
        # But for now, let's just group everything into frame 0 if failed, or error out.
        # Constructing a single frame from all images
        grouped[0] = cameras
        return grouped, [0]
        
    for cam, idx in zip(cameras, indices):
        if idx not in grouped:
            grouped[idx] = []
        grouped[idx].append(cam)
    
    # Sort frames
    sorted_frames = sorted(grouped.keys())
    return grouped, sorted_frames

def triangulate_points(matches, P1, P2):
    """
    Triangulate 2D matches to 3D points.
    matches: [N, 4] (x1, y1, x2, y2)
    P1, P2: [3, 4] Projection matrices (K @ [R|T])
    """
    # Convert points to homogeneous coordinates for opencv triangulation if needed
    # Or use cv2.triangulatePoints which takes 2xN arrays
    
    kp1 = matches[:, :2].T # 2xN
    kp2 = matches[:, 2:].T # 2xN
    
    points_4d = cv2.triangulatePoints(P1, P2, kp1, kp2)
    points_3d = points_4d[:3] / points_4d[3] # 3xN
    
    return points_3d.T # Nx3

def get_camera_matrix(camera):
    # Construct K
    # fov = 2 * atan(W / (2 * f)) -> f = W / (2 * tan(fov/2))
    f_x = camera.width / (2 * math.tan(camera.FovX / 2))
    f_y = camera.height / (2 * math.tan(camera.FovY / 2))
    
    K = np.array([
        [f_x, 0, camera.width / 2],
        [0, f_y, camera.height / 2],
        [0, 0, 1]
    ])
    
    # Construct [R|T]
    # Camera R, T are World-to-Camera
    RT = np.zeros((3, 4))
    RT[:3, :3] = camera.R
    RT[:3, 3] = camera.T
    
    # P = K @ RT
    P = K @ RT
    return P

import cv2

def run_roma_matching(model, im1_path, im2_path, device):
    """
    Run RoMA matching between two images.
    Returns: matches numpy array [N, 4] (x1, y1, x2, y2) in pixel coordinates
    """
    if not HAS_ROMA:
        # Dummy mock for testing without RoMA
        return np.empty((0, 4))

    # Perform matching
    try:
        warp, certainty = model.match(im1_path, im2_path, device=device)
        # Sample matches
        matches, certainty = model.sample(warp, certainty)
        
        # Get image sizes
        w1, h1 = Image.open(im1_path).size
        w2, h2 = Image.open(im2_path).size
        
        # Convert to pixel coordinates
        kpts1, kpts2 = model.to_pixel_coordinates(matches, h1, w1, h2, w2)
        
        # Combine
        # kpts1: [N, 2], kpts2: [N, 2]
        res = torch.cat([kpts1, kpts2], dim=1).cpu().numpy()
        return res
    except Exception as e:
        print(f"RoMA matching failed: {e}")
        return np.empty((0, 4))

def save_freetime_ply(path, xyz, motion, time, t_scale_log, spatial_scale_log, colors):
    # Prepare data for PLY
    # Ensure all are numpy arrays and flattened/aligned
    
    num_points = xyz.shape[0]
    
    # Calculate SH (DC only, rest are 0)
    # RGB2SH in utils works on torch tensors? Let's check. 
    # Usually it's (rgb - 0.5) / C0. linear. 
    # Let's assume input colors are [0, 1].
    # We will use simple numpy conversion consistent with RGB2SH
    C0 = 0.28209479177387814
    f_dc = (colors - 0.5) / C0
    
    # Opacity (inverse sigmoid of 0.1)
    # inverse_sigmoid(x) = log(x/(1-x))
    op_val = np.log(0.1 / (1 - 0.1))
    opacity = np.full(num_points, op_val, dtype='f4')
    
    # Define custom dtype
    # Conforming to FreeTimeGS_repr.md and freetimegs.md
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), # Normals (Unused by FreeTimeGS, but standard PLY)
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'), # SH DC
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'), # Static Rotation
        ('t', 'f4'),          # Time Center
        ('t_scale', 'f4'),    # Duration (Log)
        ('motion_0', 'f4'), ('motion_1', 'f4'), ('motion_2', 'f4') # Velocity
    ]
    
    # Standard 3DGS has f_rest_0 to f_rest_44. If we don't include them, 
    # standard loader might complain if it expects them. 
    # But this is a custom loader for FreeTimeGS.
    # We will include them to be safe if FreeTimeGS inherits from Standard.
    for i in range(45):
        dtype.append((f'f_rest_{i}', 'f4'))
    
    elements = np.empty(num_points, dtype=dtype)
    
    # Fill fields
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    
    # Normals (Optional, just zero or use motion if we wanted to hide it there, but we use explicit fields now)
    elements['nx'] = np.zeros(num_points, dtype='f4')
    elements['ny'] = np.zeros(num_points, dtype='f4')
    elements['nz'] = np.zeros(num_points, dtype='f4')
    
    # Motion (Velocity)
    elements['motion_0'] = motion[:, 0]
    elements['motion_1'] = motion[:, 1]
    elements['motion_2'] = motion[:, 2]
    
    # SH DC
    elements['f_dc_0'] = f_dc[:, 0]
    elements['f_dc_1'] = f_dc[:, 1]
    elements['f_dc_2'] = f_dc[:, 2]
    
    # Opacity
    elements['opacity'] = opacity
    
    # Scales
    elements['scale_0'] = spatial_scale_log
    elements['scale_1'] = spatial_scale_log
    elements['scale_2'] = spatial_scale_log
    
    # Rotation (Identity - Static Orientation)
    # FreeTimeGS uses static rotation for the Gaussian shape, but motion is purely translational (velocity)
    elements['rot_0'] = np.ones(num_points, dtype='f4')
    elements['rot_1'] = np.zeros(num_points, dtype='f4')
    elements['rot_2'] = np.zeros(num_points, dtype='f4')
    elements['rot_3'] = np.zeros(num_points, dtype='f4')
    
    # Time / Duration
    elements['t'] = time
    elements['t_scale'] = t_scale_log
    
    # Zero out f_rest
    for i in range(45):
        elements[f'f_rest_{i}'] = np.zeros(num_points, dtype='f4')
    
    # Create PLY
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    print(f"Saved PLY to {path}")

def main():
    args = parse_args()
    
    print(f"Loading dataset from {args.source_path}")
    dataset = GaussianDataset(args.source_path, resolution=args.resolution, images_folder="images")
    
    # Group cameras
    grouped_cameras, frame_indices = group_cameras_by_time(dataset.cameras)
    print(f"Found {len(frame_indices)} frames with valid cameras.")
    
    if len(frame_indices) == 0:
        print("No frames found. Exiting.")
        return

    # Init RoMA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if HAS_ROMA:
        print(f"Initializing RoMA model on {device}...")
        roma_model = roma_outdoor(device=device)
    else:
        roma_model = None
        print("Skipping RoMA initialization (not installed).")
        return

    global_point_cloud_list = [] # List of numpy arrays [N, 3]
    
    # Phase 1: Spatial Reconstruction
    print("Starting Phase 1: Spatial Reconstruction...")
    
    # Prepare progress bar
    pbar = tqdm(frame_indices, desc="Processing Frames")
    for t_idx, t in enumerate(pbar):
        current_cameras = grouped_cameras[t]
        
        # Sort cameras (spatially or by name) to ensure consistent chaining
        current_cameras.sort(key=lambda x: x.image_name)
        
        num_cams = len(current_cameras)
        if num_cams < 2:
            print(f"Frame {t} has less than 2 cameras, skipping matching.")
            global_point_cloud_list.append(np.empty((0, 3)))
            continue
            
        frame_points = []
        
        # Chain matching: 0-1, 1-2, ..., (N-1)-0
        # For simplicity, we just do linear chain.
        pairs = []
        for i in range(num_cams - 1):
            pairs.append((i, i + 1))
        # Optional: Close loop if it's a ring
        # pairs.append((num_cams - 1, 0)) 
        
        for idx_i, idx_j in pairs:
            cam_i = current_cameras[idx_i]
            cam_j = current_cameras[idx_j]
            
            # Paths
            im_path_i = cam_i._image_path
            im_path_j = cam_j._image_path
            
            # Run RoMA
            matches = run_roma_matching(roma_model, im_path_i, im_path_j, device)
            
            if len(matches) < 10:
                continue
                
            # Triangulate
            P_i = get_camera_matrix(cam_i)
            P_j = get_camera_matrix(cam_j)
            
            pts_3d = triangulate_points(matches, P_i, P_j)
            
            # Simple filtering: remove points behind cameras or too far
            # Checking Z in camera space is good but we only have 3D world here
            # We can check distance from camera centers?
            # For now, just trust RoMA and keeping it simple.
            
            frame_points.append(pts_3d)
            
        if not frame_points:
            print(f"No points generated for frame {t}")
            global_point_cloud_list.append(np.empty((0, 3)))
            continue
            
        # Aggregate frame points
        cloud_at_t = np.concatenate(frame_points, axis=0)
        
        # Random Downsample
        if cloud_at_t.shape[0] > args.points_per_frame:
            indices = np.random.choice(cloud_at_t.shape[0], args.points_per_frame, replace=False)
            cloud_at_t = cloud_at_t[indices]
            
        global_point_cloud_list.append(cloud_at_t)
        
    # Phase 2: Temporal Motion Estimation
    print("Starting Phase 2: Temporal Motion Estimation...")
    
    final_data_containers = []
    total_frames = len(frame_indices) # This assumes continuous indices, but we iterate list
    
    for t_idx, t in enumerate(frame_indices):
        curr_cloud = global_point_cloud_list[t_idx]
        if curr_cloud.shape[0] == 0:
            continue
            
        # Determine target for motion
        if t_idx < total_frames - 1:
            target_cloud = global_point_cloud_list[t_idx + 1]
            if target_cloud.shape[0] == 0:
                # Fallback if next frame empty
                velocity = np.zeros_like(curr_cloud)
            else:
                # KNN
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_cloud)
                distances, indices = nbrs.kneighbors(curr_cloud)
                nearest_points = target_cloud[indices.flatten()]
                velocity = nearest_points - curr_cloud
        else:
            # Last frame
            velocity = np.zeros_like(curr_cloud)
            # Optional: use previous velocity?
            
        # Normalize time [0, 1]
        norm_t = t_idx / max(1, total_frames - 1)
        time_array = np.full(curr_cloud.shape[0], norm_t)
        
        # Dummy colors (gray)
        colors = np.ones_like(curr_cloud) * 0.5
        
        frame_data = {
            'xyz': curr_cloud,
            'motion': velocity,
            't': time_array,
            'colors': colors
        }
        final_data_containers.append(frame_data)
        
    if not final_data_containers:
        print("No valid data generated.")
        return

    # Phase 3: Aggregation and Save
    print("Starting Phase 3: Aggregation...")
    
    all_xyz = np.concatenate([d['xyz'] for d in final_data_containers], axis=0)
    all_motion = np.concatenate([d['motion'] for d in final_data_containers], axis=0)
    all_time = np.concatenate([d['t'] for d in final_data_containers], axis=0)
    all_colors = np.concatenate([d['colors'] for d in final_data_containers], axis=0)
    
    # Calculate Spatial Scale (dist to nearest neighbor)
    print("Calculating spatial scales...")
    # Using a subset for faster KNN if needed, but we usually need per-point
    # Note: simple_knn usually uses mean of nearest 3. 
    knn_scale = NearestNeighbors(n_neighbors=4, algorithm='kd_tree').fit(all_xyz) # k=4 because 1st is self
    dists, _ = knn_scale.kneighbors(all_xyz)
    # Average of closest 3 neighbors (excluding self at index 0)
    spatial_scale = np.mean(dists[:, 1:], axis=1) 
    spatial_scale_log = np.log(spatial_scale + 1e-6)
    
    # Temporal Scale (Duration)
    # Fix from FreeTimeGS_fix.md: Initialize with log(1.0) = 0.0 or log(0.5)
    # Using 0.0 to let it cover more time initially and then shrink via regularization
    t_scale_log = np.full(all_xyz.shape[0], 0.0)
    
    # Save
    save_freetime_ply(args.output_path, all_xyz, all_motion, all_time, t_scale_log, spatial_scale_log, all_colors)
    print("Done!")

if __name__ == "__main__":
    main()
