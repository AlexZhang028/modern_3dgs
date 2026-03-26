#!/usr/bin/env python3
"""
SelfCap to FreeTimeGS Initialization Script

This script processes multi-frame point clouds (e.g. from SelfCap dataset)
to generate an initialization PLY file for FreeTimeGS training.

It performs:
1. Loading sequence of PLY files.
2. Computing motion vectors (velocity) between adjacent frames using KNN.
3. Assigning time timestamps.
4. Aggregating all points.
5. Computing initial spatial scales.
6. Saving to FreeTimeGS-compatible PLY format.

Usage:
    python tools/process_selfcap_pcd.py --source_path <path_to_dataset_root> --output_path <path_to_output.ply>
    
    Example:
    python tools/process_selfcap_pcd.py --source_path ../3DGS_test_data/bar-release --output_path init_freetime.ply
"""

import os
import argparse
import glob
import numpy as np
import torch
import yaml
from plyfile import PlyData, PlyElement
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Process SelfCap PCDs for FreeTimeGS")
    parser.add_argument("--config", type=str, default="", help="Path to YAML config file")
    parser.add_argument("--source_path", type=str, default=None, help="Path to the dataset root (containing pcds/ folder)")
    parser.add_argument("--output_path", type=str, default="init_freetime.ply", help="Output PLY name")
    parser.add_argument("--pcd_subfolder", type=str, default="pcds", help="Subfolder containing per-frame PLYs")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index (default: 0)")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame index (default: -1, meaning all)")
    parser.add_argument("--downsample_rate", type=int, default=10, help="Process every K-th frame (default: 10)")
    parser.add_argument("--downsample_points", type=int, default=10000, help="Max points per frame to load initially (random sample)")
    parser.add_argument("--max_total_points", type=int, default=100000, help="Maximum allowable points in final PLY. Will randomly downsample if exceeded.")
    parser.add_argument("--static_vel_threshold", type=float, default=0.1, help="Velocity magnitude threshold to consider a point static (background).")
    parser.add_argument("--static_keep_ratio", type=float, default=0.1, help="Ratio of static points to  keep (0.0-1.0).")
    parser.add_argument("--max_vel_threshold", type=float, default=2.0, help="Max allowable velocity; points exceeding this are discarded as noise (flying points).")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda/cpu)")
    parser.add_argument("--fps", type=float, default=60.0, help="Frames per second (default: 30.0)")
    parser.add_argument("--sh_degree", type=int, default=3, help="SH degree for initialization (default: 3)")
    parser.add_argument("--color_weight", type=float, default=0.5, help="Weight for color in KNN matching (Scheme A)")
    parser.add_argument("--smooth_k", type=int, default=5, help="KNN neighborhood size for velocity median smoothing (Scheme C)")
    
    args = parser.parse_args()
    
    # Load parameters from yaml if provided.
    # Note: YAML values overwrite command line defaults when specified.
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                for k, v in yaml_config.items():
                    if hasattr(args, k):
                        setattr(args, k, v)
    
    if args.source_path is None:
        parser.error("--source_path must be provided via command line or config yaml.")
        
    return args

def read_ply_points(path):
    plydata = PlyData.read(path)
    # Assume standard PLY with x, y, z, red, green, blue
    x = np.asarray(plydata['vertex']['x'])
    y = np.asarray(plydata['vertex']['y'])
    z = np.asarray(plydata['vertex']['z'])
    
    try:
        r = np.asarray(plydata['vertex']['red'])
        g = np.asarray(plydata['vertex']['green'])
        b = np.asarray(plydata['vertex']['blue'])
    except:
        # Fallback if no color
        N = len(x)
        r = np.ones(N) * 128
        g = np.ones(N) * 128
        b = np.ones(N) * 128
        
    points = np.stack([x, y, z], axis=1)
    colors = np.stack([r, g, b], axis=1) / 255.0 # Normalize 0-1
    return points, colors

def compute_scales(points, k=3, device='cuda'):
    """
    Compute scales using simple_knn (fast) if available, otherwise fallback to pytorch (slow).
    """
    if not torch.cuda.is_available():
        device = 'cpu'
    
    pts = torch.from_numpy(points).float().to(device)
    
    # Method 1: Try simple_knn (Standard 3DGS library) - FASTEST
    try:
        from simple_knn._C import distCUDA2
        # distCUDA2 returns the average of the squared distances to the nearest 3 neighbors
        # We need the sqrt of that to get the distance.
        dist_sq = distCUDA2(pts)
        scales = torch.sqrt(dist_sq)
        return scales.cpu().numpy()
    except ImportError:
        pass
        
    print("Warning: simple_knn not found. Using slow PyTorch fallback for scale computation.")

    # Method 2: Fallback PyTorch implementation
    # Handle large N with chunking if necessary
    N = pts.shape[0]
    if N > 30000:
        # Simple chunking approach for query
        scales = []
        chunk_size = 5000
        for i in range(0, N, chunk_size):
            chunk = pts[i:i+chunk_size]
            dist = torch.cdist(chunk.unsqueeze(0), pts.unsqueeze(0)).squeeze(0) # [M, N]
            # dist contains 0 for self.
            # topk k+1 (smallest)
            val, _ = torch.topk(dist, k=k+1, dim=1, largest=False)
            # exclude first (self) which is 0
            val = val[:, 1:]
            scales.append(val.mean(dim=1))
        scales = torch.cat(scales, dim=0)
    else:
        dist = torch.cdist(pts.unsqueeze(0), pts.unsqueeze(0)).squeeze(0)
        val, _ = torch.topk(dist, k=k+1, dim=1, largest=False)
        val = val[:, 1:]
        scales = val.mean(dim=1)
        
    return scales.cpu().numpy()

def smooth_velocities(pts_np, vel_np, k=5, device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
        
    pts = torch.from_numpy(pts_np).float().to(device)
    vel = torch.from_numpy(vel_np).float().to(device)
    N = pts.shape[0]
    
    smoothed = torch.zeros_like(vel)
    chunk_size = 5000
    for i in range(0, N, chunk_size):
        chunk = pts[i:i+chunk_size]
        dist = torch.cdist(chunk.unsqueeze(0), pts.unsqueeze(0)).squeeze(0) # [Batch, N]
        _, idx = torch.topk(dist, k=k+1, dim=1, largest=False)
        # gather neighbor velocities
        neighbor_vels = vel[idx] # [Batch, k+1, 3]
        # median smoothing (dim 1 is the neighborhood)
        median_vel, _ = torch.median(neighbor_vels, dim=1)
        smoothed[i:i+chunk_size] = median_vel
        
    return smoothed.cpu().numpy()

def find_knn_correspondence(query_np, target_np, query_col_np=None, target_col_np=None, color_weight=0.0, k=1, device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
        
    query = torch.from_numpy(query_np).float().to(device)
    target = torch.from_numpy(target_np).float().to(device)
    
    if color_weight > 0.0 and query_col_np is not None and target_col_np is not None:
        q_cols = torch.from_numpy(query_col_np).float().to(device)
        t_cols = torch.from_numpy(target_col_np).float().to(device)
        query = torch.cat([query, q_cols * (color_weight ** 0.5)], dim=1)
        target = torch.cat([target, t_cols * (color_weight ** 0.5)], dim=1)
    
    N = query.shape[0]
    M = target.shape[0]
    
    indices_list = []
    
    # Chunking query
    chunk_size = 5000
    for i in range(0, N, chunk_size):
        q_chunk = query[i:i+chunk_size]
        # Check dist against all target
        # Careful with memory if M is huge
        if M > 30000:
             # Loop over target chunks? 
             # For exact NN, we need min over all targets.
             # Easier to just rely on PyTorch cdist memory management or assume M ~ 20k
             pass
             
        dist = torch.cdist(q_chunk.unsqueeze(0), target.unsqueeze(0)).squeeze(0) # [Batch, M]
        _, idx = torch.topk(dist, k=k, dim=1, largest=False)
        indices_list.append(idx)
        
    indices = torch.cat(indices_list, dim=0)
    return indices.cpu().numpy()

def main():
    args = parse_args()
    
    pcd_dir = os.path.join(args.source_path, args.pcd_subfolder)
    if not os.path.exists(pcd_dir):
        print(f"Error: Directory {pcd_dir} not found.")
        return

    # Sort by frame index extracted from filename
    def extract_frame_idx(fname):
        base = os.path.basename(fname)
        # Find all digits
        import re
        nums = re.findall(r'\d+', base)
        if nums:
            return int(nums[-1]) # Assume last number is frame index
        return -1

    files = sorted(glob.glob(os.path.join(pcd_dir, "*.ply")), key=extract_frame_idx)
    if not files:
        print(f"Error: No .ply files found in {pcd_dir}")
        return

    # Determine Base Index (Logic: First file represents '0' offset? 
    # Or start_frame relative to sorted list?)
    # User requirement: "if first is 5200, start_frame 10 should load 5210"
    # This implies start_frame is relative to the BEGINNING of the sequence found on disk.
    
    first_file_idx = extract_frame_idx(files[0])
    target_start_idx = first_file_idx + args.start_frame
    if args.end_frame != -1:
        target_end_idx = first_file_idx + args.end_frame
    else:
        target_end_idx = -1 # All
        
    print(f"Dataset First Frame: {first_file_idx}")
    print(f"Processing Range (Absolute): {target_start_idx} - {target_end_idx if target_end_idx != -1 else 'End'}")

    filtered_files = []
    for f in files:
        idx = extract_frame_idx(f)
        if idx >= target_start_idx:
             if target_end_idx != -1 and idx >= target_end_idx:
                 continue # Don't break immediately if not strictly contiguous, but here sorted...
             filtered_files.append(f)
    files = filtered_files
    
    # Filter by frame range (if user specifies start/end frame indices, matched against filename index)
    # (Removed old logic to avoid double filtering)

    if not files:
        print(f"Error: No files found in target range.")
        return

    # Downsample frames
    files = files[::args.downsample_rate]
    print(f"Found {len(files)} frames to process (step {args.downsample_rate}).")

    global_points = []
    global_colors = []
    
    # Phase 1: Read all frames
    print("Loading point clouds...")
    frame_data_list = []
    
    for i, fpath in enumerate(files):
        print(f"Reading frame {i}/{len(files)-1}: {os.path.basename(fpath)}", end='\r')
        pts, cols = read_ply_points(fpath)
        
        # Random downsample if too many points
        if len(pts) > args.downsample_points:
            indices = np.random.choice(len(pts), args.downsample_points, replace=False)
            pts = pts[indices]
            cols = cols[indices]
        
        frame_idx = extract_frame_idx(fpath)
        frame_data_list.append({
            'pts': pts,
            'cols': cols,
            'frame_idx': frame_idx
        })
    print("\nLoading complete.")

    # Phase 2: Compute Motion and Time
    print("Computing motion (KNN)...")
    final_xyz = []
    final_rgb = []
    final_motion = []
    final_t = []
    final_t_scale = []
    
    total_frames = len(frame_data_list)
    
    # Try to load sync.json to correct timestamp if available
    sync_offset = 0.0
    
    # Time delta for velocity calculation
    dt = args.downsample_rate / args.fps
    if dt <= 0: dt = 1.0/30.0 # Safety
    
    for i in range(total_frames):
        curr_pts = frame_data_list[i]['pts']
        curr_cols = frame_data_list[i]['cols']
        frame_idx = frame_data_list[i]['frame_idx']
        
        # Time setup
        # Timestamp should be relative to the requested start_frame
        # If user asked for start_frame 10 (abs 5210), they expect t=0 to be at that point?
        # Typically yes, training starts at t=0.
        # User formula: time_sec = frameidx / 60. sync_time = time_sec - sync.
        
        # For PCD tool, we define 't' as relevant to the training Time 0.
        # Let's use (frame_idx - target_start_idx) / fps.
        # This assumes PCD filenames are synced with the "Master" timeline defined by start_frame.
        
        time_seconds = (frame_idx - target_start_idx) / args.fps
        
        # Determine target frame for motion estimation
        if i < total_frames - 1:
            target_pts = frame_data_list[i+1]['pts']
        else:
            # Last frame: use previous frame to estimate "backwards" or just zero?
            # Doc suggests: velocity = 0 or use prev. We use velocity=0 for last frame simplicity for now
            # or better matches doc: "if t < Total -1 ... else velocity = 0"
            target_pts = None 
            
        if target_pts is not None:
            # KNN to find correspondence
            target_cols = frame_data_list[i+1]['cols']
            indices = find_knn_correspondence(curr_pts, target_pts, curr_cols, target_cols, color_weight=args.color_weight, k=1, device=args.device)
            indices = indices.flatten()
            
            nearest_pts = target_pts[indices]
            displacement = nearest_pts - curr_pts
            velocity = displacement / dt
            
            if args.smooth_k > 0:
                velocity = smooth_velocities(curr_pts, velocity, k=args.smooth_k, device=args.device)
        else:
            velocity = np.zeros_like(curr_pts)
            
        # Optimization: Filter static background points
        # Calculate speed (magnitude of velocity)
        speed = np.linalg.norm(velocity, axis=1)
        is_static = speed < args.static_vel_threshold
        is_flying = speed > args.max_vel_threshold
        
        # We explicitly force static points to have strictly ZERO velocity
        # so they don't do "Brownian motion" when retained (e.g. from Frame 0)
        velocity[is_static] = 0.0
        
        # Decide which points to keep
        # Start by keeping everything
        keep_mask = np.ones(len(curr_pts), dtype=bool)
        
        # Identify static and flying indices
        static_indices = np.where(is_static)[0]
        flying_indices = np.where(is_flying)[0]
        
        # 1. Drop flying points completely (these are usually KNN mismatch noise)
        keep_mask[flying_indices] = False
        
        if len(static_indices) > 0:
            # Apply ratio to limit static density across ALL frames
            # This ensures we don't accidentally wipe out dynamic points that temporarily stop moving
            if args.static_keep_ratio < 1.0:
                num_keep = int(len(static_indices) * args.static_keep_ratio)
                keep_mask[static_indices] = False
                if num_keep > 0:
                    winners = np.random.choice(static_indices, num_keep, replace=False)
                    keep_mask[winners] = True
        
        # Filter arrays
        curr_pts_filtered = curr_pts[keep_mask]
        curr_cols_filtered = curr_cols[keep_mask]
        velocity_filtered = velocity[keep_mask]

        # Time in seconds
        # timestamp was calculated at start of loop
        # time_seconds = timestamp 
        # (var name changed in loop setup, using that)
        
        # normalized t (old behavior) -> seconds (new behavior)
        # norm_t_val = t / max(1, (total_frames - 1))
        
        # Append
        final_xyz.append(curr_pts_filtered)
        final_rgb.append(curr_cols_filtered)
        final_motion.append(velocity_filtered)
        final_t.append(np.full((len(curr_pts_filtered), 1), time_seconds))
        
        # Duration init (log scale)
        # Fix from FreeTimeGS_fix.md: Initialize with log(1.0) or log(0.5)
        # Using 0.0 to let it cover more time initially and then shrink via regularization
        final_t_scale.append(np.full((len(curr_pts_filtered), 1), -2))
        
        static_count = len(static_indices)
        dropped_count = static_count - np.sum(keep_mask[static_indices])
        dropped_flying = len(flying_indices)
        print(f"Processed frame {i}/{total_frames-1}. Dropped {dropped_count} static, {dropped_flying} flying points.", end='\r')
        
    print("\nAggregation...")
    
    all_xyz = np.concatenate(final_xyz, axis=0)
    all_rgb = np.concatenate(final_rgb, axis=0)
    all_motion = np.concatenate(final_motion, axis=0)
    all_t = np.concatenate(final_t, axis=0)
    all_t_scale = np.concatenate(final_t_scale, axis=0)
    
    # Global Downsample if needed
    total_aggregated = len(all_xyz)
    if args.max_total_points > 0 and total_aggregated > args.max_total_points:
        print(f"\nTotal points {total_aggregated} exceeds limit {args.max_total_points}. Downsampling...")
        keep_indices = np.random.choice(total_aggregated, args.max_total_points, replace=False)
        all_xyz = all_xyz[keep_indices]
        all_rgb = all_rgb[keep_indices]
        all_motion = all_motion[keep_indices]
        all_t = all_t[keep_indices]
        all_t_scale = all_t_scale[keep_indices]
        
        # Determine strict frame affiliation for scale computation is now harder after shuffle
        # So we perform scale computation BEFORE shuffle? 
        # Or just compute scales on the reduced set? 
        # Computing scales on reduced set is faster but density estimation might be slightly lower (larger scales).
        # Given we want initialization scales, "larger" is usually safer than "too small".
        # Let's compute scales on the reduced set.
    
    # Phase 3: Spatial Scaling
    print(f"Computing spatial scales for {len(all_xyz)} points... (this may take a while)")
    
    # Since we might have shuffled/merged frames, we just compute global KNN now.
    # If N is huge, chunking in compute_scales handles it.
    all_scales = compute_scales(all_xyz, device=args.device)
    
    # Log scale for storage (Standard 3DGS model expects LOG scales in parameter, but usually PLY stores raw scales? 
    # WAIT. core/gaussian_model.py load_ply reads 'scale_0' etc.
    # And then: self._gaussian_params['scaling'] = create_param(scales)
    # And get_scaling returns torch.exp(param).
    # So the parameter is log-scale.
    # The PLY loader in GaussianModel reads it as is.
    # Does standard 3DGS init save log-scale or linear scale in PLY?
    # Looking at `save_ply`: `scales = get_tensor('scaling').detach().cpu().numpy()` which is the PARAMETER (log scale).
    # So PLY usually contains LOG SCALES.
    # Converting linear distance to log.
    
    all_scales_log = np.log(np.clip(all_scales, 1e-6, None))
    # Duplicate to 3 dims
    all_scales_log_3d = np.repeat(all_scales_log[:, None], 3, axis=1)

    # Basic Rotation (Identity) [1, 0, 0, 0]
    num_points = len(all_xyz)
    all_rots = np.zeros((num_points, 4))
    all_rots[:, 0] = 1.0

    # SH Features (DC only for init, rest 0)
    # SH Rest (Higher orders)
    # If args.sh_degree > 0, we should initialize f_rest to 0 to avoid garbage colors if loaded directly
    num_rest_coeffs = ((args.sh_degree + 1) ** 2 - 1) * 3
    if num_rest_coeffs > 0:
        all_f_rest = np.zeros((num_points, num_rest_coeffs), dtype=np.float32)
    else:
        all_f_rest = None
    
    # Convert RGB to SH DC
    # C0 = 0.28209479177387814
    C0 = 0.28209479177387814
    all_f_dc = (all_rgb - 0.5) / C0
    
    # Opacity (Logit)
    # standard init is 0.1 -> inverse_sigmoid(0.1)
    from scipy.special import logit
    # We store raw opacity or logit?
    # load_ply: opacities = np.asarray(...)
    # self._gaussian_params['opacity'] = create_param(opacities)
    # And get_opacity uses sigmoid.
    # So PLY stores LOGIT opacity (inverse sigmoid).
    all_opacities = np.full((num_points, 1), logit(0.1))

    # Construct PLY
    print("Constructing PLY...")
    # Dtypes
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), # Standard normals
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]
    
    # 动态添加高阶球谐特征 f_rest
    if all_f_rest is not None:
        for i in range(all_f_rest.shape[1]):
            dtype.append((f'f_rest_{i}', 'f4'))

    dtype.extend([
        ('t', 'f4'), ('t_scale', 'f4'),
        ('motion_0', 'f4'), ('motion_1', 'f4'), ('motion_2', 'f4')
    ])
    
    elements = np.empty(num_points, dtype=dtype)
    elements['x'] = all_xyz[:, 0]
    elements['y'] = all_xyz[:, 1]
    elements['z'] = all_xyz[:, 2]
    
    # Setting normals to 0 or motion? 
    # Doc said "borrow normals for motion", but we have explicit motion_0 fields.
    # We'll just set normals to 0 for standard viewers or equal to motion for visualization.
    elements['nx'] = all_motion[:, 0]
    elements['ny'] = all_motion[:, 1]
    elements['nz'] = all_motion[:, 2]
    
    elements['red'] = (all_rgb[:, 0] * 255).astype('u1')
    elements['green'] = (all_rgb[:, 1] * 255).astype('u1')
    elements['blue'] = (all_rgb[:, 2] * 255).astype('u1')
    
    elements['opacity'] = all_opacities[:, 0]
    elements['scale_0'] = all_scales_log_3d[:, 0]
    elements['scale_1'] = all_scales_log_3d[:, 1]
    elements['scale_2'] = all_scales_log_3d[:, 2]
    
    elements['rot_0'] = all_rots[:, 0]
    elements['rot_1'] = all_rots[:, 1]
    elements['rot_2'] = all_rots[:, 2]
    elements['rot_3'] = all_rots[:, 3]
    
    elements['f_dc_0'] = all_f_dc[:, 0]
    elements['f_dc_1'] = all_f_dc[:, 1]
    elements['f_dc_2'] = all_f_dc[:, 2]
    
    if all_f_rest is not None:
        for i in range(all_f_rest.shape[1]):
            elements[f'f_rest_{i}'] = all_f_rest[:, i]
    
    elements['t'] = all_t[:, 0]
    elements['t_scale'] = all_t_scale[:, 0]
    elements['motion_0'] = all_motion[:, 0]
    elements['motion_1'] = all_motion[:, 1]
    elements['motion_2'] = all_motion[:, 2]
    
    # Save
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(args.output_path)
    print(f"Saved {num_points} points to {args.output_path}")

if __name__ == "__main__":
    main()
