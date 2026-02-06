#!/usr/bin/env python3
"""
Gaussian Distribution Visualizer

This script loads a trained Gaussian Splatting PLY file and visualizes the distribution
of its parameters (Position, Scale, Rotation, Opacity, etc.).

Usage:
    python tools/visualize_gaussian_distribution.py --ply_path <path_to_ply_file>
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
from scipy.stats import norm

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Gaussian Parameters Distribution")
    parser.add_argument("--ply_path", type=str, required=True, help="Path to the trained PLY file")
    parser.add_argument("--output_dir", type=str, default="gaussian_analysis", help="Directory to save plots")
    parser.add_argument("--bins", type=int, default=100, help="Number of bins for histograms")
    return parser.parse_args()

def load_ply(path):
    print(f"Loading PLY file: {path}...")
    plydata = PlyData.read(path)
    
    # Extract data
    vertex = plydata['vertex']
    
    data = {}
    
    # 1. Position (XYZ)
    data['x'] = np.asarray(vertex['x'])
    data['y'] = np.asarray(vertex['y'])
    data['z'] = np.asarray(vertex['z'])
    
    # 2. Scale (Usually log scale in 3DGS)
    # Check for scale names
    scale_cols = [p.name for p in vertex.properties if p.name.startswith('scale_')]
    if scale_cols:
        scales = np.stack([np.asarray(vertex[n]) for n in scale_cols], axis=1)
        data['scale'] = scales # saved as log scale
        data['scale_exp'] = np.exp(scales) # actual scale
    
    # 3. Opacity (Usually logit in 3DGS)
    if 'opacity' in vertex:
        opac = np.asarray(vertex['opacity'])
        data['opacity_logit'] = opac
        
        # Determine activation
        # Standard 3DGS uses sigmoid(opacity)
        # But if the ply is saved 'baked' it might be different, but typically it's the parameter value
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        data['opacity'] = sigmoid(opac)
        
    # 4. Rotation
    rot_cols = [p.name for p in vertex.properties if p.name.startswith('rot_')]
    if rot_cols:
        data['rot'] = np.stack([np.asarray(vertex[n]) for n in rot_cols], axis=1)
        
    # 5. Features (DC)
    dc_cols = [p.name for p in vertex.properties if p.name.startswith('f_dc_')]
    if dc_cols:
        # Just simple magnitude check
        dc = np.stack([np.asarray(vertex[n]) for n in dc_cols], axis=1)
        data['colors_dc'] = dc
        
    # FreeTimeGS specific
    # t, t_scale, motion
    if 't' in vertex:
        data['t'] = np.asarray(vertex['t'])
    
    if 't_scale' in vertex:
        # Usually stored as log duration
        data['t_scale'] = np.asarray(vertex['t_scale'])
        data['t_duration'] = np.exp(data['t_scale'])
    
    return data

def plot_distributions(data, output_dir, bins=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Generating plots in {output_dir}...")
    
    # Set style
    plt.style.use('ggplot')
    
    # 1. Spatial Distribution (2D projections)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(data['x'], data['y'], s=0.1, alpha=0.1, c='b')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('XY Projection')
    axes[0].axis('equal')
    
    axes[1].scatter(data['x'], data['z'], s=0.1, alpha=0.1, c='g')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_title('XZ Projection')
    axes[1].axis('equal')
    
    axes[2].scatter(data['y'], data['z'], s=0.1, alpha=0.1, c='r')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_title('YZ Projection')
    axes[2].axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spatial_distribution.png"), dpi=150)
    plt.close()
    
    # 2. Scale Distribution
    if 'scale' in data:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Log Scales (Actual parameters)
        scales_flat = data['scale'].flatten()
        axes[0].hist(scales_flat, bins=bins, color='purple', alpha=0.7)
        axes[0].set_title('Log Scale Parameters')
        axes[0].set_xlabel('Log Scale value')
        
        # Exp Scales (Actual World sizes)
        scales_exp_flat = data['scale_exp'].flatten()
        # Cap outliers for viewing
        p99 = np.percentile(scales_exp_flat, 99)
        scales_exp_filtered = scales_exp_flat[scales_exp_flat < p99]
        
        axes[1].hist(scales_exp_filtered, bins=bins, color='orange', alpha=0.7)
        axes[1].set_title(f'Actual Scale Sizes (99% percentile < {p99:.4f})')
        axes[1].set_xlabel('World Scale')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "scale_distribution.png"))
        plt.close()
        
        # Anisotropy (Max / Min scale)
        if data['scale_exp'].shape[1] == 3:
            max_s = np.max(data['scale_exp'], axis=1)
            min_s = np.min(data['scale_exp'], axis=1)
            anisotropy = max_s / (min_s + 1e-8)
            
            plt.figure(figsize=(10, 6))
            p99_ani = np.percentile(anisotropy, 99)
            plt.hist(anisotropy[anisotropy < p99_ani], bins=bins, color='teal')
            plt.title('Anisotropy (Max Scale / Min Scale)')
            plt.xlabel('Ratio')
            plt.savefig(os.path.join(output_dir, "anisotropy_distribution.png"))
            plt.close()

    # 3. Opacity Distribution
    if 'opacity' in data:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Logits
        axes[0].hist(data['opacity_logit'], bins=bins, color='gray', alpha=0.7)
        axes[0].set_title('Opacity Logits (Parameter)')
        
        # Sigmoid
        axes[1].hist(data['opacity'], bins=bins, color='black', alpha=0.7)
        axes[1].set_title('Actual Opacity (0.0 - 1.0)')
        axes[1].set_yscale('log') # Log scale because usually many are near 0 or 1
        axes[1].set_ylabel('Count (Log Scale)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "opacity_distribution.png"))
        plt.close()

    # 4. Rotation Distribution (Quaternions)
    if 'rot' in data:
        # Norm should be 1, but check
        rots = data['rot']
        norms = np.linalg.norm(rots, axis=1)
        
        plt.figure(figsize=(8, 6))
        plt.hist(norms, bins=50, color='brown')
        plt.title('Quaternion Norms (Should be ~1.0 if normalized)')
        plt.xlabel('Norm')
        plt.savefig(os.path.join(output_dir, "rotation_norm.png"))
        plt.close()

    # 5. Temporal (FreeTimeGS)
    if 't' in data:
        plt.figure(figsize=(10, 6))
        plt.hist(data['t'], bins=bins, color='cyan', alpha=0.7)
        plt.title('Temporal Center (t) Distribution')
        plt.xlabel('Time')
        plt.savefig(os.path.join(output_dir, "time_distribution.png"))
        plt.close()

    if 't_scale' in data:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Log t_scale
        axes[0].hist(data['t_scale'], bins=bins, color='purple', alpha=0.7)
        axes[0].set_title('Log Temporal Scale/Duration')
        axes[0].set_xlabel('Log Duration')
        
        # Actual Duration (Exp)
        # Cap outliers
        durs = data['t_duration']
        p99 = np.percentile(durs, 99)
        durs_filt = durs[durs < p99]
        
        axes[1].hist(durs_filt, bins=bins, color='magenta', alpha=0.7)
        axes[1].set_title(f'Temporal Duration (Seconds) (99% < {p99:.4f})')
        axes[1].set_xlabel('Duration')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "time_scale_distribution.png"))
        plt.close()
        
    print(f"All plots saved to {output_dir}")

def main():
    args = parse_args()
    
    if not os.path.exists(args.ply_path):
        print(f"Error: File not found: {args.ply_path}")
        return

    data = load_ply(args.ply_path)
    
    print(f"Loaded {len(data['x'])} Gaussians.")
    
    plot_distributions(data, args.output_dir, args.bins)

if __name__ == "__main__":
    main()
