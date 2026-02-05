#!/usr/bin/env python3
"""
Test Script for SelfCap Dataset Loading and Sampling
"""

import sys
import os
import argparse
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import SelfCapVideoDataset
from data.samplers import TemporalSampler

def main():
    parser = argparse.ArgumentParser(description="Test SelfCap Dataset Loading")
    parser.add_argument("--source_path", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame index")
    parser.add_argument("--show_image_stats", action="store_true", help="Print image statistics")
    args = parser.parse_args()

    print(f"Testing dataset at: {args.source_path}")
    print(f"Conf: Start={args.start_frame}, End={args.end_frame}")

    # 1. Initialize Dataset
    print("\n[Initializing SelfCapVideoDataset]...")
    try:
        dataset = SelfCapVideoDataset(
            source_path=args.source_path,
            split="train",
            resolution=-1,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
    except Exception as e:
        print(f"FAILED to initialize dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Dataset successfully loaded.")
    print(f"Total Cameras (Frames): {len(dataset)}")

    if len(dataset) == 0:
        print("Error: No cameras/frames found. Exiting.")
        return

    # Check boundaries
    print("\n[Checking Frame Boundaries]")
    
    first_item = dataset[0]
    first_cam = first_item['camera']
    print(f"First Frame (Index 0): {first_cam.image_name}")
    print(f"  Timestamp (Norm): {first_cam.timestamp:.6f}")
    print(f"  Timestamp (Sec):  {first_cam.timestamp_seconds:.6f}s")
    
    last_idx = len(dataset) - 1
    last_item = dataset[last_idx]
    last_cam = last_item['camera']
    print(f"Last Frame (Index {last_idx}): {last_cam.image_name}")
    print(f"  Timestamp (Norm): {last_cam.timestamp:.6f}")
    print(f"  Timestamp (Sec):  {last_cam.timestamp_seconds:.6f}s")
    
    # 2. Initialize Sampler
    print("\n[Initializing TemporalSampler]...")
    sampler = TemporalSampler(dataset)
    
    # 3. Test Sampling Loop
    print("\n[Testing Random Sampling]")
    num_samples = 5
    
    for i in range(num_samples):
        print(f"--- Sample {i+1}/{num_samples} ---")
        camera, timestamp = sampler.sample()
        
        print(f"  Camera Name: {camera.image_name}")
        print(f"  Returned Timestamp: {timestamp:.6f}")
        
        # Verify timestamps match
        if camera.timestamp is not None:
            diff = abs(camera.timestamp - timestamp)
            status = "MATCH" if diff < 1e-6 else "MISMATCH"
            print(f"  Camera Timestamp:   {camera.timestamp:.6f} [{status}]")
        else:
            print(f"  Camera Timestamp:   None (Error?)")
            
        print(f"  Real Time (Sec):    {camera.timestamp_seconds:.6f}s")
        
        # Verify Image Loading
        if camera.image is not None:
            c, h, w = camera.image.shape
            print(f"  Image Data: Loaded Tensor [{c}, {h}, {w}]")
            if args.show_image_stats:
                print(f"  Min: {camera.image.min():.4f}, Max: {camera.image.max():.4f}, Mean: {camera.image.mean():.4f}")
        else:
            print("  Image Data: NOT LOADED (Error?)")

    print("\nSuccess! Dataset and Sampler appear to be working.")

if __name__ == "__main__":
    main()
