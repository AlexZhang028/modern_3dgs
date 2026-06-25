"""
Standalone frame extraction tool for SelfCap video datasets.

Extracts frames from videos to {source_path}/frames/{cam_name}/{eff_idx:05d}.jpg
taking sync offsets into account. Writes frames/meta.json for verification.

Usage:
    python tools/extract_frames.py --source_path /path/to/dataset [options]
"""

import os
import sys
import json
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.selfcap_loader import read_selfcap_cameras, read_selfcap_sync


def extract_frames(
    source_path: str,
    start_frame: int = 0,
    end_frame: int = -1,
    quality: int = 95,
    overwrite: bool = False,
) -> None:
    source = Path(source_path)
    frames_root = source / "frames"

    cam_names, _, _ = read_selfcap_cameras(str(source))
    sync_map = read_selfcap_sync(str(source))

    videos_root = source / "videos"
    if not videos_root.exists():
        videos_root = source

    video_files = {Path(f).stem: str(videos_root / f)
                   for f in os.listdir(videos_root) if f.lower().endswith('.mp4')}
    if not video_files:
        raise ValueError(f"No .mp4 files found in {videos_root}")

    # Determine fps and total frames from first available video
    first_video = next(iter(video_files.values()))
    cap = cv2.VideoCapture(first_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    end = end_frame if (end_frame > 0 and end_frame <= total_frames) else total_frames
    selected_indices = list(range(start_frame, end))
    print(f"Source  : {source}")
    print(f"Frames  : {start_frame}–{end}  ({len(selected_indices)} master frames, fps={fps:.2f})")
    print(f"Cameras : {len(cam_names)}")

    # Build per-camera extraction plan
    # For each camera: list of (master_idx, eff_idx)
    plans = {}
    for name in cam_names:
        if name not in video_files:
            continue
        sync_offset = sync_map.get(name, 0.0)
        frames_dir = frames_root / name
        plan = []
        for idx in selected_indices:
            eff_idx = int(round(idx + sync_offset * fps))
            out_path = frames_dir / f"{eff_idx:05d}.jpg"
            if not overwrite and out_path.exists():
                continue
            plan.append(eff_idx)
        if plan:
            plans[name] = plan

    if not plans:
        print("All frames already extracted. Use --overwrite to re-extract.")
        return

    # Extract per video (sequential read for speed)
    for name, eff_indices in plans.items():
        video_path = video_files[name]
        frames_dir = frames_root / name
        frames_dir.mkdir(parents=True, exist_ok=True)

        needed = set(eff_indices)
        max_frame = max(needed)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open {video_path}, skipping.")
            continue

        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        current = 0
        saved = 0

        pbar = tqdm(total=max_frame + 1, desc=f"  {name}", leave=False)
        while current <= max_frame:
            if current in needed:
                ret, frame = cap.read()
                if not ret:
                    print(f"  Warning: Reached end of {name} at frame {current}")
                    break
                out_path = frames_dir / f"{current:05d}.jpg"
                cv2.imwrite(str(out_path), frame, encode_params)
                saved += 1
            else:
                cap.grab()
            current += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        print(f"  {name}: saved {saved} frames → {frames_dir}")

    # Write meta.json
    meta = {
        "fps": fps,
        "start_frame": start_frame,
        "end_frame": end,
        "quality": quality,
        "cameras": list(plans.keys()),
    }
    meta_path = frames_root / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. Meta written to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract SelfCap video frames to JPEG files")
    parser.add_argument("--source_path", required=True, help="Dataset root directory")
    parser.add_argument("--start_frame", type=int, default=0, help="First master frame index")
    parser.add_argument("--end_frame", type=int, default=-1, help="Last master frame index (-1 = all)")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if files exist")
    args = parser.parse_args()

    extract_frames(
        source_path=args.source_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        quality=args.quality,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
