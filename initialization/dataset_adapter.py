"""
Dataset adapter: prepares a COLMAP image workspace from a SelfCap source_path.

Reads the SelfCap camera list, locates the reference-frame images under
frames/{cam_name}/{eff_idx:05d}.jpg (or extracts directly from video if the
frames directory does not exist), and copies them into {work_dir}/images/ with
filenames that COLMAP can discover.

Returns a WorkspaceInfo with video metadata needed for temporal prior assignment.
"""

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class WorkspaceInfo:
    """Metadata produced by prepare_colmap_workspace."""
    image_dir: Path                           # work_dir / "images"
    fps: float
    n_frames: int                             # number of frames in training range
    duration_seconds: float                   # (n_frames - 1) / fps
    cam_names: List[str] = field(default_factory=list)
    # Known intrinsics per camera (K [3,3]), empty when unavailable.
    intrinsics: Dict[str, np.ndarray] = field(default_factory=dict)


def prepare_colmap_workspace(
    source_path: str,
    work_dir: str,
    start_frame: int = 0,
    end_frame: int = -1,
    reference_frame: Optional[int] = None,
) -> WorkspaceInfo:
    """
    Prepare a COLMAP image directory for the static branch.

    For each camera, one image (the reference frame) is placed into
    {work_dir}/images/{cam_name}.jpg.  The reference frame defaults to
    start_frame (the first frame of the training range).

    Args:
        source_path:     Dataset root (contains extri.yml / videos/).
        work_dir:        Scratch directory for COLMAP artefacts.
        start_frame:     First master frame index of the training range.
        end_frame:       Last master frame index (-1 = all frames in video).
        reference_frame: Master frame index used for static branch images.
                         Defaults to start_frame.

    Returns:
        WorkspaceInfo with image_dir, fps, n_frames, duration_seconds.
    """
    source = Path(source_path)
    work   = Path(work_dir)
    image_dir = work / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    sys_path_insert(str(Path(__file__).parent.parent))
    from data.selfcap_loader import read_selfcap_cameras, read_selfcap_sync

    cam_names, _, intrinsics_raw = read_selfcap_cameras(str(source))
    sync_map = read_selfcap_sync(str(source))

    # Locate video files
    videos_root = source / "videos"
    if not videos_root.exists():
        videos_root = source
    video_map: Dict[str, Path] = {
        Path(f).stem: videos_root / f
        for f in os.listdir(videos_root)
        if f.lower().endswith(".mp4")
    }
    if not video_map:
        raise ValueError(f"No .mp4 files found in {videos_root}")

    # Read FPS and total frames from the first available video
    first_video = next(iter(video_map.values()))
    cap = cv2.VideoCapture(str(first_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    end = end_frame if (0 < end_frame <= total) else total
    n_frames = max(end - start_frame, 1)
    duration_seconds = (n_frames - 1) / fps if n_frames > 1 else 1.0 / fps

    ref_master = reference_frame if reference_frame is not None else start_frame

    frames_root = source / "frames"

    # Parse known intrinsics (K matrix per camera) when available.
    intrinsics: Dict[str, np.ndarray] = {}
    for name, intr in intrinsics_raw.items():
        if intr and "K" in intr:
            intrinsics[name] = intr["K"]

    registered: List[str] = []
    for name in cam_names:
        if name not in video_map:
            continue

        sync_offset = sync_map.get(name, 0.0)
        eff_idx = int(round(ref_master + sync_offset * fps))
        dst = image_dir / f"{name}.jpg"

        # Prefer pre-extracted frames; fall back to on-the-fly video read.
        src_jpg = frames_root / name / f"{eff_idx:05d}.jpg"
        if src_jpg.exists():
            shutil.copy2(str(src_jpg), str(dst))
        else:
            _extract_single_frame(str(video_map[name]), eff_idx, str(dst))

        if dst.exists():
            registered.append(name)

    if not registered:
        raise RuntimeError(
            "No reference-frame images could be placed in the COLMAP workspace. "
            "Run tools/extract_frames.py first, or check that videos/ exists."
        )

    print(f"[adapter] Prepared {len(registered)} images in {image_dir}")
    print(f"[adapter] fps={fps:.2f}  n_frames={n_frames}  ref_master={ref_master}")

    return WorkspaceInfo(
        image_dir=image_dir,
        fps=fps,
        n_frames=n_frames,
        duration_seconds=duration_seconds,
        cam_names=registered,
        intrinsics=intrinsics,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_single_frame(video_path: str, frame_idx: int, dst_path: str) -> None:
    """Extract one frame from a video and save as JPEG."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  Warning: frame {frame_idx} not readable from {video_path}")
        return
    cv2.imwrite(dst_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


def sys_path_insert(path: str) -> None:
    import sys
    if path not in sys.path:
        sys.path.insert(0, path)
