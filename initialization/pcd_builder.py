"""
Assemble the final SharpTimeGS initialization PLY from static and dynamic branches.

Pipeline:
  1. Static branch  → static_init.ply  (one PLY, all frames, wide temporal scope)
  2. Dynamic branch → dynamic_init.ply (one PLY, one entry per dynamic point per frame)
  3. merge_plys     → sharptime_init.ply (concatenated)

The dynamic PLY is built by:
  - Iterating over per-frame triangulated point clouds.
  - Assigning dynamic temporal priors: t = frame_time, t_scale = log(3·Δt).
  - Assigning KNN-estimated velocities as the motion field.

All temporal fields are in **seconds** relative to start_frame.
Normalization to [0,1] is applied inside SharpTimeGaussianModel.create_from_pcd.
"""

import math
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).parent.parent))
from initialization.pcd_utils import dynamic_temporal_priors, static_temporal_priors, write_init_ply


# ---------------------------------------------------------------------------
# Build dynamic branch PLY
# ---------------------------------------------------------------------------

def build_dynamic_ply(
    frames_data: Dict[int, Dict[str, np.ndarray]],
    velocities: Dict[int, np.ndarray],
    output_path: str,
    fps: float,
    start_frame: int = 0,
    n_duplicate: int = 1,
    duplicate_jitter_sigma: float = 0.5,
) -> str:
    """
    Assemble a PLY from per-frame dynamic point clouds.

    Args:
        frames_data:            Dict master_t → {"xyz": [N,3], "rgb": [N,3]}.
        velocities:             Dict master_t → [N,3] velocity (world units/s).
        output_path:            Where to write the dynamic PLY.
        fps:                    Frames per second (used to convert frame index to seconds).
        start_frame:            Master frame index of the first training frame
                                (subtracted so that the first frame has t=0).
        n_duplicate:            Replicate each point this many times total (1 = off).
                                Each copy receives independent Gaussian XYZ jitter;
                                velocity and temporal anchor are inherited unchanged.
        duplicate_jitter_sigma: Jitter magnitude as a fraction of the per-frame mean
                                inter-point KNN distance. 0.3 = tight, 0.7 = gap-filling.

    Returns:
        Path to the written PLY file (same as output_path).
    """
    if not frames_data:
        raise ValueError("frames_data is empty — nothing to build.")

    delta_t = 1.0 / max(fps, 1e-6)

    all_xyz:   List[np.ndarray] = []
    all_rgb:   List[np.ndarray] = []
    all_t:     List[np.ndarray] = []
    all_tscl:  List[np.ndarray] = []
    all_mot:   List[np.ndarray] = []

    n_before = 0
    n_after  = 0

    for master_t in sorted(frames_data.keys()):
        xyz = frames_data[master_t]["xyz"]
        rgb = frames_data[master_t]["rgb"]
        if len(xyz) == 0:
            continue

        vel = velocities.get(master_t, np.zeros((len(xyz), 3), dtype=np.float32))

        # --- point duplication with spatial jitter ---
        n_before += len(xyz)
        if n_duplicate > 1:
            xyz, rgb, vel = _duplicate_with_jitter(
                xyz, rgb, vel,
                n=n_duplicate,
                sigma_frac=duplicate_jitter_sigma,
            )
        n_after += len(xyz)

        t_seconds = (master_t - start_frame) * delta_t
        priors    = dynamic_temporal_priors(t_seconds, delta_t, vel)

        all_xyz.append(xyz)
        all_rgb.append(rgb)
        all_t.append(priors["t"])
        all_tscl.append(priors["t_scale_log"])
        all_mot.append(priors["motion"])

    if not all_xyz:
        raise ValueError("All frames have empty point clouds.")

    if n_duplicate > 1:
        print(
            f"[pcd_builder] Dynamic duplication ×{n_duplicate}: "
            f"{n_before} triangulated → {n_after} seed points "
            f"(σ={duplicate_jitter_sigma:.2f}×mean_knn_dist per frame)"
        )

    xyz_all = np.vstack(all_xyz)
    rgb_all = np.vstack(all_rgb)
    t_all   = np.concatenate(all_t)
    ts_all  = np.concatenate(all_tscl)
    mot_all = np.vstack(all_mot)

    write_init_ply(output_path, xyz_all, rgb_all, t_all, ts_all, mot_all)
    return output_path


# ---------------------------------------------------------------------------
# Merge static + dynamic PLYs
# ---------------------------------------------------------------------------

def merge_plys(
    static_ply_path: str,
    dynamic_ply_path: str,
    output_path: str,
) -> str:
    """
    Concatenate static and dynamic PLYs into the final sharptime_init.ply.

    Both PLYs must have the same vertex fields (written by write_init_ply).

    Returns:
        Path to the merged PLY.
    """
    static_data  = _read_ply_vertices(static_ply_path)
    dynamic_data = _read_ply_vertices(dynamic_ply_path)

    merged = np.concatenate([static_data, dynamic_data])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    vertex_el = PlyElement.describe(merged, "vertex")
    PlyData([vertex_el]).write(output_path)
    print(
        f"[pcd_builder] Merged PLY: {len(static_data)} static + "
        f"{len(dynamic_data)} dynamic = {len(merged)} points → {output_path}"
    )
    return output_path


# ---------------------------------------------------------------------------
# Read fps helper (DRY with other modules)
# ---------------------------------------------------------------------------

def read_fps_from_source(source_path: str) -> float:
    """Read fps from first .mp4 under {source_path}/videos/ or source_path."""
    import cv2, os
    source = Path(source_path)
    for root in [source / "videos", source]:
        if root.exists():
            for f in os.listdir(root):
                if f.lower().endswith(".mp4"):
                    cap = cv2.VideoCapture(str(root / f))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    return fps if fps > 0 else 30.0
    return 30.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_ply_vertices(path: str) -> np.ndarray:
    """Read vertex array from a PLY file written by write_init_ply."""
    ply = PlyData.read(path)
    return ply["vertex"].data


def _duplicate_with_jitter(
    xyz: np.ndarray,      # [N, 3] float32 — original positions
    rgb: np.ndarray,      # [N, 3] uint8   — colours
    vel: np.ndarray,      # [N, 3] float32 — velocities
    n: int,               # total copies per point (1 = no-op)
    sigma_frac: float,    # jitter std as fraction of per-cloud mean KNN distance
) -> tuple:
    """
    Replicate each dynamic point n times with independent Gaussian XYZ jitter.

    Velocity and colour are copied unchanged. The jitter standard deviation is
    sigma_frac × (mean nearest-neighbour distance in this frame's cloud), so it
    adapts to the local point density rather than using a fixed world-unit value.

    This increases the seed point count by n× at zero additional compute cost.
    Training-time pruning will remove duplicates that don't reduce reconstruction
    error; useful copies survive.

    Returns:
        (xyz_out, rgb_out, vel_out) — each with n×N rows.
    """
    if n <= 1 or len(xyz) == 0:
        return xyz, rgb, vel

    from .pcd_utils import knn_mean_dist

    # Compute characteristic inter-point distance for this frame's cloud.
    # knn_mean_dist returns per-point values; take the global mean as the scale.
    k = min(3, len(xyz) - 1)
    if k > 0:
        mean_dist = float(knn_mean_dist(xyz, k=k).mean())
    else:
        mean_dist = 1e-3   # degenerate fallback

    sigma = sigma_frac * mean_dist

    xyz_copies = [xyz]
    rgb_copies = [rgb]
    vel_copies = [vel]

    for _ in range(n - 1):
        jitter = np.random.normal(0.0, sigma, size=xyz.shape).astype(np.float32)
        xyz_copies.append(xyz + jitter)
        rgb_copies.append(rgb)
        vel_copies.append(vel)

    return (
        np.vstack(xyz_copies),
        np.vstack(rgb_copies),
        np.vstack(vel_copies),
    )
