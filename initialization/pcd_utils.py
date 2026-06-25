"""
Point cloud utilities shared across initialization branches.

Functions:
  knn_mean_dist   — mean distance to k nearest neighbours per point
  compute_scales  — isotropic log-scale initialisation (vanilla 3DGS convention)
  write_init_ply  — write a fetchPly-compatible PLY with temporal fields
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors


def knn_mean_dist(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Return the mean distance to the k nearest neighbours for every point.

    Args:
        xyz: [N, 3] float array.
        k:   Number of neighbours (excluding self).

    Returns:
        [N] float array of mean distances, clipped to a minimum of 1e-7.
    """
    if len(xyz) <= k:
        return np.full(len(xyz), 1e-4, dtype=np.float32)

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(xyz)
    dists, _ = nbrs.kneighbors(xyz)          # [N, k+1]; column 0 is self (dist=0)
    mean_dist = dists[:, 1:].mean(axis=1)    # [N]
    return np.clip(mean_dist, 1e-7, None).astype(np.float32)


def compute_scales(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Isotropic log-scale initialisation: log(sqrt(mean_knn_dist)).

    Returns [N, 3] where all three axes share the same value, matching the
    vanilla 3DGS convention used in create_from_pcd.
    """
    d = knn_mean_dist(xyz, k)
    s = np.log(np.sqrt(d)).astype(np.float32)[:, None]  # [N, 1]
    return np.repeat(s, 3, axis=1)                        # [N, 3]


def write_init_ply(
    output_path: str,
    xyz: np.ndarray,          # [N, 3] float32
    rgb: np.ndarray,          # [N, 3] uint8  (0-255)
    t: np.ndarray,            # [N] or [N,1] float32 — time anchor in seconds
    t_scale_log: np.ndarray,  # [N] or [N,1] float32 — log(σ_t) in seconds
    motion: np.ndarray,       # [N, 3] float32 — velocity in world units/s
) -> None:
    """
    Write a PLY that fetchPly() can read and pass to create_from_pcd().

    Fields written:
      x, y, z              — world coordinates
      nx, ny, nz           — surface normals (set to zero)
      red, green, blue     — per-point RGB (uint8)
      t                    — time anchor (seconds)
      t_scale              — log temporal scale (seconds)
      motion_0/1/2         — velocity components (world units / second)

    The 'r' (lifespan radius logit) field is intentionally omitted; it is
    always initialised to _R_INIT_LOGIT inside SharpTimeGaussianModel.create_from_pcd.
    """
    n = len(xyz)

    t_flat        = np.asarray(t,          dtype=np.float32).ravel()
    tscale_flat   = np.asarray(t_scale_log, dtype=np.float32).ravel()
    motion_arr    = np.asarray(motion,      dtype=np.float32).reshape(n, 3)
    rgb_arr       = np.asarray(rgb,         dtype=np.uint8).reshape(n, 3)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("t", "f4"),
        ("t_scale", "f4"),
        ("motion_0", "f4"), ("motion_1", "f4"), ("motion_2", "f4"),
    ]

    elements = np.empty(n, dtype=dtype)
    elements["x"]  = xyz[:, 0]
    elements["y"]  = xyz[:, 1]
    elements["z"]  = xyz[:, 2]
    elements["nx"] = 0.0
    elements["ny"] = 0.0
    elements["nz"] = 0.0
    elements["red"]   = rgb_arr[:, 0]
    elements["green"] = rgb_arr[:, 1]
    elements["blue"]  = rgb_arr[:, 2]
    elements["t"]        = t_flat
    elements["t_scale"]  = tscale_flat
    elements["motion_0"] = motion_arr[:, 0]
    elements["motion_1"] = motion_arr[:, 1]
    elements["motion_2"] = motion_arr[:, 2]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    vertex_el = PlyElement.describe(elements, "vertex")
    PlyData([vertex_el]).write(output_path)
    print(f"[pcd_utils] Wrote {n} points → {output_path}")


def static_temporal_priors(duration_seconds: float, n_points: int) -> dict:
    """
    Temporal parameters for static Gaussians (§3.3 of SharpTimeGS paper).

    The time anchor is placed at the midpoint of the training sequence and
    the temporal scale is set wide enough that the temporal weight stays ≈ 1
    across the full sequence.

    Returns a dict with:
      t           [N]    seconds — time anchor
      t_scale_log [N]    log(σ_t) — log temporal scale in seconds
      motion      [N, 3] zeros — no initial velocity
    """
    t_mid     = duration_seconds / 2.0
    # log(3 * duration) ensures e^{-0.5*(half_dur/σ)^2} ≈ 0.97 at sequence edges
    t_scale   = math.log(max(3.0 * duration_seconds, 1e-6))

    return {
        "t":           np.full(n_points, t_mid,    dtype=np.float32),
        "t_scale_log": np.full(n_points, t_scale,  dtype=np.float32),
        "motion":      np.zeros((n_points, 3),      dtype=np.float32),
    }


def dynamic_temporal_priors(
    t_seconds: float,
    delta_t: float,
    velocity: np.ndarray,  # [N, 3]
) -> dict:
    """
    Temporal parameters for dynamic Gaussians at a specific frame.

    Args:
        t_seconds: Timestamp of this frame in seconds (relative to training start).
        delta_t:   Duration between consecutive frames (1/fps).
        velocity:  [N, 3] velocity estimated via KNN cross-frame matching.

    Returns:
        t           [N]    seconds
        t_scale_log [N]    log(σ_t) covering ≈ 3 frame durations
        motion      [N, 3] velocity in world units / second
    """
    n = len(velocity)
    # σ_t covers ~3 frame durations so the Gaussian fades between frames
    t_scale = math.log(max(3.0 * delta_t, 1e-6))

    return {
        "t":           np.full(n, t_seconds, dtype=np.float32),
        "t_scale_log": np.full(n, t_scale,   dtype=np.float32),
        "motion":      velocity.astype(np.float32),
    }
