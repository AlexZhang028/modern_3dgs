"""
KNN-based cross-frame velocity estimation for dynamic 3D points.

For each adjacent master-frame pair (t, t+1):
  - Match dynamic points by nearest-neighbour in 3D world space.
  - Velocity = (X_{t+1} - X_t) / delta_t.
  - Outlier filtering by maximum displacement distance.

Output velocities are used as the `motion` field in the dynamic branch PLY,
which SharpTimeGaussianModel uses to initialise Gaussian velocity parameters.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def estimate_velocities(
    frames_data: Dict[int, Dict[str, np.ndarray]],
    fps: float,
    *,
    k: int = 5,
    max_displacement: Optional[float] = None,
    max_speed: Optional[float] = None,
) -> Dict[int, np.ndarray]:
    """
    Estimate per-point 3D velocities from adjacent-frame point cloud pairs.

    For each master frame t with a known t+1 cloud:
      - Build a KNN tree on the t+1 cloud.
      - For each point in the t cloud, find the k nearest neighbours in t+1.
      - Velocity = (mean(X_{t+1}[neighbours]) - X_t[point]) / delta_t.
      - Points without a suitable match get velocity = [0, 0, 0].

    Two-level outlier filtering is applied:
      1. **Displacement filter** (per frame-pair): zero out points whose kNN
         centroid displacement exceeds ``max_displacement`` (default: 3× mean
         nearest-neighbour distance).
      2. **Speed cap** (per frame-pair, adaptive IQR): clip velocity magnitude
         to Q75 + 3×IQR of all non-zero per-point speeds in the frame pair.
         Additionally clipped to ``max_speed`` when provided.  Direction is
         preserved; only the magnitude is capped.

    The IQR cap handles the case where KNN matching produces spuriously large
    displacements on non-rigid bodies (e.g., matching a limb at frame t to a
    different limb at t+1).

    Args:
        frames_data:       Dict master_t → {"xyz": [N,3], "rgb": [N,3]}.
        fps:               Frames per second (used to compute delta_t = 1/fps).
        k:                 Number of nearest neighbours for velocity averaging.
        max_displacement:  Maximum valid inter-frame displacement in world units.
                           Points whose kNN centroid is farther than this get
                           zero velocity. Defaults to 3× mean kNN distance.
        max_speed:         Hard upper bound on velocity magnitude (world units/s).
                           Applied after the IQR cap.  None = no hard cap.

    Returns:
        Dict mapping master_t → velocity array [N, 3] float32.
        Frames without a t+1 cloud receive velocities of zero.
    """
    if not frames_data:
        return {}

    delta_t  = 1.0 / max(fps, 1e-6)
    t_sorted = sorted(frames_data.keys())
    velocities: Dict[int, np.ndarray] = {}

    for i, master_t in enumerate(t_sorted):
        xyz_t = frames_data[master_t]["xyz"]  # [N, 3]
        N     = len(xyz_t)

        # Find the next available frame
        master_t1 = t_sorted[i + 1] if i + 1 < len(t_sorted) else None
        if master_t1 is None or master_t1 != master_t + 1:
            velocities[master_t] = np.zeros((N, 3), dtype=np.float32)
            continue

        xyz_t1 = frames_data[master_t1]["xyz"]  # [M, 3]
        if len(xyz_t1) == 0:
            velocities[master_t] = np.zeros((N, 3), dtype=np.float32)
            continue

        # ---- build KNN on t+1 cloud ----
        k_eff = min(k, len(xyz_t1))
        nn    = NearestNeighbors(n_neighbors=k_eff, algorithm="auto", n_jobs=-1)
        nn.fit(xyz_t1)
        dists, idxs = nn.kneighbors(xyz_t)   # [N, k_eff]

        xyz_t1_matched = xyz_t1[idxs].mean(axis=1)   # [N, 3] centroid in t+1
        displacement   = xyz_t1_matched - xyz_t        # [N, 3]
        dist_to_match  = np.linalg.norm(displacement, axis=1)  # [N]

        # ---- Level-1: displacement outlier filter ----
        if max_displacement is None:
            mean_knn = dists[:, 0].mean() if dists.shape[1] > 0 else 1.0
            thresh   = 3.0 * max(mean_knn, 1e-6)
        else:
            thresh = max_displacement

        vel = displacement / delta_t   # [N, 3]  world_units / second
        outlier = dist_to_match > thresh
        vel[outlier] = 0.0

        # ---- Level-2: IQR-based speed cap ----
        speeds = np.linalg.norm(vel, axis=1)          # [N]
        nz     = speeds[speeds > 0.0]
        if len(nz) >= 4:
            q25, q75 = np.percentile(nz, [25, 75])
            iqr      = q75 - q25
            iqr_cap  = q75 + 3.0 * iqr               # Tukey upper fence
            if max_speed is not None:
                iqr_cap = min(iqr_cap, max_speed)

            too_fast = speeds > iqr_cap
            if too_fast.any():
                # Preserve direction, clip magnitude
                vel[too_fast] = (
                    vel[too_fast] / speeds[too_fast, None] * iqr_cap
                )
        elif max_speed is not None:
            too_fast = speeds > max_speed
            if too_fast.any():
                vel[too_fast] = (
                    vel[too_fast] / speeds[too_fast, None] * max_speed
                )

        velocities[master_t] = vel.astype(np.float32)

    # Last frame gets zero velocity (no successor)
    if t_sorted:
        last = t_sorted[-1]
        if last not in velocities:
            N = len(frames_data[last]["xyz"])
            velocities[last] = np.zeros((N, 3), dtype=np.float32)

    return velocities
