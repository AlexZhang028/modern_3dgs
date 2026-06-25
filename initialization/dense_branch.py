"""
Dense dynamic region reconstruction via per-frame multi-view stereo (MVS).

For each master frame in the training range:
  1. Load all camera images + SAM2 masks.
  2. For each camera pair with sufficient stereo baseline, stereo-rectify the
     pair and run StereoSGBM to get a dense disparity map.
  3. Apply the mask, reproject valid disparities to 3D, transform to world
     coordinates using the known GT camera poses.
  4. Fuse all camera-pair clouds and voxel-grid-downsample.

Return type is identical to dynamic_branch.run_dynamic_branch so it can be
used as a drop-in supplement or replacement.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.selfcap_loader import read_selfcap_cameras, read_selfcap_sync


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_dense_dynamic_branch(
    source_path: str,
    work_dir: str,
    start_frame: int = 0,
    end_frame: int = -1,
    *,
    block_size: int = 11,
    uniqueness_ratio: int = 10,
    speckle_window_size: int = 100,
    speckle_range: int = 2,
    min_depth: float = 0.3,
    max_depth: float = 10.0,
    min_baseline: float = 0.1,
    max_baseline: float = 3.0,
    voxel_size: float = 0.02,
    max_image_size: int = 960,
    overwrite: bool = False,
    masks_dir: Optional[str] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Dense triangulation of the SAM2-masked dynamic region via StereoSGBM.

    Unlike the sparse SIFT branch, this approach processes every pixel in
    the dynamic mask rather than only detected keypoints.  Regions with
    subtle texture (clothing folds, skin shading gradients) that produce no
    SIFT keypoints are handled naturally.

    Args:
        source_path:          Dataset root (extri.yml / intri.yml / frames/).
        work_dir:             Scratch dir; per-frame clouds cached under
                              {work_dir}/dense_dynamic/{t:05d}/points3D.npz.
        start_frame/end_frame: Master frame range (-1 = infer from disk).
        block_size:           SGBM block size (odd integer, 5–21).
                              Smaller = more detail / more noise.
        uniqueness_ratio:     Reject matches where the runner-up is within this
                              percentage of the winner (0 = off, 5–15 typical).
        speckle_window_size:  Remove isolated disparity blobs smaller than
                              this many pixels (50–200).
        speckle_range:        Max allowed disparity variation within a speckle.
        min_depth/max_depth:  Depth range to keep in camera-space metres.
        min_baseline:         Skip camera pairs whose optical-centre distance
                              is below this threshold (metres).
        max_baseline:         Skip camera pairs whose baseline exceeds this.
                              Large-baseline pairs (e.g. opposite-side cameras)
                              produce disparities >> image width, causing SGBM
                              buffer overflow.  Keep below ~3 m for typical scenes.
        voxel_size:           Grid cell size for output downsampling (metres).
                              0 = no downsampling.
        max_image_size:       Resize the longer image edge to this before
                              running SGBM.  K is scaled consistently so 3-D
                              output is in the original metric coordinate system.
                              Use 640–1280 depending on available RAM.
        overwrite:            Recompute even if per-frame cache files exist.
        masks_dir:            Directory with per-camera SAM2 masks.

    Returns:
        Dict master_t → {"xyz": [N,3] float32, "rgb": [N,3] uint8}.
    """
    source      = Path(source_path)
    frames_root = source / "frames"
    masks_root  = Path(masks_dir) if masks_dir else Path(work_dir) / "masks"
    dense_root  = Path(work_dir) / "dense_dynamic"

    cam_names, extrinsics, intrinsics = read_selfcap_cameras(str(source))
    sync_map = read_selfcap_sync(str(source))
    fps      = _read_fps(source)

    master_frames = _resolve_frame_range(
        frames_root, cam_names, sync_map, fps, start_frame, end_frame
    )
    if not master_frames:
        print("[dense] No frames found in the requested range.")
        return {}

    img_size = _read_image_size(frames_root, cam_names, sync_map, fps, master_frames)
    if img_size is None:
        print("[dense] Could not determine image size from any frame.")
        return {}
    H, W = img_size

    pair_params = _build_all_stereo_params(
        cam_names, extrinsics, intrinsics, H, W,
        min_baseline, block_size, uniqueness_ratio,
        speckle_window_size, speckle_range, min_depth, max_depth,
        max_image_size=max_image_size,
        max_baseline=max_baseline,
    )
    if not pair_params:
        print(f"[dense] No valid camera pairs with baseline ≥ {min_baseline} m.")
        return {}

    baselines = [sp["baseline"] for sp in pair_params.values()]
    first_sp = next(iter(pair_params.values()))
    W_s, H_s = first_sp["W_s"], first_sp["H_s"]
    scale_pct = int(round(first_sp["scale"] * 100))
    print(
        f"[dense] {len(pair_params)} stereo pairs, "
        f"baseline {min(baselines):.2f}–{max(baselines):.2f} m, "
        f"image {W}×{H} → {W_s}×{H_s} ({scale_pct}% for SGBM)"
    )

    result: Dict[int, Dict[str, np.ndarray]] = {}

    for master_t in tqdm(master_frames, desc="[dense] frames", unit="frame"):
        out_npz = dense_root / f"{master_t:05d}" / "points3D.npz"
        if out_npz.exists() and not overwrite:
            data = np.load(str(out_npz))
            result[master_t] = {"xyz": data["xyz"], "rgb": data["rgb"]}
            continue

        # Load all camera images and masks for this frame
        imgs:  Dict[str, np.ndarray]           = {}
        masks: Dict[str, Optional[np.ndarray]] = {}
        for cam in cam_names:
            sync_offset = sync_map.get(cam, 0.0)
            eff_t = int(round(master_t + sync_offset * fps))
            img_path  = frames_root / cam / f"{eff_t:05d}.jpg"
            mask_path = masks_root  / cam / f"{eff_t:05d}.png"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            imgs[cam]  = img
            masks[cam] = (
                cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_path.exists() else None
            )

        all_xyz: List[np.ndarray] = []
        all_rgb: List[np.ndarray] = []

        for (c1, c2), sp in pair_params.items():
            if c1 not in imgs or c2 not in imgs:
                continue
            mask1 = masks.get(c1)
            if mask1 is not None and int(mask1.max()) == 0:
                continue  # No dynamic pixels in reference camera

            xyz, rgb = _stereo_to_world(
                imgs[c1], imgs[c2], mask1, sp, min_depth, max_depth
            )
            if xyz is not None and len(xyz) > 0:
                all_xyz.append(xyz)
                all_rgb.append(rgb)

        if not all_xyz:
            continue

        xyz_all = np.vstack(all_xyz).astype(np.float32)
        rgb_all = np.vstack(all_rgb).astype(np.uint8)

        if voxel_size > 0:
            xyz_all, rgb_all = _voxel_downsample(xyz_all, rgb_all, voxel_size)

        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out_npz), xyz=xyz_all, rgb=rgb_all)
        result[master_t] = {"xyz": xyz_all, "rgb": rgb_all}

    n_pts = sum(d["xyz"].shape[0] for d in result.values())
    print(f"[dense] Done: {len(result)} frames, {n_pts} dense dynamic points.")
    return result


def merge_frame_data(
    sparse: Dict[int, Dict[str, np.ndarray]],
    dense:  Dict[int, Dict[str, np.ndarray]],
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Concatenate sparse (SIFT) and dense (MVS) per-frame point clouds.
    Frames present in only one source are kept as-is.
    """
    all_keys = set(sparse) | set(dense)
    merged: Dict[int, Dict[str, np.ndarray]] = {}
    for t in sorted(all_keys):
        if t in sparse and t in dense:
            merged[t] = {
                "xyz": np.vstack([sparse[t]["xyz"], dense[t]["xyz"]]),
                "rgb": np.vstack([sparse[t]["rgb"], dense[t]["rgb"]]),
            }
        elif t in sparse:
            merged[t] = sparse[t]
        else:
            merged[t] = dense[t]
    return merged


# ---------------------------------------------------------------------------
# Stereo parameter precomputation (done once per camera pair, before frame loop)
# ---------------------------------------------------------------------------

def _build_all_stereo_params(
    cam_names, extrinsics, intrinsics,
    H: int, W: int,
    min_baseline: float,
    block_size: int, uniqueness_ratio: int,
    speckle_window_size: int, speckle_range: int,
    min_depth: float, max_depth: float,
    max_image_size: int = 960,
    max_baseline: float = 3.0,
) -> Dict[Tuple[str, str], dict]:
    """Build rectification maps + SGBM instances for all qualifying pairs."""
    scale = min(1.0, max_image_size / max(H, W))
    pair_params: Dict[Tuple[str, str], dict] = {}

    for i, c1 in enumerate(cam_names):
        for c2 in cam_names[i + 1:]:
            if c1 not in extrinsics or c2 not in extrinsics:
                continue
            if c1 not in intrinsics or c2 not in intrinsics:
                continue

            R1w = np.asarray(extrinsics[c1]["R"], dtype=np.float64)
            T1w = np.asarray(extrinsics[c1]["T"], dtype=np.float64).flatten()
            R2w = np.asarray(extrinsics[c2]["R"], dtype=np.float64)
            T2w = np.asarray(extrinsics[c2]["T"], dtype=np.float64).flatten()

            # Camera centres in world space: C = -R.T @ T
            c1_pos = -(R1w.T @ T1w)
            c2_pos = -(R2w.T @ T2w)
            baseline = float(np.linalg.norm(c2_pos - c1_pos))
            if baseline < min_baseline or baseline > max_baseline:
                continue

            K1 = np.asarray(intrinsics[c1]["K"], dtype=np.float64)
            D1 = np.asarray(intrinsics[c1]["D"], dtype=np.float64).flatten()
            K2 = np.asarray(intrinsics[c2]["K"], dtype=np.float64)
            D2 = np.asarray(intrinsics[c2]["D"], dtype=np.float64).flatten()

            # Relative pose from cam1 to cam2:
            #   X_c2 = R_rel @ X_c1 + T_rel
            R_rel = R2w @ R1w.T
            T_rel = T2w - R_rel @ T1w

            sp = _precompute_stereo_pair(
                K1, D1, K2, D2, R_rel, T_rel, R1w, T1w,
                H, W, scale, block_size, uniqueness_ratio,
                speckle_window_size, speckle_range,
                min_depth, max_depth, baseline, max_image_size,
            )
            if sp is not None:
                pair_params[(c1, c2)] = sp

    return pair_params


def _precompute_stereo_pair(
    K1, D1, K2, D2,
    R_rel, T_rel, R1w, T1w,
    H: int, W: int, scale: float,
    block_size: int, uniqueness_ratio: int,
    speckle_window_size: int, speckle_range: int,
    min_depth: float, max_depth: float,
    baseline: float, max_image_size: int = 960,
) -> Optional[dict]:
    """
    Compute stereo rectification maps and an SGBM instance for one camera pair.

    Images are downscaled by `scale` before SGBM to avoid OOM on high-res inputs.
    K matrices are scaled consistently so 3-D output remains in world-metric units.

    Coordinate maths:
        stereoRectify returns Rr1 such that X_rect = Rr1 @ X_cam1.
        To recover world coordinates from X_rect:
            X_cam1  = Rr1.T @ X_rect
            X_world = R1w.T @ (X_cam1 - T1w)
                    = R1w.T @ Rr1.T @ X_rect  -  R1w.T @ T1w
        So  R_rect1_to_world = R1w.T @ Rr1.T
            T_rect1_to_world = -R1w.T @ T1w   (= camera centre in world space)
    """
    H_s = int(round(H * scale))
    W_s = int(round(W * scale))

    # Scale camera matrices to match the downsampled image resolution
    K1_s = K1.copy(); K1_s[:2, :] *= scale
    K2_s = K2.copy(); K2_s[:2, :] *= scale

    try:
        Rr1, Rr2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1_s, D1, K2_s, D2, (W_s, H_s),
            R_rel, T_rel.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,  # crop to valid-pixel region only
        )
    except cv2.error:
        return None

    # Baseline in rectified space (used to size the disparity search range)
    fx = float(P1[0, 0])
    baseline_rect = abs(float(P2[0, 3])) / (fx + 1e-9)
    if baseline_rect < 1e-6:
        return None  # degenerate: cameras essentially co-located or facing each other

    # Disparity range: d = fx * b / Z  →  d_min/max from Z_max/min
    d_max = max(16, int(fx * baseline_rect / max(min_depth, 1e-3)) + 16)
    d_min = max(0, int(fx * baseline_rect / max_depth) - 8)

    # Reject pairs where the disparity range would exceed the scaled image width.
    # Pairs with large baselines (e.g. opposite-side cameras) produce disparities
    # far larger than the image width; SGBM buffer arithmetic overflows on them.
    if d_min >= W_s:
        return None

    n_disp = max(16, ((d_max - d_min + 15) // 16) * 16)
    n_disp = min(n_disp, 256)  # keep compute tractable; 256 sufficient for 0.15-3 m baseline

    map1x, map1y = cv2.initUndistortRectifyMap(K1_s, D1, Rr1, P1, (W_s, H_s), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2_s, D2, Rr2, P2, (W_s, H_s), cv2.CV_32FC1)

    bs = block_size
    stereo = cv2.StereoSGBM_create(
        minDisparity      = d_min,
        numDisparities    = n_disp,
        blockSize         = bs,
        P1                = 8  * 3 * bs * bs,
        P2                = 32 * 3 * bs * bs,
        disp12MaxDiff     = 1,
        uniquenessRatio   = uniqueness_ratio,
        speckleWindowSize = speckle_window_size,
        speckleRange      = speckle_range,
        mode              = cv2.STEREO_SGBM_MODE_SGBM,
    )

    R_rect1_to_world = R1w.T @ Rr1.T          # [3, 3]
    T_rect1_to_world = -(R1w.T @ T1w)         # [3] — camera 1 centre in world

    return {
        "map1x": map1x, "map1y": map1y,
        "map2x": map2x, "map2y": map2y,
        "Q": Q,
        "stereo": stereo,
        "Rr1": Rr1,
        "R_rect1_to_world": R_rect1_to_world,
        "T_rect1_to_world": T_rect1_to_world,
        "d_min": d_min,
        "baseline": baseline,
        "H_s": H_s,
        "W_s": W_s,
        "scale": scale,
    }


# ---------------------------------------------------------------------------
# Per-frame, per-pair stereo processing
# ---------------------------------------------------------------------------

def _stereo_to_world(
    img1: np.ndarray,              # [H, W, 3] BGR — reference camera
    img2: np.ndarray,              # [H, W, 3] BGR — secondary camera
    mask1: Optional[np.ndarray],   # [H, W] uint8 (255 = dynamic), or None
    sp: dict,
    min_depth: float,
    max_depth: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Dense stereo on one camera pair → [M, 3] world XYZ + [M, 3] RGB.

    Steps:
      1. Rectify both images (and mask) using precomputed maps.
      2. Run StereoSGBM on greyscale rectified images.
      3. Apply mask + disparity validity.
      4. Reproject to 3D and depth-filter in cam1-space.
      5. Transform to world coordinates.
    """
    # --- downscale then rectify ---
    # Rectification maps are built for the scaled image size; resize source
    # images to match before applying remap.
    W_s, H_s = sp["W_s"], sp["H_s"]
    if sp["scale"] < 1.0:
        img1 = cv2.resize(img1, (W_s, H_s), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (W_s, H_s), interpolation=cv2.INTER_LINEAR)
        if mask1 is not None:
            mask1 = cv2.resize(mask1, (W_s, H_s), interpolation=cv2.INTER_NEAREST)

    rect1 = cv2.remap(img1, sp["map1x"], sp["map1y"], cv2.INTER_LINEAR)
    rect2 = cv2.remap(img2, sp["map2x"], sp["map2y"], cv2.INTER_LINEAR)

    if mask1 is not None:
        rect_mask = cv2.remap(mask1, sp["map1x"], sp["map1y"], cv2.INTER_NEAREST)
    else:
        rect_mask = None

    # --- dense disparity ---
    gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY)
    try:
        disp = sp["stereo"].compute(gray1, gray2).astype(np.float32) / 16.0
    except cv2.error:
        return None, None

    # --- validity mask: positive disparity ∩ dynamic region ---
    valid = disp > sp["d_min"]
    if rect_mask is not None:
        valid &= (rect_mask > 0)

    if not valid.any():
        return None, None

    # --- reproject to rectified cam1 3D space ---
    pts_rect = cv2.reprojectImageTo3D(disp, sp["Q"])   # [H, W, 3]
    pts_valid = pts_rect[valid]                         # [M, 3]

    # --- depth filter in original (non-rectified) cam1 space ---
    # X_cam1 = Rr1.T @ X_rect
    pts_cam1 = (sp["Rr1"].T @ pts_valid.T).T           # [M, 3]
    depth_ok = (pts_cam1[:, 2] > min_depth) & (pts_cam1[:, 2] < max_depth)

    if not depth_ok.any():
        return None, None

    pts_valid = pts_valid[depth_ok]

    # --- transform to world space ---
    # X_world = R_rect1_to_world @ X_rect + T_rect1_to_world
    R = sp["R_rect1_to_world"]
    T = sp["T_rect1_to_world"]
    xyz_world = (R @ pts_valid.T).T + T               # [K, 3]

    # --- RGB from rectified cam1 (BGR → RGB) ---
    rgb_rect = rect1[:, :, ::-1]
    rgb = rgb_rect[valid][depth_ok]                    # [K, 3]

    return xyz_world.astype(np.float32), rgb.astype(np.uint8)


# ---------------------------------------------------------------------------
# Voxel-grid downsampling
# ---------------------------------------------------------------------------

def _voxel_downsample(
    xyz: np.ndarray,
    rgb: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep the first point falling into each voxel cell.
    O(N log N) via np.unique on an integer key.
    """
    if len(xyz) == 0:
        return xyz, rgb

    voxels = np.floor(xyz / voxel_size).astype(np.int64)
    lo = voxels.min(axis=0)
    voxels -= lo

    hi = voxels.max(axis=0) + 1                         # [3] grid extents
    stride_y = int(hi[0])
    stride_z = int(hi[0]) * int(hi[1])
    keys = (voxels[:, 0]
            + voxels[:, 1] * stride_y
            + voxels[:, 2] * stride_z)

    _, first_idx = np.unique(keys, return_index=True)
    return xyz[first_idx], rgb[first_idx]


# ---------------------------------------------------------------------------
# Shared helpers (mirror dynamic_branch.py to stay self-contained)
# ---------------------------------------------------------------------------

def _read_fps(source: Path) -> float:
    for root in [source / "videos", source]:
        if root.exists():
            for f in os.listdir(root):
                if f.lower().endswith(".mp4"):
                    cap = cv2.VideoCapture(str(root / f))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    return fps if fps > 0 else 30.0
    return 30.0


def _read_image_size(
    frames_root: Path,
    cam_names: List[str],
    sync_map: dict,
    fps: float,
    master_frames: List[int],
) -> Optional[Tuple[int, int]]:
    """Read (H, W) from the first readable frame across all cameras."""
    for cam in cam_names:
        sync_offset = sync_map.get(cam, 0.0)
        for master_t in master_frames[:10]:
            eff_t = int(round(master_t + sync_offset * fps))
            p = frames_root / cam / f"{eff_t:05d}.jpg"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    return img.shape[:2]
    return None


def _resolve_frame_range(
    frames_root: Path,
    cam_names: List[str],
    sync_map: dict,
    fps: float,
    start_frame: int,
    end_frame: int,
) -> List[int]:
    if end_frame <= 0:
        max_eff = 0
        for cam in cam_names:
            d = frames_root / cam
            if not d.exists():
                continue
            for f in d.iterdir():
                if f.suffix == ".jpg":
                    try:
                        max_eff = max(max_eff, int(f.stem))
                    except ValueError:
                        pass
        if max_eff == 0:
            return []
        sync_offset = sync_map.get(cam_names[0], 0.0) if cam_names else 0.0
        end_frame = int(round(max_eff - sync_offset * fps))
    return list(range(start_frame, end_frame + 1))
