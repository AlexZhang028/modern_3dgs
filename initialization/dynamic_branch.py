"""
Per-frame dynamic 3D point triangulation using known camera poses.

For each master frame in the training range:
  1. Loads per-camera frame images and SAM2 dynamic masks.
  2. Extracts SIFT features inside the masked (dynamic) region.
  3. Matches features across all camera pairs (exhaustive).
  4. Triangulates matched points with known intrinsics/extrinsics from extri.yml/intri.yml.
  5. Filters by reprojection error and positive depth.
  6. Saves per-frame clouds to {work_dir}/dynamic/{t:05d}/points3D.npz.

Camera poses come directly from the SelfCap calibration files — no extra COLMAP
subprocess is needed for this step.
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

def run_dynamic_branch(
    source_path: str,
    work_dir: str,
    start_frame: int = 0,
    end_frame: int = -1,
    *,
    n_features: int = 2000,
    match_ratio: float = 0.75,
    reproj_threshold: float = 4.0,
    min_matches: int = 8,
    overwrite: bool = False,
    masks_dir: Optional[str] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Triangulate dynamic 3D points for every master frame.

    Args:
        source_path:       Dataset root with extri.yml/intri.yml and frames/.
        work_dir:          Root for intermediate artefacts; output goes to
                           {work_dir}/dynamic/{master_t:05d}/points3D.npz.
        start_frame:       First master frame of the training range.
        end_frame:         Last master frame (-1 = inferred from frames on disk).
        n_features:        Maximum SIFT keypoints per image per camera.
        match_ratio:       Lowe's ratio-test threshold for SIFT matching.
        reproj_threshold:  Maximum reprojection error (pixels) to keep a point.
        min_matches:       Minimum number of good matches per camera pair.
        overwrite:         Recompute even if output files exist.
        masks_dir:         Directory containing SAM2 masks per camera.  Defaults
                           to {work_dir}/masks if not provided.

    Returns:
        Dict mapping master_t → {"xyz": [N,3] float32, "rgb": [N,3] uint8}.
        Returns an empty dict if no cameras have masks.
    """
    source      = Path(source_path)
    frames_root = source / "frames"
    masks_root  = Path(masks_dir) if masks_dir else Path(work_dir) / "masks"
    dyn_root    = Path(work_dir) / "dynamic"

    cam_names, extrinsics, intrinsics = read_selfcap_cameras(str(source))
    sync_map = read_selfcap_sync(str(source))
    fps      = _read_fps(source)

    # ---- resolve master-frame range ----
    master_frames = _resolve_frame_range(frames_root, cam_names, sync_map, fps, start_frame, end_frame)
    if not master_frames:
        print("[dyn] No frames found in the requested range.")
        return {}

    # ---- precompute projection matrices ----
    proj_mats = _build_proj_mats(cam_names, extrinsics, intrinsics)
    active_cams = [n for n in cam_names if n in proj_mats]
    if len(active_cams) < 2:
        print("[dyn] Need at least 2 cameras with known poses and intrinsics.")
        return {}

    # ---- build SIFT detector + matcher ----
    sift    = cv2.SIFT_create(nfeatures=n_features)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    result: Dict[int, Dict[str, np.ndarray]] = {}

    for master_t in tqdm(master_frames, desc="[dyn] frames", unit="frame"):
        out_npz = dyn_root / f"{master_t:05d}" / "points3D.npz"
        if out_npz.exists() and not overwrite:
            data = np.load(str(out_npz))
            result[master_t] = {"xyz": data["xyz"], "rgb": data["rgb"]}
            continue

        frame_data = _load_frame_data(
            active_cams, frames_root, masks_root, sync_map, fps, master_t
        )
        if not frame_data:
            continue

        # Extract keypoints per camera
        kp_data: Dict[str, dict] = {}
        for cam in active_cams:
            if cam not in frame_data:
                continue
            kps, descs, img = _extract_keypoints(sift, frame_data[cam])
            if descs is not None and len(descs) >= 2:
                kp_data[cam] = {"kps": kps, "descs": descs, "img": img}

        active = list(kp_data.keys())
        if len(active) < 2:
            continue

        all_xyz: List[np.ndarray] = []
        all_rgb: List[np.ndarray] = []

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                cam1, cam2 = active[i], active[j]
                xyz, rgb = _triangulate_pair(
                    kp_data[cam1], kp_data[cam2],
                    proj_mats[cam1], proj_mats[cam2],
                    extrinsics[cam1], extrinsics[cam2],
                    intrinsics[cam1], intrinsics[cam2],
                    matcher,
                    match_ratio=match_ratio,
                    reproj_threshold=reproj_threshold,
                    min_matches=min_matches,
                )
                if xyz is not None and len(xyz):
                    all_xyz.append(xyz)
                    all_rgb.append(rgb)

        if not all_xyz:
            continue

        xyz_all = np.vstack(all_xyz).astype(np.float32)
        rgb_all = np.vstack(all_rgb).astype(np.uint8)

        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out_npz), xyz=xyz_all, rgb=rgb_all)
        result[master_t] = {"xyz": xyz_all, "rgb": rgb_all}

    n_pts_total = sum(d["xyz"].shape[0] for d in result.values())
    print(f"[dyn] Done: {len(result)} frames, {n_pts_total} total dynamic points.")
    return result


def load_dynamic_frame(work_dir: str, master_t: int) -> Optional[Dict[str, np.ndarray]]:
    """Load saved dynamic point cloud for a single master frame."""
    path = Path(work_dir) / "dynamic" / f"{master_t:05d}" / "points3D.npz"
    if not path.exists():
        return None
    data = np.load(str(path))
    return {"xyz": data["xyz"], "rgb": data["rgb"]}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_proj_mats(
    cam_names: List[str],
    extrinsics: dict,
    intrinsics: dict,
) -> Dict[str, dict]:
    """Build per-camera data dict with projection matrix and params."""
    out = {}
    for name in cam_names:
        if name not in extrinsics or name not in intrinsics:
            continue
        R = np.asarray(extrinsics[name]["R"], dtype=np.float64)
        T = np.asarray(extrinsics[name]["T"], dtype=np.float64).reshape(3)
        K = np.asarray(intrinsics[name]["K"], dtype=np.float64)
        D = np.asarray(intrinsics[name]["D"], dtype=np.float64).flatten()
        # Normalised projection matrix (no K — used after undistortPoints)
        P_norm = np.hstack([R, T.reshape(3, 1)])  # [3, 4]
        out[name] = {"R": R, "T": T, "K": K, "D": D, "P_norm": P_norm}
    return out


def _load_frame_data(
    cam_names: List[str],
    frames_root: Path,
    masks_root: Path,
    sync_map: dict,
    fps: float,
    master_t: int,
) -> Dict[str, dict]:
    """Load image + mask for each camera at master frame t."""
    out = {}
    for cam in cam_names:
        sync_offset = sync_map.get(cam, 0.0)
        eff_t = int(round(master_t + sync_offset * fps))

        img_path  = frames_root / cam / f"{eff_t:05d}.jpg"
        mask_path = masks_root / cam / f"{eff_t:05d}.png"

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask = None  # use full image

        out[cam] = {"img": img, "mask": mask}
    return out


def _extract_keypoints(
    sift: cv2.SIFT,
    frame: dict,
) -> Tuple[list, Optional[np.ndarray], np.ndarray]:
    """Detect SIFT keypoints inside the dynamic mask (or whole image if no mask)."""
    img  = frame["img"]
    mask = frame["mask"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kps, descs = sift.detectAndCompute(gray, mask)
    return kps, descs, img


def _triangulate_pair(
    d1: dict,
    d2: dict,
    cam1_params: dict,
    cam2_params: dict,
    ext1: dict,
    ext2: dict,
    int1: dict,
    int2: dict,
    matcher: cv2.BFMatcher,
    *,
    match_ratio: float,
    reproj_threshold: float,
    min_matches: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Match and triangulate one camera pair.

    Returns:
        xyz: [M, 3] float32 in world coordinates, or None.
        rgb: [M, 3] uint8 from camera 1, or None.
    """
    kps1, descs1, img1 = d1["kps"], d1["descs"], d1["img"]
    kps2, descs2, _    = d2["kps"], d2["descs"], d2["img"]

    # ---- ratio-test matching ----
    raw = matcher.knnMatch(descs1, descs2, k=2)
    good = [m for m, n in raw if len(raw[0]) == 2 and m.distance < match_ratio * n.distance]
    if len(good) < min_matches:
        return None, None

    pts1 = np.float32([kps1[m.queryIdx].pt for m in good])  # [M, 2]
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good])

    K1 = cam1_params["K"]
    D1 = cam1_params["D"]
    K2 = cam2_params["K"]
    D2 = cam2_params["D"]
    R1 = cam1_params["R"]
    T1 = cam1_params["T"]
    R2 = cam2_params["R"]
    T2 = cam2_params["T"]

    # ---- undistort to normalised camera coordinates ----
    n1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, D1)  # [M, 1, 2]
    n2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K2, D2)

    P1 = np.hstack([R1, T1.reshape(3, 1)])  # [3, 4] — no K (points already normalised)
    P2 = np.hstack([R2, T2.reshape(3, 1)])

    pts4D = cv2.triangulatePoints(P1, P2, n1.squeeze(1).T, n2.squeeze(1).T)  # [4, M]
    w = pts4D[3]
    valid_w = np.abs(w) > 1e-8
    xyz = (pts4D[:3, valid_w] / w[valid_w]).T.astype(np.float32)  # [M', 3]
    pts1_filt = pts1[valid_w]

    if len(xyz) == 0:
        return None, None

    # ---- filter: must have positive depth in both cameras ----
    xyz_c1 = (R1 @ xyz.T + T1.reshape(3, 1)).T
    xyz_c2 = (R2 @ xyz.T + T2.reshape(3, 1)).T
    depth_ok = (xyz_c1[:, 2] > 0) & (xyz_c2[:, 2] > 0)
    xyz = xyz[depth_ok]
    pts1_filt = pts1_filt[depth_ok]

    if len(xyz) == 0:
        return None, None

    # ---- filter: reprojection error ----
    pts1_proj, _ = cv2.projectPoints(xyz, R1, T1.reshape(3, 1), K1, D1)
    pts2_proj, _ = cv2.projectPoints(xyz, R2, T2.reshape(3, 1), K2, D2)
    pts2_orig = pts2[valid_w][depth_ok]

    err1 = np.linalg.norm(pts1_filt - pts1_proj.squeeze(1), axis=1)
    err2 = np.linalg.norm(pts2_orig - pts2_proj.squeeze(1), axis=1)
    reproj_ok = (err1 < reproj_threshold) & (err2 < reproj_threshold)

    xyz = xyz[reproj_ok]
    pts1_final = pts1_filt[reproj_ok].astype(int)

    if len(xyz) == 0:
        return None, None

    # ---- get RGB from camera 1 ----
    h, w = img1.shape[:2]
    pts1_final[:, 0] = pts1_final[:, 0].clip(0, w - 1)
    pts1_final[:, 1] = pts1_final[:, 1].clip(0, h - 1)
    bgr = img1[pts1_final[:, 1], pts1_final[:, 0]]
    rgb = bgr[:, ::-1].astype(np.uint8)  # BGR → RGB

    return xyz, rgb


def _resolve_frame_range(
    frames_root: Path,
    cam_names: List[str],
    sync_map: dict,
    fps: float,
    start_frame: int,
    end_frame: int,
) -> List[int]:
    """Return sorted list of master frames in [start_frame, end_frame]."""
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
