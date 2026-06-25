"""
Write COLMAP sparse-model binary files (cameras.bin, images.bin, points3D.bin)
from SelfCap GT intrinsics/extrinsics.

Used by static_branch to seed point_triangulator with known camera poses so
that the static point cloud shares the SelfCap world coordinate frame with the
dynamic branch.

Binary format reference: https://colmap.github.io/format.html
"""

import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# COLMAP model_id for the OPENCV 8-parameter model (fx,fy,cx,cy,k1,k2,p1,p2)
_OPENCV_MODEL_ID  = 4
_OPENCV_NUM_PARAMS = 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_known_pose_model(
    output_dir: str,
    cam_names: List[str],
    extrinsics: Dict[str, dict],
    intrinsics: Dict[str, dict],
    image_dir: str,
) -> Path:
    """
    Write cameras.bin, images.bin, and an empty points3D.bin for the
    point_triangulator command.

    Args:
        output_dir:  Directory where the three .bin files will be written.
                     Created if it does not exist.
        cam_names:   Ordered list of camera names (from read_selfcap_cameras).
        extrinsics:  cam_name → {"R": [3,3], "T": [3]} (world-to-camera).
        intrinsics:  cam_name → {"K": [3,3], "D": [≥4]}.
        image_dir:   Directory that contains {cam_name}.jpg reference images;
                     used to determine image width and height.

    Returns:
        Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- collect active cameras (those with both poses and intrinsics) ---
    active = [n for n in cam_names if n in extrinsics and n in intrinsics]
    if not active:
        raise ValueError("No cameras have both extrinsics and intrinsics.")

    # --- read image dimensions from the reference-frame jpegs ---
    dims: Dict[str, Tuple[int, int]] = {}  # cam_name → (width, height)
    for name in active:
        img_path = Path(image_dir) / f"{name}.jpg"
        if img_path.exists():
            h, w = cv2.imread(str(img_path)).shape[:2]
            dims[name] = (w, h)
        else:
            # Fallback: derive from K (cx*2 ≈ width)
            K = np.asarray(intrinsics[name]["K"])
            dims[name] = (int(K[0, 2] * 2), int(K[1, 2] * 2))

    _write_cameras(out / "cameras.bin", active, intrinsics, dims)
    _write_images(out / "images.bin",   active, extrinsics)
    _write_points3D_empty(out / "points3D.bin")

    return out


# ---------------------------------------------------------------------------
# Binary writers
# ---------------------------------------------------------------------------

def _write_cameras(
    path: Path,
    cam_names: List[str],
    intrinsics: Dict[str, dict],
    dims: Dict[str, Tuple[int, int]],
) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cam_names)))
        for cam_id, name in enumerate(cam_names, start=1):
            K  = np.asarray(intrinsics[name]["K"], dtype=np.float64)
            D  = np.asarray(intrinsics[name]["D"], dtype=np.float64).ravel()
            w, h = dims[name]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            # OPENCV model: [fx, fy, cx, cy, k1, k2, p1, p2]
            k1 = float(D[0]) if len(D) > 0 else 0.0
            k2 = float(D[1]) if len(D) > 1 else 0.0
            p1 = float(D[2]) if len(D) > 2 else 0.0
            p2 = float(D[3]) if len(D) > 3 else 0.0
            params = [fx, fy, cx, cy, k1, k2, p1, p2]

            f.write(struct.pack("<i",   cam_id))
            f.write(struct.pack("<i",   _OPENCV_MODEL_ID))
            f.write(struct.pack("<Q",   w))
            f.write(struct.pack("<Q",   h))
            f.write(struct.pack(f"<{_OPENCV_NUM_PARAMS}d", *params))


def _write_images(
    path: Path,
    cam_names: List[str],
    extrinsics: Dict[str, dict],
) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cam_names)))
        for img_id, name in enumerate(cam_names, start=1):
            cam_id = img_id   # one-to-one: camera i corresponds to image i
            R = np.asarray(extrinsics[name]["R"], dtype=np.float64)
            T = np.asarray(extrinsics[name]["T"], dtype=np.float64).ravel()
            qvec = _rotmat_to_qvec(R)   # [w, x, y, z]

            f.write(struct.pack("<i",     img_id))
            f.write(struct.pack("<4d",    *qvec))
            f.write(struct.pack("<3d",    *T))
            f.write(struct.pack("<i",     cam_id))
            f.write(name.encode("utf-8") + b".jpg\x00")  # relative image name
            f.write(struct.pack("<Q",     0))             # no 2D points yet


def _write_points3D_empty(path: Path) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 0))


# ---------------------------------------------------------------------------
# Rotation matrix → COLMAP quaternion
# ---------------------------------------------------------------------------

def _rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3×3 rotation matrix to a unit quaternion [w, x, y, z]
    in COLMAP's convention (real part first).

    Uses the numerically stable Shepperd method.
    """
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)
