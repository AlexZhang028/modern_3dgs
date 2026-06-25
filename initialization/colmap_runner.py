"""
Headless COLMAP subprocess wrappers.

All functions run colmap CLI commands as subprocesses and raise RuntimeError
on non-zero exit codes. No GUI is launched.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Reuse existing binary-format readers from the project
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.colmap_loader import read_points3D_binary, read_points3D_text


def find_colmap(colmap_bin: Optional[str] = None) -> str:
    """Return colmap binary path; raise FileNotFoundError if absent."""
    candidate = colmap_bin or shutil.which("colmap")
    if candidate:
        return candidate
    raise FileNotFoundError(
        "colmap binary not found in PATH. "
        "Install via: conda install -c conda-forge colmap"
    )


def _run(cmd: list, label: str) -> None:
    print(f"[COLMAP] {label}")
    print(f"  cmd: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"COLMAP step '{label}' exited with code {result.returncode}"
        )


def run_feature_extractor(
    db_path: str,
    image_path: str,
    *,
    gpu: bool = True,
    camera_model: str = "OPENCV",
    single_camera_per_folder: bool = False,
    camera_params: Optional[str] = None,
    mask_path: Optional[str] = None,
    colmap_bin: str = "colmap",
) -> None:
    """Extract SIFT features from all images in image_path into db_path."""
    cmd = [
        colmap_bin, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path",    str(image_path),
        "--FeatureExtraction.use_gpu", "1" if gpu else "0",
        "--ImageReader.camera_model", camera_model,
    ]
    if single_camera_per_folder:
        cmd += ["--ImageReader.single_camera_per_folder", "1"]
    if camera_params:
        cmd += ["--ImageReader.camera_params", camera_params]
    if mask_path:
        cmd += ["--ImageReader.mask_path", str(mask_path)]
    _run(cmd, "feature_extractor")


def run_exhaustive_matcher(
    db_path: str,
    *,
    gpu: bool = True,
    colmap_bin: str = "colmap",
) -> None:
    """Match features between all image pairs (suitable for small N_cam)."""
    cmd = [
        colmap_bin, "exhaustive_matcher",
        "--database_path", str(db_path),
        "--FeatureMatching.use_gpu", "1" if gpu else "0",
    ]
    _run(cmd, "exhaustive_matcher")


def run_mapper(
    db_path: str,
    image_path: str,
    output_path: str,
    *,
    colmap_bin: str = "colmap",
) -> Path:
    """Run incremental SfM mapper; returns path to the best sparse model dir."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap_bin, "mapper",
        "--database_path", str(db_path),
        "--image_path",    str(image_path),
        "--output_path",   str(output_path),
    ]
    _run(cmd, "mapper")
    return _best_model_dir(output_path)


def run_point_triangulator(
    db_path: str,
    image_path: str,
    input_path: str,
    output_path: str,
    *,
    min_tri_angle: float = 2.0,
    colmap_bin: str = "colmap",
) -> None:
    """Triangulate with known camera poses (used by the dynamic branch)."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    cmd = [
        colmap_bin, "point_triangulator",
        "--database_path", str(db_path),
        "--image_path",    str(image_path),
        "--input_path",    str(input_path),
        "--output_path",   str(output_path),
        "--Mapper.filter_min_tri_angle", str(min_tri_angle),
    ]
    _run(cmd, "point_triangulator")


def _best_model_dir(sparse_root: str) -> Path:
    """Return the sub-directory in sparse_root that has the largest points3D file."""
    root = Path(sparse_root)
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if not subdirs:
        raise RuntimeError(f"colmap mapper produced no model directories in {sparse_root}")

    def _size(d: Path) -> int:
        for name in ("points3D.bin", "points3D.txt"):
            f = d / name
            if f.exists():
                return f.stat().st_size
        return 0

    best = max(subdirs, key=_size)
    if _size(best) == 0:
        raise RuntimeError(
            f"colmap mapper succeeded but no points3D file found in {best}. "
            "This may mean insufficient overlap or all images failed to register."
        )
    return best


def read_points3D(sparse_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read COLMAP points3D from sparse_path.

    Returns:
        xyz: [N, 3] float32 — world coordinates
        rgb: [N, 3] uint8  — per-point RGB colors
    """
    d = Path(sparse_path)
    bin_path = d / "points3D.bin"
    txt_path = d / "points3D.txt"

    if bin_path.exists():
        xyz, rgb, _ = read_points3D_binary(str(bin_path))
    elif txt_path.exists():
        xyz, rgb, _ = read_points3D_text(str(txt_path))
    else:
        raise FileNotFoundError(f"No points3D.bin or points3D.txt in {sparse_path}")

    return xyz.astype(np.float32), rgb.astype(np.uint8)
