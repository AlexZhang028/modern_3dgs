"""
Static branch of the SharpTimeGS initialization pipeline.

Pipeline:
  1. Prepare COLMAP workspace — copy reference-frame images from the dataset.
  2. Write known-pose sparse model — cameras.bin / images.bin from GT extri/intri.yml.
  3. Run headless COLMAP (feature_extractor → exhaustive_matcher → point_triangulator).
  4. Read triangulated 3D points.
  5. Assign static temporal priors (v=0, T=mid, σ_t=log(3·duration)).
  6. Write a fetchPly-compatible init PLY.

Using GT poses (from extri.yml/intri.yml) instead of running the incremental
SfM mapper ensures that:
  - The static point cloud lives in the SelfCap world coordinate frame.
  - No coordinate-frame mismatch with the dynamic branch (which also uses GT poses).
  - Reconstruction is faster and free of SfM pose drift.
"""

import shutil
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .colmap_runner import (
    find_colmap,
    read_points3D,
    run_exhaustive_matcher,
    run_feature_extractor,
    run_point_triangulator,
)
from .colmap_writer import write_known_pose_model
from .dataset_adapter import WorkspaceInfo, prepare_colmap_workspace
from .pcd_utils import static_temporal_priors, write_init_ply

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))
from data.selfcap_loader import read_selfcap_cameras, read_selfcap_sync


def _generate_motion_masks(
    source_path: str,
    work: Path,
    cam_names: List[str],
    ref_master: int,
    fps: float,
    *,
    diff_threshold: int = 15,
    dilate_px: int = 30,
) -> Optional[str]:
    """
    Generate per-camera motion masks for the reference frame via frame differencing.

    Compares the reference frame against up to 4 neighboring frames (±1, ±2) and
    takes the union of all detected motion regions.  The resulting mask is inverted
    for COLMAP: pixels = 0 are excluded from SIFT extraction (dynamic regions),
    pixels = 255 are kept (static background).

    COLMAP mask naming convention:
        For image "{cam}.jpg"  →  mask_path/{cam}.jpg.png

    Returns:
        Path to the mask directory, or None if mask generation failed for all cameras.
    """
    frames_root = Path(source_path) / "frames"
    if not frames_root.exists():
        print("[static] Warning: frames/ not found, skipping motion mask generation.")
        return None

    sync_map = read_selfcap_sync(source_path)
    mask_dir = work / "static_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    n_success = 0
    for cam_name in cam_names:
        cam_dir = frames_root / cam_name
        if not cam_dir.exists():
            continue

        sync_offset = sync_map.get(cam_name, 0.0)
        ref_eff = int(round(ref_master + sync_offset * fps))
        ref_path = cam_dir / f"{ref_eff:05d}.jpg"

        if not ref_path.exists():
            print(f"[static] Warning: {ref_path.name} missing for {cam_name}, skipping mask.")
            continue

        ref_gray = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)

        # Collect motion mask from multiple neighboring frames (OR-union)
        motion_mask = np.zeros_like(ref_blur)
        for delta in [1, -1, 2, -2]:
            nbr_path = cam_dir / f"{ref_eff + delta:05d}.jpg"
            if not nbr_path.exists():
                continue
            nbr_gray = cv2.imread(str(nbr_path), cv2.IMREAD_GRAYSCALE)
            nbr_blur = cv2.GaussianBlur(nbr_gray, (5, 5), 0)
            diff = cv2.absdiff(ref_blur, nbr_blur)
            _, m = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.bitwise_or(motion_mask, m)

        if not motion_mask.any():
            # No neighbors found or completely static — write a fully-permissive mask
            colmap_mask = np.full_like(ref_blur, 255)
        else:
            if dilate_px > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (dilate_px, dilate_px)
                )
                motion_mask = cv2.dilate(motion_mask, kernel)
            # Invert: 255=static (keep for SIFT), 0=dynamic (mask out)
            colmap_mask = 255 - motion_mask

        out_path = mask_dir / f"{cam_name}.jpg.png"
        cv2.imwrite(str(out_path), colmap_mask)
        n_success += 1

    if n_success == 0:
        print("[static] Warning: no motion masks generated — proceeding without masking.")
        return None

    print(f"[static]   Motion masks: {n_success}/{len(cam_names)} cameras → {mask_dir}")
    return str(mask_dir)


def run_static_branch(
    source_path: str,
    work_dir: str,
    output_ply: str,
    *,
    start_frame: int = 0,
    end_frame: int = -1,
    reference_frame: Optional[int] = None,
    gpu: bool = True,
    colmap_bin: Optional[str] = None,
    camera_model: str = "OPENCV",   # kept for API compatibility; OPENCV is always used
) -> str:
    """
    Run the full static branch and write an init PLY.

    Args:
        source_path:     SelfCap dataset root (contains extri.yml / intri.yml).
        work_dir:        Scratch directory for COLMAP artefacts.
        output_ply:      Path where the final static init PLY will be written.
        start_frame:     First master frame of the training range.
        end_frame:       Last master frame (-1 = video end).
        reference_frame: Master frame used for COLMAP images.  Defaults to
                         start_frame (minimal-motion assumption).
        gpu:             Use GPU for SIFT extraction and matching.
        colmap_bin:      Path to colmap binary; auto-detected if None.
        camera_model:    Ignored — OPENCV is always used (matches extri/intri format).

    Returns:
        Path to the written PLY (same as output_ply).
    """
    colmap = find_colmap(colmap_bin)
    work   = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    db_path      = work / "colmap.db"
    sparse_input = work / "sparse_input"
    sparse_out   = work / "sparse_output"

    # Always start with a clean COLMAP workspace so that motion masks are
    # applied to a fresh feature extraction (COLMAP skips re-extraction for
    # images already present in the DB).
    for artifact in [db_path, sparse_input, sparse_out]:
        if artifact.is_file():
            artifact.unlink()
        elif artifact.is_dir():
            shutil.rmtree(artifact)

    # ------------------------------------------------------------------ #
    # Step 1 — Prepare COLMAP workspace                                   #
    # ------------------------------------------------------------------ #
    print("\n[static] Step 1 — Preparing COLMAP workspace")
    ws: WorkspaceInfo = prepare_colmap_workspace(
        source_path=source_path,
        work_dir=str(work),
        start_frame=start_frame,
        end_frame=end_frame,
        reference_frame=reference_frame,
    )

    # ------------------------------------------------------------------ #
    # Step 2 — Write known-pose sparse model from GT poses                #
    # ------------------------------------------------------------------ #
    print("\n[static] Step 2 — Writing GT-pose sparse model (cameras/images .bin)")
    _, extrinsics, intrinsics = read_selfcap_cameras(source_path)

    write_known_pose_model(
        output_dir=str(sparse_input),
        cam_names=ws.cam_names,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_dir=str(ws.image_dir),
    )
    print(f"[static]   Written to {sparse_input}")

    # ------------------------------------------------------------------ #
    # Step 2.5 — Generate motion masks for reference frame               #
    # Exclude dynamic regions from SIFT to avoid triangulating           #
    # foreground objects into the static point cloud.                    #
    # ------------------------------------------------------------------ #
    print("\n[static] Step 2.5 — Generating reference-frame motion masks")
    ref_master = reference_frame if reference_frame is not None else start_frame
    mask_path_arg = _generate_motion_masks(
        source_path=source_path,
        work=work,
        cam_names=ws.cam_names,
        ref_master=ref_master,
        fps=ws.fps,
    )

    # ------------------------------------------------------------------ #
    # Step 3 — Feature extraction                                         #
    # ------------------------------------------------------------------ #
    print("\n[static] Step 3 — Feature extraction")
    run_feature_extractor(
        db_path=str(db_path),
        image_path=str(ws.image_dir),
        gpu=gpu,
        camera_model="OPENCV",
        single_camera_per_folder=False,
        camera_params=None,   # COLMAP reads from DB; poses come from sparse_input
        mask_path=mask_path_arg,
        colmap_bin=colmap,
    )

    # ------------------------------------------------------------------ #
    # Step 4 — Exhaustive matching                                        #
    # ------------------------------------------------------------------ #
    print("\n[static] Step 4 — Exhaustive matching")
    run_exhaustive_matcher(
        db_path=str(db_path),
        gpu=gpu,
        colmap_bin=colmap,
    )

    # ------------------------------------------------------------------ #
    # Step 5 — Point triangulation with known poses                       #
    # ------------------------------------------------------------------ #
    print("\n[static] Step 5 — Point triangulation (known GT poses)")
    run_point_triangulator(
        db_path=str(db_path),
        image_path=str(ws.image_dir),
        input_path=str(sparse_input),
        output_path=str(sparse_out),
        colmap_bin=colmap,
    )

    # ------------------------------------------------------------------ #
    # Step 6 — Read reconstruction                                        #
    # ------------------------------------------------------------------ #
    print("\n[static] Step 6 — Reading points3D")
    xyz, rgb = read_points3D(str(sparse_out))
    n = len(xyz)
    print(f"[static]   Triangulated {n} points")

    if n == 0:
        raise RuntimeError(
            "COLMAP point_triangulator produced zero points. "
            "Check that the reference-frame images have sufficient overlap "
            "and that the GT poses in extri.yml are correct."
        )

    # ------------------------------------------------------------------ #
    # Step 7 — Assign temporal priors and write PLY                       #
    # ------------------------------------------------------------------ #
    print("\n[static] Step 7 — Writing init PLY")
    priors = static_temporal_priors(ws.duration_seconds, n)
    write_init_ply(
        output_path=output_ply,
        xyz=xyz,
        rgb=rgb,
        t=priors["t"],
        t_scale_log=priors["t_scale_log"],
        motion=priors["motion"],
    )
    print(
        f"[static] Done. {n} static points → {output_ply}\n"
        f"         duration={ws.duration_seconds:.2f}s  "
        f"t_mid={ws.duration_seconds/2:.2f}s  "
        f"t_scale=log({3*ws.duration_seconds:.2f})"
    )
    return output_ply
