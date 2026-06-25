"""
SharpTimeGS initialization pipeline.

This module exposes a single public function `run_pipeline(cfg)` that runs all
initialization steps driven by an InitConfig.  It can be imported from Python
code (notebooks, training scripts) or called from the CLI in
tools/sharptime_init.py.

Pipeline steps:
  1. Static COLMAP triangulation   (known GT poses from extri.yml)
  2. RAFT optical flow              (adjacent frame pairs, all cameras)
  3. SAM2 video segmentation        (seeded from flow; per-camera masks)
  4. Per-frame dynamic triangulation (SIFT in masked regions)
  5. KNN cross-frame velocity estimation
  6. Dynamic PLY assembly            (temporal priors + velocity)
  7. Static + dynamic PLY merge      → final sharptime_init.ply

Every step writes its outputs under work_dir and is idempotent when
overwrite=False, so partial runs can be resumed by re-running the script.
"""

import shutil
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from extract_frames import extract_frames as _extract_frames

from .init_config import InitConfig
from .static_branch import run_static_branch
from .optical_flow import compute_flow_for_dataset
from .segmentation import compute_masks_for_dataset
from .dynamic_branch import run_dynamic_branch
from .dense_branch import run_dense_dynamic_branch, merge_frame_data
from .velocity_estimator import estimate_velocities
from .pcd_builder import build_dynamic_ply, merge_plys, read_fps_from_source


def run_pipeline(cfg: InitConfig) -> str:
    """
    Execute the full SharpTimeGS initialization pipeline.

    Args:
        cfg: Fully populated InitConfig (call cfg.validate() before this).

    Returns:
        Absolute path to the written sharptime_init.ply.
    """
    cfg.validate()

    source     = Path(cfg.dataset.source_path)
    work_dir   = cfg.resolve_work_dir()
    work       = Path(work_dir)
    device     = cfg.device
    overwrite  = cfg.overwrite

    static_ply    = str(work / "static_init.ply")
    dynamic_ply   = str(work / "dynamic_init.ply")
    output_path   = cfg.dataset.output_path
    masks_out_dir = str(source / "masks")   # SAM2 masks go to dataset root, not work_dir

    print(cfg)

    # ------------------------------------------------------------------ #
    # Step 0 — Frame extraction (auto, if frames/ does not exist)         #
    # ------------------------------------------------------------------ #
    frames_root = source / "frames"
    if cfg.dataset.extract_frames:
        first_cam_frames = next(
            (frames_root / c for c in source.iterdir()
             if (frames_root / c.name).is_dir()), None
        ) if frames_root.exists() else None
        already_done = (
            frames_root.exists()
            and first_cam_frames is not None
            and any(first_cam_frames.iterdir())
        )
        if not already_done:
            print("\n[init] === Step 0: Frame extraction ===")
            _extract_frames(
                source_path=str(source),
                start_frame=cfg.dataset.start_frame,
                end_frame=cfg.dataset.end_frame,
                overwrite=overwrite,
            )
        else:
            print(f"[init] Step 0 (frames): already extracted at {frames_root}, skipping.")
    else:
        if not frames_root.exists():
            raise FileNotFoundError(
                f"frames/ not found at {frames_root}. "
                "Set dataset.extract_frames: true or run tools/extract_frames.py first."
            )

    # ------------------------------------------------------------------ #
    # Step 1 — Static branch (GT-pose COLMAP triangulation)               #
    # ------------------------------------------------------------------ #
    if not cfg.static.skip:
        print("\n[init] === Step 1: Static branch ===")
        run_static_branch(
            source_path=str(source),
            work_dir=work_dir,
            output_ply=static_ply,
            start_frame=cfg.dataset.start_frame,
            end_frame=cfg.dataset.end_frame,
            reference_frame=cfg.static.reference_frame,
            gpu=(device == "cuda"),
            colmap_bin=cfg.static.colmap_bin,
        )
    else:
        _skip_msg("Step 1 (static)", static_ply)

    # ------------------------------------------------------------------ #
    # Step 2 — Optical flow (RAFT)                                        #
    # ------------------------------------------------------------------ #
    if not cfg.flow.skip:
        print("\n[init] === Step 2: Optical flow (RAFT) ===")
        flow_paths = compute_flow_for_dataset(
            source_path=str(source),
            work_dir=work_dir,
            start_frame=cfg.dataset.start_frame,
            end_frame=cfg.dataset.end_frame,
            device=device,
            model_size=cfg.flow.model,
            iters=cfg.flow.iters,
            max_size=cfg.flow.max_size,
            overwrite=overwrite,
        )
        n_flow = sum(len(v) for v in flow_paths.values())
        print(f"[init] Flow: {n_flow} files written to {work_dir}/flow/")
    else:
        _skip_msg("Step 2 (flow)", str(work / "flow"))

    # ------------------------------------------------------------------ #
    # Step 3 — SAM2 video segmentation                                    #
    # ------------------------------------------------------------------ #
    if not cfg.segmentation.skip and not cfg.dynamic.skip:
        flow_dir = work / "flow"
        if not flow_dir.exists():
            print(
                "[init] Warning: no flow/ directory found — SAM2 segmentation skipped.\n"
                "         Run with flow.skip: false first."
            )
        else:
            print("\n[init] === Step 3: SAM2 video segmentation ===")
            mask_paths = compute_masks_for_dataset(
                source_path=str(source),
                work_dir=work_dir,
                start_frame=cfg.dataset.start_frame,
                end_frame=cfg.dataset.end_frame,
                model_size=cfg.segmentation.model,
                device=device,
                flow_threshold=cfg.flow.threshold,
                n_prompt_points=cfg.segmentation.n_prompt_points,
                prompt_y_min_frac=cfg.segmentation.prompt_y_min_frac,
                prompt_y_max_frac=cfg.segmentation.prompt_y_max_frac,
                n_negative_points=cfg.segmentation.n_negative_points,
                cross_cam_reproj=cfg.segmentation.cross_cam_reproj,
                mask_area_max_frac=cfg.segmentation.mask_area_max_frac,
                mask_centroid_y_max_frac=cfg.segmentation.mask_centroid_y_max_frac,
                offload_to_cpu=cfg.segmentation.offload_to_cpu,
                overwrite=overwrite,
                masks_dir=masks_out_dir,
            )
            n_masks = sum(len(v) for v in mask_paths.values())
            print(f"[init] Segmentation: {n_masks} masks written to {masks_out_dir}/")
    else:
        _skip_msg("Step 3 (segmentation)", masks_out_dir)

    # ------------------------------------------------------------------ #
    # Step 4 — Per-frame dynamic triangulation (sparse SIFT)             #
    # ------------------------------------------------------------------ #
    frames_data = {}
    if not cfg.dynamic.skip:
        if not Path(masks_out_dir).exists():
            print(
                "[init] Note: no masks/ found — dynamic triangulation will use the\n"
                "       full image (no masking). Run SAM2 first for better results."
            )
        sift_mode = "relaxed (dense branch will supplement)" if cfg.dense.enabled else "original"
        print(f"\n[init] === Step 4a: Per-frame sparse triangulation (SIFT, {sift_mode}) ===")
        frames_data = run_dynamic_branch(
            source_path=str(source),
            work_dir=work_dir,
            start_frame=cfg.dataset.start_frame,
            end_frame=cfg.dataset.end_frame,
            n_features=cfg.dynamic.n_sift_features,
            match_ratio=cfg.dynamic.match_ratio,
            reproj_threshold=cfg.dynamic.reproj_threshold,
            min_matches=cfg.dynamic.min_matches,
            overwrite=overwrite,
            masks_dir=masks_out_dir,
        )
    else:
        print("[init] Step 4a (sparse triangulation): skipped  (dynamic.skip=true)")

    # ------------------------------------------------------------------ #
    # Step 4b — Dense MVS for dynamic region (optional)                  #
    # ------------------------------------------------------------------ #
    if not cfg.dynamic.skip and cfg.dense.enabled:
        print("\n[init] === Step 4b: Per-frame dense reconstruction (StereoSGBM) ===")
        dense_data = run_dense_dynamic_branch(
            source_path=str(source),
            work_dir=work_dir,
            start_frame=cfg.dataset.start_frame,
            end_frame=cfg.dataset.end_frame,
            block_size=cfg.dense.block_size,
            uniqueness_ratio=cfg.dense.uniqueness_ratio,
            speckle_window_size=cfg.dense.speckle_window_size,
            speckle_range=cfg.dense.speckle_range,
            min_depth=cfg.dense.min_depth,
            max_depth=cfg.dense.max_depth,
            min_baseline=cfg.dense.min_baseline,
            max_baseline=cfg.dense.max_baseline,
            voxel_size=cfg.dense.voxel_size,
            max_image_size=cfg.dense.max_image_size,
            overwrite=overwrite,
            masks_dir=masks_out_dir,
        )
        if cfg.dense.merge_sparse:
            frames_data = merge_frame_data(frames_data, dense_data)
            n_total = sum(d["xyz"].shape[0] for d in frames_data.values())
            print(f"[init] Merged sparse + dense: {n_total} total dynamic points")
        else:
            frames_data = dense_data

    # ------------------------------------------------------------------ #
    # Step 5 — KNN velocity estimation                                    #
    # ------------------------------------------------------------------ #
    velocities = {}
    if not cfg.dynamic.skip and frames_data:
        print("\n[init] === Step 5: KNN velocity estimation ===")
        fps = read_fps_from_source(str(source))
        velocities = estimate_velocities(
            frames_data=frames_data,
            fps=fps,
            k=cfg.dynamic.knn_k,
            max_speed=cfg.dynamic.max_speed,
        )
        _log_velocity_stats(velocities)
        print(f"[init] Velocity: {len(velocities)} frames  (fps={fps:.1f})")

    # ------------------------------------------------------------------ #
    # Step 6 — Build dynamic PLY                                          #
    # ------------------------------------------------------------------ #
    dynamic_written = False
    if not cfg.dynamic.skip and frames_data:
        print("\n[init] === Step 6: Building dynamic PLY ===")
        fps = read_fps_from_source(str(source))
        # n_duplicate / jitter are only applied when the dense branch ran.
        # Without dense, SIFT already produces the final sparse cloud and
        # duplicating points would introduce artificial near-duplicates.
        if cfg.dense.enabled:
            n_dup   = cfg.dynamic.n_duplicate
            jitter  = cfg.dynamic.duplicate_jitter_sigma
        else:
            n_dup   = 1
            jitter  = 0.0
            if cfg.dynamic.n_duplicate > 1:
                print("[init]   dense disabled → n_duplicate reset to 1 (original SIFT mode)")
        try:
            build_dynamic_ply(
                frames_data=frames_data,
                velocities=velocities,
                output_path=dynamic_ply,
                fps=fps,
                start_frame=cfg.dataset.start_frame,
                n_duplicate=n_dup,
                duplicate_jitter_sigma=jitter,
            )
            dynamic_written = True
        except ValueError as e:
            print(f"[init] Warning: could not build dynamic PLY — {e}")

    # ------------------------------------------------------------------ #
    # Step 7 — Merge static + dynamic → final PLY                         #
    # ------------------------------------------------------------------ #
    print(f"\n[init] === Step 7: Writing final PLY → {output_path} ===")
    static_exists  = Path(static_ply).exists()
    dynamic_exists = dynamic_written and Path(dynamic_ply).exists()

    if static_exists and dynamic_exists:
        merge_plys(static_ply, dynamic_ply, output_path)
    elif static_exists:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(static_ply, output_path)
        print(f"[init] Static-only: copied {static_ply} → {output_path}")
    elif dynamic_exists:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dynamic_ply, output_path)
        print(f"[init] Dynamic-only: copied {dynamic_ply} → {output_path}")
    else:
        raise RuntimeError(
            "Neither static nor dynamic PLY was produced. "
            "Check earlier steps for errors."
        )

    print(f"\n[init] Done.  PLY: {output_path}")
    return str(Path(output_path).resolve())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _skip_msg(step: str, expected_path: str) -> None:
    exists = Path(expected_path).exists()
    status = "found" if exists else "NOT FOUND"
    print(f"[init] {step}: skipped  ({expected_path}  {status})")


def _log_velocity_stats(velocities: dict) -> None:
    """Print a compact speed distribution summary across all frames."""
    import numpy as np
    all_speeds = []
    for vel in velocities.values():
        speeds = np.linalg.norm(vel, axis=1)
        all_speeds.append(speeds)
    if not all_speeds:
        return
    all_speeds = np.concatenate(all_speeds)
    nz = all_speeds[all_speeds > 0.0]
    total = len(all_speeds)
    n_nz  = len(nz)
    if n_nz == 0:
        print(f"[init]   speed stats: all zero  (N={total})")
        return
    q25, q50, q75, q95 = (
        np.percentile(nz, 25),
        np.percentile(nz, 50),
        np.percentile(nz, 75),
        np.percentile(nz, 95),
    )
    print(
        f"[init]   speed stats (non-zero {n_nz}/{total}): "
        f"Q25={q25:.1f}  Q50={q50:.1f}  Q75={q75:.1f}  "
        f"Q95={q95:.1f}  max={nz.max():.1f}  (world_units/s)"
    )
