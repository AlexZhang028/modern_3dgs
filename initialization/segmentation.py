"""
Dynamic region segmentation using SAM2 video predictor.

Strategy
--------
For each camera, a temporary JPEG directory is created containing only the
training-range frames, named sequentially (00000.jpg, 00001.jpg, …).  SAM2's
video predictor processes this as a continuous clip, propagating masks
temporally for consistent dynamic-region labelling.

High-flow pixels from the RAFT output (Step 3) are used as positive point
prompts, seeded on the frame with the highest motion coverage.  Propagation
runs bidirectionally from the seed frame so objects that appear or disappear
mid-sequence are still captured.

Output
------
    {work_dir}/masks/{cam_name}/{eff_t:05d}.png
        Binary mask, 255 = dynamic, 0 = static.  Named by effective frame
        index (same convention as the flow files).

Checkpoints
-----------
Downloaded automatically to ~/.cache/sam2/ on first use.

    sam2.1_hiera_large.pt   (~224 MB)  default, highest quality
    sam2.1_hiera_small.pt   (~46 MB)   faster, less VRAM
    sam2.1_hiera_tiny.pt    (~39 MB)   fastest

SAM2 installation (already handled by environment setup):
    pip install sam2
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.selfcap_loader import read_selfcap_cameras, read_selfcap_sync
from initialization.optical_flow import flow_magnitude, load_flow


# ---------------------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------------------

_CONFIGS = {
    "large":     "configs/sam2.1/sam2.1_hiera_l.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "small":     "configs/sam2.1/sam2.1_hiera_s.yaml",
    "tiny":      "configs/sam2.1/sam2.1_hiera_t.yaml",
}
_CKPT_NAMES = {
    "large":     "sam2.1_hiera_large.pt",
    "base_plus": "sam2.1_hiera_base_plus.pt",
    "small":     "sam2.1_hiera_small.pt",
    "tiny":      "sam2.1_hiera_tiny.pt",
}
_BASE_URL  = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
_CACHE_DIR = Path.home() / ".cache" / "sam2"


def get_checkpoint(model_size: str = "large") -> Tuple[str, str]:
    """
    Return (model_cfg, ckpt_path) for the requested model size, downloading
    the checkpoint if it is not already cached.
    """
    if model_size not in _CONFIGS:
        raise ValueError(f"model_size must be one of {list(_CONFIGS)}, got '{model_size}'")

    ckpt_name = _CKPT_NAMES[model_size]
    ckpt_path = _CACHE_DIR / ckpt_name

    if not ckpt_path.exists():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        url = _BASE_URL + ckpt_name
        print(f"[seg] Downloading SAM2 checkpoint: {url}")
        torch.hub.download_url_to_file(url, str(ckpt_path), progress=True)

    return _CONFIGS[model_size], str(ckpt_path)


# ---------------------------------------------------------------------------
# Core segmenter
# ---------------------------------------------------------------------------

class Sam2VideoSegmenter:
    """
    Wraps the SAM2 video predictor for dynamic-region segmentation.

    Args:
        model_size:          "large" | "base_plus" | "small" | "tiny"
        device:              Torch device string.
        offload_to_cpu:      Offload video frames to CPU to save VRAM.
                             Recommended for sequences longer than ~200 frames
                             or when VRAM is under 12 GB.
    """

    def __init__(
        self,
        model_size: str = "large",
        device: str = "cuda",
        offload_to_cpu: bool = False,
    ):
        from sam2.build_sam import build_sam2_video_predictor

        cfg, ckpt = get_checkpoint(model_size)
        self.predictor      = build_sam2_video_predictor(cfg, ckpt, device=device)
        self.device         = device
        self.offload_to_cpu = offload_to_cpu
        print(f"[seg] SAM2-{model_size} ready on {device}")

    def segment_sequence(
        self,
        frame_dir: Path,
        flow_dir: Optional[Path],
        eff_indices: List[int],
        flow_threshold: float = 2.0,
        n_prompt_points: int = 5,
        y_min_frac: float = 0.0,
        y_max_frac: float = 1.0,
        n_negative_points: int = 0,
        fixed_points: Optional[np.ndarray] = None,
        fixed_labels: Optional[np.ndarray] = None,
        obj_id: int = 1,
    ) -> Dict[int, np.ndarray]:
        """
        Run SAM2 video segmentation on a sequential JPEG directory.

        Args:
            frame_dir:        Directory with frames named 00000.jpg, 00001.jpg, …
                              (sequential symlinks built by compute_masks_for_dataset).
            flow_dir:         Directory with flow_*.npy files (one per sequential index).
                              If None, a fallback empty-mask result is returned.
            eff_indices:      Effective frame indices corresponding to 00000.jpg …
                              Used to name the output masks.
            flow_threshold:   Pixels with flow magnitude > this are "dynamic".
            n_prompt_points:  Number of seed points passed to SAM2.
            obj_id:           SAM2 object ID (1 = primary dynamic object).

        Returns:
            Dict mapping eff_idx → [H, W] uint8 mask (0 / 255).
        """
        n = len(eff_indices)
        if n == 0:
            return {}

        # ---- build aggregated motion map from available flow files ----
        agg_mag, seed_seq_idx = self._aggregate_flow(flow_dir, n, flow_threshold)

        # ---- pick prompt points ----
        if fixed_points is not None:
            # Cross-camera reprompting: use caller-supplied points
            points = fixed_points
            labels = (fixed_labels if fixed_labels is not None
                      else np.ones(len(fixed_points), dtype=np.int32))
        else:
            points, labels = _pick_prompt_points(
                agg_mag, flow_threshold, n_prompt_points,
                y_min_frac=y_min_frac, y_max_frac=y_max_frac,
                n_negative_points=n_negative_points,
            )

        if points is None:
            print("[seg]   No high-flow pixels found — returning empty masks.")
            # Return empty masks
            sample = cv2.imread(str(frame_dir / "00000.jpg"))
            H, W = (sample.shape[:2] if sample is not None else (0, 0))
            return {eff: np.zeros((H, W), dtype=np.uint8) for eff in eff_indices}

        # ---- init SAM2 state ----
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            state = self.predictor.init_state(
                video_path=str(frame_dir),
                offload_video_to_cpu=self.offload_to_cpu,
            )
            H = state["video_height"]
            W = state["video_width"]

            self.predictor.add_new_points_or_box(
                state,
                frame_idx=seed_seq_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

            masks: Dict[int, np.ndarray] = {}

            # ---- forward propagation from seed ----
            for seq_idx, obj_ids, logits in self.predictor.propagate_in_video(
                state, start_frame_idx=seed_seq_idx
            ):
                if obj_id in obj_ids:
                    i = obj_ids.index(obj_id)
                    mask = (logits[i][0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                else:
                    mask = np.zeros((H, W), dtype=np.uint8)
                masks[eff_indices[seq_idx]] = mask

            # ---- backward propagation from seed ----
            if seed_seq_idx > 0:
                for seq_idx, obj_ids, logits in self.predictor.propagate_in_video(
                    state, start_frame_idx=seed_seq_idx, reverse=True
                ):
                    eff = eff_indices[seq_idx]
                    if eff in masks:
                        # already set by forward pass — keep as-is
                        continue
                    if obj_id in obj_ids:
                        i = obj_ids.index(obj_id)
                        mask = (logits[i][0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                    else:
                        mask = np.zeros((H, W), dtype=np.uint8)
                    masks[eff] = mask

            self.predictor.reset_state(state)

        return masks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aggregate_flow(
        self,
        flow_dir: Optional[Path],
        n_frames: int,
        threshold: float,
    ) -> Tuple[Optional[np.ndarray], int]:
        """
        Load all flow files, sum magnitudes, return (agg_mag, best_seed_idx).

        agg_mag:       [H, W] float32, sum of per-frame flow magnitudes.
        best_seed_idx: sequential index (0-based) of the frame with highest
                       dynamic coverage, used as the SAM2 seed frame.
        """
        if flow_dir is None or not flow_dir.exists():
            return None, 0

        flow_files = sorted(flow_dir.glob("*.npy"), key=lambda p: int(p.stem))
        if not flow_files:
            return None, 0

        agg: Optional[np.ndarray] = None
        per_frame_score: List[float] = []

        for npy in flow_files:
            try:
                flow = load_flow(str(npy))          # [H, W, 2]
                mag  = flow_magnitude(flow)          # [H, W]
            except Exception:
                per_frame_score.append(0.0)
                continue

            if agg is None:
                agg = np.zeros_like(mag)
            agg += mag
            per_frame_score.append(float((mag > threshold).sum()))

        if agg is None:
            return None, 0

        best_seq_idx = int(np.argmax(per_frame_score)) if per_frame_score else 0
        return agg, best_seq_idx


# ---------------------------------------------------------------------------
# Dataset-level mask computation
# ---------------------------------------------------------------------------

def compute_masks_for_dataset(
    source_path: str,
    work_dir: str,
    start_frame: int = 0,
    end_frame: int = -1,
    *,
    model_size: str = "large",
    device: str = "cuda",
    flow_threshold: float = 2.0,
    n_prompt_points: int = 5,
    prompt_y_min_frac: float = 0.0,
    prompt_y_max_frac: float = 1.0,
    n_negative_points: int = 0,
    cross_cam_reproj: bool = False,
    mask_area_max_frac: float = 0.35,
    mask_centroid_y_max_frac: float = 0.72,
    offload_to_cpu: bool = False,
    overwrite: bool = False,
    masks_dir: Optional[str] = None,
) -> Dict[str, List[Path]]:
    """
    Segment dynamic regions for every camera across the training range.

    For each camera:
      1. Builds a temp directory of sequentially-named JPEG symlinks covering
         [start_frame, end_frame].  SAM2 treats this as a continuous video clip.
      2. Seeds the predictor with high-flow point prompts derived from the
         aggregated RAFT flow (Step 3 output).
      3. Propagates masks bidirectionally from the highest-motion seed frame.
      4. Saves masks as {masks_dir}/{cam_name}/{eff_t:05d}.png.

    Args:
        source_path:      Dataset root (frames/ and flow/ relative subdirs expected).
        work_dir:         Root of intermediate artefacts (same as for optical_flow).
        start_frame:      First master frame of the training range.
        end_frame:        Last master frame (-1 = inferred from frames on disk).
        model_size:       "large" | "base_plus" | "small" | "tiny".
        device:           "cuda" or "cpu".
        flow_threshold:   Flow-magnitude threshold (pixels) for dynamic detection.
        n_prompt_points:  Number of SAM2 foreground prompt points per camera.
        offload_to_cpu:   Offload video frames to CPU memory (saves VRAM for long clips).
        overwrite:        Re-compute even if mask files already exist.
        masks_dir:        Directory to write masks into.  Defaults to
                          {work_dir}/masks if not provided.

    Returns:
        Dict mapping cam_name → list of saved mask Paths.
    """
    source      = Path(source_path)
    frames_root = source / "frames"
    flow_root   = Path(work_dir) / "flow"
    masks_root  = Path(masks_dir) if masks_dir else Path(work_dir) / "masks"

    cam_names, extrinsics, intrinsics = read_selfcap_cameras(str(source))
    sync_map = read_selfcap_sync(str(source))
    fps      = _read_fps(source)

    segmenter = Sam2VideoSegmenter(
        model_size=model_size,
        device=device,
        offload_to_cpu=offload_to_cpu,
    )

    result: Dict[str, List[Path]] = {}
    # raw masks kept in memory for cross-camera quality check
    all_raw_masks: Dict[str, Dict[int, np.ndarray]] = {}

    for cam_name in cam_names:
        cam_frames_dir = frames_root / cam_name
        if not cam_frames_dir.exists():
            print(f"[seg] {cam_name}: no frames directory, skipping.")
            continue

        sync_offset = sync_map.get(cam_name, 0.0)
        cam_masks_dir = masks_root / cam_name
        cam_masks_dir.mkdir(parents=True, exist_ok=True)

        # ---- build ordered list of (master_t, eff_t) pairs ----
        pairs = _build_frame_pairs(cam_frames_dir, start_frame, end_frame, sync_offset, fps)
        if not pairs:
            print(f"[seg] {cam_name}: no frames in requested range, skipping.")
            continue

        # ---- check for already-done masks ----
        missing = [(mt, et) for mt, et in pairs
                   if not (cam_masks_dir / f"{et:05d}.png").exists() or overwrite]
        if not missing:
            print(f"[seg] {cam_name}: all {len(pairs)} masks exist, skipping.")
            result[cam_name] = [cam_masks_dir / f"{et:05d}.png" for _, et in pairs]
            # Load saved masks for cross-camera quality check
            if cross_cam_reproj:
                all_raw_masks[cam_name] = {
                    et: cv2.imread(str(cam_masks_dir / f"{et:05d}.png"),
                                   cv2.IMREAD_GRAYSCALE)
                    for _, et in pairs
                }
            continue

        eff_indices = [et for _, et in pairs]

        # ---- create temp sequential symlink dir ----
        with _temp_frame_dir(cam_frames_dir, eff_indices) as tmp_dir:
            flow_dir = flow_root / cam_name if flow_root.exists() else None

            print(f"[seg] {cam_name}: segmenting {len(eff_indices)} frames …")
            masks = segmenter.segment_sequence(
                frame_dir=tmp_dir,
                flow_dir=flow_dir,
                eff_indices=eff_indices,
                flow_threshold=flow_threshold,
                n_prompt_points=n_prompt_points,
                y_min_frac=prompt_y_min_frac,
                y_max_frac=prompt_y_max_frac,
                n_negative_points=n_negative_points,
            )

        # ---- save masks ----
        saved: List[Path] = []
        for eff_t, mask in masks.items():
            out_path = cam_masks_dir / f"{eff_t:05d}.png"
            cv2.imwrite(str(out_path), mask)
            saved.append(out_path)

        for eff_t in eff_indices:
            out_path = cam_masks_dir / f"{eff_t:05d}.png"
            if not out_path.exists():
                cv2.imwrite(str(out_path), np.zeros((1, 1), dtype=np.uint8))
                saved.append(out_path)

        print(f"[seg] {cam_name}: saved {len(saved)} masks → {cam_masks_dir}")
        result[cam_name] = saved
        if cross_cam_reproj:
            all_raw_masks[cam_name] = masks

    # ------------------------------------------------------------------ #
    # Cross-camera 3D reprompting (2nd pass for bad cameras)             #
    # ------------------------------------------------------------------ #
    if cross_cam_reproj and all_raw_masks:
        print("\n[seg] === Cross-camera 3D reprompting pass ===")
        bad_cams, good_centroids = _classify_masks(
            all_raw_masks, mask_area_max_frac, mask_centroid_y_max_frac
        )
        if bad_cams:
            print(f"[seg] Bad cameras: {bad_cams}")
            X_world = _compute_3d_subject_centroid(good_centroids, extrinsics, intrinsics)
            if X_world is not None:
                print(f"[seg] Triangulated 3D subject centre: {X_world.round(2)}")
                for cam_name in bad_cams:
                    if cam_name not in extrinsics or cam_name not in intrinsics:
                        continue
                    R = np.asarray(extrinsics[cam_name]["R"], dtype=np.float64)
                    T = np.asarray(extrinsics[cam_name]["T"], dtype=np.float64).flatten()
                    K = np.asarray(intrinsics[cam_name]["K"], dtype=np.float64)
                    proj = _project_to_camera(X_world, R, T, K)
                    if proj is None:
                        print(f"[seg] {cam_name}: projected point behind camera, skipping.")
                        continue
                    px, py = proj
                    print(f"[seg] {cam_name}: re-prompting at ({px:.0f}, {py:.0f})")
                    fixed_pts    = np.array([[px, py]], dtype=np.float32)
                    fixed_labels = np.array([1], dtype=np.int32)

                    cam_frames_dir = frames_root / cam_name
                    if not cam_frames_dir.exists():
                        continue
                    sync_offset   = sync_map.get(cam_name, 0.0)
                    cam_masks_dir = masks_root / cam_name

                    pairs = _build_frame_pairs(
                        cam_frames_dir, start_frame, end_frame, sync_offset, fps
                    )
                    if not pairs:
                        continue
                    eff_indices = [et for _, et in pairs]

                    with _temp_frame_dir(cam_frames_dir, eff_indices) as tmp_dir:
                        flow_dir = flow_root / cam_name if flow_root.exists() else None
                        masks = segmenter.segment_sequence(
                            frame_dir=tmp_dir,
                            flow_dir=flow_dir,
                            eff_indices=eff_indices,
                            flow_threshold=flow_threshold,
                            n_prompt_points=n_prompt_points,
                            fixed_points=fixed_pts,
                            fixed_labels=fixed_labels,
                        )

                    saved = []
                    for eff_t, mask in masks.items():
                        out_path = cam_masks_dir / f"{eff_t:05d}.png"
                        cv2.imwrite(str(out_path), mask)
                        saved.append(out_path)
                    print(f"[seg] {cam_name}: reprompted, saved {len(saved)} masks.")
                    result[cam_name] = saved
            else:
                print("[seg] Could not triangulate 3D centroid — skipping reprompting.")
        else:
            print("[seg] All cameras passed quality check — no reprompting needed.")

    return result


def load_mask(path: str) -> np.ndarray:
    """Load a saved mask. Returns [H, W] uint8 (0=static, 255=dynamic)."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pick_prompt_points(
    agg_mag: Optional[np.ndarray],
    threshold: float,
    n_points: int,
    y_min_frac: float = 0.0,
    y_max_frac: float = 1.0,
    n_negative_points: int = 0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Select N well-spread foreground prompt points from the aggregated magnitude map.

    Positive prompts are restricted to image rows [y_lo, y_hi) to prevent floor
    pixels at the bottom from being chosen as foreground seeds.  Optionally adds
    SAM2 negative prompts (label=0) in the excluded bottom region.

    Returns (points [K, 2] float32 in (x, y) order, labels [K] int32) or
    (None, None) when no above-threshold pixels exist.
    """
    if agg_mag is None:
        return None, None

    H, W   = agg_mag.shape
    y_lo   = int(H * y_min_frac)
    y_hi   = int(H * y_max_frac)

    ys, xs = np.where(agg_mag > threshold)
    if len(ys) == 0:
        return None, None

    # --- positive prompts: restrict to vertical ROI ---
    roi    = (ys >= y_lo) & (ys < y_hi)
    p_ys   = ys[roi];  p_xs = xs[roi]
    if len(p_ys) == 0:
        # ROI has no dynamic pixels — fall back to the full image
        p_ys, p_xs = ys, xs

    p_mags = agg_mag[p_ys, p_xs]
    order  = np.argsort(p_mags)[::-1]
    min_d  = max(H, W) // max(2 * n_points, 1)

    chosen_pos = []
    for i in order:
        if len(chosen_pos) >= n_points:
            break
        py, px = int(p_ys[i]), int(p_xs[i])
        if chosen_pos:
            dists = [np.hypot(px - p[0], py - p[1]) for p in chosen_pos]
            if min(dists) < min_d:
                continue
        chosen_pos.append((px, py))

    if not chosen_pos:
        return None, None

    all_pts    = list(chosen_pos)
    all_labels = [1] * len(chosen_pos)

    # --- negative prompts: high-flow pixels in the excluded bottom region ---
    if n_negative_points > 0 and y_max_frac < 1.0:
        neg    = ys >= y_hi
        n_ys   = ys[neg];  n_xs = xs[neg]
        if len(n_ys) > 0:
            n_mags  = agg_mag[n_ys, n_xs]
            n_order = np.argsort(n_mags)[::-1]
            min_d_n = max(H, W) // max(2 * n_negative_points, 1)
            chosen_neg = []
            for i in n_order:
                if len(chosen_neg) >= n_negative_points:
                    break
                py, px = int(n_ys[i]), int(n_xs[i])
                if chosen_neg:
                    dists = [np.hypot(px - p[0], py - p[1]) for p in chosen_neg]
                    if min(dists) < min_d_n:
                        continue
                chosen_neg.append((px, py))
            all_pts.extend(chosen_neg)
            all_labels.extend([0] * len(chosen_neg))

    pts    = np.array(all_pts,    dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int32)
    return pts, labels


def _build_frame_pairs(
    cam_frames_dir: Path,
    start_frame: int,
    end_frame: int,
    sync_offset: float,
    fps: float,
) -> List[Tuple[int, int]]:
    """
    Return ordered [(master_t, eff_t)] for the requested training range.

    For each master frame t in [start_frame, end_frame], the effective frame
    index is eff_t = round(t + sync_offset * fps).  Only pairs whose .jpg
    file exists on disk are returned.
    """
    existing_eff: set = {int(p.stem) for p in cam_frames_dir.glob("*.jpg")}
    if not existing_eff:
        return []

    # Infer end_frame from available files when -1 is passed
    if end_frame <= 0:
        max_eff = max(existing_eff)
        end_frame = int(round(max_eff - sync_offset * fps))

    pairs = []
    for master_t in range(start_frame, end_frame + 1):
        eff_t = int(round(master_t + sync_offset * fps))
        if eff_t in existing_eff:
            pairs.append((master_t, eff_t))

    return pairs


class _temp_frame_dir:
    """
    Context manager that creates a temporary directory with sequential JPEG
    symlinks pointing at the requested effective-index frames.

    On exit the temporary directory is removed.
    """

    def __init__(self, cam_frames_dir: Path, eff_indices: List[int]):
        self._cam_dir    = cam_frames_dir
        self._eff_indices = eff_indices
        self._tmp: Optional[Path] = None

    def __enter__(self) -> Path:
        self._tmp = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
        for seq_i, eff_t in enumerate(self._eff_indices):
            src = self._cam_dir / f"{eff_t:05d}.jpg"
            dst = self._tmp / f"{seq_i:05d}.jpg"
            if src.exists():
                os.symlink(str(src.resolve()), str(dst))
        return self._tmp

    def __exit__(self, *_):
        if self._tmp and self._tmp.exists():
            shutil.rmtree(str(self._tmp), ignore_errors=True)


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


# ---------------------------------------------------------------------------
# Cross-camera 3D reprompting helpers
# ---------------------------------------------------------------------------

def _classify_masks(
    all_masks: Dict[str, Dict[int, np.ndarray]],
    area_max_frac: float,
    centroid_y_max_frac: float,
) -> Tuple[List[str], Dict[str, Tuple[float, float]]]:
    """
    Classify cameras as good/bad based on their aggregate mask quality.

    Returns:
        bad_cams:       List of camera names whose masks look wrong (floor/bg).
        good_centroids: Dict cam → (cx_px, cy_px) centroid for good cameras.
    """
    bad_cams: List[str] = []
    good_centroids: Dict[str, Tuple[float, float]] = {}

    for cam_name, masks in all_masks.items():
        frames = [m for m in masks.values() if m is not None and m.size > 0]
        if not frames:
            bad_cams.append(cam_name)
            continue

        H, W = frames[0].shape[:2]
        # Majority-vote aggregate: pixel is dynamic if it's dynamic in >50% of frames
        agg = np.zeros((H, W), dtype=np.float32)
        for m in frames:
            if m.shape[:2] == (H, W):
                agg += (m > 0).astype(np.float32)
        agg_bin = (agg / len(frames)) > 0.5

        area_frac = float(agg_bin.sum()) / (H * W)
        ys_m, xs_m = np.where(agg_bin)

        if len(ys_m) == 0:
            bad_cams.append(cam_name)
            continue

        cy_frac = float(ys_m.mean()) / H

        if area_frac > area_max_frac or cy_frac > centroid_y_max_frac:
            bad_cams.append(cam_name)
        else:
            good_centroids[cam_name] = (float(xs_m.mean()), float(ys_m.mean()))

    return bad_cams, good_centroids


def _compute_3d_subject_centroid(
    cam_centroids: Dict[str, Tuple[float, float]],
    extrinsics: dict,
    intrinsics: dict,
) -> Optional[np.ndarray]:
    """
    Least-squares ray intersection: finds the 3D point closest to all camera
    rays originating from the mask centroid of each good camera.

    Each camera gives a ray  O + t*d  where:
        O = -R.T @ T        (camera centre in world)
        d = R.T @ K_inv @ [cx, cy, 1]  (normalised ray direction in world)

    Closest-point-to-all-rays solution (Ax = b):
        A = sum_i (I - d_i d_i^T)
        b = sum_i (I - d_i d_i^T) @ O_i
    """
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3,      dtype=np.float64)
    n_rays = 0

    for cam_name, (cx, cy) in cam_centroids.items():
        if cam_name not in extrinsics or cam_name not in intrinsics:
            continue
        R   = np.asarray(extrinsics[cam_name]["R"], dtype=np.float64)
        T   = np.asarray(extrinsics[cam_name]["T"], dtype=np.float64).flatten()
        K   = np.asarray(intrinsics[cam_name]["K"], dtype=np.float64)

        O   = -(R.T @ T)                                 # world-space camera centre
        d_c = np.linalg.inv(K) @ np.array([cx, cy, 1.0]) # ray in camera space
        d   = R.T @ (d_c / np.linalg.norm(d_c))         # ray in world space (normalised)

        P   = np.eye(3) - np.outer(d, d)                 # projection ⊥ to ray
        A  += P
        b  += P @ O
        n_rays += 1

    if n_rays < 2:
        return None

    try:
        X, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        if rank < 3:
            return None
        return X.astype(np.float32)
    except np.linalg.LinAlgError:
        return None


def _project_to_camera(
    X_world: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    Project a world point into a camera.  Returns (x_px, y_px) or None if
    the point is behind the camera.
    """
    X_cam = R @ X_world.astype(np.float64) + T
    if X_cam[2] <= 0.0:
        return None
    x = K @ X_cam
    return (float(x[0] / x[2]), float(x[1] / x[2]))
