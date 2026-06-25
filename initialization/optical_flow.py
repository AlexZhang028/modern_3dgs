"""
Optical flow estimation using RAFT (torchvision.models.optical_flow).

For each adjacent master-frame pair (t → t+1) and each camera, computes
2-D optical flow and saves it as:
    {work_dir}/flow/{cam_name}/{master_t:05d}.npy   # shape [H, W, 2]

Values are pixel displacements in the JPEG-image coordinate space.
Flow magnitude [H, W] = norm(flow, axis=-1) is used downstream by SAM2
as a saliency map for dynamic-region detection.

Notes
-----
* RAFT requires H and W to both be divisible by 8. Images are zero-padded
  before inference and the padding is cropped from the result.
* Weights are downloaded automatically on first use (~20 MB for large model).
* No new pip/conda dependencies: torchvision >= 0.13 already bundles RAFT.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.selfcap_loader import read_selfcap_cameras, read_selfcap_sync


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

class RaftFlowEstimator:
    """
    Thin wrapper around torchvision's RAFT implementation.

    Args:
        model_size: "large" (higher quality) or "small" (faster, less VRAM).
        device:     "cuda" or "cpu".
        iters:      Number of recurrent update iterations (12–20 for inference).
    """

    def __init__(
        self,
        model_size: str = "large",
        device: str = "cuda",
        iters: int = 20,
        max_size: int = 1024,
    ):
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )

        self.device   = device
        self.iters    = iters
        self.max_size = max_size  # -1 → no resize

        if model_size == "large":
            weights    = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=weights).to(device).eval()
        elif model_size == "small":
            weights    = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=weights).to(device).eval()
        else:
            raise ValueError(f"model_size must be 'large' or 'small', got '{model_size}'")

        self.transforms = weights.transforms()
        print(f"[flow] RAFT-{model_size} ready on {device}  (iters={iters}, max_size={max_size})")

    @torch.no_grad()
    def estimate(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate forward optical flow from frame1 to frame2.

        Args:
            frame1, frame2: [H, W, 3] uint8 **RGB** arrays.

        Returns:
            flow: [H, W, 2] float32.  flow[y, x] = (du, dv) in pixels,
                  where du is the horizontal (x) displacement and dv is the
                  vertical (y) displacement.
        """
        H, W = frame1.shape[:2]

        # --- optional down-scale to fit in VRAM ---
        scale = 1.0
        if self.max_size > 0 and max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)
            new_h = int(round(H * scale))
            new_w = int(round(W * scale))
            import cv2 as _cv2
            frame1 = _cv2.resize(frame1, (new_w, new_h), interpolation=_cv2.INTER_AREA)
            frame2 = _cv2.resize(frame2, (new_w, new_h), interpolation=_cv2.INTER_AREA)

        h, w = frame1.shape[:2]   # possibly down-scaled dimensions

        # --- build batched uint8 tensors [1, 3, h, w] ---
        t1 = _to_uint8_tensor(frame1, self.device)
        t2 = _to_uint8_tensor(frame2, self.device)

        # --- normalise (uint8 → float32 in [-1, 1]) ---
        t1, t2 = self.transforms(t1, t2)

        # --- pad to multiples of 8 ---
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            t1 = F.pad(t1, [0, pad_w, 0, pad_h])
            t2 = F.pad(t2, [0, pad_w, 0, pad_h])

        # --- run RAFT ---
        flow_preds = self.model(t1, t2, num_flow_updates=self.iters)
        flow_small = flow_preds[-1][0, :, :h, :w]   # [2, h, w]

        # --- up-scale flow back to original resolution ---
        if scale < 1.0:
            flow_up = F.interpolate(
                flow_small.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
            )[0]                         # [2, H, W]
            flow_up = flow_up / scale    # rescale displacement magnitudes
        else:
            flow_up = flow_small         # [2, H, W]

        return flow_up.permute(1, 2, 0).cpu().numpy()  # [H, W, 2]


# ---------------------------------------------------------------------------
# Dataset-level flow computation
# ---------------------------------------------------------------------------

def compute_flow_for_dataset(
    source_path: str,
    work_dir: str,
    start_frame: int = 0,
    end_frame: int = -1,
    *,
    device: str = "cuda",
    model_size: str = "large",
    iters: int = 20,
    max_size: int = 1024,
    overwrite: bool = False,
) -> Dict[str, List[Path]]:
    """
    Compute forward optical flow for every adjacent frame pair across all cameras.

    For master frame t in [start_frame, end_frame-1] and each camera:
      - Load {source_path}/frames/{cam_name}/{eff_t:05d}.jpg  (frame t)
      - Load {source_path}/frames/{cam_name}/{eff_t1:05d}.jpg (frame t+1)
      - Run RAFT
      - Save to {work_dir}/flow/{cam_name}/{master_t:05d}.npy

    Args:
        source_path: Dataset root (contains extri.yml and frames/).
        work_dir:    Scratch directory; flow files are written under flow/.
        start_frame: First master frame of the training range.
        end_frame:   Last master frame (exclusive upper bound, -1 = all).
        device:      Torch device for RAFT inference.
        model_size:  "large" or "small".
        iters:       RAFT recurrent iterations.
        overwrite:   If False, skip already-computed .npy files.

    Returns:
        Dict mapping cam_name → list of Paths to the saved .npy files
        (one per master frame pair).
    """
    source     = Path(source_path)
    frames_root = source / "frames"
    flow_root   = Path(work_dir) / "flow"

    if not frames_root.exists():
        raise FileNotFoundError(
            f"frames/ directory not found at {frames_root}. "
            "Run tools/extract_frames.py first."
        )

    # --- read SelfCap metadata ---
    cam_names, _, _ = read_selfcap_cameras(str(source))
    sync_map        = read_selfcap_sync(str(source))

    # --- determine fps from first available video (for sync offset calc) ---
    fps = _read_fps(source)

    # --- determine effective frame range ---
    # Find maximum eff_idx to bound end_frame
    all_eff = _max_eff_idx(frames_root, cam_names)
    if end_frame <= 0 or end_frame > all_eff + 1:
        end_frame = all_eff + 1          # inclusive upper bound

    master_frames = list(range(start_frame, end_frame))
    if len(master_frames) < 2:
        raise ValueError("Need at least 2 frames to compute optical flow.")

    # Drop last frame (no t+1 available)
    pairs = list(zip(master_frames[:-1], master_frames[1:]))

    # --- build estimator ---
    estimator = RaftFlowEstimator(model_size=model_size, device=device, iters=iters, max_size=max_size)

    result: Dict[str, List[Path]] = {}

    for cam_name in cam_names:
        cam_frames_dir = frames_root / cam_name
        if not cam_frames_dir.exists():
            print(f"[flow] Skipping {cam_name}: no frames directory.")
            continue

        sync_offset = sync_map.get(cam_name, 0.0)
        out_dir     = flow_root / cam_name
        out_dir.mkdir(parents=True, exist_ok=True)

        saved: List[Path] = []
        skipped = 0

        for master_t, master_t1 in tqdm(
            pairs, desc=f"  {cam_name}", leave=False, unit="pair"
        ):
            out_path = out_dir / f"{master_t:05d}.npy"
            if out_path.exists() and not overwrite:
                saved.append(out_path)
                skipped += 1
                continue

            # effective frame indices accounting for sync offset
            eff_t  = int(round(master_t  + sync_offset * fps))
            eff_t1 = int(round(master_t1 + sync_offset * fps))

            path1 = cam_frames_dir / f"{eff_t:05d}.jpg"
            path2 = cam_frames_dir / f"{eff_t1:05d}.jpg"

            if not path1.exists() or not path2.exists():
                # Missing frame — write a zero-flow placeholder
                if path1.exists():
                    h, w = cv2.imread(str(path1)).shape[:2]
                else:
                    h, w = 0, 0
                if h > 0:
                    np.save(str(out_path), np.zeros((h, w, 2), dtype=np.float32))
                print(f"  [flow] Warning: missing {path2.name} for {cam_name}, wrote zeros.")
                saved.append(out_path)
                continue

            frame1 = _load_rgb(str(path1))
            frame2 = _load_rgb(str(path2))
            flow   = estimator.estimate(frame1, frame2)
            np.save(str(out_path), flow.astype(np.float32))
            saved.append(out_path)

        print(
            f"[flow] {cam_name}: {len(saved)} pairs  "
            f"({skipped} skipped, {len(saved)-skipped} computed)"
        )
        result[cam_name] = saved

    return result


def load_flow(path: str) -> np.ndarray:
    """Load a saved flow file. Returns [H, W, 2] float32."""
    return np.load(path)


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """Compute per-pixel flow magnitude. Returns [H, W] float32."""
    return np.sqrt((flow ** 2).sum(axis=-1))


def flow_to_motion_mask(
    flow: np.ndarray,
    threshold: float = 2.0,
) -> np.ndarray:
    """
    Threshold flow magnitude to produce a binary motion mask.

    Args:
        flow:      [H, W, 2] flow array.
        threshold: Pixels with magnitude > threshold are marked as dynamic.

    Returns:
        mask: [H, W] bool — True where motion exceeds the threshold.
    """
    return flow_magnitude(flow) > threshold


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_uint8_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    """[H, W, 3] uint8 RGB numpy → [1, 3, H, W] uint8 torch tensor."""
    return (
        torch.from_numpy(img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )


def _load_rgb(path: str) -> np.ndarray:
    """Load JPEG as [H, W, 3] uint8 RGB."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_fps(source: Path) -> float:
    """Read FPS from the first .mp4 in videos/ (or source root)."""
    for root in [source / "videos", source]:
        if root.exists():
            for f in os.listdir(root):
                if f.lower().endswith(".mp4"):
                    cap = cv2.VideoCapture(str(root / f))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()
                    return fps if fps > 0 else 30.0
    return 30.0


def _max_eff_idx(frames_root: Path, cam_names: list) -> int:
    """Find the maximum effective frame index stored under frames/."""
    max_idx = 0
    for name in cam_names:
        d = frames_root / name
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix == ".jpg":
                try:
                    max_idx = max(max_idx, int(f.stem))
                except ValueError:
                    pass
    return max_idx
