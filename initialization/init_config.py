"""
SharpTimeGS initialization configuration.

Mirrors the style of config/config.py — flat dataclasses per concern,
nested inside InitConfig, with from_yaml / to_yaml helpers.

Usage:
    cfg = InitConfig.from_yaml("config/sharptime_init_bar.yaml")
    # or override individual fields after loading
    cfg.dataset.source_path = "/data/bar-release"
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Paths and frame range — required fields, no defaults."""
    source_path: str = ""          # SelfCap dataset root (extri.yml / intri.yml)
    output_path: str = ""          # Final sharptime_init.ply path
    work_dir: str = ""             # Scratch dir (default: source_path/.sharptime_init_work)
    start_frame: int = 0           # First master frame of the training window
    end_frame: int = -1            # Last master frame (-1 = infer from frames on disk)
    extract_frames: bool = True    # Auto-extract frames if frames/ does not exist


@dataclass
class StaticConfig:
    """Settings for Step 1 — static COLMAP triangulation with GT poses."""
    reference_frame: Optional[int] = None  # Master frame used for reference images
                                            # (default: start_frame)
    colmap_bin: Optional[str] = None        # Path to colmap binary (auto-detected)
    skip: bool = False                      # Skip this step entirely


@dataclass
class FlowConfig:
    """Settings for Step 2 — RAFT optical flow."""
    model: str = "large"        # RAFT model size: "large" | "small"
    iters: int = 20             # Recurrent update iterations
    threshold: float = 2.0     # Flow magnitude (pixels) above which a pixel is "dynamic"
    max_size: int = 1024        # Resize long edge to this before inference, then scale flow back.
                                # Set to -1 to disable (risks OOM on high-res images).
    skip: bool = False          # Skip (requires existing flow/ dir in work_dir)


@dataclass
class SegmentationConfig:
    """Settings for Step 3 — SAM2 video segmentation."""
    model: str = "large"          # SAM2 model: "large" | "base_plus" | "small" | "tiny"
    n_prompt_points: int = 5      # High-flow seed points passed to SAM2 per camera
    offload_to_cpu: bool = False  # Offload video frames to CPU (saves VRAM for long clips)
    skip: bool = False            # Skip (requires existing masks/ dir in work_dir)

    # --- Prompt vertical ROI --------------------------------------------------
    # Positive prompts are restricted to image rows [y_min_frac*H, y_max_frac*H).
    # Prevents floor pixels at the image bottom from being selected as foreground
    # seeds when floor motion (shadows, reflections) outweighs subject motion.
    prompt_y_min_frac: float = 0.0   # exclude top fraction  (0 = no exclusion)
    prompt_y_max_frac: float = 1.0   # exclude bottom fraction (1 = no exclusion)
    # Add SAM2 negative prompts (label=0) from the excluded bottom region.
    # Explicitly tells the model "do not include floor here".
    n_negative_points: int = 0

    # --- Cross-camera 3D reprompting -----------------------------------------
    # After the first SAM2 pass, cameras with bad masks (too large or centroid
    # too low) are detected.  The mask centroids of good cameras are triangulated
    # to a 3D subject position, then re-projected into bad cameras as corrected
    # prompt points for a second SAM2 pass.
    cross_cam_reproj: bool = False          # enable 2nd-pass reprompting
    mask_area_max_frac: float = 0.35        # mask > this fraction of image → bad
    mask_centroid_y_max_frac: float = 0.72  # mask centroid below this row → bad


@dataclass
class DynamicConfig:
    """Settings for Steps 4-5 — per-frame SIFT triangulation and KNN velocity."""
    n_sift_features: int = 2000    # Max SIFT keypoints per camera per frame
    match_ratio: float = 0.75      # Lowe's ratio-test threshold
    reproj_threshold: float = 4.0  # Max reprojection error (pixels) to keep a 3D point
    min_matches: int = 8           # Min SIFT matches per camera pair to triangulate
    knn_k: int = 5                 # K nearest neighbours for cross-frame velocity estimation
    max_speed: Optional[float] = None  # Hard upper bound on init velocity magnitude
                                       # (world_units/s). IQR-based cap always applies;
                                       # this adds a scene-specific absolute ceiling.
                                       # E.g. 600 for cm-scale corgi, None = IQR only.
    # --- Point cloud densification (applied after triangulation, before writing PLY) ---
    n_duplicate: int = 1           # Replicate each dynamic point this many times (1 = off).
                                   # Each copy receives independent Gaussian XYZ jitter.
                                   # Velocity and temporal anchor are copied unchanged.
                                   # Recommended range: 2–5. Training-time pruning will
                                   # remove redundant copies; seed points cost nothing extra.
    duplicate_jitter_sigma: float = 0.5  # Jitter magnitude as a fraction of the mean
                                         # inter-point KNN distance in the per-frame cloud.
                                         # 0.3 = tight (sub-voxel), 0.7 = loose (may gap-fill).
    skip: bool = False             # Skip (output_path will be static branch only)


@dataclass
class DenseConfig:
    """Settings for Step 4b — dense dynamic region reconstruction via StereoSGBM."""
    enabled: bool = False          # Enable dense MVS (supplements or replaces sparse SIFT)
    merge_sparse: bool = True      # Keep SIFT triangulated points alongside dense cloud
    block_size: int = 11           # SGBM block size (odd, 5–21); smaller = more detail + noise
    uniqueness_ratio: int = 10     # Reject matches where runner-up is within N% of winner
    speckle_window_size: int = 100 # Remove isolated disparity regions smaller than this
    speckle_range: int = 2         # Max disparity variation within a speckle region
    min_depth: float = 0.3         # Min camera-space depth to retain (metres)
    max_depth: float = 10.0        # Max camera-space depth to retain (metres)
    min_baseline: float = 0.1      # Skip pairs whose optical-centre distance < this (metres)
    max_baseline: float = 3.0      # Skip pairs with baseline > this (metres).
                                   # Large-baseline pairs produce disparities >> image width
                                   # causing SGBM integer overflow. Keep ≤ ~3 m.
    voxel_size: float = 0.02       # Voxel grid cell size for output downsampling (metres).
                                   # 0 = no downsampling. Recommended: 0.01–0.05.
    max_image_size: int = 960      # Resize the longer image edge to this before SGBM.
                                   # K is scaled consistently; 3-D output stays metric.
                                   # Lower = less RAM. 640–1280 are typical safe values.


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class InitConfig:
    """
    Complete SharpTimeGS initialization configuration.

    Load from YAML:
        cfg = InitConfig.from_yaml("config/sharptime_init_bar.yaml")

    Save to YAML:
        cfg.to_yaml("config/sharptime_init_bar.yaml")
    """
    dataset:     DatasetConfig      = field(default_factory=DatasetConfig)
    static:      StaticConfig       = field(default_factory=StaticConfig)
    flow:        FlowConfig         = field(default_factory=FlowConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    dynamic:     DynamicConfig      = field(default_factory=DynamicConfig)
    dense:       DenseConfig        = field(default_factory=DenseConfig)

    device:    str  = "cuda"   # "cuda" | "cpu"
    overwrite: bool = False    # Re-run steps even if output files exist

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "InitConfig":
        """Load config from a YAML file.  Missing keys fall back to defaults."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}

        def _sub(klass, key):
            return klass(**{
                k: v for k, v in d.get(key, {}).items()
                if k in klass.__dataclass_fields__
            })

        return cls(
            dataset=_sub(DatasetConfig, "dataset"),
            static=_sub(StaticConfig, "static"),
            flow=_sub(FlowConfig, "flow"),
            segmentation=_sub(SegmentationConfig, "segmentation"),
            dynamic=_sub(DynamicConfig, "dynamic"),
            dense=_sub(DenseConfig, "dense"),
            device=d.get("device", "cuda"),
            overwrite=d.get("overwrite", False),
        )

    def to_yaml(self, yaml_path: str) -> None:
        """Serialize config to YAML."""
        d = {
            "dataset":      asdict(self.dataset),
            "static":       asdict(self.static),
            "flow":         asdict(self.flow),
            "segmentation": asdict(self.segmentation),
            "dynamic":      asdict(self.dynamic),
            "dense":        asdict(self.dense),
            "device":       self.device,
            "overwrite":    self.overwrite,
        }
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(d, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def resolve_work_dir(self) -> str:
        """Return work_dir, falling back to {source_path}/.sharptime_init_work."""
        if self.dataset.work_dir:
            return self.dataset.work_dir
        return str(Path(self.dataset.source_path) / ".sharptime_init_work")

    def validate(self) -> None:
        """Raise ValueError for obviously wrong settings."""
        if not self.dataset.source_path:
            raise ValueError("dataset.source_path must be set")
        if not self.dataset.output_path:
            raise ValueError("dataset.output_path must be set")
        if not Path(self.dataset.source_path).exists():
            raise ValueError(f"source_path does not exist: {self.dataset.source_path}")
        if self.flow.model not in ("large", "small"):
            raise ValueError(f"flow.model must be 'large' or 'small', got '{self.flow.model}'")
        if self.segmentation.model not in ("large", "base_plus", "small", "tiny"):
            raise ValueError(
                f"segmentation.model must be one of large/base_plus/small/tiny, "
                f"got '{self.segmentation.model}'"
            )

    def __str__(self) -> str:
        lines = ["=" * 56, "SharpTimeGS Init Config", "=" * 56]
        for section_name, section in [
            ("dataset", self.dataset),
            ("static", self.static),
            ("flow", self.flow),
            ("segmentation", self.segmentation),
            ("dynamic", self.dynamic),
        ]:
            lines.append(f"\n[{section_name}]")
            for k, v in asdict(section).items():
                lines.append(f"  {k}: {v}")
        lines += [f"\ndevice:    {self.device}", f"overwrite: {self.overwrite}", "=" * 56]
        return "\n".join(lines)
