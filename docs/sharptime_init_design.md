# SharpTimeGS Velocity-Aware Initialization — Design Document

## 1. Goal & Scope

Implement the velocity-aware initialization from SharpTimeGS (§3.3, arXiv 2602.02989v2).
The existing `freetimegs_init.py` uses RoMA for feature matching and produces a single
merged point cloud with no dynamic/static separation.  This design replaces it with a
two-branch pipeline that:

- Runs **headless COLMAP** (CLI only, no GUI) for fully automated reconstruction.
- Separates dynamic and static regions at initialization time via optical flow + SAM2
  segmentation, assigning physically meaningful temporal priors to each Gaussian.
- Produces a single SharpTimeGS-compatible PLY that `load_ply()` can read directly
  without any change to `gaussian_model.py`.

---

## 2. Pipeline Overview

```
Multi-view video (N_cam cameras × N_frames frames)
│
├─── STATIC BRANCH ──────────────────────────────────────────────────────┐
│    Frame 0, full images, all cameras                                    │
│    └─ headless COLMAP (feature_extractor + exhaustive_matcher + mapper) │
│       → static_sparse/  (cameras.bin, points3D.bin)                    │
│       → static_pcd.ply  [N_static points]                              │
│       Init: v=0, T=mid_frame, σ_t=log(3·N_frames), r=softplus⁻¹(1e-6) │
└─────────────────────────────────────────────────────────────────────────┘
│
└─── DYNAMIC BRANCH ─────────────────────────────────────────────────────┐
     For each frame t = 0 … N_frames-1:                                  │
       Step 1  RAFT optical flow (frame t → t+1, all cameras)            │
               → flow fields [H×W×2]                                     │
       Step 2  SAM2 video segmentation (flow magnitude as point prompts) │
               → binary dynamic masks  [H×W]  per (cam, frame)           │
       Step 3  Masked per-frame COLMAP triangulation                     │
               (reuse known cameras from static branch)                  │
               → dynamic_pcd_t.ply  [M_t points, XYZ + color]            │
       Step 4  KNN matching across adjacent frames                       │
               (t → t+1, k=1 in 3D space)                               │
               → v_init = (X_{t+1} - X_t) / Δt  per matched point       │
     Merge all frames → dynamic_all.ply                                  │
     Init: T=t, v=v_init, σ_t=log(3·Δt_per_frame), r=softplus⁻¹(1e-6)  │
└─────────────────────────────────────────────────────────────────────────┘
│
└─── MERGE → init_point_cloud.ply  (static + dynamic, all SharpTimeGS fields)
```

---

## 3. Static Branch

### 3.1 Input

All camera images at **frame 0** (or a chosen reference frame with minimal motion).
Camera intrinsics/extrinsics are unknown at this point and will be estimated by COLMAP.

### 3.2 Headless COLMAP Commands

```bash
COLMAP_DB=$WORK/static/colmap.db
IMAGES=$WORK/static/images/       # symlink or copy of frame-0 images
SPARSE=$WORK/static/sparse/

# 1. Feature extraction (single camera set per image for multi-camera rigs)
colmap feature_extractor \
    --database_path   $COLMAP_DB \
    --image_path      $IMAGES \
    --ImageReader.single_camera_per_folder 1 \
    --SiftExtraction.use_gpu 1

# 2. Exhaustive matching (N_cam is small, e.g. 4-21)
colmap exhaustive_matcher \
    --database_path $COLMAP_DB \
    --SiftMatching.use_gpu 1

# 3. Sparse mapping
colmap mapper \
    --database_path $COLMAP_DB \
    --image_path    $IMAGES \
    --output_path   $SPARSE

# 4. (Optional) Convert to text for debugging
colmap model_converter \
    --input_path  $SPARSE/0 \
    --output_path $SPARSE/0 \
    --output_type TXT
```

The resulting `$SPARSE/0/` contains:
- `cameras.bin` / `cameras.txt` — intrinsics per camera
- `images.bin` / `images.txt` — extrinsics (R, t) per image
- `points3D.bin` / `points3D.txt` — triangulated 3D points + RGB

### 3.3 Parameter Mapping (paper → code)

| Paper symbol | Value at init | PLY field | Notes |
|---|---|---|---|
| `X` | COLMAP point XYZ | `x, y, z` | world coords |
| `S`, `R`, `O`, `Y` | 3DGS defaults | standard fields | same as static init |
| `T` (time anchor) | `(t_start + t_end) / 2` | `t` | middle of sequence |
| `v` (velocity) | `[0, 0, 0]` | `motion_x/y/z` | static → no drift |
| `σ_t` (temporal scale) | `log(3 · N_frames · Δt)` | `t_scale` | covers full sequence |
| `r` (lifespan radius) | `softplus⁻¹(1e-6) ≈ -13.816` | `r` | near-zero flat region |

The `σ_t` init makes the Gaussian's temporal weight still ≈1 at the edges of the
full sequence, so static Gaussians never fade during training.

---

## 4. Dynamic Branch

### 4.1 Step 1 — Optical Flow (RAFT)

For each adjacent frame pair `(t, t+1)` and each camera `c`:

```python
# raft_flow.py
from raft import RAFT
model = RAFT(args).cuda()
model.load_state_dict(torch.load('raft-things.pth'))

# padder = InputPadder(image1.shape)
flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
# flow_up: [1, 2, H, W]  (u, v displacements in pixels)
```

Flow magnitude `‖flow_up‖₂` is used as the dynamic confidence map.
Threshold at `flow_threshold` (default 2.0 px) to get initial motion mask.

**Output:** `flow_{cam}_{t}.npy`  shape `[H, W, 2]`

### 4.2 Step 2 — Dynamic Mask (SAM2)

Use high-flow pixels as point prompts for SAM2 video segmentation:

```python
# sam2_mask.py
from sam2.build_sam import build_sam2_video_predictor

predictor = build_sam2_video_predictor(model_cfg, checkpoint)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path=frames_dir)
    # Seed from frame 0 high-flow regions
    predictor.add_new_points_or_box(
        state, frame_idx=0, obj_id=1,
        points=high_flow_coords,      # [K, 2]  pixel coords of K flow peaks
        labels=np.ones(K, dtype=np.int32)
    )
    for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
        # masks: bool [H, W]
        save_mask(frame_idx, masks)
```

**Output:** `mask_{cam}_{t}.png`  binary `[H, W]`, True = dynamic

### 4.3 Step 3 — Per-Frame Masked COLMAP Triangulation

Camera parameters are already known from the static branch reconstruction.
We only need to **triangulate new feature matches** in the masked images:

```bash
# For frame t, mask images from all cameras, then:
MASKED_IMAGES=$WORK/dynamic/frame_{t}/images/
MASKED_DB=$WORK/dynamic/frame_{t}/colmap.db
KNOWN_SPARSE=$WORK/static/sparse/0/      # cameras + image poses
OUT_SPARSE=$WORK/dynamic/frame_{t}/sparse/

# Extract features from masked images
colmap feature_extractor \
    --database_path $MASKED_DB \
    --image_path    $MASKED_IMAGES \
    --SiftExtraction.use_gpu 1

# Match features between cameras at same frame
colmap exhaustive_matcher \
    --database_path $MASKED_DB \
    --SiftMatching.use_gpu 1

# Triangulate with known camera poses (do NOT re-run mapper)
colmap point_triangulator \
    --database_path $MASKED_DB \
    --image_path    $MASKED_IMAGES \
    --input_path    $KNOWN_SPARSE \
    --output_path   $OUT_SPARSE \
    --Mapper.filter_min_tri_angle 2.0
```

> **Note:** `point_triangulator` requires the `images.bin` in `$KNOWN_SPARSE` to already
> have the frame-t images registered.  If the dataset follows a multi-camera rig convention
> where camera poses are shared across frames, the same `cameras.bin` / `images.bin` from
> the static branch applies — just import the per-frame images into the same DB with known
> camera IDs.

**Alternative (faster):** Triangulate directly in Python via `pycolmap`:

```python
import pycolmap

rec = pycolmap.Reconstruction(known_sparse_path)
for img_id, image in rec.images.items():
    image.registered = True  # mark as registered
pycolmap.triangulate_points(rec, masked_db_path, masked_image_path)
# rec.points3D now has dynamic points for frame t
```

**Output:** `dynamic_pcd_t.ply`  per-frame masked sparse point cloud

### 4.4 Step 4 — KNN Cross-Frame Velocity Estimation

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

def estimate_velocity(pts_t, pts_t1, delta_t, k=1):
    """
    pts_t   : [M_t,  3]  dynamic points at frame t
    pts_t1  : [M_t1, 3]  dynamic points at frame t+1
    delta_t : float       time between frames (seconds)
    Returns : v_init [M_t, 3]
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(pts_t1)
    dists, indices = nbrs.kneighbors(pts_t)        # [M_t, k]
    # Use closest match
    matched_pts_t1 = pts_t1[indices[:, 0]]         # [M_t, 3]
    displacement   = matched_pts_t1 - pts_t        # [M_t, 3]
    v_init         = displacement / delta_t         # [M_t, 3]
    return v_init
```

Unmatched points at the last frame (no t+1) receive `v = [0,0,0]`.

**Dynamic Gaussian temporal parameters:**

| Paper symbol | Value | Notes |
|---|---|---|
| `T` | frame timestamp `t` | normalized to `[0, 1]` if `normalized_t=True` |
| `v` | `v_init` from KNN | in world units/second |
| `σ_t` | `log(3 · Δt_per_frame)` | covers ≈3 frame durations initially |
| `r` | `softplus⁻¹(1e-6) ≈ -13.816` | near-zero flat region initially |

---

## 5. Merge & Output PLY

Both branches produce point clouds that are concatenated into a single init PLY.

### 5.1 PLY Field Specification

The output PLY uses the **existing SharpTimeGS PLY format** — no changes to
`gaussian_model.py` are needed:

```
vertex {
  # Standard 3DGS fields
  float x, y, z
  float opacity           # inverse_sigmoid(0.1)
  float scale_0, scale_1, scale_2   # log-space
  float rot_0, rot_1, rot_2, rot_3  # unit quaternion
  float f_dc_0, f_dc_1, f_dc_2      # DC SH (RGB → SH)
  float f_rest_0 … f_rest_44        # higher SH bands (0 if SH=3 init)

  # FreeTimeGS / SharpTimeGS temporal fields
  float t                 # time anchor T  (normalized or seconds)
  float t_scale           # log(σ_t)
  float motion_x, motion_y, motion_z   # velocity v (world units/s, normalized if applicable)

  # SharpTimeGS-only
  float r                 # lifespan radius logit (softplus⁻¹(r_val))
}
```

No new PLY fields are needed.  `detect_mode_from_ply()` already identifies this as
`"sharptime"` by the presence of the `r` field.

### 5.2 Standard Gaussian Attribute Initialization

Both static and dynamic points use the same heuristics as vanilla 3DGS:

```python
# Scale: log of mean nearest-neighbor distance
dists = compute_knn_distances(xyz, k=3)
scales = np.log(np.sqrt(dists))            # [N, 1] → broadcast to [N, 3]

# Opacity: inverse_sigmoid(0.1)
opacities = inverse_sigmoid(0.1 * np.ones((N, 1)))

# Rotation: identity quaternion
rotations = np.tile([1, 0, 0, 0], (N, 1))  # [N, 4]

# SH DC: RGB to SH coefficient
sh_dc = RGB2SH(colors)                     # [N, 3]
# SH rest: zeros
sh_rest = np.zeros((N, 3, (sh_degree+1)**2 - 1))
```

---

## 6. New Data Structures

### 6.1 `BasicPointCloud` — No Change Required

The existing `BasicPointCloud(points, colors, normals, t, t_scale, motion)` is
sufficient for the `create_from_pcd` path.  The `r` parameter is always
initialized to `_R_INIT_LOGIT` in `SharpTimeGaussianModel.create_from_pcd`.

However, if we want `create_from_pcd` to accept per-point `r` values (e.g. to
pass different init radii for static vs dynamic), we would extend it:

```python
# utils/graphics_utils.py — optional future extension
class BasicPointCloud(NamedTuple):
    points:  np.ndarray
    colors:  np.ndarray
    normals: np.ndarray
    t:       Optional[np.ndarray] = None
    t_scale: Optional[np.ndarray] = None
    motion:  Optional[np.ndarray] = None
    r:       Optional[np.ndarray] = None   # NEW: per-point lifespan logit
```

And in `SharpTimeGaussianModel.create_from_pcd`:
```python
if pcd.r is not None:
    self._gaussian_params['r'] = nn.Parameter(
        torch.from_numpy(pcd.r).float().to(self.device), requires_grad=True
    )
# else: keep the existing _R_INIT_LOGIT default
```

**Decision:** Since both static and dynamic branches use the same `r = softplus⁻¹(1e-6)`
init per the paper, this extension is **not strictly needed** for the initial implementation.
The init script writes the final PLY directly, and `load_ply` reads it.

### 6.2 Intermediate: `FrameRecon`

Internal data class used by the init script (not persisted):

```python
@dataclass
class FrameRecon:
    frame_idx:  int
    timestamp:  float                 # seconds
    xyz:        np.ndarray            # [M, 3]  world coordinates
    rgb:        np.ndarray            # [M, 3]  uint8
    cam_ids:    np.ndarray            # [M]     which camera observed each point
    is_dynamic: bool                  # True for dynamic branch output
```

### 6.3 Intermediate: `VelocityEstimate`

```python
@dataclass
class VelocityEstimate:
    xyz:    np.ndarray   # [M, 3]  positions at frame t
    v:      np.ndarray   # [M, 3]  estimated velocity (world/s)
    t:      float        # timestamp of frame t
    valid:  np.ndarray   # [M]     bool, False if no KNN match found
```

---

## 7. Reading Mechanism

The init PLY is read by `SharpTimeGaussianModel.load_ply()` via the hook chain:

```
GaussianModel.load_ply()
  └─ reads x,y,z, opacity, scale_*, rot_*, f_dc_*, f_rest_*
  └─ calls _load_extra_ply_data(plydata, num_points)
       └─ FreeTimeGaussianModel._load_extra_ply_data()
            reads: t, t_scale, motion_x/y/z
            └─ SharpTimeGaussianModel._load_extra_ply_data()
                 reads: r
```

**No changes to `gaussian_model.py` are needed** provided the PLY has all fields.

Integration with `train.py` via `config.init_point_cloud_path`:

```yaml
data:
  init_point_cloud_path: /path/to/sharptime_init.ply
```

`setup_model()` in `builder.py` already checks this field and calls `load_ply` if set.

---

## 8. Script Design: `tools/sharptime_init.py`

### 8.1 CLI

```
python tools/sharptime_init.py \
    --source_path  /data/bar-release \
    --output_path  /data/bar-release/sharptime_init.ply \
    --resolution   4 \
    --flow_threshold 2.0 \
    --knn_k        1 \
    --sam2_model   sam2_hiera_large \
    --raft_model   raft-things.pth \
    --colmap_bin   colmap \
    --work_dir     /tmp/sharptime_init_work \
    --start_frame  0 \
    --end_frame    -1 \
    --sh_degree    3 \
    --skip_static     # skip static branch if camera params already exist
    --skip_flow       # skip flow/SAM2, use existing masks
    --skip_dynamic    # skip dynamic branch, static-only init
```

### 8.2 Submodule Structure

```
tools/
  sharptime_init.py            # entry point, orchestrates pipeline
  sharptime_init/
    __init__.py
    colmap_runner.py            # headless COLMAP wrapper (subprocess calls)
    optical_flow.py             # RAFT wrapper + thresholding
    segmentation.py             # SAM2 video predictor wrapper
    velocity_estimator.py       # KNN cross-frame matching
    pcd_builder.py              # per-frame PLY → FrameRecon
    ply_writer.py               # merge FrameRecons → final SharpTimeGS PLY
    dataset_adapter.py          # reads source_path (COLMAP/SelfCap format)
```

### 8.3 `colmap_runner.py` Key Functions

```python
def run_feature_extractor(db_path, image_path, single_camera=False, gpu=True) -> None:
    """Runs `colmap feature_extractor` as subprocess."""

def run_exhaustive_matcher(db_path, gpu=True) -> None:
    """Runs `colmap exhaustive_matcher`."""

def run_sequential_matcher(db_path, overlap=10, gpu=True) -> None:
    """Runs `colmap sequential_matcher` (for large N_frames)."""

def run_mapper(db_path, image_path, output_path) -> Path:
    """Runs `colmap mapper`, returns path to best model (0/)."""

def run_point_triangulator(db_path, image_path, input_path, output_path) -> None:
    """Runs `colmap point_triangulator` with known camera poses."""

def read_colmap_points3D(sparse_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads points3D.bin → (xyz [N,3], rgb [N,3])."""
```

---

## 9. Dependencies

| Library | Purpose | Install |
|---|---|---|
| `colmap` | SfM / triangulation | system package or conda |
| `pycolmap` | Python COLMAP bindings (optional, faster than subprocess) | `pip install pycolmap` |
| RAFT | Optical flow | `git clone .../RAFT; pip install -e .` |
| SAM2 | Video segmentation | `pip install sam2` |
| `sklearn` | KNN matching | already in env |
| `plyfile` | PLY I/O | already in env |

RAFT checkpoint: `raft-things.pth` from the official RAFT release.
SAM2 checkpoint: `sam2_hiera_large.pt` from Meta's SAM2 release.

---

## 10. Open Design Decisions

| Question | Options | Recommendation |
|---|---|---|
| Static branch COLMAP target | Frame 0 / all frames / first N frames | **Frame 0** per paper; covers background before any motion |
| Dynamic COLMAP per frame | `mapper` vs `point_triangulator` | **`point_triangulator`** — faster, uses known poses from static branch |
| KNN across frames | k=1 vs k>1 (average) | **k=1** per paper; more robust than average for fast motion |
| SAM2 prompt strategy | Flow magnitude peaks / grid / bbox | **Flow magnitude peaks** — N points where ‖flow‖ > threshold |
| `normalized_t` handling | Scale `t` and `σ_t` accordingly | Use dataset's `time_info` dict, same as FreeTimeGS |
| `r` init | Same for static and dynamic | **Yes** (paper: both init to `1e-6`); learned separately via gradient |
| Reuse `freetimegs_init.py` | Refactor vs new script | **New script** (`sharptime_init.py`); old script stays for FreeTimeGS compat |

---

## 11. Implementation Order

1. `colmap_runner.py` — subprocess wrapper + `read_colmap_points3D()`
2. Static branch end-to-end → verify PLY loads in SharpTimeGS
3. `optical_flow.py` (RAFT wrapper)
4. `segmentation.py` (SAM2 wrapper)  
5. `pcd_builder.py` — masked COLMAP triangulation per frame
6. `velocity_estimator.py` — KNN velocity
7. `ply_writer.py` — merge + write final PLY
8. `sharptime_init.py` — wire everything together
9. (Optional) `BasicPointCloud.r` field extension + `create_from_pcd` support
