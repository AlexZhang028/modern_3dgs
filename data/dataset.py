"""
Dataset Class - PyTorch Dataset Implementation

Supports COLMAP, NeRF Synthetic (Blender), and SelfCap/EasyVolcap dataset formats.
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import cv2

from .camera import Camera
from utils import PILtoTorch, fov2focal, focal2fov, BasicPointCloud


class GaussianDataset(Dataset):
    """
    Base class for Gaussian Splatting Datasets.
    Provides common functionality for caching, camera management etc.
    Concrete implementations should handle specific dataset formats.
    """
    
    def __init__(
        self,
        source_path: str,
        split: str = "train",
        resolution: int = -1,
        white_background: bool = False,
        images_folder: str = "images",
        depths_folder: str = "",
        eval_mode: bool = False,
        llffhold: int = 8,
        cache_device: Optional[str] = None,  # "cuda", "cpu" or None (no cache)
        train_test_exp: bool = False,
        init_point_cloud_path: str = "",  # Custom initial point cloud path
        start_frame: int = 0,
        end_frame: int = -1,
        test_camera_names: Optional[List[str]] = None,
        train_camera_names: Optional[List[str]] = None,
        normalized_t: bool = True,
    ):
        """
        Args:
            source_path: Dataset root directory
            split: "train" or "test"
            resolution: Target resolution (-1 for original)
            white_background: Use white background
            images_folder: Images folder name
            depths_folder: Depths folder name
            eval_mode: Evaluation mode (affects train/test split)
            llffhold: LLFF hold-out interval
            cache_device: Cache device (None, "cpu", "cuda")
            train_test_exp: Use train/test exposure compensation
            init_point_cloud_path: Custom initial point cloud path (absolute or relative)
            start_frame: Start frame index for video datasets
            end_frame: End frame index for video datasets (-1 for all)
            test_camera_names: List of camera names to exclude from training (used as test set)
            train_camera_names: List of camera names to use for training (overrides default split)
            normalized_t: Whether to use normalized time [0,1] or seconds.
        """
        self.source_path = source_path
        self.split = split
        self.resolution = resolution
        self.white_background = white_background
        self.images_folder = images_folder
        self.depths_folder = depths_folder
        self.eval_mode = eval_mode
        self.llffhold = llffhold
        self.cache_device = cache_device
        self.train_test_exp = train_test_exp
        self.init_point_cloud_path = init_point_cloud_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.test_camera_names = test_camera_names if test_camera_names is not None else []
        self.train_camera_names = train_camera_names if train_camera_names is not None else []
        self.normalized_t = normalized_t

        # Store camera info
        self.cameras: List[Camera] = []
        self.image_cache: Dict[int, torch.Tensor] = {}
        
        # Scene info
        self.scene_info: Dict = {}
        self.point_cloud: Optional[BasicPointCloud] = None
        
        # Detect and load dataset
        self._load_dataset()
    
    def _load_dataset(self):
        """Override this in subclasses."""
        pass
        
    def _compute_scene_normalization(self):
        """Compute scene normalization parameters."""
        if not self.cameras:
            return
            
        cam_centers = []
        for cam in self.cameras:
            # Compute camera center
            Rt = np.zeros((4, 4))
            Rt[:3, :3] = cam.R.T
            Rt[:3, 3] = cam.T
            Rt[3, 3] = 1.0
            C2W = np.linalg.inv(Rt)
            cam_centers.append(C2W[:3, 3])
        
        cam_centers = np.array(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=0)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=1)
        diagonal = np.max(dist)
        radius = diagonal * 1.1
        translate = -center
        
        self.scene_info = {
            "translate": translate,
            "radius": radius,
            "center": center,
        }
    
    def _load_initial_point_cloud(self):
        """Unified initial point cloud loading logic."""
        from .ply_utils import fetchPly, storePly
        from .colmap_loader import read_points3D_binary, read_points3D_text
        from utils.sh_utils import SH2RGB
        
        ply_path = None
        
        # 1. Custom path
        if self.init_point_cloud_path:
            # Check if absolute or relative
            if os.path.isabs(self.init_point_cloud_path):
                candidate_path = self.init_point_cloud_path
            else:
                candidate_path = os.path.join(self.source_path, self.init_point_cloud_path)
            
            if os.path.exists(candidate_path):
                ply_path = candidate_path
                print(f"✓ Using custom initial point cloud: {ply_path}")
            else:
                print(f"Warning: Custom point cloud not found at {candidate_path}")
        
        # 2. points3d.ply at root
        if ply_path is None:
            candidate_path = os.path.join(self.source_path, "points3d.ply")
            if os.path.exists(candidate_path):
                ply_path = candidate_path
                print(f"✓ Found point cloud at root: {ply_path}")
        
        # 3. COLMAP sparse/0/points3D.ply
        if ply_path is None:
            sparse_path = os.path.join(self.source_path, "sparse/0")
            if os.path.exists(sparse_path):
                colmap_ply = os.path.join(sparse_path, "points3D.ply")
                
                # Try converting if missing
                if not os.path.exists(colmap_ply):
                    try:
                        bin_path = os.path.join(sparse_path, "points3D.bin")
                        txt_path = os.path.join(sparse_path, "points3D.txt")
                        
                        print("Converting COLMAP points3D to .ply...")
                        try:
                            xyz, rgb, _ = read_points3D_binary(bin_path)
                        except:
                            xyz, rgb, _ = read_points3D_text(txt_path)
                        storePly(colmap_ply, xyz, rgb)
                        print(f"✓ Converted point cloud saved to {colmap_ply}")
                    except Exception as e:
                        print(f"Warning: Could not convert COLMAP point cloud: {e}")
                
                if os.path.exists(colmap_ply):
                    ply_path = colmap_ply
                    print(f"✓ Found COLMAP point cloud: {ply_path}")
        
        # 4. Load if found
        if ply_path:
            try:
                self.point_cloud = fetchPly(ply_path)
                self.scene_info["ply_path"] = ply_path
                print(f"✓ Loaded {len(self.point_cloud.points)} points from point cloud")
                return
            except Exception as e:
                print(f"Warning: Could not load point cloud from {ply_path}: {e}")
        
        # 5. Generate random points
        print("No initial point cloud found, generating random points...")
        num_pts = 100_000
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        rgb = SH2RGB(shs)
        self.point_cloud = BasicPointCloud(
            points=xyz, 
            colors=rgb, 
            normals=np.zeros((num_pts, 3))
        )
        
        # Save generated points
        save_path = os.path.join(self.source_path, "points3d_random.ply")
        try:
            storePly(save_path, xyz, rgb * 255)
            self.scene_info["ply_path"] = save_path
            print(f"✓ Generated {num_pts} random points, saved to {save_path}")
        except Exception:
            pass
    
    def __len__(self) -> int:
        return len(self.cameras)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a data sample.
        
        Returns:
            Dict with keys:
                - camera: Camera object
                - image: [3, H, W] tensor
                - alpha_mask: [1, H, W] tensor
                - depth_map: [1, H, W] tensor (optional)
                - depth_mask: [1, H, W] tensor (optional)
                - uid: int
        """
        camera = self.cameras[idx]
        
        # Check cache
        if self.cache_device and idx in self.image_cache:
            image_data = self.image_cache[idx]
            camera.image = image_data["image"]
            camera.alpha_mask = image_data["alpha_mask"]
            if "depth_map" in image_data:
                camera.depth_map = image_data["depth_map"]
                camera.depth_mask = image_data["depth_mask"]
                camera.depth_reliable = image_data["depth_reliable"]
        else:
            # Lazy load
            self._load_camera_image(camera)
            
            # Cache if enabled
            if self.cache_device:
                cache_data = {
                    "image": camera.image,
                    "alpha_mask": camera.alpha_mask,
                }
                if camera.depth_map is not None:
                    cache_data["depth_map"] = camera.depth_map
                    cache_data["depth_mask"] = camera.depth_mask
                    cache_data["depth_reliable"] = camera.depth_reliable
                self.image_cache[idx] = cache_data
        
        return {
            "camera": camera,
            "image": camera.image,
            "alpha_mask": camera.alpha_mask,
            "depth_map": camera.depth_map,
            "depth_mask": camera.depth_mask,
            "uid": camera.uid,
        }
    
    def _load_camera_image(self, camera: Camera):
        """Load camera image and depth map."""
        if not hasattr(camera, '_image_path'):
            raise ValueError(f"Camera {camera.uid} does not have _image_path set")
            
        # Load RGB Image
        image = Image.open(camera._image_path)
        
        # Handle resolution
        # If resolution is 1, 2, 4, 8, treat as downscale factor
        if self.resolution in [1, 2, 4, 8]:
            resolution = (image.size[0] // self.resolution, image.size[1] // self.resolution)
        elif self.resolution > 0:
            # If resolution is not a standard factor, treat as simply forcing square or specific size?
            # Creating a square resize is usually destructive for non-square images.
            # Assuming widely used downscale factor semantic.
            resolution = (image.size[0] // self.resolution, image.size[1] // self.resolution)
        else:
            resolution = image.size
        
        # Convert to tensor
        resized_image_rgb = PILtoTorch(image, resolution)
        
        # Handle alpha channel
        if resized_image_rgb.shape[0] == 4:
            camera.image = resized_image_rgb[:3, ...]
            camera.alpha_mask = resized_image_rgb[3:4, ...]
        else:
            camera.image = resized_image_rgb
            camera.alpha_mask = torch.ones(1, resized_image_rgb.shape[1], resized_image_rgb.shape[2])
        
        # Apply white/black background (for NeRF Synthetic)
        if resized_image_rgb.shape[0] == 4:
            bg = torch.ones(3, 1, 1) if self.white_background else torch.zeros(3, 1, 1)
            camera.image = camera.image * camera.alpha_mask + bg * (1 - camera.alpha_mask)
        
        # Clamp to [0, 1]
        camera.image = camera.image.clamp(0.0, 1.0)
        
        # Load depth map (if available)
        if hasattr(camera, '_depth_path') and camera._depth_path and os.path.exists(camera._depth_path):
            self._load_depth_map(camera, resolution)
    
    def _load_depth_map(self, camera: Camera, resolution: Tuple[int, int]):
        """Load depth map."""
        depth = cv2.imread(camera._depth_path, cv2.IMREAD_UNCHANGED)
        
        if depth is not None:
            # Resize
            depth = cv2.resize(depth, resolution)
            depth = depth.astype(np.float32)
            depth[depth < 0] = 0
            
            # Apply depth parameters
            if camera._depth_params:
                scale = camera._depth_params.get("scale", 1.0)
                offset = camera._depth_params.get("offset", 0.0)
                med_scale = camera._depth_params.get("med_scale", 1.0)
                
                # Check reliability
                if scale < 0.2 * med_scale or scale > 5 * med_scale:
                    camera.depth_reliable = False
                    camera.depth_mask = torch.zeros(1, resolution[1], resolution[0])
                else:
                    camera.depth_reliable = True
                    camera.depth_mask = torch.ones(1, resolution[1], resolution[0])
                
                if scale > 0:
                    depth = depth * scale + offset
            else:
                camera.depth_reliable = True
                camera.depth_mask = torch.ones(1, resolution[1], resolution[0])
            
            # Convert to tensor
            if depth.ndim > 2:
                depth = depth[..., 0]
            camera.depth_map = torch.from_numpy(depth[None]).float()
        else:
            camera.depth_map = None
            camera.depth_mask = None
            camera.depth_reliable = False
    
    def get_cameras_extent(self) -> float:
        """Get scene extent (for densification)."""
        return self.scene_info.get("radius", 1.0)


class StaticGaussianDataset(GaussianDataset):
    """
    Dataset for Static Scenes (COLMAP, NeRF Synthetic).
    """
    def _load_dataset(self):
        """Load dataset based on directory structure."""
        if os.path.exists(os.path.join(self.source_path, "sparse")):
            print(f"Detected COLMAP dataset at {self.source_path}")
            self._load_colmap_dataset()
        elif os.path.exists(os.path.join(self.source_path, "transforms_train.json")):
            print(f"Detected NeRF Synthetic dataset at {self.source_path}")
            self._load_nerf_synthetic_dataset()
        else:
            raise ValueError(
                f"Could not recognize dataset type at {self.source_path}. "
                "Expected COLMAP (sparse/) or NeRF Synthetic (transforms_train.json)"
            )
    
    def _load_colmap_dataset(self):
        """Load COLMAP dataset."""
        from .colmap_loader import (
            read_extrinsics_binary, read_intrinsics_binary,
            read_extrinsics_text, read_intrinsics_text,
            read_points3D_binary, read_points3D_text,
            qvec2rotmat
        )
        
        sparse_path = os.path.join(self.source_path, "sparse/0")
        
        # Read camera parameters
        try:
            cam_extrinsics = read_extrinsics_binary(os.path.join(sparse_path, "images.bin"))
            cam_intrinsics = read_intrinsics_binary(os.path.join(sparse_path, "cameras.bin"))
        except:
            cam_extrinsics = read_extrinsics_text(os.path.join(sparse_path, "images.txt"))
            cam_intrinsics = read_intrinsics_text(os.path.join(sparse_path, "cameras.txt"))
        
        # Read depth parameters (if available)
        depths_params = None
        if self.depths_folder:
            depth_params_file = os.path.join(sparse_path, "depth_params.json")
            if os.path.exists(depth_params_file):
                with open(depth_params_file, 'r') as f:
                    depths_params = json.load(f)
                # Calculate median scale
                all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
                med_scale = np.median(all_scales[all_scales > 0]) if (all_scales > 0).sum() else 0
                for key in depths_params:
                    depths_params[key]["med_scale"] = med_scale
        
        # Determine test set
        test_cam_names = self.test_camera_names
        if not test_cam_names and self.eval_mode:
            if self.llffhold > 0:
                cam_names = sorted([cam_extrinsics[cam_id].name for cam_id in cam_extrinsics])
                test_cam_names = [name for idx, name in enumerate(cam_names) if idx % self.llffhold == 0]
            else:
                test_file = os.path.join(sparse_path, "test.txt")
                if os.path.exists(test_file):
                    with open(test_file, 'r') as f:
                        test_cam_names = [line.strip() for line in f]
        
        # Load camera info
        images_path = os.path.join(self.source_path, self.images_folder)
        depths_path = os.path.join(self.source_path, self.depths_folder) if self.depths_folder else ""
        
        all_cameras = []
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            
            # Compute FOV
            if intr.model == "SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, intr.height)
                FovX = focal2fov(focal_length_x, intr.width)
            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, intr.height)
                FovX = focal2fov(focal_length_x, intr.width)
            else:
                raise ValueError(f"Unsupported camera model: {intr.model}")
            
            image_name = extr.name
            image_path = os.path.join(images_path, image_name)
            
            # Depth map path
            depth_path = None
            depth_params = None
            if depths_params and depths_path:
                stem = '.'.join(image_name.split('.')[:-1])
                if stem in depths_params:
                    depth_params = depths_params[stem]
                    depth_path = os.path.join(depths_path, f"{stem}.png")
            
            is_test = image_name in test_cam_names
            is_train = image_name in self.train_camera_names if self.train_camera_names else (not is_test)
            
            # Filter by split
            if self.split == "train":
                if not is_train and not self.train_test_exp:
                    continue
            if self.split == "test" and not is_test:
                continue
            
            # Create Camera object (Images deferred)
            camera = Camera(
                uid=len(all_cameras),
                image_name=image_name,
                R=R,
                T=T,
                FovX=FovX,
                FovY=FovY,
                width=intr.width,
                height=intr.height,
                colmap_id=intr.id,
            )
            
            # Store extra info for lazy loading
            camera._image_path = image_path
            camera._depth_path = depth_path
            camera._depth_params = depth_params
            camera._is_test = is_test
            
            all_cameras.append(camera)
        
        self.cameras = all_cameras
        
        # Compute scene normalization
        self._compute_scene_normalization()
        
        # Load initial point cloud
        self._load_initial_point_cloud()
        
        print(f"Loaded {len(self.cameras)} cameras from COLMAP dataset")

    def _load_nerf_synthetic_dataset(self):
        """Load NeRF Synthetic dataset."""
        transform_file = f"transforms_{self.split}.json"
        transform_path = os.path.join(self.source_path, transform_file)
        
        if not os.path.exists(transform_path):
            print(f"Warning: {transform_file} not found, using empty dataset")
            return
        
        with open(transform_path, 'r') as f:
            transforms = json.load(f)
        
        fovx = transforms["camera_angle_x"]
        frames = transforms["frames"]
        
        all_cameras = []
        for idx, frame in enumerate(frames):
            # Camera to World transform
            c2w = np.array(frame["transform_matrix"])
            # Change coordinate system: OpenGL/Blender (Y up, Z back) -> OpenCV/COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            
            # World to Camera transform
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            
            # Image path
            file_path = frame["file_path"]
            if not file_path.endswith(".png"):
                file_path += ".png"
            image_path = os.path.join(self.source_path, file_path)
            image_name = Path(file_path).stem
            
            # Filter override
            if self.split == 'train' and self.train_camera_names:
                 if image_name not in self.train_camera_names:
                     continue
            if self.split == 'test' and self.test_camera_names:
                 if image_name not in self.test_camera_names:
                     continue
            
            # Read image size (need to read once for size, but don't cache pixels)
            image = Image.open(image_path)
            width, height = image.size
            
            # Compute FovY
            fovy = focal2fov(fov2focal(fovx, width), height)
            
            # Depth map path
            depth_path = None
            if self.depths_folder:
                depth_path = os.path.join(self.source_path, self.depths_folder, f"{image_name}.png")
                if not os.path.exists(depth_path):
                    depth_path = None
            
            camera = Camera(
                uid=idx,
                image_name=image_name,
                R=R,
                T=T,
                FovX=fovx,
                FovY=fovy,
                width=width,
                height=height,
            )
            
            camera._image_path = image_path
            camera._depth_path = depth_path
            camera._depth_params = None
            camera._is_test = (self.split == "test")
            
            all_cameras.append(camera)
        
        self.cameras = all_cameras
        
        # Compute scene normalization
        self._compute_scene_normalization()
        
        # Load or generate point cloud (Unified)
        self._load_initial_point_cloud()
        
        print(f"Loaded {len(self.cameras)} cameras from NeRF Synthetic dataset")


class SelfCapVideoDataset(GaussianDataset):
    """
    Dataset for SelfCap/EasyVolcap video-based datasets.
    Reads frames directly from video files without extraction.
    """
    
    def _load_dataset(self):
        print(f"Detected SelfCap Video dataset at {self.source_path}")
        self._load_selfcap_video_dataset()

    def _load_selfcap_video_dataset(self):
        from .selfcap_loader import read_selfcap_cameras
        
        # Load params (supports optimized/ folder)
        cam_names, extrinsics, intrinsics = read_selfcap_cameras(self.source_path)
        
        # Check for videos folder
        videos_root = os.path.join(self.source_path, "videos")
        if not os.path.exists(videos_root):
             # Maybe they are in root
             videos_root = self.source_path
        
        print(f"Loading SelfCap Video dataset from {videos_root}...")
        
        video_files = os.listdir(videos_root)
        video_map = {} # name -> path
        
        for v_file in video_files:
            if v_file.lower().endswith('.mp4'):
                stem = os.path.splitext(v_file)[0]
                video_map[stem] = os.path.join(videos_root, v_file)
        
        # Determine total frames from the first available video
        if not video_map:
            raise ValueError(f"No .mp4 video files found in {videos_root}")
            
        first_video = next(iter(video_map.values()))
        cap = cv2.VideoCapture(first_video)
        
        # Read FPS and Frames
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: 
            print("Warning: Could not read FPS, defaulting to 30.0")
            fps = 30.0
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        start = self.start_frame
        end = self.end_frame
        if end == -1 or end > total_frames:
            end = total_frames
        
        selected_indices = list(range(start, end))
        subset_frame_count = len(selected_indices)
        duration_seconds = subset_frame_count / fps
        
        print(f"Video Info: {width}x{height}, {total_frames} frames (FPS: {fps:.2f}).")
        print(f"Dataset Range: Frames {start}-{end} ({subset_frame_count} frames, {duration_seconds:.2f}s)")

        all_cameras = []
        for name in cam_names:
            is_test = name in self.test_camera_names
            # If train_camera_names is set, strictly follow it. Otherwise, use all non-test cameras.
            is_train = name in self.train_camera_names if self.train_camera_names else (not is_test)
            
            if self.split == "train" and not is_train:
                continue
            if self.split == "test" and not is_test:
                continue
            
            if name not in extrinsics:
                continue

            # Try to find video
            video_path = video_map.get(name)
            if not video_path:
                if name.isdigit() and len(name) == 2 and name[0] != '0':
                     pass
                if not video_path:
                    continue

            extr = extrinsics[name]
            # Stores World-to-Camera matrix (OpenCV)
            # Camera class expects Camera-to-World rotation (equivalent to Transposed W2C)
            R = extr['R'].transpose()
            T = extr['T']
            
            intr = intrinsics.get(name)
            # Default intrinsics
            curr_width = width
            curr_height = height
            
            FovX = 1.0
            FovY = 1.0
            if intr and 'K' in intr:
                 K = intr['K']
                 fx = K[0,0]
                 fy = K[1,1]
                 FovX = focal2fov(fx, curr_width)
                 FovY = focal2fov(fy, curr_height)

            for idx in selected_indices:
                image_name = f"{name}_frame_{idx:05d}"
                
                # Time handling
                # Normalization: [0, 1] mapped to [start_frame, end_frame]
                if subset_frame_count > 1:
                    time_norm = (idx - start) / (subset_frame_count - 1)
                else:
                    time_norm = 0.0
                
                # Valid time in seconds (relative to start_frame)
                time_sec = (idx - start) / fps

                # Calculate downscaled dimensions used for Camera object
                final_width = curr_width
                final_height = curr_height
                
                if self.resolution in [1, 2, 4, 8]:
                     final_width = curr_width // self.resolution
                     final_height = curr_height // self.resolution
                elif self.resolution > 0:
                     final_width = self.resolution
                     final_height = self.resolution

                # Select timestamp based on normalized_t flag
                selected_timestamp = time_norm if self.normalized_t else time_sec

                camera = Camera(
                    uid=len(all_cameras),
                    image_name=image_name,
                    R=R,
                    T=T,
                    FovX=FovX,
                    FovY=FovY,
                    width=final_width,
                    height=final_height,
                    timestamp=selected_timestamp,
                    timestamp_seconds=time_sec
                )
                
                # Store video info
                camera._video_path = video_path
                camera._frame_idx = idx
                
                all_cameras.append(camera)

        self.cameras = all_cameras
        self._compute_scene_normalization()
        self._load_initial_point_cloud()
        print(f"Loaded {len(self.cameras)} frames from SelfCap Video dataset")
        
        # Preload Video Frames optimization
        # If cache_images is True, we sequentially read the video now to avoid Open+Seek overhead later.
        # This transforms random access (slow) into sequential access (fast).
        if self.cache_device:
            print("Preloading video frames into memory (Sequential Read Optimization)...")
            self._preload_video_frames()

    def _preload_video_frames(self):
        """Preload all frames sequentially to avoid random seek overhead."""
        # 1. Group cameras by video file
        video_groups = {}
        for cam in self.cameras:
            if hasattr(cam, '_video_path'):
                if cam._video_path not in video_groups:
                    video_groups[cam._video_path] = []
                video_groups[cam._video_path].append(cam)
        
        # 2. Process each video
        for video_path, cams in video_groups.items():
            # Sort cams by frame index
            cams.sort(key=lambda c: c._frame_idx)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not preload {video_path}")
                continue
                
            current_frame = 0
            loaded_count = 0
            
            # Optimization: Only read frames we need. 
            # If dataset is sparse (e.g. frame 0, 10, 20), skip frames instead of decoding.
            # But cap.grab() is faster than read().
            
            # Get max frame needed
            max_frame = cams[-1]._frame_idx
            cam_indices = {c._frame_idx: i for i, c in enumerate(cams)}
            
            from tqdm import tqdm
            pbar = tqdm(total=max_frame+1, desc=f"Loading {os.path.basename(video_path)}", leave=False)
            
            while current_frame <= max_frame:
                if current_frame in cam_indices:
                     ret, frame = cap.read()
                     if not ret: 
                         break
                     
                     # Process frame
                     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                     
                     if self.resolution in [1, 2, 4, 8]:
                        new_w = frame.shape[1] // self.resolution
                        new_h = frame.shape[0] // self.resolution
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                     elif self.resolution > 0:
                        frame = cv2.resize(frame, (self.resolution, self.resolution))
                        
                     image_tensor = torch.from_numpy(frame).float() / 255.0
                     image_tensor = image_tensor.permute(2, 0, 1)
                     
                     # Check if we should move to GPU immediately?
                     # data_device usually handles this. Here we store in RAM or cache_device
                     # If cache_device is 'cuda', we move it.
                     if self.cache_device == "cuda":
                         image_tensor = image_tensor.cuda()
                     
                     # Assign to all cameras sharing this frame (unlikely but possible)
                     # In our list, multiple cams might point to same frame? 
                     # The grouping logic above put cams in a list. 
                     # Simply find the cam(s) with this index.
                     # Since we sorted, we can just look.
                     # Actually, multiple cams implies multiple views, but here video_path is unique per view? 
                     # Usually SelfCap is monocular video or multi-view videos. 
                     # In our loader, 'name' mapped to 'video_path'. So unique per camera.
                     
                     # Find camera object in our list
                     # We might have duplicates if we have multiple cameras pointing to SAME video/frame 
                     # (not typical for this dataset structure).
                     
                     # Find matches efficiently
                     # Iterate original filtered cams list
                     for c in cams:
                         if c._frame_idx == current_frame:
                             c.image = image_tensor
                             c.alpha_mask = torch.ones(1, image_tensor.shape[1], image_tensor.shape[2], device=image_tensor.device)
                             
                             # Manually populate cache dict so __getitem__ sees it
                             # Note: caching in `self.image_cache` uses dataset index as key.
                             # We need to look up the dataset index (uid is usually the index in self.cameras)
                             self.image_cache[c.uid] = {
                                 "image": c.image,
                                 "alpha_mask": c.alpha_mask
                             }
                             loaded_count += 1
                else:
                    # Skip frame
                    cap.grab()
                
                current_frame += 1
                pbar.update(1)
            
            pbar.close()
            cap.release()
            print(f"   Preloaded {loaded_count} frames for {os.path.basename(video_path)}")

    def _load_camera_image(self, camera: Camera):
        """Load specific frame from video."""
        if not hasattr(camera, '_video_path'):
             # Fallback to standard if we mixed types
             return super()._load_camera_image(camera)

        cap = cv2.VideoCapture(camera._video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, camera._frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Error reading frame {camera._frame_idx} from {camera._video_path}")
            # return black image?
            frame = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
        else:
             # BGR -> RGB
             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
             
        # Handle resolution (dataset arg)
        # If resolution is small integer (1, 2, 4, 8...), treat as downscale factor
        if self.resolution in [1, 2, 4, 8]:
            new_w = frame.shape[1] // self.resolution
            new_h = frame.shape[0] // self.resolution
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        elif self.resolution > 0:
             # Force square/specific size (legacy behavior)
             frame = cv2.resize(frame, (self.resolution, self.resolution))
             
        # To Tensor
        image_tensor = torch.from_numpy(frame).float() / 255.0 # [H, W, 3]
        image_tensor = image_tensor.permute(2, 0, 1) # [3, H, W]
        
        camera.image = image_tensor
        camera.alpha_mask = torch.ones(1, image_tensor.shape[1], image_tensor.shape[2])


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function.
    
    As 3DGS renders single view, no batching is done. Returns the first element.
    """
    return batch[0]
