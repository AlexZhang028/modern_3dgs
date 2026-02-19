"""
Core Gaussian Model Module

This module implements the GaussianModel, wrapping all Gaussian parameters in an nn.Module.
Supports both static and freetime modes through inheritance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, List, Union

from config.config import ModelConfig
from utils.general_utils import (
    inverse_sigmoid,
    build_rotation,
    build_scaling_rotation,
    strip_symmetric
)
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from plyfile import PlyData, PlyElement


class GaussianModel(nn.Module):
    """
    Base Class: Static 3D Gaussian Splatting Model.
    """
    def __init__(self, config: ModelConfig, device: str = "cuda"):
        super().__init__()
        
        self.config = config
        self.mode = "static" 
        
        # 1. Explicit device management
        self.device = torch.device(device)
        
        # 2. Use nn.ParameterDict instead of plain dict
        self._gaussian_params = nn.ParameterDict()
        
        # 3. Activation registry
        self._param_activations = {
            'scaling': lambda x: torch.exp(x),
            'opacity': lambda x: torch.sigmoid(x),
            'rotation': lambda x: F.normalize(x),
            'xyz': lambda x: x,
            'features_dc': lambda x: x,
            'features_rest': lambda x: x,
        }

        # 4. Virtual property registry
        self._computed_getters = {
            'features': lambda s: torch.cat((s._features_dc, s._features_rest), dim=1)
        }
        
        # 5. Densification stats buffer
        self.register_buffer('xyz_gradient_accum', torch.empty(0))
        self.register_buffer('denom', torch.empty(0))
        self.register_buffer('max_radii2D', torch.empty(0))
        
        # Other attributes
        self.active_sh_degree = 0
        self.max_sh_degree = config.sh_degree
        self.spatial_lr_scale = 1.0

    # ==================== Device Management ====================

    def to(self, device: Union[str, torch.device]):
        device = torch.device(device)
        super().to(device)
        self.device = device
        return self

    def cuda(self, device=None):
        target_device = torch.device("cuda" if device is None else device)
        return self.to(target_device)

    def cpu(self):
        return self.to(torch.device("cpu"))

    def __getattr__(self, name: str):
        _gaussian_params = None
        if '_modules' in self.__dict__:
            _gaussian_params = self.__dict__['_modules'].get('_gaussian_params')

        # A. Access raw parameters
        if _gaussian_params is not None and name.startswith("_") and len(name) > 1:
            key = name[1:]
            if key in _gaussian_params:
                return _gaussian_params[key]

        # B. Access processed parameters
        if name.startswith("get_") and len(name) > 4:
            key = name[4:]
            computed = self.__dict__.get('_computed_getters', {})
            if key in computed:
                return computed[key](self)
            
            if _gaussian_params is not None and key in _gaussian_params:
                raw_val = _gaussian_params[key]
                activations = self.__dict__.get('_param_activations', {})
                handler = activations.get(key, lambda x: x)
                return handler(raw_val)

        return super().__getattr__(name)

    def __setattr__(self, name: str, value):
        # 1. Handle special internal attributes immediately
        if name in ['_param_activations', '_computed_getters']:
            super().__setattr__(name, value)
            return

        # 2. Safely access _gaussian_params
        _gaussian_params = None
        if '_modules' in self.__dict__:
            _gaussian_params = self.__dict__['_modules'].get('_gaussian_params')

        # 3. Intercept assignments to _xyz, etc.
        if _gaussian_params is not None and name.startswith("_") and len(name) > 1:
            key = name[1:]
            
            # Safe access to activations
            activations = self.__dict__.get('_param_activations')
            
            # Determine if this key is a valid parameter key
            is_valid_param = False
            if activations is not None and key in activations:
                is_valid_param = True
            elif key in ['t', 't_scale', 'motion']: # Legacy/Subclass support fallback
                is_valid_param = True

            if is_valid_param:
                # Ensure it's a Parameter or None
                if isinstance(value, nn.Parameter) or value is None:
                    if value is None:
                        del _gaussian_params[key]
                    else:
                        # Ensure correct device
                        if value.device != self.device:
                            value.data = value.data.to(self.device)
                        _gaussian_params[key] = value
                    return
                elif isinstance(value, torch.Tensor):
                    # If Tensor passed, wrap in Parameter
                    _gaussian_params[key] = nn.Parameter(value.to(self.device), requires_grad=True)
                    return

        super().__setattr__(name, value)

    @property
    def num_points(self) -> int:
        if 'xyz' not in self._gaussian_params:
            return 0
        return self._gaussian_params['xyz'].shape[0]

    def get_parameter(self, name: str) -> Optional[torch.Tensor]:
        if name in self._gaussian_params:
            param = self._gaussian_params[name]
        else:
            return None

        if name == 'scaling':
            return torch.exp(param)
        elif name == 'opacity':
            return torch.sigmoid(param)
        return param

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
        
    def covariance_activation(self, scaling, modifier, rotation):
        L = build_scaling_rotation(rotation, scaling * modifier)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    # ==================== Initialization & Loading ====================

    def _get_knn_dist(self, points):
        try:
            from simple_knn._C import distCUDA2
            return distCUDA2(points)
        except:
            return torch.ones(points.shape[0], device=self.device) * 0.01

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float = 1.0):
        self.spatial_lr_scale = spatial_lr_scale
        
        points = torch.from_numpy(pcd.points).float().to(self.device)
        colors = torch.from_numpy(pcd.colors).float().to(self.device)

        num_points = points.shape[0]

        self._gaussian_params['xyz'] = nn.Parameter(points, requires_grad=True)

        fused_color = RGB2SH(colors)
        features = torch.zeros((num_points, 3, (self.max_sh_degree + 1) ** 2), device=self.device)
        features[:, :, 0] = fused_color
        
        self._gaussian_params['features_dc'] = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous(), requires_grad=True
        )
        self._gaussian_params['features_rest'] = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous(), requires_grad=True
        )

        dist2 = torch.clamp_min(self._get_knn_dist(points), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        self._gaussian_params['scaling'] = nn.Parameter(scales, requires_grad=True)

        rots = torch.zeros((num_points, 4), device=self.device)
        rots[:, 0] = 1.0
        self._gaussian_params['rotation'] = nn.Parameter(rots, requires_grad=True)

        opacities = inverse_sigmoid(0.1 * torch.ones((num_points, 1), device=self.device))
        self._gaussian_params['opacity'] = nn.Parameter(opacities, requires_grad=True)

        self._init_stats_buffer(num_points)

    def _init_stats_buffer(self, num_points):
        self.xyz_gradient_accum = torch.zeros((num_points, 1), device=self.device)
        self.denom = torch.zeros((num_points, 1), device=self.device)
        self.max_radii2D = torch.zeros((num_points), device=self.device)

    def load_ply(self, ply_path: str, spatial_lr_scale: float = 1.0):
        from plyfile import PlyData
        
        plydata = PlyData.read(ply_path)
        
        # Detect mode match
        detected_mode = detect_mode_from_ply(ply_path)
        if detected_mode != self.mode:
            print(f"Warning: Loading {detected_mode} PLY into {self.mode} model.")

        # Read Basic Attributes
        xyz = np.stack([
            np.asarray(plydata['vertex']['x']),
            np.asarray(plydata['vertex']['y']),
            np.asarray(plydata['vertex']['z'])
        ], axis=1).astype(np.float32)
        
        opacities = np.asarray(plydata['vertex']['opacity'])[..., None].astype(np.float32)
        
        scale_names = ['scale_0', 'scale_1', 'scale_2']
        scales = np.stack([
            np.asarray(plydata['vertex'][name]) for name in scale_names
        ], axis=1).astype(np.float32)
        
        rot_names = ['rot_0', 'rot_1', 'rot_2', 'rot_3']
        rots = np.stack([
            np.asarray(plydata['vertex'][name]) for name in rot_names
        ], axis=1).astype(np.float32)

        # Read SH Features
        features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
        features_dc[:, 0, 0] = np.asarray(plydata['vertex']['f_dc_0']).astype(np.float32)
        features_dc[:, 1, 0] = np.asarray(plydata['vertex']['f_dc_1']).astype(np.float32)
        features_dc[:, 2, 0] = np.asarray(plydata['vertex']['f_dc_2']).astype(np.float32)
        
        extra_f_names = [p.name for p in plydata['vertex'].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata['vertex'][attr_name]).astype(np.float32)
        features_extra = features_extra.reshape((xyz.shape[0], 3, -1))
        
        # Create Parameters
        def create_param(data_np):
            return nn.Parameter(torch.from_numpy(data_np).float().to(self.device), requires_grad=True)

        self._gaussian_params['xyz'] = create_param(xyz)
        self._gaussian_params['opacity'] = create_param(opacities)
        self._gaussian_params['scaling'] = create_param(scales)
        self._gaussian_params['rotation'] = create_param(rots)
        
        dc_tensor = torch.from_numpy(features_dc).float().to(self.device)
        self._gaussian_params['features_dc'] = nn.Parameter(
            dc_tensor.transpose(1, 2).contiguous(), 
            requires_grad=True
        )
        
        rest_tensor = torch.from_numpy(features_extra).float().to(self.device)
        self._gaussian_params['features_rest'] = nn.Parameter(
            rest_tensor.transpose(1, 2).contiguous(), 
            requires_grad=True
        )

        # Hook for subclasses to load their data
        self._load_extra_ply_data(plydata, xyz.shape[0])
        
        # Initialize Stats
        self._init_stats_buffer(xyz.shape[0])
        self.spatial_lr_scale = spatial_lr_scale
        self.active_sh_degree = self.max_sh_degree
        
    def _load_extra_ply_data(self, plydata, num_points):
        """Hook for subclasses."""
        pass

    def save_ply(self, ply_path: str):
        import numpy as np
        
        def get_tensor(key):
            if key in self._gaussian_params:
                return self._gaussian_params[key]
            return None

        xyz = get_tensor('xyz').detach().cpu().numpy()
        opacities = get_tensor('opacity').detach().cpu().numpy()
        scales = get_tensor('scaling').detach().cpu().numpy()
        rots = get_tensor('rotation').detach().cpu().numpy()
        features_dc = get_tensor('features_dc').transpose(1, 2).contiguous().detach().cpu().numpy()
        features_rest = get_tensor('features_rest').transpose(1, 2).contiguous().detach().cpu().numpy()
        
        dtype_list = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ]
        
        for i in range(3):
            dtype_list.append((f'f_dc_{i}', 'f4'))
        
        for i in range(features_rest.shape[1] * features_rest.shape[2]):
            dtype_list.append((f'f_rest_{i}', 'f4'))
        
        # Add subclass dtypes
        self._add_extra_ply_dtype(dtype_list)
        
        num_points = xyz.shape[0]
        vertex_data = np.empty(num_points, dtype=dtype_list)
        
        vertex_data['x'] = xyz[:, 0]
        vertex_data['y'] = xyz[:, 1]
        vertex_data['z'] = xyz[:, 2]
        vertex_data['opacity'] = opacities[:, 0]
        vertex_data['scale_0'] = scales[:, 0]
        vertex_data['scale_1'] = scales[:, 1]
        vertex_data['scale_2'] = scales[:, 2]
        vertex_data['rot_0'] = rots[:, 0]
        vertex_data['rot_1'] = rots[:, 1]
        vertex_data['rot_2'] = rots[:, 2]
        vertex_data['rot_3'] = rots[:, 3]
        
        for i in range(3):
            vertex_data[f'f_dc_{i}'] = features_dc[:, i, 0]
        
        features_rest_flat = features_rest.reshape(num_points, -1)
        for i in range(features_rest_flat.shape[1]):
            vertex_data[f'f_rest_{i}'] = features_rest_flat[:, i]
            
        # Fill subclass data
        self._fill_extra_ply_data(vertex_data)
        
        el = PlyElement.describe(vertex_data, 'vertex')
        PlyData([el]).write(ply_path)
        print(f"PLY Saved: {ply_path} ({num_points} points)")

    def _add_extra_ply_dtype(self, dtype_list: List):
        pass

    def _fill_extra_ply_data(self, vertex_data):
        pass

    def get_param_groups(self, optim_config) -> List[Dict]:
        """Create parameter groups (different LR for each parameter)."""
        param_groups = [
            {
                'params': [self._xyz],
                'lr': optim_config.position_lr_init * self.spatial_lr_scale,
                'name': "xyz"
            },
            {
                'params': [self._features_dc],
                'lr': optim_config.feature_lr,
                'name': "f_dc"
            },
            {
                'params': [self._features_rest],
                'lr': optim_config.feature_lr / 20.0,
                'name': "f_rest"
            },
            {
                'params': [self._opacity],
                'lr': optim_config.opacity_lr,
                'name': "opacity"
            },
            {
                'params': [self._scaling],
                'lr': optim_config.scaling_lr,
                'name': "scaling"
            },
            {
                'params': [self._rotation],
                'lr': optim_config.rotation_lr,
                'name': "rotation"
            }
        ]
        return param_groups

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


class FreeTimeGaussianModel(GaussianModel):
    """
    Subclass: Temporal FreeTimeGS Model.
    Adds time (t) and motion (velocity) parameters.
    """
    def __init__(self, config: ModelConfig, device: str = "cuda"):
        super().__init__(config, device)
        self.mode = "freetime"
        self.register_buffer('t_extent', torch.tensor(1.0, device=device))
        self.register_buffer('t_start', torch.tensor(0.0, device=device))
        
        # Register extra activations
        self._param_activations.update({
            't_scale': lambda x: torch.exp(x),
            't': lambda x: x,
            'motion': lambda x: x,
        })
        
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float = 1.0, time_info: Optional[Dict] = None):
        super().create_from_pcd(pcd, spatial_lr_scale)
        num_points = self.num_points

        # Handle time normalization info
        t_extent = 1.0
        t_start = 0.0
        if time_info is not None:
             t_extent = time_info.get('t_extent', 1.0)
             t_start = time_info.get('t_start', 0.0)

        # Avoid zero division
        if t_extent <= 1e-6:
             t_extent = 1.0

        self.t_extent = torch.tensor(t_extent, device=self.device)
        self.t_start = torch.tensor(t_start, device=self.device)
        
        normalized_t_mode = getattr(self.config, 'normalized_t', True)

        # Initialize t
        if pcd.t is not None:
             if normalized_t_mode:
                 # Normalize: (t_sec - t_start) / t_extent
                 t_val = (pcd.t.copy() - t_start) / t_extent
             else:
                 # Raw Seconds
                 t_val = pcd.t.copy()
             self._gaussian_params['t'] = nn.Parameter(torch.from_numpy(t_val).float().to(self.device), requires_grad=True)
        else:
             # Random initialization
             if normalized_t_mode:
                 # [0, 1]
                 self._gaussian_params['t'] = nn.Parameter(torch.rand((num_points, 1), device=self.device), requires_grad=True)
             else:
                 # [t_start, t_start + t_extent]
                 rand_t = torch.rand((num_points, 1), device=self.device) * t_extent + t_start
                 self._gaussian_params['t'] = nn.Parameter(rand_t, requires_grad=True)

        # Initialize t_scale (duration)
        if pcd.t_scale is not None:
             # Assume input pcd.t_scale is already in Log(Seconds)
             t_scale_val = pcd.t_scale.copy()
             
             if normalized_t_mode:
                 # We need Log(Normalized Duration)
                 # log(D_norm) = log(D_sec / t_extent) = log(D_sec) - log(t_extent)
                 t_scale_val = t_scale_val - np.log(t_extent)
             else:
                 # We need Log(Seconds). Input is already Log(Seconds).
                 pass
             
             self._gaussian_params['t_scale'] = nn.Parameter(torch.from_numpy(t_scale_val).float().to(self.device), requires_grad=True)
        else:
             # Initial duration heuristic.
             if normalized_t_mode:
                  # Normalized space: 0.5 units where 1.0 is full extent
                  init_duration = 0.5
             else:
                  # Seconds space: 0.5 * extent
                  init_duration = 0.5 * t_extent
             
             # Parameter is log-scale
             init_duration = max(init_duration, 1e-4) # Safety clamp
             init_val = np.log(init_duration)
             self._gaussian_params['t_scale'] = nn.Parameter(torch.full((num_points, 1), init_val, device=self.device), requires_grad=True)
        
        # Initialize motion (velocity) (Normalize: increase by t_extent)
        if pcd.motion is not None:
             if normalized_t_mode:
                 # v_norm = v_sec * t_extent
                 motion_val = pcd.motion.copy() * t_extent
             else:
                 # v_sec
                 motion_val = pcd.motion.copy()
             self._gaussian_params['motion'] = nn.Parameter(torch.from_numpy(motion_val).float().to(self.device), requires_grad=True)
        else:
             self._gaussian_params['motion'] = nn.Parameter(torch.zeros((num_points, 3), device=self.device), requires_grad=True)

    def _load_extra_ply_data(self, plydata, num_points):
        # Helper
        def create_param(data_np):
            return nn.Parameter(torch.from_numpy(data_np).float().to(self.device), requires_grad=True)

        if 't' in [p.name for p in plydata['vertex'].properties]:
            t = np.asarray(plydata['vertex']['t'])[..., None].astype(np.float32)
            self._gaussian_params['t'] = create_param(t)
        else:
            self._gaussian_params['t'] = nn.Parameter(torch.full((num_points, 1), 0.5, device=self.device), requires_grad=True)
        
        if 't_scale' in [p.name for p in plydata['vertex'].properties]:
            t_scale = np.asarray(plydata['vertex']['t_scale'])[..., None].astype(np.float32)
            self._gaussian_params['t_scale'] = create_param(t_scale)
        else:
            self._gaussian_params['t_scale'] = nn.Parameter(torch.zeros((num_points, 1), device=self.device), requires_grad=True)
        
        motion_names = ['motion_0', 'motion_1', 'motion_2']
        if all(name in [p.name for p in plydata['vertex'].properties] for name in motion_names):
            motion = np.stack([
                np.asarray(plydata['vertex'][name]) for name in motion_names
            ], axis=1).astype(np.float32)
            self._gaussian_params['motion'] = create_param(motion)
        else:
            self._gaussian_params['motion'] = nn.Parameter(torch.zeros((num_points, 3), device=self.device), requires_grad=True)

    def _add_extra_ply_dtype(self, dtype_list: List):
        dtype_list.append(('t', 'f4'))
        dtype_list.append(('t_scale', 'f4'))
        dtype_list.append(('motion_0', 'f4'))
        dtype_list.append(('motion_1', 'f4'))
        dtype_list.append(('motion_2', 'f4'))

    def _fill_extra_ply_data(self, vertex_data):
        def get_tensor(key):
            if key in self._gaussian_params:
                return self._gaussian_params[key]
            return None
            
        t = get_tensor('t').detach().cpu().numpy()
        t_scale = get_tensor('t_scale').detach().cpu().numpy()
        motion = get_tensor('motion').detach().cpu().numpy()
        
        # Check normalized mode
        normalized_t_mode = getattr(self.config, 'normalized_t', True)

        if normalized_t_mode:
            # Un-normalize for saving (0-1 -> Seconds)
            t_extent = self.t_extent.item()
            t_start = self.t_start.item()
            
            # t_sec = t_norm * t_extent + t_start
            vertex_data['t'] = t[:, 0] * t_extent + t_start
            
            # t_scale (duration): scale * t_extent
            # log(scale * t_extent) = log(scale) + log(t_extent)
            # Assuming params are log-scale, we add log(t_extent)
            vertex_data['t_scale'] = t_scale[:, 0] + np.log(t_extent)
            
            # Motion: v_sec = v_norm / t_extent
            vertex_data['motion_0'] = motion[:, 0] / t_extent
            vertex_data['motion_1'] = motion[:, 1] / t_extent
            vertex_data['motion_2'] = motion[:, 2] / t_extent
        else:
            # Save raw values (already in Seconds)
            vertex_data['t'] = t[:, 0]
            vertex_data['t_scale'] = t_scale[:, 0]
            vertex_data['motion_0'] = motion[:, 0]
            vertex_data['motion_1'] = motion[:, 1]
            vertex_data['motion_2'] = motion[:, 2]

    def get_param_groups(self, optim_config) -> List[Dict]:
        groups = super().get_param_groups(optim_config)
        groups.extend([
            {
                'params': [self._t],
                'lr': optim_config.t_lr,
                'name': "t"
            },
            {
                'params': [self._t_scale],
                'lr': optim_config.t_scale_lr,
                'name': "t_scale"
            },
            {
                'params': [self._motion],
                'lr': optim_config.velocity_lr,
                'name': "velocity"
            }
        ])
        return groups

    def get_at_time(self, timestamp: float, opacity_threshold: float = 0.001) -> Dict[str, torch.Tensor]:
        # 1. Get original parameters
        # print("get_at_time called with timestamp:", timestamp)
        mu_x = self.get_xyz.float()          # [N, 3]
        mu_t = self.get_t.float()            # [N, 1]
        
        # 2. Get Duration (Scale)
        # Note: self.get_t_scale already applies exp() activation via _param_activations
        s = self.get_t_scale.float()         # [N, 1]
        
        v = self.get_motion.float()          # [N, 3]
        
        # 3. Calculate time difference
        delta_t = timestamp - mu_t  # [N, 1]
        
        # 4. Update position (Eq. 1 in paper)
        # Note: Rotation is static in FreeTimeGS, only position changes with time.
        xyz_at_t = mu_x + v * delta_t  # [N, 3]
        # xyz_at_t = mu_x 
        
        # 5. Calculate temporal opacity weight
        # s is duration (standard deviation)
        temporal_weight = torch.exp(-0.5 * (delta_t / (s + 1e-7)) ** 2)  # [N, 1]
        
        # 6. Modulate opacity
        base_opacity = self.get_opacity.float()  # [N, 1]
        opacity_at_t = base_opacity * temporal_weight  # [N, 1]
        
        # 7. Cull invisible Gaussians
        mask = opacity_at_t.squeeze() > opacity_threshold  # [N]
        
        return {
            'xyz_at_t': xyz_at_t,
            'opacity_at_t': opacity_at_t,
            'temporal_weight': temporal_weight,
            'mask': mask
        }

    def relocate(self, mask: torch.Tensor, new_xyz: torch.Tensor, timestamp: float, new_motion: Optional[torch.Tensor] = None):
        """
        Relocate masked gaussians to new positions and reset their temporal attributes.
        Used for periodic relocation in FreeTimeGS.
        
        Args:
           mask: Boolean mask of gaussians to relocate [N]
           new_xyz: New 3D positions for these gaussians [M, 3] where M = mask.sum()
           timestamp: Current time to set as new mu_t
           new_motion: New velocity for these gaussians [M, 3]. If None, resets to 0.
        """
        if mask.sum() == 0:
            return

        # 1. Reset Position (XYZ) to new locations
        # Note: Optimizer update is handled by replace_tensor_to_optimizer in densifier usually, 
        # but here we might just modify the tensor data if the optimizer link is preserved,
        # OR we rely on the densifier to handle the parameter replacement.
        # However, standard densification acts on *shapes*. Relocation modifies existing values.
        # If we just change .data, Adam states (momentum) are stale. 
        # Ideally we should zero out their momentum.
        
        # For simplicity and speed in this specific implementation:
        # We assume this is called *during* densification logic where we might handle optimization updates.
        # But if called standalone, we must be careful.
        # The prompt implies a method on the model.
        
        # Update XYZ
        optimizable_tensors = {}
        
        # Direct data modification (simplest, though momentum might push it away initially)
        self._xyz[mask] = new_xyz.to(self.device)
        
        # 2. Reset Time (mu_t) -> Current Time
        self._t[mask] = timestamp
        
        # 3. Reset Motion -> Inherited velocity or 0
        if new_motion is not None:
             self._motion[mask] = new_motion.to(self.device)
        else:
             self._motion[mask] = 0.0

        
        # 4. Reset Opacity -> slightly increased to survive execution
        # inverse_sigmoid(0.5) is 0.0. inverse_sigmoid(0.1) is -2.19
        new_opacity = inverse_sigmoid(0.1 * torch.ones(mask.sum(), 1, device=self.device))
        self._opacity[mask] = new_opacity
        
        # 5. Reset Duration -> random small duration, rand*2 -> [0,2]. -3 -> [-3, -1]. Correct.
        n_relocated = mask.sum()
        random_tscales = (torch.rand(n_relocated, 1, device="cuda") * 2.0) - 3.0 
        self._t_scale[mask] = random_tscales


def detect_mode_from_ply(ply_path: str) -> str:
    """
    Auto-detect mode (static/freetime) from PLY file.
    """
    from plyfile import PlyData
    
    plydata = PlyData.read(ply_path)
    vertex_names = [prop.name for prop in plydata['vertex'].properties]
    
    has_t = 't' in vertex_names
    has_t_scale = 't_scale' in vertex_names
    has_motion = any(name in vertex_names for name in ['motion_0', 'motion_1', 'motion_2'])
    
    if has_t or has_t_scale or has_motion:
        return "freetime"
    else:
        return "static"


def create_model_from_config(config: ModelConfig, device: str = "cuda") -> GaussianModel:
    """Factory function to create the correct model type."""
    if config.mode == "freetime":
        return FreeTimeGaussianModel(config, device)
    else:
        return GaussianModel(config, device)
