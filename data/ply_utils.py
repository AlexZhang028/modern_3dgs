"""
PLY Utility Functions
"""

import numpy as np
from plyfile import PlyData, PlyElement
from utils import BasicPointCloud


def fetchPly(path):
    """Load point cloud from PLY file"""
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    
    # Load positions (Required)
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    
    # Load colors (Try multiple names)
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        try:
            # Some files use r, g, b
            colors = np.vstack([vertices['r'], vertices['g'], vertices['b']]).T / 255.0
        except:
            # If no color, use white
            print(f"Warning: No color information found in PLY, using white color")
            colors = np.ones_like(positions)
    
    # Load normals (Optional)
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        # If no normals, use zero vectors
        normals = np.zeros_like(positions)
    
    # Load Temporal Attributes (FreeTimeGS)
    t = None
    t_scale = None
    motion = None
    
    vertex_names = [p.name for p in plydata['vertex'].properties]
    
    if 't' in vertex_names:
        t = np.asarray(vertices['t'])[..., None]
        
    if 't_scale' in vertex_names:
        t_scale = np.asarray(vertices['t_scale'])[..., None]
        
    motion_names = ['motion_0', 'motion_1', 'motion_2']
    if all(name in vertex_names for name in motion_names):
        motion = np.stack([
            np.asarray(vertices[name]) for name in motion_names
        ], axis=1)

    return BasicPointCloud(
        points=positions, 
        colors=colors, 
        normals=normals,
        t=t,
        t_scale=t_scale,
        motion=motion
    )


def storePly(path, xyz, rgb):
    """
    Save point cloud to PLY file.
    
    Args:
        path: Path to PLY file
        xyz: Point coordinates [N, 3]
        rgb: Colors [N, 3], range [0, 255]
    """
    # Define PLY data structure
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create and write PLY file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def load_ply_to_points(path):
    """
    Load point cloud from PLY file, return coordinates and colors.
    
    Args:
        path: PLY file path
        
    Returns:
        points: [N, 3] numpy array, point coordinates
        colors: [N, 3] numpy array, colors range [0, 1]
    """
    point_cloud = fetchPly(path)
    return point_cloud.points, point_cloud.colors
