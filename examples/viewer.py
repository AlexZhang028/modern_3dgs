"""
FreeTimeGS / 3DGS Viewer
========================

A unified viewer for both static 3D Gaussian Splatting and dynamic FreeTimeGS models.

Features:
- Auto-detection of model type (Static vs FreeTime)
- Real-time interaction (Orbit controls)
- Live Rendering (GLFW)
- Video/Image Export
- Automatic scene centering

Usage:
    # Live Viewer (Auto-detect mode)
    python examples/viewer.py --ply path/to/model.ply

    # Export Video (for dynamic) or Image (for static)
    python examples/viewer.py --ply path/to/model.ply --output render_out
    
    # Custom Resolution
    python examples/viewer.py --ply path/to/model.ply --width 1280 --height 720
"""

import sys
import os

# ─── Linux GTK hard-crash prevention ─────────────────────────────────────────
# Problem: GLFW pulls in GTK 3 (for clipboard/Wayland support).  On RHEL-family
# systems the adwaita-icon-theme package ships *empty* SVG files.  GTK resolves
# any missing icon to the "image-missing" fallback, which is also an empty SVG,
# and then hits a fatal g_assert → SIGABRT before the viewer window opens.
#
# Root cause: os.environ changes are too late — GTK reads env vars when its .so
# is dlopen()'d (i.e. at `import glfw`), not when Python code runs.
#
# Two-stage fix (runs only once, before any other import):
#   Stage 1 — write a valid local SVG override at $XDG_DATA_HOME/icons/…  so
#              GTK finds it before the broken system file (XDG search order).
#   Stage 2 — os.execve() to *replace* the current process with a fresh one
#              that already has the GTK env vars set when its first .so loads.
#              The sentinel _VIEWER_ENV_SET prevents infinite re-exec.
if sys.platform.startswith('linux') and not os.environ.get('_VIEWER_ENV_SET'):
    from pathlib import Path as _P

    # Stage 1: local SVG override
    _svg = _P.home() / '.local/share/icons/Adwaita/scalable/status/image-missing.svg'
    if not _svg.exists() or _svg.stat().st_size == 0:
        _svg.parent.mkdir(parents=True, exist_ok=True)
        _svg.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"/>\n')
    _idx = _P.home() / '.local/share/icons/Adwaita/index.theme'
    if not _idx.exists():
        _idx.write_text(
            '[Icon Theme]\nName=Adwaita\nDirectories=scalable/status\n\n'
            '[scalable/status]\nSize=16\nType=Scalable\nMinSize=1\nMaxSize=256\n'
        )

    # Stage 2: re-exec with env vars baked in from process start
    _env = os.environ.copy()
    _env.update({
        'GTK_THEME':        'Hicolor',
        'NO_AT_BRIDGE':     '1',
        'GDK_BACKEND':      'x11',
        '_VIEWER_ENV_SET':  '1',
    })
    _cache = '/usr/lib64/gdk-pixbuf-2.0/2.10.0/loaders.cache'
    if not _env.get('GDK_PIXBUF_MODULE_FILE') and os.path.isfile(_cache):
        _env['GDK_PIXBUF_MODULE_FILE'] = _cache
    os.execve(sys.executable, [sys.executable] + sys.argv, _env)
    # os.execve replaces the process image — code below only runs in the
    # re-exec'd process where _VIEWER_ENV_SET=1 and the env is already clean.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import time
import math
import numpy as np
import torch
import cv2
from typing import Optional, Tuple, Dict
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.gaussian_model import (
    GaussianModel, FreeTimeGaussianModel, SharpTimeGaussianModel,
    detect_mode_from_ply, create_model_from_config,
)
from core.renderer import GaussianRenderer
from config.config import ModelConfig, PipelineConfig
from data.camera import Camera
from utils.general_utils import seed_everything

# Try importing GLFW for live viewing
try:
    import glfw
    from OpenGL.GL import *
    GLFW_AVAILABLE = True
except ImportError:
    GLFW_AVAILABLE = False
    print("Warning: GLFW/PyOpenGL not found. Live viewing disabled.")

# =============================================================================
# Camera Utilities
# =============================================================================

class SimpleCamera(Camera):
    """
    A simplified camera class for the viewer, inheriting from the data.camera.Camera.
    Manages extrinsic/intrinsic updates dynamically.
    """
    def __init__(self, width: int, height: int, fov_deg: float, device="cuda"):
        # Initialize basic params
        self.device = device
        fov_rad = np.radians(fov_deg)
        
        # Calculate FovX/FovY
        # Assuming FovX is given
        self.FovX = fov_rad
        self.FovY = 2 * math.atan(math.tan(self.FovX / 2) * (height / width))
        
        # Initialize parent with dummy data first, then update
        # We don't need real R/T yet, will set them via update_pose
        super().__init__(
            uid=0, 
            image_name="viewer", 
            R=np.eye(3), 
            T=np.zeros(3), 
            width=width, 
            height=height,
            FovX=self.FovX,
            FovY=self.FovY
        )
        
        # Manually ensure transforms are on device if parent puts them there
        # But data.camera.Camera usually computes them on property access or requires manual move?
        # Looking at Camera source, it computes world_view_transform on init or needs update.
        # We will override the update method.

    def update_pose(self, c2w_matrix: np.ndarray):
        """
        Update camera pose from Camera-to-World matrix.
        """
        # Invert C2W to get W2C
        w2c = np.linalg.inv(c2w_matrix)
        
        # R is 3x3, T is 3x1
        # 3DGS 的 camera.py 中，world_view_transform 的构建使用了 getWorld2View2。
        # 该函数内部会将传入的 R 进行转置 (R.transpose())。
        # 因此，为了得到正确的 View Matrix (W2C)，我们需要传入 W2C 旋转矩阵的转置
        # (或者直接理解为我们需要传入 C2W 的旋转矩阵)
        self.R = w2c[:3, :3].transpose() 
        self.T = w2c[:3, 3]
        
        # Force re-computation of projection matrices
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix
        
        self.trans = np.array([0.0, 0.0, 0.0]) 
        self.scale = 1.0 
        
        # Re-upload to GPU
        # 注意：这里 getWorld2View2 内部会再次转置 self.R，从而复原为正确的 W2C 旋转
        self.world_view_transform = torch.tensor(
            getWorld2View2(self.R, self.T, self.trans, self.scale)
        ).transpose(0, 1).to(self.device)
        
        # Force re-computation of projection matrices
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix
        
        self.trans = np.array([0.0, 0.0, 0.0]) # Scene normalization translation
        self.scale = 1.0 # Scene normalization scale
        
        # Re-upload to GPU
        self.world_view_transform = torch.tensor(
            getWorld2View2(self.R, self.T, self.trans, self.scale)
        ).transpose(0, 1).to(self.device)
        
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FovX, fovY=self.FovY
        ).transpose(0, 1).to(self.device)
        
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class ObitControl:
    """Orbital Camera Control (Y-Up)."""
    def __init__(self, look_at=np.array([0, 0, 0]), radius=3.0, theta=0.0, phi=np.pi/3):
        self.look_at = look_at
        self.radius = radius
        self.theta = theta # Azimuth (around Y)
        self.phi = phi     # Elevation (Angle from Y-axis)
        
        self.phi = np.clip(self.phi, 1e-3, np.pi - 1e-3)
        
        # Save initial state for reset
        self.init_state = {
            'look_at': look_at.copy(),
            'radius': radius,
            'theta': theta,
            'phi': phi
        }
        
    def reset(self):
        self.look_at = self.init_state['look_at'].copy()
        self.radius = self.init_state['radius']
        self.theta = self.init_state['theta']
        self.phi = self.init_state['phi']
        
    def get_c2w(self):
        """
        Get Camera-to-World matrix.
        Convention: OpenCV (X Right, Y Down, Z Forward) in Camera Frame.
        World Frame: Y-Up.
        """
        # Spherical Coordinates (Y-up convention)
        # y = r * cos(phi)
        # x = r * sin(phi) * sin(theta)
        # z = r * sin(phi) * cos(theta)
        
        x = self.radius * np.sin(self.phi) * np.sin(self.theta)
        y = self.radius * np.cos(self.phi)
        z = self.radius * np.sin(self.phi) * np.cos(self.theta)
        
        pos = self.look_at + np.array([x, y, z])
        
        # LookAt Matrix
        # Forward points from pos to look_at
        forward = self.look_at - pos
        forward /= np.linalg.norm(forward)
        
        # World Up = Y
        # world_up = np.array([0, 1.0, 0])
        # For SelfCap data, Y is downwards, so we flip the world up to -Y
        world_up = np.array([0, -1.0, 0])
        
        # Right (X_cam)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
             # Singularity
             right = np.array([1, 0, 0])
        right /= np.linalg.norm(right)
        
        # True Camera Up (Y_cam direction is DOWN for OpenCV)
        # Standard Up
        std_up = np.cross(right, forward)
        
        # OpenCV Y is Down
        cam_down = -std_up
        
        # Rotation Matrix
        R = np.eye(3)
        R[:, 0] = right     # X
        R[:, 1] = cam_down  # Y
        R[:, 2] = forward   # Z
        
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos
        return c2w

    def handle_mouse(self, dx, dy, mode='rotate'):
        sensitivity_rotate = 0.005
        sensitivity_pan = 0.001 * self.radius
        
        if mode == 'rotate':
            # Horizontal Drag (dx) -> Rotate around Y (Theta)
            # Drag Left -> Camera moves Left -> Theta decreases? 
            # Usually: Drag Left -> Rotate Scene Left -> Camera moves Right -> Theta increases?
            # Let's try standard orbit: Drag Left (dx<0) -> Theta decreases.
            self.theta -= dx * sensitivity_rotate
            
            # Vertical Drag (dy) -> Rotate around X / Elevation (Phi)
            # Drag Down (dy>0) -> Camera moves Down -> Phi increases (towards South Pole)
            self.phi += dy * sensitivity_rotate
            self.phi = np.clip(self.phi, 1e-3, np.pi - 1e-3)
            
        elif mode == 'pan':
            c2w = self.get_c2w()
            right = c2w[:3, 0]
            down = c2w[:3, 1]
            
            self.look_at -= (right * dx * sensitivity_pan)
            self.look_at -= (down * dy * sensitivity_pan)

    def handle_scroll(self, yoffset):
        self.radius *= (1.0 - yoffset * 0.1)
        self.radius = max(0.1, self.radius)


# =============================================================================
# Helper Functions
# =============================================================================

def cam_to_c2w(cam) -> np.ndarray:
    """Reconstruct 4x4 C2W matrix from a dataset Camera.

    Camera.R stores the C2W rotation (= transpose of W2C rotation).
    Camera.camera_center stores the camera position in world coordinates.
    """
    c2w = np.eye(4)
    c2w[:3, :3] = cam.R
    c2w[:3, 3] = cam.camera_center.cpu().numpy()
    return c2w


def interpolate_cameras(c2w_a: np.ndarray, c2w_b: np.ndarray, n_frames: int):
    """Return n_frames C2W matrices interpolated between c2w_a and c2w_b.

    Rotation uses SLERP; translation uses LERP.
    """
    from scipy.spatial.transform import Rotation, Slerp

    rot_a = Rotation.from_matrix(c2w_a[:3, :3])
    rot_b = Rotation.from_matrix(c2w_b[:3, :3])
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([rot_a, rot_b]))

    pos_a = c2w_a[:3, 3]
    pos_b = c2w_b[:3, 3]

    result = []
    for i in range(n_frames):
        t = i / (n_frames - 1) if n_frames > 1 else 0.0
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = slerp(t).as_matrix()
        c2w[:3, 3] = (1.0 - t) * pos_a + t * pos_b
        result.append(c2w)
    return result


def estimate_camera_init(gaussians: GaussianModel) -> Tuple[np.ndarray, float]:
    """
    Estimate initial camera position and radius based on scene bounds.
    Returns: (center, radius)
    """
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    center = np.mean(xyz, axis=0)
    
    # Calculate rough bounding box radius
    dists = np.linalg.norm(xyz - center, axis=1)
    radius = np.percentile(dists, 90) * 2.0  # 2x the 90th percentile distance
    
    print(f"Auto-center: {center}, Radius: {radius:.2f}")
    return center, radius

def draw_axes_overlay(c2w, window_w, window_h, size=100):
    """
    Draw RGB axes in the bottom-right corner.
    """
    # Viewport for axes (Bottom-Right)
    # glViewport(x, y, width, height) where (0,0) is bottom-left
    glViewport(window_w - size, 0, size, size)
    
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    # Simple ortho projection for orientation widget
    glOrtho(-1.2, 1.2, -1.2, 1.2, -10, 10)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    # Apply Camera Rotation
    # We want to rotate the world (axes) relative to camera.
    # This corresponds to the Rotation part of the View Matrix (W2C).
    w2c = np.linalg.inv(c2w)
    
    # Convert OpenCV Camera (Right, Down, Fwd) to OpenGL Camera (Right, Up, Back)
    # w2c_gl = S @ w2c_cv
    S = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    w2c_gl = S @ w2c
    
    R = w2c_gl[:3, :3]
    
    # Construct View Matrix (Rotation only)
    view_mat = np.eye(4, dtype=np.float32)
    view_mat[:3, :3] = R
    
    # GL expects column-major order. Numpy is row-major.
    # Transposing row-major numpy array gives column-major data when flattened for GL.
    glMultMatrixf(view_mat.T.flatten())
    
    # Draw Axes
    glLineWidth(3.0)
    glBegin(GL_LINES)
    
    # X - Red
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0); glVertex3f(1, 0, 0)
    
    # Y - Green
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0); glVertex3f(0, 1, 0)
    
    # Z - Blue
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0); glVertex3f(0, 0, 1)
    
    glEnd()
    glLineWidth(1.0)
    
    # Reset Color to White to avoid tinting main scene
    glColor3f(1, 1, 1)
    
    # Cleanup
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    
    # Restore Main Viewport
    glViewport(0, 0, window_w, window_h)


# =============================================================================
# Renderers
# =============================================================================

def render_preview_static(
    gaussians: GaussianModel, 
    renderer: GaussianRenderer, 
    cam_state: ObitControl,
    width, height, fov
):
    start_time = time.time()
    camera = SimpleCamera(width, height, fov)
    camera.update_pose(cam_state.get_c2w())
    
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    with torch.no_grad():
        out = renderer(gaussians, camera, bg_color)
    
    torch.cuda.synchronize()
    render_time_ms = (time.time() - start_time) * 1000
    
    img = out['render'].permute(1, 2, 0).detach().cpu().numpy()
    
    stats = {
        'render_ms': render_time_ms,
        'visible': out['visibility_filter'].sum().item(),
        'total': gaussians.num_points
    }
    return img, stats

def render_preview_dynamic(
    gaussians: FreeTimeGaussianModel, 
    renderer: GaussianRenderer, 
    cam_state: ObitControl, 
    time_val: float,
    width, height, fov
):
    start_time = time.time()
    camera = SimpleCamera(width, height, fov)
    camera.update_pose(cam_state.get_c2w())
    
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    with torch.no_grad():
        out = renderer(gaussians, camera, bg_color, timestamp=time_val, enable_culling=True) # Disable culling to get accurate visibility stats for dynamic models
    
    torch.cuda.synchronize()
    render_time_ms = (time.time() - start_time) * 1000
    
    img = out['render'].permute(1, 2, 0).detach().cpu().numpy()
    
    # Extract stats from temporal info
    visible = out['temporal_info']['visible_gaussians']
    total = out['temporal_info']['total_gaussians']
    
    stats = {
        'render_ms': render_time_ms,
        'visible': visible,
        'total': total
    }
    return img, stats


def run_live_viewer(args, gaussians, renderer, init_center, init_radius):
    if not GLFW_AVAILABLE:
        print("Error: Live viewer requires GLFW.")
        return

    # Init GLFW
    if not glfw.init():
        return
        
    window = glfw.create_window(args.width, args.height, "FreeTimeGS / 3DGS Viewer", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(0) # Disable VSync for uncapped FPS measurement
    
    # Init Controls
    # Initial pose: theta=0, phi=90deg (on equator)
    control = ObitControl(look_at=init_center, radius=init_radius, theta=0, phi=np.pi/2)
    
    # Init GL Texture
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    # Control State
    mouse_last_x, mouse_last_y = 0, 0
    mouse_btn_left = False
    mouse_btn_right = False
    
    def mouse_button_callback(window, button, action, mods):
        nonlocal mouse_btn_left, mouse_btn_right
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse_btn_left = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            mouse_btn_right = (action == glfw.PRESS)
            
    def cursor_pos_callback(window, xpos, ypos):
        nonlocal mouse_last_x, mouse_last_y
        dx = xpos - mouse_last_x
        dy = ypos - mouse_last_y
        mouse_last_x = xpos
        mouse_last_y = ypos
        
        if mouse_btn_left:
            control.handle_mouse(dx, dy, 'rotate')
        elif mouse_btn_right:
            control.handle_mouse(dx, dy, 'pan')
            
    def scroll_callback(window, xoffset, yoffset):
        control.handle_scroll(yoffset)

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    
    # Time Configuration — both freetime and sharptime are dynamic
    is_freetime = gaussians.mode in ("freetime", "sharptime")
    
    start_time = args.start_time
    end_time = args.end_time
    loop_duration = 5.0 # Default fallback
    
    if is_freetime:
        t_max_actual = gaussians.get_t.detach().max().item()
        
        # 1. Determine Start Time (Default 0.0)
        if start_time < 0:
            start_time = 0.0
            
        # 2. Determine End Time (Default floor(max_t))
        if end_time < 0:
            end_time = math.floor(t_max_actual)
            # Fallback for normalized data (e.g., max=0.9 -> floor=0)
            # If floor results in 0 (and start is 0), and actual data > 0.5, use actual max.
            if end_time <= start_time and t_max_actual > 0.5:
                 end_time = t_max_actual
        
        print(f"Time Range: [{start_time:.3f}, {end_time:.3f}]")
        
        # 3. Determine Duration (Natural duration)
        raw_duration = end_time - start_time
        if raw_duration > 0:
            loop_duration = raw_duration
        else:
            loop_duration = 1.0
    else:
        start_time = 0.0
        end_time = 1.0

    app_start_time = time.time()
    
    frame_count = 0
    last_title_time = time.time()

    paused = False
    pause_start_time = 0.0
    total_paused_duration = 0.0

    while not glfw.window_should_close(window):
        # Poll events
        glfw.poll_events()
        
        # Keyboard Inputs
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            control.reset()
            
        # Pause/Resume (Simple debounce)
        if not hasattr(run_live_viewer, "last_space_press"):
             run_live_viewer.last_space_press = 0
             
        if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
            if time.time() - run_live_viewer.last_space_press > 0.3:
                if not paused:
                    pause_start_time = time.time()
                    paused = True
                else:
                    total_paused_duration += (time.time() - pause_start_time)
                    paused = False
                run_live_viewer.last_space_press = time.time()

        current_time_clock = time.time()
        
        # Calculate scene time
        scene_t = 0.0
        if is_freetime:
            effective_elapsed = (pause_start_time if paused else current_time_clock) - app_start_time - total_paused_duration
            
            if args.loop:
                progress = (effective_elapsed % loop_duration) / loop_duration
                scene_t = start_time + progress * (end_time - start_time)
            else:
                progress = min(effective_elapsed / loop_duration, 1.0)
                scene_t = start_time + progress * (end_time - start_time)

        # Render
        if is_freetime:
            img, stats = render_preview_dynamic(gaussians, renderer, control, scene_t, args.width, args.height, args.fov)
        else:
            img, stats = render_preview_static(gaussians, renderer, control, args.width, args.height, args.fov)
            
        # Update Window Title (FPS, etc)
        frame_count += 1
        if current_time_clock - last_title_time >= 0.5: # Update every 0.5s
            fps = frame_count / (current_time_clock - last_title_time)
            
            total_points = stats['total'] if stats['total'] > 0 else 1
            vis_pct = 100 * stats['visible'] / total_points
            
            title = f"FreeTimeGS Viewer | FPS: {fps:.1f} | Gaussian: {stats['total']} | Visible: {stats['visible']} ({vis_pct:.1f}%)"
            if is_freetime:
                title += f" | Time: {scene_t:.3f}"
            
            glfw.set_window_title(window, title)
            frame_count = 0
            last_title_time = current_time_clock

        # Display
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        
        # Prepare for OpenGL (Flip Y)
        img = np.flipud(img)
        
        width = args.width
        height = args.height
        
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
        
        glClear(GL_COLOR_BUFFER_BIT)
        glEnable(GL_TEXTURE_2D)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(-1, -1)
        glTexCoord2f(1, 0); glVertex2f(1, -1)
        glTexCoord2f(1, 1); glVertex2f(1, 1)
        glTexCoord2f(0, 1); glVertex2f(-1, 1)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        draw_axes_overlay(control.get_c2w(), width, height)
        
        glfw.swap_buffers(window)
        
    glfw.terminate()

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, required=True, help="Path to input .ply file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--output", type=str, default=None, help="Output path (without extension). If set, runs offline render.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start_time", type=float, default=-1, help="Start time for dynamic playback")
    parser.add_argument("--end_time", type=float, default=-1, help="End time for dynamic playback")
    parser.add_argument("--loop", action="store_true", help="Loop the animation")
    
    # Optional arguments for rendering specific camera from dataset
    parser.add_argument("--source_path", type=str, default="", help="Path to dataset root (Req if --cam is used)")
    parser.add_argument("--cam", type=str, nargs='+', default=None,
                        help="Camera name(s) from dataset. Single: render that camera's frames. "
                             "Two (e.g. --cam 01 02): interpolate between those two cameras.")
    parser.add_argument("--interp_frames", type=int, default=60,
                        help="Number of frames when interpolating between two cameras (default 60)")
    parser.add_argument("--output_gt", action="store_true", help="If set and --cam is used, output GT video as well")
    parser.add_argument("--resolution", type=int, default=1, help="Dataset resolution factor (1, 2, 4, 8) if rendering --cam")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame for rendering dataset (only applies to --cam)")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame for rendering dataset (only applies to --cam)")
    
    args = parser.parse_args()
    
    seed_everything(0)
    
    # 1. Detect Mode & Load Model
    from plyfile import PlyData as _PlyData
    _ply_fields = {p.name for p in _PlyData.read(args.ply)['vertex'].properties}
    _is_init_pcd = 'opacity' not in _ply_fields   # init PLY has no trained Gaussian attrs

    if _is_init_pcd:
        # Init point cloud (output of sharptime_init.py) — use create_from_pcd path
        mode = "sharptime" if ('t' in _ply_fields or 'motion_0' in _ply_fields) else "static"
        print(f"Detected init point cloud (no opacity field). Visualising as {mode.upper()} gaussians...")
        config = ModelConfig(mode=mode, sh_degree=3)
        gaussians = create_model_from_config(config, args.device)
        from data.ply_utils import fetchPly
        pcd = fetchPly(args.ply)
        gaussians.create_from_pcd(pcd, spatial_lr_scale=1.0)
    else:
        mode = detect_mode_from_ply(args.ply)
        print(f"Detected Model Mode: {mode.upper()}")
        config = ModelConfig(mode=mode, sh_degree=3)
        gaussians = create_model_from_config(config, args.device)
        print(f"Loading PLY from {args.ply}...")
        gaussians.load_ply(args.ply)
    
    # 2. Setup Renderer
    render_config = PipelineConfig()
    renderer = GaussianRenderer(render_config).to(args.device)
    
    # 3. Auto-Center Camera
    center, radius = estimate_camera_init(gaussians)
    # Default Phi changed to pi/3 for better Z-up view
    control = ObitControl(look_at=center, radius=radius, theta=0, phi=np.pi/3)
    
    # 4. Run Mode
    if args.output:
        # Offline Render
        print(f"Starting Offline Render...")
        
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if args.cam and len(args.cam) == 2:
            # ── Two-camera interpolation mode ──────────────────────────────
            cam_name_a, cam_name_b = args.cam
            print(f"Interpolating between cameras '{cam_name_a}' and '{cam_name_b}'...")
            from core.builder import setup_dataset
            from config.config import DataConfig

            if not args.source_path:
                print("Error: --source_path is required when using --cam")
                return

            def _load_first_cam(cam_name):
                dc = DataConfig(
                    source_path=args.source_path,
                    resolution=args.resolution,
                    eval=True,
                    test_cameras=[cam_name],
                    start_frame=args.start_frame,
                    end_frame=args.end_frame,
                    cache_images=False,
                    # lazy_loading=True,
                    inference_only=True,
                )
                ds = setup_dataset(dc, split="test")
                if not ds or len(ds) == 0:
                    print(f"Error: No frames found for camera '{cam_name}'")
                    return None
                sorted_idx = sorted(range(len(ds)), key=lambda i: ds.cameras[i].timestamp)
                return ds.cameras[sorted_idx[0]]

            cam_a = _load_first_cam(cam_name_a)
            cam_b = _load_first_cam(cam_name_b)
            if cam_a is None or cam_b is None:
                return

            c2w_list = interpolate_cameras(cam_to_c2w(cam_a), cam_to_c2w(cam_b), args.interp_frames)

            render_w = cam_a.width
            render_h = cam_a.height
            fov_x_deg = np.degrees(cam_a.FovX)

            # Temporal interpolation range (dynamic models only)
            t_start = args.start_time if args.start_time >= 0 else (cam_a.timestamp or 0.0)
            t_end   = args.end_time   if args.end_time   >= 0 else (cam_b.timestamp or t_start)

            out_path = f"{args.output}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            interp_out = cv2.VideoWriter(out_path, fourcc, args.fps, (render_w, render_h))
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

            if mode in ("freetime", "sharptime"):
                print(f"Rendering {args.interp_frames} interpolated frames "
                      f"({render_w}x{render_h}), cam {cam_name_a}→{cam_name_b}, "
                      f"t {t_start:.4f}→{t_end:.4f}...")
            else:
                print(f"Rendering {args.interp_frames} interpolated frames "
                      f"({render_w}x{render_h}), cam {cam_name_a}→{cam_name_b}...")

            for idx, c2w in enumerate(tqdm(c2w_list)):
                alpha = idx / (args.interp_frames - 1) if args.interp_frames > 1 else 0.0
                interp_cam = SimpleCamera(render_w, render_h, fov_x_deg, device=args.device)
                interp_cam.update_pose(c2w)
                with torch.no_grad():
                    if mode in ("freetime", "sharptime"):
                        render_t = (1.0 - alpha) * t_start + alpha * t_end
                        out_dict = renderer(gaussians, interp_cam, bg_color,
                                            timestamp=render_t, enable_culling=True)
                    else:
                        out_dict = renderer(gaussians, interp_cam, bg_color)
                img_render = out_dict['render'].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                interp_out.write(cv2.cvtColor((img_render * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                del out_dict, img_render

            interp_out.release()
            print(f"Interpolation video saved to {out_path}")

        elif args.cam:
            # ── Single-camera dataset render (existing behaviour) ──────────
            cam_name = args.cam[0]
            print(f"Rendering Specific Camera '{cam_name}' from {args.source_path}...")
            from core.builder import setup_dataset
            from config.config import DataConfig

            if not args.source_path:
                print("Error: --source_path is required when using --cam")
                return

            data_config = DataConfig(
                source_path=args.source_path,
                resolution=args.resolution,
                eval=True,
                test_cameras=[cam_name],
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                cache_images=False,
                # lazy_loading=True,
                inference_only=not args.output_gt
            )
            dataset = setup_dataset(data_config, split="test")

            if not dataset or len(dataset) == 0:
                print(f"Error: No frames found for camera '{cam_name}'")
                return
                
            # sort by time
            sorted_indices = sorted(range(len(dataset)), key=lambda i: dataset.cameras[i].timestamp)
            
            n_frames = len(sorted_indices)
            first_cam = dataset.cameras[sorted_indices[0]]
            render_w = first_cam.width
            render_h = first_cam.height
            
            print(f"Found {n_frames} frames. Resolution: {render_w}x{render_h}")
            
            out_path = f"{args.output}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, args.fps, (render_w, render_h))
            
            out_gt = None
            if args.output_gt:
                out_gt_path = f"{args.output}_gt.mp4"
                out_gt = cv2.VideoWriter(out_gt_path, fourcc, args.fps, (render_w, render_h))
                
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            
            for i in tqdm(sorted_indices):
                sample = dataset[i]
                cam_data = sample['camera'].to(args.device)
                
                with torch.no_grad():
                    if mode in ("freetime", "sharptime"):
                        out_dict = renderer(gaussians, cam_data, bg_color, timestamp=cam_data.timestamp, enable_culling=True)
                    else:
                        out_dict = renderer(gaussians, cam_data, bg_color)
                
                img_render = out_dict['render'].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                out.write(cv2.cvtColor((img_render * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                if out_gt is not None:
                    img_gt = sample['image'].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                    out_gt.write(cv2.cvtColor((img_gt * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    
                # Free memory to prevent OOM
                cam_data.image = None
                cam_data.alpha_mask = None
                cam_data.depth_map = None
                cam_data.depth_mask = None
                
                # Delete local references
                del out_dict
                del img_render
                del sample
                if i % 10 == 0:
                    torch.cuda.empty_cache()
                    
            out.release()
            print(f"Saved video to {out_path}")
            if out_gt is not None:
                out_gt.release()
                print(f"Saved GT video to {out_gt_path}")
                
        elif mode == "static":
            print("Rendering Static Image...")
            img, _ = render_preview_static(gaussians, renderer, control, args.width, args.height, args.fov)
            out_path = f"{args.output}.png"
            cv2.imwrite(out_path, cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            print(f"Saved to {out_path}")
            
        elif mode in ("freetime", "sharptime"):
            print("Rendering Temporal Video...")
            frames = []
            
            # Determine time range
            t_max_actual = gaussians.get_t.max().item()
            
            # Start Time Default: 0.0
            start_t = args.start_time if args.start_time >= 0 else 0.0
            
            # End Time Default: floor(max_t)
            end_t = args.end_time
            if end_t < 0:
                end_t = math.floor(t_max_actual)
                # Fallback if range is 0 (e.g. data is [0, 0.9])
                if end_t <= start_t and t_max_actual > start_t:
                    end_t = t_max_actual
            
            duration = end_t - start_t
            if duration <= 0: duration = 1.0 # Standard fallback
            
            n_frames = int(duration * args.fps)
            
            print(f"Rendering {n_frames} frames from t={start_t:.2f} to {end_t:.2f} (Duration: {duration:.2f}s)")
            
            for i in tqdm(range(n_frames)):
                progress = i / (n_frames - 1) if n_frames > 1 else 0
                t = start_t + progress * (end_t - start_t)
                
                img, _ = render_preview_dynamic(gaussians, renderer, control, t, args.width, args.height, args.fov)
                img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                frames.append(img_uint8)
                
            # Save Video
            out_path = f"{args.output}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, args.fps, (args.width, args.height))
            
            for frame in frames:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            out.release()
            print(f"Video saved to {out_path}")

    else:
        # Live Mode
        print("Starting Live Viewer...")
        run_live_viewer(args, gaussians, renderer, center, radius)

if __name__ == "__main__":
    main()
