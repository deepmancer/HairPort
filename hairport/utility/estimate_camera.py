# Standard library
import gc
import json
import os
from pathlib import Path
from typing import Tuple

# Third-party libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import trimesh
from PIL import Image, ImageFilter, ImageOps
from plotly import graph_objects as go
from scipy.spatial.transform import Rotation

# Local imports
from hairport.core.mesh_utils import (
    rotate_glb_mesh,
    apply_rotation,
    apply_inverse_rotation,
)

from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import numpy as np
import torch
import torch.nn as nn

from typing import Optional

# --- your existing helpers (unchanged) ---------------------------------

def orthogonal_bounds_from_ortho_scale(
    ortho_scale: torch.Tensor,
    image_size=(1024, 1024),
):
    """
    Compute (left, right, bottom, top) from ortho_scale + image size,
    in the *same* way as project_points_blender_ortho.
    """
    W, H = image_size
    W = float(W)
    H = float(H)

    aspect = W / H
    if aspect >= 1.0:
        half_w = ortho_scale / 2.0
        half_h = half_w / aspect
    else:
        half_h = ortho_scale / 2.0
        half_w = half_h * aspect

    left   = -half_w
    right  = +half_w
    bottom = -half_h
    top    = +half_h
    return left, right, bottom, top


def get_orthogonal_projection_matrix(
    batch_size: int,
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float = 0.1,
    far: float = 100.0,
    device: Optional[str] = None,
) -> torch.FloatTensor:
    """
    Same as in the external repo.
    Column-vector convention: x_clip = P @ x_cam.
    """
    projection_matrix = torch.zeros(
        batch_size, 4, 4, dtype=torch.float32, device=device
    )
    projection_matrix[:, 0, 0] = 2 / (right - left)
    projection_matrix[:, 1, 1] = -2 / (top - bottom)
    projection_matrix[:, 2, 2] = -2 / (far - near)
    projection_matrix[:, 0, 3] = -(right + left) / (right - left)
    projection_matrix[:, 1, 3] = -(top + bottom) / (top - bottom)
    projection_matrix[:, 2, 3] = -(far + near) / (far - near)
    projection_matrix[:, 3, 3] = 1
    return projection_matrix


def get_ortho_projection_matrix_from_scale(
    batch_size: int,
    ortho_scale: torch.Tensor,
    image_size=(1024, 1024),
    near: float = 0.1,
    far: float = 100.0,
    device: Optional[str] = None,
) -> torch.FloatTensor:
    """
    Wrapper that gives you a 4×4 orthographic matrix with the same
    functionality/convention as get_orthogonal_projection_matrix,
    parameterised by ortho_scale + image_size.
    """
    if not isinstance(ortho_scale, torch.Tensor):
        ortho_scale = torch.tensor(ortho_scale, dtype=torch.float32, device=device)
    else:
        ortho_scale = ortho_scale.to(device=device, dtype=torch.float32)

    left, right, bottom, top = orthogonal_bounds_from_ortho_scale(
        ortho_scale, image_size
    )

    return get_orthogonal_projection_matrix(
        batch_size=batch_size,
        left=float(left),
        right=float(right),
        bottom=float(bottom),
        top=float(top),
        near=near,
        far=far,
        device=device,
    )


def rodrigues(rvec: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    theta = torch.linalg.norm(rvec, dim=-1, keepdim=True)  # (..., 1)
    k = rvec / (theta + eps)
    kx, ky, kz = k.unbind(-1)
    zeros = torch.zeros_like(kx)
    K = torch.stack([
        zeros, -kz,   ky,
        kz,    zeros, -kx,
        -ky,   kx,    zeros
    ], dim=-1).reshape(rvec.shape[:-1] + (3, 3))

    I = torch.eye(3, device=rvec.device, dtype=rvec.dtype).expand_as(K)
    theta = theta.unsqueeze(-1)
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    R = I + sin_t * K + (1.0 - cos_t) * (K @ K)
    return R


def project_points_blender_ortho(
    points_world: torch.Tensor,
    cam_location: torch.Tensor,
    cam_rotvec: torch.Tensor,
    ortho_scale: torch.Tensor,
    image_size=(1024, 1024),
) -> torch.Tensor:
    W, H = image_size
    W = float(W)
    H = float(H)
    dtype = points_world.dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cam_location = cam_location.to(device=device, dtype=dtype)
    cam_rotvec   = cam_rotvec.to(device=device, dtype=dtype)
    ortho_scale  = ortho_scale.to(device=device, dtype=dtype)

    R_world_from_cam = rodrigues(cam_rotvec)  # interpreted as R_cam2world
    pts_local = (points_world - cam_location) @ R_world_from_cam  # (N, 3)

    x_cam = pts_local[:, 0]
    y_cam = pts_local[:, 1]
    z_cam = pts_local[:, 2]
    depth = -z_cam  # optional usage

    aspect = W / H
    if aspect >= 1.0:
        half_w = ortho_scale / 2.0
        half_h = half_w / aspect
    else:
        half_h = ortho_scale / 2.0
        half_w = half_h * aspect

    x_ndc = (x_cam + half_w) / (2.0 * half_w)
    y_ndc = (y_cam + half_h) / (2.0 * half_h)

    x_pix = x_ndc * W
    y_pix = (1.0 - y_ndc) * H
    return torch.stack([x_pix, y_pix], dim=-1)  # (N, 2)


class OptimizableCameraModel(nn.Module):
    def __init__(
        self, 
        init_location: torch.Tensor,
        init_rotation: torch.Tensor,
        init_ortho_scale: float = 1.0,
        device: torch.device = None,
        image_size=(1024, 1024),
        near: float = 0.1,
        far: float = 100.0,
    ):
        """
        Spherical-position + free-rotation orthographic camera.

        Provides:
          - location          (Cartesian cam position)
          - cam_pos           (alias of location, like MVAdapter Camera.cam_pos)
          - distance          (||location||)
          - azimuth_deg       (deg, same definition as MVAdapter)
          - elevation_deg     (deg, same definition as MVAdapter)
          - c2w               (4x4 camera-to-world)
          - w2c               (4x4 world-to-camera)
          - proj_mtx          (4x4 single orthographic projection)
          - mvp_mtx           (4x4 proj_mtx @ w2c)
          - get_projection_matrix(...)  # batched proj matrix, repo-style
        """
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not isinstance(device, torch.device):
            device = torch.device(device)

        init_location = init_location.to(device=device, dtype=torch.float32)
        init_rotation = init_rotation.to(device=device, dtype=torch.float32)

        x, y, z = init_location[0], init_location[1], init_location[2]

        sphere_radius = torch.sqrt(x**2 + y**2 + z**2)
        self.sphere_radius = sphere_radius.to(device=device, dtype=torch.float32)

        # internal spherical params (your convention)
        theta = torch.atan2(x, z)
        phi = torch.asin(torch.clamp(y / (sphere_radius + 1e-8), -1.0, 1.0))

        self.theta = nn.Parameter(theta.clone().detach().to(device))
        self.phi = nn.Parameter(phi.clone().detach().to(device))
        self.rotvec = nn.Parameter(init_rotation.clone().detach().to(device))
        self.ortho_scale = nn.Parameter(
            torch.tensor(init_ortho_scale, dtype=torch.float32, device=device)
        )

        self.image_size = image_size
        self.near = float(near)
        self.far = float(far)

    # ----------------- basic position -----------------

    @property
    def location(self) -> torch.Tensor:
        theta_clamped = torch.clamp(self.theta, -np.pi/2, np.pi/2)
        phi_clamped = torch.clamp(self.phi, -np.pi/2, np.pi/2)

        x = self.sphere_radius * torch.sin(theta_clamped) * torch.cos(phi_clamped)
        y = self.sphere_radius * torch.sin(phi_clamped)
        z = self.sphere_radius * torch.cos(theta_clamped) * torch.cos(phi_clamped)

        return torch.stack([x, y, z])

    @property
    def cam_pos(self) -> torch.Tensor:
        return self.location

    # ----------------- MVAdapter-style spherical -----------------

    @property
    def distance(self) -> torch.Tensor:
        loc = self.location
        return torch.linalg.norm(loc)

    @property
    def azimuth_deg(self) -> torch.Tensor:
        loc = self.location
        x, y, z = loc[0], loc[1], loc[2]
        azim = torch.atan2(y, x)  # radians
        return azim * (180.0 / math.pi)

    @property
    def elevation_deg(self) -> torch.Tensor:
        loc = self.location
        x, y, z = loc[0], loc[1], loc[2]
        r = torch.linalg.norm(loc) + 1e-8
        elev = torch.asin(torch.clamp(z / r, -1.0, 1.0))
        return elev * (180.0 / math.pi)

    # ----------------- transforms: c2w / w2c -----------------

    def get_c2w(self) -> torch.Tensor:
        R_cam2world = rodrigues(self.rotvec)  # (3, 3)
        C = self.location                     # (3,)

        device = R_cam2world.device
        dtype = R_cam2world.dtype

        c2w = torch.eye(4, device=device, dtype=dtype)
        c2w[:3, :3] = R_cam2world
        c2w[:3, 3] = C
        return c2w

    @property
    def c2w(self) -> torch.Tensor:
        return self.get_c2w()

    def get_w2c(self) -> torch.Tensor:
        R_cam2world = rodrigues(self.rotvec)  # (3, 3)
        C = self.location                     # (3,)

        device = R_cam2world.device
        dtype = R_cam2world.dtype

        R_w2c = R_cam2world.transpose(0, 1)
        t_w2c = -R_w2c @ C

        w2c = torch.eye(4, device=device, dtype=dtype)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = t_w2c
        return w2c

    @property
    def w2c(self) -> torch.Tensor:
        return self.get_w2c()

    # ----------------- projection & MVP -----------------

    def get_proj_mtx(
        self,
        image_size=None,
        near: Optional[float] = None,
        far: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Single 4x4 orthographic projection matrix for this camera.
        """
        if image_size is None:
            image_size = self.image_size
        if near is None:
            near = self.near
        if far is None:
            far = self.far

        P = get_ortho_projection_matrix_from_scale(
            batch_size=1,
            ortho_scale=self.ortho_scale,
            image_size=image_size,
            near=near,
            far=far,
            device=self.theta.device,
        )
        return P[0]  # (4, 4)

    @property
    def proj_mtx(self) -> torch.Tensor:
        return self.get_proj_mtx()

    def get_mvp_mtx(
        self,
        image_size=None,
        near: Optional[float] = None,
        far: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Model-View-Projection: mvp = proj_mtx @ w2c
        (assuming model matrix = identity).
        """
        P = self.get_proj_mtx(image_size=image_size, near=near, far=far)
        V = self.get_w2c()
        return P @ V

    @property
    def mvp_mtx(self) -> torch.Tensor:
        return self.get_mvp_mtx()

    # ----------------- NEW: projection_matrix method -----------------

    def get_projection_matrix(
        self,
        batch_size: int = 1,
        image_size=None,
        near: Optional[float] = None,
        far: Optional[float] = None,
    ) -> torch.Tensor:
        """
        MVAdapter-style projection matrix method.

        Returns a [batch_size, 4, 4] orthographic projection matrix,
        using the same convention as get_orthogonal_projection_matrix
        in the external repo, parameterised by this camera's
        ortho_scale, image_size, near, far.

        If batch_size > 1, the same matrix is broadcasted along batch.
        """
        if image_size is None:
            image_size = self.image_size
        if near is None:
            near = self.near
        if far is None:
            far = self.far

        # base [1, 4, 4] projection matrix
        base_P = get_ortho_projection_matrix_from_scale(
            batch_size=1,
            ortho_scale=self.ortho_scale,
            image_size=image_size,
            near=near,
            far=far,
            device=self.theta.device,
        )  # (1, 4, 4)

        if batch_size == 1:
            return base_P
        else:
            return base_P.repeat(batch_size, 1, 1)

    @property
    def projection_matrix(self) -> torch.Tensor:
        """
        Convenience alias: same as get_projection_matrix(batch_size=1).
        Shape: (1, 4, 4)
        """
        return self.get_projection_matrix(batch_size=1)

    # ----------------- projection & MVP -----------------

    def get_proj_mtx(
        self,
        image_size=None,
        near: Optional[float] = None,
        far: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Single 4x4 orthographic projection matrix for this camera.
        """
        if image_size is None:
            image_size = self.image_size
        if near is None:
            near = self.near
        if far is None:
            far = self.far

        P = get_ortho_projection_matrix_from_scale(
            batch_size=1,
            ortho_scale=self.ortho_scale,
            image_size=image_size,
            near=near,
            far=far,
            device=self.theta.device,
        )
        return P[0]  # (4, 4)

    @property
    def proj_mtx(self) -> torch.Tensor:
        return self.get_proj_mtx()

    def get_mvp_mtx(
        self,
        image_size=None,
        near: Optional[float] = None,
        far: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Model-View-Projection: mvp = proj_mtx @ w2c
        (assuming model matrix = identity).
        """
        P = self.get_proj_mtx(image_size=image_size, near=near, far=far)
        V = self.get_w2c()
        return P @ V

    @property
    def mvp_mtx(self) -> torch.Tensor:
        return self.get_mvp_mtx()

    # ----------------- NEW: projection_matrix method -----------------

    def get_projection_matrix(
        self,
        batch_size: int = 1,
        image_size=None,
        near: Optional[float] = None,
        far: Optional[float] = None,
    ) -> torch.Tensor:
        """
        MVAdapter-style projection matrix method.

        Returns a [batch_size, 4, 4] orthographic projection matrix,
        using the same convention as get_orthogonal_projection_matrix
        in the external repo, parameterised by this camera's
        ortho_scale, image_size, near, far.

        If batch_size > 1, the same matrix is broadcasted along batch.
        """
        if image_size is None:
            image_size = self.image_size
        if near is None:
            near = self.near
        if far is None:
            far = self.far

        # base [1, 4, 4] projection matrix
        base_P = get_ortho_projection_matrix_from_scale(
            batch_size=1,
            ortho_scale=self.ortho_scale,
            image_size=image_size,
            near=near,
            far=far,
            device=self.theta.device,
        )  # (1, 4, 4)

        if batch_size == 1:
            return base_P
        else:
            return base_P.repeat(batch_size, 1, 1)

    @property
    def projection_matrix(self) -> torch.Tensor:
        """
        Convenience alias: same as get_projection_matrix(batch_size=1).
        Shape: (1, 4, 4)
        """
        return self.get_projection_matrix(batch_size=1)

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    
def convert_image(image):
    if isinstance(image, torch.Tensor):
        image = image.squeeze().squeeze()
        arr = image.cpu().numpy()
        if arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0))
        arr = (arr * 255 if arr.dtype != np.uint8 else arr).astype(np.uint8)
    elif isinstance(image, np.ndarray):
        arr = (image * 255 if image.dtype != np.uint8 else image).astype(np.uint8)
    else:
        return image
    mode = "RGB" 
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            mode = "RGBA"
        elif arr.shape[2] == 3:
            mode = "RGB"
        else:
            mode = "L"
    else:
        mode = "L"

    return Image.fromarray(arr).convert(mode).resize((384, 384))

def show_image(image, title=None, stats=True):
    img = convert_image(image)
    if stats:
        print(f"Image shape: {img.size}")
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def show_images(images, titles=None, ncols=None):
    n = len(images)
    cols = n if ncols is None else ncols
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(min(cols * 8, 32), min(rows * 8, 32)))
    axes = np.array(axes).reshape(-1)
    for idx, img in enumerate(images):
        ax = axes[idx]
        ax.imshow(convert_image(img))
        if titles:
            ax.set_title(titles[idx])
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def show_images_row(images, titles=None):
    show_images(images, titles=titles, ncols=len(images))

def show_images_column(images, titles=None):
    show_images(images, titles=titles, ncols=1)


def load_rgba(path: str) -> Image.Image:
    """Load an image as RGBA (ensures alpha exists)."""
    img = Image.open(path).convert("RGBA")
    return img

def split_rgba(img_rgba: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Return (RGB, A) where A is L-mode alpha."""
    r, g, b, a = img_rgba.split()
    rgb = Image.merge("RGB", (r, g, b))
    return rgb, a

def alpha_to_binary_mask(
    alpha: Image.Image,
    threshold: int = 8,
    close_radius: int = 1,
    feather_px: int = 0,
    invert: bool = False,
) -> Image.Image:
    """
    Convert an alpha channel (L-mode) into a clean binary mask.
    - threshold: alpha > threshold => 255 (foreground), else 0
    - close_radius: morphological close (dilate then erode) radius in pixels to seal pinholes
    - feather_px: optional Gaussian feather for softer transitions (keeps L-mode)
    - invert: if True, swap foreground/background
    Returns: L-mode mask in [0,255]
    """
    # Binarize
    mask = alpha.point(lambda a: 255 if a > threshold else 0, mode="L")

    # Morphological close (approx) via min/max filters
    if close_radius > 0:
        # Dilate
        mask = mask.filter(ImageFilter.MaxFilter(size=2 * close_radius + 1))
        # Erode
        mask = mask.filter(ImageFilter.MinFilter(size=2 * close_radius + 1))

    if feather_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_px))

    if invert:
        mask = ImageOps.invert(mask)

    return mask

def to_3ch(mask_l: Image.Image) -> Image.Image:
    """Duplicate L-mode mask to 3 channels (RGB) for ControlNet convenience."""
    return Image.merge("RGB", (mask_l, mask_l, mask_l))

def prepare_control_image_from_mask(mask_l: Image.Image) -> Image.Image:
    """
    ControlNet generally accepts a PIL image; for a binary mask, we pass it as a 3-channel image.
    Keep it 8-bit; Diffusers will handle normalization internally.
    """
    return to_3ch(mask_l)

def center_crop_resize(img: Image.Image, short_side: int = 1024) -> Image.Image:
    """Center-crop to square and resize to target short side (SDXL friendly)."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    if side != short_side:
        img = img.resize((short_side, short_side), Image.LANCZOS)
    return img


def compute_euler_angle_difference(euler1, euler2, seq='xyz'):
    rot1 = R.from_euler(seq, euler1)
    rot2 = R.from_euler(seq, euler2)
    
    # Compute relative rotation: R_diff = R1_inv * R2
    diff_rot = rot1.inv() * rot2
    
    # The magnitude of the rotation vector of the relative rotation 
    # is the angle of rotation required to align rot1 with rot2.
    return diff_rot.magnitude()

def visualize_3d_landmarks_and_camera(
    landmarks_3d: np.ndarray,
    camera_location: np.ndarray,
    camera_rotation: np.ndarray,
    output_path: str,
    title: str = "3D Landmarks and Camera Position"
):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=landmarks_3d[:, 0],
        y=landmarks_3d[:, 1],
        z=landmarks_3d[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.8
        ),
        name='3D Landmarks',
        text=[f'Point {i}' for i in range(len(landmarks_3d))],
        hovertemplate='<b>%{text}</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[camera_location[0]],
        y=[camera_location[1]],
        z=[camera_location[2]],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='diamond',
            opacity=1.0
        ),
        name='Camera Position',
        text=['Camera'],
        hovertemplate='<b>Camera</b><br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>'
    ))
    
    rotation_matrix = Rotation.from_euler('xyz', camera_rotation).as_matrix()
    arrow_length = 0.3
    
    axes_colors = ['crimson', 'green', 'royalblue']
    axes_names = ['Camera X', 'Camera Y', 'Camera Z']
    
    for i, (color, name) in enumerate(zip(axes_colors, axes_names)):
        direction = rotation_matrix[:, i]
        end_point = camera_location + direction * arrow_length
        
        fig.add_trace(go.Scatter3d(
            x=[camera_location[0], end_point[0]],
            y=[camera_location[1], end_point[1]],
            z=[camera_location[2], end_point[2]],
            mode='lines+markers',
            line=dict(color=color, width=6),
            marker=dict(size=[4, 8], color=color, symbol='diamond'),
            name=name,
            showlegend=True,
            hovertemplate=f'<b>{name}</b><extra></extra>'
        ))
    
    axis_length = 0.5
    world_axes = [
        ([0, axis_length], [0, 0], [0, 0], 'red', 'World X'),
        ([0, 0], [0, axis_length], [0, 0], 'green', 'World Y'),
        ([0, 0], [0, 0], [0, axis_length], 'blue', 'World Z')
    ]
    
    for x, y, z, color, name in world_axes:
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+text',
            line=dict(color=color, width=3, dash='dash'),
            text=['', name],
            textposition='top center',
            textfont=dict(size=12, color=color),
            name=name,
            showlegend=True,
            hovertemplate=f'<b>{name}</b><extra></extra>'
        ))
    
    all_points = np.vstack([landmarks_3d, camera_location.reshape(1, -1)])
    center = all_points.mean(axis=0)
    max_range = np.abs(all_points - center).max() * 1.2
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        scene=dict(
            xaxis=dict(
                title='X',
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
                showbackground=True,
                range=[center[0] - max_range, center[0] + max_range]
            ),
            yaxis=dict(
                title='Y',
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
                showbackground=True,
                range=[center[1] - max_range, center[1] + max_range]
            ),
            zaxis=dict(
                title='Z',
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
                showbackground=True,
                range=[center[2] - max_range, center[2] + max_range]
            ),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        width=1200,
        height=900,
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)
    print(f"3D visualization saved to: {output_path}")

def visualize_projection_on_image(
    image_path: str,
    landmarks_2d_proj: np.ndarray,
    landmarks_2d_pred: np.ndarray,
    point_size: int = 10,
    bbox: dict = None,
    scale_factor: float = 1.0,
):
    img = Image.open(image_path).convert("RGB")
    if img.size != (1024, 1024):
        print(f"Resizing image from {img.size} to (1024, 1024) for visualization.")
        img = img.resize((1024, 1024))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]

    if bbox is not None and scale_factor != 1.0:
        bbox_size = max(bbox['width'], bbox['height'])
        center_x = bbox['center_x']
        center_y = bbox['center_y']
        half_size = bbox_size / 2
        
        x_min = max(0, int(center_x - half_size))
        y_min = max(0, int(center_y - half_size))
        x_max = min(W, int(center_x + half_size))
        y_max = min(H, int(center_y + half_size))
        
        img_cropped = img[y_min:y_max, x_min:x_max]
        img_cropped = cv2.resize(img_cropped, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        out = img_cropped.copy()
        
        landmarks_proj_viz = landmarks_2d_proj
        landmarks_pred_viz = landmarks_2d_pred
    else:
        out = img.copy()
        landmarks_proj_viz = landmarks_2d_proj
        landmarks_pred_viz = landmarks_2d_pred
    
    viz_H, viz_W = out.shape[:2]
    
    for x, y in landmarks_proj_viz:
        if 0 <= x < viz_W and 0 <= y < viz_H:
            cv2.circle(out, (int(x), int(y)), radius=point_size, color=(255, 0, 0), thickness=-1)

    for x, y in landmarks_pred_viz:
        if 0 <= x < viz_W and 0 <= y < viz_H:
            cv2.circle(out, (int(x), int(y)), radius=point_size, color=(0, 0, 255), thickness=-1)

    legend_height = 80
    legend_width = 300
    legend_margin = 20
    legend_y = legend_margin
    legend_x = viz_W - legend_width - legend_margin
    
    overlay = out.copy()
    cv2.rectangle(overlay, 
                  (legend_x - 10, legend_y - 10), 
                  (legend_x + legend_width, legend_y + legend_height), 
                  (255, 255, 255), 
                  -1)
    out = cv2.addWeighted(overlay, 0.7, out, 0.3, 0)
    
    # legend_item_height = 25
    
    # cv2.circle(out, (legend_x + 10, legend_y + 15), radius=point_size, color=(255, 0, 0), thickness=-1)
    # cv2.putText(out, "Projected 3D Landmarks", (legend_x + 30, legend_y + 20), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # cv2.circle(out, (legend_x + 10, legend_y + 15 + legend_item_height), 
    #            radius=point_size, color=(0, 0, 255), thickness=-1)
    # cv2.putText(out, "Detected 2D Landmarks", (legend_x + 30, legend_y + 20 + legend_item_height), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    # error = np.mean(np.linalg.norm(landmarks_proj_viz - landmarks_pred_viz, axis=1))
    # cv2.putText(out, f"Mean Error: {error:.2f}px", 
    #             (legend_x + 10, legend_y + 20 + 2 * legend_item_height), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out)

def estimate_camera_single_run(
    xyz_ref: torch.Tensor,
    uv_input: torch.Tensor,
    init_camera_location: torch.Tensor = torch.tensor([0.0, 0.00, 1.3]),
    init_camera_rotation: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
    init_ortho_scale: float = 1.0,
    num_iters: int = 500,
    lr: float = 0.1,
    lr_step: int = 100,
    lr_gamma: float = 0.5,
    verbose: bool = False,
    width: int = 1024,
    device: str = "cuda",
    early_stopping: bool = True,
    patience: int = 75,
    min_delta: float = 1e-6,
    min_iterations: int = 250,
    ortho_scale_freeze_iters: int = 300,
    debug: bool = False,
    debug_dir: str = None,
    source_image_path: str = None,
    stage_name: str = "optimization",
    bbox: dict = None,
    scale_factor: float = 1.0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if debug and debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        projections_dir = debug_dir / f"{stage_name}_projections"
        projections_dir.mkdir(parents=True, exist_ok=True)
    camera_model = OptimizableCameraModel(
        init_location=init_camera_location,
        init_rotation=init_camera_rotation,
        init_ortho_scale=init_ortho_scale,
        device=device,
    )
    camera_model.to(device)
    optim_params = [
        {'params': camera_model.theta, 'lr': lr},
        {'params': camera_model.phi, 'lr': lr},
        {'params': camera_model.rotvec, 'lr': lr},
        {'params': camera_model.ortho_scale, 'lr': 0.0},
    ]
    optimizer = optim.AdamW(optim_params)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters, eta_min=5e-4)
    losses = []
    xyz_ref = xyz_ref.to(device)
    uv_input = uv_input.to(device)
    print(f"Device: {xyz_ref.device}, dtype: {xyz_ref.dtype}")
    best_loss = float('inf')
    best_params = None
    best_uv_proj = None
    patience_counter = 0
    best_iter = 0
    early_stop_triggered = False
    early_stop_reason = ""
    ortho_scale_unfrozen = False
    for i in range(num_iters):
        if i == ortho_scale_freeze_iters and not ortho_scale_unfrozen:
            for param_group in optimizer.param_groups:
                if param_group['params'][0] is camera_model.ortho_scale:
                    param_group['lr'] = 0.5 * lr
                    ortho_scale_unfrozen = True
                    if verbose:
                        print(f"\n>>> Iteration {i}: Unfreezing ortho_scale optimization (lr={0.5 * lr:.4f})")
                    break
        optimizer.zero_grad()
        width = 1024
        uv_proj = project_points_blender_ortho(
            points_world=xyz_ref,
            cam_location=camera_model.location,
            cam_rotvec=camera_model.rotvec,
            ortho_scale=camera_model.ortho_scale,
            image_size=(width, width),
        )

        l1_loss = F.l1_loss(uv_proj, uv_input.to(device))
        loss = l1_loss
        angle_limit = np.pi / 2
        angle_margin = 0.1
        theta_val = camera_model.theta
        if theta_val < -(angle_limit - angle_margin):
            angle_penalty = 10.0 * (theta_val + angle_limit - angle_margin) ** 2
            loss = loss + angle_penalty
        elif theta_val > (angle_limit - angle_margin):
            angle_penalty = 10.0 * (theta_val - angle_limit + angle_margin) ** 2
            loss = loss + angle_penalty
        phi_val = camera_model.phi
        if phi_val < -(angle_limit - angle_margin):
            angle_penalty = 10.0 * (phi_val + angle_limit - angle_margin) ** 2
            loss = loss + angle_penalty
        elif phi_val > (angle_limit - angle_margin):
            angle_penalty = 10.0 * (phi_val - angle_limit + angle_margin) ** 2
            loss = loss + angle_penalty
        if loss.item() < (best_loss - min_delta):
            best_loss = loss.item()
            best_iter = i
            best_uv_proj = uv_proj.detach().cpu().numpy()
            patience_counter = 0
            with torch.no_grad():
                best_params = {
                    "rotvec": camera_model.rotvec.detach().cpu().numpy().copy(),
                    "location": camera_model.location.detach().cpu().numpy().copy(),
                    "ortho_scale": camera_model.ortho_scale.detach().cpu().numpy().copy(),
                    "theta": camera_model.theta.detach().cpu().numpy().copy(),
                    "phi": camera_model.phi.detach().cpu().numpy().copy(),
                    "sphere_radius": camera_model.sphere_radius.detach().cpu().numpy().copy(),
                }
        else:
            patience_counter += 1
        if debug and debug_dir is not None and source_image_path is not None:
            if i % 500 == 0 or i == num_iters - 1:
                current_uv_proj = uv_proj.detach().cpu().numpy()
                current_uv_input = uv_input.cpu().numpy()
                debug_image = visualize_projection_on_image(
                    image_path=source_image_path,
                    landmarks_2d_proj=current_uv_proj,
                    landmarks_2d_pred=current_uv_input,
                    point_size=3,
                    bbox=bbox,
                    scale_factor=scale_factor,
                )
                debug_image_path = projections_dir / f"iter_{i:05d}_loss_{loss.item():.4f}.png"
                debug_image.save(debug_image_path)
                if verbose:
                    print(f"  Debug: Saved projection to {debug_image_path.name}")
        if early_stopping and i >= max(min_iterations, ortho_scale_freeze_iters):
            if patience_counter >= patience:
                early_stop_triggered = True
                early_stop_reason = f"No improvement for {patience} iterations"
                if verbose:
                    print(f"Early stopping at iteration {i}: {early_stop_reason}")
                    print(f"Best loss: {best_loss:.6f} at iteration {best_iter}")
                break
            if len(losses) >= 20:
                recent_losses = losses[-20:]
                loss_std = np.std(recent_losses)
                if loss_std < min_delta * 0.1:
                    early_stop_triggered = True
                    early_stop_reason = f"Loss converged (std: {loss_std:.2e})"
                    if verbose:
                        print(f"Early stopping at iteration {i}: {early_stop_reason}")
                        print(f"Best loss: {best_loss:.6f} at iteration {best_iter}")
                    break
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if verbose and (i % 50 == 0 or i == num_iters - 1 or early_stop_triggered):
            status = f" [EARLY STOP: {early_stop_reason}]" if early_stop_triggered else ""
            print(f"Iter {i:4d}: Loss = {loss.item():.6f}, Best = {best_loss:.6f}{status}")
    final_message = f"Training completed after {i + 1} iterations"
    if early_stop_triggered:
        final_message += f" (early stopping: {early_stop_reason})"
    if verbose:
        print(final_message)
        print(f"Best loss during optimization: {best_loss:.6f} at iteration {best_iter}")
        print(f"Final rotation: {best_params['rotvec']}")
        print(f"Final location (Cartesian): {best_params['location']}")
        print(f"Final location (Spherical): theta={best_params['theta']:.4f} rad ({np.degrees(best_params['theta']):.2f}°), "
              f"phi={best_params['phi']:.4f} rad ({np.degrees(best_params['phi']):.2f}°), r={best_params['sphere_radius']:.2f}")
        print(f"Final orthogonal scale: {best_params['ortho_scale']}")
    best_params["loss_history"] = np.array(losses)
    best_params["best_loss"] = best_loss
    best_params["best_iteration"] = best_iter
    best_params["total_iterations"] = i + 1
    best_params["early_stop_triggered"] = early_stop_triggered
    best_params["early_stop_reason"] = early_stop_reason
    return best_params, best_uv_proj

def estimate_camera_multiple_runs(
    xyz_ref: torch.Tensor,
    uv_input: torch.Tensor,
    init_camera_locations: list,
    init_camera_rotations: list,
    init_ortho_scale: float = 1.0,
    num_iters: int = 500,
    lr: float = 0.1,
    lr_step: int = 100,
    lr_gamma: float = 0.5,
    verbose: bool = False,
    width: int = 1024,
    device: str = "cuda",
    early_stopping: bool = True,
    patience: int = 75,
    min_delta: float = 1e-6,
    min_iterations: int = 250,
    debug: bool = False,
    debug_dir: str = None,
    source_image_path: str = None,
    stage_name: str = "optimization",
    bbox: dict = None,
    scale_factor: float = 1.0,
):
    if len(init_camera_locations) != len(init_camera_rotations):
        raise ValueError(
            f"Number of locations ({len(init_camera_locations)}) must match "
            f"number of rotations ({len(init_camera_rotations)})"
        )
    if len(init_camera_locations) == 0:
        raise ValueError("At least one initialization pair must be provided")
    best_overall_loss = float('inf')
    best_overall_params = None
    best_overall_uv_proj = None
    best_run_idx = -1
    print(f"\n{'='*80}")
    print(f"Running {len(init_camera_locations)} optimization runs with different initializations")
    print(f"{'='*80}\n")
    for run_idx, (init_location, init_rotation) in enumerate(
        zip(init_camera_locations, init_camera_rotations)
    ):
        if not isinstance(init_location, torch.Tensor):
            init_location = torch.tensor(init_location, dtype=torch.float32)
        if not isinstance(init_rotation, torch.Tensor):
            init_rotation = torch.tensor(init_rotation, dtype=torch.float32)
        print(f"\n--- Run {run_idx + 1}/{len(init_camera_locations)} ---")
        print(f"Initial location: {init_location.cpu().numpy()}")
        print(f"Initial rotation: {init_rotation.cpu().numpy()}")
        run_debug_dir = None
        run_stage_name = stage_name
        if debug and debug_dir is not None:
            run_debug_dir = f"{debug_dir}/run_{run_idx:02d}"
            run_stage_name = f"{stage_name}_run_{run_idx:02d}"
        try:
            params, uv_proj = estimate_camera_single_run(
                xyz_ref=xyz_ref,
                uv_input=uv_input,
                init_camera_location=init_location,
                init_camera_rotation=init_rotation,
                init_ortho_scale=init_ortho_scale,
                num_iters=num_iters,
                lr=lr,
                lr_step=lr_step,
                lr_gamma=lr_gamma,
                verbose=verbose,
                width=width,
                device=device,
                early_stopping=early_stopping,
                patience=patience,
                min_delta=min_delta,
                min_iterations=min_iterations,
                debug=debug,
                debug_dir=run_debug_dir,
                source_image_path=source_image_path,
                stage_name=run_stage_name,
                bbox=bbox,
                scale_factor=scale_factor,
            )
            current_loss = params['best_loss']
            print(f"Run {run_idx + 1} completed with loss: {current_loss:.6f}")
            if current_loss < best_overall_loss:
                best_overall_loss = current_loss
                best_overall_params = params
                best_overall_uv_proj = uv_proj
                best_run_idx = run_idx
                print(f">>> New best loss: {best_overall_loss:.6f} (Run {run_idx + 1})")
        except Exception as e:
            print(f"Run {run_idx + 1} failed with error: {e}")
            continue
    if best_overall_params is None:
        raise RuntimeError("All optimization runs failed")
    print(f"\n{'='*80}")
    print(f"Best result from Run {best_run_idx + 1}/{len(init_camera_locations)}")
    print(f"Best loss: {best_overall_loss:.6f}")
    print(f"Best location: {best_overall_params['location']}")
    print(f"Best rotation: {best_overall_params['rotvec']}")
    print(f"Best ortho scale: {best_overall_params['ortho_scale']}")
    print(f"{'='*80}\n")
    best_overall_params['num_runs'] = len(init_camera_locations)
    best_overall_params['best_run_idx'] = best_run_idx
    return best_overall_params, best_overall_uv_proj

def convert_mediapipe_to_dlib68(lmks_mp: np.ndarray) -> np.ndarray:
    MP2DLIB_CORRESPONDENCE = [
        [127], [234], [93], [132, 58], [58, 172], [136], [150], [176], [152],
        [400], [379], [365], [397, 288], [361], [323], [454], [356],
        [70], [63], [105], [66], [107],
        [336], [296], [334], [293], [300],
        [168, 6], [197, 195], [5], [4], [75], [97], [2], [326], [305],
        [33], [160], [158], [133], [153], [144],
        [362], [385], [387], [263], [373], [380],
        [61], [39], [37], [0], [267], [269], [291],
        [321], [314], [17], [84], [91],
        [78], [82], [13], [312], [308],
        [317], [14], [87],
    ]
    mp2dlib_correspondence_normalized = []
    for indices in MP2DLIB_CORRESPONDENCE:
        if len(indices) == 1:
            mp2dlib_correspondence_normalized.append([indices[0], indices[0]])
        else:
            mp2dlib_correspondence_normalized.append(indices)
    lmks_dlib = np.array([
        lmks_mp[indices].mean(axis=0) 
        for indices in mp2dlib_correspondence_normalized
    ])
    return lmks_dlib

def compute_landmarks_bbox(landmarks_2d: np.ndarray, padding: int = 128) -> dict:
    x_min = np.min(landmarks_2d[:, 0])
    x_max = np.max(landmarks_2d[:, 0])
    y_min = np.min(landmarks_2d[:, 1])
    y_max = np.max(landmarks_2d[:, 1])
    x_min = max(0, x_min - padding)
    x_max = x_max + padding
    y_min = max(0, y_min - padding)
    y_max = y_max + padding
    width = x_max - x_min
    height = y_max - y_min
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return {
        'x_min': x_min,
        'y_min': y_min,
        'x_max': x_max,
        'y_max': y_max,
        'width': width,
        'height': height,
        'center_x': center_x,
        'center_y': center_y,
    }

def crop_and_resize_landmarks(
    landmarks_2d: np.ndarray,
    bbox: dict,
    target_size: int = 1024,
    original_size: int = 1024
) -> tuple:
    bbox_size = max(bbox['width'], bbox['height'])
    center_x = bbox['center_x']
    center_y = bbox['center_y']
    half_size = bbox_size / 2
    x_min = center_x - half_size
    y_min = center_y - half_size
    x_max = center_x + half_size
    y_max = center_y + half_size
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(original_size, x_max)
    y_max = min(original_size, y_max)
    actual_bbox_size = max(x_max - x_min, y_max - y_min)
    landmarks_cropped = landmarks_2d.copy()
    landmarks_cropped[:, 0] -= x_min
    landmarks_cropped[:, 1] -= y_min
    scale_factor = target_size / actual_bbox_size
    landmarks_resized = landmarks_cropped * scale_factor
    return landmarks_resized, scale_factor

def adjust_ortho_scale_for_crop(ortho_scale: float, scale_factor: float) -> float:
    adjusted_scale = ortho_scale * np.sqrt(scale_factor)
    return adjusted_scale

def convert_camera_cv_to_blender(location_cv, euler_cv):
    location_blender = location_cv.copy()
    euler_blender = euler_cv.copy()
    return location_blender, euler_blender


def align_landmarks(
    target_mesh_path: str,
    source_image_path: str,
    view_aligned_dir: str,
    source_lmk_data: dict,
    source_rotation_euler_rad,
    lmk_3d_vertex_indices,
    init_camera_location=[0.0, 0.0, 3.0],
    init_camera_rotation_euler=[0.0, 0.0, 0.0],
    target_size: int = 1024,
    target_rotation_euler_rad=[0.0, 0.0, 0.0],
    frontalize_target: bool = False,
    debug=False,
):
    # Ensure PIL is available
    from PIL import Image
    
    target_mesh_aligned_to_source_path = Path(view_aligned_dir).parent / "aligned_target_mesh.glb"
    alignment_data_dir = Path(view_aligned_dir) / "alignment"
    alignment_data_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = None
    if debug:
        debug_dir = Path(view_aligned_dir) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"Debug mode enabled. Debug files will be saved to: {debug_dir}")
    
    # Apply rotations based on frontalize_target flag
    if frontalize_target:
        # First, frontalize the target by applying inverse rotation
        frontalized_mesh_path = Path(view_aligned_dir).parent / "frontalized_target_mesh.glb"
        print(f"Frontalizing target mesh with inverse rotation: {target_rotation_euler_rad}")
        _, frontalized_mesh = rotate_glb_mesh(
            input_glb_path=target_mesh_path,
            euler_angles_rad=target_rotation_euler_rad,
            output_glb_path=str(frontalized_mesh_path),
            rotate_fn=apply_inverse_rotation
        )
        
        # Then rotate to align with source
        print(f"Aligning frontalized mesh to source with rotation: {source_rotation_euler_rad}")
        _, source_aligned_target_mesh = rotate_glb_mesh(
            input_glb_path=str(frontalized_mesh_path),
            euler_angles_rad=source_rotation_euler_rad,
            output_glb_path=str(target_mesh_aligned_to_source_path),
            rotate_fn=apply_rotation
        )
    else:
        # Just rotate to align with source
        _, source_aligned_target_mesh = rotate_glb_mesh(
            input_glb_path=target_mesh_path,
            euler_angles_rad=source_rotation_euler_rad,
            output_glb_path=str(target_mesh_aligned_to_source_path),
            rotate_fn=apply_rotation
        )

    source_aligned_target_mesh = trimesh.load(target_mesh_aligned_to_source_path)

    # If the loaded object is a Scene, dump the geometry into a single mesh
    if isinstance(source_aligned_target_mesh, trimesh.Scene):
        # Fix: Access .geometry directly from the scene object instead of trimesh.graph
        source_aligned_target_mesh = trimesh.util.concatenate(tuple(source_aligned_target_mesh.geometry.values()))

    target_lmk_3d_478 = source_aligned_target_mesh.vertices[lmk_3d_vertex_indices.cpu().numpy()]
    target_lmk_3d_478 = target_lmk_3d_478[:, [0, 2, 1]] * np.array([1, -1, 1])
    # target_lmk_3d_478 = source_aligned_target_mesh.vertices[lmk_3d_vertex_indices.cpu().numpy()]
    source_lmk_2d_478_raw = source_lmk_data["ldm478"]
    
    # Load source image to get actual dimensions
    with Image.open(source_image_path) as src_img:
        src_w, src_h = src_img.size
        print(f"\nSource Image Dimensions: {src_w}x{src_h}")
    
    # Use landmarks in original image coordinates (no cropping)
    source_lmk_2d_478 = source_lmk_2d_478_raw.copy()
    
    print(f"\nLandmark preprocessing:")
    print(f"  Using original image size: {src_w}x{src_h}")
    print(f"  No cropping or resizing applied")
    
    source_lmk_2d_68 = convert_mediapipe_to_dlib68(source_lmk_2d_478)
    target_lmk_3d_68 = convert_mediapipe_to_dlib68(target_lmk_3d_478)
    
    if debug and debug_dir is not None:
        visualize_3d_landmarks_and_camera(
            landmarks_3d=target_lmk_3d_68,
            camera_location=np.array(init_camera_location),
            camera_rotation=np.array(init_camera_rotation_euler),
            output_path=str(debug_dir / "stage1_3d_landmarks_and_camera_initial.html"),
            title="Stage 1: 68 Landmarks and Initial Camera Position"
        )
    
    print("="*80)
    print("STAGE 1: Coarse alignment using 68 facial landmarks (orthographic)")
    print("="*80)
    
    stage1_init_locations = [
        [0.0, 0.0, 4.0],
        [2.0, 0.0, 3.0],
        [-2.0, 0.0, 3.0],
        # [0.0, -1.0, 3.0],
        # [0.0, 1.0, 3.0],
        # [2.0, 1.0, 3.0],
        # [-2.0, -1.0, 3.0],
        # [2.0, -1.0, 3.0],
        # [-2.0, 1.0, 3.0],
        # [0.0, 0.0, 3.0],
    ]
    
    stage1_init_rotations = []
    for loc in stage1_init_locations:
        x, y, z = loc
        dx, dy, dz = -x, -y, -z
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        if length > 0:
            dx, dy, dz = dx/length, dy/length, dz/length
        yaw = np.arctan2(-dx, -dz)
        pitch = np.arctan2(-dy, np.sqrt(dx**2 + dz**2))
        roll = 0.0
        stage1_init_rotations.append([pitch, yaw, roll])
    print("\nInitialization parameters:")
    for i, (loc, rot) in enumerate(zip(stage1_init_locations, stage1_init_rotations)):
        print(f"  Init {i+1}: location={loc}, rotation={[f'{r:.4f}' for r in rot]} rad "
              f"({[f'{np.degrees(r):.2f}' for r in rot]} deg)")
    
    stage1_params, stage1_projected_uv = estimate_camera_multiple_runs(
        xyz_ref=torch.tensor(target_lmk_3d_68, dtype=torch.float32),
        uv_input=torch.tensor(source_lmk_2d_68, dtype=torch.float32),
        init_camera_locations=stage1_init_locations,
        init_camera_rotations=stage1_init_rotations,
        init_ortho_scale=1.0,
        num_iters=3000,
        lr=0.05,
        width=src_w,
        verbose=True,
        early_stopping=True,
        patience=200,
        min_delta=1e-4,
        min_iterations=800,
        debug=debug,
        debug_dir=str(debug_dir) if debug_dir is not None else None,
        source_image_path=source_image_path,
        stage_name="stage1",
        bbox=None,
        scale_factor=1.0,
    )
    
   
    print(f"Stage 1 complete. Best loss: {stage1_params['best_loss']:.6f} (from run {stage1_params['best_run_idx'] + 1}/{stage1_params['num_runs']})")
    
    if debug and debug_dir is not None:
        visualize_3d_landmarks_and_camera(
            landmarks_3d=target_lmk_3d_68,
            camera_location=stage1_params['location'],
            camera_rotation=stage1_params['rotvec'],
            output_path=str(debug_dir / "stage1_3d_landmarks_and_camera_optimized.html"),
            title=f"Stage 1: 68 Landmarks and Optimized Camera (Loss: {stage1_params['best_loss']:.4f})"
        )
    
    stage1_projection_image = visualize_projection_on_image(
        image_path=source_image_path,
        landmarks_2d_proj=stage1_projected_uv,
        landmarks_2d_pred=source_lmk_2d_68,
        point_size=6,
        bbox=None,
        scale_factor=1.0,
    )
    stage1_vis_path = Path(view_aligned_dir) / "alignment" / "stage1_landmark_projection_68.png"
    stage1_projection_image.save(stage1_vis_path)
    print(f"Stage 1 visualization saved to: {stage1_vis_path}")
    
    if debug and debug_dir is not None:
        visualize_3d_landmarks_and_camera(
            landmarks_3d=target_lmk_3d_478,
            camera_location=stage1_params['location'],
            camera_rotation=stage1_params['rotvec'],
            output_path=str(debug_dir / "stage2_3d_landmarks_and_camera_initial.html"),
            title="Stage 2: 478 Landmarks and Stage 1 Camera Position"
        )
    
    print("\n" + "="*80)
    print("STAGE 2: Fine-tuning with all 478 landmarks (orthographic)")
    print("="*80)
    
    # Add small noise to stage1 parameters for better exploration
    stage1_location = torch.tensor(stage1_params['location'], dtype=torch.float32)
    stage1_rotvec = torch.tensor(stage1_params['rotvec'], dtype=torch.float32)
    stage1_ortho_scale = float(stage1_params['ortho_scale'])
    
    # Add small Gaussian noise
    location_noise = torch.randn_like(stage1_location) * 0.05  # 5% position noise
    rotvec_noise = torch.randn_like(stage1_rotvec) * 0.02  # ~1 degree rotation noise
    ortho_scale_noise = np.random.randn() * 0.02  # 2% scale noise
    
    noisy_location = stage1_location + location_noise
    noisy_rotvec = stage1_rotvec + rotvec_noise
    noisy_ortho_scale = stage1_ortho_scale * (1.0 + ortho_scale_noise)
    
    print(f"\nAdding noise to stage1 parameters:")
    print(f"  Location: {stage1_location.numpy()} -> {noisy_location.numpy()}")
    print(f"  Rotvec: {stage1_rotvec.numpy()} -> {noisy_rotvec.numpy()}")
    print(f"  Ortho scale: {stage1_ortho_scale:.4f} -> {noisy_ortho_scale:.4f}")
    
    best_camera_params, projected_uv = estimate_camera_single_run(
        xyz_ref=torch.tensor(target_lmk_3d_478, dtype=torch.float32),
        uv_input=torch.tensor(source_lmk_2d_478, dtype=torch.float32),
        num_iters=1500,
        lr=0.025,
        width=src_w,
        verbose=True,
        early_stopping=True,
        patience=100,
        min_iterations=600,
        init_camera_location=noisy_location,
        init_camera_rotation=noisy_rotvec,
        init_ortho_scale=noisy_ortho_scale,
        debug=debug,
        debug_dir=str(debug_dir) if debug_dir is not None else None,
        source_image_path=source_image_path,
        stage_name="stage2",
        bbox=None,
        scale_factor=1.0,
        min_delta=1e-4,
    )

    print(f"Stage 2 complete. Final loss: {best_camera_params['best_loss']:.6f}")
    print("="*80)
    print(f"Improvement: {stage1_params['best_loss']:.6f} -> {best_camera_params['best_loss']:.6f}")
    print("="*80)
        
    if debug and debug_dir is not None:
        visualize_3d_landmarks_and_camera(
            landmarks_3d=target_lmk_3d_478,
            camera_location=best_camera_params['location'],
            camera_rotation=best_camera_params['rotvec'],
            output_path=str(debug_dir / "stage2_3d_landmarks_and_camera_optimized.html"),
            title=f"Stage 2: 478 Landmarks and Final Optimized Camera (Loss: {best_camera_params['best_loss']:.4f})"
        )
    lmk_projection_image = visualize_projection_on_image(
        image_path=source_image_path,
        landmarks_2d_proj=projected_uv,
        landmarks_2d_pred=source_lmk_2d_478,
        point_size=2,
        bbox=None,
        scale_factor=1.0,
    )
    camera_params_path = Path(view_aligned_dir) / "camera_params.json"
    camera_params_to_save = {
        "location": best_camera_params["location"].tolist(),
        "rotvec": best_camera_params["rotvec"].tolist(),
        "ortho_scale": float(best_camera_params["ortho_scale"]),
        "source_dimensions": {"width": src_w, "height": src_h},
        "stage1_loss": float(stage1_params['best_loss']),
        "stage2_loss": float(best_camera_params['best_loss']),
    }
    with open(camera_params_path, 'w') as f:
        json.dump(camera_params_to_save, f, indent=2)
    lmk_projection_path = Path(view_aligned_dir) / "alignment" / "stage2_landmark_projection_478.png"
    lmk_projection_image.save(lmk_projection_path)
    print(f"Stage 2 visualization saved to: {lmk_projection_path}")

    return best_camera_params