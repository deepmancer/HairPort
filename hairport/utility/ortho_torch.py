"""
--------------------------------------------------------------------
3×4 calibration matrix (intrinsic K · extrinsic [R | t]) for a
**orthographic** camera - Pure Python implementation with PyTorch
--------------------------------------------------------------------
Author : AI Assistant
Date   : 2025-07-06 (Updated: 2025-11-10)
Notes  :
  • Coordinate convention: Standard computer vision system
      - Camera coordinate system: +X right, +Y down, +Z forward (into scene)
      - A camera at [0, 0, z] with rotation [0, 0, 0] looks toward the origin
      - Rotation is specified as Euler angles (rx, ry, rz) in XYZ order
  • In orthographic mode the field of view is governed by
      ortho_scale (world-units across the fitting axis)
    instead of lens focal length, and the mapping from world
    units → pixels is therefore linear (no perspective divide).
  • Pure Python implementation without bpy dependency
  • Uses PyTorch for proper gradient flow and device management
--------------------------------------------------------------------
"""

import torch
import math
from typing import Union, Tuple


def euler_to_matrix(rx: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor, order: str = "XYZ") -> torch.Tensor:
    """
    Build a 3×3 rotation matrix from Euler angles (rx, ry, rz) in radians.
    Blender's default order is XYZ: rotate about X, then Y, then Z.
    """
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    # Get device and dtype from input tensors
    device = rx.device
    dtype = rx.dtype
    
    # Create zero and one tensors
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    one = torch.tensor(1.0, device=device, dtype=dtype)
    
    # single-axis rotations
    Rx = torch.stack([
        torch.stack([one, zero, zero]),
        torch.stack([zero, cx, -sx]),
        torch.stack([zero, sx, cx])
    ])
    
    Ry = torch.stack([
        torch.stack([cy, zero, sy]),
        torch.stack([zero, one, zero]),
        torch.stack([-sy, zero, cy])
    ])
    
    Rz = torch.stack([
        torch.stack([cz, -sz, zero]),
        torch.stack([sz, cz, zero]),
        torch.stack([zero, zero, one])
    ])

    # apply in X→Y→Z order:  v′ = Rz ( Ry ( Rx v ) )
    return Rz @ (Ry @ Rx)


def get_sensor_fit(sensor_fit: str, size_x: float, size_y: float) -> str:
    """Blender's BKE_camera_sensor_fit logic."""
    if sensor_fit == "AUTO":
        return "HORIZONTAL" if size_x >= size_y else "VERTICAL"
    return sensor_fit


def get_calibration_matrix_K_ortho(
    ortho_scale: Union[float, torch.Tensor],
    resolution_x: int,
    resolution_y: int,
    sensor_fit: str = "AUTO",
    shift_x: Union[float, torch.Tensor] = 0.0,
    shift_y: Union[float, torch.Tensor] = 0.0,
    resolution_percentage: float = 100.0,
    pixel_aspect_x: float = 1.0,
    pixel_aspect_y: float = 1.0,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build the 3×3 intrinsic matrix for an ORTHO camera.

    K = [[s_u,  skew, u_0],
         [  0,   s_v, v_0],
         [  0,     0,   1]]

    where  s_u , s_v  convert world‐units to pixels and (u₀,v₀)
    is the principal point expressed in pixels.
    """
    # Determine device and dtype
    if device is None:
        if isinstance(ortho_scale, torch.Tensor):
            device = ortho_scale.device
            dtype = ortho_scale.dtype
        elif isinstance(shift_x, torch.Tensor):
            device = shift_x.device
            dtype = shift_x.dtype
        elif isinstance(shift_y, torch.Tensor):
            device = shift_y.device
            dtype = shift_y.dtype
        else:
            device = torch.device('cpu')
    
    # Convert to tensors if needed
    if not isinstance(ortho_scale, torch.Tensor):
        ortho_scale = torch.tensor(ortho_scale, device=device, dtype=dtype)
    if not isinstance(shift_x, torch.Tensor):
        shift_x = torch.tensor(shift_x, device=device, dtype=dtype)
    if not isinstance(shift_y, torch.Tensor):
        shift_y = torch.tensor(shift_y, device=device, dtype=dtype)
    
    # 1. Image resolution & pixel aspect
    scale = resolution_percentage / 100.0
    res_x_px = resolution_x * scale
    res_y_px = resolution_y * scale
    pixel_aspect = pixel_aspect_y / pixel_aspect_x

    # 2. Which dimension (H / V) should match ortho_scale ?
    fit = get_sensor_fit(sensor_fit, res_x_px, res_y_px)

    if fit == 'HORIZONTAL':
        view_fac_px = res_x_px                         # <─ matches ortho_scale
    else:  # 'VERTICAL'
        view_fac_px = pixel_aspect * res_y_px          # <─ matches ortho_scale

    # 3. Convert world-units → pixels
    pixel_size_world = ortho_scale / view_fac_px      # world-units per pixel
    s_u = 1.0 / pixel_size_world                      # px / wu  (X axis)
    s_v = s_u / pixel_aspect                          # px / wu  (Y axis)

    # 4. Principal point + lens shifts
    u_0 = res_x_px * 0.5 - shift_x * view_fac_px
    v_0 = res_y_px * 0.5 + shift_y * view_fac_px / pixel_aspect
    skew = torch.tensor(0.0, device=device, dtype=dtype)  # square pixels assumed

    K = torch.stack([
        torch.stack([s_u, skew, u_0]),
        torch.stack([torch.tensor(0.0, device=device, dtype=dtype), s_v, v_0]),
        torch.stack([torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)])
    ])

    return K


def get_view_matrix(
    cam_loc: torch.Tensor,
    cam_rot: torch.Tensor
) -> torch.Tensor:
    """
    Build the 4×4 view matrix: world → camera.
    
    Convention: Camera at [0, 0, z] with rotation [0, 0, 0] looks toward the origin.
    Camera coordinate system: +X right, +Y down, +Z forward (into scene).
    
    Args:
        cam_loc: (3,) tensor - camera position in world coordinates
        cam_rot: (3,) tensor - Euler angles in radians (XYZ order)
    
    Returns:
        4×4 view matrix transforming world coordinates to camera coordinates
    """
    device = cam_loc.device
    dtype = cam_loc.dtype
    
    # Camera-to-world rotation
    R_cam2world = euler_to_matrix(cam_rot[0], cam_rot[1], cam_rot[2])
    
    # World-to-camera rotation and translation
    R_world2cam = R_cam2world.T
    t_world2cam = -R_world2cam @ cam_loc

    # Assemble 4×4 view matrix
    view = torch.eye(4, device=device, dtype=dtype)
    view[:3, :3] = R_world2cam
    view[:3, 3] = t_world2cam
    return view


def get_coord_transform(device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Identity transform - no coordinate system change needed.
    Camera convention: +X right, +Y down, +Z forward (standard computer vision).
    """
    return torch.eye(4, device=device, dtype=dtype)


def get_projection_matrix_ortho_opengl(
    K: torch.Tensor,
    width: int,
    height: int,
    ortho_scale: Union[float, torch.Tensor],
    near: float = 0.001,
    far: float = 10.0
) -> torch.Tensor:
    """
    Build a 4×4 OpenGL-style orthographic projection matrix.
    
    For orthographic projection:
    - No perspective divide (w component remains 1)
    - Linear mapping from world coordinates to NDC
    - Uses ortho_scale to determine the view volume
    """
    device = K.device
    dtype = K.dtype
    
    # Convert ortho_scale to tensor if needed
    if not isinstance(ortho_scale, torch.Tensor):
        ortho_scale = torch.tensor(ortho_scale, device=device, dtype=dtype)
    
    # Calculate the orthographic bounds
    # ortho_scale represents the world-space size across the fitting dimension
    aspect_ratio = width / height
    
    if width >= height:  # HORIZONTAL fit
        left = -ortho_scale * 0.5
        right = ortho_scale * 0.5
        bottom = -ortho_scale * 0.5 / aspect_ratio
        top = ortho_scale * 0.5 / aspect_ratio
    else:  # VERTICAL fit
        left = -ortho_scale * 0.5 * aspect_ratio
        right = ortho_scale * 0.5 * aspect_ratio
        bottom = -ortho_scale * 0.5
        top = ortho_scale * 0.5
    
    # Build orthographic projection matrix
    P = torch.zeros((4, 4), device=device, dtype=dtype)
    P[0, 0] = 2.0 / (right - left)
    P[0, 3] = -(right + left) / (right - left)
    P[1, 1] = 2.0 / (top - bottom)
    P[1, 3] = -(top + bottom) / (top - bottom)
    P[2, 2] = -2.0 / (far - near)
    P[2, 3] = -(far + near) / (far - near)
    P[3, 3] = 1.0
    
    return P


def get_3x4_RT_matrix(
    cam_loc: torch.Tensor,
    cam_rot: torch.Tensor
) -> torch.Tensor:
    """
    Extrinsics [R | t] that map world → camera space.
    
    Convention: Camera at [0, 0, z] with rotation [0, 0, 0] looks toward the origin.
    - Camera coordinate system: +X right, +Y down, +Z forward (into the scene)
    - Rotation [0, 0, 0] means camera's +Z axis points toward -Z in world space
    """
    device = cam_loc.device
    dtype = cam_loc.dtype
    
    # Camera-to-world rotation from Euler angles
    R_cam2world = euler_to_matrix(cam_rot[0], cam_rot[1], cam_rot[2])
    
    # World-to-camera is the inverse
    R_world2cam = R_cam2world.T
    T_world2cam = -(R_world2cam @ cam_loc)

    # Assemble 3x4 [R | t] matrix
    RT = torch.zeros((3, 4), device=device, dtype=dtype)
    RT[:3, :3] = R_world2cam
    RT[:3, 3] = T_world2cam

    return RT


def get_3x4_P_matrix_ortho(
    cam_loc: Union[Tuple[float, float, float], torch.Tensor],
    cam_rot: Union[Tuple[float, float, float], torch.Tensor],
    ortho_scale: Union[float, torch.Tensor],
    resolution_x: int,
    resolution_y: int,
    sensor_fit: str = "AUTO",
    shift_x: Union[float, torch.Tensor] = 0.0,
    shift_y: Union[float, torch.Tensor] = 0.0,
    resolution_percentage: float = 100.0,
    pixel_aspect_x: float = 1.0,
    pixel_aspect_y: float = 1.0,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        P  – 3×4 projection matrix (intrinsic·extrinsic)
        K  – 3×3 intrinsics
        RT – 3×4 extrinsics
    """
    # Determine device and dtype
    if device is None:
        if isinstance(cam_loc, torch.Tensor):
            device = cam_loc.device
            dtype = cam_loc.dtype
        elif isinstance(cam_rot, torch.Tensor):
            device = cam_rot.device
            dtype = cam_rot.dtype
        elif isinstance(ortho_scale, torch.Tensor):
            device = ortho_scale.device
            dtype = ortho_scale.dtype
        else:
            device = torch.device('cpu')
    
    # Convert to tensors if needed
    if not isinstance(cam_loc, torch.Tensor):
        cam_loc = torch.tensor(cam_loc, device=device, dtype=dtype)
    if not isinstance(cam_rot, torch.Tensor):
        cam_rot = torch.tensor(cam_rot, device=device, dtype=dtype)
    
    K = get_calibration_matrix_K_ortho(
        ortho_scale=ortho_scale,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        sensor_fit=sensor_fit,
        shift_x=shift_x,
        shift_y=shift_y,
        resolution_percentage=resolution_percentage,
        pixel_aspect_x=pixel_aspect_x,
        pixel_aspect_y=pixel_aspect_y,
        device=device,
        dtype=dtype,
    )
    
    RT = get_3x4_RT_matrix(cam_loc, cam_rot)
    P = K @ RT
    
    return P, K, RT


def orthogonal_projection(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def get_calib_mat_ortho(
    cam_loc: Union[Tuple[float, float, float], torch.Tensor],
    cam_rot: Union[Tuple[float, float, float], torch.Tensor],
    ortho_scale: Union[float, torch.Tensor] = 1.0,
    input_size: int = 1024,
    sensor_fit: str = "AUTO",
    shift_x: Union[float, torch.Tensor] = 0.0,
    shift_y: Union[float, torch.Tensor] = 0.0,
    resolution_percentage: float = 100.0,
    pixel_aspect_x: float = 1.0,
    pixel_aspect_y: float = 1.0,
    near: float = 0.001,
    far: float = 100.0,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    to_blender: bool = True,
) -> torch.Tensor:
    """
    Pure-Python orthographic camera calibration matrix computation.
    Returns the 4×4 matrix that maps world coords → NDC for orthographic projection.
    
    Coordinate Convention:
    ----------------------
    - Camera coordinate system: +X right, +Y down, +Z forward (into scene)
    - A camera at position [0, 0, z] with rotation [0, 0, 0] looks toward the origin
    - Rotation euler angles (rx, ry, rz) are in radians, applied in XYZ order
    
    Example:
    --------
    Camera at [0, 0, 2] with rotation [0, 0, 0]:
      - Position: 2 units along +Z axis
      - Looking direction: toward origin (along -Z in world, +Z in camera)
      - Up direction: +Y in world space
    
    Args:
        cam_loc: Camera location (x, y, z) in world coordinates
        cam_rot: Camera rotation (rx, ry, rz) in radians, Euler XYZ order
        ortho_scale: Orthographic scale (world units across fitting axis)
        input_size: Resolution (assuming square images)
        sensor_fit: 'AUTO', 'HORIZONTAL', or 'VERTICAL'
        shift_x, shift_y: Lens shift parameters
        resolution_percentage: Render resolution percentage
        pixel_aspect_x, pixel_aspect_y: Pixel aspect ratios
        near, far: Near and far clipping planes
        device: PyTorch device
        dtype: PyTorch data type
        to_blender: Legacy parameter (deprecated, kept for compatibility)
    
    Returns:
        4×4 tensor that maps world coordinates to normalized device coordinates
    """
    # Determine device and dtype
    if device is None:
        if isinstance(cam_loc, torch.Tensor):
            device = cam_loc.device
            dtype = cam_loc.dtype
        elif isinstance(cam_rot, torch.Tensor):
            device = cam_rot.device
            dtype = cam_rot.dtype
        elif isinstance(ortho_scale, torch.Tensor):
            device = ortho_scale.device
            dtype = ortho_scale.dtype
        else:
            device = torch.device('cpu')
    
    # Convert to tensors if needed
    if not isinstance(cam_loc, torch.Tensor):
        cam_loc = torch.tensor(cam_loc, device=device, dtype=dtype)
    if not isinstance(cam_rot, torch.Tensor):
        cam_rot = torch.tensor(cam_rot, device=device, dtype=dtype)

    # 1) Intrinsics for orthographic camera
    K = get_calibration_matrix_K_ortho(
        ortho_scale=ortho_scale,
        resolution_x=input_size,
        resolution_y=input_size,
        sensor_fit=sensor_fit,
        shift_x=shift_x,
        shift_y=shift_y,
        resolution_percentage=resolution_percentage,
        pixel_aspect_x=pixel_aspect_x,
        pixel_aspect_y=pixel_aspect_y,
        device=device,
        dtype=dtype,
    )

    # 2) Orthographic projection matrix
    P_proj = get_projection_matrix_ortho_opengl(
        K, input_size, input_size, ortho_scale, near, far
    )

    # 3) View matrix
    V = get_view_matrix(cam_loc, cam_rot)

    # 4) Compose: P_final = P_proj @ V
    # Note: to_blender parameter is deprecated but kept for compatibility
    P_final = P_proj @ V
    
    return P_final
