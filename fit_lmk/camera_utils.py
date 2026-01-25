import torch
import numpy as np
from typing import Union, Tuple


def euler_to_rotation_matrix(rx: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor) -> torch.Tensor:
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    device = rx.device
    dtype = rx.dtype
    
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    one = torch.tensor(1.0, device=device, dtype=dtype)
    
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

    return Rz @ (Ry @ Rx)


def get_sensor_fit(sensor_fit: str, size_x: float, size_y: float) -> str:
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
    if device is None:
        if isinstance(ortho_scale, torch.Tensor):
            device = ortho_scale.device
            dtype = ortho_scale.dtype
        else:
            device = torch.device('cpu')
    
    if not isinstance(ortho_scale, torch.Tensor):
        ortho_scale = torch.tensor(ortho_scale, device=device, dtype=dtype)
    if not isinstance(shift_x, torch.Tensor):
        shift_x = torch.tensor(shift_x, device=device, dtype=dtype)
    if not isinstance(shift_y, torch.Tensor):
        shift_y = torch.tensor(shift_y, device=device, dtype=dtype)
    
    scale = resolution_percentage / 100.0
    res_x_px = resolution_x * scale
    res_y_px = resolution_y * scale
    pixel_aspect = pixel_aspect_y / pixel_aspect_x

    fit = get_sensor_fit(sensor_fit, res_x_px, res_y_px)

    if fit == 'HORIZONTAL':
        view_fac_px = res_x_px
    else:
        view_fac_px = res_y_px / pixel_aspect

    pixel_size_world = ortho_scale / view_fac_px
    s_u = 1.0 / pixel_size_world
    s_v = s_u / pixel_aspect

    u_0 = res_x_px * 0.5 - shift_x * view_fac_px
    v_0 = res_y_px * 0.5 + shift_y * view_fac_px / pixel_aspect
    skew = torch.tensor(0.0, device=device, dtype=dtype)

    K = torch.stack([
        torch.stack([s_u, skew, u_0]),
        torch.stack([torch.tensor(0.0, device=device, dtype=dtype), s_v, v_0]),
        torch.stack([torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(1.0, device=device, dtype=dtype)])
    ])

    return K


def get_3x4_RT_matrix(
    cam_loc: torch.Tensor,
    cam_rot: torch.Tensor
) -> torch.Tensor:
    device = cam_loc.device
    dtype = cam_loc.dtype
    
    R_bcam2cv = torch.tensor([
        [1, 0,  0],
        [0,-1,  0],
        [0, 0, -1]
    ], device=device, dtype=dtype)

    R_world2bcam = euler_to_rotation_matrix(cam_rot[0], cam_rot[1], cam_rot[2]).T
    T_world2bcam = -(R_world2bcam @ cam_loc)

    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    RT = torch.zeros((3, 4), device=device, dtype=dtype)
    RT[:3, :3] = R_world2cv
    RT[:3, 3] = T_world2cv

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
    if device is None:
        if isinstance(cam_loc, torch.Tensor):
            device = cam_loc.device
            dtype = cam_loc.dtype
        elif isinstance(cam_rot, torch.Tensor):
            device = cam_rot.device
            dtype = cam_rot.dtype
        else:
            device = torch.device('cpu')
    
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


def generate_mesh_rotations(
    num_views: int = 4,
    angle_range: float = 0.65,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generate rotation angles for the mesh (around Y-axis) for multi-view rendering.
    
    Args:
        num_views: Number of additional views (excluding the frontal view)
        angle_range: Maximum rotation angle in radians
        device: torch device
        dtype: torch dtype
    
    Returns:
        Tensor of rotation angles with shape (num_views + 1,) where first angle is 0.0 (frontal)
    """
    if device is None:
        device = torch.device('cpu')
    
    # Start with frontal view (0 rotation)
    rotations = [torch.tensor(0.0, device=device, dtype=dtype)]
    
    # Generate perturbed rotations around Y-axis
    angles = torch.linspace(-angle_range, angle_range, num_views, device=device, dtype=dtype)
    
    for angle in angles:
        rotations.append(angle)
    
    return torch.stack(rotations)


def generate_perturbed_cameras(
    cam_loc: torch.Tensor,
    cam_rot: torch.Tensor,
    num_views: int = 4,
    angle_range: float = 0.15,
    trans_range: float = 0.05,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Deprecated: Use generate_mesh_rotations instead.
    This function is kept for backward compatibility.
    """
    if device is None:
        device = cam_loc.device
        dtype = cam_loc.dtype
    
    cam_locs = [cam_loc]
    cam_rots = [cam_rot]
    
    angles = torch.linspace(-angle_range, angle_range, num_views, device=device, dtype=dtype)
    
    for i, angle in enumerate(angles):
        if i % 2 == 0:
            delta_rot = torch.tensor([0.0, angle, 0.0], device=device, dtype=dtype)
        else:
            delta_rot = torch.tensor([angle, 0.0, 0.0], device=device, dtype=dtype)
        
        new_rot = cam_rot + delta_rot
        
        trans_scale = (i + 1) / num_views * trans_range
        delta_trans = torch.tensor([
            trans_scale * (0.5 - (i % 2)),
            0.0,
            trans_scale * (0.5 - (i // 2 % 2))
        ], device=device, dtype=dtype)
        
        new_loc = cam_loc + delta_trans
        
        cam_locs.append(new_loc)
        cam_rots.append(new_rot)
    
    return torch.stack(cam_locs), torch.stack(cam_rots)
