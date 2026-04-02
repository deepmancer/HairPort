import argparse
import bpy
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import AutoencoderKL, DDPMScheduler, LCMScheduler, UNet2DConditionModel
from torchvision import transforms
# Diffusers
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionXLImg2ImgPipeline,
)

import random
import sys

from hairport.config import get_config

_cfg = get_config()
_mvadapter_path = str(_cfg.paths.mv_adapter_module)
if _mvadapter_path not in sys.path:
    sys.path.insert(0, _mvadapter_path)

from mvadapter.models.attention_processor import (
    DecoupledMVRowSelfAttnProcessor2_0,
    DecoupledMVRowColSelfAttnProcessor2_0,
)
from mvadapter.pipelines.pipeline_mvadapter_i2mv_sdxl import MVAdapterI2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils import make_image_grid, tensor_to_image
from mvadapter.utils.mesh_utils import (
    NVDiffRastContextWrapper,
    TexturedMesh,
    Camera,
    render,
    RenderOutput,
    get_orthogonal_projection_matrix,
    get_orthogonal_camera,
)

from hairport.core.bg_remover import BackgroundRemover


@dataclass
class TexturedViewConfig:
    """Configuration for generating a textured view."""
    # Model paths
    base_model: str | None = None
    vae_model: str | None = None
    adapter_path: str | None = None
    adapter_weight_name: str | None = None
    lora_models: Optional[List[str]] = None
    lora_scales: Optional[List[float]] = None

    # Generation parameters
    num_views: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    reference_conditioning_scale: float | None = None
    control_conditioning_scale: float | None = None
    negative_prompt: str | None = None

    # Output dimensions
    height: int | None = None
    width: int | None = None

    # Device settings
    device: str | None = None
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        cfg = get_config()
        rv = cfg.render_view
        # Model paths
        if self.base_model is None:
            self.base_model = cfg.models.realvis_v4
        if self.vae_model is None:
            self.vae_model = cfg.models.sdxl_vae
        if self.adapter_path is None:
            self.adapter_path = cfg.models.mv_adapter
        if self.adapter_weight_name is None:
            self.adapter_weight_name = cfg.models.mv_adapter_weight
        if self.lora_models is None:
            self.lora_models = [
                str(Path(cfg.paths.mv_adapter_module) / rv.lora_dir / f)
                for f in rv.lora_files
            ]
        if self.lora_scales is None:
            self.lora_scales = list(rv.lora_scales)
        # Generation parameters
        if self.num_views is None:
            self.num_views = rv.num_views
        if self.num_inference_steps is None:
            self.num_inference_steps = rv.num_inference_steps
        if self.guidance_scale is None:
            self.guidance_scale = rv.guidance_scale
        if self.reference_conditioning_scale is None:
            self.reference_conditioning_scale = rv.reference_conditioning_scale
        if self.control_conditioning_scale is None:
            self.control_conditioning_scale = rv.control_conditioning_scale
        if self.negative_prompt is None:
            self.negative_prompt = cfg.prompts.render_view_negative
        # Output dimensions
        if self.height is None:
            self.height = rv.height
        if self.width is None:
            self.width = rv.width
        # Device
        if self.device is None:
            self.device = cfg.device



from hairport.utility.estimate_camera import (
    OptimizableCameraModel,
    rodrigues,
    project_points_blender_ortho,
)
from scipy.spatial.transform import Rotation

BLENDER_TO_NVDIFFRAST_ROT_TORCH = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=torch.float32,
)
BLENDER_TO_NVDIFFRAST_ROT_NP = BLENDER_TO_NVDIFFRAST_ROT_TORCH.cpu().numpy()

NVDIFFRAST_TO_BLENDER_ROT_TORCH = BLENDER_TO_NVDIFFRAST_ROT_TORCH.t().contiguous()
NVDIFFRAST_TO_BLENDER_ROT_NP = NVDIFFRAST_TO_BLENDER_ROT_TORCH.cpu().numpy()

def _transform_position_blender_to_nvdiffrast(pos: np.ndarray) -> np.ndarray:
    rotation_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(np.pi / 2), -np.sin(np.pi / 2)],
        [0.0, np.sin(np.pi / 2), np.cos(np.pi / 2)]
    ])
    pos = rotation_x @ pos
    return pos


def _transform_rotvec_blender_to_nvdiffrast(rotvec: np.ndarray) -> np.ndarray:
    R_b = Rotation.from_rotvec(rotvec).as_matrix()
    
    rotation_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(np.pi / 2), -np.sin(np.pi / 2)],
        [0.0, np.sin(np.pi / 2), np.cos(np.pi / 2)]
    ])
    
    R_n = rotation_x @ R_b
    
    return Rotation.from_matrix(R_n).as_rotvec()


def _convert_cameras_nvdiffrast_to_blender(cameras: Camera) -> Camera:
    device = cameras.c2w.device
    dtype = cameras.c2w.dtype

    rot_bn = BLENDER_TO_NVDIFFRAST_ROT_TORCH.to(device=device, dtype=dtype)
    rot_nb = NVDIFFRAST_TO_BLENDER_ROT_TORCH.to(device=device, dtype=dtype)

    R_bn_4 = torch.eye(4, device=device, dtype=dtype)
    R_nb_4 = torch.eye(4, device=device, dtype=dtype)
    R_bn_4[:3, :3] = rot_bn
    R_nb_4[:3, :3] = rot_nb

    R_bn_4 = R_bn_4.unsqueeze(0)
    R_nb_4 = R_nb_4.unsqueeze(0)

    w2c_b = torch.matmul(cameras.w2c, R_bn_4)

    c2w_b = torch.matmul(R_nb_4, cameras.c2w)

    proj = cameras.proj_mtx
    mvp_b = torch.matmul(proj, w2c_b)

    cam_pos_b = c2w_b[:, :3, 3]

    cameras.c2w = c2w_b
    cameras.w2c = w2c_b
    cameras.mvp_mtx = mvp_b
    cameras.cam_pos = cam_pos_b

    return cameras


@dataclass 
class CameraParams:
    location: np.ndarray  # (3,) camera position in world space (NVDiffrast coords)
    rotvec: np.ndarray    # (3,) rotation vector (Rodrigues, NVDiffrast coords)
    ortho_scale: float    # orthographic scale
    
    @classmethod
    def from_optimizable_camera(cls, camera: OptimizableCameraModel) -> "CameraParams":
        """Create CameraParams from an OptimizableCameraModel instance."""
        with torch.no_grad():
            return cls(
                location=camera.location.detach().cpu().numpy(),
                rotvec=camera.rotvec.detach().cpu().numpy(),
                ortho_scale=float(camera.ortho_scale.detach().cpu().item()),
            )
    
    @classmethod
    def from_dict(cls, params_dict: Dict, from_blender: bool = False) -> "CameraParams":
        location = np.array(params_dict["location"])
        rotvec = np.array(params_dict["rotvec"])
        ortho_scale = float(params_dict["ortho_scale"])
        
        if from_blender:
            # Convert from Blender coords to NVDiffrast coords
            location = _transform_position_blender_to_nvdiffrast(location)
            rotvec = _transform_rotvec_blender_to_nvdiffrast(rotvec)

        return cls(
            location=location,
            rotvec=rotvec,
            ortho_scale=ortho_scale + 0.2,  # Clamp to reasonable range
        )
    
    @classmethod
    def from_blender_dict(cls, params_dict: Dict) -> "CameraParams":
        return cls.from_dict(params_dict, from_blender=True)
    
    def to_nvdiffrast_coords(self) -> "CameraParams":
        return CameraParams(
            location=_transform_position_blender_to_nvdiffrast(self.location),
            rotvec=_transform_rotvec_blender_to_nvdiffrast(self.rotvec),
            ortho_scale=self.ortho_scale,
        )


def flush_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    torch.cuda.empty_cache()


def create_camera_from_params(
    camera_params: CameraParams,
    image_size: Tuple[int, int],
    near: float | None = None,
    far: float | None = None,
    device: str | None = None,
) -> Camera:
    cfg = get_config()
    if near is None:
        near = cfg.render_view.camera_near
    if far is None:
        far = cfg.render_view.camera_far
    if device is None:
        device = cfg.device
    width, height = image_size
    
    # Convert numpy to torch tensors
    location = torch.tensor(camera_params.location, dtype=torch.float32, device=device)
    rotvec = torch.tensor(camera_params.rotvec, dtype=torch.float32, device=device)
    
    ortho_scale = torch.tensor(camera_params.ortho_scale, dtype=torch.float32, device=device)
    
    # Compute rotation matrix from rodrigues vector
    R_cam2world = rodrigues(rotvec)  # (3, 3)
    
    # Build c2w (camera-to-world) matrix
    c2w = torch.eye(4, dtype=torch.float32, device=device)
    c2w[:3, :3] = R_cam2world
    c2w[:3, 3] = location
    
    # Compute w2c (world-to-camera) matrix
    w2c = torch.linalg.inv(c2w)
    
    # Compute orthographic projection matrix
    aspect = width / height
    if aspect >= 1.0:
        half_w = ortho_scale / 2.0
        half_h = half_w / aspect
    else:
        half_h = ortho_scale / 2.0
        half_w = half_h * aspect
    
    left = -half_w
    right = half_w
    bottom = -half_h
    top = half_h
    
    proj_mtx = get_orthogonal_projection_matrix(
        batch_size=1,
        left=float(left),
        right=float(right),
        bottom=float(bottom),
        top=float(top),
        near=near,
        far=far,
        device=device,
    )[0]  # Remove batch dimension
    
    # Compute MVP matrix
    mvp_mtx = proj_mtx @ w2c
    
    # Add batch dimension for compatibility
    return Camera(
        c2w=c2w.unsqueeze(0),
        w2c=w2c.unsqueeze(0),
        proj_mtx=proj_mtx.unsqueeze(0),
        mvp_mtx=mvp_mtx.unsqueeze(0),
        cam_pos=location.unsqueeze(0),
    )

# Standard 6-view configuration for MVAdapter in NVDiffrast coords (+Z fwd, +Y up).
# These azimuth/elevation presets define cameras in NVDiffrast/OpenGL convention.
# When rendering a mesh from Blender, we rotate these cameras into Blender coords.
STANDARD_AZIMUTH_DEG = [x - 90 for x in [0, 90, 180, 270, 180, 180]]
STANDARD_ELEVATION_DEG = [0, 0, 0, 0, 89.99, -89.99]
STANDARD_DISTANCE = [1.8] * 6
STANDARD_ORTHO_LEFT = -0.55
STANDARD_ORTHO_RIGHT = 0.55
STANDARD_ORTHO_BOTTOM = -0.55
STANDARD_ORTHO_TOP = 0.55


def compute_camera_azimuth_elevation(camera_params: CameraParams) -> Tuple[float, float]:
    location = camera_params.location
    
    # Compute distance from origin
    distance = np.linalg.norm(location)
    if distance < 1e-6:
        return 0.0, 0.0
    
    # Normalize location
    loc_normalized = location / distance
    
    # Elevation: angle from XY plane using Z component.
    elevation_rad = np.arcsin(np.clip(loc_normalized[2], -1, 1))
    elevation_deg = np.degrees(elevation_rad)
    
    # Azimuth: angle in XY plane from X axis
    azimuth_rad = np.arctan2(loc_normalized[1], loc_normalized[0])
    azimuth_deg = np.degrees(azimuth_rad)
    
    return azimuth_deg, elevation_deg


def find_closest_view_index(camera_params: CameraParams) -> int:
    azimuth_deg, elevation_deg = compute_camera_azimuth_elevation(camera_params)
    
    # Only consider side views (indices 0-3) for replacement
    # Top (4) and bottom (5) are special views
    min_distance = float('inf')
    closest_idx = 0
    
    for i in range(4):  # Only first 4 views (side views)
        std_azimuth = STANDARD_AZIMUTH_DEG[i]
        std_elevation = STANDARD_ELEVATION_DEG[i]
        
        # Angular distance (considering wrap-around for azimuth)
        azimuth_diff = abs(azimuth_deg - std_azimuth)
        azimuth_diff = min(azimuth_diff, 360 - azimuth_diff)
        elevation_diff = abs(elevation_deg - std_elevation)
        
        # Combined angular distance
        distance = np.sqrt(azimuth_diff**2 + elevation_diff**2)
        
        if distance < min_distance:
            min_distance = distance
            closest_idx = i
    
    return closest_idx


def create_6view_cameras_with_custom(
    camera_params: CameraParams,
    image_size: Tuple[int, int],
    device: str = "cuda",
    mesh_in_blender_coords: bool = True,
) -> Tuple[Camera, int]:
    # 1. Get standard views in NVDiffrast coords
    azimuths = list(STANDARD_AZIMUTH_DEG)
    elevations = list(STANDARD_ELEVATION_DEG)
    
    # 2. Find closest standard view to input (in NVDiffrast coords)
    closest_idx = find_closest_view_index(camera_params)
    
    # 3. Swap closest to index 0 so input camera takes the first slot
    # This ensures the first view is the one closest to the input
    azimuths[0], azimuths[closest_idx] = azimuths[closest_idx], azimuths[0]
    elevations[0], elevations[closest_idx] = elevations[closest_idx], elevations[0]
    
    # 4. Create the exact input camera using create_camera_from_params
    # This preserves the rotvec and exact transformation from Blender
    input_camera = create_camera_from_params(
        camera_params=camera_params,
        image_size=image_size,
        device=device,
    )
    input_azimuth, input_elevation = compute_camera_azimuth_elevation(camera_params)
    azimuths[0] = input_azimuth
    elevations[0] = input_elevation
    # 5. Prepare arguments for get_orthogonal_camera for the remaining 5 views
    dist = np.linalg.norm(camera_params.location)
    ortho_scale = camera_params.ortho_scale
    
    # Calculate bounds from ortho_scale
    aspect = image_size[0] / image_size[1]
    if aspect >= 1.0:
        half_w = ortho_scale / 2.0
        half_h = half_w / aspect
    else:
        half_h = ortho_scale / 2.0
        half_w = half_h * aspect
    
    left = -half_w
    right = half_w
    bottom = -half_h
    top = half_h
    
    # Create standard cameras for views 1-5 (excluding view 0 which is the input camera)
    standard_cameras = get_orthogonal_camera(
        elevation_deg=elevations[1:],  # Skip first view
        azimuth_deg=azimuths[1:],
        distance=[dist] * 5,  # Only 5 views now
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        device=device,
    )

    # 6. If the mesh is in Blender coords, convert both cameras to that space
    if mesh_in_blender_coords:
        input_camera = _convert_cameras_nvdiffrast_to_blender(input_camera)
        standard_cameras = _convert_cameras_nvdiffrast_to_blender(standard_cameras)

    # cameras = standard_cameras
    
    # 7. Concatenate input camera (view 0) with standard cameras (views 1-5)
    cameras = Camera(
        c2w=torch.cat([input_camera.c2w, standard_cameras.c2w], dim=0),
        w2c=torch.cat([input_camera.w2c, standard_cameras.w2c], dim=0),
        proj_mtx=torch.cat([input_camera.proj_mtx, standard_cameras.proj_mtx], dim=0),
        mvp_mtx=torch.cat([input_camera.mvp_mtx, standard_cameras.mvp_mtx], dim=0),
        cam_pos=torch.cat([input_camera.cam_pos, standard_cameras.cam_pos], dim=0),
    )

    # The input/closest view is at index 0
    return cameras, 0


def load_mesh_from_blender(
    mesh_path: str,
    device: str = "cuda",
) -> TexturedMesh:
    # Clear existing scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import GLB file
    bpy.ops.import_scene.gltf(filepath=mesh_path)
    
    # Get the imported mesh object
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if not mesh_objects:
        raise ValueError(f"No mesh found in {mesh_path}")
    
    # Deselect all first
    bpy.ops.object.select_all(action='DESELECT')
    
    # If multiple meshes, combine them
    if len(mesh_objects) > 1:
        bpy.context.view_layer.objects.active = mesh_objects[0]
        for obj in mesh_objects:
            obj.select_set(True)
        bpy.ops.object.join()
        mesh_obj = bpy.context.view_layer.objects.active
    else:
        mesh_obj = mesh_objects[0]
        bpy.context.view_layer.objects.active = mesh_obj
        mesh_obj.select_set(True)
    
    mesh = mesh_obj.data
    
    # Triangulate using bmesh (more reliable than operator in this context)
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    
    # Extract vertex positions
    vertices = np.zeros((len(mesh.vertices), 3), dtype=np.float32)
    for i, v in enumerate(mesh.vertices):
        vertices[i] = v.co
    
    # Extract face indices
    faces = np.zeros((len(mesh.polygons), 3), dtype=np.int64)
    for i, poly in enumerate(mesh.polygons):
        if len(poly.vertices) != 3:
            raise ValueError(f"Face {i} is not a triangle after triangulation")
        faces[i] = poly.vertices
    
    # Convert to torch tensors
    v_pos = torch.from_numpy(vertices).float()
    t_pos_idx = torch.from_numpy(faces).long()
    
    # Extract UV coordinates and texture
    v_tex = None
    t_tex_idx = None
    texture = None
    
    if mesh.uv_layers.active:
        uv_layer = mesh.uv_layers.active.data
        
        # Build UV coordinate list (one UV per loop/corner)
        uv_coords = np.zeros((len(uv_layer), 2), dtype=np.float32)
        for i, uv in enumerate(uv_layer):
            uv_coords[i] = [uv.uv[0], 1.0 - uv.uv[1]]  # Flip V coordinate
        
        # Create per-vertex UV mapping (map loop indices to vertex indices)
        # We need to handle the fact that UVs are per-loop, not per-vertex
        loop_to_vertex = np.zeros(len(mesh.loops), dtype=np.int32)
        for i, loop in enumerate(mesh.loops):
            loop_to_vertex[i] = loop.vertex_index
        
        # Create UV indices that match face topology
        uv_indices = np.zeros((len(mesh.polygons), 3), dtype=np.int64)
        loop_idx = 0
        for poly_idx, poly in enumerate(mesh.polygons):
            for corner in range(len(poly.vertices)):
                uv_indices[poly_idx, corner] = poly.loop_start + corner
            loop_idx += len(poly.vertices)
        
        v_tex = torch.from_numpy(uv_coords).float()
        t_tex_idx = torch.from_numpy(uv_indices).long()
        
        # Extract texture image if available
        if mesh_obj.active_material and mesh_obj.active_material.use_nodes:
            mat = mesh_obj.active_material
            nodes = mat.node_tree.nodes
            
            # Find the base color texture node
            for node in nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    img = node.image
                    # Get image pixels
                    width, height = img.size
                    pixels = np.array(img.pixels[:]).reshape((height, width, 4))
                    # Flip vertically (Blender uses bottom-left origin)
                    pixels = np.flipud(pixels)
                    # Convert to RGB and torch tensor
                    texture = torch.from_numpy(pixels[:, :, :3].copy()).float()
                    break
    
    # Create TexturedMesh instance
    textured_mesh = TexturedMesh(
        v_pos=v_pos,
        t_pos_idx=t_pos_idx,
        v_tex=v_tex,
        t_tex_idx=t_tex_idx,
        texture=texture,
    )
    
    # Set stitched mesh (using same vertices/faces since Blender handles merging)
    textured_mesh.set_stitched_mesh(v_pos, t_pos_idx)
    
    # Move to device
    textured_mesh.to(device)
    
    return textured_mesh


def cleanup_blender_scene():
    """Clear all objects, meshes, materials, and images from Blender to free memory."""
    # Select all objects and delete them
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clear orphaned data blocks
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
    
    # Purge orphaned data
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)


def load_textured_mesh(
    mesh_path: str,
    rescale: bool = True,
    scale: float = 0.5,
    device: str = "cuda",
    align_coordinate_system: bool = True,
) -> TexturedMesh:
    # Load mesh using Blender API (keeps mesh in Blender coordinate system)
    textured_mesh = load_mesh_from_blender(
        mesh_path=mesh_path,
        device=device,
    )

    # NOTE: Mesh is kept in Blender coords; cameras will be transformed to match.
    return textured_mesh


def render_textured_view(
    mesh: TexturedMesh,
    camera: Camera,
    height: int,
    width: int,
    device: str = "cuda",
    render_texture: bool = True,
) -> Tuple[RenderOutput, Optional[Image.Image]]:
    ctx = NVDiffRastContextWrapper(device=device, context_type="cuda")
    
    # Render geometry (position and normals)
    render_out = render(
        ctx,
        mesh,
        camera,
        height=height,
        width=width,
        render_attr=render_texture and mesh.texture is not None,
        render_depth=True,
        render_normal=True,
        normal_background=0.0,
    )
    
    # Convert textured render to PIL image if available
    textured_image = None
    if render_out.attr is not None:
        # Squeeze batch dimension (1, H, W, C) -> (H, W, C) for tensor_to_image
        attr_squeezed = render_out.attr.squeeze(0).clamp(0, 1)
        textured_image = tensor_to_image(attr_squeezed, batched=False)
    
    return render_out, textured_image


def prepare_control_images(
    render_out: RenderOutput,
    device: str = "cuda",
) -> torch.Tensor:
    # Normalize position to [0, 1] range
    pos_normalized = (render_out.pos + 0.5).clamp(0, 1)  # (1, H, W, 3)
    
    # Normalize normal to [0, 1] range
    normal_normalized = (render_out.normal / 2 + 0.5).clamp(0, 1)  # (1, H, W, 3)
    
    # Concatenate and permute to (1, 6, H, W)
    control_images = torch.cat([pos_normalized, normal_normalized], dim=-1)  # (1, H, W, 6)
    control_images = control_images.permute(0, 3, 1, 2).to(device)  # (1, 6, H, W)
    
    return control_images


def preprocess_reference_image(
    image: Union[str, Path, Image.Image],
    height: int,
    width: int,
) -> Image.Image:
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    
    if image.mode == "RGBA":
        image_np = np.array(image)
        alpha = image_np[..., 3] > 0
        H, W = alpha.shape
        
        # Get bounding box of alpha
        y, x = np.where(alpha)
        if len(y) == 0 or len(x) == 0:
            # No alpha, just resize
            return image.convert("RGB").resize((width, height))
        
        y0, y1 = max(y.min() - 1, 0), min(y.max() + 1, H)
        x0, x1 = max(x.min() - 1, 0), min(x.max() + 1, W)
        image_center = image_np[y0:y1, x0:x1]
        
        # Resize to fit 90% of target dimensions
        H_crop, W_crop, _ = image_center.shape
        if H_crop > W_crop:
            new_W = int(W_crop * (height * 0.9) / H_crop)
            new_H = int(height * 0.9)
        else:
            new_H = int(H_crop * (width * 0.9) / W_crop)
            new_W = int(width * 0.9)
        
        image_center = np.array(Image.fromarray(image_center).resize((new_W, new_H)))
        
        # Pad to target size
        start_h = (height - new_H) // 2
        start_w = (width - new_W) // 2
        padded = np.zeros((height, width, 4), dtype=np.uint8)
        padded[start_h:start_h + new_H, start_w:start_w + new_W] = image_center
        
        # Composite on gray background
        padded = padded.astype(np.float32) / 255.0
        rgb = padded[:, :, :3] * padded[:, :, 3:4] + (1 - padded[:, :, 3:4]) * 0.5
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(rgb)
    else:
        return image.convert("RGB").resize((width, height))


class TexturedViewGenerator:
    def __init__(
        self,
        config: Optional[TexturedViewConfig] = None,
        load_pipeline: bool = True,
    ):
        self.config = config or TexturedViewConfig()
        self.pipe = None
        self.ctx = None
        
        if load_pipeline:
            self.load_pipeline()
            
        self.bg_remover = BackgroundRemover(device=self.config.device)
    
    def load_pipeline(self):
        config = self.config
        
        # Prepare pipeline kwargs
        pipe_kwargs = {}
        if config.vae_model is not None:
            pipe_kwargs["vae"] = AutoencoderKL.from_pretrained(config.vae_model)
        
        # Load base pipeline
        self.pipe = MVAdapterI2MVSDXLPipeline.from_pretrained(
            config.base_model, 
            **pipe_kwargs
        )
        
        # Configure scheduler with SNR shift
        # self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        #     self.pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
        # )
        self.pipe.scheduler = ShiftSNRScheduler.from_scheduler(
            self.pipe.scheduler,
            shift_mode="interpolated",
            shift_scale=8.0,
            scheduler_class=None,  # Use default scheduler class
        )

        # Initialize custom adapter for 6 views (standard MVAdapter configuration)
        # Use DecoupledMVRowColSelfAttnProcessor2_0 for proper multi-view attention
        self.pipe.init_custom_adapter(
            num_views=config.num_views,
            self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0,
        )

        # Load adapter weights
        self.pipe.load_custom_adapter(
            config.adapter_path,
            weight_name=config.adapter_weight_name,
        )

        if config.lora_models:
            for lora_path, lora_scale in zip(config.lora_models, config.lora_scales):
                model_, name_ = lora_path.rsplit("/", 1)
                self.pipe.load_lora_weights(
                    model_,
                    weight_name=name_,
                    scale=lora_scale,
                )

        # Move to device
        self.pipe.to(device=config.device, dtype=config.dtype)
        self.pipe.cond_encoder.to(device=config.device, dtype=config.dtype)
        
        # Enable memory optimization
        self.pipe.enable_vae_slicing()
        
        # Initialize rasterization context
        self.ctx = NVDiffRastContextWrapper(device=config.device, context_type="cuda")

    def generate_view(
        self,
        mesh_path: str,
        reference_image: Union[str, Path, Image.Image],
        camera_params: CameraParams,
        person_prompt: str,
        hair_prompt: str,
        seed: int = -1,
        return_intermediates: bool = False,
        blender_coordinate_system: bool = True,
        intermediate_save_dir: Optional[Union[str, Path]] = None,
    ) -> Union[Image.Image, Tuple[Image.Image, Dict]]:
        config = self.config
        # Construct combined prompt
        prompt = f"{person_prompt} 8k, master piece , ultra-detailed, high quality, soft diffuse lighting, uniform illumination, balanced exposure."
        prompt2 = f"{person_prompt} {hair_prompt} 8k, ultra-detailed, high quality, soft diffuse lighting, uniform illumination, balanced exposure."

        
        if isinstance(reference_image, (str, Path)):
            reference_image = Image.open(reference_image).convert("RGB")
        
        reference_image = self.bg_remover.remove_background(reference_image)[0] # RGBA
        
        # Load textured mesh (kept in Blender coordinate system)
        mesh = load_textured_mesh(
            mesh_path,
            device=config.device,
            align_coordinate_system=False,  # No mesh axis correction
        )
    
        # Create 6-view cameras with custom camera replacing the closest standard view.
        # Cameras are built in NVDiffrast coords and rotated into Blender coords when
        # blender_coordinate_system is True.
        cameras, target_view_idx = create_6view_cameras_with_custom(
            camera_params,
            image_size=(config.width, config.height),
            device=config.device,
            mesh_in_blender_coords=blender_coordinate_system,
        )
        
        print(
            f"Custom camera replaces view index {target_view_idx} "
            f"(standard views: front=0, right=1, back=2, left=3, top=4, bottom=5)"
        )
        
        # Render geometry for all 6 views (mesh and cameras are now in the same frame)
        render_out = render(
            self.ctx,
            mesh,
            cameras,
            height=config.height,
            width=config.width,
            render_attr=False,  # Don't render texture for control images
            render_depth=True,
            render_normal=True,
            normal_background=0.0,
        )
        
        # Prepare control images (position + normal) for all 6 views
        # Shape: (6, H, W, 3) for pos and normal
        pos_normalized = (render_out.pos + 0.5).clamp(0, 1)  # (6, H, W, 3)
        normal_normalized = (render_out.normal / 2 + 0.5).clamp(0, 1)  # (6, H, W, 3)

        # Save per-view pos/normal immediately after rendering (before diffusion)
        # Only save if intermediate_save_dir is explicitly provided
        if intermediate_save_dir is not None:
            save_dir_path = Path(intermediate_save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)
            for view_idx in range(pos_normalized.shape[0]):
                tensor_to_image(pos_normalized[view_idx], batched=False).save(
                    save_dir_path / f"pos_view_{view_idx}.png"
                )
                tensor_to_image(normal_normalized[view_idx], batched=False).save(
                    save_dir_path / f"normal_view_{view_idx}.png"
                )
            
            # Save textured mesh (vertices and faces only) as PLY (for debugging)
            ply_path = save_dir_path / "textured_mesh.ply"
            vertices = mesh.v_pos.detach().cpu().numpy()
            vertices_centered = vertices - vertices.mean(axis=0, keepdims=True)
            faces = mesh.t_pos_idx.detach().cpu().numpy()
            
            with open(ply_path, 'w') as f:
                # Write PLY header
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(vertices)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write(f"element face {len(faces)}\n")
                f.write("property list uchar int vertex_indices\n")
                f.write("end_header\n")
                
                # Write vertices
                for v in vertices_centered:
                    f.write(f"{v[0]} {v[1]} {v[2]}\n")
                
                # Write faces
                for face in faces:
                    f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        # Concatenate and permute to (6, 6, H, W)
        control_images = torch.cat([pos_normalized, normal_normalized], dim=-1)  # (6, H, W, 6)
        control_images = control_images.permute(0, 3, 1, 2).to(config.device)  # (6, 6, H, W)
        
        # Preprocess reference image
        ref_image = preprocess_reference_image(
            reference_image,
            height=config.height,
            width=config.width,
        )
        
        # Set up generator for reproducibility
        pipe_kwargs = {}
        if seed != -1 and isinstance(seed, int):
            pipe_kwargs["generator"] = torch.Generator(device=config.device).manual_seed(seed)
        
        # Run the pipeline with 6 views
        output = self.pipe(
            prompt,
            prompt2=prompt2,
            height=config.height,
            width=config.width,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            num_images_per_prompt=config.num_views,  # 6 views
            control_image=control_images,
            control_conditioning_scale=config.control_conditioning_scale,
            reference_image=ref_image,
            reference_conditioning_scale=config.reference_conditioning_scale,
            negative_prompt=config.negative_prompt,
            **pipe_kwargs,
        )

        # Extract the target view (the one with our custom camera)
        generated_image = output.images[target_view_idx]
        
        if return_intermediates:
            # Get position and normal maps for the target view
            pos_target = pos_normalized[target_view_idx]
            normal_target = normal_normalized[target_view_idx]
            
            # Also render textured view for the target camera
            target_camera = cameras[target_view_idx]
            _, textured_render = render_textured_view(
                mesh,
                target_camera,
                height=config.height,
                width=config.width,
                device=config.device,
                render_texture=True,
            )
            
            intermediates = {
                "textured_render": textured_render,
                "position_map": tensor_to_image(pos_target, batched=False),
                "normal_map": tensor_to_image(normal_target, batched=False),
                "reference_image": ref_image,
                "prompt": prompt,
                "target_view_idx": target_view_idx,
                "all_generated_images": output.images,  # All 6 generated views
                "all_pos_images": tensor_to_image(pos_normalized, batched=True),
                "all_normal_images": tensor_to_image(normal_normalized, batched=True),
                "intermediate_save_dir": intermediate_save_dir,  # Pass through for saving
            }
            return generated_image, intermediates
        
        return generated_image
    
    def generate_view_from_dict(
        self,
        mesh_path: str,
        reference_image: Union[str, Path, Image.Image],
        camera_dict: Dict,
        person_prompt: str,
        hair_prompt: str,
        from_blender: bool = True,
        blender_coordinate_system: bool = True,
        intermediate_save_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Union[Image.Image, Tuple[Image.Image, Dict]]:
        camera_params = CameraParams.from_dict(camera_dict, from_blender=from_blender)
        return self.generate_view(
            mesh_path=mesh_path,
            reference_image=reference_image,
            camera_params=camera_params,
            person_prompt=person_prompt,
            hair_prompt=hair_prompt,
            blender_coordinate_system=blender_coordinate_system,
            intermediate_save_dir=intermediate_save_dir,
            **kwargs,
        )
    
    def unload(self):
        """Unload pipeline and free GPU memory."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.ctx is not None:
            del self.ctx
            self.ctx = None
        flush_gpu_memory()


# Convenience function for quick inference
def generate_view_ig2mv(
    mesh_path: str,
    reference_image: Union[str, Path, Image.Image],
    camera_params: Union[CameraParams, Dict],
    person_prompt: str,
    hair_prompt: str,
    config: Optional[TexturedViewConfig] = None,
    seed: int = -1,
    return_intermediates: bool = False,
    from_blender: bool = True,
    blender_coordinate_system: bool = True,
    intermediate_save_dir: Optional[Union[str, Path]] = None,
    generator: Optional[TexturedViewGenerator] = None,
) -> Union[Image.Image, Tuple[Image.Image, Dict]]:
    should_unload = False
    if generator is None:
        generator = TexturedViewGenerator(config=config, load_pipeline=True)
        should_unload = True
    
    try:
        if isinstance(camera_params, dict):
            camera_params = CameraParams.from_dict(camera_params, from_blender=from_blender)
        
        result = generator.generate_view(
            mesh_path=mesh_path,
            reference_image=reference_image,
            camera_params=camera_params,
            person_prompt=person_prompt,
            hair_prompt=hair_prompt,
            seed=seed,
            return_intermediates=return_intermediates,
            blender_coordinate_system=blender_coordinate_system,
            intermediate_save_dir=intermediate_save_dir,
        )
        return result
    finally:
        if should_unload:
            generator.unload()
        cleanup_blender_scene()


def process_view_aligned_folder(
    folder_path: Path,
    data_dir: Path,
    seed: int = -1,
    bald_version: str = "w_seg",
    save_intermediates: bool = True,
    from_blender: bool = True,
    generator: Optional[TexturedViewGenerator] = None,
) -> bool:
    """Process a single view-aligned folder. Returns True if successful."""
    
    # Check for required files
    camera_json_path = folder_path / bald_version / "camera_params.json"
    mesh_path = folder_path / "aligned_target_mesh.glb"
    output_path = folder_path / "alignment" / "target_image.png"
    folder_name = folder_path.name
    
    parent_folder_name = folder_path.parent.name
    
    parts = folder_name.split("_to_")
    if len(parts) != 2:
        print(f"Skipping {folder_name}: invalid folder name format")
        return False
    
    target_id, source_id = parts
    if not camera_json_path.exists():
        return False
    
    if not mesh_path.exists():
        mesh_path = data_dir / "lmk_3d" / parent_folder_name / target_id / "postprocessed_textured_mesh.glb"
        if not mesh_path.exists():
            print(f"Skipping {folder_name}: mesh file not found")
            return False
    
    # Skip if already processed
    if output_path.exists():
        print(f"Skipping {folder_path.name}: already processed")
        return False
    
    # Parse folder name to extract target_id and source_id
    # Format: {target_id}_to_{source_id}
    # Construct paths
    dataset_name = Path(data_dir).name
    print(f"Starting inference on dataset: {dataset_name}")
    
    if dataset_name == "celeba_reduced":
        image_folder = "image_outpainted"
    else:
        image_folder = "image"
    
    image_dir_path = Path(data_dir) / image_folder
    
    reference_image_path = image_dir_path / f"{target_id}.png"
    prompt_path = data_dir / "prompt" / f"{target_id}.json"
    
    if not reference_image_path.exists():
        print(f"Skipping {folder_name}: reference image not found")
        return False
    
    if not prompt_path.exists():
        print(f"Skipping {folder_name}: prompt file not found")
        return False
    
    # Load camera parameters
    with open(camera_json_path, 'r') as f:
        camera_dict = json.load(f)
    
    # Load prompts
    with open(prompt_path, 'r') as f:
        prompt_data = json.load(f)
    person_prompt = prompt_data.get("subject", [{}])[0].get("description", "A high-quality photo of a person")
    person_prompt = person_prompt.replace(" no background", "")
    hair_prompt = prompt_data.get("subject", [{}])[0].get("hair_description", "detailed hair")
    
    print(f"Processing {folder_name}")
    print(f"  Person: {person_prompt}")
    print(f"  Hair: {hair_prompt}")
    
    try:
        result = generate_view_ig2mv(
            mesh_path=str(mesh_path),
            reference_image=str(reference_image_path),
            camera_params=camera_dict,
            person_prompt=person_prompt,
            hair_prompt=hair_prompt,
            seed=seed,
            return_intermediates=save_intermediates,
            from_blender=from_blender,
            blender_coordinate_system=True,
            intermediate_save_dir=None,  # Don't save extra intermediates by default
            generator=generator,
        )
        
        if save_intermediates:
            output_image, intermediates = result
        else:
            output_image = result
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_image.save(output_path)
        print(f"  Generated image saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"  Error processing {folder_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Generate textured views for all view-aligned folders")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="outputs/",
        help="Root data directory containing view_aligned folders"
    )
    parser.add_argument(
        "--shape_provider",
        type=str,
        default="hi3dgen",
        choices=["hunyuan", "hi3dgen", "direct3d_s2"],
        help="Shape provider used in view alignment"
    )
    parser.add_argument(
        "--texture_provider",
        type=str,
        default="mvadapter",
        choices=["hunyuan", "mvadapter"],
        help="Texture provider used in view alignment"
    )
    parser.add_argument(
        "--bald_version", default="w_seg", type=str, choices=["w_seg", "wo_seg", "all"]
    )
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument(
        "--save_intermediates",
        action="store_true",
        default=False,
        help="Whether to save intermediate images"
    )
    parser.add_argument(
        "--from_blender",
        dest="from_blender",
        action="store_true",
        default=True,
        help="Camera parameters are from Blender coordinate system"
    )
    parser.add_argument(
        "--from_nvdiffrast",
        dest="from_blender",
        action="store_false",
        help="Camera parameters are in NVDiffrast coordinate system"
    )
    args = parser.parse_args()
    
    # Construct view_aligned directory path
    data_dir = Path(args.data_dir)
    provider_subdir = f"shape_{args.shape_provider}__texture_{args.texture_provider}"
    view_aligned_dir = data_dir / "view_aligned" / provider_subdir
    
    if not view_aligned_dir.exists():
        print(f"View aligned directory not found: {view_aligned_dir}")
        exit(1)
    
    # Get all subdirectories
    all_folders = [f for f in view_aligned_dir.iterdir() if f.is_dir()]
    
    # Seed random by timestamp and shuffle folders
    timestamp_seed = int(time.time())
    random.seed(timestamp_seed)
    random.shuffle(all_folders)
    print(f"Shuffled folders using timestamp seed: {timestamp_seed}")
    print(f"Found {len(all_folders)} view-aligned folders\n")
    
    # Create a single TexturedViewGenerator instance to reuse across all folders
    print("Loading TexturedViewGenerator pipeline...")
    generator = TexturedViewGenerator(config=None, load_pipeline=True)
    print("Pipeline loaded successfully!\n")
    
    # Process each folder
    processed_count = 0
    skipped_count = 0
    error_count = 0

    # Process each pair
    if args.bald_version == "all":
        bald_versions = ["w_seg", "wo_seg"]
    else:
        bald_versions = [args.bald_version]
    
    try:
        for bald_version in bald_versions:
            for i, folder in enumerate(all_folders, 1):
                # Check if already processed (check inside loop for concurrent execution safety)
                output_path = folder  / "alignment" / "target_image.png"
                if output_path.exists():
                    print(f"[{i}/{len(all_folders)}] Skipping {folder.name}: already processed")
                    skipped_count += 1
                    continue
                
                # Check if camera_params.json exists before processing
                camera_json_path = folder / bald_version / "camera_params.json"
                if not camera_json_path.exists():
                    print(f"[{i}/{len(all_folders)}] Skipping {folder.name}: camera_params.json not found")
                    skipped_count += 1
                    continue
                
                print(f"\n[{i}/{len(all_folders)}] Processing {folder.name}")
                
                result = process_view_aligned_folder(
                    folder_path=folder,
                    data_dir=data_dir,
                    bald_version=bald_version,
                    seed=args.seed,
                    save_intermediates=args.save_intermediates,
                    from_blender=args.from_blender,
                    generator=generator,
                )
                
                if result is True:
                    processed_count += 1
                else:
                    # Processing failed or returned False
                    error_count += 1
    finally:
        # Clean up generator
        print("\nUnloading pipeline...")
        generator.unload()
    
    print(f"\n{'='*60}")
    print(f"Processing complete:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped (already done): {skipped_count}")
    print(f"  Errors/Missing files: {error_count}")
    print(f"{'='*60}")