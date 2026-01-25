import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm
from scipy.spatial.transform import Rotation


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def _setup_directories(
    output_dir: Path, 
    shape_provider: str = "hi3dgen",
    texture_provider: str = "mvadapter"
) -> Tuple[Path, Path, Path]:
    """Create required output directories."""
    matted_image_dir = output_dir / 'matted_image'
    
    # Create clearer directory structure: lmk_3d/shape_{shape_provider}__texture_{texture_provider}
    lmk_3d_dir = output_dir / 'lmk_3d' / f"shape_{shape_provider}__texture_{texture_provider}"
    
    if texture_provider == "mvadapter":
        textured_mesh_dir = output_dir / "mvadapter" / shape_provider
    elif texture_provider == "hunyuan":
        textured_mesh_dir = output_dir / "hunyuan"
    else:
        raise ValueError(f"Unsupported texture_provider: {texture_provider}")

    for directory in [matted_image_dir, lmk_3d_dir, textured_mesh_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    return matted_image_dir, lmk_3d_dir, textured_mesh_dir


def _rotate_mesh_vertices(mesh, rotation_matrix):
    """Apply rotation matrix to mesh vertices and normals around center."""
    mesh = mesh.copy()
    mesh_center = mesh.vertices.mean(axis=0)
    vertices_centered = mesh.vertices - mesh_center
    mesh.vertices = (vertices_centered @ rotation_matrix.T) + mesh_center
    
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
        mesh.vertex_normals = mesh.vertex_normals @ rotation_matrix.T
    if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
        mesh.face_normals = mesh.face_normals @ rotation_matrix.T
    
    return mesh


def apply_inverse_rotation(mesh, euler_angles_rad):
    """Apply inverse rotation to mesh using Euler angles."""
    rotation = Rotation.from_euler('xyz', euler_angles_rad)
    inverse_rotation_matrix = rotation.inv().as_matrix()
    return _rotate_mesh_vertices(mesh, inverse_rotation_matrix)


def apply_rotation(mesh, euler_angles_rad):
    """Apply rotation to mesh using Euler angles."""
    rotation = Rotation.from_euler('xyz', euler_angles_rad)
    rotation_matrix = rotation.as_matrix()
    return _rotate_mesh_vertices(mesh, rotation_matrix)


def align_target_to_source_view(target_mesh, target_euler_rad, source_euler_rad):
    """Align target mesh to source view by applying inverse then forward rotation."""
    frontal_mesh = apply_inverse_rotation(target_mesh, target_euler_rad)
    aligned_mesh = apply_rotation(frontal_mesh, source_euler_rad)
    return aligned_mesh


def apply_glb_to_target_transform(mesh):
    """Transform mesh from GLB coordinate system to target system (90° X-axis rotation)."""
    rotation_matrix = Rotation.from_euler('x', np.pi / 2).as_matrix()
    return _rotate_mesh_vertices(mesh, rotation_matrix)


def apply_target_to_glb_transform(mesh):
    """Transform mesh from target coordinate system back to GLB system (-90° X-axis rotation)."""
    rotation_matrix = Rotation.from_euler('x', -np.pi / 2).as_matrix()
    return _rotate_mesh_vertices(mesh, rotation_matrix)

def rotate_glb_mesh(
    input_glb_path: str,
    output_glb_path: str = None,
    euler_angles_rad: list = [0.0, 0.0, 0.0],
    rotate_fn=apply_inverse_rotation,
    to_normalize_vertice_ids: np.ndarray = None,
    target_landmark_extent: float = None,
):
    """Rotate a GLB mesh file and optionally save the result.
    
    Args:
        input_glb_path: Path to input GLB file
        output_glb_path: Path to save output GLB file (optional)
        euler_angles_rad: Euler angles for rotation
        rotate_fn: Function to apply rotation
        to_normalize_vertice_ids: Vertex indices for centering (landmarks)
        target_landmark_extent: Target max extent (per axis) of landmarks in meters.
            If provided, mesh will be scaled so landmark bounding box max extent equals this value.
    
    Returns:
        result_scene: The processed trimesh Scene
        frontalized_mesh: The intermediate frontalized mesh
        final_landmark_coords: 3D coordinates of landmarks after scaling/centering (or None)
    """
    input_path = Path(input_glb_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input GLB file does not exist: {input_glb_path}")

    if output_glb_path is not None:
        output_path = Path(output_glb_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    loaded = trimesh.load(str(input_path), force='scene')
    final_landmark_coords = None
    
    if isinstance(loaded, trimesh.Scene):
        for mesh_name, mesh in loaded.geometry.items():
            if isinstance(mesh, trimesh.Trimesh):
                mesh_in_target = apply_glb_to_target_transform(mesh)
                frontalized_mesh = rotate_fn(mesh_in_target, euler_angles_rad)
                mesh_back_to_glb = apply_target_to_glb_transform(frontalized_mesh)
                if to_normalize_vertice_ids is not None:
                    vertices = mesh_back_to_glb.vertices.copy()
                    landmark_vertices = vertices[to_normalize_vertice_ids]
                    
                    # Scale based on landmark bounding box if target extent is specified
                    if target_landmark_extent is not None:
                        landmark_min = landmark_vertices.min(axis=0)
                        landmark_max = landmark_vertices.max(axis=0)
                        landmark_extent = landmark_max - landmark_min
                        max_extent = landmark_extent.max()
                        
                        if max_extent > 0:
                            scale_factor = target_landmark_extent / max_extent
                            vertices *= scale_factor
                            landmark_vertices = vertices[to_normalize_vertice_ids]
                    
                    # Center using landmark mean
                    centroid = landmark_vertices.mean(axis=0)
                    vertices -= centroid
                    mesh_back_to_glb.vertices = vertices
                    
                    # Compute final landmark coordinates
                    final_landmark_coords = vertices[to_normalize_vertice_ids]
                loaded.geometry[mesh_name] = mesh_back_to_glb
        
        result_scene = loaded
        result_mesh = list(loaded.geometry.values())[0] if loaded.geometry else None

    elif isinstance(loaded, trimesh.Trimesh):
        mesh_in_target = apply_glb_to_target_transform(loaded)
        frontalized_mesh = rotate_fn(mesh_in_target, euler_angles_rad)
        result_mesh = apply_target_to_glb_transform(frontalized_mesh)
        if to_normalize_vertice_ids is not None:
            vertices = result_mesh.vertices.copy()
            landmark_vertices = vertices[to_normalize_vertice_ids]
            
            # Scale based on landmark bounding box if target extent is specified
            if target_landmark_extent is not None:
                landmark_min = landmark_vertices.min(axis=0)
                landmark_max = landmark_vertices.max(axis=0)
                landmark_extent = landmark_max - landmark_min
                max_extent = landmark_extent.max()
                
                if max_extent > 0:
                    scale_factor = target_landmark_extent / max_extent
                    vertices *= scale_factor
                    landmark_vertices = vertices[to_normalize_vertice_ids]
            
            # Center using landmark mean
            centroid = landmark_vertices.mean(axis=0)
            vertices -= centroid
            result_mesh.vertices = vertices
            
            # Compute final landmark coordinates
            final_landmark_coords = vertices[to_normalize_vertice_ids]
        
        result_scene = trimesh.Scene(result_mesh)
    else:
        raise ValueError(f"Unexpected type loaded: {type(loaded)}")
    
    if output_glb_path is not None:
        result_scene.export(str(output_path), file_type='glb')
    
    return result_scene, frontalized_mesh, final_landmark_coords


def process_shape_mesh(
    target_textured_mesh_path: str,
    target_rotation_euler_rad: list,
    frontalize: bool = False,
):
    if frontalize:
        logger.info(f"Frontalizing mesh with inverse rotation: {target_rotation_euler_rad}")
        rotate_glb_mesh(
            input_glb_path=target_textured_mesh_path,
            output_glb_path=target_textured_mesh_path,
            euler_angles_rad=target_rotation_euler_rad,
            rotate_fn=apply_rotation,
        )

def process_shape_meshes(
    textured_mesh_dir: Path,
    texture_provider: str,
    pixel3dmm_output_dir: Path,
    data_dir: Path,
    frontalize: bool = False,
):
    """Process 3D landmark fitting for all samples in random order."""
    all_sample_ids = os.listdir(str(textured_mesh_dir))
    sample_ids = all_sample_ids
    
    # Randomize the order of sample processing for parallel runs
    random.seed(int(os.times().elapsed * 1000))
    random.shuffle(sample_ids)
    
    if not sample_ids:
        logger.warning(f"No samples to process")
        return
    to_frontalize_ids = []
    meshes_dir = "/workspace/outputs/hi3dgen_copy"
    for folder in os.listdir(meshes_dir):
        shape_mesh_path = os.path.join(meshes_dir, folder, "shape_mesh.glb")
        if os.path.exists(shape_mesh_path):
            file_size_mb = os.path.getsize(shape_mesh_path) / (1024 * 1024)
            if file_size_mb >= 25:
                to_frontalize_ids.append(folder)

    for sample_id in tqdm(sample_ids, desc="Frontalizing Meshes", unit="sample"):
        target_textured_mesh_path = textured_mesh_dir / sample_id / "shape_mesh.glb"
        if sample_id in to_frontalize_ids:
            frontalize = True
        else:
            frontalize = False

        if not target_textured_mesh_path.exists():
            continue

        head_orientation_path = pixel3dmm_output_dir / sample_id / "head_orientation.json"
        if not head_orientation_path.exists():
            logger.warning(f"Orientation file not found for {sample_id}, skipping")
            continue
        
        with open(head_orientation_path, 'r') as f:
            head_orientation_dict = json.load(f)
            euler_rad = head_orientation_dict.get("euler_angles_xyz_radians", [[0.0, 0.0, 0.0]])[0]
            # euler_rad[0] = 0.0
            # euler_rad[2] = 0.0
            # Disable all rotation adjustments
            # euler_rad = [0.0, 0.0, 0.0]
            
            process_shape_mesh(
                target_textured_mesh_path=str(target_textured_mesh_path),
                target_rotation_euler_rad=euler_rad,
                frontalize=frontalize,
            )

def main(
    data_dir: str, 
    shape_provider: str = "hunyuan",
    texture_provider: str = "mvadapter",
    frontalize: bool = False,
) -> None:
    """Main processing pipeline for 3D landmark fitting."""
    # Validate provider combinations
    if shape_provider == "hi3dgen" and texture_provider == "hunyuan":
        raise ValueError(
            "Invalid combination: texture_provider cannot be 'hunyuan' when shape_provider is 'hi3dgen'"
        )
    
    pixel3dmm_output_dir = Path(data_dir) / "pixel3dmm_output"
    logger.info(f"Starting 3D landmark fitting with shape_provider='{shape_provider}', "
                f"texture_provider='{texture_provider}'")
    
    output_dir = Path(data_dir)
    
    process_shape_meshes(
        data_dir=output_dir,
        texture_provider=texture_provider,
        pixel3dmm_output_dir=pixel3dmm_output_dir,
        textured_mesh_dir=output_dir / shape_provider
    )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Landmark Fitting Post-Processing')
    parser.add_argument('--data_dir', type=str, default="/workspace/outputs",
                        help='Path to the data directory')
    parser.add_argument('--shape_provider', type=str, default='hi3dgen', 
                        choices=['hunyuan', 'hi3dgen'],
                        help='Shape provider name (hunyuan or hi3dgen)')
    parser.add_argument('--texture_provider', type=str, default='mvadapter',
                        choices=['mvadapter', 'hunyuan'],
                        help='Texture provider name (mvadapter or hunyuan)')
    parser.add_argument('--frontalize', action='store_true',
                        help='Whether to frontalize the mesh')
    args = parser.parse_args()
    main(
        data_dir=args.data_dir, 
        shape_provider=args.shape_provider,
        texture_provider=args.texture_provider,
        frontalize=args.frontalize
    )
