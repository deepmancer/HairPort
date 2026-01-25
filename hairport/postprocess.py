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

from fit_lmk.run_standalone import estimate_3d_landmarks_standalone


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


# Target extent for landmark bounding box normalization (in meters)
TARGET_LANDMARK_EXTENT = 0.4


def fit_3d_landmarks(
    lmk_3d_output_dir: str,
    target_textured_mesh_path: str,
    target_rotation_euler_rad: list,
    camera_location: list = [0.0, -4.0, 0.0],
    camera_rotation: list = [1.5708, 0.0, 0.0],
    camera_ortho_scale: float = 1.75,
    frontalize: bool = False,
):
    """Fit 3D landmarks to a textured mesh."""
    lmk_3d_output_dir = Path(lmk_3d_output_dir)
    vertex_output_file_path = lmk_3d_output_dir / "vertex_indices.npy"
    
    # Define output path for postprocessed mesh in the lmk_3d directory
    postprocessed_mesh_path = lmk_3d_output_dir / "postprocessed_textured_mesh.glb"
    # Output path for aligned landmark 3D coordinates
    landmark_coords_path = lmk_3d_output_dir / "landmarks_3d.npy"

    if vertex_output_file_path.exists() and postprocessed_mesh_path.exists() and landmark_coords_path.exists():
        return
    
    if frontalize:
        logger.info(f"Frontalizing mesh with inverse rotation: {target_rotation_euler_rad}")
        frontalized_mesh_path = Path(target_textured_mesh_path).parent / "frontalized_temp.glb"
        _, frontalized_mesh, _ = rotate_glb_mesh(
            input_glb_path=target_textured_mesh_path,
            output_glb_path=str(frontalized_mesh_path),
            euler_angles_rad=target_rotation_euler_rad,
            rotate_fn=apply_inverse_rotation,
        )
    else:
        frontalized_mesh_path = target_textured_mesh_path
        # Load the mesh directly if not frontalizing, and transform to target coordinate system
        # to be consistent with the frontalize=True case
        loaded = trimesh.load(str(target_textured_mesh_path), force='scene')
        if isinstance(loaded, trimesh.Scene):
            mesh_glb = list(loaded.geometry.values())[0] if loaded.geometry else None
            frontalized_mesh = apply_glb_to_target_transform(mesh_glb) if mesh_glb is not None else None
        elif isinstance(loaded, trimesh.Trimesh):
            frontalized_mesh = apply_glb_to_target_transform(loaded)
        else:
            frontalized_mesh = None

    try:
        estimate_3d_landmarks_standalone(
            mesh_path=frontalized_mesh_path,
            cam_loc=camera_location,
            cam_rot=camera_rotation,
            ortho_scale=camera_ortho_scale,
            output_dir=lmk_3d_output_dir,
            num_perturbations=0,
            resolution=1024,
            optimize=True,
            device='cuda',
        )
        
        vertex_indices = np.load(vertex_output_file_path)
        _, _, final_landmark_coords = rotate_glb_mesh(
            input_glb_path=frontalized_mesh_path,
            output_glb_path=str(postprocessed_mesh_path),
            rotate_fn=apply_rotation,
            to_normalize_vertice_ids=vertex_indices,
            target_landmark_extent=TARGET_LANDMARK_EXTENT,
        )
        
        # Save the aligned landmark 3D coordinates
        if final_landmark_coords is not None:
            np.save(landmark_coords_path, final_landmark_coords)
            logger.info(f"Saved aligned landmark coordinates to {landmark_coords_path}")
            
        logger.info(f"Saved scaled and centered mesh to {postprocessed_mesh_path}")
            
    except Exception as e:
        logger.error(f"Error during 3D landmark fitting for {target_textured_mesh_path}: {e}")
    finally:
        if frontalize and frontalized_mesh_path != target_textured_mesh_path:
            try:
                frontalized_mesh_path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete temporary frontalized mesh: {e}")

def _process_lmk_3d_fitting(
    lmk_3d_output_dir: Path,
    pixel3dmm_output_dir: Path,
    textured_mesh_dir: Path,
    texture_provider: str,
    data_dir: Path,
    frontalize: bool = False,
):
    """Process 3D landmark fitting for all samples in random order."""
    all_sample_ids = os.listdir(str(pixel3dmm_output_dir))
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

    for sample_id in tqdm(sample_ids, desc="Fitting 3D landmarks", unit="sample"):
        # Determine mesh path based on texture provider
        # if sample_id.startswith('n') or sample_id.startswith('sample_'):
        #     # Skip invalid sample IDs
        #     logger.warning(f"Skipping invalid sample ID: {sample_id}")
        #     continue
        target_textured_mesh_path = textured_mesh_dir / sample_id / "textured_mesh.glb"
        # if sample_id in to_frontalize_ids:
        #     frontalize = True
        # else:
        #     frontalize = False

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
            
            sample_lmk_3d_output_dir = lmk_3d_output_dir / sample_id
            sample_lmk_3d_output_dir.mkdir(parents=True, exist_ok=True)

            fit_3d_landmarks(
                lmk_3d_output_dir=str(sample_lmk_3d_output_dir),
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
    
    logger.info(f"Starting 3D landmark fitting with shape_provider='{shape_provider}', "
                f"texture_provider='{texture_provider}'")
    
    output_dir = Path(data_dir)
    _, lmk_3d_dir, textured_mesh_dir = _setup_directories(
        output_dir, 
        shape_provider=shape_provider,
        texture_provider=texture_provider
    )

    pixel3dmm_output_dir = output_dir / 'pixel3dmm_output'
    _process_lmk_3d_fitting(
        lmk_3d_output_dir=lmk_3d_dir,
        pixel3dmm_output_dir=pixel3dmm_output_dir,
        textured_mesh_dir=textured_mesh_dir,
        texture_provider=texture_provider,
        data_dir=output_dir,
        frontalize=frontalize,
    )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Landmark Fitting Post-Processing')
    parser.add_argument('--data_dir', type=str, default="/workspace/celeba_reduced",
                        help='Path to the data directory')
    parser.add_argument('--shape_provider', type=str, default='hi3dgen', 
                        choices=['hunyuan', 'hi3dgen', 'direct3d_s2'],
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
