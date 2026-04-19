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

from hairport.fit_lmk.run_standalone import estimate_3d_landmarks_standalone
from hairport.core.mesh_utils import (
    rotate_mesh_vertices as _rotate_mesh_vertices,
    apply_inverse_rotation,
    apply_rotation,
    align_target_to_source_view,
    apply_glb_to_target_transform,
    apply_target_to_glb_transform,
    rotate_glb_mesh,
    TARGET_LANDMARK_EXTENT,
)


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
    textured_mesh_dir: Path,
    texture_provider: str,
    data_dir: Path,
    frontalize: bool = False,
):
    """Process 3D landmark fitting for all samples in random order.

    Head orientation is computed via FLAMEFitter (SHeaP) and cached under
    ``<data_dir>/head_orientation/<sample_id>/head_orientation.json``.
    """
    from hairport.core.flame_fitting import compute_head_orientation

    all_sample_ids = os.listdir(str(textured_mesh_dir))
    sample_ids = all_sample_ids
    
    # Randomize the order of sample processing for parallel runs
    random.seed(int(os.times().elapsed * 1000))
    random.shuffle(sample_ids)
    
    if not sample_ids:
        logger.warning(f"No samples to process")
        return
    to_frontalize_ids = []
    meshes_dir = str(Path(data_dir) / "hi3dgen_copy")
    if os.path.isdir(meshes_dir):
        for folder in os.listdir(meshes_dir):
            shape_mesh_path = os.path.join(meshes_dir, folder, "shape_mesh.glb")
            if os.path.exists(shape_mesh_path):
                file_size_mb = os.path.getsize(shape_mesh_path) / (1024 * 1024)
                if file_size_mb >= 25:
                    to_frontalize_ids.append(folder)

    fitter = None  # lazily initialised FLAMEFitter

    for sample_id in tqdm(sample_ids, desc="Fitting 3D landmarks", unit="sample"):
        target_textured_mesh_path = textured_mesh_dir / sample_id / "textured_mesh.glb"

        if not target_textured_mesh_path.exists():
            continue
        
        # Compute head orientation via FLAMEFitter (cached)
        image_path = data_dir / "image" / f"{sample_id}.png"
        cache_dir = data_dir / "head_orientation" / sample_id
        if not image_path.exists():
            logger.warning(f"Source image not found for {sample_id}, skipping orientation")
            continue

        try:
            head_orientation_dict = compute_head_orientation(
                image_path=image_path,
                cache_dir=cache_dir,
                fitter=fitter,
            )
        except Exception as e:
            logger.warning(f"Could not compute head orientation for {sample_id}: {e}")
            continue

        euler_rad = head_orientation_dict.get("euler_angles_xyz_radians", [[0.0, 0.0, 0.0]])[0]
        
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

    _process_lmk_3d_fitting(
        lmk_3d_output_dir=lmk_3d_dir,
        textured_mesh_dir=textured_mesh_dir,
        texture_provider=texture_provider,
        data_dir=output_dir,
        frontalize=frontalize,
    )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Landmark Fitting Post-Processing')
    parser.add_argument('--data_dir', type=str, default="outputs",
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
