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
    meshes_dir = str(Path(data_dir) / "hi3dgen_copy")
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
    parser.add_argument('--data_dir', type=str, default="outputs",
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
