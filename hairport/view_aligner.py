import argparse
import csv
import itertools
import json
import math
import os
import gc
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from hairport.utility.blender_rendering import render_mesh
from hairport.utility.estimate_camera import align_landmarks, convert_camera_cv_to_blender
from hairport.core import BackgroundRemover, FacialLandmarkDetector
from hairport.utility.uncrop_sdxl.uncrop_sdxl import Uncropper
from hairport.config import get_config

@dataclass
class Config:
    # Angle threshold for determining if 3D lifting is needed (in degrees)
    ANGLE_THRESHOLD_FOR_3D_LIFT: float | None = None
    
    # Rendering settings
    RENDER_RESOLUTION: int | None = None
    
    # Directory structure
    DIR_MATTED_IMAGE: str | None = None
    DIR_FLAME: str | None = None
    DIR_LANDMARKS: str | None = None
    DIR_LANDMARKS_3D: str | None = None
    DIR_VIEW_ALIGNED: str | None = None
    DIR_SRC_OUTPAINTED: str | None = None
    DIR_PROMPTS: str | None = None

    # File names
    FILE_HEAD_ORIENTATION: str | None = None
    FILE_LANDMARKS: str | None = None
    FILE_VERTEX_INDICES: str | None = None
    FILE_TEXTURED_MESH: str | None = None
    FILE_ALIGNED_MESH: str | None = None
    FILE_CAMERA_PARAMS: str | None = None
    FILE_ENHANCED_RENDER: str | None = None

    # ID filters
    EXCLUDED_ID_PREFIXES: tuple = ()

    def __post_init__(self):
        cfg = get_config()
        av = cfg.align_view
        ds = cfg.dataset
        if self.ANGLE_THRESHOLD_FOR_3D_LIFT is None:
            self.ANGLE_THRESHOLD_FOR_3D_LIFT = av.angle_threshold_3d_lift
        if self.RENDER_RESOLUTION is None:
            self.RENDER_RESOLUTION = av.render_resolution
        if self.DIR_MATTED_IMAGE is None:
            self.DIR_MATTED_IMAGE = ds.dir_matted_image
        if self.DIR_FLAME is None:
            self.DIR_FLAME = ds.dir_pixel3dmm
        if self.DIR_LANDMARKS is None:
            self.DIR_LANDMARKS = ds.dir_landmarks
        if self.DIR_LANDMARKS_3D is None:
            self.DIR_LANDMARKS_3D = ds.dir_landmarks_3d
        if self.DIR_VIEW_ALIGNED is None:
            self.DIR_VIEW_ALIGNED = ds.dir_view_aligned
        if self.DIR_SRC_OUTPAINTED is None:
            self.DIR_SRC_OUTPAINTED = ds.dir_source_outpainted
        if self.DIR_PROMPTS is None:
            self.DIR_PROMPTS = ds.dir_prompts
        if self.FILE_HEAD_ORIENTATION is None:
            self.FILE_HEAD_ORIENTATION = ds.file_head_orientation
        if self.FILE_LANDMARKS is None:
            self.FILE_LANDMARKS = ds.file_landmarks
        if self.FILE_VERTEX_INDICES is None:
            self.FILE_VERTEX_INDICES = ds.file_vertex_indices
        if self.FILE_TEXTURED_MESH is None:
            self.FILE_TEXTURED_MESH = ds.file_textured_mesh
        if self.FILE_ALIGNED_MESH is None:
            self.FILE_ALIGNED_MESH = ds.file_aligned_mesh
        if self.FILE_CAMERA_PARAMS is None:
            self.FILE_CAMERA_PARAMS = ds.file_camera_params
        if self.FILE_ENHANCED_RENDER is None:
            self.FILE_ENHANCED_RENDER = ds.file_enhanced_render

def compute_euler_angle_difference(euler1, euler2, seq='xyz'):
    rot1 = R.from_euler(seq, euler1)
    rot2 = R.from_euler(seq, euler2)
    
    # Compute relative rotation: R_diff = R1_inv * R2
    diff_rot = rot1.inv() * rot2
    
    # The magnitude of the rotation vector of the relative rotation 
    # is the angle of rotation required to align rot1 with rot2.
    return diff_rot.magnitude()


def compute_lift_3d(
    data_dir: Union[str, Path],
    target_id: str,
    source_id: str,
    config: Config = None,
) -> tuple[bool, float]:
    """Compute whether 3D lifting is needed based on angle difference between target and source.
    
    Returns:
        tuple[bool, float]: (lift_3d_decision, angle_difference_in_radians)
    """
    if config is None:
        config = Config()
    
    data_dir = str(data_dir)
    flame_dir = os.path.join(data_dir, config.DIR_FLAME)
    
    # Load orientation data
    source_head_orientation_path = os.path.join(flame_dir, source_id, config.FILE_HEAD_ORIENTATION)
    target_head_orientation_path = os.path.join(flame_dir, target_id, config.FILE_HEAD_ORIENTATION)
    
    with open(source_head_orientation_path, 'r') as f:
        source_head_orientation_data = json.load(f)
    with open(target_head_orientation_path, 'r') as f:
        target_head_orientation_data = json.load(f)
    
    target_euler_rad = target_head_orientation_data['euler_angles_xyz_radians'][0]
    # target_euler_rad[0] = 0.0  # Ignore pitch for lift_3d computation
    # target_euler_rad[2] = 0.0  # Ignore yaw for lift_3d computation
    source_euler_rad = source_head_orientation_data['euler_angles_xyz_radians'][0]
    # source_euler_rad[0] = 0.0  # Ignore pitch for lift_3d computation
    # source_euler_rad[2] = 0.0  # Ignore yaw for lift_3d computation
    
    cross_head_angle_diff = compute_euler_angle_difference(
        target_euler_rad,
        source_euler_rad,
        seq='xyz'
    )
    cross_head_angle_diff_deg = math.degrees(cross_head_angle_diff)
    
    return cross_head_angle_diff_deg >= config.ANGLE_THRESHOLD_FOR_3D_LIFT, cross_head_angle_diff


def get_textured_mesh_dir(data_dir: str, shape_provider: str, texture_provider: str) -> str:
    if shape_provider == "hunyuan" and texture_provider == "hunyuan":
        return os.path.join(data_dir, "hunyuan")
    return os.path.join(data_dir, texture_provider, shape_provider)


def get_provider_subdir(shape_provider: str, texture_provider: str) -> str:
    return f"shape_{shape_provider}__texture_{texture_provider}"


def compute_outpainting(
    data_dir: Union[str, Path],
    target_id: str,
    source_id: str,
    shape_provider: str = "hunyuan",
    texture_provider: str = "hunyuan",
    bald_version: str = "w_seg",
    config: Config = None,
    uncropper: Uncropper = None,
    facial_landmark_detector: FacialLandmarkDetector = None,
    enable_outpainting: bool = True,
) -> bool:
    """Compute outpainting for a source-target pair.
    
    Args:
        enable_outpainting: If True, run the outpainting pipeline. If False,
            directly copy the source bald image and create an identity resize_info.
    
    Returns:
        bool: True if outpainting was computed, False if skipped (already exists)
    """
    if config is None:
        config = Config()
    
    data_dir = str(data_dir)
    provider_subdir = get_provider_subdir(shape_provider, texture_provider)

    # Define directory paths
    lmk_dir = os.path.join(data_dir, config.DIR_LANDMARKS)
    view_aligned_dir = os.path.join(data_dir, config.DIR_VIEW_ALIGNED, provider_subdir)
    prompt_dir = os.path.join(data_dir, config.DIR_PROMPTS)

    os.makedirs(view_aligned_dir, exist_ok=True)

    # Create view-aligned directory for this pair
    pair_view_aligned_dir = os.path.join(view_aligned_dir, f"{target_id}_to_{source_id}", bald_version)
    os.makedirs(pair_view_aligned_dir, exist_ok=True)

    # Check if outpainting already exists (image, landmarks, and resize_info)
    source_uncropped_path = os.path.join(pair_view_aligned_dir, config.DIR_SRC_OUTPAINTED, "outpainted_image.png")
    source_outpainted_lmk_path = os.path.join(pair_view_aligned_dir, config.DIR_SRC_OUTPAINTED, "landmarks.npy")
    resize_info_path = os.path.join(pair_view_aligned_dir, config.DIR_SRC_OUTPAINTED, "resize_info.json")
    
    if os.path.exists(source_uncropped_path) and os.path.exists(source_outpainted_lmk_path) and os.path.exists(resize_info_path):
        print(f"Skipping outpainting (already exists): {target_id} -> {source_id}")
        return False

    # Load images and landmarks
    source_image_path = os.path.join(data_dir, "bald", bald_version, "image", f"{source_id}.png")
    source_lmk_path = os.path.join(lmk_dir, source_id, config.FILE_LANDMARKS)
    target_lmk_path = os.path.join(lmk_dir, target_id, config.FILE_LANDMARKS)
    source_prompt_path = os.path.join(prompt_dir, f"{source_id}.json")

    source_image = Image.open(source_image_path).convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    os.makedirs(os.path.dirname(source_uncropped_path), exist_ok=True)

    if not enable_outpainting:
        # Directly copy the source image without outpainting
        # Create an identity resize_info (image fills entire canvas with no transformation)
        source_image.save(source_uncropped_path)
        source_uncropped = source_image  # Use source_image as source_uncropped for landmark detection
        print(f"Saved source image directly (outpainting disabled): {source_uncropped_path}")
        
        # Identity resize_info: image fills the entire 1024x1024 canvas
        resize_info = {
            "new_width": 1024,
            "new_height": 1024,
            "margin_x": 0,
            "margin_y": 0,
            "scale_factor": 1.0,
            "resize_factor": 1.0,
            "scaled_width": 1024,
            "scaled_height": 1024,
            "landmark_offset_x": None,
            "landmark_offset_y": None,
        }
        with open(resize_info_path, 'w') as f:
            json.dump(resize_info, f, indent=2)
    else:
        # Run outpainting pipeline
        if uncropper is None:
            uncropper = Uncropper()
            uncropper.load_pipeline()

        source_prompts = json.load(open(source_prompt_path, "r"))
        source_prompt = source_prompts["subject"][0].get("description_no_hair", "High-quality photo of a bald person, high resolution, detailed, 4k").replace(", no background", "")
        source_background_prompt = source_prompts.get("background", "")

        source_uncropped, resize_info, _ = uncropper.uncrop_matching_source(
            target_image=source_image,
            target_landmark_path=source_lmk_path,
            source_landmark_path=target_lmk_path,
            prompt=source_prompt + " " + source_background_prompt,
            safeguard_resolution=5.0,
        )
        
        source_uncropped.resize((1024, 1024), Image.Resampling.LANCZOS).save(source_uncropped_path)
        print(f"Saved outpainted image to: {source_uncropped_path}")
        # Save resize info to JSON
        with open(resize_info_path, 'w') as f:
            json.dump(resize_info, f, indent=2)
    
    if facial_landmark_detector is None:
        facial_landmark_detector = FacialLandmarkDetector(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    source_outpainted_lmk_data = facial_landmark_detector.get_lmk_full(source_uncropped)
    np.save(source_outpainted_lmk_path, source_outpainted_lmk_data)
    
    return True


def run_camera_optimization(
    data_dir: Union[str, Path],
    target_id: str,
    source_id: str,
    shape_provider: str = "hunyuan",
    texture_provider: str = "hunyuan",
    bald_version: str = "w_seg",
    debug: bool = False,
    config: Config = None,
) -> bool:
    """Run camera optimization for a source-target pair that requires 3D lifting.
    Assumes outpainting has already been computed.
    Returns:
        bool: True if optimization was run, False if skipped (already exists)
    """
    if config is None:
        config = Config()
    
    data_dir = str(data_dir)
    provider_subdir = get_provider_subdir(shape_provider, texture_provider)

    # Define directory paths
    flame_dir = os.path.join(data_dir, config.DIR_FLAME)
    view_aligned_dir = os.path.join(data_dir, config.DIR_VIEW_ALIGNED, provider_subdir)
    lmk_3d_dir = os.path.join(data_dir, config.DIR_LANDMARKS_3D, provider_subdir)

    # Create view-aligned directory for this pair
    pair_view_aligned_dir = os.path.join(view_aligned_dir, f"{target_id}_to_{source_id}", bald_version)

    # Skip if camera params already exist
    camera_params_path = os.path.join(pair_view_aligned_dir, config.FILE_CAMERA_PARAMS)
    if os.path.exists(camera_params_path):
        print(f"Skipping camera optimization (already exists): {target_id} -> {source_id}")
        return False

    # Load orientation data
    source_head_orientation_path = os.path.join(flame_dir, source_id, config.FILE_HEAD_ORIENTATION)
    target_head_orientation_path = os.path.join(flame_dir, target_id, config.FILE_HEAD_ORIENTATION)

    with open(source_head_orientation_path, 'r') as f:
        source_head_orientation_data = json.load(f)
    with open(target_head_orientation_path, 'r') as f:
        target_head_orientation_data = json.load(f)

    target_euler_rad = target_head_orientation_data['euler_angles_xyz_radians'][0]
    source_euler_rad = source_head_orientation_data['euler_angles_xyz_radians'][0]

    # Load 3D landmark vertex indices
    target_lmk_3d_dir = os.path.join(lmk_3d_dir, target_id)
    lmk_3d_vertex_indices_path = os.path.join(target_lmk_3d_dir, config.FILE_VERTEX_INDICES)
    lmk_3d_vertex_indices = torch.from_numpy(np.load(lmk_3d_vertex_indices_path))

    # Load outpainted source data
    source_uncropped_path = os.path.join(pair_view_aligned_dir, config.DIR_SRC_OUTPAINTED, "outpainted_image.png")
    source_outpainted_lmk_path = os.path.join(pair_view_aligned_dir, config.DIR_SRC_OUTPAINTED, "landmarks.npy")
    
    if not os.path.exists(source_uncropped_path) or not os.path.exists(source_outpainted_lmk_path):
        raise FileNotFoundError(f"Outpainting not found for {target_id} -> {source_id}. Run outpainting first.")
    
    source_lmk_data = np.load(source_outpainted_lmk_path, allow_pickle=True).item()
    target_mesh_path = os.path.join(lmk_3d_dir, target_id, config.FILE_TEXTURED_MESH)

    print("Running camera optimization for 3D lifted render.")

    # Align landmarks and optimize camera
    align_landmarks(
        target_mesh_path=target_mesh_path,
        source_image_path=source_uncropped_path,
        view_aligned_dir=pair_view_aligned_dir,
        source_lmk_data=source_lmk_data,
        source_rotation_euler_rad=source_euler_rad,
        target_rotation_euler_rad=target_euler_rad,
        lmk_3d_vertex_indices=lmk_3d_vertex_indices,
        frontalize_target=False,
        debug=debug,
    )
    
    return True


def align_target_to_source_lift_3d(
    data_dir: Union[str, Path],
    target_id: str,
    source_id: str,
    lift_3d: bool,
    shape_provider: str = "hunyuan",
    texture_provider: str = "hunyuan",
    bald_version: str = "w_seg",
    debug: bool = False,
    config: Config = None,
    uncropper: Uncropper = None,
    facial_landmark_detector: FacialLandmarkDetector = None,
):
    """Legacy function that combines outpainting and camera optimization.
    For new code, prefer using compute_outpainting and run_camera_optimization separately.
    """
    if config is None:
        config = Config()
    
    # Compute outpainting
    compute_outpainting(
        data_dir=data_dir,
        target_id=target_id,
        source_id=source_id,
        shape_provider=shape_provider,
        texture_provider=texture_provider,
        bald_version=bald_version,
        config=config,
        uncropper=uncropper,
        facial_landmark_detector=facial_landmark_detector,
    )
    
    # Run camera optimization if 3D lifting is needed
    if lift_3d:
        run_camera_optimization(
            data_dir=data_dir,
            target_id=target_id,
            source_id=source_id,
            shape_provider=shape_provider,
            texture_provider=texture_provider,
            bald_version=bald_version,
            debug=debug,
            config=config,
        )


def filter_ids(all_ids: list, config: Config) -> list:
    return [
        id_name for id_name in all_ids 
        if not any(id_name.startswith(prefix) for prefix in config.EXCLUDED_ID_PREFIXES)
    ]


def prepare_pairs(data_dir: str, config: Config, pairs_csv_file: str = None) -> list:
    # Check if we should read from CSV
    csv_path = None
    if pairs_csv_file is not None:
        csv_path = pairs_csv_file
    else:
        default_csv = os.path.join(data_dir, "pairs.csv")
        if os.path.exists(default_csv):
            csv_path = default_csv
    
    # If CSV file is available, read pairs from it
    if csv_path is not None:
        print(f"Reading pairs from CSV: {csv_path}")
        pairs = []
        rows_data = []
        has_lift_3d_column = False
        has_head_diff_angle_column = False
        needs_update = False
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            has_lift_3d_column = 'lift_3d' in fieldnames if fieldnames else False
            has_head_diff_angle_column = 'head_diff_angle' in fieldnames if fieldnames else False
            
            for row in reader:
                target_id = row['target_id']
                source_id = row['source_id']
                
                # Check if lift_3d and head_diff_angle need to be computed
                if (has_lift_3d_column and 'lift_3d' in row and row['lift_3d'] != '' and
                    has_head_diff_angle_column and 'head_diff_angle' in row and row['head_diff_angle'] != ''):
                    lift_3d = row['lift_3d'].lower() in ('true', '1', 'yes')
                    head_diff_angle = float(row['head_diff_angle'])
                else:
                    # Compute lift_3d and head_diff_angle
                    try:
                        lift_3d, head_diff_angle = compute_lift_3d(data_dir, target_id, source_id, config)
                        needs_update = True
                    except Exception as e:
                        print(f"Warning: Could not compute lift_3d for {target_id} -> {source_id}: {e}")
                        lift_3d = False
                        head_diff_angle = 0.0
                        needs_update = True
                
                pairs.append((target_id, source_id, lift_3d))
                row_data = {
                    'target_id': target_id, 
                    'source_id': source_id, 
                    'lift_3d': str(lift_3d),
                    'head_diff_angle': str(head_diff_angle)
                }
                rows_data.append(row_data)
        
        # Update CSV if lift_3d or head_diff_angle column was missing or had empty values
        if needs_update or not has_lift_3d_column or not has_head_diff_angle_column:
            print(f"Updating CSV with lift_3d and head_diff_angle columns: {csv_path}")
            with open(csv_path, 'w', newline='') as f:
                fieldnames = ['target_id', 'source_id', 'lift_3d', 'head_diff_angle']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_data)
        
        print(f"Loaded {len(pairs)} pairs from CSV")
        return pairs
    
    # Fall back to generating pairs from all IDs
    print("No CSV file found, generating pairs from all IDs")
    all_ids = os.listdir(os.path.join(data_dir, config.DIR_MATTED_IMAGE))
    all_ids = [f.split(".")[0] for f in all_ids]
    
    target_ids = filter_ids(all_ids, config)
    source_ids = filter_ids(all_ids, config)

    # target_ids = [t for t in target_ids if not t.startswith("sample_") and not t.startswith("side") and not t.startswith("n")]
    # source_ids = [s for s in source_ids if not s.startswith("sample_") and not s.startswith("side") and not s.startswith("n")]

    random.shuffle(target_ids)
    random.shuffle(source_ids)

    # Create all possible pairs where source and target are different    
    pairs = list(itertools.product(target_ids, source_ids))
    pairs = [(target, source) for target, source in pairs if target != source]

    random.shuffle(pairs)
    
    # Compute lift_3d for each pair
    pairs_with_lift_3d = []
    for target_id, source_id in pairs:
        try:
            lift_3d, head_diff_angle = compute_lift_3d(data_dir, target_id, source_id, config)
        except Exception as e:
            print(f"Warning: Could not compute lift_3d for {target_id} -> {source_id}: {e}")
            lift_3d = False
        pairs_with_lift_3d.append((target_id, source_id, lift_3d))
    
    return pairs_with_lift_3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align target hairstyle to source view using landmark optimization"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="outputs/",
    )
    parser.add_argument(
        "--shape_provider", default="hi3dgen", type=str, choices=["hunyuan", "hi3dgen", "direct3d_s2"]
    )
    parser.add_argument(
        "--texture_provider", default="mvadapter", type=str, choices=["hunyuan", "mvadapter"]
    )
    parser.add_argument(
        "--bald_version", default="w_seg", type=str, choices=["wo_seg", "w_seg", "all"]
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode with 3D visualizations and intermediate projection saves"
    )
    parser.add_argument(
        "--pairs_csv_file",
        type=str,
        default=None,
        help="Path to CSV file containing pairs (columns: source_id, target_id). If not provided, will look for pairs.csv in data_dir"
    )
    parser.add_argument(
        "--enable_outpainting",
        action="store_true",
        default=False,
        help="Enable outpainting to expand source images. When disabled, source images are copied directly with identity transformation."
    )
    parser.add_argument(
        "--disable_outpainting",
        action="store_true",
        default=False,
        help="Disable outpainting (source images are copied directly with identity transformation)"
    )
    args = parser.parse_args()
    
    # Handle the disable_outpainting flag (overrides enable_outpainting)
    if args.disable_outpainting:
        args.enable_outpainting = False
    
    config = Config()

    random_seed = int(time.time())
    random.seed(random_seed)
    np.random.seed(random_seed)
    print(f"Using random seed: {random_seed}")

    # Prepare pairs
    pairs = prepare_pairs(args.data_dir, config, args.pairs_csv_file)
    random.shuffle(pairs)
    print(f"Processing {len(pairs)} pairs")
    success_count = 0
    failure_count = 0
    MAX_SUCCESS = 10000
    
    # Only initialize uncropper if outpainting is enabled
    if args.enable_outpainting:
        uncropper = Uncropper()
        uncropper.load_pipeline()
    else:
        uncropper = None
        print("Outpainting disabled - source images will be copied directly")
    
    facial_landmark_detector = FacialLandmarkDetector(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    # Process each pair
    if args.bald_version == "all":
        bald_versions = ["w_seg", "wo_seg"]
    else:
        bald_versions = [args.bald_version]

    # Phase 1: Compute outpainting for ALL samples
    print("=" * 80)
    print("PHASE 1: Computing outpainting for all samples")
    print("=" * 80)
    
    outpaint_success_count = 0
    outpaint_skip_count = 0
    outpaint_failure_count = 0
    
    for bald_version in bald_versions:
        for i, (target_id, source_id, lift_3d) in enumerate(pairs):
            try:
                print(f"[Outpainting] Processing pair {i+1}/{len(pairs)}: {source_id} <-- {target_id} (bald_version={bald_version})")
                was_computed = compute_outpainting(
                    data_dir=args.data_dir,
                    target_id=target_id,
                    source_id=source_id,
                    shape_provider=args.shape_provider,
                    texture_provider=args.texture_provider,
                    bald_version=bald_version,
                    config=config,
                    uncropper=uncropper,
                    facial_landmark_detector=facial_landmark_detector,
                    enable_outpainting=args.enable_outpainting,
                )
                if was_computed:
                    outpaint_success_count += 1
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    outpaint_skip_count += 1
            except Exception as e:
                print(f"[Outpainting] Error processing pair {target_id} -> {source_id}: {e}")
                outpaint_failure_count += 1
    
    print(f"\nOutpainting phase complete: {outpaint_success_count} computed, {outpaint_skip_count} skipped, {outpaint_failure_count} failed")
    
    # Release GPU resources by deleting the Uncropper (only if it was initialized)
    if uncropper is not None:
        print("\nReleasing Uncropper GPU resources...")
        del uncropper
        torch.cuda.empty_cache()
        gc.collect()
        print("Uncropper deleted and GPU memory freed.")
    
    # Phase 2: Run camera optimization for samples requiring 3D lifting
    print("\n" + "=" * 80)
    print("PHASE 2: Running camera optimization for 3D lifting samples")
    print("=" * 80)
    
    # Filter pairs that require 3D lifting
    lift_3d_pairs = [(t, s, l) for t, s, l in pairs if l]
    print(f"Found {len(lift_3d_pairs)} pairs requiring 3D lifting")
    
    camera_opt_success_count = 0
    camera_opt_skip_count = 0
    camera_opt_failure_count = 0
    
    for bald_version in bald_versions:
        for i, (target_id, source_id, lift_3d) in enumerate(lift_3d_pairs):
            try:
                print(f"[Camera Opt] Processing pair {i+1}/{len(lift_3d_pairs)}: {source_id} <-- {target_id} (bald_version={bald_version})")
                was_computed = run_camera_optimization(
                    data_dir=args.data_dir,
                    target_id=target_id,
                    source_id=source_id,
                    shape_provider=args.shape_provider,
                    texture_provider=args.texture_provider,
                    bald_version=bald_version,
                    debug=args.debug,
                    config=config,
                )
                if was_computed:
                    camera_opt_success_count += 1
                    torch.cuda.empty_cache()
                    gc.collect()
                    success_count += 1
                    if success_count >= MAX_SUCCESS:
                        print(f"Reached maximum success count of {MAX_SUCCESS}. Stopping.")
                        break
                else:
                    camera_opt_skip_count += 1
            except Exception as e:
                print(f"[Camera Opt] Error processing pair {target_id} -> {source_id}: {e}")
                camera_opt_failure_count += 1
                failure_count += 1
    
    print(f"\nCamera optimization phase complete: {camera_opt_success_count} computed, {camera_opt_skip_count} skipped, {camera_opt_failure_count} failed")
    print(f"\nTotal: {success_count} successful, {failure_count} failed")