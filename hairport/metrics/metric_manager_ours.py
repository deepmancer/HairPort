"""
Evaluation pipeline manager for hair transfer methods.

This module provides the main evaluation pipeline that orchestrates:
- Loading samples from different hair transfer methods
- Computing metrics using the metrics module
- Creating visualizations using the visualizers module
- Generating LaTeX reports using the latex_builder module

All LaTeX outputs are written to the evaluation directory with subdirectories:
- hairport/: HairPort method results
- baselines/: Baseline method results
- comparison/: Cross-method comparison visualizations
- latex_report/: Publication-ready LaTeX tables and figures

Failure Detection:
The pipeline includes automatic failure case detection and filtering:
- Technical failures: NaN values in critical metrics (face/hair detection failures)
- Extreme outliers: Values beyond 4 standard deviations from the mean
- Failure cases are excluded from final metrics, visualizations, and reports
- Maximum failure rate capped at 10% to ensure data quality
- Detailed failure reports saved for transparency and debugging
"""

from __future__ import annotations

import os
import json
import glob
import random
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm

# Import from local modules
from hairport.metrics.metrics import (
    Sample,
    Metric,
    MetricSuite,
    CLIPIMetric,
    FIDMetric,
    FIDCLIPMetric,
    SSIMMetric,
    PSNRMetric,
    LPIPSMetric,
    DINOv2HairSimilarityMetric,
    DreamSimMetric,
    IDSMetric,
    DINOv3HairSimilarityMetric,
    PixioHairSimilarityMetric,
    HairPromptSimilarityMetric,
    _to_rgb,
    _resize_like,
    _pil_to_torch_float01,
    _intersected_nonhair_weights,
    _apply_nonhair_mask_to_image,
)

from hairport.metrics.visualizers import (
    create_method_visualizations,
    create_comparison_visualizations,
    save_summary_statistics,
)

from hairport.metrics.latex_builder import (
    generate_all_latex_outputs,
    get_method_display_name,
    METRIC_CATEGORIES,
    GLOBAL_METRICS,
)

# SAM3 for mask extraction
from utils.sam_mask_extractor import SAMMaskExtractor
SAM3_AVAILABLE = True

# CaptionerPipeline for prompt generation (lazy import)
from hairport.captioner_pipeline import CaptionerPipeline
CAPTIONER_AVAILABLE = True



# -----------------------------
# Default Path Configuration
# -----------------------------

BASE_DIR = "/workspace/outputs"

PATHS = {
    # Source/reference images are directly in image/ as {id}.png
    "source_image": f"{BASE_DIR}/image/{{source_id}}.png",
    "reference_image": f"{BASE_DIR}/image/{{target_id}}.png",
    # Hair masks are in CelebAMask-HQ/CelebAMask-HQ-mask-anno/{0-14 folder}/{id}_hair.png
    "source_hair_mask": {
        "annot_dir": f"{BASE_DIR}/hair_mask/",
        "filename": "{source_id}.png",
    },
    "reference_hair_mask": {
        "annot_dir": f"{BASE_DIR}/hair_mask/",
        "filename": "{target_id}.png",
    },
    # Baseline methods (grouped under 'baselines' key)
    'baselines': {
        'hairfusion': {
            "base_dir": f"{BASE_DIR}/baselines/hairfusion/",
            "transferred": f"{BASE_DIR}/baselines/hairfusion/{{target_id}}_to_{{source_id}}/transferred.png",
            "hair_mask": f"{BASE_DIR}/baselines/hairfusion/{{target_id}}_to_{{source_id}}/hair_mask.png",
        },
        'AnyDoor': {
            "base_dir": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter",
            "transferred": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/3d_aware/transferred_anydoor/hair_restored.png",
            "hair_mask": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/3d_aware/transferred_anydoor/hair_restored_mask.png",
        },
        'MimicBrush': {
            "base_dir": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter",
            "transferred": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/3d_aware/transferred_mimicbrush/hair_restored.png",
            "hair_mask": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/3d_aware/transferred_mimicbrush/hair_restored_mask.png",
        },
        # 'Ours (InsertAnything, 3D-Aware)': {
        #     "base_dir": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter",
        #     "transferred": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/3d_aware/transferred/hair_restored.png",
        #     "hair_mask": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/3d_aware/transferred/hair_restored_mask.png",
        # },
        # 'Ours (FLUX.2-klein-9B, 3D-Unaware)': {
        #     "base_dir": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter",
        #     "transferred": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/3d_unaware/transferred_klein/hair_restored.png",
        #     "hair_mask": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/3d_unaware/transferred_klein/hair_restored_mask.png",
        # },
        # 'Ours (FLUX.2-klein-9B, 3D-Aware)': {
        #     "base_dir": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter",
        #     "transferred": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/3d_aware/transferred_klein/hair_restored.png",
        #     "hair_mask": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/3d_aware/transferred_klein/hair_restored_mask.png",
        # },
        # 'Ours (FLUX.2, 3D-unaware)': {
        #     "base_dir": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter",
        #     "transferred": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/3d_unaware/transferred_kleinn/hair_restored.png",
        #     "hair_mask": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/3d_unaware/transferred_kleinn/hair_restored_mask.png",
        # },
        'Ours (InsertAnything)': {
            "base_dir": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter",
            "transferred": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/3d_aware/transferred/hair_restored.png",
            "hair_mask": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/3d_aware/transferred/hair_restored_mask.png",
        }
    },
    'hairport': {
        'Ours (FLUX.2-klein-9B)': {
            "base_dir": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter",
            "transferred": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/{{awareness}}/transferred_klein/hair_restored.png",
            "hair_mask": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/w_seg/{{awareness}}/transferred_klein/hair_restored_mask.png",
        },
        # 'Ours (FLUX.2, WO Segmentation)': {
        #     "base_dir": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter",
        #     "transferred": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/wo_seg/{{awareness}}/transferred_klein/hair_restored.png",
        #     "hair_mask": f"{BASE_DIR}/view_aligned/shape_hi3dgen__texture_mvadapter/{{target_id}}_to_{{source_id}}/wo_seg/{{awareness}}/transferred_klein/hair_restored_mask.png",
        # },
    },
}

# -----------------------------
# Path Helper Functions
# -----------------------------

def _get_method_paths(paths: Dict[str, Any], method: str) -> Dict[str, str]:
    """
    Get the paths dictionary for a specific method, handling the nested structure
    where baselines are grouped under 'baselines' key and hairport variants under 'hairport' key.
    """
    # Check if method is directly in paths (for backward compatibility)
    if method in paths and isinstance(paths[method], dict) and 'base_dir' in paths[method]:
        return paths[method]
    
    # Check in 'baselines' nested dict
    if 'baselines' in paths and method in paths['baselines']:
        return paths['baselines'][method]
    
    # Check in 'hairport' nested dict
    if 'hairport' in paths and method in paths['hairport']:
        return paths['hairport'][method]
    
    # Build list of available methods for error message
    available_methods = []
    for key, value in paths.items():
        if isinstance(value, dict) and 'base_dir' in value:
            available_methods.append(key)
    if 'baselines' in paths:
        available_methods.extend(paths['baselines'].keys())
    if 'hairport' in paths:
        available_methods.extend(paths['hairport'].keys())
    
    raise KeyError(
        f"Method '{method}' not found in paths configuration. "
        f"Available methods: {sorted(available_methods)}"
    )


def _parse_pair_ids(folder_name: str) -> Tuple[str, str]:
    """Parse folder name in format 'target_id_to_source_id' -> (target_id, source_id)"""
    parts = folder_name.split("_to_")
    if len(parts) != 2:
        raise ValueError(f"Invalid folder name format: {folder_name}. Expected 'target_id_to_source_id'")
    return parts[0], parts[1]


def _find_source_hair_mask(source_id: str, annot_dir: str, filename_pattern: str) -> str:
    """Find source hair mask. First checks BASE_DIR/hair_mask/, then annot_dir/*/filename structure."""
    # First, check the simpler path: BASE_DIR/hair_mask/{source_id}.{ext}
    for ext in ['png', 'jpg', 'jpeg']:
        simple_path = os.path.join(BASE_DIR, "hair_mask", f"{source_id}.{ext}")
        if os.path.exists(simple_path):
            return simple_path
    
    # Fall back to original annotation directory structure
    filename = filename_pattern.format(source_id=source_id)
    file_path = os.path.join(annot_dir, filename)
    return file_path


def _find_reference_hair_mask(target_id: str, annot_dir: str, filename_pattern: str) -> Optional[str]:
    """Find reference (target) hair mask. First checks BASE_DIR/hair_mask/, then annot_dir/*/filename structure."""
    # First, check the simpler path: BASE_DIR/hair_mask/{target_id}.{ext}
    for ext in ['png', 'jpg', 'jpeg']:
        simple_path = os.path.join(BASE_DIR, "hair_mask", f"{target_id}.{ext}")
        if os.path.exists(simple_path):
            return simple_path
    
    # Fall back to original annotation directory structure
    filename = filename_pattern.format(target_id=target_id)
    file_path = os.path.join(annot_dir, filename)
    return file_path

def _discover_pairs(base_dir: str) -> List[Tuple[str, str]]:
    """Discover all valid pairs by listing subdirectories in base_dir."""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    pairs = []
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):
            try:
                target_id, source_id = _parse_pair_ids(folder_name)
                pairs.append((target_id, source_id))
            except ValueError as e:
                print(f"Warning: Skipping folder {folder_name}: {e}")
                continue
    
    return sorted(pairs)


# Type alias for pair info: (target_id, source_id, head_diff_angle)
PairInfo = Tuple[str, str, float]


def _get_awareness_mode(head_diff_angle: float, angle_threshold: float) -> str:
    """
    Determine awareness mode based on head angle difference.
    
    Args:
        head_diff_angle: The head pose angle difference in radians
        angle_threshold: Threshold for switching between 3d_aware and 3d_unaware
    
    Returns:
        '3d_aware' if angle > threshold, '3d_unaware' otherwise
    """
    return '3d_aware' if head_diff_angle > angle_threshold else '3d_unaware'


def _load_pairs_from_csv(
    data_dir: str,
    to_include_angle_threshold: float = 0.1,
) -> List[PairInfo]:
    """
    Load all pairs from pairs.csv with their head_diff_angle values.
    
    Args:
        data_dir: Directory containing pairs.csv (e.g., /workspace/celeba_reduced)
        to_include_angle_threshold: Angle threshold used for awareness mode selection.
            Pairs with head_diff_angle > threshold use '3d_aware' paths,
            pairs with head_diff_angle <= threshold use '3d_unaware' paths.
    
    Returns:
        List of (target_id, source_id, head_diff_angle) tuples for ALL pairs.
    
    Raises:
        FileNotFoundError: If pairs.csv is not found in data_dir
        ValueError: If required columns are missing from pairs.csv
    """
    csv_path = os.path.join(data_dir, "pairs.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"pairs.csv not found at: {csv_path}")
    
    print(f"\nLoading pairs from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_columns = ['source_id', 'target_id', 'lift_3d', 'head_diff_angle']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in pairs.csv: {missing_columns}. "
            f"Required columns: {required_columns}. "
            f"Found columns: {list(df.columns)}"
        )
    
    n_total = len(df)
    print(f"  Total pairs in CSV: {n_total}")
    
    # Count pairs by awareness mode
    n_3d_aware = len(df[df['head_diff_angle'] > to_include_angle_threshold])
    n_3d_unaware = n_total - n_3d_aware
    
    print(f"  Angle threshold: {to_include_angle_threshold:.4f} radians")
    print(f"  Pairs using 3d_aware (angle > threshold): {n_3d_aware}")
    print(f"  Pairs using 3d_unaware (angle <= threshold): {n_3d_unaware}")
    
    # Convert to list of tuples (target_id, source_id, head_diff_angle)
    # Note: Ensure IDs are strings for consistent handling
    pairs = [
        (str(row['target_id']), str(row['source_id']), float(row['head_diff_angle']))
        for _, row in df.iterrows() 
    ]
    # if row['lift_3d'] == True
    return sorted(pairs, key=lambda x: (x[0], x[1]))


def _filter_pairs_by_existence(
    pairs: List[PairInfo],
    method_paths: Dict[str, str],
    method: str,
    angle_threshold: float = 0.1,
) -> List[PairInfo]:
    """
    Filter pairs to only include those that have generated outputs for a method.
    
    For HairPort methods, uses the appropriate awareness mode path based on head angle.
    
    Args:
        pairs: List of (target_id, source_id, head_diff_angle) tuples
        method_paths: Dictionary containing 'transferred' path template
        method: Method name (used to determine if awareness mode applies)
        angle_threshold: Threshold for awareness mode selection
    
    Returns:
        Filtered list of pairs that have existing transferred images
    """
    existing_pairs = []
    uses_awareness = is_hairport_method(method)
    
    for target_id, source_id, head_diff_angle in pairs:
        if uses_awareness:
            awareness = _get_awareness_mode(head_diff_angle, angle_threshold)
            transferred_path = method_paths["transferred"].format(
                target_id=target_id, source_id=source_id, awareness=awareness
            )
        else:
            transferred_path = method_paths["transferred"].format(
                target_id=target_id, source_id=source_id
            )
        
        if os.path.exists(transferred_path):
            existing_pairs.append((target_id, source_id, head_diff_angle))
    
    return existing_pairs


# -----------------------------
# Sample Loading
# -----------------------------

def _load_sample(
    target_id: str, 
    source_id: str, 
    paths: Dict[str, Any], 
    method: str,
    awareness: Optional[str] = None,
) -> Sample:
    """
    Load a single sample given target_id, source_id, paths config, and method name.
    
    Args:
        target_id: ID of the hair donor (reference) image
        source_id: ID of the source (identity) image
        paths: Path configuration dictionary
        method: Method name (e.g., 'hairport_hi3dgen', 'hairfastgan')
        awareness: For HairPort methods, '3d_aware' or '3d_unaware'. 
                   Ignored for baseline methods.
    """
    # Load source image
    source_path = paths["source_image"].format(source_id=source_id)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source image not found: {source_path}")
    source_img = Image.open(source_path).convert("RGB").resize((768, 768), Image.LANCZOS)
    
    # Load reference image
    reference_path = paths["reference_image"].format(target_id=target_id)
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference image not found: {reference_path}")
    reference_img = Image.open(reference_path).convert("RGB").resize((768, 768), Image.LANCZOS)
    
    # Get method-specific paths
    method_paths = _get_method_paths(paths, method)
    
    # Load generated image (use awareness for HairPort methods)
    if is_hairport_method(method) and awareness is not None:
        generated_path = method_paths["transferred"].format(
            target_id=target_id, source_id=source_id, awareness=awareness
        )
    else:
        generated_path = method_paths["transferred"].format(
            target_id=target_id, source_id=source_id
        )
    if not os.path.exists(generated_path):
        raise FileNotFoundError(f"Generated image not found: {generated_path}")
    generated_img = Image.open(generated_path).convert("RGB").resize((768, 768), Image.LANCZOS)
    
    # Load source hair mask
    source_mask_config = paths["source_hair_mask"]
    source_mask_path = _find_source_hair_mask(
        source_id,
        source_mask_config["annot_dir"],
        source_mask_config["filename"]
    )
    source_hair_mask = Image.open(source_mask_path).convert("L").resize((768, 768), Image.NEAREST)
    
    # Load generated hair mask (use awareness for HairPort methods)
    if is_hairport_method(method) and awareness is not None:
        generated_mask_path = method_paths["hair_mask"].format(
            target_id=target_id, source_id=source_id, awareness=awareness
        )
    else:
        generated_mask_path = method_paths["hair_mask"].format(
            target_id=target_id, source_id=source_id
        )
    if not os.path.exists(generated_mask_path):
        raise FileNotFoundError(f"Generated hair mask not found: {generated_mask_path}")
    generated_hair_mask = Image.open(generated_mask_path).convert("L").resize((768, 768), Image.NEAREST)
    
    # Load reference hair mask (optional)
    reference_hair_mask = None
    if "reference_hair_mask" in paths:
        ref_mask_config = paths["reference_hair_mask"]
        ref_mask_path = _find_reference_hair_mask(
            target_id,
            ref_mask_config["annot_dir"],
            ref_mask_config["filename"]
        )
        if ref_mask_path is not None:
            reference_hair_mask = Image.open(ref_mask_path).convert("L").resize((768, 768), Image.NEAREST)
    
    return Sample(
        source=source_img,
        generated=generated_img,
        reference=reference_img,
        hair_mask_source=source_hair_mask,
        hair_mask_generated=generated_hair_mask,
        hair_mask_reference=reference_hair_mask
    )


# -----------------------------
# Hair Mask Generation
# -----------------------------

def _ensure_hair_masks_exist(
    pairs: List[PairInfo],
    paths: Dict[str, Any],
    method: str,
    angle_threshold: float = 0.1,
    device: str = "cuda",
) -> Tuple[int, int]:
    """
    Ensure hair masks exist for all pairs. Generate missing masks using SAMMaskExtractor.
    
    For HairPort methods, uses awareness mode based on head angle.
    """
    print("\nChecking for missing hair masks...")
    
    method_paths = _get_method_paths(paths, method)
    uses_awareness = is_hairport_method(method)
    
    missing_masks = []
    existing_count = 0
    
    for target_id, source_id, head_diff_angle in pairs:
        if uses_awareness:
            awareness = _get_awareness_mode(head_diff_angle, angle_threshold)
            mask_path = method_paths["hair_mask"].format(
                target_id=target_id, source_id=source_id, awareness=awareness
            )
            transferred_path = method_paths["transferred"].format(
                target_id=target_id, source_id=source_id, awareness=awareness
            )
        else:
            mask_path = method_paths["hair_mask"].format(target_id=target_id, source_id=source_id)
            transferred_path = method_paths["transferred"].format(target_id=target_id, source_id=source_id)
        
        if os.path.exists(mask_path):
            existing_count += 1
        elif os.path.exists(transferred_path):
            missing_masks.append({
                'target_id': target_id,
                'source_id': source_id,
                'transferred_path': transferred_path,
                'mask_path': mask_path,
            })
    
    print(f"  Existing masks: {existing_count}")
    print(f"  Missing masks: {len(missing_masks)}")
    
    if len(missing_masks) == 0:
        print("  All hair masks already exist!")
        return existing_count, 0
    
    if not SAM3_AVAILABLE:
        print("  Warning: SAM3 not available, cannot generate missing masks")
        return existing_count, 0
    
    print(f"\nGenerating {len(missing_masks)} missing hair masks using SAMMaskExtractor...")
    mask_extractor = SAMMaskExtractor(confidence_threshold=0.4)
    
    generated_count = 0
    failed_count = 0
    
    for mask_info in tqdm(missing_masks, desc="Generating hair masks", unit="mask"):
        try:
            transferred_img = Image.open(mask_info['transferred_path']).convert("RGB")
            hair_mask = mask_extractor(transferred_img)[0]
            
            os.makedirs(os.path.dirname(mask_info['mask_path']), exist_ok=True)
            hair_mask.save(mask_info['mask_path'])
            
            generated_count += 1
            
        except Exception as e:
            tqdm.write(f"✗ Error for {mask_info['target_id']}_to_{mask_info['source_id']}: {e}")
            failed_count += 1
    
    del mask_extractor
    if device == "cuda":
        torch.cuda.empty_cache()
    
    print(f"\nHair mask generation complete:")
    print(f"  Generated: {generated_count}")
    print(f"  Failed: {failed_count}")
    
    return existing_count, generated_count


# -----------------------------
# Prompt Generation
# -----------------------------

def _ensure_prompts_exist(
    pairs: List[PairInfo],
    paths: Dict[str, Any],
    method: str,
    angle_threshold: float = 0.1,
    device: str = "cuda",
) -> Tuple[int, int]:
    """
    Ensure prompt.json exists for all pairs. Generate missing prompts using CaptionerPipeline with hair-only mode.
    
    For HairPort methods, uses awareness mode based on head angle.
    """
    print("\nChecking for missing prompt.json files...")
    
    method_paths = _get_method_paths(paths, method)
    uses_awareness = is_hairport_method(method)
    
    missing_prompts = []
    existing_count = 0
    
    for target_id, source_id, head_diff_angle in pairs:
        if uses_awareness:
            awareness = _get_awareness_mode(head_diff_angle, angle_threshold)
            transferred_path = method_paths["transferred"].format(
                target_id=target_id, source_id=source_id, awareness=awareness
            )
        else:
            transferred_path = method_paths["transferred"].format(target_id=target_id, source_id=source_id)
        
        prompt_path = os.path.join(os.path.dirname(transferred_path), "prompt.json")
        
        if os.path.exists(prompt_path):
            existing_count += 1
        elif os.path.exists(transferred_path):
            missing_prompts.append({
                'target_id': target_id,
                'source_id': source_id,
                'transferred_path': transferred_path,
                'prompt_path': prompt_path,
            })
    
    print(f"  Existing prompts: {existing_count}")
    print(f"  Missing prompts: {len(missing_prompts)}")
    
    if len(missing_prompts) == 0:
        print("  All prompt.json files already exist!")
        return existing_count, 0
    
    if not CAPTIONER_AVAILABLE:
        print("  Warning: CaptionerPipeline not available, cannot generate missing prompts")
        return existing_count, 0
    
    print(f"\nGenerating {len(missing_prompts)} missing prompt.json files using CaptionerPipeline (hair-only mode)...")
    captioner = CaptionerPipeline(use_flash_attention=True)
    
    generated_count = 0
    failed_count = 0
    batch_size = 16
    
    # Process in batches for efficiency
    for batch_start in tqdm(range(0, len(missing_prompts), batch_size), desc="Generating prompts", unit="batch"):
        batch = missing_prompts[batch_start:batch_start + batch_size]
        batch_paths = [info['transferred_path'] for info in batch]
        
        try:
            # Generate hair descriptions in batch using CaptionerPipeline
            hair_descriptions = captioner.caption_images_batch(
                batch_paths,
                prompt_type="hair",
                use_true_batch=True,
            )
            
            # Save each result in the batch
            for prompt_info, hair_description in zip(batch, hair_descriptions):
                try:
                    # Create structured JSON matching captioner_pipeline.py output format
                    data = {
                        "scene": "",
                        "subject": [
                            {
                                "description": "",
                                "hair_description": hair_description,
                                "position": "",
                                "action": "",
                            }
                        ],
                        "style": "",
                        "lighting": "",
                        "background": "",
                        "composition": "",
                        "camera": {
                            "angle": "",
                            "lens": "",
                            "depth_of_field": "",
                        },
                    }
                    
                    os.makedirs(os.path.dirname(prompt_info['prompt_path']), exist_ok=True)
                    with open(prompt_info['prompt_path'], 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    generated_count += 1
                    
                except Exception as e:
                    tqdm.write(f"✗ Error saving {prompt_info['target_id']}_to_{prompt_info['source_id']}: {e}")
                    failed_count += 1
                    
        except Exception as e:
            tqdm.write(f"✗ Batch error: {e}")
            failed_count += len(batch)
    
    # Clean up captioner model
    del captioner
    if device == "cuda":
        torch.cuda.empty_cache()
    
    print(f"\nPrompt generation complete:")
    print(f"  Generated: {generated_count}")
    print(f"  Failed: {failed_count}")
    
    return existing_count, generated_count


def _ensure_reference_prompts_exist(
    pairs: List[PairInfo],
    paths: Dict[str, Any],
    device: str = "cuda",
) -> Tuple[int, int]:
    """
    Ensure prompt.json exists for all reference (hair donor) images.
    Reference images are identified by target_id and stored in {BASE_DIR}/prompt/{target_id}.json
    """
    print("\nChecking for missing reference image prompts...")
    
    # Collect unique target_ids (hair donor images) - pairs are now (target_id, source_id, angle)
    unique_target_ids = sorted(set(target_id for target_id, _, _ in pairs))
    
    prompt_dir = os.path.join(BASE_DIR, "prompt")
    os.makedirs(prompt_dir, exist_ok=True)
    
    missing_prompts = []
    existing_count = 0
    
    for target_id in unique_target_ids:
        prompt_path = os.path.join(prompt_dir, f"{target_id}.json")
        image_path = paths["reference_image"].format(target_id=target_id)
        
        if os.path.exists(prompt_path):
            existing_count += 1
        elif os.path.exists(image_path):
            missing_prompts.append({
                'target_id': target_id,
                'image_path': image_path,
                'prompt_path': prompt_path,
            })
    
    print(f"  Existing reference prompts: {existing_count}")
    print(f"  Missing reference prompts: {len(missing_prompts)}")
    
    if len(missing_prompts) == 0:
        print("  All reference prompt.json files already exist!")
        return existing_count, 0
    
    if not CAPTIONER_AVAILABLE:
        print("  Warning: CaptionerPipeline not available, cannot generate missing reference prompts")
        return existing_count, 0
    
    print(f"\nGenerating {len(missing_prompts)} missing reference prompts using CaptionerPipeline (hair-only mode)...")
    captioner = CaptionerPipeline(use_flash_attention=True)
    
    generated_count = 0
    failed_count = 0
    batch_size = 8
    
    # Process in batches for efficiency
    for batch_start in tqdm(range(0, len(missing_prompts), batch_size), desc="Generating reference prompts", unit="batch"):
        batch = missing_prompts[batch_start:batch_start + batch_size]
        batch_paths = [info['image_path'] for info in batch]
        
        try:
            # Generate hair descriptions in batch using CaptionerPipeline
            hair_descriptions = captioner.caption_images_batch(
                batch_paths,
                prompt_type="hair",
                use_true_batch=True,
            )
            
            # Save each result in the batch
            for prompt_info, hair_description in zip(batch, hair_descriptions):
                try:
                    # Create structured JSON matching captioner_pipeline.py output format
                    data = {
                        "scene": "",
                        "subject": [
                            {
                                "description": "",
                                "hair_description": hair_description,
                                "position": "",
                                "action": "",
                            }
                        ],
                        "style": "",
                        "lighting": "",
                        "background": "",
                        "composition": "",
                        "camera": {
                            "angle": "",
                            "lens": "",
                            "depth_of_field": "",
                        },
                    }
                    
                    with open(prompt_info['prompt_path'], 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    
                    generated_count += 1
                    
                except Exception as e:
                    tqdm.write(f"✗ Error saving reference {prompt_info['target_id']}: {e}")
                    failed_count += 1
                    
        except Exception as e:
            tqdm.write(f"✗ Batch error: {e}")
            failed_count += len(batch)
    
    # Clean up captioner model
    del captioner
    if device == "cuda":
        torch.cuda.empty_cache()
    
    print(f"\nReference prompt generation complete:")
    print(f"  Generated: {generated_count}")
    print(f"  Failed: {failed_count}")
    
    return existing_count, generated_count


# -----------------------------
# Output Directory Structure
# -----------------------------

# HairPort method prefix - all methods starting with this are considered "ours"
HAIRPORT_METHOD_PREFIX = 'hairport'

# Methods that use awareness mode (require {awareness} placeholder in paths)
# These are methods grouped under 'hairport' in the PATHS configuration
AWARENESS_METHODS = set()  # Populated dynamically from PATHS


def _get_awareness_methods(paths: Dict[str, Any] = None) -> set:
    """Get the set of methods that use awareness mode from paths configuration."""
    if paths is None:
        paths = PATHS
    if 'hairport' in paths:
        return set(paths['hairport'].keys())
    return set()


def is_hairport_method(method: str, paths: Dict[str, Any] = None) -> bool:
    """
    Check if a method is a HairPort variant (uses awareness mode).
    
    A method is considered a HairPort variant if:
    1. It's listed under the 'hairport' key in the paths configuration, OR
    2. Its name starts with 'hairport' (for backward compatibility)
    
    HairPort variants use the {awareness} placeholder in their path templates
    and require awareness mode selection based on head angle.
    """
    awareness_methods = _get_awareness_methods(paths)
    return method in awareness_methods or method.startswith(HAIRPORT_METHOD_PREFIX)


def _get_output_dirs(base_output_dir: str, method: str) -> Dict[str, str]:
    """
    Get output directory paths for a method.
    
    Structure:
    - evaluation/
      - hairport/           # All HairPort variants go here
        - hairport/         # Base HairPort method
        - hairport_hi3dgen/
        - hairport_direct3d_s2/
      - baselines/          # All baseline methods
        - hairfastgan/
        - stablehair/
        - ...
      - comparison/
      - latex_report/
    """
    if is_hairport_method(method):
        # All HairPort variants go under hairport/ directory
        method_dir = os.path.join(base_output_dir, "hairport", method)
    else:
        # Baselines go under baselines/ directory
        method_dir = os.path.join(base_output_dir, "baselines", method)
    
    return {
        'method': method_dir,
        'comparison': os.path.join(base_output_dir, "comparison"),
        'latex_report': os.path.join(base_output_dir, "latex_report"),
    }


# -----------------------------
# Expected Metrics Configuration
# -----------------------------

# All metrics that should be computed for each method
EXPECTED_METRICS = {
    # Global metrics (computed once for entire dataset)
    'global': ['clip_i', 'fid', 'fid_clip'],
    # Per-sample metrics (computed for each pair)
    'per_sample': [
        'ssim_nonhair_intersection',
        'psnr_nonhair_intersection',
        'lpips',  # LPIPS on non-hair intersection region
        # 'dreamsim',  # DreamSim on hair region (reference vs generated)
        'ids',
        'clip_i_per_sample',
        'dinov3_hair_similarity', 
        'dinov2_hair_similarity',
        # 'pixio_hair_similarity',  # Pixio on hair region (reference vs generated)
        # 'hair_prompt_similarity',
    ],
}

# Metadata columns (not metrics)
METADATA_COLS = ['target_id', 'source_id', 'pair_name']


# -----------------------------
# Failure Detection and Filtering
# -----------------------------

def _detect_failure_cases(
    df_results: pd.DataFrame,
    method: str,
    max_failure_rate: float = 0.10,
    detect_outliers: bool = True,
    outlier_std_threshold: float = 4.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Detect and filter out failure cases from evaluation results.
    
    Failure criteria (applied in order):
    1. Technical failures: NaN values in critical metrics (IDS, identity-related)
    2. Face detection failures: IDS is NaN (face not detected)
    3. Hair detection failures: DINOv2 hair similarity is NaN (hair region too small/missing)
    4. Optional: Extreme outliers beyond outlier_std_threshold standard deviations
    
    Args:
        df_results: DataFrame with evaluation results
        method: Method name for logging
        max_failure_rate: Maximum allowed failure rate (default 0.10 = 10%)
        detect_outliers: Whether to detect and remove extreme outliers
        outlier_std_threshold: Number of standard deviations for outlier detection
        
    Returns:
        Tuple of (filtered_df, failures_df, summary_dict)
    """
    print(f"\n{'='*60}")
    print(f"Detecting Failure Cases for {method}")
    print(f"{'='*60}")
    
    n_total = len(df_results)
    metadata_cols = METADATA_COLS
    metric_cols = [col for col in df_results.columns if col not in metadata_cols]
    
    # Track failure reasons for each sample
    failure_mask = pd.Series([False] * n_total, index=df_results.index)
    failure_reasons = {idx: [] for idx in df_results.index}
    
    # 1. Check for critical NaN values (face/identity detection failures)
    critical_metrics = ['ids']  # Identity similarity is critical - indicates face detection failure
    for metric in critical_metrics:
        if metric in df_results.columns:
            nan_mask = df_results[metric].isna()
            n_nan = nan_mask.sum()
            if n_nan > 0:
                print(f"  Found {n_nan} samples with NaN in {metric} (face detection failed)")
                failure_mask |= nan_mask
                for idx in df_results[nan_mask].index:
                    failure_reasons[idx].append(f"{metric}_nan")
    
    # 2. Check for hair detection failures (DINOv3 hair similarity NaN)
    hair_metrics = ['dinov3_hair_similarity']
    for metric in hair_metrics:
        if metric in df_results.columns:
            nan_mask = df_results[metric].isna()
            n_nan = nan_mask.sum()
            if n_nan > 0:
                print(f"  Found {n_nan} samples with NaN in {metric} (hair region too small/missing)")
                failure_mask |= nan_mask
                for idx in df_results[nan_mask].index:
                    if f"{metric}_nan" not in failure_reasons[idx]:
                        failure_reasons[idx].append(f"{metric}_nan")
    
    # 3. Optional: Detect extreme outliers
    if detect_outliers:
        print(f"\n  Detecting extreme outliers (>{outlier_std_threshold}σ)...")
        # Only check per-sample metrics (not global metrics)
        per_sample_metric_cols = [m for m in metric_cols if m not in GLOBAL_METRICS]
        
        for metric in per_sample_metric_cols:
            if metric not in df_results.columns or df_results[metric].isna().all():
                continue
            
            # Compute statistics on non-failed samples only
            valid_values = df_results.loc[~failure_mask, metric].dropna()
            if len(valid_values) < 10:  # Need at least 10 samples for meaningful statistics
                continue
            
            mean_val = valid_values.mean()
            std_val = valid_values.std()
            
            if std_val == 0 or np.isnan(std_val):
                continue
            
            # Detect outliers based on direction
            higher_is_better = METRIC_HIGHER_IS_BETTER.get(metric, True)
            
            # For metrics where lower is better, flag extremely high values
            # For metrics where higher is better, flag extremely low values
            if higher_is_better:
                # Flag values that are too low (e.g., IDS < mean - 4*std)
                outlier_mask = (df_results[metric] < (mean_val - outlier_std_threshold * std_val)) & ~failure_mask
            else:
                # Flag values that are too high (e.g., LPIPS > mean + 4*std)
                outlier_mask = (df_results[metric] > (mean_val + outlier_std_threshold * std_val)) & ~failure_mask
            
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                print(f"    {metric}: {n_outliers} extreme outliers detected")
                failure_mask |= outlier_mask
                for idx in df_results[outlier_mask].index:
                    failure_reasons[idx].append(f"{metric}_outlier")
    
    # Calculate failure statistics
    n_failures = failure_mask.sum()
    failure_rate = n_failures / n_total if n_total > 0 else 0.0
    
    print(f"\n  Total failures detected: {n_failures}/{n_total} ({failure_rate*100:.2f}%)")
    
    # Check if failure rate exceeds maximum
    if failure_rate > max_failure_rate:
        print(f"  ⚠ Warning: Failure rate ({failure_rate*100:.2f}%) exceeds maximum ({max_failure_rate*100:.2f}%)")
        print(f"  Consider adjusting outlier detection threshold or investigating data quality")
    
    # Create failure DataFrame with reasons
    failures_df = df_results[failure_mask].copy()
    failures_df['failure_reasons'] = failures_df.index.map(lambda idx: ', '.join(failure_reasons[idx]))
    
    # Create filtered DataFrame (excluding failures)
    filtered_df = df_results[~failure_mask].copy()
    
    # Summary statistics
    summary = {
        'method': method,
        'n_total': int(n_total),  # Convert to native Python int
        'n_failures': int(n_failures),  # Convert to native Python int
        'n_valid': int(len(filtered_df)),  # Convert to native Python int
        'failure_rate': float(failure_rate),  # Convert to native Python float
        'failure_breakdown': {},
    }
    
    # Count failures by type
    from collections import Counter
    all_reasons = []
    for reasons in failure_reasons.values():
        all_reasons.extend(reasons)
    # Convert Counter values to native Python int
    summary['failure_breakdown'] = {k: int(v) for k, v in Counter(all_reasons).items()}
    
    print(f"  Valid samples retained: {len(filtered_df)}/{n_total}")
    if summary['failure_breakdown']:
        print(f"\n  Failure breakdown:")
        for reason, count in sorted(summary['failure_breakdown'].items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")
    
    return filtered_df, failures_df, summary


def _save_failure_report(
    failures_df: pd.DataFrame,
    summary: Dict[str, Any],
    output_dir: str,
    method: str,
):
    """Save detailed failure report to disk."""
    if len(failures_df) == 0:
        print(f"  No failures to report for {method}")
        return
    
    # Save failures CSV
    failures_csv = os.path.join(output_dir, "failure_cases.csv")
    failures_df.to_csv(failures_csv, index=False)
    print(f"  Failure cases saved to: {failures_csv}")
    
    # Save failure summary JSON
    summary_json = os.path.join(output_dir, "failure_summary.json")
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Failure summary saved to: {summary_json}")
    
    # Create failure report text file
    report_path = os.path.join(output_dir, "failure_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Failure Detection Report: {method}\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total samples: {summary['n_total']}\n")
        f.write(f"Failed samples: {summary['n_failures']}\n")
        f.write(f"Valid samples: {summary['n_valid']}\n")
        f.write(f"Failure rate: {summary['failure_rate']*100:.2f}%\n\n")
        
        if summary['failure_breakdown']:
            f.write("Failure Breakdown:\n")
            f.write("-"*40 + "\n")
            for reason, count in sorted(summary['failure_breakdown'].items(), key=lambda x: -x[1]):
                pct = (count / summary['n_failures']) * 100 if summary['n_failures'] > 0 else 0
                f.write(f"  {reason}: {count} ({pct:.1f}%)\n")
            f.write("\n")
        
        f.write("\nFailed Samples:\n")
        f.write("-"*40 + "\n")
        for _, row in failures_df.iterrows():
            f.write(f"  {row['pair_name']}: {row['failure_reasons']}\n")
    
    print(f"  Failure report saved to: {report_path}")


# Metric direction configuration (imported from latex_builder but defined here for independence)
METRIC_HIGHER_IS_BETTER = {
    'clip_i': True,
    'clip_i_per_sample': True,
    'fid': False,
    'fid_clip': False,
    'ssim_nonhair_intersection': True,
    'psnr_nonhair_intersection': True,
    'lpips': False,
    'dreamsim': False,
    'ids': True,
    'dinov3_hair_similarity': True,
    'pixio_hair_similarity': True,
    'hair_prompt_similarity': True,
}


# Metrics that require prompt generation (disabled by default)
PROMPT_DEPENDENT_METRICS = ['hair_prompt_similarity']


def _get_missing_metrics(
    existing_df: Optional[pd.DataFrame],
    enable_prompt_metrics: bool = False,
) -> Dict[str, List[str]]:
    """
    Determine which metrics are missing from existing results.
    
    Args:
        existing_df: DataFrame with existing results, or None if no results exist
        enable_prompt_metrics: If True, include prompt-dependent metrics (default: False)
        
    Returns:
        Dictionary with 'global' and 'per_sample' lists of missing metric names
    """
    # Filter out prompt-dependent metrics if disabled
    expected_global = EXPECTED_METRICS['global'].copy()
    expected_per_sample = EXPECTED_METRICS['per_sample'].copy()
    
    if not enable_prompt_metrics:
        expected_per_sample = [m for m in expected_per_sample if m not in PROMPT_DEPENDENT_METRICS]
    
    if existing_df is None:
        return {
            'global': expected_global,
            'per_sample': expected_per_sample,
        }
    
    existing_cols = set(existing_df.columns)
    
    missing_global = [m for m in expected_global if m not in existing_cols]
    missing_per_sample = [m for m in expected_per_sample if m not in existing_cols]
    
    return {
        'global': missing_global,
        'per_sample': missing_per_sample,
    }


def _load_existing_results(output_dir: str, method: str) -> Optional[pd.DataFrame]:
    """
    Load existing per-pair results for a method if they exist.
    
    Returns:
        DataFrame with existing results, or None if no results exist
    """
    output_dirs = _get_output_dirs(output_dir, method)
    method_output_dir = output_dirs['method']
    per_pair_csv = os.path.join(method_output_dir, "per_pair_results.csv")
    
    if not os.path.exists(per_pair_csv):
        return None
    
    try:
        df = pd.read_csv(per_pair_csv)
        return df
    except Exception as e:
        print(f"  Warning: Could not load existing results: {e}")
        return None


# -----------------------------
# Main Evaluation Functions
# -----------------------------

def evaluate_method(
    output_dir: str,
    base_dir: str = BASE_DIR,
    paths: Dict[str, Any] = None,
    method: str = "hairport",
    device: str = "cuda",
    batch_size: int = 16,
    force_recompute: bool = False,
    enable_prompt_metrics: bool = False,
    to_include_angle_threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Evaluate a hair transfer method on all available pairs.
    
    This function intelligently skips metrics that have already been computed,
    only computing missing metrics and merging them with existing results.
    
    Pair Selection and Awareness Mode:
    Pairs are loaded from pairs.csv in base_dir, which must contain columns:
    source_id, target_id, lift_3d, and head_diff_angle. ALL pairs are included
    in evaluation.
    
    For HairPort methods, the awareness mode is determined by head_diff_angle:
    - head_diff_angle > to_include_angle_threshold: uses '3d_aware' path
    - head_diff_angle <= to_include_angle_threshold: uses '3d_unaware' path
    
    For baseline methods, the angle threshold has no effect on path selection.
    
    Args:
        output_dir: Base evaluation directory
        base_dir: Base directory containing images and pairs.csv
        paths: Path configuration dictionary (default: PATHS)
        method: Method name to evaluate
        device: Device for computation
        batch_size: Batch size for metric computation
        force_recompute: If True, recompute all metrics even if they exist
        enable_prompt_metrics: If True, generate prompts and compute hair_prompt_similarity
            metric. Requires CaptionerPipeline. Default: False
        to_include_angle_threshold: Angle threshold (in radians) for HairPort awareness mode.
            Pairs with head_diff_angle > threshold use '3d_aware' paths,
            pairs with head_diff_angle <= threshold use '3d_unaware' paths.
            Default: 0.1 radians.
    
    Returns:
        Dictionary containing aggregated results and paths to saved files
    """
    if paths is None:
        paths = PATHS
    
    print("="*80)
    print(f"EVALUATING METHOD: {method}")
    print("="*80)
    
    # Get output directories
    output_dirs = _get_output_dirs(output_dir, method)
    method_output_dir = output_dirs['method']
    os.makedirs(method_output_dir, exist_ok=True)
    print(f"Output directory: {method_output_dir}")
    
    # Check for existing results and determine missing metrics
    existing_df = None if force_recompute else _load_existing_results(output_dir, method)
    missing_metrics = _get_missing_metrics(existing_df, enable_prompt_metrics=enable_prompt_metrics)
    
    all_missing = missing_metrics['global'] + missing_metrics['per_sample']
    
    if existing_df is not None and len(all_missing) == 0:
        print("\n✓ All metrics already computed. Nothing to do.")
        # Return existing results
        metadata_cols = METADATA_COLS
        metric_cols = [col for col in existing_df.columns if col not in metadata_cols]
        df_metrics = existing_df[metric_cols]
        aggregate_results = {
            'mean': df_metrics.mean().to_dict(),
            'std': df_metrics.std().to_dict(),
            'median': df_metrics.median().to_dict(),
            'min': df_metrics.min().to_dict(),
            'max': df_metrics.max().to_dict(),
        }
        global_metrics_values = {
            'clip_i': existing_df['clip_i'].iloc[0] if 'clip_i' in existing_df.columns else None,
            'fid': existing_df['fid'].iloc[0] if 'fid' in existing_df.columns else None,
            'fid_clip': existing_df['fid_clip'].iloc[0] if 'fid_clip' in existing_df.columns else None,
        }
        # Check if failure summary exists
        failure_summary_path = os.path.join(method_output_dir, "failure_summary.json")
        n_failures = 0
        failure_rate = 0.0
        if os.path.exists(failure_summary_path):
            try:
                with open(failure_summary_path, 'r') as f:
                    fs = json.load(f)
                    n_failures = fs.get('n_failures', 0)
                    failure_rate = fs.get('failure_rate', 0.0)
            except:
                pass
        
        return {
            'method': method,
            'n_pairs_evaluated': len(existing_df),
            'n_pairs_failed': 0,
            'n_failure_cases_removed': n_failures,
            'failure_rate': failure_rate,
            'aggregate_results': aggregate_results,
            'global_metrics': global_metrics_values,
            'output_dir': method_output_dir,
            'per_pair_csv': os.path.join(method_output_dir, "per_pair_results.csv"),
            'aggregate_json': os.path.join(method_output_dir, "aggregate_results.json"),
            'per_pair_df': existing_df,
        }
    
    if existing_df is not None:
        print(f"\n→ Found existing results with {len(existing_df)} pairs")
        print(f"  Missing global metrics: {missing_metrics['global'] or 'None'}")
        print(f"  Missing per-sample metrics: {missing_metrics['per_sample'] or 'None'}")
    else:
        print("\n→ No existing results found. Computing all metrics.")
    
    # Get method-specific paths
    method_paths = _get_method_paths(paths, method)
    
    # Load ALL pairs from CSV (no filtering by angle - awareness mode handles this)
    print(f"\nLoading pairs from pairs.csv (angle threshold = {to_include_angle_threshold} for awareness mode)")
    all_pairs = _load_pairs_from_csv(
        data_dir=base_dir,
        to_include_angle_threshold=to_include_angle_threshold,
    )
    
    # Filter to pairs that exist for this method (awareness-aware for HairPort)
    pairs = _filter_pairs_by_existence(all_pairs, method_paths, method, angle_threshold=to_include_angle_threshold)
    print(f"Pairs with existing outputs for {method}: {len(pairs)}/{len(all_pairs)}")
    
    if len(pairs) == 0:
        raise ValueError(
            f"No valid pairs found for method '{method}'. "
            f"Checked {len(all_pairs)} pairs from pairs.csv against {method_paths['base_dir']}"
        )
    
    # Ensure hair masks exist (awareness-aware for HairPort)
    n_existing, n_generated = _ensure_hair_masks_exist(
        pairs, paths, method, angle_threshold=to_include_angle_threshold, device=device
    )
    
    # Ensure prompt.json files exist (only if prompt metrics are enabled)
    if enable_prompt_metrics:
        # Ensure prompt.json files exist for transferred images
        n_prompts_existing, n_prompts_generated = _ensure_prompts_exist(
            pairs, paths, method, angle_threshold=to_include_angle_threshold, device=device
        )
        
        # Ensure prompt.json files exist for reference (hair donor) images
        n_ref_prompts_existing, n_ref_prompts_generated = _ensure_reference_prompts_exist(pairs, paths, device=device)
    else:
        print("\nSkipping prompt generation (enable_prompt_metrics=False)")
    # Initialize device
    device = device if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Check if we need to load samples (only if there are missing metrics)
    samples = None
    sample_metadata = None
    failed_pairs = []
    uses_awareness = is_hairport_method(method)
    
    if len(all_missing) > 0:
        # Load all samples
        print("\nLoading samples...")
        samples = []
        sample_metadata = []
        
        for target_id, source_id, head_diff_angle in tqdm(pairs, desc="Loading samples", unit="pair"):
            try:
                # Determine awareness mode for HairPort methods
                awareness = _get_awareness_mode(head_diff_angle, to_include_angle_threshold) if uses_awareness else None
                sample = _load_sample(target_id, source_id, paths, method, awareness=awareness)
                samples.append(sample)
                sample_metadata.append({
                    'target_id': target_id,
                    'source_id': source_id,
                    'pair_name': f"{target_id}_to_{source_id}",
                    'head_diff_angle': head_diff_angle,
                    'awareness': awareness,
                })
            except Exception as e:
                tqdm.write(f"✗ Error loading {target_id}_to_{source_id}: {e}")
                failed_pairs.append({
                    'target_id': target_id,
                    'source_id': source_id,
                    'pair_name': f"{target_id}_to_{source_id}",
                    'error': str(e)
                })
        
        if len(samples) == 0:
            raise RuntimeError("No samples were successfully loaded")
        
        print(f"\nSuccessfully loaded: {len(samples)}/{len(pairs)} samples")
    
    # Compute missing metrics
    print("\nComputing missing metrics...")
    all_metrics = {}
    per_sample_metrics = {}
    
    # --- Global metrics ---
    if 'clip_i' in missing_metrics['global']:
        print("  Computing CLIP-I...")
        clip_i_metric = CLIPIMetric(device=device, batch_size=batch_size)
        all_metrics['clip_i'] = clip_i_metric.compute(samples)
        # Also get per-sample CLIP-I if needed
        if 'clip_i_per_sample' in missing_metrics['per_sample']:
            per_sample_metrics['clip_i_per_sample'] = clip_i_metric.compute_per_sample(samples)
    elif existing_df is not None and 'clip_i' in existing_df.columns:
        all_metrics['clip_i'] = existing_df['clip_i'].iloc[0]
    
    if 'fid' in missing_metrics['global']:
        print("  Computing FID (Inception V3)...")
        fid_metric = FIDMetric(device=device, batch_size=batch_size)
        all_metrics['fid'] = fid_metric.compute(samples)
    elif existing_df is not None and 'fid' in existing_df.columns:
        all_metrics['fid'] = existing_df['fid'].iloc[0]
    
    if 'fid_clip' in missing_metrics['global']:
        print("  Computing FID_CLIP (CLIP encoder)...")
        fid_clip_metric = FIDCLIPMetric(device=device, batch_size=batch_size)
        all_metrics['fid_clip'] = fid_clip_metric.compute(samples)
    elif existing_df is not None and 'fid_clip' in existing_df.columns:
        all_metrics['fid_clip'] = existing_df['fid_clip'].iloc[0]
    
    # --- Per-sample metrics ---
    
    # CLIP-I per sample (if not already computed with global CLIP-I)
    if 'clip_i_per_sample' in missing_metrics['per_sample'] and 'clip_i_per_sample' not in per_sample_metrics:
        print("  Computing CLIP-I (per-sample)...")
        clip_i_metric = CLIPIMetric(device=device, batch_size=batch_size)
        per_sample_metrics['clip_i_per_sample'] = clip_i_metric.compute_per_sample(samples)
    
    # SSIM (masked)
    if 'ssim_nonhair_intersection' in missing_metrics['per_sample']:
        ssim_metric = SSIMMetric()
        ssim_results = [
            ssim_metric.compute([s]) 
            for s in tqdm(samples, desc="Computing SSIM (masked)", unit="sample", leave=False)
        ]
        per_sample_metrics['ssim_nonhair_intersection'] = ssim_results
    
    # PSNR (masked)
    if 'psnr_nonhair_intersection' in missing_metrics['per_sample']:
        psnr_metric = PSNRMetric()
        psnr_results = [
            psnr_metric.compute([s]) 
            for s in tqdm(samples, desc="Computing PSNR (masked)", unit="sample", leave=False)
        ]
        per_sample_metrics['psnr_nonhair_intersection'] = psnr_results
    
    # LPIPS (masked)
    if 'lpips' in missing_metrics['per_sample']:
        lpips_metric = LPIPSMetric(device=device, net="alex", region="nonhair_intersection")
        lpips_metric._load_model()
        lpips_results = []
        for sample in tqdm(samples, desc="Computing LPIPS (masked)", unit="sample", leave=False):
            src = _to_rgb(sample.source)
            gen = _resize_like(_to_rgb(sample.generated), src, resample=Image.BICUBIC)
            w = _intersected_nonhair_weights(src, gen, sample.hair_mask_source, sample.hair_mask_generated)
            src2, gen2 = lpips_metric._composite_to_localize_diff(src, gen, w)
            with torch.inference_mode():
                x = _pil_to_torch_float01(src2, lpips_metric.device) * 2.0 - 1.0
                y = _pil_to_torch_float01(gen2, lpips_metric.device) * 2.0 - 1.0
                d = lpips_metric.model(x, y)
                lpips_results.append(float(d.detach().cpu().item()))
        lpips_metric._unload_model()
        per_sample_metrics['lpips'] = lpips_results
    
    # DreamSim (on hair region: reference_hair vs generated_hair)
    if 'dreamsim' in missing_metrics['per_sample']:
        print("  Computing DreamSim (hair region)...")
        dreamsim_metric = DreamSimMetric(device=device, extraction_mode="masked")
        dreamsim_results = dreamsim_metric.compute_per_sample(samples)
        dreamsim_results = [r if r is not None else float('nan') for r in dreamsim_results]
        per_sample_metrics['dreamsim'] = dreamsim_results
    
    # IDS
    if 'ids' in missing_metrics['per_sample']:
        print("  Computing IDS...")
        ids_metric = IDSMetric(device=device)
        ids_results = ids_metric.compute_per_sample(samples)
        ids_results = [r if r is not None else float('nan') for r in ids_results]
        per_sample_metrics['ids'] = ids_results
    
    if 'dinov3_hair_similarity' in missing_metrics['per_sample']:
        print("  Computing DINOv3 Hair Similarity...")
        dinov3_hair_metric = DINOv3HairSimilarityMetric(
            device=device, 
            model_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
            similarity_mode="patch",
            patch_aggregation="pool_then_compare",
            extraction_mode="cropped",
        )
        dinov3_hair_results = dinov3_hair_metric.compute_per_sample(samples)
        dinov3_hair_results = [r if r is not None else float('nan') for r in dinov3_hair_results]
        per_sample_metrics['dinov3_hair_similarity'] = dinov3_hair_results
    
    # DINOv2 Hair Similarity
    if 'dinov2_hair_similarity' in missing_metrics['per_sample']:
        print("  Computing DINOv2 Hair Similarity...")
        dinov2_hair_metric = DINOv2HairSimilarityMetric(
            device=device, 
            model_id="facebook/dinov2-large",
            similarity_mode="patch",
            patch_aggregation="pool_then_compare",
            extraction_mode="cropped",
        )
        dinov2_hair_results = dinov2_hair_metric.compute_per_sample(samples)
        dinov2_hair_results = [r if r is not None else float('nan') for r in dinov2_hair_results]
        per_sample_metrics['dinov2_hair_similarity'] = dinov2_hair_results
    
    # Pixio Hair Similarity
    if 'pixio_hair_similarity' in missing_metrics['per_sample']:
        print("  Computing Pixio Hair Similarity...")
        pixio_hair_metric = PixioHairSimilarityMetric(
            device=device, 
            model_id="facebook/pixio-vit1b16",
            similarity_mode="patch",
            patch_aggregation="pool_then_compare",
            extraction_mode="masked",
            use_normalized_features=False,
            feature_combination="patch_only",
        )
        pixio_hair_results = pixio_hair_metric.compute_per_sample(samples)
        pixio_hair_results = [r if r is not None else float('nan') for r in pixio_hair_results]
        per_sample_metrics['pixio_hair_similarity'] = pixio_hair_results
    # Hair Prompt Similarity (text embedding based) - only if prompt metrics enabled
    if enable_prompt_metrics and 'hair_prompt_similarity' in missing_metrics['per_sample']:
        print("  Computing Hair Prompt Similarity...")
        # Load hair prompts from prompt.json files
        prompts_ref = []  # Reference hair descriptions (from source hair)
        prompts_gen = []  # Generated hair descriptions (from transferred image)
        valid_indices = []  # Track which samples have valid prompts
        
        for idx, metadata in enumerate(sample_metadata):
            target_id = metadata['target_id']
            source_id = metadata['source_id']
            
            # Load reference hair prompt from source image's prompt.json
            # Note: reference hair comes from the target_id (hair donor)
            ref_prompt_path = os.path.join(BASE_DIR, "prompt", f"{target_id}.json")
            
            # Load generated hair prompt from the transferred image's folder
            transferred_path = method_paths["transferred"].format(target_id=target_id, source_id=source_id)
            gen_prompt_path = os.path.join(os.path.dirname(transferred_path), "prompt.json")
            
            ref_hair_desc = None
            gen_hair_desc = None
            
            # Try to load reference prompt
            if os.path.exists(ref_prompt_path):
                try:
                    with open(ref_prompt_path, 'r', encoding='utf-8') as f:
                        ref_data = json.load(f)
                    if 'subject' in ref_data and len(ref_data['subject']) > 0:
                        ref_hair_desc = ref_data['subject'][0].get('hair_description', '')
                except Exception as e:
                    tqdm.write(f"  Warning: Could not load reference prompt for {target_id}: {e}")
            
            # Try to load generated prompt
            if os.path.exists(gen_prompt_path):
                try:
                    with open(gen_prompt_path, 'r', encoding='utf-8') as f:
                        gen_data = json.load(f)
                    if 'subject' in gen_data and len(gen_data['subject']) > 0:
                        gen_hair_desc = gen_data['subject'][0].get('hair_description', '')
                except Exception as e:
                    tqdm.write(f"  Warning: Could not load generated prompt for {target_id}_to_{source_id}: {e}")
            
            # Only include if both prompts are available and non-empty
            if ref_hair_desc and gen_hair_desc:
                prompts_ref.append(ref_hair_desc)
                prompts_gen.append(gen_hair_desc)
                valid_indices.append(idx)
        
        print(f"  Found {len(valid_indices)}/{len(sample_metadata)} pairs with valid hair prompts")
        
        if len(valid_indices) > 0:
            # Compute similarities
            hair_prompt_metric = HairPromptSimilarityMetric(device=device, batch_size=batch_size)
            prompt_similarities = hair_prompt_metric.compute_per_sample(samples, prompts_ref, prompts_gen)
            
            # Map results back to all samples (NaN for missing)
            hair_prompt_results = [float('nan')] * len(sample_metadata)
            for i, valid_idx in enumerate(valid_indices):
                if prompt_similarities[i] is not None:
                    hair_prompt_results[valid_idx] = prompt_similarities[i]
            
            per_sample_metrics['hair_prompt_similarity'] = hair_prompt_results
        else:
            print("  Warning: No valid hair prompts found, skipping hair_prompt_similarity metric")
            per_sample_metrics['hair_prompt_similarity'] = [float('nan')] * len(sample_metadata)
    
    print("  All missing metrics computed!")
    
    # Build or merge results
    if existing_df is not None and samples is not None:
        # Merge new metrics with existing results
        print("\nMerging new metrics with existing results...")
        df_results = existing_df.copy()
        
        # Create a mapping from pair_name to index for efficient lookup
        pair_to_idx = {row['pair_name']: i for i, row in df_results.iterrows()}
        
        # Add new global metrics
        if 'clip_i' in missing_metrics['global']:
            df_results['clip_i'] = all_metrics['clip_i']
        if 'fid' in missing_metrics['global']:
            df_results['fid'] = all_metrics['fid']
        if 'fid_clip' in missing_metrics['global']:
            df_results['fid_clip'] = all_metrics['fid_clip']
        
        # Add new per-sample metrics
        for metric_name, values in per_sample_metrics.items():
            # Create a new column initialized with NaN
            if metric_name not in df_results.columns:
                df_results[metric_name] = float('nan')
            
            # Map computed values to existing rows
            for idx, metadata in enumerate(sample_metadata):
                pair_name = metadata['pair_name']
                if pair_name in pair_to_idx:
                    df_results.loc[pair_to_idx[pair_name], metric_name] = values[idx]
    else:
        # Build results from scratch
        per_pair_results = []
        for idx, metadata in enumerate(sample_metadata):
            pair_metrics = {
                'target_id': metadata['target_id'],
                'source_id': metadata['source_id'],
                'pair_name': metadata['pair_name'],
            }
            
            # Add global metrics
            if 'clip_i' in all_metrics:
                pair_metrics['clip_i'] = all_metrics['clip_i']
            if 'fid' in all_metrics:
                pair_metrics['fid'] = all_metrics['fid']
            if 'fid_clip' in all_metrics:
                pair_metrics['fid_clip'] = all_metrics['fid_clip']
            
            # Add per-sample metrics (only those that were actually computed)
            for metric_name, values in per_sample_metrics.items():
                pair_metrics[metric_name] = values[idx]
            
            per_pair_results.append(pair_metrics)
        
        df_results = pd.DataFrame(per_pair_results)
    
    # -----------------------------
    # Detect and Filter Failure Cases
    # -----------------------------
    print("\n" + "="*80)
    print("POST-PROCESSING: Failure Detection")
    print("="*80)
    
    # Detect failures (technical failures + optional outliers)
    # df_filtered, df_failures, failure_summary = _detect_failure_cases(
    #     df_results,
    #     method=method,
    #     max_failure_rate=0.10,  # Maximum 10% failure rate
    #     detect_outliers=False,   # Detect extreme outliers
    #     outlier_std_threshold=4.0,  # 4 standard deviations
    # )
    
    # # Save failure report
    # if len(df_failures) > 0:
    #     _save_failure_report(df_failures, failure_summary, method_output_dir, method)
    
    # Update df_results to use filtered data
    # df_results = df_filtered
    
    # Separate metadata from metrics
    metadata_cols = ['target_id', 'source_id', 'pair_name']
    metric_cols = [col for col in df_results.columns if col not in metadata_cols]
    global_metrics_list = GLOBAL_METRICS
    per_pair_metric_cols = [col for col in metric_cols if col not in global_metrics_list]
    
    # Save per-pair results
    per_pair_csv = os.path.join(method_output_dir, "per_pair_results.csv")
    df_results.to_csv(per_pair_csv, index=False)
    print(f"\nPer-pair results saved to: {per_pair_csv}")
    
    # Save failed pairs
    if failed_pairs:
        failed_csv = os.path.join(method_output_dir, "failed_pairs.csv")
        pd.DataFrame(failed_pairs).to_csv(failed_csv, index=False)
        print(f"Failed pairs logged to: {failed_csv}")
    
    # Compute aggregate statistics
    df_metrics = df_results[metric_cols]
    aggregate_results = {
        'mean': df_metrics.mean().to_dict(),
        'std': df_metrics.std().to_dict(),
        'median': df_metrics.median().to_dict(),
        'min': df_metrics.min().to_dict(),
        'max': df_metrics.max().to_dict(),
    }
    
    # Save aggregate results
    aggregate_json = os.path.join(method_output_dir, "aggregate_results.json")
    with open(aggregate_json, 'w') as f:
        json.dump(aggregate_results, f, indent=2)
    print(f"Aggregate results saved to: {aggregate_json}")
    
    # Save summary statistics
    print("\nGenerating summary statistics...")
    df_per_pair_metrics = df_results[per_pair_metric_cols] if per_pair_metric_cols else df_metrics
    save_summary_statistics(df_per_pair_metrics, method_output_dir, method)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    if per_pair_metric_cols:
        create_method_visualizations(df_per_pair_metrics, method_output_dir, method)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults summary for {method}:")
    
    # Print failure statistics if any
    print("\nGlobal Metrics:")
    for metric in global_metrics_list:
        if metric in df_results.columns:
            print(f"  {metric:30s}: {df_results[metric].iloc[0]:8.4f}")
    
    print("\nPer-Pair Metrics (mean ± std):")
    for metric in per_pair_metric_cols:
        mean_val = aggregate_results['mean'][metric]
        std_val = aggregate_results['std'][metric]
        print(f"  {metric:30s}: {mean_val:8.4f} ± {std_val:8.4f}")
    
    global_metrics_values = {
        'clip_i': all_metrics.get('clip_i') or (df_results['clip_i'].iloc[0] if 'clip_i' in df_results.columns else None),
        'fid': all_metrics.get('fid') or (df_results['fid'].iloc[0] if 'fid' in df_results.columns else None),
        'fid_clip': all_metrics.get('fid_clip') or (df_results['fid_clip'].iloc[0] if 'fid_clip' in df_results.columns else None),
    }
    
    return {
        'method': method,
        'n_pairs_evaluated': len(df_results),
        'n_pairs_failed': len(failed_pairs),
        'aggregate_results': aggregate_results,
        'global_metrics': global_metrics_values,
        'output_dir': method_output_dir,
        'per_pair_csv': per_pair_csv,
        'aggregate_json': aggregate_json,
        'per_pair_df': df_results,
    }


def _is_method_already_evaluated(output_dir: str, method: str) -> bool:
    """Check if a method has already been evaluated (per_pair_results.csv exists)."""
    output_dirs = _get_output_dirs(output_dir, method)
    method_output_dir = output_dirs['method']
    
    per_pair_csv = os.path.join(method_output_dir, "per_pair_results.csv")
    return os.path.exists(per_pair_csv)


def _load_previous_results(output_dir: str, method: str) -> Optional[Dict[str, Any]]:
    """Load previously computed results for a method."""
    output_dirs = _get_output_dirs(output_dir, method)
    method_output_dir = output_dirs['method']
    
    per_pair_csv = os.path.join(method_output_dir, "per_pair_results.csv")
    aggregate_json = os.path.join(method_output_dir, "aggregate_results.json")
    
    if not os.path.exists(per_pair_csv) or not os.path.exists(aggregate_json):
        return None
    
    try:
        df_results = pd.read_csv(per_pair_csv)
        
        with open(aggregate_json, 'r') as f:
            aggregate_results = json.load(f)
        
        global_metrics = {}
        if 'clip_i' in df_results.columns:
            global_metrics['clip_i'] = df_results['clip_i'].iloc[0]
        if 'fid' in df_results.columns:
            global_metrics['fid'] = df_results['fid'].iloc[0]
        if 'fid_clip' in df_results.columns:
            global_metrics['fid_clip'] = df_results['fid_clip'].iloc[0]
        
        print(f"  Loaded {len(df_results)} pairs from previous evaluation")
        
        return {
            'method': method,
            'per_pair_df': df_results,
            'aggregate_results': aggregate_results,
            'global_metrics': global_metrics,
            'output_dir': method_output_dir,
        }
        
    except Exception as e:
        print(f"  Error loading previous results: {e}")
        return None


# -----------------------------
# Available Methods Discovery
# -----------------------------

def get_all_available_methods(paths: Dict[str, Any] = None) -> List[str]:
    """
    Get list of all available method names from the paths configuration.
    
    Returns:
        List of method names (e.g., ['hairport_hi3dgen', 'hairfastgan', 'stablehair', ...])
    """
    if paths is None:
        paths = PATHS
    
    methods = []
    
    # Add top-level methods (for backward compatibility)
    for key, value in paths.items():
        if isinstance(value, dict) and 'base_dir' in value:
            methods.append(key)
    
    # Add baseline methods
    if 'baselines' in paths:
        methods.extend(paths['baselines'].keys())
    
    # Add hairport variant methods
    if 'hairport' in paths:
        methods.extend(paths['hairport'].keys())
    
    return sorted(methods)


def _resolve_methods(
    methods: Optional[List[str]] | str,
    paths: Dict[str, Any] = None,
) -> List[str]:
    """
    Resolve methods parameter to a list of method names.
    
    Args:
        methods: Either "all", a list of method names, or None (defaults to ['hairfastgan', 'hairport'])
        paths: Path configuration dictionary
        
    Returns:
        List of method names
    """
    if methods == "all":
        return get_all_available_methods(paths)
    elif methods is None:
        return ['hairfastgan', 'hairport']
    elif isinstance(methods, str):
        return [methods]
    else:
        return list(methods)


# -----------------------------
# Separate Evaluation and Comparison Functions
# -----------------------------

def evaluate_methods(
    output_dir: str = "/workspace/celeba_reduced/evaluation",
    methods: Optional[List[str]] | str = None,
    device: str = "cuda",
    batch_size: int = 8,
    force_recompute: bool = False,
    randomize_order: bool = True,
    enable_prompt_metrics: bool = False,
    to_include_angle_threshold: float = 0.1,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
    """
    Evaluate multiple hair transfer methods and compute metrics.
    
    This function only computes metrics for each method without generating
    comparison visualizations or LaTeX outputs. Use `generate_comparisons`
    to create those outputs from the evaluation results.
    
    Features:
    - Intelligent incremental evaluation: skips already-computed metrics
    - New metrics added to EXPECTED_METRICS will be computed for existing methods
    - Use force_recompute=True to recompute all metrics from scratch
    
    Pair Selection and Awareness Mode:
    ALL pairs from pairs.csv are included in evaluation. For HairPort methods,
    the awareness mode is determined by head_diff_angle:
    - head_diff_angle > to_include_angle_threshold: uses '3d_aware' path
    - head_diff_angle <= to_include_angle_threshold: uses '3d_unaware' path
    
    Args:
        output_dir: Base output directory for saving results
        methods: Method names to evaluate. Can be:
            - "all": Evaluate all available methods
            - List of method names: ['hairfastgan', 'hairport', ...]
            - None: Defaults to ['hairfastgan', 'hairport']
        device: Device for computation ('cuda' or 'cpu')
        batch_size: Batch size for metric computation
        force_recompute: If True, recompute all metrics even if results exist
        randomize_order: If True, shuffle methods to randomize evaluation order
        enable_prompt_metrics: If True, generate prompts and compute hair_prompt_similarity
            metric. Requires CaptionerPipeline. Default: False
        to_include_angle_threshold: Angle threshold (in radians) for HairPort awareness mode.
            Pairs with head_diff_angle > threshold use '3d_aware' paths,
            pairs with head_diff_angle <= threshold use '3d_unaware' paths.
            Default: 0.1 radians.
    
    Returns:
        Tuple of (results_dict, global_metrics_dict):
            - results_dict: Dict mapping method names to per-pair DataFrames
            - global_metrics_dict: Dict mapping method names to global metrics dicts
    """
    resolved_methods = _resolve_methods(methods)
    
    if randomize_order:
        resolved_methods = list(resolved_methods)
        random.shuffle(resolved_methods)
        print(f"Evaluating methods in randomized order: {resolved_methods}")
    else:
        print(f"Evaluating methods: {resolved_methods}")
    
    results_dict = {}
    global_metrics_dict = {}
    
    methods_pbar = tqdm(resolved_methods, desc="Evaluating methods", unit="method")
    for method in methods_pbar:
        methods_pbar.set_postfix_str(method)
        try:
            tqdm.write(f"\n{'#'*80}")
            tqdm.write(f"# Method: {method}")
            tqdm.write(f"{'#'*80}")
            
            results = evaluate_method(
                output_dir=output_dir,
                method=method,
                device=device,
                batch_size=batch_size,
                force_recompute=force_recompute,
                enable_prompt_metrics=enable_prompt_metrics,
                to_include_angle_threshold=to_include_angle_threshold,
            )
            
            results_dict[method] = results['per_pair_df']
            global_metrics_dict[method] = results['global_metrics']
            
        except Exception as e:
            tqdm.write(f"Error evaluating {method}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE: {len(results_dict)}/{len(resolved_methods)} methods succeeded")
    print(f"{'='*80}")
    
    return results_dict, global_metrics_dict


def generate_comparisons(
    output_dir: str = "/workspace/celeba_reduced/evaluation",
    methods: Optional[List[str]] | str = None,
    results_dict: Optional[Dict[str, pd.DataFrame]] = None,
    global_metrics_dict: Optional[Dict[str, Dict[str, float]]] = None,
    metrics: Optional[List[str]] = None,
    paper_title: str = "Quantitative Evaluation of Hair Transfer Methods",
    our_method: str = "hairport",
    min_angle_threshold: Optional[float] = None,
    data_dir: str = BASE_DIR,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
    """
    Generate comparison visualizations and LaTeX outputs from evaluation results.
    
    This function can either:
    1. Use provided results_dict and global_metrics_dict directly
    2. Load previously saved results from output_dir for specified methods
    
    Args:
        output_dir: Base output directory containing method results
        methods: Method names to include in comparison. Can be:
            - "all": Include all available methods with saved results
            - List of method names: ['hairfastgan', 'hairport', ...]
            - None: Defaults to ['hairfastgan', 'hairport']
            Ignored if results_dict is provided.
        results_dict: Pre-computed results (optional, will load from disk if not provided)
        global_metrics_dict: Pre-computed global metrics (optional)
        metrics: List of metric names to include in comparison outputs.
            - None: Include all computed metrics (default)
            - List of metric names: ['clip_i', 'fid', 'ssim_nonhair_intersection', ...]
        paper_title: Title for LaTeX report
        our_method: Method name to highlight as "ours" in tables
        min_angle_threshold: If specified, only include samples with head_diff_angle > this threshold
            in the comparison. Requires pairs.csv to be present in data_dir.
            If None, all samples are included (default).
        data_dir: Directory containing pairs.csv for angle filtering. Default: BASE_DIR.
    
    Returns:
        Tuple of (results_dict, global_metrics_dict) used for comparison
    """
    # Load results from disk if not provided
    if results_dict is None:
        resolved_methods = _resolve_methods(methods)
        print(f"Loading results for methods: {resolved_methods}")
        
        results_dict = {}
        global_metrics_dict = {} if global_metrics_dict is None else global_metrics_dict
        
        for method in resolved_methods:
            result = _load_previous_results(output_dir, method)
            if result is not None:
                results_dict[method] = result['per_pair_df']
                if method not in global_metrics_dict:
                    global_metrics_dict[method] = result['global_metrics']
                print(f"  ✓ Loaded {method}: {len(result['per_pair_df'])} pairs")
            else:
                print(f"  ✗ No results found for {method}")
    
    if len(results_dict) == 0:
        print("No results available for comparison. Run evaluate_methods first.")
        return {}, {}
    
    # Filter samples by head angle threshold if specified
    if min_angle_threshold is not None:
        print(f"\nFiltering samples by head_diff_angle > {min_angle_threshold} radians...")
        
        # Load pairs.csv to get head_diff_angle values
        csv_path = os.path.join(data_dir, "pairs.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"pairs.csv not found at: {csv_path}. "
                f"Cannot filter by angle threshold without pairs.csv."
            )
        
        pairs_df = pd.read_csv(csv_path)
        required_columns = ['source_id', 'target_id', 'head_diff_angle']
        missing_columns = [col for col in required_columns if col not in pairs_df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in pairs.csv: {missing_columns}. "
                f"Required columns for angle filtering: {required_columns}"
            )
        
        # Create a set of valid (target_id, source_id) pairs that meet the angle threshold
        valid_pairs = set()
        for _, row in pairs_df.iterrows():
            if row['head_diff_angle'] > min_angle_threshold:
                # Store as strings for consistent matching
                valid_pairs.add((str(row['target_id']), str(row['source_id'])))
        
        print(f"  Valid pairs with angle > {min_angle_threshold}: {len(valid_pairs)}/{len(pairs_df)}")
        
        # Filter each method's results
        filtered_results_dict = {}
        for method, df in results_dict.items():
            # Ensure target_id and source_id are strings for matching
            df = df.copy()
            df['target_id'] = df['target_id'].astype(str)
            df['source_id'] = df['source_id'].astype(str)
            
            # Filter to only include valid pairs
            mask = df.apply(
                lambda row: (row['target_id'], row['source_id']) in valid_pairs, 
                axis=1
            )
            filtered_df = df[mask].copy()
            
            n_before = len(df)
            n_after = len(filtered_df)
            print(f"  {method}: {n_after}/{n_before} samples retained")
            
            if n_after > 0:
                filtered_results_dict[method] = filtered_df
            else:
                print(f"    ⚠ Warning: No samples remaining for {method} after filtering")
        
        results_dict = filtered_results_dict
        
        if len(results_dict) == 0:
            print("\nNo samples remaining after angle filtering. Adjust threshold or check data.")
            return {}, {}
    
    # Filter metrics if specified
    if metrics is not None:
        print(f"\nFiltering to specified metrics: {metrics}")
        filtered_results_dict = {}
        filtered_global_metrics_dict = {}
        
        for method, df in results_dict.items():
            # Keep metadata columns + specified metrics
            cols_to_keep = METADATA_COLS + [m for m in metrics if m in df.columns]
            filtered_results_dict[method] = df[cols_to_keep].copy()
            
            # Filter global metrics
            if global_metrics_dict and method in global_metrics_dict:
                filtered_global_metrics_dict[method] = {
                    k: v for k, v in global_metrics_dict[method].items() 
                    if k in metrics
                }
        
        results_dict = filtered_results_dict
        global_metrics_dict = filtered_global_metrics_dict
        
        print(f"  Filtered results contain {len(metrics)} metrics")
    
    print(f"\nGenerating comparisons for {len(results_dict)} methods: {list(results_dict.keys())}")
    
    # Create comparison visualizations
    print(f"\n{'#'*80}")
    print(f"# Creating Comparison Visualizations")
    print(f"{'#'*80}")
    
    comparison_dir = os.path.join(output_dir, "comparison")
    create_comparison_visualizations(results_dict, global_metrics_dict, comparison_dir, our_method=our_method)
    print(f"\nComparison visualizations saved to: {comparison_dir}")
    
    # Generate LaTeX outputs
    print(f"\n{'#'*80}")
    print(f"# Generating LaTeX Outputs for Publication")
    print(f"{'#'*80}")
    
    generate_all_latex_outputs(
        results_dict=results_dict,
        global_metrics_dict=global_metrics_dict,
        output_dir=output_dir,
        paper_title=paper_title,
        our_method=our_method,
    )
    
    print(f"\n{'='*80}")
    print(f"COMPARISON GENERATION COMPLETE")
    print(f"{'='*80}")
    
    return results_dict, global_metrics_dict


def evaluate_and_compare(
    output_dir: str = "/workspace/celeba_reduced/evaluation",
    methods: Optional[List[str]] | str = None,
    device: str = "cuda",
    batch_size: int = 8,
    force_recompute: bool = False,
    randomize_order: bool = True,
    metrics: Optional[List[str]] = None,
    paper_title: str = "Quantitative Evaluation of Hair Transfer Methods",
    our_method: str = "hairport",
    enable_prompt_metrics: bool = False,
    to_include_angle_threshold: float = 0.1,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
    """
    Evaluate multiple methods and create comparison visualizations.
    
    This is a convenience function that combines `evaluate_methods` and
    `generate_comparisons` into a single call.
    
    Features:
    - Intelligent incremental evaluation: skips already-computed metrics
    - New metrics added to EXPECTED_METRICS will be computed for existing methods
    - Use force_recompute=True to recompute all metrics from scratch
    
    Pair Selection and Awareness Mode:
    ALL pairs from pairs.csv are included in evaluation. For HairPort methods,
    the awareness mode is determined by head_diff_angle:
    - head_diff_angle > to_include_angle_threshold: uses '3d_aware' path
    - head_diff_angle <= to_include_angle_threshold: uses '3d_unaware' path
    
    Output structure:
    - output_dir/
      - hairport/          # HairPort results
      - baselines/         # Baseline results
        - hairfastgan/
        - stablehair/
        - ...
      - comparison/        # Comparison visualizations
      - latex_report/      # LaTeX tables and reports
    
    Args:
        output_dir: Base output directory
        methods: Method names to evaluate. Can be:
            - "all": Evaluate all available methods
            - List of method names: ['hairfastgan', 'hairport', ...]
            - None: Defaults to ['hairfastgan', 'hairport']
        device: Device for computation
        batch_size: Batch size for metric computation
        force_recompute: If True, recompute all metrics even if results exist
        randomize_order: If True, shuffle methods to randomize evaluation order
        metrics: List of metric names to include in comparison outputs.
            - None: Include all computed metrics (default)
            - List of metric names: ['clip_i', 'fid', 'ssim_nonhair_intersection', ...]
        paper_title: Title for LaTeX report
        our_method: Method name to highlight as "ours" in tables
        enable_prompt_metrics: If True, generate prompts and compute hair_prompt_similarity
            metric. Requires CaptionerPipeline. Default: False
        to_include_angle_threshold: Angle threshold (in radians) for HairPort awareness mode.
            Pairs with head_diff_angle > threshold use '3d_aware' paths,
            pairs with head_diff_angle <= threshold use '3d_unaware' paths.
            Default: 0.1 radians.
    
    Returns:
        Tuple of (results_dict, global_metrics_dict)
    """
    # Step 1: Evaluate methods
    results_dict, global_metrics_dict = evaluate_methods(
        output_dir=output_dir,
        methods=methods,
        device=device,
        batch_size=batch_size,
        force_recompute=force_recompute,
        randomize_order=randomize_order,
        enable_prompt_metrics=enable_prompt_metrics,
        to_include_angle_threshold=to_include_angle_threshold,
    )
    
    # Step 2: Generate comparisons
    if len(results_dict) >= 1:
        generate_comparisons(
            output_dir=output_dir,
            results_dict=results_dict,
            global_metrics_dict=global_metrics_dict,
            metrics=metrics,
            paper_title=paper_title,
            our_method=our_method,
        )
    
    return results_dict, global_metrics_dict

if __name__ == "__main__":
    # results_dict, global_metrics_dict = evaluate_methods(
    #     output_dir="/workspace/outputs/evaluation",
    #     methods="all",
    #     device="cuda",
    #     batch_size=32,
    #     # force_recompute=True,
    #     to_include_angle_threshold=0.75,  # Only include pairs with head_diff_angle > 0.25 radians
    # )
    
    results_dict, global_metrics_dict = generate_comparisons(
        output_dir="/workspace/outputs/evaluation",
        methods="all",
        metrics=['ssim_nonhair_intersection', 'psnr_nonhair_intersection', 'fid', 'fid_clip', 'ids', 'dinov3_hair_similarity', 'dinov2_hair_similarity', 'lpips'],
        min_angle_threshold=0,  # Only include samples with head_diff_angle > 0.15 radians
        data_dir="/workspace/outputs",  # Directory containing pairs.csv
    )

    # # Default: Evaluate specific methods and generate comparisons
    # results_dict, global_metrics_dict = evaluate_and_compare(
    #     output_dir="/workspace/celeba_reduced/evaluation",
    #     methods=['hairfastgan', 'stablehair', 'hairclipv2', 'hairclip', 'hairfusion', 'hairport'],
    #     device="cuda",
    #     batch_size=16,
    #     to_include_angle_threshold=0.1,
    # )

