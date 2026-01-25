"""
Evaluation pipeline manager for bald conversion methods.

This module provides the evaluation pipeline for bald conversion task:
- Loading samples from different bald conversion methods
- Computing metrics (IDS, SSIM, PSNR, FID, FID_CLIP on non-hair regions)
- Separating evaluation for aligned vs unaligned image methods

Metrics computed:
- IDS: Identity Similarity (face identity preservation)
- SSIM: Structural Similarity on non-hair region
- PSNR: Peak Signal-to-Noise Ratio on non-hair region
- FID: Fréchet Inception Distance on non-hair region
- FID_CLIP: FID using CLIP encoder on non-hair region
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
    DreamSimMetric,
    IDSMetric,
    DINOv3HairSimilarityMetric,
    HairPromptSimilarityMetric,
    _to_rgb,
    _resize_like,
    _pil_to_torch_float01,
    _intersected_nonhair_weights,
    _apply_nonhair_mask_to_image,
)

from hairport.metrics.visualizers import (
    create_method_visualizations,
    save_summary_statistics,
)

# SAM3 for mask extraction
from utils.sam_mask_extractor import SAMMaskExtractor
SAM3_AVAILABLE = True


# -----------------------------
# Default Path Configuration
# -----------------------------

BASE_DIR = "/workspace/bald_dataset/"
OUTPUTS_DIR = "/workspace/outputs"
PATHS = {
    # Source images (original with hair)
    "image": f"{OUTPUTS_DIR}/image/{{sample_id}}.png",
    # Hair masks for source images
    "hair_mask": f"{OUTPUTS_DIR}/hair_mask/{{sample_id}}.png",
    # Aligned version of source images
    "aligned_image": f"{OUTPUTS_DIR}/aligned_image/{{sample_id}}.png",
    # Aligned hair masks
    "aligned_hair_mask": f"{OUTPUTS_DIR}/aligned_hair_mask/{{sample_id}}.png",
    'baselines': {
        'stablehair': {
            "file_path": f"{BASE_DIR}/{{sample_id}}/StableHair.png",
            "is_aligned": True,
        },
        'hairclipv2': {
            "file_path": f"{BASE_DIR}/{{sample_id}}/HairCLIP_v2.png",
            "is_aligned": True,
        },
        'hairmapper': {
            "file_path": f"{BASE_DIR}/{{sample_id}}/HairMapper.png",
            "is_aligned": True,
        },
        'qwen': {
            "file_path": f"{BASE_DIR}/{{sample_id}}/Qwen-Image-Edit-2509.png",
            "is_aligned": False,
        },
        'nano_banana': {
            "file_path": f"{OUTPUTS_DIR}/bald/nano_banana/image/{{sample_id}}.png",
            "is_aligned": False,
        },
    },
    'hairport': {
        "w_seg_aligned": {
            "file_path": f"{BASE_DIR}/{{sample_id}}/HairPort_w__seg.png",
            "is_aligned": True,
        },
        "wo_seg_aligned": {
            "file_path": f"{BASE_DIR}/{{sample_id}}/HairPort_w_o_seg.png",
            "is_aligned": True,
        },
        "w_seg": {
            "file_path": f"{OUTPUTS_DIR}/bald/w_seg/image/{{sample_id}}.png",
            "is_aligned": False,
        },
        "wo_seg": {
            "file_path": f"{OUTPUTS_DIR}/bald/wo_seg/image/{{sample_id}}.png",
            "is_aligned": False,
        },
    },
}


# -----------------------------
# Expected Metrics Configuration
# -----------------------------

EXPECTED_METRICS = {
    # Global metrics (computed once for entire dataset)
    'global': ['fid_nonhair', 'fid_clip_nonhair'],
    # Per-sample metrics (computed for each sample)
    'per_sample': [
        'ids',
        'ssim_nonhair',
        'psnr_nonhair',
    ],
}

# Metadata columns (not metrics)
METADATA_COLS = ['sample_id']


# -----------------------------
# Helper Functions
# -----------------------------

def _get_method_config(paths: Dict[str, Any], method: str) -> Dict[str, Any]:
    """
    Get the configuration for a specific method.
    Returns dict with 'file_path' and 'is_aligned' keys.
    """
    # Check in hairport
    if 'hairport' in paths and method in paths['hairport']:
        return paths['hairport'][method]
    
    # Check in baselines
    if 'baselines' in paths and method in paths['baselines']:
        return paths['baselines'][method]
    
    # List available methods for error message
    available_methods = []
    if 'hairport' in paths:
        available_methods.extend(paths['hairport'].keys())
    if 'baselines' in paths:
        available_methods.extend(paths['baselines'].keys())
    
    raise KeyError(
        f"Method '{method}' not found in paths configuration. "
        f"Available methods: {sorted(available_methods)}"
    )


def _discover_sample_ids(paths: Dict[str, Any]) -> List[str]:
    """Discover all valid sample IDs by listing the bald_dataset directories."""
    sample_ids = []
    
    # List directories in BASE_DIR that contain method outputs
    for item in os.listdir(BASE_DIR):
        item_path = os.path.join(BASE_DIR, item)
        if os.path.isdir(item_path) and item not in ['image', 'aligned_image', 'prompt']:
            # Check if this directory has any method output files
            has_outputs = any(
                os.path.exists(os.path.join(item_path, f))
                for f in os.listdir(item_path) if f.endswith('.png')
            )
            if has_outputs:
                sample_ids.append(item)
    
    return sorted(sample_ids)


def _get_all_methods(paths: Dict[str, Any] = None) -> Dict[str, List[str]]:
    """
    Get all available methods grouped by alignment type.
    
    Returns:
        Dictionary with 'aligned' and 'unaligned' lists of method names.
    """
    if paths is None:
        paths = PATHS
    
    aligned_methods = []
    unaligned_methods = []
    
    # Process hairport methods
    if 'hairport' in paths:
        for method_name, config in paths['hairport'].items():
            if config.get('is_aligned', False):
                aligned_methods.append(method_name)
            else:
                unaligned_methods.append(method_name)
    
    # Process baseline methods
    if 'baselines' in paths:
        for method_name, config in paths['baselines'].items():
            if config.get('is_aligned', False):
                aligned_methods.append(method_name)
            else:
                unaligned_methods.append(method_name)
    
    return {
        'aligned': sorted(aligned_methods),
        'unaligned': sorted(unaligned_methods),
    }


def _load_bald_sample(
    sample_id: str,
    method: str,
    paths: Dict[str, Any],
    image_size: int = 512,
) -> Sample:
    """
    Load a sample for bald evaluation.
    
    For bald task:
    - source: original image with hair
    - generated: bald result
    - reference: None (no reference for bald conversion)
    - hair_mask_source: hair mask of original image
    - hair_mask_generated: empty mask (bald head has no hair)
    """
    method_config = _get_method_config(paths, method)
    is_aligned = method_config.get('is_aligned', False)
    
    # Determine which source image and hair mask to use based on alignment
    if is_aligned:
        source_path = paths["aligned_image"].format(sample_id=sample_id)
        hair_mask_path = paths.get("aligned_hair_mask", paths["hair_mask"]).format(sample_id=sample_id)
    else:
        source_path = paths["image"].format(sample_id=sample_id)
        hair_mask_path = paths["hair_mask"].format(sample_id=sample_id)
    
    # Load source image
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source image not found: {source_path}")
    source_img = Image.open(source_path).convert("RGB").resize((image_size, image_size), Image.LANCZOS)
    
    # Load generated (bald) image
    generated_path = method_config["file_path"].format(sample_id=sample_id)
    if not os.path.exists(generated_path):
        raise FileNotFoundError(f"Generated image not found: {generated_path}")
    generated_img = Image.open(generated_path).convert("RGB").resize((image_size, image_size), Image.LANCZOS)
    
    # Load source hair mask
    if not os.path.exists(hair_mask_path):
        raise FileNotFoundError(f"Hair mask not found: {hair_mask_path}")
    source_hair_mask = Image.open(hair_mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)
    
    # For bald conversion, generated image should have no hair, so empty mask
    generated_hair_mask = Image.new("L", (image_size, image_size), 0)
    
    return Sample(
        source=source_img,
        generated=generated_img,
        reference=None,  # No reference for bald conversion
        hair_mask_source=source_hair_mask,
        hair_mask_generated=generated_hair_mask,
        hair_mask_reference=None,
    )


# -----------------------------
# Hair Mask Generation
# -----------------------------

def _ensure_hair_masks_exist(
    sample_ids: List[str],
    paths: Dict[str, Any],
    is_aligned: bool = False,
    device: str = "cuda",
) -> Tuple[int, int]:
    """Ensure hair masks exist for all samples. Generate missing masks using SAMMaskExtractor."""
    print(f"\nChecking for missing hair masks ({'aligned' if is_aligned else 'unaligned'})...")
    
    if is_aligned:
        image_key = "aligned_image"
        mask_key = "aligned_hair_mask"
    else:
        image_key = "image"
        mask_key = "hair_mask"
    
    missing_masks = []
    existing_count = 0
    
    for sample_id in sample_ids:
        mask_path = paths[mask_key].format(sample_id=sample_id)
        image_path = paths[image_key].format(sample_id=sample_id)
        
        if os.path.exists(mask_path):
            existing_count += 1
        elif os.path.exists(image_path):
            missing_masks.append({
                'sample_id': sample_id,
                'image_path': image_path,
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
            img = Image.open(mask_info['image_path']).convert("RGB")
            hair_mask = mask_extractor(img)[0]
            
            os.makedirs(os.path.dirname(mask_info['mask_path']), exist_ok=True)
            hair_mask.save(mask_info['mask_path'])
            
            generated_count += 1
            
        except Exception as e:
            tqdm.write(f"✗ Error for {mask_info['sample_id']}: {e}")
            failed_count += 1
    
    del mask_extractor
    if device == "cuda":
        torch.cuda.empty_cache()
    
    print(f"\nHair mask generation complete:")
    print(f"  Generated: {generated_count}")
    print(f"  Failed: {failed_count}")
    
    return existing_count, generated_count


# -----------------------------
# Output Directory Structure
# -----------------------------

def _get_output_dirs(base_output_dir: str, method: str, is_aligned: bool) -> Dict[str, str]:
    """Get output directory paths for a method."""
    alignment_suffix = "aligned" if is_aligned else "unaligned"
    method_dir = os.path.join(base_output_dir, alignment_suffix, method)
    
    return {
        'method': method_dir,
        'comparison': os.path.join(base_output_dir, alignment_suffix, "comparison"),
    }


def _get_missing_metrics(existing_df: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
    """Determine which metrics are missing from existing results."""
    if existing_df is None:
        return {
            'global': EXPECTED_METRICS['global'].copy(),
            'per_sample': EXPECTED_METRICS['per_sample'].copy(),
        }
    
    existing_cols = set(existing_df.columns)
    
    missing_global = [m for m in EXPECTED_METRICS['global'] if m not in existing_cols]
    missing_per_sample = [m for m in EXPECTED_METRICS['per_sample'] if m not in existing_cols]
    
    return {
        'global': missing_global,
        'per_sample': missing_per_sample,
    }


def _load_existing_results(output_dir: str, method: str, is_aligned: bool) -> Optional[pd.DataFrame]:
    """Load existing per-sample results for a method if they exist."""
    output_dirs = _get_output_dirs(output_dir, method, is_aligned)
    method_output_dir = output_dirs['method']
    per_sample_csv = os.path.join(method_output_dir, "per_sample_results.csv")
    
    if not os.path.exists(per_sample_csv):
        return None
    
    try:
        return pd.read_csv(per_sample_csv)
    except Exception as e:
        print(f"Warning: Could not load existing results: {e}")
        return None


# -----------------------------
# Non-Hair Region FID Computation
# -----------------------------

def _compute_nonhair_fid(
    samples: List[Sample],
    device: str = "cuda",
    batch_size: int = 16,
    use_clip: bool = False,
) -> float:
    """
    Compute FID on non-hair regions.
    
    For bald conversion, we compare:
    - Source images with hair masked out (non-hair region)
    - Generated bald images with original hair region masked out
    
    This measures how well the non-hair regions are preserved.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchvision import transforms
    
    if use_clip:
        from transformers import CLIPModel, CLIPProcessor
        # Use CLIP encoder
        model_name = "openai/clip-vit-large-patch14"
        clip_model = CLIPModel.from_pretrained(model_name).to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained(model_name)
        
        # Collect features
        source_features = []
        generated_features = []
        
        for sample in tqdm(samples, desc=f"Computing FID_CLIP features", leave=False):
            src = _to_rgb(sample.source)
            gen = _resize_like(_to_rgb(sample.generated), src)
            
            # Apply non-hair mask (use source hair mask for both)
            src_masked = _apply_nonhair_mask_to_image(src, sample.hair_mask_source, sample.hair_mask_source)
            gen_masked = _apply_nonhair_mask_to_image(gen, sample.hair_mask_source, sample.hair_mask_source)
            
            with torch.inference_mode():
                src_inputs = clip_processor(images=src_masked, return_tensors="pt").to(device)
                gen_inputs = clip_processor(images=gen_masked, return_tensors="pt").to(device)
                
                src_feat = clip_model.get_image_features(**src_inputs)
                gen_feat = clip_model.get_image_features(**gen_inputs)
                
                source_features.append(src_feat.cpu().numpy())
                generated_features.append(gen_feat.cpu().numpy())
        
        source_features = np.vstack(source_features)
        generated_features = np.vstack(generated_features)
        
        # Compute FID from features
        def compute_stats(features):
            mu = np.mean(features, axis=0)
            sigma = np.cov(features, rowvar=False)
            return mu, sigma
        
        def compute_fid_from_stats(mu1, sigma1, mu2, sigma2):
            from scipy import linalg
            diff = mu1 - mu2
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))
        
        mu1, sigma1 = compute_stats(source_features)
        mu2, sigma2 = compute_stats(generated_features)
        fid_value = compute_fid_from_stats(mu1, sigma1, mu2, sigma2)
        
        del clip_model
        torch.cuda.empty_cache()
        
        return fid_value
    else:
        # Use Inception V3
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])
        
        for sample in tqdm(samples, desc="Computing FID features", leave=False):
            src = _to_rgb(sample.source)
            gen = _resize_like(_to_rgb(sample.generated), src)
            
            # Apply non-hair mask
            src_masked = _apply_nonhair_mask_to_image(src, sample.hair_mask_source, sample.hair_mask_source)
            gen_masked = _apply_nonhair_mask_to_image(gen, sample.hair_mask_source, sample.hair_mask_source)
            
            src_tensor = transform(src_masked).unsqueeze(0).to(device)
            gen_tensor = transform(gen_masked).unsqueeze(0).to(device)
            
            fid.update(src_tensor, real=True)
            fid.update(gen_tensor, real=False)
        
        fid_value = float(fid.compute().cpu().item())
        
        del fid
        torch.cuda.empty_cache()
        
        return fid_value


# -----------------------------
# Main Evaluation Function
# -----------------------------

def evaluate_method(
    output_dir: str,
    method: str,
    paths: Dict[str, Any] = None,
    device: str = "cuda",
    batch_size: int = 16,
    force_recompute: bool = False,
    image_size: int = 512,
) -> Dict[str, Any]:
    """
    Evaluate a bald conversion method on all available samples.
    
    Args:
        output_dir: Base evaluation directory
        method: Method name to evaluate
        paths: Path configuration dictionary (default: PATHS)
        device: Device for computation
        batch_size: Batch size for metric computation
        force_recompute: If True, recompute all metrics even if they exist
        image_size: Size to resize images to
    
    Returns:
        Dictionary containing aggregated results and paths to saved files
    """
    if paths is None:
        paths = PATHS
    
    # Get method configuration
    method_config = _get_method_config(paths, method)
    is_aligned = method_config.get('is_aligned', False)
    alignment_type = "aligned" if is_aligned else "unaligned"
    
    print("=" * 80)
    print(f"EVALUATING BALD METHOD: {method} ({alignment_type})")
    print("=" * 80)
    
    # Get output directories
    output_dirs = _get_output_dirs(output_dir, method, is_aligned)
    method_output_dir = output_dirs['method']
    os.makedirs(method_output_dir, exist_ok=True)
    print(f"Output directory: {method_output_dir}")
    
    # Check for existing results
    existing_df = None if force_recompute else _load_existing_results(output_dir, method, is_aligned)
    missing_metrics = _get_missing_metrics(existing_df)
    
    all_missing = missing_metrics['global'] + missing_metrics['per_sample']
    
    if existing_df is not None and len(all_missing) == 0:
        print("\n✓ All metrics already computed. Nothing to do.")
        return _build_results_dict(existing_df, method, method_output_dir)
    
    if existing_df is not None:
        print(f"\n→ Found existing results with {len(existing_df)} samples")
        print(f"  Missing global metrics: {missing_metrics['global'] or 'None'}")
        print(f"  Missing per-sample metrics: {missing_metrics['per_sample'] or 'None'}")
    else:
        print("\n→ No existing results found. Computing all metrics.")
    
    # Discover all sample IDs
    sample_ids = _discover_sample_ids(paths)
    print(f"\nFound {len(sample_ids)} sample IDs")
    
    if len(sample_ids) == 0:
        raise ValueError("No valid samples found")
    
    # Ensure hair masks exist
    _ensure_hair_masks_exist(sample_ids, paths, is_aligned=is_aligned, device=device)
    
    # Initialize device
    device = device if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Load samples
    print("\nLoading samples...")
    samples = []
    sample_metadata = []
    failed_samples = []
    
    for sample_id in tqdm(sample_ids, desc="Loading samples", unit="sample"):
        try:
            sample = _load_bald_sample(sample_id, method, paths, image_size)
            samples.append(sample)
            sample_metadata.append({'sample_id': sample_id})
        except Exception as e:
            tqdm.write(f"✗ Error loading {sample_id}: {e}")
            failed_samples.append({'sample_id': sample_id, 'error': str(e)})
    
    if len(samples) == 0:
        raise RuntimeError("No samples were successfully loaded")
    
    print(f"\nSuccessfully loaded: {len(samples)}/{len(sample_ids)} samples")
    
    # Compute metrics
    print("\nComputing metrics...")
    all_metrics = {}
    per_sample_metrics = {}
    
    # --- Global metrics ---
    if 'fid_nonhair' in missing_metrics['global']:
        print("  Computing FID (non-hair region)...")
        all_metrics['fid_nonhair'] = _compute_nonhair_fid(
            samples, device=device, batch_size=batch_size, use_clip=False
        )
        print(f"    FID (non-hair): {all_metrics['fid_nonhair']:.4f}")
    
    if 'fid_clip_nonhair' in missing_metrics['global']:
        print("  Computing FID_CLIP (non-hair region)...")
        all_metrics['fid_clip_nonhair'] = _compute_nonhair_fid(
            samples, device=device, batch_size=batch_size, use_clip=True
        )
        print(f"    FID_CLIP (non-hair): {all_metrics['fid_clip_nonhair']:.4f}")
    
    # --- Per-sample metrics ---
    
    # IDS (Identity Similarity)
    if 'ids' in missing_metrics['per_sample']:
        print("  Computing IDS...")
        ids_metric = IDSMetric(device=device)
        ids_results = ids_metric.compute_per_sample(samples)
        ids_results = [r if r is not None else float('nan') for r in ids_results]
        per_sample_metrics['ids'] = ids_results
    
    # SSIM (non-hair region)
    if 'ssim_nonhair' in missing_metrics['per_sample']:
        print("  Computing SSIM (non-hair region)...")
        ssim_metric = SSIMMetric()
        ssim_results = []
        for sample in tqdm(samples, desc="Computing SSIM", unit="sample", leave=False):
            # SSIMMetric already handles non-hair masking internally
            ssim_val = ssim_metric.compute([sample])
            ssim_results.append(ssim_val)
        per_sample_metrics['ssim_nonhair'] = ssim_results
    
    # PSNR (non-hair region)
    if 'psnr_nonhair' in missing_metrics['per_sample']:
        print("  Computing PSNR (non-hair region)...")
        psnr_metric = PSNRMetric()
        psnr_results = []
        for sample in tqdm(samples, desc="Computing PSNR", unit="sample", leave=False):
            # PSNRMetric already handles non-hair masking internally
            psnr_val = psnr_metric.compute([sample])
            psnr_results.append(psnr_val)
        per_sample_metrics['psnr_nonhair'] = psnr_results
    
    print("  All metrics computed!")
    
    # Build results DataFrame
    results_data = []
    for idx, metadata in enumerate(sample_metadata):
        row = {'sample_id': metadata['sample_id']}
        
        # Add global metrics (same for all samples)
        for metric_name, value in all_metrics.items():
            row[metric_name] = value
        
        # Add per-sample metrics
        for metric_name, values in per_sample_metrics.items():
            row[metric_name] = values[idx]
        
        results_data.append(row)
    
    df_results = pd.DataFrame(results_data)
    
    # Save results
    per_sample_csv = os.path.join(method_output_dir, "per_sample_results.csv")
    df_results.to_csv(per_sample_csv, index=False)
    print(f"\nPer-sample results saved to: {per_sample_csv}")
    
    # Compute aggregate statistics
    metric_cols = [col for col in df_results.columns if col not in METADATA_COLS]
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
    
    # Save failed samples if any
    if failed_samples:
        failed_json = os.path.join(method_output_dir, "failed_samples.json")
        with open(failed_json, 'w') as f:
            json.dump(failed_samples, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults summary for {method} ({alignment_type}):")
    
    print("\nGlobal Metrics:")
    for metric in EXPECTED_METRICS['global']:
        if metric in all_metrics:
            print(f"  {metric}: {all_metrics[metric]:.4f}")
    
    print("\nPer-Sample Metrics (mean ± std):")
    for metric in EXPECTED_METRICS['per_sample']:
        if metric in per_sample_metrics:
            mean_val = np.nanmean(per_sample_metrics[metric])
            std_val = np.nanstd(per_sample_metrics[metric])
            print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    return {
        'method': method,
        'alignment_type': alignment_type,
        'n_samples_evaluated': len(df_results),
        'n_samples_failed': len(failed_samples),
        'aggregate_results': aggregate_results,
        'global_metrics': all_metrics,
        'output_dir': method_output_dir,
        'per_sample_csv': per_sample_csv,
        'aggregate_json': aggregate_json,
        'per_sample_df': df_results,
    }


def _build_results_dict(df: pd.DataFrame, method: str, output_dir: str) -> Dict[str, Any]:
    """Build results dictionary from existing DataFrame."""
    metric_cols = [col for col in df.columns if col not in METADATA_COLS]
    df_metrics = df[metric_cols]
    
    aggregate_results = {
        'mean': df_metrics.mean().to_dict(),
        'std': df_metrics.std().to_dict(),
        'median': df_metrics.median().to_dict(),
        'min': df_metrics.min().to_dict(),
        'max': df_metrics.max().to_dict(),
    }
    
    global_metrics = {}
    for metric in EXPECTED_METRICS['global']:
        if metric in df.columns:
            global_metrics[metric] = df[metric].iloc[0]
    
    return {
        'method': method,
        'n_samples_evaluated': len(df),
        'n_samples_failed': 0,
        'aggregate_results': aggregate_results,
        'global_metrics': global_metrics,
        'output_dir': output_dir,
        'per_sample_csv': os.path.join(output_dir, "per_sample_results.csv"),
        'aggregate_json': os.path.join(output_dir, "aggregate_results.json"),
        'per_sample_df': df,
    }


# -----------------------------
# Batch Evaluation Functions
# -----------------------------

def evaluate_all_methods(
    output_dir: str = "/workspace/bald_dataset/evaluation",
    methods: Optional[List[str]] = None,
    alignment_filter: Optional[str] = None,
    paths: Dict[str, Any] = None,
    device: str = "cuda",
    batch_size: int = 16,
    force_recompute: bool = False,
    randomize_order: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate multiple bald conversion methods.
    
    Args:
        output_dir: Base output directory for saving results
        methods: List of method names to evaluate, or None for all
        alignment_filter: Filter by alignment type ('aligned', 'unaligned', or None for both)
        paths: Path configuration dictionary
        device: Device for computation
        batch_size: Batch size for metric computation
        force_recompute: If True, recompute all metrics
        randomize_order: If True, shuffle methods to randomize evaluation order
    
    Returns:
        Dictionary mapping method names to their evaluation results
    """
    if paths is None:
        paths = PATHS
    
    # Get all methods grouped by alignment
    all_methods = _get_all_methods(paths)
    
    # Filter methods
    if methods is None:
        if alignment_filter == 'aligned':
            methods_to_evaluate = all_methods['aligned']
        elif alignment_filter == 'unaligned':
            methods_to_evaluate = all_methods['unaligned']
        else:
            methods_to_evaluate = all_methods['aligned'] + all_methods['unaligned']
    else:
        methods_to_evaluate = methods
    
    if randomize_order:
        random.shuffle(methods_to_evaluate)
    
    print(f"\nMethods to evaluate: {methods_to_evaluate}")
    
    results = {}
    
    for method in tqdm(methods_to_evaluate, desc="Evaluating methods", unit="method"):
        try:
            result = evaluate_method(
                output_dir=output_dir,
                method=method,
                paths=paths,
                device=device,
                batch_size=batch_size,
                force_recompute=force_recompute,
            )
            results[method] = result
        except Exception as e:
            print(f"\n✗ Error evaluating {method}: {e}")
            results[method] = {'error': str(e)}
    
    return results


def generate_comparison_table(
    output_dir: str = "/workspace/bald_dataset/evaluation",
    paths: Dict[str, Any] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate comparison tables for aligned and unaligned methods.
    
    Returns:
        Tuple of (aligned_df, unaligned_df) with aggregated metrics per method
    """
    if paths is None:
        paths = PATHS
    
    all_methods = _get_all_methods(paths)
    
    def load_results_for_methods(methods: List[str], is_aligned: bool) -> pd.DataFrame:
        rows = []
        for method in methods:
            output_dirs = _get_output_dirs(output_dir, method, is_aligned)
            aggregate_json = os.path.join(output_dirs['method'], "aggregate_results.json")
            
            if os.path.exists(aggregate_json):
                with open(aggregate_json, 'r') as f:
                    agg = json.load(f)
                
                row = {'method': method}
                for metric in EXPECTED_METRICS['global'] + EXPECTED_METRICS['per_sample']:
                    if metric in agg['mean']:
                        row[f'{metric}_mean'] = agg['mean'][metric]
                        row[f'{metric}_std'] = agg['std'].get(metric, 0)
                rows.append(row)
        
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    aligned_df = load_results_for_methods(all_methods['aligned'], is_aligned=True)
    unaligned_df = load_results_for_methods(all_methods['unaligned'], is_aligned=False)
    
    # Save comparison tables
    if not aligned_df.empty:
        aligned_csv = os.path.join(output_dir, "aligned", "comparison_table.csv")
        os.makedirs(os.path.dirname(aligned_csv), exist_ok=True)
        aligned_df.to_csv(aligned_csv, index=False)
        print(f"Aligned comparison table saved to: {aligned_csv}")
    
    if not unaligned_df.empty:
        unaligned_csv = os.path.join(output_dir, "unaligned", "comparison_table.csv")
        os.makedirs(os.path.dirname(unaligned_csv), exist_ok=True)
        unaligned_df.to_csv(unaligned_csv, index=False)
        print(f"Unaligned comparison table saved to: {unaligned_csv}")
    
    return aligned_df, unaligned_df


# -----------------------------
# Main Entry Point
# -----------------------------

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


# -----------------------------
# Method Display Names
# -----------------------------

METHOD_DISPLAY_NAMES = {
    # Baselines
    'stablehair': 'Stable-Hair',
    'hairclipv2': 'HairCLIPv2',
    'hairmapper': 'HairMapper',
    'qwen': 'Qwen-Edit',
    'nano_banana': 'Nano Banana 🍌',
    # HairPort variants
    'w_seg_aligned': 'HairPort (w/ seg)',
    'wo_seg_aligned': 'HairPort (w/o seg)',
    'w_seg': 'HairPort (w/ seg)',
    'wo_seg': 'HairPort (w/o seg)',
}

METRIC_DISPLAY_NAMES = {
    'ids': 'IDS ↑',
    'ssim_nonhair': 'SSIM ↑',
    'psnr_nonhair': 'PSNR ↑',
    'fid_nonhair': 'FID ↓',
    'fid_clip_nonhair': 'FID-CLIP ↓',
}

# Metrics where higher is better
HIGHER_IS_BETTER = {'ids', 'ssim_nonhair', 'psnr_nonhair'}

# Color palettes
HAIRPORT_COLOR = '#2E86AB'  # Blue for our method
BASELINE_COLORS = ['#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#95190C']


def get_method_display_name(method: str) -> str:
    """Get display name for a method."""
    return METHOD_DISPLAY_NAMES.get(method, method)


def get_metric_display_name(metric: str) -> str:
    """Get display name for a metric."""
    return METRIC_DISPLAY_NAMES.get(metric, metric)


# -----------------------------
# Visualization Functions
# -----------------------------

def create_bar_comparison_chart(
    results_df: pd.DataFrame,
    metrics: List[str],
    output_path: str,
    title: str = "Bald Conversion Methods Comparison",
    our_methods: List[str] = None,
    figsize: Tuple[float, float] = None,
) -> None:
    """
    Create a grouped bar chart comparing methods across multiple metrics.
    
    Args:
        results_df: DataFrame with 'method' column and metric columns
        metrics: List of metric names to plot
        output_path: Path to save the figure
        title: Figure title
        our_methods: List of method names that are "ours" (highlighted differently)
        figsize: Figure size (width, height)
    """
    if our_methods is None:
        our_methods = ['w_seg_aligned', 'wo_seg_aligned', 'w_seg', 'wo_seg']
    
    n_metrics = len(metrics)
    n_methods = len(results_df)
    
    if figsize is None:
        figsize = (max(8, n_methods * 1.2), 4 * ((n_metrics + 1) // 2))
    
    # Create subplots
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    # Color assignment
    colors = []
    for method in results_df['method']:
        if method in our_methods:
            colors.append(HAIRPORT_COLOR)
        else:
            idx = len([m for m in results_df['method'][:list(results_df['method']).index(method)] 
                      if m not in our_methods])
            colors.append(BASELINE_COLORS[idx % len(BASELINE_COLORS)])
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        
        if mean_col not in results_df.columns:
            continue
        
        x = np.arange(n_methods)
        means = results_df[mean_col].values
        stds = results_df[std_col].values if std_col in results_df.columns else np.zeros(n_methods)
        
        # Create bars
        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors, 
                      edgecolor='black', linewidth=0.5, alpha=0.85)
        
        # Highlight best method
        if metric in HIGHER_IS_BETTER:
            best_idx = np.argmax(means)
        else:
            best_idx = np.argmin(means)
        bars[best_idx].set_edgecolor('#FFD700')
        bars[best_idx].set_linewidth(2.5)
        
        # Labels and formatting
        ax.set_ylabel(get_metric_display_name(metric))
        ax.set_xticks(x)
        ax.set_xticklabels([get_method_display_name(m) for m in results_df['method']], 
                          rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            if metric in ['fid_nonhair', 'fid_clip_nonhair']:
                label = f'{mean:.1f}'
            else:
                label = f'{mean:.3f}'
            ax.annotate(label,
                       xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.01 * max(means)),
                       ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=HAIRPORT_COLOR, edgecolor='black', label='Ours'),
        mpatches.Patch(facecolor=BASELINE_COLORS[0], edgecolor='black', label='Baselines'),
        mpatches.Patch(facecolor='white', edgecolor='#FFD700', linewidth=2.5, label='Best'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Bar comparison chart saved to: {output_path}")


def create_radar_chart(
    results_df: pd.DataFrame,
    metrics: List[str],
    output_path: str,
    title: str = "Multi-Metric Comparison",
    our_methods: List[str] = None,
    normalize: bool = True,
) -> None:
    """
    Create a radar/spider chart comparing methods across multiple metrics.
    
    Args:
        results_df: DataFrame with 'method' column and metric columns
        metrics: List of metric names to plot
        output_path: Path to save the figure
        title: Figure title
        our_methods: List of method names that are "ours"
        normalize: Whether to normalize metrics to [0, 1] range
    """
    if our_methods is None:
        our_methods = ['w_seg_aligned', 'wo_seg_aligned', 'w_seg', 'wo_seg']
    
    n_metrics = len(metrics)
    n_methods = len(results_df)
    
    # Prepare data
    values_dict = {}
    for _, row in results_df.iterrows():
        method = row['method']
        values = []
        for metric in metrics:
            mean_col = f'{metric}_mean'
            if mean_col in row:
                values.append(row[mean_col])
            else:
                values.append(0)
        values_dict[method] = values
    
    # Normalize values
    if normalize:
        for i, metric in enumerate(metrics):
            col_values = [values_dict[m][i] for m in values_dict]
            min_val, max_val = min(col_values), max(col_values)
            range_val = max_val - min_val if max_val != min_val else 1
            
            for method in values_dict:
                normalized = (values_dict[method][i] - min_val) / range_val
                # Invert for metrics where lower is better
                if metric not in HIGHER_IS_BETTER:
                    normalized = 1 - normalized
                values_dict[method][i] = normalized
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot each method
    color_idx = 0
    for method, values in values_dict.items():
        values_plot = values + values[:1]  # Complete the loop
        
        if method in our_methods:
            color = HAIRPORT_COLOR
            linewidth = 2.5
            alpha = 0.3
            linestyle = '-'
        else:
            color = BASELINE_COLORS[color_idx % len(BASELINE_COLORS)]
            color_idx += 1
            linewidth = 1.5
            alpha = 0.1
            linestyle = '--'
        
        ax.plot(angles, values_plot, 'o-', linewidth=linewidth, 
               label=get_method_display_name(method), color=color, linestyle=linestyle)
        ax.fill(angles, values_plot, alpha=alpha, color=color)
    
    # Set labels
    metric_labels = [get_metric_display_name(m).replace(' ↑', '').replace(' ↓', '') 
                     for m in metrics]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, size=10)
    
    # Set y-axis
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title(title, size=14, fontweight='bold', y=1.08)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Radar chart saved to: {output_path}")


def create_box_plot_comparison(
    output_dir: str,
    methods: List[str],
    is_aligned: bool,
    metrics: List[str] = None,
    output_path: str = None,
    title: str = None,
    our_methods: List[str] = None,
    paths: Dict[str, Any] = None,
) -> None:
    """
    Create box plots showing the distribution of per-sample metrics.
    
    Args:
        output_dir: Base evaluation directory
        methods: List of method names
        is_aligned: Whether methods are aligned
        metrics: List of metrics to plot (defaults to per-sample metrics)
        output_path: Path to save figure
        title: Figure title
        our_methods: List of method names that are "ours"
        paths: Path configuration
    """
    if paths is None:
        paths = PATHS
    if metrics is None:
        metrics = EXPECTED_METRICS['per_sample']
    if our_methods is None:
        our_methods = ['w_seg_aligned', 'wo_seg_aligned', 'w_seg', 'wo_seg']
    if title is None:
        alignment_str = "Aligned" if is_aligned else "Unaligned"
        title = f"Per-Sample Metric Distributions ({alignment_str} Methods)"
    
    # Load per-sample data for each method
    all_data = []
    for method in methods:
        method_dirs = _get_output_dirs(output_dir, method, is_aligned)
        csv_path = os.path.join(method_dirs['method'], "per_sample_results.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for metric in metrics:
                if metric in df.columns:
                    for val in df[metric].dropna():
                        all_data.append({
                            'method': get_method_display_name(method),
                            'metric': get_metric_display_name(metric),
                            'value': val,
                            'is_ours': method in our_methods,
                        })
    
    if not all_data:
        print("No data available for box plots")
        return
    
    plot_df = pd.DataFrame(all_data)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        metric_display = get_metric_display_name(metric)
        metric_data = plot_df[plot_df['metric'] == metric_display]
        
        if metric_data.empty:
            continue
        
        # Create color palette
        method_order = metric_data.groupby('method')['value'].mean().sort_values(
            ascending=(metric not in HIGHER_IS_BETTER)
        ).index.tolist()
        
        palette = {}
        baseline_idx = 0
        for method in method_order:
            # Check if this is our method by looking up original name
            original_name = [k for k, v in METHOD_DISPLAY_NAMES.items() if v == method]
            if original_name and original_name[0] in our_methods:
                palette[method] = HAIRPORT_COLOR
            else:
                palette[method] = BASELINE_COLORS[baseline_idx % len(BASELINE_COLORS)]
                baseline_idx += 1
        
        sns.boxplot(
            data=metric_data,
            x='method',
            y='value',
            order=method_order,
            palette=palette,
            ax=ax,
            width=0.6,
            linewidth=1,
            fliersize=3,
        )
        
        ax.set_xlabel('')
        ax.set_ylabel(metric_display)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path is None:
        alignment_str = "aligned" if is_aligned else "unaligned"
        output_path = os.path.join(output_dir, alignment_str, "box_plot_comparison.png")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Box plot comparison saved to: {output_path}")


def create_latex_table(
    results_df: pd.DataFrame,
    metrics: List[str],
    output_path: str,
    caption: str = "Quantitative comparison of bald conversion methods.",
    label: str = "tab:bald_comparison",
    our_methods: List[str] = None,
    highlight_best: bool = True,
) -> str:
    """
    Generate a LaTeX table for the results.
    
    Args:
        results_df: DataFrame with results
        metrics: List of metrics to include
        output_path: Path to save the .tex file
        caption: Table caption
        label: Table label for referencing
        our_methods: Methods to mark as "Ours"
        highlight_best: Whether to bold the best result
    
    Returns:
        LaTeX table string
    """
    if our_methods is None:
        our_methods = ['w_seg_aligned', 'wo_seg_aligned', 'w_seg', 'wo_seg']
    
    # Build header
    metric_headers = [get_metric_display_name(m).replace('↑', '$\\uparrow$').replace('↓', '$\\downarrow$') 
                      for m in metrics]
    header = "Method & " + " & ".join(metric_headers) + " \\\\"
    
    # Find best values for each metric
    best_values = {}
    for metric in metrics:
        mean_col = f'{metric}_mean'
        if mean_col in results_df.columns:
            if metric in HIGHER_IS_BETTER:
                best_values[metric] = results_df[mean_col].max()
            else:
                best_values[metric] = results_df[mean_col].min()
    
    # Build rows
    rows = []
    for _, row in results_df.iterrows():
        method = row['method']
        display_name = get_method_display_name(method)
        
        # Mark our methods
        if method in our_methods:
            display_name = f"\\textbf{{{display_name}}} (Ours)"
        
        cells = [display_name]
        
        for metric in metrics:
            mean_col = f'{metric}_mean'
            std_col = f'{metric}_std'
            
            if mean_col in row and not pd.isna(row[mean_col]):
                mean_val = row[mean_col]
                std_val = row[std_col] if std_col in row and not pd.isna(row[std_col]) else 0
                
                # Format based on metric type
                if metric in ['fid_nonhair', 'fid_clip_nonhair']:
                    cell = f"{mean_val:.2f}"
                    if std_val > 0:
                        cell += f" $\\pm$ {std_val:.2f}"
                else:
                    cell = f"{mean_val:.4f}"
                    if std_val > 0:
                        cell += f" $\\pm$ {std_val:.4f}"
                
                # Highlight best
                if highlight_best and metric in best_values:
                    if abs(mean_val - best_values[metric]) < 1e-6:
                        cell = f"\\textbf{{{cell}}}"
            else:
                cell = "-"
            
            cells.append(cell)
        
        rows.append(" & ".join(cells) + " \\\\")
    
    # Assemble table
    n_cols = len(metrics) + 1
    col_spec = "l" + "c" * len(metrics)
    
    latex = f"""\\begin{{table}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{{col_spec}}}
\\toprule
{header}
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}"""
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"LaTeX table saved to: {output_path}")
    return latex


def create_combined_comparison_figure(
    output_dir: str = "/workspace/bald_dataset/evaluation",
    paths: Dict[str, Any] = None,
) -> None:
    """
    Create a comprehensive comparison figure with multiple panels.
    
    Generates:
    1. Bar charts for aligned methods
    2. Bar charts for unaligned methods
    3. Radar charts
    4. Box plots
    5. LaTeX tables
    """
    if paths is None:
        paths = PATHS
    
    all_methods = _get_all_methods(paths)
    all_metrics = EXPECTED_METRICS['global'] + EXPECTED_METRICS['per_sample']
    
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    # Generate aligned method comparisons
    if all_methods['aligned']:
        aligned_df, _ = generate_comparison_table(output_dir, paths)
        
        if not aligned_df.empty:
            # Bar chart
            create_bar_comparison_chart(
                aligned_df,
                all_metrics,
                os.path.join(output_dir, "aligned", "bar_comparison.png"),
                title="Bald Conversion: Aligned Methods Comparison",
            )
            
            # Radar chart
            create_radar_chart(
                aligned_df,
                all_metrics,
                os.path.join(output_dir, "aligned", "radar_comparison.png"),
                title="Aligned Methods: Multi-Metric View",
            )
            
            # Box plot
            create_box_plot_comparison(
                output_dir,
                all_methods['aligned'],
                is_aligned=True,
                output_path=os.path.join(output_dir, "aligned", "box_comparison.png"),
            )
            
            # LaTeX table
            create_latex_table(
                aligned_df,
                all_metrics,
                os.path.join(output_dir, "aligned", "comparison_table.tex"),
                caption="Quantitative comparison of bald conversion methods (aligned images).",
                label="tab:bald_aligned",
            )
    
    # Generate unaligned method comparisons
    if all_methods['unaligned']:
        _, unaligned_df = generate_comparison_table(output_dir, paths)
        
        if not unaligned_df.empty:
            # Bar chart
            create_bar_comparison_chart(
                unaligned_df,
                all_metrics,
                os.path.join(output_dir, "unaligned", "bar_comparison.png"),
                title="Bald Conversion: Unaligned Methods Comparison",
            )
            
            # Radar chart
            create_radar_chart(
                unaligned_df,
                all_metrics,
                os.path.join(output_dir, "unaligned", "radar_comparison.png"),
                title="Unaligned Methods: Multi-Metric View",
            )
            
            # Box plot
            create_box_plot_comparison(
                output_dir,
                all_methods['unaligned'],
                is_aligned=False,
                output_path=os.path.join(output_dir, "unaligned", "box_comparison.png"),
            )
            
            # LaTeX table
            create_latex_table(
                unaligned_df,
                all_metrics,
                os.path.join(output_dir, "unaligned", "comparison_table.tex"),
                caption="Quantitative comparison of bald conversion methods (unaligned images).",
                label="tab:bald_unaligned",
            )
    
    print("\n✓ All visualizations generated successfully!")


def create_qualitative_comparison_grid(
    sample_ids: List[str],
    methods: List[str],
    output_path: str,
    paths: Dict[str, Any] = None,
    n_samples: int = 5,
    figsize: Tuple[float, float] = None,
    title: str = "Qualitative Comparison",
) -> None:
    """
    Create a grid of qualitative comparisons showing input and outputs.
    
    Args:
        sample_ids: List of sample IDs to show
        methods: List of methods to compare
        output_path: Path to save the figure
        paths: Path configuration
        n_samples: Number of samples to show
        figsize: Figure size
        title: Figure title
    """
    if paths is None:
        paths = PATHS
    
    # Select samples
    if len(sample_ids) > n_samples:
        selected_ids = random.sample(sample_ids, n_samples)
    else:
        selected_ids = sample_ids[:n_samples]
    
    n_cols = len(methods) + 1  # +1 for input
    n_rows = len(selected_ids)
    
    if figsize is None:
        figsize = (2.5 * n_cols, 2.5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Column headers
    headers = ['Input'] + [get_method_display_name(m) for m in methods]
    
    for row_idx, sample_id in enumerate(selected_ids):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            
            if col_idx == 0:
                # Input image - determine if aligned or unaligned based on first method
                method_config = _get_method_config(paths, methods[0])
                is_aligned = method_config.get('is_aligned', False)
                
                if is_aligned:
                    img_path = paths["aligned_image"].format(sample_id=sample_id)
                else:
                    img_path = paths["image"].format(sample_id=sample_id)
            else:
                method = methods[col_idx - 1]
                method_config = _get_method_config(paths, method)
                img_path = method_config["file_path"].format(sample_id=sample_id)
            
            if os.path.exists(img_path):
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12)
            
            ax.axis('off')
            
            # Add column header for first row
            if row_idx == 0:
                ax.set_title(headers[col_idx], fontsize=10, fontweight='bold', pad=5)
            
            # Add row label for first column
            if col_idx == 0:
                ax.set_ylabel(sample_id, fontsize=9, rotation=0, ha='right', va='center',
                             labelpad=10)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Qualitative comparison grid saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bald Conversion Evaluation")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/bald_dataset/evaluation",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=None,
        help="Methods to evaluate (default: all)",
    )
    parser.add_argument(
        "--alignment",
        type=str,
        choices=["aligned", "unaligned", "both"],
        default="both",
        help="Filter by alignment type",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for metric computation",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Recompute all metrics even if results exist",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation and only generate visualizations",
    )
    parser.add_argument(
        "--generate_visuals",
        action="store_true",
        help="Generate comparison visualizations after evaluation",
    )
    parser.add_argument(
        "--qualitative_samples",
        type=int,
        default=5,
        help="Number of samples for qualitative comparison",
    )
    
    args = parser.parse_args()
    
    alignment_filter = None if args.alignment == "both" else args.alignment
    
    # Evaluate all methods (unless skipped)
    if not args.skip_eval:
        results = evaluate_all_methods(
            output_dir=args.output_dir,
            methods=args.methods,
            alignment_filter=alignment_filter,
            device=args.device,
            batch_size=args.batch_size,
            force_recompute=args.force_recompute,
        )
    
    # Generate comparison tables
    aligned_df, unaligned_df = generate_comparison_table(output_dir=args.output_dir)
    
    # Generate visualizations
    if args.generate_visuals or args.skip_eval:
        create_combined_comparison_figure(output_dir=args.output_dir)
        
        # Generate qualitative comparisons
        sample_ids = _discover_sample_ids(PATHS)
        all_methods = _get_all_methods(PATHS)
        
        if all_methods['aligned']:
            create_qualitative_comparison_grid(
                sample_ids=sample_ids,
                methods=all_methods['aligned'],
                output_path=os.path.join(args.output_dir, "aligned", "qualitative_comparison.png"),
                n_samples=args.qualitative_samples,
                title="Qualitative Comparison: Aligned Methods",
            )
        
        if all_methods['unaligned']:
            create_qualitative_comparison_grid(
                sample_ids=sample_ids,
                methods=all_methods['unaligned'],
                output_path=os.path.join(args.output_dir, "unaligned", "qualitative_comparison.png"),
                n_samples=args.qualitative_samples,
                title="Qualitative Comparison: Unaligned Methods",
            )
    
    print("\n" + "=" * 80)
    print("ALL EVALUATIONS COMPLETE")
    print("=" * 80)
    
    if not aligned_df.empty:
        print("\nAligned Methods Comparison:")
        print(aligned_df.to_string(index=False))
    
    if not unaligned_df.empty:
        print("\nUnaligned Methods Comparison:")
        print(unaligned_df.to_string(index=False))