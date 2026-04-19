# ================================
# Standard Library
# ================================
import argparse
import gc
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

# ================================
# Third-Party Libraries
# ================================
import numpy as np
import torch
from PIL import Image
import cv2
from typing import List, Tuple, Optional

# ================================
# Local Imports
# ================================
from hairport.utility.warper import (
    resize_landmarks,
    align_images_for_max_iou,
    align_images_for_min_lmk_diff,
    transform_image_and_mask,
)
from hairport.core import BackgroundRemover, SAMMaskExtractor, CodeFormerEnhancer, FacialLandmarkDetector, FLAMEFitter
from hairport.config import get_config

def flush():
    gc.collect()
    torch.cuda.empty_cache()


def validate_landmarks_data(lmk_data: dict, lmk_path: Path, context: str = "landmarks") -> None:
    """Validate that landmark data is not None and contains required keys.
    
    Args:
        lmk_data: The loaded landmark data dictionary.
        lmk_path: Path to the landmarks file (for error messages).
        context: Description of which landmarks (e.g., 'source', 'target') for error messages.
    
    Raises:
        ValueError: If landmarks data is None or missing required keys.
    """
    if lmk_data is None:
        raise ValueError(
            f"Invalid {context} landmarks file: {lmk_path}\n"
            f"The file exists but contains None (likely from a failed landmark extraction).\n"
            f"Please delete this file and re-run landmark extraction for the corresponding image."
        )
    
    required_keys = ['ldm468', 'image_height', 'image_width']
    missing_keys = [k for k in required_keys if k not in lmk_data]
    
    if missing_keys:
        raise ValueError(
            f"Invalid {context} landmarks file: {lmk_path}\n"
            f"Missing required keys: {missing_keys}\n"
            f"Available keys: {list(lmk_data.keys()) if isinstance(lmk_data, dict) else 'N/A'}\n"
            f"Please delete this file and re-run landmark extraction."
        )
    
    if lmk_data['ldm468'] is None:
        raise ValueError(
            f"Invalid {context} landmarks file: {lmk_path}\n"
            f"The 'ldm468' key exists but is None (landmark detection likely failed).\n"
            f"Please delete this file and re-run landmark extraction."
        )

def get_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get the bounding box of the mask region."""
    h, w = mask.shape[0], mask.shape[1]
    
    if mask.sum() < 10:
        return 0, h, 0, w
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return (y1, y2, x1, x2)


def create_soft_blend_mask(
    hair_mask: np.ndarray,
    dilation_kernel: int = 15,
    dilation_iterations: int = 2,
    blur_radius: int = 21
) -> np.ndarray:
    """Create a soft blend mask with smooth transitions."""
    # Ensure binary mask
    mask = (hair_mask > 0.5).astype(np.uint8)
    
    # Dilate to create transition zone
    kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=dilation_iterations)
    
    # Create soft edges with Gaussian blur
    soft_mask = cv2.GaussianBlur(dilated.astype(np.float32), (blur_radius, blur_radius), 0)
    
    # Ensure original hair region stays at 1.0
    soft_mask = np.maximum(soft_mask, mask.astype(np.float32))
    
    return soft_mask


def create_distance_soft_blend_mask(
    hair_mask: np.ndarray,
    *,
    dilation_px: int = 0,
    dilation_iterations: int = 1,
    feather_px: int = 12,
    mask_threshold: float = 0.5,
) -> np.ndarray:
    """Create a distance-based soft blend mask (one-sided feather outside the hair region)."""
    mask = (hair_mask > mask_threshold).astype(np.uint8)
    if int(mask.sum()) == 0:
        return np.zeros_like(mask, dtype=np.float32)

    if dilation_px > 0 and dilation_iterations > 0:
        k = 2 * int(dilation_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=int(dilation_iterations))

    feather_px = int(max(0, feather_px))
    if feather_px == 0:
        return mask.astype(np.float32)

    # Distance in *outside* region to the nearest hair pixel.
    # OpenCV's distanceTransform returns distance to the nearest zero pixel.
    outside = (1 - mask).astype(np.uint8)
    dist_outside = cv2.distanceTransform(outside, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)

    # One-sided feather: inside hair is always 1; outside decays to 0 over `feather_px`.
    outside_alpha = np.clip(1.0 - (dist_outside / float(feather_px)), 0.0, 1.0)
    soft = np.where(mask > 0, 1.0, outside_alpha)
    return soft.astype(np.float32)


def create_hierarchical_blend_mask(
    hair_mask: np.ndarray,
    num_levels: int = 4
) -> List[np.ndarray]:
    """Create multi-scale blend masks for hierarchical blending."""
    masks = []
    current_mask = (hair_mask > 0.5).astype(np.float32)
    
    for level in range(num_levels):
        # Progressively more blur at each level
        blur_size = 5 + level * 8  # 5, 13, 21, 29
        if blur_size % 2 == 0:
            blur_size += 1
        
        # Dilation increases with level
        kernel_size = 3 + level * 4  # 3, 7, 11, 15
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1)
        
        # Apply blur
        blurred = cv2.GaussianBlur(dilated.astype(np.float32), (blur_size, blur_size), 0)
        masks.append(blurred)
    
    return masks


def poisson_blend(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    center: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Apply Poisson blending for natural composition."""
    # Ensure proper types
    source = source.astype(np.uint8)
    target = target.astype(np.uint8)
    mask_255 = (mask * 255).astype(np.uint8)
    
    # Find center if not provided
    if center is None:
        bbox = get_bbox_from_mask(mask)
        y1, y2, x1, x2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
    
    try:
        # NORMAL_CLONE preserves texture better than MIXED_CLONE for hair
        result = cv2.seamlessClone(source, target, mask_255, center, cv2.NORMAL_CLONE)
        return result
    except cv2.error:
        # Fallback to alpha blending if Poisson fails
        alpha = mask[:, :, np.newaxis] if mask.ndim == 2 else mask
        return (source * alpha + target * (1 - alpha)).astype(np.uint8)


def multi_scale_blend(
    generated: np.ndarray,
    source: np.ndarray,
    masks: List[np.ndarray],
    use_laplacian: bool = True
) -> np.ndarray:
    """Multi-scale blending using Laplacian pyramids."""
    if not use_laplacian or len(masks) < 2:
        # Simple alpha blend with first mask
        alpha = masks[0][:, :, np.newaxis]
        return (generated * alpha + source * (1 - alpha)).astype(np.uint8)
    
    num_levels = len(masks)
    
    # Build Laplacian pyramids
    def build_laplacian_pyramid(img, levels):
        pyramid = []
        current = img.astype(np.float32)
        for i in range(levels - 1):
            down = cv2.pyrDown(current)
            up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
            laplacian = current - up
            pyramid.append(laplacian)
            current = down
        pyramid.append(current)
        return pyramid
    
    # Build pyramids
    gen_pyr = build_laplacian_pyramid(generated.astype(np.float32), num_levels)
    src_pyr = build_laplacian_pyramid(source.astype(np.float32), num_levels)
    
    # Blend at each level
    blended_pyr = []
    for i in range(num_levels):
        h, w = gen_pyr[i].shape[:2]
        mask_resized = cv2.resize(masks[i], (w, h))
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]
        blended = gen_pyr[i] * mask_resized + src_pyr[i] * (1 - mask_resized)
        blended_pyr.append(blended)
    
    # Reconstruct from pyramid
    result = blended_pyr[-1]
    for i in range(num_levels - 2, -1, -1):
        result = cv2.pyrUp(result, dstsize=(blended_pyr[i].shape[1], blended_pyr[i].shape[0]))
        result = result + blended_pyr[i]
    
    return np.clip(result, 0, 255).astype(np.uint8)



def composite_hair_onto_bald(
    hair_restored_np: np.ndarray,
    bald_np: np.ndarray,
    hair_mask_np: np.ndarray,
    use_multiscale: bool = True,
    feather_px: int = 12,
) -> np.ndarray:
    """Composite hair region from hair_restored onto the bald source image."""
    # Ensure mask is normalized to 0-1
    if hair_mask_np.max() > 1:
        hair_mask_np = hair_mask_np.astype(np.float32) / 255.0
    
    # Ensure images are same size
    h, w = bald_np.shape[:2]
    if hair_restored_np.shape[:2] != (h, w):
        hair_restored_np = cv2.resize(hair_restored_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
    if hair_mask_np.shape[:2] != (h, w):
        hair_mask_np = cv2.resize(hair_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create binary mask - NO dilation to avoid including face pixels
    binary_mask = (hair_mask_np > 0.5).astype(np.float32)
    
    if use_multiscale:
        # Multi-scale Laplacian blending with proper mask handling
        # First, create a composite base: hair where mask=1, bald where mask=0
        binary_mask_3d = binary_mask[:, :, np.newaxis]
        base_composite = (hair_restored_np * binary_mask_3d + bald_np * (1 - binary_mask_3d)).astype(np.uint8)
        
        # Create soft mask for Laplacian blending (feather INWARD to soften hair edges)
        # Use erosion + blur to create soft edges that stay within hair region
        kernel_size = max(3, feather_px // 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded_mask = cv2.erode(binary_mask.astype(np.uint8), kernel, iterations=1).astype(np.float32)
        
        # Blur to create soft transition
        blur_size = max(5, feather_px)
        if blur_size % 2 == 0:
            blur_size += 1
        soft_mask = cv2.GaussianBlur(binary_mask, (blur_size, blur_size), 0)
        
        # Ensure core hair region stays at 1.0
        soft_mask = np.maximum(soft_mask, eroded_mask)
        
        # Build Laplacian pyramids for smooth blending
        num_levels = 4
        blend_masks = []
        for level in range(num_levels):
            # Progressively more blur at each level
            level_blur = 5 + level * 8
            if level_blur % 2 == 0:
                level_blur += 1
            blurred = cv2.GaussianBlur(soft_mask, (level_blur, level_blur), 0)
            blend_masks.append(blurred)
        
        # Use multi-scale blend between base_composite and bald
        # This smooths the transition without introducing gray boundaries
        composited = multi_scale_blend(
            base_composite,
            bald_np,
            blend_masks,
            use_laplacian=True
        )
    else:
        # Simple alpha blending with inward feathering
        # Create soft mask that feathers INWARD from the hair boundary
        blur_size = max(5, feather_px)
        if blur_size % 2 == 0:
            blur_size += 1
        
        # Blur the binary mask to soften edges
        soft_mask = cv2.GaussianBlur(binary_mask, (blur_size, blur_size), 0)
        
        # First create hard composite (no gray boundaries)
        binary_mask_3d = binary_mask[:, :, np.newaxis]
        base_composite = hair_restored_np * binary_mask_3d + bald_np * (1 - binary_mask_3d)
        
        # Then blend the base composite with bald using soft mask for smooth edges
        soft_mask_3d = soft_mask[:, :, np.newaxis]
        composited = (base_composite * soft_mask_3d + bald_np * (1 - soft_mask_3d)).astype(np.uint8)
    
    return composited


@dataclass
class BlendingConfig:
    # Resolution settings
    RESOLUTION: int | None = None
    OPTIMIZATION_RESOLUTION: int | None = None
    
    # Enhancement settings
    CODEFORMER_UPSCALE: int | None = None
    CODEFORMER_FIDELITY: float | None = None
    
    # Alignment settings
    ALIGNMENT_IOU_WEIGHT: float | None = None
    ALIGNMENT_LANDMARK_WEIGHT: float | None = None
    
    # SAM settings
    SAM_CONFIDENCE_THRESHOLD: float | None = None
    
    # Directory structure
    DIR_MATTED_IMAGE: str | None = None
    DIR_BALD: str | None = None
    DIR_VIEW_ALIGNED: str | None = None
    DIR_SRC_OUTPAINTED: str | None = None
    
    # Top-level directories for 3D aware/unaware processing
    DIR_3D_AWARE: str | None = None
    DIR_3D_UNAWARE: str | None = None
    
    # Subdirectories within 3d_aware/ and 3d_unaware/
    SUBDIR_WARPING: str | None = None
    SUBDIR_BLENDING: str | None = None
    SUBDIR_ALIGNMENT: str | None = None
    SUBDIR_BALD_IMAGE: str | None = None
    SUBDIR_BALD_LMK: str | None = None
    
    # File names
    FILE_LANDMARKS: str | None = None
    FILE_TARGET_IMAGE_GENERATED: str | None = None
    FILE_OUTPAINTED_IMAGE: str | None = None
    FILE_WARPED_TARGET_IMAGE: str = "warped_target_image.png"
    FILE_TARGET_HEAD_MASK: str = "target_head_mask.png"
    FILE_TARGET_HAIR_MASK: str = "target_hair_mask.png"
    FILE_TARGET_HAIR_ENHANCED: str = "target_hair_enhanced.png"
    FILE_WARPING_PARAMS: str = "warping_params.json"
    FILE_ALPHA_BLENDED: str = "alpha_blended.png"
    FILE_POISSON_BLENDED: str | None = None
    FILE_FLAME_SEGMENTATION: str = "flame_segmentation.png"
    FILE_FLAME_OVERLAY: str = "flame_overlay.png"
    FILE_HEAD_ORIENTATION: str | None = None
    
    POISSON_BLEND_STRENGTH: float | None = None

    def __post_init__(self):
        cfg = get_config()
        bh = cfg.blend_hair
        ds = cfg.dataset
        if self.RESOLUTION is None:
            self.RESOLUTION = bh.resolution
        if self.OPTIMIZATION_RESOLUTION is None:
            self.OPTIMIZATION_RESOLUTION = bh.optimization_resolution
        if self.CODEFORMER_UPSCALE is None:
            self.CODEFORMER_UPSCALE = bh.codeformer_upscale
        if self.CODEFORMER_FIDELITY is None:
            self.CODEFORMER_FIDELITY = bh.codeformer_fidelity
        if self.ALIGNMENT_IOU_WEIGHT is None:
            self.ALIGNMENT_IOU_WEIGHT = bh.alignment_iou_weight
        if self.ALIGNMENT_LANDMARK_WEIGHT is None:
            self.ALIGNMENT_LANDMARK_WEIGHT = bh.alignment_landmark_weight
        if self.SAM_CONFIDENCE_THRESHOLD is None:
            self.SAM_CONFIDENCE_THRESHOLD = bh.sam_confidence_threshold
        if self.DIR_MATTED_IMAGE is None:
            self.DIR_MATTED_IMAGE = ds.dir_matted_image
        if self.DIR_BALD is None:
            self.DIR_BALD = ds.dir_bald
        if self.DIR_VIEW_ALIGNED is None:
            self.DIR_VIEW_ALIGNED = ds.dir_view_aligned
        if self.DIR_SRC_OUTPAINTED is None:
            self.DIR_SRC_OUTPAINTED = ds.dir_source_outpainted
        if self.DIR_3D_AWARE is None:
            self.DIR_3D_AWARE = ds.dir_3d_aware
        if self.DIR_3D_UNAWARE is None:
            self.DIR_3D_UNAWARE = ds.dir_3d_unaware
        if self.SUBDIR_WARPING is None:
            self.SUBDIR_WARPING = ds.subdir_warping
        if self.SUBDIR_BLENDING is None:
            self.SUBDIR_BLENDING = ds.subdir_blending
        if self.SUBDIR_ALIGNMENT is None:
            self.SUBDIR_ALIGNMENT = ds.subdir_alignment
        if self.SUBDIR_BALD_IMAGE is None:
            self.SUBDIR_BALD_IMAGE = ds.subdir_bald_image
        if self.SUBDIR_BALD_LMK is None:
            self.SUBDIR_BALD_LMK = ds.subdir_bald_lmk
        if self.FILE_LANDMARKS is None:
            self.FILE_LANDMARKS = ds.file_landmarks
        if self.FILE_TARGET_IMAGE_GENERATED is None:
            self.FILE_TARGET_IMAGE_GENERATED = ds.file_target_phase1
        if self.FILE_OUTPAINTED_IMAGE is None:
            self.FILE_OUTPAINTED_IMAGE = "outpainted_image.png"
        if self.FILE_POISSON_BLENDED is None:
            self.FILE_POISSON_BLENDED = ds.file_poisson_blended
        if self.FILE_HEAD_ORIENTATION is None:
            self.FILE_HEAD_ORIENTATION = ds.file_head_orientation
        if self.POISSON_BLEND_STRENGTH is None:
            self.POISSON_BLEND_STRENGTH = bh.poisson_blend_strength


def requires_3d_lifting(folder_path: Path, bald_version: str) -> bool:
    """Check if 3D lifting was required by looking for camera_params.json."""
    camera_params_path = folder_path / bald_version / "camera_params.json"
    return camera_params_path.exists()


def get_or_compute_flame_segmentation(
    image: Union[np.ndarray, Image.Image, str, Path],
    output_dir: Path,
    flame_fitter: 'FLAMEFitter',
    precomputed_path: Optional[Path] = None,
    config: BlendingConfig = None,
) -> Optional[np.ndarray]:
    """Get FLAME segmentation from precomputed path or compute using FLAMEFitter.
    
    Args:
        image: Input image (numpy array, PIL Image, or path).
        output_dir: Directory to save computed segmentation.
        flame_fitter: FLAMEFitter instance.
        precomputed_path: Optional path to precomputed segmentation.
        config: BlendingConfig instance.
    
    Returns:
        FLAME segmentation mask as numpy array, or None if computation fails.
    """
    if config is None:
        config = BlendingConfig()
    
    # Try to load precomputed segmentation
    if precomputed_path is not None and precomputed_path.exists():
        print(f"Loading precomputed FLAME segmentation from: {precomputed_path}")
        try:
            flame_mask = np.array(Image.open(precomputed_path).convert('L'))
            return flame_mask
        except Exception as e:
            print(f"Warning: Failed to load precomputed FLAME segmentation: {e}")
    
    # Check if segmentation already exists in output_dir
    output_seg_path = output_dir / config.FILE_FLAME_SEGMENTATION
    if output_seg_path.exists():
        print(f"Loading existing FLAME segmentation from: {output_seg_path}")
        try:
            flame_mask = np.array(Image.open(output_seg_path).convert('L'))
            return flame_mask
        except Exception as e:
            print(f"Warning: Failed to load existing FLAME segmentation: {e}")
    
    # Compute FLAME segmentation using FLAMEFitter
    print(f"Computing FLAME segmentation...")
    try:
        # Load image if it's a path
        if isinstance(image, (str, Path)):
            image_array = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image_array = np.array(image.convert('RGB'))
        else:
            image_array = image
        
        result = flame_fitter.fit_flame(image_array)
        
        if result is None:
            print("Warning: FLAME fitting failed (no face detected or fitting error)")
            return None
        
        flame_mask, result_dict = result
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FLAME segmentation
        Image.fromarray(flame_mask).save(output_seg_path)
        print(f"Saved FLAME segmentation to: {output_seg_path}")
        
        # Save overlay image
        overlay = flame_fitter.create_overlay(image_array, flame_mask, alpha=0.4)
        overlay_path = output_dir / config.FILE_FLAME_OVERLAY
        Image.fromarray(overlay).save(overlay_path)
        print(f"Saved FLAME overlay to: {overlay_path}")
        
        # Save head orientation
        orientation_path = output_dir / config.FILE_HEAD_ORIENTATION
        with open(orientation_path, 'w') as f:
            json.dump(result_dict['head_orientation'], f, indent=2)
        print(f"Saved head orientation to: {orientation_path}")
        
        return flame_mask
        
    except Exception as e:
        print(f"Error computing FLAME segmentation: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_landmarks(image_path: Path, output_path: Path, landmark_detector: FacialLandmarkDetector) -> Path:
    """Extract landmarks from image and save to specific output path."""
    if output_path.exists():
        print(f"Landmarks already exist at {output_path}, skipping extraction.")
        return output_path
    
    try:
        # Use FacialLandmarkDetector to extract all landmark formats
        result = landmark_detector.get_lmk_full(str(image_path))
        if result is None:
            print(f"Failed to detect landmarks for {image_path}")
            return None
        
        # Save landmarks
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, result)
        print(f"Saved landmarks to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error extracting landmarks: {e}")
        return None


def load_source_data(
    folder_path: Path, 
    bald_version: str, 
    config: BlendingConfig,
    landmark_detector: FacialLandmarkDetector = None,
) -> dict:
    """Load source (outpainted bald) image and landmarks from the pair folder.
    
    If landmarks are missing or invalid, attempts to generate them using the landmark_detector.
    
    Args:
        folder_path: Path to the view-aligned folder.
        bald_version: Bald version subdirectory (e.g., 'w_seg', 'wo_seg').
        config: BlendingConfig instance.
        landmark_detector: Optional FacialLandmarkDetector for generating missing landmarks.
    
    Returns:
        Dictionary with 'image' and 'landmarks' keys.
    
    Raises:
        FileNotFoundError: If source image not found.
        ValueError: If landmarks are invalid and cannot be generated.
    """
    bald_folder_path = folder_path / bald_version
    source_outpainted_dir = bald_folder_path / config.DIR_SRC_OUTPAINTED
    
    # Load outpainted source image
    source_image_path =  source_outpainted_dir / config.FILE_OUTPAINTED_IMAGE
    if not source_image_path.exists():
        raise FileNotFoundError(f"Outpainted source image not found: {source_image_path}")
    source_image = Image.open(source_image_path).convert('RGB')
    
    # Load source landmarks
    source_lmk_path = source_outpainted_dir / config.FILE_LANDMARKS
    source_lmk_data = None
    needs_generation = False
    
    if source_lmk_path.exists():
        try:
            source_lmk_data = np.load(source_lmk_path, allow_pickle=True).item()
            # Validate landmarks data
            validate_landmarks_data(source_lmk_data, source_lmk_path, context="source outpainted")
        except (ValueError, TypeError) as e:
            print(f"Warning: Invalid source landmarks file: {e}")
            print(f"Will attempt to regenerate landmarks...")
            needs_generation = True
            # Delete the invalid file
            source_lmk_path.unlink()
    else:
        print(f"Source landmarks not found at {source_lmk_path}")
        needs_generation = True
    
    # Attempt to generate landmarks if needed
    if needs_generation:
        if landmark_detector is None:
            raise ValueError(
                f"Source landmarks are missing/invalid at {source_lmk_path} and no landmark_detector provided.\n"
                f"Cannot generate landmarks without a FacialLandmarkDetector instance."
            )
        
        print(f"Generating landmarks for source image: {source_image_path}")
        try:
            result = landmark_detector.get_lmk_full(str(source_image_path))
            if result is None:
                raise ValueError(
                    f"Failed to detect landmarks for source image: {source_image_path}\n"
                    f"Landmark detection returned None (no face detected in the image).\n"
                    f"Skipping this folder."
                )
            
            # Save the generated landmarks
            source_lmk_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(source_lmk_path, result)
            print(f"Saved generated landmarks to: {source_lmk_path}")
            source_lmk_data = result
            
            # Validate the newly generated landmarks
            validate_landmarks_data(source_lmk_data, source_lmk_path, context="source outpainted (generated)")
            
        except Exception as e:
            raise ValueError(
                f"Failed to generate landmarks for source image: {source_image_path}\n"
                f"Error: {e}\n"
                f"Skipping this folder."
            )
    
    return {
        'image': source_image,
        'landmarks': source_lmk_data,
    }


def load_target_data(
    folder_path: Path, 
    data_dir: Path,
    bald_version: str, 
    use_3d_lifting: bool,
    codeformer_enhancer: CodeFormerEnhancer,
    landmark_detector: FacialLandmarkDetector,
    config: BlendingConfig
) -> dict:
    """Load target image and landmarks from the appropriate location."""
    bald_folder_path = folder_path / bald_version
    target_id = folder_path.name.split("_to_")[0]
    
    if use_3d_lifting:
        # 3D lifting case: use alignment/target_image.png
        alignment_dir = folder_path / config.SUBDIR_ALIGNMENT
        target_image_path = alignment_dir / config.FILE_TARGET_IMAGE_GENERATED
        
        if not target_image_path.exists():
            raise FileNotFoundError(f"Target image not found: {target_image_path}")
        
        target_image =  Image.open(target_image_path).convert('RGB')

        # Compute landmarks for enhanced image and save to alignment/landmarks.npy
        target_lmk_path = alignment_dir / "landmarks.npy"
        target_lmk_path = extract_landmarks(target_image_path, target_lmk_path, landmark_detector)
        
        if target_lmk_path is None or not target_lmk_path.exists():
            raise FileNotFoundError(f"Failed to extract landmarks for enhanced target image")
        
        target_lmk_data = np.load(target_lmk_path, allow_pickle=True).item()
        
        # Validate landmarks data
        validate_landmarks_data(target_lmk_data, target_lmk_path, context="target (3D lifted)")
        
    else:
        # Non-3D lifting case: use hair_aligned_image or original image
        hair_aligned_path = data_dir / "image" / f"{target_id}.png"
        # original_image_path = data_dir / "image" / f"{target_id}.png"
        
        if hair_aligned_path.exists():
            target_image_path = hair_aligned_path
            print(f"Using hair-aligned target image: {target_image_path}")
        # elif original_image_path.exists():
        #     target_image_path = original_image_path
        #     print(f"Using original target image: {target_image_path}")
        else:
            raise FileNotFoundError(
                f"Target image not found at:\n"
                f"  {hair_aligned_path}\n"
            )
        
        target_image = Image.open(target_image_path).convert('RGB')
        target_image = target_image.resize(
            (config.RESOLUTION, config.RESOLUTION), 
            Image.Resampling.LANCZOS
        )
        
        # Load existing landmarks from data_dir (lmk folder)
        target_lmk_path = data_dir / "lmk" / target_id / "landmarks.npy"
        
        if not target_lmk_path.exists():
            raise FileNotFoundError(
                f"Target landmarks not found at: {target_lmk_path}\n"
                f"For non-3D lifting cases, landmarks must already exist in the data_dir/lmk folder."
            )
        
        target_lmk_data = np.load(target_lmk_path, allow_pickle=True).item()
        
        # Validate landmarks data
        validate_landmarks_data(target_lmk_data, target_lmk_path, context="target")

    return {
        'image': target_image,
        'landmarks': target_lmk_data,
    }

def create_white_background_image(image_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create image with white background where mask is 0 (non-foreground)."""
    # Ensure mask is binary (0 or 1)
    if mask.max() > 1:
        mask = (mask > 127).astype(np.float32)
    else:
        mask = mask.astype(np.float32)
    
    # Expand mask to 3 channels if needed
    if mask.ndim == 2:
        mask_3d = mask[:, :, np.newaxis]
    else:
        mask_3d = mask
    
    # Create white background (255, 255, 255)
    white_bg = np.ones_like(image_np) * 255
    
    # Composite: foreground where mask=1, white where mask=0
    result = (image_np * mask_3d + white_bg * (1 - mask_3d)).astype(np.uint8)
    return result


def prepare_images_and_masks(source_data: dict, target_data: dict, bg_remover: BackgroundRemover, config: BlendingConfig) -> dict:
    output_resolution = config.RESOLUTION
    opt_resolution = config.OPTIMIZATION_RESOLUTION
    
    # Prepare full resolution images (for final output)
    source_image_full = source_data['image'].convert('RGB').resize(
        (output_resolution, output_resolution), Image.Resampling.LANCZOS
    )
    target_image_full = target_data['image'].convert('RGB').resize(
        (output_resolution, output_resolution), Image.Resampling.LANCZOS
    )
    source_image_np = np.array(source_image_full)
    target_image_np = np.array(target_image_full)
    
    # Extract silhouette masks at full resolution for alignment
    _, source_mask_full = bg_remover.remove_background(source_image_full, refine_foreground=False)
    _, target_mask_full = bg_remover.remove_background(target_image_full, refine_foreground=False)
    
    # Masks at optimization resolution
    source_mask_opt = np.array(
        source_mask_full.resize((opt_resolution, opt_resolution), Image.Resampling.NEAREST)
    ) > 127
    target_mask_opt = np.array(
        target_mask_full.resize((opt_resolution, opt_resolution), Image.Resampling.NEAREST)
    ) > 127
    
    # Prepare optimization resolution data (for faster alignment)
    # Use white-background images for alignment to focus on foreground alignment
    source_image_opt_raw = np.array(
        source_data['image'].convert('RGB').resize((opt_resolution, opt_resolution), Image.Resampling.LANCZOS)
    )
    target_image_opt_raw = np.array(
        target_data['image'].convert('RGB').resize((opt_resolution, opt_resolution), Image.Resampling.LANCZOS)
    )
    
    # Create white-background versions for alignment
    source_image_opt = create_white_background_image(source_image_opt_raw, source_mask_opt)
    target_image_opt = create_white_background_image(target_image_opt_raw, target_mask_opt)
    
    # Masks at full resolution
    source_mask_np = np.array(source_mask_full) > 127
    target_mask_np = np.array(target_mask_full) > 127
    
    # Landmarks at optimization resolution
    source_lmk_opt = resize_landmarks(
        source_data['landmarks']['ldm468'],
        [source_data['landmarks']['image_height'], source_data['landmarks']['image_width']],
        [opt_resolution, opt_resolution]
    )
    target_lmk_opt = resize_landmarks(
        target_data['landmarks']['ldm468'],
        [target_data['landmarks']['image_height'], target_data['landmarks']['image_width']],
        [opt_resolution, opt_resolution]
    )
    
    return {
        # Full resolution data (for final output)
        'source_image': source_image_np,
        'target_image': target_image_np,
        'source_mask': source_mask_np,
        'target_mask': target_mask_np,
        # Optimization resolution data
        'source_image_opt': source_image_opt,
        'target_image_opt': target_image_opt,
        'source_mask_opt': source_mask_opt,
        'target_mask_opt': target_mask_opt,
        'source_landmarks_opt': source_lmk_opt,
        'target_landmarks_opt': target_lmk_opt,
        # Resolution info for parameter scaling
        'optimization_resolution': opt_resolution,
        'output_resolution': output_resolution,
    }


def align_target_to_source(
    prepared_data: dict, 
    bg_remover: BackgroundRemover, 
    config: BlendingConfig,
    source_flame_mask: Optional[np.ndarray] = None,
    target_flame_mask: Optional[np.ndarray] = None,
) -> tuple:
    """Perform alignment optimization at lower resolution, then scale parameters for output resolution.
    
    Prioritizes IOU-based alignment using FLAME masks when available, falls back to landmark-based.
    """
    opt_res = prepared_data['optimization_resolution']
    out_res = prepared_data['output_resolution']
    scale_factor = out_res / opt_res
    
    use_iou_alignment = False
    
    # Try IOU-based alignment if FLAME masks are provided
    if source_flame_mask is not None and target_flame_mask is not None:
        try:
            # Resize FLAME masks to optimization resolution
            source_flame_opt = cv2.resize(
                source_flame_mask.astype(np.float32),
                (opt_res, opt_res),
                interpolation=cv2.INTER_NEAREST
            ) > 127
            target_flame_opt = cv2.resize(
                target_flame_mask.astype(np.float32),
                (opt_res, opt_res),
                interpolation=cv2.INTER_NEAREST
            ) > 127
            
            # Use IoU + landmark alignment with FLAME masks
            print("Using FLAME-based IoU + landmark alignment")
            _, _, _, params_opt, iou = align_images_for_max_iou(
                prepared_data['source_image_opt'],
                source_flame_opt,
                prepared_data['target_image_opt'],
                target_flame_opt,
                source_flame_opt,
                target_flame_opt,
                source_lmk=prepared_data['source_landmarks_opt'],
                target_lmk=prepared_data['target_landmarks_opt'],
                iou_weight=config.ALIGNMENT_IOU_WEIGHT,
                landmark_weight=config.ALIGNMENT_LANDMARK_WEIGHT
            )
            params_opt['iou'] = iou
            use_iou_alignment = True
            print(f"FLAME-based IOU alignment successful, IoU: {iou:.4f}")
            
        except Exception as e:
            print(f"Warning: FLAME-based IOU alignment failed: {e}")
            print("Falling back to landmark-only alignment...")
            use_iou_alignment = False
    else:
        print("FLAME masks not available, using landmark-only alignment")
    
    # Fallback to landmark-only alignment
    if not use_iou_alignment:
        print("Using landmark-only alignment (Procrustes + refinement)")
        _, params_opt, lmk_dist = align_images_for_min_lmk_diff(
            prepared_data['source_image_opt'],
            prepared_data['target_image_opt'],
            prepared_data['source_landmarks_opt'],
            prepared_data['target_landmarks_opt'],
            allow_rotation=True,
            allow_nonuniform_scale=True,
        )
        params_opt['iou'] = None
    
    # Scale translation parameters to output resolution
    # Scale and angle remain the same, only translation needs scaling
    params_scaled = {
        'scale_x': params_opt['scale_x'],
        'scale_y': params_opt['scale_y'],
        'angle_rad': params_opt['angle_rad'],
        'angle_deg': params_opt['angle_deg'],
        'tx': params_opt['tx'] * scale_factor,  # Scale translation
        'ty': params_opt['ty'] * scale_factor,  # Scale translation
        'iou': params_opt.get('iou'),
        'landmark_distance': params_opt['landmark_distance'] * scale_factor if params_opt.get('landmark_distance') is not None else None,
    }
    
    print(f"\nScaled parameters from {opt_res}x{opt_res} to {out_res}x{out_res}:")
    print(f"  Translation: ({params_opt['tx']:.2f}, {params_opt['ty']:.2f}) -> ({params_scaled['tx']:.2f}, {params_scaled['ty']:.2f}) pixels")
    
    # Apply scaled transformation to full resolution images
    warped_image, warped_mask, _ = transform_image_and_mask(
        prepared_data['target_image'],
        prepared_data['target_mask'],
        prepared_data['target_mask'],
        params_scaled['scale_x'],
        params_scaled['scale_y'],
        params_scaled['angle_rad'],
        params_scaled['tx'],
        params_scaled['ty']
    )
    
    # Refine warped mask using background removal
    print("Refining warped mask with background removal...")
    warped_image_pil = Image.fromarray(warped_image)
    _, silhouette_mask = bg_remover.remove_background(warped_image_pil, refine_foreground=False)
    silhouette_mask_np = (np.array(silhouette_mask) > 127).astype(np.float32)
    
    # Perform pixel-wise AND between silhouette and warped mask
    refined_warped_mask = warped_mask * silhouette_mask_np
    print(f"Mask refinement: Original mask area = {warped_mask.sum():.0f}, Refined mask area = {refined_warped_mask.sum():.0f}")
    
    # Create matted warped image with white background
    warped_image_matted = create_white_background_image(warped_image, refined_warped_mask)
    
    return warped_image_matted, refined_warped_mask, params_scaled, params_scaled.get('iou')


def create_fallback_warped_image(
    target_image: Image.Image,
    bg_remover: BackgroundRemover,
    config: BlendingConfig,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Create a fallback warped image when alignment fails (no landmarks or FLAME fitting failure).
    
    This simply uses the target image as-is with identity transformation parameters,
    and extracts a silhouette mask using the background remover.
    
    Args:
        target_image: The target image to use as fallback.
        bg_remover: Background remover for mask extraction.
        config: Blending configuration.
    
    Returns:
        Tuple of (warped_image_np, warped_mask_np, params_dict)
    """
    print("=" * 60)
    print("WARNING: Using fallback - target image used directly without warping")
    print("=" * 60)
    
    # Resize target image to output resolution
    target_resized = target_image.convert('RGB').resize(
        (config.RESOLUTION, config.RESOLUTION), 
        Image.Resampling.LANCZOS
    )
    target_np = np.array(target_resized)
    
    # Extract silhouette mask
    _, silhouette_mask = bg_remover.remove_background(target_resized, refine_foreground=False)
    mask_np = (np.array(silhouette_mask) > 127).astype(np.float32)
    
    # Create white-background version
    warped_image = create_white_background_image(target_np, mask_np)
    
    # Identity transformation parameters
    params = {
        'scale_x': 1.0,
        'scale_y': 1.0,
        'angle_rad': 0.0,
        'angle_deg': 0.0,
        'tx': 0.0,
        'ty': 0.0,
        'iou': None,
        'landmark_distance': None,
        'fallback': True,  # Flag to indicate this was a fallback
    }
    
    return warped_image, mask_np, params


def save_warping_results(
    folder_path: Path,
    bald_version: str,
    warped_image: np.ndarray,
    warped_mask: np.ndarray,
    params: dict,
    iou: float,
    config: BlendingConfig,
    mode_dir: str = None,
) -> Path:
    """Save warping results to {folder_path}/{bald_version}/{mode_dir}/warping/."""
    if mode_dir is None:
        mode_dir = config.DIR_3D_UNAWARE
    warping_dir = folder_path / bald_version / mode_dir / config.SUBDIR_WARPING
    warping_dir.mkdir(parents=True, exist_ok=True)
    
    # Save warped image
    warped_image_pil = Image.fromarray(warped_image)
    warped_image_path = warping_dir / config.FILE_WARPED_TARGET_IMAGE
    warped_image_pil.save(warped_image_path)
    
    # Save warped mask
    warped_mask_pil = Image.fromarray((warped_mask * 255).astype(np.uint8))
    warped_mask_path = warping_dir / config.FILE_TARGET_HEAD_MASK
    warped_mask_pil.save(warped_mask_path)
    
    # Save parameters
    params_to_save = {
        "scale_x": float(params['scale_x']),
        "scale_y": float(params['scale_y']),
        "angle_rad": float(params['angle_rad']),
        "angle_deg": float(np.degrees(params['angle_rad'])),
        "tx": float(params['tx']),
        "ty": float(params['ty']),
        "final_iou": float(iou) if iou is not None else None,
        "landmark_distance": float(params['landmark_distance']) if params.get('landmark_distance') is not None else None,
    }
    params_path = warping_dir / config.FILE_WARPING_PARAMS
    with open(params_path, 'w') as f:
        json.dump(params_to_save, f, indent=2)
    
    return warped_image_path


def extract_hair_mask(warped_image: Image.Image, output_path: Path, bg_remover: BackgroundRemover, sam_extractor: SAMMaskExtractor, config: BlendingConfig) -> Image.Image:
    """Extract hair mask from warped image using SAM."""
    foreground, silh_mask = bg_remover.remove_background(warped_image, refine_foreground=True)
    hair_mask, confidence = sam_extractor(foreground, prompt="head hair")
    hair_mask_np = np.array(hair_mask)
    silh_mask_np = np.array(silh_mask)
    hair_mask_np = np.where(silh_mask_np > 0, hair_mask_np, 0)
    hair_mask = Image.fromarray(hair_mask_np)
    hair_mask.save(output_path)
    
    print(f"Saved hair mask to: {output_path}")
    print(f"Confidence score: {confidence:.4f}")
    
    return hair_mask


def create_hair_only_image(warped_image: Image.Image, hair_mask: Image.Image, config: BlendingConfig) -> Image.Image:
    image_np = np.array(warped_image.resize((config.RESOLUTION, config.RESOLUTION), Image.Resampling.LANCZOS))
    mask_np = (np.array(hair_mask.resize((config.RESOLUTION, config.RESOLUTION), Image.Resampling.NEAREST)) / 255.0).astype(np.uint8)
    hair_only_np = image_np * mask_np[..., None]
    return Image.fromarray(hair_only_np).convert("RGB")


def blend_hair_onto_source(
    source_image: np.ndarray,
    hair_image: Image.Image,
    hair_mask: Image.Image,
    output_dir: Path,
    config: BlendingConfig,
) -> dict:
    """Blend hair onto source image using alpha and multi-scale Laplacian blending."""
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure consistent sizes
    target_size = (config.RESOLUTION, config.RESOLUTION)
    if hair_image.size != target_size:
        hair_image = hair_image.resize(target_size, Image.Resampling.LANCZOS)
    if hair_mask.size != target_size:
        hair_mask = hair_mask.resize(target_size, Image.Resampling.NEAREST)
    
    # Ensure source is correct size
    if source_image.shape[:2] != (config.RESOLUTION, config.RESOLUTION):
        source_image = cv2.resize(source_image, target_size, interpolation=cv2.INTER_LANCZOS4)

    hair_image_np = np.array(hair_image.convert("RGB"))
    hair_mask_np = np.array(hair_mask)
    
    results = {}
    
    # Define blending configurations: (method_name, filename, use_multiscale)
    blending_configs = [
        ("alpha_blending", config.FILE_ALPHA_BLENDED, False), 
        ("multiscale_blending", config.FILE_POISSON_BLENDED, True),
    ]
    
    for method_name, filename, use_multiscale in blending_configs:
        print(f"Performing {method_name} (use_multiscale={use_multiscale})...")
        
        # Use composite_hair_onto_bald for blending
        blended_np = composite_hair_onto_bald(
            hair_restored_np=hair_image_np,
            bald_np=source_image,
            hair_mask_np=hair_mask_np,
            use_multiscale=use_multiscale,
            feather_px=21,
        )
        
        blended = Image.fromarray(blended_np)
        
        output_path = output_dir / filename
        blended.save(output_path)
        print(f"Saved {method_name} image to: {output_path}")
        
        # Store result
        result_key = "alpha_blended" if not use_multiscale else "multiscale_blended"
        results[result_key] = blended
    
    return results

def enhance_hair_region(warped_image: Image.Image, hair_mask: Image.Image, codeformer_enhancer: CodeFormerEnhancer, config: BlendingConfig) -> Image.Image:
    """Apply CodeFormer enhancement only to the hair region of the warped image."""
    # Enhance the entire image
    enhanced_image = codeformer_enhancer.enhance(
        warped_image, 
        upscale=1,  # Keep same size
        codeformer_fidelity=0.5, 
        face_upsample=False
    )
    
    # Convert to numpy for masking
    warped_np = np.array(warped_image.convert('RGB'))
    enhanced_np = np.array(enhanced_image.convert('RGB'))
    hair_mask_np = np.array(hair_mask.resize(warped_image.size, Image.Resampling.NEAREST)) / 255.0
    
    # Apply enhanced version only to hair region
    # Expand mask for broadcasting
    hair_mask_3d = hair_mask_np[..., np.newaxis]
    
    # Blend: use enhanced where mask is 1, original where mask is 0
    result_np = (enhanced_np * hair_mask_3d + warped_np * (1 - hair_mask_3d)).astype(np.uint8)
    
    return Image.fromarray(result_np)


def process_view_aligned_folder(
    folder_path: Union[str, Path],
    data_dir: Union[str, Path],
    bald_version: str,
    config: BlendingConfig = None,
    codeformer_enhancer: CodeFormerEnhancer = None,
    bg_remover: BackgroundRemover = None,
    landmark_detector: FacialLandmarkDetector = None,
    sam_extractor: SAMMaskExtractor = None,
    flame_fitter: FLAMEFitter = None,
) -> bool:
    if config is None:
        config = BlendingConfig()
    
    folder_path = Path(folder_path)
    data_dir = Path(data_dir)
    target_id, source_id = folder_path.name.split("_to_")
    bald_folder_path = folder_path / bald_version
    
    # Initialize models once if not provided
    should_cleanup = False
    if codeformer_enhancer is None:
        print("Initializing CodeFormer enhancer...")
        codeformer_enhancer = CodeFormerEnhancer(device='cuda', ultrasharp=True)
        should_cleanup = True
    
    if bg_remover is None:
        print("Initializing BackgroundRemover...")
        bg_remover = BackgroundRemover()
        should_cleanup = True
    
    if landmark_detector is None:
        print("Initializing FacialLandmarkDetector...")
        landmark_detector = FacialLandmarkDetector(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        should_cleanup = True
    
    if sam_extractor is None:
        print("Initializing SAMMaskExtractor...")
        sam_extractor = SAMMaskExtractor(confidence_threshold=config.SAM_CONFIDENCE_THRESHOLD)
        should_cleanup = True
    
    if flame_fitter is None:
        print("Initializing FLAMEFitter...")
        flame_fitter = FLAMEFitter()
        should_cleanup = True

    # Check if source outpainted image exists
    source_outpainted_dir = bald_folder_path / config.DIR_SRC_OUTPAINTED
    source_image_path = source_outpainted_dir / config.FILE_OUTPAINTED_IMAGE
    if not source_image_path.exists():
        print(f"Source outpainted image not found: {source_image_path}")
        return False
    
    # Determine if 3D lifting was used (camera_params.json exists)
    use_3d_lifting = requires_3d_lifting(folder_path, bald_version)
    print(f"3D lifting required: {use_3d_lifting}")
    
    # Determine which modes to process
    # For 3D lifting samples: perform BOTH 3D aware and 3D unaware
    # For non-3D lifting samples: only perform 3D unaware
    modes_to_process = []
    
    if use_3d_lifting:
        # Check 3D aware mode
        dir_3d_aware = bald_folder_path / config.DIR_3D_AWARE
        blending_3d_aware = dir_3d_aware / config.SUBDIR_BLENDING
        alpha_3d_aware = blending_3d_aware / config.FILE_ALPHA_BLENDED
        poisson_3d_aware = blending_3d_aware / config.FILE_POISSON_BLENDED
        if not (alpha_3d_aware.exists() and poisson_3d_aware.exists()):
            modes_to_process.append('3d_aware')
        else:
            print(f"3D aware blending already exists in {blending_3d_aware}")
        
        # Check 3D unaware mode
        dir_3d_unaware = bald_folder_path / config.DIR_3D_UNAWARE
        blending_3d_unaware = dir_3d_unaware / config.SUBDIR_BLENDING
        alpha_3d_unaware = blending_3d_unaware / config.FILE_ALPHA_BLENDED
        poisson_3d_unaware = blending_3d_unaware / config.FILE_POISSON_BLENDED
        if not (alpha_3d_unaware.exists() and poisson_3d_unaware.exists()):
            modes_to_process.append('3d_unaware')
        else:
            print(f"3D unaware blending already exists in {blending_3d_unaware}")
    else:
        pass
        # Non-3D lifting: only 3D unaware
        # dir_3d_unaware = bald_folder_path / config.DIR_3D_UNAWARE
        # blending_3d_unaware = dir_3d_unaware / config.SUBDIR_BLENDING
        # alpha_3d_unaware = blending_3d_unaware / config.FILE_ALPHA_BLENDED
        # poisson_3d_unaware = blending_3d_unaware / config.FILE_POISSON_BLENDED
        # if not (alpha_3d_unaware.exists() and poisson_3d_unaware.exists()):
        #     modes_to_process.append('3d_unaware')
        # else:
        #     print(f"3D unaware blending already exists in {blending_3d_unaware}")
    
    # Check if all processing is already done
    if not modes_to_process:
        print(f"All blending outputs already exist, skipping processing.")
        if should_cleanup:
            del codeformer_enhancer
            del bg_remover
            del landmark_detector
            del sam_extractor
            flush()
        return True
    
    try:
        # Step 1: Load source data (same for both modes)
        source_data = load_source_data(folder_path, bald_version, config, landmark_detector=landmark_detector)
        
        # Prepare source image at full resolution
        source_image_full = source_data['image'].convert('RGB').resize(
            (config.RESOLUTION, config.RESOLUTION), Image.Resampling.LANCZOS
        )
        source_image_np = np.array(source_image_full)
        
        # Process each mode with its own warping
        for mode in modes_to_process:
            print(f"\n{'='*60}")
            print(f"Processing {mode} mode...")
            print(f"{'='*60}")
            
            # Determine mode directory and target source
            mode_dir = config.DIR_3D_AWARE if mode == '3d_aware' else config.DIR_3D_UNAWARE
            mode_base_dir = bald_folder_path / mode_dir
            warping_dir = mode_base_dir / config.SUBDIR_WARPING
            blending_dir = mode_base_dir / config.SUBDIR_BLENDING
            
            if mode == '3d_aware':
                # Load view-aligned target from 3D lifting
                # For 3D aware: target is alignment/target_image_phase_1.png
                alignment_dir = folder_path / config.SUBDIR_ALIGNMENT
                target_image_path_3d = alignment_dir / config.FILE_TARGET_IMAGE_GENERATED
                
                use_fallback = False
                target_data = None
                fallback_target_image = None
                
                try:
                    target_data = load_target_data(
                        folder_path, data_dir, bald_version, use_3d_lifting, 
                        codeformer_enhancer, landmark_detector, config
                    )
                    print(f"Using view-aligned target image for 3D aware warping")
                except (FileNotFoundError, ValueError) as e:
                    print(f"Warning: Failed to load target data for 3D aware mode: {e}")
                    use_fallback = True
                    # Load the target image directly for fallback
                    if target_image_path_3d.exists():
                        fallback_target_image = Image.open(target_image_path_3d).convert('RGB')
                        print(f"Will use fallback with target image: {target_image_path_3d}")
                    else:
                        print(f"Warning: Fallback target image not found at {target_image_path_3d}, skipping {mode}")
                        continue
            else:
                # Load original target image directly
                dataset_name = data_dir.name.lower()
                if dataset_name == "celeba_reduced":
                    image_folder = "image_outpainted"
                else:
                    image_folder = "image"
                    
                original_target_path = data_dir / image_folder / f"{target_id}.png"
                if not original_target_path.exists():
                    print(f"Warning: Original target image not found at {original_target_path}, skipping {mode}")
                    continue
                
                print(f"Loading original target image from: {original_target_path}")
                original_target = Image.open(original_target_path).convert('RGB')
                original_target = original_target.resize(
                    (config.RESOLUTION, config.RESOLUTION), 
                    Image.Resampling.LANCZOS
                )
                
                # Try to load/generate landmarks, set use_fallback if it fails
                use_fallback = False
                target_data = None
                fallback_target_image = None
                
                target_lmk_path = data_dir / f"{image_folder}_lmk" / target_id / "landmarks.npy"
                target_lmk_data = None
                needs_generation = False
                
                if target_lmk_path.exists():
                    try:
                        target_lmk_data = np.load(target_lmk_path, allow_pickle=True).item()
                        validate_landmarks_data(target_lmk_data, target_lmk_path, context="target")
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Invalid target landmarks file: {e}")
                        print(f"Will attempt to regenerate landmarks...")
                        needs_generation = True
                        try:
                            target_lmk_path.unlink()
                        except:
                            pass
                else:
                    print(f"Target landmarks not found at {target_lmk_path}")
                    needs_generation = True
                
                if needs_generation:
                    print(f"Generating landmarks for target image: {original_target_path}")
                    try:
                        result = landmark_detector.get_lmk_full(str(original_target_path))
                        if result is None:
                            print(f"Warning: Failed to detect landmarks for target image (no face detected)")
                            print(f"Will use fallback with target image directly")
                            use_fallback = True
                            fallback_target_image = original_target
                        else:
                            target_lmk_path.parent.mkdir(parents=True, exist_ok=True)
                            np.save(target_lmk_path, result)
                            print(f"Saved generated landmarks to: {target_lmk_path}")
                            target_lmk_data = result
                    except Exception as e:
                        print(f"Warning: Failed to generate landmarks for target image: {e}")
                        print(f"Will use fallback with target image directly")
                        use_fallback = True
                        fallback_target_image = original_target
                
                if not use_fallback:
                    target_data = {
                        'image': original_target,
                        'landmarks': target_lmk_data,
                    }
                    print(f"Using original target image for 3D unaware warping")
            
            # Define paths for warping outputs
            warped_image_path = warping_dir / config.FILE_WARPED_TARGET_IMAGE
            warped_mask_path = warping_dir / config.FILE_TARGET_HEAD_MASK
            warping_params_path = warping_dir / config.FILE_WARPING_PARAMS
            hair_mask_path = warping_dir / config.FILE_TARGET_HAIR_MASK
            hair_only_image_path = warping_dir / "target_hair_only_image.png"
            hair_enhanced_path = warping_dir / config.FILE_TARGET_HAIR_ENHANCED
            
            # Step 2 & 3 & 4: Prepare images, align target to source, and save warping results
            if warped_image_path.exists() and warped_mask_path.exists() and warping_params_path.exists():
                print(f"Warping outputs already exist in {warping_dir}, loading from disk...")
                warped_image = np.array(Image.open(warped_image_path).convert('RGB'))
                warped_image_pil = Image.fromarray(warped_image)
            elif use_fallback:
                # Use fallback: target image directly without warping
                print(f"Using fallback mode - skipping alignment due to previous failures")
                warped_image, warped_mask, params = create_fallback_warped_image(
                    fallback_target_image, bg_remover, config
                )
                iou = None
                
                # Save warping results (with fallback flag)
                warped_image_path = save_warping_results(
                    folder_path, bald_version, warped_image, warped_mask, params, iou, config,
                    mode_dir=mode_dir
                )
                warped_image_pil = Image.fromarray(warped_image)
                print(f"Saved fallback warping results to {warping_dir}")
            else:
                # Normal warping flow
                # Step 2: Prepare images and masks
                try:
                    prepared_data = prepare_images_and_masks(source_data, target_data, bg_remover, config)
                except Exception as e:
                    print(f"Warning: Failed to prepare images and masks: {e}")
                    print(f"Using fallback mode with target image directly")
                    warped_image, warped_mask, params = create_fallback_warped_image(
                        target_data['image'], bg_remover, config
                    )
                    iou = None
                    warped_image_path = save_warping_results(
                        folder_path, bald_version, warped_image, warped_mask, params, iou, config,
                        mode_dir=mode_dir
                    )
                    warped_image_pil = Image.fromarray(warped_image)
                    print(f"Saved fallback warping results to {warping_dir}")
                    # Continue to next steps (hair mask extraction, etc.)
                    # Skip the else block below
                    prepared_data = None
                
                if prepared_data is not None:
                    # Get or compute FLAME segmentations for IOU-based alignment
                    source_flame_mask = None
                    target_flame_mask = None
                    
                    # Source FLAME segmentation (computed on outpainted image)
                    print("Getting FLAME segmentation for source image...")
                    source_flame_mask = get_or_compute_flame_segmentation(
                        image=source_data['image'],
                        output_dir=source_outpainted_dir,
                        flame_fitter=flame_fitter,
                        precomputed_path=None,  # Always compute for source outpainted
                        config=config,
                    )
                    
                    # Check if source FLAME failed - trigger fallback
                    if source_flame_mask is None:
                        print(f"Warning: FLAME fitting failed for source image")
                        print(f"Using fallback mode with target image directly")
                        warped_image, warped_mask, params = create_fallback_warped_image(
                            target_data['image'], bg_remover, config
                        )
                        iou = None
                        warped_image_path = save_warping_results(
                            folder_path, bald_version, warped_image, warped_mask, params, iou, config,
                            mode_dir=mode_dir
                        )
                        warped_image_pil = Image.fromarray(warped_image)
                        print(f"Saved fallback warping results to {warping_dir}")
                    else:
                        # Target FLAME segmentation
                        print("Getting FLAME segmentation for target image...")
                        if mode == '3d_aware':
                            # For 3D aware: compute and save in alignment folder
                            alignment_dir = folder_path / config.SUBDIR_ALIGNMENT
                            target_flame_mask = get_or_compute_flame_segmentation(
                                image=target_data['image'],
                                output_dir=alignment_dir,
                                flame_fitter=flame_fitter,
                                precomputed_path=None,  # Compute for view-aligned target
                                config=config,
                            )
                        else:
                            # For 3D unaware: compute FLAME segmentation directly via FLAMEFitter
                            target_flame_output_dir = data_dir / f"{image_folder}_flame_output" / target_id
                            target_flame_mask = get_or_compute_flame_segmentation(
                                image=target_data['image'],
                                output_dir=target_flame_output_dir,
                                flame_fitter=flame_fitter,
                                precomputed_path=None,
                                config=config,
                            )
                        
                        # Check if target FLAME failed - trigger fallback
                        if target_flame_mask is None:
                            print(f"Warning: FLAME fitting failed for target image")
                            print(f"Using fallback mode with target image directly")
                            warped_image, warped_mask, params = create_fallback_warped_image(
                                target_data['image'], bg_remover, config
                            )
                            iou = None
                            warped_image_path = save_warping_results(
                                folder_path, bald_version, warped_image, warped_mask, params, iou, config,
                                mode_dir=mode_dir
                            )
                            warped_image_pil = Image.fromarray(warped_image)
                            print(f"Saved fallback warping results to {warping_dir}")
                        else:
                            # Perform alignment (IOU-based with FLAME masks, or fallback to landmark-based)
                            print(f"Performing alignment for {mode}...")
                            try:
                                warped_image, warped_mask, params, iou = align_target_to_source(
                                    prepared_data, 
                                    bg_remover, 
                                    config,
                                    source_flame_mask=source_flame_mask,
                                    target_flame_mask=target_flame_mask,
                                )
                                
                                # Save warping results
                                warped_image_path = save_warping_results(
                                    folder_path, bald_version, warped_image, warped_mask, params, iou, config,
                                    mode_dir=mode_dir
                                )
                                warped_image_pil = Image.fromarray(warped_image)
                                print(f"Saved warping results to {warping_dir}")
                            except Exception as e:
                                print(f"Warning: Alignment failed: {e}")
                                print(f"Using fallback mode with target image directly")
                                warped_image, warped_mask, params = create_fallback_warped_image(
                                    target_data['image'], bg_remover, config
                                )
                                iou = None
                                warped_image_path = save_warping_results(
                                    folder_path, bald_version, warped_image, warped_mask, params, iou, config,
                                    mode_dir=mode_dir
                                )
                                warped_image_pil = Image.fromarray(warped_image)
                                print(f"Saved fallback warping results to {warping_dir}")
            
            # Step 5: Extract hair mask for warped image (skip if already exists)
            if hair_mask_path.exists():
                print(f"Hair mask already exists at {hair_mask_path}, loading from disk...")
                hair_mask = Image.open(hair_mask_path).convert('L')
            else:
                hair_mask = extract_hair_mask(warped_image_pil, hair_mask_path, bg_remover, sam_extractor, config)
            
            # Step 6: Apply CodeFormer enhancement to hair region if not already done
            if hair_enhanced_path.exists():
                print(f"Enhanced hair image already exists at {hair_enhanced_path}, loading from disk...")
                warped_image_for_hair = Image.open(hair_enhanced_path).convert('RGB')
            elif mode == '3d_aware' and use_3d_lifting:
                # For 3D aware, target was already enhanced during view alignment
                warped_image_for_hair = warped_image_pil
            else:
                # For 3D unaware (or 3D aware without prior enhancement), enhance the hair region
                print("Applying CodeFormer enhancement to hair region...")
                warped_image_for_hair = enhance_hair_region(warped_image_pil, hair_mask, codeformer_enhancer, config)
                warped_image_for_hair.save(hair_enhanced_path)
                print(f"Saved enhanced hair image to: {hair_enhanced_path}")
            
            # Step 7: Create hair-only image (for visualization/debugging only)
            hair_only_image = create_hair_only_image(warped_image_for_hair, hair_mask, config)
            hair_only_image.save(hair_only_image_path)
            
            # Step 8: Perform blending
            print(f"\nPerforming blending for {mode}...")
            print(f"Output directory: {blending_dir}")
            
            blend_hair_onto_source(
                source_image_np,
                warped_image_for_hair,
                hair_mask,
                blending_dir,
                config,
            )
        
        # Cleanup if we initialized models locally
        if should_cleanup:
            del codeformer_enhancer
            del bg_remover
            del landmark_detector
            del sam_extractor
            flush()
        
        return True
        
    except Exception as e:
        print(f"Error during processing for folder {folder_path}: {e}")
        import traceback
        traceback.print_exc()
        # Cleanup if we initialized models locally
        if should_cleanup:
            del codeformer_enhancer
            del bg_remover
            del landmark_detector
            del sam_extractor
            flush()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blend hair from view-aligned folders")
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
        "--bald_version",
        type=str,
        default="w_seg",
        choices=["wo_seg", "w_seg", "all"],
        help="Bald version to process"
    )
    args = parser.parse_args()
    
    config = BlendingConfig()
    data_dir = Path(args.data_dir)
    provider_subdir = f"shape_{args.shape_provider}__texture_{args.texture_provider}"
    view_aligned_dir = data_dir / config.DIR_VIEW_ALIGNED / provider_subdir
    
    if not view_aligned_dir.exists():
        print(f"View aligned directory not found: {view_aligned_dir}")
        exit(1)
    
    all_folders = [f for f in view_aligned_dir.iterdir() if f.is_dir()]
    
    timestamp_seed = int(time.time())
    random.seed(timestamp_seed)
    random.shuffle(all_folders)
    print(f"Shuffled folders using timestamp seed: {timestamp_seed}")
    print(f"Found {len(all_folders)} view-aligned folders\n")
    
    # Process each bald version
    if args.bald_version == "all":
        bald_versions = ["w_seg", "wo_seg"]
    else:
        bald_versions = [args.bald_version]
    
    processed_count = 0
    error_count = 0
    
    # Initialize models once for batch processing
    print("Initializing models for batch processing...")
    # codeformer_enhancer = CodeFormerEnhancer(device='cuda', ultrasharp=True)
    bg_remover = BackgroundRemover()
    landmark_detector = FacialLandmarkDetector(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    sam_extractor = SAMMaskExtractor(confidence_threshold=config.SAM_CONFIDENCE_THRESHOLD)
    flame_fitter = FLAMEFitter()
    print("Models initialized.\n")
    
    try:
        for bald_version in bald_versions:
            print(f"\n{'='*60}")
            print(f"Processing bald_version: {bald_version}")
            print(f"{'='*60}\n")
            
            for i, folder in enumerate(all_folders, 1):
                print(f"\n[{i}/{len(all_folders)}] Processing {folder.name} (bald_version={bald_version})")
                
                result = process_view_aligned_folder(
                    folder_path=folder,
                    data_dir=data_dir,
                    bald_version=bald_version,
                    config=config,
                    codeformer_enhancer=None,
                    bg_remover=bg_remover,
                    landmark_detector=landmark_detector,
                    sam_extractor=sam_extractor,
                    flame_fitter=flame_fitter,
                )
                
                if result:
                    processed_count += 1
                else:
                    error_count += 1
    finally:
        # Cleanup models
        print("\nCleaning up models...")
        # del codeformer_enhancer
        del bg_remover
        del landmark_detector
        del sam_extractor
        flush()
    
    print(f"\n{'='*60}")
    print(f"Processing complete:")
    print(f"  Processed: {processed_count}")
    print(f"  Errors/Missing files: {error_count}")
    print(f"{'='*60}")