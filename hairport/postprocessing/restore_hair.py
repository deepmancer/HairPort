"""Advanced Hair Transfer Pipeline using InsertAnything with RF-Inversion."""

# ================================
# Standard Library
# ================================
import argparse
import gc
import json
import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ================================
# Third-Party Libraries
# ================================
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from tqdm import tqdm

# ================================
# Diffusers Imports
# ================================
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline

from utils.sam_mask_extractor import SAMMaskExtractor

from utils.bg_remover import BackgroundRemover
from hairport.utility.uncrop_sdxl import ImageUncropper


def flush():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()


# ================================
# Singleton Wrappers for Heavy Models
# ================================

class SAMMaskExtractorSingleton:
    """
    Singleton wrapper for SAMMaskExtractor to avoid repeated instantiation.
    
    SAMMaskExtractor loads heavy models (SAM) which is expensive. This singleton
    ensures the model is loaded once and reused across all calls.
    
    Usage:
        sam = SAMMaskExtractorSingleton.get_instance()
        mask, score = sam(image, prompt="hair")
    """
    _instance: Optional[SAMMaskExtractor] = None
    _confidence_threshold: float = 0.4
    _detection_threshold: float = 0.5
    
    @classmethod
    def get_instance(
        cls,
        confidence_threshold: float = 0.4,
        detection_threshold: float = 0.5,
        force_reload: bool = False,
    ) -> SAMMaskExtractor:
        """
        Get or create the singleton SAMMaskExtractor instance.
        
        Args:
            confidence_threshold: Confidence threshold for SAM
            detection_threshold: Detection threshold for SAM
            force_reload: If True, reload the model even if already loaded
            
        Returns:
            SAMMaskExtractor instance
        """
        if SAMMaskExtractor is None:
            raise RuntimeError("SAMMaskExtractor is not available. Cannot extract hair mask.")
        
        # Check if we need to reload (thresholds changed or forced)
        thresholds_changed = (
            cls._confidence_threshold != confidence_threshold or
            cls._detection_threshold != detection_threshold
        )
        
        if cls._instance is None or force_reload or thresholds_changed:
            # Clean up existing instance if any
            if cls._instance is not None:
                del cls._instance
                cls._instance = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Create new instance
            cls._instance = SAMMaskExtractor(
                confidence_threshold=confidence_threshold,
                detection_threshold=detection_threshold
            )
            cls._confidence_threshold = confidence_threshold
            cls._detection_threshold = detection_threshold
            print(f"[SAMMaskExtractorSingleton] Initialized with confidence={confidence_threshold}, detection={detection_threshold}")
        
        return cls._instance
    
    @classmethod
    def release(cls):
        """Release the singleton instance and free GPU memory."""
        if cls._instance is not None:
            del cls._instance
            cls._instance = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[SAMMaskExtractorSingleton] Released instance and cleared GPU memory")
    
    @classmethod
    def is_available(cls) -> bool:
        """Check if SAMMaskExtractor is available."""
        return SAMMaskExtractor is not None


class BackgroundRemoverSingleton:
    """
    Singleton wrapper for BackgroundRemover to avoid repeated instantiation.
    
    BackgroundRemover loads models which can be expensive. This singleton
    ensures the model is loaded once and reused across all calls.
    
    Usage:
        bg_remover = BackgroundRemoverSingleton.get_instance()
        image, mask = bg_remover.remove_background(input_image)
    """
    _instance: Optional[BackgroundRemover] = None
    
    @classmethod
    def get_instance(cls, force_reload: bool = False) -> BackgroundRemover:
        """
        Get or create the singleton BackgroundRemover instance.
        
        Args:
            force_reload: If True, reload the model even if already loaded
            
        Returns:
            BackgroundRemover instance
        """
        if cls._instance is None or force_reload:
            # Clean up existing instance if any
            if cls._instance is not None:
                del cls._instance
                cls._instance = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            cls._instance = BackgroundRemover()
            print("[BackgroundRemoverSingleton] Initialized")
        
        return cls._instance
    
    @classmethod
    def release(cls):
        """Release the singleton instance and free GPU memory."""
        if cls._instance is not None:
            del cls._instance
            cls._instance = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[BackgroundRemoverSingleton] Released instance and cleared GPU memory")


def extract_hair_mask(
    image: Image.Image,
    confidence_threshold: float = 0.4,
    detection_threshold: float = 0.5,
    prompt: str = "hair",
    sam_instance: Optional[SAMMaskExtractor] = None,
    bg_remover_instance: Optional[BackgroundRemover] = None,
) -> tuple[Image.Image, float]:
    """Extract hair mask from an image using SAM.
    
    Args:
        image: PIL Image to extract hair mask from
        confidence_threshold: Confidence threshold for SAM (default: 0.4)
        detection_threshold: Detection threshold for SAM (default: 0.5)
        prompt: Text prompt for SAM (default: "hair")
        sam_instance: Optional pre-instantiated SAMMaskExtractor (uses singleton if None)
        bg_remover_instance: Optional pre-instantiated BackgroundRemover (uses singleton if None)
        
    Returns:
        tuple: (hair_mask_image, confidence_score)
        
    Raises:
        RuntimeError: If SAMMaskExtractor is not available
    """
    if not SAMMaskExtractorSingleton.is_available():
        raise RuntimeError("SAMMaskExtractor is not available. Cannot extract hair mask.")
    
    # Use provided instances or get from singletons
    sam = sam_instance if sam_instance is not None else SAMMaskExtractorSingleton.get_instance(
        confidence_threshold=confidence_threshold,
        detection_threshold=detection_threshold
    )
    bg_remover = bg_remover_instance if bg_remover_instance is not None else BackgroundRemoverSingleton.get_instance()
    
    _, silh_mask_pil = bg_remover.remove_background(image)
    hair_mask_pil, score = sam(image, prompt=prompt)
    
    # Ensure hair mask does not extend beyond the silhouette mask
    hair_mask_np = np.array(hair_mask_pil).astype(np.float32) / 255.0
    silh_mask_np = np.array(silh_mask_pil).astype(np.float32) / 255.0
    # Multiply masks to constrain hair to within silhouette
    constrained_hair_mask = (hair_mask_np * silh_mask_np * 255.0).astype(np.uint8)
    hair_mask_pil = Image.fromarray(constrained_hair_mask)
    
    return hair_mask_pil, score


# ================================
# Configuration
# ================================

@dataclass
class HairTransferConfig:
    """Configuration for hair transfer pipeline."""
    
    # Model settings
    FLUX_FILL_MODEL: str = "black-forest-labs/FLUX.1-Fill-dev"
    FLUX_REDUX_MODEL: str = "black-forest-labs/FLUX.1-Redux-dev"
    LORA_WEIGHTS_PATH: str = "/workspace/HairPort/Hairdar/insert_anything_lora.safetensors"
    LORA_SCALE: float = 1.0  # LoRA weight scale (lower = less InsertAnything influence, more Redux influence)
    
    # Processing resolution
    PROCESSING_RESOLUTION: int = 768
    OUTPUT_RESOLUTION: int = 1024
    
    # Generation parameters
    SEED: int = 42
    GUIDANCE_SCALE: float = 30.0
    NUM_INFERENCE_STEPS: int = 50
    MAX_SEQUENCE_LENGTH: int = 512
    
    # Reference image processing
    REF_EXPAND_RATIO: float = 1.3
    
    # Mask processing
    HAIR_MASK_DILATION_KERNEL: int = 25
    HAIR_MASK_DILATION_ITERATIONS: int = 2
    BLEND_MASK_DILATION_KERNEL: int = 20
    BLEND_MASK_DILATION_ITERATIONS: int = 2
    BLEND_MASK_BLUR_RADIUS: int = 15
    DIPTYCH_MASK_BLUR_RADIUS: int = 21  # Gaussian blur radius for diptych mask to avoid hard boundaries

    # Second-pass mask processing (tighter to preserve identity/background)
    SECOND_PASS_HAIR_MASK_DILATION_KERNEL: int = 24
    SECOND_PASS_HAIR_MASK_DILATION_ITERATIONS: int = 2
    SECOND_PASS_BLEND_MASK_DILATION_KERNEL: int = 20
    SECOND_PASS_BLEND_MASK_DILATION_ITERATIONS: int = 2
    SECOND_PASS_BLEND_MASK_BLUR_RADIUS: int = 10
    
    # Bounding box expansion
    BBOX_EXPAND_RATIO: float = 1.2
    CROP_EXPAND_RATIO: float = 2.5
    
    # Blending parameters
    POISSON_BLEND: bool = True
    MULTI_SCALE_BLEND: bool = True
    BLEND_FEATHER_SIZE: int = 25
    
    # Latent-space blending parameters (FLUX Flow Matching compatible)
    LATENT_BLEND: bool = True  # Use latent-space blending instead of pixel blending
    LATENT_BLEND_START_STEP: float = 0.0  # Start blending immediately (was 0.3) - critical for identity
    LATENT_BLEND_STRENGTH: float = 0.9  # Higher strength (was 0.5) - better identity preservation
    LATENT_BLEND_SCHEDULE: str = "constant"  # Use constant (was flux_linear) - consistent preservation
    
    # Conditioning freedom parameters (allows natural hair-scalp integration)
    # In later denoising steps, reduce image conditioning strength to allow
    # the model more freedom to naturally integrate hair with the scalp
    COND_FREEDOM_ENABLED: bool = False  # Enable conditioning freedom in later steps
    COND_FREEDOM_START_RATIO: float = 0.8  # Start reducing conditioning at 60% of denoising
    COND_FREEDOM_END_STRENGTH: float = 0.4  # Final image cond strength (0=text-only, 1=full image cond)
    COND_FREEDOM_SCHEDULE: str = "cosine"  # "linear", "cosine", or "step"
    
    # Directory structure
    DIR_VIEW_ALIGNED: str = "view_aligned"
    DIR_ALIGNMENT: str = "alignment"
    DIR_BALD: str = "bald"
    DIR_POSTPROCESSING: str = "fill_processed"
    DIR_PROMPTS: str = "prompt"
    
    # Top-level directories for 3D aware/unaware processing
    DIR_3D_AWARE: str = "3d_aware"
    DIR_3D_UNAWARE: str = "3d_unaware"
    
    # Subdirectories within 3d_aware/ and 3d_unaware/
    SUBDIR_WARPING: str = "warping"
    SUBDIR_BLENDING: str = "blending"
    SUBDIR_TRANSFERRED: str = "transferred"  # Output directory for hair restoration
    
    MATTED_IMAGE_SUBDIR: str = "matted_image"  # Legacy
    MATTED_IMAGE_HAIR_MASK_SUBDIR: str = "matted_image_mask"  # Legacy
    
    # Redux image source directories (in priority order)
    DIR_HAIR_ALIGNED_IMAGE: str = "hair_aligned_image"  # Primary: hair-aligned images
    DIR_IMAGE: str = "image"  # Fallback: original images
    
    # File names
    FILE_VIEW_ALIGNED_IMAGE: str = "target_image.png"
    FILE_WARPED_TARGET_IMAGE: str = "warped_target_image.png"
    FILE_WARPED_HAIR_MASK: str = "target_hair_mask.png"
    FILE_TARGET_HAIR_MASK: str = "target_image_hair_mask.png"
    FILE_CAMERA_PARAMS: str = "camera_params.json"
    
    # Output file naming (simple names, directory structure encodes settings)
    FILE_HAIR_RESTORED: str = "hair_restored.png"
    FILE_HAIR_RESTORED_MASK: str = "hair_restored_mask.png"
    FILE_HAIR_RESTORED_FINAL: str = "hair_restored_final.png"
    FILE_HAIR_RESTORED_FINAL_MASK: str = "hair_restored_final_mask.png"
    
    # Legacy output file naming (for backward compatibility)
    FILE_OUTPUT_PREFIX: str = "transferred_fill"
    FILE_OUTPUT_MASK_SUFFIX: str = "_mask"
    FILE_OUTPUT_FINAL_SUFFIX: str = "_final"
    
    # Outpainted source image paths
    DIR_SOURCE_OUTPAINTED: str = "source_outpainted"
    FILE_OUTPAINTED_IMAGE: str = "outpainted_image.png"
    FILE_RESIZE_INFO: str = "resize_info.json"
    
    SUBDIR_BALD_IMAGE: str = "image"

# ================================
# Image Processing Utilities
# ================================

def f(r, T=0.6, beta=0.1):
    return np.where(r < T, beta + (1 - beta) / T * r, 1)

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


def expand_bbox(
    mask: np.ndarray,
    yyxx: Tuple[int, int, int, int],
    ratio: float,
    min_crop: int = 0
) -> Tuple[int, int, int, int]:
    """Expand the bounding box by a given ratio."""
    y1, y2, x1, x2 = yyxx
    H, W = mask.shape[0], mask.shape[1]
    
    yyxx_area = (y2 - y1 + 1) * (x2 - x1 + 1)
    r1 = yyxx_area / (H * W)
    r2 = f(r1)
    ratio = math.sqrt(r2 / r1)
    
    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2 - y1 + 1)
    w = ratio * (x2 - x1 + 1)
    h = max(h, min_crop)
    w = max(w, min_crop)
    
    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)
    
    x1 = max(0, x1)
    x2 = min(W, x2)
    y1 = max(0, y1)
    y2 = min(H, y2)
    return (y1, y2, x1, x2)


def pad_to_square(
    image: np.ndarray,
    pad_value: int = 255,
    random_pad: bool = False
) -> np.ndarray:
    """Pad the image to a square shape."""
    H, W = image.shape[0], image.shape[1]
    if H == W:
        return image
    
    padd = abs(H - W)
    if random_pad:
        padd_1 = int(np.random.randint(0, padd))
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1
    
    if len(image.shape) == 2:
        if H > W:
            pad_param = ((0, 0), (padd_1, padd_2))
        else:
            pad_param = ((padd_1, padd_2), (0, 0))
    elif len(image.shape) == 3:
        if H > W:
            pad_param = ((0, 0), (padd_1, padd_2), (0, 0))
        else:
            pad_param = ((padd_1, padd_2), (0, 0), (0, 0))
    
    image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
    return image


def expand_image_mask(
    image: np.ndarray,
    mask: np.ndarray,
    ratio: float = 1.4
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand the image and mask with padding."""
    h, w = image.shape[0], image.shape[1]
    H, W = int(h * ratio), int(w * ratio)
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W - w) // 2)
    w2 = W - w - w1
    
    pad_param_image = ((h1, h2), (w1, w2), (0, 0))
    pad_param_mask = ((h1, h2), (w1, w2))
    image = np.pad(image, pad_param_image, 'constant', constant_values=255)
    mask = np.pad(mask, pad_param_mask, 'constant', constant_values=0)
    return image, mask


def box2square(
    image: np.ndarray,
    box: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """Convert the bounding box to a square shape."""
    H, W = image.shape[0], image.shape[1]
    y1, y2, x1, x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h, w = y2 - y1, x2 - x1
    
    if h >= w:
        x1 = cx - h // 2
        x2 = cx + h // 2
    else:
        y1 = cy - w // 2
        y2 = cy + w // 2
    
    x1 = max(0, x1)
    x2 = min(W, x2)
    y1 = max(0, y1)
    y2 = min(H, y2)
    return (y1, y2, x1, x2)


def crop_back(
    pred: np.ndarray,
    tar_image: np.ndarray,
    extra_sizes: np.ndarray,
    tar_box_yyxx_crop: np.ndarray
) -> np.ndarray:
    """Crop the predicted image back to the original image."""
    H1, W1, H2, W2 = extra_sizes
    y1, y2, x1, x2 = tar_box_yyxx_crop
    pred = cv2.resize(pred, (W2, H2))
    m = 2  # margin_pixel
    
    if W1 == H1:
        if m != 0:
            tar_image[y1 + m:y2 - m, x1 + m:x2 - m, :] = pred[m:-m, m:-m]
        else:
            tar_image[y1:y2, x1:x2, :] = pred[:, :]
        return tar_image
    
    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:, pad1:-pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1:-pad2, :, :]
    
    gen_image = tar_image.copy()
    if m != 0:
        gen_image[y1 + m:y2 - m, x1 + m:x2 - m, :] = pred[m:-m, m:-m]
    else:
        gen_image[y1:y2, x1:x2, :] = pred[:, :]
    
    return gen_image


# ================================
# Advanced Blending Utilities
# ================================

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
    """Create a distance-based soft blend mask."""
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
    
    if use_multiscale:
        # Use multi-scale Laplacian blending for natural transitions
        blend_masks = create_hierarchical_blend_mask(hair_mask_np, num_levels=4)
        composited = multi_scale_blend(
            hair_restored_np,
            bald_np,
            blend_masks,
            use_laplacian=True
        )
    else:
        # Simple soft alpha blending
        soft_mask = create_distance_soft_blend_mask(
            hair_mask_np,
            dilation_px=3,
            dilation_iterations=1,
            feather_px=feather_px,
        )
        alpha = soft_mask[:, :, np.newaxis]
        composited = (hair_restored_np * alpha + bald_np * (1 - alpha)).astype(np.uint8)
    
    return composited



# ================================
# Inversion Utilities for Identity Preservation
# ================================

class LatentInverter:
    """Helper for VAE latent encoding/decoding."""
    
    def __init__(self, pipeline: FluxFillPipeline, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.pipe = pipeline
        self.device = device
        self.dtype = dtype
    
    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode an image to latent space using the VAE."""
        # Preprocess image
        image_tensor = self.pipe.image_processor.preprocess(image)
        # Use the same dtype as the VAE for encoding
        vae_dtype = next(self.pipe.vae.parameters()).dtype
        image_tensor = image_tensor.to(device=self.device, dtype=vae_dtype)
        
        # Encode with VAE
        latent = self.pipe.vae.encode(image_tensor).latent_dist.sample()
        latent = (latent - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        
        return latent.to(self.dtype)
    
    @torch.no_grad()
    def encode_for_blending(self, image: Image.Image, target_size: Tuple[int, int]) -> torch.Tensor:
        """Encode an image for latent-space blending."""
        # Resize image to match expected latent dimensions
        # VAE downsamples by 8x, so target latent size * 8 = image size
        expected_image_size = (target_size[1] * 8, target_size[0] * 8)  # (W, H) for PIL
        
        if image.size != expected_image_size:
            image = image.resize(expected_image_size, Image.Resampling.LANCZOS)
        
        return self.encode_image(image)
    
    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """
        Decode a latent tensor back to an image.
        """
        # Use the same dtype as the VAE
        vae_dtype = next(self.pipe.vae.parameters()).dtype
        
        # Denormalize
        latent = latent.to(vae_dtype)
        latent = (latent / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
        
        # Decode
        image = self.pipe.vae.decode(latent, return_dict=False)[0]
        
        # Post-process
        image = self.pipe.image_processor.postprocess(image)[0]
        return image
    


# ================================
# Latent-Space Blending Callback
# ================================

def _flux_unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Unpack FLUX sequence latents back to spatial format."""
    batch_size = latents.shape[0]
    
    # FLUX uses 2x2 patches on top of VAE's 8x downsampling
    # So effective patch size is 16x16 in image space
    latent_h = height // 16
    latent_w = width // 16
    
    # Reshape from [batch, seq, 64] to [batch, h, w, 16, 2, 2]
    # The 64 = 16 channels * 2 * 2 (patch)
    latents = latents.view(batch_size, latent_h, latent_w, 16, 2, 2)
    
    # Rearrange to [batch, 16, h*2, w*2] = [batch, 16, H/8, W/8]
    latents = latents.permute(0, 3, 1, 4, 2, 5).contiguous()
    latents = latents.view(batch_size, 16, latent_h * 2, latent_w * 2)
    
    return latents


def _flux_pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack spatial latents into FLUX sequence format."""
    batch_size, channels, h, w = latents.shape
    
    # h and w should be divisible by 2 (FLUX patch size)
    assert h % 2 == 0 and w % 2 == 0, f"Latent dimensions must be even, got {h}x{w}"
    
    # Reshape to [batch, 16, h//2, 2, w//2, 2]
    latents = latents.view(batch_size, channels, h // 2, 2, w // 2, 2)
    
    # Rearrange to [batch, h//2, w//2, 16, 2, 2] then flatten
    latents = latents.permute(0, 2, 4, 1, 3, 5).contiguous()
    latents = latents.view(batch_size, (h // 2) * (w // 2), channels * 4)
    
    return latents


class LatentBlendCallback:
    """Callback for latent-space blending during FLUX denoising."""
    
    def __init__(
        self,
        source_latents: torch.Tensor,
        blend_mask: torch.Tensor,
        start_step_ratio: float = 0.0,
        blend_strength: float = 0.85,
        schedule: str = "constant",
        total_steps: int = 50,
        generator: torch.Generator = None,
        image_height: int = 768,
        image_width: int = 768
    ):
        self.source_latents_clean = source_latents
        self.blend_mask = blend_mask
        self.start_step_ratio = start_step_ratio
        self.blend_strength = blend_strength
        self.schedule = schedule
        self.total_steps = total_steps
        self.generator = generator
        self.current_step = 0
        
        self.image_height = image_height
        self.image_width = image_width
        self._cached_noise = None
        
    def _get_or_create_noise(self, shape: tuple, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get cached noise or create new noise matching the source latents."""
        if self._cached_noise is None or self._cached_noise.shape != shape:
            if self.generator is not None:
                self._cached_noise = torch.randn(shape, generator=self.generator, device=device, dtype=dtype)
            else:
                self._cached_noise = torch.randn(shape, device=device, dtype=dtype)
        return self._cached_noise.to(device=device, dtype=dtype)

    def _infer_flow_t(self, pipe, step: int, timestep: Union[torch.Tensor, float, int]) -> float:
        """Infer continuous flow time t in [0,1] for FLUX/flow-matching."""
        # Extract scalar
        if isinstance(timestep, torch.Tensor):
            t_value = float(timestep.detach().flatten()[0].item())
        else:
            t_value = float(timestep)

        # Prefer scheduler-provided range (most reliable across implementations)
        scheduler = getattr(pipe, "scheduler", None)
        timesteps = getattr(scheduler, "timesteps", None) if scheduler is not None else None
        try:
            if timesteps is not None:
                ts = timesteps.detach().float().cpu().flatten()
                if ts.numel() >= 2:
                    t_min = float(ts.min().item())
                    t_max = float(ts.max().item())
                    if t_max != t_min:
                        t_norm = (t_value - t_min) / (t_max - t_min)
                        # Clamp (numerical safety)
                        return float(min(1.0, max(0.0, t_norm)))
        except Exception:
            # Fall back below
            pass

        # Heuristic fallback: if it already looks like [0,1], use it.
        if 0.0 <= t_value <= 1.0:
            return t_value

        # Final fallback: approximate by step index. Denoising solves from t=1 -> t=0.
        if self.total_steps <= 1:
            return 0.0
        frac = step / (self.total_steps - 1)
        return float(1.0 - min(1.0, max(0.0, frac)))
    
    def _get_blend_weight(self, denoise_progress: float) -> float:
        """Compute blending weight as a function of denoising progress."""
        denoise_progress = float(min(1.0, max(0.0, denoise_progress)))

        if denoise_progress < self.start_step_ratio:
            return 0.0

        adjusted = (denoise_progress - self.start_step_ratio) / (1.0 - self.start_step_ratio)
        adjusted = float(min(1.0, max(0.0, adjusted)))

        if self.schedule == "flux_linear":
            return adjusted * self.blend_strength
        if self.schedule == "flux_cosine":
            return (1 - math.cos(adjusted * math.pi)) / 2 * self.blend_strength
        if self.schedule == "constant":
            return self.blend_strength
        return adjusted * self.blend_strength
    
    def _add_noise_to_source(
        self, 
        source_clean: torch.Tensor, 
        t: float,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Forward-noise the source to match flow time t."""
        noise = self._get_or_create_noise(source_clean.shape, device, dtype)
        t = float(min(1.0, max(0.0, t)))
        source_noisy = (1.0 - t) * source_clean + t * noise
        return source_noisy
    
    def __call__(
        self,
        pipe,
        step: int,
        timestep: torch.Tensor,
        callback_kwargs: dict
    ) -> dict:
        """
        Callback invoked at each denoising step.
        
        For FLUX compatibility:
        1. Unpack sequence latents to spatial format
        2. Blend in spatial format (easier for mask operations)
        3. Repack to sequence format
        """
        self.current_step = step
        latents = callback_kwargs.get("latents")
        
        if latents is None:
            return callback_kwargs
        
        # Infer continuous flow time and compute denoising progress.
        # For flow matching: t=1 is noise, t=0 is data => denoise_progress = 1 - t
        t = self._infer_flow_t(pipe, step=step, timestep=timestep)
        denoise_progress = 1.0 - t
        weight = self._get_blend_weight(denoise_progress)
        
        if weight <= 0:
            return callback_kwargs
        
        device = latents.device
        dtype = latents.dtype
        
        # Check if latents are in FLUX packed format [batch, seq, hidden]
        if latents.dim() != 3:
            # Not packed format, skip
            return callback_kwargs
        
        batch_size, seq_len, hidden_dim = latents.shape
        
        # For diptych, width is doubled
        diptych_width = self.image_width * 2
        diptych_height = self.image_height
        
        # Verify sequence length matches expected diptych size
        expected_seq_len = (diptych_height // 16) * (diptych_width // 16)
        if seq_len != expected_seq_len:
            # Sequence length doesn't match, might be different resolution
            # Try to infer dimensions
            # For FLUX, seq_len = (H/16) * (W/16), hidden = 64
            # We know diptych has 2:1 aspect ratio
            # So if seq_len = h * w where w = 2*h, then seq_len = 2*h^2
            # h = sqrt(seq_len / 2)
            h_patches = int(math.sqrt(seq_len / 2))
            w_patches = seq_len // h_patches
            if h_patches * w_patches != seq_len:
                print(f"[LatentBlend] Cannot infer dimensions from seq_len={seq_len}")
                return callback_kwargs
            diptych_height = h_patches * 16
            diptych_width = w_patches * 16
        
        try:
            # Unpack FLUX latents to spatial format
            latents_spatial = _flux_unpack_latents(latents, diptych_height, diptych_width)
            # Now shape is [batch, 16, H/8, W/8] = [batch, 16, diptych_h/8, diptych_w/8]
            
            # Get source latents and forward-noise to the current flow time
            source_clean = self.source_latents_clean.to(device=device, dtype=dtype)
            source_noisy = self._add_noise_to_source(source_clean, t=t, device=device, dtype=dtype)
            
            # Source is single image, diptych is [ref | target]
            # We only blend into the right half (target)
            single_w = latents_spatial.shape[-1] // 2
            
            # Verify source matches target half dimensions
            if source_noisy.shape[-2:] != (latents_spatial.shape[-2], single_w):
                # Resize source to match
                source_noisy = F.interpolate(
                    source_noisy,
                    size=(latents_spatial.shape[-2], single_w),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Prepare mask
            mask = self.blend_mask.to(device=device, dtype=dtype)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(0)
            
            # Resize mask to match latent spatial dimensions
            mask = F.interpolate(
                mask,
                size=(latents_spatial.shape[-2], single_w),
                mode='bilinear',
                align_corners=False
            )
            
            # Expand mask to all channels
            mask = mask.expand(batch_size, latents_spatial.shape[1], -1, -1)
            
            # Split diptych: [reference | target]
            left_half = latents_spatial[..., :single_w]   # Reference (keep)
            right_half = latents_spatial[..., single_w:]  # Target (blend)
            
            # Blend: inv_mask where 1=non-hair (preserve source), 0=hair (keep generated)
            inv_mask = 1.0 - mask
            
            # Per-pixel blend strength (preserve source in non-hair regions)
            blend_weight = weight * inv_mask  # weight in non-hair regions, 0 in hair
            
            blended_right = right_half * (1 - blend_weight) + source_noisy * blend_weight
            
            # Reconstruct diptych
            blended_spatial = torch.cat([left_half, blended_right], dim=-1)
            
            # Repack to FLUX sequence format
            latents = _flux_pack_latents(blended_spatial)
            
            callback_kwargs["latents"] = latents
            
        except Exception as e:
            print(f"[LatentBlend] Error during blending: {e}")
            # Return original latents on error
            
        return callback_kwargs


# ================================
# Conditioning Freedom Callback
# ================================

class ConditioningFreedomCallback:
    """Callback that reduces image conditioning influence in later denoising steps."""
    
    def __init__(
        self,
        start_ratio: float = 0.6,
        end_strength: float = 0.3,
        schedule: str = "cosine",
        total_steps: int = 50,
        image_height: int = 768,
        image_width: int = 768,
        generator: torch.Generator = None,
    ):
        self.start_ratio = start_ratio
        self.end_strength = end_strength
        self.schedule = schedule
        self.total_steps = total_steps
        self.image_height = image_height
        self.image_width = image_width
        self.generator = generator
        
        # Cache noise for consistent perturbation
        self._cached_noise = None
        
    def _get_conditioning_strength(self, denoise_progress: float) -> float:
        """Compute the conditioning strength as a function of denoising progress."""
        if denoise_progress < self.start_ratio:
            return 1.0  # Full conditioning in early steps
        
        # Progress through the fade region [start_ratio, 1.0]
        fade_progress = (denoise_progress - self.start_ratio) / (1.0 - self.start_ratio)
        fade_progress = max(0.0, min(1.0, fade_progress))
        
        if self.schedule == "linear":
            strength = 1.0 - fade_progress * (1.0 - self.end_strength)
        elif self.schedule == "cosine":
            # Smooth cosine transition
            strength = self.end_strength + (1.0 - self.end_strength) * (1.0 + math.cos(fade_progress * math.pi)) / 2
        elif self.schedule == "step":
            # Abrupt transition at midpoint
            strength = 1.0 if fade_progress < 0.5 else self.end_strength
        else:
            strength = 1.0 - fade_progress * (1.0 - self.end_strength)
            
        return strength
    
    def _infer_denoise_progress(self, pipe, step: int, timestep) -> float:
        """
        Infer denoising progress (0=start/noisy, 1=end/clean).
        
        For FLUX flow matching, the scheduler goes from t=1 (noise) to t=0 (clean).
        """
        if isinstance(timestep, torch.Tensor):
            t_value = float(timestep.detach().flatten()[0].item())
        else:
            t_value = float(timestep)
        
        # Try to use scheduler timesteps for accurate mapping
        scheduler = getattr(pipe, "scheduler", None)
        timesteps = getattr(scheduler, "timesteps", None) if scheduler else None
        
        if timesteps is not None:
            try:
                ts = timesteps.detach().float().cpu().flatten()
                if ts.numel() >= 2:
                    t_min, t_max = float(ts.min()), float(ts.max())
                    if t_max != t_min:
                        # t normalized to [0, 1] where 0=noise, 1=clean in scheduler space
                        t_norm = (t_value - t_min) / (t_max - t_min)
                        # For flow matching going t=1->0, denoise_progress = 1 - t_norm
                        return 1.0 - max(0.0, min(1.0, t_norm))
            except Exception:
                pass
        
        # Fallback to step-based progress
        if self.total_steps <= 1:
            return 1.0
        return step / (self.total_steps - 1)
    
    def _get_or_create_noise(self, shape, device, dtype):
        """Get cached noise or create new noise."""
        if self._cached_noise is None or self._cached_noise.shape != shape:
            if self.generator is not None:
                self._cached_noise = torch.randn(shape, generator=self.generator, device=device, dtype=dtype)
            else:
                self._cached_noise = torch.randn(shape, device=device, dtype=dtype)
        return self._cached_noise.to(device=device, dtype=dtype)
    
    def __call__(
        self,
        pipe,
        step: int,
        timestep: torch.Tensor,
        callback_kwargs: dict
    ) -> dict:
        """
        Callback invoked at each denoising step.
        
        In later steps, we add noise to the latents to reduce the influence
        of image conditioning. This is mathematically equivalent to reducing
        the SNR of the conditioning signal, making the model rely more on
        text/prior conditioning.
        """
        latents = callback_kwargs.get("latents")
        
        if latents is None:
            return callback_kwargs
        
        # Compute denoising progress and conditioning strength
        denoise_progress = self._infer_denoise_progress(pipe, step, timestep)
        cond_strength = self._get_conditioning_strength(denoise_progress)
        
        # If full conditioning, no modification needed
        if cond_strength >= 0.99:
            return callback_kwargs
        
        device = latents.device
        dtype = latents.dtype
        
        # Only process if latents are in FLUX packed format [batch, seq, hidden]
        if latents.dim() != 3:
            return callback_kwargs
        
        batch_size, seq_len, hidden_dim = latents.shape
        
        # Infer diptych dimensions (width = 2 * height for diptych)
        # seq_len = (H/16) * (W/16) where W = 2*H
        # so seq_len = (H/16) * (2H/16) = 2 * (H/16)^2
        h_patches = int(math.sqrt(seq_len / 2))
        w_patches = 2 * h_patches
        
        if h_patches * w_patches != seq_len:
            # Can't determine dimensions, skip
            return callback_kwargs
        
        try:
            # Compute noise amount: more noise = less conditioning
            # noise_amount = 0 means full conditioning, 1 means pure noise
            noise_amount = 1.0 - cond_strength
            
            # Generate noise matching latent shape
            noise = self._get_or_create_noise(latents.shape, device, dtype)
            
            # Add noise to reduce conditioning influence
            # This is like adding a small epsilon to make the model "forget" some 
            # of the image conditioning, allowing text conditioning to dominate
            # 
            # The formula: latents_new = cond_strength * latents + noise_amount * noise * scale
            # where scale controls the noise magnitude
            noise_scale = 0.1  # Small scale to avoid destroying structure
            perturbed_latents = latents + noise_amount * noise_scale * noise
            
            callback_kwargs["latents"] = perturbed_latents
            
            # Log progress (optional, can be removed)
            if step % 10 == 0:
                print(f"[CondFreedom] Step {step}: progress={denoise_progress:.2f}, "
                      f"cond_strength={cond_strength:.2f}")
            
        except Exception as e:
            print(f"[CondFreedom] Error at step {step}: {e}")
            # Return original on error
            
        return callback_kwargs


class CombinedCallback:
    """
    Combines multiple callbacks into one.
    
    This allows using both LatentBlendCallback and ConditioningFreedomCallback
    together during generation.
    """
    
    def __init__(self, callbacks: List):
        """
        Args:
            callbacks: List of callback objects with __call__ method
        """
        self.callbacks = [cb for cb in callbacks if cb is not None]
    
    def __call__(
        self,
        pipe,
        step: int,
        timestep: torch.Tensor,
        callback_kwargs: dict
    ) -> dict:
        """Apply all callbacks in sequence."""
        for callback in self.callbacks:
            callback_kwargs = callback(pipe, step, timestep, callback_kwargs)
        return callback_kwargs


# ================================
# Main Pipeline
# ================================

class AdvancedHairTransferPipeline:
    """Advanced pipeline for hair transfer combining InsertAnything and RF-Inversion."""
    
    def __init__(
        self,
        config: Optional[HairTransferConfig] = None,
        device: str = "cuda"
    ):
        if config is None:
            config = HairTransferConfig()
        
        self.config = config
        self.device = device
        self.dtype = torch.bfloat16
        self.size = (config.PROCESSING_RESOLUTION, config.PROCESSING_RESOLUTION)
        
        self._load_models()
    
    def _load_models(self):
        """Load and initialize all required models."""
        print("Loading FLUX Fill pipeline...")
        self.pipe = FluxFillPipeline.from_pretrained(
            self.config.FLUX_FILL_MODEL,
            torch_dtype=self.dtype
        ).to(self.device)
        
        print(f"Loading LoRA weights from {self.config.LORA_WEIGHTS_PATH}...")
        if os.path.exists(self.config.LORA_WEIGHTS_PATH):
            self.pipe.load_lora_weights(self.config.LORA_WEIGHTS_PATH, adapter_name="insert_anything")
            # Set LoRA scale - lower values reduce InsertAnything's diptych-copy behavior,
            # allowing more influence from Redux semantic embeddings
            self.pipe.set_adapters(["insert_anything"], adapter_weights=[self.config.LORA_SCALE])
            print(f"  LoRA scale set to {self.config.LORA_SCALE}")
        else:
            print(f"Warning: LoRA weights not found at {self.config.LORA_WEIGHTS_PATH}")
        
        print("Loading FLUX Redux pipeline...")
        self.redux = FluxPriorReduxPipeline.from_pretrained(
            self.config.FLUX_REDUX_MODEL,
            # text_encoder=self.pipe.text_encoder,
            # text_encoder_2=self.pipe.text_encoder_2,
            # tokenizer=self.pipe.tokenizer,
            # tokenizer_2=self.pipe.tokenizer_2,
            torch_dtype=self.dtype,
        ).to(self.device)
        
        # Initialize latent inverter with matching dtype
        self.inverter = LatentInverter(self.pipe, self.device, self.dtype)
        
        print("Pipeline initialized successfully!")

    @contextmanager
    def _override_config(self, **overrides: Any):
        """Temporarily override config values for a single transfer pass."""
        if not overrides:
            yield
            return
        original = self.config
        self.config = replace(self.config, **overrides)
        try:
            yield
        finally:
            self.config = original
    
    def _prepare_reference_image(
        self,
        reference_image: np.ndarray,
        reference_mask: np.ndarray
    ) -> np.ndarray:
        """
        Prepare the reference hair image for encoding.
        
        Extracts the hair region, removes background, and centers it.
        """
        # Ensure binary mask
        ref_mask = (reference_mask > 0.5).astype(np.uint8)
        
        # Get bounding box
        ref_box = get_bbox_from_mask(ref_mask)
        
        # Create masked reference (white background)
        ref_mask_3 = np.stack([ref_mask] * 3, axis=-1)
        masked_ref = reference_image * ref_mask_3 + 255 * (1 - ref_mask_3)
        
        # Crop to bounding box
        y1, y2, x1, x2 = ref_box
        masked_ref = masked_ref[y1:y2, x1:x2, :]
        ref_mask_crop = ref_mask[y1:y2, x1:x2]
        
        # Expand with padding
        masked_ref, _ = expand_image_mask(
            masked_ref,
            ref_mask_crop,
            ratio=self.config.REF_EXPAND_RATIO
        )
        
        # Pad to square
        masked_ref = pad_to_square(masked_ref, pad_value=255, random_pad=False)
        
        return masked_ref.astype(np.uint8)
    
    def _prepare_target_region(
        self,
        source_image: np.ndarray,
        target_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Tuple, Tuple]:
        """
        Prepare the target region for generation.
        
        Crops to the hair region with context and returns metadata
        for compositing back.
        """
        # Dilate mask to ensure coverage
        kernel = np.ones(
            (self.config.HAIR_MASK_DILATION_KERNEL, self.config.HAIR_MASK_DILATION_KERNEL),
            np.uint8
        )
        tar_mask = cv2.dilate(
            target_mask.astype(np.uint8),
            kernel,
            iterations=self.config.HAIR_MASK_DILATION_ITERATIONS
        )
        
        # Compute crop region
        tar_box = get_bbox_from_mask(tar_mask)
        tar_box_expanded = expand_bbox(tar_mask, tar_box, ratio=self.config.BBOX_EXPAND_RATIO)
        tar_box_crop = expand_bbox(source_image, tar_box_expanded, ratio=self.config.CROP_EXPAND_RATIO)
        tar_box_crop = box2square(source_image, tar_box_crop)
        
        # Crop
        y1, y2, x1, x2 = tar_box_crop
        cropped_image = source_image[y1:y2, x1:x2, :].copy()
        cropped_mask = tar_mask[y1:y2, x1:x2].copy()
        
        # Get dimensions for restoration
        H1, W1 = cropped_image.shape[:2]
        
        # Pad to square
        cropped_mask = pad_to_square(cropped_mask, pad_value=0)
        cropped_image = pad_to_square(cropped_image, pad_value=255)
        
        H2, W2 = cropped_image.shape[:2]
        
        return cropped_image, cropped_mask, (H1, W1, H2, W2), tar_box_crop
    
    def _create_diptych_inputs(
        self,
        reference_prepared: np.ndarray,
        target_image: np.ndarray,
        target_mask: np.ndarray
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Create diptych format inputs for InsertAnything.
        
        The diptych concatenates reference (left) and target (right) horizontally.
        """
        # Resize to processing resolution
        ref_resized = cv2.resize(reference_prepared, self.size)
        tar_resized = cv2.resize(target_image, self.size)
        mask_resized = cv2.resize(target_mask, self.size)
        
        # # Apply Gaussian blur to smooth mask edges and avoid hard boundaries
        # blur_radius = self.config.DIPTYCH_MASK_BLUR_RADIUS
        # if blur_radius > 0:
        #     # Ensure blur radius is odd
        #     if blur_radius % 2 == 0:
        #         blur_radius += 1
        #     mask_resized = cv2.GaussianBlur(
        #         mask_resized.astype(np.float32), 
        #         (blur_radius, blur_radius), 
        #         0
        #     )
        #     # Normalize back to 0-1 range
        #     mask_resized = np.clip(mask_resized, 0, 1)
        
        # Create diptych image
        diptych_image = np.concatenate([ref_resized, tar_resized], axis=1)
        
        # Create diptych mask (black on left, mask on right)
        mask_black = np.zeros_like(tar_resized)
        mask_3ch = np.stack([mask_resized] * 3, axis=-1) * 255
        diptych_mask = np.concatenate([mask_black, mask_3ch], axis=1)
        
        return Image.fromarray(diptych_image), Image.fromarray(diptych_mask.astype(np.uint8))
    def _encode_text_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        max_sequence_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a text prompt using FLUX's T5 and CLIP text encoders.
        
        In FLUX architecture:
        - prompt (first) -> CLIP text encoder -> pooled_prompt_embeds [1, 768]
        - prompt_2 (second) -> T5 text encoder -> prompt_embeds [1, seq_len, 4096]
        
        For hair description, we want the detailed text to go to T5 (prompt_2)
        since T5 handles the rich semantic content, while CLIP provides global context.
        
        Args:
            prompt: Text prompt to encode (goes to T5 via prompt_2)
            max_sequence_length: Maximum sequence length for T5 encoder
            
        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds)
            - prompt_embeds: T5 embeddings [1, seq_len, 4096]
            - pooled_prompt_embeds: CLIP pooled embeddings [1, 768]
        """
        # Use FluxFillPipeline's encode_prompt method
        # prompt_2 -> T5 encoder (main sequence embeddings for cross-attention)
        # prompt -> CLIP encoder (pooled embeddings for global conditioning)
        prompt_embeds, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
            prompt=prompt,       # CLIP: provides pooled embeddings
            prompt_2=prompt_2,     # T5: provides sequence embeddings (hair description)
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        return prompt_embeds, pooled_prompt_embeds

    def _create_dual_conditioned_embeds(
        self,
        redux_image: Image.Image,
        text_prompt: Optional[str] = None,
        text_prompt2: Optional[str] = None,
        text_weight: float = 1.0,
        image_weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Create dual-conditioned embeddings combining Redux image features and text prompt.
        
        This follows the InsertAnything architecture where:
        - prompt_embeds = [text_embeds (512 tokens), image_embeds (from Redux)]
        - pooled_prompt_embeds = CLIP text pooled embeddings
        
        The InsertAnything paper uses Redux for visual feature extraction and concatenates
        these with text embeddings for the transformer's cross-attention.
        
        Args:
            redux_image: PIL Image for Redux encoding (hair reference)
            text_prompt: Optional text description of the hair
            text_weight: Weight for text embeddings (default 1.0)
            image_weight: Weight for image embeddings (default 1.0)
            
        Returns:
            Dict with 'prompt_embeds' and 'pooled_prompt_embeds' for the pipeline
        """
        batch_size = 1
        max_seq_len = self.config.MAX_SEQUENCE_LENGTH  # 512
        
        # ============================================
        # Step 1: Get Redux image embeddings
        # ============================================
        # Redux encodes the image and returns embeddings that capture visual features
        # The image_embedder projects CLIP image features to the text embedding space
        image_latents = self.redux.encode_image(redux_image, self.device, 1)
        image_embeds = self.redux.image_embedder(image_latents).image_embeds
        image_embeds = image_embeds.to(device=self.device, dtype=self.dtype)
        # image_embeds shape: [1, num_image_tokens, 4096]
        
        # ============================================
        # Step 2: Get text embeddings (or zeros if no prompt)
        # ============================================
        if text_prompt and text_prompt.strip():
            # Encode text prompt using T5 and CLIP
            text_embeds, pooled_text_embeds = self._encode_text_prompt(
                text_prompt,
                text_prompt2,
                max_sequence_length=max_seq_len,
            )
            # text_embeds shape: [1, seq_len, 4096] (padded to max_seq_len)
            # pooled_text_embeds shape: [1, 768]
            
            # Ensure text_embeds is exactly max_seq_len tokens
            if text_embeds.shape[1] < max_seq_len:
                # Pad with zeros if shorter
                padding = torch.zeros(
                    (batch_size, max_seq_len - text_embeds.shape[1], 4096),
                    device=self.device,
                    dtype=self.dtype,
                )
                text_embeds = torch.cat([text_embeds, padding], dim=1)
            elif text_embeds.shape[1] > max_seq_len:
                # Truncate if longer
                text_embeds = text_embeds[:, :max_seq_len, :]
            
            # Apply text weight
            text_embeds = text_embeds * text_weight
            pooled_prompt_embeds = pooled_text_embeds * text_weight
        else:
            # No text prompt - use zeros (original InsertAnything behavior)
            text_embeds = torch.zeros(
                (batch_size, max_seq_len, 4096),
                device=self.device,
                dtype=self.dtype,
            )
            pooled_prompt_embeds = torch.zeros(
                (batch_size, 768),
                device=self.device,
                dtype=self.dtype,
            )
        
        # ============================================
        # Step 3: Apply image weight and concatenate
        # ============================================
        # Following InsertAnything's image_project.py:
        # prompt_embeds = [text_embeds, image_embeds] concatenated along sequence dimension
        image_embeds = image_embeds * image_weight
        prompt_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        
        # ============================================
        # Step 4: Return in pipeline-compatible format
        # ============================================
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
    
    def transfer_hair(
        self,
        source_image_path: str,
        reference_image_path: str,
        reference_mask_path: Optional[str] = None,
        target_mask_path: Optional[str] = None,
        output_path: str = "output.png",
        hair_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        redux_image_path: Optional[str] = None,
        redux_mask_path: Optional[str] = None,
    ) -> str:
        """
        Perform hair transfer from reference to source image.
        
        Args:
            source_image_path: Path to source (bald) image
            reference_image_path: Path to reference (hair donor) image
            reference_mask_path: Path to reference hair mask (optional, will auto-compute with SAM if not provided)
            target_mask_path: Path to target hair mask (optional, will auto-compute with SAM if not provided)
            output_path: Path to save the result
            hair_prompt: Text description of hair attributes
            seed: Random seed for reproducibility
            redux_image_path: Path to matted reference image for Redux encoding (optional, uses reference_image if not provided)
            redux_mask_path: Path to hair mask for the Redux image (optional, uses reference_mask if not provided)
        
        Returns:
            Path to the saved output image
        """
        if seed is None:
            seed = self.config.SEED
        
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator(self.device).manual_seed(seed)
        
        print(f"Processing with seed={seed}")
        
        # ============================================
        # Phase 1: Load and Prepare Images
        # ============================================
        print("Phase 1: Loading and preparing images...")
        
        # Load images
        source_pil = Image.open(source_image_path).convert("RGB").resize(
            self.size, Image.Resampling.LANCZOS
        )
        reference_pil = Image.open(reference_image_path).convert("RGB").resize(
            self.size, Image.Resampling.LANCZOS
        )
        
        # Load or compute reference mask
        if reference_mask_path and os.path.exists(reference_mask_path):
            ref_mask_pil = Image.open(reference_mask_path).convert("L").resize(
                self.size, Image.Resampling.NEAREST
            )
        else:
            print(f"Reference mask not found at {reference_mask_path}, computing using SAMMaskExtractor (singleton)...")
            if not SAMMaskExtractorSingleton.is_available():
                raise RuntimeError(
                    "SAMMaskExtractor unavailable; provide a valid --ref_mask path or ensure SAMMaskExtractor import works."
                )
            ref_mask_pil, ref_score = extract_hair_mask(reference_pil, confidence_threshold=0.4, detection_threshold=0.5, prompt="hair")
            print(f"SAM mask extraction score (reference): {ref_score:.3f}")
            # Optionally save the computed mask for future use
            if reference_mask_path:
                os.makedirs(os.path.dirname(reference_mask_path), exist_ok=True)
                ref_mask_pil.save(reference_mask_path)
                print(f"Saved computed reference mask to {reference_mask_path}")
            ref_mask_pil = ref_mask_pil.resize(self.size, Image.Resampling.NEAREST)
        
        # Load or compute target mask
        if target_mask_path and os.path.exists(target_mask_path):
            tar_mask_pil = Image.open(target_mask_path).convert("L").resize(
                self.size, Image.Resampling.NEAREST
            )
        else:
            tar_mask_pil = ref_mask_pil.copy()
            # print(f"Target mask not found at {target_mask_path}, computing using SAMMaskExtractor...")
            # if SAMMaskExtractor is None:
            #     raise RuntimeError(
            #         "SAMMaskExtractor unavailable; provide a valid --tar_mask path or ensure SAMMaskExtractor import works."
            #     )
            # tar_mask_pil, tar_score = extract_hair_mask(reference_pil, confidence_threshold=0.4, detection_threshold=0.4, prompt="hair")
            
            # print(f"SAM mask extraction score (target): {tar_score:.3f}")
            # # Optionally save the computed mask for future use
            # if target_mask_path:
            #     os.makedirs(os.path.dirname(target_mask_path), exist_ok=True)
            #     tar_mask_pil.save(target_mask_path)
            #     print(f"Saved computed target mask to {target_mask_path}")
            # tar_mask_pil = tar_mask_pil.resize(self.size, Image.Resampling.NEAREST)
        
        # Convert to numpy
        source_np = np.asarray(source_pil)
        reference_np = np.asarray(reference_pil)
        ref_mask_np = (np.asarray(ref_mask_pil) > 128).astype(np.uint8)
        tar_mask_np = (np.asarray(tar_mask_pil) > 128).astype(np.uint8)
        
        # Mask the reference image by the target mask - keep masked pixels, make non-masked white
        tar_mask_3ch = np.stack([tar_mask_np] * 3, axis=-1)
        reference_np = reference_np * tar_mask_3ch + 255 * (1 - tar_mask_3ch)
        reference_np = reference_np.astype(np.uint8)
        
        # Save original source for final composition
        original_source = source_np.copy()
        
        # Prepare reference hair
        masked_ref = self._prepare_reference_image(reference_np, ref_mask_np)
        
        # Prepare target region
        target_image, target_mask, extra_sizes, tar_box_crop = self._prepare_target_region(
            source_np, tar_mask_np
        )
        
        # ============================================
        # Phase 2: Reference Encoding with Redux
        # ============================================
        print("Phase 2: Encoding reference hair attributes...")
        
        # Prepare Redux input image
        print(redux_image_path)
        redux_pil = Image.open(redux_image_path).convert("RGB").resize(
                self.size, Image.Resampling.LANCZOS
            )
        redux_np = np.asarray(redux_pil)
        # If redux_image_path is provided, use the matted reference image instead of the warped reference
        redux_input = cv2.resize(redux_np, self.size)
        # Resize for pipeline (for diptych - still uses warped reference)
        masked_ref_resized = cv2.resize(masked_ref, self.size)
        
        # Build conditioning prompt
        if hair_prompt is None:
            hair_prompt = "detailed hair, high quality, realistic, natural lighting"
        
        base_prompt = "Add hair with strand-level details. Connect the hair naturally to the scalp and face."
        
        # Get Redux embeddings using the Redux input (matted reference or warped reference)
        pipe_prior_output = self._create_dual_conditioned_embeds(
            redux_image=Image.fromarray(redux_input),
            text_prompt=base_prompt,
            text_prompt2=hair_prompt,
            text_weight=1.0,
            image_weight=1.0,
        )
        
        # ============================================
        # Phase 3: Identity-Preserving Generation
        # ============================================
        print("Phase 3: Generating with identity preservation...")
        
        # Create diptych inputs
        target_crop_resized = cv2.resize(target_image, self.size)
        target_mask_resized = cv2.resize(target_mask, self.size)
        
        diptych_image, diptych_mask = self._create_diptych_inputs(
            masked_ref_resized,
            target_crop_resized,
            target_mask_resized
        )
        
        # ============================================
        # Latent-Space Blending Setup (if enabled)
        # ============================================
        # 
        # This implements the key insight from "Blended Diffusion" (Avrahami et al., 2022):
        # Instead of blending pixels after generation, we blend latents during denoising.
        # 
        # Benefits for hair transfer:
        # 1. Model has full freedom in hair region (InsertAnything controls hair style)
        # 2. Non-hair regions naturally preserve source identity
        # 3. Diffusion process harmonizes the boundary automatically
        # 4. No hard edges or compositing artifacts
        #
        # The mask interpretation:
        # - mask=1: Hair region - let InsertAnything generate freely
        # - mask=0: Non-hair region - guide towards source preservation
        
        callback = None
        latent_blend_callback = None
        cond_freedom_callback = None
        
        if self.config.LATENT_BLEND:
            print("  Setting up latent-space blending...")
            
            # Encode the target crop (source region) to latent space
            # This will be used to guide non-hair regions during denoising
            target_crop_pil = Image.fromarray(target_crop_resized)
            source_latents = self.inverter.encode_image(target_crop_pil)
            
            # Create soft blend mask for latent space
            # Use TIGHT mask - minimal dilation to avoid halo artifacts
            # The diffusion model will naturally harmonize boundaries
            blend_mask_np = create_distance_soft_blend_mask(
                target_mask_resized,
                dilation_px=2,  # roughly matches a 5x5 dilation kernel
                dilation_iterations=1,
                feather_px=8,  # tight transition band to reduce halos
            )
            
            # Convert mask to tensor
            blend_mask_tensor = torch.from_numpy(blend_mask_np).float()
            
            # Create the latent blending callback with FLUX-compatible settings
            # Pass image dimensions for proper latent unpacking
            latent_blend_callback = LatentBlendCallback(
                source_latents=source_latents,
                blend_mask=blend_mask_tensor,
                start_step_ratio=self.config.LATENT_BLEND_START_STEP,
                blend_strength=self.config.LATENT_BLEND_STRENGTH,
                schedule=self.config.LATENT_BLEND_SCHEDULE,
                total_steps=self.config.NUM_INFERENCE_STEPS,
                generator=generator,  # Pass generator for reproducible noise
                image_height=self.size[1],  # Height of single target image
                image_width=self.size[0],   # Width of single target image
            )
            print(f"  Latent blend (FLUX-compatible): start={self.config.LATENT_BLEND_START_STEP:.1%}, "
                  f"strength={self.config.LATENT_BLEND_STRENGTH:.1%}, "
                  f"schedule={self.config.LATENT_BLEND_SCHEDULE}")
        
        # ============================================
        # Conditioning Freedom Setup (if enabled)
        # ============================================
        # 
        # This reduces image conditioning influence in later denoising steps,
        # allowing the model more freedom to naturally integrate hair with the scalp.
        # Without this, the hair can look "pasted" - not naturally connected to the head.
        #
        if self.config.COND_FREEDOM_ENABLED:
            print("  Setting up conditioning freedom for natural integration...")
            
            cond_freedom_callback = ConditioningFreedomCallback(
                start_ratio=self.config.COND_FREEDOM_START_RATIO,
                end_strength=self.config.COND_FREEDOM_END_STRENGTH,
                schedule=self.config.COND_FREEDOM_SCHEDULE,
                total_steps=self.config.NUM_INFERENCE_STEPS,
                image_height=self.size[1],
                image_width=self.size[0],
                generator=generator,
            )
            print(f"  Conditioning freedom: start={self.config.COND_FREEDOM_START_RATIO:.1%}, "
                  f"end_strength={self.config.COND_FREEDOM_END_STRENGTH:.1%}, "
                  f"schedule={self.config.COND_FREEDOM_SCHEDULE}")
        
        # Combine callbacks if both are enabled
        if latent_blend_callback and cond_freedom_callback:
            callback = CombinedCallback([latent_blend_callback, cond_freedom_callback])
        elif latent_blend_callback:
            callback = latent_blend_callback
        elif cond_freedom_callback:
            callback = cond_freedom_callback
        
        # Save FLUX Fill inputs for debugging/visualization
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            diptych_image.save(os.path.join(output_dir, "flux_fill_input_diptych.png"))
            # diptych_mask.save(os.path.join(output_dir, "flux_fill_input_mask.png"))
            # Also save the Redux input image
            Image.fromarray(redux_input).save(os.path.join(output_dir, "flux_redux_input.png"))
            print(f"  Savediptych_imaged FLUX Fill inputs to {output_dir}")
        # Run generation with InsertAnything
        # The callback (if enabled) will blend latents during denoising
        edited_image = self.pipe(
            image=diptych_image,
            mask_image=diptych_mask,
            height=diptych_mask.size[1],
            width=diptych_mask.size[0],
            max_sequence_length=self.config.MAX_SEQUENCE_LENGTH,
            generator=generator,
            guidance_scale=self.config.GUIDANCE_SCALE,
            num_inference_steps=self.config.NUM_INFERENCE_STEPS,
            # callback_on_step_end=callback,
            # callback_on_step_end_tensor_inputs=["latents"] if callback else None,
            **pipe_prior_output,
        ).images[0]
        
        # Extract right half (generated region)
        width, height = edited_image.size
        generated_crop = edited_image.crop((width // 2, 0, width, height))
        generated_np = np.array(generated_crop)
        
        # ============================================
        # Phase 4: Post-Generation Blending (Optional)
        # ============================================
        # When using latent blending, pixel blending is less critical but can still help
        # smooth any remaining boundary artifacts
        
        # Get source crop at same resolution
        source_crop = target_crop_resized
    
        # ============================================
        # Phase 5: Composite Back to Original
        # ============================================
        print("Phase 5: Compositing to original image...")
        blended = generated_np
        # Crop back to original image
        final_result = crop_back(
            blended,
            original_source.copy(),
            np.array(extra_sizes),
            np.array(tar_box_crop)
        )
        
        # Convert to PIL and resize to output resolution
        result_pil = Image.fromarray(final_result)
        result_pil = result_pil.resize(
            (self.config.OUTPUT_RESOLUTION, self.config.OUTPUT_RESOLUTION),
            Image.Resampling.LANCZOS
        )
        
        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_pil.save(output_path)
        print(f"Output saved to: {output_path}")
        
        return output_path
    
    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        if hasattr(self, 'pipe'):
            del self.pipe
        if hasattr(self, 'redux'):
            del self.redux
        flush()


def _derive_second_pass_paths(
    first_output_path: Union[str, Path],
    config: HairTransferConfig,
) -> Tuple[Path, Path, Path]:
    """Derive mask/final output paths for the second pass."""
    first_output_path = Path(first_output_path)
    output_dir = first_output_path.parent
    if first_output_path.name == config.FILE_OUTPUT:
        mask_path = output_dir / config.FILE_OUTPUT_MASK
        final_path = output_dir / config.FILE_OUTPUT_FINAL
        final_mask_path = output_dir / config.FILE_OUTPUT_FINAL_MASK
    else:
        mask_path = output_dir / f"{first_output_path.stem}_mask{first_output_path.suffix}"
        final_path = output_dir / f"{first_output_path.stem}_final{first_output_path.suffix}"
        final_mask_path = output_dir / f"{final_path.stem}_mask{final_path.suffix}"
    return mask_path, final_path, final_mask_path


def _compute_hair_mask(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    *,
    confidence_threshold: float = 0.4,
    detection_threshold: float = 0.5,
    prompt: str = "hair",
) -> float:
    """Compute and save a hair mask from an image using SAM (uses singleton)."""
    if not SAMMaskExtractorSingleton.is_available():
        raise RuntimeError(
            "SAMMaskExtractor unavailable; cannot compute hair mask for second pass."
        )
    image_path = Path(image_path)
    mask_path = Path(mask_path)
    image = Image.open(image_path).convert("RGB")
    hair_mask_pil, score = extract_hair_mask(
        image,
        confidence_threshold=confidence_threshold,
        detection_threshold=detection_threshold,
        prompt=prompt,
    )
    os.makedirs(mask_path.parent, exist_ok=True)
    hair_mask_pil.save(mask_path)
    return score


def _composite_with_bald(
    hair_image_path: Union[str, Path],
    hair_mask_path: Union[str, Path],
    bald_image_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    use_multiscale: bool = True,
) -> None:
    """Composite hair from a generated image onto the bald source image."""
    hair_image = Image.open(hair_image_path).convert("RGB")
    hair_mask = Image.open(hair_mask_path).convert("L")
    bald_image = Image.open(bald_image_path).convert("RGB")

    if bald_image.size != hair_image.size:
        bald_image = bald_image.resize(hair_image.size, Image.Resampling.LANCZOS)
    if hair_mask.size != hair_image.size:
        hair_mask = hair_mask.resize(hair_image.size, Image.Resampling.NEAREST)

    hair_np = np.asarray(hair_image)
    bald_np = np.asarray(bald_image)
    mask_np = np.asarray(hair_mask)

    composited = composite_hair_onto_bald(
        hair_np,
        bald_np,
        mask_np,
        use_multiscale=use_multiscale,
        feather_px=2,
    )

    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    Image.fromarray(composited).save(output_path)


# ================================
# Output Filename Helpers
# ================================

def build_output_filename(
    config: HairTransferConfig,
    bald_version: str,
    is_3d_aware: bool,
    suffix: str = "",
) -> str:
    """Build output filename based on configuration.
    
    NOTE: This function is kept for backward compatibility.
    The new output structure uses simple filenames (hair_restored.png)
    and encodes settings in the directory path instead.
    
    Args:
        config: HairTransferConfig instance
        bald_version: Version of bald image (w_seg or wo_seg)
        is_3d_aware: Whether aligned image is actually used
        suffix: Additional suffix (e.g., '_mask', '_final')
    
    Returns:
        Filename like 'hair_restored_transferred_fill_w_seg_3d_aware.png'
    """
    parts = [config.FILE_OUTPUT_PREFIX]
    parts.append(bald_version)
    parts.append("3d_aware" if is_3d_aware else "3d_unaware")
    return "hair_restored_" + "_".join(parts) + suffix + ".png"


def get_output_dir(
    pair_dir: Path,
    config: HairTransferConfig,
    use_3d_aware: bool,
) -> Path:
    """Get output directory path based on conditioning mode.
    
    New structure: {pair_dir}/{3d_aware|3d_unaware}/transferred/
    
    Args:
        pair_dir: Path to the pair directory (e.g., .../w_seg/)
        config: HairTransferConfig instance
        use_3d_aware: Whether 3D aware mode is used
    
    Returns:
        Path to output directory
    """
    mode_dir = config.DIR_3D_AWARE if use_3d_aware else config.DIR_3D_UNAWARE
    return pair_dir / mode_dir / config.SUBDIR_TRANSFERRED


# ================================
# Batch Processing
# ================================

def process_sample(
    folder: Path,
    pipeline: "AdvancedHairTransferPipeline",
    data_dir: Path,
    config: HairTransferConfig,
    bald_version: str,
    conditioning_mode: str,
    skip_existing: bool = True,
) -> bool:
    """Process a single sample folder for a given bald_version and conditioning_mode."""
    folder_name = folder.name
    
    # Parse folder name
    if "_to_" not in folder_name:
        print(f"Skipping {folder_name}: invalid format")
        return False
    
    try:
        target_id, source_id = folder_name.split("_to_")
    except ValueError:
        print(f"Invalid directory name format: {folder_name}")
        return False
    
    # Pair directory for this bald version
    pair_dir = folder / bald_version
    if not pair_dir.exists():
        print(f"Pair directory not found: {pair_dir}, skipping...")
        return False
    
    # Check for warped outputs from view_blender (new pipeline)
    # Select mode directory and warping directory based on conditioning_mode
    mode_dir = config.DIR_3D_AWARE if conditioning_mode == "3d_aware" else config.DIR_3D_UNAWARE
    warping_dir = pair_dir / mode_dir / config.SUBDIR_WARPING
    
    warped_image_path = warping_dir / config.FILE_WARPED_TARGET_IMAGE
    warped_hair_mask_path = warping_dir / config.FILE_WARPED_HAIR_MASK
    has_warped_outputs = warped_image_path.exists() and warped_hair_mask_path.exists()
    
    # Determine 3D lifting status (for alignment folder, used for view-aligned image)
    camera_params_path = pair_dir / config.FILE_CAMERA_PARAMS
    alignment_dir = pair_dir / config.DIR_ALIGNMENT
    view_aligned_image_path = alignment_dir / config.FILE_VIEW_ALIGNED_IMAGE
    lift_3d_applied = camera_params_path.exists()
    aligned_image_exists = lift_3d_applied and view_aligned_image_path.exists()
    
    # Determine processing mode based on conditioning_mode
    if conditioning_mode == "3d_aware":
        # 3D aware mode: use warped outputs if available, otherwise fall back to aligned image
        use_3d_aware = has_warped_outputs or aligned_image_exists
        if not use_3d_aware:
            reason = "no warped outputs and no aligned image"
            print(f"No 3D lifting for {folder_name}/{bald_version} ({reason}), skipping 3d_aware mode")
            return False
    elif conditioning_mode == "3d_unaware":
        use_3d_aware = False
    else:
        raise ValueError(f"Invalid conditioning_mode: {conditioning_mode}")
    
    # Build output paths using new directory structure:
    # {pair_dir}/{3d_aware|3d_unaware}/transferred/hair_restored.png
    output_dir = get_output_dir(pair_dir, config, use_3d_aware)
    output_path = output_dir / config.FILE_HAIR_RESTORED
    output_mask_path = output_dir / config.FILE_HAIR_RESTORED_MASK
    output_final_path = output_dir / config.FILE_HAIR_RESTORED_FINAL
    output_final_mask_path = output_dir / config.FILE_HAIR_RESTORED_FINAL_MASK
    
    # Skip if output exists
    if skip_existing and output_path.exists():
        print(f"Output already exists for {folder_name}/{bald_version}/{conditioning_mode}, skipping...")
        return True
    
    try:
        # Resolve source image path (prefer outpainted, fallback to bald)
        outpainted_source_path = pair_dir / config.DIR_SOURCE_OUTPAINTED / config.FILE_OUTPAINTED_IMAGE
        resize_info_path = pair_dir / config.DIR_SOURCE_OUTPAINTED / config.FILE_RESIZE_INFO
        
        if outpainted_source_path.exists():
            source_image_path = outpainted_source_path
            resize_info = resize_info_path if resize_info_path.exists() else None
        else:
            bald_image_dir = data_dir / config.DIR_BALD / bald_version / config.SUBDIR_BALD_IMAGE
            source_image_path = bald_image_dir / f"{source_id}.png"
            resize_info = None
        
        # Fallback to image directory if bald not found
        if not source_image_path.exists():
            alternative_source_path = data_dir / "image" / f"{source_id}.png"
            if alternative_source_path.exists():
                source_image_path = alternative_source_path
            else:
                print(f"  Warning: Source not found: {source_image_path}")
                return False
        
        # Reference image and masks - prefer warped outputs from view_blender
        if has_warped_outputs:
            # Use warped image as reference, warped mask as both reference and target mask
            # This works for BOTH 3d_aware and 3d_unaware modes now since both have warped outputs
            reference_image = warped_image_path
            reference_mask = warped_hair_mask_path
            target_mask = warped_hair_mask_path  # Same mask for both
            mode_desc = "3D aware" if use_3d_aware else "3D unaware"
            print(f"  Using warped outputs from view_blender ({mode_desc})")
        elif use_3d_aware and aligned_image_exists:
            # Fallback to view-aligned image (legacy path for 3D aware mode)
            reference_image = view_aligned_image_path
            reference_mask = reference_image.parent / (reference_image.stem + "_hair_mask.png")
            if not reference_mask.exists():
                # Try to compute hair mask
                ref_pil = Image.open(reference_image).convert("RGB")
                ref_mask_pil, score = extract_hair_mask(ref_pil, confidence_threshold=0.4, detection_threshold=0.5, prompt="hair")
                os.makedirs(reference_mask.parent, exist_ok=True)
                ref_mask_pil.save(reference_mask)
                print(f"  Computed and saved reference hair mask (SAM score: {score:.3f})")
            target_mask = reference_mask
            print(f"  Using view-aligned image (legacy path)")
        else:
            # Use matted image as reference (fallback for 3D unaware mode without warped outputs)
            reference_image = data_dir / config.MATTED_IMAGE_SUBDIR / f"{target_id}.png"
            reference_mask = data_dir / config.MATTED_IMAGE_HAIR_MASK_SUBDIR / f"{target_id}.png"
            if not reference_mask.exists():
                # Try to compute hair mask
                ref_pil = Image.open(reference_image).convert("RGB")
                ref_mask_pil, score = extract_hair_mask(ref_pil, confidence_threshold=0.4, detection_threshold=0.5, prompt="hair")
                os.makedirs(reference_mask.parent, exist_ok=True)
                ref_mask_pil.save(reference_mask)
                print(f"  Computed and saved reference hair mask (SAM score: {score:.3f})")
            target_mask = reference_mask
            print(f"  Using matted image (3D unaware fallback - no warped outputs)")
        
        if not reference_image.exists():
            print(f"  Warning: Reference image not found: {reference_image}")
            return False
        
        if not reference_mask.exists():
            print(f"  Warning: Reference mask not found: {reference_mask}")
            return False
        
        # Redux image: try hair_aligned_image first, then fall back to image directory
        # Always apply background remover and compute hair mask with SAM
        redux_image_path = None
        hair_aligned_image_path = data_dir / config.DIR_HAIR_ALIGNED_IMAGE / f"{target_id}.png"
        original_image_path = data_dir / config.DIR_IMAGE / f"{target_id}.png"
        
        if hair_aligned_image_path.exists():
            redux_image_path = hair_aligned_image_path
            print(f"  Using hair_aligned_image for Redux: {redux_image_path}")
        elif original_image_path.exists():
            redux_image_path = original_image_path
            print(f"  Using original image for Redux: {redux_image_path}")
        else:
            print(f"  Warning: No Redux source image found for {target_id}")
            redux_image_path = None
        
        # Process Redux image: apply background remover and compute hair mask
        redux_image_processed = None
        redux_hair_mask = None
        
        if redux_image_path is not None:
            try:
                # Load image and ensure RGB (3 channels)
                redux_raw = Image.open(redux_image_path).convert("RGB")
                redux_raw_np = np.array(redux_raw).astype(np.float32)
                
                # Ensure we have exactly 3 channels
                if redux_raw_np.ndim == 2:
                    redux_raw_np = np.stack([redux_raw_np] * 3, axis=-1)
                elif redux_raw_np.shape[-1] == 4:
                    redux_raw_np = redux_raw_np[:, :, :3]  # Drop alpha channel
                
                # Apply background remover to get foreground mask (using singleton)
                bg_remover = BackgroundRemoverSingleton.get_instance()
                _, fg_mask = bg_remover.remove_background(redux_raw)
                
                # Convert foreground mask to numpy and normalize
                fg_mask_np = np.array(fg_mask).astype(np.float32) / 255.0
                if fg_mask_np.ndim == 3:
                    fg_mask_np = fg_mask_np[:, :, 0]  # Take first channel if multi-channel
                
                # Whiten non-foreground pixels
                fg_mask_3ch = np.stack([fg_mask_np] * 3, axis=-1)
                redux_matted_np = (redux_raw_np * fg_mask_3ch + 255.0 * (1.0 - fg_mask_3ch)).astype(np.uint8)
                redux_image_processed = Image.fromarray(redux_matted_np).convert("RGB")
                print(f"  Applied background removal for Redux image")
                
                # Compute hair mask using SAM (using singleton)
                if SAMMaskExtractorSingleton.is_available():
                    redux_hair_mask_pil, sam_score = extract_hair_mask(
                        redux_image_processed,
                        confidence_threshold=0.4,
                        detection_threshold=0.5,
                        prompt="hair",
                    )
                    redux_hair_mask = redux_hair_mask_pil
                    print(f"  Computed Redux hair mask with SAM (score: {sam_score:.3f})")
                else:
                    print(f"  Warning: SAMMaskExtractor not available, skipping Redux hair mask")
                    
            except Exception as e:
                print(f"  Warning: Failed to process Redux image: {e}")
                redux_image_processed = None
                redux_hair_mask = None
        
        # Load hair prompt
        prompts_dir = data_dir / config.DIR_PROMPTS
        hair_prompt = None
        try:
            prompt_file = prompts_dir / f"{target_id}.json"
            if prompt_file.exists():
                with open(prompt_file, 'r') as f:
                    prompt_data = json.load(f)
                hair_prompt = prompt_data.get("subject", [{}])[0].get("hair_description")
                if hair_prompt:
                    print(f"  Using hair prompt: {hair_prompt[:50]}...")
        except Exception as e:
            print(f"  Warning: Could not load prompt: {e}")
        
        print(f"Source image path: {source_image_path}", flush=True)
        print(f"Reference image path: {reference_image}", flush=True)
        print(f"Reference mask path: {reference_mask}", flush=True)
        print(f"Target mask path: {target_mask}", flush=True)
        print(f"Has warped outputs: {has_warped_outputs}", flush=True)
        print(f"3D lifting applied: {lift_3d_applied}", flush=True)
        print(f"Using 3D aware: {use_3d_aware}", flush=True)
        if resize_info:
            print(f"Resize info path: {resize_info}", flush=True)
        
        # Save processed Redux image temporarily if needed
        # If Redux processing failed, fall back to the reference image
        redux_temp_path = None
        redux_mask_temp_path = None
        
        if redux_image_processed is not None:
            # Save temporarily for pipeline use, will be cleaned up
            redux_temp_path = output_dir / ".redux_input_processed.tmp.png"
            os.makedirs(output_dir, exist_ok=True)
            redux_image_processed.save(redux_temp_path)
        else:
            # Fallback: use reference image as Redux input
            print(f"  Warning: Redux processing failed, using reference image as fallback")
            redux_temp_path = reference_image
        
        # Run hair transfer
        pipeline.transfer_hair(
            source_image_path=str(source_image_path),
            reference_image_path=str(reference_image),
            reference_mask_path=str(reference_mask),
            target_mask_path=str(target_mask),
            output_path=str(output_path),
            hair_prompt=hair_prompt,
            redux_image_path=str(redux_temp_path) if redux_temp_path else None,
            redux_mask_path=str(redux_mask_temp_path) if redux_mask_temp_path else None,
        )
        
        # Clean up temporary Redux file
        temp_redux_file = output_dir / ".redux_input_processed.tmp.png"
        if temp_redux_file.exists():
            temp_redux_file.unlink()
        
        # Compute hair mask for output
        if output_path.exists():
            score = _compute_hair_mask(output_path, output_mask_path)
            print(f"  Saved hair_restored_mask.png (SAM score: {score:.3f})")
        
        return True
        
    except Exception as e:
        print(f"  Error processing {folder_name}: {e}")
        return False


def process_view_aligned_folders(
    data_dir: Union[str, Path],
    shape_provider: str = "hi3dgen",
    texture_provider: str = "mvadapter",
    config: Optional[HairTransferConfig] = None,
    skip_existing: bool = True,
    bald_version: str = "w_seg",
    conditioning_mode: str = "3d_aware",
) -> Dict[str, int]:
    """Process all view-aligned folders with the advanced hair transfer pipeline."""
    if config is None:
        config = HairTransferConfig()
    
    data_dir = Path(data_dir)
    provider_subdir = f"shape_{shape_provider}__texture_{texture_provider}"
    view_aligned_dir = data_dir / config.DIR_VIEW_ALIGNED / provider_subdir
    
    if not view_aligned_dir.exists():
        raise ValueError(f"View aligned directory not found: {view_aligned_dir}")
    
    # Determine bald versions to process
    if bald_version == "all":
        bald_versions = ["w_seg", "wo_seg"]
    else:
        bald_versions = [bald_version]
    
    # Determine conditioning modes to process
    if conditioning_mode == "all":
        conditioning_modes = ["3d_aware", "3d_unaware"]
    else:
        conditioning_modes = [conditioning_mode]
    
    # Log configuration
    print(f"\nConfiguration:")
    print(f"  bald_version(s): {bald_versions}")
    print(f"  conditioning_mode(s): {conditioning_modes}")
    
    # Find all folders
    all_folders = [f for f in view_aligned_dir.iterdir() if f.is_dir()]
    
    if not all_folders:
        print("No folders found!")
        return {"processed": 0, "skipped": 0, "errors": 0}
    
    # Shuffle for distributed processing
    timestamp_seed = int(time.time())
    random.seed(timestamp_seed)
    random.shuffle(all_folders)
    print(f"Found {len(all_folders)} samples (shuffle seed: {timestamp_seed})")
    
    # Initialize pipeline
    print("\n" + "=" * 60)
    print("Initializing Advanced Hair Transfer Pipeline...")
    print("=" * 60)
    pipeline = AdvancedHairTransferPipeline(config)
    
    overall_success = 0
    overall_total = 0
    
    try:
        # Process samples for each bald_version and conditioning_mode combination
        for bv in bald_versions:
            for cm in conditioning_modes:
                print(f"\n{'='*60}")
                print(f"Processing: bald_version={bv}, conditioning_mode={cm}")
                print(f"{'='*60}")
                
                success_count = 0
                for i, folder in enumerate(all_folders, 1):
                    print(f"\n[{i}/{len(all_folders)}] Processing {folder.name} ({bv}/{cm})")
                    
                    if process_sample(
                        folder,
                        pipeline,
                        data_dir,
                        config,
                        bald_version=bv,
                        conditioning_mode=cm,
                        skip_existing=skip_existing,
                    ):
                        success_count += 1
                
                print(f"\n✓ {bv}/{cm}: {success_count}/{len(all_folders)} samples processed")
                overall_success += success_count
                overall_total += len(all_folders)
        
        print(f"\n{'='*60}")
        print(f"✓ All processing complete! {overall_success}/{overall_total} total samples processed")
        print(f"{'='*60}")

    finally:
        del pipeline
        # Release singleton instances to free GPU memory
        SAMMaskExtractorSingleton.release()
        BackgroundRemoverSingleton.release()
        flush()
    
    return {"processed": overall_success, "skipped": 0, "errors": overall_total - overall_success}


# ================================
# CLI Entry Point
# ================================

def main():
    """Main entry point for the advanced hair transfer pipeline."""
    parser = argparse.ArgumentParser(
        description="Advanced Hair Transfer Pipeline with InsertAnything + RF-Inversion"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch"],
        default="batch",
        help="Processing mode: single image or batch"
    )
    
    # Single mode arguments
    parser.add_argument("--source", type=str, help="Source (bald) image path")
    parser.add_argument("--reference", type=str, help="Reference (hair) image path")
    parser.add_argument("--ref_mask", type=str, help="Reference hair mask path (optional, auto-computed with SAM if not provided)")
    parser.add_argument("--tar_mask", type=str, help="Target hair mask path (optional, auto-computed with SAM if not provided)")
    parser.add_argument("--redux_image", type=str, help="Matted reference image for Redux encoding (optional, uses reference if not provided)")
    parser.add_argument("--redux_mask", type=str, help="Hair mask for Redux image (optional, auto-computed with SAM if not provided)")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument("--hair_prompt", type=str, help="Hair description prompt")
    
    # Batch mode arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspace/outputs",
        help="Root data directory for batch processing (default: /workspace/outputs)"
    )
    parser.add_argument(
        "--shape_provider",
        type=str,
        default="hi3dgen",
        choices=["hunyuan", "hi3dgen", "direct3d_s2"],
        help="Shape provider name (default: hi3dgen)"
    )
    parser.add_argument(
        "--texture_provider",
        type=str,
        default="mvadapter",
        choices=["hunyuan", "mvadapter"],
        help="Texture provider name (default: mvadapter)"
    )
    
    # Bald image settings (matching restore_hair_flux2.py)
    parser.add_argument(
        "--bald_version",
        type=str,
        default="w_seg",
        choices=["w_seg", "wo_seg", "all"],
        help="Bald version to use: w_seg, wo_seg, or all (default: w_seg)"
    )
    
    # Conditioning mode settings (matching restore_hair_flux2.py)
    parser.add_argument(
        "--conditioning_mode",
        type=str,
        default="3d_aware",
        choices=["3d_aware", "3d_unaware", "all"],
        help="Conditioning mode: 3d_aware (use aligned image when available), "
             "3d_unaware (never use aligned image), or all (default: 3d_aware)"
    )
    
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip already processed folders"
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        help="Process all folders"
    )
    
    # Pipeline parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--guidance_scale", type=float, default=30.0, help="Guidance scale")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--no_poisson_blend", type=bool, default=False, help="Disable Poisson blending")
    parser.add_argument("--no_multiscale_blend", type=bool, default=False, help="Disable multi-scale blending")
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="LoRA weight scale (lower = less InsertAnything diptych-copy, more Redux influence). Try 0.5-0.8 for more Redux effect. Default: 1.0"
    )
    
    # Latent-space blending parameters (new)
    parser.add_argument(
        "--latent_blend",
        action="store_true",
        default=True,
        help="Enable latent-space blending (recommended for natural hair adaptation)"
    )
    parser.add_argument(
        "--no_latent_blend",
        action="store_false",
        dest="latent_blend",
        help="Disable latent-space blending (use pixel blending only)"
    )
    parser.add_argument(
        "--latent_blend_start",
        type=float,
        default=0.45,
        help="When to start latent blending (0-1, lower=more freedom). Default: 0.3"
    )
    parser.add_argument(
        "--latent_blend_strength",
        type=float,
        default=0.90,
        help="Latent blend strength for non-hair regions (0=full freedom, 1=strict preservation). Default: 0.5"
    )
    parser.add_argument(
        "--latent_blend_schedule",
        type=str,
        default="flux_linear",
        choices=["flux_linear", "flux_cosine", "constant"],
        help="How blend strength changes over denoising steps (FLUX-compatible). Default: flux_linear"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = HairTransferConfig()
    config.SEED = args.seed
    config.GUIDANCE_SCALE = args.guidance_scale
    config.NUM_INFERENCE_STEPS = args.num_steps
    config.POISSON_BLEND = not args.no_poisson_blend
    config.MULTI_SCALE_BLEND = not args.no_multiscale_blend
    config.LORA_SCALE = args.lora_scale
    
    # Latent blending config
    config.LATENT_BLEND = args.latent_blend
    config.LATENT_BLEND_START_STEP = args.latent_blend_start
    config.LATENT_BLEND_STRENGTH = args.latent_blend_strength
    config.LATENT_BLEND_SCHEDULE = args.latent_blend_schedule
    
    if args.mode == "single":
        # Single image processing
        if not all([args.source, args.reference, args.output]):
            parser.error("Single mode requires --source, --reference, and --output (masks are optional, will auto-compute with SAM)")
        
        pipeline = AdvancedHairTransferPipeline(config)
        # try:
        pipeline.transfer_hair(
            source_image_path=args.source,
            reference_image_path=args.reference,
            reference_mask_path=args.ref_mask,
            target_mask_path=args.tar_mask,
            output_path=args.output,
            hair_prompt=args.hair_prompt,
            redux_image_path=args.redux_image,
            redux_mask_path=args.redux_mask,
        )
        output_mask_path, output_final_path, output_final_mask_path = _derive_second_pass_paths(
            args.output, config
        )
        score = _compute_hair_mask(args.output, output_mask_path)
        print(f"Saved computed hair mask to {output_mask_path} (SAM score: {score:.3f})")
        second_pass_overrides = {
            "HAIR_MASK_DILATION_KERNEL": config.SECOND_PASS_HAIR_MASK_DILATION_KERNEL,
            "HAIR_MASK_DILATION_ITERATIONS": config.SECOND_PASS_HAIR_MASK_DILATION_ITERATIONS,
            "BLEND_MASK_DILATION_KERNEL": config.SECOND_PASS_BLEND_MASK_DILATION_KERNEL,
            "BLEND_MASK_DILATION_ITERATIONS": config.SECOND_PASS_BLEND_MASK_DILATION_ITERATIONS,
            "BLEND_MASK_BLUR_RADIUS": config.SECOND_PASS_BLEND_MASK_BLUR_RADIUS,
            "LATENT_BLEND": False,
        }
        with pipeline._override_config(**second_pass_overrides):
            pipeline.transfer_hair(
                source_image_path=args.source,
                reference_image_path=str(Path(args.output)),
                reference_mask_path=str(output_mask_path),
                target_mask_path=str(output_mask_path),
                output_path=str(output_final_path),
                hair_prompt=args.hair_prompt,
                redux_image_path=args.redux_image,
                redux_mask_path=args.redux_mask,
            )
        score = _compute_hair_mask(output_final_path, output_final_mask_path)
        print(f"Saved computed final hair mask to {output_final_mask_path} (SAM score: {score:.3f})")
        _composite_with_bald(
            hair_image_path=output_final_path,
            hair_mask_path=output_final_mask_path,
            bald_image_path=args.source,
            output_path=output_final_path,
            use_multiscale=config.MULTI_SCALE_BLEND,
        )
        
        # Cleanup
        del pipeline
        SAMMaskExtractorSingleton.release()
        BackgroundRemoverSingleton.release()
        flush()
    
    else:
        # Batch processing
        results = process_view_aligned_folders(
            data_dir=args.data_dir,
            shape_provider=args.shape_provider,
            texture_provider=args.texture_provider,
            config=config,
            skip_existing=args.skip_existing,
            bald_version=args.bald_version,
            conditioning_mode=args.conditioning_mode,
        )
        
        print(f"\n{'=' * 60}")
        print("Processing Complete:")
        print(f"  Processed: {results['processed']}")
        print(f"  Skipped:   {results['skipped']}")
        print(f"  Errors:    {results['errors']}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
