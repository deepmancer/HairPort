"""
Hair Restoration Pipeline using AnyDoor Framework.

This module implements hair transfer using the AnyDoor zero-shot object-level
image customization model instead of FLUX/InsertAnything.

AnyDoor uses:
- Stable Diffusion 2.1 backbone with ControlNet
- DINOv2 for visual conditioning
- Collage-based approach with Sobel edge detection
- DDIM sampling at 512x512 resolution

Reference: https://github.com/ali-vilab/AnyDoor
"""

import argparse
import gc
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import shutil

# Add AnyDoor to path
ANYDOOR_PATH = Path(__file__).parent.parent.parent.parent / "AnyDoor"
if str(ANYDOOR_PATH) not in sys.path:
    sys.path.insert(0, str(ANYDOOR_PATH))

# AnyDoor imports
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import (
    get_bbox_from_mask,
    expand_bbox,
    box2squre,
    pad_to_square,
    expand_image_mask,
    box_in_box,
    sobel,
)

# Local imports - same as restore_hair_new.py
from utils.sam_mask_extractor import SAMMaskExtractor
from utils.bg_remover import BackgroundRemover
from hairport.utility.masking_utils import expand_bbox_for_hair
from hairport.utility.blending_utils import (
    create_distance_soft_blend_mask,
    create_hierarchical_blend_mask,
    multi_scale_blend,
)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def flush():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# Singleton Classes for SAM and Background Remover (same as restore_hair_new.py)
# ============================================================================

class SAMMaskExtractorSingleton:
    """Singleton for SAM mask extraction to avoid reloading models."""
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
        if SAMMaskExtractor is None:
            raise RuntimeError("SAMMaskExtractor is not available.")

        thresholds_changed = (
            cls._confidence_threshold != confidence_threshold
            or cls._detection_threshold != detection_threshold
        )

        if cls._instance is None or force_reload or thresholds_changed:
            if cls._instance is not None:
                del cls._instance
                cls._instance = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            cls._instance = SAMMaskExtractor(
                confidence_threshold=confidence_threshold,
                detection_threshold=detection_threshold,
            )
            cls._confidence_threshold = confidence_threshold
            cls._detection_threshold = detection_threshold
            print(f"[SAMMaskExtractorSingleton] Initialized with confidence={confidence_threshold}, detection={detection_threshold}")

        return cls._instance

    @classmethod
    def release(cls):
        """Release singleton instance and free GPU memory."""
        if cls._instance is not None:
            del cls._instance
            cls._instance = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[SAMMaskExtractorSingleton] Released instance")

    @classmethod
    def is_available(cls) -> bool:
        return SAMMaskExtractor is not None


class BackgroundRemoverSingleton:
    """Singleton for background removal to avoid reloading models."""
    _instance: Optional[BackgroundRemover] = None

    @classmethod
    def get_instance(cls, force_reload: bool = False) -> BackgroundRemover:
        if cls._instance is None or force_reload:
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
        """Release singleton instance and free GPU memory."""
        if cls._instance is not None:
            del cls._instance
            cls._instance = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[BackgroundRemoverSingleton] Released instance")


def extract_hair_mask(
    image: Image.Image,
    confidence_threshold: float = 0.4,
    detection_threshold: float = 0.5,
    prompt: str = "head hair",
    sam_instance: Optional[SAMMaskExtractor] = None,
    bg_remover_instance: Optional[BackgroundRemover] = None,
) -> Tuple[Image.Image, float]:
    """Extract hair mask using SAM, constrained by silhouette."""
    if not SAMMaskExtractorSingleton.is_available():
        raise RuntimeError("SAMMaskExtractor is not available.")

    sam = sam_instance if sam_instance is not None else SAMMaskExtractorSingleton.get_instance(
        confidence_threshold=confidence_threshold,
        detection_threshold=detection_threshold,
    )
    bg_remover = bg_remover_instance if bg_remover_instance is not None else BackgroundRemoverSingleton.get_instance()

    _, silh_mask_pil = bg_remover.remove_background(image)
    hair_mask_pil, score = sam(image, prompt=prompt)

    # Constrain hair mask within silhouette
    hair_mask_np = np.array(hair_mask_pil).astype(np.float32) / 255.0
    silh_mask_np = np.array(silh_mask_pil).astype(np.float32) / 255.0
    constrained_hair_mask = (hair_mask_np * silh_mask_np * 255.0).astype(np.uint8)
    hair_mask_pil = Image.fromarray(constrained_hair_mask)

    return hair_mask_pil, score


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AnyDoorHairTransferConfig:
    """Configuration for AnyDoor-based hair transfer pipeline."""

    # Model paths
    ANYDOOR_CONFIG: str = str(ANYDOOR_PATH / "configs" / "anydoor.yaml")
    ANYDOOR_CHECKPOINT: str = str(ANYDOOR_PATH / "epoch=1-step=8687.ckpt")
    DINOV2_WEIGHTS: str = str(ANYDOOR_PATH / "dinov2_vitg14_pretrain.pth")
    ISEG_MODEL_PATH: str = str(ANYDOOR_PATH / "iseg" / "coarse_mask_refine.pth")

    # Processing resolution (AnyDoor uses 512x512)
    PROCESSING_RESOLUTION: int = 512
    OUTPUT_RESOLUTION: int = 1024
    REF_IMAGE_SIZE: int = 224  # Reference image size for DINOv2

    # Generation parameters
    SEED: int = 42
    GUIDANCE_SCALE: float = 4.5  # AnyDoor default
    CONTROL_STRENGTH: float = 1.0
    DDIM_STEPS: int = 50
    ETA: float = 0.0

    # Shape control (whether to use target mask for shape guidance)
    ENABLE_SHAPE_CONTROL: bool = True  # Important for hair transfer

    # Memory optimization
    SAVE_MEMORY: bool = False

    # Directory structure (same as restore_hair_new.py)
    DIR_VIEW_ALIGNED: str = "view_aligned"
    DIR_ALIGNMENT: str = "alignment"
    DIR_BALD: str = "bald"
    DIR_PROMPTS: str = "prompt"

    DIR_3D_AWARE: str = "3d_aware"
    DIR_3D_UNAWARE: str = "3d_unaware"

    SUBDIR_WARPING: str = "warping"
    SUBDIR_BLENDING: str = "blending"
    SUBDIR_TRANSFERRED: str = "transferred_anydoor"  # Different output dir

    DIR_IMAGE: str = "image"

    # File names
    FILE_VIEW_ALIGNED_IMAGE: str = "target_image_phase_1.png"
    FILE_WARPED_TARGET_IMAGE: str = "warped_target_image.png"
    FILE_WARPED_HAIR_MASK: str = "target_hair_mask.png"
    FILE_TARGET_HAIR_MASK: str = "target_image_hair_mask.png"

    FILE_HAIR_RESTORED: str = "hair_restored.png"
    FILE_HAIR_RESTORED_MASK: str = "hair_restored_mask.png"

    # Outpainted source paths
    DIR_SOURCE_OUTPAINTED: str = "source_outpainted"
    FILE_OUTPAINTED_IMAGE: str = "outpainted_image.png"
    FILE_RESIZE_INFO: str = "resize_info.json"

    SUBDIR_BALD_IMAGE: str = "image"


# ============================================================================
# AnyDoor Processing Functions
# ============================================================================

def crop_back(pred: np.ndarray, tar_image: np.ndarray, extra_sizes: np.ndarray, tar_box_yyxx_crop: np.ndarray) -> np.ndarray:
    """Crop generated image back to original target image space."""
    H1, W1, H2, W2 = extra_sizes
    y1, y2, x1, x2 = tar_box_yyxx_crop
    pred = cv2.resize(pred, (W2, H2))
    m = 3  # margin pixels

    if W1 == H1:
        tar_image[y1 + m : y2 - m, x1 + m : x2 - m, :] = pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:, pad1:-pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1:-pad2, :, :]

    tar_image[y1 + m : y2 - m, x1 + m : x2 - m, :] = pred[m:-m, m:-m]
    return tar_image


def process_pairs_for_hair(
    ref_image: np.ndarray,
    ref_mask: np.ndarray,
    tar_image: np.ndarray,
    tar_mask: np.ndarray,
    enable_shape_control: bool = True,
    ref_expand_ratio: float = 1.3,
) -> Dict[str, Any]:
    """
    Process reference (hair) and target (bald) image pairs for AnyDoor inference.
    
    This is adapted from AnyDoor's process_pairs but optimized for hair transfer:
    - Reference: the hair region to transfer (should be pre-masked with white background)
    - Target: the bald person's head region where hair should be placed
    
    Note: ref_image should ideally be pre-masked before calling this function,
    but we apply masking here as well for safety.
    """
    # ========= Reference (Hair) Processing ===========
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # Ensure ref_mask is binary float for proper masking arithmetic
    ref_mask_float = ref_mask.astype(np.float32)
    ref_mask_3 = np.stack([ref_mask_float, ref_mask_float, ref_mask_float], axis=-1)
    
    # Mask reference image (white background for non-hair)
    # Use float arithmetic to avoid uint8 overflow issues
    ref_image_float = ref_image.astype(np.float32)
    masked_ref_image = (ref_image_float * ref_mask_3 + 255.0 * (1.0 - ref_mask_3)).astype(np.uint8)

    # Crop to bounding box
    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
    ref_mask = ref_mask[y1:y2, x1:x2]

    # Expand with padding
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ref_expand_ratio)
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)

    # Pad to square and resize to 224x224 (DINOv2 input size)
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)
    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), (224, 224)).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value=0, random=False)
    ref_mask_3 = cv2.resize(ref_mask_3.astype(np.uint8), (224, 224)).astype(np.uint8)
    ref_mask = ref_mask_3[:, :, 0]

    # Create collage reference (Sobel edge extraction)
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose / 255)

    # ========= Target (Bald Head) Processing ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1, 1.3])
    tar_box_yyxx_full = tar_box_yyxx

    # Expand crop region for context
    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.3, 3.0])
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2, x1:x2, :]
    cropped_tar_mask = tar_mask[y1:y2, x1:x2]

    # Get relative box position within crop
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx

    # Create collage: place reference edge map in target region
    ref_image_collage = cv2.resize(ref_image_collage.astype(np.uint8), (x2 - x1, y2 - y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2 - x1, y2 - y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy()
    collage[y1:y2, x1:x2, :] = ref_image_collage

    # Create collage mask
    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2, x1:x2, :] = 1.0

    if enable_shape_control:
        # Use target mask for shape guidance (important for hair)
        collage_mask = np.stack([cropped_tar_mask, cropped_tar_mask, cropped_tar_mask], -1)

    # Size before padding
    H1, W1 = collage.shape[0], collage.shape[1]

    # Pad to square
    cropped_target_image = pad_to_square(cropped_target_image, pad_value=0, random=False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value=0, random=False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value=2, random=False).astype(np.uint8)

    # Size after padding
    H2, W2 = collage.shape[0], collage.shape[1]

    # Resize to 512x512 (AnyDoor processing size)
    cropped_target_image = cv2.resize(cropped_target_image.astype(np.uint8), (512, 512)).astype(np.float32)
    collage = cv2.resize(collage.astype(np.uint8), (512, 512)).astype(np.float32)
    collage_mask = cv2.resize(collage_mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    collage_mask[collage_mask == 2] = -1  # Mark padding regions

    # Normalize
    masked_ref_image = masked_ref_image / 255
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0
    collage = np.concatenate([collage, collage_mask[:, :, :1]], -1)

    item = dict(
        ref=masked_ref_image.copy(),
        jpg=cropped_target_image.copy(),
        hint=collage.copy(),
        extra_sizes=np.array([H1, W1, H2, W2]),
        tar_box_yyxx_crop=np.array(tar_box_yyxx_crop),
        tar_box_yyxx=np.array(tar_box_yyxx_full),
    )
    return item


def composite_hair_onto_bald(
    hair_restored_np: np.ndarray,
    bald_np: np.ndarray,
    hair_mask_np: np.ndarray,
    use_multiscale: bool = True,
    feather_px: int = 12,
) -> np.ndarray:
    """Composite hair region from restored image onto bald source."""
    if hair_mask_np.max() > 1:
        hair_mask_np = hair_mask_np.astype(np.float32) / 255.0

    h, w = bald_np.shape[:2]
    if hair_restored_np.shape[:2] != (h, w):
        hair_restored_np = cv2.resize(hair_restored_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
    if hair_mask_np.shape[:2] != (h, w):
        hair_mask_np = cv2.resize(hair_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

    if use_multiscale:
        blend_masks = create_hierarchical_blend_mask(hair_mask_np, num_levels=4)
        composited = multi_scale_blend(
            hair_restored_np,
            bald_np,
            blend_masks,
            use_laplacian=True,
        )
    else:
        soft_mask = create_distance_soft_blend_mask(
            hair_mask_np,
            dilation_px=3,
            dilation_iterations=1,
            feather_px=feather_px,
        )
        alpha = soft_mask[:, :, np.newaxis]
        composited = (hair_restored_np * alpha + bald_np * (1 - alpha)).astype(np.uint8)

    return composited


# ============================================================================
# AnyDoor Hair Restoration Pipeline
# ============================================================================

class AnyDoorHairRestorationPipeline:
    """Pipeline for hair transfer using AnyDoor framework."""

    def __init__(
        self,
        config: Optional[AnyDoorHairTransferConfig] = None,
        device: str = "cuda",
    ):
        if config is None:
            config = AnyDoorHairTransferConfig()

        self.config = config
        self.device = device
        self.size = (config.PROCESSING_RESOLUTION, config.PROCESSING_RESOLUTION)

        # Disable verbosity
        disable_verbosity()
        if config.SAVE_MEMORY:
            enable_sliced_attention()

        self._load_models()

    def _load_models(self):
        """Load and initialize AnyDoor model."""
        print("Loading AnyDoor model...")

        # Update config with correct paths
        anydoor_config = OmegaConf.load(self.config.ANYDOOR_CONFIG)

        # Update DINOv2 weight path in config
        if hasattr(anydoor_config.model.params, 'cond_stage_config'):
            anydoor_config.model.params.cond_stage_config.weight = self.config.DINOV2_WEIGHTS

        # Create model
        self.model = create_model(self.config.ANYDOOR_CONFIG).cpu()

        # Load checkpoint
        if os.path.exists(self.config.ANYDOOR_CHECKPOINT):
            self.model.load_state_dict(load_state_dict(self.config.ANYDOOR_CHECKPOINT, location='cuda'))
            print(f"Loaded AnyDoor checkpoint from {self.config.ANYDOOR_CHECKPOINT}")
        else:
            raise FileNotFoundError(f"AnyDoor checkpoint not found at {self.config.ANYDOOR_CHECKPOINT}")

        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)

        print("AnyDoor pipeline initialized successfully!")

    def mask_image(
        self,
        image: Image.Image,
        mask_mode: str = "hair",
        background_value: Tuple[int, int, int] = (255, 255, 255),
        hair_mask: Optional[Image.Image] = None,
        mode: str = "RGB",
    ) -> Tuple[Image.Image, Image.Image]:
        """Mask image by hair or subject silhouette."""
        image_np = np.array(image).astype(np.float32)

        if mask_mode == "hair":
            if hair_mask is not None:
                mask_pil = hair_mask
                if mask_pil.size != image.size:
                    mask_pil = mask_pil.resize(image.size, Image.Resampling.NEAREST)
            else:
                mask_pil, _ = extract_hair_mask(image)
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        elif mask_mode == "matte":
            bg_remover = BackgroundRemoverSingleton.get_instance()
            _, mask_pil = bg_remover.remove_background(image)
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unknown mask_mode: {mask_mode}")

        background = np.full_like(image_np, background_value, dtype=np.float32)
        mask_3ch = mask_np[:, :, np.newaxis]
        masked_np = (image_np * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)

        if mask_pil.mode != "L":
            mask_pil = mask_pil.convert("L")

        return Image.fromarray(masked_np).convert(mode), mask_pil

    def transfer_hair(
        self,
        source_image_path: str,
        source_mask_path: str,
        reference_image_path: str,
        reference_mask_path: str,
        output_dir: str = "output",
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        control_strength: Optional[float] = None,
        ddim_steps: Optional[int] = None,
        enable_shape_control: Optional[bool] = None,
    ) -> str:
        """
        Transfer hair from reference to source (bald) image using AnyDoor.

        Args:
            source_image_path: Path to source (bald) image
            source_mask_path: Path to target mask (where hair should go)
            reference_image_path: Path to reference (hair donor) image
            reference_mask_path: Path to reference hair mask
            output_dir: Output directory
            seed: Random seed
            guidance_scale: Classifier-free guidance scale
            control_strength: ControlNet strength
            ddim_steps: Number of DDIM steps
            enable_shape_control: Use target mask for shape guidance

        Returns:
            Path to output image
        """
        if seed is None:
            seed = self.config.SEED
        if guidance_scale is None:
            guidance_scale = self.config.GUIDANCE_SCALE
        if control_strength is None:
            control_strength = self.config.CONTROL_STRENGTH
        if ddim_steps is None:
            ddim_steps = self.config.DDIM_STEPS
        if enable_shape_control is None:
            enable_shape_control = self.config.ENABLE_SHAPE_CONTROL

        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"Processing with seed={seed}, guidance_scale={guidance_scale}, steps={ddim_steps}")
        print("Loading and preparing images...")

        # Load images
        source_pil = Image.open(source_image_path).convert("RGB")
        reference_pil = Image.open(reference_image_path).convert("RGB")
        reference_mask_pil = Image.open(reference_mask_path).convert("L")
        source_mask_pil = Image.open(source_mask_path).convert("L")

        # CRITICAL: Ensure masks match image sizes before any processing
        if reference_mask_pil.size != reference_pil.size:
            print(f"  Warning: Resizing reference mask from {reference_mask_pil.size} to {reference_pil.size}")
            reference_mask_pil = reference_mask_pil.resize(reference_pil.size, Image.Resampling.NEAREST)
        if source_mask_pil.size != source_pil.size:
            print(f"  Warning: Resizing source mask from {source_mask_pil.size} to {source_pil.size}")
            source_mask_pil = source_mask_pil.resize(source_pil.size, Image.Resampling.NEAREST)

        # Convert to numpy (AnyDoor works with numpy arrays)
        tar_image = np.asarray(source_pil)
        ref_image = np.asarray(reference_pil).copy()  # Make writable copy
        tar_mask = np.asarray(source_mask_pil)
        tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)
        ref_mask = np.asarray(reference_mask_pil)
        ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

        if tar_mask.sum() == 0:
            raise ValueError("No mask for the target (bald) image!")
        if ref_mask.sum() == 0:
            raise ValueError("No mask for the reference (hair) image!")

        # CRITICAL: Pre-mask the reference image to ensure ONLY hair pixels are visible
        # This prevents any non-hair regions from leaking into the model's visual conditioning
        # Set non-hair regions to white (255) before any further processing
        ref_mask_3ch = np.stack([ref_mask, ref_mask, ref_mask], axis=-1)
        ref_image_masked = ref_image * ref_mask_3ch + (255 * (1 - ref_mask_3ch)).astype(np.uint8)

        # Keep original for crop_back
        raw_background = tar_image.copy()

        # Process pairs for AnyDoor - pass the pre-masked reference image
        item = process_pairs_for_hair(
            ref_image_masked,  # Use pre-masked image instead of raw reference
            ref_mask,
            tar_image,
            tar_mask,
            enable_shape_control=enable_shape_control,
        )

        ref = item['ref']
        hint = item['hint']
        num_samples = 1

        # Prepare control input
        control = torch.from_numpy(hint.copy()).float().cuda()
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        # Prepare CLIP/DINOv2 input
        clip_input = torch.from_numpy(ref.copy()).float().cuda()
        clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
        clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

        H, W = 512, 512

        # Prepare conditioning
        cond = {
            "c_concat": [control],
            "c_crossattn": [self.model.get_learned_conditioning(clip_input)],
        }
        un_cond = {
            "c_concat": [control],
            "c_crossattn": [self.model.get_learned_conditioning([torch.zeros((1, 3, 224, 224)).cuda()] * num_samples)],
        }
        shape = (4, H // 8, W // 8)

        if self.config.SAVE_MEMORY:
            self.model.low_vram_shift(is_diffusing=True)

        # Set control scales
        self.model.control_scales = [control_strength] * 13

        # Sample
        samples, _ = self.ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=self.config.ETA,
            unconditional_guidance_scale=guidance_scale,
            unconditional_conditioning=un_cond,
        )

        if self.config.SAVE_MEMORY:
            self.model.low_vram_shift(is_diffusing=False)

        # Decode
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()

        pred = x_samples[0]
        pred = np.clip(pred, 0, 255).astype(np.uint8)

        # Crop back to original image space
        sizes = item['extra_sizes']
        tar_box_yyxx_crop = item['tar_box_yyxx_crop']
        gen_image = crop_back(pred, tar_image.copy(), sizes, tar_box_yyxx_crop)

        # Keep background unchanged outside the target region
        y1, y2, x1, x2 = item['tar_box_yyxx']
        raw_background[y1:y2, x1:x2, :] = gen_image[y1:y2, x1:x2, :]
        gen_image = raw_background

        # Resize to output resolution
        edited_image = Image.fromarray(gen_image)
        edited_image = edited_image.resize(
            (self.config.OUTPUT_RESOLUTION, self.config.OUTPUT_RESOLUTION),
            Image.Resampling.LANCZOS,
        )

        # Save output
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, self.config.FILE_HAIR_RESTORED)
        edited_image.save(output_path)
        print(f"Saved hair-restored image to {output_path}")

        return output_path

    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'ddim_sampler'):
            del self.ddim_sampler
        flush()


# ============================================================================
# Helper Functions
# ============================================================================

def _compute_hair_mask(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    *,
    confidence_threshold: float = 0.4,
    detection_threshold: float = 0.5,
    prompt: str = "head hair",
) -> float:
    """Compute and save hair mask using SAM."""
    if not SAMMaskExtractorSingleton.is_available():
        raise RuntimeError("SAMMaskExtractor unavailable.")

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


def get_output_dir(
    pair_dir: Path,
    config: AnyDoorHairTransferConfig,
    use_3d_aware: bool,
) -> Path:
    """Get output directory for AnyDoor results."""
    mode_dir = config.DIR_3D_AWARE if use_3d_aware else config.DIR_3D_UNAWARE
    return pair_dir / mode_dir / config.SUBDIR_TRANSFERRED


# ============================================================================
# Batch Processing
# ============================================================================

def process_sample(
    folder: Path,
    pipeline: AnyDoorHairRestorationPipeline,
    data_dir: Path,
    config: AnyDoorHairTransferConfig,
    bald_version: str,
    conditioning_mode: str,
    skip_existing: bool = True,
) -> bool:
    """Process a single sample folder."""
    folder_name = folder.name

    if "_to_" not in folder_name:
        print(f"Skipping {folder_name}: invalid format")
        return False

    try:
        target_id, source_id = folder_name.split("_to_")
    except ValueError:
        print(f"Invalid directory name format: {folder_name}")
        return False

    pair_dir = folder / bald_version
    if not pair_dir.exists():
        print(f"Pair directory not found: {pair_dir}, skipping...")
        return False

    dataset_name = Path(data_dir).name
    use_3d_aware = (conditioning_mode == "3d_aware")

    mode_dir = config.DIR_3D_AWARE if use_3d_aware else config.DIR_3D_UNAWARE
    warping_dir = pair_dir / mode_dir / config.SUBDIR_WARPING

    warped_image_path = warping_dir / config.FILE_WARPED_TARGET_IMAGE
    warped_hair_mask_path = warping_dir / config.FILE_WARPED_HAIR_MASK
    has_warped_outputs = warped_image_path.exists() and warped_hair_mask_path.exists()

    if not has_warped_outputs:
        print(f"  Missing warped outputs for {folder_name}/{bald_version}/{conditioning_mode}, skipping...")
        return False

    output_dir = get_output_dir(pair_dir, config, use_3d_aware)
    output_path = output_dir / config.FILE_HAIR_RESTORED
    output_mask_path = output_dir / config.FILE_HAIR_RESTORED_MASK

    if skip_existing and output_path.exists():
        print(f"Output already exists for {folder_name}/{bald_version}/{conditioning_mode}, skipping...")
        return True

    try:
        # Find source (bald) image
        source_image_path = pair_dir / config.DIR_SOURCE_OUTPAINTED / config.FILE_OUTPAINTED_IMAGE
        if not source_image_path.exists():
            bald_image_dir = data_dir / config.DIR_BALD / bald_version / config.SUBDIR_BALD_IMAGE
            source_image_path = bald_image_dir / f"{source_id}.png"

        if not source_image_path.exists():
            print(f"  Warning: Source image not found: {source_image_path}")
            return False

        # Find reference (hair) image and mask
        alignment_dir = folder / config.DIR_ALIGNMENT
        phase1_image_path = alignment_dir / config.FILE_VIEW_ALIGNED_IMAGE
        phase1_mask_path = alignment_dir / config.FILE_TARGET_HAIR_MASK

        if phase1_image_path.exists():
            reference_image_path = phase1_image_path
            reference_mask_path = phase1_mask_path
            if not reference_mask_path.exists():
                print(f"  Computing hair mask for {phase1_image_path}...")
                _compute_hair_mask(
                    image_path=phase1_image_path,
                    mask_path=reference_mask_path,
                    confidence_threshold=0.3,
                    detection_threshold=0.4,
                    prompt="head hair",
                )
            print(f"  Using target_image_phase_1.png as reference")
        else:
            reference_image_path = warped_image_path
            reference_mask_path = warped_hair_mask_path
            print(f"  Fallback: Using warped_target_image as reference")

        target_mask_path = warped_hair_mask_path

        print(f"  Source: {source_image_path}")
        print(f"  Reference: {reference_image_path}")
        print(f"  Mode: {'3D-Aware' if use_3d_aware else '3D-Unaware'}")

        os.makedirs(output_dir, exist_ok=True)

        try:
            pipeline.transfer_hair(
                source_image_path=str(source_image_path),
                source_mask_path=str(target_mask_path),
                reference_image_path=str(reference_image_path),
                reference_mask_path=str(reference_mask_path),
                output_dir=str(output_dir),
            )
        except Exception as e:
            print(f"  Error during transfer_hair: {e}")
            if output_dir.exists():
                shutil.rmtree(output_dir)
                print(f"  Removed incomplete output directory: {output_dir}")
            return False

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
    config: Optional[AnyDoorHairTransferConfig] = None,
    skip_existing: bool = True,
    bald_version: str = "w_seg",
    conditioning_mode: str = "3d_aware",
) -> Dict[str, int]:
    """Process all view-aligned folders in batch."""
    if config is None:
        config = AnyDoorHairTransferConfig()

    data_dir = Path(data_dir)
    provider_subdir = f"shape_{shape_provider}__texture_{texture_provider}"
    view_aligned_dir = data_dir / config.DIR_VIEW_ALIGNED / provider_subdir

    if not view_aligned_dir.exists():
        raise ValueError(f"View aligned directory not found: {view_aligned_dir}")

    bald_versions = ["w_seg", "wo_seg"] if bald_version == "all" else [bald_version]
    conditioning_modes = ["3d_aware", "3d_unaware"] if conditioning_mode == "all" else [conditioning_mode]

    print(f"\nConfiguration:")
    print(f"  bald_version(s): {bald_versions}")
    print(f"  conditioning_mode(s): {conditioning_modes}")

    all_folders = [f for f in view_aligned_dir.iterdir() if f.is_dir()]

    if not all_folders:
        print("No folders found!")
        return {"processed": 0, "skipped": 0, "errors": 0}

    timestamp_seed = int(time.time())
    random.seed(timestamp_seed)
    random.shuffle(all_folders)
    print(f"Found {len(all_folders)} samples (shuffle seed: {timestamp_seed})")

    print("\n" + "=" * 60)
    print("Initializing AnyDoor Hair Restoration Pipeline...")
    print("=" * 60)
    pipeline = AnyDoorHairRestorationPipeline(config)

    overall_success = 0
    overall_total = 0

    for bv in bald_versions:
        for cm in conditioning_modes:
            print(f"\n{'=' * 60}")
            print(f"Processing: bald_version={bv}, conditioning_mode={cm}")
            print(f"{'=' * 60}")

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

    print(f"\n{'=' * 60}")
    print(f"✓ All processing complete! {overall_success}/{overall_total} total samples processed")
    print(f"{'=' * 60}")

    return {"processed": overall_success, "skipped": 0, "errors": overall_total - overall_success}


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main entry point for AnyDoor-based hair restoration pipeline."""
    parser = argparse.ArgumentParser(
        description="Hair Restoration Pipeline using AnyDoor Framework"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch"],
        default="batch",
        help="Processing mode: single image or batch",
    )

    # Single mode arguments
    parser.add_argument("--source", type=str, help="Source (bald) image path")
    parser.add_argument("--reference", type=str, help="Reference (hair) image path")
    parser.add_argument("--ref_mask", type=str, help="Reference hair mask path")
    parser.add_argument("--tar_mask", type=str, help="Target hair mask path")
    parser.add_argument("--output", type=str, help="Output path")

    # Batch mode arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspace/outputs",
        help="Root data directory for batch processing",
    )
    parser.add_argument(
        "--shape_provider",
        type=str,
        default="hi3dgen",
        choices=["hunyuan", "hi3dgen", "direct3d_s2"],
        help="Shape provider name",
    )
    parser.add_argument(
        "--texture_provider",
        type=str,
        default="mvadapter",
        choices=["hunyuan", "mvadapter"],
        help="Texture provider name",
    )
    parser.add_argument(
        "--bald_version",
        type=str,
        default="w_seg",
        choices=["w_seg", "wo_seg", "all"],
        help="Bald version to use",
    )
    parser.add_argument(
        "--conditioning_mode",
        type=str,
        default="3d_aware",
        choices=["3d_aware", "3d_unaware", "all"],
        help="Conditioning mode",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip already processed folders",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        help="Process all folders",
    )

    # AnyDoor parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.5,
        help="Guidance scale (default: 4.5)",
    )
    parser.add_argument(
        "--control_strength",
        type=float,
        default=1.0,
        help="ControlNet strength (default: 1.0)",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="Number of DDIM steps (default: 50)",
    )
    parser.add_argument(
        "--enable_shape_control",
        action="store_true",
        default=True,
        help="Enable shape control using target mask",
    )
    parser.add_argument(
        "--no_shape_control",
        action="store_false",
        dest="enable_shape_control",
        help="Disable shape control",
    )
    parser.add_argument(
        "--save_memory",
        action="store_true",
        default=False,
        help="Enable memory saving mode",
    )

    args = parser.parse_args()

    # Create config
    config = AnyDoorHairTransferConfig()
    config.SEED = args.seed
    config.GUIDANCE_SCALE = args.guidance_scale
    config.CONTROL_STRENGTH = args.control_strength
    config.DDIM_STEPS = args.ddim_steps
    config.ENABLE_SHAPE_CONTROL = args.enable_shape_control
    config.SAVE_MEMORY = args.save_memory

    if args.mode == "single":
        if not all([args.source, args.reference, args.output]):
            parser.error("Single mode requires --source, --reference, and --output")

        pipeline = AnyDoorHairRestorationPipeline(config)

        output_path = Path(args.output)
        output_dir = output_path.parent

        pipeline.transfer_hair(
            source_image_path=args.source,
            source_mask_path=args.tar_mask if args.tar_mask else args.ref_mask,
            reference_image_path=args.reference,
            reference_mask_path=args.ref_mask,
            output_dir=str(output_dir),
            guidance_scale=args.guidance_scale,
            control_strength=args.control_strength,
            ddim_steps=args.ddim_steps,
            enable_shape_control=args.enable_shape_control,
        )

        # Compute hair mask for output
        output_file = output_dir / config.FILE_HAIR_RESTORED
        output_mask_path = output_dir / config.FILE_HAIR_RESTORED_MASK
        if output_file.exists():
            score = _compute_hair_mask(output_file, output_mask_path)
            print(f"Saved computed hair mask to {output_mask_path} (SAM score: {score:.3f})")

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
