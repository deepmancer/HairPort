"""
Hair Restoration Pipeline using MimicBrush Framework.

This module implements hair transfer using the MimicBrush zero-shot
image editing with reference imitation model.

MimicBrush uses:
- Stable Diffusion 1.5 inpainting backbone
- CLIP image encoder for reference conditioning
- ReferenceNet for visual feature injection
- Depth Anything for optional shape control
- DDIM sampling at 512x512 resolution

Reference: https://github.com/ali-vilab/MimicBrush
Paper: "Zero-shot Image Editing with Reference Imitation" (arXiv:2406.07547)
"""

import argparse
import gc
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

# ============================================================================
# Diffusers Compatibility Patches
# ============================================================================
# The MimicBrush codebase was developed with an older version of diffusers
# that had some classes that no longer exist in stable versions.
# We patch these before importing MimicBrush modules.

def _patch_diffusers_for_mimicbrush():
    """
    Patch diffusers modules for MimicBrush compatibility.
    This must be called BEFORE importing any MimicBrush modules.
    
    Changes in newer diffusers:
    - PositionNet was removed from diffusers.models.embeddings
    - unet_2d_blocks moved from diffusers.models to diffusers.models.unets
    - LoRALinearLayer moved or was removed
    """
    import diffusers.models.embeddings as embeddings_module
    
    # === Patch 1: Add missing PositionNet ===
    if not hasattr(embeddings_module, 'PositionNet'):
        class PositionNet(nn.Module):
            """
            Stub PositionNet for diffusers compatibility.
            This class is used in gated attention mechanisms but is not present
            in newer diffusers versions. For MimicBrush hair transfer, this code
            path is typically not executed as we use standard attention.
            """
            def __init__(
                self,
                positive_len: int = 768,
                out_dim: int = 768,
                feature_type: str = "text-only",
            ):
                super().__init__()
                self.positive_len = positive_len
                self.out_dim = out_dim if isinstance(out_dim, int) else out_dim[0]
                self.feature_type = feature_type
                
                # Simple projection layers
                self.linears = nn.Sequential(
                    nn.Linear(self.positive_len, 512),
                    nn.SiLU(),
                    nn.Linear(512, self.out_dim),
                )
            
            def forward(self, x):
                return self.linears(x)
        
        embeddings_module.PositionNet = PositionNet
        print("[MimicBrush Compat] Injected PositionNet stub into diffusers.models.embeddings")
    
    # Also check for other potentially missing imports
    if not hasattr(embeddings_module, 'ImageHintTimeEmbedding'):
        class ImageHintTimeEmbedding(nn.Module):
            """Stub for ImageHintTimeEmbedding."""
            def __init__(self, *args, **kwargs):
                super().__init__()
            def forward(self, x, *args, **kwargs):
                return x
        embeddings_module.ImageHintTimeEmbedding = ImageHintTimeEmbedding
        print("[MimicBrush Compat] Injected ImageHintTimeEmbedding stub")
    
    # === Patch 2: Create diffusers.models.unet_2d_blocks alias ===
    # In newer diffusers, this moved to diffusers.models.unets.unet_2d_blocks
    import diffusers.models
    if not hasattr(diffusers.models, 'unet_2d_blocks'):
        try:
            from diffusers.models.unets import unet_2d_blocks
            diffusers.models.unet_2d_blocks = unet_2d_blocks
            sys.modules['diffusers.models.unet_2d_blocks'] = unet_2d_blocks
            print("[MimicBrush Compat] Aliased diffusers.models.unets.unet_2d_blocks -> diffusers.models.unet_2d_blocks")
        except ImportError:
            print("[MimicBrush Compat] Warning: Could not find unet_2d_blocks in either location")
    
    # === Patch 3: Handle LoRALinearLayer if missing ===
    try:
        from diffusers.models.lora import LoRALinearLayer
    except ImportError:
        # Create a stub module for diffusers.models.lora
        import types
        lora_module = types.ModuleType('diffusers.models.lora')
        
        class LoRALinearLayer(nn.Module):
            """Stub LoRALinearLayer for compatibility."""
            def __init__(self, *args, **kwargs):
                super().__init__()
            def forward(self, x):
                return x
        
        lora_module.LoRALinearLayer = LoRALinearLayer
        sys.modules['diffusers.models.lora'] = lora_module
        diffusers.models.lora = lora_module
        print("[MimicBrush Compat] Injected LoRALinearLayer stub")

# Apply patches before any MimicBrush imports
_patch_diffusers_for_mimicbrush()

# Add MimicBrush to path
MIMICBRUSH_PATH = Path(__file__).parent.parent.parent.parent / "MimicBrush"
if str(MIMICBRUSH_PATH) not in sys.path:
    sys.path.insert(0, str(MIMICBRUSH_PATH))

# MimicBrush imports (MUST come after patching)
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from models.pipeline_mimicbrush import MimicBrushPipeline
from models.ReferenceNet import ReferenceNet
from models.depth_guider import DepthGuider
from mimicbrush import MimicBrush_RefNet
from torchvision.transforms import Compose

# Depth Anything imports
DEPTHANYTHING_PATH = MIMICBRUSH_PATH / "depthanything"
if str(DEPTHANYTHING_PATH) not in sys.path:
    sys.path.insert(0, str(DEPTHANYTHING_PATH))

from depthanything.fast_import import depth_anything_model
from depthanything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# Local imports - same as other restore_hair files
from utils.sam_mask_extractor import SAMMaskExtractor
from utils.bg_remover import BackgroundRemover
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
# Singleton Classes for SAM and Background Remover (same as other restore_hair files)
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
class MimicBrushHairTransferConfig:
    """Configuration for MimicBrush-based hair transfer pipeline."""

    # Model paths (relative to MimicBrush installation)
    MIMICBRUSH_CKPT: str = str(MIMICBRUSH_PATH / "MimicBrush" / "mimicbrush" / "mimicbrush.bin")
    REFERENCE_MODEL_PATH: str = str(MIMICBRUSH_PATH / "cleansd" / "stable-diffusion-v1-5")
    INPAINTING_MODEL_PATH: str = str(MIMICBRUSH_PATH / "cleansd" / "stable-diffusion-inpainting")
    VAE_MODEL_PATH: str = str(MIMICBRUSH_PATH / "MimicBrush" / "sd-vae-ft-mse")
    IMAGE_ENCODER_PATH: str = str(MIMICBRUSH_PATH / "MimicBrush" / "image_encoder")
    DEPTH_MODEL_PATH: str = str(MIMICBRUSH_PATH / "MimicBrush" / "depth_model" / "depth_anything_vitb14.pth")

    # Processing resolution (MimicBrush uses 512x512)
    PROCESSING_RESOLUTION: int = 512
    OUTPUT_RESOLUTION: int = 1024

    # Generation parameters
    SEED: int = 42
    GUIDANCE_SCALE: float = 5.0  # MimicBrush default
    NUM_INFERENCE_STEPS: int = 50
    ETA: float = 0.0

    # Shape control (whether to use depth for shape guidance)
    # For hair transfer, typically we want shape control enabled
    # to preserve the target head shape
    ENABLE_SHAPE_CONTROL: bool = True

    # Directory structure (same as other restore_hair files)
    DIR_VIEW_ALIGNED: str = "view_aligned"
    DIR_ALIGNMENT: str = "alignment"
    DIR_BALD: str = "bald"
    DIR_PROMPTS: str = "prompt"
    DIR_3D_AWARE: str = "3d_aware"
    DIR_3D_UNAWARE: str = "3d_unaware"
    SUBDIR_WARPING: str = "warping"
    SUBDIR_BLENDING: str = "blending"
    SUBDIR_TRANSFERRED: str = "transferred_mimicbrush"  # Unique for MimicBrush
    SUBDIR_BALD_IMAGE: str = "image"

    DIR_IMAGE: str = "image"

    # File names (must match restore_hair_new.py)
    FILE_VIEW_ALIGNED_IMAGE: str = "target_image_phase_1.png"
    FILE_WARPED_TARGET_IMAGE: str = "warped_target_image.png"
    FILE_WARPED_HAIR_MASK: str = "target_hair_mask.png"
    FILE_TARGET_HAIR_MASK: str = "target_image_phase_1_mask.png"
    FILE_HAIR_RESTORED: str = "hair_restored.png"
    FILE_HAIR_RESTORED_MASK: str = "hair_restored_mask.png"

    # Outpainted source paths
    DIR_SOURCE_OUTPAINTED: str = "source_outpainted"
    FILE_OUTPAINTED_IMAGE: str = "outpainted_image.png"
    FILE_RESIZE_INFO: str = "resize_info.json"

    # Blending settings
    USE_MULTISCALE_BLEND: bool = True
    BLEND_FEATHER_PX: int = 12


# ============================================================================
# MimicBrush Helper Functions
# ============================================================================

def pad_img_to_square(original_image: Image.Image, is_mask: bool = False) -> Image.Image:
    """Pad image to square with white (image) or black (mask) background."""
    width, height = original_image.size

    if height == width:
        return original_image

    if height > width:
        padding = (height - width) // 2
        new_size = (height, height)
    else:
        padding = (width - height) // 2
        new_size = (width, width)

    if is_mask:
        new_image = Image.new("RGB", new_size, "black")
    else:
        new_image = Image.new("RGB", new_size, "white")

    if height > width:
        new_image.paste(original_image, (padding, 0))
    else:
        new_image.paste(original_image, (0, padding))
    return new_image


def collage_region(low: Image.Image, high: Image.Image, mask: Image.Image) -> Image.Image:
    """Create collage: black out masked region in preparation for inpainting."""
    mask_np = (np.array(mask) > 128).astype(np.uint8)
    low_np = np.array(low).astype(np.uint8)
    # Set masked region to black (will be inpainted)
    low_np = (low_np * 0).astype(np.uint8)
    high_np = np.array(high).astype(np.uint8)
    mask_3 = mask_np
    collage = low_np * mask_3 + high_np * (1 - mask_3)
    return Image.fromarray(collage)


def crop_padding_and_resize(ori_image: np.ndarray, square_image: np.ndarray) -> np.ndarray:
    """Crop and resize square output back to original aspect ratio."""
    ori_height, ori_width = ori_image.shape[:2]
    scale = max(ori_height / square_image.shape[0], ori_width / square_image.shape[1])
    resized_square_image = cv2.resize(
        square_image, 
        (int(square_image.shape[1] * scale), int(square_image.shape[0] * scale))
    )
    padding_size = max(
        resized_square_image.shape[0] - ori_height, 
        resized_square_image.shape[1] - ori_width
    )
    if ori_height < ori_width:
        top = padding_size // 2
        bottom = resized_square_image.shape[0] - (padding_size - top)
        cropped_image = resized_square_image[top:bottom, :, :]
    else:
        left = padding_size // 2
        right = resized_square_image.shape[1] - (padding_size - left)
        cropped_image = resized_square_image[:, left:right, :]
    return cropped_image


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
        hair_mask_np = cv2.resize(hair_mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    if use_multiscale:
        blend_mask = create_hierarchical_blend_mask(
            (hair_mask_np * 255).astype(np.uint8),
            feather_px=feather_px,
        )
        composited = multi_scale_blend(bald_np, hair_restored_np, blend_mask)
    else:
        blend_mask = create_distance_soft_blend_mask(
            (hair_mask_np * 255).astype(np.uint8),
            feather_px=feather_px,
        )
        mask_3ch = np.stack([blend_mask] * 3, axis=-1)
        composited = (
            bald_np.astype(np.float32) * (1 - mask_3ch) +
            hair_restored_np.astype(np.float32) * mask_3ch
        ).astype(np.uint8)

    return composited


# ============================================================================
# MimicBrush Hair Restoration Pipeline
# ============================================================================

class MimicBrushHairRestorationPipeline:
    """Pipeline for hair transfer using MimicBrush framework."""

    def __init__(
        self,
        config: Optional[MimicBrushHairTransferConfig] = None,
        device: str = "cuda",
    ):
        if config is None:
            config = MimicBrushHairTransferConfig()

        self.config = config
        self.device = device
        self.size = (config.PROCESSING_RESOLUTION, config.PROCESSING_RESOLUTION)

        self._load_models()

    def _load_models(self):
        """Load and initialize MimicBrush model components."""
        print("Loading MimicBrush model...")

        # Initialize depth transform
        self.depth_transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        # Load Depth Anything model
        print("  Loading Depth Anything model...")
        if os.path.exists(self.config.DEPTH_MODEL_PATH):
            depth_anything_model.load_state_dict(
                torch.load(self.config.DEPTH_MODEL_PATH, map_location='cpu')
            )
        else:
            print(f"  Warning: Depth model not found at {self.config.DEPTH_MODEL_PATH}")
            print("  Shape control will be disabled.")

        # Initialize scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        # Load VAE
        print(f"  Loading VAE from {self.config.VAE_MODEL_PATH}...")
        self.vae = AutoencoderKL.from_pretrained(
            self.config.VAE_MODEL_PATH,
            use_safetensors=False,  # MimicBrush uses .bin files
        ).to(dtype=torch.float16)

        # Load UNet with modified input channels (13 = 4 latent + 1 mask + 4 masked_latent + 4 depth)
        print(f"  Loading UNet from {self.config.INPAINTING_MODEL_PATH}...")
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.INPAINTING_MODEL_PATH,
            subfolder="unet",
            in_channels=13,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            use_safetensors=False,  # MimicBrush uses .bin files
        ).to(dtype=torch.float16)

        # Create MimicBrush pipeline
        print(f"  Creating MimicBrush pipeline from {self.config.INPAINTING_MODEL_PATH}...")
        self.pipe = MimicBrushPipeline.from_pretrained(
            self.config.INPAINTING_MODEL_PATH,
            torch_dtype=torch.float16,
            scheduler=self.noise_scheduler,
            vae=self.vae,
            unet=self.unet,
            feature_extractor=None,
            safety_checker=None,
            use_safetensors=False,  # MimicBrush uses .bin files
        )

        # Load DepthGuider
        print("  Loading DepthGuider...")
        self.depth_guider = DepthGuider()

        # Load ReferenceNet
        print(f"  Loading ReferenceNet from {self.config.REFERENCE_MODEL_PATH}...")
        self.referencenet = ReferenceNet.from_pretrained(
            self.config.REFERENCE_MODEL_PATH,
            subfolder="unet",
            use_safetensors=False,  # MimicBrush uses .bin files
        ).to(dtype=torch.float16)

        # Create full MimicBrush model with RefNet
        print(f"  Loading MimicBrush checkpoint from {self.config.MIMICBRUSH_CKPT}...")
        self.mimicbrush_model = MimicBrush_RefNet(
            self.pipe,
            self.config.IMAGE_ENCODER_PATH,
            self.config.MIMICBRUSH_CKPT,
            depth_anything_model,
            self.depth_guider,
            self.referencenet,
            self.device,
        )

        # Mask processor for preprocessing
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=1,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

        print("MimicBrush pipeline initialized successfully!")

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

    def _infer_single(
        self,
        ref_image: np.ndarray,
        target_image: np.ndarray,
        target_mask: np.ndarray,
        seed: int = -1,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        enable_shape_control: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run MimicBrush inference on a single image.

        Args:
            ref_image: Reference image (hair donor) as RGB numpy array
            target_image: Target image (bald) as RGB numpy array
            target_mask: Binary mask (0/1) indicating where to inpaint
            seed: Random seed (-1 for random)
            num_inference_steps: Number of DDIM steps
            guidance_scale: Classifier-free guidance scale
            enable_shape_control: Use depth for shape preservation

        Returns:
            Tuple of (generated image, depth prediction) as numpy arrays
        """
        ref_image = ref_image.astype(np.uint8)
        target_image = target_image.astype(np.uint8)
        target_mask = target_mask.astype(np.uint8)

        # Prepare reference image (pad to square)
        ref_image_pil = Image.fromarray(ref_image)
        ref_image_pil = pad_img_to_square(ref_image_pil)

        # Prepare target image (pad to square)
        target_image_pil = pad_img_to_square(Image.fromarray(target_image))
        target_image_low = target_image_pil

        # Prepare mask (expand to 3 channels, pad to square)
        target_mask_3ch = np.stack([target_mask, target_mask, target_mask], -1).astype(np.uint8) * 255
        target_mask_pil = Image.fromarray(target_mask_3ch)
        target_mask_pil = pad_img_to_square(target_mask_pil, is_mask=True)

        # Store original target for depth computation
        target_image_ori = target_image_pil.copy()

        # Create collage (black out masked region)
        target_image_pil = collage_region(target_image_low, target_image_pil, target_mask_pil)

        # Prepare depth image
        depth_image = np.array(target_image_ori)
        depth_image = self.depth_transform({'image': depth_image})['image']
        depth_image = torch.from_numpy(depth_image).unsqueeze(0) / 255

        # Disable shape control if requested
        if not enable_shape_control:
            depth_image = depth_image * 0

        # Preprocess mask for pipeline
        mask_pt = self.mask_processor.preprocess(
            target_mask_pil, 
            height=self.config.PROCESSING_RESOLUTION, 
            width=self.config.PROCESSING_RESOLUTION
        )

        # Generate
        if seed == -1:
            seed = np.random.randint(10000)

        pred, depth_pred = self.mimicbrush_model.generate(
            pil_image=ref_image_pil,
            depth_image=depth_image,
            num_samples=1,
            num_inference_steps=num_inference_steps,
            seed=seed,
            image=target_image_pil,
            mask_image=mask_pt,
            strength=1.0,
            guidance_scale=guidance_scale,
        )

        # Post-process depth prediction
        depth_pred = F.interpolate(
            depth_pred, 
            size=(self.config.PROCESSING_RESOLUTION, self.config.PROCESSING_RESOLUTION), 
            mode='bilinear', 
            align_corners=True
        )[0][0]
        depth_pred = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min()) * 255.0
        depth_pred = depth_pred.detach().cpu().numpy().astype(np.uint8)
        depth_pred = cv2.applyColorMap(depth_pred, cv2.COLORMAP_INFERNO)[:, :, ::-1]

        pred = pred[0]
        pred = np.array(pred).astype(np.uint8)

        return pred, depth_pred.astype(np.uint8)

    def transfer_hair(
        self,
        source_image_path: str,
        source_mask_path: str,
        reference_image_path: str,
        reference_mask_path: str,
        output_dir: str = "output",
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        enable_shape_control: Optional[bool] = None,
    ) -> str:
        """
        Transfer hair from reference to source (bald) image using MimicBrush.

        In MimicBrush for hair transfer:
        - Reference: Hair donor image (the hair we want to transfer)
        - Target: Bald person image with mask indicating where hair should go
        - The model inpaints the masked region using reference appearance

        Args:
            source_image_path: Path to source (bald) image
            source_mask_path: Path to target mask (where hair should go)
            reference_image_path: Path to reference (hair donor) image
            reference_mask_path: Path to reference hair mask
            output_dir: Output directory
            seed: Random seed
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of DDIM steps
            enable_shape_control: Use depth for shape preservation

        Returns:
            Path to output image
        """
        if seed is None:
            seed = self.config.SEED
        if guidance_scale is None:
            guidance_scale = self.config.GUIDANCE_SCALE
        if num_inference_steps is None:
            num_inference_steps = self.config.NUM_INFERENCE_STEPS
        if enable_shape_control is None:
            enable_shape_control = self.config.ENABLE_SHAPE_CONTROL

        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"Processing with seed={seed}, guidance_scale={guidance_scale}, steps={num_inference_steps}")
        print("Loading and preparing images...")

        # Load images
        source_pil = Image.open(source_image_path).convert("RGB")
        reference_pil = Image.open(reference_image_path).convert("RGB")
        reference_mask_pil = Image.open(reference_mask_path).convert("L")
        source_mask_pil = Image.open(source_mask_path).convert("L")

        # Store original size for later
        original_size = source_pil.size

        # Convert to numpy arrays
        tar_image = np.asarray(source_pil)
        ref_image = np.asarray(reference_pil)
        tar_mask = np.asarray(source_mask_pil)
        tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)
        ref_mask = np.asarray(reference_mask_pil)
        ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

        # Dilate target mask to ensure full hair coverage (same as InsertAnything)
        kernel = np.ones((3, 3), np.uint8)
        tar_mask = cv2.dilate(tar_mask, kernel, iterations=1)

        if tar_mask.sum() == 0:
            raise ValueError("No mask for the target (bald) image!")
        if ref_mask.sum() == 0:
            raise ValueError("No mask for the reference (hair) image!")

        print("Running MimicBrush inference...")

        # Run inference - pass reference image without masking
        pred, depth_pred = self._infer_single(
            ref_image=ref_image,
            target_image=tar_image,
            target_mask=tar_mask,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            enable_shape_control=enable_shape_control,
        )

        # Crop back to original aspect ratio
        pred = crop_padding_and_resize(tar_image, pred)
        depth_pred = crop_padding_and_resize(tar_image, depth_pred)

        # Ensure output matches original size
        if pred.shape[:2] != (original_size[1], original_size[0]):
            pred = cv2.resize(pred, original_size, interpolation=cv2.INTER_LANCZOS4)

        # Gaussian blur on mask for soft blending edge
        tar_mask_blend = np.stack([tar_mask, tar_mask, tar_mask], -1).astype(np.uint8) * 255
        for _ in range(10):
            tar_mask_blend = cv2.GaussianBlur(tar_mask_blend, (3, 3), 0)
        mask_alpha = tar_mask_blend / 255.0

        # Resize mask if needed
        if mask_alpha.shape[:2] != pred.shape[:2]:
            mask_alpha = cv2.resize(mask_alpha, (pred.shape[1], pred.shape[0]))

        # Blend: use generated content in masked region, original elsewhere
        synthesis = pred * mask_alpha + tar_image * (1 - mask_alpha)
        synthesis = synthesis.astype(np.uint8)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Save output
        output_path = os.path.join(output_dir, self.config.FILE_HAIR_RESTORED)
        Image.fromarray(synthesis).save(output_path)
        print(f"Saved hair restored image to {output_path}")

        # Save depth visualization
        depth_output_path = os.path.join(output_dir, "depth_pred.png")
        Image.fromarray(depth_pred).save(depth_output_path)

        # Also save the raw generated image before blending
        raw_output_path = os.path.join(output_dir, "raw_generated.png")
        Image.fromarray(pred).save(raw_output_path)

        return output_path

    def __del__(self):
        """Cleanup when pipeline is deleted."""
        if hasattr(self, 'mimicbrush_model'):
            del self.mimicbrush_model
        if hasattr(self, 'pipe'):
            del self.pipe
        if hasattr(self, 'vae'):
            del self.vae
        if hasattr(self, 'unet'):
            del self.unet
        if hasattr(self, 'referencenet'):
            del self.referencenet
        if hasattr(self, 'depth_guider'):
            del self.depth_guider
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
        raise RuntimeError("SAMMaskExtractor is not available.")

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
    config: MimicBrushHairTransferConfig,
    use_3d_aware: bool,
) -> Path:
    """Get output directory for MimicBrush results."""
    mode_dir = config.DIR_3D_AWARE if use_3d_aware else config.DIR_3D_UNAWARE
    return pair_dir / mode_dir / config.SUBDIR_TRANSFERRED


# ============================================================================
# Batch Processing
# ============================================================================

def process_sample(
    folder: Path,
    pipeline: MimicBrushHairRestorationPipeline,
    data_dir: Path,
    config: MimicBrushHairTransferConfig,
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
    config: Optional[MimicBrushHairTransferConfig] = None,
    skip_existing: bool = True,
    bald_version: str = "w_seg",
    conditioning_mode: str = "3d_aware",
) -> Dict[str, int]:
    """Process all view-aligned folders in batch."""
    if config is None:
        config = MimicBrushHairTransferConfig()

    data_dir = Path(data_dir)
    provider_subdir = f"shape_{shape_provider}__texture_{texture_provider}"
    view_aligned_dir = data_dir / config.DIR_VIEW_ALIGNED / provider_subdir

    if not view_aligned_dir.exists():
        raise FileNotFoundError(f"View aligned directory not found: {view_aligned_dir}")

    bald_versions = ["w_seg", "wo_seg"] if bald_version == "all" else [bald_version]
    # MimicBrush only supports 3D-aware mode (requires warped outputs)
    conditioning_modes = ["3d_aware"]

    print(f"\nConfiguration:")
    print(f"  bald_version(s): {bald_versions}")
    print(f"  conditioning_mode: 3d_aware (only mode supported)")

    all_folders = [f for f in view_aligned_dir.iterdir() if f.is_dir()]

    if not all_folders:
        print(f"No folders found in {view_aligned_dir}")
        return {"processed": 0, "skipped": 0, "errors": 0}

    timestamp_seed = int(time.time())
    random.seed(timestamp_seed)
    random.shuffle(all_folders)
    print(f"Found {len(all_folders)} samples (shuffle seed: {timestamp_seed})")

    print("\n" + "=" * 60)
    print("Initializing MimicBrush Hair Restoration Pipeline...")
    print("=" * 60)
    pipeline = MimicBrushHairRestorationPipeline(config)

    overall_success = 0
    overall_total = 0

    for bv in bald_versions:
        for cm in conditioning_modes:
            print(f"\n{'=' * 60}")
            print(f"Processing: bald_version={bv}, conditioning_mode={cm}")
            print(f"{'=' * 60}")

            success_count = 0
            for folder in tqdm(all_folders, desc=f"{bv}/{cm}"):
                result = process_sample(
                    folder=folder,
                    pipeline=pipeline,
                    data_dir=data_dir,
                    config=config,
                    bald_version=bv,
                    conditioning_mode=cm,
                    skip_existing=skip_existing,
                )
                if result:
                    success_count += 1

            overall_success += success_count
            overall_total += len(all_folders)
            print(f"Completed {success_count}/{len(all_folders)} for {bv}/{cm}")

    print(f"\n{'=' * 60}")
    print(f"✓ All processing complete! {overall_success}/{overall_total} total samples processed")
    print(f"{'=' * 60}")

    return {"processed": overall_success, "skipped": 0, "errors": overall_total - overall_success}


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main entry point for MimicBrush-based hair restoration pipeline."""
    parser = argparse.ArgumentParser(
        description="Hair Restoration Pipeline using MimicBrush Framework"
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
    parser.add_argument("--output", type=str, default="output", help="Output directory")

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
        choices=["3d_aware"],
        help="Conditioning mode (only 3d_aware supported, requires warped outputs)",
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

    # MimicBrush parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale (default: 5.0)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help="Number of DDIM steps (default: 50)",
    )
    parser.add_argument(
        "--enable_shape_control",
        action="store_true",
        default=True,
        help="Enable shape control using depth (preserve head shape)",
    )
    parser.add_argument(
        "--no_shape_control",
        action="store_false",
        dest="enable_shape_control",
        help="Disable shape control",
    )

    args = parser.parse_args()

    # Create config
    config = MimicBrushHairTransferConfig()
    config.SEED = args.seed
    config.GUIDANCE_SCALE = args.guidance_scale
    config.NUM_INFERENCE_STEPS = args.num_steps
    config.ENABLE_SHAPE_CONTROL = args.enable_shape_control

    if args.mode == "single":
        if not args.source or not args.reference:
            parser.error("Single mode requires --source and --reference")

        # Check for masks
        if not args.tar_mask:
            parser.error("Single mode requires --tar_mask (target/source mask)")
        if not args.ref_mask:
            parser.error("Single mode requires --ref_mask (reference hair mask)")

        pipeline = MimicBrushHairRestorationPipeline(config)
        output_path = pipeline.transfer_hair(
            source_image_path=args.source,
            source_mask_path=args.tar_mask,
            reference_image_path=args.reference,
            reference_mask_path=args.ref_mask,
            output_dir=args.output,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            enable_shape_control=args.enable_shape_control,
        )
        
        print(30*"=")
        print(f"\nOutput saved to: {output_path}")
        print(30*"=")

    else:
        # Batch mode
        process_view_aligned_folders(
            data_dir=args.data_dir,
            shape_provider=args.shape_provider,
            texture_provider=args.texture_provider,
            config=config,
            skip_existing=args.skip_existing,
            bald_version=args.bald_version,
            conditioning_mode=args.conditioning_mode,
        )


if __name__ == "__main__":
    main()
