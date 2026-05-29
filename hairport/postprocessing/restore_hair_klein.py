"""Hair transfer using FLUX.2 Klein 9B with 3D-aware and 3D-unaware modes."""

import argparse
import gc
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffusers import Flux2KleinPipeline
from rembg import remove, new_session

from hairport.core import SAMMaskExtractor
from hairport.utility.uncrop_sdxl import Uncropper
from easy_dwpose import DWposeDetector
from hairport.config import get_config
from hairport.data import DatasetManager
from hairport.runtime import build_provenance, can_reuse_artifact, derive_seed, write_provenance


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class HairTransferKleinConfig:
    # Model settings
    FLUX_KLEIN_MODEL: str | None = None

    # Processing resolution
    PROCESSING_RESOLUTION: int | None = None
    OUTPUT_RESOLUTION: int | None = None
    
    # Generation parameters (optimized for distilled model)
    SEED: int | None = None
    GUIDANCE_SCALE: float | None = None
    NUM_INFERENCE_STEPS: int | None = None
    
    # Directory structure
    DIR_VIEW_ALIGNED: str | None = None
    DIR_ALIGNMENT: str | None = None
    DIR_BALD: str | None = None
    DIR_PROMPTS: str | None = None
    
    # Top-level directories for 3D aware/unaware processing
    DIR_3D_AWARE: str | None = None
    DIR_3D_UNAWARE: str | None = None
    
    # Subdirectories
    SUBDIR_WARPING: str | None = None
    SUBDIR_TRANSFERRED: str | None = None
    
    # Redux image source directories (in priority order)
    DIR_HAIR_ALIGNED_IMAGE: str = "image"
    DIR_IMAGE: str = "image"
    DIR_IMAGE_OUTPAINTED: str = "image_outpainted"
    
    # File names
    FILE_VIEW_ALIGNED_IMAGE: str | None = None
    FILE_VIEW_ALIGNED_MASK: str | None = None

    # Blending file (alternative to masked view-aligned)
    SUBDIR_BLENDING: str | None = None
    FILE_POISSON_BLENDED: str | None = None
    
    # Output file naming
    FILE_HAIR_RESTORED: str | None = None
    FILE_HAIR_RESTORED_MASK: str | None = None
    
    # Source paths
    DIR_SOURCE_OUTPAINTED: str | None = None
    FILE_OUTPAINTED_IMAGE: str = "outpainted_image.png"
    SUBDIR_BALD_IMAGE: str | None = None
    
    # Masking colors
    BACKGROUND_COLOR: Tuple[int, int, int] | None = None
    NON_HAIR_FOREGROUND_COLOR: Tuple[int, int, int] | None = None
    
    # Uncropping settings for 3D-aware mode
    UNCROP_HAIR_THRESHOLD: float | None = None
    UNCROP_BORDER_THRESHOLD: float | None = None
    UNCROP_RESIZE_PERCENTAGE: float | None = None
    UNCROP_PROMPT: str | None = None

    def __post_init__(self):
        cfg = get_config()
        th = cfg.transfer_hair
        ds = cfg.dataset
        if self.FLUX_KLEIN_MODEL is None:
            self.FLUX_KLEIN_MODEL = cfg.models.flux_klein
        if self.PROCESSING_RESOLUTION is None:
            self.PROCESSING_RESOLUTION = th.processing_resolution
        if self.OUTPUT_RESOLUTION is None:
            self.OUTPUT_RESOLUTION = th.output_resolution
        if self.SEED is None:
            self.SEED = cfg.seed
        if self.GUIDANCE_SCALE is None:
            self.GUIDANCE_SCALE = th.guidance_scale
        if self.NUM_INFERENCE_STEPS is None:
            self.NUM_INFERENCE_STEPS = th.num_inference_steps
        if self.DIR_VIEW_ALIGNED is None:
            self.DIR_VIEW_ALIGNED = ds.dir_view_aligned
        if self.DIR_ALIGNMENT is None:
            self.DIR_ALIGNMENT = ds.subdir_alignment
        if self.DIR_BALD is None:
            self.DIR_BALD = ds.dir_bald
        if self.DIR_PROMPTS is None:
            self.DIR_PROMPTS = ds.dir_prompts
        if self.DIR_3D_AWARE is None:
            self.DIR_3D_AWARE = ds.dir_3d_aware
        if self.DIR_3D_UNAWARE is None:
            self.DIR_3D_UNAWARE = ds.dir_3d_unaware
        if self.SUBDIR_WARPING is None:
            self.SUBDIR_WARPING = ds.subdir_warping
        if self.SUBDIR_TRANSFERRED is None:
            self.SUBDIR_TRANSFERRED = ds.subdir_transferred
        if self.FILE_VIEW_ALIGNED_IMAGE is None:
            phase = int(cfg.enhance_view.conditioning_phase)
            self.FILE_VIEW_ALIGNED_IMAGE = f"target_image_phase_{phase}.png"
        if self.FILE_VIEW_ALIGNED_MASK is None:
            phase = int(cfg.enhance_view.conditioning_phase)
            self.FILE_VIEW_ALIGNED_MASK = f"target_image_phase_{phase}_mask.png"
        if self.SUBDIR_BLENDING is None:
            self.SUBDIR_BLENDING = ds.subdir_blending
        if self.FILE_POISSON_BLENDED is None:
            self.FILE_POISSON_BLENDED = ds.file_poisson_blended
        if self.FILE_HAIR_RESTORED is None:
            self.FILE_HAIR_RESTORED = ds.file_hair_restored
        if self.FILE_HAIR_RESTORED_MASK is None:
            self.FILE_HAIR_RESTORED_MASK = ds.file_hair_restored_mask
        if self.DIR_SOURCE_OUTPAINTED is None:
            self.DIR_SOURCE_OUTPAINTED = ds.dir_source_outpainted
        if self.SUBDIR_BALD_IMAGE is None:
            self.SUBDIR_BALD_IMAGE = ds.subdir_bald_image
        if self.BACKGROUND_COLOR is None:
            self.BACKGROUND_COLOR = tuple(th.bg_color)
        if self.NON_HAIR_FOREGROUND_COLOR is None:
            self.NON_HAIR_FOREGROUND_COLOR = tuple(th.non_hair_fg_color)
        if self.UNCROP_HAIR_THRESHOLD is None:
            self.UNCROP_HAIR_THRESHOLD = th.uncrop_hair_threshold
        if self.UNCROP_BORDER_THRESHOLD is None:
            self.UNCROP_BORDER_THRESHOLD = th.uncrop_border_threshold
        if self.UNCROP_RESIZE_PERCENTAGE is None:
            self.UNCROP_RESIZE_PERCENTAGE = th.uncrop_resize_percentage
        if self.UNCROP_PROMPT is None:
            self.UNCROP_PROMPT = cfg.prompts.transfer_uncrop


# ============================================================================
# Prompts
# ============================================================================

# Prompts are read from centralized config at runtime via get_config().prompts.*

def _prompt_3d_aware():
    return get_config().prompts.transfer_3d_aware

def _prompt_3d_aware_wo_bald():
    return get_config().prompts.transfer_3d_aware_wo_bald

def _prompt_3d_unaware():
    return get_config().prompts.transfer_3d_unaware

# ============================================================================
# Utility Functions
# ============================================================================

def flush_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    torch.cuda.empty_cache()

from rembg import remove, new_session

class BackgroundRemover:
    def __init__(self, device: str | torch.device | None = None):
        if device is None:
            device = get_config().device
        if not torch.cuda.is_available() and "cuda" in str(device):
            device = "cpu"
        self.device = torch.device(device)
        self._session = new_session(get_config().models.rembg_session)
    
    def remove_background(self, image: Image.Image, refine_foreground: bool = False) -> tuple[Image.Image, Image.Image]:
        foreground = remove(image, session=self._session)
        alpha = foreground.getchannel('A')
        mask = ((np.array(alpha) / 255.0) > 0.8).astype(np.uint8) * 255
        return foreground, Image.fromarray(mask)

class DWPoseSingleton:
    _instance: Optional[DWposeDetector] = None
    _device: str = "cuda"
    
    @classmethod
    def get_instance(
        cls,
        device: str | None = None,
        force_reload: bool = False,
    ) -> DWposeDetector:
        if device is None:
            device = get_config().device
        if not torch.cuda.is_available() and "cuda" in str(device):
            device = "cpu"
        device = str(device)
        if cls._instance is None or force_reload or cls._device != device:
            if cls._instance is not None:
                del cls._instance
                cls._instance = None
                flush_gpu_memory()
            
            cls._instance = DWposeDetector(device=device)
            cls._device = device
            print(f"[DWPoseSingleton] Initialized on {device}")
        
        return cls._instance
    
    @classmethod
    def release(cls):
        if cls._instance is not None:
            del cls._instance
            cls._instance = None
            flush_gpu_memory()
            print("[DWPoseSingleton] Released")


class UncropperSingleton:
    _uncropper: Optional[Uncropper] = None
    
    @classmethod
    def get_uncropper(
        cls,
        force_reload: bool = False,
    ) -> Uncropper:
        if cls._uncropper is None or force_reload:
            if cls._uncropper is not None:
                del cls._uncropper
                cls._uncropper = None
                flush_gpu_memory()
            
            from hairport.utility.uncrop_sdxl.uncrop_sdxl import Uncropper, UncropperConfig
            uncrop_cfg = get_config().uncrop
            config = UncropperConfig(
                width=uncrop_cfg.width,
                height=uncrop_cfg.height,
                alignment="Middle",
                overlap_percentage=uncrop_cfg.overlap_percentage,
                num_inference_steps=uncrop_cfg.num_inference_steps,
            )
            cls._uncropper = Uncropper(config)
            cls._uncropper.load_pipeline()
            print("[UncropperSingleton] SDXL Uncropper initialized")
        
        return cls._uncropper
    
    @classmethod
    def release(cls):
        if cls._uncropper is not None:
            del cls._uncropper
            cls._uncropper = None
            flush_gpu_memory()
            print("[UncropperSingleton] SDXL Uncropper released")


def compute_mask_bounding_box(
    mask: Image.Image,
) -> Tuple[int, int, int, int]:
    mask_np = np.array(mask)
    if mask_np.ndim == 3:
        mask_np = mask_np[:, :, 0]
    
    # Find non-zero pixels
    rows = np.any(mask_np > 127, axis=1)
    cols = np.any(mask_np > 127, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # Empty mask
        return (0, 0, 0, 0)
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    
    return (int(x_min), int(y_min), int(width), int(height))


def should_uncrop_for_3d_aware(
    view_aligned_hair_mask: Image.Image,
    processing_resolution: int,
    size_threshold: float = 0.60,
    border_threshold: float = 0.10,
) -> bool:
    x, y, w, h = compute_mask_bounding_box(view_aligned_hair_mask)
    
    if w == 0 or h == 0:
        return False
    
    # Check 1: Hair bbox size exceeds threshold
    size_threshold_pixels = processing_resolution * size_threshold
    exceeds_size = (w > size_threshold_pixels) or (h > size_threshold_pixels)
    
    # Check 2: Hair mask is too close to any border
    border_threshold_pixels = processing_resolution * border_threshold
    dist_left = x
    dist_top = y
    dist_right = processing_resolution - (x + w)
    dist_bottom = processing_resolution - (y + h)
    min_border_dist = min(dist_left, dist_top, dist_right, dist_bottom)
    too_close_to_border = min_border_dist < border_threshold_pixels
    
    needs_uncrop = exceeds_size or too_close_to_border
    
    if needs_uncrop:
        reasons = []
        if exceeds_size:
            reasons.append(f"size {w}x{h} > {size_threshold_pixels:.0f}px")
        if too_close_to_border:
            reasons.append(f"border dist {min_border_dist:.0f}px < {border_threshold_pixels:.0f}px")
        print(f"    Hair mask bbox: {w}x{h} at ({x},{y}) -> UNCROP NEEDED ({', '.join(reasons)})")
    else:
        print(f"    Hair mask bbox: {w}x{h} at ({x},{y}), border dist {min_border_dist:.0f}px -> no uncrop needed")
    
    return needs_uncrop


class BackgroundRemoverSingleton:
    _instance = None
    _session = None
    _device: str | None = None
    
    @classmethod
    def get_instance(cls, device: str | None = None, force_reload: bool = False):
        if device is None:
            device = get_config().device
        if not torch.cuda.is_available() and "cuda" in str(device):
            device = "cpu"
        if cls._instance is None or force_reload or cls._device != str(device):
            if cls._instance is not None:
                del cls._instance
                cls._instance = None
                flush_gpu_memory()
            
            cls._session = BackgroundRemover(device=device)
            cls._instance = cls
            cls._device = str(device)
            print(f"[BackgroundRemoverSingleton] Initialized on {device}")
        
        return cls._instance
    
    @classmethod
    def remove_background(cls, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if cls._session is None:
            cls.get_instance()
        
        foreground, mask = cls._session.remove_background(image.convert("RGB"))
        return foreground, mask
    
    @classmethod
    def release(cls):
        if cls._instance is not None:
            cls._session = None
            cls._instance = None
            cls._device = None
            flush_gpu_memory()
            print("[BackgroundRemoverSingleton] Released")


class SAMMaskExtractorSingleton:
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
        thresholds_changed = (
            cls._confidence_threshold != confidence_threshold or
            cls._detection_threshold != detection_threshold
        )
        
        if cls._instance is None or force_reload or thresholds_changed:
            if cls._instance is not None:
                del cls._instance
                cls._instance = None
                flush_gpu_memory()
            
            cls._instance = SAMMaskExtractor(
                confidence_threshold=confidence_threshold,
                detection_threshold=detection_threshold
            )
            cls._confidence_threshold = confidence_threshold
            cls._detection_threshold = detection_threshold
            print(f"[SAMMaskExtractorSingleton] Initialized")
        
        return cls._instance
    
    @classmethod
    def release(cls):
        """Release resources."""
        if cls._instance is not None:
            del cls._instance
            cls._instance = None
            flush_gpu_memory()
            print("[SAMMaskExtractorSingleton] Released")
    
    @classmethod
    def is_available(cls) -> bool:
        return SAMMaskExtractor is not None


def extract_hair_mask(
    image: Image.Image,
    confidence_threshold: float = 0.4,
    detection_threshold: float = 0.5,
    primary_prompt: str = "head hair",
    secondary_prompt: str = "hair tie, hair clip, headband",
) -> Tuple[Image.Image, float]:
    sam = SAMMaskExtractorSingleton.get_instance(
        confidence_threshold=confidence_threshold,
        detection_threshold=detection_threshold
    )
    
    _, silh_mask_pil = BackgroundRemoverSingleton.remove_background(image)
    silh_mask_np = np.array(silh_mask_pil).astype(np.float32) / 255.0
    
    hair_mask_pil, primary_score = sam(image, prompt=primary_prompt)
    hair_mask_np = np.array(hair_mask_pil).astype(np.float32) / 255.0
    
    try:
        accessories_mask_pil, accessories_score = sam(image, prompt=secondary_prompt)
        accessories_mask_np = np.array(accessories_mask_pil).astype(np.float32) / 255.0
        combined_mask_np = np.maximum(hair_mask_np, accessories_mask_np)
    except Exception as e:
        combined_mask_np = hair_mask_np
    constrained_mask = (combined_mask_np * silh_mask_np * 255.0).astype(np.uint8)
    
    return Image.fromarray(constrained_mask), primary_score


def prepare_hair_only_image(
    image: Image.Image,
    hair_mask: Optional[Image.Image] = None,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    non_hair_color: Tuple[int, int, int] = (180, 180, 180),
    target_size: Tuple[int, int] = (1024, 1024),
    include_body: bool = True,
    device: str | None = None,
) -> Image.Image:
    """Layers: white bg -> gray non-hair -> skeleton -> hair (color)."""
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    _, silhouette_mask = BackgroundRemoverSingleton.remove_background(image)
    if silhouette_mask.size != target_size:
        silhouette_mask = silhouette_mask.resize(target_size, Image.Resampling.NEAREST)
    
    if hair_mask is None:
        hair_mask, _ = extract_hair_mask(image)
    else:
        hair_mask = hair_mask.convert("L")
    if hair_mask.size != target_size:
        hair_mask = hair_mask.resize(target_size, Image.Resampling.NEAREST)
    
    dwpose_detector = DWPoseSingleton.get_instance(device=device)
    pose_result = dwpose_detector(image, include_hands=False, include_body=include_body, detect_resolution=1024)
    
    if isinstance(pose_result, tuple):
        pose_image = pose_result[0]
    else:
        pose_image = pose_result
    
    if isinstance(pose_image, np.ndarray):
        pose_image = Image.fromarray(pose_image)
    
    if pose_image.size != target_size:
        pose_image = pose_image.resize(target_size, Image.Resampling.BILINEAR)
    image_np = np.array(image).astype(np.float32)
    pose_np = np.array(pose_image.convert("RGB")).astype(np.float32)
    silhouette_np = np.array(silhouette_mask).astype(np.float32) / 255.0
    hair_np = np.array(hair_mask).astype(np.float32) / 255.0
    hair_np = (hair_np > 0.5).astype(np.float32)
    output_np = np.full_like(image_np, background_color, dtype=np.float32)
    
    non_hair_fg_mask = silhouette_np * (1.0 - hair_np)
    non_hair_color_np = np.array(non_hair_color, dtype=np.float32)
    for c in range(3):
        output_np[:, :, c] = (
            output_np[:, :, c] * (1.0 - non_hair_fg_mask) +
            non_hair_color_np[c] * non_hair_fg_mask
        )
    
    pose_mask = np.any(pose_np > 10, axis=2).astype(np.float32)
    pose_mask_3ch = pose_mask[:, :, np.newaxis]
    output_np = output_np * (1.0 - pose_mask_3ch) + pose_np * pose_mask_3ch
    
    hair_mask_3ch = hair_np[:, :, np.newaxis]
    output_np = output_np * (1.0 - hair_mask_3ch) + image_np * hair_mask_3ch
    
    return Image.fromarray(output_np.astype(np.uint8))


def prepare_bald_source_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (1024, 1024),
    apply_matting: bool = True,
) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    if image.size != target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    if apply_matting:
        foreground, mask = BackgroundRemoverSingleton.remove_background(image)
        white_bg = Image.new("RGBA", target_size, (255, 255, 255, 255))
        result = Image.alpha_composite(white_bg, foreground)
        return result.convert("RGB")
    
    return image


# ============================================================================
# Hair Transfer Pipeline
# ============================================================================

class HairTransferKleinPipeline:
    def __init__(
        self,
        config: Optional[HairTransferKleinConfig] = None,
        device: str | None = None,
    ):
        if config is None:
            config = HairTransferKleinConfig()
        
        self.config = config
        self.device = device or get_config().device
        if not torch.cuda.is_available() and "cuda" in str(self.device):
            self.device = "cpu"
        self.dtype = torch.bfloat16
        self.pipe = None
        
        self._load_models()
    
    def _load_models(self):
        print(f"Loading FLUX.2 Klein from {self.config.FLUX_KLEIN_MODEL}...")
        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.config.FLUX_KLEIN_MODEL,
            revision=get_config().models.flux_klein_revision,
            torch_dtype=self.dtype,
        )
        self.pipe.to(self.device)
        BackgroundRemoverSingleton.get_instance(device=self.device)
        
        print("Pipeline initialized successfully!")
    
    def transfer_hair(
        self,
        source_bald_image: Union[str, Path, Image.Image],
        reference_image: Union[str, Path, Image.Image],
        view_aligned_image: Optional[Union[str, Path, Image.Image]] = None,
        reference_hair_mask: Optional[Union[str, Path, Image.Image]] = None,
        view_aligned_hair_mask: Optional[Union[str, Path, Image.Image]] = None,
        use_3d_aware: bool = False,
        conditioning_source: str = "enhanced",
        output_dir: Optional[str] = None,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
    ) -> Image.Image:
        if seed is None:
            seed = self.config.SEED
        if seed is None or seed < 0:
            seed = int(time.time())
        if num_inference_steps is None:
            num_inference_steps = self.config.NUM_INFERENCE_STEPS
        if guidance_scale is None:
            guidance_scale = self.config.GUIDANCE_SCALE
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator(self.device).manual_seed(seed)
        
        target_size = (self.config.PROCESSING_RESOLUTION, self.config.PROCESSING_RESOLUTION)
        if isinstance(source_bald_image, (str, Path)):
            source_bald_image = Image.open(source_bald_image).convert("RGB")
        if isinstance(reference_image, (str, Path)):
            reference_image = Image.open(reference_image).convert("RGB")
        if view_aligned_image is not None and isinstance(view_aligned_image, (str, Path)):
            view_aligned_image = Image.open(view_aligned_image).convert("RGB")
        
        if reference_hair_mask is not None and isinstance(reference_hair_mask, (str, Path)):
            reference_hair_mask = Image.open(reference_hair_mask).convert("L")
        if view_aligned_hair_mask is not None and isinstance(view_aligned_hair_mask, (str, Path)):
            view_aligned_hair_mask = Image.open(view_aligned_hair_mask).convert("L")
        
        if conditioning_source not in {"enhanced", "blended"}:
            raise ValueError(f"Unknown conditioning_source: {conditioning_source!r}")
        print(f"Mode: {'3D-Aware' if use_3d_aware else '3D-Unaware'} ({conditioning_source})")
        print(f"Seed: {seed}, Steps: {num_inference_steps}, CFG: {guidance_scale}")
        needs_uncrop = False
        resize_info = None
        original_bald_size = source_bald_image.size
        
        if use_3d_aware:
            va_mask_for_check = view_aligned_hair_mask
            if isinstance(va_mask_for_check, (str, Path)):
                va_mask_for_check = Image.open(va_mask_for_check).convert("L")

            if isinstance(va_mask_for_check, Image.Image):
                needs_uncrop = should_uncrop_for_3d_aware(
                    va_mask_for_check,
                    self.config.PROCESSING_RESOLUTION,
                    size_threshold=self.config.UNCROP_HAIR_THRESHOLD,
                    border_threshold=self.config.UNCROP_BORDER_THRESHOLD,
                )
            else:
                print("    No view-aligned hair mask available; skipping uncrop check.")
                needs_uncrop = False
            
            if needs_uncrop:
                print("  Uncropping source bald image using SDXL...")
                uncropper = UncropperSingleton.get_uncropper()
                
                uncropped_bald, resize_info = uncropper.uncrop(
                    source_bald_image,
                    prompt=self.config.UNCROP_PROMPT,
                    resize_percentage=self.config.UNCROP_RESIZE_PERCENTAGE,
                )
                source_bald_image = uncropped_bald
                print(f"    Uncropped bald: resize={self.config.UNCROP_RESIZE_PERCENTAGE}%, "
                      f"margin=({resize_info['margin_x']}, {resize_info['margin_y']}), "
                      f"size=({resize_info['new_width']}, {resize_info['new_height']})")
        
        print("  Preparing bald source image...")
        img1_bald = prepare_bald_source_image(source_bald_image, target_size, apply_matting=False)
        
        print("  Preparing reference hair image...")
        img_reference = prepare_hair_only_image(
            reference_image,
            hair_mask=reference_hair_mask,
            background_color=self.config.BACKGROUND_COLOR,
            non_hair_color=self.config.NON_HAIR_FOREGROUND_COLOR,
            target_size=target_size,
            device=self.device,
        )
        
        if use_3d_aware:
            if view_aligned_image is None:
                raise ValueError("view_aligned_image is required when use_3d_aware=True")
            
            print(f"  Preparing {conditioning_source} view-aligned hair image...")
            img_view_aligned = prepare_hair_only_image(
                view_aligned_image,
                hair_mask=view_aligned_hair_mask,
                background_color=self.config.BACKGROUND_COLOR,
                non_hair_color=self.config.NON_HAIR_FOREGROUND_COLOR,
                target_size=target_size,
                device=self.device,
            )
            # img1_bald = Image.open("/workspace/outputs/image/side10.png").convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
            image_list = [img1_bald, img_view_aligned, img_reference]
            prompt = _prompt_3d_aware()
            print(f"  Passing 3 images to FLUX Klein")
        else:
            image_list = [img1_bald, img_reference]
            prompt = _prompt_3d_unaware()
            print(f"  Passing 2 images to FLUX Klein")
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            for i, img in enumerate(image_list):
                debug_path = os.path.join(output_dir, f"input_{i+1}.png")
                img.save(debug_path)
                print(f"  Saved debug: {debug_path}")
        
        print("  Running FLUX.2 Klein inference...")
        result = self.pipe(
            prompt=prompt,
            image=image_list,
            height=target_size[1],
            width=target_size[0],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        output_image = result.images[0]
        # result = self.pipe(
        #     prompt="Match the texture and color of the hair in image 1 as closely as possible to the hair in image 2. Strictly preserve overall hairstyle, hair shape, hair length, hair volume, and hair details from image 1. Only adjust the hair texture and color to closely match image 2, without changing any other attributes of the hair. Keep all non-hair areas unchanged from image 1, including face, body, background, lighting, and photographic style.",
        #     image=[output_image, reference_image],
        #     height=target_size[1],
        #     width=target_size[0],
        #     num_inference_steps=num_inference_steps,
        #     guidance_scale=guidance_scale,
        #     generator=generator,
        # )
        output_image = result.images[0]
        if output_image.mode != "RGB":
            output_image = output_image.convert("RGB")
        
        if needs_uncrop and resize_info is not None:
            print("  Cropping result back to original region...")
            uncropper = UncropperSingleton.get_uncropper()
            if output_dir is not None:
                uncropped_path = os.path.join(output_dir, "hair_restored_uncropped.png")
                output_image.save(uncropped_path)
                print(f"  Saved uncropped: {uncropped_path}")
            
            output_image = uncropper.crop(
                output_image,
                resize_info,
                output_size=original_bald_size,
            )
            print(f"  Cropped back to: {output_image.size}")
        if output_dir is not None:
            output_path = os.path.join(output_dir, self.config.FILE_HAIR_RESTORED)
            output_image.save(output_path)
            print(f"  Saved: {output_path}")
        
        return output_image
    
    def unload(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        BackgroundRemoverSingleton.release()
        SAMMaskExtractorSingleton.release()
        DWPoseSingleton.release()
        UncropperSingleton.release()
        flush_gpu_memory()
        print("Pipeline unloaded")


# ============================================================================
# Batch Processing
# ============================================================================

def process_sample(
    folder: Path,
    pipeline: HairTransferKleinPipeline,
    data_dir: Path,
    config: HairTransferKleinConfig,
    bald_version: str,
    skip_existing: bool = True,
    conditioning_source: str = "enhanced",
    include_3d_unaware: bool = True,
    shape_provider: str = "hi3dgen",
    texture_provider: str = "mvadapter",
) -> Dict[str, bool]:
    results = {
        '3d_aware': False,
        '3d_unaware': False,
        'skipped_3d_aware': False,
        'skipped_3d_unaware': False,
        'error': None,
    }
    folder_name = folder.name
    
    if "_to_" not in folder_name:
        print(f"Skipping {folder_name}: invalid format")
        results["error"] = f"Invalid pair folder format: {folder_name}"
        return results
    
    try:
        target_id, source_id = folder_name.split("_to_")
    except ValueError:
        print(f"Invalid directory name format: {folder_name}")
        results["error"] = f"Invalid pair folder format: {folder_name}"
        return results
    
    dm = DatasetManager(data_dir)
    pair_dir = dm.transfer_dir(
        target_id, source_id, shape_provider, texture_provider
    ) / bald_version
    if not pair_dir.exists():
        print(f"  Pair directory not found: {pair_dir}")
        results["error"] = f"Pair directory not found: {pair_dir}"
        return results
    
    dataset_name = data_dir.name
    
    try:
        source_image_path = dm.source_outpainted_file(
            target_id, source_id, bald_version, shape_provider, texture_provider
        )
        if not source_image_path.exists():
            source_image_path = dm.bald_image(source_id, bald_version)
        
        if not source_image_path.exists():
            print(f"  Source bald image not found")
            results["error"] = f"Source bald image not found for {folder_name}:{bald_version}"
            return results
        
        if dataset_name == "outputs":
            image_folder = config.DIR_HAIR_ALIGNED_IMAGE
        else:
            image_folder = config.DIR_IMAGE_OUTPAINTED
        
        reference_image_path = dm.source_image(target_id, image_folder)
        if not reference_image_path.exists():
            reference_image_path = dm.source_image(target_id, config.DIR_IMAGE)
        
        if not reference_image_path.exists():
            print(f"  Reference image not found for {target_id}")
            results["error"] = f"Reference image not found for {target_id}"
            return results
        alignment_dir = dm.alignment_dir(
            target_id, source_id, bald_version, shape_provider, texture_provider
        )
        view_aligned_image_path = alignment_dir / config.FILE_VIEW_ALIGNED_IMAGE
        view_aligned_mask_path = alignment_dir / config.FILE_VIEW_ALIGNED_MASK
        source_outpainted_dir = dm.source_outpainted_dir(
            target_id, source_id, bald_version, shape_provider, texture_provider
        )
        resize_info_path = source_outpainted_dir / "resize_info.json"
        
        # Blended conditioning is an explicit alternative to the enhanced view.
        blending_image_path = dm.poisson_blended_file(
            target_id, source_id, bald_version, "3d_aware", shape_provider, texture_provider
        )
        
        has_view_aligned = view_aligned_image_path.exists()
        has_blending = blending_image_path.exists()
        has_camera = dm.camera_params_file(
            target_id, source_id, bald_version, shape_provider, texture_provider
        ).exists()
        
        modes_to_run = ['3d_unaware'] if include_3d_unaware else []
        if conditioning_source == "blended" and has_blending:
            modes_to_run.append('3d_aware')
        elif conditioning_source == "enhanced" and has_view_aligned:
            modes_to_run.append('3d_aware')
        elif has_camera:
            results["error"] = (
                f"Required {conditioning_source} 3D-aware conditioning artifact is missing "
                f"for {folder_name}:{bald_version}."
            )
        
        print(f"  Source: {source_image_path}")
        print(f"  Reference: {reference_image_path}")
        if conditioning_source == "blended":
            if has_blending:
                print(f"  Blending: {blending_image_path}")
                print(f"  Modes: 3D-Aware (blending) + 3D-Unaware")
            else:
                print(f"  Blending: Not available")
                print(f"  Modes: 3D-Unaware only")
        elif has_view_aligned:
            print(f"  View-aligned: {view_aligned_image_path}")
            print(f"  Modes: 3D-Aware + 3D-Unaware")
        else:
            print(f"  View-aligned: Not available")
            print(f"  Modes: 3D-Unaware only")
        
        for mode in modes_to_run:
            use_3d_aware = (mode == '3d_aware')
            mode_name = "3D-Aware" if use_3d_aware else "3D-Unaware"
            
            output_path = dm.hair_restored_file(
                target_id,
                source_id,
                bald_version,
                mode,
                conditioning_source if use_3d_aware else None,
                shape_provider,
                texture_provider,
            )
            output_dir = output_path.parent
            va_input_path = (
                blending_image_path if use_3d_aware and conditioning_source == "blended"
                else view_aligned_image_path if use_3d_aware
                else reference_image_path
            )
            item_seed = derive_seed(
                config.SEED, "transfer_hair", folder_name, bald_version,
                conditioning_source if use_3d_aware else "unaware", "generate",
            )
            provenance_inputs = [source_image_path, reference_image_path, va_input_path]
            if resize_info_path.exists():
                provenance_inputs.append(resize_info_path)
            provenance = build_provenance(
                "transfer_hair", seed=item_seed,
                inputs=provenance_inputs,
                metadata={
                    "bald_version": bald_version,
                    "conditioning_source": conditioning_source if use_3d_aware else "unaware",
                    "mode": mode,
                },
            )
            if skip_existing and can_reuse_artifact(output_path, provenance):
                print(f"    [{mode_name}] Output exists, skipping...")
                results[mode] = True
                results[f"skipped_{mode}"] = True
                continue
            
            print(f"    [{mode_name}] Generating...")
            
            os.makedirs(output_dir, exist_ok=True)
            
            va_image = None
            va_mask = None
            if use_3d_aware:
                # Ensure view-aligned mask exists (needed for uncropping check)
                mask_provenance = build_provenance(
                    "transfer_hair_conditioning_mask",
                    seed=item_seed,
                    inputs=[view_aligned_image_path],
                    metadata={"bald_version": bald_version, "folder": folder_name},
                )
                if (view_aligned_image_path.exists()
                        and not can_reuse_artifact(view_aligned_mask_path, mask_provenance)):
                    print(f"    Computing view-aligned hair mask...")
                    va_img_for_mask = Image.open(view_aligned_image_path).convert("RGB")
                    computed_mask, mask_score = extract_hair_mask(va_img_for_mask)
                    computed_mask.save(view_aligned_mask_path)
                    write_provenance(view_aligned_mask_path, mask_provenance)
                    print(f"    Saved mask to {view_aligned_mask_path} (SAM score: {mask_score:.3f})")
                
                if conditioning_source == "blended" and has_blending:
                    va_image = str(blending_image_path)
                    # Still need mask for uncropping check even when using blending
                    va_mask = str(view_aligned_mask_path) if view_aligned_mask_path.exists() else None
                else:
                    va_image = str(view_aligned_image_path)
                    va_mask = str(view_aligned_mask_path) if view_aligned_mask_path.exists() else None
            
            result = pipeline.transfer_hair(
                source_bald_image=str(source_image_path),
                reference_image=str(reference_image_path),
                view_aligned_image=va_image,
                view_aligned_hair_mask=va_mask,
                use_3d_aware=use_3d_aware,
                conditioning_source=conditioning_source,
                output_dir=str(output_dir),
                seed=item_seed,
            )
            
            if result is not None:
                # Check if resize_info.json exists in source outpainted folder
                if resize_info_path.exists():
                    print(f"    [{mode_name}] Found resize_info.json, cropping result to original size...")
                    with open(resize_info_path, 'r') as f:
                        source_resize_info = json.load(f)
                    uncropper = UncropperSingleton.get_uncropper()
                    # Get original size from resize_info
                    original_width = source_resize_info.get('original_width', result.width)
                    original_height = source_resize_info.get('original_height', result.height)
                    original_size = (original_width, original_height)
                    result = uncropper.crop(result, source_resize_info, output_size=original_size)
                    print(f"    [{mode_name}] Cropped result to original size: {result.size}")
                    # Save the cropped result to output_dir
                    cropped_output_path = output_dir / config.FILE_HAIR_RESTORED
                    result.save(cropped_output_path)
                
                output_mask_path = output_dir / config.FILE_HAIR_RESTORED_MASK
                hair_mask, score = extract_hair_mask(result)
                hair_mask.save(output_mask_path)
                write_provenance(output_path, provenance)
                write_provenance(
                    output_mask_path,
                    build_provenance(
                        "transfer_hair_output_mask",
                        seed=item_seed,
                        inputs=[output_path],
                        metadata={"bald_version": bald_version, "mode": mode},
                    ),
                )
                print(f"    [{mode_name}] Saved (SAM score: {score:.3f})")
                results[mode] = True
        
        return results
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)
        return results


def process_view_aligned_folders(
    data_dir: Union[str, Path],
    shape_provider: str = "hi3dgen",
    texture_provider: str = "mvadapter",
    config: Optional[HairTransferKleinConfig] = None,
    skip_existing: bool = True,
    bald_version: str = "w_seg",
    conditioning_source: str = "enhanced",
    include_3d_unaware: bool = True,
    device: str | None = None,
) -> Dict[str, int]:
    if config is None:
        config = HairTransferKleinConfig()
    
    data_dir = Path(data_dir)
    dm = DatasetManager(data_dir)
    view_aligned_dir = dm.view_aligned_root(shape_provider, texture_provider)
    
    if not view_aligned_dir.exists():
        raise ValueError(f"View aligned directory not found: {view_aligned_dir}")
    
    if bald_version == "all":
        bald_versions = ["w_seg", "wo_seg"]
    else:
        bald_versions = [bald_version]
    
    print(f"\nConfiguration:")
    print(f"  bald_version(s): {bald_versions}")
    if conditioning_source not in {"enhanced", "blended"}:
        raise ValueError(f"Unknown conditioning_source: {conditioning_source!r}")
    print(f"  conditioning_source: {conditioning_source}")
    print(f"  Mode selection: Automatic (3D-aware when view-aligned available, always 3D-unaware)")
    all_folders = [f for f in view_aligned_dir.iterdir() if f.is_dir()]
    
    if not all_folders:
        print("No folders found!")
        return {
            "processed_3d_aware": 0,
            "processed_3d_unaware": 0,
            "completed_3d_aware": 0,
            "completed_3d_unaware": 0,
            "skipped_3d_aware": 0,
            "skipped_3d_unaware": 0,
            "total_samples": 0,
            "errors": [],
        }

    timestamp_seed = config.SEED if config.SEED is not None and config.SEED >= 0 else int(time.time())
    config.SEED = timestamp_seed
    random.seed(timestamp_seed)
    random.shuffle(all_folders)
    print(f"Found {len(all_folders)} samples (shuffle seed: {timestamp_seed})")
    print("\n" + "=" * 60)
    print("Initializing FLUX.2 Klein Hair Transfer Pipeline...")
    print("=" * 60)
    pipeline = HairTransferKleinPipeline(config, device=device)
    
    stats = {
        '3d_aware_success': 0,
        '3d_unaware_success': 0,
        '3d_aware_completed': 0,
        '3d_unaware_completed': 0,
        '3d_aware_skipped': 0,
        '3d_unaware_skipped': 0,
        'total_samples': 0,
        'errors': [],
    }
    
    try:
        for bv in bald_versions:
            print(f"\n{'='*60}")
            print(f"Processing: bald_version={bv}")
            print(f"{'='*60}")
            
            for i, folder in enumerate(all_folders, 1):
                print(f"\n[{i}/{len(all_folders)}] {folder.name} ({bv})")
                sample_results = process_sample(
                    folder,
                    pipeline,
                    data_dir,
                    config,
                    bald_version=bv,
                    skip_existing=skip_existing,
                    conditioning_source=conditioning_source,
                    include_3d_unaware=include_3d_unaware,
                    shape_provider=shape_provider,
                    texture_provider=texture_provider,
                )
                
                stats['total_samples'] += 1
                if sample_results['3d_aware']:
                    stats['3d_aware_success'] += 1
                    if sample_results['skipped_3d_aware']:
                        stats['3d_aware_skipped'] += 1
                    else:
                        stats['3d_aware_completed'] += 1
                if sample_results['3d_unaware']:
                    stats['3d_unaware_success'] += 1
                    if sample_results['skipped_3d_unaware']:
                        stats['3d_unaware_skipped'] += 1
                    else:
                        stats['3d_unaware_completed'] += 1
                if sample_results.get("error"):
                    stats["errors"].append(f"{folder.name}:{bv}: {sample_results['error']}")
            
            print(f"\n✓ {bv}: 3D-Aware={stats['3d_aware_success']}, 3D-Unaware={stats['3d_unaware_success']}")
    
    finally:
        pipeline.unload()
    
    print(f"\n{'='*60}")
    print(f"✓ All processing complete!")
    print(f"  3D-Aware successful: {stats['3d_aware_success']}")
    print(f"  3D-Unaware successful: {stats['3d_unaware_success']}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"{'='*60}")
    
    return {
        "processed_3d_aware": stats['3d_aware_success'],
        "processed_3d_unaware": stats['3d_unaware_success'],
        "completed_3d_aware": stats['3d_aware_completed'],
        "completed_3d_unaware": stats['3d_unaware_completed'],
        "skipped_3d_aware": stats['3d_aware_skipped'],
        "skipped_3d_unaware": stats['3d_unaware_skipped'],
        "total_samples": stats['total_samples'],
        "errors": stats["errors"],
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hair Transfer using FLUX.2 Klein 9B"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch"],
        default="batch",
        help="Processing mode"
    )
    
    # Single mode arguments
    parser.add_argument("--source", type=str, help="Source bald image path")
    parser.add_argument("--reference", type=str, help="Reference hair image path")
    parser.add_argument("--view_aligned", type=str, help="View-aligned image (for 3D-aware)")
    parser.add_argument("--output", type=str, help="Output directory")
    
    # Batch mode arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="outputs",
        help="Root data directory"
    )
    parser.add_argument(
        "--shape_provider",
        type=str,
        default="hi3dgen",
        choices=["hunyuan", "hi3dgen", "direct3d_s2"],
        help="Shape provider name"
    )
    parser.add_argument(
        "--texture_provider",
        type=str,
        default="mvadapter",
        choices=["hunyuan", "mvadapter"],
        help="Texture provider name"
    )
    parser.add_argument(
        "--bald_version",
        type=str,
        default="w_seg",
        choices=["w_seg", "wo_seg", "all"],
        help="Bald version to use"
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
    parser.add_argument(
        "--conditioning_source",
        choices=["enhanced", "blended"],
        default="enhanced",
        help="3D-aware conditioning evidence to use"
    )
    
    # Pipeline parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=4,
        help="Number of inference steps (default: 4 for Klein)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale (default: 1.0 for Klein)"
    )
    parser.add_argument("--device", type=str, default=None, help="Compute device")
    
    args = parser.parse_args()
    
    # Create config
    config = HairTransferKleinConfig()
    config.SEED = args.seed
    config.NUM_INFERENCE_STEPS = args.num_steps
    config.GUIDANCE_SCALE = args.guidance_scale

    if args.mode == "single":
        if not all([args.source, args.reference, args.output]):
            parser.error("Single mode requires --source, --reference, and --output")
        
        pipeline = HairTransferKleinPipeline(config, device=args.device)
        
        try:
            use_3d_aware = args.view_aligned is not None
            
            modes_to_run = ['3d_unaware']
            if use_3d_aware:
                modes_to_run.append('3d_aware')
            
            for mode in modes_to_run:
                is_3d_aware = (mode == '3d_aware')
                mode_name = "3D-Aware" if is_3d_aware else "3D-Unaware"
                
                mode_output_dir = os.path.join(args.output, mode)
                os.makedirs(mode_output_dir, exist_ok=True)
                
                print(f"\n[{mode_name}] Generating...")
                
                result = pipeline.transfer_hair(
                    source_bald_image=args.source,
                    reference_image=args.reference,
                    view_aligned_image=args.view_aligned if is_3d_aware else None,
                    use_3d_aware=is_3d_aware,
                    conditioning_source=args.conditioning_source,
                    output_dir=mode_output_dir,
                )
                output_mask_path = os.path.join(mode_output_dir, config.FILE_HAIR_RESTORED_MASK)
                hair_mask, score = extract_hair_mask(result)
                hair_mask.save(output_mask_path)
                print(f"[{mode_name}] Saved hair mask (SAM score: {score:.3f})")
            
        finally:
            pipeline.unload()
    
    else:
        results = process_view_aligned_folders(
            data_dir=args.data_dir,
            shape_provider=args.shape_provider,
            texture_provider=args.texture_provider,
            config=config,
            skip_existing=args.skip_existing,
            bald_version=args.bald_version,
            conditioning_source=args.conditioning_source,
            device=args.device,
        )
        
        print(f"\n{'=' * 60}")
        print("Processing Complete:")
        print(f"  3D-Aware processed:   {results['processed_3d_aware']}")
        print(f"  3D-Unaware processed: {results['processed_3d_unaware']}")
        print(f"  Total samples:        {results['total_samples']}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
