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

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from tqdm import tqdm


from diffusers import FluxFillPipeline, FluxPriorReduxPipeline
from diffusers.pipelines.flux.pipeline_flux_prior_redux import FluxPriorReduxPipelineOutput, PipelineImageInput
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel

from utils.sam_mask_extractor import SAMMaskExtractor

from utils.bg_remover import BackgroundRemover
from hairport.utility.uncrop_sdxl import ImageUncropper
from hairport.utility.masking_utils import get_bbox_from_mask, expand_bbox, expand_bbox_for_hair, pad_to_square, box2squre, crop_back, expand_image_mask
from hairport.utility.blending_utils import create_distance_soft_blend_mask, create_hierarchical_blend_mask, multi_scale_blend
from hairport.postprocessing.multi_cond_redux import FluxMultiPriorReduxPipeline
from typing import List, Optional, Union, Tuple

import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import shutil


def flush():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()

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
    _instance: Optional[BackgroundRemover] = None
    
    @classmethod
    def get_instance(cls, force_reload: bool = False) -> BackgroundRemover:
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
    prompt: str = "head hair",
    sam_instance: Optional[SAMMaskExtractor] = None,
    bg_remover_instance: Optional[BackgroundRemover] = None,
) -> tuple[Image.Image, float]:
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

@dataclass
class HairTransferConfig:
    """Configuration for hair transfer pipeline."""
    
    # Model settings
    FLUX_FILL_MODEL: str = "black-forest-labs/FLUX.1-Fill-dev"
    FLUX_REDUX_MODEL: str = "black-forest-labs/FLUX.1-Redux-dev"
    LORA_WEIGHTS_PATH: str = "/workspace/HairPort/Hairdar/insert_anything_lora.safetensors"
    
    # Processing resolution
    PROCESSING_RESOLUTION: int = 768
    OUTPUT_RESOLUTION: int = 1024
    
    # Generation parameters
    SEED: int = 42
    GUIDANCE_SCALE: float = 30.0
    NUM_INFERENCE_STEPS: int = 50
    MAX_SEQUENCE_LENGTH: int = 512
    
    # Latent-space blending parameters (FLUX Flow Matching compatible)
    LATENT_BLEND: bool = True  # Use latent-space blending instead of pixel blending
    LATENT_BLEND_START_STEP: float = 0.0  # Start blending immediately (was 0.3) - critical for identity
    LATENT_BLEND_STRENGTH: float = 0.9  # Higher strength (was 0.5) - better identity preservation
    LATENT_BLEND_SCHEDULE: str = "constant"  # Use constant (was flux_linear) - consistent preservation

    # Directory structure
    DIR_VIEW_ALIGNED: str = "view_aligned"
    DIR_ALIGNMENT: str = "alignment"
    DIR_BALD: str = "bald"
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
    FILE_VIEW_ALIGNED_FALLBACK_IMAGE: str = "target_image_phase_2.png"
    FILE_WARPED_TARGET_IMAGE: str = "warped_target_image.png"
    FILE_WARPED_HAIR_MASK: str = "target_hair_mask.png"
    FILE_TARGET_HAIR_MASK: str = "target_image_hair_mask.png"
    FILE_CAMERA_PARAMS: str = "camera_params.json"
    
    # Output file naming (simple names, directory structure encodes settings)
    FILE_HAIR_RESTORED: str = "hair_restored.png"
    FILE_HAIR_RESTORED_MASK: str = "hair_restored_mask.png"
    
    # Uncropping settings
    ENABLE_UNCROP: bool = False  # Whether to uncrop the output to original image space

    # Outpainted source image paths
    DIR_SOURCE_OUTPAINTED: str = "source_outpainted"
    FILE_OUTPAINTED_IMAGE: str = "outpainted_image.png"
    FILE_RESIZE_INFO: str = "resize_info.json"
    
    SUBDIR_BALD_IMAGE: str = "image"


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
        """Callback invoked at each denoising step."""
        self.current_step = step
        latents = callback_kwargs.get("latents")
        
        if latents is None:
            return callback_kwargs
        
        t = self._infer_flow_t(pipe, step=step, timestep=timestep)
        denoise_progress = 1.0 - t
        weight = self._get_blend_weight(denoise_progress)
        
        if weight <= 0:
            return callback_kwargs
        
        device = latents.device
        dtype = latents.dtype
        
        if latents.dim() != 3:
            return callback_kwargs
        
        batch_size, seq_len, hidden_dim = latents.shape
        
        diptych_width = self.image_width * 2
        diptych_height = self.image_height
        
        expected_seq_len = (diptych_height // 16) * (diptych_width // 16)
        if seq_len != expected_seq_len:
            h_patches = int(math.sqrt(seq_len / 2))
            w_patches = seq_len // h_patches
            if h_patches * w_patches != seq_len:
                print(f"[LatentBlend] Cannot infer dimensions from seq_len={seq_len}")
                return callback_kwargs
            diptych_height = h_patches * 16
            diptych_width = w_patches * 16
        
        try:
            latents_spatial = _flux_unpack_latents(latents, diptych_height, diptych_width)
            
            source_clean = self.source_latents_clean.to(device=device, dtype=dtype)
            source_noisy = self._add_noise_to_source(source_clean, t=t, device=device, dtype=dtype)
            
            single_w = latents_spatial.shape[-1] // 2
            
            if source_noisy.shape[-2:] != (latents_spatial.shape[-2], single_w):
                source_noisy = F.interpolate(
                    source_noisy,
                    size=(latents_spatial.shape[-2], single_w),
                    mode='bilinear',
                    align_corners=False
                )
            
            mask = self.blend_mask.to(device=device, dtype=dtype)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(0)
            
            mask = F.interpolate(
                mask,
                size=(latents_spatial.shape[-2], single_w),
                mode='bilinear',
                align_corners=False
            )
            
            mask = mask.expand(batch_size, latents_spatial.shape[1], -1, -1)
            
            left_half = latents_spatial[..., :single_w]
            right_half = latents_spatial[..., single_w:]
            
            inv_mask = 1.0 - mask
            blend_weight = weight * inv_mask
            
            blended_right = right_half * (1 - blend_weight) + source_noisy * blend_weight
            
            blended_spatial = torch.cat([left_half, blended_right], dim=-1)
            latents = _flux_pack_latents(blended_spatial)
            
            callback_kwargs["latents"] = latents
            
        except Exception as e:
            print(f"[LatentBlend] Error during blending: {e}")
            
        return callback_kwargs

class HairRestorationPipeline:
    """Pipeline for hair transfer combining InsertAnything and RF-Inversion."""
    
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
            self.pipe.load_lora_weights(self.config.LORA_WEIGHTS_PATH)
        else:
            raise FileNotFoundError(f"LoRA weights not found at {self.config.LORA_WEIGHTS_PATH}")
        
        print("Loading FLUX Redux pipeline...")
        self.redux = FluxMultiPriorReduxPipeline.from_pretrained(
            self.config.FLUX_REDUX_MODEL,
            torch_dtype=self.dtype,
        ).to(self.device)
        
        self.inverter = LatentInverter(self.pipe, self.device, self.dtype)
        print("Pipeline initialized successfully!")

    def mask_image(
        self,
        image: Image.Image,
        mask_mode: str = "hair",
        background_value: Tuple[int, int, int] = (255, 255, 255),
        hair_mask: Optional[Image.Image] = None,
        mode: str = "RGB",
    ) -> Tuple[Image.Image, Image.Image]:
        """Mask image by hair or subject silhouette. Returns (masked_image, mask)."""
        image_np = np.array(image).astype(np.float32)
        
        if mask_mode == "hair":
            # Use provided hair mask or compute with SAM
            if hair_mask is not None:
                mask_pil = hair_mask
                if mask_pil.size != image.size:
                    mask_pil = mask_pil.resize(image.size, Image.Resampling.NEAREST)
            else:
                mask_pil, _ = extract_hair_mask(image)
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        elif mask_mode == "matte":
            # Use background removal for subject silhouette
            bg_remover = BackgroundRemoverSingleton.get_instance()
            _, mask_pil = bg_remover.remove_background(image)
            mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        else:
            raise ValueError(f"Unknown mask_mode: {mask_mode}. Expected 'hair' or 'matte'.")
        
        # Create background array
        background = np.full_like(image_np, background_value, dtype=np.float32)
        
        # Blend: foreground where mask is 1, background where mask is 0
        mask_3ch = mask_np[:, :, np.newaxis]
        masked_np = (image_np * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)
        
        # Ensure mask_pil is in "L" mode
        if mask_pil.mode != "L":
            mask_pil = mask_pil.convert("L")
        
        return Image.fromarray(masked_np).convert(mode), mask_pil

    def mask_image_by_hair(self, image: Image.Image, mode="RGB") -> Image.Image:
        """Mask out non-hair regions in the image using SAM.
        
        Deprecated: Use mask_image(image, mask_mode="hair", background_value=(0,0,0)) instead.
        """
        masked_img, _ = self.mask_image(image, mask_mode="hair", background_value=(0, 0, 0), mode=mode)
        return masked_img
    
    def matte_image(self, image: Image.Image, mode="RGB") -> Image.Image:
        """Matte the image to isolate the subject using background removal.
        
        Deprecated: Use mask_image(image, mask_mode="matte", background_value=(0,0,0)) instead.
        """
        masked_img, _ = self.mask_image(image, mask_mode="matte", background_value=(0, 0, 0), mode=mode)
        return masked_img
    
    def transfer_hair(
        self,
        source_image_path: str,
        source_mask_path: str,
        reference_image_path: str,
        reference_mask_path: str,
        output_dir: str = "output",
        hair_prompt: str = "",
        subject_prompt: str = "",
        seed: Optional[int] = None,
        redux_images: Optional[List[Dict[str, Any]]] = None,
        guidance_scale: Optional[float] = 30.0,
        resize_info_path: Optional[str] = None,
    ) -> str:
        """Transfer hair from reference to source image."""
        if seed is None:
            seed = self.config.SEED
        
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator(self.device).manual_seed(seed)
        
        print(f"Processing with seed={seed}")
        print("Loading and preparing images...")
        
        source_pil = Image.open(source_image_path).convert("RGB").resize(self.size, Image.Resampling.LANCZOS)
        reference_pil = Image.open(reference_image_path).convert("RGB").resize(self.size, Image.Resampling.LANCZOS)
        reference_mask_pil = Image.open(reference_mask_path).convert("L").resize(self.size, Image.Resampling.NEAREST)
        source_mask_pil = Image.open(source_mask_path).convert("L").resize(self.size, Image.Resampling.NEAREST)
        
        # Matte the reference image (white background)
        reference_pil, _ = self.mask_image(reference_pil, mask_mode="matte", background_value=(255, 255, 255), mode="RGB")
        
        flux_redux_input_images = []
        flux_redux_input_masks = []
        
        if redux_images is not None and len(redux_images) > 0:
            print(f"Processing {len(redux_images)} Redux images...")
            for idx, redux_item in enumerate(redux_images):
                # Load image
                img = redux_item.get('image')
                if isinstance(img, (str, Path)):
                    img = Image.open(img).convert("RGB")
                img = img.resize(self.size, Image.Resampling.LANCZOS)
                
                mask_by_hair = redux_item.get('mask_by_hair', False)
                
                if mask_by_hair:
                    # Mask by hair region (black background for non-hair)
                    hair_mask = redux_item.get('hair_mask')
                    if hair_mask is not None and isinstance(hair_mask, (str, Path)):
                        hair_mask = Image.open(hair_mask).convert("L")
                    # Use mask_image with hair mode (will compute mask via SAM if hair_mask is None)
                    processed_img, processed_mask = self.mask_image(
                        img,
                        mask_mode="hair",
                        background_value=(255, 255, 255),
                        hair_mask=hair_mask,
                        mode="RGB"
                    )
                    print(f"  Redux image {idx+1}: masked by hair")
                else:
                    # Matte with white background (full subject visible)
                    processed_img, processed_mask = self.mask_image(
                        img,
                        mask_mode="matte",
                        background_value=(255, 255, 255),
                        mode="RGB"
                    )
                    print(f"  Redux image {idx+1}: matted with white background")
                
                flux_redux_input_images.append(processed_img)
                flux_redux_input_masks.append(processed_mask)
        else:
            print("No Redux images provided, using reference image as fallback")
            flux_redux_input_images.append(reference_pil)
            flux_redux_input_masks.append(reference_mask_pil)
            
        # for i, img in enumerate(flux_redux_input_images):
        #     save_dir = os.path.join(output_dir, f"redux_input_{i+1}.png")
        #     os.makedirs(output_dir, exist_ok=True)
        #     img.save(save_dir)
        #     print(f"Saved Redux input image {i+1} to {save_dir}")
        
        # # Save Redux masks for debugging
        # for i, mask in enumerate(flux_redux_input_masks):
        #     mask_save_dir = os.path.join(output_dir, f"redux_input_mask_{i+1}.png")
        #     mask.save(mask_save_dir)
        #     print(f"Saved Redux input mask {i+1} to {mask_save_dir}")
        
        tar_image = source_pil
        ref_image = reference_pil
        tar_mask = source_mask_pil
        ref_mask = reference_mask_pil
        
        tar_image = np.asarray(tar_image)
        tar_mask = np.asarray(tar_mask)
        tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)

        ref_image = np.asarray(ref_image)
        ref_mask = np.asarray(ref_mask)
        ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

        if tar_mask.sum() == 0:
            raise 'No mask for the background image.Please check mask button!'

        if ref_mask.sum() == 0:
            raise 'No mask for the reference image.Please check mask button!'

        ref_box_yyxx = get_bbox_from_mask(ref_mask)
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) 
        y1,y2,x1,x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
        ref_mask = ref_mask[y1:y2,x1:x2] 
        ratio = 1.3
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)


        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False) 

        kernel = np.ones((11, 11), np.uint8)
        iterations = 3
        tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)

        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        tar_box_yyxx_expanded, needs_outpaint = expand_bbox_for_hair(
            tar_mask, 
            tar_box_yyxx, 
            min_ratio=1.4,
            coverage_threshold=0.5
        )
        
        if needs_outpaint:
            print("  Note: Hair mask fills most of image. Consider using outpainted source for better results.")
        
        tar_box_yyxx_crop, _ = expand_bbox_for_hair(
            tar_image[:, :, 0] if len(tar_image.shape) == 3 else tar_image,
            tar_box_yyxx_expanded,
            min_ratio=1.45,  # Larger minimum for the crop region
            coverage_threshold=0.5
        )
        tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
        y1,y2,x1,x2 = tar_box_yyxx_crop


        old_tar_image = tar_image.copy()
        tar_image = tar_image[y1:y2,x1:x2,:]
        tar_mask = tar_mask[y1:y2,x1:x2]

        H1, W1 = tar_image.shape[0], tar_image.shape[1]

        tar_mask = pad_to_square(tar_mask, pad_value=0)
        tar_mask = cv2.resize(tar_mask, self.size)

        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), self.size).astype(np.uint8)
        
        prompt = "Transfer the exact hair to the scalp/head of the bald person. Blend the hair naturally and realistically to the head/scalp region."
        prompt_2_parts = [prompt]
        if subject_prompt:
            prompt_2_parts.append(subject_prompt)
        if hair_prompt:
            prompt_2_parts.append(hair_prompt)
        prompt_2 = " ".join(prompt_2_parts)
        
        print(f"Using {len(flux_redux_input_images)} images for Redux conditioning")
        
        prompt_embeds = []
        pooled_prompt_embeds = []
        redux_outs = self.redux(
            image=flux_redux_input_images,
            mask=flux_redux_input_masks,
            downsampling_factor=1.0,
            mode="autocrop with mask",
        )
        prompt_embeds = redux_outs.prompt_embeds
        pooled_prompt_embeds = redux_outs.pooled_prompt_embeds
        print(f"Combined Redux prompt embeds shape: {prompt_embeds.shape}, Pooled embeds shape: {pooled_prompt_embeds.shape}")

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            lora_scale=None,
        )

        tar_image = pad_to_square(tar_image, pad_value=255)

        H2, W2 = tar_image.shape[0], tar_image.shape[1]

        tar_image = cv2.resize(tar_image, self.size)
        diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)


        tar_mask = np.stack([tar_mask,tar_mask,tar_mask],-1)
        mask_black = np.ones_like(tar_image) * 0
        mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)


        diptych_ref_tar = Image.fromarray(diptych_ref_tar)
        mask_diptych[mask_diptych == 1] = 255
        mask_diptych = Image.fromarray(mask_diptych)

        # Generate timestamp for unique filenames
        
        os.makedirs(output_dir, exist_ok=True)
        # masked_ref_image_pil = Image.fromarray(masked_ref_image)
        # masked_ref_image_pil.save(f'{output_dir}/input_masked_ref.png')
        # diptych_ref_tar.save(f'{output_dir}/input_diptych.png')
        # mask_diptych.save(f'{output_dir}/input_mask_diptych.png')

        generator = torch.Generator("cuda").manual_seed(seed)
        edited_images = self.pipe(
            image=diptych_ref_tar,
            mask_image=mask_diptych,
            height=mask_diptych.size[1],
            guidance_scale=guidance_scale,
            width=mask_diptych.size[0],
            max_sequence_length=512,
            generator=generator,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        ).images
        print(f"Generated {len(edited_images)} edited images.")
        edited_image = edited_images[0]

        width, height = edited_image.size
        left = width // 2
        right = width
        top = 0
        bottom = height
        edited_image = edited_image.crop((left, top, right, bottom))

        edited_image = np.array(edited_image)
        edited_image = crop_back(edited_image, old_tar_image, np.array([H1, W1, H2, W2]), np.array(tar_box_yyxx_crop)) 
        edited_image = Image.fromarray(edited_image)
        edited_image = edited_image.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        if self.config.ENABLE_UNCROP and resize_info_path is not None:
            try:
                with open(resize_info_path, 'r') as f:
                    resize_info = json.load(f)
                
                # Extract uncrop parameters from resize_info
                original_size = resize_info.get('original_size')  # [width, height]
                crop_box = resize_info.get('crop_box')  # [x1, y1, x2, y2] or similar
                scale = resize_info.get('scale', 1.0)
                offset = resize_info.get('offset', [0, 0])
                
                if original_size is not None:
                    original_w, original_h = original_size
                    
                    # Create white background at original size
                    uncropped = Image.new('RGB', (original_w, original_h), (255, 255, 255))
                    
                    # Calculate paste position based on resize_info
                    if crop_box is not None:
                        x1, y1, x2, y2 = crop_box
                        crop_w = x2 - x1
                        crop_h = y2 - y1
                        
                        # Resize edited image to match crop region
                        resized_edited = edited_image.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
                        uncropped.paste(resized_edited, (x1, y1))
                    else:
                        # Use scale and offset
                        new_w = int(edited_image.width / scale)
                        new_h = int(edited_image.height / scale)
                        resized_edited = edited_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        paste_x = int(offset[0])
                        paste_y = int(offset[1])
                        uncropped.paste(resized_edited, (paste_x, paste_y))
                    
                    edited_image = uncropped
                    print(f"  Uncropped to original size: {original_w}x{original_h}")
                    
            except Exception as e:
                print(f"  Warning: Failed to uncrop image: {e}")
                
        output_path = os.path.join(output_dir, f'hair_restored.png')
        edited_image.save(output_path)
        print(f"Saved hair-restored image to {output_path}")
        return output_path

    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        if hasattr(self, 'pipe'):
            del self.pipe
        if hasattr(self, 'redux'):
            del self.redux
        flush()


def _compute_hair_mask(
    image_path: Union[str, Path],
    mask_path: Union[str, Path],
    *,
    confidence_threshold: float = 0.4,
    detection_threshold: float = 0.5,
    prompt: str = "head hair",
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


def get_output_dir(
    pair_dir: Path,
    config: HairTransferConfig,
    use_3d_aware: bool,
) -> Path:
    """Get output directory: {pair_dir}/{3d_aware|3d_unaware}/transferred/"""
    mode_dir = config.DIR_3D_AWARE if use_3d_aware else config.DIR_3D_UNAWARE
    return pair_dir / mode_dir / config.SUBDIR_TRANSFERRED


def process_sample(
    folder: Path,
    pipeline: "HairRestorationPipeline",
    data_dir: Path,
    config: HairTransferConfig,
    bald_version: str,
    conditioning_mode: str,
    skip_existing: bool = True,
) -> bool:
    """Process a single sample folder for given bald_version and conditioning_mode."""
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
    
    dataset_name = Path(data_dir).name
    
    # Set mode flag
    use_3d_aware = (conditioning_mode == "3d_aware")
    
    # Select mode directory based on conditioning_mode
    mode_dir = config.DIR_3D_AWARE if use_3d_aware else config.DIR_3D_UNAWARE
    warping_dir = pair_dir / mode_dir / config.SUBDIR_WARPING
    blending_dir = pair_dir / mode_dir / config.SUBDIR_BLENDING
    
    # Check for required warped outputs
    warped_image_path = warping_dir / config.FILE_WARPED_TARGET_IMAGE
    warped_hair_mask_path = warping_dir / config.FILE_WARPED_HAIR_MASK
    has_warped_outputs = warped_image_path.exists() and warped_hair_mask_path.exists()
    
    if not has_warped_outputs:
        print(f"  Missing warped outputs for {folder_name}/{bald_version}/{conditioning_mode}, skipping...")
        return False
    
    # Build output paths:
    # {pair_dir}/{3d_aware|3d_unaware}/transferred/hair_restored.png
    output_dir = get_output_dir(pair_dir, config, use_3d_aware)
    output_path = output_dir / config.FILE_HAIR_RESTORED
    output_mask_path = output_dir / config.FILE_HAIR_RESTORED_MASK
    
    # Skip if output exists
    if skip_existing and output_path.exists():
        print(f"Output already exists for {folder_name}/{bald_version}/{conditioning_mode}, skipping...")
        return True
    
    try:
        source_image_path = pair_dir / config.DIR_SOURCE_OUTPAINTED / config.FILE_OUTPAINTED_IMAGE
        
        if not source_image_path.exists():
            bald_image_dir = data_dir / config.DIR_BALD / bald_version / config.SUBDIR_BALD_IMAGE
            source_image_path = bald_image_dir / f"{source_id}.png"
        
        if not source_image_path.exists():
            print(f"  Warning: Source image not found: {source_image_path}")
            return False
        
        # Use target_image_phase_1.png from alignment directory as reference
        alignment_dir = folder / config.DIR_ALIGNMENT
        phase1_image_path = alignment_dir / config.FILE_VIEW_ALIGNED_IMAGE
        phase1_mask_path = alignment_dir / config.FILE_TARGET_HAIR_MASK
        
        # Check if phase1 exists, fallback to warped if not
        if phase1_image_path.exists():
            reference_image_path = phase1_image_path
            reference_mask_path = phase1_mask_path
            if not reference_mask_path.exists():
                # Compute hair mask for phase1 image if missing
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
            # Fallback to warped outputs
            reference_image_path = warped_image_path
            reference_mask_path = warped_hair_mask_path
            print(f"  Fallback: Using warped_target_image as reference")
        
        target_mask_path = warped_hair_mask_path
        
        print(f"  Reference image: {reference_image_path}")
        print(f"  Reference mask: {reference_mask_path}")
        
        if dataset_name == "outputs":
            original_ref_folder = "hair_aligned_image"
        else:
            original_ref_folder = "image_outpainted"
        
        original_reference_path = data_dir / original_ref_folder / f"{target_id}.png"
        
        if not original_reference_path.exists():
            original_reference_path = data_dir / config.DIR_IMAGE / f"{target_id}.png"
        
        if not original_reference_path.exists():
            print(f"  Warning: Original reference image not found for {target_id}")
            original_reference_path = None
        
        redux_images = []
        
        if use_3d_aware:
            # Use target_image_phase_1.png for Redux (same as diptych reference)
            if phase1_image_path.exists():
                redux_images.append({
                    'image': str(phase1_image_path),
                    'mask_by_hair': True,
                    'hair_mask': str(phase1_mask_path) if phase1_mask_path.exists() else None,
                })
                print(f"  Redux image 1: target_image_phase_1 (masked by hair)")
            else:
                # Fallback to warped image
                redux_images.append({
                    'image': str(warped_image_path),
                    'mask_by_hair': True,
                    'hair_mask': str(warped_hair_mask_path),
                })
                print(f"  Redux image 1 (fallback): warped_target_image (masked by hair)")
            
            poisson_blended_path = blending_dir / "poisson_blended.png"
            if poisson_blended_path.exists():
                redux_images.append({
                    'image': str(poisson_blended_path),
                    'mask_by_hair': False,  # Matte with white background only
                })
                print(f"  Redux image 2: poisson_blended (matted, not masked by hair)")
            else:
                print(f"  Warning: poisson_blended.png not found at {poisson_blended_path}")
            
            # 3. Original reference image (masked by hair)
            # if original_reference_path is not None:
            #     redux_images.append({
            #         'image': str(original_reference_path),
            #         'mask_by_hair': True,
            #         'hair_mask': None,  # Will be computed by SAM
            #     })
            #     print(f"  Redux image 3: original reference (masked by hair)")
        else:
            # 3D-Unaware Transfer: 1 Redux image
            # 1. Original reference image (masked by hair)
            if original_reference_path is not None:
                redux_images.append({
                    'image': str(original_reference_path),
                    'mask_by_hair': True,
                    'hair_mask': None,  # Will be computed by SAM
                })
                print(f"  Redux image 1: original reference (masked by hair)")
            else:
                # Fallback: use warped image
                redux_images.append({
                    'image': str(warped_image_path),
                    'mask_by_hair': True,
                    'hair_mask': str(warped_hair_mask_path),
                })
                print(f"  Redux image 1 (fallback): warped_target_image (masked by hair)")
        
        prompts_dir = data_dir / config.DIR_PROMPTS
        hair_prompt = ""
        subject_prompt = ""
        
        try:
            target_prompt_file = prompts_dir / f"{target_id}.json"
            if target_prompt_file.exists():
                with open(target_prompt_file, 'r') as f:
                    prompt_data = json.load(f)
                hair_prompt = prompt_data.get("subject", [{}])[0].get("hair_description", "")
                if hair_prompt:
                    print(f"  Hair prompt: {hair_prompt[:50]}...")
        except Exception as e:
            print(f"  Warning: Could not load target prompt: {e}")
        
        try:
            source_prompt_file = prompts_dir / f"{source_id}.json"
            if source_prompt_file.exists():
                with open(source_prompt_file, 'r') as f:
                    prompt_data = json.load(f)
                subject_prompt = prompt_data.get("subject", [{}])[0].get("description_no_hair", "").replace(" no background", "")
                if subject_prompt:
                    print(f"  Subject prompt: {subject_prompt[:50]}...")
        except Exception as e:
            print(f"  Warning: Could not load source prompt: {e}")
        
        print(f"  Source: {source_image_path}")
        print(f"  Mode: {'3D-Aware' if use_3d_aware else '3D-Unaware'}")
        print(f"  Redux images count: {len(redux_images)}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        resize_info_path = pair_dir / config.DIR_SOURCE_OUTPAINTED / config.FILE_RESIZE_INFO
        resize_info_str = str(resize_info_path) if resize_info_path.exists() else None
        
        try:
            pipeline.transfer_hair(
                source_image_path=str(source_image_path),
                source_mask_path=str(target_mask_path),
                reference_image_path=str(reference_image_path),
                reference_mask_path=str(reference_mask_path),
                output_dir=str(output_dir),
                hair_prompt=hair_prompt,
                subject_prompt="",
                redux_images=redux_images,
                resize_info_path=resize_info_str,
            )
        except Exception as e:
            print(f"  Error during transfer_hair: {e}")
            # Remove the transferred folder if it exists due to error
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
    config: Optional[HairTransferConfig] = None,
    skip_existing: bool = True,
    bald_version: str = "w_seg",
    conditioning_mode: str = "3d_aware",
) -> Dict[str, int]:
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
    print("Initializing Hair Restoration Pipeline...")
    print("=" * 60)
    pipeline = HairRestorationPipeline(config)
    
    overall_success = 0
    overall_total = 0
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

    # finally:
    #     del pipeline
    #     # Release singleton instances to free GPU memory
    #     SAMMaskExtractorSingleton.release()
    #     BackgroundRemoverSingleton.release()
    #     flush()
    
    return {"processed": overall_success, "skipped": 0, "errors": overall_total - overall_success}


# ================================
# CLI Entry Point
# ================================

def main():
    """Main entry point for the hair restoration pipeline."""
    parser = argparse.ArgumentParser(
        description="Hair Restoration Pipeline with InsertAnything + RF-Inversion"
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
    
    # Conditioning mode settings
    parser.add_argument(
        "--conditioning_mode",
        type=str,
        default="3d_aware",
        choices=["3d_aware", "3d_unaware", "all"],
        help="Conditioning mode: 3d_aware (use view-aligned target), "
             "3d_unaware (use original target), or all (default: all)"
    )
    
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
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
    parser.add_argument("--num_steps", type=int, default=60, help="Number of inference steps")
    
    # Uncropping option
    parser.add_argument(
        "--enable_uncrop",
        action="store_true",
        default=False,
        help="Enable uncropping output to original image space using resize_info.json"
    )
    parser.add_argument(
        "--no_uncrop",
        action="store_false",
        dest="enable_uncrop",
        help="Disable uncropping (default)"
    )
    
    # Latent-space blending parameters
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
        help="When to start latent blending (0-1, lower=more freedom). Default: 0.45"
    )
    parser.add_argument(
        "--latent_blend_strength",
        type=float,
        default=0.90,
        help="Latent blend strength for non-hair regions (0=full freedom, 1=strict preservation). Default: 0.90"
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
    config.ENABLE_UNCROP = args.enable_uncrop
    
    # Latent blending config
    config.LATENT_BLEND = args.latent_blend
    config.LATENT_BLEND_START_STEP = args.latent_blend_start
    config.LATENT_BLEND_STRENGTH = args.latent_blend_strength
    config.LATENT_BLEND_SCHEDULE = args.latent_blend_schedule
    
    if args.mode == "single":
        # Single image processing
        if not all([args.source, args.reference, args.output]):
            parser.error("Single mode requires --source, --reference, and --output (masks are optional, will auto-compute with SAM)")
        
        pipeline = HairRestorationPipeline(config)
        
        # Build redux_images list from CLI arguments
        redux_images = []
        if args.redux_image:
            redux_images.append({
                'image': args.redux_image,
                'mask_by_hair': True,  # Default to masking by hair for single mode
                'hair_mask': args.redux_mask,
            })
        
        # Get output directory from output path
        output_path = Path(args.output)
        output_dir = output_path.parent
        
        pipeline.transfer_hair(
            source_image_path=args.source,
            source_mask_path=args.tar_mask if args.tar_mask else args.ref_mask,
            reference_image_path=args.reference,
            reference_mask_path=args.ref_mask,
            output_dir=str(output_dir),
            hair_prompt=args.hair_prompt or "",
            redux_images=redux_images if redux_images else None,
            guidance_scale=args.guidance_scale,
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
