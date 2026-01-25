#!/usr/bin/env python3
"""
Identity Restoration using FLUX Kontext + RF-Inversion

This script takes the output of restore_hair.py (hair_restored.png) and performs
identity restoration to better preserve the source person's facial features while
keeping the transferred hair.

Workflow:
1. Load hair_restored.png (output from restore_hair.py)
2. Compute and save hair mask as hair_restored_hair_mask.png
3. Run identity restoration using RF-Inversion
4. Resize output to 1024x1024
5. Compute and save final hair mask as id_restored_hair_mask.png

Notes:
- Uses official diffusers APIs only (diffusers==0.36.0)
- Does NOT apply CodeFormer enhancement (handled separately if needed)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import cv2
except Exception as e:
    raise RuntimeError("OpenCV (cv2) is required for mask dilation/resizing in this script.") from e

# --------------------------------------------------------------------------------------
# Optional project-local dependencies
# --------------------------------------------------------------------------------------
SAMMaskExtractor = None

try:
    sys.path.append("/workspace/HairPort/Hairdar/")
    from utils.sam_mask_extractor import SAMMaskExtractor  # type: ignore
except Exception:
    print("[WARN] Could not import SAMMaskExtractor. Auto hair-mask extraction will be disabled.")
from utils.bg_remover import BackgroundRemover  # type: ignore
from utils.super_resolution import CodeFormerEnhancer

# Import blending utilities and config from restore_hair for hair compositing
try:
    from hairport.postprocessing.restore_hair import (
        create_distance_soft_blend_mask,
        create_hierarchical_blend_mask,
        multi_scale_blend,
        HairTransferConfig,
        build_output_filename,
    )
    _HAS_RESTORE_HAIR_CONFIG = True
except ImportError:
    _HAS_RESTORE_HAIR_CONFIG = False
    # Fallback: define minimal blending functions locally
    def create_distance_soft_blend_mask(
        hair_mask: np.ndarray,
        dilation_px: int = 5,
        dilation_iterations: int = 1,
        feather_px: int = 15,
    ) -> np.ndarray:
        """Create soft blend mask using distance transform."""
        mask = (hair_mask > 0.5).astype(np.uint8)
        if dilation_px > 0 and dilation_iterations > 0:
            kernel = np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
        outside = (1 - mask).astype(np.uint8)
        dist_outside = cv2.distanceTransform(outside, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
        outside_alpha = np.clip(1.0 - (dist_outside / float(feather_px)), 0.0, 1.0)
        soft = np.where(mask > 0, 1.0, outside_alpha)
        return soft.astype(np.float32)

    def create_hierarchical_blend_mask(hair_mask: np.ndarray, num_levels: int = 4) -> list:
        """Create multi-scale blend masks."""
        masks = []
        current_mask = (hair_mask > 0.5).astype(np.float32)
        for level in range(num_levels):
            blur_size = 5 + level * 8
            if blur_size % 2 == 0:
                blur_size += 1
            kernel_size = 3 + level * 4
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1)
            blurred = cv2.GaussianBlur(dilated.astype(np.float32), (blur_size, blur_size), 0)
            masks.append(blurred)
        return masks

    def multi_scale_blend(
        generated: np.ndarray,
        source: np.ndarray,
        masks: list,
        use_laplacian: bool = True
    ) -> np.ndarray:
        """Multi-scale blending using Laplacian pyramids."""
        if not use_laplacian or len(masks) < 2:
            alpha = masks[0][:, :, np.newaxis]
            return (generated * alpha + source * (1 - alpha)).astype(np.uint8)
        num_levels = len(masks)
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
        gen_pyr = build_laplacian_pyramid(generated.astype(np.float32), num_levels)
        src_pyr = build_laplacian_pyramid(source.astype(np.float32), num_levels)
        blended_pyr = []
        for i in range(num_levels):
            h, w = gen_pyr[i].shape[:2]
            mask_resized = cv2.resize(masks[i], (w, h))
            if mask_resized.ndim == 2:
                mask_resized = mask_resized[:, :, np.newaxis]
            blended = gen_pyr[i] * mask_resized + src_pyr[i] * (1 - mask_resized)
            blended_pyr.append(blended)
        result = blended_pyr[-1]
        for i in range(num_levels - 2, -1, -1):
            result = cv2.pyrUp(result, dstsize=(blended_pyr[i].shape[1], blended_pyr[i].shape[0]))
            result = result + blended_pyr[i]
        return np.clip(result, 0, 255).astype(np.uint8)

    # Fallback HairTransferConfig if restore_hair import failed
    from dataclasses import dataclass as _fallback_dataclass
    @_fallback_dataclass
    class HairTransferConfig:
        """Fallback configuration for identity restoration pipeline."""
        # Directory structure
        DIR_VIEW_ALIGNED: str = "view_aligned"
        DIR_ALIGNMENT: str = "alignment"
        DIR_BALD: str = "bald"
        DIR_POSTPROCESSING: str = "fill_processed"  # Legacy
        DIR_PROMPTS: str = "prompt"
        # Top-level directories for 3D aware/unaware processing
        DIR_3D_AWARE: str = "3d_aware"
        DIR_3D_UNAWARE: str = "3d_unaware"
        # Subdirectories
        SUBDIR_WARPING: str = "warping"
        SUBDIR_TRANSFERRED: str = "transferred"  # New output directory
        SUBDIR_BALD_IMAGE: str = "image"
        MATTED_IMAGE_SUBDIR: str = "matted_image"  # Legacy
        MATTED_IMAGE_HAIR_MASK_SUBDIR: str = "matted_image_mask"  # Legacy
        # Redux image source directories (in priority order)
        DIR_HAIR_ALIGNED_IMAGE: str = "hair_aligned_image"  # Primary: hair-aligned images
        DIR_IMAGE: str = "image"  # Fallback: original images
        # File names
        FILE_VIEW_ALIGNED_IMAGE: str = "target_image.png"
        FILE_WARPED_TARGET_IMAGE: str = "warped_target_image.png"
        FILE_WARPED_HAIR_MASK: str = "target_hair_mask.png"
        FILE_CAMERA_PARAMS: str = "camera_params.json"
        # New output file naming (simple names)
        FILE_HAIR_RESTORED: str = "hair_restored.png"
        FILE_HAIR_RESTORED_MASK: str = "hair_restored_mask.png"
        # Legacy output file naming (for backward compatibility)
        FILE_OUTPUT_PREFIX: str = "transferred_fill"
        FILE_OUTPUT_MASK_SUFFIX: str = "_mask"
        # Outpainted source
        DIR_SOURCE_OUTPAINTED: str = "source_outpainted"
        FILE_OUTPAINTED_IMAGE: str = "outpainted_image.png"

    def build_output_filename(
        config: HairTransferConfig,
        bald_version: str,
        is_3d_aware: bool,
        suffix: str = "",
    ) -> str:
        """Build output filename based on configuration."""
        parts = [config.FILE_OUTPUT_PREFIX]
        parts.append(bald_version)
        parts.append("3d_aware" if is_3d_aware else "3d_unaware")
        return "hair_restored_" + "_".join(parts) + suffix + ".png"

# --------------------------------------------------------------------------------------
# Diffusers imports (official package only)
# --------------------------------------------------------------------------------------
try:
    from diffusers import FluxKontextPipeline
except Exception:
    from diffusers import FluxPipeline as FluxKontextPipeline  # type: ignore


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------
def _crop_to_multiple(img: Image.Image, multiple: int) -> Image.Image:
    w, h = img.size
    return img.crop((0, 0, w - w % multiple, h - h % multiple))


def preprocess_img(
    path: str,
    pipe: FluxKontextPipeline,
    device: torch.device,
    target_hw: Optional[Tuple[int, int]] = None,
    return_pil: bool = False,
) -> torch.Tensor | Image.Image:
    img = Image.open(path).convert("RGB")

    multiple = int(getattr(pipe, "vae_scale_factor", 8)) * 2
    img = _crop_to_multiple(img, multiple)

    if target_hw is not None:
        W, H = target_hw
        img = img.resize((W, H), Image.Resampling.LANCZOS)

    if return_pil:
        return img

    tens = pipe.image_processor.preprocess(img)
    tens = tens.to(device=device, dtype=torch.bfloat16)
    return tens


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return float(image_seq_len * m + b)

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    import inspect

    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` may be provided.")

    if timesteps is not None:
        accept_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_timesteps:
            raise ValueError(f"{scheduler.__class__.__name__}.set_timesteps does not support custom `timesteps`.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)

    if sigmas is not None:
        params = inspect.signature(scheduler.set_timesteps).parameters
        accept_sigmas = "sigmas" in params
        if not accept_sigmas:
            raise ValueError(f"{scheduler.__class__.__name__}.set_timesteps does not support custom `sigmas`.")

        # ---- FIX: also pass num_inference_steps if supported/needed ----
        try:
            if "num_inference_steps" in params:
                scheduler.set_timesteps(num_inference_steps=num_inference_steps, sigmas=sigmas, device=device, **kwargs)
            else:
                # some schedulers use the first positional arg for num_inference_steps
                scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device, **kwargs)
        except TypeError:
            # fallback if scheduler truly ignores num_inference_steps with custom sigmas
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)

        return scheduler.timesteps, len(scheduler.timesteps)

    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)



def lowpass_fft_latents(
    x: torch.Tensor,
    cutoff: float = 0.30,
    alpha: float = 1.0,
) -> torch.Tensor:
    assert x.ndim == 4, f"Expected [B,C,H,W], got {tuple(x.shape)}"
    assert 0 < cutoff <= 1
    assert alpha >= 0

    x_dtype = x.dtype
    x_f = x.float()

    X = torch.fft.fft2(x_f, dim=(-2, -1))
    X = torch.fft.fftshift(X, dim=(-2, -1))

    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device, dtype=x_f.dtype),
        torch.linspace(-1, 1, W, device=x.device, dtype=x_f.dtype),
        indexing="ij",
    )
    rr = torch.sqrt(xx * xx + yy * yy)
    hp_proj = (rr > cutoff).to(x_f.dtype)[None, None, :, :]

    Y = X - alpha * (X * hp_proj)

    Y = torch.fft.ifftshift(Y, dim=(-2, -1))
    x_out = torch.fft.ifft2(Y, dim=(-2, -1)).real
    return x_out.to(x_dtype)


def _autocast_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.bfloat16


# --------------------------------------------------------------------------------------
# Reference Hair Statistics Transfer (AdaIN-style)
# --------------------------------------------------------------------------------------
def transfer_hair_statistics(
    target_latents: torch.Tensor,
    ref_latents: torch.Tensor,
    target_hair_mask: torch.Tensor,
    ref_hair_mask: torch.Tensor,
    strength: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Transfer hair appearance via Adaptive Instance Normalization.
    
    This transfers COLOR and TEXTURE statistics (mean, std) from the reference
    hair region to the target hair region WITHOUT transferring spatial structure.
    Works regardless of mask position/shape differences.
    
    Args:
        target_latents: Target image latents [B, C, H, W]
        ref_latents: Reference image latents [B, C, H, W]
        target_hair_mask: Binary mask for target hair [1, 1, H, W] or [B, 1, H, W]
        ref_hair_mask: Binary mask for reference hair [1, 1, H, W] or [B, 1, H, W]
        strength: Blend strength (0 = no transfer, 1 = full transfer)
        eps: Small constant for numerical stability
        
    Returns:
        Modified target latents with reference hair statistics in hair region
    """
    # Ensure masks have correct shape
    if target_hair_mask.dim() == 3:
        target_hair_mask = target_hair_mask.unsqueeze(0)
    if ref_hair_mask.dim() == 3:
        ref_hair_mask = ref_hair_mask.unsqueeze(0)
    
    # Expand masks to match latent channels
    B, C, H, W = target_latents.shape
    
    # Compute statistics from reference hair region (per-channel)
    ref_masked = ref_latents * ref_hair_mask
    ref_count = ref_hair_mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1)
    ref_mean = ref_masked.sum(dim=(-2, -1), keepdim=True) / ref_count
    
    ref_var = ((ref_latents - ref_mean) ** 2 * ref_hair_mask).sum(dim=(-2, -1), keepdim=True) / ref_count
    ref_std = (ref_var + eps).sqrt()
    
    # Compute statistics from target hair region (per-channel)
    tar_masked = target_latents * target_hair_mask
    tar_count = target_hair_mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1)
    tar_mean = tar_masked.sum(dim=(-2, -1), keepdim=True) / tar_count
    
    tar_var = ((target_latents - tar_mean) ** 2 * target_hair_mask).sum(dim=(-2, -1), keepdim=True) / tar_count
    tar_std = (tar_var + eps).sqrt()
    
    # Normalize target hair region, then re-scale with reference statistics
    # Only apply to hair region pixels
    normalized = (target_latents - tar_mean) / (tar_std + eps)
    transferred = normalized * ref_std + ref_mean
    
    # Blend: only apply transfer in target hair region, with strength control
    blend_mask = target_hair_mask * strength
    result = target_latents * (1 - blend_mask) + transferred * blend_mask
    
    return result


def compute_injection_strength_schedule(
    t: torch.Tensor,
    injection_start_t: float = 400.0,
    injection_end_t: float = 50.0,
    max_strength: float = 0.4,
) -> float:
    """
    Compute time-dependent injection strength.
    
    - High t (early denoising): Structure forming → no injection
    - Medium t: Transition phase → ramp up
    - Low t (late denoising): Details forming → max injection
    
    Args:
        t: Current timestep
        injection_start_t: Start injecting below this timestep
        injection_end_t: Full strength below this timestep
        max_strength: Maximum injection strength
        
    Returns:
        Injection strength in [0, max_strength]
    """
    t_val = float(t.item()) if isinstance(t, torch.Tensor) else float(t)
    
    if t_val > injection_start_t:
        return 0.0
    elif t_val < injection_end_t:
        return max_strength
    else:
        # Linear ramp from 0 to max_strength
        progress = (injection_start_t - t_val) / (injection_start_t - injection_end_t)
        return max_strength * progress


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _dilate_mask(mask01: np.ndarray, k: int = 5, iters: int = 1) -> np.ndarray:
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask01, kernel, iterations=iters)


def _blur_mask(mask01: np.ndarray, k: int = 5) -> np.ndarray:
    """Apply Gaussian blur to mask for soft edges without extending reach."""
    if k % 2 == 0:
        k += 1  # Kernel must be odd
    return cv2.GaussianBlur(mask01, (k, k), 0)


def _resize_mask(mask01: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(mask01, (w, h), interpolation=cv2.INTER_NEAREST)


def _latent_hw_for_output(pipe: FluxKontextPipeline, out_h: int, out_w: int) -> Tuple[int, int]:
    vsf = int(getattr(pipe, "vae_scale_factor", 8))
    latent_h = 2 * (int(out_h) // (vsf * 2))
    latent_w = 2 * (int(out_w) // (vsf * 2))
    return latent_h, latent_w


def _scheduler_index_for_timestep(scheduler, t: torch.Tensor) -> int:
    if hasattr(scheduler, "index_for_timestep"):
        return int(scheduler.index_for_timestep(t))

    if hasattr(scheduler, "_init_step_index") and hasattr(scheduler, "step_index"):
        scheduler._init_step_index(t)
        return int(scheduler.step_index)

    timesteps = scheduler.timesteps
    t_val = t.to(device=timesteps.device)
    idxs = (timesteps == t_val).nonzero(as_tuple=False)
    if idxs.numel() == 0:
        diffs = (timesteps - t_val).abs()
        return int(torch.argmin(diffs).item())
    return int(idxs[0].item())


def extract_hair_mask(
    image: Image.Image,
    confidence_threshold: float = 0.4,
    prompt: str = "hair",
) -> tuple[Image.Image, float]:
    """Extract hair mask from an image using SAM.
    
    Args:
        image: PIL Image to extract hair mask from
        confidence_threshold: Confidence threshold for SAM (default: 0.4)
        prompt: Text prompt for SAM (default: "hair")
        
    Returns:
        tuple: (hair_mask_image, confidence_score)
        
    Raises:
        RuntimeError: If SAMMaskExtractor is not available
    """
    if SAMMaskExtractor is None:
        raise RuntimeError("SAMMaskExtractor is not available. Cannot extract hair mask.")
    
    sam = SAMMaskExtractor(confidence_threshold=0.5, detection_threshold=0.5)
    bg_remover = BackgroundRemover()
    try:
        _, silh_mask_pil = bg_remover.remove_background(image)
        hair_mask_pil, score = sam(image, prompt=prompt)
        # Ensure hair mask does not extend beyond the silhouette mask
        hair_mask_np = np.array(hair_mask_pil).astype(np.float32) / 255.0
        silh_mask_np = np.array(silh_mask_pil).astype(np.float32) / 255.0
        # Multiply masks to constrain hair to within silhouette
        constrained_hair_mask = (hair_mask_np * silh_mask_np * 255.0).astype(np.uint8)
        hair_mask_pil = Image.fromarray(constrained_hair_mask)  # Removed deprecated mode='L'
        return hair_mask_pil, score
    finally:
        del sam
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def truncate_hair_prompt(
    prompt: str,
    max_tokens: int = 70,  # Leave some buffer below 77
    verbose: bool = True,
) -> str:
    """Truncate hair prompt to fit within CLIP's token limit.
    
    CLIP text encoder has a 77 token limit. This function truncates
    the prompt at a sentence/clause boundary to avoid mid-word cuts.
    
    Args:
        prompt: The hair description prompt
        max_tokens: Maximum tokens (default 70 to leave buffer)
        verbose: Whether to print warnings
        
    Returns:
        Truncated prompt string
    """
    if not prompt:
        return prompt
    
    # Rough estimate: ~4 characters per token on average
    # This is a heuristic; actual tokenization varies
    estimated_tokens = len(prompt.split())
    
    if estimated_tokens <= max_tokens:
        return prompt
    
    # Try to truncate at natural boundaries (semicolon, comma, period)
    words = prompt.split()
    
    # Take approximately max_tokens words
    truncated_words = words[:max_tokens]
    truncated = ' '.join(truncated_words)
    
    # Try to end at a natural boundary
    for boundary in [';', ',', '.', ' and ', ' with ']:
        last_idx = truncated.rfind(boundary)
        if last_idx > len(truncated) * 0.6:  # Don't cut too much
            truncated = truncated[:last_idx + (1 if boundary in ';,.' else len(boundary))]
            break
    
    if verbose and len(words) > max_tokens:
        print(f"  [WARN] Hair prompt truncated from {len(words)} to ~{len(truncated.split())} words (CLIP 77 token limit)")
    
    return truncated.strip()


def load_hair_prompt_from_json(
    prompt_file: str,
    verbose: bool = True,
) -> Optional[str]:
    """Load hair description prompt from a JSON file.
    
    The JSON file is expected to have a structure like:
    {
        "subject": [
            {
                "hair_description": "long wavy blonde hair"
            }
        ]
    }
    
    Args:
        prompt_file: Path to the prompt JSON file
        verbose: Whether to print status messages
        
    Returns:
        Hair description string if found, None otherwise
    """
    if not os.path.exists(prompt_file):
        if verbose:
            print(f"  Prompt file not found: {prompt_file}")
        return None
    
    try:
        with open(prompt_file, 'r') as f:
            prompt_data = json.load(f)
        
        hair_prompt = prompt_data.get("subject", [{}])[0].get("hair_description")
        if hair_prompt and verbose:
            # Truncate for display only (T5 encoder supports 512+ tokens)
            display_prompt = hair_prompt[:50] + "..." if len(hair_prompt) > 50 else hair_prompt
            print(f"  Loaded hair prompt: {display_prompt}")
        return hair_prompt
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not load prompt from {prompt_file}: {e}")
        return None


# --------------------------------------------------------------------------------------
# Core editor
# --------------------------------------------------------------------------------------
@dataclass
class EditArgs:
    image_a_prime: str
    image_a: str
    image_b: str
    output: str
    height: int = 1024
    width: int = 1024
    steps: int = 50
    src_guidance_scale: float = 1.5
    tar_guidance_scale: float = 8.0
    stop_times: List[int] = None
    alphas: List[float] = None
    seed: int = 18
    src_prompt: str = "Same hair, change face"
    tar_prompt: str = "Same hair, change face"
    mask_file: Optional[str] = None
    mask_dilation_k: int = 7
    mask_blend_start_t: int = 100

    # RF-Inversion forward controller strength (gamma in the paper / official code)
    rf_gamma: float = 0.5
    
    # Reference hair image for dual conditioning + statistics transfer
    reference_hair_image: Optional[str] = None
    reference_hair_mask: Optional[str] = None
    
    # Statistics transfer parameters
    stats_transfer_strength: float = 0.35  # Max strength for AdaIN transfer
    stats_injection_start_t: float = 350.0  # Start transferring below this t
    stats_injection_end_t: float = 50.0     # Full strength below this t
    use_dual_conditioning: bool = True      # Concatenate reference as 2nd cond image


class FluxKontextRFInversionEditor:
    def __init__(self, pipe: FluxKontextPipeline, device: torch.device):
        self.pipe = pipe
        self.device = device

        self.pipe.vae.to(device=self.device, dtype=torch.float32)
        self.pipe.vae.enable_tiling()

    @torch.inference_mode()
    def _encode_vae_mode(self, img_tensor: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", enabled=False):
            img_f32 = img_tensor.float()
            lat = self.pipe.vae.encode(img_f32).latent_dist.mode()
            lat = (lat - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor
        lat = lat.to(torch.bfloat16)  # diffusion in fp16/bf16
        return lat

    def _pack(self, latents_spatial: torch.Tensor, num_channels_latents: int) -> torch.Tensor:
        b, c, h, w = latents_spatial.shape
        return self.pipe._pack_latents(latents_spatial, b, num_channels_latents, h, w)

    def _make_ids(self, latent_h: int, latent_w: int, image_ids: bool, dtype: torch.dtype) -> torch.Tensor:
        ids = self.pipe._prepare_latent_image_ids(
            batch_size=1,
            height=latent_h // 2,
            width=latent_w // 2,
            device=self.device,
            dtype=dtype,
        )
        if image_ids:
            ids = ids.clone()
            ids[..., 0] = 1
        return ids

    def _predict_v(
        self,
        latents_seq: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled: torch.Tensor,
        guidance: Optional[torch.Tensor],
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        timestep = t.expand(latents_seq.shape[0]).to(latents_seq.dtype)
        noise_pred = self.pipe.transformer(
            hidden_states=latents_seq,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            pooled_projections=pooled,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        return noise_pred

    @torch.no_grad()
    def _rf_controlled_forward_step(
        self,
        z_edit: torch.Tensor,                 # [B, seq_edit, d]
        v_edit: torch.Tensor,                 # [B, seq_edit, d] base vector field u_t
        y1: torch.Tensor,                     # [B, seq_edit, d] terminal noise sample
        t_frac: torch.Tensor,                 # scalar tensor in [0,1)
        gamma: float,
        step_sigma: torch.Tensor,             # scalar tensor (sigma_prev - sigma_i)
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        RF-Inversion controlled forward ODE (Algorithm 1):
            u_cond = (y1 - Y_t) / (1 - t)
            u_hat  = u + gamma * (u_cond - u)
            Y_{t+Δ} = Y_t + u_hat * Δσ
        """
        denom = (1.0 - t_frac).clamp_min(eps)
        u_cond = (y1 - z_edit) / denom
        u_hat = v_edit + float(gamma) * (u_cond - v_edit)
        return z_edit + u_hat * step_sigma

    def process_single_sample(
        self,
        args: EditArgs,
        target_hair_prompt: Optional[str] = None,
        preserve_mask: bool = False,
        codeformer_enhancer: Optional[object] = None,
        codeformer_fidelity: float = 0.5,
        verbose: bool = True,
    ) -> None:
        if args.stop_times is None:
            args.stop_times = [900]
        if args.alphas is None:
            args.alphas = [0.0]

        # Reproducibility (kept)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        generator = torch.Generator(device=self.device).manual_seed(args.seed)

        H, W = args.height, args.width
        if verbose:
            print("Loading images...")
        image_A = preprocess_img(args.image_a, self.pipe, self.device, target_hw=(W, H))
        image_Ap = preprocess_img(args.image_a_prime, self.pipe, self.device, target_hw=(W, H))
        image_B = preprocess_img(args.image_b, self.pipe, self.device, target_hw=(W, H))

        # Load reference hair image if provided (for dual conditioning + statistics transfer)
        x0_ref = None
        ref_hair_mask_lat = None
        has_reference = args.reference_hair_image is not None and os.path.exists(args.reference_hair_image)
        
        if has_reference:
            if verbose:
                print(f"Loading reference hair image: {args.reference_hair_image}")
            image_ref = preprocess_img(args.reference_hair_image, self.pipe, self.device, target_hw=(W, H))
            x0_ref = self._encode_vae_mode(image_ref)
            
            # Load or compute reference hair mask
            if args.reference_hair_mask and os.path.exists(args.reference_hair_mask):
                ref_mask_pil = Image.open(args.reference_hair_mask).convert("L").resize((W, H), Image.Resampling.NEAREST)
                ref_mask01 = np.array(ref_mask_pil).astype(np.float32) / 255.0
                if verbose:
                    print(f"Loaded reference hair mask from: {args.reference_hair_mask}")
            else:
                # Compute reference hair mask using SAM
                if SAMMaskExtractor is not None:
                    pil_ref = preprocess_img(args.reference_hair_image, self.pipe, self.device, target_hw=(W, H), return_pil=True)
                    ref_mask_pil, ref_score = extract_hair_mask(pil_ref, confidence_threshold=0.3)
                    ref_mask01 = np.array(ref_mask_pil).astype(np.float32) / 255.0
                    if verbose:
                        print(f"Computed reference hair mask (SAM score: {ref_score:.3f})")
                    # Save computed mask if path provided
                    if args.reference_hair_mask:
                        _ensure_dir(os.path.dirname(args.reference_hair_mask))
                        ref_mask_pil.save(args.reference_hair_mask)
                        if verbose:
                            print(f"Saved reference hair mask to: {args.reference_hair_mask}")
                else:
                    print("[WARN] No reference hair mask and SAMMaskExtractor unavailable, disabling reference features")
                    has_reference = False
                    ref_mask01 = None
            
            if has_reference and ref_mask01 is not None:
                latent_h, latent_w = _latent_hw_for_output(self.pipe, H, W)
                ref_mask01_lat = _resize_mask(ref_mask01, latent_w, latent_h)
                ref_hair_mask_lat = torch.from_numpy(ref_mask01_lat).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Masks (kept exactly)
        if verbose:
            print("Preparing hair mask...")
        if args.mask_file is not None:
            m = Image.open(args.mask_file).convert("L").resize((W, H), Image.Resampling.NEAREST)
            mask01 = (np.array(m).astype(np.float32) / 255.0)
            # Use Gaussian blur for soft edges instead of dilation (avoids halo artifacts)
            mask01_soft = _blur_mask(mask01, k=args.mask_dilation_k)
        else:
            if SAMMaskExtractor is None:
                raise RuntimeError("No --mask_file provided and SAMMaskExtractor could not be imported.")
            pil_for_sam = preprocess_img(args.image_a_prime, self.pipe, self.device, target_hw=(W, H), return_pil=True)
            sam = SAMMaskExtractor(confidence_threshold=0.25)  # Lower threshold for higher res
            hair_mask_pil, score = sam(pil_for_sam, prompt="hair")
            if verbose:
                print(f"SAM mask score: {score:.3f}")
            mask01 = (np.array(hair_mask_pil).astype(np.float32) / 255.0)
            # Use Gaussian blur for soft edges instead of dilation (avoids halo artifacts)
            mask01_soft = _blur_mask(mask01, k=args.mask_dilation_k)

        latent_h, latent_w = _latent_hw_for_output(self.pipe, H, W)
        mask01_lat = _resize_mask(mask01, latent_w, latent_h)
        mask01_soft_lat = _resize_mask(mask01_soft, latent_w, latent_h)

        mask_phase1 = torch.from_numpy(mask01_lat).unsqueeze(0).unsqueeze(0).to(self.device)
        mask_phase1_soft = torch.from_numpy(mask01_soft_lat).unsqueeze(0).unsqueeze(0).to(self.device)

        # Encode images (kept)
        if verbose:
            print("Encoding images to latent space...")
        x0_A = self._encode_vae_mode(image_A)
        x0_B = self._encode_vae_mode(image_B)
        x0_Ap = self._encode_vae_mode(image_Ap)

        # Pack + ids (kept)
        num_channels_latents = self.pipe.transformer.config.in_channels // 4
        xA_packed = self._pack(x0_A, num_channels_latents)
        xB_packed = self._pack(x0_B, num_channels_latents)
        zt_edit = self._pack(x0_Ap, num_channels_latents)

        # Pack reference if available (for dual conditioning)
        xRef_packed = None
        if has_reference and x0_ref is not None:
            xRef_packed = self._pack(x0_ref, num_channels_latents)
            if verbose:
                print(f"Reference hair latents packed for dual conditioning")

        ids_lat = self._make_ids(
            latent_h=x0_Ap.shape[2], latent_w=x0_Ap.shape[3], image_ids=False, dtype=zt_edit.dtype
        )
        ids_img = self._make_ids(
            latent_h=x0_A.shape[2], latent_w=x0_A.shape[3], image_ids=True, dtype=zt_edit.dtype
        )
        
        # Create IDs for reference image (distinct from primary conditioning)
        # Use id value of 2 to distinguish from primary conditioning (id=1)
        ids_ref = None
        if has_reference and xRef_packed is not None:
            ids_ref = self._make_ids(
                latent_h=x0_ref.shape[2], latent_w=x0_ref.shape[3], image_ids=True, dtype=zt_edit.dtype
            )
            # Mark reference with different ID to help model distinguish
            ids_ref = ids_ref.clone()
            ids_ref[..., 0] = 2  # Different from primary (1) and latents (0)

        # Scheduler / timesteps (kept)
        scheduler = self.pipe.scheduler
        sigmas_np = np.linspace(1.0, 1 / args.steps, args.steps).tolist()

        base_seq = int(getattr(scheduler.config, "base_image_seq_len", 256)) if hasattr(scheduler, "config") else 256
        max_seq = int(getattr(scheduler.config, "max_image_seq_len", 4096)) if hasattr(scheduler, "config") else 4096
        base_shift = float(getattr(scheduler.config, "base_shift", 0.5)) if hasattr(scheduler, "config") else 0.5
        max_shift = float(getattr(scheduler.config, "max_shift", 1.15)) if hasattr(scheduler, "config") else 1.15

        image_seq_len = int(zt_edit.shape[1])
        mu = calculate_shift(image_seq_len, base_seq, max_seq, base_shift, max_shift)

        timesteps, _ = retrieve_timesteps(
            scheduler,
            args.steps,
            self.device,
            sigmas=sigmas_np,
            mu=mu,
        )
        sigmas = scheduler.sigmas.to(self.device)

        # Encode prompts (kept)
        if verbose:
            print("Encoding prompts...")
        with torch.no_grad():
            src_prompt_embeds, src_pooled, src_txt_ids = self.pipe.encode_prompt(
                prompt=args.src_prompt,
                prompt_2=None,
                device=self.device,
            )

            # CLIP (prompt): Short instruction (77 token limit)
            # T5 (prompt_2): Full hair description (512+ token limit)
            clip_prompt = "Add hair with strand-level details. Connect the hair naturally to the scalp and face."
            t5_prompt = target_hair_prompt  # T5 can handle full description
            tar_prompt_embeds, tar_pooled, tar_txt_ids = self.pipe.encode_prompt(
                prompt=clip_prompt,
                prompt_2=t5_prompt,
                device=self.device,
            )

            if self.pipe.transformer.config.guidance_embeds:
                src_guidance = torch.full([1], args.src_guidance_scale, device=self.device, dtype=torch.float32).expand(
                    zt_edit.shape[0]
                )
                tar_guidance = torch.full([1], args.tar_guidance_scale, device=self.device, dtype=torch.float32).expand(
                    zt_edit.shape[0]
                )
            else:
                src_guidance = None
                tar_guidance = None

        
        max_stop_time = max(args.stop_times)

        def _nearest_timestep_index(timesteps_tensor: torch.Tensor, stop_time_val: float) -> int:
            diffs = (timesteps_tensor.float() - float(stop_time_val)).abs()
            return int(torch.argmin(diffs).item())

        # Cache inversion latents by scheduler index (Fix B)
        inversion_latents_by_sidx: List[Optional[torch.Tensor]] = [None] * len(timesteps)

        # -----------------------------
        # Compute the *maximum* boundary we need to invert to (supports multiple stop_times)
        # -----------------------------
        max_stop_t_idx = _nearest_timestep_index(timesteps, float(max_stop_time))
        t_max_stop = timesteps[max_stop_t_idx]
        sidx_max_stop = _scheduler_index_for_timestep(scheduler, t_max_stop)

        # Inversion runs from low-noise -> high-noise, i.e. reverse order of timesteps (which are usually high->low)
        rev_timesteps = torch.flip(timesteps, dims=[0])

        # Find where t_max_stop sits in the reversed list
        # If timesteps are strictly reversed, this is simply:
        rev_stop_idx = (len(timesteps) - 1) - max_stop_t_idx

        # Only invert the subset needed up to the max stop boundary (no break; tqdm count is correct)
        inv_timesteps = rev_timesteps[: rev_stop_idx + 1]

        if verbose:
            print(
                f"Phase A (RF-Inversion): Controlled forward ODE up to requested max_stop_time={max_stop_time} "
                f"(nearest t={float(t_max_stop.item()):.3f}, gamma={args.rf_gamma})..."
            )
            print(f"  Inversion will run {len(inv_timesteps)}/{len(timesteps)} scheduler steps.")

        amp_dtype = _autocast_dtype()

        with torch.no_grad():
            y1 = torch.randn(
                zt_edit.shape,
                device=self.device,
                dtype=zt_edit.dtype,
                generator=generator,
            )

            for t in tqdm(inv_timesteps, desc="RF-Inversion (forward)"):
                sidx = _scheduler_index_for_timestep(scheduler, t)
                prev_sidx = max(sidx - 1, 0)

                step_sigma = (sigmas[prev_sidx] - sigmas[sidx]).to(dtype=torch.float32)
                t_frac = (t.to(torch.float32) / 1000.0).clamp(0.0, 0.999999)

                z_with_cond = torch.cat((zt_edit, xA_packed), dim=1)
                ids_with_cond = torch.cat((ids_lat, ids_img), dim=0)

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                    v_full = self._predict_v(
                        latents_seq=z_with_cond,
                        prompt_embeds=src_prompt_embeds,
                        pooled=src_pooled,
                        guidance=src_guidance,
                        txt_ids=src_txt_ids,
                        img_ids=ids_with_cond,
                        t=t,
                    )

                v_edit = v_full[:, : zt_edit.shape[1]]

                zt_edit = self._rf_controlled_forward_step(
                    z_edit=zt_edit.float(),
                    v_edit=v_edit.float(),
                    y1=y1.float(),
                    t_frac=t_frac,
                    gamma=args.rf_gamma,
                    step_sigma=step_sigma,
                ).to(v_full.dtype)

                inversion_latents_by_sidx[sidx] = zt_edit.detach().clone()

        # -----------------------------
        # Phase B: Generation / denoise
        # Fix: include the boundary timestep itself (stop_t_idx, not stop_t_idx+1)
        # -----------------------------
        with torch.inference_mode():
            for stop_time in args.stop_times:
                stop_t_idx = _nearest_timestep_index(timesteps, float(stop_time))
                t_stop = timesteps[stop_t_idx]
                sidx_stop = _scheduler_index_for_timestep(scheduler, t_stop)

                for alpha in args.alphas:
                    if verbose:
                        print(
                            f"Phase B: Generating with target prompt "
                            f"(stop_time={stop_time} -> nearest t={float(t_stop.item()):.3f}, alpha={alpha})"
                        )
                        print(f"  Generation will run {len(timesteps) - stop_t_idx}/{len(timesteps)} scheduler steps.")

                    z0 = inversion_latents_by_sidx[sidx_stop]
                    if z0 is None:
                        raise RuntimeError(
                            f"Missing inversion latent at stop boundary (scheduler_idx={sidx_stop}, "
                            f"t={float(t_stop.item())})."
                        )
                    zt_edit = z0.clone()

                    # Apply low-pass ONCE at the boundary if desired
                    if alpha != 0.0:
                        unpacked = self.pipe._unpack_latents(zt_edit, H, W, self.pipe.vae_scale_factor)
                        unpacked = lowpass_fft_latents(unpacked, cutoff=0.1, alpha=alpha)
                        zt_edit = self.pipe._pack_latents(
                            unpacked,
                            unpacked.shape[0],
                            num_channels_latents,
                            unpacked.shape[2],
                            unpacked.shape[3],
                        )

                    # IMPORTANT FIX:
                    # Start denoising at t_stop itself so we apply the sigma step from sidx_stop -> next_sidx
                    for t in tqdm(timesteps[stop_t_idx:], desc="Generation"):
                        sidx = _scheduler_index_for_timestep(scheduler, t)
                        next_sidx = min(sidx + 1, len(sigmas) - 1)
                        t_i, t_im1 = sigmas[sidx], sigmas[next_sidx]

                        if t > args.mask_blend_start_t:
                            old = inversion_latents_by_sidx[sidx]
                            if old is not None:
                                new_zt = self.pipe._unpack_latents(zt_edit, H, W, self.pipe.vae_scale_factor)
                                old_zt = self.pipe._unpack_latents(old, H, W, self.pipe.vae_scale_factor)

                                if preserve_mask:
                                    # Use NON-blurred mask for hair injection (preserves detail at edges)
                                    # Hair region: 100% from inversion (no generation smoothing)
                                    new_zt = new_zt * (1 - mask_phase1) + old_zt * mask_phase1
                                else:
                                    new_zt = old_zt * (1 - mask_phase1_soft) + new_zt * mask_phase1_soft

                                zt_edit = self.pipe._pack_latents(
                                    new_zt,
                                    new_zt.shape[0],
                                    num_channels_latents,
                                    new_zt.shape[2],
                                    new_zt.shape[3],
                                )
                        
                        # -----------------------------------------------------------------
                        # Reference Hair Statistics Transfer (AdaIN-style)
                        # Transfer color/texture from reference without spatial structure
                        # -----------------------------------------------------------------
                        if has_reference and x0_ref is not None and ref_hair_mask_lat is not None:
                            injection_strength = compute_injection_strength_schedule(
                                t,
                                injection_start_t=args.stats_injection_start_t,
                                injection_end_t=args.stats_injection_end_t,
                                max_strength=args.stats_transfer_strength,
                            )
                            
                            if injection_strength > 0:
                                # Unpack current latents to spatial format
                                zt_spatial = self.pipe._unpack_latents(zt_edit, H, W, self.pipe.vae_scale_factor)
                                
                                # Apply statistics transfer
                                zt_spatial = transfer_hair_statistics(
                                    target_latents=zt_spatial,
                                    ref_latents=x0_ref,
                                    target_hair_mask=mask_phase1,
                                    ref_hair_mask=ref_hair_mask_lat,
                                    strength=injection_strength,
                                )
                                
                                # Re-pack
                                zt_edit = self.pipe._pack_latents(
                                    zt_spatial,
                                    zt_spatial.shape[0],
                                    num_channels_latents,
                                    zt_spatial.shape[2],
                                    zt_spatial.shape[3],
                                )

                        # -----------------------------------------------------------------
                        # Dual Conditioning: Concatenate reference as 2nd conditioning image
                        # -----------------------------------------------------------------
                        if has_reference and args.use_dual_conditioning and xRef_packed is not None and ids_ref is not None:
                            # Dual conditioning: [latents, bald_cond, ref_hair_cond]
                            z_with_cond = torch.cat((zt_edit, xB_packed, xRef_packed), dim=1)
                            ids_with_cond = torch.cat((ids_lat, ids_img, ids_ref), dim=0)
                        else:
                            # Standard single conditioning: [latents, bald_cond]
                            z_with_cond = torch.cat((zt_edit, xB_packed), dim=1)
                            ids_with_cond = torch.cat((ids_lat, ids_img), dim=0)

                        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
                            v_full = self._predict_v(
                                latents_seq=z_with_cond,
                                prompt_embeds=tar_prompt_embeds,
                                pooled=tar_pooled,
                                guidance=tar_guidance,
                                txt_ids=tar_txt_ids,
                                img_ids=ids_with_cond,
                                t=t,
                            )

                        v_edit = v_full[:, : zt_edit.shape[1]]
                        dt = (t_im1 - t_i).float()
                        
                        # CRITICAL FIX: Apply velocity only to FACE region, not hair
                        # This prevents the bald-conditioned model from smoothing/saturating the hair
                        if preserve_mask and t > args.mask_blend_start_t:
                            # Unpack to spatial format for masking
                            zt_unpacked = self.pipe._unpack_latents(zt_edit, H, W, self.pipe.vae_scale_factor)
                            v_unpacked = self.pipe._unpack_latents(
                                v_edit.unsqueeze(0) if v_edit.dim() == 2 else v_edit, 
                                H, W, self.pipe.vae_scale_factor
                            )
                            
                            # Apply velocity only to non-hair regions (face/background)
                            # Hair region gets NO denoising - stays exactly as inverted
                            zt_new = zt_unpacked.float() + dt * v_unpacked.float() * (1 - mask_phase1)
                            
                            zt_edit = self.pipe._pack_latents(
                                zt_new.to(v_full.dtype),
                                zt_new.shape[0],
                                num_channels_latents,
                                zt_new.shape[2],
                                zt_new.shape[3],
                            )
                        else:
                            # Early timesteps or no preservation: normal full denoising
                            zt_edit = (zt_edit.float() + dt * v_edit.float()).to(v_full.dtype)

                    # Decode
                    if verbose:
                        print("Decoding latents to image...")
                    unpacked = self.pipe._unpack_latents(zt_edit, H, W, self.pipe.vae_scale_factor)
                    x0_denorm = (unpacked / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor

                    with torch.autocast(device_type="cuda", dtype=torch.float32, enabled=torch.cuda.is_available()):
                        img = self.pipe.vae.decode(x0_denorm, return_dict=False)[0]

                    result_image = self.pipe.image_processor.postprocess(img)[0]

                    out_dir = os.path.dirname(args.output)
                    _ensure_dir(out_dir)

                    if len(args.stop_times) > 1 or len(args.alphas) > 1:
                        out_path = f"{args.output.rsplit('.', 1)[0]}_{stop_time}_{alpha}.png"
                    else:
                        out_path = args.output

                    result_image.save(out_path)
                    if verbose:
                        print(f"Output saved to: {out_path}")
                        print(f"Output resolution: {result_image.size}")


# --------------------------------------------------------------------------------------
# Hair Compositing Utilities
# --------------------------------------------------------------------------------------
def composite_hair_onto_bald(
    hair_restored_np: np.ndarray,
    bald_np: np.ndarray,
    hair_mask_np: np.ndarray,
    use_multiscale: bool = True,
    feather_px: int = 12,
) -> np.ndarray:
    """
    Composite hair region from hair_restored onto the bald source image.
    
    This creates a better starting point for identity restoration by keeping
    the bald image's identity/background intact while only transferring hair.
    
    Args:
        hair_restored_np: Hair restored image as numpy array (H, W, 3)
        bald_np: Bald source image as numpy array (H, W, 3)
        hair_mask_np: Binary hair mask as numpy array (H, W), values 0-255 or 0-1
        use_multiscale: Whether to use multi-scale Laplacian blending
        feather_px: Feather size for soft mask edges
        
    Returns:
        Composited image as numpy array (H, W, 3)
    """
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
        blend_masks = _create_hierarchical_blend_mask(hair_mask_np, num_levels=4)
        composited = _multi_scale_blend(
            hair_restored_np,
            bald_np,
            blend_masks,
            use_laplacian=True
        )
    else:
        # Simple soft alpha blending
        soft_mask = _create_distance_soft_blend_mask(
            hair_mask_np,
            dilation_px=3,
            dilation_iterations=1,
            feather_px=feather_px,
        )
        alpha = soft_mask[:, :, np.newaxis]
        composited = (hair_restored_np * alpha + bald_np * (1 - alpha)).astype(np.uint8)
    
    return composited


def _create_distance_soft_blend_mask(
    hair_mask: np.ndarray,
    dilation_px: int = 5,
    dilation_iterations: int = 1,
    feather_px: int = 15,
) -> np.ndarray:
    """Create soft blend mask using distance transform."""
    mask = (hair_mask > 0.5).astype(np.uint8)
    if dilation_px > 0 and dilation_iterations > 0:
        kernel = np.ones((dilation_px * 2 + 1, dilation_px * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
    outside = (1 - mask).astype(np.uint8)
    dist_outside = cv2.distanceTransform(outside, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)
    outside_alpha = np.clip(1.0 - (dist_outside / float(feather_px)), 0.0, 1.0)
    soft = np.where(mask > 0, 1.0, outside_alpha)
    return soft.astype(np.float32)


def _create_hierarchical_blend_mask(hair_mask: np.ndarray, num_levels: int = 4) -> list:
    """Create multi-scale blend masks."""
    masks = []
    current_mask = (hair_mask > 0.5).astype(np.float32)
    for level in range(num_levels):
        blur_size = 5 + level * 8
        if blur_size % 2 == 0:
            blur_size += 1
        kernel_size = 3 + level * 4
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1)
        blurred = cv2.GaussianBlur(dilated.astype(np.float32), (blur_size, blur_size), 0)
        masks.append(blurred)
    return masks


def _multi_scale_blend(
    generated: np.ndarray,
    source: np.ndarray,
    masks: list,
    use_laplacian: bool = True
) -> np.ndarray:
    """Multi-scale blending using Laplacian pyramids."""
    if not use_laplacian or len(masks) < 2:
        alpha = masks[0][:, :, np.newaxis]
        return (generated * alpha + source * (1 - alpha)).astype(np.uint8)
    num_levels = len(masks)

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

    gen_pyr = build_laplacian_pyramid(generated.astype(np.float32), num_levels)
    src_pyr = build_laplacian_pyramid(source.astype(np.float32), num_levels)
    blended_pyr = []
    for i in range(num_levels):
        h, w = gen_pyr[i].shape[:2]
        mask_resized = cv2.resize(masks[i], (w, h))
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]
        blended = gen_pyr[i] * mask_resized + src_pyr[i] * (1 - mask_resized)
        blended_pyr.append(blended)
    result = blended_pyr[-1]
    for i in range(num_levels - 2, -1, -1):
        result = cv2.pyrUp(result, dstsize=(blended_pyr[i].shape[1], blended_pyr[i].shape[0]))
        result = result + blended_pyr[i]
    return np.clip(result, 0, 255).astype(np.uint8)




# --------------------------------------------------------------------------------------
# Phase wrappers
# --------------------------------------------------------------------------------------
def run_id_restoration(
    editor: FluxKontextRFInversionEditor,
    image_a_prime_path: str,
    image_a_path: str,
    image_b_path: str,
    output_path: str,
    height: int = 1024,
    width: int = 1024,
    steps: int = 40,
    src_guidance_scale: float = 5.5,
    tar_guidance_scale: float = 5.5,
    stop_times: List[int] = None,
    alphas: List[float] = None,
    seed: int = 18,
    mask_file: Optional[str] = None,
    rf_gamma: float = 0.5,
    hair_prompt: Optional[str] = None,
    # Reference hair parameters for style transfer
    reference_hair_image: Optional[str] = None,
    reference_hair_mask: Optional[str] = None,
    stats_transfer_strength: float = 0.15,
    stats_injection_start_t: float = 350.0,
    stats_injection_end_t: float = 50.0,
    use_dual_conditioning: bool = True,
    verbose: bool = True,
):
    args = EditArgs(
        image_a_prime=image_a_prime_path,
        image_a=image_a_path,
        image_b=image_b_path,
        output=output_path,
        height=height,
        width=width,
        steps=35,  # Tuned default for 1024x1024; override via run_id_restoration(..., steps=...)
        src_guidance_scale=1.5,
        tar_guidance_scale=2.5,
        stop_times=stop_times or [840],  # Tuned for 1024x1024 (was 750 at 1024)
        alphas=alphas or [0.0],
        seed=seed,
        src_prompt="",
        tar_prompt="Add hair with strand-level details. Connect the hair naturally to the scalp and face.",
        mask_blend_start_t=0.0,  # ~250 * (1024/1024)
        mask_dilation_k=40,  # ~10 * (1024/1024)
        rf_gamma=0.85,
        # Reference hair parameters
        reference_hair_image=reference_hair_image,
        reference_hair_mask=reference_hair_mask,
        stats_transfer_strength=stats_transfer_strength,
        stats_injection_start_t=stats_injection_start_t,
        stats_injection_end_t=stats_injection_end_t,
        use_dual_conditioning=True,
    )

    if verbose:
        print("\nIdentity Restoration Configuration:")
        print(f"  image_a_prime: {args.image_a_prime}")
        print(f"  image_a:       {args.image_a}")
        print(f"  image_b:       {args.image_b}")
        print(f"  output:        {args.output}")
        print(f"  HxW:           {args.height}x{args.width}")
        print(f"  rf_gamma:      {args.rf_gamma}")
        print(f"  hair_prompt:   {args.tar_prompt}")
        if mask_file:
            print(f"  mask_file:     {mask_file}")
        if reference_hair_image:
            print(f"  reference_hair_image: {reference_hair_image}")
            print(f"  reference_hair_mask:  {reference_hair_mask}")
            print(f"  stats_transfer_strength: {stats_transfer_strength}")
            print(f"  use_dual_conditioning: {use_dual_conditioning}")

    editor.process_single_sample(
        args=args,
        target_hair_prompt=hair_prompt,
        preserve_mask=True,
        codeformer_enhancer=None,
        codeformer_fidelity=1.0,
        verbose=verbose,
    )

# --------------------------------------------------------------------------------------
# Processing logic
# --------------------------------------------------------------------------------------
def process_single_sample(
    editor: FluxKontextRFInversionEditor,
    hair_restored_file: str,
    source_bald_file: str,
    output_dir: str,
    sample_name: str = "sample",
    rf_gamma: float = 0.5,
    hair_prompt: Optional[str] = None,
    # Reference hair parameters
    reference_hair_image: Optional[str] = None,
    reference_hair_mask: Optional[str] = None,
    stats_transfer_strength: float = 0.35,
    use_dual_conditioning: bool = True,
):
    """
    Process a single sample for identity restoration with reference hairstyle support.
    
    Pipeline:
    1. Extract hair mask from hair_restored.png
    2. Composite hair region onto bald source using pixel-space blending
    3. Pass composited image to identity restoration with optional reference hairstyle
    4. Extract final hair mask from output
    
    Args:
        editor: FluxKontextRFInversionEditor instance
        hair_restored_file: Path to hair_restored.png (output from restore_hair.py)
        source_bald_file: Path to the source bald image
        output_dir: Directory to save outputs
        sample_name: Name for logging
        rf_gamma: RF-Inversion gamma parameter
        hair_prompt: Optional text description of hair to guide generation
        reference_hair_image: Path to reference hairstyle image (different person)
        reference_hair_mask: Path to reference hair mask (will be computed if not provided)
        stats_transfer_strength: Strength of statistics transfer from reference (0-1)
        use_dual_conditioning: Whether to use reference as dual conditioning image
    """
    _ensure_dir(output_dir)
    
    # Create postprocessing subdirectory
    postprocessing_dir = os.path.join(output_dir, "postprocessing")
    _ensure_dir(postprocessing_dir)
    
    # Define output paths
    hair_restored_mask_path = os.path.join(postprocessing_dir, "hair_restored_mask.png")
    composited_input_path = os.path.join(postprocessing_dir, "composited_input.png")
    id_restoration_output = os.path.join(postprocessing_dir, "id_restored.png")
    id_restored_mask_path = os.path.join(postprocessing_dir, "id_restored_mask.png")
    
    # Skip if both output files already exist
    if os.path.exists(id_restoration_output) and os.path.exists(id_restored_mask_path):
        print(f"Skipping {sample_name}: id_restored.png and id_restored_mask.png already exist")
        return
    
    # Check input files exist
    if not os.path.exists(hair_restored_file):
        raise FileNotFoundError(f"Input file not found: {hair_restored_file}")
    if not os.path.exists(source_bald_file):
        raise FileNotFoundError(f"Source bald file not found: {source_bald_file}")
    
    print(f"\n{'='*60}")
    print(f"Processing: {sample_name}")
    print(f"{'='*60}")
    
    # Load images
    hair_restored_image = Image.open(hair_restored_file).convert("RGB")
    bald_image = Image.open(source_bald_file).convert("RGB")
    
    # Resize to common resolution for processing
    target_size = (1024, 1024)
    hair_restored_resized = hair_restored_image.resize(target_size, Image.Resampling.LANCZOS)
    bald_resized = bald_image.resize(target_size, Image.Resampling.LANCZOS)
    
    hair_restored_np = np.array(hair_restored_resized)
    bald_np = np.array(bald_resized)
    print(bald_np.shape)
    print(hair_restored_np.shape)
    # Step 1: Extract hair mask from hair_restored image
    print("\n--- Step 1: Extracting hair mask from hair_restored image ---")
    hair_mask_np = None
    if os.path.exists(hair_restored_mask_path):
        # Load existing mask instead of recomputing
        hair_mask_pil = Image.open(hair_restored_mask_path).convert("L")
        hair_mask_np = np.array(hair_mask_pil).astype(np.float32) / 255.0
        print(f"Loaded existing hair mask from: {hair_restored_mask_path}")
    elif SAMMaskExtractor is not None:
        try:
            hair_mask_pil, mask_score = extract_hair_mask(hair_restored_resized, confidence_threshold=0.4)
            hair_mask_pil.save(hair_restored_mask_path)
            hair_mask_np = np.array(hair_mask_pil).astype(np.float32) / 255.0
            print(f"Hair mask saved to: {hair_restored_mask_path} (score: {mask_score:.3f})")
        except Exception as e:
            print(f"Warning: Failed to extract hair mask: {e}")
    else:
        print("Warning: SAMMaskExtractor not available")
    
    # Step 2: Composite hair onto bald source
    print("\n--- Step 2: Compositing hair onto bald source image ---")
    if hair_mask_np is not None:
        composited_np = composite_hair_onto_bald(
            hair_restored_np,
            bald_np,
            hair_mask_np,
            use_multiscale=True,
            feather_px=9,
        )
        print(f"Composited image shape: {composited_np.shape}")
        composited_pil = Image.fromarray(composited_np)
        composited_pil.save(composited_input_path)
        print(f"Composited input saved to: {composited_input_path}")
        input_for_restoration = composited_input_path
    else:
        # Fallback: use hair_restored directly if mask extraction failed
        print("Using hair_restored directly (mask extraction failed)")
        input_for_restoration = hair_restored_file
    # Step 3: Run identity restoration on composited image
    print("\n--- Step 3: Running identity restoration ---")
    if hair_prompt:
        print(f"Using hair prompt: {hair_prompt}")
    if reference_hair_image and os.path.exists(reference_hair_image):
        print(f"Using reference hairstyle: {reference_hair_image}")
        print(f"  Stats transfer strength: {stats_transfer_strength}")
        print(f"  Dual conditioning: {use_dual_conditioning}")
    run_id_restoration(
        editor=editor,
        image_a_prime_path=input_for_restoration,  # Composited image (hair on bald)
        image_a_path=source_bald_file,             # Condition on bald during inversion
        image_b_path=source_bald_file,             # Condition on bald during generation
        output_path=id_restoration_output,
        rf_gamma=rf_gamma,
        hair_prompt=hair_prompt,
        # Reference hair parameters
        reference_hair_image=reference_hair_image,
        reference_hair_mask=reference_hair_mask,
        stats_transfer_strength=stats_transfer_strength,
        use_dual_conditioning=use_dual_conditioning,
        verbose=True,
    )
    
    # Step 4: Compute hair mask on id_restored image
    print("\n--- Step 4: Computing hair mask on id_restored image ---")
    if os.path.exists(id_restoration_output) and SAMMaskExtractor is not None:
        try:
            id_restored_image = Image.open(id_restoration_output).convert("RGB")
            hair_mask_pil, mask_score = extract_hair_mask(id_restored_image, confidence_threshold=0.5)
            hair_mask_pil.save(id_restored_mask_path)
            print(f"Hair mask saved to: {id_restored_mask_path} (score: {mask_score:.3f})")
        except Exception as e:
            print(f"Warning: Failed to extract hair mask from output: {e}")
    elif not os.path.exists(id_restoration_output):
        print(f"Warning: id_restored.png not found at {id_restoration_output}")
    else:
        print("Warning: SAMMaskExtractor not available, skipping output hair mask extraction")
    
    print(f"\n✓ Completed {sample_name}")


# --------------------------------------------------------------------------------------
# Identity Restoration Output Filename Builder
# --------------------------------------------------------------------------------------
def build_id_restoration_output_filename(
    config: HairTransferConfig,
    bald_version: str,
    is_3d_aware: bool,
    suffix: str = "",
) -> str:
    """Build output filename for identity restoration based on configuration.
    
    Args:
        config: HairTransferConfig instance
        bald_version: Version of bald image (w_seg or wo_seg)
        is_3d_aware: Whether aligned image is actually used
        suffix: Additional suffix (e.g., '_mask')
    
    Returns:
        Filename like 'id_restored_w_seg_3d_aware.png'
    """
    parts = [bald_version]
    parts.append("3d_aware" if is_3d_aware else "3d_unaware")
    return "id_restored_" + "_".join(parts) + suffix + ".png"


# --------------------------------------------------------------------------------------
# Process Single Sample (for batch processing)
# --------------------------------------------------------------------------------------
def process_sample_for_setting(
    editor: FluxKontextRFInversionEditor,
    folder: str,
    data_dir: str,
    config: HairTransferConfig,
    bald_version: str,
    conditioning_mode: str,
    rf_gamma: float = 0.5,
    hair_prompt_override: Optional[str] = None,
    stats_transfer_strength: float = 0.35,
    use_dual_conditioning: bool = True,
    skip_existing: bool = True,
) -> bool:
    """
    Process a single sample folder for a given bald_version and conditioning_mode.
    
    This function mirrors the structure from restore_hair.py's process_sample function.
    
    Args:
        editor: FluxKontextRFInversionEditor instance
        folder: Path to the sample folder (e.g., /workspace/outputs/view_aligned/.../target_to_source)
        data_dir: Root data directory
        config: HairTransferConfig instance
        bald_version: Version of bald image (w_seg or wo_seg)
        conditioning_mode: '3d_aware' or '3d_unaware'
        rf_gamma: RF-Inversion gamma parameter
        hair_prompt_override: Optional hair prompt (overrides JSON)
        stats_transfer_strength: Strength of statistics transfer from reference
        use_dual_conditioning: Whether to use reference as dual conditioning image
        skip_existing: Whether to skip already processed samples
    
    Returns:
        True if processing succeeded, False otherwise
    """
    from pathlib import Path
    folder = Path(folder)
    data_dir = Path(data_dir)
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
    
    # Determine 3D aware status (same logic as restore_hair.py)
    mode_dir = config.DIR_3D_AWARE if conditioning_mode == "3d_aware" else config.DIR_3D_UNAWARE
    warping_dir = pair_dir / mode_dir / config.SUBDIR_WARPING
    
    warped_image_path = warping_dir / config.FILE_WARPED_TARGET_IMAGE
    warped_hair_mask_path = warping_dir / config.FILE_WARPED_HAIR_MASK
    has_warped_outputs = warped_image_path.exists() and warped_hair_mask_path.exists()
    
    camera_params_path = pair_dir / config.FILE_CAMERA_PARAMS
    alignment_dir = pair_dir / config.DIR_ALIGNMENT
    view_aligned_image_path = alignment_dir / config.FILE_VIEW_ALIGNED_IMAGE
    lift_3d_applied = camera_params_path.exists()
    aligned_image_exists = lift_3d_applied and view_aligned_image_path.exists()
    
    # Determine processing mode based on conditioning_mode
    if conditioning_mode == "3d_aware":
        use_3d_aware = has_warped_outputs or aligned_image_exists
        if not use_3d_aware:
            print(f"No 3D lifting for {folder_name}/{bald_version}, skipping 3d_aware")
            return False
    elif conditioning_mode == "3d_unaware":
        use_3d_aware = False
    else:
        raise ValueError(f"Invalid conditioning_mode: {conditioning_mode}")
    
    # Build paths to hair_restored output from restore_hair.py
    # New structure: {pair_dir}/{3d_aware|3d_unaware}/transferred/hair_restored.png
    mode_dir_name = config.DIR_3D_AWARE if use_3d_aware else config.DIR_3D_UNAWARE
    transferred_dir = pair_dir / mode_dir_name / config.SUBDIR_TRANSFERRED
    
    # Input files from restore_hair.py (new simple naming)
    hair_restored_file = transferred_dir / config.FILE_HAIR_RESTORED
    hair_restored_mask_file = transferred_dir / config.FILE_HAIR_RESTORED_MASK
    
    # Build output paths for identity restoration (same directory)
    id_restored_output = transferred_dir / "id_restored.png"
    id_restored_mask_output = transferred_dir / "id_restored_mask.png"
    composited_input_path = transferred_dir / "composited_input.png"
    
    # Check if input hair_restored file exists
    if not hair_restored_file.exists():
        print(f"Skipping {folder_name}/{bald_version}/{conditioning_mode}: missing {hair_restored_file}")
        return False
    
    # Skip if output already exists
    if skip_existing and id_restored_output.exists() and id_restored_mask_output.exists():
        print(f"Output already exists for {folder_name}/{bald_version}/{conditioning_mode}, skipping...")
        return True
    
    # Resolve source bald image path (prefer outpainted, fallback to bald)
    outpainted_source_path = pair_dir / config.DIR_SOURCE_OUTPAINTED / config.FILE_OUTPAINTED_IMAGE
    
    if outpainted_source_path.exists():
        source_bald_file = outpainted_source_path
    else:
        bald_image_dir = data_dir / config.DIR_BALD / bald_version / config.SUBDIR_BALD_IMAGE
        source_bald_file = bald_image_dir / f"{source_id}.png"
    
    # Fallback to image directory if bald not found
    if not source_bald_file.exists():
        alternative_source_path = data_dir / "image" / f"{source_id}.png"
        if alternative_source_path.exists():
            source_bald_file = alternative_source_path
        else:
            print(f"  Warning: Source not found: {source_bald_file}")
            return False
    
    # Reference hair image and mask paths (for reference hairstyle transfer)
    reference_hair_image = None
    reference_hair_mask = None
    
    if has_warped_outputs:
        reference_hair_image = warped_image_path
        reference_hair_mask = warped_hair_mask_path
    elif use_3d_aware and aligned_image_exists:
        reference_hair_image = view_aligned_image_path
        reference_hair_mask = reference_hair_image.parent / (reference_hair_image.stem + "_hair_mask.png")
    else:
        matted_image_path = data_dir / config.MATTED_IMAGE_SUBDIR / f"{target_id}.png"
        matted_mask_path = data_dir / config.MATTED_IMAGE_HAIR_MASK_SUBDIR / f"{target_id}.png"
        if matted_image_path.exists():
            reference_hair_image = matted_image_path
            reference_hair_mask = matted_mask_path if matted_mask_path.exists() else None
    
    # Load hair prompt
    hair_prompt = hair_prompt_override
    if hair_prompt is None:
        prompts_dir = data_dir / config.DIR_PROMPTS
        try:
            prompt_file = prompts_dir / f"{target_id}.json"
            if prompt_file.exists():
                with open(prompt_file, 'r') as f:
                    prompt_data = json.load(f)
                hair_prompt = prompt_data.get("subject", [{}])[0].get("hair_description")
        except Exception as e:
            print(f"  Warning: Could not load prompt: {e}")
    
    sample_name = f"{folder_name}/{bald_version}/{'3d_aware' if use_3d_aware else '3d_unaware'}"
    
    print(f"\n{'='*60}")
    print(f"Processing: {sample_name}")
    print(f"{'='*60}")
    print(f"  Hair restored input: {hair_restored_file}")
    print(f"  Source bald file: {source_bald_file}")
    print(f"  Output: {id_restored_output}")
    if reference_hair_image:
        print(f"  Reference hair image: {reference_hair_image}")
    if hair_prompt:
        print(f"  Hair prompt: {hair_prompt[:50]}...")
    
    try:
        _ensure_dir(str(fill_processed_dir))
        
        # Load images
        hair_restored_image = Image.open(hair_restored_file).convert("RGB")
        bald_image = Image.open(source_bald_file).convert("RGB")
        
        # Resize to common resolution for processing
        target_size = (1024, 1024)
        hair_restored_resized = hair_restored_image.resize(target_size, Image.Resampling.LANCZOS)
        bald_resized = bald_image.resize(target_size, Image.Resampling.LANCZOS)
        
        hair_restored_np = np.array(hair_restored_resized)
        bald_np = np.array(bald_resized)
        
        # Step 1: Extract or load hair mask from hair_restored image
        print("\n--- Step 1: Getting hair mask from hair_restored image ---")
        hair_mask_np = None
        if hair_restored_mask_file.exists():
            hair_mask_pil = Image.open(hair_restored_mask_file).convert("L")
            hair_mask_np = np.array(hair_mask_pil).astype(np.float32) / 255.0
            print(f"Loaded existing hair mask from: {hair_restored_mask_file}")
        elif SAMMaskExtractor is not None:
            try:
                hair_mask_pil, mask_score = extract_hair_mask(hair_restored_resized, confidence_threshold=0.4)
                hair_mask_pil.save(str(hair_restored_mask_file))
                hair_mask_np = np.array(hair_mask_pil).astype(np.float32) / 255.0
                print(f"Hair mask computed and saved to: {hair_restored_mask_file} (score: {mask_score:.3f})")
            except Exception as e:
                print(f"Warning: Failed to extract hair mask: {e}")
        else:
            print("Warning: SAMMaskExtractor not available")
        
        # Step 2: Composite hair onto bald source
        print("\n--- Step 2: Compositing hair onto bald source image ---")
        if hair_mask_np is not None:
            composited_np = composite_hair_onto_bald(
                hair_restored_np,
                bald_np,
                hair_mask_np,
                use_multiscale=True,
                feather_px=9,
            )
            composited_pil = Image.fromarray(composited_np)
            composited_pil.save(str(composited_input_path))
            print(f"Composited input saved to: {composited_input_path}")
            input_for_restoration = str(composited_input_path)
        else:
            print("Using hair_restored directly (mask extraction failed)")
            input_for_restoration = str(hair_restored_file)
        
        # Step 3: Run identity restoration
        print("\n--- Step 3: Running identity restoration ---")
        run_id_restoration(
            editor=editor,
            image_a_prime_path=input_for_restoration,
            image_a_path=str(source_bald_file),
            image_b_path=str(source_bald_file),
            output_path=str(id_restored_output),
            rf_gamma=rf_gamma,
            hair_prompt=hair_prompt,
            reference_hair_image=str(reference_hair_image) if reference_hair_image and reference_hair_image.exists() else None,
            reference_hair_mask=str(reference_hair_mask) if reference_hair_mask and reference_hair_mask.exists() else None,
            stats_transfer_strength=stats_transfer_strength,
            use_dual_conditioning=use_dual_conditioning,
            verbose=True,
        )
        
        # Step 4: Compute hair mask on id_restored image
        print("\n--- Step 4: Computing hair mask on id_restored image ---")
        if id_restored_output.exists() and SAMMaskExtractor is not None:
            try:
                id_restored_image = Image.open(id_restored_output).convert("RGB")
                hair_mask_pil, mask_score = extract_hair_mask(id_restored_image, confidence_threshold=0.5)
                hair_mask_pil.save(str(id_restored_mask_output))
                print(f"Hair mask saved to: {id_restored_mask_output} (score: {mask_score:.3f})")
            except Exception as e:
                print(f"Warning: Failed to extract hair mask from output: {e}")
        
        print(f"\n✓ Completed {sample_name}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed processing {sample_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_batch(
    editor: FluxKontextRFInversionEditor,
    data_dir: str,
    shape_provider: str,
    texture_provider: str,
    config: Optional[HairTransferConfig] = None,
    bald_version: str = "w_seg",
    conditioning_mode: str = "3d_aware",
    rf_gamma: float = 0.5,
    hair_prompt: Optional[str] = None,
    stats_transfer_strength: float = 0.35,
    use_dual_conditioning: bool = True,
    skip_existing: bool = True,
) -> dict:
    """
    Process all samples in batch mode with reference hairstyle support.
    
    This function mirrors the structure from restore_hair.py's process_view_aligned_folders.
    
    Args:
        editor: FluxKontextRFInversionEditor instance
        data_dir: Root data directory
        shape_provider: Shape provider name (e.g., 'hi3dgen')
        texture_provider: Texture provider name (e.g., 'mvadapter')
        config: HairTransferConfig instance (uses default if None)
        bald_version: 'w_seg', 'wo_seg', or 'all'
        conditioning_mode: '3d_aware', '3d_unaware', or 'all'
        rf_gamma: RF-Inversion gamma parameter
        hair_prompt: Optional hair description to use for all samples (overrides JSON)
        stats_transfer_strength: Strength of statistics transfer from reference (0-1)
        use_dual_conditioning: Whether to use reference as dual conditioning image
        skip_existing: Whether to skip already processed samples
    
    Returns:
        Dictionary with processing statistics {'processed': int, 'skipped': int, 'errors': int}
    """
    from pathlib import Path
    
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
    print(f"  data_dir: {data_dir}")
    print(f"  shape_provider: {shape_provider}")
    print(f"  texture_provider: {texture_provider}")
    print(f"  bald_version(s): {bald_versions}")
    print(f"  conditioning_mode(s): {conditioning_modes}")
    print(f"  rf_gamma: {rf_gamma}")
    print(f"  stats_transfer_strength: {stats_transfer_strength}")
    print(f"  use_dual_conditioning: {use_dual_conditioning}")
    print(f"  skip_existing: {skip_existing}")
    if hair_prompt:
        print(f"  hair_prompt (override): {hair_prompt[:50]}...")
    
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
    
    overall_success = 0
    overall_total = 0
    
    # Process samples for each bald_version and conditioning_mode combination
    for bv in bald_versions:
        for cm in conditioning_modes:
            print(f"\n{'='*60}")
            print(f"Processing: bald_version={bv}, conditioning_mode={cm}")
            print(f"{'='*60}")
            
            success_count = 0
            for folder in all_folders:
                overall_total += 1
                result = process_sample_for_setting(
                    editor=editor,
                    folder=str(folder),
                    data_dir=str(data_dir),
                    config=config,
                    bald_version=bv,
                    conditioning_mode=cm,
                    rf_gamma=rf_gamma,
                    hair_prompt_override=hair_prompt,
                    stats_transfer_strength=stats_transfer_strength,
                    use_dual_conditioning=use_dual_conditioning,
                    skip_existing=skip_existing,
                )
                if result:
                    success_count += 1
                    overall_success += 1
            
            print(f"\nCompleted {bv}/{cm}: {success_count}/{len(all_folders)} samples")
    
    print(f"\n{'='*60}")
    print(f"✓ All processing complete! {overall_success}/{overall_total} total samples processed")
    print(f"{'='*60}")
    
    return {"processed": overall_success, "skipped": 0, "errors": overall_total - overall_success}


# --------------------------------------------------------------------------------------
# Main / CLI
# --------------------------------------------------------------------------------------
def _load_pipe(model_id: str, device: torch.device) -> FluxKontextPipeline:
    pipe = FluxKontextPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        # custom_pipeline="pipeline_flux_kontext_multiple_images",
    )
    return pipe.to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Identity Restoration - Process hair_restored.png from restore_hair.py"
    )
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-Kontext-dev")

    parser.add_argument("--mode", type=str, choices=["batch", "single"], default="batch")
    parser.add_argument("--data_dir", type=str, default="/workspace/outputs/",
                        help="Root data directory for batch processing")
    parser.add_argument("--shape_provider", type=str, default="hi3dgen",
                        choices=["hunyuan", "hi3dgen"],
                        help="Shape provider name (default: hi3dgen)")
    parser.add_argument("--texture_provider", type=str, default="mvadapter",
                        choices=["hunyuan", "mvadapter"],
                        help="Texture provider name (default: mvadapter)")

    # Bald image settings (matching restore_hair.py)
    parser.add_argument(
        "--bald_version",
        type=str,
        default="w_seg",
        choices=["w_seg", "wo_seg", "all"],
        help="Bald version to use: w_seg, wo_seg, or all (default: w_seg)"
    )
    
    # Conditioning mode settings (matching restore_hair.py)
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

    # RF-Inversion param (gamma)
    parser.add_argument("--rf_gamma", type=float, default=0.5, 
                        help="RF-Inversion forward controller strength (gamma). Lower = more source fidelity and texture preservation.")
    
    # Hair prompt for guiding generation
    parser.add_argument("--hair_prompt", type=str, default=None,
                        help="Text description of hair to guide generation "
                             "(e.g., 'long wavy blonde hair', 'short black curly hair'). "
                             "Overrides prompts from JSON files.")
    
    # Reference hairstyle parameters (Statistics Transfer + Dual Conditioning)
    parser.add_argument("--stats_transfer_strength", type=float, default=0.2,
                        help="Strength of AdaIN statistics transfer from reference hair (0-1). "
                             "Higher = more color/texture from reference.")
    parser.add_argument("--use_dual_conditioning", type=lambda x: x.lower() == 'true', default=True,
                        help="Whether to use reference hair as dual conditioning image (true/false).")
    parser.add_argument("--stats_injection_start_t", type=float, default=400.0,
                        help="Start statistics injection below this timestep.")
    parser.add_argument("--stats_injection_end_t", type=float, default=50.0,
                        help="Full statistics injection strength below this timestep.")

    # Single-mode inputs
    parser.add_argument("--hair_restored_file", type=str, default=None,
                        help="Path to hair_restored.png (output from restore_hair.py)")
    parser.add_argument("--bald_file", type=str, default=None,
                        help="Path to source bald image")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--sample_name", type=str, default="single_sample",
                        help="Sample name for logging")
    parser.add_argument("--reference_hair_image", type=str, default=None,
                        help="Path to reference hairstyle image (for single mode)")
    parser.add_argument("--reference_hair_mask", type=str, default=None,
                        help="Path to reference hair mask (computed if not provided)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading FLUX Kontext pipeline (official diffusers)...")
    pipe = _load_pipe(args.model_id, device)
    editor = FluxKontextRFInversionEditor(pipe, device)

    if args.mode == "batch":
        print("Running in BATCH mode...")
        print(f"  bald_version: {args.bald_version}")
        print(f"  conditioning_mode: {args.conditioning_mode}")
        print(f"  RF gamma: {args.rf_gamma}")
        print(f"  Stats transfer strength: {args.stats_transfer_strength}")
        print(f"  Dual conditioning: {args.use_dual_conditioning}")
        print(f"  Skip existing: {args.skip_existing}")
        if args.hair_prompt:
            print(f"  Hair prompt (override): {args.hair_prompt}")
        
        process_batch(
            editor=editor,
            data_dir=args.data_dir,
            shape_provider=args.shape_provider,
            texture_provider=args.texture_provider,
            config=HairTransferConfig(),
            bald_version=args.bald_version,
            conditioning_mode=args.conditioning_mode,
            rf_gamma=args.rf_gamma,
            hair_prompt=args.hair_prompt,
            stats_transfer_strength=args.stats_transfer_strength,
            use_dual_conditioning=args.use_dual_conditioning,
            skip_existing=args.skip_existing,
        )
        return

    print("Running in SINGLE mode...")
    print(f"RF gamma: {args.rf_gamma}")
    if args.hair_prompt:
        print(f"Hair prompt: {args.hair_prompt}")
    if args.reference_hair_image:
        print(f"Reference hair image: {args.reference_hair_image}")
    if not all([args.hair_restored_file, args.bald_file, args.output_dir]):
        print("Error: single mode requires --hair_restored_file --bald_file --output_dir")
        return

    _ensure_dir(args.output_dir)
    process_single_sample(
        editor=editor,
        hair_restored_file=args.hair_restored_file,
        source_bald_file=args.bald_file,
        output_dir=args.output_dir,
        sample_name=args.sample_name,
        rf_gamma=args.rf_gamma,
        hair_prompt=args.hair_prompt,
        reference_hair_image=args.reference_hair_image,
        reference_hair_mask=args.reference_hair_mask,
        stats_transfer_strength=args.stats_transfer_strength,
        use_dual_conditioning=args.use_dual_conditioning,
    )


if __name__ == "__main__":
    main()
