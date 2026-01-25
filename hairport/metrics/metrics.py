"""
Metric implementations for hair transfer evaluation.

This module contains all metric classes for evaluating hair transfer quality:
- CLIP-I: Cosine similarity between CLIP embeddings of reference and generated images
- FID: Fréchet Inception Distance (using Inception V3)
- FID_CLIP: Fréchet Inception Distance using CLIP encoder instead of Inception V3
- SSIM: Structural Similarity Index (masked and full)
- PSNR: Peak Signal-to-Noise Ratio (masked and full)
- LPIPS: Learned Perceptual Image Patch Similarity (between source and generated, non-hair region)
- DreamSim: DreamSim perceptual distance
- IDS: Identity Similarity using InsightFace
- DINOv2 Hair Similarity: Hair region similarity using DINOv2
- CLIP Hair Similarity: Hair region similarity using CLIP
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import math

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# SSIM
from skimage.metrics import structural_similarity

# FID
from torchmetrics.image.fid import FrechetInceptionDistance
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# CLIP
from transformers import CLIPModel, CLIPProcessor

# LPIPS
import lpips

# DreamSim
import inspect
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from dreamsim import dreamsim as _load_dreamsim


# InsightFace for Identity Similarity
from insightface.app import FaceAnalysis

# DINOv2 for Hair Similarity
from transformers import AutoImageProcessor, AutoModel

# Sentence Transformers for text embedding similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# -----------------------------
# Data containers
# -----------------------------

@dataclass
class Sample:
    """One evaluation sample."""
    source: Image.Image                 # source image (RGB)
    generated: Image.Image              # edited/generated image (RGB)
    reference: Optional[Image.Image]    # reference image (RGB) for CLIP-I (may be None if not used)
    hair_mask_source: Image.Image       # hair mask of source (L; binary/soft)
    hair_mask_generated: Image.Image    # hair mask of generated (L; binary/soft)
    hair_mask_reference: Optional[Image.Image] = None  # hair mask of reference (L; binary/soft) for hair-only metrics


# -----------------------------
# Common utilities
# -----------------------------

def _to_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img


def _to_l(img: Image.Image) -> Image.Image:
    return img.convert("L") if img.mode != "L" else img


def _resize_like(img: Image.Image, ref: Image.Image, resample=Image.BILINEAR) -> Image.Image:
    return img if img.size == ref.size else img.resize(ref.size, resample)


def _pil_to_np_uint8(img: Image.Image) -> np.ndarray:
    """(H,W,3) uint8"""
    img = _to_rgb(img)
    arr = np.array(img, dtype=np.uint8)
    return arr


def _pil_mask_to_np_float(mask: Image.Image, ref_img: Image.Image) -> np.ndarray:
    """
    Convert mask to float in [0,1], shape (H,W).
    Resizes mask to match ref_img.
    """
    mask = _to_l(mask)
    mask = _resize_like(mask, ref_img, resample=Image.BILINEAR)
    m = np.asarray(mask, dtype=np.float32) / 255.0
    return np.clip(m, 0.0, 1.0)


def _intersected_nonhair_weights(
    source_img: Image.Image,
    gen_img: Image.Image,
    hair_src: Image.Image,
    hair_gen: Image.Image,
    eps_area: float = 1e-6,
) -> np.ndarray:
    """
    W = (1 - hair_src) * (1 - hair_gen), resized to match source/gen.
    Returns float weights (H,W) in [0,1].
    """
    gen_img_r = _resize_like(_to_rgb(gen_img), _to_rgb(source_img), resample=Image.BICUBIC)

    hs = _pil_mask_to_np_float(hair_src, source_img)
    hg = _pil_mask_to_np_float(hair_gen, gen_img_r)

    if hg.shape != hs.shape:
        hg_img = Image.fromarray((hg * 255).astype(np.uint8), mode="L")
        hg = _pil_mask_to_np_float(hg_img, source_img)

    w = (1.0 - hs) * (1.0 - hg)
    area = float(np.sum(w))
    if area < eps_area:
        raise ValueError(
            f"Intersected non-hair region is empty/too small (sum={area:.6f}). "
            "Check masks alignment or thresholding."
        )
    return w


def _weighted_mean(x: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> float:
    wsum = float(np.sum(w))
    if wsum < eps:
        return float("nan")
    return float(np.sum(x * w) / (wsum + eps))


def _pil_to_torch_float01(img: Image.Image, device: str) -> torch.Tensor:
    """(1,3,H,W) float32 in [0,1]"""
    arr = _pil_to_np_uint8(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device)
    return t


def _pil_to_torch_uint8(img: Image.Image, device: str) -> torch.Tensor:
    """(1,3,H,W) uint8 in [0,255]"""
    arr = _pil_to_np_uint8(img)
    t = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=torch.uint8)


def _extract_hair_region(
    img: Image.Image,
    hair_mask: Image.Image,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    crop_to_bbox: bool = True,
    min_size: int = 64,
) -> Image.Image:
    """
    Extract hair region from image using the provided mask.
    
    Args:
        img: RGB image
        hair_mask: Binary/soft mask (L mode) where hair pixels are bright
        background_color: Color for non-hair pixels (default: black)
        crop_to_bbox: If True, crop to bounding box of hair region
        min_size: Minimum size for the extracted region
    
    Returns:
        RGB image with only hair region visible (rest is background_color)
    """
    img = _to_rgb(img)
    mask = _to_l(hair_mask)
    mask = _resize_like(mask, img, resample=Image.BILINEAR)
    
    img_np = np.array(img, dtype=np.uint8)
    mask_np = np.array(mask, dtype=np.float32) / 255.0
    
    bg = np.array(background_color, dtype=np.uint8).reshape(1, 1, 3)
    bg = np.broadcast_to(bg, img_np.shape).copy()
    
    mask_3d = mask_np[..., None]
    out_np = (mask_3d * img_np + (1 - mask_3d) * bg).astype(np.uint8)
    
    if crop_to_bbox:
        binary_mask = mask_np > 0.5
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return Image.fromarray(out_np).resize((min_size, min_size), Image.BILINEAR)
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        pad = 5
        y_min = max(0, y_min - pad)
        y_max = min(out_np.shape[0] - 1, y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(out_np.shape[1] - 1, x_max + pad)
        
        out_np = out_np[y_min:y_max+1, x_min:x_max+1]
        
        h, w = out_np.shape[:2]
        if h < min_size or w < min_size:
            scale = max(min_size / h, min_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            out_img = Image.fromarray(out_np)
            out_img = out_img.resize((new_w, new_h), Image.BILINEAR)
            return out_img
    
    return Image.fromarray(out_np)


def _extract_hair_region_with_context(
    img: Image.Image,
    hair_mask: Image.Image,
    context_ratio: float = 0.3,
) -> Image.Image:
    """
    Extract hair region with some surrounding context for better feature extraction.
    """
    img = _to_rgb(img)
    mask = _to_l(hair_mask)
    mask = _resize_like(mask, img, resample=Image.BILINEAR)
    
    img_np = np.array(img, dtype=np.float32)
    mask_np = np.array(mask, dtype=np.float32) / 255.0
    
    mask_3d = mask_np[..., None]
    weight = mask_3d + (1 - mask_3d) * context_ratio
    out_np = (img_np * weight).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(out_np)


# -----------------------------
# Metric base class
# -----------------------------

class Metric:
    """Abstract metric interface."""
    name: str

    def compute(self, samples: List[Sample]) -> float:
        raise NotImplementedError


# -----------------------------
# CLIP-I (image-image cosine similarity)
# -----------------------------

class CLIPIMetric(Metric):
    """
    CLIP-I: cosine similarity between CLIP image embeddings of
    reference image (hair donor) and generated/transferred image.
    
    This measures how well the generated image preserves visual similarity
    to the reference image from which the hairstyle was transferred.
    """
    name = "clip_i"

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "openai/clip-vit-base-patch32",
        batch_size: int = 16,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model_id = model_id
        self.processor = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.processor = CLIPProcessor.from_pretrained(self.model_id, use_fast=True)
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self.model.eval()

    def _unload_model(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    @torch.inference_mode()
    def compute(self, samples: List[Sample]) -> float:
        self._load_model()
        
        refs: List[Image.Image] = []
        gens: List[Image.Image] = []
        for s in samples:
            if s.reference is None:
                raise ValueError("CLIP-I requires Sample.reference to be provided.")
            refs.append(_to_rgb(s.reference))
            gens.append(_to_rgb(s.generated))

        sims: List[float] = []
        for i in range(0, len(samples), self.batch_size):
            rb = refs[i:i + self.batch_size]
            gb = gens[i:i + self.batch_size]
            r_inputs = self.processor(images=rb, return_tensors="pt").to(self.device)
            g_inputs = self.processor(images=gb, return_tensors="pt").to(self.device)

            r_feat = self.model.get_image_features(**r_inputs)
            g_feat = self.model.get_image_features(**g_inputs)

            r_feat = F.normalize(r_feat, dim=-1)
            g_feat = F.normalize(g_feat, dim=-1)

            cos = torch.sum(r_feat * g_feat, dim=-1)
            sims.extend(cos.detach().cpu().tolist())

        result = float(np.mean(sims))
        self._unload_model()
        return result

    @torch.inference_mode()
    def compute_per_sample(self, samples: List[Sample]) -> List[float]:
        """Compute CLIP-I for each sample individually."""
        self._load_model()
        
        sims: List[float] = []
        for s in samples:
            if s.reference is None:
                sims.append(float('nan'))
                continue
            
            ref = _to_rgb(s.reference)
            gen = _to_rgb(s.generated)
            
            r_inputs = self.processor(images=[ref], return_tensors="pt").to(self.device)
            g_inputs = self.processor(images=[gen], return_tensors="pt").to(self.device)

            r_feat = self.model.get_image_features(**r_inputs)
            g_feat = self.model.get_image_features(**g_inputs)

            r_feat = F.normalize(r_feat, dim=-1)
            g_feat = F.normalize(g_feat, dim=-1)

            cos = torch.sum(r_feat * g_feat, dim=-1)
            sims.append(float(cos.detach().cpu().item()))
        
        self._unload_model()
        return sims


# -----------------------------
# FID (on non-hair intersected region)
# -----------------------------

def _apply_nonhair_mask_to_image(
    img: Image.Image,
    hair_mask_source: Image.Image,
    hair_mask_generated: Image.Image,
    background_color: Tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """
    Apply non-hair intersection mask to image.
    Pixels in hair regions (either source or generated) are replaced with background_color.
    
    W = (1 - hair_src) * (1 - hair_gen)
    """
    img = _to_rgb(img)
    hs = _pil_mask_to_np_float(hair_mask_source, img)
    hg = _pil_mask_to_np_float(hair_mask_generated, img)
    
    # Non-hair weight: 1 where neither source nor generated has hair
    w = (1.0 - hs) * (1.0 - hg)
    
    img_np = np.array(img, dtype=np.float32)
    bg = np.array(background_color, dtype=np.float32).reshape(1, 1, 3)
    
    # Apply mask: keep non-hair regions, replace hair regions with background
    w3 = w[..., None]
    out_np = img_np * w3 + bg * (1.0 - w3)
    out_np = np.clip(out_np, 0, 255).astype(np.uint8)
    
    return Image.fromarray(out_np)


class FIDMetric(Metric):
    """
    FID (Fréchet Inception Distance) between source images and generated/transferred images.
    
    FID measures the distance between the feature distributions of two image sets
    using Inception V3 features. Lower FID indicates more similar distributions,
    suggesting higher quality and more realistic generated images.
    
    This metric computes FID on FULL images (no masking) to measure overall
    image quality and distribution similarity between source and generated images.
    
    By default: "real" = source images, "fake" = generated/transferred images.
    """
    name = "fid"

    def __init__(
        self,
        device: str = "cuda",
        feature_dim: int = 2048,
        batch_size: int = 16,
        normalize: bool = False,
    ):
        self.device = device
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        self.normalize = normalize
        self.fid = None

    def _load_model(self):
        if self.fid is None:
            self.fid = FrechetInceptionDistance(feature=self.feature_dim, normalize=self.normalize).to(self.device)

    def _unload_model(self):
        if self.fid is not None:
            del self.fid
            self.fid = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    @torch.inference_mode()
    def compute(self, samples: List[Sample]) -> float:
        """
        Compute FID between source images (real) and generated images (fake).
        
        Uses full images without any masking to measure overall distribution quality.
        """
        self._load_model()
        self.fid.reset()

        # Process source images (real) - full images, no masking
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i:i + self.batch_size]
            source_images = [_to_rgb(s.source) for s in batch]
            x = torch.cat([_pil_to_torch_uint8(img, self.device) for img in source_images], dim=0)
            self.fid.update(x, real=True)

        # Process generated images (fake) - full images, no masking
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i:i + self.batch_size]
            generated_images = [
                _resize_like(_to_rgb(s.generated), _to_rgb(s.source), resample=Image.BICUBIC)
                for s in batch
            ]
            x = torch.cat([_pil_to_torch_uint8(img, self.device) for img in generated_images], dim=0)
            self.fid.update(x, real=False)

        result = float(self.fid.compute().detach().cpu().item())
        self._unload_model()
        return result


# -----------------------------
# FID_CLIP (FID using CLIP encoder)
# -----------------------------

class FIDCLIPMetric(Metric):
    """
    FID_CLIP: Fréchet Inception Distance computed using CLIP image encoder
    instead of Inception V3.
    
    Compares distribution of source images ("real") to generated images ("fake")
    using CLIP image embeddings. This provides a distribution-level quality measure
    that captures semantic similarity as understood by CLIP.
    
    This metric computes FID on FULL images (no masking) to measure overall
    semantic distribution similarity between source and generated images.
    """
    name = "fid_clip"

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "openai/clip-vit-base-patch32",
        batch_size: int = 16,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model_id = model_id
        self.processor = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.processor = CLIPProcessor.from_pretrained(self.model_id, use_fast=True)
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self.model.eval()

    def _unload_model(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    @torch.inference_mode()
    def _get_features(self, images: List[Image.Image]) -> np.ndarray:
        """Extract CLIP features for a list of images."""
        all_features = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            features = self.model.get_image_features(**inputs)
            # Do NOT normalize features for FID computation - normalization removes variance
            # which is essential for measuring distribution distance
            all_features.append(features.cpu().numpy())
        return np.concatenate(all_features, axis=0)

    def _compute_fid_from_features(
        self, 
        real_features: np.ndarray, 
        fake_features: np.ndarray,
        eps: float = 1e-6
    ) -> float:
        """
        Compute FID between two sets of features.
        
        FID = ||mu_r - mu_f||^2 + Tr(C_r + C_f - 2*sqrt(C_r*C_f))
        """
        # Compute mean and covariance for real features
        mu_r = np.mean(real_features, axis=0)
        sigma_r = np.cov(real_features, rowvar=False)
        
        # Compute mean and covariance for fake features
        mu_f = np.mean(fake_features, axis=0)
        sigma_f = np.cov(fake_features, rowvar=False)
        
        # Compute squared difference of means
        diff = mu_r - mu_f
        diff_sq = np.dot(diff, diff)
        
        # Compute sqrt of product of covariances using scipy
        from scipy import linalg
        
        # Add small epsilon to diagonal for numerical stability
        sigma_r = sigma_r + eps * np.eye(sigma_r.shape[0])
        sigma_f = sigma_f + eps * np.eye(sigma_f.shape[0])
        
        # Compute sqrt(sigma_r @ sigma_f)
        covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
        
        # Handle potential numerical issues
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # Compute FID
        trace = np.trace(sigma_r) + np.trace(sigma_f) - 2 * np.trace(covmean)
        fid = diff_sq + trace
        
        return float(fid)

    @torch.inference_mode()
    def compute(self, samples: List[Sample]) -> float:
        """
        Compute FID_CLIP between source images (real) and generated images (fake).
        
        Uses full images without any masking to measure overall semantic distribution quality.
        """
        self._load_model()
        
        # Source images (real) - full images, no masking
        source_images = [_to_rgb(s.source) for s in samples]
        
        # Generated images (fake) - full images, no masking
        generated_images = [
            _resize_like(_to_rgb(s.generated), _to_rgb(s.source), resample=Image.BICUBIC)
            for s in samples
        ]
        
        # Extract features from full images
        real_features = self._get_features(source_images)
        fake_features = self._get_features(generated_images)
        
        # Compute FID
        result = self._compute_fid_from_features(real_features, fake_features)
        
        self._unload_model()
        return result


# -----------------------------
# SSIM on intersected non-hair region
# -----------------------------

class SSIMMetric(Metric):
    """
    SSIM computed on intersected non-hair region:
      W = (1 - hair_src) * (1 - hair_gen)
    """
    name = "ssim_nonhair_intersection"

    def __init__(self, data_range: float = 255.0):
        self.data_range = data_range

    def compute(self, samples: List[Sample]) -> float:
        vals: List[float] = []
        for s in samples:
            src = _to_rgb(s.source)
            gen = _resize_like(_to_rgb(s.generated), src, resample=Image.BICUBIC)

            w = _intersected_nonhair_weights(src, gen, s.hair_mask_source, s.hair_mask_generated)

            a = _pil_to_np_uint8(src).astype(np.float32)
            b = _pil_to_np_uint8(gen).astype(np.float32)

            ssim_val, ssim_map = structural_similarity(
                a, b,
                channel_axis=2,
                data_range=self.data_range,
                full=True
            )
            if ssim_map.ndim == 3:
                ssim_map = np.mean(ssim_map, axis=2)
            
            vals.append(_weighted_mean(ssim_map.astype(np.float32), w.astype(np.float32)))

        return float(np.mean(vals))

# -----------------------------
# PSNR on intersected non-hair region
# -----------------------------

class PSNRMetric(Metric):
    """
    PSNR computed on intersected non-hair region:
      W = (1 - hair_src) * (1 - hair_gen)
    """
    name = "psnr_nonhair_intersection"

    def __init__(self, max_val: float = 255.0, eps: float = 1e-12):
        self.max_val = max_val
        self.eps = eps

    def compute(self, samples: List[Sample]) -> float:
        vals: List[float] = []
        for s in samples:
            src = _to_rgb(s.source)
            gen = _resize_like(_to_rgb(s.generated), src, resample=Image.BICUBIC)
            w = _intersected_nonhair_weights(src, gen, s.hair_mask_source, s.hair_mask_generated)

            a = _pil_to_np_uint8(src).astype(np.float32)
            b = _pil_to_np_uint8(gen).astype(np.float32)

            diff2 = np.mean((a - b) ** 2, axis=2)
            mse = _weighted_mean(diff2.astype(np.float32), w.astype(np.float32), eps=self.eps)
            mse = max(mse, self.eps)

            psnr = 10.0 * math.log10((self.max_val ** 2) / mse)
            vals.append(float(psnr))

        return float(np.mean(vals))


# -----------------------------
# LPIPS
# -----------------------------

class LPIPSMetric(Metric):
    """
    LPIPS distance between source and generated images.
    
    When region='nonhair_intersection', computes LPIPS only on the
    intersected non-hair region: W = (1 - hair_src) * (1 - hair_gen).
    This measures how well the non-hair regions (face, background) are preserved.
    """
    name = "lpips"

    def __init__(
        self,
        device: str = "cuda",
        net: str = "alex",
        region: str = "full",
    ):
        self.device = device
        self.region = region
        self.net = net
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.model = lpips.LPIPS(net=self.net).to(self.device)
            self.model.eval()

    def _unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    @staticmethod
    def _composite_to_localize_diff(
        a: Image.Image,
        b: Image.Image,
        w: np.ndarray,
    ) -> Tuple[Image.Image, Image.Image]:
        """Make outside-region identical to reduce impact outside region."""
        a_np = _pil_to_np_uint8(a).astype(np.float32)
        b_np = _pil_to_np_uint8(b).astype(np.float32)
        w3 = w[..., None].astype(np.float32)

        out_a = a_np
        out_b = a_np * (1.0 - w3) + b_np * w3

        out_a = np.clip(out_a, 0, 255).astype(np.uint8)
        out_b = np.clip(out_b, 0, 255).astype(np.uint8)
        return Image.fromarray(out_a), Image.fromarray(out_b)

    @torch.inference_mode()
    def compute(self, samples: List[Sample]) -> float:
        self._load_model()
        
        vals: List[float] = []
        for s in samples:
            src = _to_rgb(s.source)
            gen = _resize_like(_to_rgb(s.generated), src, resample=Image.BICUBIC)

            if self.region == "nonhair_intersection":
                w = _intersected_nonhair_weights(src, gen, s.hair_mask_source, s.hair_mask_generated)
                src2, gen2 = self._composite_to_localize_diff(src, gen, w)
            else:
                src2, gen2 = src, gen

            x = _pil_to_torch_float01(src2, self.device) * 2.0 - 1.0
            y = _pil_to_torch_float01(gen2, self.device) * 2.0 - 1.0
            d = self.model(x, y)
            vals.append(float(d.detach().cpu().item()))

        result = float(np.mean(vals))
        self._unload_model()
        return result


# -----------------------------
# DreamSim (on hair region)
# -----------------------------
class DreamSimMetric(Metric):
    name = "dreamsim"

    def __init__(
        self,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
        dreamsim_type: str = "dinov2_vitb14",
        use_patch_model: bool = True,
        extraction_mode: str = "masked",
        mask_threshold: float = 0.5,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        min_size: int = 64,
    ):
        self.device = device
        self.cache_dir = cache_dir
        self.dreamsim_type = dreamsim_type
        self.use_patch_model = use_patch_model

        self.extraction_mode = extraction_mode
        self.mask_threshold = float(mask_threshold)
        self.background_color = background_color
        self.min_size = int(min_size)

        self.model = None
        self.preprocess = None

    def _load_model(self):
        if self.model is not None:
            return

        kwargs = {"pretrained": True, "device": self.device}
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir

        sig = inspect.signature(_load_dreamsim)
        if "dreamsim_type" in sig.parameters:
            kwargs["dreamsim_type"] = self.dreamsim_type
        if "use_patch_model" in sig.parameters:
            kwargs["use_patch_model"] = self.use_patch_model

        self.model, self.preprocess = _load_dreamsim(**kwargs)
        self.model.eval()

    def _unload_model(self):
        if self.model is not None:
            del self.model
            del self.preprocess
            self.model = None
            self.preprocess = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def _to_rgb(self, img: Image.Image) -> Image.Image:
        return img.convert("RGB") if img.mode != "RGB" else img

    def _to_l(self, img: Image.Image) -> Image.Image:
        return img.convert("L") if img.mode != "L" else img

    def _resize_like(self, src: Image.Image, ref: Image.Image, resample=Image.BILINEAR) -> Image.Image:
        return src.resize(ref.size, resample=resample)

    def _extract_hair_region_with_aligned_mask(
        self,
        img: Image.Image,
        hair_mask: Image.Image,
        crop_to_bbox: bool,
    ) -> Tuple[Image.Image, Image.Image]:
        img = self._to_rgb(img)
        mask = self._to_l(hair_mask)
        mask = self._resize_like(mask, img, resample=Image.BILINEAR)

        img_np = np.array(img, dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.float32) / 255.0

        bg = np.array(self.background_color, dtype=np.uint8).reshape(1, 1, 3)
        bg = np.broadcast_to(bg, img_np.shape).copy()

        out_np = (mask_np[..., None] * img_np + (1.0 - mask_np[..., None]) * bg).astype(np.uint8)
        out_mask_np = (mask_np * 255.0).astype(np.uint8)

        out_img = Image.fromarray(out_np)
        out_mask = Image.fromarray(out_mask_np, mode="L")

        if not crop_to_bbox:
            return out_img, out_mask

        binary = mask_np > self.mask_threshold
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)

        if not np.any(rows) or not np.any(cols):
            out_img = out_img.resize((self.min_size, self.min_size), Image.BILINEAR)
            out_mask = out_mask.resize((self.min_size, self.min_size), Image.BILINEAR)
            return out_img, out_mask

        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]

        pad = 5
        y0 = max(0, y0 - pad)
        y1 = min(out_np.shape[0] - 1, y1 + pad)
        x0 = max(0, x0 - pad)
        x1 = min(out_np.shape[1] - 1, x1 + pad)

        out_img = out_img.crop((x0, y0, x1 + 1, y1 + 1))
        out_mask = out_mask.crop((x0, y0, x1 + 1, y1 + 1))

        if out_img.size[0] < self.min_size or out_img.size[1] < self.min_size:
            w, h = out_img.size
            scale = max(self.min_size / max(h, 1), self.min_size / max(w, 1))
            new_w = int(max(self.min_size, round(w * scale)))
            new_h = int(max(self.min_size, round(h * scale)))
            out_img = out_img.resize((new_w, new_h), Image.BILINEAR)
            out_mask = out_mask.resize((new_w, new_h), Image.BILINEAR)

        return out_img, out_mask

    def _extract_hair_image(self, img: Image.Image, mask: Image.Image) -> Image.Image:
        if self.extraction_mode == "cropped":
            out_img, _ = self._extract_hair_region_with_aligned_mask(img, mask, crop_to_bbox=True)
            return out_img

        if self.extraction_mode == "context":
            img = self._to_rgb(img)
            m = self._to_l(mask)
            m = self._resize_like(m, img, resample=Image.BILINEAR)
            m_np = (np.array(m, dtype=np.float32) / 255.0) > self.mask_threshold
            rows = np.any(m_np, axis=1)
            cols = np.any(m_np, axis=0)
            if np.any(rows) and np.any(cols):
                y0, y1 = np.where(rows)[0][[0, -1]]
                x0, x1 = np.where(cols)[0][[0, -1]]
                h = (y1 - y0 + 1)
                w = (x1 - x0 + 1)
                pad_y = int(0.2 * h)
                pad_x = int(0.2 * w)
                y0 = max(0, y0 - pad_y)
                y1 = min(m_np.shape[0] - 1, y1 + pad_y)
                x0 = max(0, x0 - pad_x)
                x1 = min(m_np.shape[1] - 1, x1 + pad_x)
                img = img.crop((x0, y0, x1 + 1, y1 + 1))
                mask = m.crop((x0, y0, x1 + 1, y1 + 1))
            out_img, _ = self._extract_hair_region_with_aligned_mask(img, mask, crop_to_bbox=False)
            return out_img

        out_img, _ = self._extract_hair_region_with_aligned_mask(img, mask, crop_to_bbox=False)
        return out_img

    def _prep(self, img: Image.Image) -> torch.Tensor:
        t = self.preprocess(self._to_rgb(img)).to(self.device)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        return t

    @torch.inference_mode()
    def compute(self, samples: List[Sample]) -> float:
        self._load_model()
        dists: List[float] = []

        for s in samples:
            if s.reference is None or s.generated is None:
                raise ValueError("Requires Sample.reference and Sample.generated.")
            if s.hair_mask_reference is None or s.hair_mask_generated is None:
                raise ValueError("Requires Sample.hair_mask_reference and Sample.hair_mask_generated.")

            ref_hair = self._extract_hair_image(s.reference, s.hair_mask_reference)
            gen_hair = self._extract_hair_image(s.generated, s.hair_mask_generated)

            ref_t = self._prep(ref_hair)
            gen_t = self._prep(gen_hair)

            dist = self.model(ref_t, gen_t)
            if isinstance(dist, (tuple, list)):
                dist = dist[0]
            dist_val = float(dist.detach().float().mean().cpu().item())
            dists.append(dist_val)

        out = float(np.mean(dists)) if dists else 0.0
        self._unload_model()
        return out

    @torch.inference_mode()
    def compute_per_sample(self, samples: List[Sample]) -> List[Optional[float]]:
        self._load_model()
        outs: List[Optional[float]] = []

        for s in samples:
            if s.reference is None or s.generated is None or s.hair_mask_reference is None or s.hair_mask_generated is None:
                outs.append(None)
                continue

            try:
                ref_hair = self._extract_hair_image(s.reference, s.hair_mask_reference)
                gen_hair = self._extract_hair_image(s.generated, s.hair_mask_generated)

                ref_t = self._prep(ref_hair)
                gen_t = self._prep(gen_hair)

                dist = self.model(ref_t, gen_t)
                if isinstance(dist, (tuple, list)):
                    dist = dist[0]
                outs.append(float(dist.detach().float().mean().cpu().item()))
            except Exception:
                outs.append(None)

        self._unload_model()
        return outs


# -----------------------------
# InsightFace Identity Similarity (IDS)
# -----------------------------

class IDSMetric(Metric):
    """
    Identity Similarity (IDS) using InsightFace.
    Computes cosine similarity between face embeddings of source and generated images.
    """
    name = "ids"

    def __init__(
        self,
        device: str = "cuda",
        det_size: Tuple[int, int] = (640, 640),
    ):
        self.device = device
        self.det_size = det_size
        self.app = None

    def _load_model(self):
        if self.app is None:
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=self.det_size)

    def _unload_model(self):
        if self.app is not None:
            del self.app
            self.app = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def _get_embedding(self, img: Image.Image) -> Optional[np.ndarray]:
        """Extract face embedding from image."""
        img_np = np.array(_to_rgb(img))
        img_bgr = img_np[:, :, ::-1].copy()
        faces = self.app.get(img_bgr)
        if len(faces) == 0:
            return None
        return faces[0].embedding

    def compute(self, samples: List[Sample]) -> float:
        self._load_model()
        
        sims: List[float] = []
        for s in samples:
            src_emb = self._get_embedding(s.source)
            gen_emb = self._get_embedding(s.generated)
            
            if src_emb is None or gen_emb is None:
                continue
            
            src_emb = src_emb / np.linalg.norm(src_emb)
            gen_emb = gen_emb / np.linalg.norm(gen_emb)
            sim = float(np.dot(src_emb, gen_emb))
            sims.append(sim)
        
        result = float(np.mean(sims)) if sims else 0.0
        self._unload_model()
        return result

    def compute_per_sample(self, samples: List[Sample]) -> List[Optional[float]]:
        """Compute IDS for each sample individually."""
        self._load_model()
        
        results: List[Optional[float]] = []
        for s in samples:
            src_emb = self._get_embedding(s.source)
            gen_emb = self._get_embedding(s.generated)
            
            if src_emb is None or gen_emb is None:
                results.append(None)
                continue
            
            src_emb = src_emb / np.linalg.norm(src_emb)
            gen_emb = gen_emb / np.linalg.norm(gen_emb)
            sim = float(np.dot(src_emb, gen_emb))
            results.append(sim)
        
        self._unload_model()
        return results


class _HairPatchMaskingMixin:
    def _mask_to_patch_weights(
        self,
        inputs_pixel_values: torch.Tensor,  # (1,3,H,W)
        hair_mask_pil: Image.Image,         # aligned with the image fed into processor
        patch_size: int,
    ) -> torch.Tensor:
        _, _, H, W = inputs_pixel_values.shape
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(f"Processed size {(H, W)} not divisible by patch_size={patch_size}")

        # Use bilinear to preserve fractional coverage; normalize to [0,1]
        m = hair_mask_pil.convert("L").resize((W, H), resample=Image.BILINEAR)
        m_np = np.array(m, dtype=np.float32) / 255.0
        m_t = torch.from_numpy(m_np).to(inputs_pixel_values.device).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        pooled = F.avg_pool2d(m_t, kernel_size=patch_size, stride=patch_size)[0, 0]  # (Gh,Gw)
        weights = pooled.flatten().clamp_(0.0, 1.0)  # (N,)
        return weights

    def _select_patches(
        self,
        patch_tokens: torch.Tensor,  # (1,N,D)
        weights: torch.Tensor,      # (N,)
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if patch_tokens.shape[0] != 1:
            raise ValueError("Expected batch size 1 for patch_tokens.")

        N = patch_tokens.shape[1]
        if weights.shape[0] != N:
            raise ValueError(f"Token/weight mismatch: tokens={N}, weights={weights.shape[0]}")

        keep = weights > threshold
        if keep.any():
            idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
            return patch_tokens[:, idx, :], weights[idx]

        # Fallback 1: keep any patch with non-zero coverage
        keep = weights > 0.0
        if keep.any():
            idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
            return patch_tokens[:, idx, :], weights[idx]

        # Fallback 2: keep the single best patch
        top = torch.argmax(weights).item()
        return patch_tokens[:, top:top + 1, :], weights[top:top + 1]

    def _weighted_pool_then_cosine(
        self,
        patches_a: torch.Tensor,  # (1,N,D)
        weights_a: torch.Tensor,  # (N,)
        patches_b: torch.Tensor,  # (1,M,D)
        weights_b: torch.Tensor,  # (M,)
        eps: float = 1e-8,
    ) -> float:
        patches_a = F.normalize(patches_a, dim=-1)
        patches_b = F.normalize(patches_b, dim=-1)

        wa = weights_a.clamp(min=0.0)
        wb = weights_b.clamp(min=0.0)

        wa_sum = wa.sum().clamp(min=eps)
        wb_sum = wb.sum().clamp(min=eps)

        emb_a = (patches_a[0] * wa[:, None]).sum(dim=0) / wa_sum
        emb_b = (patches_b[0] * wb[:, None]).sum(dim=0) / wb_sum

        emb_a = F.normalize(emb_a, dim=-1)
        emb_b = F.normalize(emb_b, dim=-1)

        return float(torch.clamp(torch.dot(emb_a, emb_b), -1.0, 1.0).item())

    def _pairwise_similarity(
        self,
        patches_a: torch.Tensor,  # (1,N,D)
        patches_b: torch.Tensor,  # (1,M,D)
        mode: str,
    ) -> float:
        patches_a = F.normalize(patches_a, dim=-1)
        patches_b = F.normalize(patches_b, dim=-1)

        sim = torch.bmm(patches_a, patches_b.transpose(1, 2))  # (1,N,M)

        if mode == "mean":
            return float(sim.mean().item())

        if mode == "max":
            # symmetric best-match average
            a_to_b = sim.max(dim=-1).values.mean()  # (1,N) -> scalar
            b_to_a = sim.max(dim=-2).values.mean()  # (1,M) -> scalar
            return float(((a_to_b + b_to_a) * 0.5).item())

        # default
        return float(sim.mean().item())

    def _cosine_cls(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return float(torch.clamp(torch.sum(a * b, dim=-1), -1.0, 1.0).item())


# -----------------------------
# DINOv3 Hair Similarity
# -----------------------------

class DINOv3HairSimilarityMetric(Metric, _HairPatchMaskingMixin):
    """
    Hair similarity metric using DINOv3 (facebook/dinov3-*) vision transformer.
    
    DINOv3 produces high-quality dense features that achieve outstanding performance
    on various vision tasks. This metric computes similarity between hair regions
    in reference and generated images using patch-level or CLS-token features.
    
    Key differences from DINOv2:
    - DINOv3 ViT models use patch_size=16 (configurable)
    - DINOv3 may have register tokens (num_register_tokens in config)
    - Output structure: [CLS, register_tokens..., patch_tokens...]
    
    Available models (LVD-1689M pretrained):
    - facebook/dinov3-vits16-pretrain-lvd1689m (21M params)
    - facebook/dinov3-vits16plus-pretrain-lvd1689m (29M params)
    - facebook/dinov3-vitb16-pretrain-lvd1689m (86M params)
    - facebook/dinov3-vitl16-pretrain-lvd1689m (300M params)
    - facebook/dinov3-vith16plus-pretrain-lvd1689m (840M params)
    - facebook/dinov3-vit7b16-pretrain-lvd1689m (7B params)
    
    Args:
        device: Device to run the model on ('cuda' or 'cpu').
        model_id: HuggingFace model identifier for DINOv3.
        batch_size: Batch size for processing (currently unused, single-sample).
        similarity_mode: How to compute similarity:
            - 'cls': Use only CLS token cosine similarity
            - 'patch': Use weighted patch-level similarity (default)
            - 'both': Average of CLS and patch similarity
        patch_aggregation: How to aggregate patch similarities:
            - 'pool_then_compare': Weighted pool patches, then cosine (default)
            - 'mean': Mean of all pairwise patch similarities
            - 'max': Symmetric best-match average
        extraction_mode: How to extract hair region:
            - 'masked': Full image with non-hair masked to background
            - 'cropped': Crop to hair bounding box
            - 'context': Include some context around hair region
        mask_threshold: Threshold for considering a patch as "hair" (0-1).
    """
    name = "dinov3_hair_similarity"

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        batch_size: int = 16,
        similarity_mode: str = "patch",
        patch_aggregation: str = "pool_then_compare",
        extraction_mode: str = "masked",
        mask_threshold: float = 0.5,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        min_hair_size: int = 64,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model_id = model_id
        self.similarity_mode = similarity_mode
        self.patch_aggregation = patch_aggregation
        self.extraction_mode = extraction_mode
        self.mask_threshold = mask_threshold
        self.background_color = background_color
        self.min_hair_size = min_hair_size

        self.processor = None
        self.model = None
        self.num_register_tokens = None
        self.patch_size = None
        self.image_size = None

    def _load_model(self):
        if self.model is None:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            
            # Extract model configuration
            config = self.model.config
            self.num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)
            self.patch_size = int(getattr(config, "patch_size", 16))
            # Get image size from processor or config
            if hasattr(self.processor, "size"):
                size_dict = self.processor.size
                if isinstance(size_dict, dict):
                    self.image_size = size_dict.get("height", size_dict.get("shortest_edge", 224))
                else:
                    self.image_size = size_dict
            else:
                self.image_size = int(getattr(config, "image_size", 224))

    def _unload_model(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.num_register_tokens = None
            self.patch_size = None
            self.image_size = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def _extract_hair_image(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Extract hair region from image with aligned mask.
        
        Returns:
            Tuple of (extracted_image, aligned_mask) where both are the same size
            and the mask is aligned with the extracted image coordinates.
        """
        if self.extraction_mode == "cropped":
            return self._extract_hair_region_with_aligned_mask(img, mask, crop_to_bbox=True)
        if self.extraction_mode == "context":
            extracted = _extract_hair_region_with_context(img, mask, context_ratio=0.2)
            aligned_mask = _resize_like(_to_l(mask), extracted, resample=Image.BILINEAR)
            return extracted, aligned_mask
        # Default "masked" mode: keep full image, mask non-hair to background
        return self._extract_hair_region_with_aligned_mask(img, mask, crop_to_bbox=False)

    def _extract_hair_region_with_aligned_mask(
        self,
        img: Image.Image,
        hair_mask: Image.Image,
        crop_to_bbox: bool = True,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Extract hair region with properly aligned mask.
        
        The mask is kept aligned with the output image through all transformations
        (resizing, cropping) so patch weights can be computed correctly.
        """
        img = _to_rgb(img)
        mask = _to_l(hair_mask)
        mask = _resize_like(mask, img, resample=Image.BILINEAR)

        img_np = np.array(img, dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.float32) / 255.0

        bg = np.array(self.background_color, dtype=np.uint8).reshape(1, 1, 3)
        bg = np.broadcast_to(bg, img_np.shape).copy()

        # Apply mask: hair visible, non-hair replaced with background
        out_np = (mask_np[..., None] * img_np + (1.0 - mask_np[..., None]) * bg).astype(np.uint8)
        out_mask_np = (mask_np * 255.0).astype(np.uint8)

        if crop_to_bbox:
            binary = mask_np > 0.5
            rows = np.any(binary, axis=1)
            cols = np.any(binary, axis=0)

            if not np.any(rows) or not np.any(cols):
                # No hair detected - return small placeholder
                out_img = Image.fromarray(out_np).resize(
                    (self.min_hair_size, self.min_hair_size), Image.BILINEAR
                )
                out_mask = Image.fromarray(out_mask_np, mode="L").resize(
                    (self.min_hair_size, self.min_hair_size), Image.BILINEAR
                )
                return out_img, out_mask

            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            # Add padding around bounding box
            pad = 5
            y_min = max(0, y_min - pad)
            y_max = min(out_np.shape[0] - 1, y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(out_np.shape[1] - 1, x_max + pad)

            # Crop both image and mask identically
            out_np = out_np[y_min:y_max + 1, x_min:x_max + 1]
            out_mask_np = out_mask_np[y_min:y_max + 1, x_min:x_max + 1]

            h, w = out_np.shape[:2]
            if h < self.min_hair_size or w < self.min_hair_size:
                scale = max(self.min_hair_size / h, self.min_hair_size / w)
                new_h, new_w = int(h * scale), int(w * scale)
                out_img = Image.fromarray(out_np).resize((new_w, new_h), Image.BILINEAR)
                out_mask = Image.fromarray(out_mask_np, mode="L").resize((new_w, new_h), Image.BILINEAR)
                return out_img, out_mask

        return Image.fromarray(out_np), Image.fromarray(out_mask_np, mode="L")

    def _align_mask_to_preprocessed(
        self,
        mask: Image.Image,
        pixel_values: torch.Tensor,
    ) -> Image.Image:
        """
        Align the hair mask to match the preprocessed image dimensions.
        
        The DINOv3 processor resizes images to a fixed size (typically 224x224).
        The mask must be resized to match this processed size for correct
        patch-to-mask correspondence.
        
        Args:
            mask: Hair mask aligned with the original extracted image.
            pixel_values: Preprocessed image tensor of shape (1, 3, H, W).
            
        Returns:
            Mask resized to match the preprocessed image dimensions.
        """
        _, _, H, W = pixel_values.shape
        return mask.convert("L").resize((W, H), resample=Image.BILINEAR)

    def _get_features(
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Image.Image]:
        """
        Extract DINOv3 features and align mask to preprocessed dimensions.
        
        Args:
            img: Input image (already has hair extracted/masked).
            mask: Hair mask aligned with img.
            
        Returns:
            Tuple of (cls_token, patch_tokens, pixel_values, aligned_mask)
            where aligned_mask matches the preprocessed image size.
        """
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        x = outputs.last_hidden_state

        # Extract CLS token (always first)
        cls = x[:, 0, :]
        
        # Extract patch tokens (after CLS and any register tokens)
        # DINOv3 structure: [CLS, register_tokens..., patch_tokens...]
        patch_start = 1 + self.num_register_tokens
        patches = x[:, patch_start:, :]
        
        # Align mask to preprocessed image dimensions
        aligned_mask = self._align_mask_to_preprocessed(mask, inputs["pixel_values"])

        return cls, patches, inputs["pixel_values"], aligned_mask

    def _compute_patch_similarity_masked(
        self,
        patches_ref: torch.Tensor,
        pv_ref: torch.Tensor,
        mask_ref: Image.Image,
        patches_gen: torch.Tensor,
        pv_gen: torch.Tensor,
        mask_gen: Image.Image,
    ) -> float:
        """
        Compute similarity between patch features weighted by hair mask coverage.
        
        Args:
            patches_ref: Reference patch tokens (1, N, D).
            pv_ref: Reference preprocessed pixel values (1, 3, H, W).
            mask_ref: Reference hair mask (already aligned to pv_ref size).
            patches_gen: Generated patch tokens (1, M, D).
            pv_gen: Generated preprocessed pixel values (1, 3, H, W).
            mask_gen: Generated hair mask (already aligned to pv_gen size).
            
        Returns:
            Similarity score in [-1, 1].
        """
        # Compute patch weights from masks
        w_ref = self._mask_to_patch_weights(pv_ref, mask_ref, self.patch_size)
        w_gen = self._mask_to_patch_weights(pv_gen, mask_gen, self.patch_size)

        if self.patch_aggregation == "pool_then_compare":
            return self._weighted_pool_then_cosine(patches_ref, w_ref, patches_gen, w_gen)

        # Select patches above threshold for pairwise comparison
        pr, wr = self._select_patches(patches_ref, w_ref, threshold=self.mask_threshold)
        pg, wg = self._select_patches(patches_gen, w_gen, threshold=self.mask_threshold)
        return self._pairwise_similarity(pr, pg, mode=self.patch_aggregation)

    @torch.inference_mode()
    def compute(self, samples: List[Sample]) -> float:
        """
        Compute mean hair similarity across all samples.
        
        Args:
            samples: List of Sample objects with reference, generated images and masks.
            
        Returns:
            Mean similarity score across all samples.
        """
        self._load_model()
        sims: List[float] = []

        for s in samples:
            if s.reference is None or s.generated is None:
                raise ValueError("Requires Sample.reference and Sample.generated.")
            if s.hair_mask_reference is None or s.hair_mask_generated is None:
                raise ValueError("Requires Sample.hair_mask_reference and Sample.hair_mask_generated.")

            # Extract hair regions with aligned masks
            ref_img, ref_mask = self._extract_hair_image(s.reference, s.hair_mask_reference)
            gen_img, gen_mask = self._extract_hair_image(s.generated, s.hair_mask_generated)

            # Get features with masks aligned to preprocessed dimensions
            cls_ref, patches_ref, pv_ref, aligned_mask_ref = self._get_features(ref_img, ref_mask)
            cls_gen, patches_gen, pv_gen, aligned_mask_gen = self._get_features(gen_img, gen_mask)

            if self.similarity_mode == "cls":
                sim = self._cosine_cls(cls_ref, cls_gen)
            elif self.similarity_mode == "patch":
                sim = self._compute_patch_similarity_masked(
                    patches_ref, pv_ref, aligned_mask_ref,
                    patches_gen, pv_gen, aligned_mask_gen
                )
            elif self.similarity_mode == "both":
                cls_sim = self._cosine_cls(cls_ref, cls_gen)
                patch_sim = self._compute_patch_similarity_masked(
                    patches_ref, pv_ref, aligned_mask_ref,
                    patches_gen, pv_gen, aligned_mask_gen
                )
                sim = 0.5 * (cls_sim + patch_sim)
            else:
                sim = self._cosine_cls(cls_ref, cls_gen)

            sims.append(sim)

        out = float(np.mean(sims)) if sims else 0.0
        self._unload_model()
        return out

    @torch.inference_mode()
    def compute_per_sample(self, samples: List[Sample]) -> List[Optional[float]]:
        """
        Compute hair similarity for each sample individually.
        
        Args:
            samples: List of Sample objects.
            
        Returns:
            List of similarity scores (None for failed samples).
        """
        self._load_model()
        results: List[Optional[float]] = []

        for s in samples:
            if s.reference is None or s.generated is None:
                results.append(None)
                continue
            if s.hair_mask_reference is None or s.hair_mask_generated is None:
                results.append(None)
                continue

            try:
                ref_img, ref_mask = self._extract_hair_image(s.reference, s.hair_mask_reference)
                gen_img, gen_mask = self._extract_hair_image(s.generated, s.hair_mask_generated)

                cls_ref, patches_ref, pv_ref, aligned_mask_ref = self._get_features(ref_img, ref_mask)
                cls_gen, patches_gen, pv_gen, aligned_mask_gen = self._get_features(gen_img, gen_mask)

                if self.similarity_mode == "cls":
                    sim = self._cosine_cls(cls_ref, cls_gen)
                elif self.similarity_mode == "patch":
                    sim = self._compute_patch_similarity_masked(
                        patches_ref, pv_ref, aligned_mask_ref,
                        patches_gen, pv_gen, aligned_mask_gen
                    )
                elif self.similarity_mode == "both":
                    cls_sim = self._cosine_cls(cls_ref, cls_gen)
                    patch_sim = self._compute_patch_similarity_masked(
                        patches_ref, pv_ref, aligned_mask_ref,
                        patches_gen, pv_gen, aligned_mask_gen
                    )
                    sim = 0.5 * (cls_sim + patch_sim)
                else:
                    sim = self._cosine_cls(cls_ref, cls_gen)

                results.append(sim)
            except Exception:
                results.append(None)

        self._unload_model()
        return results


# -----------------------------
# DINOv2 Hair Similarity
# -----------------------------

class DINOv2HairSimilarityMetric(Metric, _HairPatchMaskingMixin):
    """
    Hair similarity metric using DINOv2 (facebook/dinov2-*) vision transformer.
    
    DINOv2 produces high-quality dense features for various vision tasks.
    This metric computes similarity between hair regions in reference and 
    generated images using patch-level or CLS-token features.
    
    Key features:
    - DINOv2 ViT models use patch_size=14
    - DINOv2-with-registers variants have register tokens (num_register_tokens in config)
    - Output structure: [CLS, register_tokens..., patch_tokens...]
    
    Available models:
    - facebook/dinov2-small, facebook/dinov2-base, facebook/dinov2-large, facebook/dinov2-giant
    - facebook/dinov2-with-registers-small, facebook/dinov2-with-registers-base, 
      facebook/dinov2-with-registers-large, facebook/dinov2-with-registers-giant
    
    Args:
        device: Device to run the model on ('cuda' or 'cpu').
        model_id: HuggingFace model identifier for DINOv2.
        batch_size: Batch size for processing (currently unused, single-sample).
        similarity_mode: How to compute similarity:
            - 'cls': Use only CLS token cosine similarity
            - 'patch': Use weighted patch-level similarity (default)
            - 'both': Average of CLS and patch similarity
        patch_aggregation: How to aggregate patch similarities:
            - 'pool_then_compare': Weighted pool patches, then cosine (default)
            - 'mean': Mean of all pairwise patch similarities
            - 'max': Symmetric best-match average
        extraction_mode: How to extract hair region:
            - 'masked': Full image with non-hair masked to background
            - 'cropped': Crop to hair bounding box
            - 'context': Include some context around hair region
        mask_threshold: Threshold for considering a patch as "hair" (0-1).
        background_color: RGB color for non-hair pixels in masked/cropped modes.
        min_hair_size: Minimum size for extracted hair region.
    """
    name = "dinov2_hair_similarity"

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "facebook/dinov2-with-registers-large",
        batch_size: int = 16,
        similarity_mode: str = "patch",
        patch_aggregation: str = "pool_then_compare",
        extraction_mode: str = "masked",
        mask_threshold: float = 0.5,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        min_hair_size: int = 64,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model_id = model_id
        self.similarity_mode = similarity_mode
        self.patch_aggregation = patch_aggregation
        self.extraction_mode = extraction_mode
        self.mask_threshold = mask_threshold
        self.background_color = background_color
        self.min_hair_size = min_hair_size

        self.processor = None
        self.model = None
        self.patch_size = None
        self.num_register_tokens = None
        self.image_size = None

    def _load_model(self):
        if self.model is None:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            
            # Extract model configuration
            config = self.model.config
            self.patch_size = int(getattr(config, "patch_size", 14))
            self.num_register_tokens = int(getattr(config, "num_register_tokens", 0) or 0)
            # Get image size from processor or config
            if hasattr(self.processor, "size"):
                size_dict = self.processor.size
                if isinstance(size_dict, dict):
                    self.image_size = size_dict.get("height", size_dict.get("shortest_edge", 518))
                else:
                    self.image_size = size_dict
            else:
                self.image_size = int(getattr(config, "image_size", 518))

    def _unload_model(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.patch_size = None
            self.num_register_tokens = None
            self.image_size = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def _extract_hair_image(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Extract hair region from image with aligned mask.
        
        Returns:
            Tuple of (extracted_image, aligned_mask) where both are the same size
            and the mask is aligned with the extracted image coordinates.
        """
        if self.extraction_mode == "cropped":
            return self._extract_hair_region_with_aligned_mask(img, mask, crop_to_bbox=True)
        if self.extraction_mode == "context":
            extracted = _extract_hair_region_with_context(img, mask, context_ratio=0.2)
            aligned_mask = _resize_like(_to_l(mask), extracted, resample=Image.BILINEAR)
            return extracted, aligned_mask
        # Default "masked" mode: keep full image, mask non-hair to background
        return self._extract_hair_region_with_aligned_mask(img, mask, crop_to_bbox=False)

    def _extract_hair_region_with_aligned_mask(
        self,
        img: Image.Image,
        hair_mask: Image.Image,
        crop_to_bbox: bool = True,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Extract hair region with properly aligned mask.
        
        The mask is kept aligned with the output image through all transformations
        (resizing, cropping) so patch weights can be computed correctly.
        """
        img = _to_rgb(img)
        mask = _to_l(hair_mask)
        mask = _resize_like(mask, img, resample=Image.BILINEAR)

        img_np = np.array(img, dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.float32) / 255.0

        bg = np.array(self.background_color, dtype=np.uint8).reshape(1, 1, 3)
        bg = np.broadcast_to(bg, img_np.shape).copy()

        # Apply mask: hair visible, non-hair replaced with background
        out_np = (mask_np[..., None] * img_np + (1.0 - mask_np[..., None]) * bg).astype(np.uint8)
        out_mask_np = (mask_np * 255.0).astype(np.uint8)

        if crop_to_bbox:
            binary = mask_np > 0.5
            rows = np.any(binary, axis=1)
            cols = np.any(binary, axis=0)

            if not np.any(rows) or not np.any(cols):
                # No hair detected - return small placeholder
                out_img = Image.fromarray(out_np).resize(
                    (self.min_hair_size, self.min_hair_size), Image.BILINEAR
                )
                out_mask = Image.fromarray(out_mask_np, mode="L").resize(
                    (self.min_hair_size, self.min_hair_size), Image.BILINEAR
                )
                return out_img, out_mask

            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            # Add padding around bounding box
            pad = 5
            y_min = max(0, y_min - pad)
            y_max = min(out_np.shape[0] - 1, y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(out_np.shape[1] - 1, x_max + pad)

            # Crop both image and mask identically
            out_np = out_np[y_min:y_max + 1, x_min:x_max + 1]
            out_mask_np = out_mask_np[y_min:y_max + 1, x_min:x_max + 1]

            h, w = out_np.shape[:2]
            if h < self.min_hair_size or w < self.min_hair_size:
                scale = max(self.min_hair_size / h, self.min_hair_size / w)
                new_h, new_w = int(h * scale), int(w * scale)
                out_img = Image.fromarray(out_np).resize((new_w, new_h), Image.BILINEAR)
                out_mask = Image.fromarray(out_mask_np, mode="L").resize((new_w, new_h), Image.BILINEAR)
                return out_img, out_mask

        return Image.fromarray(out_np), Image.fromarray(out_mask_np, mode="L")

    def _align_mask_to_preprocessed(
        self,
        mask: Image.Image,
        pixel_values: torch.Tensor,
    ) -> Image.Image:
        """
        Align the hair mask to match the preprocessed image dimensions.
        
        The DINOv2 processor resizes images to a fixed size (typically 518x518).
        The mask must be resized to match this processed size for correct
        patch-to-mask correspondence.
        
        Args:
            mask: Hair mask aligned with the original extracted image.
            pixel_values: Preprocessed image tensor of shape (1, 3, H, W).
            
        Returns:
            Mask resized to match the preprocessed image dimensions.
        """
        _, _, H, W = pixel_values.shape
        return mask.convert("L").resize((W, H), resample=Image.BILINEAR)

    def _get_features(
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Image.Image]:
        """
        Extract DINOv2 features and align mask to preprocessed dimensions.
        
        Args:
            img: Input image (already has hair extracted/masked).
            mask: Hair mask aligned with img.
            
        Returns:
            Tuple of (cls_token, patch_tokens, pixel_values, aligned_mask)
            where aligned_mask matches the preprocessed image size.
        """
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        x = outputs.last_hidden_state

        # Extract CLS token (always first)
        cls = x[:, 0, :]
        
        # Extract patch tokens (after CLS and any register tokens)
        # DINOv2 structure: [CLS, register_tokens..., patch_tokens...]
        patch_start = 1 + self.num_register_tokens
        patches = x[:, patch_start:, :]
        
        # Align mask to preprocessed image dimensions
        aligned_mask = self._align_mask_to_preprocessed(mask, inputs["pixel_values"])

        return cls, patches, inputs["pixel_values"], aligned_mask

    def _compute_patch_similarity_masked(
        self,
        patches_ref: torch.Tensor,
        pv_ref: torch.Tensor,
        mask_ref: Image.Image,
        patches_gen: torch.Tensor,
        pv_gen: torch.Tensor,
        mask_gen: Image.Image,
    ) -> float:
        """
        Compute similarity between patch features weighted by hair mask coverage.
        
        Args:
            patches_ref: Reference patch tokens (1, N, D).
            pv_ref: Reference preprocessed pixel values (1, 3, H, W).
            mask_ref: Reference hair mask (already aligned to pv_ref size).
            patches_gen: Generated patch tokens (1, M, D).
            pv_gen: Generated preprocessed pixel values (1, 3, H, W).
            mask_gen: Generated hair mask (already aligned to pv_gen size).
            
        Returns:
            Similarity score in [-1, 1].
        """
        # Compute patch weights from masks
        w_ref = self._mask_to_patch_weights(pv_ref, mask_ref, self.patch_size)
        w_gen = self._mask_to_patch_weights(pv_gen, mask_gen, self.patch_size)

        if self.patch_aggregation == "pool_then_compare":
            return self._weighted_pool_then_cosine(patches_ref, w_ref, patches_gen, w_gen)

        # Select patches above threshold for pairwise comparison
        pr, _ = self._select_patches(patches_ref, w_ref, threshold=self.mask_threshold)
        pg, _ = self._select_patches(patches_gen, w_gen, threshold=self.mask_threshold)
        return self._pairwise_similarity(pr, pg, mode=self.patch_aggregation)

    @torch.inference_mode()
    def compute(self, samples: List[Sample]) -> float:
        """
        Compute mean hair similarity across all samples.
        
        Args:
            samples: List of Sample objects with reference, generated images and masks.
            
        Returns:
            Mean similarity score across all samples.
        """
        self._load_model()
        sims: List[float] = []

        for s in samples:
            if s.reference is None or s.generated is None:
                raise ValueError("Requires Sample.reference and Sample.generated.")
            if s.hair_mask_reference is None or s.hair_mask_generated is None:
                raise ValueError("Requires Sample.hair_mask_reference and Sample.hair_mask_generated.")

            # Extract hair regions with aligned masks
            ref_img, ref_mask = self._extract_hair_image(s.reference, s.hair_mask_reference)
            gen_img, gen_mask = self._extract_hair_image(s.generated, s.hair_mask_generated)

            # Get features with masks aligned to preprocessed dimensions
            cls_ref, patches_ref, pv_ref, aligned_mask_ref = self._get_features(ref_img, ref_mask)
            cls_gen, patches_gen, pv_gen, aligned_mask_gen = self._get_features(gen_img, gen_mask)

            if self.similarity_mode == "cls":
                sim = self._cosine_cls(cls_ref, cls_gen)
            elif self.similarity_mode == "patch":
                sim = self._compute_patch_similarity_masked(
                    patches_ref, pv_ref, aligned_mask_ref,
                    patches_gen, pv_gen, aligned_mask_gen
                )
            elif self.similarity_mode == "both":
                cls_sim = self._cosine_cls(cls_ref, cls_gen)
                patch_sim = self._compute_patch_similarity_masked(
                    patches_ref, pv_ref, aligned_mask_ref,
                    patches_gen, pv_gen, aligned_mask_gen
                )
                sim = 0.5 * (cls_sim + patch_sim)
            else:
                sim = self._cosine_cls(cls_ref, cls_gen)

            sims.append(sim)

        out = float(np.mean(sims)) if sims else 0.0
        self._unload_model()
        return out

    @torch.inference_mode()
    def compute_per_sample(self, samples: List[Sample]) -> List[Optional[float]]:
        """
        Compute hair similarity for each sample individually.
        
        Args:
            samples: List of Sample objects.
            
        Returns:
            List of similarity scores (None for failed samples).
        """
        self._load_model()
        results: List[Optional[float]] = []

        for s in samples:
            if s.reference is None or s.generated is None:
                results.append(None)
                continue
            if s.hair_mask_reference is None or s.hair_mask_generated is None:
                results.append(None)
                continue

            try:
                ref_img, ref_mask = self._extract_hair_image(s.reference, s.hair_mask_reference)
                gen_img, gen_mask = self._extract_hair_image(s.generated, s.hair_mask_generated)

                cls_ref, patches_ref, pv_ref, aligned_mask_ref = self._get_features(ref_img, ref_mask)
                cls_gen, patches_gen, pv_gen, aligned_mask_gen = self._get_features(gen_img, gen_mask)

                if self.similarity_mode == "cls":
                    sim = self._cosine_cls(cls_ref, cls_gen)
                elif self.similarity_mode == "patch":
                    sim = self._compute_patch_similarity_masked(
                        patches_ref, pv_ref, aligned_mask_ref,
                        patches_gen, pv_gen, aligned_mask_gen
                    )
                elif self.similarity_mode == "both":
                    cls_sim = self._cosine_cls(cls_ref, cls_gen)
                    patch_sim = self._compute_patch_similarity_masked(
                        patches_ref, pv_ref, aligned_mask_ref,
                        patches_gen, pv_gen, aligned_mask_gen
                    )
                    sim = 0.5 * (cls_sim + patch_sim)
                else:
                    sim = self._cosine_cls(cls_ref, cls_gen)

                results.append(sim)
            except Exception:
                results.append(None)

        self._unload_model()
        return results


# -----------------------------
# Pixio Hair Similarity
# -----------------------------
class PixioHairSimilarityMetric(Metric, _HairPatchMaskingMixin):
    name = "pixio_hair_similarity"

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "facebook/pixio-vit1b16",
        batch_size: int = 16,
        similarity_mode: str = "patch",
        patch_aggregation: str = "pool_then_compare",
        extraction_mode: str = "masked",
        use_normalized_features: bool = False,
        feature_combination: str = "patch_only",  # cls_only | patch_only | concat_avg | pixio_style
        mask_threshold: float = 0.5,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model_id = model_id

        self.similarity_mode = similarity_mode
        self.patch_aggregation = patch_aggregation
        self.extraction_mode = extraction_mode

        self.use_normalized_features = use_normalized_features
        self.feature_combination = feature_combination
        self.mask_threshold = float(mask_threshold)

        self.processor = None
        self.model = None
        self.patch_size = None
        self.n_cls_tokens = None

    def _load_model(self):
        if self.model is None:
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            self.patch_size = int(getattr(self.model.config, "patch_size", 16))
            self.n_cls_tokens = int(getattr(self.model.config, "n_cls_tokens", 8))

    def _unload_model(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.patch_size = None
            self.n_cls_tokens = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def _extract_hair_image(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.extraction_mode == "cropped":
            return self._extract_hair_region_with_aligned_mask(img, mask, crop_to_bbox=True)
        if self.extraction_mode == "context":
            extracted = _extract_hair_region_with_context(img, mask, context_ratio=0.2)
            aligned_mask = _resize_like(_to_l(mask), extracted, resample=Image.BILINEAR)
            return extracted, aligned_mask
        extracted = _extract_hair_region(img, mask, crop_to_bbox=False)
        aligned_mask = _resize_like(_to_l(mask), extracted, resample=Image.BILINEAR)
        return extracted, aligned_mask

    def _extract_hair_region_with_aligned_mask(
        self,
        img: Image.Image,
        hair_mask: Image.Image,
        crop_to_bbox: bool = True,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        min_size: int = 64,
    ) -> Tuple[Image.Image, Image.Image]:
        img = _to_rgb(img)
        mask = _to_l(hair_mask)
        mask = _resize_like(mask, img, resample=Image.BILINEAR)

        img_np = np.array(img, dtype=np.uint8)
        mask_np = np.array(mask, dtype=np.float32) / 255.0

        bg = np.array(background_color, dtype=np.uint8).reshape(1, 1, 3)
        bg = np.broadcast_to(bg, img_np.shape).copy()

        out_np = (mask_np[..., None] * img_np + (1.0 - mask_np[..., None]) * bg).astype(np.uint8)
        out_mask_np = (mask_np * 255.0).astype(np.uint8)

        if crop_to_bbox:
            binary = mask_np > self.mask_threshold
            rows = np.any(binary, axis=1)
            cols = np.any(binary, axis=0)

            if not np.any(rows) or not np.any(cols):
                out_img = Image.fromarray(out_np).resize((min_size, min_size), Image.BILINEAR)
                out_mask = Image.fromarray(out_mask_np, mode="L").resize((min_size, min_size), Image.BILINEAR)
                return out_img, out_mask

            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            pad = 5
            y_min = max(0, y_min - pad)
            y_max = min(out_np.shape[0] - 1, y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(out_np.shape[1] - 1, x_max + pad)

            out_np = out_np[y_min:y_max + 1, x_min:x_max + 1]
            out_mask_np = out_mask_np[y_min:y_max + 1, x_min:x_max + 1]

            h, w = out_np.shape[:2]
            if h < min_size or w < min_size:
                scale = max(min_size / h, min_size / w)
                new_h, new_w = int(h * scale), int(w * scale)
                out_img = Image.fromarray(out_np).resize((new_w, new_h), Image.BILINEAR)
                out_mask = Image.fromarray(out_mask_np, mode="L").resize((new_w, new_h), Image.BILINEAR)
                return out_img, out_mask

        return Image.fromarray(out_np), Image.fromarray(out_mask_np, mode="L")

    def _get_features(self, img: Image.Image):
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)

        if self.use_normalized_features:
            x = outputs.last_hidden_state
        else:
            x = outputs.hidden_states[-1]

        cls_tokens = x[:, : self.n_cls_tokens, :]
        patch_tokens = x[:, self.n_cls_tokens :, :]

        return cls_tokens, patch_tokens, inputs["pixel_values"]

    def _compute_cls_similarity(self, cls_ref: torch.Tensor, cls_gen: torch.Tensor) -> float:
        ref = cls_ref.mean(dim=1)
        gen = cls_gen.mean(dim=1)
        return self._cosine_cls(ref, gen)

    def _weighted_pool_then_cosine_pixio(
        self,
        patches_a: torch.Tensor,   # (1,N,D)
        weights_a: torch.Tensor,   # (N,)
        patches_b: torch.Tensor,   # (1,M,D)
        weights_b: torch.Tensor,   # (M,)
        eps: float = 1e-8,
    ) -> float:
        # IMPORTANT: For Pixio, avoid normalizing patch tokens before pooling.
        # Pool raw (preferably pre-LN) tokens with mask weights, then normalize once.
        if patches_a.shape[0] != 1 or patches_b.shape[0] != 1:
            raise ValueError("Expected batch size 1 for patch tensors.")

        wa = weights_a.clamp(min=0.0).to(patches_a.device, dtype=patches_a.dtype)
        wb = weights_b.clamp(min=0.0).to(patches_b.device, dtype=patches_b.dtype)

        wa_sum = wa.sum().clamp(min=eps)
        wb_sum = wb.sum().clamp(min=eps)

        emb_a = (patches_a[0] * wa[:, None]).sum(dim=0) / wa_sum
        emb_b = (patches_b[0] * wb[:, None]).sum(dim=0) / wb_sum

        emb_a = F.normalize(emb_a.float(), dim=-1)
        emb_b = F.normalize(emb_b.float(), dim=-1)

        return float(torch.clamp(torch.dot(emb_a, emb_b), -1.0, 1.0).item())

    def _compute_patch_similarity_masked(
        self,
        patches_ref: torch.Tensor,
        pv_ref: torch.Tensor,
        mask_ref: Image.Image,
        patches_gen: torch.Tensor,
        pv_gen: torch.Tensor,
        mask_gen: Image.Image,
    ) -> float:
        w_ref = self._mask_to_patch_weights(pv_ref, mask_ref, self.patch_size)
        w_gen = self._mask_to_patch_weights(pv_gen, mask_gen, self.patch_size)

        # Sanity check: patch token count must match patch grid size
        _, _, H_ref, W_ref = pv_ref.shape
        expected_ref = (H_ref // self.patch_size) * (W_ref // self.patch_size)
        if patches_ref.shape[1] != expected_ref:
            raise ValueError(f"Pixio patch count mismatch (ref): {patches_ref.shape[1]} vs {expected_ref}")

        _, _, H_gen, W_gen = pv_gen.shape
        expected_gen = (H_gen // self.patch_size) * (W_gen // self.patch_size)
        if patches_gen.shape[1] != expected_gen:
            raise ValueError(f"Pixio patch count mismatch (gen): {patches_gen.shape[1]} vs {expected_gen}")

        if self.patch_aggregation == "pool_then_compare":
            return self._weighted_pool_then_cosine_pixio(patches_ref, w_ref, patches_gen, w_gen)

        pr, _ = self._select_patches(patches_ref, w_ref, threshold=self.mask_threshold)
        pg, _ = self._select_patches(patches_gen, w_gen, threshold=self.mask_threshold)
        return self._pairwise_similarity(pr, pg, mode=self.patch_aggregation)

    def _compute_pixio_style_similarity(
        self,
        cls_ref: torch.Tensor,
        patches_ref: torch.Tensor,
        pv_ref: torch.Tensor,
        mask_ref: Image.Image,
        cls_gen: torch.Tensor,
        patches_gen: torch.Tensor,
        pv_gen: torch.Tensor,
        mask_gen: Image.Image,
    ) -> float:
        w_ref = self._mask_to_patch_weights(pv_ref, mask_ref, self.patch_size)
        w_gen = self._mask_to_patch_weights(pv_gen, mask_gen, self.patch_size)

        cls_ref_avg = cls_ref.mean(dim=1, keepdim=True)
        cls_gen_avg = cls_gen.mean(dim=1, keepdim=True)

        cls_ref_exp = cls_ref_avg.expand(-1, patches_ref.shape[1], -1)
        cls_gen_exp = cls_gen_avg.expand(-1, patches_gen.shape[1], -1)

        comb_ref = torch.cat([patches_ref, cls_ref_exp], dim=-1)
        comb_gen = torch.cat([patches_gen, cls_gen_exp], dim=-1)

        return self._weighted_pool_then_cosine_pixio(comb_ref, w_ref, comb_gen, w_gen)

    def _compute_similarity(
        self,
        cls_ref: torch.Tensor,
        patches_ref: torch.Tensor,
        pv_ref: torch.Tensor,
        mask_ref: Image.Image,
        cls_gen: torch.Tensor,
        patches_gen: torch.Tensor,
        pv_gen: torch.Tensor,
        mask_gen: Image.Image,
    ) -> float:
        # feature_combination takes precedence and is unambiguous
        if self.feature_combination == "pixio_style":
            return self._compute_pixio_style_similarity(
                cls_ref, patches_ref, pv_ref, mask_ref,
                cls_gen, patches_gen, pv_gen, mask_gen
            )

        if self.feature_combination == "patch_only":
            return self._compute_patch_similarity_masked(patches_ref, pv_ref, mask_ref, patches_gen, pv_gen, mask_gen)

        if self.feature_combination == "cls_only":
            return self._compute_cls_similarity(cls_ref, cls_gen)

        if self.feature_combination == "concat_avg":
            w_ref = self._mask_to_patch_weights(pv_ref, mask_ref, self.patch_size)
            w_gen = self._mask_to_patch_weights(pv_gen, mask_gen, self.patch_size)

            eps = 1e-8
            wr_sum = w_ref.sum().clamp(min=eps)
            wg_sum = w_gen.sum().clamp(min=eps)

            # IMPORTANT: do not normalize patch tokens before pooling for Pixio
            patch_ref_avg = (patches_ref[0] * w_ref[:, None]).sum(dim=0) / wr_sum
            patch_gen_avg = (patches_gen[0] * w_gen[:, None]).sum(dim=0) / wg_sum

            cls_ref_avg = cls_ref.mean(dim=1)[0]
            cls_gen_avg = cls_gen.mean(dim=1)[0]

            # Normalize components once, then concatenate and normalize once more
            patch_ref_avg = F.normalize(patch_ref_avg.float(), dim=-1)
            patch_gen_avg = F.normalize(patch_gen_avg.float(), dim=-1)
            cls_ref_avg = F.normalize(cls_ref_avg.float(), dim=-1)
            cls_gen_avg = F.normalize(cls_gen_avg.float(), dim=-1)

            ref = torch.cat([cls_ref_avg, patch_ref_avg], dim=-1)
            gen = torch.cat([cls_gen_avg, patch_gen_avg], dim=-1)

            ref = F.normalize(ref, dim=-1)
            gen = F.normalize(gen, dim=-1)
            return float(torch.clamp(torch.dot(ref, gen), -1.0, 1.0).item())

        # Fallback to similarity_mode only when feature_combination is unrecognized
        if self.similarity_mode == "patch":
            return self._compute_patch_similarity_masked(patches_ref, pv_ref, mask_ref, patches_gen, pv_gen, mask_gen)
        if self.similarity_mode == "both":
            cls_sim = self._compute_cls_similarity(cls_ref, cls_gen)
            patch_sim = self._compute_patch_similarity_masked(patches_ref, pv_ref, mask_ref, patches_gen, pv_gen, mask_gen)
            return 0.5 * (cls_sim + patch_sim)
        return self._compute_cls_similarity(cls_ref, cls_gen)

    @torch.inference_mode()
    def compute(self, samples: List[Sample]) -> float:
        self._load_model()
        sims: List[float] = []

        for s in samples:
            if s.reference is None or s.generated is None:
                raise ValueError("Requires Sample.reference and Sample.generated.")
            if s.hair_mask_reference is None or s.hair_mask_generated is None:
                raise ValueError("Requires Sample.hair_mask_reference and Sample.hair_mask_generated.")

            ref_img, ref_mask = self._extract_hair_image(s.reference, s.hair_mask_reference)
            gen_img, gen_mask = self._extract_hair_image(s.generated, s.hair_mask_generated)

            cls_ref, patches_ref, pv_ref = self._get_features(ref_img)
            cls_gen, patches_gen, pv_gen = self._get_features(gen_img)

            sim = self._compute_similarity(
                cls_ref, patches_ref, pv_ref, ref_mask,
                cls_gen, patches_gen, pv_gen, gen_mask
            )
            sims.append(sim)

        out = float(np.mean(sims)) if sims else 0.0
        self._unload_model()
        return out

    @torch.inference_mode()
    def compute_per_sample(self, samples: List[Sample]) -> List[Optional[float]]:
        self._load_model()
        results: List[Optional[float]] = []

        for s in samples:
            if s.reference is None or s.generated is None:
                results.append(None)
                continue
            if s.hair_mask_reference is None or s.hair_mask_generated is None:
                results.append(None)
                continue

            try:
                ref_img, ref_mask = self._extract_hair_image(s.reference, s.hair_mask_reference)
                gen_img, gen_mask = self._extract_hair_image(s.generated, s.hair_mask_generated)

                cls_ref, patches_ref, pv_ref = self._get_features(ref_img)
                cls_gen, patches_gen, pv_gen = self._get_features(gen_img)

                sim = self._compute_similarity(
                    cls_ref, patches_ref, pv_ref, ref_mask,
                    cls_gen, patches_gen, pv_gen, gen_mask
                )
                results.append(sim)
            except Exception:
                results.append(None)

        self._unload_model()
        return results

# -----------------------------
# Hair Prompt Similarity (Text Embedding)
# -----------------------------

class HairPromptSimilarityMetric(Metric):
    """
    Hair Prompt Similarity: cosine similarity between text embeddings of
    hair descriptions from reference image and generated image.
    
    
    Uses the Qwen3-Embedding-8B model for high-quality text embeddings.
    Hair descriptions are loaded from prompt.json files generated by the
    CaptionerPipeline.
    
    This metric measures semantic similarity between the hair descriptions,
    providing a text-based assessment of hair transfer quality.
    """
    name = "hair_prompt_similarity"

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "Qwen/Qwen3-Embedding-8B",
        batch_size: int = 16,
    ):
        self.device = device
        self.batch_size = batch_size
        self.model_id = model_id
        self.model = None

    def _load_model(self):
        if self.model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers is required for HairPromptSimilarityMetric. "
                    "Install with: pip install sentence-transformers>=2.7.0"
                )
            print(f"  Loading {self.model_id}...")
            self.model = SentenceTransformer(
                self.model_id,
                model_kwargs={"torch_dtype": torch.float16},
                tokenizer_kwargs={"padding_side": "left"},
            )
            print(f"  Model loaded successfully.")

    def _unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text string."""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0]

    def _get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a batch of texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=self.batch_size)
        return embeddings

    @torch.inference_mode()
    def compute(self, samples: List[Any], prompts_ref: List[str], prompts_gen: List[str]) -> float:
        """
        Compute average cosine similarity between reference and generated hair prompts.
        
        Args:
            samples: List of samples (not used directly, kept for API consistency)
            prompts_ref: List of reference hair description strings
            prompts_gen: List of generated hair description strings
            
        Returns:
            Mean cosine similarity across all pairs
        """
        if len(prompts_ref) != len(prompts_gen):
            raise ValueError(
                f"Number of reference prompts ({len(prompts_ref)}) must match "
                f"number of generated prompts ({len(prompts_gen)})"
            )
        
        self._load_model()
        
        # Get embeddings for all prompts
        ref_embeddings = self._get_embeddings_batch(prompts_ref)
        gen_embeddings = self._get_embeddings_batch(prompts_gen)
        
        # Normalize embeddings
        ref_embeddings = ref_embeddings / np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
        gen_embeddings = gen_embeddings / np.linalg.norm(gen_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarities
        similarities = np.sum(ref_embeddings * gen_embeddings, axis=1)
        
        result = float(np.mean(similarities))
        self._unload_model()
        return result

    @torch.inference_mode()
    def compute_per_sample(
        self, 
        samples: List[Any], 
        prompts_ref: List[str], 
        prompts_gen: List[str]
    ) -> List[Optional[float]]:
        """
        Compute cosine similarity for each sample pair.
        
        Args:
            samples: List of samples (not used directly, kept for API consistency)
            prompts_ref: List of reference hair description strings
            prompts_gen: List of generated hair description strings
            
        Returns:
            List of cosine similarities for each pair
        """
        if len(prompts_ref) != len(prompts_gen):
            raise ValueError(
                f"Number of reference prompts ({len(prompts_ref)}) must match "
                f"number of generated prompts ({len(prompts_gen)})"
            )
        
        self._load_model()
        
        results: List[Optional[float]] = []
        
        # Process in batches
        for i in range(0, len(prompts_ref), self.batch_size):
            batch_ref = prompts_ref[i:i + self.batch_size]
            batch_gen = prompts_gen[i:i + self.batch_size]
            
            # Get embeddings
            ref_embeddings = self._get_embeddings_batch(batch_ref)
            gen_embeddings = self._get_embeddings_batch(batch_gen)
            
            # Normalize embeddings
            ref_embeddings = ref_embeddings / np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
            gen_embeddings = gen_embeddings / np.linalg.norm(gen_embeddings, axis=1, keepdims=True)
            
            # Compute cosine similarities
            batch_similarities = np.sum(ref_embeddings * gen_embeddings, axis=1)
            
            for sim in batch_similarities:
                # Handle any NaN values that might occur
                if np.isnan(sim):
                    results.append(None)
                else:
                    results.append(float(sim))
        
        self._unload_model()
        return results


# -----------------------------
# Convenience runner
# -----------------------------

class MetricSuite:
    """
    Run multiple single-metric classes and return a dict.
    """
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics

    def compute_all(self, samples: List[Sample]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for m in self.metrics:
            out[m.name] = m.compute(samples)
        return out
