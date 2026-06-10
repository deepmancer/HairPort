"""Composite the model's bald plate back into the original frame (no color match).

The bald plate is composited as-is; a hair-removal matte changes only the hair /
scalp / revealed-background region and leaves everything else byte-identical.
Arrays are uint8 RGB H×W×3 at native plate resolution.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from hairport.fitting.compat import fill_mask_holes

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Change-region alpha matte (hair removal)
# --------------------------------------------------------------------------- #

def _keep_components_overlapping(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Keep only connected components of *mask* that overlap *seed* (uint8 0/255)."""
    num, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    if num <= 1:
        return (mask > 0).astype(np.uint8) * 255
    keep_labels = np.unique(labels[seed > 0])
    keep_labels = keep_labels[keep_labels != 0]
    out = np.isin(labels, keep_labels).astype(np.uint8) * 255
    return out


def build_change_alpha(
    orig: np.ndarray,
    bald: np.ndarray,
    hair_mask: np.ndarray,
    matte_dilate_px: int = 6,
    extend_band_frac: float = 0.06,
    extend_diff_threshold: int = 12,
    feather_px: int = 5,
    border_zero_frac: float = 0.02,
) -> np.ndarray:
    """Soft removal matte: grow the hair seed through the model-changed band
    (``|orig-bald|>thr`` within ``extend_band_frac×side``) to swallow wisps the
    segmenter misses, then feather outward only (α=1 over all hair → no halo)."""
    h, w = hair_mask.shape[:2]
    side = max(h, w)
    seed = (hair_mask > 0).astype(np.uint8) * 255
    if matte_dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (matte_dilate_px * 2 + 1,) * 2)
        seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, k)
    seed = fill_mask_holes(seed)

    # Outward search band around the hair seed.
    r = max(1, int(round(extend_band_frac * side)))
    band_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1,) * 2)
    band = cv2.dilate(seed, band_k)

    # Region the model actually changed (the removed hair, incl. wisps).
    diff = np.abs(bald.astype(np.int16) - orig.astype(np.int16)).mean(axis=2)
    changed = (diff > extend_diff_threshold).astype(np.uint8) * 255

    core = ((seed > 0) | ((changed > 0) & (band > 0))).astype(np.uint8) * 255
    # close small gaps, fill holes, keep only what connects to the hair seed
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, matte_dilate_px) * 2 + 1,) * 2)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, close_k)
    core = fill_mask_holes(core)
    core = _keep_components_overlapping(core, seed)

    # Outward-only feather: distance OUTSIDE the core ramps α from 1→0 in
    # background; α stays exactly 1 over all hair/wisps (no inward ramp).
    if feather_px > 0:
        dist = cv2.distanceTransform((core == 0).astype(np.uint8), cv2.DIST_L2, 3)
        alpha = 1.0 - np.clip(dist / float(feather_px), 0.0, 1.0)
        alpha[core > 0] = 1.0
    else:
        alpha = (core > 0).astype(np.float32)

    # Zero the border band so paste is seamless (matte never reaches the edge).
    bz = max(1, int(round(border_zero_frac * side)))
    border = np.zeros((h, w), np.float32)
    border[bz : h - bz, bz : w - bz] = 1.0
    alpha = alpha * border

    return np.clip(alpha, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Grain matching
# --------------------------------------------------------------------------- #

def match_grain(
    comp: np.ndarray, orig: np.ndarray, alpha: np.ndarray, max_sigma: float = 6.0
) -> Tuple[np.ndarray, float]:
    """Add Gaussian grain matched to the original's noise into the changed region."""
    # Estimate noise sigma from the invariant region via a high-pass residual.
    inv = (alpha < 0.05) & (orig.sum(axis=2) > 0)
    if inv.sum() < 256:
        return comp, 0.0
    gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
    sigma = float(np.clip(np.std(hp[inv]), 0.0, max_sigma))
    if sigma < 0.5:
        return comp, sigma
    noise = np.random.default_rng(0).normal(0.0, sigma, comp.shape[:2]).astype(np.float32)
    a = (alpha > 0.05)[..., None]
    out = comp.astype(np.float32) + noise[..., None] * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), sigma


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #

def composite_plate(
    orig: np.ndarray,
    bald: np.ndarray,
    hair_mask: np.ndarray,
    *,
    seam_poisson: bool = True,
    grain_match: bool = True,
    matte_dilate_px: int = 6,
    extend_band_frac: float = 0.06,
    extend_diff_threshold: int = 12,
    feather_px: int = 5,
    border_zero_frac: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Composite the bald plate into the original plate (no color matching).

    Returns ``(composited_uint8, alpha_float32, comp_params)``.
    """
    params: Dict[str, Any] = {
        "seam_poisson": seam_poisson, "grain_match": grain_match,
        "matte_dilate_px": matte_dilate_px, "extend_band_frac": extend_band_frac,
        "extend_diff_threshold": extend_diff_threshold,
        "feather_px": feather_px, "border_zero_frac": border_zero_frac,
    }

    alpha = build_change_alpha(
        orig, bald, hair_mask,
        matte_dilate_px=matte_dilate_px, extend_band_frac=extend_band_frac,
        extend_diff_threshold=extend_diff_threshold, feather_px=feather_px,
        border_zero_frac=border_zero_frac,
    )

    a = alpha[..., None]
    comp = (a * bald.astype(np.float32) + (1.0 - a) * orig.astype(np.float32))
    comp = np.clip(comp + 0.5, 0, 255).astype(np.uint8)

    if seam_poisson:
        core = (alpha > 0.5).astype(np.uint8) * 255
        if core.any():
            from hairport.utility.warper import _screened_poisson_blend

            try:
                comp = _screened_poisson_blend(
                    dst_img=orig, src_img=comp, mask_u8=core,
                    lam=1.0, mixed_gradients=False,
                )
            except Exception:
                logger.warning("screened-Poisson seam blend failed; keeping alpha comp.", exc_info=True)

    if grain_match:
        comp, sigma = match_grain(comp, orig, alpha)
        params["grain_sigma"] = sigma

    return comp, alpha, params
