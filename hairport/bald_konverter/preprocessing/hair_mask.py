"""Hair mask computation for the bald converter.

Orchestrates :class:`BackgroundRemover` (BEN2) and :class:`SAMMaskExtractor`
(SAM3) to produce the hair mask of a portrait image, constrained to the
foreground silhouette.  The bald body/head segmentation is provided separately
by the SMPL-X fitting backend (see :mod:`hairport.fitting`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from ..config.defaults import SAM_HAIR_CONFIDENCE_THRESHOLD
from .background import BackgroundRemover
from .sam_extractor import SAMMaskExtractor

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Container for the masks produced by :class:`HairMaskPipeline`."""

    hair_mask: np.ndarray
    """Binary uint8 mask (0/255) of hair, constrained to the foreground."""

    silhouette: Optional[np.ndarray] = None
    """Binary uint8 mask (0/255) of the BEN2 foreground silhouette."""

    foreground: Optional[Image.Image] = None
    """RGBA foreground image (background removed)."""


class HairMaskPipeline:
    """Compute the hair mask from a single portrait image.

    Pipeline steps:

    1. **Foreground matte** (BEN2) → binary silhouette
    2. **Hair mask** (SAM3, text prompt ``"hair"``) ∩ silhouette

    Parameters
    ----------
    device : str | torch.device
        Compute device for all sub-models.
    sam_confidence : float
        Confidence threshold for the SAM hair extractor (default:
        :data:`~..config.defaults.SAM_HAIR_CONFIDENCE_THRESHOLD`).
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        sam_confidence: float = SAM_HAIR_CONFIDENCE_THRESHOLD,
    ):
        self.device = str(device)
        self.bg_remover = BackgroundRemover(device=self.device, alpha_threshold=0.5)
        self.sam_extractor = SAMMaskExtractor(
            confidence_threshold=sam_confidence,
            device=self.device,
        )
        logger.info("HairMaskPipeline initialised on %s", self.device)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def preprocess(
        self,
        image: Image.Image | np.ndarray,
        return_foreground: bool = False,
    ) -> PreprocessingResult:
        """Compute the hair mask for *image*.

        Parameters
        ----------
        image : PIL Image or ndarray
            Input portrait image.
        return_foreground : bool
            If ``True``, include the RGBA foreground in the result.

        Returns
        -------
        PreprocessingResult
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        rgb = image.convert("RGB")

        # 1) Foreground silhouette ------------------------------------------------
        foreground, silh_pil = self.bg_remover.remove_background(rgb)
        silh = (np.array(silh_pil) > 50).astype(np.uint8)

        # Sanity checks
        if silh.sum() == 0:
            h, w = silh.shape
            silh[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1
            logger.warning("Empty silhouette — using centre fallback region.")
        elif silh.sum() < 0.01 * silh.size:
            logger.warning("Very small silhouette detected.")
        elif silh.sum() > 0.95 * silh.size:
            logger.warning("Very large silhouette detected.")

        # 2) Hair mask via SAM ----------------------------------------------------
        hair_pil, _score = self.sam_extractor(rgb, prompt="hair")
        hair_mask = (np.array(hair_pil) > 127).astype(np.uint8)

        # Ensure spatial match
        if hair_mask.shape != silh.shape:
            hair_mask = cv2.resize(
                hair_mask, (silh.shape[1], silh.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Constrain to silhouette
        hair_mask = (hair_mask * silh).astype(np.uint8)
        hair_out = (hair_mask * 255).astype(np.uint8)
        silh_out = (silh * 255).astype(np.uint8)

        torch.cuda.empty_cache()

        return PreprocessingResult(
            hair_mask=hair_out,
            silhouette=silh_out,
            foreground=foreground if return_foreground else None,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def to_device(self, device: str | torch.device | None = None) -> None:
        """Move all sub-models onto *device* (default: the configured device)."""
        device = device if device is not None else self.device
        self.bg_remover.to_device(device)
        self.sam_extractor.to_device(device)

    def offload(self) -> None:
        """Park all sub-models in CPU RAM (no-op under ``resident`` policy)."""
        self.bg_remover.offload()
        self.sam_extractor.offload()

    def teardown(self) -> None:
        """Release GPU memory for all sub-models."""
        self.bg_remover.teardown()
        self.sam_extractor.teardown()
        torch.cuda.empty_cache()

    def __del__(self) -> None:
        try:
            self.teardown()
        except Exception:
            pass  # interpreter shutdown: torch/cuda may already be gone
