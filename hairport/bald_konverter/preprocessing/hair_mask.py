"""End-to-end hair / body mask computation pipeline.

Orchestrates :class:`BackgroundRemover`, :class:`SAMMaskExtractor`, and
:class:`FaceParser` to produce hair masks and body masks from portrait images.
Optionally uses :class:`FLAMESegmenter` for FLAME-based head masks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .background import BackgroundRemover
from .face_parser import FaceParser
from .sam_extractor import SAMMaskExtractor

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Container for all masks produced by :class:`HairMaskPipeline`."""

    hair_mask: np.ndarray
    """Binary uint8 mask (0/255) of hair."""

    body_mask: np.ndarray
    """Binary uint8 mask (0/255) of body (neck-down silhouette)."""

    foreground: Optional[Image.Image] = None
    """RGBA foreground image (background removed)."""

    segformer_labels: Optional[np.ndarray] = None
    """Per-pixel Segformer class labels (uint8)."""


class HairMaskPipeline:
    """Compute hair and body masks from a single portrait image.

    Pipeline steps:

    1. **Foreground matte** (BEN2) → binary silhouette
    2. **Hair mask** (SAM3, text prompt ``"hair"``)
    3. **Face parsing** (Segformer) → neck-position detection
    4. **Body mask** = silhouette with everything above neck zeroed-out

    Parameters
    ----------
    device : str | torch.device
        Compute device for all sub-models.
    sam_confidence : float
        Confidence threshold for the SAM hair extractor.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        sam_confidence: float = 0.25,
    ):
        self.device = str(device)
        self.bg_remover = BackgroundRemover(device=self.device, alpha_threshold=0.5)
        self.sam_extractor = SAMMaskExtractor(
            confidence_threshold=sam_confidence,
            device=self.device,
        )
        self.face_parser = FaceParser(device=self.device)
        logger.info("HairMaskPipeline initialised on %s", self.device)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def preprocess(
        self,
        image: Image.Image | np.ndarray,
        return_foreground: bool = False,
        return_segformer: bool = False,
    ) -> PreprocessingResult:
        """Compute hair and body masks for *image*.

        Parameters
        ----------
        image : PIL Image or ndarray
            Input portrait image.
        return_foreground : bool
            If ``True``, include the RGBA foreground in the result.
        return_segformer : bool
            If ``True``, include the raw Segformer label map in the result.

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

        # 3) Segformer face parsing for neck detection ----------------------------
        segf_labels = self.face_parser.parse(rgb)
        neck_y = self.face_parser.find_neck_y(segf_labels, silh.shape[0])

        # 4) Body mask = silhouette below neck ------------------------------------
        body = silh.copy()
        body[:neck_y, :] = 0

        # Dilate to fill thin neck / shoulder gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        body = cv2.dilate(body, kernel, iterations=1)

        # Scale to 0/255 for consistency
        hair_out = (hair_mask * 255).astype(np.uint8)
        body_out = (body * 255).astype(np.uint8)

        torch.cuda.empty_cache()

        return PreprocessingResult(
            hair_mask=hair_out,
            body_mask=body_out,
            foreground=foreground if return_foreground else None,
            segformer_labels=segf_labels if return_segformer else None,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def teardown(self) -> None:
        """Release GPU memory for all sub-models."""
        self.bg_remover.teardown()
        self.sam_extractor.teardown()
        self.face_parser.teardown()
        torch.cuda.empty_cache()

    def __del__(self) -> None:
        self.teardown()
