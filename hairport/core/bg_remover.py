"""Canonical background removal using BEN2.

This is the single shared ``BackgroundRemover`` for the HairPort framework
(outside of ``bald_konverter``, which maintains its own independent copy).

Usage::

    from hairport.core.bg_remover import BackgroundRemover

    remover = BackgroundRemover()
    foreground, mask = remover.remove_background(pil_image)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class BackgroundRemover:
    """Remove image backgrounds using BEN2.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID (default from :pydata:`hairport.config`).
    alpha_threshold : float
        Alpha-channel threshold for binary mask generation.
    device : str
        Compute device.
    """

    def __init__(
        self,
        model_id: str | None = None,
        alpha_threshold: float | None = None,
        device: str | None = None,
    ):
        from hairport.config import get_config

        cfg = get_config()
        self.model_id = model_id or cfg.models.ben2
        self.alpha_threshold = alpha_threshold if alpha_threshold is not None else cfg.bg_removal.alpha_threshold
        resolved_device = device if device is not None else cfg.device
        if not torch.cuda.is_available() and "cuda" in resolved_device:
            resolved_device = "cpu"
        self.device = torch.device(resolved_device)
        self._model: nn.Module = self._init_model()

    def _init_model(self) -> nn.Module:
        from ben2 import BEN_Base

        torch.set_float32_matmul_precision("high")
        model = BEN_Base.from_pretrained(self.model_id).to(self.device)
        model.eval()
        logger.info("BackgroundRemover (BEN2) loaded on %s", self.device)
        return model

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def remove_background(
        self,
        image: Image.Image,
        refine_foreground: bool = False,
    ) -> Tuple[Image.Image, Image.Image]:
        """Remove the background from *image*.

        Returns
        -------
        foreground : PIL.Image.Image
            RGBA image with background removed.
        mask : PIL.Image.Image
            Grayscale binary mask (0 / 255).
        """
        foreground = self._model.inference(image, refine_foreground=refine_foreground)
        alpha = foreground.getchannel("A")
        mask = ((np.array(alpha) / 255.0) > self.alpha_threshold).astype(np.uint8) * 255
        return foreground, Image.fromarray(mask)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def teardown(self) -> None:
        """Release GPU memory."""
        if hasattr(self, "_model"):
            del self._model
        torch.cuda.empty_cache()

    def __del__(self) -> None:
        self.teardown()


__all__ = ["BackgroundRemover"]
