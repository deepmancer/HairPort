"""Background removal using BEN2 (Background Extraction Network v2).

Wraps the ``PramaLLC/BEN2`` model to produce foreground-matted RGBA images
and binary silhouette masks.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class BackgroundRemover:
    """Remove backgrounds from portrait images using BEN2.

    Parameters
    ----------
    model_id : str
        Hugging Face model ID for BEN2.
    device : str | torch.device
        Compute device.
    alpha_threshold : float
        Threshold on the alpha channel (0–1) to binarize the silhouette mask.
    """

    def __init__(
        self,
        model_id: str | None = None,
        device: str | torch.device = "cuda",
        alpha_threshold: float = 0.5,
        revision: str | None = None,
    ):
        from ben2 import BEN_Base
        from hairport.config import get_config

        cfg = get_config()
        model_id = model_id or cfg.models.ben2
        revision = revision if revision is not None else cfg.models.ben2_revision
        self.device = torch.device(device)
        self.alpha_threshold = alpha_threshold
        self._model: nn.Module = BEN_Base.from_pretrained(
            model_id, revision=revision
        ).to(self.device)
        self._model.eval()
        logger.info("BackgroundRemover (BEN2) loaded on %s", self.device)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def remove_background(
        self,
        image: Image.Image,
        refine_foreground: bool = False,
    ) -> Tuple[Image.Image, Image.Image]:
        """Extract foreground and produce a binary silhouette mask.

        Parameters
        ----------
        image : PIL.Image.Image
            Input RGB image.
        refine_foreground : bool
            If ``True``, applies BEN2's built-in foreground refinement.

        Returns
        -------
        foreground : PIL.Image.Image
            RGBA image with background removed.
        mask : PIL.Image.Image
            Grayscale binary mask (0 / 255) of the foreground silhouette.
        """
        foreground: Image.Image = self._model.inference(image, refine_foreground=refine_foreground)
        alpha = foreground.getchannel("A")
        mask_np = ((np.array(alpha) / 255.0) > self.alpha_threshold).astype(np.uint8) * 255
        return foreground, Image.fromarray(mask_np)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def to_device(self, device: str | torch.device | None = None) -> None:
        """Move the model onto *device* (default: the configured device)."""
        if hasattr(self, "_model"):
            self._model.to(device if device is not None else self.device)

    def offload(self) -> None:
        """Park the model in CPU RAM (no-op under ``resident`` policy)."""
        from hairport import memory

        if hasattr(self, "_model"):
            memory.offload(self._model)

    def teardown(self) -> None:
        """Free GPU memory held by the model."""
        if hasattr(self, "_model"):
            del self._model
        torch.cuda.empty_cache()

    def __del__(self) -> None:
        try:
            self.teardown()
        except Exception:
            pass  # interpreter shutdown: torch/cuda may already be gone
