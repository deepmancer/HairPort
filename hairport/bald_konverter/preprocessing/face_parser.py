"""Segformer-based face parsing for neck detection and body mask computation.

Uses the ``jonathandinu/face-parsing`` model (19 classes) to locate the neck
position, which is used to split the silhouette into head vs. body regions.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..config.segmentation import SegformerClass

logger = logging.getLogger(__name__)


class FaceParser:
    """Detect face components using Segformer for neck-based body mask cropping.

    Parameters
    ----------
    model_id : str
        Hugging Face model ID for the Segformer face parser.
    device : str | torch.device
        Compute device.
    """

    def __init__(
        self,
        model_id: str = "jonathandinu/face-parsing",
        device: str | torch.device = "cuda",
    ):
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        self.device = torch.device(device)
        self._processor = SegformerImageProcessor.from_pretrained(model_id)
        self._model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self._model.to(self.device).eval()
        logger.info("FaceParser (Segformer) loaded on %s", self.device)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def parse(self, image: Image.Image) -> np.ndarray:
        """Run face parsing on *image*.

        Parameters
        ----------
        image : PIL.Image.Image
            Input RGB image.

        Returns
        -------
        labels : np.ndarray
            ``(H, W)`` uint8 array of per-pixel :class:`SegformerClass` IDs.
        """
        image = image.convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self._model(**inputs).logits
            upsampled = F.interpolate(
                logits,
                size=image.size[::-1],  # (H, W)
                mode="bilinear",
                align_corners=True,
            )
            labels = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        return labels

    def find_neck_y(self, labels: np.ndarray, image_height: int) -> int:
        """Return the y-coordinate of the top of the neck region.

        Falls back to chin / lower-lip detection, and ultimately to 65 % of
        the image height if no anatomical landmark is found.

        Parameters
        ----------
        labels : np.ndarray
            Segformer parsing output.
        image_height : int
            Height of the original image (for fallback computation).

        Returns
        -------
        int
            The topmost y-coordinate of the neck region.
        """
        neck_mask = (
            (labels == int(SegformerClass.NECK))
            | (labels == int(SegformerClass.NECK_L))
        )

        if np.any(neck_mask):
            return int(np.where(neck_mask)[0].min())

        # Fallback: chin / lip bottom + 3 %
        for cls in (SegformerClass.L_LIP, SegformerClass.U_LIP, SegformerClass.MOUTH):
            region = labels == int(cls)
            if np.any(region):
                return int(np.where(region)[0].max()) + int(0.03 * image_height)

        # Final fallback
        return int(0.65 * image_height)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def teardown(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_processor"):
            del self._processor
        torch.cuda.empty_cache()

    def __del__(self) -> None:
        self.teardown()
