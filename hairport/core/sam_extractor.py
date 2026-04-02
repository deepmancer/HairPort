"""Canonical SAM 3.1 mask extraction for the HairPort framework.

This is the single shared ``SAMMaskExtractor`` for HairPort (outside of
``bald_konverter``, which maintains its own independent copy).

Usage::

    from hairport.core.sam_extractor import SAMMaskExtractor

    extractor = SAMMaskExtractor()
    mask, score = extractor(pil_image, prompt="hair")
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class SAMMaskExtractor:
    """Text-prompted instance segmentation using SAM 3.1.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID.  Defaults to the value in
        :pydata:`hairport.config.SAMConfig`.
    confidence_threshold : float
        Threshold applied to mask logits.
    detection_threshold : float
        Object-detection confidence threshold.
    device : str
        Compute device.
    """

    def __init__(
        self,
        model_id: str | None = None,
        confidence_threshold: float | None = None,
        detection_threshold: float | None = None,
        device: str | None = None,
    ):
        from hairport.config import get_config

        cfg = get_config()
        self.model_id = model_id or cfg.models.sam
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else cfg.sam.confidence_threshold
        )
        self.detection_threshold = (
            detection_threshold if detection_threshold is not None else cfg.sam.detection_threshold
        )
        resolved_device = device if device is not None else cfg.device
        if not torch.cuda.is_available() and "cuda" in resolved_device:
            resolved_device = "cpu"
        self.device = resolved_device

        from transformers import Sam31Processor, Sam31Model

        self._model = Sam31Model.from_pretrained(self.model_id).to(self.device)
        self._model.eval()
        self._processor = Sam31Processor.from_pretrained(self.model_id)
        logger.info("SAMMaskExtractor (SAM 3.1) loaded on %s", self.device)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        image: Image.Image | np.ndarray,
        prompt: str = "head hair",
    ) -> Tuple[Image.Image, float]:
        """Segment *image* with a text *prompt* and return a binary mask.

        Parameters
        ----------
        image : PIL Image or ndarray
            Input RGB image.
        prompt : str
            Text prompt describing the region (e.g. ``"hair"``).

        Returns
        -------
        mask : PIL.Image.Image
            Grayscale binary mask (0 / 255).
        score : float
            Average detection confidence.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Resolve target sizes with fallback
        if "original_sizes" in inputs and inputs["original_sizes"] is not None:
            target_sizes = inputs["original_sizes"].tolist()
        else:
            target_sizes = [(image.height, image.width)]

        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=self.detection_threshold,
            mask_threshold=self.confidence_threshold,
            target_sizes=target_sizes,
        )[0]

        masks = results["masks"].cpu().detach().numpy()
        scores = results["scores"]

        # Handle empty detections
        if masks.shape[0] == 0:
            logger.warning("SAM detected no objects for prompt '%s'", prompt)
            h, w = target_sizes[0]
            return Image.fromarray(np.zeros((h, w), dtype=np.uint8)), 0.0

        # Progressive squeeze to remove singleton dims
        while masks.ndim == 4 and masks.shape[0] == 1:
            masks = masks.squeeze(0)
        while masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        while masks.ndim == 3 and masks.shape[0] == 1:
            masks = masks.squeeze(0)

        # Combine multiple detected masks via logical OR
        if masks.ndim == 3 and masks.shape[0] > 1:
            logger.debug("Combining %d masks with logical OR", masks.shape[0])
            combined = np.any(masks > 0.5, axis=0)
            mask_np = (combined * 255).astype(np.uint8)
            avg_score = float(np.mean(scores.cpu().numpy())) if len(scores) > 0 else 0.0
        elif masks.ndim == 2:
            mask_np = ((masks > 0.5) * 255).astype(np.uint8)
            avg_score = scores[0].item() if len(scores) > 0 else 0.0
        else:
            raise ValueError(f"Unexpected mask shape after processing: {masks.shape}")

        return Image.fromarray(mask_np), avg_score

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def teardown(self) -> None:
        """Release GPU memory."""
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_processor"):
            del self._processor
        torch.cuda.empty_cache()

    def __del__(self) -> None:
        self.teardown()


__all__ = ["SAMMaskExtractor"]
