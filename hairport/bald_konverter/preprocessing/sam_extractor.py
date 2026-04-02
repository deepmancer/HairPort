"""Hair / head mask extraction using SAM3 (Segment Anything Model 3).

Wraps ``facebook/sam3`` via the HuggingFace ``transformers`` library for
text-prompted instance segmentation of hair and head regions.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class SAMMaskExtractor:
    """Extract binary masks from images using text-prompted SAM3.

    Parameters
    ----------
    model_id : str
        Hugging Face model ID (default ``facebook/sam3``).
    confidence_threshold : float
        Threshold applied to mask logits.
    detection_threshold : float
        Object detection confidence threshold.
    device : str
        Compute device.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        confidence_threshold: float = 0.35,
        detection_threshold: float = 0.4,
        device: str = "cuda",
    ):
        from ..config.defaults import SAM_CONFIDENCE_THRESHOLD, SAM_DETECTION_THRESHOLD

        confidence_threshold = confidence_threshold if confidence_threshold != 0.35 else SAM_CONFIDENCE_THRESHOLD
        detection_threshold = detection_threshold if detection_threshold != 0.4 else SAM_DETECTION_THRESHOLD
        from transformers import Sam3Model, Sam3Processor

        self.confidence_threshold = confidence_threshold
        self.detection_threshold = detection_threshold
        self.device = device

        self._model = Sam3Model.from_pretrained(model_id).to(self.device)
        self._model.eval()
        self._processor = Sam3Processor.from_pretrained(model_id)
        logger.info("SAMMaskExtractor (SAM3) loaded on %s", self.device)

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
            Text prompt describing the region to segment (e.g. ``"hair"``).

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

        # Progressive squeeze to remove singleton dims (matches original logic)
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
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_processor"):
            del self._processor
        torch.cuda.empty_cache()

    def __del__(self) -> None:
        self.teardown()
