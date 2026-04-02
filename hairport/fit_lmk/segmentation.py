"""Face segmentation using SAM 3.1 and BEN2 background removal.

This replaces the old CDGNet-based segmentation (which is no longer maintained)
with the canonical ``hairport.core`` utilities.
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from hairport.core.bg_remover import BackgroundRemover
from hairport.core.sam_extractor import SAMMaskExtractor


class FaceSegmenter:
    """Segment face, hair, and body regions from a portrait image.

    Uses BEN2 for background removal (silhouette) and SAM 3.1 for hair
    segmentation (replacing CDGNet + LIPClass.HAIR).
    """

    def __init__(self, device: str | None = None):
        from hairport.config import get_config
        if device is None:
            device = get_config().device
        self.device = device
        self.bg_remover = BackgroundRemover(device=device)
        self.sam_extractor = SAMMaskExtractor(device=device)

    def extract_masks(self, image_path):
        """Return dict of binary masks: face_mask, hair_mask, bg_mask, body_mask."""
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert("RGB")
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        else:
            image = image_path

        # Background removal → silhouette
        rgba, silh_mask = self.bg_remover.remove_background(image)
        silh = (np.array(silh_mask) > 50).astype(np.uint8)

        # Hair segmentation via SAM 3.1 with "hair" prompt
        hair_pil_mask, _ = self.sam_extractor.extract_mask(image, text_prompt="hair")
        hair_mask = (np.array(hair_pil_mask) > 127).astype(np.uint8)

        # Combine: hair only within silhouette
        hair_mask = (hair_mask * silh).astype(np.uint8)
        face_mask = (silh & ~hair_mask).astype(np.uint8)
        bg_mask = 1 - silh

        return {
            "face_mask": face_mask,
            "hair_mask": hair_mask,
            "bg_mask": bg_mask,
            "body_mask": silh,
        }
