"""Regression tests for repaired stage API boundaries."""

from __future__ import annotations

import inspect
import unittest
from importlib.util import find_spec

import numpy as np
from PIL import Image


@unittest.skipUnless(
    find_spec("torch") and find_spec("omegaconf") and find_spec("diffusers"),
    "HairPort stage dependencies not installed",
)
class StageApiContractTests(unittest.TestCase):
    def test_enhancer_batch_helper_accepts_stage_parameters(self) -> None:
        from hairport.view_enhancer import process_view_aligned_folder

        parameters = inspect.signature(process_view_aligned_folder).parameters
        self.assertIn("num_inference_steps", parameters)
        self.assertIn("guidance_scale", parameters)
        self.assertIn("conditioning_phase", parameters)

    def test_face_segmenter_uses_callable_sam_extractor_contract(self) -> None:
        from hairport.fit_lmk.segmentation import FaceSegmenter

        class DummyBackground:
            def remove_background(self, image):
                return image.convert("RGBA"), Image.new("L", image.size, 255)

        class DummySam:
            def __call__(self, image, prompt="hair"):
                self.prompt = prompt
                return Image.new("L", image.size, 255), 1.0

        segmenter = FaceSegmenter.__new__(FaceSegmenter)
        segmenter.bg_remover = DummyBackground()
        segmenter.sam_extractor = DummySam()
        masks = segmenter.extract_masks(Image.new("RGB", (4, 4), "white"))
        self.assertEqual(segmenter.sam_extractor.prompt, "hair")
        self.assertTrue(np.all(masks["hair_mask"] == 1))


if __name__ == "__main__":
    unittest.main()
