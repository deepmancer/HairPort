"""Contract tests for non-square framing + VFX compositing (CPU-only).

No GPU / model load: validates the geometric and compositing invariants that
guarantee a seamless, lossless full-frame composite-back.
"""

from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


@unittest.skipUnless(find_spec("cv2") and find_spec("numpy"), "cv2/numpy not installed")
class FramingContractTests(unittest.TestCase):
    def _rng(self):
        import numpy as np
        return np.random.default_rng(0)

    def test_plate_is_square_and_paste_is_identity(self) -> None:
        import numpy as np
        from hairport.bald_konverter.framing import plan_framing

        img = self._rng().integers(0, 255, (300, 500, 3), dtype=np.uint8)  # non-square
        hair = np.zeros((300, 500), np.uint8)
        hair[80:160, 200:300] = 255
        fr = plan_framing(img, hair, crop_scale=1.8, model_size=256)

        plate = fr.extract_native(img)
        self.assertEqual(plate.shape[0], plate.shape[1])  # square
        # Pasting the unmodified plate reproduces the original exactly.
        self.assertTrue(np.array_equal(fr.paste(img, plate), img))

    def test_plate_never_extends_beyond_original_pixels(self) -> None:
        # Hard constraint: the square plate must always lie fully inside the image,
        # so no padding (mirror or edge) is ever introduced — regardless of how the
        # subject fills a non-square frame.
        import numpy as np
        from hairport.bald_konverter.framing import plan_framing

        cases = [
            ((941, 1672, 3), (295, 2, 938, 938)),    # landscape, head fills height
            ((1672, 941, 3), (60, 400, 820, 900)),   # portrait, large subject
            ((941, 1672, 3), (700, 300, 200, 220)),  # small head, off-centre
        ]
        for shape, face_bbox in cases:
            img = self._rng().integers(0, 255, shape, dtype=np.uint8)
            h, w = shape[:2]
            fr = plan_framing(img, hair_mask=None, face_bbox=face_bbox,
                              crop_scale=1.8, model_size=768)
            self.assertEqual(fr.side, fr.extract_native(img).shape[0])
            self.assertLessEqual(fr.side, min(h, w))            # fits the short side
            self.assertGreaterEqual(fr.px, 0)
            self.assertGreaterEqual(fr.py, 0)
            self.assertLessEqual(fr.px + fr.side, w)
            self.assertLessEqual(fr.py + fr.side, h)
            self.assertEqual(fr._pads(), (0, 0, 0, 0))          # never beyond bounds

    def test_plate_near_border_reflect_pads_and_pastes_identity(self) -> None:
        import numpy as np
        from hairport.bald_konverter.framing import Framing

        img = self._rng().integers(0, 255, (200, 200, 3), dtype=np.uint8)
        # Plate pushed past the top-left corner (negative origin).
        fr = Framing(orig_h=200, orig_w=200, px=-40, py=-30, side=160, model_size=128)
        plate = fr.extract_native(img)
        self.assertEqual(plate.shape[:2], (160, 160))
        self.assertTrue(np.array_equal(fr.paste(img, plate), img))

    def test_framing_json_roundtrip(self) -> None:
        from hairport.bald_konverter.framing import Framing

        fr = Framing(orig_h=300, orig_w=500, px=12, py=-8, side=240, model_size=256)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "framing.json"
            fr.to_json(p)
            fr2 = Framing.from_json(p)
        self.assertEqual(fr.to_dict(), fr2.to_dict())

    def test_composite_changes_only_inside_region_and_alpha_border_zero(self) -> None:
        import numpy as np
        from hairport.bald_konverter import compositing

        rng = self._rng()
        orig = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        bald = orig.copy()
        bald[40:120, 90:160] = [200, 180, 170]
        hair = np.zeros((256, 256), np.uint8)
        hair[50:110, 100:150] = 255

        comp, alpha, params = compositing.composite_plate(
            orig, bald, hair, seam_poisson=True, grain_match=False,
        )
        # alpha is zero in the border band → seamless paste guarantee
        self.assertEqual(float(alpha[:6].max()), 0.0)
        self.assertEqual(float(alpha[-6:].max()), 0.0)
        # untouched far region (no hair, no model change) keeps the original
        far = np.zeros((256, 256), bool)
        far[180:240, 20:80] = True
        self.assertTrue(np.array_equal(comp[far], orig[far]))
        self.assertNotIn("color", params)

    def test_matte_extends_through_model_changed_wisp_band_no_halo(self) -> None:
        """The halo fix: hair the segmenter missed (but the model removed) must be
        covered by the matte so no original-hair residue remains."""
        import numpy as np
        from hairport.bald_konverter import compositing

        h = w = 256
        orig = np.full((h, w, 3), 30, np.uint8)            # dark "wall"
        # true hair = a disc; SAM seed under-segments it (smaller disc).
        yy, xx = np.mgrid[0:h, 0:w]
        true_hair = ((yy - 128) ** 2 + (xx - 128) ** 2) <= 60 ** 2
        seed_disc = ((yy - 128) ** 2 + (xx - 128) ** 2) <= 45 ** 2
        orig[true_hair] = [220, 210, 190]                  # blonde hair over wall
        hair_mask = (seed_disc.astype(np.uint8)) * 255     # under-segmented seed

        bald = np.full((h, w, 3), 30, np.uint8)            # model removed ALL hair
        bald[((yy - 128) ** 2 + (xx - 128) ** 2) <= 40 ** 2] = [200, 175, 160]  # small bald scalp

        comp, alpha, _ = compositing.composite_plate(
            orig, bald, hair_mask, seam_poisson=False, grain_match=False,
            extend_band_frac=0.2, extend_diff_threshold=12, feather_px=3,
        )
        # The wisp ring (true hair beyond the seed) must be covered (alpha≈1)…
        ring = true_hair & ~seed_disc
        self.assertGreater(float(alpha[ring].mean()), 0.95)
        # …and must contain NO residual original hair colour in the composite.
        resid = (np.abs(comp[ring].astype(int) - np.array([220, 210, 190])).sum(1) < 60)
        self.assertLess(resid.mean(), 0.02)


if __name__ == "__main__":
    unittest.main()
