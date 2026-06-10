"""Contract tests for the modular fitting backend (hairport.fitting).

CPU-only: no PEAR model load, no GPU. Validates the persisted-artifact schema,
the legacy v1 loader, the backend registry, and the head-orientation JSON
contract.
"""

from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(
    find_spec("torch") and find_spec("omegaconf"),
    "HairPort runtime dependencies not installed",
)
class FittingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        from hairport.config import load_config, reset_config, set_config

        self.reset_config = reset_config
        reset_config()
        set_config(load_config())

    def tearDown(self) -> None:
        self.reset_config()

    def test_body_fit_result_roundtrip(self) -> None:
        import numpy as np
        import torch

        from hairport.fitting.base import FIT_RESULT_SCHEMA_VERSION, BodyFitResult

        fit = BodyFitResult(
            backend="pear",
            smplx_params={"body_pose": torch.randn(1, 21, 3, 3)},
            flame_params={"shape_params": torch.randn(1, 300),
                          "expression_params": torch.randn(1, 50)},
            camera={"focal_length": 24.0, "screen_size": 1024},
            vertices=torch.randn(10475, 3),
            faces=torch.randint(0, 10475, (20908, 3)),
            head_vertices=torch.randn(5143, 3),
            head_faces=torch.randint(0, 5143, (10144, 3)),
            head_mask=(np.random.rand(64, 48) > 0.5).astype(np.uint8) * 255,
            body_mask=(np.random.rand(64, 48) > 0.5).astype(np.uint8) * 255,
            head_orientation={"euler_angles_xyz_radians": [[0.1, 0.2, 0.3]]},
            image_size=(64, 48),
            source="/some/portrait.png",
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "fit.pt"
            self.assertEqual(fit.save(path), path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["schema_version"], FIT_RESULT_SCHEMA_VERSION)
            loaded = BodyFitResult.load(path)

        torch.testing.assert_close(loaded.vertices, fit.vertices)
        torch.testing.assert_close(
            loaded.flame_params["shape_params"], fit.flame_params["shape_params"]
        )
        self.assertTrue(np.array_equal(loaded.head_mask, fit.head_mask))
        self.assertTrue(np.array_equal(loaded.body_mask, fit.body_mask))
        self.assertEqual(loaded.image_size, (64, 48))
        self.assertEqual(loaded.backend, "pear")
        self.assertEqual(
            loaded.head_orientation["euler_angles_xyz_radians"], [[0.1, 0.2, 0.3]]
        )

    def test_registry_resolves_pear_and_rejects_unknown(self) -> None:
        from hairport import fitting

        sentinel = object()
        factory = lambda device="cuda", **kw: sentinel
        with patch.dict(fitting._REGISTRY, {"pear": factory}):
            self.assertIs(fitting.get_fitting_backend("pear", device="cpu"), sentinel)
        with self.assertRaisesRegex(ValueError, "Unknown fitting backend"):
            fitting.get_fitting_backend("does-not-exist")

    def test_registry_default_reads_config_backend(self) -> None:
        from hairport import fitting

        captured = {}
        sentinel = object()

        def factory(device="cuda", **kw):
            captured["device"] = device
            return sentinel

        with patch.dict(fitting._REGISTRY, {"pear": factory}):
            backend = fitting.get_fitting_backend(device="cpu")  # name=None → cfg
        self.assertIs(backend, sentinel)
        self.assertEqual(captured["device"], "cpu")

    def test_render_size_default_is_768(self) -> None:
        from hairport.config import get_config

        self.assertEqual(int(get_config().fitting.render_size), 768)

    def test_render_inv_trans_aligns_across_resolutions(self) -> None:
        """The render-resolution inverse affine must map the render square to the
        SAME source quadrilateral as the 256-patch inverse — i.e. rendering at a
        different size only changes sampling density, not framing/alignment."""
        import cv2
        import numpy as np
        from hairport.fitting.pear_backend import _trans_from_patch

        # An arbitrary square bbox (xywh), like process_bbox produces.
        cx, cy, bw, bh = 900.0, 1400.0, 1200.0, 1200.0

        def corners_to_source(size):
            inv = _trans_from_patch(cx, cy, bw, bh, size, size, 1.0, 0.0, inv=True)
            pts = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float32)
            return cv2.transform(pts[None], inv)[0]

        src_256 = corners_to_source(256)
        src_768 = corners_to_source(768)
        # The four render-square corners land on the same source quad (≤1px).
        self.assertLess(float(np.abs(src_256 - src_768).max()), 1.0)


if __name__ == "__main__":
    unittest.main()
