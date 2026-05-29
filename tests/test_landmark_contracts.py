"""Regression tests for supported landmark-projection behavior."""

from __future__ import annotations

import unittest
from importlib.util import find_spec

import numpy as np


@unittest.skipUnless(find_spec("torch") and find_spec("scipy"), "Landmark runtime dependencies not installed")
class LandmarkContractTests(unittest.TestCase):
    def test_glb_import_transform_is_renderer_consistent(self) -> None:
        from hairport.fit_lmk.transforms import GLB_IMPORT_ROTATION_X_DEG, blender_import_rotation

        matrix = blender_import_rotation(".glb")
        self.assertEqual(GLB_IMPORT_ROTATION_X_DEG, 90.0)
        np.testing.assert_allclose(matrix @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
