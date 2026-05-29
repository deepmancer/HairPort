"""Canonical mesh import transforms shared by rendering and projection."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

GLB_IMPORT_ROTATION_X_DEG = 90.0
OBJ_IMPORT_ROTATION_Y_DEG = 180.0


def blender_import_rotation(mesh_extension: str) -> np.ndarray:
    """Return the rotation Blender applies to imported mesh geometry."""
    extension = mesh_extension.lower()
    if extension in {".glb", ".gltf"}:
        return Rotation.from_euler("x", GLB_IMPORT_ROTATION_X_DEG, degrees=True).as_matrix()
    if extension in {".obj", ".ply"}:
        return Rotation.from_euler("y", OBJ_IMPORT_ROTATION_Y_DEG, degrees=True).as_matrix()
    return np.eye(3)
