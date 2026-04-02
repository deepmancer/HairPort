"""Shared mesh rotation and coordinate-system transform utilities.

Extracted from ``hairport/postprocess.py`` and ``hairport/postprocess_shape_mesh.py``
which both contained identical copies of these helpers.

Usage::

    from hairport.core.mesh_utils import (
        apply_rotation,
        apply_inverse_rotation,
        rotate_glb_mesh,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


# ------------------------------------------------------------------ #
# Low-level helpers
# ------------------------------------------------------------------ #

def rotate_mesh_vertices(mesh: trimesh.Trimesh, rotation_matrix: np.ndarray) -> trimesh.Trimesh:
    """Apply a 3×3 rotation matrix to *mesh* vertices and normals around the mesh centre."""
    mesh = mesh.copy()
    centre = mesh.vertices.mean(axis=0)
    mesh.vertices = (mesh.vertices - centre) @ rotation_matrix.T + centre

    if hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None:
        mesh.vertex_normals = mesh.vertex_normals @ rotation_matrix.T
    if hasattr(mesh, "face_normals") and mesh.face_normals is not None:
        mesh.face_normals = mesh.face_normals @ rotation_matrix.T

    return mesh


def apply_inverse_rotation(mesh: trimesh.Trimesh, euler_angles_rad) -> trimesh.Trimesh:
    """Apply the *inverse* of the given XYZ Euler rotation to *mesh*."""
    rot = Rotation.from_euler("xyz", euler_angles_rad)
    return rotate_mesh_vertices(mesh, rot.inv().as_matrix())


def apply_rotation(mesh: trimesh.Trimesh, euler_angles_rad) -> trimesh.Trimesh:
    """Apply the given XYZ Euler rotation to *mesh*."""
    rot = Rotation.from_euler("xyz", euler_angles_rad)
    return rotate_mesh_vertices(mesh, rot.as_matrix())


def align_target_to_source_view(
    target_mesh: trimesh.Trimesh,
    target_euler_rad,
    source_euler_rad,
) -> trimesh.Trimesh:
    """Align *target_mesh* from its own view to *source* view via frontal."""
    frontal = apply_inverse_rotation(target_mesh, target_euler_rad)
    return apply_rotation(frontal, source_euler_rad)


# ------------------------------------------------------------------ #
# GLB ↔ target coordinate-system transforms
# ------------------------------------------------------------------ #

def glb_to_target(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Transform mesh from GLB coord system to target (90° X rotation)."""
    return rotate_mesh_vertices(mesh, Rotation.from_euler("x", np.pi / 2).as_matrix())


def target_to_glb(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Transform mesh from target coord system back to GLB (−90° X rotation)."""
    return rotate_mesh_vertices(mesh, Rotation.from_euler("x", -np.pi / 2).as_matrix())


# Backward-compatible aliases used by postprocess scripts
apply_glb_to_target_transform = glb_to_target
apply_target_to_glb_transform = target_to_glb


# ------------------------------------------------------------------ #
# High-level GLB rotate + normalise
# ------------------------------------------------------------------ #

# Default extent for landmark bounding-box normalisation (metres).
TARGET_LANDMARK_EXTENT: float = 0.4


def rotate_glb_mesh(
    input_glb_path: str | Path,
    output_glb_path: str | Path | None = None,
    euler_angles_rad: list | None = None,
    rotate_fn: Callable = apply_inverse_rotation,
    to_normalize_vertice_ids: np.ndarray | None = None,
    target_landmark_extent: float | None = None,
) -> Tuple[trimesh.Scene, Optional[trimesh.Trimesh], Optional[np.ndarray]]:
    """Rotate a GLB mesh and optionally save the result.

    Returns
    -------
    result_scene
        The processed :class:`trimesh.Scene`.
    frontalized_mesh
        The intermediate mesh after rotation (in target coord space).
    final_landmark_coords
        3-D landmark coords after scale/centre (or ``None``).
    """
    if euler_angles_rad is None:
        euler_angles_rad = [0.0, 0.0, 0.0]

    input_path = Path(input_glb_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input GLB file does not exist: {input_path}")

    if output_glb_path is not None:
        output_path = Path(output_glb_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    loaded = trimesh.load(str(input_path), force="scene")
    final_landmark_coords = None
    frontalized_mesh = None

    def _normalise(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, Optional[np.ndarray]]:
        """Scale and centre *mesh* via landmark vertices."""
        if to_normalize_vertice_ids is None:
            return mesh, None
        verts = mesh.vertices.copy()
        lm = verts[to_normalize_vertice_ids]
        if target_landmark_extent is not None:
            extent = (lm.max(axis=0) - lm.min(axis=0)).max()
            if extent > 0:
                scale = target_landmark_extent / extent
                verts *= scale
                lm = verts[to_normalize_vertice_ids]
        verts -= lm.mean(axis=0)
        mesh.vertices = verts
        return mesh, verts[to_normalize_vertice_ids]

    if isinstance(loaded, trimesh.Scene):
        for name, mesh in loaded.geometry.items():
            if isinstance(mesh, trimesh.Trimesh):
                mesh_target = glb_to_target(mesh)
                frontalized_mesh = rotate_fn(mesh_target, euler_angles_rad)
                mesh_glb = target_to_glb(frontalized_mesh)
                mesh_glb, final_landmark_coords = _normalise(mesh_glb)
                loaded.geometry[name] = mesh_glb
        result_scene = loaded
    elif isinstance(loaded, trimesh.Trimesh):
        mesh_target = glb_to_target(loaded)
        frontalized_mesh = rotate_fn(mesh_target, euler_angles_rad)
        result_mesh = target_to_glb(frontalized_mesh)
        result_mesh, final_landmark_coords = _normalise(result_mesh)
        result_scene = trimesh.Scene(result_mesh)
    else:
        raise ValueError(f"Unexpected type from trimesh.load: {type(loaded)}")

    if output_glb_path is not None:
        result_scene.export(str(output_path), file_type="glb")

    return result_scene, frontalized_mesh, final_landmark_coords


__all__ = [
    "rotate_mesh_vertices",
    "apply_inverse_rotation",
    "apply_rotation",
    "align_target_to_source_view",
    "glb_to_target",
    "target_to_glb",
    "apply_glb_to_target_transform",
    "apply_target_to_glb_transform",
    "rotate_glb_mesh",
    "TARGET_LANDMARK_EXTENT",
]
