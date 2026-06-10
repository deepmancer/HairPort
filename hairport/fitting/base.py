"""Backend-agnostic fitting result and backend protocol.

:class:`BodyFitResult` is the single persisted artifact for all fitting
backends (schema v2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import torch

logger = logging.getLogger(__name__)

#: Bump when the on-disk layout of :class:`BodyFitResult` changes.
FIT_RESULT_SCHEMA_VERSION = 2


@dataclass
class BodyFitResult:
    """A single image's fitted geometry (persistable). Stable cross-backend
    contracts: ``head_mask`` (HxW uint8 0/255) and ``head_orientation``
    (``euler_angles_xyz_radians`` + forward/up/right basis); the rest is
    analysis data (params, meshes, camera)."""

    backend: str  # e.g. "pear"
    smplx_params: Dict[str, torch.Tensor] = field(default_factory=dict)  # CPU; empty if head-only
    flame_params: Dict[str, torch.Tensor] = field(default_factory=dict)  # CPU
    camera: Dict[str, Any] = field(default_factory=dict)  # R, T, focal_length, image_size, inv_trans
    vertices: Optional[torch.Tensor] = None       # full posed mesh, e.g. SMPL-X (10475, 3)
    faces: Optional[torch.Tensor] = None
    head_vertices: Optional[torch.Tensor] = None  # FLAME head submesh
    head_faces: Optional[torch.Tensor] = None
    head_mask: Optional[np.ndarray] = None        # FLAME head silhouette, source res
    body_mask: Optional[np.ndarray] = None        # full SMPL-X silhouette, source res
    head_orientation: Dict[str, Any] = field(default_factory=dict)
    image_size: Tuple[int, int] = (0, 0)          # (height, width)
    source: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Plain-types payload (loadable with ``torch.load`` sans hairport)."""
        return {
            "schema_version": FIT_RESULT_SCHEMA_VERSION,
            "backend": self.backend,
            "smplx_params": self.smplx_params,
            "flame_params": self.flame_params,
            "camera": self.camera,
            "vertices": self.vertices,
            "faces": self.faces,
            "head_vertices": self.head_vertices,
            "head_faces": self.head_faces,
            "head_mask": self.head_mask,
            "body_mask": self.body_mask,
            "head_orientation": self.head_orientation,
            "image_size": tuple(self.image_size),
            "source": self.source,
            "extra": self.extra,
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Atomically persist the fit to *path* (``.pt``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.to_dict(), tmp)
        tmp.replace(path)
        logger.info("Saved %s fit → %s", self.backend, path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BodyFitResult":
        """Load a fit written by :meth:`save`."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        version = payload.get("schema_version")
        if version != FIT_RESULT_SCHEMA_VERSION:
            raise ValueError(
                f"{path} has schema version {version}; expected "
                f"{FIT_RESULT_SCHEMA_VERSION}."
            )
        return cls(
            backend=payload.get("backend", "unknown"),
            smplx_params=payload.get("smplx_params", {}),
            flame_params=payload.get("flame_params", {}),
            camera=payload.get("camera", {}),
            vertices=payload.get("vertices"),
            faces=payload.get("faces"),
            head_vertices=payload.get("head_vertices"),
            head_faces=payload.get("head_faces"),
            head_mask=payload.get("head_mask"),
            body_mask=payload.get("body_mask"),
            head_orientation=payload.get("head_orientation", {}),
            image_size=tuple(payload.get("image_size", (0, 0))),
            source=payload.get("source"),
            extra=payload.get("extra", {}),
        )


@runtime_checkable
class FittingBackend(Protocol):
    """Interface every fitting backend implements.

    Lifecycle methods mirror the preprocessing models so backends participate
    in the :mod:`hairport.memory` residency policy.
    """

    def fit(
        self,
        image: Union[str, Path, np.ndarray, "PIL.Image.Image"],  # noqa: F821
        source: Optional[str] = None,
    ) -> BodyFitResult:
        """Fit the parametric model(s) to *image*."""
        ...

    def to_device(self, device: Optional[str] = None) -> None: ...

    def offload(self) -> None: ...

    def teardown(self) -> None: ...
