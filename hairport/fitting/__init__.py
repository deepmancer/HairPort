"""Modular human/head fitting backends.

A backend recovers SMPL-X/FLAME geometry from an image as a backend-agnostic
:class:`BodyFitResult`. Resolve via :func:`get_fitting_backend` (keyed by
``cfg.fitting.backend``); only ``"pear"`` is registered today.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from .base import BodyFitResult, FittingBackend

__all__ = [
    "BodyFitResult",
    "FittingBackend",
    "get_fitting_backend",
    "register_backend",
]

# name -> factory(device, **kwargs) -> FittingBackend
_REGISTRY: Dict[str, Callable[..., FittingBackend]] = {}


def register_backend(name: str, factory: Callable[..., FittingBackend]) -> None:
    """Register a fitting-backend factory under *name*."""
    _REGISTRY[name] = factory


def _pear_factory(device: str = "cuda", **kwargs) -> FittingBackend:
    # Imported lazily: PEAR pulls in heavy deps (pytorch3d, ultralytics) that
    # should not be required to merely import hairport.fitting.
    from .pear_backend import PearFittingBackend

    return PearFittingBackend(device=device, **kwargs)


register_backend("pear", _pear_factory)


def get_fitting_backend(
    name: Optional[str] = None,
    device: str = "cuda",
    **kwargs,
) -> FittingBackend:
    """Instantiate a fitting backend (``name`` defaults to ``cfg.fitting.backend``)."""
    if name is None:
        try:
            from hairport.config import get_config

            name = str(get_config().fitting.backend)
        except Exception:
            name = "pear"
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown fitting backend {name!r}; registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](device=device, **kwargs)
