"""Shared compatibility helpers for fitting backends."""

from __future__ import annotations

import numpy as np


def ensure_chumpy_numpy_compat() -> None:
    """Restore numpy aliases removed in numpy>=1.24 that chumpy 0.70 imports.

    FLAME / SMPL-X ``.pkl`` files store ``chumpy`` arrays, so unpickling them
    triggers ``import chumpy``, which does ``from numpy import bool, int, ...``
    and crashes on modern numpy.  Call this before importing any code that
    unpickles a FLAME/SMPL pickle.
    """
    compat = {
        "bool": bool, "int": int, "float": float, "complex": complex,
        "object": object, "str": str, "unicode": str,
    }
    for name, value in compat.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask (e.g. eye sockets / open mouth).

    Keeps the largest background component; everything else flips to
    foreground.  Input/output are uint8 with 0 / 255 values.
    """
    from scipy import ndimage

    inverted = 255 - mask
    labeled, num_features = ndimage.label(inverted)
    if num_features > 1:
        sizes = ndimage.sum(inverted, labeled, range(1, num_features + 1))
        largest = int(np.argmax(sizes)) + 1
        mask = np.where(labeled == largest, 0, 255).astype(np.uint8)
    return mask
