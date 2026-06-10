"""Head-orientation computation over the fitting-backend registry.

Produces the stable head_orientation JSON contract consumed by landmark_3d and
view_aligner (backend-agnostic).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

_ZERO_ORIENTATION: Dict[str, Any] = {
    "euler_angles_xyz_radians": [[0.0, 0.0, 0.0]],
    "forward": [0.0, 0.0, -1.0],
    "up": [0.0, 1.0, 0.0],
    "right": [1.0, 0.0, 0.0],
}


def compute_head_orientation(
    image_path: Union[str, Path],
    cache_dir: Union[str, Path, None] = None,
    backend: Optional[Any] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Compute head orientation for a portrait image, with optional JSON caching.

    Parameters
    ----------
    image_path : str | Path
        Path to the input portrait image.
    cache_dir : str | Path | None
        If given, ``head_orientation.json`` is read/written here.
    backend : FittingBackend | None
        Reusable fitting backend.  One is created (and kept alive) when
        ``None`` is passed.
    force : bool
        Re-compute even when a cached file exists.

    Returns
    -------
    dict
        ``{"euler_angles_xyz_radians": [[x, y, z]], "forward", "up", "right"}``
        — the stable head-orientation schema.
    """
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "head_orientation.json"
        if not force and cache_path.exists():
            with open(cache_path, "r") as f:
                return json.load(f)

    if backend is None:
        from hairport.fitting import get_fitting_backend

        backend = get_fitting_backend()

    try:
        fit = backend.fit(image_path)
        orientation = fit.head_orientation or _ZERO_ORIENTATION
    except Exception:
        logger.warning(
            "Head fitting failed for %s — returning zero orientation.",
            image_path, exc_info=True,
        )
        orientation = _ZERO_ORIENTATION

    if cache_path is not None:
        with open(cache_path, "w") as f:
            json.dump(orientation, f, indent=2)
        logger.info("Cached head orientation → %s", cache_path)

    return orientation
