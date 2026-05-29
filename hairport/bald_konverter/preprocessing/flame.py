"""Optional FLAME-based head segmentation via SHeaP.

All ``sheap`` imports are behind a try / except so the core package can work
without the ``flame`` extra installed.

Install the extra with::

    pip install bald-konverter[flame]
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from hairport.flame_assets import ensure_flame_model_runtime

logger = logging.getLogger(__name__)

_SHEAP_AVAILABLE = False
try:
    from sheap import inference_images_list, load_sheap_model, render_mesh  # noqa: F401
    from sheap.flame_segmentation import create_binary_mask_texture  # noqa: F401
    from sheap.tiny_flame import TinyFlame, pose_components_to_rotmats  # noqa: F401

    _SHEAP_AVAILABLE = True
except ImportError:
    pass


def _require_sheap() -> None:
    if not _SHEAP_AVAILABLE:
        raise ImportError(
            "SHeaP is required for FLAME-based segmentation but is not installed.\n"
            "Install with: pip install bald-konverter[flame]"
        )


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask (e.g., open-mouth region)."""
    from scipy import ndimage

    inverted = 255 - mask
    labeled, num_features = ndimage.label(inverted)
    if num_features > 1:
        sizes = ndimage.sum(inverted, labeled, range(1, num_features + 1))
        largest = int(np.argmax(sizes)) + 1
        mask = np.where(labeled == largest, 0, 255).astype(np.uint8)
    return mask


class FLAMESegmenter:
    """Produce a binary head mask from a portrait image via FLAME fitting.

    Uses the SHeaP model to fit a FLAME mesh and renders the head
    segmentation mask.

    Parameters
    ----------
    flame_model_runtime : str | Path | None
        FLAME runtime model `.pt` path. Auto-generated from source if missing.
    flame_model_source : str | Path | None
        FLAME source `.pkl` path used for runtime conversion.
    flame_masks_path : str | Path | None
        FLAME segmentation mask file path (``FLAME_masks.pkl``).
    flame_eyelids_path : str | Path | None
        FLAME eyelids blendshape `.pt` path.
    model_type : str
        SHeaP model variant (``"expressive"`` or ``"neutral"``).
    device : str | torch.device
        Compute device.
    max_expansion_iterations : int
        Maximum number of crop-expansion iterations when the rendered mask
        touches the crop border.
    """

    def __init__(
        self,
        flame_model_runtime: Union[str, Path, None] = None,
        flame_model_source: Union[str, Path, None] = None,
        flame_masks_path: Union[str, Path, None] = None,
        flame_eyelids_path: Union[str, Path, None] = None,
        flame_dir: Union[str, Path, None] = None,
        model_type: str = "expressive",
        device: Optional[Union[str, torch.device]] = None,
        max_expansion_iterations: int = 2,
    ):
        _require_sheap()

        from hairport.config import get_config

        cfg = get_config()
        if flame_dir is not None:
            flame_dir = Path(flame_dir)
            default_runtime = flame_dir / "parametric_models" / "generic_model.pt"
            default_source = flame_dir / "parametric_models" / "generic_model.pkl"
            default_masks = flame_dir / "vertex_mappings" / "FLAME_masks.pkl"
        else:
            default_runtime = Path(cfg.paths.flame_model_runtime)
            default_source = Path(cfg.paths.flame_model_source)
            default_masks = Path(cfg.paths.flame_masks)

        self.flame_model_source = Path(flame_model_source or default_source)
        self.flame_model_runtime = Path(flame_model_runtime or default_runtime)
        self.flame_masks_path = Path(flame_masks_path or default_masks)
        self.flame_eyelids_path = Path(flame_eyelids_path or cfg.paths.flame_eyelids)
        self.flame_model_runtime = ensure_flame_model_runtime(
            self.flame_model_source,
            self.flame_model_runtime,
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_type = model_type
        self.max_expansion_iterations = max_expansion_iterations

        # Lazy-loaded heavy objects
        self._sheap_model = None
        self._flame_model = None

        self._c2w = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------ #
    # Lazy properties
    # ------------------------------------------------------------------ #

    @property
    def sheap_model(self):
        if self._sheap_model is None:
            self._sheap_model = load_sheap_model(model_type=self.model_type).to(self.device)
        return self._sheap_model

    @property
    def flame_model(self) -> "TinyFlame":
        if self._flame_model is None:
            self._flame_model = TinyFlame(
                self.flame_model_runtime,
                eyelids_ckpt=self.flame_eyelids_path,
            )
        return self._flame_model

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def segment(self, image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Return a binary head mask (uint8, 0/255) for *image*.

        Parameters
        ----------
        image : path, ndarray, or PIL Image
            Input portrait image.

        Returns
        -------
        np.ndarray
            ``(H, W)`` binary mask with head region = 255.
        """
        if isinstance(image, (str, Path)):
            image_array = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image_array = np.array(image.convert("RGB"))
        else:
            image_array = image

        h, w = image_array.shape[:2]
        render_size = max(h, w)

        # Run SHeaP inference (writes temp file, as required by API)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
            tmp_path = Path(tmp.name)
            Image.fromarray(image_array).save(tmp_path, quality=95)

            with torch.no_grad():
                predictions = inference_images_list(
                    model=self.sheap_model,
                    device=self.device,
                    image_paths=[tmp_path],
                )

        # Build FLAME mesh
        verts = self.flame_model(
            shape=predictions["shape_from_facenet"],
            expression=predictions["expr"],
            pose=pose_components_to_rotmats(predictions),
            eyelids=predictions["eyelids"],
            translation=predictions["cam_trans"],
        )

        # Render binary head mask
        mask_verts, mask_faces, mask_colors = create_binary_mask_texture(
            verts[0],
            self.flame_model.faces,
            flame_masks_path=self.flame_masks_path,
        )
        mask_render, _ = render_mesh(
            verts=mask_verts,
            faces=mask_faces,
            c2w=self._c2w,
            img_width=render_size,
            img_height=render_size,
            render_normals=False,
            render_segmentation=True,
            vertex_colors=mask_colors,
            black_background=True,
        )

        mask_gray = mask_render[:, :, 0]
        mask_filled = _fill_mask_holes(mask_gray)

        # Resize to original image dimensions if needed
        if mask_filled.shape[0] != h or mask_filled.shape[1] != w:
            mask_filled = cv2.resize(mask_filled, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_filled = (mask_filled > 127).astype(np.uint8) * 255

        return mask_filled

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def teardown(self) -> None:
        if self._sheap_model is not None:
            del self._sheap_model
            self._sheap_model = None
        if self._flame_model is not None:
            del self._flame_model
            self._flame_model = None
        torch.cuda.empty_cache()

    def __del__(self) -> None:
        self.teardown()
