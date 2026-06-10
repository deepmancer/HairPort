"""Head-centric square plate framing for non-square bald conversion.

A square *plate* is cropped around the head, the model runs on it, and the result
is composited back into the full-resolution original (see :mod:`compositing`).
The plate rect may extend past the image bounds: RGB is reflect-padded, masks are
zero-padded, and ``paste`` writes back only the in-image portion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np


def _to_np(image) -> np.ndarray:
    from PIL import Image

    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    return image


@dataclass
class Framing:
    """Geometry of a square head plate (top-left ``px,py``, side ``side``) in original coords."""

    orig_h: int
    orig_w: int
    px: int
    py: int
    side: int
    model_size: int  # model resolution the plate is prepared for (metadata)

    def _pads(self) -> Tuple[int, int, int, int]:
        """Pad amounts (left, top, right, bottom) needed to contain the plate."""
        left = max(0, -self.px)
        top = max(0, -self.py)
        right = max(0, self.px + self.side - self.orig_w)
        bottom = max(0, self.py + self.side - self.orig_h)
        return left, top, right, bottom

    def extract_native(self, image, border_mode: str = "reflect", fill: int = 0) -> np.ndarray:
        """``side×side`` native plate; ``border_mode`` ``reflect`` (RGB) or ``constant`` (masks)."""
        arr = _to_np(image)
        left, top, right, bottom = self._pads()
        if border_mode == "reflect":
            padded = cv2.copyMakeBorder(arr, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        else:
            padded = cv2.copyMakeBorder(
                arr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill
            )
        y0, x0 = self.py + top, self.px + left
        return padded[y0 : y0 + self.side, x0 : x0 + self.side].copy()

    def map_mask_into_plate(self, mask: np.ndarray) -> np.ndarray:
        """Map a full-res original-coords mask into native plate coords (zero-padded)."""
        return self.extract_native(mask, border_mode="constant", fill=0)

    def _intersection(self) -> Tuple[int, int, int, int, int, int, int, int]:
        """In-image rect (ox0,oy0,ox1,oy1) and matching plate rect (sx0,sy0,sx1,sy1)."""
        ox0, oy0 = max(0, self.px), max(0, self.py)
        ox1 = min(self.orig_w, self.px + self.side)
        oy1 = min(self.orig_h, self.py + self.side)
        sx0, sy0 = ox0 - self.px, oy0 - self.py
        sx1, sy1 = sx0 + (ox1 - ox0), sy0 + (oy1 - oy0)
        return ox0, oy0, ox1, oy1, sx0, sy0, sx1, sy1

    def paste(self, original, plate_native: np.ndarray) -> np.ndarray:
        """Paste the in-image portion of the ``side×side`` plate into a copy of *original*."""
        out = _to_np(original).copy()
        ox0, oy0, ox1, oy1, sx0, sy0, sx1, sy1 = self._intersection()
        out[oy0:oy1, ox0:ox1] = plate_native[sy0:sy1, sx0:sx1]
        return out

    def plate_to_original(self, plate_native: np.ndarray, fill: int = 0) -> np.ndarray:
        """Place a plate-space array onto a full-size canvas (for original-frame masks)."""
        if plate_native.ndim == 2:
            canvas = np.full((self.orig_h, self.orig_w), fill, dtype=plate_native.dtype)
        else:
            canvas = np.full(
                (self.orig_h, self.orig_w, plate_native.shape[2]), fill, dtype=plate_native.dtype
            )
        ox0, oy0, ox1, oy1, sx0, sy0, sx1, sy1 = self._intersection()
        canvas[oy0:oy1, ox0:ox1] = plate_native[sy0:sy1, sx0:sx1]
        return canvas

    def to_json(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Framing":
        return cls(**{k: int(v) for k, v in d.items()})

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Framing":
        return cls.from_dict(json.loads(Path(path).read_text()))


def plan_framing(
    image,
    hair_mask: np.ndarray,
    face_bbox: Tuple[int, int, int, int] | None = None,
    foreground_mask: np.ndarray | None = None,
    crop_scale: float = 1.8,
    model_size: int = 768,
) -> Framing:
    """Plan a square plate = ``crop_scale × max(dims)`` around the head (hair ∪ face bbox).

    Falls back to the upper foreground silhouette, then the whole frame, if no
    hair/face is found.
    """
    arr = _to_np(image)
    h, w = arr.shape[:2]

    boxes = []
    if hair_mask is not None and hair_mask.any():
        ys, xs = np.where(hair_mask > 0)
        boxes.append((xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    if face_bbox is not None:
        boxes.append(tuple(int(v) for v in face_bbox))

    if boxes:
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
    elif foreground_mask is not None and foreground_mask.any():
        ys, xs = np.where(foreground_mask > 0)
        x0, x1 = xs.min(), xs.max() + 1
        y0 = ys.min()
        y1 = y0 + int(0.6 * (ys.max() - ys.min() + 1))  # upper silhouette ≈ head
    else:
        x0, y0, x1, y1 = 0, 0, w, h

    bw, bh = x1 - x0, y1 - y0
    cx, cy = x0 + bw / 2.0, y0 + bh / 2.0
    side = max(int(round(crop_scale * max(bw, bh))), 16)
    px = int(round(cx - side / 2.0))
    py = int(round(cy - side / 2.0))
    return Framing(orig_h=h, orig_w=w, px=px, py=py, side=side, model_size=model_size)
