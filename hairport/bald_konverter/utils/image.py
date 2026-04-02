"""Image manipulation helpers for grid assembly, cropping, and resizing."""

from __future__ import annotations

import numpy as np
from PIL import Image


def resize_to_square(image: Image.Image, size: int) -> Image.Image:
    """Resize *image* to (*size* × *size*) using Lanczos resampling."""
    return image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


# --------------------------------------------------------------------------- #
# Two-panel helpers (wo_seg mode)
# --------------------------------------------------------------------------- #

def create_two_panel(left: Image.Image, right: Image.Image) -> Image.Image:
    """Create a side-by-side ``[LEFT | RIGHT]`` image.

    Both panels are resized to the same height; the combined width is doubled.
    """
    assert left.size == right.size, "Both panels must have the same size."
    w, h = left.size
    combined = Image.new("RGB", (w * 2, h))
    combined.paste(left, (0, 0))
    combined.paste(right, (w, 0))
    return combined


def crop_right_half(image: Image.Image) -> Image.Image:
    """Return the right half of *image*."""
    w, h = image.size
    return image.crop((w // 2, 0, w, h))


# --------------------------------------------------------------------------- #
# Four-panel helpers (w_seg mode)
# --------------------------------------------------------------------------- #

def create_four_panel(
    top_left: Image.Image,
    top_right: Image.Image,
    bottom_left: Image.Image,
    bottom_right: Image.Image,
) -> Image.Image:
    """Assemble a 2×2 grid from four equal-sized panels.

    Layout::

        ┌────────────┬────────────┐
        │  top_left   │  top_right  │
        ├────────────┼────────────┤
        │ bottom_left │ bottom_right│
        └────────────┴────────────┘

    All panels are expected to share the same size.
    """
    w, h = top_left.size
    grid = Image.new("RGB", (w * 2, h * 2))
    grid.paste(top_left, (0, 0))
    grid.paste(top_right, (w, 0))
    grid.paste(bottom_left, (0, h))
    grid.paste(bottom_right, (w, h))
    return grid


def crop_bottom_right_quadrant(image: Image.Image) -> Image.Image:
    """Return the bottom-right quadrant of a 2×2 grid image."""
    w, h = image.size
    return image.crop((w // 2, h // 2, w, h))


# --------------------------------------------------------------------------- #
# Segmentation visualisation
# --------------------------------------------------------------------------- #

def create_combined_seg_image(
    hair_mask: np.ndarray,
    body_mask: np.ndarray,
    size: int | None = None,
) -> Image.Image:
    """Render a segmentation visualization: hair = red, body = green.

    Parameters
    ----------
    hair_mask : np.ndarray
        Binary uint8 mask (0 or 255) for hair.
    body_mask : np.ndarray
        Binary uint8 mask (0 or 255) for body/skin.
    size : int, optional
        If given, both masks are resized to (*size* × *size*) first.

    Returns
    -------
    PIL.Image.Image
        RGB image with red (hair), green (body), and black (background).
    """
    import cv2

    if size is not None:
        hair_mask = cv2.resize(hair_mask, (size, size), interpolation=cv2.INTER_NEAREST)
        body_mask = cv2.resize(body_mask, (size, size), interpolation=cv2.INTER_NEAREST)

    h, w = hair_mask.shape[:2]
    combined = np.zeros((h, w, 3), dtype=np.uint8)

    # Body (non-hair) first → green
    body_only = (body_mask > 0) & (hair_mask == 0)
    combined[body_only] = [0, 255, 0]

    # Hair → red (overrides body where they overlap)
    combined[hair_mask > 0] = [255, 0, 0]

    return Image.fromarray(combined)


def create_body_green_image(
    body_mask: np.ndarray,
    size: int | None = None,
) -> Image.Image:
    """Render *body_mask* as green-on-black RGB."""
    import cv2

    if size is not None:
        body_mask = cv2.resize(body_mask, (size, size), interpolation=cv2.INTER_NEAREST)

    h, w = body_mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[body_mask > 0] = [0, 255, 0]
    return Image.fromarray(rgb)


# --------------------------------------------------------------------------- #
# Inpainting mask tensors
# --------------------------------------------------------------------------- #

def make_right_half_mask(width: int, height: int):
    """Return a ``(1, H, W)`` float Tensor masking the right half."""
    import torch

    mask = torch.zeros(1, height, width)
    mask[:, :, width // 2 :] = 1.0
    return mask


def make_bottom_right_mask(width: int, height: int):
    """Return a ``(1, H, W)`` float Tensor masking the bottom-right quadrant."""
    import torch

    mask = torch.zeros(1, height, width)
    mask[:, height // 2 :, width // 2 :] = 1.0
    return mask
