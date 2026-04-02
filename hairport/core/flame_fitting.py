"""Canonical FLAME model fitting via SHeaP for the HairPort framework.

Combines the best patterns from the previous three implementations:
- ``hairport/utility/flame_fitting.py`` (clean parameterisation)
- ``bald_konverter/preprocessing/flame.py`` (try/except guarded imports)
- ``utils/flame_fitting.py`` (extra features)

All ``sheap`` imports are behind a try / except so downstream code can
gracefully degrade when SHeaP is not installed.

Usage::

    from hairport.core.flame_fitting import FLAMEFitter

    fitter = FLAMEFitter()  # reads paths from HairPortConfig
    flame_mask, result_dict = fitter.fit_flame(image)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from roma import rotvec_to_rotmat
from scipy import ndimage

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Optional SHeaP imports (guarded)
# --------------------------------------------------------------------------- #

_SHEAP_AVAILABLE = False
try:
    from sheap import inference_images_list, load_sheap_model, render_mesh
    from sheap.flame_segmentation import create_binary_mask_texture
    from sheap.tiny_flame import TinyFlame, pose_components_to_rotmats

    _SHEAP_AVAILABLE = True
except ImportError:
    pass

os.environ["PYOPENGL_PLATFORM"] = "egl"


def _require_sheap() -> None:
    if not _SHEAP_AVAILABLE:
        raise ImportError(
            "SHeaP is required for FLAME fitting but is not installed.\n"
            "Install with:  cd modules/SHeaP && pip install -e ."
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in binary mask (e.g., open mouth region)."""
    inverted = 255 - mask
    labeled, num_features = ndimage.label(inverted)
    if num_features > 1:
        component_sizes = ndimage.sum(inverted, labeled, range(1, num_features + 1))
        largest_component_label = np.argmax(component_sizes) + 1
        largest_component_mask = labeled == largest_component_label
        mask = np.where(largest_component_mask, 0, 255).astype(np.uint8)
    return mask


def _rotmat_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to XYZ Euler angles (radians)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


# --------------------------------------------------------------------------- #
# FLAMEFitter
# --------------------------------------------------------------------------- #

class FLAMEFitter:
    """FLAME model fitting with face cropping, SHeaP inference, and mesh rendering.

    Parameters
    ----------
    flame_dir : Path or str or None
        Directory containing ``generic_model.pt``, ``eyelids.pt``, etc.
        If *None*, resolved from :pydata:`hairport.config`.
    device : torch.device or None
        Compute device.
    model_type : str
        SHeaP model variant: ``"paper"`` or ``"expressive"``.
    padding_ratio : float
        Face-crop padding ratio.
    min_detection_confidence : float
        MediaPipe face-detection confidence threshold.
    """

    def __init__(
        self,
        flame_dir: Union[str, Path, None] = None,
        device: Optional[torch.device] = None,
        model_type: str | None = None,
        padding_ratio: float | None = None,
        min_detection_confidence: float | None = None,
    ):
        _require_sheap()

        from hairport.config import get_config
        cfg = get_config()

        self.flame_dir = Path(flame_dir) if flame_dir else Path(cfg.paths.flame_dir)
        self.device = device or torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model_type = model_type or cfg.flame.model_type
        self.padding_ratio = padding_ratio if padding_ratio is not None else cfg.flame.padding_ratio
        self.min_detection_confidence = (
            min_detection_confidence if min_detection_confidence is not None
            else cfg.flame.detection_confidence
        )

        # Lazy-loaded heavy models
        self._sheap_model = None
        self._flame_model = None
        self._face_detector = None

        # Camera-to-world transform (identity with z-offset)
        self._c2w = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------ #
    # Lazy-loaded properties
    # ------------------------------------------------------------------ #

    @property
    def sheap_model(self):
        if self._sheap_model is None:
            self._sheap_model = load_sheap_model(model_type=self.model_type).to(self.device)
        return self._sheap_model

    @property
    def flame_model(self) -> TinyFlame:
        if self._flame_model is None:
            self._flame_model = TinyFlame(
                self.flame_dir / "generic_model.pt",
                eyelids_ckpt=self.flame_dir / "eyelids.pt",
            )
        return self._flame_model

    @property
    def face_detector(self):
        if self._face_detector is None:
            from hairport.core.facial_landmark_detector import FacialLandmarkDetector

            self._face_detector = FacialLandmarkDetector(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
            )
        return self._face_detector

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _crop_face_with_padding(
        self, image: np.ndarray
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Crop face region from image with specified padding margin."""
        height, width = image.shape[:2]
        bbox = self.face_detector.get_face_bounding_box(image, return_format="xywh")
        if bbox is None:
            return None

        x, y, w, h = bbox
        pad_x = int(w * self.padding_ratio)
        pad_y = int(h * self.padding_ratio)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)

        cropped = image[y1:y2, x1:x2].copy()
        crop_params = {
            "crop_coords": [x1, y1, x2, y2],
            "original_size": [width, height],
            "cropped_size": [cropped.shape[1], cropped.shape[0]],
            "face_bbox": [int(x), int(y), int(w), int(h)],
        }
        return cropped, crop_params

    def _uncrop_mask_to_original(
        self,
        mask: np.ndarray,
        crop_params: Dict[str, Any],
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """Map a mask from cropped image space back to original coords."""
        x1, y1, x2, y2 = crop_params["crop_coords"]
        orig_width, orig_height = crop_params["original_size"]
        crop_width, crop_height = crop_params["cropped_size"]

        mask_h, mask_w = mask.shape[:2]
        if mask_w != crop_width or mask_h != crop_height:
            mask_resized = cv2.resize(
                mask.astype(np.float32), (crop_width, crop_height), interpolation=interpolation
            )
            mask_resized = (mask_resized > 127).astype(np.uint8) * 255
        else:
            mask_resized = mask

        full_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        src_x1 = max(0, -x1) if x1 < 0 else 0
        src_y1 = max(0, -y1) if y1 < 0 else 0
        src_x2 = crop_width - max(0, x2 - orig_width) if x2 > orig_width else crop_width
        src_y2 = crop_height - max(0, y2 - orig_height) if y2 > orig_height else crop_height
        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(orig_width, x2)
        dst_y2 = min(orig_height, y2)
        full_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask_resized[src_y1:src_y2, src_x1:src_x2]
        return full_mask

    def _extract_head_orientation(self, predictions: Dict, frame_idx: int = 0) -> Dict[str, Any]:
        """Extract head orientation as Euler angles and direction vectors."""
        torso_rotvec = predictions["torso_pose"][frame_idx].cpu()
        neck_rotvec = predictions["neck_pose"][frame_idx].cpu()
        torso_rotmat = rotvec_to_rotmat(torso_rotvec.unsqueeze(0))[0]
        neck_rotmat = rotvec_to_rotmat(neck_rotvec.unsqueeze(0))[0]
        head_rotmat = (torso_rotmat @ neck_rotmat).numpy()
        euler_xyz = _rotmat_to_euler_xyz(head_rotmat)
        return {
            "euler_xyz_radians": euler_xyz.tolist(),
            "forward": (-head_rotmat[:, 2]).tolist(),
            "up": head_rotmat[:, 1].tolist(),
            "right": head_rotmat[:, 0].tolist(),
        }

    def _render_flame_mask(self, verts: torch.Tensor, output_size: int = 512) -> np.ndarray:
        """Render binary FLAME mask with holes filled."""
        mask_verts, mask_faces, mask_colors = create_binary_mask_texture(
            verts, self.flame_model.faces, flame_masks_path=self.flame_dir / "FLAME_masks.pkl"
        )
        mask_render, _ = render_mesh(
            verts=mask_verts,
            faces=mask_faces,
            c2w=self._c2w,
            img_width=output_size,
            img_height=output_size,
            render_normals=False,
            render_segmentation=True,
            vertex_colors=mask_colors,
            black_background=True,
        )
        mask_gray = mask_render[:, :, 0]
        return _fill_mask_holes(mask_gray)

    def _extract_flame_parameters(self, predictions: Dict, frame_idx: int = 0) -> Dict[str, Any]:
        """Extract FLAME parameters from SHeaP predictions."""
        return {
            "shape": predictions["shape_from_facenet"][frame_idx].cpu().numpy().tolist(),
            "expression": predictions["expr"][frame_idx].cpu().numpy().tolist(),
            "eyelids": predictions["eyelids"][frame_idx].cpu().numpy().tolist(),
            "translation": predictions["cam_trans"][frame_idx].cpu().numpy().tolist(),
            "torso_pose": predictions["torso_pose"][frame_idx].cpu().numpy().tolist(),
            "neck_pose": predictions["neck_pose"][frame_idx].cpu().numpy().tolist(),
            "jaw_pose": predictions["jaw_pose"][frame_idx].cpu().numpy().tolist(),
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit_flame(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Fit FLAME model to an input image.

        Returns ``(flame_mask, result_dict)`` or ``None`` if no face is detected.
        """
        if isinstance(image, (str, Path)):
            image_array = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            image_array = np.array(image.convert("RGB"))
        else:
            image_array = image

        crop_result = self._crop_face_with_padding(image_array)
        if crop_result is None:
            return None
        cropped_image, crop_params = crop_result

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            Image.fromarray(cropped_image).save(tmp_path, quality=95)

        try:
            with torch.no_grad():
                predictions = inference_images_list(
                    model=self.sheap_model,
                    device=self.device,
                    image_paths=[tmp_path],
                )
        finally:
            tmp_path.unlink(missing_ok=True)

        verts = self.flame_model(
            shape=predictions["shape_from_facenet"],
            expression=predictions["expr"],
            pose=pose_components_to_rotmats(predictions),
            eyelids=predictions["eyelids"],
            translation=predictions["cam_trans"],
        )

        crop_width, crop_height = crop_params["cropped_size"]
        render_size = max(crop_width, crop_height)
        mask_cropped = self._render_flame_mask(verts=verts[0], output_size=render_size)

        if crop_width != crop_height:
            mask_cropped = cv2.resize(mask_cropped, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR)
            mask_cropped = (mask_cropped > 127).astype(np.uint8) * 255

        flame_mask = self._uncrop_mask_to_original(mask_cropped, crop_params)
        flame_parameters = self._extract_flame_parameters(predictions, frame_idx=0)
        head_orientation = self._extract_head_orientation(predictions, frame_idx=0)

        result_dict = {
            "cropped_image": cropped_image,
            "cropping_params": crop_params,
            "flame_parameters": flame_parameters,
            "head_orientation": head_orientation,
        }
        return flame_mask, result_dict

    def fit_flame_batch(
        self,
        image_paths: List[Union[str, Path]],
    ) -> List[Optional[Tuple[np.ndarray, Dict[str, Any]]]]:
        """Fit FLAME model to a batch of images."""
        cropped_images: list[np.ndarray] = []
        crop_params_list: list[Dict] = []
        valid_indices: list[int] = []

        for i, img_path in enumerate(image_paths):
            image_array = np.array(Image.open(img_path).convert("RGB"))
            crop_result = self._crop_face_with_padding(image_array)
            if crop_result is None:
                logger.warning("No face detected in %s, skipping.", img_path)
                continue
            cropped_image, crop_params = crop_result
            cropped_images.append(cropped_image)
            crop_params_list.append(crop_params)
            valid_indices.append(i)

        if not cropped_images:
            return [None] * len(image_paths)

        temp_dir = Path(tempfile.mkdtemp())
        temp_paths: list[Path] = []
        try:
            for i, cropped_img in enumerate(cropped_images):
                temp_path = temp_dir / f"crop_{i}.jpg"
                Image.fromarray(cropped_img).save(temp_path, quality=95)
                temp_paths.append(temp_path)

            with torch.no_grad():
                predictions = inference_images_list(
                    model=self.sheap_model,
                    device=self.device,
                    image_paths=temp_paths,
                )

            verts = self.flame_model(
                shape=predictions["shape_from_facenet"],
                expression=predictions["expr"],
                pose=pose_components_to_rotmats(predictions),
                eyelids=predictions["eyelids"],
                translation=predictions["cam_trans"],
            )

            results: list[Optional[Tuple[np.ndarray, Dict[str, Any]]]] = [None] * len(image_paths)
            for batch_idx, orig_idx in enumerate(valid_indices):
                crop_params = crop_params_list[batch_idx]
                crop_width, crop_height = crop_params["cropped_size"]
                render_size = max(crop_width, crop_height)
                mask_cropped = self._render_flame_mask(verts=verts[batch_idx], output_size=render_size)

                if crop_width != crop_height:
                    mask_cropped = cv2.resize(mask_cropped, (crop_width, crop_height), interpolation=cv2.INTER_LINEAR)
                    mask_cropped = (mask_cropped > 127).astype(np.uint8) * 255

                flame_mask = self._uncrop_mask_to_original(mask_cropped, crop_params)
                flame_parameters = self._extract_flame_parameters(predictions, frame_idx=batch_idx)
                head_orientation = self._extract_head_orientation(predictions, frame_idx=batch_idx)

                results[orig_idx] = (
                    flame_mask,
                    {
                        "cropped_image": cropped_images[batch_idx],
                        "cropping_params": crop_params,
                        "flame_parameters": flame_parameters,
                        "head_orientation": head_orientation,
                    },
                )
            return results
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_overlay(
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.4,
        color: Tuple[int, int, int] = (255, 64, 64),
    ) -> np.ndarray:
        """Create an overlay of the FLAME mask on an image."""
        mask_rgb = np.zeros_like(image)
        mask_rgb[:, :, 0] = (mask > 127).astype(np.uint8) * color[0]
        mask_rgb[:, :, 1] = (mask > 127).astype(np.uint8) * color[1]
        mask_rgb[:, :, 2] = (mask > 127).astype(np.uint8) * color[2]

        overlay = image.copy().astype(np.float32)
        mask_norm = (mask > 127).astype(np.float32)[:, :, np.newaxis]
        overlay = overlay * (1 - alpha * mask_norm) + mask_rgb.astype(np.float32) * alpha * mask_norm
        return np.clip(overlay, 0, 255).astype(np.uint8)


__all__ = ["FLAMEFitter", "_SHEAP_AVAILABLE"]
