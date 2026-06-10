"""PEAR-based fitting backend (SMPL-X body + FLAME head EHM recovery).

Wraps the vendored PEAR submodule (``modules/PEAR``) — the EHM_v2 model plus
its ViTPose backbone + SMPL-X transformer head — behind HairPort's
:class:`~hairport.fitting.base.FittingBackend` contract.

PEAR uses CWD-relative asset/config paths and imports its own ``models``/
``utils`` top-level packages, so all PEAR calls happen inside
:meth:`PearFittingBackend._pear_context` (sys.path + chdir into the submodule).

Reference inference flow reproduced from ``modules/PEAR/inference_images.py``:
detect person bbox → aspect-preserving crop to 256×256 → EHM forward
(``pd_cam``, ``body_param``, ``flame_param``) → pose the fused SMPL-X+FLAME
mesh → render the FLAME head submesh with PEAR's GS camera → affine-warp the
silhouette back to the source frame.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from .base import BodyFitResult
from .compat import ensure_chumpy_numpy_compat, fill_mask_holes

logger = logging.getLogger(__name__)

# PEAR's GS camera convention (inference_images.py:build_cameras_kwargs).
_PEAR_FOCAL = 24.0
# Model detection-patch size (EHM ViT input — trained at 256, do not change).
_CROP = 256
# Default SMPL-X silhouette render resolution (overridable via cfg.fitting.render_size).
_DEFAULT_RENDER_SIZE = 768


def _euler_and_basis_from_rotmat(R: np.ndarray) -> Dict[str, Any]:
    """Euler angles (XYZ) + basis vectors for the head_orientation JSON contract."""
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return {
        "euler_angles_xyz_radians": [[float(x), float(y), float(z)]],
        "forward": (-R[:, 2]).tolist(),
        "up": R[:, 1].tolist(),
        "right": R[:, 0].tolist(),
    }


def _kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Optimal rotation mapping *src* onto *dst* (both (N,3), mean-centered)."""
    src = src - src.mean(0)
    dst = dst - dst.mean(0)
    H = src.T @ dst
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


class PearFittingBackend:
    """Fit SMPL-X body + FLAME head to a single image using PEAR EHM."""

    def __init__(
        self,
        device: str = "cuda",
        pear_module: Optional[Union[str, Path]] = None,
        repo_id: Optional[str] = None,
        checkpoint: Optional[str] = None,
        detector_weights: Optional[str] = None,
        matting: Optional[bool] = None,
        render_size: Optional[int] = None,
    ):
        from hairport.config import get_config

        cfg = get_config()
        fcfg = getattr(cfg, "fitting", None)
        self.device = device
        self.render_size = int(
            render_size if render_size is not None
            else getattr(fcfg, "render_size", _DEFAULT_RENDER_SIZE)
        )
        self.pear_root = Path(
            pear_module or (getattr(fcfg, "pear_module", None) or "modules/PEAR")
        ).resolve()
        self.repo_id = repo_id or getattr(fcfg, "pear_repo_id", "BestWJH/PEAR_models")
        self.checkpoint = checkpoint or getattr(
            fcfg, "pear_checkpoint", "ehm_model_stage1.pt"
        )
        self.detector_weights = detector_weights or getattr(
            fcfg, "detector_weights", "model_zoo/yolov8x.pt"
        )
        self.matting = matting if matting is not None else bool(
            getattr(fcfg, "matting", True)
        )
        self.config_name = getattr(fcfg, "pear_config", "infer")

        # Lazy-loaded heavy objects
        self._ehm_model = None     # Ehm_Pipeline (backbone + head)
        self._ehm = None           # EHM_v2 (SMPL-X + FLAME parametric model)
        self._renderer = None      # PEAR Renderer2
        self._detector = None      # YOLO
        self._bg_remover = None    # HairPort BEN2 matting
        self._head_idx = None      # smplx -> flame vertex indices
        self._head_faces = None    # FLAME head faces

    # ------------------------------------------------------------------ #
    # PEAR import / CWD context
    # ------------------------------------------------------------------ #

    @contextlib.contextmanager
    def _pear_context(self):
        """Make PEAR's top-level packages importable and its rel-paths resolve."""
        ensure_chumpy_numpy_compat()
        root = str(self.pear_root)
        prev_cwd = os.getcwd()
        added = root not in sys.path
        if added:
            sys.path.insert(0, root)
        try:
            os.chdir(root)
            yield
        finally:
            os.chdir(prev_cwd)
            if added and root in sys.path:
                sys.path.remove(root)

    # ------------------------------------------------------------------ #
    # Lazy loaders
    # ------------------------------------------------------------------ #

    def _ensure_models(self):
        if self._ehm_model is not None:
            return
        from huggingface_hub import hf_hub_download

        with self._pear_context():
            from models.modules.ehm import EHM_v2
            from models.pipeline.ehm_pipeline import Ehm_Pipeline
            from models.modules.renderer.body_renderer import Renderer2
            from utils.general_utils import ConfigDict, add_extra_cfgs

            meta_cfg = add_extra_cfgs(
                ConfigDict(model_config_path=os.path.join("configs", f"{self.config_name}.yaml"))
            )
            ehm_model = Ehm_Pipeline(meta_cfg)
            ckpt = hf_hub_download(repo_id=self.repo_id, filename=self.checkpoint)
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            ehm_model.backbone.load_state_dict(state["backbone"], strict=False)
            ehm_model.head.load_state_dict(state["head"], strict=False)
            self._ehm_model = ehm_model.to(self.device).eval()

            self._ehm = EHM_v2("assets/FLAME", "assets/SMPLX").to(self.device).eval()
            self._renderer = Renderer2("assets/SMPLX", self.render_size, focal_length=_PEAR_FOCAL).to(self.device)

            self._head_idx = torch.as_tensor(
                self._ehm.smplx.smplx2flame_ind, device=self.device
            ).long()
            self._head_faces = self._ehm.flame.faces_tensor.to(self.device)

    def _ensure_detector(self):
        if self._detector is None:
            from ultralytics import YOLO

            weights = self.pear_root / self.detector_weights
            self._detector = YOLO(str(weights) if weights.exists() else "yolov8x.pt")

    def _ensure_bg(self):
        if self._bg_remover is None and self.matting:
            from hairport.bald_konverter.preprocessing.background import BackgroundRemover

            self._bg_remover = BackgroundRemover(device=self.device, alpha_threshold=0.5)

    # ------------------------------------------------------------------ #
    # Pre/post helpers (reuse PEAR's own affine utilities)
    # ------------------------------------------------------------------ #

    def _person_bbox(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Largest-person xywh bbox via YOLO; None if no detection."""
        self._ensure_detector()
        res = self._detector.predict(
            img_rgb, device=self.device, classes=0, conf=0.5, save=False, verbose=False
        )[0].boxes.xyxy.detach().cpu().numpy()
        if len(res) == 0:
            return None
        areas = (res[:, 2] - res[:, 0]) * (res[:, 3] - res[:, 1])
        x1, y1, x2, y2 = res[int(np.argmax(areas))]
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    def _silhouette_bbox(self, mask: np.ndarray) -> Optional[np.ndarray]:
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            return None
        return np.array([xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()],
                        dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        source: Optional[str] = None,
    ) -> BodyFitResult:
        if isinstance(image, (str, Path)):
            if source is None:
                source = str(image)
            img_rgb = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            img_rgb = np.array(image.convert("RGB"))
        else:
            img_rgb = image[..., :3].copy()
        h, w = img_rgb.shape[:2]

        self._ensure_models()

        # Matte to black background (user requirement: feed PEAR the matted image).
        model_input = img_rgb
        silh = None
        if self.matting:
            self._ensure_bg()
            fg, silh_pil = self._bg_remover.remove_background(Image.fromarray(img_rgb))
            silh = (np.array(silh_pil) > 127).astype(np.uint8) * 255
            matted = img_rgb.copy()
            matted[silh == 0] = 0
            model_input = matted

        with self._pear_context():
            from utils.graphics_utils import GS_Camera
            import torchvision.transforms as T
            # PEAR's bbox/affine helpers live in inference_images; replicate the
            # exact transform here to avoid importing its YOLO-bound main module.
            bbox_xywh = self._person_bbox(model_input)
            if bbox_xywh is None and silh is not None:
                bbox_xywh = self._silhouette_bbox(silh)
            if bbox_xywh is None:  # last resort: full frame
                bbox_xywh = np.array([0, 0, w, h], dtype=np.float32)

            bbox = _process_bbox(bbox_xywh, w, h, (_CROP, _CROP), ratio=1.25)
            patch, trans, inv_trans = _generate_patch_image(
                model_input.astype(np.float32), bbox, 1.0, 0.0, False, (_CROP, _CROP)
            )
            x = (T.ToTensor()(patch.astype(np.float32)) / 255.0).unsqueeze(0).to(self.device)

            # Inverse affine for the render-resolution silhouette: same square
            # bbox as the model patch, mapped from a render_size×render_size grid
            # back to the source frame (so the high-res render is not collapsed).
            bb_cx = float(bbox[0] + 0.5 * bbox[2])
            bb_cy = float(bbox[1] + 0.5 * bbox[3])
            inv_trans_render = _trans_from_patch(
                bb_cx, bb_cy, float(bbox[2]), float(bbox[3]),
                self.render_size, self.render_size, 1.0, 0.0, inv=True,
            )

            with torch.no_grad():
                outputs = self._ehm_model(x)
                smplx_dict = self._ehm(outputs["body_param"], outputs["flame_param"], pose_type="aa")

                verts = smplx_dict["vertices"]                  # [1, 10475, 3]
                head_verts = verts[:, self._head_idx]            # [1, Nhead, 3]
                cam = GS_Camera(
                    focal_length=torch.tensor([[_PEAR_FOCAL]], device=self.device),
                    principal_point=torch.zeros(1, 2, device=self.device),
                    image_size=torch.tensor(
                        [[self.render_size, self.render_size]], device=self.device
                    ).float(),
                    R=outputs["pd_cam"][0:1, :3, :3],
                    T=outputs["pd_cam"][0:1, :3, 3],
                    device=self.device,
                )
                # FLAME head submesh → head_mask; full SMPL-X mesh → body_mask.
                head_mask = self._render_silhouette(
                    head_verts, self._head_faces, cam, inv_trans_render, h, w
                )
                body_mask = self._render_silhouette(
                    verts, self._ehm.smplx.faces_tensor, cam, inv_trans_render, h, w
                )

        # Orientation from the rigid rotation of the head submesh vs. its
        # neutral (identity-pose) configuration — backend-agnostic.
        head_orient = self._orientation(outputs)

        cpu = lambda d: {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in d.items()}
        return BodyFitResult(
            backend="pear",
            smplx_params=cpu(outputs["body_param"]),
            flame_params=cpu(outputs["flame_param"]),
            camera={
                "R": outputs["pd_cam"][0, :3, :3].detach().cpu(),
                "T": outputs["pd_cam"][0, :3, 3].detach().cpu(),
                "focal_length": _PEAR_FOCAL,
                "screen_size": self.render_size,
                "inv_trans": inv_trans_render,
                "bbox_xywh": bbox.tolist(),
            },
            vertices=verts[0].detach().cpu(),
            faces=self._ehm.smplx.faces_tensor.detach().cpu(),
            head_vertices=head_verts[0].detach().cpu(),
            head_faces=self._head_faces.detach().cpu(),
            head_mask=head_mask,
            body_mask=body_mask,
            head_orientation=head_orient,
            image_size=(h, w),
            source=source,
        )

    def _render_silhouette(self, verts, faces, cam, inv_trans, h, w) -> np.ndarray:
        """Render *verts*/*faces* with *cam*, warp the silhouette to the source
        frame, and return a hole-filled uint8 0/255 mask of shape ``(h, w)``.

        PEAR's ``render_mesh`` returns ``[B, 4, S, S]`` (RGB + alpha, white bg,
        ×255) at the render resolution ``S = self.render_size``; the alpha is the
        silhouette. The soft alpha is warped (bilinear) to the source frame and
        thresholded — no intermediate down-sample, so edges stay crisp.
        """
        img = self._renderer.render_mesh(verts, cameras=cam, faces=faces)
        alpha = img[0, 3].clamp(0, 255).byte().cpu().numpy()  # (render_size, render_size)
        mask = cv2.warpAffine(
            alpha, inv_trans, (w, h), flags=cv2.INTER_LINEAR, borderValue=0
        )
        mask = (mask > 127).astype(np.uint8) * 255
        return fill_mask_holes(mask)

    def _orientation(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Head world rotation from a neutral-vs-posed Kabsch alignment."""
        with torch.no_grad():
            neutral = self._ehm(
                outputs["body_param"], outputs["flame_param"],
                zero_expression=True, zero_jaw=True, zero_shape=True, pose_type="aa",
            )["vertices"][:, self._head_idx][0].detach().cpu().numpy()
            posed = self._ehm(
                outputs["body_param"], outputs["flame_param"], pose_type="aa",
            )["vertices"][:, self._head_idx][0].detach().cpu().numpy()
        R = _kabsch(neutral, posed)
        return _euler_and_basis_from_rotmat(R)

    # ------------------------------------------------------------------ #
    # Memory residency
    # ------------------------------------------------------------------ #

    def to_device(self, device: Optional[str] = None) -> None:
        from hairport import memory

        device = device or self.device
        for m in (self._ehm_model, self._ehm, self._renderer):
            memory.move_to(m, device)

    def offload(self) -> None:
        from hairport import memory

        for m in (self._ehm_model, self._ehm, self._renderer):
            memory.offload(m)
        if self._bg_remover is not None:
            self._bg_remover.offload()

    def teardown(self) -> None:
        from hairport import memory

        self._ehm_model = self._ehm = self._renderer = None
        self._detector = None
        if self._bg_remover is not None:
            self._bg_remover.teardown()
            self._bg_remover = None
        memory.flush()

    def __del__(self):
        try:
            self.teardown()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# PEAR bbox/affine utilities (faithful copies of inference_images.py, kept here
# so we don't import that YOLO-bound module). Behaviour-identical.
# --------------------------------------------------------------------------- #

def _sanitize_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w - 1, x1 + max(0, w - 1))
    y2 = min(img_h - 1, y1 + max(0, h - 1))
    if w * h > 0 and x2 > x1 and y2 > y1:
        return np.array([x1, y1, x2 - x1, y2 - y1])
    return None


def _process_bbox(bbox, img_w, img_h, input_shape, ratio=1.25):
    bbox = _sanitize_bbox(bbox, img_w, img_h)
    if bbox is None:
        return np.array([0, 0, img_w, img_h], dtype=np.float32)
    w, h = bbox[2], bbox[3]
    c_x, c_y = bbox[0] + w / 2.0, bbox[1] + h / 2.0
    aspect = input_shape[1] / input_shape[0]
    if w > aspect * h:
        h = w / aspect
    elif w < aspect * h:
        w = h * aspect
    bbox[2], bbox[3] = w * ratio, h * ratio
    bbox[0], bbox[1] = c_x - bbox[2] / 2.0, c_y - bbox[3] / 2.0
    return bbox.astype(np.float32)


def _rotate_2d(pt, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return np.array([pt[0] * cs - pt[1] * sn, pt[0] * sn + pt[1] * cs], dtype=np.float32)


def _trans_from_patch(c_x, c_y, src_w, src_h, dst_w, dst_h, scale, rot, inv=False):
    src_w, src_h = src_w * scale, src_h * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    rot_rad = np.pi * rot / 180
    src_down = _rotate_2d(np.array([0, src_h * 0.5], np.float32), rot_rad)
    src_right = _rotate_2d(np.array([src_w * 0.5, 0], np.float32), rot_rad)
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_down = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_right = np.array([dst_w * 0.5, 0], dtype=np.float32)
    src = np.stack([src_center, src_center + src_down, src_center + src_right])
    dst = np.stack([dst_center, dst_center + dst_down, dst_center + dst_right])
    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src)).astype(np.float32)
    return cv2.getAffineTransform(np.float32(src), np.float32(dst)).astype(np.float32)


def _generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape):
    img = cvimg.copy()
    img_h, img_w = img.shape[:2]
    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_w, bb_h = float(bbox[2]), float(bbox[3])
    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_w - bb_c_x - 1
    trans = _trans_from_patch(bb_c_x, bb_c_y, bb_w, bb_h, out_shape[1], out_shape[0], scale, rot)
    patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    inv_trans = _trans_from_patch(bb_c_x, bb_c_y, bb_w, bb_h, out_shape[1], out_shape[0], scale, rot, inv=True)
    return patch.astype(np.float32), trans, inv_trans
