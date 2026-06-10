"""End-to-end bald conversion: framing → FLUX wo_seg/w_seg on the plate → composite back."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from .config.defaults import (
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_STRENGTH,
)
from hairport.fitting import BodyFitResult
from .framing import Framing, plan_framing
from . import compositing
from .utils.image import (
    create_body_green_image,
    create_combined_seg_image,
    create_four_panel,
    resize_to_square,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class BaldResult:
    """Bald-conversion output. ``bald_image`` is full-res original-aspect (head
    changed only); the rest are intermediates (populated when requested). Masks
    are in original-frame coords; ``head_fit`` is the SMPL-X+FLAME fit."""

    bald_image: Image.Image
    plate: Optional[np.ndarray] = None              # native square head plate
    bald_plate: Optional[np.ndarray] = None         # model bald output, pre-composite
    change_alpha: Optional[np.ndarray] = None       # compositing matte (plate coords)
    bald_image_wo_seg: Optional[Image.Image] = None
    hair_mask: Optional[np.ndarray] = None
    body_mask: Optional[np.ndarray] = None
    head_mask: Optional[np.ndarray] = None
    smplx_body_mask: Optional[np.ndarray] = None
    flux_input_wo_seg: Optional[Image.Image] = None
    flux_input_w_seg: Optional[Image.Image] = None
    foreground: Optional[Image.Image] = None
    framing: Optional[Framing] = None
    comp_params: Optional[dict] = None
    head_fit: Optional[BodyFitResult] = None


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


class BaldKonverterPipeline:
    """Bald conversion. ``mode``: ``wo_seg`` (2-panel), ``w_seg`` (4-panel,
    SMPL-X-guided), or ``auto`` (wo_seg then w_seg). ``w_seg``/``auto`` require
    the fitting backend (``cfg.fitting.backend``, default PEAR)."""

    def __init__(
        self,
        mode: str = "auto",
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        lora_path_wo_seg: Optional[str] = None,
        lora_path_w_seg: Optional[str] = None,
        base_model: str | None = None,
        base_model_revision: str | None = None,
        lora_repo: str | None = None,
        lora_revision: str | None = None,
        wo_seg_image_size: int | None = None,
        w_seg_image_size: int | None = None,
    ):
        from hairport.config import get_config

        cfg = get_config()
        if mode not in ("wo_seg", "w_seg", "auto"):
            raise ValueError(f"Invalid mode '{mode}'. Choose from: wo_seg, w_seg, auto")

        self.mode = mode
        self.device = device
        self.dtype = dtype if dtype is not None else getattr(torch, cfg.baldify.dtype)
        self.wo_seg_image_size = (
            wo_seg_image_size if wo_seg_image_size is not None
            else cfg.baldify.wo_seg_image_size
        )
        self.w_seg_image_size = (
            w_seg_image_size if w_seg_image_size is not None
            else cfg.baldify.w_seg_image_size
        )
        self.framing_cfg = cfg.baldify.framing
        self.compositing_cfg = cfg.baldify.compositing

        # ---- Lazy-loaded components ----------------------------------------
        self._pipe = None  # shared FluxInpaintPipeline
        self._preproc: Optional[object] = None
        self._backend: Optional[object] = None  # head-fitting backend
        self._face_detector: Optional[object] = None  # framing face bbox

        self._lora_wo_seg = lora_path_wo_seg
        self._lora_w_seg = lora_path_w_seg
        self._base_model = base_model or cfg.models.flux_kontext
        self._base_model_revision = (
            base_model_revision if base_model_revision is not None
            else cfg.models.flux_kontext_revision
        )
        self._lora_repo = lora_repo or cfg.models.bald_konverter_repo
        self._lora_revision = (
            lora_revision if lora_revision is not None
            else cfg.models.bald_konverter_revision
        )

        self._active_lora: Optional[str] = None  # track which LoRA is loaded

    # ------------------------------------------------------------------ #
    # Lazy loaders — avoid loading models until first use
    # ------------------------------------------------------------------ #

    def _get_base_pipe(self):
        from hairport import memory

        if self._pipe is None:
            from .models.konverter import load_base_pipeline

            self._pipe = load_base_pipeline(
                base_model=self._base_model,
                revision=self._base_model_revision,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            # May have been parked in CPU RAM between usage windows
            # (memory.policy=exclusive); bring it back before use.
            memory.move_to(self._pipe, self.device)
        return self._pipe

    def _load_lora(self, variant: str) -> None:
        """Swap LoRA weights on the shared pipeline if needed."""
        if self._active_lora == variant:
            return
        pipe = self._get_base_pipe()
        if self._active_lora is not None:
            pipe.unload_lora_weights()
        from .models.hub import download_checkpoint

        lora_path = (
            self._lora_wo_seg if variant == "wo_seg" else self._lora_w_seg
        )
        if lora_path is None:
            lora_path = download_checkpoint(
                variant, repo_id=self._lora_repo, revision=self._lora_revision
            )
        pipe.load_lora_weights(lora_path)
        self._active_lora = variant
        logger.info("Loaded %s LoRA weights", variant)

    def _get_preprocessor(self):
        if self._preproc is None:
            from .preprocessing.hair_mask import HairMaskPipeline

            self._preproc = HairMaskPipeline(device=self.device)
        return self._preproc

    def _get_backend(self):
        if self._backend is None:
            from hairport.fitting import get_fitting_backend

            self._backend = get_fitting_backend(device=self.device)
        return self._backend

    def _face_bbox(self, image_np: np.ndarray):
        """Best-effort face bbox ``(x, y, w, h)`` for framing; ``None`` on failure."""
        try:
            if self._face_detector is None:
                from hairport.core import FacialLandmarkDetector

                self._face_detector = FacialLandmarkDetector(
                    static_image_mode=True, max_num_faces=1,
                    refine_landmarks=False, min_detection_confidence=0.5,
                )
            bbox = self._face_detector.get_face_bounding_box(image_np, return_format="xywh")
            if bbox is None:
                return None
            return tuple(int(v) for v in bbox)
        except Exception:
            logger.debug("Face detection unavailable for framing; using mask bbox.", exc_info=True)
            return None

    # ------------------------------------------------------------------ #
    # Core generation methods
    # ------------------------------------------------------------------ #

    def _run_wo_seg(
        self,
        image: Image.Image,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
    ) -> tuple[Image.Image, Image.Image]:
        """Two-panel wo_seg generation; returns (bald_plate, flux_input_2panel)."""
        self._load_lora("wo_seg")
        pipe = self._get_base_pipe()

        from .config.defaults import PROMPT_WO_SEG
        from .utils.image import (
            create_two_panel,
            crop_right_half,
            make_right_half_mask,
            resize_to_square,
        )

        size = self.wo_seg_image_size
        img = resize_to_square(image, size)
        combined = create_two_panel(img, img)
        mask = make_right_half_mask(combined.size[0], combined.size[1])

        output = pipe(
            prompt=PROMPT_WO_SEG,
            image=combined,
            mask_image=mask,
            guidance_scale=guidance_scale,
            height=combined.size[1],
            width=combined.size[0],
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]

        return crop_right_half(output), combined

    def _run_w_seg(
        self,
        image: Image.Image,
        bald_wo_seg: Image.Image,
        hair_mask: np.ndarray,
        body_mask: np.ndarray,
        seed: int,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        smplx_body_mask: np.ndarray,
        head_mask: Optional[np.ndarray] = None,
    ) -> tuple[Image.Image, Image.Image]:
        """Four-panel w_seg generation; returns (bald_plate, grid). Panels:
        top-left = SAM3 hair (red) over body∪head (green); top-right = SMPL-X
        silhouette (green); bottom = original plate + wo_seg bald."""
        self._load_lora("w_seg")
        pipe = self._get_base_pipe()

        from .config.defaults import PROMPT_W_SEG
        from .utils.image import (
            crop_bottom_right_quadrant,
            make_bottom_right_mask,
            resize_to_square,
        )

        size = self.w_seg_image_size
        half = size // 2

        # Top-left body = BEN2 bald silhouette ∪ precomputed head mask.
        final_body = body_mask.copy()
        if head_mask is not None:
            import cv2

            if head_mask.shape != final_body.shape:
                head_mask = cv2.resize(
                    head_mask,
                    (final_body.shape[1], final_body.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            final_body = np.maximum(final_body, head_mask)

        # Build panels
        combined_seg = create_combined_seg_image(hair_mask, final_body, size=half)
        body_green = create_body_green_image(smplx_body_mask, size=half)
        orig_panel = resize_to_square(image, half)
        bald_panel = resize_to_square(bald_wo_seg, half)

        grid = create_four_panel(combined_seg, body_green, orig_panel, bald_panel)
        if grid.size != (size, size):
            grid = resize_to_square(grid, size)
        mask = make_bottom_right_mask(grid.size[0], grid.size[1])

        output = pipe(
            prompt=PROMPT_W_SEG,
            image=grid,
            mask_image=mask,
            guidance_scale=guidance_scale,
            height=grid.size[1],
            width=grid.size[0],
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]

        return crop_bottom_right_quadrant(output), grid

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        image: Union[str, Path, Image.Image],
        seed: int = DEFAULT_SEED,
        refine_seed: int | None = None,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        strength: float = DEFAULT_STRENGTH,
        return_intermediates: bool = False,
    ) -> BaldResult:
        """Bald-convert *image* (path or PIL). ``refine_seed`` defaults to
        ``seed``; ``return_intermediates`` populates the optional BaldResult fields."""
        import cv2
        from hairport import memory

        # Load image (keep both a full-res ndarray and PIL view).
        if isinstance(image, (str, Path)):
            source = str(image)
            image = Image.open(image).convert("RGB")
        else:
            source = None
            image = image.convert("RGB")
        img_np = np.array(image)

        wo_gen_kwargs = dict(
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        )
        ri = return_intermediates

        # ---- Preprocess on the ORIGINAL (full-res) -------------------------
        # hair matte + foreground silhouette in true image coordinates; needed
        # by every mode for framing and the composite.
        preproc = self._get_preprocessor()
        preproc.to_device(self.device)
        prep_result = preproc.preprocess(image, return_foreground=ri)
        hair_full = prep_result.hair_mask
        silh_full = prep_result.silhouette
        face_bbox = self._face_bbox(img_np)
        preproc.offload()

        # ---- Framing — head-centric square plate ---------------------------
        fr = plan_framing(
            img_np, hair_full, face_bbox=face_bbox, foreground_mask=silh_full,
            crop_scale=float(self.framing_cfg.crop_scale),
            model_size=self.wo_seg_image_size,
        )
        plate_pil = Image.fromarray(
            fr.extract_native(img_np, border_mode=str(self.framing_cfg.border_pad_mode))
        )

        # ---- Step 1: wo_seg on the plate -----------------------------------
        bald_plate_pil, flux_input_wo = self._run_wo_seg(plate_pil, **wo_gen_kwargs)
        bald_wo_plate = bald_plate_pil  # keep the wo_seg plate for intermediates

        grid = None
        head_fit = None
        body_mask_plate = head_mask_plate = smplx_body_plate = None

        # ---- Steps 2-5: w_seg refinement (w_seg / auto) --------------------
        if self.mode != "wo_seg":
            memory.offload(self._pipe)
            preproc.to_device(self.device)
            # body silhouette (BEN2) on the bald plate — top-left panel base
            _, bald_silh_pil = preproc.bg_remover.remove_background(bald_plate_pil)
            body_mask_plate = np.array(bald_silh_pil).astype(np.uint8)
            # hair matte resampled into plate coords (top-left red)
            hair_plate = fr.map_mask_into_plate(hair_full)
            # SMPL-X head + body fit on the bald plate
            backend = self._get_backend()
            backend.to_device(self.device)
            head_fit = backend.fit(bald_plate_pil, source=source)
            head_mask_plate = head_fit.head_mask
            smplx_body_plate = head_fit.body_mask
            backend.offload()
            preproc.offload()

            bald_plate_pil, grid = self._run_w_seg(
                image=plate_pil,
                bald_wo_seg=bald_plate_pil,
                hair_mask=hair_plate,
                body_mask=body_mask_plate,
                head_mask=head_mask_plate,
                smplx_body_mask=smplx_body_plate,
                seed=seed if refine_seed is None else refine_seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
            )

        # ---- Composite the bald plate back into the original frame ---------
        # The bald plate is composited as-is (no color matching); the matte
        # grows the hair seed through the model-changed wisp band to avoid a
        # residual-hair halo.
        side = fr.side
        orig_plate = fr.extract_native(img_np, border_mode="reflect")
        bald_plate_native = cv2.resize(
            np.array(bald_plate_pil), (side, side), interpolation=cv2.INTER_LANCZOS4
        )
        hair_plate_native = fr.map_mask_into_plate(hair_full)
        ccfg = self.compositing_cfg

        comp_plate, alpha_plate, comp_params = compositing.composite_plate(
            orig_plate, bald_plate_native, hair_plate_native,
            seam_poisson=bool(ccfg.seam_poisson),
            grain_match=bool(ccfg.grain_match),
            matte_dilate_px=int(ccfg.matte_dilate_px),
            extend_band_frac=float(ccfg.extend_band_frac),
            extend_diff_threshold=int(ccfg.extend_diff_threshold),
            feather_px=int(ccfg.feather_px),
            border_zero_frac=float(ccfg.border_zero_frac),
        )
        comp_params["seed"] = seed
        comp_params["refine_seed"] = refine_seed

        final_pil = Image.fromarray(fr.paste(img_np, comp_plate))

        # ---- Map plate-space masks back to the ORIGINAL frame -------------- #
        def _to_orig(mask_plate):
            if mask_plate is None:
                return None
            m = cv2.resize(mask_plate, (side, side), interpolation=cv2.INTER_NEAREST) \
                if mask_plate.shape[:2] != (side, side) else mask_plate
            return fr.plate_to_original(m.astype(np.uint8))

        if head_fit is not None:
            head_fit.extra["framing"] = fr.to_dict()

        return BaldResult(
            bald_image=final_pil,
            plate=orig_plate if ri else None,
            bald_plate=bald_plate_native if ri else None,
            change_alpha=(alpha_plate * 255).astype(np.uint8) if ri else None,
            bald_image_wo_seg=bald_wo_plate if ri else None,
            hair_mask=hair_full if ri else None,
            body_mask=_to_orig(body_mask_plate) if ri else None,
            head_mask=_to_orig(head_mask_plate) if ri else None,
            smplx_body_mask=_to_orig(smplx_body_plate) if ri else None,
            flux_input_wo_seg=flux_input_wo,
            flux_input_w_seg=grid,
            foreground=prep_result.foreground if ri else None,
            framing=fr,
            comp_params=comp_params,
            head_fit=head_fit,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def teardown(self) -> None:
        """Release all GPU resources (idempotent; everything reloads lazily)."""
        if self._preproc is not None:
            self._preproc.teardown()
            self._preproc = None
        if self._backend is not None:
            self._backend.teardown()
            self._backend = None
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        self._active_lora = None
        torch.cuda.empty_cache()
