"""End-to-end bald-conversion pipeline.

Orchestrates preprocessing (hair/body masks), FLUX LoRA inference, and
optional FLAME segmentation into a single callable.

Usage::

    from hairport.bald_konverter import BaldKonverterPipeline

    pipeline = BaldKonverterPipeline(mode="auto")
    result = pipeline("portrait.jpg")
    result.bald_image.save("bald.png")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    W_SEG_IMAGE_SIZE,
    WO_SEG_IMAGE_SIZE,
)
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
    """Output from :class:`BaldKonverterPipeline`.

    Attributes
    ----------
    bald_image : PIL.Image.Image
        The generated bald portrait.
    hair_mask : np.ndarray | None
        Binary hair mask (only in ``w_seg`` / ``auto`` mode).
    body_mask : np.ndarray | None
        Binary body mask (only in ``w_seg`` / ``auto`` mode).
    segmentation_map : np.ndarray | None
        Segformer per-pixel labels (if ``return_intermediates=True``).
    flux_input : PIL.Image.Image | None
        The assembled grid fed to FLUX (if ``return_intermediates=True``).
    foreground : PIL.Image.Image | None
        RGBA foreground (if ``return_intermediates=True``).
    flame_mask : np.ndarray | None
        FLAME head mask (if ``use_flame=True`` and ``return_intermediates=True``).
    """

    bald_image: Image.Image
    hair_mask: Optional[np.ndarray] = None
    body_mask: Optional[np.ndarray] = None
    segmentation_map: Optional[np.ndarray] = None
    flux_input: Optional[Image.Image] = None
    foreground: Optional[Image.Image] = None
    flame_mask: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


class BaldKonverterPipeline:
    """High-level API for bald conversion.

    Parameters
    ----------
    mode : ``"wo_seg"`` | ``"w_seg"`` | ``"auto"``
        * ``wo_seg`` — fast 2-panel conversion (no preprocessing needed).
        * ``w_seg`` — segmentation-guided 4-panel conversion (higher quality).
        * ``auto`` — runs ``wo_seg`` first, then refines with ``w_seg``.
    device : str
        Compute device.
    dtype : torch.dtype
        Model precision.
    use_flame : bool
        If ``True``, use SHeaP FLAME fitting for the head mask (requires
        ``pip install bald-konverter[flame]`` and the FLAME2020 model files).
    flame_dir : str | Path, optional
        Path to the ``FLAME2020/`` directory (only used if ``use_flame=True``).
    lora_path_wo_seg : str, optional
        Custom LoRA path for the wo_seg model.
    lora_path_w_seg : str, optional
        Custom LoRA path for the w_seg model.
    """

    def __init__(
        self,
        mode: str = "auto",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_flame: bool = False,
        flame_dir: Optional[Union[str, Path]] = None,
        lora_path_wo_seg: Optional[str] = None,
        lora_path_w_seg: Optional[str] = None,
    ):
        if mode not in ("wo_seg", "w_seg", "auto"):
            raise ValueError(f"Invalid mode '{mode}'. Choose from: wo_seg, w_seg, auto")

        self.mode = mode
        self.device = device
        self.dtype = dtype
        self.use_flame = use_flame

        # ---- Lazy-loaded components ----------------------------------------
        self._pipe = None  # shared FluxInpaintPipeline
        self._konverter_wo: Optional[object] = None
        self._konverter_w: Optional[object] = None
        self._preproc: Optional[object] = None
        self._flame: Optional[object] = None

        self._lora_wo_seg = lora_path_wo_seg
        self._lora_w_seg = lora_path_w_seg
        self._flame_dir = flame_dir

        self._active_lora: Optional[str] = None  # track which LoRA is loaded

    # ------------------------------------------------------------------ #
    # Lazy loaders — avoid loading models until first use
    # ------------------------------------------------------------------ #

    def _get_base_pipe(self):
        if self._pipe is None:
            from .models.konverter import load_base_pipeline

            self._pipe = load_base_pipeline(device=self.device, dtype=self.dtype)
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
            lora_path = download_checkpoint(variant)
        pipe.load_lora_weights(lora_path)
        self._active_lora = variant
        logger.info("Loaded %s LoRA weights", variant)

    def _get_preprocessor(self):
        if self._preproc is None:
            from .preprocessing.hair_mask import HairMaskPipeline

            self._preproc = HairMaskPipeline(device=self.device)
        return self._preproc

    def _get_flame_segmenter(self):
        if self._flame is None:
            from .preprocessing.flame import FLAMESegmenter

            self._flame = FLAMESegmenter(
                flame_dir=self._flame_dir or "FLAME2020",
                device=self.device,
            )
        return self._flame

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
        """Two-panel generation (wo_seg).

        Returns
        -------
        tuple[Image.Image, Image.Image]
            ``(bald_image, flux_input)`` — the cropped bald result and the
            assembled 2-panel image that was fed to FLUX.
        """
        self._load_lora("wo_seg")
        pipe = self._get_base_pipe()

        from .config.defaults import PROMPT_WO_SEG
        from .utils.image import (
            create_two_panel,
            crop_right_half,
            make_right_half_mask,
            resize_to_square,
        )

        size = WO_SEG_IMAGE_SIZE
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
        flame_mask: Optional[np.ndarray] = None,
    ) -> tuple[Image.Image, Image.Image]:
        """Four-panel generation (w_seg). Returns (bald_image, grid_image)."""
        self._load_lora("w_seg")
        pipe = self._get_base_pipe()

        from .config.defaults import PROMPT_W_SEG
        from .utils.image import (
            crop_bottom_right_quadrant,
            make_bottom_right_mask,
            resize_to_square,
        )

        size = W_SEG_IMAGE_SIZE
        half = size // 2

        # Compute final body mask (union with FLAME or SAM head mask)
        final_body = body_mask.copy()
        if flame_mask is not None:
            final_body = np.maximum(final_body, flame_mask)
        else:
            # Use SAM "head" prompt on the bald image as fallback
            try:
                preproc = self._get_preprocessor()
                head_pil, _ = preproc.sam_extractor(bald_wo_seg, prompt="head")
                head_mask = (np.array(head_pil) > 127).astype(np.uint8) * 255
                import cv2

                if head_mask.shape != final_body.shape:
                    head_mask = cv2.resize(
                        head_mask,
                        (final_body.shape[1], final_body.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                final_body = np.maximum(final_body, head_mask)
            except Exception:
                logger.warning("SAM head-mask fallback failed; using body mask only.")

        # Build panels
        combined_seg = create_combined_seg_image(hair_mask, final_body, size=half)
        body_green = create_body_green_image(final_body, size=half)
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
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        strength: float = DEFAULT_STRENGTH,
        return_intermediates: bool = False,
    ) -> BaldResult:
        """Generate a bald portrait from *image*.

        Parameters
        ----------
        image : path or PIL Image
            Input portrait.
        seed : int
            Random seed for reproducibility.
        num_inference_steps : int
            Number of diffusion steps.
        guidance_scale : float
            Classifier-free guidance scale.
        strength : float
            Inpainting strength (0–1).
        return_intermediates : bool
            If ``True``, populate optional fields in :class:`BaldResult`.

        Returns
        -------
        BaldResult
        """
        # Load image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        gen_kwargs = dict(
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        )

        # ---- wo_seg only ----------------------------------------------------
        if self.mode == "wo_seg":
            bald, flux_input_wo = self._run_wo_seg(image, **gen_kwargs)
            return BaldResult(bald_image=bald, flux_input=flux_input_wo)

        # ---- w_seg or auto --------------------------------------------------
        # Step 1: initial bald via wo_seg
        bald_wo, flux_input_wo = self._run_wo_seg(image, **gen_kwargs)

        # Step 2: preprocessing — hair mask from original image
        preproc = self._get_preprocessor()
        prep_result = preproc.preprocess(
            image,
            return_foreground=return_intermediates,
            return_segformer=return_intermediates,
        )

        # Step 3: body mask from BEN2 on the *bald* image (full silhouette)
        _, bald_silh_pil = preproc.bg_remover.remove_background(bald_wo)
        body_mask = np.array(bald_silh_pil).astype(np.uint8)

        # Step 4: optional FLAME head mask
        flame_mask = None
        if self.use_flame:
            flame_seg = self._get_flame_segmenter()
            flame_mask = flame_seg.segment(bald_wo)

        # Step 5: w_seg generation
        bald_w, grid = self._run_w_seg(
            image=image,
            bald_wo_seg=bald_wo,
            hair_mask=prep_result.hair_mask,
            body_mask=body_mask,
            flame_mask=flame_mask,
            **gen_kwargs,
        )

        return BaldResult(
            bald_image=bald_w,
            hair_mask=prep_result.hair_mask if return_intermediates else None,
            body_mask=body_mask if return_intermediates else None,
            segmentation_map=prep_result.segformer_labels if return_intermediates else None,
            flux_input=grid,
            foreground=prep_result.foreground if return_intermediates else None,
            flame_mask=flame_mask if return_intermediates else None,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def teardown(self) -> None:
        """Release all GPU resources."""
        if self._preproc is not None:
            self._preproc.teardown()
        if self._flame is not None:
            self._flame.teardown()
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        torch.cuda.empty_cache()
