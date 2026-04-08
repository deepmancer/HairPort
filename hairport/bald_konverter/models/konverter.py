"""FLUX-based bald-conversion inference wrappers.

Two LoRA-based generators:

* :class:`BaldKonverter` — *wo_seg* mode (2-panel, no segmentation input)
* :class:`BaldKonverterWithSeg` — *w_seg* mode (4-panel, segmentation-guided)

Both share the same heavy ``FluxInpaintPipeline`` base model and simply swap
LoRA weights, avoiding the need to load ~12 GB twice.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from .toolkit.pipeline_flux_inpaint import FluxInpaintPipeline
from PIL import Image

from ..config.defaults import (
    BASE_MODEL_ID,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_STRENGTH,
    PROMPT_W_SEG,
    PROMPT_WO_SEG,
    W_SEG_IMAGE_SIZE,
    WO_SEG_IMAGE_SIZE,
)
from ..utils.image import (
    create_two_panel,
    crop_bottom_right_quadrant,
    crop_right_half,
    make_bottom_right_mask,
    make_right_half_mask,
    resize_to_square,
)
from .hub import download_checkpoint

logger = logging.getLogger(__name__)


def load_base_pipeline(
    base_model: str = BASE_MODEL_ID,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> FluxInpaintPipeline:
    """Load the base FLUX inpainting pipeline (no LoRA weights attached).

    This can be shared between :class:`BaldKonverter` and
    :class:`BaldKonverterWithSeg` via :meth:`from_pipeline`.
    """
    torch.set_float32_matmul_precision("high")
    pipe = FluxInpaintPipeline.from_pretrained(base_model, torch_dtype=dtype)
    pipe.to(device)
    logger.info("Base FLUX pipeline loaded on %s (dtype=%s)", device, dtype)
    return pipe


# --------------------------------------------------------------------------- #
# wo_seg mode — 2-panel conversion
# --------------------------------------------------------------------------- #


class BaldKonverter:
    """Generate a bald image using the 2-panel (wo_seg) FLUX LoRA.

    The source image is placed on the left; FLUX inpaints the right panel
    as the bald version using the prompt cue and a LoRA-finetuned adapter.

    Parameters
    ----------
    lora_path : str, optional
        Path to the LoRA ``.safetensors`` file.  If ``None``, it is
        automatically downloaded from the Hugging Face Hub.
    base_model : str
        Hugging Face ID for the base FLUX model.
    device : str
        Compute device.
    dtype : torch.dtype
        Model precision.
    """

    def __init__(
        self,
        lora_path: Optional[str] = None,
        base_model: str = BASE_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.pipe = load_base_pipeline(base_model, device, dtype)
        lora = lora_path or download_checkpoint("wo_seg")
        self.pipe.load_lora_weights(lora)
        logger.info("BaldKonverter (wo_seg) LoRA loaded from %s", lora)

    @classmethod
    def from_pipeline(
        cls,
        pipe: FluxInpaintPipeline,
        lora_path: Optional[str] = None,
    ) -> "BaldKonverter":
        """Create from an *existing* pipeline to avoid reloading the base model."""
        obj = cls.__new__(cls)
        obj.device = str(pipe.device)
        obj.pipe = pipe
        lora = lora_path or download_checkpoint("wo_seg")
        pipe.load_lora_weights(lora)
        logger.info("BaldKonverter (wo_seg) LoRA attached from %s", lora)
        return obj

    def generate(
        self,
        image: Image.Image,
        prompt: str = PROMPT_WO_SEG,
        size: int = WO_SEG_IMAGE_SIZE,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        strength: float = DEFAULT_STRENGTH,
        seed: int = DEFAULT_SEED,
    ) -> Image.Image:
        """Run bald conversion on *image*.

        Parameters
        ----------
        image : PIL.Image.Image
            Source portrait.
        prompt : str
            Generation prompt.
        size : int
            Each panel is resized to (*size* × *size*).
        guidance_scale, num_inference_steps, strength, seed
            FLUX generation parameters.

        Returns
        -------
        PIL.Image.Image
            The generated bald image (*size* × *size*).
        """
        img = resize_to_square(image, size)
        combined = create_two_panel(img, img)
        mask = make_right_half_mask(combined.size[0], combined.size[1])

        output = self.pipe(
            prompt=prompt,
            image=combined,
            mask_image=mask,
            guidance_scale=guidance_scale,
            height=combined.size[1],
            width=combined.size[0],
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]

        return crop_right_half(output)


# --------------------------------------------------------------------------- #
# w_seg mode — 4-panel conversion
# --------------------------------------------------------------------------- #


class BaldKonverterWithSeg:
    """Generate a bald image using the 4-panel (w_seg) FLUX LoRA.

    Expects a pre-assembled 2×2 grid image where the *bottom-right* quadrant
    is to be inpainted as the bald version, informed by the segmentation context
    in the top two panels.

    Parameters
    ----------
    lora_path : str, optional
        Path to the LoRA ``.safetensors`` file.  If ``None``, downloaded
        from the Hub.
    base_model : str
        Hugging Face ID for the base FLUX model.
    device : str
        Compute device.
    dtype : torch.dtype
        Model precision.
    """

    def __init__(
        self,
        lora_path: Optional[str] = None,
        base_model: str = BASE_MODEL_ID,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.pipe = load_base_pipeline(base_model, device, dtype)
        lora = lora_path or download_checkpoint("w_seg")
        self.pipe.load_lora_weights(lora)
        logger.info("BaldKonverterWithSeg (w_seg) LoRA loaded from %s", lora)

    @classmethod
    def from_pipeline(
        cls,
        pipe: FluxInpaintPipeline,
        lora_path: Optional[str] = None,
    ) -> "BaldKonverterWithSeg":
        """Create from an *existing* pipeline to avoid reloading the base model."""
        obj = cls.__new__(cls)
        obj.device = str(pipe.device)
        obj.pipe = pipe
        lora = lora_path or download_checkpoint("w_seg")
        pipe.load_lora_weights(lora)
        logger.info("BaldKonverterWithSeg (w_seg) LoRA attached from %s", lora)
        return obj

    def generate(
        self,
        grid_image: Image.Image,
        prompt: str = PROMPT_W_SEG,
        size: int = W_SEG_IMAGE_SIZE,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        strength: float = DEFAULT_STRENGTH,
        seed: int = DEFAULT_SEED,
    ) -> Image.Image:
        """Run bald conversion on a pre-assembled 2×2 grid.

        Parameters
        ----------
        grid_image : PIL.Image.Image
            2×2 panel grid (see :func:`~bald_konverter.utils.image.create_four_panel`).
        prompt : str
            Generation prompt.
        size : int
            The grid is resized to (*size* × *size*) before inference.
        guidance_scale, num_inference_steps, strength, seed
            FLUX generation parameters.

        Returns
        -------
        PIL.Image.Image
            Cropped bottom-right quadrant — the bald result.
        """
        grid = resize_to_square(grid_image, size)
        mask = make_bottom_right_mask(grid.size[0], grid.size[1])

        output = self.pipe(
            prompt=prompt,
            image=grid,
            mask_image=mask,
            guidance_scale=guidance_scale,
            height=grid.size[1],
            width=grid.size[0],
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]

        return crop_bottom_right_quadrant(output)
