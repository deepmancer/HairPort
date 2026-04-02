"""Stage 2 — Caption: Generate text descriptions and outpaint bald images.

Delegates to ``hairport.utility.uncrop_qwen`` for Qwen-based outpainting.

Usage::

    # Programmatic
    from hairport.stages.caption import CaptionStage
    stage = CaptionStage()
    stage.run(data_dir="outputs")

    # CLI
    python -m hairport.stages.caption --data_dir outputs
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from hairport.config import get_config, add_config_args, load_config_from_args

logger = logging.getLogger(__name__)


class CaptionStage:
    """Outpaint bald images using Qwen Image-Edit.

    Parameters
    ----------
    device : str
        Compute device.
    dtype : torch.dtype
        Model dtype (default BFloat16).
    """

    def __init__(self, device: str | None = None, dtype: torch.dtype = torch.bfloat16):
        cfg = get_config()
        self.device = device if device is not None else cfg.device
        self.dtype = dtype
        self._pipe = None

    def _ensure_pipeline(self):
        if self._pipe is not None:
            return
        from diffusers import QwenImageEditPipeline
        from hairport.utility.uncrop_qwen import QwenImageTransformer2DModel

        cfg = get_config()
        self._pipe = QwenImageEditPipeline.from_pretrained(
            cfg.models.qwen_image_edit, torch_dtype=self.dtype
        ).to(self.device)
        self._pipe.transformer.__class__ = QwenImageTransformer2DModel
        self._pipe.load_lora_weights(
            cfg.models.qwen_lightning_lora,
            weight_name=cfg.models.qwen_lightning_lora_weight,
        )
        self._pipe.fuse_lora()
        logger.info("Qwen outpainting pipeline loaded")

    def run(
        self,
        data_dir: str | Path,
        bald_subdir: str | None = None,
        images_subdir: str = "image",
        output_subdir: str = "image_outpainted",
    ) -> dict:
        """Run outpainting on all images in the bald directory.

        Returns
        -------
        dict
            Summary with count keys.
        """
        from hairport.utility.uncrop_qwen import prepare_image_and_mask

        self._ensure_pipeline()
        cfg = get_config()
        if bald_subdir is None:
            bald_subdir = f"bald/{cfg.pipeline.bald_version}"
        data_dir = Path(data_dir)
        images_dir = data_dir / bald_subdir / images_subdir
        output_dir = data_dir / bald_subdir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = get_config()
        _cap = cfg.caption

        target_ids = [p.stem for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        logger.info(f"Found {len(target_ids)} images to outpaint")

        stats = {"processed": 0, "skipped": 0, "failed": 0}
        for target_id in tqdm(target_ids, desc="Outpainting"):
            output_path = output_dir / f"{target_id}.png"
            if output_path.exists():
                stats["skipped"] += 1
                continue
            try:
                input_image = Image.open(images_dir / f"{target_id}.png").convert("RGB")
                input_prepared, mask = prepare_image_and_mask(
                    image=input_image,
                    width=_cap.width, height=_cap.height,
                    overlap_percentage=_cap.overlap_percentage,
                    resize_option="Custom",
                    custom_resize_percentage=_cap.resize_percentage,
                    alignment="Middle",
                    overlap_left=True, overlap_right=True,
                    overlap_top=True, overlap_bottom=True,
                )
                result = self._pipe(
                    image=input_prepared,
                    prompt=cfg.prompts.caption_outpaint,
                    true_cfg_scale=_cap.true_cfg_scale,
                    negative_prompt=" ",
                    num_inference_steps=_cap.num_inference_steps,
                    max_sequence_length=_cap.max_sequence_length,
                    height=_cap.height, width=_cap.width,
                ).images[0]
                result.convert("RGB").resize((_cap.width, _cap.height), Image.Resampling.LANCZOS).save(output_path)
                stats["processed"] += 1
            except Exception as e:
                logger.error(f"Failed to outpaint {target_id}: {e}")
                stats["failed"] += 1

        logger.info(f"Caption/outpaint complete: {stats}")
        return stats

    def unload(self):
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="hairport-caption", description="Outpaint bald images")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--bald_subdir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    cfg = get_config()
    data_dir = args.data_dir or "outputs"
    bald_subdir = args.bald_subdir or f"bald/{cfg.pipeline.bald_version}"

    stage = CaptionStage(device=args.device)
    result = stage.run(data_dir=data_dir, bald_subdir=bald_subdir)
    print(f"Caption: {result}")


if __name__ == "__main__":
    main()
