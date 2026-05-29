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
import json
import logging
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from hairport.config import get_config, add_config_args, load_config_from_args
from hairport.runtime import (
    StageSummary,
    build_provenance,
    can_reuse_artifact,
    derive_seed,
    write_provenance,
)

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
        if not torch.cuda.is_available() and "cuda" in str(self.device):
            self.device = "cpu"
        self.dtype = dtype
        self._pipe = None
        self._captioner = None

    def _ensure_pipeline(self):
        if self._pipe is not None:
            return
        from diffusers import QwenImageEditPipeline
        from hairport.utility.uncrop_qwen import QwenImageTransformer2DModel

        cfg = get_config()
        self._pipe = QwenImageEditPipeline.from_pretrained(
            cfg.models.qwen_image_edit,
            revision=cfg.models.qwen_image_edit_revision,
            torch_dtype=self.dtype,
        ).to(self.device)
        self._pipe.transformer.__class__ = QwenImageTransformer2DModel
        self._pipe.load_lora_weights(
            cfg.models.qwen_lightning_lora,
            weight_name=cfg.models.qwen_lightning_lora_weight,
            revision=cfg.models.qwen_lightning_lora_revision,
        )
        self._pipe.fuse_lora()
        logger.info("Qwen outpainting pipeline loaded")

    def _ensure_captioner(self):
        if self._captioner is not None:
            return
        from hairport.core import CaptionerPipeline

        self._captioner = CaptionerPipeline(dtype=self.dtype, device_map=self.device)
        logger.info("Qwen captioner pipeline loaded")

    def run(
        self,
        data_dir: str | Path,
        bald_subdir: str | None = None,
        bald_version: str | None = None,
        images_subdir: str = "image",
        output_subdir: str = "image_outpainted",
        generate_prompts: bool = True,
        run_outpainting: bool = True,
        seed: int | None = None,
    ) -> StageSummary:
        """Generate prompt JSON and outpaint images in the bald directory.

        Returns
        -------
        StageSummary
            Summary with per-item generation outcomes.
        """
        cfg = get_config()
        if bald_subdir is None:
            if bald_version is None:
                bald_version = cfg.pipeline.bald_version
            bald_subdirs = (
                ["bald/w_seg", "bald/wo_seg"]
                if bald_version == "all"
                else [f"bald/{bald_version}"]
            )
        else:
            bald_subdirs = [bald_subdir]
        data_dir = Path(data_dir)
        if seed is None:
            seed = cfg.seed
        summary = StageSummary(metadata={"bald_subdirs": bald_subdirs})

        if generate_prompts:
            prompt_summary = self._generate_prompt_json(data_dir, seed)
            summary.attempted += prompt_summary.attempted
            summary.completed += prompt_summary.completed
            summary.skipped += prompt_summary.skipped
            summary.failures.extend(prompt_summary.failures)
            summary.artifacts.extend(prompt_summary.artifacts)

        if not run_outpainting:
            logger.info(f"Caption complete: {summary.to_dict()}")
            return summary

        from hairport.utility.uncrop_qwen import prepare_image_and_mask

        self._ensure_pipeline()
        cfg = get_config()
        _cap = cfg.caption

        for current_bald_subdir in bald_subdirs:
            images_dir = data_dir / current_bald_subdir / images_subdir
            output_dir = data_dir / current_bald_subdir / output_subdir
            output_dir.mkdir(parents=True, exist_ok=True)

            if not images_dir.exists():
                logger.warning(f"Bald image directory not found for outpainting: {images_dir}")
                summary.add_failure(current_bald_subdir, f"Missing input directory: {images_dir}")
                continue

            image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
            logger.info(f"Found {len(image_paths)} images to outpaint in {images_dir}")

            for image_path in tqdm(image_paths, desc=f"Outpainting {current_bald_subdir}"):
                summary.attempted += 1
                target_id = image_path.stem
                output_path = output_dir / f"{target_id}.png"
                bv = Path(current_bald_subdir).name
                item_seed = derive_seed(seed, "caption", target_id, bv, operation_name="outpaint")
                provenance = build_provenance(
                    "caption_outpaint", seed=item_seed, inputs=[image_path],
                    metadata={"bald_version": bv},
                )
                if can_reuse_artifact(output_path, provenance):
                    summary.skipped += 1
                    continue
                try:
                    input_image = Image.open(image_path).convert("RGB")
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
                        generator=torch.Generator(device=self.device).manual_seed(item_seed),
                    ).images[0]
                    result.convert("RGB").resize((_cap.width, _cap.height), Image.Resampling.LANCZOS).save(output_path)
                    write_provenance(output_path, provenance)
                    summary.completed += 1
                    summary.add_artifacts([output_path])
                except Exception as e:
                    logger.error(f"Failed to outpaint {target_id}: {e}")
                    summary.add_failure(f"{target_id}:{bv}:outpaint", e)

        logger.info(f"Caption/outpaint complete: {summary.to_dict()}")
        return summary

    def _generate_prompt_json(self, data_dir: Path, seed: int) -> StageSummary:
        image_dir = data_dir / "image"
        prompt_dir = data_dir / "prompt"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        summary = StageSummary()
        if not image_dir.exists():
            logger.warning(f"Image directory not found for captioning: {image_dir}")
            summary.add_failure("prompts", f"Missing input directory: {image_dir}")
            return summary

        image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        logger.info(f"Found {len(image_paths)} images to caption")
        if not image_paths:
            return summary

        self._ensure_captioner()
        for image_path in tqdm(image_paths, desc="Captioning"):
            summary.attempted += 1
            prompt_path = prompt_dir / f"{image_path.stem}.json"
            prompt_types = (
                "description", "description_no_hair", "hair_description", "background"
            )
            operation_seeds = {
                prompt_type: derive_seed(
                    seed, "caption", image_path.stem, operation_name=prompt_type
                )
                for prompt_type in prompt_types
            }
            provenance = build_provenance(
                "caption_prompt", seed=seed, inputs=[image_path],
                metadata={"sampling": "seeded", "operation_seeds": operation_seeds},
            )
            if can_reuse_artifact(prompt_path, provenance):
                summary.skipped += 1
                continue
            try:
                def caption(prompt_type: str) -> str:
                    return self._captioner.caption_image(
                        image_path, prompt_type=prompt_type, seed=operation_seeds[prompt_type]
                    )

                description = caption("description")
                description_no_hair = caption("description_no_hair")
                hair_description = caption("hair_description")
                background = caption("background")
                payload = {
                    "subject": [
                        {
                            "description": description,
                            "description_no_hair": description_no_hair,
                            "hair_description": hair_description,
                        }
                    ],
                    "background": background,
                }
                with open(prompt_path, "w") as f:
                    json.dump(payload, f, indent=2)
                write_provenance(prompt_path, provenance)
                summary.completed += 1
                summary.add_artifacts([prompt_path])
            except Exception as e:
                logger.error(f"Failed to caption {image_path.name}: {e}")
                summary.add_failure(f"{image_path.stem}:caption", e)
        return summary

    def unload(self):
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if self._captioner is not None:
            del self._captioner
            self._captioner = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="hairport-caption", description="Outpaint bald images")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--bald_subdir", type=str, default=None)
    parser.add_argument("--bald_version", type=str, default=None, choices=["w_seg", "wo_seg", "all"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_generate_prompts", action="store_true")
    parser.add_argument("--no_outpainting", action="store_true")
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    cfg = get_config()
    data_dir = args.data_dir or "outputs"
    bald_subdir = args.bald_subdir

    stage = CaptionStage(device=args.device)
    result = stage.run(
        data_dir=data_dir,
        bald_subdir=bald_subdir,
        bald_version=args.bald_version or cfg.pipeline.bald_version,
        generate_prompts=not args.no_generate_prompts,
        run_outpainting=not args.no_outpainting,
        seed=cfg.seed,
    )
    print(f"Caption: {result}")


if __name__ == "__main__":
    main()
