"""Stage 1 — Baldify: bald portraits via ``BaldKonverterPipeline``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from hairport.config import get_config, add_config_args, load_config_from_args
from hairport.runtime import (
    StageSummary,
    build_provenance,
    can_reuse_artifact,
    derive_seed,
    write_provenance,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class BaldifyStage:
    """Generate bald portraits. ``mode``: ``auto``/``wo_seg``/``w_seg``;
    ``persist_intermediates`` defaults to ``cfg.baldify.persist_intermediates``."""

    def __init__(
        self,
        mode: str | None = None,
        device: str | None = None,
        persist_intermediates: bool | None = None,
    ):
        cfg = get_config()
        self.mode = mode if mode is not None else cfg.baldify.mode
        self._mode_explicit = mode is not None
        self.device = device if device is not None else cfg.device
        if not torch.cuda.is_available() and "cuda" in str(self.device):
            self.device = "cpu"
        self.persist_intermediates = (
            persist_intermediates if persist_intermediates is not None
            else bool(cfg.baldify.persist_intermediates)
        )
        self._pipeline = None

    def _resolve_mode(self, bald_version: str) -> str:
        if bald_version not in {"wo_seg", "w_seg"}:
            raise ValueError(f"bald_version must be 'wo_seg' or 'w_seg', got {bald_version!r}")
        if self._mode_explicit and self.mode not in {"auto", bald_version}:
            raise ValueError(
                f"Baldify mode {self.mode!r} does not match output bald_version "
                f"{bald_version!r}. Use matching values to avoid mislabeled outputs."
            )
        return bald_version

    def _ensure_pipeline(self, mode: str):
        if self._pipeline is not None and self.mode != mode:
            self.unload()
        if self._pipeline is None:
            from hairport.bald_konverter.pipeline import BaldKonverterPipeline

            self.mode = mode
            self._pipeline = BaldKonverterPipeline(
                mode=self.mode,
                device=self.device,
            )

    @staticmethod
    def _persist_artifacts(result, path_for) -> None:
        """Write every populated intermediate + a manifest. ``path_for(kind, ext)``
        resolves each destination."""
        import json
        import numpy as np
        from PIL import Image

        written: dict[str, str] = {}

        def _img(obj, kind):
            if obj is None:
                return
            p = path_for(kind, "png")
            p.parent.mkdir(parents=True, exist_ok=True)
            (Image.fromarray(obj) if isinstance(obj, np.ndarray) else obj).save(p)
            written[kind] = p.name

        _img(result.plate, "plate")
        _img(result.bald_plate, "bald_plate")
        _img(result.change_alpha, "alpha")
        _img(result.bald_image_wo_seg, "bald_wo_seg")
        _img(result.hair_mask, "mask_hair")
        _img(result.body_mask, "mask_body")
        _img(result.head_mask, "mask_head")
        _img(result.smplx_body_mask, "mask_smplx")
        _img(result.flux_input_wo_seg, "flux_input")
        _img(result.flux_input_w_seg, "flux_input_w_seg")
        _img(result.foreground, "foreground")

        if result.framing is not None:
            p = path_for("framing", "json")
            p.parent.mkdir(parents=True, exist_ok=True)
            result.framing.to_json(p)
            written["framing"] = p.name
        if result.head_fit is not None:
            p = path_for("head_fit", "pt")
            p.parent.mkdir(parents=True, exist_ok=True)
            result.head_fit.save(p)
            written["head_fit"] = p.name

        man = path_for("manifest", "json")
        man.parent.mkdir(parents=True, exist_ok=True)
        man.write_text(json.dumps({
            "comp_params": result.comp_params,
            "framing": result.framing.to_dict() if result.framing is not None else None,
            "artifacts": written,
        }, indent=2, default=str))

    def run(
        self,
        data_dir: str | Path | None = None,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        bald_version: str | None = None,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        strength: float | None = None,
    ) -> StageSummary:
        """Run baldification via ``data_dir`` (pipeline), ``input_dir``/``output_dir``
        (batch), or ``input_path``/``output_path`` (single). Returns a StageSummary."""
        cfg = get_config()
        if bald_version is None:
            bald_version = cfg.pipeline.bald_version
        if seed is None:
            seed = cfg.baldify.seed
        if num_inference_steps is None:
            num_inference_steps = cfg.baldify.num_inference_steps
        if guidance_scale is None:
            guidance_scale = cfg.baldify.guidance_scale
        if strength is None:
            strength = cfg.baldify.strength

        # Single-image mode
        if input_path:
            resolved_mode = self._resolve_mode(bald_version)
            self._ensure_pipeline(resolved_mode)
            input_path = Path(input_path)
            if output_path is None:
                output_path = input_path.with_name(f"{input_path.stem}_bald.png")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary = StageSummary(attempted=1)
            item_seed = derive_seed(seed, "baldify", input_path.stem, bald_version, operation_name="wo_seg")
            refine_seed = derive_seed(seed, "baldify", input_path.stem, bald_version, operation_name="w_seg")
            provenance = build_provenance(
                "baldify", seed=item_seed, inputs=[input_path],
                metadata={"bald_version": bald_version, "refine_seed": refine_seed},
            )
            try:
                result = self._pipeline(
                    image=input_path, seed=item_seed, refine_seed=refine_seed,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale, strength=strength,
                    return_intermediates=self.persist_intermediates,
                )
                result.bald_image.save(output_path)
                if self.persist_intermediates:
                    stem = input_path.stem
                    self._persist_artifacts(
                        result,
                        lambda kind, ext="png": output_path.parent / f"{stem}_{kind}.{ext}",
                    )
                write_provenance(output_path, provenance)
                summary.completed = 1
                summary.add_artifacts([output_path])
                return summary
            except Exception as e:
                logger.error(f"Failed: {e}")
                summary.add_failure(input_path.stem, e)
                return summary

        # Resolve input_dir from data_dir when needed
        if data_dir is not None and input_dir is None:
            data_dir = Path(data_dir)
            input_dir = data_dir / "image"
            data_dir_for_outputs = data_dir
        else:
            data_dir_for_outputs = None

        if not input_dir:
            raise ValueError("Provide data_dir, input_dir, or input_path")

        input_dir = Path(input_dir)
        images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        logger.info(f"Found {len(images)} images in {input_dir}")

        summary = StageSummary()
        bald_versions = ["w_seg", "wo_seg"] if bald_version == "all" else [bald_version]

        for bv in bald_versions:
            resolved_mode = self._resolve_mode(bv)
            self._ensure_pipeline(resolved_mode)

            dm_for_outputs = None
            if data_dir_for_outputs is not None:
                version_output_dir = data_dir_for_outputs / "bald" / bv / "image"
                dm_for_outputs = DatasetManager(data_dir_for_outputs)
            else:
                if output_dir is not None and len(bald_versions) > 1:
                    raise ValueError("bald_version='all' requires data_dir or per-version output directories")
                version_output_dir = Path(output_dir) if output_dir is not None else input_dir.parent / f"{input_dir.name}_bald"

            version_output_dir.mkdir(parents=True, exist_ok=True)

            for img_path in images:
                summary.attempted += 1
                out_path = version_output_dir / f"{img_path.stem}.png"
                item_id = f"{img_path.stem}:{bv}"
                item_seed = derive_seed(seed, "baldify", img_path.stem, bv, operation_name="wo_seg")
                refine_seed = derive_seed(seed, "baldify", img_path.stem, bv, operation_name="w_seg")
                provenance = build_provenance(
                    "baldify", seed=item_seed, inputs=[img_path],
                    metadata={"bald_version": bv, "refine_seed": refine_seed},
                )
                if can_reuse_artifact(out_path, provenance):
                    summary.skipped += 1
                    continue
                try:
                    result = self._pipeline(
                        image=img_path, seed=item_seed, refine_seed=refine_seed,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale, strength=strength,
                        return_intermediates=self.persist_intermediates,
                    )
                    result.bald_image.save(out_path)
                    if self.persist_intermediates:
                        stem = img_path.stem
                        if dm_for_outputs is not None:
                            path_for = (
                                lambda kind, ext="png", _s=stem, _bv=bv:
                                dm_for_outputs.bald_artifact(_s, kind, _bv, ext)
                            )
                        else:
                            path_for = (
                                lambda kind, ext="png", _s=stem:
                                version_output_dir / f"{_s}_{kind}.{ext}"
                            )
                        self._persist_artifacts(result, path_for)
                    write_provenance(out_path, provenance)
                    summary.completed += 1
                    summary.add_artifacts([out_path])
                except Exception as e:
                    logger.error(f"Failed to process {img_path.name} ({bv}): {e}")
                    summary.add_failure(item_id, e)

        summary.metadata["bald_versions"] = bald_versions
        logger.info(f"Baldify complete: {summary.to_dict()}")
        return summary

    def unload(self):
        """Release GPU resources."""
        if self._pipeline is not None:
            self._pipeline.teardown()
            self._pipeline = None


def main(argv: list[str] | None = None):
    """CLI entry point for baldify stage."""
    parser = argparse.ArgumentParser(
        prog="hairport-baldify",
        description="Generate bald versions of portrait images.",
    )
    parser.add_argument("--data_dir", type=str, help="Root data directory (pipeline mode)")
    parser.add_argument("--input-dir", type=str, help="Input image directory (batch mode)")
    parser.add_argument("--output-dir", type=str, help="Output directory (batch mode)")
    parser.add_argument("--input", type=str, help="Single input image path")
    parser.add_argument("--output", type=str, help="Single output image path")
    parser.add_argument(
        "--mode", type=str, default=None,
        choices=["wo_seg", "w_seg", "auto"],
    )
    parser.add_argument("--bald_version", type=str, default=None,
                        choices=["wo_seg", "w_seg", "all"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--strength", type=float, default=None)
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    stage = BaldifyStage(
        mode=args.mode, device=args.device,
    )
    result = stage.run(
        data_dir=args.data_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_path=args.input,
        output_path=args.output,
        bald_version=args.bald_version,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
    )
    print(f"Baldify: {result}")


if __name__ == "__main__":
    main()
