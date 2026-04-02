"""Stage 9 — Transfer Hair: Final hair transfer using FLUX.2 Klein 9B.

Delegates to ``hairport.postprocessing.restore_hair_klein``.

Usage::

    # Programmatic
    from hairport.stages.transfer_hair import TransferHairStage
    stage = TransferHairStage()
    stage.run(data_dir="outputs", shape_provider="hi3dgen")

    # CLI
    python -m hairport.stages.transfer_hair --data_dir outputs --shape_provider hi3dgen
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from hairport.config import get_config, add_config_args, load_config_from_args

logger = logging.getLogger(__name__)


class TransferHairStage:
    """Transfer hair using FLUX.2 Klein 9B with 3D-aware and 3D-unaware modes.

    Parameters
    ----------
    seed : int
        Random seed.
    num_inference_steps : int
        Klein inference steps (default 4).
    guidance_scale : float
        Guidance scale (default 1.0 for distilled model).
    """

    def __init__(
        self,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ):
        cfg = get_config()
        self.seed = seed if seed is not None else cfg.transfer_hair.seed
        self.num_inference_steps = (
            num_inference_steps if num_inference_steps is not None
            else cfg.transfer_hair.num_inference_steps
        )
        self.guidance_scale = (
            guidance_scale if guidance_scale is not None
            else cfg.transfer_hair.guidance_scale
        )

    def run(
        self,
        data_dir: str | Path,
        shape_provider: str | None = None,
        texture_provider: str | None = None,
        bald_version: str | None = None,
        skip_existing: bool = True,
        use_blending: bool = False,
    ) -> dict:
        """Run batch hair transfer on all view-aligned folders.

        Returns
        -------
        dict
            Summary with processed counts.
        """
        from hairport.postprocessing.restore_hair_klein import (
            HairTransferKleinConfig,
            process_view_aligned_folders,
        )

        cfg = get_config()
        if shape_provider is None:
            shape_provider = cfg.pipeline.shape_provider
        if texture_provider is None:
            texture_provider = cfg.pipeline.texture_provider
        if bald_version is None:
            bald_version = cfg.pipeline.bald_version

        config = HairTransferKleinConfig()
        config.SEED = self.seed
        config.NUM_INFERENCE_STEPS = self.num_inference_steps
        config.GUIDANCE_SCALE = self.guidance_scale

        logger.info(f"TransferHair: data_dir={data_dir}, shape={shape_provider}, "
                     f"texture={texture_provider}, bald={bald_version}")

        results = process_view_aligned_folders(
            data_dir=str(data_dir),
            shape_provider=shape_provider,
            texture_provider=texture_provider,
            config=config,
            skip_existing=skip_existing,
            bald_version=bald_version,
            use_blending=use_blending,
        )

        logger.info(f"TransferHair complete: {results}")
        return results

    def run_single(
        self,
        source: str | Path,
        reference: str | Path,
        output: str | Path,
        view_aligned: str | Path | None = None,
    ) -> dict:
        """Run single-pair hair transfer.

        Returns
        -------
        dict
            ``{"3d_aware": path_or_None, "3d_unaware": path_or_None}``
        """
        import os
        from hairport.postprocessing.restore_hair_klein import (
            HairTransferKleinConfig,
            HairTransferKleinPipeline,
            extract_hair_mask,
        )

        config = HairTransferKleinConfig()
        config.SEED = self.seed
        config.NUM_INFERENCE_STEPS = self.num_inference_steps
        config.GUIDANCE_SCALE = self.guidance_scale

        pipeline = HairTransferKleinPipeline(config)
        output_dir = Path(output)
        results = {}

        try:
            modes = ["3d_unaware"]
            if view_aligned is not None:
                modes.append("3d_aware")

            for mode in modes:
                is_3d = mode == "3d_aware"
                mode_dir = output_dir / mode
                mode_dir.mkdir(parents=True, exist_ok=True)

                result = pipeline.transfer_hair(
                    source_bald_image=str(source),
                    reference_image=str(reference),
                    view_aligned_image=str(view_aligned) if is_3d else None,
                    use_3d_aware=is_3d,
                    output_dir=str(mode_dir),
                )

                mask, score = extract_hair_mask(result)
                mask.save(str(mode_dir / config.FILE_HAIR_RESTORED_MASK))
                results[mode] = str(mode_dir / config.FILE_HAIR_RESTORED)
        finally:
            pipeline.unload()

        return results


def main(argv: list[str] | None = None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="hairport-transfer-hair", description="Hair transfer with FLUX.2 Klein 9B")
    parser.add_argument("--mode", type=str, choices=["single", "batch"], default="batch")

    # Single mode
    parser.add_argument("--source", type=str, help="Source bald image")
    parser.add_argument("--reference", type=str, help="Reference hair image")
    parser.add_argument("--view_aligned", type=str, help="View-aligned image (3D-aware)")
    parser.add_argument("--output", type=str, help="Output directory")

    # Batch mode
    parser.add_argument("--data_dir", type=str, default="outputs")
    parser.add_argument("--shape_provider", type=str, default=None, choices=["hunyuan", "hi3dgen", "direct3d_s2"])
    parser.add_argument("--texture_provider", type=str, default=None, choices=["hunyuan", "mvadapter"])
    parser.add_argument("--bald_version", type=str, default=None, choices=["w_seg", "wo_seg", "all"])
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    parser.add_argument("--use_blending", action="store_true", default=False)

    # Pipeline params
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    stage = TransferHairStage(
        seed=args.seed,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
    )

    if args.mode == "single":
        if not all([args.source, args.reference, args.output]):
            parser.error("Single mode requires --source, --reference, and --output")
        result = stage.run_single(
            source=args.source,
            reference=args.reference,
            output=args.output,
            view_aligned=args.view_aligned,
        )
    else:
        result = stage.run(
            data_dir=args.data_dir,
            shape_provider=args.shape_provider,
            texture_provider=args.texture_provider,
            bald_version=args.bald_version,
            skip_existing=args.skip_existing,
            use_blending=args.use_blending,
        )

    print(f"TransferHair: {result}")


if __name__ == "__main__":
    main()
