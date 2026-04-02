"""Stage 1 — Baldify: Generate bald versions of portrait images.

Delegates to ``bald_konverter.pipeline.BaldKonverterPipeline``.

Usage::

    # Programmatic
    from hairport.stages.baldify import BaldifyStage
    stage = BaldifyStage()
    stage.run(data_dir="outputs")                          # pipeline mode
    stage.run(input_path="photo.png", output_path="bald.png")  # single image

    # CLI
    python -m hairport.stages.baldify --data_dir outputs
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hairport.config import get_config, add_config_args, load_config_from_args

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class BaldifyStage:
    """Generate bald portraits using bald_konverter.

    Parameters
    ----------
    mode : str
        Conversion mode: ``"auto"``, ``"wo_seg"``, or ``"w_seg"``.
    device : str
        Compute device.
    use_flame : bool
        Whether to use SHeaP FLAME fitting for head segmentation.
    flame_dir : str | Path | None
        Path to FLAME2020/ model directory (only with *use_flame*).
    """

    def __init__(
        self,
        mode: str | None = None,
        device: str | None = None,
        use_flame: bool = False,
        flame_dir: str | Path | None = None,
    ):
        cfg = get_config()
        self.mode = mode if mode is not None else cfg.baldify.mode
        self.device = device if device is not None else cfg.device
        self.use_flame = use_flame
        self.flame_dir = flame_dir
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            from hairport.bald_konverter.pipeline import BaldKonverterPipeline

            self._pipeline = BaldKonverterPipeline(
                mode=self.mode,
                device=self.device,
                use_flame=self.use_flame,
                flame_dir=self.flame_dir,
            )

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
    ) -> dict:
        """Run baldification.

        Supports three invocation modes:

        1. **data_dir** (pipeline mode): reads from ``<data_dir>/image/``
           and writes to ``<data_dir>/bald/<bald_version>/image/``.
        2. **input_dir / output_dir** (explicit batch).
        3. **input_path / output_path** (single image).

        Returns
        -------
        dict
            Summary with ``processed``, ``skipped``, ``failed`` counts.
        """
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

        self._ensure_pipeline()

        # Single-image mode
        if input_path:
            input_path = Path(input_path)
            if output_path is None:
                output_path = input_path.with_name(f"{input_path.stem}_bald.png")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                result = self._pipeline(
                    image=input_path, seed=seed,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale, strength=strength,
                )
                result.bald_image.save(output_path)
                return {"processed": 1, "skipped": 0, "failed": 0}
            except Exception as e:
                logger.error(f"Failed: {e}")
                return {"processed": 0, "skipped": 0, "failed": 1}

        # Resolve input_dir / output_dir from data_dir when needed
        if data_dir is not None and input_dir is None:
            data_dir = Path(data_dir)
            input_dir = data_dir / "image"
            output_dir = data_dir / "bald" / bald_version / "image"

        if not input_dir:
            raise ValueError("Provide data_dir, input_dir, or input_path")

        input_dir = Path(input_dir)
        if output_dir is None:
            output_dir = input_dir.parent / f"{input_dir.name}_bald"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        logger.info(f"Found {len(images)} images in {input_dir}")

        stats = {"processed": 0, "skipped": 0, "failed": 0}
        for img_path in images:
            out_path = output_dir / f"{img_path.stem}.png"
            if out_path.exists():
                stats["skipped"] += 1
                continue
            try:
                result = self._pipeline(
                    image=img_path, seed=seed,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale, strength=strength,
                )
                result.bald_image.save(out_path)
                stats["processed"] += 1
            except Exception as e:
                logger.error(f"Failed to process {img_path.name}: {e}")
                stats["failed"] += 1

        logger.info(f"Baldify complete: {stats}")
        return stats

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
                        choices=["wo_seg", "w_seg"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--use-flame", action="store_true")
    parser.add_argument("--flame-dir", type=str, default=None)
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    stage = BaldifyStage(
        mode=args.mode, device=args.device,
        use_flame=args.use_flame, flame_dir=args.flame_dir,
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
