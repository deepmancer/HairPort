"""Command-line interface for bald-konverter.

Usage::

    # Single image
    bald-konverter --input photo.jpg --output bald.png

    # Batch processing
    bald-konverter --input-dir ./faces/ --output-dir ./bald/

    # Fast mode (no segmentation)
    bald-konverter --input photo.jpg --output bald.png --mode wo_seg

    # With FLAME fitting
    bald-konverter --input photo.jpg --output bald.png --use-flame --flame-dir FLAME2020/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger("hairport.bald_konverter")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bald-konverter",
        description="Generate bald versions of portrait images using FLUX LoRA models.",
    )

    # I/O
    io_group = parser.add_mutually_exclusive_group(required=True)
    io_group.add_argument(
        "--input", type=str,
        help="Path to a single input image.",
    )
    io_group.add_argument(
        "--input-dir", type=str,
        help="Directory containing input images (batch mode).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for a single image (default: <input>_bald.<ext>).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for batch mode (default: <input-dir>_bald/).",
    )

    # Pipeline
    parser.add_argument(
        "--mode", type=str, default="auto",
        choices=["wo_seg", "w_seg", "auto"],
        help="Conversion mode (default: auto = wo_seg + w_seg refinement).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--steps", type=int, default=35,
        help="Number of inference steps (default: 35).",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=1.0,
        help="Guidance scale (default: 1.0).",
    )
    parser.add_argument(
        "--strength", type=float, default=1.0,
        help="Inpainting strength (default: 1.0).",
    )

    # Extras
    parser.add_argument(
        "--save-intermediates", action="store_true",
        help="Save intermediate masks and FLUX inputs alongside the output.",
    )
    parser.add_argument(
        "--use-flame", action="store_true",
        help="Use SHeaP FLAME fitting for head segmentation (requires bald-konverter[flame]).",
    )
    parser.add_argument(
        "--flame-dir", type=str, default=None,
        help="Path to FLAME2020/ model directory (only with --use-flame).",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device (default: cuda).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args(argv)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def _save_intermediates(result, stem: str, out_dir: Path) -> None:
    """Save masks and FLUX input alongside the bald image."""
    import numpy as np
    from PIL import Image

    if result.hair_mask is not None:
        Image.fromarray(result.hair_mask).save(out_dir / f"{stem}_hair_mask.png")
    if result.body_mask is not None:
        Image.fromarray(result.body_mask).save(out_dir / f"{stem}_body_mask.png")
    if result.flux_input_wo_seg is not None:
        result.flux_input_wo_seg.save(out_dir / f"{stem}_flux_input_wo_seg.png")
    if result.flux_input_w_seg is not None:
        result.flux_input_w_seg.save(out_dir / f"{stem}_flux_input_w_seg.png")
    if result.foreground is not None:
        result.foreground.save(out_dir / f"{stem}_foreground.png")
    if result.flame_mask is not None:
        Image.fromarray(result.flame_mask).save(out_dir / f"{stem}_flame_mask.png")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from .pipeline import BaldKonverterPipeline

    pipeline = BaldKonverterPipeline(
        mode=args.mode,
        device=args.device,
        use_flame=args.use_flame,
        flame_dir=args.flame_dir,
    )

    # ---- Single image -------------------------------------------------------
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            sys.exit(1)

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.with_name(f"{input_path.stem}_bald{input_path.suffix}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Processing %s → %s", input_path, output_path)
        result = pipeline(
            image=input_path,
            seed=args.seed,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            return_intermediates=args.save_intermediates,
        )
        result.bald_image.save(output_path)
        logger.info("Saved %s", output_path)

        if args.save_intermediates:
            _save_intermediates(result, input_path.stem, output_path.parent)

    # ---- Batch mode ---------------------------------------------------------
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            logger.error("Input directory not found: %s", input_dir)
            sys.exit(1)

        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = input_dir.parent / f"{input_dir.name}_bald"

        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            p for p in input_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not image_files:
            logger.warning("No images found in %s", input_dir)
            sys.exit(0)

        logger.info("Processing %d images from %s → %s", len(image_files), input_dir, output_dir)

        for img_path in tqdm(image_files, desc="Bald-converting", unit="img"):
            out_path = output_dir / f"{img_path.stem}.png"
            if out_path.exists():
                logger.debug("Skipping %s (already exists)", out_path.name)
                continue

            try:
                result = pipeline(
                    image=img_path,
                    seed=args.seed,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    strength=args.strength,
                    return_intermediates=args.save_intermediates,
                )
                result.bald_image.save(out_path)

                if args.save_intermediates:
                    _save_intermediates(result, img_path.stem, output_dir)

            except Exception:
                logger.exception("Failed to process %s", img_path.name)

    pipeline.teardown()
    logger.info("Done.")


if __name__ == "__main__":
    main()
