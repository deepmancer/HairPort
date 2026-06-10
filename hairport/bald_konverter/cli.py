"""Command-line interface for bald-konverter.

Usage::

    bald-konverter --input photo.jpg --output bald.png          # w_seg/auto run the SMPL-X fit
    bald-konverter --input-dir ./faces/ --output-dir ./bald/    # batch
    bald-konverter --input photo.jpg --output bald.png --mode wo_seg   # fast, no fit
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
        help="Embedded guidance scale (default: 1.0, matching LoRA training).",
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
        "--device", type=str, default="cuda",
        help="Compute device (default: cuda).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging.",
    )

    return parser.parse_args(argv)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def _save_head_fit(result, stem: str, out_dir: Path) -> None:
    """Persist the fitted head model (independent of --save-intermediates)."""
    if result.head_fit is not None:
        result.head_fit.save(out_dir / f"{stem}_head_fit.pt")


def _save_intermediates(result, stem: str, out_dir: Path) -> None:
    """Save all intermediates (plate, bald plate, alpha, masks, FLUX grids,
    framing JSON, SMPL-X fit) + a ``{stem}_manifest.json`` to reconstruct the edit."""
    import json
    import numpy as np
    from PIL import Image

    written: dict[str, str] = {}

    def _save_img(arr_or_img, suffix: str):
        if arr_or_img is None:
            return
        name = f"{stem}_{suffix}"
        if isinstance(arr_or_img, np.ndarray):
            Image.fromarray(arr_or_img).save(out_dir / f"{name}.png")
        else:
            arr_or_img.save(out_dir / f"{name}.png")
        written[suffix] = f"{name}.png"

    _save_img(result.plate, "plate")
    _save_img(result.bald_plate, "bald_plate")
    _save_img(result.change_alpha, "alpha")
    _save_img(result.bald_image_wo_seg, "bald_wo_seg")
    _save_img(result.hair_mask, "hair_mask")
    _save_img(result.body_mask, "body_mask")
    _save_img(result.head_mask, "head_mask")
    _save_img(result.smplx_body_mask, "smplx_body_mask")
    _save_img(result.flux_input_wo_seg, "flux_input_wo_seg")
    _save_img(result.flux_input_w_seg, "flux_input_w_seg")
    _save_img(result.foreground, "foreground")

    if result.framing is not None:
        result.framing.to_json(out_dir / f"{stem}_framing.json")
        written["framing"] = f"{stem}_framing.json"
    # head_fit (.pt) is written by _save_head_fit (always, before this call).
    if result.head_fit is not None:
        written["head_fit"] = f"{stem}_head_fit.pt"

    manifest = {
        "stem": stem,
        "output": f"{stem}.png",
        "comp_params": result.comp_params,
        "framing": result.framing.to_dict() if result.framing is not None else None,
        "artifacts": written,
    }
    (out_dir / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


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

        _save_head_fit(result, input_path.stem, output_path.parent)
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

                _save_head_fit(result, img_path.stem, output_dir)
                if args.save_intermediates:
                    _save_intermediates(result, img_path.stem, output_dir)

            except Exception:
                logger.exception("Failed to process %s", img_path.name)

    pipeline.teardown()
    logger.info("Done.")


if __name__ == "__main__":
    main()
