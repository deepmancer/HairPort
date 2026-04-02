"""Stage 5 — Align View: Align target hairstyle to source view.

Delegates to ``hairport.view_aligner`` functions.

Usage::

    # Programmatic
    from hairport.stages.align_view import AlignViewStage
    stage = AlignViewStage()
    stage.run(data_dir="outputs", shape_provider="hi3dgen")

    # CLI
    python -m hairport.stages.align_view --data_dir outputs --shape_provider hi3dgen
"""

from __future__ import annotations

import argparse
import gc
import logging
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from hairport.config import get_config, add_config_args, load_config_from_args

logger = logging.getLogger(__name__)


class AlignViewStage:
    """Align target hairstyle to source view using landmark optimisation.

    This stage:
    1. Computes outpainting for source images (Phase 1)
    2. Runs camera optimisation for 3D lifting (Phase 2)

    Models loaded: BackgroundRemover, FacialLandmarkDetector, Uncropper (optional).
    """

    def run(
        self,
        data_dir: str | Path,
        shape_provider: str | None = None,
        texture_provider: str | None = None,
        bald_version: str | None = None,
        pairs_csv_file: str | None = None,
        enable_outpainting: bool = False,
        debug: bool = False,
    ) -> dict:
        """Run view alignment.

        Returns
        -------
        dict
            Summary with phase-level counts.
        """
        from hairport.view_aligner import (
            Config, prepare_pairs,
            compute_outpainting, run_camera_optimization,
        )
        from hairport.core import BackgroundRemover, FacialLandmarkDetector

        cfg = get_config()
        if shape_provider is None:
            shape_provider = cfg.pipeline.shape_provider
        if texture_provider is None:
            texture_provider = cfg.pipeline.texture_provider
        if bald_version is None:
            bald_version = cfg.pipeline.bald_version

        config = Config()
        data_dir = str(data_dir)

        # Seed
        random_seed = int(time.time())
        random.seed(random_seed)
        np.random.seed(random_seed)
        logger.info(f"AlignView: seed={random_seed}")

        # Prepare pairs
        pairs = prepare_pairs(data_dir, config, pairs_csv_file)
        random.shuffle(pairs)
        logger.info(f"AlignView: {len(pairs)} pairs")

        bald_versions = ["w_seg", "wo_seg"] if bald_version == "all" else [bald_version]

        # Models
        facial_landmark_detector = FacialLandmarkDetector(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5,
        )

        uncropper = None
        if enable_outpainting:
            from hairport.utility.uncrop_sdxl.uncrop_sdxl import Uncropper
            uncropper = Uncropper()
            uncropper.load_pipeline()

        stats = {
            "outpaint_processed": 0, "outpaint_skipped": 0, "outpaint_failed": 0,
            "camera_processed": 0, "camera_skipped": 0, "camera_failed": 0,
        }

        # Phase 1: Outpainting
        for bv in bald_versions:
            for target_id, source_id, lift_3d in pairs:
                try:
                    was_computed = compute_outpainting(
                        data_dir=data_dir, target_id=target_id, source_id=source_id,
                        shape_provider=shape_provider, texture_provider=texture_provider,
                        bald_version=bv, config=config, uncropper=uncropper,
                        facial_landmark_detector=facial_landmark_detector,
                        enable_outpainting=enable_outpainting,
                    )
                    stats["outpaint_processed" if was_computed else "outpaint_skipped"] += 1
                except Exception as e:
                    logger.error(f"Outpainting error {target_id}->{source_id}: {e}")
                    stats["outpaint_failed"] += 1

        # Release uncropper
        if uncropper is not None:
            del uncropper
            torch.cuda.empty_cache()
            gc.collect()

        # Phase 2: Camera optimisation (3D lifting)
        lift_pairs = [(t, s, l) for t, s, l in pairs if l]
        for bv in bald_versions:
            for target_id, source_id, lift_3d in lift_pairs:
                try:
                    was_computed = run_camera_optimization(
                        data_dir=data_dir, target_id=target_id, source_id=source_id,
                        shape_provider=shape_provider, texture_provider=texture_provider,
                        bald_version=bv, debug=debug, config=config,
                    )
                    stats["camera_processed" if was_computed else "camera_skipped"] += 1
                except Exception as e:
                    logger.error(f"Camera opt error {target_id}->{source_id}: {e}")
                    stats["camera_failed"] += 1

        logger.info(f"AlignView complete: {stats}")
        return stats


def main(argv: list[str] | None = None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="hairport-align-view", description="Align target to source view")
    parser.add_argument("--data_dir", type=str, default="outputs/")
    parser.add_argument("--shape_provider", type=str, default=None, choices=["hunyuan", "hi3dgen", "direct3d_s2"])
    parser.add_argument("--texture_provider", type=str, default=None, choices=["hunyuan", "mvadapter"])
    parser.add_argument("--bald_version", type=str, default=None, choices=["wo_seg", "w_seg", "all"])
    parser.add_argument("--pairs_csv_file", type=str, default=None)
    parser.add_argument("--enable_outpainting", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    stage = AlignViewStage()
    result = stage.run(
        data_dir=args.data_dir,
        shape_provider=args.shape_provider,
        texture_provider=args.texture_provider,
        bald_version=args.bald_version,
        pairs_csv_file=args.pairs_csv_file,
        enable_outpainting=args.enable_outpainting,
        debug=args.debug,
    )
    print(f"AlignView: {result}")


if __name__ == "__main__":
    main()
