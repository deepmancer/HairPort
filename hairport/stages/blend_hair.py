"""Stage 8 — Blend Hair: Warp and blend hair onto bald heads (Poisson).

Delegates to ``hairport.view_blender`` functions.

Usage::

    # Programmatic
    from hairport.stages.blend_hair import BlendHairStage
    stage = BlendHairStage()
    stage.run(data_dir="outputs", shape_provider="hi3dgen")

    # CLI
    python -m hairport.stages.blend_hair --data_dir outputs --shape_provider hi3dgen
"""

from __future__ import annotations

import argparse
import gc
import logging
import random
import time
from pathlib import Path

import torch

from hairport.config import get_config, add_config_args, load_config_from_args

logger = logging.getLogger(__name__)


class BlendHairStage:
    """Warp, align, and Poisson-blend hair from source onto target.

    Loads: BackgroundRemover, FacialLandmarkDetector, SAMMaskExtractor, FLAMEFitter.
    """

    def run(
        self,
        data_dir: str | Path,
        shape_provider: str | None = None,
        texture_provider: str | None = None,
        bald_version: str | None = None,
    ) -> dict:
        """Run hair blending for all view-aligned folders.

        Returns
        -------
        dict
            Summary with ``processed`` and ``error`` counts.
        """
        from hairport.view_blender import (
            BlendingConfig,
            process_view_aligned_folder,
            flush,
        )
        from hairport.core import (
            BackgroundRemover,
            FacialLandmarkDetector,
            SAMMaskExtractor,
            FLAMEFitter,
        )

        cfg = get_config()
        if shape_provider is None:
            shape_provider = cfg.pipeline.shape_provider
        if texture_provider is None:
            texture_provider = cfg.pipeline.texture_provider
        if bald_version is None:
            bald_version = cfg.pipeline.bald_version

        config = BlendingConfig()
        data_dir = Path(data_dir)
        provider_subdir = f"shape_{shape_provider}__texture_{texture_provider}"
        view_aligned_dir = data_dir / config.DIR_VIEW_ALIGNED / provider_subdir

        if not view_aligned_dir.exists():
            raise FileNotFoundError(f"View aligned directory not found: {view_aligned_dir}")

        all_folders = [f for f in view_aligned_dir.iterdir() if f.is_dir()]
        timestamp_seed = int(time.time())
        random.seed(timestamp_seed)
        random.shuffle(all_folders)
        logger.info(f"BlendHair: {len(all_folders)} folders, seed={timestamp_seed}")

        bald_versions = ["w_seg", "wo_seg"] if bald_version == "all" else [bald_version]
        stats = {"processed": 0, "error": 0}

        # Init models once
        bg_remover = BackgroundRemover()
        landmark_detector = FacialLandmarkDetector(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5,
        )
        sam_extractor = SAMMaskExtractor(confidence_threshold=config.SAM_CONFIDENCE_THRESHOLD)
        flame_fitter = FLAMEFitter()

        try:
            for bv in bald_versions:
                for folder in all_folders:
                    result = process_view_aligned_folder(
                        folder_path=folder,
                        data_dir=data_dir,
                        bald_version=bv,
                        config=config,
                        codeformer_enhancer=None,
                        bg_remover=bg_remover,
                        landmark_detector=landmark_detector,
                        sam_extractor=sam_extractor,
                        flame_fitter=flame_fitter,
                    )
                    stats["processed" if result else "error"] += 1
        finally:
            del bg_remover, landmark_detector, sam_extractor
            flush()

        logger.info(f"BlendHair complete: {stats}")
        return stats


def main(argv: list[str] | None = None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="hairport-blend-hair", description="Blend hair onto bald heads")
    parser.add_argument("--data_dir", type=str, default="outputs/")
    parser.add_argument("--shape_provider", type=str, default=None, choices=["hunyuan", "hi3dgen", "direct3d_s2"])
    parser.add_argument("--texture_provider", type=str, default=None, choices=["hunyuan", "mvadapter"])
    parser.add_argument("--bald_version", type=str, default=None, choices=["wo_seg", "w_seg", "all"])
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    stage = BlendHairStage()
    result = stage.run(
        data_dir=args.data_dir,
        shape_provider=args.shape_provider,
        texture_provider=args.texture_provider,
        bald_version=args.bald_version,
    )
    print(f"BlendHair: {result}")


if __name__ == "__main__":
    main()
