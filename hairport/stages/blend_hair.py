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
from hairport.data import DatasetManager
from hairport.runtime import StageSummary, build_provenance, can_reuse_artifact, derive_seed, write_provenance

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
        seed: int | None = None,
    ) -> StageSummary:
        """Run hair blending for all view-aligned folders.

        Returns
        -------
        StageSummary
            Summary with blended artifacts and per-pair failures.
        """
        from hairport.view_blender import (
            BlendingConfig,
            process_view_aligned_folder,
            flush,
        )
        from hairport.core import (
            BackgroundRemover,
            CodeFormerEnhancer,
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
        dm = DatasetManager(data_dir)
        view_aligned_dir = dm.view_aligned_root(shape_provider, texture_provider)

        if not view_aligned_dir.exists():
            raise FileNotFoundError(f"View aligned directory not found: {view_aligned_dir}")

        all_folders = [
            (target_id, source_id, dm.transfer_dir(target_id, source_id, shape_provider, texture_provider))
            for target_id, source_id in dm.list_pairs(shape_provider, texture_provider)
        ]
        timestamp_seed = seed if seed is not None and seed >= 0 else int(time.time())
        random.seed(timestamp_seed)
        random.shuffle(all_folders)
        logger.info(f"BlendHair: {len(all_folders)} folders, seed={timestamp_seed}")

        bald_versions = ["w_seg", "wo_seg"] if bald_version == "all" else [bald_version]
        summary = StageSummary(metadata={"seed": timestamp_seed, "bald_versions": bald_versions})
        output_phase = int(cfg.enhance_view.conditioning_phase)
        unaware_target_image_folder = "image_outpainted" if data_dir.name.lower() == "celeba_reduced" else "image"

        # Init models once
        bg_remover = BackgroundRemover()
        codeformer_device = cfg.device
        if not torch.cuda.is_available() and "cuda" in str(codeformer_device):
            codeformer_device = "cpu"
        codeformer_enhancer = CodeFormerEnhancer(device=codeformer_device, ultrasharp=True)
        landmark_detector = FacialLandmarkDetector(
            static_image_mode=True, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5,
        )
        sam_extractor = SAMMaskExtractor(confidence_threshold=config.SAM_CONFIDENCE_THRESHOLD)
        flame_fitter = FLAMEFitter()

        try:
            for bv in bald_versions:
                for target_id, source_id, folder in all_folders:
                    summary.attempted += 1
                    item_id = f"{folder.name}:{bv}"
                    item_seed = derive_seed(timestamp_seed, "blend_hair", folder.name, bv, operation_name="blend")
                    source_outpainted = dm.source_outpainted_file(
                        target_id, source_id, bv, shape_provider, texture_provider
                    )
                    camera_params = dm.camera_params_file(
                        target_id, source_id, bv, shape_provider, texture_provider
                    )
                    modes_to_track = ["3d_unaware"]
                    if camera_params.exists():
                        modes_to_track.append("3d_aware")

                    output_paths = {
                        mode: dm.poisson_blended_file(
                            target_id, source_id, bv, mode, shape_provider, texture_provider
                        )
                        for mode in modes_to_track
                    }
                    target_image_path_unaware = data_dir / unaware_target_image_folder / f"{target_id}.png"
                    aware_input_path = dm.enhanced_view_file(
                        target_id, source_id, bv, shape_provider, texture_provider, phase=output_phase
                    )
                    legacy_unaware_landmarks = (
                        data_dir / f"{unaware_target_image_folder}_lmk" / target_id / "landmarks.npy"
                    )
                    provenance_by_mode = {
                        "3d_aware": build_provenance(
                            "blend_hair", seed=item_seed,
                            inputs=[aware_input_path, source_outpainted, camera_params],
                            metadata={
                                "bald_version": bv,
                                "folder": folder.name,
                                "mode": "3d_aware",
                                "conditioning_phase": output_phase,
                            },
                        ),
                        "3d_unaware": build_provenance(
                            "blend_hair", seed=item_seed,
                            inputs=[
                                target_image_path_unaware,
                                source_outpainted,
                                dm.landmarks_file(target_id),
                                legacy_unaware_landmarks,
                            ],
                            metadata={"bald_version": bv, "folder": folder.name, "mode": "3d_unaware"},
                        ),
                    }
                    if all(
                        can_reuse_artifact(output_paths[mode], provenance_by_mode[mode])
                        for mode in modes_to_track
                    ):
                        summary.skipped += 1
                        continue
                    try:
                        result = process_view_aligned_folder(
                            folder_path=folder,
                            data_dir=data_dir,
                            bald_version=bv,
                            config=config,
                            codeformer_enhancer=codeformer_enhancer,
                            bg_remover=bg_remover,
                            landmark_detector=landmark_detector,
                            sam_extractor=sam_extractor,
                            flame_fitter=flame_fitter,
                            force_recompute=True,
                        )
                        if not result:
                            summary.add_failure(item_id, "Blending did not produce an output.")
                            continue

                        produced_modes = [
                            mode for mode in modes_to_track if output_paths[mode].exists()
                        ]
                        missing_modes = [mode for mode in modes_to_track if mode not in produced_modes]
                        if missing_modes:
                            if produced_modes:
                                for mode in produced_modes:
                                    write_provenance(output_paths[mode], provenance_by_mode[mode])
                                summary.add_artifacts([output_paths[mode] for mode in produced_modes])
                            summary.add_failure(
                                item_id,
                                "Blending completed with missing expected outputs for mode(s): "
                                + ", ".join(missing_modes),
                            )
                            continue

                        for mode in modes_to_track:
                            write_provenance(output_paths[mode], provenance_by_mode[mode])
                        summary.add_artifacts([output_paths[mode] for mode in modes_to_track])
                        summary.completed += 1
                    except Exception as e:
                        logger.error(f"BlendHair error on {folder.name} ({bv}): {e}")
                        summary.add_failure(item_id, e)
        finally:
            del bg_remover, codeformer_enhancer, landmark_detector, sam_extractor
            flush()

        logger.info(f"BlendHair complete: {summary.to_dict()}")
        return summary


def main(argv: list[str] | None = None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="hairport-blend-hair", description="Blend hair onto bald heads")
    parser.add_argument("--data_dir", type=str, default="outputs/")
    parser.add_argument("--shape_provider", type=str, default=None, choices=["hunyuan", "hi3dgen", "direct3d_s2"])
    parser.add_argument("--texture_provider", type=str, default=None, choices=["hunyuan", "mvadapter"])
    parser.add_argument("--bald_version", type=str, default=None, choices=["wo_seg", "w_seg", "all"])
    parser.add_argument("--seed", type=int, default=None)
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    stage = BlendHairStage()
    result = stage.run(
        data_dir=args.data_dir,
        shape_provider=args.shape_provider,
        texture_provider=args.texture_provider,
        bald_version=args.bald_version,
        seed=args.seed,
    )
    print(f"BlendHair: {result}")


if __name__ == "__main__":
    main()
