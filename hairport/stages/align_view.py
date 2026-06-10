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
from hairport.data import DatasetManager
from hairport.runtime import (
    StageSummary,
    build_provenance,
    can_reuse_artifact,
    derive_seed,
    write_provenance,
)

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
        seed: int | None = None,
    ) -> StageSummary:
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
        random_seed = seed if seed is not None and seed >= 0 else int(time.time())
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        logger.info(f"AlignView: seed={random_seed}")

        # Prepare pairs
        dm = DatasetManager(data_dir)
        pairs = prepare_pairs(
            data_dir, config, pairs_csv_file,
            decision_manifest_path=dm.pair_decisions_file(shape_provider, texture_provider),
        )
        decisions_path = dm.pair_decisions_file(shape_provider, texture_provider)
        decisions_provenance = build_provenance(
            "align_view_pair_decisions", seed=random_seed,
            inputs=[Path(pairs_csv_file) if pairs_csv_file else Path(data_dir) / "pairs.csv"],
            metadata={"shape_provider": shape_provider, "texture_provider": texture_provider},
        )
        write_provenance(decisions_path, decisions_provenance)
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

        summary = StageSummary(
            metadata={"seed": random_seed, "pair_decisions": str(decisions_path)}
        )

        # Phase 1: Outpainting
        for bv in bald_versions:
            for target_id, source_id, lift_3d in pairs:
                summary.attempted += 1
                item_id = f"{target_id}_to_{source_id}:{bv}:outpaint"
                source_input = Path(data_dir) / "bald" / bv / "image" / f"{source_id}.png"
                outpainted = (dm.transfer_dir(target_id, source_id, shape_provider, texture_provider)
                              / bv / config.DIR_SRC_OUTPAINTED / "outpainted_image.png")
                outpaint_bundle = [
                    outpainted,
                    outpainted.parent / "landmarks.npy",
                    outpainted.parent / "resize_info.json",
                ]
                item_seed = derive_seed(random_seed, "align_view", f"{target_id}_to_{source_id}", bv, operation_name="outpaint")
                provenance = build_provenance(
                    "align_view_outpaint", seed=item_seed,
                    inputs=[
                        source_input,
                        Path(data_dir) / "prompt" / f"{source_id}.json",
                    ],
                    metadata={"enable_outpainting": enable_outpainting},
                )
                if all(can_reuse_artifact(path, provenance) for path in outpaint_bundle):
                    summary.skipped += 1
                    continue
                try:
                    random.seed(item_seed)
                    np.random.seed(item_seed)
                    torch.manual_seed(item_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(item_seed)
                    was_computed = compute_outpainting(
                        data_dir=data_dir, target_id=target_id, source_id=source_id,
                        shape_provider=shape_provider, texture_provider=texture_provider,
                        bald_version=bv, config=config, uncropper=uncropper,
                        facial_landmark_detector=facial_landmark_detector,
                        enable_outpainting=enable_outpainting,
                        force_recompute=True,
                        seed=item_seed,
                    )
                    if was_computed:
                        for path in outpaint_bundle:
                            write_provenance(path, provenance)
                        summary.completed += 1
                        summary.add_artifacts(outpaint_bundle)
                    else:
                        summary.skipped += 1
                except Exception as e:
                    logger.error(f"Outpainting error {target_id}->{source_id}: {e}")
                    summary.add_failure(item_id, e)

        # Release uncropper
        if uncropper is not None:
            del uncropper
            torch.cuda.empty_cache()
            gc.collect()

        # Phase 2: Camera optimisation (3D lifting)
        lift_pairs = [(t, s, l) for t, s, l in pairs if l]
        self._validate_3d_lift_inputs(
            data_dir=Path(data_dir),
            shape_provider=shape_provider,
            texture_provider=texture_provider,
            lift_pairs=lift_pairs,
            config=config,
        )
        for bv in bald_versions:
            for target_id, source_id, lift_3d in lift_pairs:
                summary.attempted += 1
                item_id = f"{target_id}_to_{source_id}:{bv}:camera"
                camera_output = dm.camera_params_file(
                    target_id, source_id, bv, shape_provider, texture_provider
                )
                camera_source_outpainted = (
                    dm.source_outpainted_dir(
                        target_id, source_id, bv, shape_provider, texture_provider
                    ) / "outpainted_image.png"
                )
                camera_source_lmk = camera_source_outpainted.parent / "landmarks.npy"
                camera_resize_info = camera_source_outpainted.parent / "resize_info.json"
                item_seed = derive_seed(random_seed, "align_view", f"{target_id}_to_{source_id}", bv, operation_name="camera")
                provenance = build_provenance(
                    "align_view_camera", seed=item_seed,
                    inputs=[Path(data_dir) / "image" / f"{target_id}.png",
                            Path(data_dir) / "image" / f"{source_id}.png",
                            dm.vertex_indices_file(target_id, shape_provider, texture_provider),
                            dm.landmarks_3d_file(target_id, shape_provider, texture_provider),
                            camera_source_outpainted, camera_source_lmk, camera_resize_info],
                    metadata={"shape_provider": shape_provider, "texture_provider": texture_provider},
                )
                if can_reuse_artifact(camera_output, provenance):
                    summary.skipped += 1
                    continue
                try:
                    random.seed(item_seed)
                    np.random.seed(item_seed)
                    torch.manual_seed(item_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(item_seed)
                    was_computed = run_camera_optimization(
                        data_dir=data_dir, target_id=target_id, source_id=source_id,
                        shape_provider=shape_provider, texture_provider=texture_provider,
                        bald_version=bv, debug=debug, config=config,
                        force_recompute=True,
                    )
                    if was_computed:
                        write_provenance(camera_output, provenance)
                        summary.completed += 1
                        summary.add_artifacts([camera_output])
                    else:
                        summary.skipped += 1
                except Exception as e:
                    logger.error(f"Camera opt error {target_id}->{source_id}: {e}")
                    summary.add_failure(item_id, e)

        logger.info(f"AlignView complete: {summary.to_dict()}")
        return summary

    @staticmethod
    def _validate_3d_lift_inputs(
        data_dir: Path,
        shape_provider: str,
        texture_provider: str,
        lift_pairs: list[tuple[str, str, bool]],
        config,
    ) -> None:
        if not lift_pairs:
            return

        provider_subdir = f"shape_{shape_provider}__texture_{texture_provider}"
        lmk_root = data_dir / config.DIR_LANDMARKS_3D / provider_subdir
        if texture_provider == "hunyuan":
            textured_root = data_dir / "hunyuan"
        else:
            textured_root = data_dir / texture_provider / shape_provider

        missing: list[str] = []
        for target_id in sorted({target_id for target_id, _source_id, _lift in lift_pairs}):
            out_dir = lmk_root / target_id
            required = [
                out_dir / config.FILE_TEXTURED_MESH,
                out_dir / "landmarks_3d.npy",
                out_dir / config.FILE_VERTEX_INDICES,
            ]
            absent = [str(path) for path in required if not path.exists()]
            if absent:
                expected_input = textured_root / target_id / "textured_mesh.glb"
                missing.append(
                    f"{target_id}: missing {absent}; expected Landmark3D input {expected_input}"
                )

        if missing:
            details = "\n  ".join(missing)
            raise FileNotFoundError(
                "align_view requires Landmark3D postprocessed textured mesh outputs "
                "for all 3D-lift targets. Missing:\n  "
                f"{details}"
            )

    def unload(self):
        """Stage-API symmetry: models are scoped to run() (the SDXL uncropper
        is already released after Phase 1); this only flushes leftover cache."""
        from hairport import memory

        memory.flush()


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
    parser.add_argument("--seed", type=int, default=None)
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
        seed=args.seed,
    )
    print(f"AlignView: {result}")


if __name__ == "__main__":
    main()
