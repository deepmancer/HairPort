"""Stage 7 — Enhance View: Refine rendered views with FLUX.2 Klein + CodeFormer.

Delegates to ``hairport.view_enhancer.ViewEnhancer``.

Usage::

    # Programmatic
    from hairport.stages.enhance_view import EnhanceViewStage
    stage = EnhanceViewStage()
    stage.run(data_dir="outputs", shape_provider="hi3dgen")

    # CLI
    python -m hairport.stages.enhance_view --data_dir outputs --shape_provider hi3dgen
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
from hairport.runtime import StageSummary, build_provenance, can_reuse_artifact, derive_seed

logger = logging.getLogger(__name__)


class EnhanceViewStage:
    """Enhance rendered multi-view images.

    Uses FLUX.2 Klein 9B for img2img refinement and CodeFormer
    for face super-resolution.

    Parameters
    ----------
    device : str
        Compute device.
    seed : int
        Random seed (-1 for timestamp-based).
    num_inference_steps : int
        Number of Klein inference steps.
    guidance_scale : float
        Guidance scale for Klein.
    """

    def __init__(
        self,
        device: str | None = None,
        seed: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
    ):
        cfg = get_config()
        self.device = device if device is not None else cfg.device
        if not torch.cuda.is_available() and "cuda" in str(self.device):
            self.device = "cpu"
        self.seed = seed if seed is not None else cfg.seed
        self.num_inference_steps = (
            num_inference_steps if num_inference_steps is not None
            else cfg.enhance_view.num_inference_steps
        )
        self.guidance_scale = (
            guidance_scale if guidance_scale is not None
            else cfg.enhance_view.guidance_scale
        )
        self._enhancer = None

    def _ensure_enhancer(self):
        if self._enhancer is None:
            from hairport.view_enhancer import ViewEnhancer
            self._enhancer = ViewEnhancer(device=self.device, load_pipeline=True)
            logger.info("ViewEnhancer pipeline loaded")

    def run(
        self,
        data_dir: str | Path,
        shape_provider: str | None = None,
        texture_provider: str | None = None,
        bald_version: str | None = None,
        seed: int | None = None,
    ) -> StageSummary:
        """Run view enhancement for all view-aligned folders.

        Returns
        -------
        StageSummary
            Summary with enhanced views and per-pair failures.
        """
        data_dir = Path(data_dir)

        cfg = get_config()
        if shape_provider is None:
            shape_provider = cfg.pipeline.shape_provider
        if texture_provider is None:
            texture_provider = cfg.pipeline.texture_provider
        if bald_version is None:
            bald_version = cfg.pipeline.bald_version

        dm = DatasetManager(data_dir)
        view_aligned_dir = dm.view_aligned_root(shape_provider, texture_provider)

        if not view_aligned_dir.exists():
            raise FileNotFoundError(f"View aligned directory not found: {view_aligned_dir}")

        all_folders = [
            (target_id, source_id, dm.transfer_dir(target_id, source_id, shape_provider, texture_provider))
            for target_id, source_id in dm.list_pairs(shape_provider, texture_provider)
        ]
        base_seed = self.seed if seed is None else seed
        _seed = base_seed if base_seed >= 0 else int(time.time())
        random.seed(_seed)
        random.shuffle(all_folders)
        logger.info(f"EnhanceView: {len(all_folders)} folders, seed={_seed}")

        bald_versions = ["w_seg", "wo_seg"] if bald_version == "all" else [bald_version]
        summary = StageSummary(metadata={"seed": _seed, "bald_versions": bald_versions})
        output_phase = int(cfg.enhance_view.conditioning_phase)
        pending: list[tuple[str, str, Path, str, int, dict]] = []

        for bv in bald_versions:
            for target_id, source_id, folder in all_folders:
                summary.attempted += 1
                item_seed = derive_seed(
                    _seed, "enhance_view", folder.name, bv, operation_name="enhance"
                )
                input_path = dm.rendered_view_file(
                    target_id, source_id, bv, shape_provider, texture_provider
                )
                if not input_path.exists():
                    summary.skipped += 1
                    continue
                output_path = dm.enhanced_view_file(
                    target_id, source_id, bv, shape_provider, texture_provider,
                    phase=output_phase,
                )
                provenance = build_provenance(
                    "enhance_view", seed=item_seed,
                    inputs=[input_path, dm.source_image(target_id)],
                    metadata={"bald_version": bv, "folder": folder.name},
                )
                if can_reuse_artifact(output_path, provenance):
                    summary.skipped += 1
                    continue
                pending.append((target_id, source_id, folder, bv, item_seed, provenance))

        if not pending:
            logger.info(f"EnhanceView complete: {summary.to_dict()}")
            return summary

        from hairport.view_enhancer import process_view_aligned_folder

        self._ensure_enhancer()
        try:
            for target_id, source_id, folder, bv, item_seed, provenance in pending:
                output_path = dm.enhanced_view_file(
                    target_id, source_id, bv, shape_provider, texture_provider,
                    phase=output_phase,
                )
                try:
                    result = process_view_aligned_folder(
                        folder_path=folder,
                        data_dir=data_dir,
                        bald_version=bv,
                        enhancer=self._enhancer,
                        seed=item_seed,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        conditioning_phase=output_phase,
                        provenance=provenance,
                    )
                    if result:
                        summary.completed += 1
                        summary.add_artifacts([output_path])
                    else:
                        summary.add_failure(
                            f"{folder.name}:{bv}",
                            "Enhancement prerequisites missing or enhancement returned no output.",
                        )
                except Exception as e:
                    logger.error(f"EnhanceView error on {folder.name}: {e}")
                    summary.add_failure(f"{folder.name}:{bv}", e)
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"EnhanceView complete: {summary.to_dict()}")
        return summary

    def unload(self):
        if self._enhancer is not None:
            del self._enhancer
            self._enhancer = None
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="hairport-enhance-view", description="Enhance rendered views")
    parser.add_argument("--data_dir", type=str, default="outputs/")
    parser.add_argument("--shape_provider", type=str, default=None, choices=["hunyuan", "hi3dgen", "direct3d_s2"])
    parser.add_argument("--texture_provider", type=str, default=None, choices=["hunyuan", "mvadapter"])
    parser.add_argument("--bald_version", type=str, default=None, choices=["w_seg", "wo_seg", "all"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    stage = EnhanceViewStage(
        device=args.device,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    result = stage.run(
        data_dir=args.data_dir,
        shape_provider=args.shape_provider,
        texture_provider=args.texture_provider,
        bald_version=args.bald_version,
    )
    print(f"EnhanceView: {result}")


if __name__ == "__main__":
    main()
