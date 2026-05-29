"""Stage 6 — Render View: Generate textured multi-views with MV-Adapter.

Delegates to ``hairport.view_generator.TexturedViewGenerator``.

Usage::

    # Programmatic
    from hairport.stages.render_view import RenderViewStage
    stage = RenderViewStage()
    stage.run(data_dir="outputs", shape_provider="hi3dgen")

    # CLI
    python -m hairport.stages.render_view --data_dir outputs --shape_provider hi3dgen
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


class RenderViewStage:
    """Generate textured multi-view images using MV-Adapter SDXL.

    Parameters
    ----------
    device : str
        Compute device.
    seed : int
        Random seed (-1 for timestamp-based).
    """

    def __init__(self, device: str | None = None, seed: int | None = None):
        cfg = get_config()
        self.device = device if device is not None else cfg.device
        if not torch.cuda.is_available() and "cuda" in str(self.device):
            self.device = "cpu"
        self.seed = seed if seed is not None else cfg.seed
        self._generator = None

    def _ensure_generator(self):
        if self._generator is None:
            from hairport.view_generator import TexturedViewConfig, TexturedViewGenerator
            self._generator = TexturedViewGenerator(
                config=TexturedViewConfig(device=self.device),
                load_pipeline=True,
            )
            logger.info("TexturedViewGenerator pipeline loaded")

    def run(
        self,
        data_dir: str | Path,
        shape_provider: str | None = None,
        texture_provider: str | None = None,
        bald_version: str | None = None,
        from_blender: bool = True,
        save_intermediates: bool = False,
        seed: int | None = None,
    ) -> StageSummary:
        """Run multi-view rendering for all view-aligned folders.

        Returns
        -------
        StageSummary
            Summary with generated views and per-pair failures.
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

        # Seed and shuffle
        base_seed = self.seed if seed is None else seed
        _seed = base_seed if base_seed >= 0 else int(time.time())
        random.seed(_seed)
        random.shuffle(all_folders)
        logger.info(f"RenderView: {len(all_folders)} folders, seed={_seed}")

        bald_versions = ["w_seg", "wo_seg"] if bald_version == "all" else [bald_version]
        summary = StageSummary(metadata={"seed": _seed, "bald_versions": bald_versions})
        pending: list[tuple[str, str, Path, str, int, dict]] = []

        for bv in bald_versions:
            for target_id, source_id, folder in all_folders:
                summary.attempted += 1
                camera_path = dm.camera_params_file(
                    target_id, source_id, bv, shape_provider, texture_provider
                )
                if not camera_path.exists():
                    summary.skipped += 1
                    continue
                item_seed = derive_seed(
                    _seed, "render_view", folder.name, bv, operation_name="generate"
                )
                output_path = dm.rendered_view_file(
                    target_id, source_id, bv, shape_provider, texture_provider
                )
                mesh_path = dm.aligned_mesh_file(
                    target_id, source_id, shape_provider, texture_provider
                )
                if not mesh_path.exists():
                    mesh_path = dm.postprocessed_mesh_file(
                        target_id, shape_provider, texture_provider
                    )
                provenance = build_provenance(
                    "render_view", seed=item_seed,
                    inputs=[
                        camera_path,
                        mesh_path,
                        dm.source_image(target_id),
                        dm.prompt_file(target_id),
                    ],
                    metadata={"bald_version": bv, "folder": folder.name},
                )
                if can_reuse_artifact(output_path, provenance):
                    summary.skipped += 1
                    continue
                pending.append((target_id, source_id, folder, bv, item_seed, provenance))

        if not pending:
            logger.info(f"RenderView complete: {summary.to_dict()}")
            return summary

        from hairport.view_generator import process_view_aligned_folder

        self._ensure_generator()
        try:
            for target_id, source_id, folder, bv, item_seed, provenance in pending:
                output_path = dm.rendered_view_file(
                    target_id, source_id, bv, shape_provider, texture_provider
                )
                try:
                    result = process_view_aligned_folder(
                        folder_path=folder,
                        data_dir=data_dir,
                        bald_version=bv,
                        generator=self._generator,
                        from_blender=from_blender,
                        save_intermediates=save_intermediates,
                        seed=item_seed,
                        provenance=provenance,
                    )
                    if result:
                        summary.completed += 1
                        summary.add_artifacts([output_path])
                    else:
                        summary.add_failure(
                            f"{folder.name}:{bv}",
                            "Rendering prerequisites missing or rendering returned no output.",
                        )
                except Exception as e:
                    logger.error(f"RenderView error on {folder.name}: {e}")
                    summary.add_failure(f"{folder.name}:{bv}", e)
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"RenderView complete: {summary.to_dict()}")
        return summary

    def unload(self):
        if self._generator is not None:
            del self._generator
            self._generator = None
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="hairport-render-view", description="Generate textured multi-views")
    parser.add_argument("--data_dir", type=str, default="outputs/")
    parser.add_argument("--shape_provider", type=str, default=None, choices=["hunyuan", "hi3dgen", "direct3d_s2"])
    parser.add_argument("--texture_provider", type=str, default=None, choices=["hunyuan", "mvadapter"])
    parser.add_argument("--bald_version", type=str, default=None, choices=["w_seg", "wo_seg", "all"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_intermediates", action="store_true", default=False)
    parser.add_argument("--from_blender", action="store_true", default=True)
    parser.add_argument("--from_nvdiffrast", dest="from_blender", action="store_false")
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    stage = RenderViewStage(device=args.device, seed=args.seed)
    result = stage.run(
        data_dir=args.data_dir,
        shape_provider=args.shape_provider,
        texture_provider=args.texture_provider,
        bald_version=args.bald_version,
        from_blender=args.from_blender,
        save_intermediates=args.save_intermediates,
    )
    print(f"RenderView: {result}")


if __name__ == "__main__":
    main()
