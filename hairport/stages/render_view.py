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
        self.seed = seed if seed is not None else cfg.seed
        self._generator = None

    def _ensure_generator(self):
        if self._generator is None:
            from hairport.view_generator import TexturedViewGenerator
            self._generator = TexturedViewGenerator(config=None, load_pipeline=True)
            logger.info("TexturedViewGenerator pipeline loaded")

    def run(
        self,
        data_dir: str | Path,
        shape_provider: str | None = None,
        texture_provider: str | None = None,
        bald_version: str | None = None,
        from_blender: bool = True,
        save_intermediates: bool = False,
    ) -> dict:
        """Run multi-view rendering for all view-aligned folders.

        Returns
        -------
        dict
            Summary with ``processed``, ``skipped``, ``error`` counts.
        """
        from hairport.view_generator import process_view_aligned_folder

        self._ensure_generator()
        data_dir = Path(data_dir)

        cfg = get_config()
        if shape_provider is None:
            shape_provider = cfg.pipeline.shape_provider
        if texture_provider is None:
            texture_provider = cfg.pipeline.texture_provider
        if bald_version is None:
            bald_version = cfg.pipeline.bald_version

        provider_subdir = f"shape_{shape_provider}__texture_{texture_provider}"
        view_aligned_dir = data_dir / "view_aligned" / provider_subdir

        if not view_aligned_dir.exists():
            raise FileNotFoundError(f"View aligned directory not found: {view_aligned_dir}")

        all_folders = [f for f in view_aligned_dir.iterdir() if f.is_dir()]

        # Seed and shuffle
        _seed = self.seed if self.seed >= 0 else int(time.time())
        random.seed(_seed)
        random.shuffle(all_folders)
        logger.info(f"RenderView: {len(all_folders)} folders, seed={_seed}")

        bald_versions = ["w_seg", "wo_seg"] if bald_version == "all" else [bald_version]
        stats = {"processed": 0, "skipped": 0, "error": 0}

        try:
            for bv in bald_versions:
                for folder in all_folders:
                    try:
                        result = process_view_aligned_folder(
                            folder_path=folder,
                            data_dir=data_dir,
                            bald_version=bv,
                            generator=self._generator,
                            from_blender=from_blender,
                            save_intermediates=save_intermediates,
                        )
                        stats["processed" if result else "skipped"] += 1
                    except Exception as e:
                        logger.error(f"RenderView error on {folder.name}: {e}")
                        stats["error"] += 1
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"RenderView complete: {stats}")
        return stats

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
    parser.add_argument("--save_intermediates", action="store_true", default=False)
    parser.add_argument("--from_blender", action="store_true", default=True)
    parser.add_argument("--from_nvdiffrast", dest="from_blender", action="store_false")
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    stage = RenderViewStage(seed=args.seed)
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
