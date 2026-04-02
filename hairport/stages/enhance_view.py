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
            self._enhancer = ViewEnhancer(load_pipeline=True)
            logger.info("ViewEnhancer pipeline loaded")

    def run(
        self,
        data_dir: str | Path,
        shape_provider: str | None = None,
        texture_provider: str | None = None,
        bald_version: str | None = None,
    ) -> dict:
        """Run view enhancement for all view-aligned folders.

        Returns
        -------
        dict
            Summary with ``processed``, ``skipped``, ``error`` counts.
        """
        from hairport.view_enhancer import process_view_aligned_folder

        self._ensure_enhancer()
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
        _seed = self.seed if self.seed >= 0 else int(time.time())
        random.seed(_seed)
        random.shuffle(all_folders)
        logger.info(f"EnhanceView: {len(all_folders)} folders, seed={_seed}")

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
                            enhancer=self._enhancer,
                            seed=_seed,
                            num_inference_steps=self.num_inference_steps,
                            guidance_scale=self.guidance_scale,
                        )
                        stats["processed" if result else "skipped"] += 1
                    except Exception as e:
                        logger.error(f"EnhanceView error on {folder.name}: {e}")
                        stats["error"] += 1
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"EnhanceView complete: {stats}")
        return stats

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
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    stage = EnhanceViewStage(
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
