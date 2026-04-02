"""Stage 3 — Shape Mesh: Simplify and frontalize 3D head meshes.

Delegates to ``hairport.postprocess_shape_mesh.main``.

Usage::

    # Programmatic
    from hairport.stages.shape_mesh import ShapeMeshStage
    stage = ShapeMeshStage()
    stage.run(data_dir="outputs", shape_provider="hi3dgen")

    # CLI
    python -m hairport.stages.shape_mesh --data_dir outputs --shape_provider hi3dgen
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from hairport.config import get_config, add_config_args, load_config_from_args

logger = logging.getLogger(__name__)


class ShapeMeshStage:
    """Simplify GLB meshes and optionally frontalize them.

    This stage:
    1. Loads the shapes produced by hi3dgen / hunyuan
    2. Simplifies them (quadric decimation)
    3. Optionally frontalizes using pixel3dmm head orientation
    """

    def run(
        self,
        data_dir: str | Path,
        shape_provider: str | None = None,
        texture_provider: str | None = None,
        frontalize: bool = False,
    ) -> None:
        """Run mesh post-processing.

        Parameters
        ----------
        data_dir : str | Path
            Root data directory.
        shape_provider : str
            ``"hi3dgen"`` or ``"hunyuan"``.
        texture_provider : str
            ``"mvadapter"`` or ``"hunyuan"``.
        frontalize : bool
            Whether to frontalize meshes using pixel3dmm orientation data.
        """
        from hairport.postprocess_shape_mesh import main as _run

        cfg = get_config()
        if shape_provider is None:
            shape_provider = cfg.pipeline.shape_provider
        if texture_provider is None:
            texture_provider = cfg.pipeline.texture_provider

        logger.info(f"ShapeMesh: data_dir={data_dir}, shape={shape_provider}, "
                     f"texture={texture_provider}, frontalize={frontalize}")
        _run(
            data_dir=str(data_dir),
            shape_provider=shape_provider,
            texture_provider=texture_provider,
            frontalize=frontalize,
        )
        logger.info("ShapeMesh stage complete")


def main(argv: list[str] | None = None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="hairport-shape-mesh", description="Simplify + frontalize meshes")
    parser.add_argument("--data_dir", type=str, default="outputs")
    parser.add_argument("--shape_provider", type=str, default=None, choices=["hunyuan", "hi3dgen"])
    parser.add_argument("--texture_provider", type=str, default=None, choices=["mvadapter", "hunyuan"])
    parser.add_argument("--frontalize", action="store_true")
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    stage = ShapeMeshStage()
    stage.run(
        data_dir=args.data_dir,
        shape_provider=args.shape_provider,
        texture_provider=args.texture_provider,
        frontalize=args.frontalize,
    )


if __name__ == "__main__":
    main()
