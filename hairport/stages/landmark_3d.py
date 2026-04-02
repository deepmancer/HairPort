"""Stage 4 — Landmark 3D: Estimate 3D facial landmarks via multi-view fusion.

Delegates to ``fit_lmk.run_standalone.estimate_3d_landmarks_standalone``.

Usage::

    # Programmatic — batch (pipeline mode)
    from hairport.stages.landmark_3d import Landmark3DStage
    stage = Landmark3DStage()
    stage.run(data_dir="outputs", shape_provider="hi3dgen", texture_provider="mvadapter")

    # Programmatic — single mesh
    stage.run(mesh_path="output/mesh.glb", output_dir="output/lmk_3d")

    # CLI
    python -m hairport.stages.landmark_3d --data_dir outputs --shape_provider hi3dgen
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from hairport.config import get_config, add_config_args, load_config_from_args

logger = logging.getLogger(__name__)


class Landmark3DStage:
    """Estimate 3D facial landmarks from head meshes.

    Uses multi-view Blender renders + MediaPipe + multi-view fusion
    with optional CodeFormer super-resolution.
    """

    def run(
        self,
        data_dir: str | Path | None = None,
        mesh_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        shape_provider: str | None = None,
        texture_provider: str | None = None,
        cam_loc: list[float] | None = None,
        cam_rot: list[float] | None = None,
        ortho_scale: float | None = None,
        num_perturbations: int | None = None,
        angle_range: float | None = None,
        trans_range: float | None = None,
        resolution: int | None = None,
        optimize: bool | None = None,
        device: str | None = None,
        debug: bool = False,
        debug_dir: str = "./debug_outputs",
        super_resolution: bool | None = None,
    ) -> dict:
        """Run 3D landmark estimation.

        Supports two invocation modes:

        1. **data_dir** (pipeline/batch mode): iterates over all identity
           meshes under ``<data_dir>/<texture_provider>/<shape_provider>/``
           and writes landmarks to ``<data_dir>/lmk_3d/<provider_subdir>/<id>/``.
        2. **mesh_path / output_dir** (single mesh mode).

        Returns
        -------
        dict
            In single-mesh mode: ``landmarks_3d``, ``vertex_indices``, etc.
            In batch mode: ``{"processed": N, "skipped": N, "failed": N}``.
        """
        if cam_loc is None:
            cam_loc = list(get_config().landmark_3d.default_cam_location)
        if cam_rot is None:
            cam_rot = list(get_config().landmark_3d.default_cam_rotation)

        cfg = get_config()
        _lmk = cfg.landmark_3d
        if shape_provider is None:
            shape_provider = cfg.pipeline.shape_provider
        if texture_provider is None:
            texture_provider = cfg.pipeline.texture_provider
        if ortho_scale is None:
            ortho_scale = _lmk.ortho_scale
        if num_perturbations is None:
            num_perturbations = _lmk.num_perturbations
        if angle_range is None:
            angle_range = _lmk.angle_range
        if trans_range is None:
            trans_range = _lmk.trans_range
        if resolution is None:
            resolution = _lmk.resolution
        if optimize is None:
            optimize = _lmk.optimize
        if device is None:
            device = cfg.device
        if super_resolution is None:
            super_resolution = _lmk.super_resolution

        # Single-mesh mode
        if mesh_path is not None:
            return self._run_single(
                mesh_path=mesh_path,
                output_dir=output_dir or "./output_landmarks",
                cam_loc=cam_loc, cam_rot=cam_rot,
                ortho_scale=ortho_scale,
                num_perturbations=num_perturbations,
                angle_range=angle_range, trans_range=trans_range,
                resolution=resolution, optimize=optimize,
                device=device, debug=debug, debug_dir=debug_dir,
                super_resolution=super_resolution,
            )

        # Batch mode — iterate over all identity meshes
        if data_dir is None:
            raise ValueError("Provide data_dir (batch) or mesh_path (single)")

        data_dir = Path(data_dir)
        provider_subdir = f"shape_{shape_provider}__texture_{texture_provider}"

        # Locate mesh directories
        if texture_provider == "hunyuan":
            mesh_root = data_dir / "hunyuan"
        else:
            mesh_root = data_dir / texture_provider / shape_provider

        if not mesh_root.exists():
            raise FileNotFoundError(f"Mesh root directory not found: {mesh_root}")

        lmk_out_root = data_dir / "lmk_3d" / provider_subdir

        identity_dirs = sorted(
            d for d in mesh_root.iterdir()
            if d.is_dir() and (d / "shape_mesh.glb").exists()
        )
        logger.info(f"Landmark3D batch: {len(identity_dirs)} identities in {mesh_root}")

        stats = {"processed": 0, "skipped": 0, "failed": 0}
        for id_dir in identity_dirs:
            identity_id = id_dir.name
            mesh_file = id_dir / "shape_mesh.glb"
            out = lmk_out_root / identity_id
            marker = out / "landmarks_3d.npy"

            if marker.exists():
                stats["skipped"] += 1
                continue

            try:
                self._run_single(
                    mesh_path=mesh_file, output_dir=out,
                    cam_loc=cam_loc, cam_rot=cam_rot,
                    ortho_scale=ortho_scale,
                    num_perturbations=num_perturbations,
                    angle_range=angle_range, trans_range=trans_range,
                    resolution=resolution, optimize=optimize,
                    device=device, debug=debug, debug_dir=debug_dir,
                    super_resolution=super_resolution,
                )
                stats["processed"] += 1
            except Exception as e:
                logger.error(f"Landmark3D failed for {identity_id}: {e}")
                stats["failed"] += 1

        logger.info(f"Landmark3D batch complete: {stats}")
        return stats

    @staticmethod
    def _run_single(
        mesh_path, output_dir, cam_loc, cam_rot, ortho_scale,
        num_perturbations, angle_range, trans_range, resolution,
        optimize, device, debug, debug_dir, super_resolution,
    ) -> dict:
        from hairport.fit_lmk.run_standalone import estimate_3d_landmarks_standalone

        result = estimate_3d_landmarks_standalone(
            mesh_path=str(mesh_path),
            cam_loc=cam_loc, cam_rot=cam_rot,
            ortho_scale=ortho_scale,
            output_dir=str(output_dir),
            num_perturbations=num_perturbations,
            angle_range=angle_range, trans_range=trans_range,
            resolution=resolution,
            optimize=optimize, device=device,
            debug=debug, debug_dir=debug_dir,
            super_resolution=super_resolution,
        )
        return result


def main(argv: list[str] | None = None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="hairport-landmark-3d", description="Estimate 3D facial landmarks")
    # Batch mode
    parser.add_argument("--data_dir", type=str, default=None, help="Root data directory (batch mode)")
    parser.add_argument("--shape_provider", type=str, default=None,
                        choices=["hunyuan", "hi3dgen", "direct3d_s2"])
    parser.add_argument("--texture_provider", type=str, default=None,
                        choices=["hunyuan", "mvadapter"])
    # Single mode
    parser.add_argument("--mesh_path", type=str, default=None, help="Single mesh GLB path")
    parser.add_argument("--output_dir", type=str, default="./output_landmarks")
    # Shared
    parser.add_argument("--cam_loc", type=float, nargs=3, default=None)
    parser.add_argument("--cam_rot", type=float, nargs=3, default=None)
    parser.add_argument("--ortho_scale", type=float, default=None)
    parser.add_argument("--num_perturbations", type=int, default=None)
    parser.add_argument("--angle_range", type=float, default=None)
    parser.add_argument("--trans_range", type=float, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--no_optimize", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_dir", type=str, default="./debug_outputs")
    parser.add_argument("--no_super_resolution", action="store_true")
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    if not args.data_dir and not args.mesh_path:
        parser.error("Provide --data_dir (batch) or --mesh_path (single)")

    stage = Landmark3DStage()
    result = stage.run(
        data_dir=args.data_dir,
        mesh_path=args.mesh_path,
        output_dir=args.output_dir,
        shape_provider=args.shape_provider,
        texture_provider=args.texture_provider,
        cam_loc=args.cam_loc, cam_rot=args.cam_rot,
        ortho_scale=args.ortho_scale,
        num_perturbations=args.num_perturbations,
        angle_range=args.angle_range, trans_range=args.trans_range,
        resolution=args.resolution,
        optimize=False if args.no_optimize else None,
        device=args.device,
        debug=args.debug, debug_dir=args.debug_dir,
        super_resolution=False if args.no_super_resolution else None,
    )
    print(f"Landmark3D: {result}")


if __name__ == "__main__":
    main()
