"""Stage 4 — Landmark 3D: postprocess textured meshes and fit landmarks.

Delegates to ``hairport.postprocess.fit_3d_landmarks`` so the exported
``postprocessed_textured_mesh.glb`` is the normalized MVAdapter/Hunyuan
textured mesh used by downstream 3D-aware inference.

Usage::

    # Programmatic — batch (pipeline mode)
    from hairport.stages.landmark_3d import Landmark3DStage
    stage = Landmark3DStage()
    stage.run(data_dir="outputs", shape_provider="hi3dgen", texture_provider="mvadapter")

    # Programmatic — single textured mesh
    stage.run(mesh_path="output/textured_mesh.glb", output_dir="output/lmk_3d")

    # CLI
    python -m hairport.stages.landmark_3d --data_dir outputs --shape_provider hi3dgen
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from hairport.config import get_config, add_config_args, load_config_from_args
from hairport.runtime import StageSummary, build_provenance, can_reuse_artifact, write_provenance

logger = logging.getLogger(__name__)


class Landmark3DStage:
    """Fit 3D landmarks and export postprocessed textured meshes."""

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
        frontalize: bool = False,
        target_rotation_euler_rad: list[float] | None = None,
    ) -> StageSummary:
        """Run 3D landmark estimation.

        Supports two invocation modes:

        1. **data_dir** (pipeline/batch mode): iterates over all identity
           textured meshes under ``<data_dir>/<texture_provider>/<shape_provider>/``
           and writes postprocessed meshes/landmarks to
           ``<data_dir>/lmk_3d/<provider_subdir>/<id>/``.
        2. **mesh_path / output_dir** (single textured mesh mode).

        Returns
        -------
        StageSummary
            Summary with generated landmark artifacts and per-identity failures.
        """
        cfg = get_config()
        _lmk = cfg.landmark_3d
        if shape_provider is None:
            shape_provider = cfg.pipeline.shape_provider
        if texture_provider is None:
            texture_provider = cfg.pipeline.texture_provider
        if cam_loc is None:
            cam_loc = list(_lmk.default_cam_location)
        if cam_rot is None:
            cam_rot = list(_lmk.default_cam_rotation)
        if ortho_scale is None:
            ortho_scale = _lmk.textured_mesh_ortho_scale
        if num_perturbations is None:
            num_perturbations = _lmk.num_perturbations
        if num_perturbations != 0:
            raise ValueError(
                "Landmark3D supports only single frontal-view inference; "
                "num_perturbations must be 0."
            )
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
        from hairport.fit_lmk.ray_intersector import RayMeshIntersector
        ray_backend = RayMeshIntersector.preflight_backend()
        logger.info("Landmark3D ray backend: %s", ray_backend)

        # Single-mesh mode
        if mesh_path is not None:
            result = self._run_single(
                mesh_path=mesh_path,
                output_dir=output_dir or "./output_landmarks",
                target_rotation_euler_rad=target_rotation_euler_rad or [0.0, 0.0, 0.0],
                cam_loc=cam_loc, cam_rot=cam_rot,
                ortho_scale=ortho_scale,
                num_perturbations=num_perturbations,
                angle_range=angle_range, trans_range=trans_range,
                resolution=resolution, optimize=optimize,
                device=device, debug=debug, debug_dir=debug_dir,
                super_resolution=super_resolution,
                frontalize=frontalize,
                force_recompute=True,
            )
            summary = StageSummary(
                attempted=1, completed=1,
                metadata={"result": result, "ray_backend": ray_backend},
            )
            summary.add_artifacts(
                [result[key] for key in ("landmarks_3d", "vertex_indices", "postprocessed_mesh")]
            )
            return summary

        # Batch mode — iterate over all identity meshes
        if data_dir is None:
            raise ValueError("Provide data_dir (batch) or mesh_path (single)")

        data_dir = Path(data_dir)
        provider_subdir = f"shape_{shape_provider}__texture_{texture_provider}"

        # Locate textured mesh directories
        if texture_provider == "hunyuan":
            mesh_root = data_dir / "hunyuan"
        else:
            mesh_root = data_dir / texture_provider / shape_provider

        if not mesh_root.exists():
            raise FileNotFoundError(f"Textured mesh root directory not found: {mesh_root}")

        lmk_out_root = data_dir / "lmk_3d" / provider_subdir

        identity_dirs = sorted(
            d for d in mesh_root.iterdir()
            if d.is_dir() and (d / _lmk.textured_mesh_filename).exists()
        )
        if not identity_dirs:
            expected_input = (
                data_dir
                / cfg.shape_mesh.input_mesh_dir
                / "<id>"
                / cfg.shape_mesh.input_mesh_filename
            )
            raise FileNotFoundError(
                f"No {_lmk.textured_mesh_filename} files were found in {mesh_root}. "
                "Landmark3D requires textured meshes generated by ShapeMesh/MVAdapter and "
                f"does not fall back to shape meshes. Expected ShapeMesh inputs at {expected_input} "
                f"and outputs at {mesh_root}/<id>/{_lmk.textured_mesh_filename}."
            )
        logger.info(f"Landmark3D batch: {len(identity_dirs)} identities in {mesh_root}")

        summary = StageSummary(metadata={"ray_backend": ray_backend, "landmark_mode": "single_frontal"})
        for id_dir in identity_dirs:
            summary.attempted += 1
            identity_id = id_dir.name
            mesh_file = id_dir / _lmk.textured_mesh_filename
            out = lmk_out_root / identity_id
            expected_outputs = [
                out / "landmarks_3d.npy",
                out / "vertex_indices.npy",
                out / _lmk.postprocessed_mesh_filename,
            ]

            provenance = build_provenance(
                "landmark_3d", seed=cfg.seed,
                inputs=[mesh_file, data_dir / "image" / f"{identity_id}.png"],
                metadata={"identity_id": identity_id, "num_perturbations": 0},
            )
            if all(can_reuse_artifact(path, provenance) for path in expected_outputs):
                summary.skipped += 1
                continue

            try:
                euler_rad = self._load_head_orientation(data_dir, identity_id)
                self._run_single(
                    mesh_path=mesh_file, output_dir=out,
                    target_rotation_euler_rad=euler_rad,
                    cam_loc=cam_loc, cam_rot=cam_rot,
                    ortho_scale=ortho_scale,
                    num_perturbations=num_perturbations,
                    angle_range=angle_range, trans_range=trans_range,
                    resolution=resolution, optimize=optimize,
                    device=device, debug=debug, debug_dir=debug_dir,
                    super_resolution=super_resolution,
                    frontalize=frontalize,
                    force_recompute=True,
                )
                for path in expected_outputs:
                    write_provenance(path, provenance)
                summary.completed += 1
                summary.add_artifacts(expected_outputs)
            except Exception as e:
                logger.error(f"Landmark3D failed for {identity_id}: {e}")
                summary.add_failure(identity_id, e)

        logger.info(f"Landmark3D batch complete: {summary.to_dict()}")
        return summary

    @staticmethod
    def _run_single(
        mesh_path, output_dir, target_rotation_euler_rad, cam_loc, cam_rot, ortho_scale,
        num_perturbations, angle_range, trans_range, resolution,
        optimize, device, debug, debug_dir, super_resolution, frontalize, force_recompute,
    ) -> dict:
        from hairport.postprocess import fit_3d_landmarks

        result = fit_3d_landmarks(
            lmk_3d_output_dir=str(output_dir),
            target_textured_mesh_path=str(mesh_path),
            target_rotation_euler_rad=target_rotation_euler_rad,
            camera_location=cam_loc,
            camera_rotation=cam_rot,
            camera_ortho_scale=ortho_scale,
            num_perturbations=num_perturbations,
            angle_range=angle_range, trans_range=trans_range,
            resolution=resolution,
            optimize=optimize,
            device=device,
            debug=debug, debug_dir=debug_dir,
            super_resolution=super_resolution,
            frontalize=frontalize,
            force_recompute=force_recompute,
        )
        return result

    @staticmethod
    def _load_head_orientation(data_dir: Path, identity_id: str) -> list[float]:
        from hairport.fitting.orientation import compute_head_orientation

        image_path = data_dir / "image" / f"{identity_id}.png"
        if not image_path.exists():
            raise FileNotFoundError(
                f"Source image required for head orientation was not found: {image_path}"
            )

        head_orientation = compute_head_orientation(
            image_path=image_path,
            cache_dir=data_dir / "head_orientation" / identity_id,
            force=False,
        )
        return head_orientation.get("euler_angles_xyz_radians", [[0.0, 0.0, 0.0]])[0]


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
    parser.add_argument("--frontalize", action="store_true")
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
        frontalize=args.frontalize,
    )
    print(f"Landmark3D: {result}")


if __name__ == "__main__":
    main()
