"""Stage 3 — Shape Mesh: run MVAdapter texturing from canonical shape meshes.

For the official inference path (``texture_provider=mvadapter``), this stage:
1. Reads user meshes from ``<data_dir>/shape_mesh/<id>/shape_mesh.glb``.
2. Creates a temporary alias tree expected by MVAdapter's legacy script.
3. Invokes ``python -m scripts.texture_i2tex`` in the MVAdapter module.
4. Validates textured outputs at
   ``<data_dir>/mvadapter/<shape_provider>/<id>/textured_mesh.glb``.

For non-MVAdapter texture providers, this stage validates precomputed
``textured_mesh.glb`` files and does not perform texturing.

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
import csv
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from hairport.config import get_config, add_config_args, load_config_from_args
from hairport.runtime import (
    StageSummary,
    build_provenance,
    can_reuse_artifact,
    derive_seed,
    write_provenance,
)

logger = logging.getLogger(__name__)


class ShapeMeshStage:
    """Run canonical shape-mesh texturing for pipeline stage 3."""

    def __init__(self, device: str | None = None, seed: int | None = None):
        cfg = get_config()
        self.device = device if device is not None else cfg.device
        self.seed = seed if seed is not None else cfg.seed

    @staticmethod
    def _collect_shape_meshes(shape_mesh_root: Path, shape_mesh_filename: str) -> dict[str, Path]:
        if not shape_mesh_root.exists():
            return {}
        meshes: dict[str, Path] = {}
        for identity_dir in sorted(d for d in shape_mesh_root.iterdir() if d.is_dir()):
            mesh_path = identity_dir / shape_mesh_filename
            if mesh_path.exists():
                meshes[identity_dir.name] = mesh_path
        return meshes

    @staticmethod
    def _parse_target_ids_from_pairs_csv(data_dir: Path) -> set[str] | None:
        pairs_csv = data_dir / "pairs.csv"
        if not pairs_csv.exists():
            return None
        with open(pairs_csv, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "target_id" not in reader.fieldnames:
                return None
            return {row["target_id"] for row in reader if row.get("target_id")}

    @staticmethod
    def _link_or_copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
        except OSError:
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    @staticmethod
    def _validate_precomputed_textured_meshes(
        data_dir: Path,
        shape_provider: str,
        texture_provider: str,
        textured_mesh_filename: str,
    ) -> dict:
        if texture_provider == "hunyuan":
            mesh_root = data_dir / "hunyuan"
        else:
            mesh_root = data_dir / texture_provider / shape_provider

        if not mesh_root.exists():
            raise FileNotFoundError(
                f"Precomputed textured mesh root not found: {mesh_root}. "
                f"Expected <root>/<id>/{textured_mesh_filename}."
            )

        identity_dirs = [
            d for d in sorted(mesh_root.iterdir())
            if d.is_dir() and (d / textured_mesh_filename).exists()
        ]
        if not identity_dirs:
            raise FileNotFoundError(
                f"No precomputed textured meshes were found in {mesh_root}. "
                f"Expected <root>/<id>/{textured_mesh_filename}."
            )
        return {"processed": 0, "skipped": len(identity_dirs), "failed": 0}

    def run(
        self,
        data_dir: str | Path,
        shape_provider: str | None = None,
        texture_provider: str | None = None,
        prompt_dir: str | Path | None = None,
        input_mesh_dir: str | None = None,
        input_image_dir: str | None = None,
        shape_mesh_filename: str | None = None,
        textured_mesh_filename: str | None = None,
        variant: str | None = None,
        sdxl_model_id: str | None = None,
        reference_conditioning_scale: float | None = None,
        align_input_mesh: bool | None = None,
        align_output_mesh: bool | None = None,
        preprocess_mesh: bool | None = None,
        remove_bg: bool | None = None,
        default_prompt_text: str | None = None,
        device: str | None = None,
        seed: int | None = None,
    ) -> StageSummary:
        """Run MVAdapter texturing from canonical ``shape_mesh`` inputs.

        Parameters
        ----------
        data_dir : str | Path
            Root data directory.
        shape_provider : str
            ``"hi3dgen"`` or ``"hunyuan"``.
        texture_provider : str
            ``"mvadapter"`` or ``"hunyuan"``.
        """
        cfg = get_config()
        shape_cfg = cfg.shape_mesh
        data_dir = Path(data_dir).expanduser().resolve()
        if shape_provider is None:
            shape_provider = cfg.pipeline.shape_provider
        if texture_provider is None:
            texture_provider = cfg.pipeline.texture_provider
        if input_mesh_dir is None:
            input_mesh_dir = shape_cfg.input_mesh_dir
        if input_image_dir is None:
            input_image_dir = shape_cfg.input_image_dir
        if shape_mesh_filename is None:
            shape_mesh_filename = shape_cfg.input_mesh_filename
        if textured_mesh_filename is None:
            textured_mesh_filename = shape_cfg.output_mesh_filename
        if prompt_dir is None:
            prompt_dir = Path(data_dir) / shape_cfg.prompt_subdir
        else:
            prompt_dir = Path(prompt_dir).expanduser().resolve()
        if variant is None:
            variant = shape_cfg.variant
        if sdxl_model_id is None:
            sdxl_model_id = shape_cfg.sdxl_model_id
        if reference_conditioning_scale is None:
            reference_conditioning_scale = shape_cfg.reference_conditioning_scale
        if align_input_mesh is None:
            align_input_mesh = shape_cfg.align_input_mesh
        if align_output_mesh is None:
            align_output_mesh = shape_cfg.align_output_mesh
        if preprocess_mesh is None:
            preprocess_mesh = shape_cfg.preprocess_mesh
        if remove_bg is None:
            remove_bg = shape_cfg.remove_bg
        if default_prompt_text is None:
            default_prompt_text = shape_cfg.default_prompt_text
        if device is None:
            device = self.device
        if seed is None:
            seed = self.seed

        if texture_provider != "mvadapter":
            logger.info(
                "ShapeMesh: texture_provider=%s uses precomputed textured meshes",
                texture_provider,
            )
            stats = self._validate_precomputed_textured_meshes(
                data_dir=data_dir,
                shape_provider=shape_provider,
                texture_provider=texture_provider,
                textured_mesh_filename=textured_mesh_filename,
            )
            return StageSummary(
                attempted=stats["skipped"], skipped=stats["skipped"],
                metadata={"texture_provider": texture_provider, "precomputed": True},
            )

        shape_mesh_root = data_dir / input_mesh_dir
        mesh_by_id = self._collect_shape_meshes(shape_mesh_root, shape_mesh_filename)
        if not mesh_by_id:
            raise FileNotFoundError(
                f"No input shape meshes found under {shape_mesh_root}. "
                f"Expected {input_mesh_dir}/<id>/{shape_mesh_filename}."
            )

        target_ids_from_pairs = self._parse_target_ids_from_pairs_csv(data_dir)
        if target_ids_from_pairs is None:
            expected_ids = sorted(mesh_by_id.keys())
        else:
            expected_ids = sorted([sample_id for sample_id in mesh_by_id if sample_id in target_ids_from_pairs])

        if not expected_ids:
            logger.warning(
                "ShapeMesh: no shape-mesh identities intersect with pairs.csv target_id values"
            )
            return StageSummary(metadata={"texture_provider": texture_provider})

        image_root = data_dir / input_image_dir
        missing_images = [sample_id for sample_id in expected_ids if not (image_root / f"{sample_id}.png").exists()]
        if missing_images:
            preview = ", ".join(missing_images[:5])
            if len(missing_images) > 5:
                preview = f"{preview}, ..."
            raise FileNotFoundError(
                f"Missing texturing input images in {image_root}. "
                f"Expected {input_image_dir}/<id>.png for ids: {preview}"
            )

        mv_module = Path(cfg.paths.mv_adapter_module)
        if not mv_module.exists():
            raise FileNotFoundError(f"MVAdapter module path not found: {mv_module}")

        output_root = data_dir / texture_provider / shape_provider
        item_seeds = {
            sample_id: derive_seed(seed, "shape_mesh", sample_id, operation_name="texture")
            for sample_id in expected_ids
        }
        provenance_by_id = {
            sample_id: build_provenance(
                "shape_mesh",
                seed=item_seeds[sample_id],
                inputs=[
                    mesh_by_id[sample_id],
                    image_root / f"{sample_id}.png",
                    Path(prompt_dir) / f"{sample_id}.json",
                ],
                metadata={
                    "shape_provider": shape_provider,
                    "texture_provider": texture_provider,
                    "variant": variant,
                    "operation": "texture",
                },
            )
            for sample_id in expected_ids
        }
        preexisting = {
            sample_id: can_reuse_artifact(
                output_root / sample_id / textured_mesh_filename,
                provenance_by_id[sample_id],
            )
            for sample_id in expected_ids
        }
        if all(preexisting.values()):
            logger.info("ShapeMesh: all textured meshes already exist; skipping MVAdapter run")
            return StageSummary(
                attempted=len(expected_ids), skipped=len(expected_ids),
                metadata={"texture_provider": texture_provider},
            )

        # MV-Adapter skips any existing output internally. Remove only derived
        # outputs whose provenance does not validate so they are regenerated.
        for sample_id in expected_ids:
            output_mesh = output_root / sample_id / textured_mesh_filename
            if output_mesh.exists() and not preexisting[sample_id]:
                logger.info("ShapeMesh: regenerating unvalidated artifact %s", output_mesh)
                output_mesh.unlink()

        for sample_id in expected_ids:
            if preexisting[sample_id]:
                continue
            with tempfile.TemporaryDirectory(prefix="hairport_mvadapter_") as tmp_dir:
                tmp_root = Path(tmp_dir)
                self._link_or_copy(
                    image_root / f"{sample_id}.png",
                    tmp_root / input_image_dir / f"{sample_id}.png",
                )
                self._link_or_copy(
                    mesh_by_id[sample_id],
                    tmp_root / shape_provider / sample_id / shape_mesh_filename,
                )

                cmd = [
                    sys.executable,
                    "-m",
                    "scripts.texture_i2tex",
                    "--data_dir",
                    str(tmp_root),
                    "--prompt_dir",
                    str(prompt_dir),
                    "--output_dir",
                    str(data_dir / texture_provider),
                    "--shape_provider",
                    shape_provider,
                    "--device",
                    str(device),
                    "--seed",
                    str(item_seeds[sample_id]),
                    "--variant",
                    str(variant),
                    "--sdxl_model_id",
                    str(sdxl_model_id),
                    "--reference_conditioning_scale",
                    str(reference_conditioning_scale),
                    "--default_text",
                    str(default_prompt_text),
                ]

                if align_input_mesh:
                    cmd.append("--align_input_mesh")
                if align_output_mesh:
                    cmd.append("--align_output_mesh")
                if preprocess_mesh:
                    cmd.append("--preprocess_mesh")
                if remove_bg:
                    cmd.append("--remove_bg")

                logger.info(
                    "ShapeMesh: launching MVAdapter texture_i2tex for %s (seed=%d)",
                    sample_id,
                    item_seeds[sample_id],
                )
                proc = subprocess.run(
                    cmd,
                    cwd=str(mv_module),
                    capture_output=True,
                    text=True,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        "MVAdapter texturing failed.\n"
                        f"Command: {' '.join(cmd)}\n"
                        f"stdout:\n{proc.stdout}\n"
                        f"stderr:\n{proc.stderr}"
                    )

        missing_outputs = [
            sample_id for sample_id in expected_ids
            if not (output_root / sample_id / textured_mesh_filename).exists()
        ]
        if missing_outputs:
            preview = ", ".join(missing_outputs[:5])
            if len(missing_outputs) > 5:
                preview = f"{preview}, ..."
            raise FileNotFoundError(
                f"MVAdapter did not generate {textured_mesh_filename} for ids: {preview}. "
                f"Expected outputs under {output_root}/<id>/{textured_mesh_filename}."
            )

        processed = sum(
            1 for sample_id in expected_ids
            if (not preexisting[sample_id]) and (output_root / sample_id / textured_mesh_filename).exists()
        )
        skipped = sum(1 for sample_id in expected_ids if preexisting[sample_id])
        generated_outputs = [
            output_root / sample_id / textured_mesh_filename
            for sample_id in expected_ids if not preexisting[sample_id]
        ]
        for sample_id in expected_ids:
            if not preexisting[sample_id]:
                write_provenance(
                    output_root / sample_id / textured_mesh_filename,
                    provenance_by_id[sample_id],
                )
        summary = StageSummary(
            attempted=len(expected_ids),
            completed=processed,
            skipped=skipped,
            artifacts=[str(path) for path in generated_outputs],
            metadata={"texture_provider": texture_provider},
        )
        logger.info("ShapeMesh complete: %s", summary.to_dict())
        return summary


def main(argv: list[str] | None = None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="hairport-shape-mesh",
        description="Generate textured meshes from canonical shape_mesh inputs",
    )
    parser.add_argument("--data_dir", type=str, default="outputs")
    parser.add_argument("--shape_provider", type=str, default=None, choices=["hunyuan", "hi3dgen"])
    parser.add_argument("--texture_provider", type=str, default=None, choices=["mvadapter", "hunyuan"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    stage = ShapeMeshStage(device=args.device, seed=args.seed)
    result = stage.run(
        data_dir=args.data_dir,
        shape_provider=args.shape_provider,
        texture_provider=args.texture_provider,
    )
    print(f"ShapeMesh: {result}")


if __name__ == "__main__":
    main()
