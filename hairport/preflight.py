"""Preflight validation for HairPort inference dependencies and assets."""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path
from typing import Iterable

from hairport.config import add_config_args, get_config, load_config_from_args

logger = logging.getLogger(__name__)


def validate_preflight(stages: Iterable[str] | None = None) -> list[str]:
    """Validate filesystem/runtime prerequisites for selected pipeline stages.

    Returns informational warnings. Required missing inputs raise immediately
    with a consolidated actionable message.
    """
    cfg = get_config()
    selected = set(stages or ())
    if not selected:
        selected = {
            "baldify", "caption", "shape_mesh", "landmark_3d", "align_view",
            "render_view", "enhance_view", "blend_hair", "transfer_hair",
        }

    required: list[tuple[Path, str]] = []
    missing_runtime: list[str] = []
    flame_users = selected & {"landmark_3d", "align_view", "blend_hair"}
    if flame_users:
        flame_dir = Path(cfg.paths.flame_dir)
        required.extend(
            [
                (flame_dir / cfg.models.flame_model, "user-supplied FLAME model"),
                (flame_dir / "eyelids.pt", "user-supplied FLAME eyelids asset"),
                (Path(cfg.paths.sheap_module), "SHeaP module"),
            ]
        )
        if importlib.util.find_spec("sheap") is None:
            missing_runtime.append(
                "importable Python package 'sheap' "
                f"(install with: pip install -e {Path(cfg.paths.sheap_module)})"
            )
    if "landmark_3d" in selected:
        required.append(
            (Path(cfg.paths.mediapipe_flame_embedding), "MediaPipe/FLAME embedding")
        )
    if selected & {"shape_mesh", "render_view"}:
        required.append((Path(cfg.paths.mv_adapter_module), "MV-Adapter module"))
    if "render_view" in selected:
        required.extend(
            (
                Path(cfg.paths.mv_adapter_module) / cfg.render_view.lora_dir / filename,
                "MV-Adapter render LoRA",
            )
            for filename in cfg.render_view.lora_files
        )
    if selected & {"enhance_view", "blend_hair"}:
        required.append((Path(cfg.paths.codeformer_module), "CodeFormer module"))

    missing = [f"{label}: {path}" for path, label in required if not path.exists()]
    missing.extend(missing_runtime)
    if missing:
        raise FileNotFoundError(
            "HairPort preflight failed. Install/provide required inference assets:\n  "
            + "\n  ".join(missing)
        )

    warnings: list[str] = []
    if "landmark_3d" in selected:
        from hairport.fit_lmk.ray_intersector import RayMeshIntersector

        backend = RayMeshIntersector.preflight_backend()
        if backend != "embree":
            warnings.append(
                "Embree is unavailable; Landmark3D will use the slower trimesh triangle backend."
            )
    revision_requirements = {
        "baldify": [
            "flux_kontext_revision", "bald_konverter_revision",
            "sam_bald_konverter_revision", "ben2_revision", "face_parser_revision",
        ],
        "caption": [
            "captioner_revision", "qwen_image_edit_revision",
            "qwen_lightning_lora_revision",
        ],
        "landmark_3d": ["sam_revision"],
        "render_view": ["realvis_v4_revision", "sdxl_vae_revision", "mv_adapter_revision", "ben2_revision"],
        "align_view": [
            "realvis_v5_lightning_revision", "sdxl_vae_revision",
            "controlnet_union_revision",
        ],
        "blend_hair": ["sam_revision", "ben2_revision"],
        "enhance_view": ["flux_klein_revision"],
        "transfer_hair": ["flux_klein_revision", "sam_revision", "ben2_revision"],
    }
    missing_revisions = sorted(
        {
            field
            for stage, fields in revision_requirements.items()
            if stage in selected
            for field in fields
            if not getattr(cfg.models, field)
        }
    )
    if missing_revisions:
        warnings.append(
            "Model revisions are not pinned for this run: "
            + ", ".join(missing_revisions)
            + ". Set immutable snapshot commits before recording paper results."
        )
    for warning in warnings:
        logger.warning(warning)
    return warnings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify HairPort inference prerequisites.")
    parser.add_argument("--stages", nargs="*", default=None)
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)
    warnings = validate_preflight(args.stages)
    print("HairPort preflight passed.")
    for warning in warnings:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
