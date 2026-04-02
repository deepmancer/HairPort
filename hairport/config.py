"""Centralized configuration for the HairPort framework.

Loads ``configs/default.yaml``, merges optional user overrides and CLI
dot-list flags, and exposes a typed :class:`HairPortConfig` object via
:func:`get_config` / :func:`set_config`.

Usage::

    from hairport.config import get_config

    cfg = get_config()                  # auto-loads configs/default.yaml
    print(cfg.models.flux_klein)        # "black-forest-labs/FLUX.2-klein-9B"
    print(cfg.paths.flame_dir)          # resolved absolute path

Override at startup::

    from hairport.config import load_config, set_config
    cfg = load_config("configs/my_experiment.yaml",
                       overrides=["device=cpu", "baldify.seed=123"])
    set_config(cfg)

CLI helper for argparse::

    from hairport.config import add_config_args, load_config_from_args
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_config_from_args(args)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf


# --------------------------------------------------------------------------- #
#  Structured config dataclasses  (schema for OmegaConf)
# --------------------------------------------------------------------------- #

@dataclass
class PathsConfig:
    assets_dir: str = "assets"
    modules_dir: str = "modules"
    output_dir: str = "outputs"
    flame_dir: str = "${paths.assets_dir}/flame/FLAME2020"
    codeformer_module: str = "${paths.modules_dir}/CodeFormer"
    codeformer_weights: str = "${paths.assets_dir}/weights/codeformer"
    mv_adapter_module: str = "${paths.modules_dir}/MV-Adapter"
    sheap_module: str = "${paths.modules_dir}/SHeaP"
    hi3dgen_module: str = "${paths.modules_dir}/Hi3DGen"
    mediapipe_flame_embedding: str = (
        "${paths.assets_dir}/body_models/landmarks/flame/"
        "mediapipe_landmark_embedding.npz"
    )


@dataclass
class ModelsConfig:
    # Diffusion
    flux_klein: str = "black-forest-labs/FLUX.2-klein-9B"
    flux_kontext: str = "black-forest-labs/FLUX.1-Kontext-dev"
    realvis_v4: str = "SG161222/RealVisXL_V4.0"
    realvis_v5_lightning: str = "SG161222/RealVisXL_V5.0_Lightning"
    sdxl_vae: str = "madebyollin/sdxl-vae-fp16-fix"
    controlnet_union: str = "xinsir/controlnet-union-sdxl-1.0"
    # MV-Adapter
    mv_adapter: str = "huanngzh/mv-adapter"
    mv_adapter_weight: str = "mvadapter_ig2mv_sdxl.safetensors"
    # Segmentation
    sam: str = "facebook/sam3.1"
    sam_bald_konverter: str = "facebook/sam3"
    ben2: str = "PramaLLC/BEN2"
    face_parser: str = "jonathandinu/face-parsing"
    # VL / captioning
    captioner: str = "Qwen/Qwen3-VL-8B-Instruct"
    qwen_image_edit: str = "Qwen/Qwen-Image-Edit"
    qwen_lightning_lora: str = "lightx2v/Qwen-Image-Lightning"
    qwen_lightning_lora_weight: str = "Qwen-Image-Lightning-8steps-V1.1.safetensors"
    # Bald konverter
    bald_konverter_repo: str = "deepmancer/bald_konverter"
    bald_lora_wo_seg: str = "bald_konvertor_wo_seg_000003400.safetensors"
    bald_lora_w_seg: str = "bald_konvertor_w_seg_000004900.safetensors"
    # FLAME
    flame_model: str = "generic_model.pt"
    # Enhancement
    lora_detail_xl: str = "add-detail-xl.safetensors"
    # Misc
    rembg_session: str = "birefnet-general"


@dataclass
class SAMSectionConfig:
    confidence_threshold: float = 0.35
    detection_threshold: float = 0.4
    hair_confidence_threshold: float = 0.25


@dataclass
class BGRemovalConfig:
    alpha_threshold: float = 0.8


@dataclass
class FlameConfig:
    model_type: str = "expressive"
    detection_confidence: float = 0.5
    padding_ratio: float = 0.1


@dataclass
class CodeFormerConfig:
    face_size: int = 512
    upscale: int = 2
    bg_tile: int = 100
    bg_tile_pad: int = 10


@dataclass
class FacialLandmarksConfig:
    detection_confidence: float = 0.5
    fallback_confidences: List[float] = field(
        default_factory=lambda: [0.3, 0.2, 0.1]
    )
    min_face_size: int = 64
    target_face_size: int = 256


@dataclass
class BaldifyConfig:
    mode: str = "auto"
    guidance_scale: float = 1.0
    num_inference_steps: int = 35
    strength: float = 1.0
    seed: int = 42
    dtype: str = "bfloat16"
    wo_seg_image_size: int = 768
    w_seg_image_size: int = 1024


@dataclass
class CaptionConfig:
    resize_percentage: int = 30
    num_inference_steps: int = 8
    max_sequence_length: int = 512
    overlap_percentage: int = 5
    true_cfg_scale: float = 1.0
    height: int = 1024
    width: int = 1024


@dataclass
class ShapeMeshConfig:
    target_faces: int = 150000
    quality_threshold: float = 0.8
    extra_tex_coord_weight: float = 4.0
    min_size_mb: float = 20.0
    frontalize_min_size_mb: float = 25.0


@dataclass
class Landmark3DConfig:
    ortho_scale: float = 1.1
    num_perturbations: int = 4
    angle_range: float = 0.15
    trans_range: float = 0.05
    resolution: int = 1024
    optimize: bool = True
    super_resolution: bool = True
    codeformer_fidelity: float = 0.0
    codeformer_upscale: int = 2
    default_cam_location: List[float] = field(
        default_factory=lambda: [0.0, -1.45, 0.0]
    )
    default_cam_rotation: List[float] = field(
        default_factory=lambda: [1.5708, 0.0, 0.0]
    )


@dataclass
class AlignViewConfig:
    angle_threshold_3d_lift: float = 10.0
    render_resolution: int = 1024


@dataclass
class RenderViewConfig:
    num_views: int = 6
    num_inference_steps: int = 50
    guidance_scale: float = 3.0
    reference_conditioning_scale: float = 1.0
    control_conditioning_scale: float = 1.0
    height: int = 1024
    width: int = 1024
    dtype: str = "float16"
    lora_dir: str = "loras"
    lora_files: List[str] = field(
        default_factory=lambda: ["add-detail-xl.safetensors"]
    )
    lora_scales: List[float] = field(default_factory=lambda: [0.8])
    ortho_scale_offset: float = 0.2
    camera_near: float = 0.1
    camera_far: float = 100.0


@dataclass
class EnhanceViewConfig:
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    height: int = 1024
    width: int = 1024
    max_image_size: int = 1024
    bg_color: List[int] = field(default_factory=lambda: [255, 255, 255])
    padding_ratio: float = 0.05


@dataclass
class BlendHairConfig:
    resolution: int = 1024
    optimization_resolution: int = 1024
    codeformer_upscale: int = 2
    codeformer_fidelity: float = 0.5
    alignment_iou_weight: float = 1.0
    alignment_landmark_weight: float = 1.0
    sam_confidence_threshold: float = 0.4
    poisson_blend_strength: float = 0.5
    dilation_size: int = 15
    blur_size: int = 2
    gaussian_sigma: int = 21
    feather_px: int = 12
    mask_threshold: float = 0.5


@dataclass
class TransferHairConfig:
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
    processing_resolution: int = 1024
    output_resolution: int = 1024
    seed: int = 42
    bg_color: List[int] = field(default_factory=lambda: [255, 255, 255])
    non_hair_fg_color: List[int] = field(default_factory=lambda: [200, 200, 200])
    uncrop_hair_threshold: float = 0.75
    uncrop_border_threshold: float = 0.025
    uncrop_resize_percentage: float = 80.0


@dataclass
class UncropConfig:
    width: int = 1024
    height: int = 1024
    overlap_percentage: int = 5
    num_inference_steps: int = 12
    default_resize_percentage: float = 75.0
    blend_pixels: int = 21
    face_to_width_ratio: float = 0.45
    min_resize_percentage: float = 30.0
    max_resize_percentage: float = 100.0
    dtype: str = "float16"


@dataclass
class RenderingConfig:
    engine: str = "CYCLES"
    cycles_samples: int = 512
    adaptive_threshold: float = 0.01
    adaptive_min_samples: int = 64
    tile_size: int = 256
    max_bounces: int = 12
    diffuse_bounces: int = 4
    glossy_bounces: int = 4
    transmission_bounces: int = 12
    volume_bounces: int = 0
    preview_samples: int = 32
    resolution_percentage: int = 100
    default_camera_location: List[float] = field(
        default_factory=lambda: [0.0, -1.2, 1.82]
    )
    default_camera_rotation_deg: List[float] = field(
        default_factory=lambda: [90.0, 0.0, 0.0]
    )
    default_ortho_scale: float = 1.0


@dataclass
class RenderingFitLmkConfig:
    cycles_samples: int = 256


@dataclass
class DatasetConfig:
    provider_pattern: str = "shape_{shape}__texture_{texture}"
    dir_image: str = "image"
    dir_matted_image: str = "matted_image"
    dir_landmarks: str = "lmk"
    dir_landmarks_3d: str = "lmk_3d"
    dir_pixel3dmm: str = "pixel3dmm_output"
    dir_view_aligned: str = "view_aligned"
    dir_source_outpainted: str = "source_outpainted"
    dir_bald: str = "bald"
    dir_prompts: str = "prompt"
    dir_3d_aware: str = "3d_aware"
    dir_3d_unaware: str = "3d_unaware"
    subdir_warping: str = "warping"
    subdir_blending: str = "blending"
    subdir_transferred: str = "transferred_klein"
    subdir_alignment: str = "alignment"
    subdir_bald_image: str = "image"
    subdir_bald_lmk: str = "lmk"
    file_head_orientation: str = "head_orientation.json"
    file_landmarks: str = "landmarks.npy"
    file_vertex_indices: str = "vertex_indices.npy"
    file_textured_mesh: str = "postprocessed_textured_mesh.glb"
    file_shape_mesh: str = "shape_mesh.glb"
    file_aligned_mesh: str = "aligned_target_mesh.glb"
    file_camera_params: str = "camera_params.json"
    file_enhanced_render: str = "source_alignment.png"
    file_hair_restored: str = "hair_restored.png"
    file_hair_restored_mask: str = "hair_restored_mask.png"
    file_poisson_blended: str = "poisson_blended.png"
    file_target_phase1: str = "target_image_phase_1.png"
    file_target_phase1_mask: str = "target_image_phase_1_mask.png"


@dataclass
class PipelineSectionConfig:
    shape_provider: str = "hi3dgen"
    texture_provider: str = "mvadapter"
    bald_version: str = "w_seg"


@dataclass
class PromptsConfig:
    baldify_wo_seg: str = ""
    baldify_w_seg: str = ""
    enhance_first_phase: str = ""
    enhance_second_phase: str = ""
    transfer_3d_aware: str = ""
    transfer_3d_aware_wo_bald: str = ""
    transfer_3d_unaware: str = ""
    transfer_uncrop: str = ""
    caption_outpaint: str = ""
    uncrop_default: str = ""
    uncrop_negative: str = ""
    render_view_negative: str = ""


# ---- Master config ---- #

@dataclass
class HairPortConfig:
    device: str = "cuda"
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    sam: SAMSectionConfig = field(default_factory=SAMSectionConfig)
    bg_removal: BGRemovalConfig = field(default_factory=BGRemovalConfig)
    flame: FlameConfig = field(default_factory=FlameConfig)
    codeformer: CodeFormerConfig = field(default_factory=CodeFormerConfig)
    facial_landmarks: FacialLandmarksConfig = field(
        default_factory=FacialLandmarksConfig
    )
    baldify: BaldifyConfig = field(default_factory=BaldifyConfig)
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    shape_mesh: ShapeMeshConfig = field(default_factory=ShapeMeshConfig)
    landmark_3d: Landmark3DConfig = field(default_factory=Landmark3DConfig)
    align_view: AlignViewConfig = field(default_factory=AlignViewConfig)
    render_view: RenderViewConfig = field(default_factory=RenderViewConfig)
    enhance_view: EnhanceViewConfig = field(default_factory=EnhanceViewConfig)
    blend_hair: BlendHairConfig = field(default_factory=BlendHairConfig)
    transfer_hair: TransferHairConfig = field(default_factory=TransferHairConfig)
    uncrop: UncropConfig = field(default_factory=UncropConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    rendering_fit_lmk: RenderingFitLmkConfig = field(
        default_factory=RenderingFitLmkConfig
    )
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    pipeline: PipelineSectionConfig = field(default_factory=PipelineSectionConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)


# --------------------------------------------------------------------------- #
#  Repo-root detection
# --------------------------------------------------------------------------- #

def _detect_root() -> Path:
    """Walk up from this file to find the repo root (contains pyproject.toml)."""
    current = Path(__file__).resolve().parent  # hairport/
    for parent in [current, current.parent, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent


# --------------------------------------------------------------------------- #
#  Default YAML path
# --------------------------------------------------------------------------- #

def _default_yaml_path() -> Path:
    """Return ``<repo_root>/configs/default.yaml``."""
    return _detect_root() / "configs" / "default.yaml"


# --------------------------------------------------------------------------- #
#  Loading & merging
# --------------------------------------------------------------------------- #

def load_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Build the merged configuration.

    Merge order (later wins):
      1. Structured dataclass defaults  (``HairPortConfig``)
      2. ``configs/default.yaml``
      3. User-supplied *config_path* YAML (optional)
      4. CLI dot-list *overrides* (optional)

    After merging, relative paths in ``paths.*`` are resolved to absolute
    paths against the detected repo root.

    Returns a frozen :class:`DictConfig`.
    """
    # 1) Structured defaults
    schema = OmegaConf.structured(HairPortConfig)

    # 2) Default YAML
    default_yaml = _default_yaml_path()
    if default_yaml.exists():
        yaml_cfg = OmegaConf.load(default_yaml)
    else:
        yaml_cfg = OmegaConf.create()

    # 3) Optional user override YAML
    if config_path is not None:
        user_cfg = OmegaConf.load(str(config_path))
    else:
        user_cfg = OmegaConf.create()

    # 4) CLI overrides
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
    else:
        cli_cfg = OmegaConf.create()

    # Merge
    merged = OmegaConf.merge(schema, yaml_cfg, user_cfg, cli_cfg)

    # Environment variable overrides
    if root := os.environ.get("HAIRPORT_ROOT"):
        OmegaConf.update(merged, "paths.assets_dir", f"{root}/assets")
        OmegaConf.update(merged, "paths.modules_dir", f"{root}/modules")
        OmegaConf.update(merged, "paths.output_dir", f"{root}/outputs")
    if device := os.environ.get("HAIRPORT_DEVICE"):
        OmegaConf.update(merged, "device", device)

    # Resolve interpolations
    OmegaConf.resolve(merged)

    # Resolve relative paths to absolute
    root_dir = _detect_root()
    _resolve_paths(merged, root_dir)

    # Freeze
    OmegaConf.set_readonly(merged, True)
    return merged


def _resolve_paths(cfg: DictConfig, root: Path) -> None:
    """Make every ``paths.*`` value absolute if it is relative."""
    for key in list(cfg.paths):
        val = cfg.paths[key]
        if isinstance(val, str):
            p = Path(val)
            if not p.is_absolute():
                OmegaConf.update(cfg, f"paths.{key}", str(root / p))


# --------------------------------------------------------------------------- #
#  Singleton
# --------------------------------------------------------------------------- #

_default_config: Optional[DictConfig] = None


def get_config() -> DictConfig:
    """Return the module-level default config, creating it on first call."""
    global _default_config
    if _default_config is None:
        _default_config = load_config()
    return _default_config


def set_config(config: DictConfig) -> None:
    """Override the module-level default config singleton."""
    global _default_config
    _default_config = config


def reset_config() -> None:
    """Clear the cached singleton so the next :func:`get_config` reloads."""
    global _default_config
    _default_config = None


# --------------------------------------------------------------------------- #
#  Argparse helper
# --------------------------------------------------------------------------- #

def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--config`` and ``--set`` arguments to an argparse parser.

    ``--config`` accepts a path to an override YAML file.
    ``--set`` accepts one or more ``key=value`` pairs in OmegaConf dot-list
    notation, e.g. ``--set device=cpu baldify.seed=123``.
    """
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to an override YAML config file.",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        nargs="*",
        default=None,
        help="Override config values via dot-list, e.g. --set device=cpu",
    )


def load_config_from_args(args: argparse.Namespace) -> DictConfig:
    """Build config from parsed argparse namespace and install as singleton."""
    cfg = load_config(
        config_path=getattr(args, "config", None),
        overrides=getattr(args, "config_overrides", None),
    )
    set_config(cfg)
    return cfg


# --------------------------------------------------------------------------- #
#  Convenience accessors
# --------------------------------------------------------------------------- #

def get_path(name: str) -> Path:
    """Return an absolute ``Path`` for a key in ``cfg.paths``."""
    return Path(get_config().paths[name])


def get_model(name: str) -> str:
    """Return a model ID string from ``cfg.models``."""
    return str(get_config().models[name])
