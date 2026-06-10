"""Centralized configuration for the HairPort framework.

Loads ``configs/default.yaml``, merges optional user overrides and CLI
dot-list flags, and exposes a typed :class:`HairPortConfig` object via
:func:`get_config` / :func:`set_config`.

Usage::

    from hairport.config import get_config

    cfg = get_config()                  # auto-loads configs/default.yaml
    print(cfg.models.flux_klein)        # "black-forest-labs/FLUX.2-klein-9B"
    print(cfg.fitting.backend)          # "pear"

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
    codeformer_module: str = "${paths.modules_dir}/CodeFormer"
    codeformer_weights: str = "${paths.assets_dir}/weights/codeformer"
    mv_adapter_module: str = "${paths.modules_dir}/MV-Adapter"
    pear_module: str = "${paths.modules_dir}/PEAR"
    mediapipe_flame_embedding: str = (
        "${paths.assets_dir}/landmarks/flame/"
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
    # VL / captioning
    captioner: str = "Qwen/Qwen3-VL-8B-Instruct"
    qwen_image_edit: str = "Qwen/Qwen-Image-Edit"
    qwen_lightning_lora: str = "lightx2v/Qwen-Image-Lightning"
    qwen_lightning_lora_weight: str = "Qwen-Image-Lightning-8steps-V1.1.safetensors"
    # Bald konverter
    bald_konverter_repo: str = "deepmancer/bald_konverter"
    bald_lora_wo_seg: str = "bald_konvertor_wo_seg_000003400.safetensors"
    bald_lora_w_seg: str = "bald_konvertor_w_seg_000004900.safetensors"
    # Enhancement
    lora_detail_xl: str = "add-detail-xl.safetensors"
    # Misc
    rembg_session: str = "birefnet-general"
    # Immutable Hugging Face revisions for publication runs. Fill with snapshot commits.
    flux_klein_revision: Optional[str] = None
    flux_kontext_revision: Optional[str] = None
    realvis_v4_revision: Optional[str] = None
    realvis_v5_lightning_revision: Optional[str] = None
    sdxl_vae_revision: Optional[str] = None
    controlnet_union_revision: Optional[str] = None
    mv_adapter_revision: Optional[str] = None
    sam_revision: Optional[str] = None
    sam_bald_konverter_revision: Optional[str] = None
    ben2_revision: Optional[str] = None
    captioner_revision: Optional[str] = None
    qwen_image_edit_revision: Optional[str] = None
    qwen_lightning_lora_revision: Optional[str] = None
    bald_konverter_revision: Optional[str] = None


@dataclass
class SAMSectionConfig:
    confidence_threshold: float = 0.35
    detection_threshold: float = 0.4
    hair_confidence_threshold: float = 0.25


@dataclass
class BGRemovalConfig:
    alpha_threshold: float = 0.8


@dataclass
class FittingConfig:
    """Human/head fitting backend (see :mod:`hairport.fitting`)."""

    # Registered backend key. Currently only "pear" (SMPL-X + FLAME EHM).
    backend: str = "pear"
    # Vendored PEAR submodule root (CWD the adapter chdirs into).
    pear_module: str = "${paths.modules_dir}/PEAR"
    # PEAR model checkpoint on the Hugging Face Hub.
    pear_repo_id: str = "BestWJH/PEAR_models"
    pear_checkpoint: str = "ehm_model_stage1.pt"
    pear_config: str = "infer"
    # Person detector weights, relative to the PEAR module root.
    detector_weights: str = "model_zoo/yolov8x.pt"
    # Feed PEAR the BEN2-matted (background-removed) image.
    matting: bool = True
    # SMPL-X silhouette render resolution (square). The model patch stays 256.
    render_size: int = 768


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
class FramingConfig:
    """Head-centric square-plate framing (non-square input support)."""

    # Square plate side = crop_scale × max(head_bbox_w, head_bbox_h).
    crop_scale: float = 1.8
    # Border fill for plate pixels outside the image: "reflect" | "constant".
    border_pad_mode: str = "reflect"


@dataclass
class CompositingConfig:
    """VFX composite of the bald plate back into the original frame.

    Hair-*removal* matte: the SAM hair seed is grown outward through the pixels
    the model actually changed (the wisp band) so no residual-hair halo remains.
    The bald plate is composited as-is (no color matching — the model's direct
    output already matches the input).
    """

    seam_poisson: bool = True            # screened-Poisson seam pass
    grain_match: bool = True             # add noise matched to the original
    matte_dilate_px: int = 6             # seed close / small safety dilation
    extend_band_frac: float = 0.06       # outward search band = frac × plate side
    extend_diff_threshold: int = 12      # |orig−bald| above this = model-changed
    feather_px: int = 5                  # outward-only matte feather
    border_zero_frac: float = 0.02       # zero the matte in this border band


@dataclass
class BaldifyConfig:
    mode: str = "auto"
    # Generation parameters matching the LoRA training configuration.
    guidance_scale: float = 1.0
    num_inference_steps: int = 35
    strength: float = 1.0
    seed: int = 42
    dtype: str = "bfloat16"
    # wo_seg: per-panel size (2-panel input is 2*size × size = 1536×768).
    wo_seg_image_size: int = 768
    # w_seg: full 2×2 grid size (each panel is size/2 = 512×512).
    w_seg_image_size: int = 1024
    framing: FramingConfig = field(default_factory=FramingConfig)
    compositing: CompositingConfig = field(default_factory=CompositingConfig)
    # Persist intermediates (plate, bald_plate, masks, matte, framing, head fit,
    # manifest) next to outputs for later access/analysis.
    persist_intermediates: bool = True


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
    input_mesh_dir: str = "shape_mesh"
    input_mesh_filename: str = "shape_mesh.glb"
    input_image_dir: str = "matted_image_centered"
    prompt_subdir: str = "prompt"
    output_mesh_filename: str = "textured_mesh.glb"
    variant: str = "sdxl"
    sdxl_model_id: str = "SG161222/RealVisXL_V4.0"
    reference_conditioning_scale: float = 1.0
    align_input_mesh: bool = False
    align_output_mesh: bool = True
    preprocess_mesh: bool = True
    remove_bg: bool = False
    default_prompt_text: str = (
        "high quality photo, photograph of a person, ultra-detailed, "
        "strand-level hair, 8k, realistic hair texture"
    )


@dataclass
class Landmark3DConfig:
    ortho_scale: float = 1.1
    textured_mesh_ortho_scale: float = 0.5
    num_perturbations: int = 0
    angle_range: float = 0.15
    trans_range: float = 0.05
    resolution: int = 1024
    optimize: bool = True
    super_resolution: bool = True
    textured_mesh_filename: str = "textured_mesh.glb"
    postprocessed_mesh_filename: str = "postprocessed_textured_mesh.glb"
    target_landmark_extent: float = 0.4
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
    conditioning_phase: int = 2


@dataclass
class BlendHairConfig:
    resolution: int = 1024
    optimization_resolution: int = 1024
    alignment_iou_weight: float = 1.0
    alignment_landmark_weight: float = 1.0
    sam_confidence_threshold: float = 0.4
    # Folder under <data_dir>/ providing target images for 3D-unaware blending:
    #   "auto"             — "image_outpainted" for celeba_reduced datasets,
    #                        "image" otherwise (legacy name-based behavior)
    #   "image"            — always use original images
    #   "image_outpainted" — always use outpainted images
    target_image_folder: str = "auto"


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
    conditioning_sources: List[str] = field(
        default_factory=lambda: ["enhanced", "blended"]
    )


@dataclass
class CacheConfig:
    """Policy for reusing generated inference artifacts."""

    policy: str = "validated"


@dataclass
class MemoryConfig:
    """GPU model-residency policy (see :mod:`hairport.memory`)."""

    # "exclusive": at most one large model on GPU at a time — others are
    #              parked in CPU RAM between usage windows (default).
    # "resident":  legacy behavior — models stay where they were loaded.
    policy: str = "exclusive"
    # Component-level diffusers offload for the big pipelines
    # (FLUX.1-Kontext / FLUX.2 Klein / SDXL): none | model | sequential.
    flux_offload: str = "none"


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
    dir_head_orientation: str = "head_orientation"
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
    fitting: FittingConfig = field(default_factory=FittingConfig)
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
    cache: CacheConfig = field(default_factory=CacheConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
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
    _validate_config(merged)

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


def _validate_config(cfg: DictConfig) -> None:
    """Reject published-pipeline settings that cannot be run correctly."""
    if int(cfg.landmark_3d.num_perturbations) != 0:
        raise ValueError(
            "landmark_3d.num_perturbations must be 0: the supported HairPort "
            "inference pipeline uses single frontal-view landmark projection."
        )
    conditioning_sources = list(cfg.transfer_hair.conditioning_sources)
    valid_sources = {"enhanced", "blended"}
    invalid = sorted(set(conditioning_sources) - valid_sources)
    if not conditioning_sources or invalid:
        raise ValueError(
            "transfer_hair.conditioning_sources must contain one or more of "
            f"{sorted(valid_sources)}; invalid values: {invalid}"
        )
    if str(cfg.cache.policy) != "validated":
        raise ValueError("cache.policy must be 'validated' for reproducible inference.")
    if int(cfg.enhance_view.conditioning_phase) not in (1, 2):
        raise ValueError("enhance_view.conditioning_phase must be 1 or 2.")
    if str(cfg.blend_hair.target_image_folder) not in ("auto", "image", "image_outpainted"):
        raise ValueError(
            "blend_hair.target_image_folder must be 'auto', 'image', or "
            "'image_outpainted'."
        )
    if str(cfg.memory.policy) not in ("exclusive", "resident"):
        raise ValueError("memory.policy must be 'exclusive' or 'resident'.")
    if str(cfg.memory.flux_offload) not in ("none", "model", "sequential"):
        raise ValueError(
            "memory.flux_offload must be 'none', 'model', or 'sequential'."
        )
    if str(cfg.fitting.backend) not in ("pear",):
        raise ValueError("fitting.backend must be 'pear'.")


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
