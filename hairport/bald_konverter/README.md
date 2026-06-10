# BaldKonverter

Generate bald versions of portrait images using FLUX LoRA models.

## Quick Start

### Installation

BaldKonverter ships as part of the HairPort repository (package path:
`hairport.bald_konverter`). From the repository root:

```bash
pip install -r requirements.txt   # runtime dependencies
pip install -e .                  # installs `hairport` + the `bald-konverter` CLI

# Head fitting backend (PEAR submodule + assets) and pytorch3d:
bash scripts/setup_submodules.sh
```

> **Note** — `black-forest-labs/FLUX.1-Kontext-dev` is a gated Hugging Face
> repo: accept its license on huggingface.co and authenticate with
> `huggingface-cli login` before first use.

### Python API

```python
from hairport.bald_konverter import BaldKonverterPipeline

# Full quality (auto = wo_seg → w_seg refinement)
pipeline = BaldKonverterPipeline(mode="auto")
result = pipeline("portrait.jpg")
result.bald_image.save("bald.png")

# Fast mode (no segmentation preprocessing)
pipeline = BaldKonverterPipeline(mode="wo_seg")
result = pipeline("portrait.jpg")
result.bald_image.save("bald_fast.png")

# With intermediates (masks, FLUX inputs)
result = pipeline("portrait.jpg", return_intermediates=True)
result.bald_image.save("bald.png")
result.hair_mask  # numpy array (H, W), 0/255
result.body_mask  # numpy array (H, W), 0/255
result.flux_input_wo_seg.save("flux_two_panel.png")   # 2-panel wo_seg input
result.flux_input_w_seg.save("flux_grid.png")         # 4-panel w_seg input
```

### Command Line

Installed as `bald-konverter` via `pip install -e .`
(equivalently: `python -m hairport.bald_konverter.cli`).

```bash
# Single image
bald-konverter --input photo.jpg --output bald.png

# Batch processing
bald-konverter --input-dir ./faces/ --output-dir ./bald/

# Fast mode
bald-konverter --input photo.jpg --output bald.png --mode wo_seg

# Save intermediate masks
bald-konverter --input photo.jpg --output bald.png --save-intermediates
```

> `w_seg` / `auto` modes always run the SMPL-X head/body fit (requires
> `modules/PEAR` — `bash scripts/setup_submodules.sh`). `wo_seg` needs no fitting.

### Fitted head/body models

In `w_seg` / `auto` modes the pipeline runs the configured fitting backend
(`cfg.fitting.backend`, default [PEAR](https://github.com/Pixel-Talk/PEAR) —
SMPL-X + FLAME EHM recovery) on the intermediate bald image. The fit drives the
head mask in the 4-panel grid and is **persisted by default** next to each
output as `<stem>_head_fit.pt` (pipeline stage:
`<data_dir>/bald/<version>/head_fit/<id>.pt`, resolvable via
`DatasetManager.bald_head_fit(...)`).

The artifact is a plain `torch.save` dict (schema v2 — `backend`,
`smplx_params`, `flame_params`, `camera`, full `vertices`/`faces`, FLAME
`head_vertices`/`head_faces`, rendered `head_mask`, `head_orientation`,
`image_size`, `source`), so it loads anywhere with `torch.load(path)`. For
typed access:

```python
from hairport.bald_konverter import BodyFitResult

fit = BodyFitResult.load("outputs/bald/w_seg/head_fit/person_001.pt")
fit.flame_params["expression_params"]  # (1, 50) FLAME expression
fit.smplx_params["body_pose"]          # (1, 21, 3, 3) SMPL-X body pose
fit.head_vertices                      # (~5143, 3) posed FLAME head submesh
fit.head_mask                          # (H, W) uint8 FLAME head silhouette
fit.body_mask                          # (H, W) uint8 full SMPL-X silhouette (top-right cell)
fit.head_orientation                   # {"euler_angles_xyz_radians": [[x,y,z]], ...}
```

Configuration: `baldify.persist_head_fits` (default `true`) in
`configs/default.yaml`. `w_seg` / `auto` modes always run the fit.

## Architecture

### Non-square inputs & VFX composite-back

The FLUX LoRA is square/portrait-trained, so the model always sees a square
canvas — but the converter accepts **any aspect ratio / resolution** and returns
the **same-dimensioned original with only the head changed**:

1. **Preprocess** the original (BEN2 foreground + SAM3 hair) at full resolution.
2. **Framing** (`framing.py`) — extract a *head-centric square plate* (face box ∪
   hair box, scaled by `baldify.framing.crop_scale`), reflect-padded at borders.
   The model runs on this plate; `Framing` stores the exact extract/paste maps.
3. **Composite** (`compositing.py`) the bald plate back into the original at native
   resolution. The bald plate is composited **as-is** (no color matching — the
   model's direct output already matches the input). The matte is built for hair
   *removal*: the SAM hair seed is grown outward through the pixels the model
   actually changed (``|orig − bald| > extend_diff_threshold`` within
   ``extend_band_frac × side``) so the wispy strands the segmenter misses are
   covered (no residual-hair halo), with an **outward-only** feather so the blend
   ramp lives in clean background and never re-introduces hair → alpha comp →
   screened-Poisson seam → grain match. The matte is zero at the plate border, so
   untouched pixels are **byte-identical** to the input.

Everything is configurable under `baldify.framing` / `baldify.compositing`, and
every intermediate (plate, bald plate, alpha matte, all masks, framing JSON,
SMPL-X `.pt`, and a `manifest.json` of the exact matte/grain/blend coefficients)
is saved with `--save-intermediates` / `baldify.persist_intermediates` for later
access. The model-core stages below run on the **plate**:

### Stage 1: wo_seg (two-panel FLUX generation)
- Source plate is placed on the left of a side-by-side panel
- FLUX inpaints the right half as the bald version using a LoRA adapter
- Fast — but still composites back (needs SAM3 + BEN2 for the matte)

### Stage 2: Preprocessing (for w_seg mode)
- **SAM3** (`facebook/sam3`) — text-prompted hair mask extraction
- **BEN2** (`PramaLLC/BEN2`) — foreground / background separation
- **PEAR** (SMPL-X + FLAME EHM) — head fit + the **SMPL-X body silhouette** used
  as the bald segmentation target. The silhouettes are rendered at
  `fitting.render_size` (default **768**) and warped (bilinear) to the source
  frame — the model detection patch stays 256 (the EHM ViT's trained size).

### Stage 3: w_seg (four-panel FLUX generation)
- Assembles a 2×2 grid:
  - **top-left** — SAM3 hair (red) over the bald body silhouette (green)
  - **top-right** — the **SMPL-X full-mesh silhouette** (green): bald seg target
  - **bottom-left** — original image; **bottom-right** — inpainted bald result
- FLUX inpaints the bottom-right quadrant guided by the segmentation context
- Higher quality than wo_seg alone

## Modes

| Mode | Speed | Quality | Preprocessing |
|------|-------|---------|---------------|
| `wo_seg` | Fast | Good | None |
| `w_seg` | Slow | Best | SAM3 + BEN2 + PEAR (SMPL-X) |
| `auto` | Slow | Best | Runs wo_seg then w_seg |

## Model Checkpoints

LoRA checkpoints are hosted on [Hugging Face Hub](https://huggingface.co/deepmancer/bald_konverter)
and downloaded automatically on first use:

- `bald_konvertor_wo_seg_000003400.safetensors` — 2-panel LoRA
- `bald_konvertor_w_seg_000004900.safetensors` — 4-panel LoRA

Base model: [`black-forest-labs/FLUX.1-Kontext-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)

## Package Structure

```
hairport/bald_konverter/
├── __init__.py              # Public API
├── pipeline.py              # End-to-end orchestrator
├── cli.py                   # Command-line interface
├── config/
│   └── defaults.py          # Constants (prompts, sizes, model IDs)
├── models/
│   ├── hub.py               # HF Hub download helpers
│   ├── konverter.py         # BaldKonverter, BaldKonverterWithSeg
│   └── toolkit/             # Vendored ai-toolkit snapshot (training-time code).
│                            # Only pipeline_flux_inpaint.py is used at inference;
│                            # the remaining files are NOT importable here.
├── preprocessing/
│   ├── background.py        # BackgroundRemover (BEN2)
│   ├── hair_mask.py         # HairMaskPipeline (BEN2 silhouette + SAM3 hair)
│   └── sam_extractor.py     # SAMMaskExtractor (SAM3)
└── utils/
    └── image.py             # Grid assembly, crop, mask helpers
```

Head fitting lives in the repo-level [`hairport.fitting`](../fitting/) package
(backend-agnostic; PEAR by default), not in this module.

## Requirements

- Python ≥ 3.10
- CUDA GPU with ≥ 24 GB VRAM (for FLUX model)
- Core: `torch`, `diffusers`, `transformers`, `ben2`, `safetensors`, `huggingface-hub`
- Head fitting: `ultralytics`, `lightning`, `kornia`, `smplx`, `pytorch3d` + modules/PEAR

## License

This component follows the repository license: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0). Third-party dependencies and model weights retain their own upstream licenses and terms.
