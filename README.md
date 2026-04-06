# HairPort

**In-context 3D-Aware Hair Import and Transfer for Images**

> Bald Converters Model Weights: https://huggingface.co/deepmancer/bald_konverter

> Baldy Dataset: https://huggingface.co/datasets/deepmancer/baldy

Our codebase is still being finalized and will be runnable soon.

---

## Abstract

Transferring hairstyles between images is an important but challenging task in computer graphics, computer vision, and visual effects. It enables users to explore new looks without physically altering their hair, with applications in virtual try-on systems, augmented reality, and entertainment. Most prior works operate best under small pose gaps, and they fall short under large viewpoint and scale differences, where missing hair content must be synthesized rather than transferred.

We propose **HairPort**, a 3D-aware hairstyle transfer framework that addresses these issues by explicitly separating hair removal from transfer and enforcing geometric consistency before synthesis. We introduce a **Bald Converter**, which produces realistic bald versions of faces through LoRA-based in-context adaptation of FLUX. To train the Bald Converter, we introduce a new dataset, **Baldy**, containing 6,400 paired bald and original images across diverse identities and conditions. We also use a **3D-Aware Transfer Pipeline** that reconstructs and re-renders the reference hairstyle from the target viewpoint before compositing it onto the source image. Being 3D-aware, our method supports large pose and scale discrepancies between the source and target. With these components in place, we employ a conditional flow-matching generator to synthesize the final image conditioned on the bald source, the pose-aligned hair rendering, the original reference image, and a text prompt. Together, our method enables accurate, pose-consistent, and identity-preserving hairstyle transfer, outperforming existing methods both qualitatively and quantitatively.

---

## Pipeline Overview

HairPort processes images through a nine-stage pipeline:

| # | Stage | Description |
|---|-------|-------------|
| 1 | **Baldify** | Generate realistic bald portraits using FLUX LoRA in-context editing (via the Bald Converter). |
| 2 | **Caption** | Outpaint the bald images and generate text descriptions using Qwen Image-Edit. |
| 3 | **Shape Mesh** | Simplify and frontalize 3D head meshes from Hi3DGen or Hunyuan. |
| 4 | **Landmark 3D** | Estimate 3D facial landmarks via multi-view Blender renders + MediaPipe fusion. |
| 5 | **Align View** | Align the target hairstyle to the source viewpoint through camera optimization. |
| 6 | **Render View** | Generate textured multi-view images of the target hair using MV-Adapter + SDXL. |
| 7 | **Enhance View** | Refine rendered views with FLUX.2 Klein 9B img2img + CodeFormer face SR. |
| 8 | **Blend Hair** | Warp and Poisson-blend the enhanced hair onto the bald source. |
| 9 | **Transfer Hair** | Final compositing via conditional FLUX.2 Klein 9B generation (3D-aware + 3D-unaware). |

---

## Project Structure

```
HairPort/
├── configs/
│   └── default.yaml                # Centralized YAML configuration
├── scripts/
│   └── setup_submodules.sh         # Clone external git dependencies
├── assets/                         # Model weights & FLAME data (not in git)
│   ├── flame/FLAME2020/
│   ├── body_models/landmarks/flame/
│   └── checkpoints/
├── hairport/
│   ├── config.py                   # OmegaConf-based config system
│   ├── pipeline.py                 # Pipeline orchestrator (HairPortPipeline)
│   ├── data.py                     # DatasetManager — type-safe path resolution
│   ├── stages/                     # 9 pipeline stages (each a runnable module)
│   ├── core/                       # Shared components (BG removal, SAM, FLAME, etc.)
│   ├── bald_konverter/             # Bald conversion subpackage
│   ├── fit_lmk/                    # 3D facial landmark estimation
│   ├── utility/                    # Blender rendering, warping, outpainting
│   └── postprocessing/             # Hair transfer & mask helpers
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Requirements

- **Python** ≥ 3.10
- **CUDA GPU** — most stages require a CUDA-capable GPU (≥24 GB VRAM recommended)
- **Blender** ≥ 4.0 — used for multi-view rendering in stages 4 and 6 (installed separately or via `pip install bpy`)

---

## Installation

### 1. Create a conda environment

```bash
conda create -n hairport python=3.11 -y
conda activate hairport
```

### 2. Install PyTorch (with CUDA)

Follow the [official instructions](https://pytorch.org/get-started/locally/) for your CUDA version. For example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install HairPort

```bash
git clone https://github.com/deepmancer/HairPort.git
cd HairPort
pip install -e .
```

### 4. Set up external submodules

This clones CodeFormer, MV-Adapter, Hi3DGen, and SHeaP into `modules/`:

```bash
bash scripts/setup_submodules.sh
```

### 5. Download FLAME assets

1. Register and download **FLAME 2020** from [https://flame.is.tue.mpg.de](https://flame.is.tue.mpg.de).
2. Place the files under `assets/flame/FLAME2020/`:
   ```
   assets/flame/FLAME2020/
   ├── generic_model.pt
   ├── eyelids.pt
   └── FLAME_masks.pkl
   ```
3. Copy the MediaPipe–FLAME landmark mapping:
   ```
   assets/body_models/landmarks/flame/mediapipe_landmark_embedding.npz
   ```

### 6. HuggingFace models (auto-downloaded)

All other model weights (FLUX.2 Klein, SAM 3.1, BEN2, Qwen3-VL, etc.) are automatically downloaded from HuggingFace Hub on first use. Make sure you are logged in if any models require access:

```bash
huggingface-cli login
```

---

## Configuration

HairPort uses a centralized YAML configuration system backed by [OmegaConf](https://omegaconf.readthedocs.io/).

The default configuration lives in [`configs/default.yaml`](configs/default.yaml) and covers:
- **Global settings**: `device`, `seed`
- **Paths**: asset directories, external module locations
- **Models**: HuggingFace model IDs, LoRA weights, checkpoints
- **Per-stage parameters**: resolution, inference steps, thresholds, etc.
- **Prompts**: all text prompts used across stages

### Overriding configuration

There are three ways to override defaults, applied in this order of priority:

**1. Custom YAML file:**

```bash
hairport --config configs/my_experiment.yaml
```

**2. CLI dot-list overrides:**

```bash
hairport --set device=cpu seed=123 enhance_view.num_inference_steps=6
```

**3. Programmatic override:**

```python
from hairport.config import load_config, set_config

cfg = load_config(
    "configs/my_experiment.yaml",
    overrides=["device=cpu", "baldify.seed=123"],
)
set_config(cfg)
```

---

## Quick Start: Transferring a Hairstyle

Suppose you have:
- **source.png** — the person whose face you want to keep
- **reference.png** — the person whose hairstyle you want to transfer

Here's how to get the final transferred image.

### Step 1: Prepare the data directory

Create a working directory with the required input layout:

```bash
mkdir -p my_project/image
mkdir -p my_project/matted_image

# Place ALL images (both source and reference) in the image/ folder.
# The filename stem becomes the identity ID.
cp source.png    my_project/image/source.png
cp reference.png my_project/image/reference.png
```

### Step 2: Generate 3D head meshes (external prerequisite)

HairPort requires a textured 3D head mesh for each identity. These are generated **externally** using [Hi3DGen](https://github.com/Stable-X/Hi3DGen) (or Hunyuan3D) from the input images:

```bash
# Run Hi3DGen on each image to produce a .glb mesh, then place them as:
mkdir -p my_project/mvadapter/hi3dgen/source
mkdir -p my_project/mvadapter/hi3dgen/reference

# Each folder must contain a shape_mesh.glb file:
# my_project/mvadapter/hi3dgen/source/shape_mesh.glb
# my_project/mvadapter/hi3dgen/reference/shape_mesh.glb
```

> **Note:** Stage 3 (Shape Mesh) expects these meshes to already exist. It simplifies them but does not generate them from scratch.

### Step 3: Create a pairs file

Create `my_project/pairs.csv` to specify which transfers to perform. Each row defines a (target hairstyle → source face) pair:

```csv
target_id,source_id
reference,source
```

Here `target_id` is the identity whose **hair** you want (the reference), and `source_id` is the identity whose **face** you want to keep (the source). You can list multiple pairs.

### Step 4: Run the pipeline

```bash
hairport --data_dir my_project --shape_provider hi3dgen --texture_provider mvadapter --bald_version w_seg
```

This runs all nine stages sequentially. Depending on your GPU, the full pipeline takes roughly 5–15 minutes per pair.

### Step 5: Find the result

The final transferred image is saved at:

```
my_project/view_aligned/shape_hi3dgen__texture_mvadapter/reference_to_source/w_seg/3d_aware/transferred_klein/hair_restored.png
```

The intermediate outputs are also available within the same hierarchy:

```
my_project/
├── bald/w_seg/image/source.png                  # Bald source (Stage 1)
├── bald/w_seg/image_outpainted/source.png       # Outpainted bald (Stage 2)
├── mvadapter/hi3dgen/source/shape_mesh.glb      # Simplified mesh (Stage 3)
├── lmk_3d/shape_hi3dgen__texture_mvadapter/     # 3D landmarks (Stage 4)
│   └── source/landmarks_3d.npy
└── view_aligned/shape_hi3dgen__texture_mvadapter/
    └── reference_to_source/
        ├── alignment/                            # Aligned + rendered views (Stages 5–6)
        │   ├── target_image.png                  # MV-Adapter render
        │   └── target_image_phase_1.png          # Enhanced render (Stage 7)
        └── w_seg/
            └── 3d_aware/
                ├── warping/                      # Hair warping (Stage 8)
                ├── blending/poisson_blended.png  # Poisson-blended (Stage 8)
                └── transferred_klein/
                    └── hair_restored.png         # ✅ Final result (Stage 9)
```

### Using the Python API

```python
from hairport.pipeline import HairPortPipeline

pipeline = HairPortPipeline(
    data_dir="my_project",
    shape_provider="hi3dgen",
    texture_provider="mvadapter",
    bald_version="w_seg",
)
results = pipeline.run()

for r in results:
    status = "OK" if r.success else "FAIL"
    print(f"[{status}] {r.stage:20s}  {r.duration_seconds:.1f}s")
```

---

## Usage

### Full pipeline

Run all nine stages end-to-end:

```bash
hairport --data_dir outputs
```

The `--data_dir` argument points to a directory containing the input data (source images, target meshes, etc.) and where all intermediate and final outputs are written.

### Common options

```bash
# Choose shape and texture providers
hairport --data_dir outputs --shape_provider hi3dgen --texture_provider mvadapter

# Select bald conversion mode
hairport --data_dir outputs --bald_version w_seg    # with segmentation (higher quality)
hairport --data_dir outputs --bald_version wo_seg   # without segmentation (faster)

# Set device and seed
hairport --data_dir outputs --device cuda --seed 42

# Use a custom config
hairport --data_dir outputs --config configs/my_config.yaml --set seed=0
```

### Running specific stages

```bash
# Resume from a specific stage
hairport --data_dir outputs --start render_view

# Run up to a specific stage
hairport --data_dir outputs --end caption

# Run a range of stages
hairport --data_dir outputs --start render_view --end blend_hair

# Run only specific stages
hairport --data_dir outputs --only blend_hair transfer_hair

# Skip specific stages
hairport --data_dir outputs --skip shape_mesh landmark_3d

# Continue even if a stage fails
hairport --data_dir outputs --no-stop-on-error
```

### Running individual stages

Each stage has its own CLI entry point:

```bash
hairport-baldify     --data_dir outputs
hairport-caption     --data_dir outputs
hairport-shape-mesh  --data_dir outputs
hairport-landmark-3d --data_dir outputs
hairport-align-view  --data_dir outputs
hairport-render-view --data_dir outputs
hairport-enhance-view --data_dir outputs
hairport-blend-hair  --data_dir outputs
hairport-transfer-hair --data_dir outputs
```

### Python API

```python
from hairport.pipeline import HairPortPipeline

pipeline = HairPortPipeline(
    data_dir="outputs",
    shape_provider="hi3dgen",
    texture_provider="mvadapter",
    bald_version="w_seg",
)
results = pipeline.run()

# Run a subset of stages
results = pipeline.run(start="render_view", end="blend_hair")

# Run a single stage
results = pipeline.run(only=["transfer_hair"])
```

### Bald Converter (standalone)

The Bald Converter can be used independently:

```python
from hairport.bald_konverter import BaldKonverterPipeline

pipeline = BaldKonverterPipeline(mode="auto")
result = pipeline("portrait.jpg")
result.bald_image.save("bald.png")
```

Or via CLI:

```bash
# Single image
hairport-bald --input photo.jpg --output bald.png

# Batch processing
hairport-bald --input-dir ./faces/ --output-dir ./bald/

# With segmentation for higher quality
hairport-bald --input photo.jpg --output bald.png --mode w_seg
```

### 3D Landmark Estimation (standalone)

```python
from hairport.fit_lmk import estimate_3d_landmarks

results = estimate_3d_landmarks(
    mesh_path="head.glb",
    cam_loc=[0.0, -1.45, 0.0],
    cam_rot=[1.5708, 0.0, 0.0],
    output_dir="./landmarks_output",
)
# results['landmarks_3d'] → (N, 3) tensor
# results['vertex_indices'] → (N,) tensor
# results['confidences'] → (N,) tensor
```

---

## Data Layout

The pipeline expects and produces the following directory structure under `data_dir`:

```
data_dir/
├── image/                          # ← INPUT: All portrait images (source + reference)
│   ├── source.png
│   └── reference.png
├── matted_image/                   # Background-removed images (auto-generated)
├── pairs.csv                       # ← INPUT: Transfer pairs (target_id, source_id)
│
├── mvadapter/hi3dgen/              # ← INPUT: 3D head meshes from Hi3DGen
│   ├── source/shape_mesh.glb
│   └── reference/shape_mesh.glb
│
├── bald/                           # Generated by Stage 1–2
│   └── w_seg/
│       ├── image/                  # Bald portraits
│       └── image_outpainted/       # Outpainted bald images
├── lmk_3d/                         # Generated by Stage 4
│   └── shape_hi3dgen__texture_mvadapter/
│       └── <identity>/landmarks_3d.npy
│
└── view_aligned/                   # Generated by Stages 5–9
    └── shape_hi3dgen__texture_mvadapter/
        └── <target>_to_<source>/
            ├── alignment/          # Rendered + enhanced views
            └── <bald_version>/
                └── 3d_aware/
                    ├── warping/    # Warped hair
                    ├── blending/   # Poisson-blended composite
                    └── transferred_klein/
                        └── hair_restored.png   # ← FINAL OUTPUT
```

---

## External Dependencies

| Module | Repository | Purpose |
|--------|-----------|---------|
| CodeFormer | [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) | Face super-resolution |
| MV-Adapter | [huanngzh/MV-Adapter](https://github.com/huanngzh/MV-Adapter) | Multi-view generation adapter for SDXL |
| Hi3DGen | [Stable-X/Hi3DGen](https://github.com/Stable-X/Hi3DGen) | 3D head reconstruction |
| SHeaP | [deepmancer/SHeaP](https://github.com/deepmancer/SHeaP) | FLAME-based head segmentation |

These are cloned automatically by `scripts/setup_submodules.sh`.

### Key HuggingFace Models

| Model | ID | Used For |
|-------|----|----------|
| FLUX.2 Klein 9B | `black-forest-labs/FLUX.2-klein-9B` | Image enhancement & hair transfer |
| FLUX.1 Kontext | `black-forest-labs/FLUX.1-Kontext-dev` | Bald conversion (in-context editing) |
| SAM 3.1 | `facebook/sam3.1` | Hair/head mask segmentation |
| BEN2 | `PramaLLC/BEN2` | Background removal |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | Image captioning |
| Qwen Image-Edit | `Qwen/Qwen-Image-Edit` | Image outpainting |
| RealVisXL V4.0 | `SG161222/RealVisXL_V4.0` | SDXL base model for rendering |
| MV-Adapter | `huanngzh/mv-adapter` | Multi-view adapter weights |

---

## License

This project is released under the [MIT License](LICENSE).

Copyright (c) 2026
