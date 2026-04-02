# BaldKonverter

Generate bald versions of portrait images using FLUX LoRA models.

## Quick Start

### Installation

```bash
pip install bald-konverter

# With optional FLAME-based segmentation:
pip install bald-konverter[flame]
```

### Python API

```python
from bald_konverter import BaldKonverterPipeline

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
result.flux_input.save("flux_grid.png")
```

### Command Line

```bash
# Single image
bald-konverter --input photo.jpg --output bald.png

# Batch processing
bald-konverter --input-dir ./faces/ --output-dir ./bald/

# Fast mode
bald-konverter --input photo.jpg --output bald.png --mode wo_seg

# Save intermediate masks
bald-konverter --input photo.jpg --output bald.png --save-intermediates

# With FLAME fitting
bald-konverter --input photo.jpg --output bald.png --use-flame --flame-dir FLAME2020/
```

## Architecture

The pipeline runs in up to three stages:

### Stage 1: wo_seg (two-panel FLUX generation)
- Source image is placed on the left of a side-by-side panel
- FLUX inpaints the right half as the bald version using a LoRA adapter
- Fast — no preprocessing models required

### Stage 2: Preprocessing (for w_seg mode)
- **SAM3** (`facebook/sam3`) — text-prompted hair mask extraction
- **BEN2** (`PramaLLC/BEN2`) — foreground / background separation
- **Segformer** (`jonathandinu/face-parsing`) — neck detection for body mask
- Optionally **SHeaP + FLAME** — 3D head mesh fitting for precise segmentation

### Stage 3: w_seg (four-panel FLUX generation)
- Assembles a 2×2 grid: segmentation masks (top) + original/bald images (bottom)
- FLUX inpaints the bottom-right quadrant guided by segmentation context
- Higher quality than wo_seg alone

## Modes

| Mode | Speed | Quality | Preprocessing |
|------|-------|---------|---------------|
| `wo_seg` | Fast | Good | None |
| `w_seg` | Slow | Best | SAM3 + BEN2 + Segformer |
| `auto` | Slow | Best | Runs wo_seg then w_seg |

## Model Checkpoints

LoRA checkpoints are hosted on [Hugging Face Hub](https://huggingface.co/deepmancer/bald_konverter)
and downloaded automatically on first use:

- `bald_konvertor_wo_seg_000003400.safetensors` — 2-panel LoRA
- `bald_konvertor_w_seg_000004900.safetensors` — 4-panel LoRA

Base model: [`black-forest-labs/FLUX.1-Kontext-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)

## Package Structure

```
src/bald_konverter/
├── __init__.py              # Public API
├── pipeline.py              # End-to-end orchestrator
├── cli.py                   # Command-line interface
├── config/
│   ├── defaults.py          # Constants (prompts, sizes, model IDs)
│   └── segmentation.py      # SegformerClass, ExtendedLIPClass, color palettes
├── models/
│   ├── hub.py               # HF Hub download helpers
│   └── konverter.py         # BaldKonverter, BaldKonverterWithSeg
├── preprocessing/
│   ├── background.py        # BackgroundRemover (BEN2)
│   ├── face_parser.py       # FaceParser (Segformer)
│   ├── flame.py             # FLAMESegmenter (optional SHeaP)
│   ├── hair_mask.py         # HairMaskPipeline (orchestrator)
│   └── sam_extractor.py     # SAMMaskExtractor (SAM3)
└── utils/
    └── image.py             # Grid assembly, crop, mask helpers
```

## Requirements

- Python ≥ 3.10
- CUDA GPU with ≥ 24 GB VRAM (for FLUX model)
- Core: `torch`, `diffusers`, `transformers`, `ben2`, `safetensors`, `huggingface-hub`
- Optional: `sheap`, `mediapipe`, `roma`, `scipy` (for FLAME fitting)

## License

MIT
