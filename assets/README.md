# HairPort Assets Directory

This directory holds all model weights, checkpoints, and configuration files
required by the HairPort framework. It is **not checked into Git** (see
`.gitignore`).

## Expected layout

```
assets/
├── base_models/
│   └── flame/
│       ├── parametric_models/
│       │   ├── generic_model.pkl
│       │   └── generic_model.pt
│       └── vertex_mappings/
│           └── FLAME_masks.pkl
├── landmarks/
│   └── flame/
│       ├── eyelids.pt
│       ├── mediapipe_landmark_embedding.npz
│       └── flame_landmark_idxs_barys.pt
└── checkpoints/
    └── (additional model checkpoints)
```

## How to populate

1. **FLAME base model + masks** — Place `generic_model.pkl` at
   `assets/base_models/flame/parametric_models/` and `FLAME_masks.pkl` at
   `assets/base_models/flame/vertex_mappings/`.

2. **Landmark assets** — Place `eyelids.pt` and
   `mediapipe_landmark_embedding.npz` under `assets/landmarks/flame/`.
   `flame_landmark_idxs_barys.pt` is tracked for interoperability but not
   currently consumed by the inference runtime.

3. **Runtime conversion** — `generic_model.pt` is auto-generated from
   `generic_model.pkl` during setup and preflight if missing.

4. **Other checkpoints** — Will be auto-downloaded by HuggingFace Hub the
   first time the corresponding pipeline is loaded (BEN2, SAM 3.1, etc.).
