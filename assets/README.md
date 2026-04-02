# HairPort Assets Directory

This directory holds all model weights, checkpoints, and configuration files
required by the HairPort framework. It is **not checked into Git** (see
`.gitignore`).

## Expected layout

```
assets/
├── flame/
│   └── FLAME2020/
│       ├── generic_model.pt
│       ├── eyelids.pt
│       └── FLAME_masks.pkl
├── body_models/
│   └── landmarks/
│       └── flame/
│           └── mediapipe_landmark_embedding.npz
└── checkpoints/
    └── (additional model checkpoints)
```

## How to populate

1. **FLAME 2020** — Download from https://flame.is.tue.mpg.de and place files
   under `assets/flame/FLAME2020/`.

2. **MediaPipe–FLAME mapping** — Copy or symlink
   `mediapipe_landmark_embedding.npz` into
   `assets/body_models/landmarks/flame/`.

3. **Other checkpoints** — Will be auto-downloaded by HuggingFace Hub the
   first time the corresponding pipeline is loaded (BEN2, SAM 3.1, Qwen3-VL,
   etc.).
