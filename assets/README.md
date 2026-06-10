# HairPort Assets Directory

This directory holds landmark embeddings and (optionally) checkpoint weights
used by the HairPort framework. Large weights are **not checked into Git** (see
`.gitignore`); a few small landmark files and the README figures are tracked
directly.

> **Note** — the head/body parametric models (SMPL-X, SMPL, FLAME) live with the
> **PEAR** fitting submodule under `modules/PEAR/assets/`, not here. They are
> downloaded by `scripts/setup_submodules.sh`. See the repository README §5.

## Expected layout

```
assets/
├── landmarks/
│   └── flame/
│       ├── mediapipe_landmark_embedding.npz   # MediaPipe ↔ FLAME landmark map
│       ├── flame_static_embedding_68_v4.npz
│       └── landmark_embedding.npy
├── images/                                    # README figures (tracked)
└── checkpoints/
    └── (optional model checkpoints, e.g. pear/ from older runs — gitignored)
```

## How to populate

**One step (recommended):**

```bash
python scripts/download_assets.py        # landmark embeddings into assets/landmarks/
bash   scripts/setup_submodules.sh        # PEAR parametric models into modules/PEAR/assets/
```

By downloading the parametric models you agree to the
[FLAME](https://flame.is.tue.mpg.de) and
[SMPL-X](https://smpl-x.is.tue.mpg.de) license terms.

**Other checkpoints** (BEN2, SAM3, FLUX LoRAs, PEAR EHM) are
auto-downloaded from the Hugging Face Hub the first time the corresponding
pipeline is loaded.
