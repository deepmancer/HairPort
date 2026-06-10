# HairPort Assets Directory

This directory holds landmark embeddings and (optionally) checkpoint weights
used by the HairPort framework. Large weights are **not checked into Git** (see
`.gitignore`); a few small landmark files and the README figures are tracked
directly.

> **Note** — the head/body parametric models (SMPL-X, FLAME) used by the **PEAR**
> fitting submodule live under `modules/PEAR/assets/`, not here. They ship in the
> same `hairport_data.zip` bundle and are placed by `scripts/download_assets.py`.
> See the repository README §5.

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

`scripts/download_assets.py` fetches `hairport_data.zip` from the Hugging Face Hub
and places **both** the landmark embeddings (`assets/landmarks/`) and the PEAR
runtime models (`modules/PEAR/assets/{FLAME,SMPLX}`):

```bash
python scripts/download_assets.py        # all data assets
bash   scripts/setup_submodules.sh        # submodules + pytorch3d (also runs the above)
```

By downloading the parametric models you agree to the
[FLAME](https://flame.is.tue.mpg.de) and
[SMPL-X](https://smpl-x.is.tue.mpg.de) license terms.

**Other checkpoints** (BEN2, SAM3, FLUX LoRAs, PEAR EHM) are
auto-downloaded from the Hugging Face Hub the first time the corresponding
pipeline is loaded.
