#!/usr/bin/env python3
"""Download and place every data asset HairPort + PEAR need at runtime.

Fetches ``hairport_data.zip`` from https://huggingface.co/deepmancer/bald_konverter
and extracts it into the repository, placing each file where the code expects it:

    assets/landmarks/...                 # MediaPipe/FLAME landmark embeddings
    modules/PEAR/assets/FLAME/...        # PEAR FLAME runtime assets
    modules/PEAR/assets/SMPLX/...        # PEAR SMPL-X runtime assets

The bundle holds only the files the pipeline actually loads (PEAR head/body
fitting + silhouette render). Model weights that auto-download at first run are
NOT bundled: the PEAR EHM checkpoint (``BestWJH/PEAR_models``, fetched by
``hairport.fitting``) and the YOLO detector (``yolov8x.pt``, fetched by
ultralytics).

Idempotent (``--force`` to overwrite). The FLAME / SMPL-X models are under their
own licenses (https://flame.is.tue.mpg.de, https://smpl-x.is.tue.mpg.de) —
downloading implies acceptance.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ID = "deepmancer/bald_konverter"
ZIP_FILENAME = "hairport_data.zip"
# Top-level trees inside the zip, extracted relative to the repo root.
WANTED_PREFIXES = ("assets/", "modules/")


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root,
        help="Repository root to extract into (default: the HairPort checkout).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite files that already exist."
    )
    args = parser.parse_args(argv)

    from huggingface_hub import hf_hub_download

    print(f"Downloading {REPO_ID}/{ZIP_FILENAME} (cached after first run)...")
    zip_path = hf_hub_download(repo_id=REPO_ID, filename=ZIP_FILENAME)

    root: Path = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    extracted, skipped = 0, 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or not name.startswith(WANTED_PREFIXES):
                continue
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"Refusing unsafe zip path: {name}")
            if target.exists() and not args.force:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted += 1

    print(f"Done: {extracted} files extracted, {skipped} already present.")
    print(f"Placed under: {root}/assets and {root}/modules/PEAR/assets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
