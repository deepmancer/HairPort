#!/usr/bin/env python3
"""Download HairPort landmark/FLAME assets for the landmark & alignment stages.

Fetches ``hairport_data.zip`` from https://huggingface.co/deepmancer/bald_konverter
and extracts its ``base_models/`` and ``landmarks/`` trees into ``assets/`` (the
layout in ``configs/default.yaml``, e.g. ``landmarks/flame/mediapipe_landmark_embedding.npz``).
The fitting backend (PEAR) ships its own SMPL-X/FLAME models under
``modules/PEAR/assets`` (see ``scripts/setup_submodules.sh``) — separate from these.

Idempotent (``--force`` to overwrite). The FLAME model is under its own license
(https://flame.is.tue.mpg.de) — downloading implies acceptance.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ID = "deepmancer/bald_konverter"
ZIP_FILENAME = "hairport_data.zip"
# Top-level trees inside the zip that we extract into assets/.
WANTED_PREFIXES = ("base_models/", "landmarks/")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "assets",
        help="Destination assets directory (default: <repo>/assets).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite files that already exist."
    )
    args = parser.parse_args(argv)

    from huggingface_hub import hf_hub_download

    print(f"Downloading {REPO_ID}/{ZIP_FILENAME} (cached after first run)...")
    zip_path = hf_hub_download(repo_id=REPO_ID, filename=ZIP_FILENAME)

    assets_dir: Path = args.assets_dir
    assets_dir.mkdir(parents=True, exist_ok=True)

    extracted, skipped = 0, 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if info.is_dir() or not name.startswith(WANTED_PREFIXES):
                continue
            target = assets_dir / name
            if not target.resolve().is_relative_to(assets_dir.resolve()):
                raise RuntimeError(f"Refusing unsafe zip path: {name}")
            if target.exists() and not args.force:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            extracted += 1

    print(f"Done: {extracted} files extracted, {skipped} already present.")
    print(f"Assets root: {assets_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
