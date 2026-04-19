"""hairport.data — Type-safe dataset path resolution and manifest management.

Provides :class:`DatasetManager` that encapsulates all directory/file path
logic for the HairPort pipeline, decoupling pipeline code from filesystem
layout decisions.

Usage::

    from hairport.data import DatasetManager

    dm = DatasetManager("outputs")
    img = dm.source_image("person_001")
    bald = dm.bald_image("person_001", bald_version="w_seg")
    pair_dir = dm.transfer_dir("tgt_001", "src_001", shape="hi3dgen", texture="mvadapter")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DatasetManager:
    """Centralised, type-safe path resolution for the HairPort data layout.

    All pipeline stages should call methods on this class instead of
    constructing paths ad-hoc.  This makes it trivial to migrate to a
    different directory layout in the future — only this file needs to change.

    Parameters
    ----------
    root : str | Path
        Root data directory (e.g. ``outputs/``).
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def provider_subdir(shape_provider: str, texture_provider: str) -> str:
        """Canonical provider directory name."""
        return f"shape_{shape_provider}__texture_{texture_provider}"

    @staticmethod
    def pair_folder_name(target_id: str, source_id: str) -> str:
        """Canonical pair folder name."""
        return f"{target_id}_to_{source_id}"

    # ------------------------------------------------------------------
    # Per-identity paths
    # ------------------------------------------------------------------

    def source_image(self, identity_id: str, folder: str = "image") -> Path:
        """Original source image."""
        return self.root / folder / f"{identity_id}.png"

    def matted_image_dir(self, identity_id: str) -> Path:
        """Matted (background-removed) image directory."""
        return self.root / "matted_image" / identity_id

    def landmarks_dir(self, identity_id: str) -> Path:
        """2D landmark directory."""
        return self.root / "lmk" / identity_id

    def landmarks_file(self, identity_id: str) -> Path:
        """2D landmarks .npy file."""
        return self.landmarks_dir(identity_id) / "landmarks.npy"

    def prompt_file(self, identity_id: str) -> Path:
        """Caption / prompt JSON."""
        return self.root / "prompt" / f"{identity_id}.json"

    def head_orientation_dir(self, identity_id: str) -> Path:
        """Head orientation cache directory for an identity."""
        return self.root / "head_orientation" / identity_id

    def head_orientation_file(self, identity_id: str) -> Path:
        """Head orientation JSON (computed via FLAMEFitter)."""
        return self.head_orientation_dir(identity_id) / "head_orientation.json"

    # ------------------------------------------------------------------
    # Bald images
    # ------------------------------------------------------------------

    def bald_dir(self, bald_version: str = "w_seg") -> Path:
        """Bald image root for a version."""
        return self.root / "bald" / bald_version

    def bald_image(self, identity_id: str, bald_version: str = "w_seg") -> Path:
        """Bald image for an identity."""
        return self.bald_dir(bald_version) / "image" / f"{identity_id}.png"

    def bald_outpainted_dir(self, bald_version: str = "w_seg") -> Path:
        """Outpainted bald images directory."""
        return self.bald_dir(bald_version) / "image_outpainted"

    # ------------------------------------------------------------------
    # 3D mesh & landmarks
    # ------------------------------------------------------------------

    def shape_mesh_dir(
        self, identity_id: str,
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        """Directory containing shape mesh for an identity."""
        if texture_provider == "hunyuan":
            return self.root / "hunyuan" / identity_id
        return self.root / texture_provider / shape_provider / identity_id

    def shape_mesh_file(
        self, identity_id: str,
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        """The simplified shape mesh GLB."""
        return self.shape_mesh_dir(identity_id, shape_provider, texture_provider) / "shape_mesh.glb"

    def landmarks_3d_dir(
        self, identity_id: str,
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        """3D landmarks output directory."""
        return (self.root / "lmk_3d"
                / self.provider_subdir(shape_provider, texture_provider)
                / identity_id)

    def landmarks_3d_file(
        self, identity_id: str,
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        return self.landmarks_3d_dir(identity_id, shape_provider, texture_provider) / "landmarks_3d.npy"

    def vertex_indices_file(
        self, identity_id: str,
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        return self.landmarks_3d_dir(identity_id, shape_provider, texture_provider) / "vertex_indices.npy"

    def postprocessed_mesh_file(
        self, identity_id: str,
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        return self.landmarks_3d_dir(identity_id, shape_provider, texture_provider) / "postprocessed_textured_mesh.glb"

    # ------------------------------------------------------------------
    # View-aligned / transfer hierarchy
    # ------------------------------------------------------------------

    def view_aligned_root(
        self,
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        """Root of view_aligned for a provider combination."""
        return self.root / "view_aligned" / self.provider_subdir(shape_provider, texture_provider)

    def transfer_dir(
        self,
        target_id: str,
        source_id: str,
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        """Root pair directory for a target→source transfer."""
        return (self.view_aligned_root(shape_provider, texture_provider)
                / self.pair_folder_name(target_id, source_id))

    def alignment_dir(
        self,
        target_id: str, source_id: str,
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        """Alignment subdirectory (rendered/enhanced views)."""
        return self.transfer_dir(target_id, source_id, shape_provider, texture_provider) / "alignment"

    def source_outpainted_dir(
        self,
        target_id: str, source_id: str,
        bald_version: str = "w_seg",
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        """Source outpainted images for a transfer pair."""
        return (self.transfer_dir(target_id, source_id, shape_provider, texture_provider)
                / bald_version / "source_outpainted")

    def camera_params_file(
        self,
        target_id: str, source_id: str,
        bald_version: str = "w_seg",
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        """Camera parameters JSON for a transfer pair."""
        return (self.transfer_dir(target_id, source_id, shape_provider, texture_provider)
                / bald_version / "camera_params.json")

    def warping_dir(
        self,
        target_id: str, source_id: str,
        bald_version: str = "w_seg",
        mode: str = "3d_aware",
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        """Warping results directory."""
        return (self.transfer_dir(target_id, source_id, shape_provider, texture_provider)
                / bald_version / mode / "warping")

    def blending_dir(
        self,
        target_id: str, source_id: str,
        bald_version: str = "w_seg",
        mode: str = "3d_aware",
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        """Blending results directory."""
        return (self.transfer_dir(target_id, source_id, shape_provider, texture_provider)
                / bald_version / mode / "blending")

    def transferred_dir(
        self,
        target_id: str, source_id: str,
        bald_version: str = "w_seg",
        mode: str = "3d_aware",
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        """Final transferred hair results directory."""
        return (self.transfer_dir(target_id, source_id, shape_provider, texture_provider)
                / bald_version / mode / "transferred_klein")

    # ------------------------------------------------------------------
    # Specific output files
    # ------------------------------------------------------------------

    def rendered_view_file(
        self,
        target_id: str, source_id: str,
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        """MV-Adapter rendered view."""
        return self.alignment_dir(target_id, source_id, shape_provider, texture_provider) / "target_image.png"

    def enhanced_view_file(
        self,
        target_id: str, source_id: str,
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
        phase: int = 1,
    ) -> Path:
        """Enhanced view (phase 1 or 2)."""
        return (self.alignment_dir(target_id, source_id, shape_provider, texture_provider)
                / f"target_image_phase_{phase}.png")

    def poisson_blended_file(
        self,
        target_id: str, source_id: str,
        bald_version: str = "w_seg",
        mode: str = "3d_aware",
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        """Poisson-blended image."""
        return self.blending_dir(
            target_id, source_id, bald_version, mode, shape_provider, texture_provider
        ) / "poisson_blended.png"

    def hair_restored_file(
        self,
        target_id: str, source_id: str,
        bald_version: str = "w_seg",
        mode: str = "3d_aware",
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> Path:
        """Final hair-restored image."""
        return self.transferred_dir(
            target_id, source_id, bald_version, mode, shape_provider, texture_provider
        ) / "hair_restored.png"

    # ------------------------------------------------------------------
    # Discovery / listing
    # ------------------------------------------------------------------

    def list_identities(self, folder: str = "image") -> list[str]:
        """List all identity IDs from a given image folder."""
        img_dir = self.root / folder
        if not img_dir.exists():
            return []
        return sorted(p.stem for p in img_dir.iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg"})

    def list_pairs(
        self,
        shape_provider: str = "hi3dgen",
        texture_provider: str = "mvadapter",
    ) -> list[tuple[str, str]]:
        """List all (target_id, source_id) pairs from view_aligned directories."""
        va_root = self.view_aligned_root(shape_provider, texture_provider)
        if not va_root.exists():
            return []
        pairs = []
        for d in sorted(va_root.iterdir()):
            if d.is_dir() and "_to_" in d.name:
                parts = d.name.split("_to_", 1)
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        return pairs

    def load_pairs_csv(self, csv_path: str | Path | None = None) -> list[tuple[str, str]]:
        """Load pairs from a CSV file.

        Falls back to ``data_dir/pairs.csv`` if no path provided.
        """
        import csv

        path = Path(csv_path) if csv_path else self.root / "pairs.csv"
        if not path.exists():
            logger.warning(f"Pairs CSV not found: {path}")
            return []
        pairs = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append((row["target_id"], row["source_id"]))
        return pairs

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def write_manifest(self, manifest_path: str | Path | None = None) -> Path:
        """Write a ``dataset.json`` manifest summarising the dataset.

        Returns the path to the written manifest.
        """
        path = Path(manifest_path) if manifest_path else self.root / "dataset.json"
        identities = self.list_identities()
        manifest = {
            "root": str(self.root),
            "num_identities": len(identities),
            "identities": identities,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest written: {path} ({len(identities)} identities)")
        return path

    def load_manifest(self, manifest_path: str | Path | None = None) -> dict:
        """Load a ``dataset.json`` manifest."""
        path = Path(manifest_path) if manifest_path else self.root / "dataset.json"
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"DatasetManager(root={str(self.root)!r})"
