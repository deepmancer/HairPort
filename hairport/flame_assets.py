"""FLAME asset path helpers and deterministic model conversion utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def _load_pkl_format_flame_model(path: str | Path) -> Dict[str, torch.Tensor]:
    """Load FLAME model parameters from a pickle file into torch tensors."""
    path = Path(path)
    with open(path, "rb") as handle:
        flame_data = pickle.load(handle, encoding="latin1")

    params: dict[str, torch.Tensor] = {}
    params["faces"] = torch.from_numpy(np.asarray(flame_data["f"], dtype=np.int64))
    kintree = torch.from_numpy(np.asarray(flame_data["kintree_table"], dtype=np.int64))
    kintree[kintree > 100] = -1
    params["kintree"] = kintree

    j_regressor = flame_data["J_regressor"]
    if hasattr(j_regressor, "toarray"):
        j_regressor = j_regressor.toarray()
    params["J_regressor"] = torch.from_numpy(np.asarray(j_regressor, dtype=np.float32))

    for key in ("shapedirs", "J", "weights", "posedirs", "v_template"):
        params[key] = torch.from_numpy(np.asarray(flame_data[key], dtype=np.float32))
    return params


def convert_flame_model_pkl_to_pt(
    model_source: str | Path,
    model_runtime: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert FLAME pickle model to torch `.pt` format at a target path."""
    source_path = Path(model_source)
    runtime_path = Path(model_runtime)
    if runtime_path.exists() and not overwrite:
        return runtime_path
    if not source_path.exists():
        raise FileNotFoundError(f"FLAME model source pickle not found: {source_path}")

    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_load_pkl_format_flame_model(source_path), runtime_path)
    return runtime_path


def ensure_flame_model_runtime(
    model_source: str | Path,
    model_runtime: str | Path,
) -> Path:
    """Ensure torch runtime FLAME model exists, converting from source pickle if needed."""
    runtime_path = Path(model_runtime)
    if runtime_path.exists():
        return runtime_path
    return convert_flame_model_pkl_to_pt(model_source, runtime_path, overwrite=False)

