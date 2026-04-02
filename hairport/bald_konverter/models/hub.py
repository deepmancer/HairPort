"""Hugging Face Hub helpers for downloading LoRA checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

from ..config.defaults import HF_REPO_ID, LORA_FILENAME_W_SEG, LORA_FILENAME_WO_SEG

logger = logging.getLogger(__name__)

_VARIANT_MAP = {
    "wo_seg": LORA_FILENAME_WO_SEG,
    "w_seg": LORA_FILENAME_W_SEG,
}


def download_checkpoint(
    variant: str = "wo_seg",
    repo_id: str = HF_REPO_ID,
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> str:
    """Download a LoRA checkpoint from the Hub and return the local path.

    Parameters
    ----------
    variant : ``"wo_seg"`` or ``"w_seg"``
        Which LoRA checkpoint to fetch.
    repo_id : str
        Hugging Face repository ID.
    cache_dir : str | Path, optional
        Override for the local cache directory.
    token : str, optional
        Hugging Face auth token (for gated repos).

    Returns
    -------
    str
        Absolute local path to the downloaded ``.safetensors`` file.
    """
    if variant not in _VARIANT_MAP:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(_VARIANT_MAP)}")

    filename = _VARIANT_MAP[variant]
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir,
        token=token,
    )
    logger.info("Downloaded %s checkpoint → %s", variant, local_path)
    return local_path
