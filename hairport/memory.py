"""GPU model-residency management: keep at most one large model on the GPU.

Stages use :func:`on_gpu` windows (or :func:`offload`). Two config knobs:
``memory.policy`` — ``"exclusive"`` (default; park models in CPU RAM between
windows) or ``"resident"`` (move/offload become no-ops). ``memory.flux_offload``
— diffusers offload: ``"none"`` (default), ``"model"`` (per-component), or
``"sequential"`` (lowest VRAM, slowest).
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch

logger = logging.getLogger(__name__)

VALID_POLICIES = ("exclusive", "resident")
VALID_OFFLOAD_MODES = ("none", "model", "sequential")


# --------------------------------------------------------------------------- #
# Config access
# --------------------------------------------------------------------------- #

def memory_policy() -> str:
    """Current ``memory.policy`` (safe default: ``"exclusive"``)."""
    try:
        from hairport.config import get_config

        return str(get_config().memory.policy)
    except Exception:  # config not initialised (e.g. unit tests)
        return "exclusive"


def exclusive_enabled() -> bool:
    """True when models should be offloaded between usage windows."""
    return memory_policy() == "exclusive"


def flux_offload_mode() -> str:
    """Current ``memory.flux_offload`` (safe default: ``"none"``)."""
    try:
        from hairport.config import get_config

        return str(get_config().memory.flux_offload)
    except Exception:
        return "none"


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

def flush() -> None:
    """Release Python + CUDA allocator garbage (single canonical helper)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _has_accelerate_hooks(obj: Any) -> bool:
    """True when a diffusers pipeline manages its own placement via accelerate.

    ``enable_model_cpu_offload`` / ``enable_sequential_cpu_offload`` populate
    ``pipe._all_hooks``; such pipelines must not be moved with ``.to``.
    """
    return bool(getattr(obj, "_all_hooks", None))


def move_to(obj: Any, device: str | torch.device) -> Any:
    """Move a ``torch.nn.Module`` or diffusers pipeline to *device*.

    Safe no-op for pipelines whose placement is managed by accelerate
    offload hooks (``enable_model_cpu_offload`` etc.) and for objects
    without a ``.to`` method.
    """
    if obj is None:
        return obj
    if _has_accelerate_hooks(obj):
        return obj
    to = getattr(obj, "to", None)
    if callable(to):
        obj = to(device) or obj
    return obj


def offload(obj: Any) -> Any:
    """Park *obj* in CPU RAM and release the freed VRAM.

    No-op under the ``resident`` policy so call sites do not need their own
    policy checks.
    """
    if obj is None or not exclusive_enabled():
        return obj
    obj = move_to(obj, "cpu")
    flush()
    return obj


@contextmanager
def on_gpu(
    obj: Any,
    device: str | torch.device,
    exclusive: Optional[bool] = None,
) -> Iterator[Any]:
    """Context manager: *obj* on *device* inside, offloaded to CPU after.

    Parameters
    ----------
    obj : nn.Module | DiffusionPipeline | None
        Model to manage (``None`` yields ``None``).
    device : str | torch.device
        Target device for the usage window.
    exclusive : bool, optional
        Override the global policy (default: ``memory.policy == "exclusive"``).
    """
    if obj is None:
        yield None
        return
    exclusive = exclusive_enabled() if exclusive is None else exclusive
    move_to(obj, device)
    try:
        yield obj
    finally:
        if exclusive:
            move_to(obj, "cpu")
            flush()


# --------------------------------------------------------------------------- #
# Diffusers pipeline placement
# --------------------------------------------------------------------------- #

def apply_offload_mode(pipe: Any, mode: str, device: str | torch.device) -> Any:
    """Place a diffusers pipeline according to ``memory.flux_offload``.

    ``"none"`` → ``pipe.to(device)``; ``"model"`` / ``"sequential"`` →
    the corresponding accelerate-managed offload (after which the pipeline
    must NOT be moved with ``.to`` — :func:`move_to` already guards this).
    """
    if mode not in VALID_OFFLOAD_MODES:
        raise ValueError(
            f"Invalid offload mode {mode!r}; choose from {VALID_OFFLOAD_MODES}"
        )
    if mode == "none":
        pipe.to(device)
    elif mode == "model":
        pipe.enable_model_cpu_offload(device=device)
        logger.info("Pipeline using model-level CPU offload on %s", device)
    else:
        pipe.enable_sequential_cpu_offload(device=device)
        logger.info("Pipeline using sequential CPU offload on %s", device)
    return pipe
