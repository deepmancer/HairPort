"""Runtime contracts for reliable, reproducible HairPort inference."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from omegaconf import OmegaConf

from hairport.config import get_config

logger = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = 2


@dataclass
class ItemFailure:
    """Failure associated with a concrete sample or pipeline operation."""

    item_id: str
    error: str


@dataclass
class StageSummary:
    """Uniform output contract for all top-level inference stages."""

    attempted: int = 0
    completed: int = 0
    skipped: int = 0
    failures: list[ItemFailure] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return not self.failures

    def add_failure(self, item_id: str, error: Exception | str) -> None:
        self.failures.append(ItemFailure(item_id=item_id, error=str(error)))

    def add_artifacts(self, paths: Iterable[str | Path]) -> None:
        self.artifacts.extend(str(path) for path in paths)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["success"] = self.success
        return payload


def derive_seed(
    global_seed: int | None,
    stage_name: str,
    item_id: str = "",
    bald_version: str = "",
    conditioning_source: str = "",
    operation_name: str = "",
) -> int:
    """Derive an order-independent Torch-compatible seed from run identity."""
    seed = -1 if global_seed is None else int(global_seed)
    if seed < 0:
        return seed
    token = "\0".join(
        (
            str(seed),
            stage_name,
            item_id,
            bald_version,
            conditioning_source,
            operation_name,
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % (2**31 - 1)


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _determinism_settings() -> dict[str, Any]:
    try:
        import torch

        return {
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
    except ImportError:
        return {"torch_available": False}


@lru_cache(maxsize=1)
def _code_revisions() -> dict[str, Any]:
    """Capture repository/module revisions used to generate artifacts."""
    repo_root = Path(__file__).resolve().parent.parent

    def revision(path: Path) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    modules = {
        name: revision(repo_root / "modules" / name)
        for name in ("CodeFormer", "MV-Adapter", "PEAR")
    }
    dirty = None
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"repository": revision(repo_root), "dirty": dirty, "modules": modules}


def build_provenance(
    stage: str,
    *,
    seed: int | None,
    inputs: Iterable[str | Path] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic artifact provenance for cache validation."""
    cfg = OmegaConf.to_container(get_config(), resolve=True)
    input_payload = []
    for input_path in inputs:
        path = Path(input_path)
        input_payload.append({"path": str(path), "sha256": _file_sha256(path)})
    payload: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "stage": stage,
        "seed": seed,
        "inputs": input_payload,
        "config": cfg,
        "code_revisions": _code_revisions(),
        "determinism": _determinism_settings(),
        "metadata": dict(metadata or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    payload["signature"] = hashlib.sha256(encoded).hexdigest()
    return payload


def provenance_path(artifact_path: str | Path) -> Path:
    artifact = Path(artifact_path)
    return artifact.with_name(f"{artifact.name}.provenance.json")


def can_reuse_artifact(artifact_path: str | Path, expected: Mapping[str, Any]) -> bool:
    """Return true only for an existing artifact with matching provenance."""
    if expected.get("code_revisions", {}).get("dirty"):
        # A dirty checkout does not identify the code that made an artifact.
        # Publication-grade cache reuse resumes once changes are committed.
        return False
    artifact = Path(artifact_path)
    sidecar = provenance_path(artifact)
    if not artifact.exists() or not sidecar.exists():
        return False
    try:
        with open(sidecar) as handle:
            actual = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return actual.get("signature") == expected.get("signature")


def write_provenance(artifact_path: str | Path, provenance: Mapping[str, Any]) -> Path:
    """Atomically record provenance once an artifact has been generated."""
    output = provenance_path(artifact_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(f"{output.suffix}.tmp")
    with open(tmp, "w") as handle:
        json.dump(dict(provenance), handle, indent=2, sort_keys=True, default=str)
    tmp.replace(output)
    return output
