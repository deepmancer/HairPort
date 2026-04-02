"""hairport.pipeline — Orchestrate the full HairPort processing pipeline.

Provides :class:`HairPortPipeline` that sequences the nine stages defined in
``hairport.stages``, with support for:

- Running the full pipeline or a subset of stages.
- Resuming from a given stage.
- Per-stage hooks for logging / monitoring.
- Shared ``PipelineContext`` passed across stages.

Usage::

    from hairport.pipeline import HairPortPipeline

    pipeline = HairPortPipeline(data_dir="outputs")
    pipeline.run()                           # run all stages
    pipeline.run(start="render_view")        # resume from render_view
    pipeline.run(only=["blend_hair"])        # run single stage
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from hairport.config import get_config

logger = logging.getLogger(__name__)


# ============================================================================
# Stage Registry
# ============================================================================

#: Ordered list of canonical stage names.
STAGE_ORDER: list[str] = [
    "baldify",
    "caption",
    "shape_mesh",
    "landmark_3d",
    "align_view",
    "render_view",
    "enhance_view",
    "blend_hair",
    "transfer_hair",
]


def _get_stage_class(name: str):
    """Lazy-import a stage class by canonical name."""
    if name == "baldify":
        from hairport.stages.baldify import BaldifyStage
        return BaldifyStage
    if name == "caption":
        from hairport.stages.caption import CaptionStage
        return CaptionStage
    if name == "shape_mesh":
        from hairport.stages.shape_mesh import ShapeMeshStage
        return ShapeMeshStage
    if name == "landmark_3d":
        from hairport.stages.landmark_3d import Landmark3DStage
        return Landmark3DStage
    if name == "align_view":
        from hairport.stages.align_view import AlignViewStage
        return AlignViewStage
    if name == "render_view":
        from hairport.stages.render_view import RenderViewStage
        return RenderViewStage
    if name == "enhance_view":
        from hairport.stages.enhance_view import EnhanceViewStage
        return EnhanceViewStage
    if name == "blend_hair":
        from hairport.stages.blend_hair import BlendHairStage
        return BlendHairStage
    if name == "transfer_hair":
        from hairport.stages.transfer_hair import TransferHairStage
        return TransferHairStage
    raise ValueError(f"Unknown stage: {name!r}. Valid: {STAGE_ORDER}")


# ============================================================================
# Pipeline Context & Result
# ============================================================================

@dataclass
class PipelineContext:
    """Shared context passed through pipeline stages.

    Attributes
    ----------
    data_dir : Path
        Root data directory (e.g., ``outputs/``).
    shape_provider : str
        Shape provider name (``"hi3dgen"`` or ``"hunyuan"``).
    texture_provider : str
        Texture provider name (``"mvadapter"`` or ``"hunyuan"``).
    bald_version : str
        Bald version mode (``"w_seg"``, ``"wo_seg"``, ``"all"``).
    device : str
        Compute device.
    seed : int
        Random seed (-1 for timestamp-based).
    extra : dict
        Arbitrary per-stage overrides.
    """

    data_dir: Path | None = None
    shape_provider: str | None = None
    texture_provider: str | None = None
    bald_version: str | None = None
    device: str | None = None
    seed: int | None = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        cfg = get_config()
        if self.data_dir is None:
            self.data_dir = Path("outputs")
        if self.shape_provider is None:
            self.shape_provider = cfg.pipeline.shape_provider
        if self.texture_provider is None:
            self.texture_provider = cfg.pipeline.texture_provider
        if self.bald_version is None:
            self.bald_version = cfg.pipeline.bald_version
        if self.device is None:
            self.device = cfg.device
        if self.seed is None:
            self.seed = cfg.seed

    def stage_kwargs(self, stage_name: str) -> dict:
        """Build keyword arguments for a specific stage's ``run()``."""
        base: dict = {}

        # baldify doesn't need data_dir — it uses input_dir/output_dir
        # but supports data_dir for pipeline mode
        if stage_name == "baldify":
            base["data_dir"] = self.data_dir
            base["bald_version"] = self.bald_version
            return {**base, **self.extra.get(stage_name, {})}

        # All other stages use data_dir
        base["data_dir"] = self.data_dir

        # caption needs bald_version mapped to bald_subdir
        if stage_name == "caption":
            base["bald_subdir"] = f"bald/{self.bald_version}"
            base.update(self.extra.get(stage_name, {}))
            return base

        # Stages that need shape/texture providers
        if stage_name in ("shape_mesh", "landmark_3d", "align_view",
                          "render_view", "enhance_view", "blend_hair",
                          "transfer_hair"):
            base["shape_provider"] = self.shape_provider
            base["texture_provider"] = self.texture_provider

        # Stages that need bald_version
        if stage_name in ("align_view", "render_view", "enhance_view",
                          "blend_hair", "transfer_hair"):
            base["bald_version"] = self.bald_version

        # Merge stage-specific overrides from extra
        base.update(self.extra.get(stage_name, {}))
        return base


@dataclass
class StageResult:
    """Result from a single pipeline stage execution."""

    stage: str
    success: bool
    duration_seconds: float
    result: Any = None
    error: Optional[str] = None


# ============================================================================
# Pipeline
# ============================================================================

class HairPortPipeline:
    """Orchestrate the full hair transfer pipeline.

    Parameters
    ----------
    data_dir : str | Path
        Root data directory.
    shape_provider : str
        Shape provider name.
    texture_provider : str
        Texture provider name.
    bald_version : str
        Bald version mode.
    device : str
        Compute device.
    seed : int
        Random seed.
    extra : dict | None
        Per-stage overrides. Keys are stage names, values are dicts of kwargs.
    """

    def __init__(
        self,
        data_dir: str | Path | PipelineContext = None,
        shape_provider: str | None = None,
        texture_provider: str | None = None,
        bald_version: str | None = None,
        device: str | None = None,
        seed: int | None = None,
        extra: Dict[str, Dict[str, Any]] | None = None,
    ):
        if isinstance(data_dir, PipelineContext):
            self.ctx = data_dir
        else:
            self.ctx = PipelineContext(
                data_dir=Path(data_dir) if data_dir is not None else None,
                shape_provider=shape_provider,
                texture_provider=texture_provider,
                bald_version=bald_version,
                device=device,
                seed=seed,
                extra=extra or {},
            )
        self._results: list[StageResult] = []
        self._hooks: list[Callable[[StageResult], None]] = []

    def add_hook(self, fn: Callable[[StageResult], None]) -> None:
        """Register a callback invoked after each stage completes."""
        self._hooks.append(fn)

    @property
    def results(self) -> list[StageResult]:
        """Results from the last ``run()``."""
        return list(self._results)

    def run(
        self,
        start: str | None = None,
        end: str | None = None,
        only: Sequence[str] | None = None,
        skip: Sequence[str] | None = None,
        stop_on_error: bool = True,
    ) -> list[StageResult]:
        """Execute pipeline stages.

        Parameters
        ----------
        start : str | None
            First stage to execute (inclusive). Skips all prior stages.
        end : str | None
            Last stage to execute (inclusive). Stops after this stage.
        only : sequence of str | None
            If provided, run *only* these stages (in pipeline order).
        skip : sequence of str | None
            Stage names to skip.
        stop_on_error : bool
            If True, halt on first stage failure.

        Returns
        -------
        list[StageResult]
            Results for each executed stage.
        """
        stages = self._resolve_stages(start=start, end=end, only=only, skip=skip)
        self._results = []

        logger.info(f"Pipeline: running stages {[s for s in stages]}")
        pipeline_start = time.time()

        for stage_name in stages:
            logger.info(f"--- Stage: {stage_name} ---")
            t0 = time.time()
            stage_instance = None
            try:
                cls = _get_stage_class(stage_name)
                stage_instance = cls()
                kwargs = self.ctx.stage_kwargs(stage_name)
                result = stage_instance.run(**kwargs)
                sr = StageResult(
                    stage=stage_name, success=True,
                    duration_seconds=time.time() - t0, result=result,
                )
            except Exception as e:
                logger.error(f"Stage {stage_name} failed: {e}", exc_info=True)
                sr = StageResult(
                    stage=stage_name, success=False,
                    duration_seconds=time.time() - t0, error=str(e),
                )

            self._results.append(sr)
            for hook in self._hooks:
                try:
                    hook(sr)
                except Exception:
                    logger.warning(f"Hook error after {stage_name}", exc_info=True)

            if not sr.success and stop_on_error:
                logger.error(f"Pipeline stopped at {stage_name} (stop_on_error=True)")
                break

            # Try to release GPU after each stage
            if stage_instance is not None and hasattr(stage_instance, "unload"):
                stage_instance.unload()

        total = time.time() - pipeline_start
        logger.info(f"Pipeline finished in {total:.1f}s — "
                     f"{sum(1 for r in self._results if r.success)}/{len(self._results)} succeeded")
        return self._results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_stages(
        start: str | None = None,
        end: str | None = None,
        only: Sequence[str] | None = None,
        skip: Sequence[str] | None = None,
    ) -> list[str]:
        """Resolve which stages to run and in what order."""
        if only:
            # Maintain pipeline ordering
            stages = [s for s in STAGE_ORDER if s in only]
            if not stages:
                raise ValueError(f"None of {only} are valid stages. Valid: {STAGE_ORDER}")
            return stages

        stages = list(STAGE_ORDER)

        if start:
            if start not in STAGE_ORDER:
                raise ValueError(f"Unknown start stage: {start!r}")
            idx = STAGE_ORDER.index(start)
            stages = stages[idx:]

        if end:
            if end not in STAGE_ORDER:
                raise ValueError(f"Unknown end stage: {end!r}")
            idx = stages.index(end)
            stages = stages[: idx + 1]

        if skip:
            stages = [s for s in stages if s not in skip]

        return stages


# ============================================================================
# CLI
# ============================================================================

def main(argv: list[str] | None = None):
    """Run the full HairPort pipeline from the command line."""
    import argparse
    from hairport.config import add_config_args, load_config_from_args

    parser = argparse.ArgumentParser(
        prog="hairport",
        description="Run the HairPort hair transfer pipeline.",
    )
    parser.add_argument("--data_dir", type=str, default="outputs")
    parser.add_argument("--shape_provider", type=str, default=None,
                        choices=["hunyuan", "hi3dgen", "direct3d_s2"])
    parser.add_argument("--texture_provider", type=str, default=None,
                        choices=["hunyuan", "mvadapter"])
    parser.add_argument("--bald_version", type=str, default=None,
                        choices=["wo_seg", "w_seg", "all"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start", type=str, default=None,
                        help="First stage to execute")
    parser.add_argument("--end", type=str, default=None,
                        help="Last stage to execute")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        help="Run only these stages")
    parser.add_argument("--skip", type=str, nargs="+", default=None,
                        help="Skip these stages")
    parser.add_argument("--no-stop-on-error", action="store_true",
                        help="Continue on stage failure")
    add_config_args(parser)
    args = parser.parse_args(argv)
    load_config_from_args(args)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    pipeline = HairPortPipeline(
        data_dir=args.data_dir,
        shape_provider=args.shape_provider,
        texture_provider=args.texture_provider,
        bald_version=args.bald_version,
        device=args.device,
        seed=args.seed,
    )
    results = pipeline.run(
        start=args.start,
        end=args.end,
        only=args.only,
        skip=args.skip,
        stop_on_error=not args.no_stop_on_error,
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("Pipeline Summary")
    print(f"{'=' * 60}")
    for r in results:
        status = "OK" if r.success else "FAIL"
        print(f"  [{status}] {r.stage:20s}  {r.duration_seconds:7.1f}s"
              + (f"  error={r.error}" if r.error else ""))
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
