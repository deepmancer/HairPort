"""hairport.core — Canonical shared utilities for the HairPort framework.

This package consolidates deduplicated implementations of common utilities
used across ``hairport``, ``fit_lmk``, and ``utils`` modules.
"""

from hairport.core.bg_remover import BackgroundRemover
from hairport.core.sam_extractor import SAMMaskExtractor


# Lazy imports for heavier modules (avoid import-time overhead)
def __getattr__(name: str):
    if name == "FLAMEFitter":
        from hairport.core.flame_fitting import FLAMEFitter
        return FLAMEFitter
    if name == "CaptionerPipeline":
        from hairport.core.captioner import CaptionerPipeline
        return CaptionerPipeline
    if name == "FacialLandmarkDetector":
        from hairport.core.facial_landmark_detector import FacialLandmarkDetector
        return FacialLandmarkDetector
    if name == "CodeFormerEnhancer":
        from hairport.core.super_resolution import CodeFormerEnhancer
        return CodeFormerEnhancer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BackgroundRemover",
    "CaptionerPipeline",
    "CodeFormerEnhancer",
    "FacialLandmarkDetector",
    "FLAMEFitter",
    "SAMMaskExtractor",
]
