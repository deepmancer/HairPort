"""HairPort — 3D-aware hair import & transfer framework.

Public API surface:

- **Configuration**: ``HairPortConfig``, ``get_config``, ``set_config`` from ``hairport.config``
- **Core utilities**: ``BackgroundRemover``, ``SAMMaskExtractor``, etc. from ``hairport.core``
- **Pipeline**: ``HairPortPipeline`` from ``hairport.pipeline``
- **Data management**: ``DatasetManager`` from ``hairport.data``
- **Stages**: ``hairport.stages.*`` for individual pipeline stages
"""

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy re-exports to avoid heavy import-time dependencies."""
    # Config
    if name == "HairPortConfig":
        from hairport.config import HairPortConfig
        return HairPortConfig
    if name == "get_config":
        from hairport.config import get_config
        return get_config
    if name == "set_config":
        from hairport.config import set_config
        return set_config
    if name == "load_config":
        from hairport.config import load_config
        return load_config
    # Pipeline
    if name == "HairPortPipeline":
        from hairport.pipeline import HairPortPipeline
        return HairPortPipeline
    # Data
    if name == "DatasetManager":
        from hairport.data import DatasetManager
        return DatasetManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "DatasetManager",
    "HairPortConfig",
    "HairPortPipeline",
    "get_config",
    "set_config",
]
