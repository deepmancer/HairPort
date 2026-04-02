"""BaldKonverter — Generate bald versions of portrait images using FLUX LoRA models.

Quick start::

    from hairport.bald_konverter import BaldKonverterPipeline

    pipeline = BaldKonverterPipeline(mode="auto")
    result = pipeline("portrait.jpg")
    result.bald_image.save("bald.png")

For lower-level control, use the model classes directly::

    from hairport.bald_konverter import BaldKonverter, BaldKonverterWithSeg
"""

from .models.konverter import BaldKonverter, BaldKonverterWithSeg
from .pipeline import BaldKonverterPipeline, BaldResult

__version__ = "0.1.0"

__all__ = [
    "BaldKonverter",
    "BaldKonverterPipeline",
    "BaldKonverterWithSeg",
    "BaldResult",
    "__version__",
]
