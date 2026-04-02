"""
Segmentation label configuration for Segformer and ExtendedLIP label spaces.

Provides enums, color palettes, and mapping utilities consumed by the
face parser, hair mask pipeline, and 4-panel grid renderer.

Note: The original LIPClass (20-class CDGNet labels) has been removed because
CDGNet is no longer used—SAM3 replaced it for hair mask extraction.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Segformer face-parsing classes (19 classes from jonathandinu/face-parsing)
# --------------------------------------------------------------------------- #

class SegformerClass(IntEnum):
    """Segformer face parsing classes (19 classes)."""
    BACKGROUND = 0
    SKIN = 1
    NOSE = 2
    EYE_G = 3        # eyeglasses
    L_EYE = 4
    R_EYE = 5
    L_BROW = 6
    R_BROW = 7
    L_EAR = 8
    R_EAR = 9
    MOUTH = 10
    U_LIP = 11
    L_LIP = 12
    HAIR = 13
    HAT = 14
    EAR_R = 15       # earring
    NECK_L = 16      # necklace / left neck region
    NECK = 17
    CLOTH = 18


# --------------------------------------------------------------------------- #
# Extended LIP label space (32 classes) — the *unified* output format
# --------------------------------------------------------------------------- #

class ExtendedLIPClass(IntEnum):
    """Extended LIP Dataset with detailed facial features (32 classes)."""
    BACKGROUND = 0
    HAT = 1
    HAIR = 2
    GLOVE = 3
    SUNGLASSES = 4
    UPPER_CLOTHES = 5
    DRESS = 6
    COAT = 7
    SOCKS = 8
    PANTS = 9
    JUMPSUITS = 10
    SCARF = 11
    SKIRT = 12
    FACE = 13
    LEFT_ARM = 14
    RIGHT_ARM = 15
    LEFT_LEG = 16
    RIGHT_LEG = 17
    LEFT_SHOE = 18
    RIGHT_SHOE = 19
    # Extended facial features
    NOSE = 20
    LEFT_EYE = 21
    RIGHT_EYE = 22
    LEFT_BROW = 23
    RIGHT_BROW = 24
    LEFT_EAR = 25
    RIGHT_EAR = 26
    MOUTH = 27
    UPPER_LIP = 28
    LOWER_LIP = 29
    NECK = 30
    BODY = 31


# --------------------------------------------------------------------------- #
# Class config dataclass & base config
# --------------------------------------------------------------------------- #

@dataclass
class ClassConfig:
    """Configuration for a single segmentation class."""
    class_id: int
    name: str
    color: Tuple[int, int, int]
    description: Optional[str] = None


class SegmentationConfig:
    """Base configuration class for segmentation setups."""

    def __init__(self, classes: Dict[int, ClassConfig]):
        self.classes = classes
        self._id_to_color = {c.class_id: c.color for c in classes.values()}
        self._color_to_id = {c.color: c.class_id for c in classes.values()}
        self._name_to_id = {c.name: c.class_id for c in classes.values()}
        self._id_to_name = {c.class_id: c.name for c in classes.values()}

    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        return self._id_to_color.get(class_id, (128, 128, 128))

    def get_class_id(self, color: Tuple[int, int, int]) -> Optional[int]:
        return self._color_to_id.get(color)

    def get_class_name(self, class_id: int) -> str:
        return self._id_to_name.get(class_id, f"unknown_{class_id}")

    def get_class_id_by_name(self, name: str) -> Optional[int]:
        return self._name_to_id.get(name)

    def get_color_palette(self) -> List[Tuple[int, int, int]]:
        max_id = max(self.classes.keys()) if self.classes else 0
        return [self.get_color(i) for i in range(max_id + 1)]

    def get_class_names(self) -> Dict[int, str]:
        return self._id_to_name.copy()


# --------------------------------------------------------------------------- #
# ExtendedLIP concrete config with color palette
# --------------------------------------------------------------------------- #

class ExtendedLIPSegmentationConfig(SegmentationConfig):
    """Extended LIP dataset configuration with detailed facial features."""

    def __init__(self):
        classes = {
            ExtendedLIPClass.BACKGROUND:   ClassConfig(0,  "background",   (0, 0, 0)),
            ExtendedLIPClass.HAT:          ClassConfig(1,  "hat",          (255, 0, 128)),
            ExtendedLIPClass.HAIR:         ClassConfig(2,  "hair",         (205, 169, 23)),
            ExtendedLIPClass.GLOVE:        ClassConfig(3,  "glove",        (0, 255, 0)),
            ExtendedLIPClass.SUNGLASSES:   ClassConfig(4,  "sunglasses",   (75, 0, 130)),
            ExtendedLIPClass.UPPER_CLOTHES:ClassConfig(5,  "upper_clothes",(255, 215, 0)),
            ExtendedLIPClass.DRESS:        ClassConfig(6,  "dress",        (138, 43, 226)),
            ExtendedLIPClass.COAT:         ClassConfig(7,  "coat",         (0, 191, 255)),
            ExtendedLIPClass.SOCKS:        ClassConfig(8,  "socks",        (255, 20, 147)),
            ExtendedLIPClass.PANTS:        ClassConfig(9,  "pants",        (0, 100, 0)),
            ExtendedLIPClass.JUMPSUITS:    ClassConfig(10, "jumpsuits",    (255, 140, 0)),
            ExtendedLIPClass.SCARF:        ClassConfig(11, "scarf",        (70, 130, 180)),
            ExtendedLIPClass.SKIRT:        ClassConfig(12, "skirt",        (255, 105, 180)),
            ExtendedLIPClass.FACE:         ClassConfig(13, "face",         (255, 192, 203)),
            ExtendedLIPClass.LEFT_ARM:     ClassConfig(14, "left_arm",     (0, 206, 209)),
            ExtendedLIPClass.RIGHT_ARM:    ClassConfig(15, "right_arm",    (72, 209, 204)),
            ExtendedLIPClass.LEFT_LEG:     ClassConfig(16, "left_leg",     (50, 205, 50)),
            ExtendedLIPClass.RIGHT_LEG:    ClassConfig(17, "right_leg",    (154, 205, 50)),
            ExtendedLIPClass.LEFT_SHOE:    ClassConfig(18, "left_shoe",    (255, 0, 255)),
            ExtendedLIPClass.RIGHT_SHOE:   ClassConfig(19, "right_shoe",   (128, 0, 128)),
            ExtendedLIPClass.NOSE:         ClassConfig(20, "nose",         (255, 182, 193)),
            ExtendedLIPClass.LEFT_EYE:     ClassConfig(21, "left_eye",     (30, 144, 255)),
            ExtendedLIPClass.RIGHT_EYE:    ClassConfig(22, "right_eye",    (0, 100, 255)),
            ExtendedLIPClass.LEFT_BROW:    ClassConfig(23, "left_brow",    (139, 69, 19)),
            ExtendedLIPClass.RIGHT_BROW:   ClassConfig(24, "right_brow",   (160, 82, 45)),
            ExtendedLIPClass.LEFT_EAR:     ClassConfig(25, "left_ear",     (255, 160, 122)),
            ExtendedLIPClass.RIGHT_EAR:    ClassConfig(26, "right_ear",    (255, 127, 80)),
            ExtendedLIPClass.MOUTH:        ClassConfig(27, "mouth",        (220, 20, 60)),
            ExtendedLIPClass.UPPER_LIP:    ClassConfig(28, "upper_lip",    (255, 99, 71)),
            ExtendedLIPClass.LOWER_LIP:    ClassConfig(29, "lower_lip",    (255, 39, 0)),
            ExtendedLIPClass.NECK:         ClassConfig(30, "neck",         (245, 222, 179)),
            ExtendedLIPClass.BODY:         ClassConfig(31, "body",         (205, 133, 63)),
        }
        super().__init__(classes)


# --------------------------------------------------------------------------- #
# Segformer → ExtendedLIP mapping
# --------------------------------------------------------------------------- #

class ModelMappingConfig:
    """Maps Segformer 19-class output to the ExtendedLIP 32-class label space."""

    def __init__(self):
        self.segformer_to_extended_lip: Dict[SegformerClass, ExtendedLIPClass] = {
            SegformerClass.BACKGROUND: ExtendedLIPClass.BACKGROUND,
            SegformerClass.SKIN:       ExtendedLIPClass.FACE,
            SegformerClass.NOSE:       ExtendedLIPClass.NOSE,
            SegformerClass.EYE_G:      ExtendedLIPClass.SUNGLASSES,
            SegformerClass.L_EYE:      ExtendedLIPClass.LEFT_EYE,
            SegformerClass.R_EYE:      ExtendedLIPClass.RIGHT_EYE,
            SegformerClass.L_BROW:     ExtendedLIPClass.LEFT_BROW,
            SegformerClass.R_BROW:     ExtendedLIPClass.RIGHT_BROW,
            SegformerClass.L_EAR:      ExtendedLIPClass.LEFT_EAR,
            SegformerClass.R_EAR:      ExtendedLIPClass.RIGHT_EAR,
            SegformerClass.MOUTH:      ExtendedLIPClass.MOUTH,
            SegformerClass.U_LIP:      ExtendedLIPClass.UPPER_LIP,
            SegformerClass.L_LIP:      ExtendedLIPClass.LOWER_LIP,
            SegformerClass.HAIR:       ExtendedLIPClass.HAIR,
            SegformerClass.HAT:        ExtendedLIPClass.HAT,
            SegformerClass.EAR_R:      ExtendedLIPClass.RIGHT_EAR,
            SegformerClass.NECK_L:     ExtendedLIPClass.NECK,
            SegformerClass.NECK:       ExtendedLIPClass.NECK,
            SegformerClass.CLOTH:      ExtendedLIPClass.UPPER_CLOTHES,
        }

    def as_int_dict(self) -> Dict[int, int]:
        """Return {segformer_id: extended_lip_id} with plain ints."""
        return {int(k): int(v) for k, v in self.segformer_to_extended_lip.items()}


# --------------------------------------------------------------------------- #
# Convenience class groupings
# --------------------------------------------------------------------------- #

class ClassGroups:
    """Predefined class groupings for common mask operations."""

    HAIR = [ExtendedLIPClass.HAIR]

    FACIAL_FEATURES = [
        ExtendedLIPClass.NOSE,
        ExtendedLIPClass.LEFT_EYE,
        ExtendedLIPClass.RIGHT_EYE,
        ExtendedLIPClass.LEFT_BROW,
        ExtendedLIPClass.RIGHT_BROW,
        ExtendedLIPClass.MOUTH,
        ExtendedLIPClass.UPPER_LIP,
        ExtendedLIPClass.LOWER_LIP,
    ]

    FACE_ALL = (
        [ExtendedLIPClass.FACE]
        + FACIAL_FEATURES
        + [ExtendedLIPClass.LEFT_EAR, ExtendedLIPClass.RIGHT_EAR, ExtendedLIPClass.NECK]
    )

    BODY_LIMBS = [
        ExtendedLIPClass.LEFT_ARM,
        ExtendedLIPClass.RIGHT_ARM,
        ExtendedLIPClass.LEFT_LEG,
        ExtendedLIPClass.RIGHT_LEG,
    ]

    BODY_GENERIC = [ExtendedLIPClass.BODY]

    CLOTHING = [
        ExtendedLIPClass.UPPER_CLOTHES,
        ExtendedLIPClass.DRESS,
        ExtendedLIPClass.COAT,
        ExtendedLIPClass.PANTS,
        ExtendedLIPClass.JUMPSUITS,
        ExtendedLIPClass.SCARF,
        ExtendedLIPClass.SKIRT,
        ExtendedLIPClass.SOCKS,
    ]

    ACCESSORIES = [
        ExtendedLIPClass.HAT,
        ExtendedLIPClass.SUNGLASSES,
        ExtendedLIPClass.GLOVE,
        ExtendedLIPClass.LEFT_SHOE,
        ExtendedLIPClass.RIGHT_SHOE,
    ]

    HUMAN_ALL = HAIR + FACE_ALL + BODY_LIMBS + BODY_GENERIC + CLOTHING + ACCESSORIES
    BODY_AND_HAIR = HAIR + FACE_ALL + BODY_LIMBS + BODY_GENERIC


# --------------------------------------------------------------------------- #
# Module-level singletons
# --------------------------------------------------------------------------- #

EXTENDED_LIP_CONFIG = ExtendedLIPSegmentationConfig()
MODEL_MAPPING = ModelMappingConfig()
