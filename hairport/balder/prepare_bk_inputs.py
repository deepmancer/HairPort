import argparse
import gc
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import (
    AutoModelForImageSegmentation,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from ben2 import BEN_Base
from utils.bg_remover import BackgroundRemover
# Import SAMMaskExtractor for hair mask extraction
import sys
from utils.sam_mask_extractor import SAMMaskExtractor

torch.set_float32_matmul_precision(['high', 'highest'][0])

"""
Segmentation Configuration Module

This module provides clean, flexible configuration classes for managing
segmentation classes, colors, and mappings used across different models.
"""

from enum import Enum, IntEnum
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image
import cv2


class LIPClass(IntEnum):
    """LIP Dataset standard classes (20 classes)"""
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


class ExtendedLIPClass(IntEnum):
    """Extended LIP Dataset with detailed facial features (31 classes)"""
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


class SegformerClass(IntEnum):
    """Segformer face parsing classes (19 classes)"""
    BACKGROUND = 0
    SKIN = 1
    NOSE = 2
    EYE_G = 3
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
    EAR_R = 15
    NECK_L = 16
    NECK = 17
    CLOTH = 18


@dataclass
class ClassConfig:
    """Configuration for a single segmentation class"""
    class_id: int
    name: str
    color: Tuple[int, int, int]
    description: Optional[str] = None


class SegmentationConfig:
    """Base configuration class for segmentation setups"""
    
    def __init__(self, classes: Dict[int, ClassConfig]):
        self.classes = classes
        self._id_to_color = {cls.class_id: cls.color for cls in classes.values()}
        self._color_to_id = {cls.color: cls.class_id for cls in classes.values()}
        self._name_to_id = {cls.name: cls.class_id for cls in classes.values()}
        self._id_to_name = {cls.class_id: cls.name for cls in classes.values()}
    
    def get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for a given class ID"""
        return self._id_to_color.get(class_id, (128, 128, 128))  # Default gray
    
    def get_class_id(self, color: Tuple[int, int, int]) -> Optional[int]:
        """Get class ID for a given color"""
        return self._color_to_id.get(color)
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name for a given class ID"""
        return self._id_to_name.get(class_id, f"unknown_{class_id}")
    
    def get_class_id_by_name(self, name: str) -> Optional[int]:
        """Get class ID for a given class name"""
        return self._name_to_id.get(name)
    
    def get_color_palette(self) -> List[Tuple[int, int, int]]:
        """Get ordered color palette"""
        max_id = max(self.classes.keys()) if self.classes else 0
        palette = []
        for i in range(max_id + 1):
            palette.append(self.get_color(i))
        return palette
    
    def get_class_names(self) -> Dict[int, str]:
        """Get mapping of class IDs to names"""
        return self._id_to_name.copy()


class LIPSegmentationConfig(SegmentationConfig):
    """Standard LIP dataset configuration"""
    
    def __init__(self):
        classes = {
            LIPClass.BACKGROUND: ClassConfig(0, 'background', (0, 0, 0)),
            LIPClass.HAT: ClassConfig(1, 'hat', (255, 0, 128)),
            LIPClass.HAIR: ClassConfig(2, 'hair', (205, 169, 23)),
            LIPClass.GLOVE: ClassConfig(3, 'glove', (0, 255, 0)),
            LIPClass.SUNGLASSES: ClassConfig(4, 'sunglasses', (75, 0, 130)),
            LIPClass.UPPER_CLOTHES: ClassConfig(5, 'upper_clothes', (255, 215, 0)),
            LIPClass.DRESS: ClassConfig(6, 'dress', (138, 43, 226)),
            LIPClass.COAT: ClassConfig(7, 'coat', (0, 191, 255)),
            LIPClass.SOCKS: ClassConfig(8, 'socks', (255, 20, 147)),
            LIPClass.PANTS: ClassConfig(9, 'pants', (0, 100, 0)),
            LIPClass.JUMPSUITS: ClassConfig(10, 'jumpsuits', (255, 140, 0)),
            LIPClass.SCARF: ClassConfig(11, 'scarf', (70, 130, 180)),
            LIPClass.SKIRT: ClassConfig(12, 'skirt', (255, 105, 180)),
            LIPClass.FACE: ClassConfig(13, 'face', (255, 192, 203)),
            LIPClass.LEFT_ARM: ClassConfig(14, 'left_arm', (0, 206, 209)),
            LIPClass.RIGHT_ARM: ClassConfig(15, 'right_arm', (72, 209, 204)),
            LIPClass.LEFT_LEG: ClassConfig(16, 'left_leg', (50, 205, 50)),
            LIPClass.RIGHT_LEG: ClassConfig(17, 'right_leg', (154, 205, 50)),
            LIPClass.LEFT_SHOE: ClassConfig(18, 'left_shoe', (255, 0, 255)),
            LIPClass.RIGHT_SHOE: ClassConfig(19, 'right_shoe', (128, 0, 128)),
        }
        super().__init__(classes)


class ExtendedLIPSegmentationConfig(SegmentationConfig):
    """Extended LIP dataset configuration with detailed facial features"""
    
    def __init__(self):
        # Start with basic LIP classes
        classes = {
            ExtendedLIPClass.BACKGROUND: ClassConfig(0, 'background', (0, 0, 0)),
            ExtendedLIPClass.HAT: ClassConfig(1, 'hat', (255, 0, 128)),
            ExtendedLIPClass.HAIR: ClassConfig(2, 'hair', (205, 169, 23)),
            ExtendedLIPClass.GLOVE: ClassConfig(3, 'glove', (0, 255, 0)),
            ExtendedLIPClass.SUNGLASSES: ClassConfig(4, 'sunglasses', (75, 0, 130)),
            ExtendedLIPClass.UPPER_CLOTHES: ClassConfig(5, 'upper_clothes', (255, 215, 0)),
            ExtendedLIPClass.DRESS: ClassConfig(6, 'dress', (138, 43, 226)),
            ExtendedLIPClass.COAT: ClassConfig(7, 'coat', (0, 191, 255)),
            ExtendedLIPClass.SOCKS: ClassConfig(8, 'socks', (255, 20, 147)),
            ExtendedLIPClass.PANTS: ClassConfig(9, 'pants', (0, 100, 0)),
            ExtendedLIPClass.JUMPSUITS: ClassConfig(10, 'jumpsuits', (255, 140, 0)),
            ExtendedLIPClass.SCARF: ClassConfig(11, 'scarf', (70, 130, 180)),
            ExtendedLIPClass.SKIRT: ClassConfig(12, 'skirt', (255, 105, 180)),
            ExtendedLIPClass.FACE: ClassConfig(13, 'face', (255, 192, 203)),
            ExtendedLIPClass.LEFT_ARM: ClassConfig(14, 'left_arm', (0, 206, 209)),
            ExtendedLIPClass.RIGHT_ARM: ClassConfig(15, 'right_arm', (72, 209, 204)),
            ExtendedLIPClass.LEFT_LEG: ClassConfig(16, 'left_leg', (50, 205, 50)),
            ExtendedLIPClass.RIGHT_LEG: ClassConfig(17, 'right_leg', (154, 205, 50)),
            ExtendedLIPClass.LEFT_SHOE: ClassConfig(18, 'left_shoe', (255, 0, 255)),
            ExtendedLIPClass.RIGHT_SHOE: ClassConfig(19, 'right_shoe', (128, 0, 128)),
            # Extended facial features
            ExtendedLIPClass.NOSE: ClassConfig(20, 'nose', (255, 182, 193)),
            ExtendedLIPClass.LEFT_EYE: ClassConfig(21, 'left_eye', (30, 144, 255)),
            ExtendedLIPClass.RIGHT_EYE: ClassConfig(22, 'right_eye', (0, 100, 255)),
            ExtendedLIPClass.LEFT_BROW: ClassConfig(23, 'left_brow', (139, 69, 19)),
            ExtendedLIPClass.RIGHT_BROW: ClassConfig(24, 'right_brow', (160, 82, 45)),
            ExtendedLIPClass.LEFT_EAR: ClassConfig(25, 'left_ear', (255, 160, 122)),
            ExtendedLIPClass.RIGHT_EAR: ClassConfig(26, 'right_ear', (255, 127, 80)),
            ExtendedLIPClass.MOUTH: ClassConfig(27, 'mouth', (220, 20, 60)),
            ExtendedLIPClass.UPPER_LIP: ClassConfig(28, 'upper_lip', (255, 99, 71)),
            ExtendedLIPClass.LOWER_LIP: ClassConfig(29, 'lower_lip', (255, 39, 0)),
            ExtendedLIPClass.NECK: ClassConfig(30, 'neck', (245, 222, 179)),
            ExtendedLIPClass.BODY: ClassConfig(31, 'body', (205, 133, 63)),
        }
        super().__init__(classes)


class ModelMappingConfig:
    """Configuration for mapping between different model outputs"""
    
    def __init__(self):
        # Mapping from Segformer classes to Extended LIP classes
        self.segformer_to_extended_lip = {
            SegformerClass.BACKGROUND: ExtendedLIPClass.BACKGROUND,
            SegformerClass.SKIN: ExtendedLIPClass.FACE,
            SegformerClass.NOSE: ExtendedLIPClass.NOSE,
            SegformerClass.EYE_G: ExtendedLIPClass.SUNGLASSES,
            SegformerClass.L_EYE: ExtendedLIPClass.LEFT_EYE,
            SegformerClass.R_EYE: ExtendedLIPClass.RIGHT_EYE,
            SegformerClass.L_BROW: ExtendedLIPClass.LEFT_BROW,
            SegformerClass.R_BROW: ExtendedLIPClass.RIGHT_BROW,
            SegformerClass.L_EAR: ExtendedLIPClass.LEFT_EAR,
            SegformerClass.R_EAR: ExtendedLIPClass.RIGHT_EAR,
            SegformerClass.MOUTH: ExtendedLIPClass.MOUTH,
            SegformerClass.U_LIP: ExtendedLIPClass.UPPER_LIP,
            SegformerClass.L_LIP: ExtendedLIPClass.LOWER_LIP,
            SegformerClass.HAIR: ExtendedLIPClass.HAIR,
            SegformerClass.HAT: ExtendedLIPClass.HAT,
            SegformerClass.EAR_R: ExtendedLIPClass.RIGHT_EAR,
            SegformerClass.NECK_L: ExtendedLIPClass.NECK,
            SegformerClass.NECK: ExtendedLIPClass.NECK,
            SegformerClass.CLOTH: ExtendedLIPClass.UPPER_CLOTHES,
        }
    
    def get_segformer_to_extended_lip_mapping(self) -> Dict[int, int]:
        """Get mapping dictionary from Segformer to Extended LIP class IDs"""
        return {int(seg_cls): int(lip_cls) for seg_cls, lip_cls in self.segformer_to_extended_lip.items()}


# Predefined configurations
LIP_CONFIG = LIPSegmentationConfig()
EXTENDED_LIP_CONFIG = ExtendedLIPSegmentationConfig()
MODEL_MAPPING = ModelMappingConfig()


# Common class groupings for convenience
class ClassGroups:
    """Predefined class groupings for common use cases"""
    
    # Hair-related classes
    HAIR = [ExtendedLIPClass.HAIR]
    
    # Facial features
    FACIAL_FEATURES = [
        ExtendedLIPClass.NOSE, ExtendedLIPClass.LEFT_EYE, ExtendedLIPClass.RIGHT_EYE,
        ExtendedLIPClass.LEFT_BROW, ExtendedLIPClass.RIGHT_BROW, ExtendedLIPClass.MOUTH,
        ExtendedLIPClass.UPPER_LIP, ExtendedLIPClass.LOWER_LIP
    ]
    
    # Face and facial features combined
    FACE_ALL = [ExtendedLIPClass.FACE] + FACIAL_FEATURES + [
        ExtendedLIPClass.LEFT_EAR, ExtendedLIPClass.RIGHT_EAR, ExtendedLIPClass.NECK
    ]
    
    # Body parts (arms and legs)
    BODY_LIMBS = [
        ExtendedLIPClass.LEFT_ARM, ExtendedLIPClass.RIGHT_ARM,
        ExtendedLIPClass.LEFT_LEG, ExtendedLIPClass.RIGHT_LEG
    ]
    
    # Generic body class (fallback for unclassified body regions)
    BODY_GENERIC = [ExtendedLIPClass.BODY]
    
    # Clothing
    CLOTHING = [
        ExtendedLIPClass.UPPER_CLOTHES, ExtendedLIPClass.DRESS, ExtendedLIPClass.COAT,
        ExtendedLIPClass.PANTS, ExtendedLIPClass.JUMPSUITS, ExtendedLIPClass.SCARF,
        ExtendedLIPClass.SKIRT, ExtendedLIPClass.SOCKS
    ]
    
    # Accessories
    ACCESSORIES = [
        ExtendedLIPClass.HAT, ExtendedLIPClass.SUNGLASSES, ExtendedLIPClass.GLOVE,
        ExtendedLIPClass.LEFT_SHOE, ExtendedLIPClass.RIGHT_SHOE
    ]
    
    # Human (everything except background)
    HUMAN_ALL = (HAIR + FACE_ALL + BODY_LIMBS + BODY_GENERIC + CLOTHING + ACCESSORIES)
    
    # Body and hair (for traditional body mask)
    BODY_AND_HAIR = HAIR + FACE_ALL + BODY_LIMBS + BODY_GENERIC

class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class BasePreprocessor(metaclass=SingletonMeta):
    def teardown(self):
        pass
    
    def preprocess(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def __del__(self):
        self.teardown()

__all__ = ["BasePreprocessor"]


class HairMaskPipeline(BasePreprocessor):
    DEFAULT_CDGNET_CKPT = Path("assets/checkpoints/CDGNet/LIP_epoch_149.pth")
    CDGNET_TRANS = transforms.Compose([
        # transforms.Lambda(lambda img: Image.fromarray(
        #     cv2.detailEnhance(np.array(img), sigma_s=3, sigma_r=0.1)
        # )),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    def __init__(
        self,
        cdgnet_ckpt: Path | None = None,
        device: str | torch.device = 'cuda',
        sam_confidence_threshold: float = None,
    ):
        self.device = torch.device(device)
        self.cdgnet_ckpt = cdgnet_ckpt or self.DEFAULT_CDGNET_CKPT

        self.birefnet = self._init_birefnet()
        self.cdgnet = self._init_cdgnet()
        self.segformer_processor, self.segformer_model = self._init_segformer()
        
        # Initialize SAM for hair mask extraction
        self.sam_extractor = SAMMaskExtractor(confidence_threshold=0.25)

        self.scales = [0.66, 0.80, 1.0]
        self.flip_mapping = torch.tensor((15, 14, 17, 16, 19, 18), device=self.device)
        self._upsampler: nn.Upsample | None = None

    def _init_birefnet(self) -> AutoModelForImageSegmentation:
        # GPU optimization: Ensure model is in eval mode and use half precision if possible
        model = BEN_Base.from_pretrained("PramaLLC/BEN2").to(self.device)
        model.eval()  # Explicitly set to eval mode for GPU optimization
        return model

    def _init_cdgnet(self) -> nn.Module:
        sys.path.append('/workspace/HairPort/Hairdar/modules/CDGNet')
        sys.path.append('/workspace/HairPort/Hairdar/')
        from networks.CDGNet import Res_Deeplab

        model = Res_Deeplab(num_classes=20)
        state = torch.load(self.cdgnet_ckpt, map_location='cpu')
        target = model.state_dict()

        # align checkpoint keys automatically
        new_state = {}
        for key in target:
            ckpt_key = 'module.' + key if 'module.' + key in state else key
            new_state[key] = deepcopy(state.get(ckpt_key, target[key]))

        model.load_state_dict(new_state)
        model = model.to(self.device)
        model.eval()  # Explicitly set to eval mode for GPU optimization
        return model

    def _init_segformer(self) -> tuple[SegformerImageProcessor, SegformerForSemanticSegmentation]:
        """Initialize Segformer model for detailed face/body parsing."""
        processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        model = model.to(self.device)
        model.eval()
        return processor, model

    def _preprocess_biref(self, image: Image.Image) -> torch.Tensor:
        return transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(image).unsqueeze(0).to(self.device)

    def _matte_image(self, image: Image.Image) -> tuple[Image.Image, Image.Image]:
        foreground = self.birefnet.inference(image, refine_foreground=False)
        alpha = foreground.getchannel('A')
        mask = (np.array(alpha) / 255.0 > 0.5).astype(np.uint8) * 255
        return foreground, Image.fromarray(mask)

    def _ensure_upsampler(self, input_size: tuple[int, int]) -> None:
        if not self._upsampler or self._upsampler.size != input_size:
            self._upsampler = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

    def _parse_batch(
        self,
        model: nn.Module,
        batch: torch.Tensor,
        output_size: tuple[int, int],
    ) -> np.ndarray:
        self._ensure_upsampler(batch.shape[-2:])
        
        with torch.no_grad():  # GPU optimization: disable gradient computation
            multi_scale_preds = []
            for scale in self.scales:
                scaled = F.interpolate(batch, scale_factor=scale, mode='bilinear', align_corners=True)
                out = model(scaled)
                pred = out[0][-1]
                single, flipped = pred[0], pred[1]
                flipped[14:20, :, :] = flipped[self.flip_mapping, :, :]
                
                single += flipped.flip(dims=[-1])
                single *= 0.5
                
                single = self._upsampler(single.unsqueeze(0))                 
                multi_scale_preds.append(single[0])
                
                # GPU optimization: clean up intermediate tensors
                del scaled, out, pred, single, flipped
                    
            fused_prediction = torch.stack(multi_scale_preds)
            fused_prediction = fused_prediction.mean(0)
            fused_prediction = F.interpolate(fused_prediction[None], size=output_size, mode='bicubic')[0]
            fused_prediction = fused_prediction.permute(1, 2, 0)  # HWC
            fused_prediction = torch.argmax(fused_prediction, dim=2)
            
            # GPU optimization: move to CPU and convert to numpy immediately
            parsed = fused_prediction.cpu().numpy().astype(np.uint8)
            
            # GPU optimization: clean up GPU tensors
            del multi_scale_preds, fused_prediction
                
        return parsed

    def _compute_hair_bbox(self, hair_mask: np.ndarray, margin: float = 0.1) -> tuple[int, int, int, int]:
        """Compute bounding box around hair region with margin."""
        coords = np.column_stack(np.where(hair_mask > 0))
        if len(coords) == 0:
            # If no hair detected, return full image bounds
            h, w = hair_mask.shape
            return 0, 0, w, h
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        # Add margin
        h, w = hair_mask.shape
        margin_x = int((x_max - x_min) * margin)
        margin_y = int((y_max - y_min) * margin)
        
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(w, x_max + margin_x)
        y_max = min(h, y_max + margin_y)
        
        return x_min, y_min, x_max, y_max



    def _predict_logits(
        self, model: nn.Module, batch: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        """Multi-scale+flip TTA, return fused logits at output_size (C,H,W)"""
        # NOTE: prefer bilinear, align_corners=False for logits
        scales = self.scales

        # upsampler to the original (unscaled) input resolution for each TTA branch
        in_h, in_w = batch.shape[-2:]
        up_to_input = lambda x: F.interpolate(x, size=(in_h, in_w), mode="bilinear", align_corners=True)

        per_scale = []
        for s in scales:
            with torch.no_grad():  # GPU optimization: disable gradient computation
                scaled = F.interpolate(batch, scale_factor=s, mode='bilinear', align_corners=True)
                out = model(scaled)                  # CDGNet returns list(s); last is fine
                logits = out[0][-1]                 # (B=2, C, h, w) for [orig, flipped]
                single, flipped = logits[0], logits[1]
                # remap left/right on the flipped branch BEFORE horizontal unflip
                flipped[14:20, :, :] = flipped[self.flip_mapping, :, :]
                # unflip horizontally and average the two
                single = 0.5 * (single + flipped.flip(dims=[-1]))
                upsampled = up_to_input(single.unsqueeze(0))[0]  # (C, in_h, in_w)
                per_scale.append(upsampled)
                
                # GPU optimization: clean up intermediate tensors
                del scaled, out, logits, single, flipped, upsampled
                
        with torch.no_grad():  # GPU optimization: disable gradient computation for fusion
            fused = torch.stack(per_scale, dim=0).mean(0)  # (C, in_h, in_w)
            # final resize to output_size in logits-space
            fused = F.interpolate(fused.unsqueeze(0), size=output_size, mode='bicubic')[0]
            
            # GPU optimization: clean up per_scale list
            del per_scale
            
        return fused  # (C, H, W)

    def _process_cdgnet_logits(self, image: Image.Image) -> torch.Tensor:
        """Return fused logits (C,H,W) for an RGB PIL image."""
        with torch.no_grad():  # GPU optimization: disable gradient computation
            resized = self.CDGNET_TRANS(image.convert("RGB")).unsqueeze(0).to(self.device)
            # horizontal flip TTA (stack: original + flipped)
            batch = torch.cat([resized, resized.flip(-1)], dim=0)
            logits = self._predict_logits(self.cdgnet, batch, image.size[::-1])  # (C,H,W)
            
            # GPU optimization: clean up intermediate tensors
            del resized, batch
            
        return logits

    def _process_cdgnet(self, image: Image.Image) -> np.ndarray:
        logits = self._process_cdgnet_logits(image)  # (C,H,W)
        # GPU optimization: move to CPU immediately after argmax
        parsing = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)
        
        # GPU optimization: clean up GPU tensor
        del logits
        torch.cuda.empty_cache()  # Free unused GPU memory
        
        return parsing
    
    def _process_segformer(self, image: Image.Image) -> np.ndarray:
        """Process image with Segformer for detailed face/body parsing."""
        with torch.no_grad():
            # Ensure image is RGB PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image = image.convert('RGB')
            
            # Process image with Segformer processor
            inputs = self.segformer_processor(
                images=image, return_tensors="pt",
            ).to(self.device)

            # Run inference
            outputs = self.segformer_model(**inputs)
            logits = outputs.logits
            
            # Resize output to match input image dimensions
            upsampled_logits = F.interpolate(
                logits,
                size=image.size[::-1],  # H x W
                mode='bilinear',
                align_corners=True
            )
            
            # Get label masks and move to CPU immediately
            labels = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            
            # GPU optimization: clean up GPU tensors
            del inputs, outputs, logits, upsampled_logits
            
        return labels
        """Fill transparent regions with a color perceptually far from the foreground."""
        if isinstance(rgba, Image.Image):
            rgba_img = rgba.copy()
        else:
            rgba_img = Image.fromarray(rgba, mode='RGBA')
        if isinstance(silh_mask, np.ndarray):
            silh = silh_mask
        else:
            silh = np.array(silh_mask)
        
        rgb = np.array(rgba_img)[:, :, :3]
        rgb = rgb * silh[:, :, None]
        rgba_img = np.dstack((rgb, silh * 255)).astype(np.uint8)
        rgba_img = Image.fromarray(rgba_img, mode='RGBA')
        rgba_arr = np.array(rgba_img)
        if rgba_arr.ndim != 3 or rgba_arr.shape[-1] != 4:
            return rgba_img

        alpha = rgba_arr[:, :, 3]
        fg_mask = alpha > 16
        bg_mask = ~fg_mask
        if not np.any(bg_mask):
            return rgba_img

        rgb = rgba_arr[:, :, :3]
        lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        fg_lab = lab_img[fg_mask]
        if fg_lab.size == 0:
            distinctive_color = np.array([45, 197, 244], dtype=np.uint8)
        else:
            step = max(1, fg_lab.shape[0] // 4000)
            sampled_lab = fg_lab[::step]
            sampled_rgb = rgb[fg_mask][::step].astype(np.float32)

            mean_rgb = (sampled_rgb.mean(axis=0) / 255.0).tolist()
            mean_h, _, _ = colorsys.rgb_to_hsv(*mean_rgb)
            complement_rgb = np.clip(
                np.array(
                    colorsys.hsv_to_rgb(
                        (mean_h + 0.5) % 1.0,
                        0.75,
                        0.9,
                    )
                )
                * 255.0,
                0,
                255,
            ).astype(np.uint8)

            predefined = np.array([
                (255, 255, 255),
                (0, 0, 0),
                (255, 140, 0),
                (0, 191, 255),
                (123, 104, 238),
                (64, 224, 208),
                (186, 85, 211),
                (60, 179, 113),
                (255, 99, 71),
            ], dtype=np.uint8)

            candidates = np.concatenate((predefined, complement_rgb[None]), axis=0)
            candidates = np.unique(candidates, axis=0)
            candidates_lab = cv2.cvtColor(
                candidates.reshape(-1, 1, 3),
                cv2.COLOR_RGB2LAB,
            ).reshape(-1, 3).astype(np.float32)

            best_idx = 0
            best_score = -1.0
            for idx, cand_lab in enumerate(candidates_lab):
                diff = sampled_lab - cand_lab
                dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
                min_dist = float(dist.min())
                mean_dist = float(dist.mean())
                score = min_dist + 0.1 * mean_dist
                if score > best_score:
                    best_score = score
                    best_idx = idx

            distinctive_color = candidates[best_idx]

            if best_score < 30.0:
                centroid_lab = sampled_lab.mean(axis=0)
                target_lab = np.array([
                    100.0 - centroid_lab[0],
                    -centroid_lab[1],
                    -centroid_lab[2],
                ], dtype=np.float32)
                target_lab[0] = np.clip(target_lab[0], 40.0, 90.0)
                target_lab[1:] = np.clip(target_lab[1:], -110.0, 110.0)
                lab_for_cv = np.array([
                    target_lab[0] * 255.0 / 100.0,
                    target_lab[1] + 128.0,
                    target_lab[2] + 128.0,
                ], dtype=np.float32).reshape(1, 1, 3)
                fallback_rgb = cv2.cvtColor(lab_for_cv, cv2.COLOR_Lab2RGB).reshape(3)
                distinctive_color = np.clip(fallback_rgb, 0, 255).astype(np.uint8)

        rgba_arr[bg_mask, :3] = distinctive_color

        zero_alpha_mask = bg_mask & (alpha == 0)
        rgba_arr[zero_alpha_mask, 3] = 255

        return Image.fromarray(rgba_arr, mode='RGBA')

    def _create_head_mask(self, parsing: np.ndarray, hair_mask: np.ndarray) -> np.ndarray:
        """Create a complete head mask including hair and face but excluding neck.
        
        Args:
            parsing: Segmentation parsing array with class labels (LIP format)
            hair_mask: Binary hair mask
            
        Returns:
            Binary mask of the complete head (hair + face + facial features + ears, without neck)
        """
        head_mask = np.zeros_like(parsing, dtype=np.uint8)
        
        # Include hair (from the refined hair mask for better quality)
        head_mask[hair_mask > 0] = 1
        
        # Include face
        head_mask[parsing == int(LIPClass.FACE)] = 1
        
        # Include hat if present
        head_mask[parsing == int(LIPClass.HAT)] = 1
        
        # Include sunglasses if present
        head_mask[parsing == int(LIPClass.SUNGLASSES)] = 1
        
        # Note: We explicitly exclude neck (LIPClass doesn't have separate facial features
        # like eyes, nose, ears - they're all part of FACE class in the standard LIP)
        # If using extended LIP with more detailed facial features, we could include those here
        
        return head_mask

    def _extract_hair_mask_sam(self, image: Image.Image) -> np.ndarray:
        """Extract hair mask using SAMMaskExtractor.
        
        Args:
            image: RGB PIL Image
            
        Returns:
            Binary hair mask as numpy array (uint8, values 0 or 1)
        """
        hair_mask_pil, score = self.sam_extractor(image, prompt="hair")
        hair_mask = np.array(hair_mask_pil)
        
        # Convert to binary mask (0 or 1)
        hair_mask = (hair_mask > 127).astype(np.uint8)
        
        return hair_mask

    def preprocess(self, image: Image.Image | np.ndarray, verbose: bool = False) -> dict[str, np.ndarray]:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        rgb = image.convert('RGB')

        # 1) Foreground matte / silhouette
        rgba, silh_mask = self._matte_image(rgb)
        silh = (np.array(silh_mask) > 50).astype(np.uint8)

        # Silhouette sanity checks
        if np.sum(silh) == 0:
            h, w = silh.shape
            silh[h//4:3*h//4, w//4:3*w//4] = 1
        elif np.sum(silh) < 0.01 * silh.size:
            print("Warning: Very small silhouette detected.")
        elif np.sum(silh) > 0.95 * silh.size:
            print("Warning: Very large silhouette detected.")

        # 2) Extract hair mask using SAM (replaces CDGNet for hair)
        hair_mask = self._extract_hair_mask_sam(rgb)
        
        # Ensure hair mask matches silhouette size
        if hair_mask.shape != silh.shape:
            hair_mask = cv2.resize(hair_mask, (silh.shape[1], silh.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        # Apply silhouette mask to hair
        hair_mask = (hair_mask * silh).astype(np.uint8)

        # 3) Run Segformer for face parsing (to detect neck for body mask)
        segf_parsing = self._process_segformer(rgb)

        # 5) Crop body mask to keep entire neck and body (remove head/face/lips)
        H, W = silh.shape
        
        # Detect neck from Segformer output
        # Combine NECK and NECK_L (left neck) classes
        neck_mask = ((segf_parsing == int(SegformerClass.NECK)) |
                     (segf_parsing == int(SegformerClass.NECK_L))).astype(np.uint8)

        if np.any(neck_mask):
            # Find the topmost pixel of the neck (where face ends and neck begins)
            ys = np.where(neck_mask)[0]
            y_threshold = int(np.min(ys))  # Top of neck
            print(f"Neck top detected at y={y_threshold}, keeping neck and everything below")
        else:
            # Fallback: if no neck detected, use chin/lower lip to estimate where neck should start
            print("Warning: No neck detected, falling back to chin/lip detection")
            ll_mask = (segf_parsing == int(SegformerClass.L_LIP))
            if not np.any(ll_mask):
                ll_mask = (segf_parsing == int(SegformerClass.U_LIP))
            if not np.any(ll_mask):
                ll_mask = (segf_parsing == int(SegformerClass.MOUTH))
            
            if np.any(ll_mask):
                ys = np.where(ll_mask)[0]
                # Use bottom of lip/mouth + small offset to start of neck
                y_threshold = int(np.max(ys)) + int(0.03 * H)  # Add ~3% of image height
            else:
                # Final fallback: assume neck starts ~65% down the image
                y_threshold = int(0.65 * H)
            print(f"Using estimated neck position at y={y_threshold}")

        # Zero everything ABOVE where the neck starts; keep entire neck and body below
        silh[:y_threshold, :] = 0
        
        # Dilate silhouette to restore thin neck / shoulder pixels
        kernel_size = 7
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        silh = cv2.dilate(silh, kernel, iterations=1)

        torch.cuda.empty_cache()

        results = {
            'body_mask': silh,      # body from top of neck down
            'hair_mask': hair_mask, # refined hair (from SAM)
        }
        return results

    def teardown(self):
        """Clean up resources, especially GPU memory."""
        if hasattr(self, 'sam_extractor'):
            del self.sam_extractor
        if hasattr(self, 'birefnet'):
            del self.birefnet
        if hasattr(self, 'cdgnet'):
            del self.cdgnet
        if hasattr(self, 'segformer_model'):
            del self.segformer_model
        if hasattr(self, 'segformer_processor'):
            del self.segformer_processor
        torch.cuda.empty_cache()


def run_all(data_dir, device='cuda', size=1024):
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
        
    input_dir = data_dir / "image"
    output_dir = data_dir / "balder_input"
    flame_seg_dir = None
    # data_dir / "pixel3dmm_output"
    bald_input_dir = data_dir / "bald/wo_seg/image"
    
    hair_dir = output_dir / "seg"
    body_dir = output_dir / "body_img"
    final_dir = output_dir / "final_mask"
    combined_dir = output_dir / "combined_hair_body"
    bald_final_dir = output_dir / "bald_final_mask"
    dataset_dir = output_dir / "dataset"

    for d in (hair_dir, body_dir, final_dir, combined_dir, bald_final_dir, dataset_dir):
        d.mkdir(parents=True, exist_ok=True)

    pipeline = HairMaskPipeline(device=device)

    # Check for pairs.csv to filter samples
    pairs_csv_path = data_dir / "pairs.csv"
    source_ids = None
    if pairs_csv_path.exists():
        try:
            pairs_df = pd.read_csv(pairs_csv_path)
            if 'source_id' in pairs_df.columns:
                source_ids = list(set(pairs_df['source_id'].unique()))
                source_ids = [str(sid) for sid in source_ids]
                print(f"📋 Found pairs.csv with {len(source_ids)} unique source_ids")
                print(f"   Only processing samples listed in source_id column")
            else:
                print(f"⚠️  pairs.csv exists but has no 'source_id' column, processing all images")
        except Exception as e:
            print(f"⚠️  Error reading pairs.csv: {e}, processing all images")
    else:
        print(f"ℹ️  No pairs.csv found at {pairs_csv_path}, processing all images")
    
    
    # Get all image files from input directory (supports common image formats)
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.webp']
    all_image_files = []
    for ext in image_extensions:
        all_image_files.extend(input_dir.glob(ext))

    all_image_files = sorted(all_image_files)
    
    # Filter by source_ids if pairs.csv was found
    if source_ids is not None:
        image_files = [f for f in all_image_files if f.stem in source_ids]
        print(f"   Filtered: {len(image_files)}/{len(all_image_files)} images match source_ids")
    else:
        image_files = all_image_files
    
    # Randomize order with timestamp seed
    random_seed = int(time.time())
    random.seed(random_seed)
    random.shuffle(image_files)
    print(f"🎲 Randomized processing order with seed: {random_seed}")
    
    # if not image_files:
    #     print(f"⚠️  No images found in {input_dir}" + 
    #           (" matching source_ids from pairs.csv" if source_ids is not None else ""))
    #     return

    print(f"\n{'='*60}")
    print(f"Processing original images from: {input_dir}")
    print(f"{'='*60}\n")

    processed_count = 0
    skipped_count = 0
    background_remover = BackgroundRemover()
    for img_path in image_files:
        name = img_path.stem  # filename without extension (e.g., "sample_001")
        
        # Check if already processed (dataset grid image exists)
        dataset_path = dataset_dir / f"{name}.png"
        if dataset_path.exists():
            skipped_count += 1
            print(f"⏭️  Skipping {name}: already processed ({skipped_count} skipped so far)")
            continue
        
        # define output paths using filename
        hair_path = hair_dir / f"{name}.png"
        body_path = body_dir / f"{name}.png"
        final_path = final_dir / f"{name}.png"
        combined_path = combined_dir / f"{name}.png"

        # load and convert original image (for hair mask)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"❌  Could not read {img_path}")
            continue
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")

        # First, compute hair mask from ORIGINAL image
        try:
            original_results = pipeline.preprocess(pil_img)
            hair_mask = (original_results["hair_mask"] * 255).astype(np.uint8)
        except Exception as e:
            print(f"❌  Error processing hair mask for {name}: {e}")
            continue

        # For body mask computation, use bald image if available
        body_input_img = pil_img  # default to original image
        use_bald = False
        if bald_input_dir is not None:
            bald_img_path = bald_input_dir / f"{name}.png"
            if bald_img_path.exists():
                bald_bgr = cv2.imread(str(bald_img_path), cv2.IMREAD_COLOR)
                if bald_bgr is not None:
                    body_input_img = Image.fromarray(cv2.cvtColor(bald_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
                    use_bald = True
                    print(f"Using bald image for body mask computation: {name}")
                else:
                    print(f"⚠️  Could not read bald image for {name}, using original image")
            else:
                print(f"⚠️  Bald image not found for {name}, using original image")

        # Compute body mask from bald image (or original if bald not available)
        try:
            if use_bald:
                body_mask = background_remover.remove_background(body_input_img)[1]
                # pipeline.preprocess(body_input_img)
                body_mask = np.array(body_mask).astype(np.uint8)
            else:
                # If no bald image, use body mask from original image processing
                body_mask = (original_results["body_mask"] * 255).astype(np.uint8)
        except Exception as e:
            print(f"❌  Error processing body mask for {name}: {e}")
            continue

        Image.fromarray(hair_mask).resize((size, size), Image.Resampling.NEAREST).save(hair_path)
        
        # Compute union with flame segmentation if available, otherwise use SAM head mask on bald image
        final_mask = body_mask.copy()
        if flame_seg_dir is not None:
            flame_seg_path = flame_seg_dir / name / "flame_segmentation.png"
            if flame_seg_path.exists():
                try:
                    # Load flame segmentation
                    flame_seg = cv2.imread(str(flame_seg_path), cv2.IMREAD_GRAYSCALE)
                    if flame_seg is not None:
                        # Resize flame segmentation to match body mask size
                        if flame_seg.shape != body_mask.shape:
                            flame_seg = cv2.resize(flame_seg, (body_mask.shape[1], body_mask.shape[0]), 
                                                   interpolation=cv2.INTER_NEAREST)
                        
                        # Compute union (logical OR): any pixel that is mask in either image
                        final_mask = np.maximum(body_mask, flame_seg)
                        print(f"✅ Processed {name}: computed union with flame segmentation")
                    else:
                        print(f"⚠️  Could not read flame segmentation for {name}, using SAM head mask fallback")
                        # Fallback: use SAM with "head" prompt on bald image
                        if use_bald:
                            head_mask_pil, _ = pipeline.sam_extractor(body_input_img, prompt="head")
                            head_mask = (np.array(head_mask_pil) > 127).astype(np.uint8) * 255
                            if head_mask.shape != body_mask.shape:
                                head_mask = cv2.resize(head_mask, (body_mask.shape[1], body_mask.shape[0]), 
                                                       interpolation=cv2.INTER_NEAREST)
                            final_mask = np.maximum(body_mask, head_mask)
                            print(f"  → Used SAM 'head' mask on bald image for {name}")
                except Exception as e:
                    print(f"⚠️  Error loading flame segmentation for {name}: {e}, using SAM head mask fallback")
                    # Fallback: use SAM with "head" prompt on bald image
                    if use_bald:
                        try:
                            head_mask_pil, _ = pipeline.sam_extractor(body_input_img, prompt="head")
                            head_mask = (np.array(head_mask_pil) > 127).astype(np.uint8) * 255
                            if head_mask.shape != body_mask.shape:
                                head_mask = cv2.resize(head_mask, (body_mask.shape[1], body_mask.shape[0]), 
                                                       interpolation=cv2.INTER_NEAREST)
                            final_mask = np.maximum(body_mask, head_mask)
                            print(f"  → Used SAM 'head' mask on bald image for {name}")
                        except Exception as sam_e:
                            print(f"  → SAM head mask also failed: {sam_e}, using body mask only")
            else:
                print(f"⚠️  Flame segmentation not found for {name} at {flame_seg_path}, using SAM head mask fallback")
                # Fallback: use SAM with "head" prompt on bald image
                if use_bald:
                    try:
                        head_mask_pil, _ = pipeline.sam_extractor(body_input_img, prompt="head")
                        head_mask = (np.array(head_mask_pil) > 127).astype(np.uint8) * 255
                        if head_mask.shape != body_mask.shape:
                            head_mask = cv2.resize(head_mask, (body_mask.shape[1], body_mask.shape[0]), 
                                                   interpolation=cv2.INTER_NEAREST)
                        final_mask = np.maximum(body_mask, head_mask)
                        print(f"  → Used SAM 'head' mask on bald image for {name}")
                    except Exception as sam_e:
                        print(f"  → SAM head mask also failed: {sam_e}, using body mask only")
                else:
                    print(f"  → No bald image available, using body mask only")
        else:
            print(f"✅ Processed {name}: body_mask and hair_mask saved (no flame segmentation)")

        # Save final mask (grayscale)
        Image.fromarray(final_mask).resize((size, size), Image.Resampling.NEAREST).save(final_path)

        # Save body mask as green color on black background (using final_mask)
        final_mask_resized = cv2.resize(final_mask, (size, size), interpolation=cv2.INTER_NEAREST)
        body_mask_rgb = np.zeros((size, size, 3), dtype=np.uint8)
        body_mask_rgb[final_mask_resized > 0] = [0, 255, 0]  # Green color for final mask regions
        Image.fromarray(body_mask_rgb).save(body_path)

        # Save combined mask: hair (red) + final_mask non-hair (green) + background (black)
        hair_mask_resized = cv2.resize(hair_mask, (size, size), interpolation=cv2.INTER_NEAREST)
        combined_mask = np.zeros((size, size, 3), dtype=np.uint8)
        
        # First, set final_mask regions (non-hair) to green
        final_only = (final_mask_resized > 0) & (hair_mask_resized == 0)
        combined_mask[final_only] = [0, 255, 0]  # Green for final mask (non-hair)
        
        # Then, set hair regions to red (overrides body where hair exists)
        combined_mask[hair_mask_resized > 0] = [255, 0, 0]  # Red for hair
        
        Image.fromarray(combined_mask).save(combined_path)

        # Create 2x2 grid visualization (512x512 each cell = 1024x1024 total)
        grid_size = 512
        grid_img = np.zeros((grid_size * 2, grid_size * 2, 3), dtype=np.uint8)
        
        # Top-left: combined_hair_body (already RGB)
        combined_resized = cv2.resize(combined_mask, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
        grid_img[0:grid_size, 0:grid_size] = combined_resized
        
        # Top-right: body_img (already RGB green)
        body_rgb_resized = cv2.resize(body_mask_rgb, (grid_size, grid_size), interpolation=cv2.INTER_NEAREST)
        grid_img[0:grid_size, grid_size:grid_size*2] = body_rgb_resized
        
        # Bottom-left: original image
        original_resized = cv2.resize(np.array(pil_img), (grid_size, grid_size), interpolation=cv2.INTER_LINEAR)
        grid_img[grid_size:grid_size*2, 0:grid_size] = original_resized
        
        # Bottom-right: bald image (if available)
        if use_bald:
            bald_resized = cv2.resize(np.array(body_input_img), (grid_size, grid_size), interpolation=cv2.INTER_LINEAR)
            grid_img[grid_size:grid_size*2, grid_size:grid_size*2] = bald_resized[:, :, :3]
        else:
            # If no bald image, show original again or black
            grid_img[grid_size:grid_size*2, grid_size:grid_size*2] = original_resized
        
        # Save grid
        Image.fromarray(grid_img).save(dataset_path)
        processed_count += 1

        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n🎉 Completed mask computation: {processed_count} processed, {skipped_count} skipped, {len(image_files)} total")

    # Process bald images if directory is provided
    # For bald images, we only use flame segmentation (no body mask computation)
    if bald_input_dir is not None and flame_seg_dir is not None:
        print(f"\n{'='*60}")
        print(f"Processing bald images from: {bald_input_dir}")
        print(f"Using flame segmentation only (no body mask computation)")
        print(f"{'='*60}\n")

        # Get all image files from bald input directory
        all_bald_image_files = []
        for ext in image_extensions:
            all_bald_image_files.extend(bald_input_dir.glob(ext))
        
        all_bald_image_files = sorted(all_bald_image_files)
        
        # Filter by source_ids if pairs.csv was found
        # if source_ids is not None:
        #     bald_image_files = [f for f in all_bald_image_files if f.stem in source_ids]
        #     print(f"   Filtered: {len(bald_image_files)}/{len(all_bald_image_files)} bald images match source_ids")
        # else:
        bald_image_files = all_bald_image_files
        
        # Randomize bald images order with same seed (for consistency)
        random.shuffle(bald_image_files)
        print(f"🎲 Randomized bald images processing order")
        
        if not bald_image_files:
            print(f"⚠️  No bald images found in {bald_input_dir}" +
                  (" matching source_ids from pairs.csv" if source_ids is not None else ""))
        else:
            bald_processed_count = 0
            bald_skipped_count = 0
            
            for img_path in bald_image_files:
                name = img_path.stem  # filename without extension (e.g., "sample_001")
                
                # define output path for bald images
                bald_final_path = bald_final_dir / f"{name}.png"
                
                # Skip if already processed
                if bald_final_path.exists():
                    bald_skipped_count += 1
                    print(f"⏭️  Skipping bald {name}: already processed ({bald_skipped_count} skipped so far)")
                    continue

                # Load flame segmentation directly (no body mask computation for bald images)
                flame_seg_path = flame_seg_dir / name / "flame_segmentation.png"
                if flame_seg_path.exists():
                    try:
                        # Load flame segmentation
                        flame_seg = cv2.imread(str(flame_seg_path), cv2.IMREAD_GRAYSCALE)
                        if flame_seg is not None:
                            # Resize flame segmentation to target size
                            flame_seg_resized = cv2.resize(flame_seg, (size, size), 
                                                           interpolation=cv2.INTER_NEAREST)
                            
                            # Save as final mask (only flame segmentation, no body mask)
                            Image.fromarray(flame_seg_resized).save(bald_final_path)
                            bald_processed_count += 1
                            print(f"✅ Processed bald {name}: saved flame segmentation")
                        else:
                            print(f"❌  Could not read flame segmentation for bald {name}")
                    except Exception as e:
                        print(f"❌  Error loading flame segmentation for bald {name}: {e}")
                else:
                    print(f"⚠️  Flame segmentation not found for bald {name} at {flame_seg_path}")

                torch.cuda.empty_cache()
                gc.collect()

            print(f"\n🎉 Completed bald images: {bald_processed_count} processed, {bald_skipped_count} skipped, {len(bald_image_files)} total")

    del pipeline
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hair mask pipeline on a directory of images")
    parser.add_argument("--data_dir", type=str,
                        default="/workspace/celeba_reduced",
                        help="Base data directory containing input images and outputs")
    # parser.add_argument("--input_dir", type=str,
    #                     default="/workspace/celeba_subset/image",
    #                     help="Input directory containing original images")
    # parser.add_argument("--bald_input_dir", type=str,
    #                     default=None,
    #                     help="Input directory containing bald images")
    # parser.add_argument("--output_dir", type=str, 
    #                     default="/workspace/celeba_subset/balder_input",
    #                     help="Output directory for masks and segmentations")
    # parser.add_argument("--flame_seg_dir", type=str,
    #                     default="/workspace/celeba_subset/pixel3dmm_output",
    #                     help="Directory containing flame segmentation subdirectories (e.g., sample_001/flame_segmentation.png)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--size", type=int, default=1024, help="Output mask size")
    
    args = parser.parse_args()
    
    run_all(
        data_dir=args.data_dir,
        device=args.device,
        size=args.size
    )
