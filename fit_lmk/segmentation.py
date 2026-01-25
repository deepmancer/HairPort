import numpy as np
import torch
from PIL import Image
from pathlib import Path
import sys

from utils.bg_remover import BackgroundRemover
from utils.mask_extractor import MaskExtractor
from config.segmentation_config import LIPClass


class FaceSegmenter:
    def __init__(self, device='cuda'):
        self.device = device
        self.bg_remover = BackgroundRemover(device=device)
        self.mask_extractor = MaskExtractor(device=device)
    
    def extract_masks(self, image_path):
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert('RGB')
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        else:
            image = image_path
        
        rgba, silh_mask = self.bg_remover.remove_background(image)
        silh = (np.array(silh_mask) > 50).astype(np.uint8)
        
        parsing = self.mask_extractor.extract_parsing(image)
        hair_mask = (parsing == int(LIPClass.HAIR)).astype(np.uint8)
        
        hair_mask = (hair_mask * silh).astype(np.uint8)
        face_mask = (silh & ~hair_mask).astype(np.uint8)
        bg_mask = 1 - silh
        
        return {
            'face_mask': face_mask,
            'hair_mask': hair_mask,
            'bg_mask': bg_mask,
            'body_mask': silh
        }
