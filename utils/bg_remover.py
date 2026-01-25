
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ben2 import BEN_Base

torch.set_float32_matmul_precision(['high', 'highest'][0])

class BackgroundRemover:
    
    def __init__(self, device: str | torch.device = 'cuda'):
        self.device = torch.device(device)
        self.birefnet = self._init_birefnet()
    
    def _init_birefnet(self) -> nn.Module:
        model = BEN_Base.from_pretrained("PramaLLC/BEN2").to(self.device)
        model.eval()
        return model

    def remove_background(self, image: Image.Image, refine_foreground: bool = False) -> tuple[Image.Image, Image.Image]:
        foreground = self.birefnet.inference(image, refine_foreground=refine_foreground)
        alpha = foreground.getchannel('A')
        mask = ((np.array(alpha) / 255.0) > 0.8).astype(np.uint8) * 255
        return foreground, Image.fromarray(mask)

__all__ = ['BackgroundRemover']
