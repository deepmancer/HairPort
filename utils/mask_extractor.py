import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config.segmentation_config import LIPSegmentationConfig, LIPClass

torch.set_float32_matmul_precision(['high', 'highest'][0])

class MaskExtractor:
    
    DEFAULT_CDGNET_CKPT = Path("assets/checkpoints/CDGNet/LIP_epoch_149.pth")
    CDGNET_TRANS = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    def __init__(
        self,
        cdgnet_ckpt: Path | None = None,
        device: str | torch.device = 'cuda',
    ):
        self.device = torch.device(device)
        self.cdgnet_ckpt = cdgnet_ckpt or self.DEFAULT_CDGNET_CKPT
        self.lip_config = LIPSegmentationConfig()
        self.cdgnet = self._init_cdgnet()
        self.scales = [0.66, 0.80, 1.0]
        self.flip_mapping = torch.tensor((15, 14, 17, 16, 19, 18), device=self.device)
        self._upsampler: nn.Upsample | None = None
    
    def _init_cdgnet(self) -> nn.Module:
        sys.path.append('modules/CDGNet')
        from networks.CDGNet import Res_Deeplab

        model = Res_Deeplab(num_classes=20)
        state = torch.load(self.cdgnet_ckpt, map_location='cpu')
        target = model.state_dict()

        new_state = {}
        for key in target:
            ckpt_key = 'module.' + key if 'module.' + key in state else key
            new_state[key] = deepcopy(state.get(ckpt_key, target[key]))

        model.load_state_dict(new_state)
        model = model.to(self.device)
        model.eval()
        return model
    
    def _ensure_upsampler(self, input_size: tuple[int, int]) -> None:
        if not self._upsampler or self._upsampler.size != input_size:
            self._upsampler = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    
    def _predict_logits(
        self, model: nn.Module, batch: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        scales = self.scales
        in_h, in_w = batch.shape[-2:]
        up_to_input = lambda x: F.interpolate(x, size=(in_h, in_w), mode="bilinear", align_corners=True)

        per_scale = []
        for s in scales:
            with torch.no_grad():
                scaled = F.interpolate(batch, scale_factor=s, mode='bilinear', align_corners=True)
                out = model(scaled)
                logits = out[0][-1]
                single, flipped = logits[0], logits[1]
                flipped[14:20, :, :] = flipped[self.flip_mapping, :, :]
                single = 0.5 * (single + flipped.flip(dims=[-1]))
                upsampled = up_to_input(single.unsqueeze(0))[0]
                per_scale.append(upsampled)
                del scaled, out, logits, single, flipped, upsampled
                
        with torch.no_grad():
            fused = torch.stack(per_scale, dim=0).mean(0)
            fused = F.interpolate(fused.unsqueeze(0), size=output_size, mode='bicubic')[0]
            del per_scale
            
        return fused
    
    def extract_parsing(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            resized = self.CDGNET_TRANS(image.convert("RGB")).unsqueeze(0).to(self.device)
            batch = torch.cat([resized, resized.flip(-1)], dim=0)
            logits = self._predict_logits(self.cdgnet, batch, image.size[::-1])
            parsing = torch.argmax(logits, dim=0).cpu().numpy().astype(np.uint8)
            del resized, batch, logits
            torch.cuda.empty_cache()
        return parsing
    
    def extract_logits(self, image: Image.Image) -> torch.Tensor:
        with torch.no_grad():
            resized = self.CDGNET_TRANS(image.convert("RGB")).unsqueeze(0).to(self.device)
            batch = torch.cat([resized, resized.flip(-1)], dim=0)
            logits = self._predict_logits(self.cdgnet, batch, image.size[::-1])
            del resized, batch
        return logits

    def compute_mask(
        self,
        image: Image.Image,
        label: int,
        return_pil: bool = False,
    ) -> np.ndarray:
        image_rgb = image.convert('RGB')
        parsing = self.extract_parsing(image_rgb)
        mask = (parsing == label).astype(np.uint8)
        
        # Zero-out mask pixels where alpha channel is 0
        if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
            alpha = np.array(image.split()[-1])
            mask = mask * (alpha > 0).astype(np.uint8)
        
        if return_pil:
            return Image.fromarray(mask * 255)
        return mask
    def compute_hair_mask(
        self,
        image: Image.Image,
        return_pil: bool = False,
    ) -> np.ndarray:
        hair_label = LIPClass.HAIR.value
        return self.compute_mask(image, hair_label, return_pil)
