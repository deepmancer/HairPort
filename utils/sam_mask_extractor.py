import torch
from PIL import Image
import numpy as np

# from sam3.model_builder import build_sam3_image_model
# from sam3.model.sam3_image_processor import Sam3Processor

from transformers import Sam3Processor, Sam3Model

class SAMMaskExtractor:
    def __init__(self, confidence_threshold: float = 0.35, detection_threshold=0.4, device: str = "cuda"):
        self.confidence_threshold = confidence_threshold
        self.detection_threshold = detection_threshold
        self.device = device
        self.sam_model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.sam_model.eval()
        self.sam_processor = Sam3Processor.from_pretrained("facebook/sam3")

    def __call__(self, image, prompt: str = "head hair"):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise TypeError("Image must be PIL.Image or numpy.ndarray")
        
        image = image.convert("RGB")
        inputs = self.sam_processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        results = self.sam_processor.post_process_instance_segmentation(
            outputs,
            threshold=self.detection_threshold,
            mask_threshold=self.confidence_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        
        masks, boxes, scores = results["masks"], results["boxes"], results["scores"]
        masks = masks.cpu().detach().numpy()
        
        # Remove leading batch dimension if present
        while masks.ndim == 4 and masks.shape[0] == 1:
            masks = masks.squeeze(0)
        
        # Remove trailing singleton dimensions (e.g., [N, 1, H, W] -> [N, H, W])
        while masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        
        # Now remove leading singleton from 3D masks (e.g., [1, H, W] -> [H, W])
        while masks.ndim == 3 and masks.shape[0] == 1:
            masks = masks.squeeze(0)        
        
        # Handle multiple masks by combining them with logical OR
        if masks.ndim == 3 and masks.shape[0] > 1:
            # Multiple masks detected: [N, H, W]
            print(f"Detected {masks.shape[0]} masks, combining with logical OR")
            combined_mask = np.any(masks > 0.5, axis=0)
            mask_np = (combined_mask * 255).astype(np.uint8)
            avg_score = float(np.mean(scores.cpu().numpy())) if len(scores) > 0 else 0.0
        elif masks.ndim == 2:
            # Already 2D mask: [H, W]
            mask_np = (masks * 255).astype(np.uint8)
            avg_score = scores[0].item() if len(scores) > 0 else 0.0
        else:
            raise ValueError(f"Unexpected mask shape after processing: {masks.shape}")
        
        mask_pil = Image.fromarray(mask_np)
        
        return mask_pil, avg_score
    
    def __del__(self):
        if hasattr(self, 'sam_model'):
            del self.sam_model
        if hasattr(self, 'sam_processor'):
            del self.sam_processor
        torch.cuda.empty_cache()
