
import argparse
import json
import os
import traceback
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm


from utils.bg_remover import BackgroundRemover
from hairport.utility.uncrop_sdxl.controlnet_union import ControlNetModel_Union
from hairport.utility.uncrop_sdxl.pipeline_fill_sd_xl import StableDiffusionXLFillPipeline
from hairport.utility.uncrop_sdxl.image_uncropper import (
    ImageUncropper,
    compute_dynamic_resize_percentage,
    load_landmarks,
    compute_landmark_centroid,
    compute_alignment_offset,
)


@dataclass
class UncropperConfig:
    # Output dimensions
    width: int = 1024
    height: int = 1024
    
    # Overlap settings
    overlap_percentage: int = 5
    overlap_left: bool = True
    overlap_right: bool = True
    overlap_top: bool = True
    overlap_bottom: bool = True
    
    # Alignment
    alignment: str = "Middle"
    
    # Inference settings
    num_inference_steps: int = 12
    resize_option: str = "Custom"
    default_resize_percentage: float = 75.0
    
    # Blending settings for compositing original back onto result
    blend_pixels: int = 21  # Number of pixels for feathered edge blending
    
    # Face size configuration for dynamic resizing
    face_to_width_ratio: float = 0.45
    min_resize_percentage: float = 30.0
    max_resize_percentage: float = 100.0
    
    # Prompts
    negative_prompt: str = (
        "(cgi, 3d, grayscale, render, monochrome, sketch, pixelated, blurry, low quality, naked, nude, nudity, ugly drawing:1.8), "
        "face asymmetry, eyes asymmetry, deformed eyes, open mouth, text, cropped, out of frame, worst quality, "
        "low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, "
        "poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, "
        "bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, "
        "missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"
        "underexposed, harsh shadows, dramatic lighting, vignette, color cast, oversaturated, undersaturated"
    )
    default_prompt: str = "high-quality photo of a person, realistic, high quality, 4k, ultra-detailed"
    
    # Model settings
    dtype: str = "float16"
    device: str = "cuda"
    base_model: str = "SG161222/RealVisXL_V5.0_Lightning"
    vae_model: str = "madebyollin/sdxl-vae-fp16-fix"
    controlnet_repo: str = "xinsir/controlnet-union-sdxl-1.0"


class Uncropper:
    def __init__(self, config: Optional[UncropperConfig] = None):
        self.config = config or UncropperConfig()
        self.pipe = None
        self.image_uncropper = ImageUncropper(
            target_width=self.config.width,
            target_height=self.config.height,
            alignment=self.config.alignment,
            overlap_percentage=self.config.overlap_percentage,
            overlap_left=self.config.overlap_left,
            overlap_right=self.config.overlap_right,
            overlap_top=self.config.overlap_top,
            overlap_bottom=self.config.overlap_bottom,
        )
        self._device = self.config.device if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self.config.dtype == "float16" else torch.float32
    
    def load_pipeline(self) -> None:
        print("Loading models...")
        
        # Load ControlNet
        config_file = hf_hub_download(
            self.config.controlnet_repo,
            filename="config_promax.json",
        )
        config = ControlNetModel_Union.load_config(config_file)
        controlnet = ControlNetModel_Union.from_config(config)
        
        model_file = hf_hub_download(
            self.config.controlnet_repo,
            filename="diffusion_pytorch_model_promax.safetensors",
        )
        state_dict = load_state_dict(model_file)
        controlnet.load_state_dict(state_dict)
        controlnet.to(device=self._device, dtype=self._dtype)
        
        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            self.config.vae_model, torch_dtype=self._dtype
        ).to(self._device)
        
        # Load pipeline
        self.pipe = StableDiffusionXLFillPipeline.from_pretrained(
            self.config.base_model,
            torch_dtype=self._dtype,
            vae=vae,
            controlnet=controlnet,
            variant="fp16",
        ).to(self._device)
        
        self.pipe.scheduler = TCDScheduler.from_config(self.pipe.scheduler.config)
        
        print("Models loaded successfully!")
    
    @staticmethod
    def _can_expand(
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
        alignment: str
    ) -> bool:
        if alignment in ("Left", "Right") and source_width >= target_width:
            return False
        if alignment in ("Top", "Bottom") and source_height >= target_height:
            return False
        return True
    
    def _prepare_image_and_mask(
        self,
        image: Image.Image,
        resize_percentage: float,
        landmark_offset_x: Optional[int] = None,
        landmark_offset_y: Optional[int] = None,
    ) -> Tuple[Image.Image, Image.Image, Image.Image, Tuple[int, int, int, int]]:
        # Prepare image on canvas with mask. Returns: (background, mask, resized_source, (margin_x, margin_y, new_width, new_height))
        target_size = (self.config.width, self.config.height)
        
        # Calculate the scaling factor to fit the image within the target size
        scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        
        # Resize the source image to fit within target size
        source = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Apply resize percentage
        resize_factor = resize_percentage / 100
        new_width = int(source.width * resize_factor)
        new_height = int(source.height * resize_factor)
        
        # Resize the image
        source = source.resize((new_width, new_height), Image.LANCZOS)
        
        # Calculate the overlap in pixels
        overlap_x = max(int(new_width * (self.config.overlap_percentage / 100)), 1)
        overlap_y = max(int(new_height * (self.config.overlap_percentage / 100)), 1)
        
        # Calculate margins based on landmark offsets or alignment
        if landmark_offset_x is not None and landmark_offset_y is not None:
            # Use landmark-based placement (asymmetric)
            base_margin_x = (target_size[0] - new_width) // 2
            base_margin_y = (target_size[1] - new_height) // 2
            margin_x = base_margin_x + landmark_offset_x
            margin_y = base_margin_y + landmark_offset_y
        else:
            # Use alignment-based placement (symmetric)
            alignment = self.config.alignment
            if alignment == "Middle":
                margin_x = (target_size[0] - new_width) // 2
                margin_y = (target_size[1] - new_height) // 2
            elif alignment == "Left":
                margin_x = 0
                margin_y = (target_size[1] - new_height) // 2
            elif alignment == "Right":
                margin_x = target_size[0] - new_width
                margin_y = (target_size[1] - new_height) // 2
            elif alignment == "Top":
                margin_x = (target_size[0] - new_width) // 2
                margin_y = 0
            elif alignment == "Bottom":
                margin_x = (target_size[0] - new_width) // 2
                margin_y = target_size[1] - new_height
            else:
                margin_x = (target_size[0] - new_width) // 2
                margin_y = (target_size[1] - new_height) // 2
        
        # Adjust margins to eliminate gaps and keep image within canvas
        margin_x = max(0, min(margin_x, target_size[0] - new_width))
        margin_y = max(0, min(margin_y, target_size[1] - new_height))
        
        # Create background and paste
        background = Image.new('RGB', target_size, (255, 255, 255))
        background.paste(source, (margin_x, margin_y))
        
        # Create the mask
        mask = Image.new('L', target_size, 255)
        mask_draw = ImageDraw.Draw(mask)
        
        # Calculate overlap areas
        white_gaps_patch = 2
        
        left_overlap = margin_x + overlap_x if self.config.overlap_left else margin_x + white_gaps_patch
        right_overlap = margin_x + new_width - overlap_x if self.config.overlap_right else margin_x + new_width - white_gaps_patch
        top_overlap = margin_y + overlap_y if self.config.overlap_top else margin_y + white_gaps_patch
        bottom_overlap = margin_y + new_height - overlap_y if self.config.overlap_bottom else margin_y + new_height - white_gaps_patch
        
        # Only apply edge-specific overlap adjustments for alignment-based placement
        # (not for landmark-based asymmetric placement)
        if landmark_offset_x is None and landmark_offset_y is None:
            alignment = self.config.alignment
            if alignment == "Left":
                left_overlap = margin_x + overlap_x if self.config.overlap_left else margin_x
            elif alignment == "Right":
                right_overlap = margin_x + new_width - overlap_x if self.config.overlap_right else margin_x + new_width
            elif alignment == "Top":
                top_overlap = margin_y + overlap_y if self.config.overlap_top else margin_y
            elif alignment == "Bottom":
                bottom_overlap = margin_y + new_height - overlap_y if self.config.overlap_bottom else margin_y + new_height
        
        # Draw the mask (black rectangle where original image is)
        mask_draw.rectangle([
            (left_overlap, top_overlap),
            (right_overlap, bottom_overlap)
        ], fill=0)
        
        placement_info = (margin_x, margin_y, new_width, new_height)
        return background, mask, source, placement_info
    
    @staticmethod
    def _create_feathered_mask(width: int, height: int, blend_pixels: int) -> Image.Image:
        # Create feathered alpha mask (white center, gradient edges) for seamless blending
        if blend_pixels <= 0:
            return Image.new('L', (width, height), 255)
        
        # Create gradient arrays for each edge
        mask_array = np.ones((height, width), dtype=np.float32)
        
        # Clamp blend_pixels to not exceed half the dimension
        blend_x = min(blend_pixels, width // 2)
        blend_y = min(blend_pixels, height // 2)
        
        # Create linear gradients for each edge
        for i in range(blend_y):
            alpha = i / blend_y
            mask_array[i, :] *= alpha  # Top edge
            mask_array[height - 1 - i, :] *= alpha  # Bottom edge
        
        for i in range(blend_x):
            alpha = i / blend_x
            mask_array[:, i] *= alpha  # Left edge
            mask_array[:, width - 1 - i] *= alpha  # Right edge
        
        # Convert to PIL Image
        mask_array = (mask_array * 255).astype(np.uint8)
        return Image.fromarray(mask_array, mode='L')
    
    def _composite_original_onto_result(
        self,
        result: Image.Image,
        source_resized: Image.Image,
        margin_x: int,
        margin_y: int,
        blend_pixels: int,
    ) -> Image.Image:
        # Paste original image back onto result using feathered mask to preserve center perfectly
        # Create feathered mask for the source image
        feather_mask = self._create_feathered_mask(
            source_resized.width, source_resized.height, blend_pixels
        )
        
        # Create output by copying result
        output = result.copy()
        
        # Paste original back using feathered mask for blending
        output.paste(source_resized, (margin_x, margin_y), feather_mask)
        
        return output
    
    def uncrop(
        self,
        image: Image.Image,
        prompt: str,
        resize_percentage: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        landmark_offset_x: Optional[int] = None,
        landmark_offset_y: Optional[int] = None,
        centerize: bool = False,
        landmark_path: Optional[str] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        if self.pipe is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")
        
        resize_percentage = resize_percentage or self.config.default_resize_percentage
        negative_prompt = negative_prompt or self.config.negative_prompt
        
        # If centerize is True, compute horizontal offset to center facial landmarks
        if centerize and landmark_path is not None and os.path.exists(landmark_path):
            try:
                landmarks = load_landmarks(landmark_path)
                centroid_x, centroid_y = compute_landmark_centroid(landmarks)
                
                # Compute scale factors
                scale_factor = min(
                    self.config.width / image.width,
                    self.config.height / image.height
                )
                resize_factor = resize_percentage / 100
                total_scale = scale_factor * resize_factor
                
                # Compute where centroid would be with default (middle) alignment
                new_width = int(image.width * total_scale)
                new_height = int(image.height * total_scale)
                default_margin_x = (self.config.width - new_width) // 2
                
                # Scaled centroid position relative to image origin
                scaled_centroid_x = centroid_x * total_scale
                
                # Current centroid position on canvas with default alignment
                current_centroid_canvas_x = default_margin_x + scaled_centroid_x
                
                # Target position is horizontal center of canvas
                target_centroid_x = self.config.width / 2
                
                # Compute offset needed (this will be added to base margin)
                centerize_offset_x = int(target_centroid_x - current_centroid_canvas_x)
                
                # Override or combine with existing offset
                if landmark_offset_x is None:
                    landmark_offset_x = centerize_offset_x
                else:
                    landmark_offset_x += centerize_offset_x
                    
            except Exception as e:
                print(f"Warning: Failed to compute centerize offset: {e}")
        
        # Prepare image and mask, get placement info and resized source
        background, mask, source_resized, placement = self._prepare_image_and_mask(
            image, resize_percentage, landmark_offset_x, landmark_offset_y
        )
        margin_x, margin_y, new_width, new_height = placement
        
        # Check if expansion is possible
        # Note: background is already at target size, so we check the resized source dimensions
        if not self._can_expand(
            new_width, new_height,
            self.config.width, self.config.height,
            self.config.alignment
        ):
            # Fall back to middle alignment if can't expand
            # This condition indicates the image is too large to expand in the chosen direction
            pass
        
        # Create controlnet input image
        cnet_image = background.copy()
        cnet_image.paste(0, (0, 0), mask)
        
        # Prepare prompt
        final_prompt = f"{prompt}, high quality, 4k"
        
        # Encode prompts
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(final_prompt, self._device, True, negative_prompt=negative_prompt)
        
        # Run inference
        result = None
        for output in self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=cnet_image,
            num_inference_steps=self.config.num_inference_steps
        ):
            result = output
        
        # Composite original back onto final result with feathered blending
        # This ensures the center (original image) is pixel-perfect preserved
        result = self._composite_original_onto_result(
            result=result,
            source_resized=source_resized,
            margin_x=margin_x,
            margin_y=margin_y,
            blend_pixels=self.config.blend_pixels,
        )
        
        # Compute resize info with landmark offsets
        resize_info = self.image_uncropper.compute_resize_info(
            image, resize_percentage, landmark_offset_x, landmark_offset_y
        )
        
        return result, resize_info
    
    def crop(
        self,
        uncropped_image: Image.Image,
        resize_info: Dict[str, Any],
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        return self.image_uncropper.crop_from_uncropped(
            uncropped_image, resize_info, output_size
        )
    
    def compute_resize_percentage_from_landmarks(
        self,
        landmark_path: str,
    ) -> Tuple[float, Optional[float], bool]:
        if not os.path.exists(landmark_path):
            return self.config.default_resize_percentage, None, False
        
        try:
            resize_percentage, face_size = compute_dynamic_resize_percentage(
                landmark_path=landmark_path,
                target_width=self.config.width,
                face_to_width_ratio=self.config.face_to_width_ratio,
                min_percentage=self.config.min_resize_percentage,
                max_percentage=self.config.max_resize_percentage,
            )
            return resize_percentage, face_size, True
        except Exception as e:
            print(f"Warning: Error computing resize percentage: {e}")
            return self.config.default_resize_percentage, None, False
    
    def save_resize_info(
        self,
        resize_info: Dict[str, Any],
        output_path: str,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.image_uncropper.save_resize_info(resize_info, output_path, extra_info)
    
    @staticmethod
    def load_resize_info(json_path: str) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        _, resize_info, extra_info = ImageUncropper.load_resize_info(json_path)
        return resize_info, extra_info

    @staticmethod
    def compute_face_size(landmark_path: str) -> float:
        landmarks = load_landmarks(landmark_path)
        return ImageUncropper.compute_face_size_from_landmarks(landmarks)

    @staticmethod
    def compute_source_face_ratio(
        source_landmark_path: str,
        source_image_width: int,
    ) -> float:
        # Compute face-to-width ratio from source landmarks
        source_face_size = Uncropper.compute_face_size(source_landmark_path)
        return source_face_size / source_image_width

    def compute_landmark_alignment_offset(
        self,
        source_landmark_path: str,
        source_image_width: int,
        source_image_height: int,
        target_landmark_path: str,
        target_image_width: int,
        target_image_height: int,
        target_resize_percentage: float,
        source_resize_percentage: float = 100.0,
    ) -> Tuple[int, int]:
        # Compute (offset_x, offset_y) to align target landmarks with source landmarks on canvas
        # Accounts for both source and target resize percentages
        # Load landmarks
        source_landmarks = load_landmarks(source_landmark_path)
        target_landmarks = load_landmarks(target_landmark_path)
        
        # Compute the final resize factor for target
        target_scale = min(
            self.config.width / target_image_width,
            self.config.height / target_image_height
        )
        target_resize_factor = target_scale * (target_resize_percentage / 100)
        
        # Compute alignment offset
        offset_x, offset_y = compute_alignment_offset(
            source_landmarks=source_landmarks,
            target_landmarks=target_landmarks,
            target_resize_factor=target_resize_factor,
            source_image_width=source_image_width,
            source_image_height=source_image_height,
            target_image_width=target_image_width,
            target_image_height=target_image_height,
            target_canvas_width=self.config.width,
            target_canvas_height=self.config.height,
            source_resize_percentage=source_resize_percentage,
        )
        
        return offset_x, offset_y

    def compute_resize_percentage_matching_source(
        self,
        source_landmark_path: str,
        source_image_width: int,
        target_landmark_path: str,
        safeguard_resolution: float = 0.0,
    ) -> Tuple[float, float, float]:
        # Compute target resize % to match source face ratio. Returns: (resize_%, source_ratio, target_face_size)
        source_face_ratio = self.compute_source_face_ratio(
            source_landmark_path, source_image_width
        )
        target_face_size = self.compute_face_size(target_landmark_path)
        
        resize_percentage = ImageUncropper.compute_resize_percentage(
            face_size_original=target_face_size,
            target_width=self.config.width,
            face_to_width_ratio=source_face_ratio,
            min_percentage=self.config.min_resize_percentage,
            max_percentage=self.config.max_resize_percentage,
        )
        # Apply safeguard and ensure it stays within bounds
        resize_percentage = max(
            self.config.min_resize_percentage,
            min(self.config.max_resize_percentage, resize_percentage - safeguard_resolution)
        )
        return resize_percentage, source_face_ratio, target_face_size

    def uncrop_matching_source(
        self,
        target_image: Image.Image,
        target_landmark_path: str,
        source_landmark_path: str,
        source_image_width: int = 1024,
        source_image_height: int = 1024,
        prompt: str = "High quality photo of a person, high resolution, high quality, 4k, ultra-detailed, sharp focus, professional lighting, studio lighting, uniform lighting",
        negative_prompt: Optional[str] = None,
        safeguard_resolution: float = 0.0,
        source_resize_percentage: float = 100.0,
    ) -> Tuple[Image.Image, Dict[str, Any], Dict[str, Any]]:
        # Asymmetric uncrop: match target face ratio to source + align landmarks
        # safeguard_resolution reduces resize % for generation (more context) but not for crop-back
        # source_resize_percentage is critical for correct landmark alignment
        # Returns: (uncropped_image, resize_info, extra_info)
        # Compute resize percentage matching the source
        resize_percentage, source_face_ratio, target_face_size = \
            self.compute_resize_percentage_matching_source(
                source_landmark_path=source_landmark_path,
                source_image_width=source_image_width,
                target_landmark_path=target_landmark_path,
                safeguard_resolution=safeguard_resolution,
            )
        
        # Compute landmark alignment offsets using the computed percentage
        offset_x, offset_y = self.compute_landmark_alignment_offset(
            source_landmark_path=source_landmark_path,
            source_image_width=source_image_width,
            source_image_height=source_image_height,
            target_landmark_path=target_landmark_path,
            target_image_width=target_image.width,
            target_image_height=target_image.height,
            target_resize_percentage=resize_percentage,
            source_resize_percentage=source_resize_percentage,
        )
        
        # Perform uncropping
        result, resize_info = self.uncrop(
            image=target_image,
            prompt=prompt,
            resize_percentage=resize_percentage,
            negative_prompt=negative_prompt,
            landmark_offset_x=offset_x,
            landmark_offset_y=offset_y,
        )
        
        extra_info = {
            "face_size_original": target_face_size,
            "face_to_width_ratio": source_face_ratio,
            "custom_resize_percentage": resize_percentage,
            "safeguard_resolution": safeguard_resolution,
            "landmarks_found": True,
            "source_landmark_path": source_landmark_path,
            "source_image_width": source_image_width,
            "source_image_height": source_image_height,
            "source_resize_percentage": source_resize_percentage,
            "landmark_offset_x": offset_x,
            "landmark_offset_y": offset_y,
        }
        
        return result, resize_info, extra_info


def load_prompt_from_json(
    prompt_path: str,
    default_prompt: str = "high-quality photo of a person, realistic, high quality, 4k, ultra-detailed",
    prompt_key: str = "description",
) -> str:
    if not os.path.exists(prompt_path):
        return default_prompt
    
    try:
        with open(prompt_path, 'r') as f:
            prompt_data = json.load(f)
        description = prompt_data.get("subject", [{}])[0].get(prompt_key, default_prompt)
        # background = prompt_data.get("background", "")
        description = description.replace(", no background", ", a plain background")
        return f"{description} High quality, 4k, ultra-detailed".strip()
    except Exception:
        return default_prompt


def process_batch(
    uncropper: Uncropper,
    images_dir: str,
    output_dir: str,
    resize_info_dir: str,
    prompt_dir: Optional[str] = None,
    landmarks_dir: Optional[str] = None,
    skip_existing: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(resize_info_dir, exist_ok=True)
    import random
    import time
    # Get list of images to process
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    target_ids = [os.path.splitext(f)[0] for f in image_files]

    timestamp = int(time.time())
    random.seed(timestamp)
    random.shuffle(target_ids)

    print(f"Processing {len(target_ids)} images...")
    bg_remover = BackgroundRemover()
    for target_id in tqdm(target_ids):
        output_path = os.path.join(output_dir, f"{target_id}.png")
        resize_info_path = os.path.join(resize_info_dir, f"{target_id}.json")
        
        if skip_existing and os.path.exists(output_path):
            continue
        
        # Find input image
        input_image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(images_dir, f"{target_id}{ext}")
            if os.path.exists(candidate):
                input_image_path = candidate
                break
        
        if input_image_path is None:
            print(f"Warning: Image not found for {target_id}")
            continue
        
        try:
            input_image = Image.open(input_image_path).convert("RGB")
            
            
            # input_image_rgba, silh_mask_pil = bg_remover.remove_background(input_image, refine_foreground=True)
            # Make non-foreground (background) white using the silhouette mask
            # silh_mask_pil is the foreground mask (white = foreground, black = background)
            # input_image = input_image_rgba.convert("RGB")
            # Compute resize percentage from landmarks if available
            resize_percentage = uncropper.config.default_resize_percentage
            face_size = None
            landmarks_found = False
            
            if landmarks_dir:
                landmark_path = os.path.join(landmarks_dir, target_id, "landmarks.npy")
                resize_percentage, face_size, landmarks_found = \
                    uncropper.compute_resize_percentage_from_landmarks(landmark_path)
            
            # Load prompt if available
            prompt = uncropper.config.default_prompt
            if prompt_dir:
                prompt_path = os.path.join(prompt_dir, f"{target_id}.json")
                prompt = load_prompt_from_json(prompt_path, uncropper.config.default_prompt)
            
            print(f"Processing {target_id}: prompt={prompt[:60]}...")
            # Perform uncropping
            result, resize_info = uncropper.uncrop(
                image=input_image,
                prompt=prompt,
                resize_percentage=resize_percentage,
                centerize=True,
            )
            
            # Save output
            out_image = result.convert("RGB").resize(
                (uncropper.config.width, uncropper.config.height),
                Image.Resampling.LANCZOS
            )
            out_image.save(output_path)
            
            # Save resize info
            extra_info = {
                "target_id": target_id,
                "face_size_original": face_size,
                "face_to_width_ratio": uncropper.config.face_to_width_ratio,
                "custom_resize_percentage": resize_percentage,
                "landmarks_found": landmarks_found,
            }
            uncropper.save_resize_info(resize_info, resize_info_path, extra_info)

        except Exception as e:
            print(f"Error processing {target_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("Batch processing complete!")


def main():
    parser = argparse.ArgumentParser(
        description="SDXL-based Image Uncropping Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspace/celeba_reduced/",
        help="Directory containing input images",
    )
    # parser.add_argument(
    #     "--output-dir",
    #     type=str,
    #     required=True,
    #     help="Directory to save outpainted images",
    # )
    
    # # Optional directories
    # parser.add_argument(
    #     "--resize-info-dir",
    #     type=str,
    #     default=None,
    #     help="Directory to save resize info (defaults to output-dir/resize_info)",
    # )
    # parser.add_argument(
    #     "--prompt-dir",
    #     type=str,
    #     default=None,
    #     help="Directory containing prompt JSON files",
    # )
    # parser.add_argument(
    #     "--landmarks-dir",
    #     type=str,
    #     default=None,
    #     help="Directory containing landmark files for dynamic resizing",
    # )
    
    # Image processing options
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height",
    )
    parser.add_argument(
        "--alignment",
        type=str,
        default="Middle",
        choices=["Middle", "Left", "Right", "Top", "Bottom"],
        help="Alignment of original image on canvas",
    )
    parser.add_argument(
        "--overlap-percentage",
        type=int,
        default=5,
        help="Overlap percentage for seamless blending",
    )
    parser.add_argument(
        "--default-resize-percentage",
        type=float,
        default=50.0,
        help="Default resize percentage when landmarks not available",
    )
    parser.add_argument(
        "--blend-pixels",
        type=int,
        default=21,
        help="Number of pixels for feathered edge blending (0 to disable)",
    )
    
    # Face size configuration
    parser.add_argument(
        "--face-to-width-ratio",
        type=float,
        default=0.35,
        help="Target face-to-width ratio for dynamic resizing",
    )
    parser.add_argument(
        "--min-resize-percentage",
        type=float,
        default=40.0,
        help="Minimum resize percentage",
    )
    parser.add_argument(
        "--max-resize-percentage",
        type=float,
        default=100.0,
        help="Maximum resize percentage",
    )
    
    # Inference options
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=8,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Data type for model weights",
    )
    
    # Model options
    parser.add_argument(
        "--base-model",
        type=str,
        default="SG161222/RealVisXL_V5.0_Lightning",
        help="Base SDXL model to use",
    )
    
    # Processing options
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Process all images even if output exists",
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = UncropperConfig(
        width=args.width,
        height=args.height,
        alignment=args.alignment,
        overlap_percentage=args.overlap_percentage,
        default_resize_percentage=100.0,
        blend_pixels=args.blend_pixels,
        face_to_width_ratio=args.face_to_width_ratio,
        min_resize_percentage=args.min_resize_percentage,
        max_resize_percentage=args.max_resize_percentage,
        num_inference_steps=args.num_inference_steps,
        device=args.device,
        dtype=args.dtype,
        base_model=args.base_model,
    )
    
    # Initialize uncropper
    uncropper = Uncropper(config)
    uncropper.load_pipeline()
    
    # Process batch
    
    images_dir = os.path.join(args.data_dir, "relighted_image")
    output_dir = os.path.join(args.data_dir, "image_outpainted")
    resize_info_dir = os.path.join(args.data_dir, "outpainting_params")
    prompts_dir = os.path.join(args.data_dir, "prompt")
    landmarks_dir = os.path.join(args.data_dir, "lmk")
    process_batch(
        uncropper=uncropper,
        images_dir=images_dir,
        output_dir=output_dir,
        resize_info_dir=resize_info_dir,
        prompt_dir=prompts_dir,
        landmarks_dir=landmarks_dir,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()