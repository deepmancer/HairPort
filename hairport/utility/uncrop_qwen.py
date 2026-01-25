import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    FluxFillPipeline,
    QwenImageEditPipeline,
)
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from tqdm import tqdm

from hairport.utility.qwenimage.pipeline_qwen_image_edit import QwenImageEditPipeline
from hairport.utility.qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
# from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3
# from optimization import optimize_pipeline_
def can_expand(source_width, source_height, target_width, target_height, alignment):
    """Checks if the image can be expanded based on the alignment."""
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True

def prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
    """Prepares the image with white margins and creates a mask for outpainting."""
    target_size = (width, height)
    
    # Calculate the scaling factor to fit the image within the target size
    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)
    
    # Resize the source image to fit within target size
    source = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Apply resize option using percentages
    if resize_option == "Full":
        resize_percentage = 100
    elif resize_option == "50%":
        resize_percentage = 50
    elif resize_option == "33%":
        resize_percentage = 33
    elif resize_option == "25%":
        resize_percentage = 25
    else:  # Custom
        resize_percentage = custom_resize_percentage
    
    # Calculate new dimensions based on percentage
    resize_factor = resize_percentage / 100
    new_width = int(source.width * resize_factor)
    new_height = int(source.height * resize_factor)
    
    # Ensure minimum size of 64 pixels
    new_width = max(new_width, 64)
    new_height = max(new_height, 64)
    
    # Resize the image
    source = source.resize((new_width, new_height), Image.LANCZOS)
    
    # Calculate the overlap in pixels based on the percentage
    overlap_x = int(new_width * (overlap_percentage / 100))
    overlap_y = int(new_height * (overlap_percentage / 100))
    
    # Ensure minimum overlap of 1 pixel
    overlap_x = max(overlap_x, 1)
    overlap_y = max(overlap_y, 1)
    
    # Calculate margins based on alignment
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
    
    # Adjust margins to eliminate gaps
    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))
    
    # Create a new background image with white margins and paste the resized source image
    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))
    
    # Create the mask
    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)
    
    # Calculate overlap areas
    white_gaps_patch = 2
    left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
    right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
    top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
    bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch
    
    if alignment == "Left":
        left_overlap = margin_x + overlap_x if overlap_left else margin_x
    elif alignment == "Right":
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
    elif alignment == "Top":
        top_overlap = margin_y + overlap_y if overlap_top else margin_y
    elif alignment == "Bottom":
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height
    
    # Draw the mask
    mask_draw.rectangle([
        (left_overlap, top_overlap),
        (right_overlap, bottom_overlap)
    ], fill=0)
    
    return background, mask


data_dir = "/workspace/outputs/bald/w_seg/"
images_dir = data_dir + "image/"
# pairs_df ="/workspace/outputs/pairs
output_dir = data_dir + "image_outpainted/"
os.makedirs(output_dir, exist_ok=True)

# pairs_df = pd.read_csv(pairs_df)
# all_target_ids = pairs_df['target_id'].tolist()
unique_target_ids = list(os.listdir(images_dir))
unique_target_ids = [fname.split(".")[0] for fname in unique_target_ids]
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Scheduler configuration for Lightning
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# Initialize scheduler with Lightning config
# scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
from diffusers import QwenImageEditPlusPipeline

# --- Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit", torch_dtype=dtype).to(device)
pipe.transformer.__class__ = QwenImageTransformer2DModel
# pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())


pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning", 
        weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors",
)
pipe.fuse_lora()

for target_id in tqdm(unique_target_ids):
    negative_prompt = " "
    # Set up the generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(42)
    output_path = os.path.join(output_dir, f"{target_id}.png")
    if os.path.exists(output_path):
        print(f"Skipping {target_id}, already exists.")
        continue

    print(f"Processing {target_id}...")
    input_image_path = os.path.join(images_dir, f"{target_id}.png")
    input_image = Image.open(input_image_path).convert("RGB")
    outpainting_params=dict(    
        width=1024,
        height=1024,
        overlap_percentage=5,
        resize_option="Custom",
        custom_resize_percentage=30,
        alignment="Middle",
        overlap_left=True,
        overlap_right=True,
        overlap_top=True,
        overlap_bottom=True,
    )
    input_image_prepared, mask = prepare_image_and_mask(
        image=input_image,
        **outpainting_params
    )    
    result = pipe(
        image=input_image_prepared,
        prompt="Replace the white margins with coherent content that matches the existing image. Preserve the identity of the person and do not alter the inner content. ",
        true_cfg_scale=1.0,
        negative_prompt=" ",
        num_inference_steps=8,
        max_sequence_length=512,
        height=1024,
        width=1024,
        # rewrite_prompt=True,
        # api_name="/infer"
    ).images[0]
    out_image = result.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    out_image.save(output_path)
    