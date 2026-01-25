import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm

input_images_dir = "/workspace/outputs/image"
aligned_images_dir = "/workspace/outputs/aligned_image"
bald_dataset_dir = "/workspace/bald_dataset"

METHOD_DIRS = {
    "HairPort w/ seg": "/workspace/outputs/bald/w_seg/aligned_image/",
    "HairPort w/o seg": "/workspace/outputs/bald/wo_seg/aligned_image/",
    "HairMapper": "/workspace/baselines_outputs/bald/hairmapper/",
    "HairCLIP v2": "/workspace/baselines_outputs/bald/hairclipv2/",
    "StableHair": "/workspace/baselines_outputs/bald/stable_hair/",
}

all_image_ids = os.listdir(input_images_dir)
valid_extensions = {'.png', '.jpg', '.jpeg'}
dirs_to_check = [
    input_images_dir,
    *METHOD_DIRS.values()
]

filtered_ids = []

for img_id in all_image_ids:
    # Check if the file exists in all directories with any valid extension
    exists_in_all = True
    for directory in dirs_to_check:
        found_in_dir = False
        for ext in valid_extensions:
            # Assuming img_id might already have an extension or we need to append it. 
            # Usually os.listdir returns filenames with extensions.
            # If img_id is just the stem, we check stem + ext.
            # If img_id is filename, we check if it exists directly.
            
            # Case 1: img_id is a filename (e.g., '001.png')
            if os.path.exists(os.path.join(directory, img_id)):
                found_in_dir = True
                break
            
            # Case 2: img_id is a stem, try appending extensions (less likely given os.listdir usually returns full names)
            # But let's handle the case where the filename in other dirs might have different extensions
            stem = os.path.splitext(img_id)[0]
            if os.path.exists(os.path.join(directory, stem + ext)):
                found_in_dir = True
                break
        
        if not found_in_dir:
            exists_in_all = False
            break
    
    if exists_in_all:
        filtered_ids.append(img_id)

all_available_image_ids = filtered_ids

print(f"Total images in input directory: {len(all_image_ids)}")
print(f"Total valid images found across all directories: {len(all_available_image_ids)}")


import torch
from PIL import Image
# Note: We use the 'Plus' pipeline for multi-image support in Qwen-2509
try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:
    raise ImportError("Please upgrade diffusers to support QwenImageEditPlusPipeline: pip install git+https://github.com/huggingface/diffusers.git")

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class QwenBalderPipeline:
    def __init__(self, model_id="Qwen/Qwen-Image-Edit-2509"):
        print(f"Loading Qwen-Image-Edit Plus from {model_id}...")
        # Initialize the specialized pipeline for multi-image input
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16
        ).to(DEVICE)
        # self.pipe.load_lora_weights(
        #     "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"
        # )
        # Optional: enable progress bar visibility
        self.pipe.set_progress_bar_config(disable=None)

    def generate(
        self, 
        image_input, 
        prompt_text,
        num_inference_steps=40,
        guidance_scale=1.0,
        true_cfg_scale=4.0,
        seed=42
    ):
        # 1. Prepare Inputs (ensure they are RGB PIL Images)
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        image_inputs = [img]
        
        print(f"Running Qwen Multi-Image Generation with prompt: {prompt_text[:100]}...")

        # 3. Run Inference
        # Using inference_mode for efficiency
        with torch.inference_mode():
            output = self.pipe(
                prompt=prompt_text,
                image=image_inputs,
                negative_prompt=" ",
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale, 
                guidance_scale=guidance_scale,
                generator=torch.Generator(DEVICE).manual_seed(seed)
            ).images[0]
            
        return output
    
qwen_balder_pipeline = QwenBalderPipeline()
prompt_balding = "Remove all hair from this person to make them bald while preserving their facial identity and keeping the background unchanged."
method_name = "Qwen-Image-Edit-2509"

# Configuration
BATCH_SIZE = 1
output_dir = bald_dataset_dir

# Process images in batches
for batch_start in tqdm(range(0, len(all_available_image_ids), BATCH_SIZE)):
    batch_ids = all_available_image_ids[batch_start:batch_start + BATCH_SIZE]
    
    for image_id in batch_ids:
        stem = os.path.splitext(image_id)[0]
        
        # Create output folder for this image
        image_output_dir = os.path.join(output_dir, stem)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Check if already processed
        safe_method_name = method_name.replace(" ", "_").replace("/", "_")
        output_path = os.path.join(image_output_dir, f"{safe_method_name}.png")
        if os.path.exists(output_path):
            print(f"Skipping {image_id} - already processed")
            continue
        
        # Construct input path from aligned_images_dir
        input_path = None
        if os.path.exists(os.path.join(input_images_dir, image_id)):
            input_path = os.path.join(input_images_dir, image_id)
        else:
            # Check variations with valid extensions
            for ext in valid_extensions:
                candidate = os.path.join(input_images_dir, stem + ext)
                if os.path.exists(candidate):
                    input_path = candidate
                    break
        
        if input_path is None:
            print(f"Warning: Could not find input image for {image_id}")
            continue

        # Generate bald version
        output_image = qwen_balder_pipeline.generate(
            image_input=input_path,
            prompt_text=prompt_balding,
            num_inference_steps=40,
            guidance_scale=1.0,
            true_cfg_scale=4.0,
            seed=42
        )
        
        # Save result
        output_image.save(output_path)

print(f"Processing complete. Results saved to {output_dir}")
