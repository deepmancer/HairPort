#!/usr/bin/env python3
"""
Hair Transfer Restoration Script
Restores hair from aligned views onto bald source images using Flux 2.

Usage:
    python restore_flux.py --data_dir /workspace/outputs
"""

import torch
import os
import argparse
import cv2
from PIL import Image
import random
import time
import numpy as np
import json

from utils.bg_remover import BackgroundRemover
from utils.sam_mask_extractor import SAMMaskExtractor
from hairport.utility.uncrop_sdxl import ImageUncropper

PROMPTS = {
    "3D_AWARE": (
        f"Transfer only the hair from the reference subject in image 2 and image 3 onto the scalp of the bald person in image 1. The two reference images show the same hairstyle. use image 3 as the primary hair donor; use image 2 as an alignment/shape reference. "
        f"Strictly keep unchanged the bald person’s facial identity, body, and all non-hair areas of image 1, including background, lighting, camera/framing, and overall photographic rendering. "
        f"Replicate the hair in the reference image 3, including texture, color, shape, length, volume, hairline, parting, or intericate and fine-grained details (strand-level details and variations if existing) "
        f"Preserve the apparent hair length and volume, brow-to-hairline relative distance, and matching relative placement, proportions, and orientation to facial/body keypoints and head/body pose, as it is in the reference images. "
        # f"Seamlessly integthe hair onto the person in image 1 with physically consistent shading, lighting, and interaction with the scalp and face. "
        f"Match the added hair to image 1's visual medium, lighting conditions, and resolution. "
    ),
    "3D_UNAWARE": (
        f"Transfer only the hairstyle from the reference subject in image 2 onto the scalp of the bald person in image 1. "
        f"Strictly keep unchanged the bald person’s facial identity, body, and all non-hair areas of image 1, including background, lighting, camera/framing, and overall photographic rendering. "
        f"Replicate the hair from image 2, including texture, color, shape, length, volume, hairline, parting, or intericate and fine-grained details (strand-level details and variations if existing) "
        f"Preserve the apparent hair length and volume, brow-to-hairline relative distance, and matching relative placement, proportions, and orientation to facial/body keypoints and head/body pose, as it is in the reference image. "
        # f"Seamlessly blend and harmonize the hair onto the person in image 1 with physically consistent shading, lighting, and interaction with the scalp and face. "
        f"Match the added hair to image 1's visual medium, lighting conditions, and resolution. "
    ),
}


def load_flux_pipeline(device="cpu"):
    """Load Flux 2 pipeline with turbo LoRA."""
    from diffusers import Flux2Pipeline
    
    pipe = Flux2Pipeline.from_pretrained(
        "black-forest-labs/FLUX.2-dev", 
        # text_encoder=None,
        torch_dtype=torch.bfloat16
    ).to(device)
    
    pipe.load_lora_weights(
        "fal/FLUX.2-dev-Turbo", 
        weight_name="flux.2-turbo-lora.safetensors"
    )
    # pipe.fuse_lora()
    # pipe.unload_lora_weights()
    pipe.to(device)
    # pipe.enable_model_cpu_offload()
    
    return pipe


def run_flux_inference(pipe, prompt, images, height=1024, width=1024, guidance_scale=1.5):
    """Run inference with Flux 2 pipeline."""
    # Pre-shifted custom sigmas for 8-step turbo inference
    TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]
    
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=images,
            sigmas=TURBO_SIGMAS,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            num_inference_steps=8,
        ).images[0]
    
    return result



def build_output_filename(bald_version, actual_use_aligned):
    """Build output filename based on configuration.
    
    Args:
        bald_version: Version of bald image
        actual_use_aligned: Whether aligned image is actually used (after checking availability)
    """
    parts = ["transferred_flux"]
    
    # Add bald version
    parts.append(bald_version)
    
    # Add 3D awareness indicator based on actual usage
    if actual_use_aligned:
        parts.append("3d_aware")
    else:
        parts.append("3d_unaware")
    
    return "_".join(parts) + ".png"


def extract_hair_masked_image(
    image,
    bg_remover,
    sam_mask_extractor,
    resolution,
    do_centering=True,
    input_size=None,          # if None, uses min(H, W) to compute padding
    padding_ratio=0.15,
    mask_hair=True,
    color_foreground=False,   # if True, colors non-hair foreground region with gray
):
    """
    Mask the hair region in the image using SAM and remove background.

    If do_centering=True:
      - "Fit" (zoom in/out) the hair foreground so its bounding box fits inside
        a padded content box whose padding is (input_size * padding_ratio).
      - Center the fitted foreground.

    If color_foreground=True:
      - Colors the non-hair foreground region (face/body) with gray instead of white.

    The padding constraint is implemented as:
      content_width  = W - 2*pad
      content_height = H - 2*pad
      scale = min(content_width / bbox_w, content_height / bbox_h)
    """

    # Remove background
    image_no_bg, silh = bg_remover.remove_background(image)

    # Get hair mask using SAM (first mask)
    if mask_hair:
        hair_mask = sam_mask_extractor(image_no_bg, prompt="hair")[0]
    else:
        hair_mask = silh

    # Convert images to arrays
    image_array = np.asarray(image)
    hair_mask_array = np.asarray(hair_mask)
    silh_array = np.asarray(silh)

    # Ensure masks are single-channel
    if hair_mask_array.ndim == 3:
        hair_mask_array = hair_mask_array[..., 0]
    if silh_array.ndim == 3:
        silh_array = silh_array[..., 0]

    # Combine hair mask with silhouette and binarize
    hair_mask_bin = (hair_mask_array > 0) & (silh_array > 0)
    
    # Compute non-hair foreground mask (face/body without hair)
    foreground_mask = silh_array > 0
    non_hair_foreground = foreground_mask & ~hair_mask_bin

    # Create white background and apply hair mask
    masked_image = np.full_like(image_array, 255)
    
    # If color_foreground is True, color non-hair foreground with gray
    if color_foreground:
        if image_array.ndim == 2:
            masked_image[non_hair_foreground] = 128
        else:
            masked_image[non_hair_foreground, :] = 128
    
    # Apply hair pixels
    if image_array.ndim == 2:
        masked_image[hair_mask_bin] = image_array[hair_mask_bin]
    else:
        masked_image[hair_mask_bin, :] = image_array[hair_mask_bin, :]
    
    # Ensure RGB output (3 channels) with white background
    if masked_image.ndim == 2:
        masked_image = np.stack([masked_image] * 3, axis=-1)
    elif masked_image.shape[-1] == 4:
        # Handle RGBA: composite onto white background
        alpha = masked_image[..., 3:4] / 255.0
        rgb = masked_image[..., :3]
        white_bg = np.full_like(rgb, 255)
        masked_image = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)

    if not do_centering:
        result = Image.fromarray(masked_image.astype(np.uint8))
        if result.mode != "RGB":
            result = result.convert("RGB")
        return result.resize((resolution, resolution), Image.Resampling.LANCZOS)

    # Compute bbox of hair foreground
    ys, xs = np.where(hair_mask_bin)
    if ys.size == 0 or xs.size == 0:
        return Image.fromarray(masked_image.astype(np.uint8)).resize((resolution, resolution), Image.Resampling.LANCZOS)

    H, W = masked_image.shape[:2]
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    bbox_w = (x_max - x_min) + 1
    bbox_h = (y_max - y_min) + 1

    # Padding in pixels
    base = int(input_size) if input_size is not None else min(H, W)
    pad = int(round(base * float(padding_ratio)))

    # Content box size (must be positive)
    content_w = max(1, W - 2 * pad)
    content_h = max(1, H - 2 * pad)

    # Scale factor to fit bbox inside content box (zoom-in if bbox is small, zoom-out if bbox is large)
    # (Uniform scaling keeps aspect ratio.)
    scale = min(content_w / float(bbox_w), content_h / float(bbox_h))

    # Centers
    bbox_cx = (x_min + x_max) / 2.0
    bbox_cy = (y_min + y_max) / 2.0
    img_cx = (W - 1) / 2.0
    img_cy = (H - 1) / 2.0

    # Affine transform: scale around origin + translation that maps bbox center to image center
    # new_x = scale * x + tx  with tx = img_cx - scale*bbox_cx
    # new_y = scale * y + ty  with ty = img_cy - scale*bbox_cy
    tx = img_cx - scale * bbox_cx
    ty = img_cy - scale * bbox_cy
    M = np.float32([[scale, 0.0, tx],
                    [0.0, scale, ty]])

    border_value = 255 if masked_image.ndim == 2 else (255, 255, 255)
    masked_image = cv2.warpAffine(
        masked_image,
        M,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
        
    enhanced_image = Image.fromarray(masked_image.astype(np.uint8))
    # codeformer_enhancer.enhance(enhanced_image, background_enhance=False, face_upsample=False, upscale=4.0, codeformer_fidelity=0.25)
    # enhanced_image = enhanced_image.resize((W, H), Image.Resampling.LANCZOS)
    enhanced_image, silh = bg_remover.remove_background(enhanced_image, refine_foreground=False)
    # Convert enhanced image and silhouette to arrays
    enhanced_array = np.asarray(enhanced_image)
    silh_array = np.asarray(silh)
    
    # Ensure silhouette is single-channel
    if silh_array.ndim == 3:
        silh_array = silh_array[..., 0]
    
    # Create binary mask from silhouette
    silh_mask = silh_array > 0
    
    # Create white RGB background (ensure 3 channels)
    if enhanced_array.ndim == 2:
        result_image = np.full((enhanced_array.shape[0], enhanced_array.shape[1], 3), 255, dtype=np.uint8)
        for c in range(3):
            result_image[..., c][silh_mask] = enhanced_array[silh_mask]
    elif enhanced_array.shape[-1] == 4:
        # Handle RGBA: composite onto white background
        result_image = np.full((enhanced_array.shape[0], enhanced_array.shape[1], 3), 255, dtype=np.uint8)
        result_image[silh_mask, :] = enhanced_array[silh_mask, :3]
    else:
        result_image = np.full_like(enhanced_array, 255)
        result_image[silh_mask, :] = enhanced_array[silh_mask, :]
    
    enhanced_image = Image.fromarray(result_image.astype(np.uint8))
    if enhanced_image.mode != "RGB":
        enhanced_image = enhanced_image.convert("RGB")
    return enhanced_image.resize((resolution, resolution), Image.Resampling.LANCZOS)



def process_3d_aware(
    pipe,
    source_image,
    target_image_matted,
    view_aligned_image_path,
    background_remover,
    sam_mask_extractor,
    resolution,
    output_path,
    resize_info_path=None,
):
    view_aligned_image = Image.open(view_aligned_image_path).convert("RGB").resize((resolution, resolution))
    view_aligned_image_matted = background_remover.remove_background(view_aligned_image)[0]
    # Convert view_aligned_image_matted to RGB with white background for non-foreground pixels
    view_aligned_matted_array = np.asarray(view_aligned_image_matted)
    view_aligned_no_bg, view_aligned_alpha = background_remover.remove_background(view_aligned_image)
    view_aligned_alpha_array = np.asarray(view_aligned_alpha)
    if view_aligned_alpha_array.ndim == 3:
        view_aligned_alpha_array = view_aligned_alpha_array[..., 0]
    view_aligned_fg_mask = view_aligned_alpha_array > 0
    view_aligned_white_bg = np.full_like(view_aligned_matted_array, 255)
    view_aligned_white_bg[view_aligned_fg_mask, :] = view_aligned_matted_array[view_aligned_fg_mask, :]
    view_aligned_image_matted = Image.fromarray(view_aligned_white_bg.astype(np.uint8))
    view_aligned_image_masked = extract_hair_masked_image(
        view_aligned_image_matted, background_remover, sam_mask_extractor, resolution,
        do_centering=False, padding_ratio=0.025,
    )
    
    target_image_masked = extract_hair_masked_image(
        target_image_matted, background_remover, sam_mask_extractor, resolution,
        do_centering=False, padding_ratio=0.025, color_foreground=True,
    )
    
    print(f"View aligned image path: {view_aligned_image_path}", flush=True)
    print(f"Using 3D_AWARE prompt with 3 images", flush=True)
    
    images = [source_image, view_aligned_image_masked, target_image_masked]
    prompt = PROMPTS["3D_AWARE"]
    
    result = run_flux_inference(pipe, prompt, images, resolution, resolution)
    
    # Crop result back to original size if resize_info is available
    if resize_info_path and os.path.exists(resize_info_path):
        uncropper = ImageUncropper()
        resize_info = json.load(open(resize_info_path, "r"))
        result = uncropper.crop_from_uncropped(result, resize_info, output_size=(resolution, resolution))
        print(f"Cropped result using resize_info from: {resize_info_path}", flush=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)
    print(f"Saved output image at: {output_path}", flush=True)
    
    # Save flux input images (image 2 and 3, excluding source)
    output_dir = os.path.dirname(output_path)
    view_aligned_image_masked.save(os.path.join(output_dir, "flux_input_view_aligned.png"))
    target_image_masked.save(os.path.join(output_dir, "flux_input_target.png"))
    print(f"Saved flux input images to: {output_dir}", flush=True)
    
    return True


def process_3d_unaware(
    pipe,
    source_image,
    resolution,
    target_image_matted,
    background_remover,
    sam_mask_extractor,
    output_path,
    lift_3d_applied=False,
    aligned_image_exists=False,
    resize_info_path=None,
):
    print(f"Using 3D_UNAWARE prompt with 2 images (lift_3d_applied={lift_3d_applied}, aligned_exists={aligned_image_exists})", flush=True)
    target_image_masked = extract_hair_masked_image(
        target_image_matted, background_remover, sam_mask_extractor, resolution,
        do_centering=False, padding_ratio=0.025, color_foreground=True,
    )
    images = [source_image, target_image_masked]
    prompt = PROMPTS["3D_UNAWARE"]
    
    result = run_flux_inference(pipe, prompt, images, resolution, resolution)
    
    # Crop result back to original size if resize_info is available
    if resize_info_path and os.path.exists(resize_info_path):
        uncropper = ImageUncropper()
        resize_info = json.load(open(resize_info_path, "r"))
        result = uncropper.crop_from_uncropped(result, resize_info, output_size=(resolution, resolution))
        print(f"Cropped result using resize_info from: {resize_info_path}", flush=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.save(output_path)
    print(f"Saved output image at: {output_path}", flush=True)
    
    # Save flux input images (image 2 and 3, excluding source)
    output_dir = os.path.dirname(output_path)
    target_image_masked.save(os.path.join(output_dir, "flux_input_target.png"))
    return True


def process_sample(
    sampled_dir,
    pipe,
    background_remover,
    sam_mask_extractor,
    data_dir,
    view_aligned_dir,
    image_dir,
    bald_image_dir,
    resolution=1024,
    bald_version="w_seg",
    conditioning_mode="3d_aware",
):
    try:
        target_id, source_id = sampled_dir.split("_to_")
    except ValueError:
        print(f"Invalid directory name format: {sampled_dir}")
        return False
    
    pair_dir = os.path.join(view_aligned_dir, sampled_dir, bald_version)
    if not os.path.exists(pair_dir):
        print(f"Pair directory not found: {pair_dir}, skipping...")
        return False
    
    # Determine 3D lifting status
    camera_params_path = os.path.join(pair_dir, "camera_params.json")
    view_aligned_image_path = os.path.join(pair_dir, "alignment", "target_image.png")
    lift_3d_applied = os.path.exists(camera_params_path)
    aligned_image_exists = lift_3d_applied and os.path.exists(view_aligned_image_path)
    
    # Determine processing mode based on conditioning_mode
    if conditioning_mode == "3d_aware":
        use_3d_aware = aligned_image_exists
        if not aligned_image_exists:
            reason = "angle diff below threshold" if not lift_3d_applied else "aligned image not found"
            print(f"No 3D lifting for {sampled_dir}/{bald_version} ({reason}), using 3D_UNAWARE")
    elif conditioning_mode == "3d_unaware":
        use_3d_aware = False
    else:
        raise ValueError(f"Invalid conditioning_mode: {conditioning_mode}")
    
    # Build output path
    output_filename = build_output_filename(bald_version, use_3d_aware)
    output_path = os.path.join(pair_dir, "flux2_processed", output_filename)
    
    if os.path.exists(output_path):
        print(f"Output image already exists for {sampled_dir}/{bald_version}, skipping...")
        return True
    
    # Handle copying existing 3D_UNAWARE result when forcing 3d_unaware mode
    if conditioning_mode == "3d_unaware" and not aligned_image_exists:
        unaware_filename = build_output_filename(bald_version, False)
        source_unaware_path = os.path.join(pair_dir, "flux2_processed", unaware_filename)
        if os.path.exists(source_unaware_path) and source_unaware_path != output_path:
            print(f"Copying existing 3D_UNAWARE result for {sampled_dir}/{bald_version}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            import shutil
            shutil.copy2(source_unaware_path, output_path)
            return True

    try:
        # Resolve source image path and resize_info
        outpainted_source_path = os.path.join(pair_dir, "source_outpainted", "outpainted_image.png")
        resize_info_path = os.path.join(pair_dir, "source_outpainted", "resize_info.json")
        
        if os.path.exists(outpainted_source_path):
            source_image_path = outpainted_source_path
            # Only use resize_info if it exists alongside the outpainted image
            if not os.path.exists(resize_info_path):
                resize_info_path = None
        else:
            source_image_path = os.path.join(bald_image_dir, f"{source_id}.png")
            resize_info_path = None  # No cropping needed for non-outpainted images
        
        target_image_path = os.path.join(image_dir, f"{target_id}.png")

        # Load images
        try:
            source_image = Image.open(source_image_path).convert("RGB").resize((resolution, resolution))
        except Exception as e:
            alternative_source_path = os.path.join(data_dir, "image", f"{source_id}.png")
            source_image_path = alternative_source_path
            source_image = Image.open(source_image_path).convert("RGB").resize((resolution, resolution))

        target_image = Image.open(target_image_path).convert("RGB").resize((resolution, resolution))
        target_image_matted, alpha_mask = background_remover.remove_background(target_image)
        # Convert target_image_matted to RGB with white background for non-foreground pixels
        target_image_matted_array = np.asarray(target_image_matted)
        alpha_mask_array = np.asarray(alpha_mask)
        if alpha_mask_array.ndim == 3:
            alpha_mask_array = alpha_mask_array[..., 0]
        foreground_mask = alpha_mask_array > 0
        white_bg_image = np.full_like(target_image_matted_array, 255)
        white_bg_image[foreground_mask, :] = target_image_matted_array[foreground_mask, :]
        target_image_matted = Image.fromarray(white_bg_image.astype(np.uint8))
        print(f"Source image path: {source_image_path}", flush=True)
        print(f"Target image path: {target_image_path}", flush=True)
        print(f"3D lifting applied: {lift_3d_applied}", flush=True)
        print(f"Resize info path: {resize_info_path}", flush=True)

        if use_3d_aware:
            return process_3d_aware(
                pipe, source_image, target_image_matted, view_aligned_image_path,
                background_remover, sam_mask_extractor, resolution, output_path,
                resize_info_path=resize_info_path
            )
        else:
            return process_3d_unaware(
                pipe, source_image, target_image_matted, resolution, background_remover, sam_mask_extractor, output_path,
                lift_3d_applied, aligned_image_exists,
                resize_info_path=resize_info_path
            )

    except Exception as e:
        # raise e
        print(f"Error processing {sampled_dir}: {e}")
        return False


def main(args):
    """Main entry point."""
    # Set HF token if provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    # Setup paths
    data_dir = args.data_dir
    image_dir = os.path.join(data_dir, "matted_image/")
    if not os.path.exists(image_dir):
        image_dir = os.path.join(data_dir, "matted_image/")

    view_aligned_dir = os.path.join(
        data_dir, "view_aligned", 
        f"shape_{args.shape_provider}__texture_{args.texture_provider}/"
    )

    # Load models
    print("Loading Flux 2 pipeline...")
    pipe = load_flux_pipeline(device=args.device)
    print("Pipeline loaded successfully!")

    print("Loading background remover...")
    background_remover = BackgroundRemover()
    
    print("Loading SAM mask extractor...")
    sam_mask_extractor = SAMMaskExtractor(
        detection_threshold=args.detection_threshold,
        confidence_threshold=args.confidence_threshold,
    )

    # Determine bald versions to process
    if args.bald_version == "all":
        bald_versions = ["w_seg", "wo_seg"]
    else:
        bald_versions = [args.bald_version]
    
    # Determine conditioning modes to process
    if args.conditioning_mode == "all":
        conditioning_modes = ["3d_aware", "3d_unaware"]
    else:
        conditioning_modes = [args.conditioning_mode]

    # Log configuration
    print(f"\nConfiguration:")
    print(f"  bald_version(s): {bald_versions}")
    print(f"  conditioning_mode(s): {conditioning_modes}")

    # Get all folders and shuffle
    all_folders = os.listdir(view_aligned_dir)
    shuffle_seed = int(time.time())
    random.seed(shuffle_seed)
    random.shuffle(all_folders)
    print(f"Found {len(all_folders)} samples (shuffle seed: {shuffle_seed})")

    # Process samples for each bald_version and conditioning_mode combination
    total_combinations = len(bald_versions) * len(conditioning_modes)
    overall_success = 0
    overall_total = 0
    
    for bald_version in bald_versions:
        bald_image_dir = os.path.join(data_dir, "bald", bald_version, "image/")
        
        for conditioning_mode in conditioning_modes:
            print(f"\n{'='*60}")
            print(f"Processing: bald_version={bald_version}, conditioning_mode={conditioning_mode}")
            print(f"{'='*60}")
            
            success_count = 0
            for sampled_dir in all_folders:
                if process_sample(
                    sampled_dir,
                    pipe,
                    background_remover,
                    sam_mask_extractor,
                    data_dir,
                    view_aligned_dir,
                    image_dir,
                    bald_image_dir,
                    resolution=args.resolution,
                    bald_version=bald_version,
                    conditioning_mode=conditioning_mode,
                ):
                    success_count += 1
            
            print(f"\n✓ {bald_version}/{conditioning_mode}: {success_count}/{len(all_folders)} samples processed")
            overall_success += success_count
            overall_total += len(all_folders)

    print(f"\n{'='*60}")
    print(f"✓ All processing complete! {overall_success}/{overall_total} total samples processed")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hair Transfer Restoration using Flux 2"
    )

    # Data paths
    parser.add_argument("--data_dir", type=str, default="/workspace/outputs",
                        help="Base data directory (default: /workspace/outputs)")
    parser.add_argument("--shape_provider", type=str, default="hi3dgen",
                        help="Shape provider name (default: hi3dgen)")
    parser.add_argument("--texture_provider", type=str, default="mvadapter",
                        help="Texture provider name (default: mvadapter)")

    # Model settings
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (default: cpu)")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token for model access")

    # Bald image settings
    parser.add_argument("--bald_version", type=str, default="w_seg",
                        choices=["w_seg", "wo_seg", "all"],
                        help="Bald version to use: w_seg, wo_seg, or all (default: w_seg)")

    # Conditioning mode settings
    parser.add_argument("--conditioning_mode", type=str, default="3d_aware",
                        choices=["3d_aware", "3d_unaware", "all"],
                        help="Conditioning mode: 3d_aware (use aligned image when available), "
                             "3d_unaware (never use aligned image), or all (default: 3d_aware)")

    # Processing parameters
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Image resolution (default: 1024)")
    parser.add_argument("--detection_threshold", type=float, default=0.3,
                        help="SAM detection threshold (default: 0.3)")
    parser.add_argument("--confidence_threshold", type=float, default=0.4,
                        help="SAM confidence threshold (default: 0.4)")

    args = parser.parse_args()
    
    main(args)
    

