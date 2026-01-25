import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from toolkit.pipeline_flux_inpaint import FluxInpaintPipeline
from PIL import Image
from tqdm import tqdm


torch.set_float32_matmul_precision('high')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "/workspace/HairPort/Hairdar/assets/checkpoints/BaldKonverter/bald_konvertor_w_seg_000004900.safetensors"
DEFAULT_PROMPT_FILE = "/workspace/HairPort/Hairdar/hairport/balder/prompt_w_seg.txt"
DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_FLUX_INPUT_SIZE = 1024
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_NUM_INFERENCE_STEPS = 35
DEFAULT_STRENGTH = 1.0
DEFAULT_SEED = 42


class BaldKonverterWithSeg:
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        base_model: str = DEFAULT_BASE_MODEL,
        device: str = "cuda"
    ):
        self.device = device
        self.pipe = FluxInpaintPipeline.from_pretrained(
            base_model, torch_dtype=torch.bfloat16
        )
        self.pipe.to(self.device)
        self.pipe.load_lora_weights(model_path)
        logger.info(f"Loaded BaldKonverter (with seg) model from {model_path}")
    
    def prepare_mask(self, image: Image.Image) -> torch.Tensor:
        mask = torch.zeros(1, image.size[1], image.size[0])
        if mask.shape[1] == mask.shape[2]:
            mask[:, mask.shape[1]//2:, mask.shape[2]//2:] = 1
        elif 2 * mask.shape[1] == mask.shape[2]:
            mask[:, :, mask.shape[2]//2:] = 1
        return mask
    
    def generate_bald_image(
        self,
        image: Image.Image,
        prompt: str,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        strength: float = DEFAULT_STRENGTH,
        seed: int = DEFAULT_SEED
    ) -> Image.Image:
        # Resize input to FLUX expected size
        img = image.convert('RGB').resize((DEFAULT_FLUX_INPUT_SIZE, DEFAULT_FLUX_INPUT_SIZE))
        mask = self.prepare_mask(img)
        
        # Generate inpainted image
        output = self.pipe(
            prompt=prompt,
            mask_image=mask,
            image=img,
            guidance_scale=guidance_scale,
            height=img.size[1],
            width=img.size[0],
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=torch.Generator("cpu").manual_seed(seed)
        ).images[0]
        
        print(f"FLUX output image size: {output.size}")
        bottom_right = output.crop((
            DEFAULT_FLUX_INPUT_SIZE // 2, 
            DEFAULT_FLUX_INPUT_SIZE // 2, 
            DEFAULT_FLUX_INPUT_SIZE, 
            DEFAULT_FLUX_INPUT_SIZE
        ))
        
        print(f"Cropped output size: {bottom_right.size}")
        
        return bottom_right


def _load_prompt(prompt_file: Path) -> str:
    with open(prompt_file, 'r', encoding='utf-8') as file:
        return file.readline().strip()


def _create_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_image_from_path(img_path: Path) -> Image.Image | None:
    try:
        return Image.open(img_path)
    except Exception as e:
        logger.error(f"Failed to load image {img_path}: {e}")
        return None


def _process_single_image(
    data_dir: Path,
    konverter: BaldKonverterWithSeg,
    image_path: Path,
    bald_image_path: Path | None,
    flame_seg_dir: Path | None,
    output_path: Path,
    prompt: str,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    strength: float = DEFAULT_STRENGTH,
    seed: int = DEFAULT_SEED
) -> None:
    image_file_name = image_path.stem
    flux_input_path =  data_dir / "balder_input/dataset" / f"{image_file_name}.png"
    flux_input_image = Image.open(flux_input_path)

    flux_input_image.save(flux_input_path, 'PNG')
    
    bald_output = konverter.generate_bald_image(
        image=flux_input_image,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        seed=seed
    )
    
    print(f"Final bald_output size before saving: {bald_output.size}")
    bald_output.save(output_path, 'PNG')
    print(f"Saved bald output to: {output_path}")


def process_directory(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    bald_input_dir: Optional[Path] = None,
    flame_seg_dir: Optional[Path] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    prompt_file: Path = Path(DEFAULT_PROMPT_FILE),
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    strength: float = DEFAULT_STRENGTH,
    seed: int = DEFAULT_SEED,
    mask_pipeline_device: str = "cuda"
) -> None:
    images_dir = data_dir / 'image'
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    
    if output_dir is None:
        output_dir = data_dir / 'bald' / 'w_seg' / 'image'

    if bald_input_dir is None:
        bald_input_dir = data_dir / 'bald' / 'wo_seg' / 'image'
    
    _create_directory(output_dir)
    
    # # Check for pairs.csv to filter samples
    pairs_csv_path = data_dir / "pairs.csv"
    source_ids = None
    if pairs_csv_path.exists():
        try:
            pairs_df = pd.read_csv(pairs_csv_path)
            if 'source_id' in pairs_df.columns:
                source_ids = list(set(pairs_df['source_id'].unique()))
                source_ids = [str(sid) for sid in source_ids]
                logger.info(f"📋 Found pairs.csv with {len(source_ids)} unique source_ids")
                logger.info(f"   Only processing samples listed in source_id column")
            else:
                logger.warning(f"⚠️  pairs.csv exists but has no 'source_id' column, processing all images")
        except Exception as e:
            logger.warning(f"⚠️  Error reading pairs.csv: {e}, processing all images")
    else:
        logger.info(f"ℹ️  No pairs.csv found at {pairs_csv_path}, processing all images")
    
    all_image_files = list(images_dir.glob('*.png'))
    
    # Filter by source_ids if pairs.csv was found
    if source_ids is not None:
        image_files = [f for f in all_image_files if f.stem in source_ids]
        logger.info(f"   Filtered: {len(image_files)}/{len(all_image_files)} images match source_ids")
    else:
        image_files = all_image_files
    
    random.seed()
    random.shuffle(image_files)
    
    if not image_files:
        logger.warning(f"No PNG images found in: {images_dir}" +
                      (" matching source_ids from pairs.csv" if source_ids is not None else ""))
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    prompt = _load_prompt(prompt_file)
    logger.info(f"Using prompt: {prompt}")
    
    mask_pipeline = None
    konverter = BaldKonverterWithSeg(model_path=model_path)
    
    # Filter out already processed images
    image_files = [
        img_path for img_path in image_files 
        if not (output_dir / f"{img_path.stem}.png").exists()
    ]
    
    if not image_files:
        logger.info("All images have already been processed. Nothing to do.")
        return
    
    logger.info(f"{len(image_files)} images remaining to process")
    
    for image_path in tqdm(image_files, desc="Generating bald images with segmentation", unit="image"):
        output_path = output_dir / f"{image_path.stem}.png"
        
        if output_path.exists():
            logger.debug(f"Skipping existing output: {output_path}")
            continue
        
        bald_image_path = None
        if bald_input_dir is not None and bald_input_dir.exists():
            bald_image_path = bald_input_dir / f"{image_path.stem}.png"
        try:
            _process_single_image(
                data_dir=data_dir,
                konverter=konverter,
                image_path=image_path,
                bald_image_path=bald_image_path,
                flame_seg_dir=flame_seg_dir,
                output_path=output_path,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=strength,
                seed=seed
            )
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {str(e)}")
    
    del mask_pipeline
    torch.cuda.empty_cache()
    
    logger.info(f"Completed processing. Outputs saved to: {output_dir}")


def main(
    data_dir: str,
    output_dir: Optional[str] = None,
    bald_input_dir: Optional[str] = None,
    flame_seg_dir: Optional[str] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    prompt_file: str = DEFAULT_PROMPT_FILE,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    strength: float = DEFAULT_STRENGTH,
    seed: int = DEFAULT_SEED
) -> None:
    logger.info("Starting BaldKonverter (with segmentation) processing pipeline")
    
    data_path = Path(data_dir)
    output_path = Path(output_dir) if output_dir else None
    bald_input_path = Path(bald_input_dir) if bald_input_dir else None
    flame_seg_path = Path(flame_seg_dir) if flame_seg_dir else None
    prompt_path = Path(prompt_file)
    
    process_directory(
        data_dir=data_path,
        output_dir=output_path,
        bald_input_dir=bald_input_path,
        flame_seg_dir=flame_seg_path,
        model_path=model_path,
        prompt_file=prompt_path,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        seed=seed
    )
    
    logger.info("Pipeline completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BaldKonverter with Segmentation: Generate bald versions using segmentation masks")
    parser.add_argument(
        '--data_dir', type=str, default="/workspace/celeba_subset",
        help='Path to the data directory containing image/ subfolder'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory for bald images (default: {data_dir}/bald/w_seg/image/)'
    )
    parser.add_argument(
        '--bald_input_dir', type=str, default=None,
        help='Input directory containing bald images from wo_seg (default: {data_dir}/bald/wo_seg/image/)'
    )
    parser.add_argument(
        '--flame_seg_dir', type=str, default=None,
        help='Directory containing flame segmentation subdirectories'
    )
    parser.add_argument(
        '--model_path', type=str, default=DEFAULT_MODEL_PATH,
        help='Path to the BaldKonverter model weights'
    )
    parser.add_argument(
        '--prompt_file', type=str, default=DEFAULT_PROMPT_FILE,
        help='Path to the prompt text file'
    )
    parser.add_argument(
        '--guidance_scale', type=float, default=DEFAULT_GUIDANCE_SCALE,
        help='Guidance scale for generation'
    )
    parser.add_argument(
        '--num_inference_steps', type=int, default=DEFAULT_NUM_INFERENCE_STEPS,
        help='Number of inference steps'
    )
    parser.add_argument(
        '--strength', type=float, default=DEFAULT_STRENGTH,
        help='Inpainting strength'
    )
    parser.add_argument(
        '--seed', type=int, default=DEFAULT_SEED,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        bald_input_dir=args.bald_input_dir,
        flame_seg_dir=args.flame_seg_dir,
        model_path=args.model_path,
        prompt_file=args.prompt_file,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        strength=args.strength,
        seed=args.seed
    )
