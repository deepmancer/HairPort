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

DEFAULT_MODEL_PATH = "/workspace/HairPort/Hairdar/assets/checkpoints/BaldKonverter/bald_konvertor_wo_seg_000003400.safetensors"
DEFAULT_PROMPT_FILE = "/workspace/HairPort/Hairdar/hairport/balder/prompt_wo_seg.txt"
DEFAULT_BASE_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
DEFAULT_IMAGE_SIZE = 768
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_NUM_INFERENCE_STEPS = 35
DEFAULT_STRENGTH = 1.0
DEFAULT_SEED = 42


class BaldKonverter:
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
        logger.info(f"Loaded BaldKonverter model from {model_path}")
    
    def prepare_image_and_mask(self, image: Image.Image, target_size: int = DEFAULT_IMAGE_SIZE):
        img = image.convert('RGB').resize((target_size, target_size))
        w, h = img.size
        
        combined = Image.new("RGB", (w * 2, h))
        combined.paste(img, (0, 0))
        combined.paste(img, (w, 0))
        
        mask = torch.zeros(1, combined.size[1], combined.size[0])
        if mask.shape[1] == mask.shape[2]:
            mask[:, mask.shape[1]//2:, mask.shape[2]//2:] = 1
        elif 2 * mask.shape[1] == mask.shape[2]:
            mask[:, :, mask.shape[2]//2:] = 1
        
        return combined, mask
    
    def generate_bald_image(
        self,
        image: Image.Image,
        prompt: str,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        strength: float = DEFAULT_STRENGTH,
        seed: int = DEFAULT_SEED
    ) -> Image.Image:
        combined_img, mask = self.prepare_image_and_mask(image)
        
        output = self.pipe(
            prompt=prompt,
            mask_image=mask,
            image=combined_img,
            guidance_scale=guidance_scale,
            height=combined_img.size[1],
            width=combined_img.size[0],
            num_inference_steps=num_inference_steps,
            strength=strength,
            generator=torch.Generator("cpu").manual_seed(seed)
        ).images[0]
        
        right_image = output.crop((768, 0, 1536, 768))
        
        return right_image


def _load_prompt(prompt_file: Path) -> str:
    with open(prompt_file, 'r', encoding='utf-8') as file:
        return file.readline().strip()


def _create_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _process_single_image(
    konverter: BaldKonverter,
    image_path: Path,
    output_path: Path,
    flux_input_path: Path,
    prompt: str,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    strength: float = DEFAULT_STRENGTH,
    seed: int = DEFAULT_SEED
) -> None:
    image = Image.open(image_path)
    
    combined_img, _ = konverter.prepare_image_and_mask(image)
    combined_img.save(flux_input_path, 'PNG')
    
    bald_image = konverter.generate_bald_image(
        image=image,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        seed=seed
    )
    
    bald_image.save(output_path, 'PNG')


def process_directory(
    data_dir: Path,
    output_dir: Optional[Path] = None,
    flux_input_dir: Optional[Path] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    prompt_file: Path = Path(DEFAULT_PROMPT_FILE),
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    strength: float = DEFAULT_STRENGTH,
    seed: int = DEFAULT_SEED
) -> None:
    images_dir = data_dir / 'image'
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    
    if output_dir is None:
        output_dir = data_dir / 'bald' / 'wo_seg' / 'image'
    
    if flux_input_dir is None:
        flux_input_dir = data_dir / 'bald' / 'wo_seg' / 'flux_input'
    
    _create_directory(output_dir)
    _create_directory(flux_input_dir)
    
    # Check for pairs.csv to filter samples
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
    # # Filter by source_ids if pairs.csv was found
    if source_ids is not None:
        image_files = []
        for f in all_image_files:
            # print(f.stem)
            if f.stem in source_ids:
                image_files.append(f)
        image_files = [f for f in all_image_files if f.stem in source_ids]
        logger.info(f"   Filtered: {len(image_files)}/{len(all_image_files)} images match source_ids")
    else:
        image_files = all_image_files
    import time
    random.seed(int(time.time()))
    random.shuffle(image_files)
    
    if not image_files:
        logger.warning(f"No PNG images found in: {images_dir}" +
                      (" matching source_ids from pairs.csv" if source_ids is not None else ""))
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    prompt = _load_prompt(prompt_file)
    logger.info(f"Using prompt: {prompt}")
    
    konverter = BaldKonverter(model_path=model_path)
    
    for image_path in tqdm(image_files, desc="Generating bald images", unit="image"):
        output_path = output_dir / f"{image_path.stem}.png"
        flux_input_path = flux_input_dir / f"{image_path.stem}.png"
        
        if output_path.exists() and flux_input_path.exists():
            logger.debug(f"Skipping existing output: {output_path}")
            continue
        
        try:
            _process_single_image(
                konverter=konverter,
                image_path=image_path,
                output_path=output_path,
                flux_input_path=flux_input_path,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=strength,
                seed=seed
            )
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {str(e)}")
    
    logger.info(f"Completed processing. Output saved to: {output_dir}")
    logger.info(f"FLUX inputs saved to: {flux_input_dir}")


def main(
    data_dir: str,
    output_dir: Optional[str] = None,
    flux_input_dir: Optional[str] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    prompt_file: str = DEFAULT_PROMPT_FILE,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    strength: float = DEFAULT_STRENGTH,
    seed: int = DEFAULT_SEED
) -> None:
    logger.info("Starting BaldKonverter processing pipeline")
    
    data_path = Path(data_dir)
    output_path = Path(output_dir) if output_dir else None
    flux_input_path = Path(flux_input_dir) if flux_input_dir else None
    prompt_path = Path(prompt_file)
    
    process_directory(
        data_dir=data_path,
        output_dir=output_path,
        flux_input_dir=flux_input_path,
        model_path=model_path,
        prompt_file=prompt_path,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        strength=strength,
        seed=seed
    )
    
    logger.info("Pipeline completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BaldKonverter: Generate bald versions of images")
    parser.add_argument(
        '--data_dir', type=str, default="/workspace/celeba_reduced",
        help='Path to the data directory containing image/ subfolder'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory (default: {data_dir}/bald/wo_seg/image/)'
    )
    parser.add_argument(
        '--flux_input_dir', type=str, default=None,
        help='Output directory for FLUX input images (default: {data_dir}/bald/wo_seg/flux_input/)'
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
        flux_input_dir=args.flux_input_dir,
        model_path=args.model_path,
        prompt_file=args.prompt_file,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        strength=args.strength,
        seed=args.seed
    )
