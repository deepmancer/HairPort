import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class CaptionerPipeline:
    """Pipeline for multi-field image captioning using Qwen3-VL Instruct model.

    This pipeline can generate focused, concise descriptions for:
      - scene
      - subject.description
      - subject.hair_description
      - subject.position
      - subject.action
      - style
      - lighting
      - background
      - composition
      - camera_angle
    """

    # Canonical prompts for each logical field. All are designed to be
    # short, precise, and easy for the model to follow.
    CANONICAL_PROMPTS = {
        "scene": (
            "Instruction: Provide a clear, concise description of the overall scene in this image. "
            "Describe what a viewer notices at a glance, including the main subject and setting. "
            "Use exactly one short sentence (no more than 25 words). "
            "Avoid technical camera terms and do not list separate attributes."
        ),
        "description": (
            "Instruction: Describe the main person in this image in one clear, objective sentence. "
            "Include gender (only if visually obvious), hair style and color, skin tone, one notable facial feature, "
            "and visible clothing. Mention the background only if it strongly affects how the person appears; "
            "otherwise omit it. Use neutral language and avoid emotions, artistic style, camera details, or speculation."
        ),
        "hair_description": (
            "Instruction: Describe only the subject's hair in one concise sentence. "
            "Mention hair color (and any highlights), length, texture, parting, and overall style, "
            "and how the hair falls relative to the head, neck, and shoulders. "
            "Do NOT mention any other body parts, clothing, accessories, facial features, or background elements."
        ),
        "position": (
            "Instruction: Describe where the main subject appears within the frame in one concise sentence. "
            "Refer to the subject's placement (centered, left, right, high, low) and cropping (for example, "
            "\"close-up on the face\", \"upper body cropped at the chest\"). "
            "Do not describe pose, style, lighting, or background."
        ),
        "action": (
            "Instruction: In one concise sentence, describe what the main subject is doing or how they are posed. "
            "Mention body orientation and gaze direction (for example, \"facing the camera\", \"turned slightly left\"). "
            "If the subject is still, describe their pose rather than saying they are doing nothing. "
            "Do not mention composition, style, or camera angle."
        ),
        "style": (
            "Instruction: In one concise phrase (not a full sentence), describe only the artistic style of the image "
            "(for example, \"realistic studio portrait\", \"flat graphic illustration\"). "
            "Do not mention lighting, camera angle, composition, or specific scene details."
        ),
        "lighting": (
            "Instruction: In one concise sentence, describe the lighting of the scene: direction, quality "
            "(soft or hard), overall brightness, and any noticeable color tint "
            "(for example, \"soft, even front lighting with a slightly warm tone\"). "
            "Do not describe the subject's pose, style, or composition."
        ),
        "background": (
            "Instruction: In one concise sentence, describe the background behind the subject. "
            "State whether it is plain, gradient, textured, blurred, or detailed, and briefly mention any major elements. "
            "If the background is transparent, removed, or pure solid color, say that directly."
        ),
        "composition": (
            "Instruction: In one concise sentence, describe the composition and framing of the image. "
            "Mention shot type (for example, close-up, medium shot), amount of negative space, "
            "and whether the framing feels centered or off-center. "
            "Do not describe lighting, artistic style, or the subject's action."
        ),
        "camera_angle": (
            "Instruction: In one concise sentence, describe the camera angle and distance relative to the subject "
            "(for example, \"eye-level close-up\", \"slightly high angle looking down\"). "
            "Mention only angle and distance, not composition, lighting, or style."
        ),
    }

    # Expose prompts under both the new field names and the legacy names
    # ("general" and "hair") for backwards compatibility.
    PROMPTS = dict(CANONICAL_PROMPTS)
    PROMPTS["general"] = CANONICAL_PROMPTS["description"]
    PROMPTS["hair"] = CANONICAL_PROMPTS["hair_description"]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        dtype: torch.dtype = torch.bfloat16,
        device_map: Union[str, dict] = "cuda",
        use_flash_attention: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize the captioning pipeline.

        Args:
            model_name: Hugging Face model identifier.
            dtype: Default torch dtype to use when torch_dtype is not provided.
            device_map: Device mapping for the model (e.g. "cuda" or a HF device map).
            use_flash_attention: Whether to enable FlashAttention v2.
            torch_dtype: Optional explicit dtype for the model weights. If None,
                `dtype` is used.
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve dtype to use for model weights.
        if torch_dtype is None:
            torch_dtype = dtype

        print(f"Loading {model_name}...")
        print(f"Device (for inputs): {self.device}")
        print(f"Device map (for model): {device_map}")
        print(f"torch_dtype: {torch_dtype}")

        if use_flash_attention:
            # Use FlashAttention 2 when requested and supported.
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map=device_map,
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )

        self.processor = AutoProcessor.from_pretrained(model_name)

        # Set padding side to 'left' for decoder-only architecture
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        print("Model and processor loaded successfully!")

    # ------------------------------------------------------------------
    # Core generation utilities
    # ------------------------------------------------------------------
    def _build_messages(self, image_input, prompt_text: str):
        """Build a single-turn chat message with image + text prompt."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_input},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

    def _extract_final_answer(self, text: str) -> str:
        """Extract the final answer from the model output.

        For Instruct-style models, the output is direct without hidden
        reasoning tags, so we simply strip whitespace.
        """
        return text.strip()

    # ------------------------------------------------------------------
    # Public captioning APIs
    # ------------------------------------------------------------------
    def caption_image(
        self,
        image: Union[str, Path, Image.Image],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        return_full_output: bool = False,
        prompt_type: str = "general",
        prompt: Optional[str] = None,
        presence_penalty: float = 1.5,  # kept for API compatibility, not used
    ) -> str:
        """Caption a single image with a specific prompt type.

        Args:
            image: Path to an image file or a PIL.Image instance.
            max_new_tokens: Maximum number of tokens to generate.
            temperature, top_p, top_k, repetition_penalty, do_sample:
                Standard sampling parameters.
            return_full_output: If True, return the raw decoded text from
                the model. Otherwise, return a cleaned, concise answer.
            prompt_type: Key into `PROMPTS` (e.g. "scene", "description",
                "hair_description", etc.).
            prompt: Optional custom prompt string that overrides `prompt_type`.
            presence_penalty: Present for API compatibility; not used in this
                Hugging Face generation call.

        Returns:
            A caption string.
        """
        # Normalize image input
        if isinstance(image, (str, Path)):
            image_input = str(image)
        elif isinstance(image, Image.Image):
            image_input = image
        else:
            raise ValueError(
                f"Image must be a file path (str/Path) or PIL.Image, got {type(image)}"
            )

        # Select prompt
        if prompt is None:
            prompt = self.PROMPTS.get(prompt_type, self.PROMPTS["general"])

        # Build chat message and turn into model inputs
        messages = self._build_messages(image_input, prompt)
        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            chat_text,
            images=[image_input],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.pad_token_id
                if hasattr(self.processor, "tokenizer")
                else None,
            )

        # Trim the prompt tokens from the output
        input_ids = inputs["input_ids"][0]
        output_ids = generated_ids[0]
        generated_ids_trimmed = output_ids[len(input_ids) :]

        full_output = self.processor.decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if return_full_output:
            return full_output

        return self._extract_final_answer(full_output)

    def caption_images_batch(
        self,
        images: Sequence[Union[str, Path, Image.Image]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        return_full_output: bool = False,
        use_true_batch: bool = True,
        prompt_type: str = "general",
        prompt: Optional[Union[str, Sequence[str]]] = None,
        presence_penalty: float = 1.5,  # kept for API compatibility, not used
    ) -> List[str]:
        """Caption a batch of images.

        Args:
            images: Sequence of image paths or PIL.Image instances.
            max_new_tokens, temperature, top_p, top_k, repetition_penalty,
            do_sample: Sampling parameters as in `caption_image`.
            return_full_output: If True, returns raw decoded strings.
            use_true_batch: If True, encodes all images into a single
                padded batch. If False, falls back to per-image calls.
            prompt_type: Key into `PROMPTS` if `prompt` is None.
            prompt: Optional custom prompt (single string for all images or
                a sequence of strings, one per image).
            presence_penalty: Present for API compatibility; not used.

        Returns:
            A list of caption strings, one per image.
        """
        if not images:
            return []

        # Resolve prompts for the batch
        if prompt is None:
            base_prompt = self.PROMPTS.get(prompt_type, self.PROMPTS["general"])
            prompts = [base_prompt] * len(images)
        elif isinstance(prompt, str):
            prompts = [prompt] * len(images)
        else:
            # Assume sequence-like
            prompts = list(prompt)
            if len(prompts) != len(images):
                raise ValueError(
                    f"Expected {len(images)} prompts, got {len(prompts)}."
                )

        # Fallback: sequential per-image processing (simple, robust)
        if not use_true_batch:
            results: List[str] = []
            for img, p in zip(images, prompts):
                results.append(
                    self.caption_image(
                        image=img,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        do_sample=do_sample,
                        return_full_output=return_full_output,
                        prompt=p,
                    )
                )
            return results

        # TRUE batch path: encode all images and prompts together
        image_inputs = []
        chat_texts = []
        for img, p in zip(images, prompts):
            if isinstance(img, (str, Path)):
                image_input = str(img)
            elif isinstance(img, Image.Image):
                image_input = img
            else:
                raise ValueError(
                    f"Image must be a file path (str/Path) or PIL.Image, got {type(img)}"
                )
            image_inputs.append(image_input)

            messages = self._build_messages(image_input, p)
            chat_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            chat_texts.append(chat_text)

        inputs = self.processor(
            chat_texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.pad_token_id
                if hasattr(self.processor, "tokenizer")
                else None,
            )

        # Trim prompt tokens per example and decode
        generated_ids_trimmed = []
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids):
            generated_ids_trimmed.append(out_ids[len(in_ids) :])

        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        results: List[str] = []
        for text in output_texts:
            if return_full_output:
                results.append(text)
            else:
                results.append(self._extract_final_answer(text))

        return results

    def __repr__(self) -> str:
        return f"CaptionerPipeline(model='{self.model_name}', device='{self.device}')"


# ----------------------------------------------------------------------
# CLI: walk a directory of images and write JSON captions per image
# ----------------------------------------------------------------------
def main(
    input_dir: str = "/workspace/outputs_new/matted_image",
    output_dir: str = "/workspace/outputs_new/prompt_json",
    batch_size: int = 8,
) -> None:
    """Run the captioning pipeline over a directory of images.

    For each image, this will generate a JSON file named after the
    sample_id (the image stem) with the following structure:

    {
      "scene": "...",
      "subject": [
        {
          "description": "...",
          "hair_description": "...",
          "position": "...",
          "action": "..."
        }
      ],
      "style": "...",
      "lighting": "...",
      "background": "...",
      "composition": "...",
      "camera_angle": "..."
    }
    """
    from tqdm import tqdm

    pipeline = CaptionerPipeline()

    images_dir = Path(input_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Input images directory does not exist: {images_dir}")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Collect image files
    image_extensions = {".png", ".jpg", ".jpeg"}
    all_image_files = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        ]
    )

    if not all_image_files:
        print(f"No images found in directory: {images_dir}")
        return

    # Determine which images still need JSON captions
    images_to_process = []
    already_done = 0
    for img_path in all_image_files:
        sample_id = img_path.stem
        json_path = output_dir_path / f"{sample_id}.json"
        if json_path.exists():
            already_done += 1
        else:
            images_to_process.append(img_path)

    print(f"Found {len(all_image_files)} images total.")
    print(f"  Existing JSON caption files: {already_done}")
    print(f"  Images to process: {len(images_to_process)}")

    if not images_to_process:
        print("Nothing to do: all images already have JSON captions.")
        return

    # Fields we will generate for each image (mapped to prompt types)
    field_prompt_types = {
        "scene": "scene",
        "description": "description",            # subject[0].description
        "hair_description": "hair_description",  # subject[0].hair_description
        "position": "position",                  # subject[0].position
        "action": "action",                      # subject[0].action
        "style": "style",
        "lighting": "lighting",
        "background": "background",
        "composition": "composition",
        "camera_angle": "camera_angle",
    }

    print("\nGenerating captions...")
    for batch_start in tqdm(
        range(0, len(images_to_process), batch_size), desc="Batches"
    ):
        batch_files = images_to_process[batch_start : batch_start + batch_size]
        batch_paths = [str(p) for p in batch_files]
        sample_ids = [p.stem for p in batch_files]

        # Prepare per-image storage for field outputs
        batch_captions = {sid: {} for sid in sample_ids}

        # Generate captions for each field using batched calls
        for field_name, prompt_type in field_prompt_types.items():
            captions = pipeline.caption_images_batch(
                images=batch_paths,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.0,
                do_sample=True,
                return_full_output=False,
                use_true_batch=True,
                prompt_type=prompt_type,
            )
            for sid, caption in zip(sample_ids, captions):
                batch_captions[sid][field_name] = caption

        # Write JSON files per image in this batch
        for img_path in batch_files:
            sid = img_path.stem
            fields = batch_captions[sid]

            data = {
                "scene": fields["scene"],
                "subject": [
                    {
                        "description": fields["description"],
                        "hair_description": fields["hair_description"],
                        "position": fields["position"],
                        "action": fields["action"],
                    }
                ],
                "style": fields["style"],
                "lighting": fields["lighting"],
                "background": fields["background"],
                "composition": fields["composition"],
                "camera_angle": fields["camera_angle"],
            }

            json_path = output_dir_path / f"{sid}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    print("\nCompleted! JSON captions saved to:")
    print(f"  {output_dir_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-field Image Captioning Pipeline")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/workspace/outputs_new/matted_image",
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/workspace/outputs_new/prompt_json",
        help="Directory to save per-image JSON captions",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of images to process per model batch",
    )
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir, batch_size=args.batch_size)
