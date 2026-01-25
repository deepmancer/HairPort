import argparse
import json
import os
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class CaptionerPipeline:
    """Pipeline for image captioning using Qwen3-VL Instruct model."""

    PROMPTS = {
        # ---------- Existing prompts ----------
        # "hair": (
        #     "Produce a single, tightly written sentence that describes only the hairstyle as a complete physical "
        #     "form. Specify color(s), length, texture, volume, and overall shape, then define how the hair originates at the "
        #     "scalp, how it is parted or directed, how volume is distributed, and how strands or sections behave "
        #     "around the front hairline, sides, crown, and back of the head. Clearly state where the hair ends in "
        #     "relation to the neck and shoulders so the hairstyle can be reconstructed from any viewing angle. "
        #     "Use only the head, neck, and shoulders as spatial references, and exclude all non-hair details."
        # ),

        "hair": (
            "This is a near-frontal upper-body photograph of a person. In exactly one clear, information-dense sentence, "
            "describe only the hair with exhaustive visual specificity. Explicitly state the dominant hair color and any "
            "secondary tones, highlights, lowlights, or color gradients; clearly classify the texture using concrete terms "
            "(e.g., straight, loosely wavy, tightly wavy, curly, coiled, kinked, or mixed textures), and describe strand "
            "thickness and density if visually apparent. Specify the hair length relative to facial landmarks, neck, "
            "shoulders, or upper torso, and precisely describe the style or arrangement (e.g., loose, layered, slicked "
            "back, parted, tied, braided, cropped, or voluminous). Describe root volume, direction of growth, and how the "
            "strands emerge from the scalp and fall, drape, cluster, or rest against the head, neck, shoulders, or upper "
            "torso. Use concrete, compact language; reference only the head, neck, upper torso (including shoulders), and "
            "facial landmarks (such as forehead, temples, and ears) as spatial anchors; do not mention any other body parts "
            "or any non-hair elements, including facial features, clothing, accessories, or background."
        ),
        "general": (
            "Instruction: Analyze this upper-body photograph. Output exactly one objective sentence using "
            "this template:\n"
            "\"High-quality photograph of a {gender} with {hair descriptors}, {skin tone} skin, and "
            "{facial feature}, wearing a {clothing type and color}, no background.\"\n\n"
            "Rules:\n"
            "1. STRICT ORDER: Follow template sequence exactly.\n"
            "2. VISIBLE TRAITS ONLY: Hair, skin tone, facial features, clothing.\n"
            "3. OBJECTIVE LANGUAGE: Use concrete terms. Avoid subjective/emotional descriptions.\n"
            "4. NO EXTRAS: Omit actions, environment details, inferred emotions, or artistic interpretations."
        ),
        
        "general_no_hair": (
            "Instruction: Analyze this upper-body photograph. Output exactly one objective sentence using "
            "this template:\n"
            "\"High-quality photograph of a {gender} with {skin tone} skin and {facial feature}, wearing a "
            "{clothing type and color}, no background.\"\n\n"
            "Rules:\n"
            "1. STRICT ORDER: Follow template sequence exactly.\n"
            "2. VISIBLE TRAITS ONLY: Skin tone, facial features, clothing.\n"
            "3. NO HAIR REFERENCES: Do not mention hair in any form, including style, length, color, texture, "
            "coverage, shaved head, or baldness. Do not use synonyms or indirect references (e.g., "
            "\"closely cropped,\" \"receding,\" \"clean-shaven scalp\").\n"
            "4. OBJECTIVE LANGUAGE: Use concrete terms. Avoid subjective/emotional descriptions.\n"
            "5. NO EXTRAS: Omit actions, environment details, inferred emotions, or artistic interpretations.\n"
            "6. ONE SENTENCE ONLY: Output exactly one sentence; no lists, no additional sentences."
        ),
        # ---------- New prompts for structured fields ----------

        # Overall scene
        "scene": (
            "In one concise sentence, objectively describe the overall scene shown in the image, "
            "mentioning the setting and the main visible elements without referring to camera settings "
            "or emotions."
        ),

        # Subject-related fields (besides the existing 'general' and 'hair' prompts)
        "subject_position": (
            "In a short phrase, describe where the main subject appears in the frame "
            "(for example, 'centered', 'left third', 'right third', or 'slightly below center')."
        ),
        "subject_action": (
            "In one concise sentence, describe what the main subject is doing or how they are posed, "
            "using only clearly visible actions or gestures."
        ),

        # Style
        "style": (
            "Instruction: Classify the image style. Output exactly one word from this allowed set:\n"
            "\"realistic\", \"cartoon\", \"anime\".\n\n"
            "Rules:\n"
            "1. ONE TOKEN OUTPUT: Return only the single chosen word, lowercase, with no punctuation.\n"
            "2. NO EXPLANATION: Do not add reasoning, qualifiers, or additional text.\n"
            "3. CHOOSE BEST MATCH: Pick the closest category based on the overall visual rendering style."
        ),

        # Lighting
        "lighting": (
            "In one concise sentence, describe the lighting in the image, including its direction, "
            "softness or hardness, overall brightness, and any noticeable color tint "
            "(for example, 'soft neutral front lighting' or 'strong warm side light')."
        ),

        # Background
        "background": (
            "In one concise sentence, describe the background of the image, including its type "
            "(plain, studio, indoor, outdoor) and any notable elements or patterns, and how visually "
            "simple or busy it appears."
        ),

        # Composition
        "composition": (
            "In one concise sentence, describe the composition and framing of the image, such as "
            "crop distance (close-up, medium shot), framing of the subject, and any noticeable "
            "symmetry or leading lines."
        ),

        # Camera angle
        "camera_angle": (
            "In a short phrase, describe the camera angle relative to the subject "
            "(for example, 'eye-level', 'slightly above', 'low angle', or 'three-quarter view')."
        ),

        # Camera lens appearance
        "camera_lens": (
            "In a short phrase, describe the apparent lens look based only on the image, such as "
            "'wide-angle', 'standard', 'telephoto', or an approximate focal length like 'around 35mm' "
            "or 'around 85mm portrait'. Do not mention camera brands or explicit settings."
        ),

        # Depth of field
        "camera_depth_of_field": (
            "In a short phrase, describe the depth of field visible in the image, such as "
            "'very shallow depth of field with strongly blurred background', "
            "'moderate depth of field', or 'deep focus with most elements sharp'."
        ),
    }

    def __init__(
        self,
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        dtype=torch.bfloat16,
        device_map="cuda",
        use_flash_attention=True,  # Enable by default for H100
        torch_dtype=None,
    ):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {model_name}...")
        print(f"Device: {self.device}")

        if use_flash_attention:
            if torch_dtype is None:
                torch_dtype = torch.bfloat16
            print("Using flash_attention_2 for improved performance")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype="auto", device_map="auto",
                # dtype=torch_dtype,
                # use_flash_attention_2=True,
                attn_implementation="flash_attention_2",
                # device_map=device_map,
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                # dtype=dtype,
                # device_map=device_map,
                dtype="auto", device_map="auto",
            )

        self.processor = AutoProcessor.from_pretrained(model_name)

        # Set padding side to 'left' for decoder-only architecture
        self.processor.tokenizer.padding_side = "left"

        print("Model loaded successfully!")

    def _resize_image(self, image, target_size: int = 512) -> Image.Image:
        """Resize image to target_size x target_size for efficient processing."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        if not isinstance(image, Image.Image):
            raise ValueError(
                f"Image must be a file path (str/Path) or PIL Image, got {type(image)}"
            )
        
        image = image.convert("RGB")
        
        # Resize to target_size x target_size
        if image.size != (target_size, target_size):
            image = image.resize((target_size, target_size), Image.LANCZOS)
        
        return image

    @torch.no_grad()
    def caption_image(
        self,
        image,
        max_new_tokens=16384,  # Half of original 32768
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
        do_sample=True,
        return_full_output=False,
        prompt_type="general",
        prompt=None,
        presence_penalty=1.5,
    ):
        # Resize image to 512x512 for efficient processing
        image_input = self._resize_image(image, target_size=512)

        if prompt is None:
            prompt = self.PROMPTS.get(prompt_type, self.PROMPTS["general"])

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_input},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Two-step processing: get formatted text, then use processor for tokenization
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=[image_input],
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                # greedy=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        full_output = output_text[0] if output_text else ""

        if return_full_output:
            return full_output

        return self._extract_final_answer(full_output)

    def _extract_final_answer(self, text):
        """
        Extract the final answer from the model output.
        For Instruct models, the output is direct without thinking process.
        """
        # Simply return the stripped text as Instruct models don't have thinking tags
        return text.strip()

    @torch.no_grad()
    def caption_images_batch(
        self,
        images,
        max_new_tokens=16384,  # Half of original 32768
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
        do_sample=True,
        return_full_output=False,
        use_true_batch=True,
        prompt_type="general",
        prompt=None,  # Changed default from string to None
        presence_penalty=1.5,
    ):
        if prompt is None:  # Fixed: was "if prompt is not None"
            prompt = self.PROMPTS.get(prompt_type, self.PROMPTS["general"])

        if isinstance(prompt, str):
            prompts = [prompt] * len(images)
        elif isinstance(prompt, list):
            prompts = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        if len(images) != len(prompts):
            raise ValueError(
                f"Number of images ({len(images)}) must match number of prompts ({len(prompts)})"
            )

        if not use_true_batch:
            captions = []
            for image, single_prompt in zip(images, prompts):
                caption = self.caption_image(
                    image,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    return_full_output=return_full_output,
                    prompt=single_prompt,
                )
                captions.append(caption)
            return captions

        # Resize all images to 512x512 for efficient processing
        image_inputs = [self._resize_image(image, target_size=512) for image in images]

        # Process all images in TRUE batches (not one-by-one)
        # Step 1: Use apply_chat_template with tokenize=False to get formatted text strings
        # Step 2: Use processor.__call__ to properly batch texts and images together
        batch_texts = []
        for image_input, prompt_text in zip(image_inputs, prompts):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_input},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            # Get formatted text for this conversation (tokenize=False returns string)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_texts.append(text)

        # Process in proper batches with padding
        all_captions = []

        # Batch process: pass all texts and images to processor together
        # The processor will handle tokenization and proper batching with padding
        inputs = self.processor(
            text=batch_texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,  # Enable padding for true batching
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for output_text in output_texts:
            if return_full_output:
                all_captions.append(output_text)
            else:
                all_captions.append(self._extract_final_answer(output_text))

        return all_captions

    def __repr__(self):
        return f"CaptionerPipeline(model='{self.model_name}', device='{self.device}')"


def main(
    input_dir: str = "/workspace/outputs_new/matted_image",
    output_dir: str = "/workspace/outputs_new/prompt/",
    modes: Union[str, list] = "all",
) -> None:
    """
    For each image in `input_dir`, generate a structured JSON file in `output_dir` with the form:

    {
      "scene": "overall scene description",
      "subject": [
        {
          "description": "detailed subject description (from 'general' prompt)",
          "hair_description": "detailed description of the hair of the subject (from 'hair' prompt)",
          "position": "where in frame",
          "action": "what they're doing"
        }
      ],
      "style": "artistic style",
      "lighting": "lighting description",
      "background": "background details",
      "composition": "framing and layout",
      "camera": {
        "angle": "camera angle",
        "lens": "apparent lens description",
        "depth_of_field": "depth of field description"
      }
    }

    The JSON filename is derived from the image filename (same base name, .json extension).
    """
    from tqdm import tqdm
    import random
    import time

    pipeline = CaptionerPipeline()

    # Batch size for generation
    batch_size = 8

    # Determine which modes to generate
    all_modes = [
        "scene", "general", "general_no_hair", "hair", "subject_position", "subject_action",
        "style", "lighting", "background", "composition",
        "camera_angle", "camera_lens", "camera_depth_of_field"
    ]
    if modes == "all":
        active_modes = set(all_modes)
    elif isinstance(modes, str):
        active_modes = {modes}
    else:
        active_modes = set(modes)

    print(f"Active modes: {active_modes}")

    images_dir = Path(input_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Input images directory does not exist: {images_dir}")

    os.makedirs(output_dir, exist_ok=True)

    all_image_files = sorted(
        [f for f in os.listdir(images_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
    )

    images_to_process = []
    processed = 0

    for img_file in all_image_files:
        json_file = os.path.join(
            output_dir,
            img_file.replace(".png", ".json")
            .replace(".jpg", ".json")
            .replace(".jpeg", ".json"),
        )
        if os.path.exists(json_file):
            processed += 1
        else:
            images_to_process.append(img_file)

    random.seed(time.time())
    random.shuffle(images_to_process)
    total_images = len(all_image_files)

    print(f"Total images: {total_images}")
    print(f"Already processed (JSON): {processed}/{total_images}")
    print(f"Images needing processing: {len(images_to_process)}")

    if len(images_to_process) == 0:
        print("All images have already been processed!")
        return

    print(f"\nProcessing {len(images_to_process)} images in batches of {batch_size}...")

    for i in tqdm(range(0, len(images_to_process), batch_size), desc="Generating JSON captions"):
        batch_files = images_to_process[i : i + batch_size]
        
        # Filter out samples that already have prompt files (re-check in case of concurrent processing)
        batch_files = [
            f for f in batch_files
            if not os.path.exists(
                os.path.join(
                    output_dir,
                    f.replace(".png", ".json").replace(".jpg", ".json").replace(".jpeg", ".json"),
                )
            )
        ]
        
        # Skip if all samples in this batch already have prompts
        if not batch_files:
            continue
        
        batch_paths = [os.path.join(images_dir, f) for f in batch_files]

        # Generate captions for each structured field.
        # We reuse the existing 'general' and 'hair' prompts for the subject's description
        # and hair_description, and use new concise prompts for the other fields.
        # Only generate captions for active modes.

        empty_batch = [""] * len(batch_files)

        scenes = pipeline.caption_images_batch(batch_paths, prompt_type="scene") if "scene" in active_modes else empty_batch
        subject_descriptions = pipeline.caption_images_batch(batch_paths, prompt_type="general") if "general" in active_modes else empty_batch
        subject_descriptions_no_hair = pipeline.caption_images_batch(batch_paths, prompt_type="general_no_hair") if "general_no_hair" in active_modes else empty_batch
        hair_descriptions = pipeline.caption_images_batch(batch_paths, prompt_type="hair") if "hair" in active_modes else empty_batch
        positions = pipeline.caption_images_batch(batch_paths, prompt_type="subject_position") if "subject_position" in active_modes else empty_batch
        actions = pipeline.caption_images_batch(batch_paths, prompt_type="subject_action") if "subject_action" in active_modes else empty_batch
        styles = pipeline.caption_images_batch(batch_paths, prompt_type="style") if "style" in active_modes else empty_batch
        lightings = pipeline.caption_images_batch(batch_paths, prompt_type="lighting") if "lighting" in active_modes else empty_batch
        backgrounds = pipeline.caption_images_batch(batch_paths, prompt_type="background") if "background" in active_modes else empty_batch
        compositions = pipeline.caption_images_batch(batch_paths, prompt_type="composition") if "composition" in active_modes else empty_batch
        camera_angles = pipeline.caption_images_batch(batch_paths, prompt_type="camera_angle") if "camera_angle" in active_modes else empty_batch
        camera_lenses = pipeline.caption_images_batch(batch_paths, prompt_type="camera_lens") if "camera_lens" in active_modes else empty_batch
        camera_dofs = pipeline.caption_images_batch(batch_paths, prompt_type="camera_depth_of_field") if "camera_depth_of_field" in active_modes else empty_batch

        for idx, image_file in enumerate(batch_files):
            data = {
                "scene": scenes[idx],
                "subject": [
                    {
                        "description": subject_descriptions[idx],
                        "description_no_hair": subject_descriptions_no_hair[idx],
                        "hair_description": hair_descriptions[idx],
                        "position": positions[idx],
                        "action": actions[idx],
                    }
                ],
                "style": styles[idx],
                "lighting": lightings[idx],
                "background": backgrounds[idx],
                "composition": compositions[idx],
                "camera": {
                    "angle": camera_angles[idx],
                    "lens": camera_lenses[idx],
                    "depth_of_field": camera_dofs[idx],
                },
            }

            json_file = os.path.join(
                output_dir,
                image_file.replace(".png", ".json")
                .replace(".jpg", ".json")
                .replace(".jpeg", ".json"),
            )
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nCompleted! JSON captions saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning Pipeline")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/workspace/outputs/",
        help="Dataset directory",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="*",
        default="all",
        help="List of prompt types to generate (e.g., 'hair', 'general hair', or 'all' for all types). "
             "Available modes: scene, general, hair, subject_position, subject_action, style, "
             "lighting, background, composition, camera_angle, camera_lens, camera_depth_of_field",
    )
    args = parser.parse_args()
    
    input_dir = os.path.join(args.data_dir, "image/")
    output_dir = os.path.join(args.data_dir, "prompt/")
   
    main(
        input_dir=input_dir,
        output_dir=output_dir,
        modes=["hair", "general", "general_no_hair", "style", "lighting", "background"],
    )
