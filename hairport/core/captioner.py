"""Canonical multi-field image captioning pipeline (Qwen3-VL).

Consolidates ``hairport/captioner_pipeline.py`` and
``utils/captioner_pipeline.py`` into a single, well-typed implementation.

Usage::

    from hairport.core.captioner import CaptionerPipeline

    pipeline = CaptionerPipeline()
    caption = pipeline.caption_image("photo.jpg", prompt_type="hair")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Prompt library
# ------------------------------------------------------------------ #

CANONICAL_PROMPTS: Dict[str, str] = {
    # Overall scene
    "scene": (
        "Instruction: Provide a clear, concise description of the overall scene in this image. "
        "Describe what a viewer notices at a glance, including the main subject and setting. "
        "Use exactly one short sentence (no more than 25 words). "
        "Avoid technical camera terms and do not list separate attributes."
    ),
    # Subject description (with hair)
    "description": (
        "Instruction: Describe the main person in this image in one clear, objective sentence. "
        "Include gender (only if visually obvious), hair style and color, skin tone, one notable "
        "facial feature, and visible clothing. Mention the background only if it strongly affects "
        "how the person appears; otherwise omit it. Use neutral language and avoid emotions, "
        "artistic style, camera details, or speculation."
    ),
    # Subject description (WITHOUT hair)
    "description_no_hair": (
        "Instruction: Analyze this upper-body photograph. Output exactly one objective sentence "
        "using this template:\n"
        "\"High-quality photograph of a {gender} with {skin tone} skin and {facial feature}, "
        "wearing a {clothing type and color}, no background.\"\n\n"
        "Rules:\n"
        "1. STRICT ORDER: Follow template sequence exactly.\n"
        "2. VISIBLE TRAITS ONLY: Skin tone, facial features, clothing.\n"
        "3. NO HAIR REFERENCES: Do not mention hair in any form, including style, length, "
        "color, texture, coverage, shaved head, or baldness. Do not use synonyms or indirect "
        "references (e.g., \"closely cropped,\" \"receding,\" \"clean-shaven scalp\").\n"
        "4. OBJECTIVE LANGUAGE: Use concrete terms. Avoid subjective/emotional descriptions.\n"
        "5. NO EXTRAS: Omit actions, environment details, inferred emotions, or artistic "
        "interpretations.\n"
        "6. ONE SENTENCE ONLY: Output exactly one sentence; no lists, no additional sentences."
    ),
    # Hair only
    "hair_description": (
        "This is a near-frontal upper-body photograph of a person. In exactly one clear, "
        "information-dense sentence, describe only the hair with exhaustive visual specificity. "
        "Explicitly state the dominant hair color and any secondary tones, highlights, lowlights, "
        "or color gradients; clearly classify the texture using concrete terms (e.g., straight, "
        "loosely wavy, tightly wavy, curly, coiled, kinked, or mixed textures), and describe "
        "strand thickness and density if visually apparent. Specify the hair length relative to "
        "facial landmarks, neck, shoulders, or upper torso, and precisely describe the style or "
        "arrangement (e.g., loose, layered, slicked back, parted, tied, braided, cropped, or "
        "voluminous). Describe root volume, direction of growth, and how the strands emerge from "
        "the scalp and fall, drape, cluster, or rest against the head, neck, shoulders, or upper "
        "torso. Use concrete, compact language; reference only the head, neck, upper torso "
        "(including shoulders), and facial landmarks (such as forehead, temples, and ears) as "
        "spatial anchors; do not mention any other body parts or any non-hair elements, including "
        "facial features, clothing, accessories, or background."
    ),
    # Subject position in frame
    "position": (
        "Instruction: Describe where the main subject appears within the frame in one concise "
        "sentence. Refer to the subject's placement (centered, left, right, high, low) and "
        "cropping (for example, \"close-up on the face\", \"upper body cropped at the chest\"). "
        "Do not describe pose, style, lighting, or background."
    ),
    # Subject action/pose
    "action": (
        "Instruction: In one concise sentence, describe what the main subject is doing or how "
        "they are posed. Mention body orientation and gaze direction (for example, \"facing the "
        "camera\", \"turned slightly left\"). If the subject is still, describe their pose rather "
        "than saying they are doing nothing. Do not mention composition, style, or camera angle."
    ),
    # Artistic style
    "style": (
        "Instruction: In one concise phrase (not a full sentence), describe only the artistic "
        "style of the image (for example, \"realistic studio portrait\", \"flat graphic "
        "illustration\"). Do not mention lighting, camera angle, composition, or specific scene "
        "details."
    ),
    # Lighting
    "lighting": (
        "Instruction: In one concise sentence, describe the lighting of the scene: direction, "
        "quality (soft or hard), overall brightness, and any noticeable color tint (for example, "
        "\"soft, even front lighting with a slightly warm tone\"). Do not describe the subject's "
        "pose, style, or composition."
    ),
    # Background
    "background": (
        "Instruction: In one concise sentence, describe the background behind the subject. "
        "State whether it is plain, gradient, textured, blurred, or detailed, and briefly mention "
        "any major elements. If the background is transparent, removed, or pure solid color, say "
        "that directly."
    ),
    # Composition / framing
    "composition": (
        "Instruction: In one concise sentence, describe the composition and framing of the image. "
        "Mention shot type (for example, close-up, medium shot), amount of negative space, and "
        "whether the framing feels centered or off-center. Do not describe lighting, artistic "
        "style, or the subject's action."
    ),
    # Camera angle
    "camera_angle": (
        "Instruction: In one concise sentence, describe the camera angle and distance relative to "
        "the subject (for example, \"eye-level close-up\", \"slightly high angle looking down\"). "
        "Mention only angle and distance, not composition, lighting, or style."
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
        "'very shallow depth of field with strongly blurred background', 'moderate depth of field', "
        "or 'deep focus with most elements sharp'."
    ),
}

# Legacy aliases
CANONICAL_PROMPTS["general"] = CANONICAL_PROMPTS["description"]
CANONICAL_PROMPTS["general_no_hair"] = CANONICAL_PROMPTS["description_no_hair"]
CANONICAL_PROMPTS["hair"] = CANONICAL_PROMPTS["hair_description"]
CANONICAL_PROMPTS["subject_position"] = CANONICAL_PROMPTS["position"]
CANONICAL_PROMPTS["subject_action"] = CANONICAL_PROMPTS["action"]


# ------------------------------------------------------------------ #
# Pipeline
# ------------------------------------------------------------------ #


class CaptionerPipeline:
    """Multi-field image captioning using Qwen3-VL Instruct."""

    PROMPTS = CANONICAL_PROMPTS

    def __init__(
        self,
        model_name: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device_map: Union[str, dict] = "cuda",
        use_flash_attention: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        from hairport.config import get_config

        cfg = get_config()
        self.model_name = model_name or cfg.models.captioner
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        resolved_dtype = torch_dtype or dtype
        logger.info("Loading %s (dtype=%s, flash_attn=%s)", self.model_name, resolved_dtype, use_flash_attention)

        extra_kwargs: dict = {}
        if use_flash_attention:
            extra_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=resolved_dtype,
            device_map=device_map,
            **extra_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_messages(image_input, prompt_text: str) -> list:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_input},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        return text.strip()

    def _resolve_image(self, image) -> Union[str, Image.Image]:
        if isinstance(image, (str, Path)):
            return str(image)
        if isinstance(image, Image.Image):
            return image
        raise ValueError(f"Image must be a file path (str/Path) or PIL.Image, got {type(image)}")

    # ------------------------------------------------------------------ #
    # Single-image caption
    # ------------------------------------------------------------------ #

    @torch.no_grad()
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
    ) -> str:
        """Caption a single image."""
        image_input = self._resolve_image(image)
        if prompt is None:
            prompt = self.PROMPTS.get(prompt_type, self.PROMPTS["general"])

        messages = self._build_messages(image_input, prompt)
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.processor(
            chat_text, images=[image_input], return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None),
        )

        input_len = inputs["input_ids"].shape[1]
        trimmed = generated_ids[0][input_len:]
        full_output = self.processor.decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        if return_full_output:
            return full_output
        return self._extract_final_answer(full_output)

    # ------------------------------------------------------------------ #
    # Batch caption
    # ------------------------------------------------------------------ #

    @torch.no_grad()
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
    ) -> List[str]:
        """Caption a batch of images."""
        if not images:
            return []

        # Resolve prompts
        if prompt is None:
            base_prompt = self.PROMPTS.get(prompt_type, self.PROMPTS["general"])
            prompts = [base_prompt] * len(images)
        elif isinstance(prompt, str):
            prompts = [prompt] * len(images)
        else:
            prompts = list(prompt)
            if len(prompts) != len(images):
                raise ValueError(f"Expected {len(images)} prompts, got {len(prompts)}.")

        # Fallback: sequential
        if not use_true_batch:
            return [
                self.caption_image(
                    img,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    return_full_output=return_full_output,
                    prompt=p,
                )
                for img, p in zip(images, prompts)
            ]

        # True batch
        image_inputs = [self._resolve_image(img) for img in images]
        chat_texts = [
            self.processor.apply_chat_template(
                self._build_messages(img_in, p), tokenize=False, add_generation_prompt=True,
            )
            for img_in, p in zip(image_inputs, prompts)
        ]

        inputs = self.processor(
            chat_texts, images=image_inputs, padding=True, return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=getattr(getattr(self.processor, "tokenizer", None), "pad_token_id", None),
        )

        trimmed = [
            out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)
        ]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )

        if return_full_output:
            return decoded
        return [self._extract_final_answer(t) for t in decoded]

    def __repr__(self) -> str:
        return f"CaptionerPipeline(model='{self.model_name}', device='{self.device}')"


__all__ = ["CaptionerPipeline", "CANONICAL_PROMPTS"]
