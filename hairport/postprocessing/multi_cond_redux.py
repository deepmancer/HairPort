from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import FluxPriorReduxPipeline
from diffusers.pipelines.flux.pipeline_flux_prior_redux import FluxPriorReduxPipelineOutput, PipelineImageInput
from PIL import ImageChops

@dataclass
class FluxPriorReduxPipelineOutputWithDebug:
    prompt_embeds: torch.FloatTensor
    pooled_prompt_embeds: torch.FloatTensor
    # Optional debug artifacts (ComfyUI node outputs processed image + mask for inspection)
    processed_images: Optional[List[Image.Image]] = None
    processed_masks: Optional[List[Image.Image]] = None


class FluxMultiPriorReduxPipeline(FluxPriorReduxPipeline):
    """
    Multi-reference, ComfyUI_AdvancedRefluxControl-aligned Redux prior pipeline.

    Adds:
      - Multi-image fusion (weighted average by default) without multiplying text prompt.
      - Comfy-aligned 'mode' preprocessing:
          * "center crop (square)"
          * "keep aspect ratio"
          * "autocrop with mask"
      - Token-space masking (mask resized to patch grid and applied to Redux tokens).
      - Token downsample->upsample "strength" control via downsampling_factor/function.
      - Redux token scaling by weight**2 (per reference workflow semantics).

    Intended output is compatible with FluxPipeline(**pipe_prior_output).
    """

    # -------------------------
    # Public API
    # -------------------------
    @torch.no_grad()
    def __call__(
        self,
        image: Union["PipelineImageInput", List["PipelineImageInput"]],
        prompt: Union[str, List[str], None] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        # Advanced Reflux Control-style knobs
        downsampling_factor: float = 1.0,
        downsampling_function: str = "area",
        mode: str = "center crop (square)",
        weight: Union[float, List[float]] = 1.0,
        mask: Optional[Union["PipelineImageInput", List["PipelineImageInput"]]] = None,
        autocrop_margin: float = 0.1,
        # Multi-reference fusion controls
        fuse_mode: str = "weighted_average",  # "weighted_average" or "sum"
        prior_weights: Optional[List[float]] = None,  # additional per-image weights (optional)
        # Output controls
        return_debug: bool = False,
        return_dict: bool = True,
    ):
        """
        Args:
            image:
                Single reference image or list of reference images (multi-image conditioning).
            mask:
                Optional single mask or list of masks aligned 1:1 with `image` list.
                Mask values expected in [0,1] (white = apply conditioning).
            downsampling_factor:
                1 (strongest) ... 9 (weakest-ish). Implemented as token-grid low-pass:
                downsample->upsample by this factor.
            downsampling_function:
                One of: "area", "bilinear", "bicubic", "nearest", "nearest_exact".
            mode:
                "center crop (square)" | "keep aspect ratio" | "autocrop with mask"
            weight:
                Redux token scale; applied as weight**2 (per reference workflow).
                Can be scalar or per-image list.
            autocrop_margin:
                Only used when mode == "autocrop with mask". Margin is relative to full image size.
            fuse_mode:
                How to combine multiple references. "weighted_average" is recommended.
            prior_weights:
                Optional extra per-image multipliers (e.g., confidence per reference).
            return_debug:
                If True, returns processed (cropped/padded) images and masks for inspection.

        Returns:
            FluxPriorReduxPipelineOutput-compatible output (prompt_embeds, pooled_prompt_embeds),
            optionally with debug images/masks.
        """

        device = self._execution_device

        # ---- Normalize inputs to lists (multi-reference)
        images = image if isinstance(image, list) else [image]
        num_refs = len(images)

        if mask is None:
            masks = [None] * num_refs
        else:
            masks = mask if isinstance(mask, list) else [mask]
            if len(masks) != num_refs:
                raise ValueError(f"`mask` must be None, a single mask, or a list of {num_refs} masks.")

        if isinstance(weight, (int, float)):
            weights = [float(weight)] * num_refs
        else:
            if len(weight) != num_refs:
                raise ValueError(f"`weight` must be a float or a list of {num_refs} floats.")
            weights = [float(w) for w in weight]

        if prior_weights is None:
            prior_weights = [1.0] * num_refs
        else:
            if len(prior_weights) != num_refs:
                raise ValueError(f"`prior_weights` must be None or a list of {num_refs} floats.")
            prior_weights = [float(w) for w in prior_weights]

        # ---- Preprocess per reference (Comfy-aligned)
        processed_images: List[Image.Image] = []
        processed_masks: List[Optional[Image.Image]] = []

        for img_i, m_i in zip(images, masks):
            pil_img = self._to_pil_rgb(img_i)
            pil_mask = None if m_i is None else self._to_pil_mask(m_i)

            pil_img, pil_mask = self._preprocess_mode(
                pil_img=pil_img,
                pil_mask=pil_mask,
                mode=mode,
                autocrop_margin=autocrop_margin,
            )

            processed_images.append(pil_img)
            processed_masks.append(pil_mask)

        # ---- Encode images via SIGLIP -> ReduxImageEncoder
        # We encode in a batch for efficiency, but we do NOT treat them as separate prompts.
        image_latents = self.encode_image(processed_images, device=device, num_images_per_prompt=1)
        image_embeds = self.image_embedder(image_latents).image_embeds.to(device=device)
        # image_embeds: (B=num_refs, T_img, D)

        # ---- Apply mask in token space (if provided)
        # Resize masks to token grid and multiply.
        image_embeds = self._apply_token_masks(image_embeds, processed_masks, device=device)

        # ---- Apply token downsampling strength control (low-pass)
        image_embeds = self._token_lowpass(
            image_embeds=image_embeds,
            downsampling_factor=downsampling_factor,
            downsampling_function=downsampling_function,
        )

        # ---- Apply weight semantics: scale Redux tokens by weight**2 (per repo)
        w = torch.tensor([wi ** 2 for wi in weights], device=device, dtype=image_embeds.dtype)  # (B,)
        pw = torch.tensor(prior_weights, device=device, dtype=image_embeds.dtype)               # (B,)
        per_ref_scale = (w * pw).view(num_refs, 1, 1)
        image_embeds = image_embeds * per_ref_scale

        # ---- Fuse multiple references into ONE Redux token block
        if fuse_mode not in {"weighted_average", "sum"}:
            raise ValueError("`fuse_mode` must be 'weighted_average' or 'sum'.")

        if fuse_mode == "sum":
            fused_image_embeds = image_embeds.sum(dim=0, keepdim=True)  # (1, T_img, D)
        else:
            denom = (w * pw).sum().clamp_min(torch.tensor(1e-8, device=device, dtype=image_embeds.dtype))
            fused_image_embeds = image_embeds.sum(dim=0, keepdim=True) / denom  # (1, T_img, D)

        # ---- Encode text ONCE (avoid multi-ref text amplification)
        batch_size_out = 1
        if hasattr(self, "text_encoder") and self.text_encoder is not None:
            prompt_list = prompt
            prompt2_list = prompt_2

            # If user provides prompt as list, we take the first (typical usage is a single prompt).
            if isinstance(prompt_list, list):
                prompt_list = prompt_list[0]
            if isinstance(prompt2_list, list):
                prompt2_list = prompt2_list[0]

            (text_prompt_embeds, pooled_prompt_embeds, _) = self.encode_prompt(
                prompt=prompt_list,
                prompt_2=prompt2_list,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=512,
                lora_scale=None,
            )
        else:
            # text encoders not loaded; mirror base behavior (dummy embeddings)
            if prompt is not None:
                # Keeping warning behavior aligned to base
                import warnings

                warnings.warn(
                    "prompt input is ignored when text encoders are not loaded to the pipeline. "
                    "Load text encoders to enable prompt conditioning."
                )
            # max_sequence_length is 512, t5 encoder hidden size is 4096
            text_prompt_embeds = torch.zeros((batch_size_out, 512, 4096), device=device, dtype=fused_image_embeds.dtype)
            pooled_prompt_embeds = torch.zeros((batch_size_out, 768), device=device, dtype=fused_image_embeds.dtype)

        # ---- Concatenate text + fused Redux tokens
        prompt_embeds_out = torch.cat([text_prompt_embeds, fused_image_embeds], dim=1)

        # Offload all models (base pipeline behavior)
        self.maybe_free_model_hooks()

        if not return_dict:
            return (prompt_embeds_out, pooled_prompt_embeds)

        if return_debug:
            # Convert masks None->None, else keep as PIL for inspection
            debug_masks = [m for m in processed_masks]
            return FluxPriorReduxPipelineOutputWithDebug(
                prompt_embeds=prompt_embeds_out,
                pooled_prompt_embeds=pooled_prompt_embeds,
                processed_images=processed_images,
                processed_masks=debug_masks,
            )

        return FluxPriorReduxPipelineOutput(prompt_embeds=prompt_embeds_out, pooled_prompt_embeds=pooled_prompt_embeds)

    # -------------------------
    # Helpers: image/mask conversion
    # -------------------------
    def _to_pil_rgb(self, x) -> Image.Image:
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        if isinstance(x, np.ndarray):
            arr = x
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        if torch.is_tensor(x):
            t = x.detach().cpu()
            if t.ndim == 4:
                t = t[0]
            if t.shape[0] in (1, 3):  # (C,H,W)
                t = t.permute(1, 2, 0)
            t = t.float()
            t = t.clamp(0, 1)
            arr = (t.numpy() * 255.0).astype(np.uint8)
            return Image.fromarray(arr).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(x)}")

    def _to_pil_mask(self, x) -> Image.Image:
        # Produces single-channel L mask in [0..255]
        if isinstance(x, Image.Image):
            return x.convert("L")
        if isinstance(x, np.ndarray):
            arr = x
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                arr = arr[..., 0]
            if arr.dtype != np.uint8:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            return Image.fromarray(arr).convert("L")
        if torch.is_tensor(x):
            t = x.detach().cpu()
            if t.ndim == 4:
                t = t[0]
            if t.ndim == 3 and t.shape[0] in (1, 3):
                t = t[0]  # take first channel
            t = t.float().clamp(0, 1)
            arr = (t.numpy() * 255.0).astype(np.uint8)
            return Image.fromarray(arr).convert("L")
        raise TypeError(f"Unsupported mask type: {type(x)}")

    # -------------------------
    # Helpers: Comfy-aligned preprocessing modes
    # -------------------------
    def _preprocess_mode(
        self,
        pil_img: Image.Image,
        pil_mask: Optional[Image.Image],
        mode: str,
        autocrop_margin: float,
    ) -> Tuple[Image.Image, Optional[Image.Image]]:
        mode = mode.strip().lower()

        if mode == "center crop (square)":
            pil_img = self._center_crop_square(pil_img)
            if pil_mask is not None:
                pil_mask = self._center_crop_square(pil_mask)
            return pil_img, pil_mask

        if mode == "keep aspect ratio":
            pil_img, pad_mask = self._pad_to_square(pil_img)
            if pil_mask is None:
                pil_mask = pad_mask
            else:
                pil_mask, _ = self._pad_to_square(pil_mask)
                pil_mask = ImageChops.multiply(pil_mask, pad_mask)
            return pil_img, pil_mask

        if mode == "autocrop with mask":
            if pil_mask is None:
                raise ValueError("mode='autocrop with mask' requires a mask.")
            pil_img, pil_mask = self._autocrop_with_mask(pil_img, pil_mask, autocrop_margin)
            # After autocrop, the repo behavior is still to provide square input to Redux
            pil_img, pad_mask = self._pad_to_square(pil_img)
            pil_mask, _ = self._pad_to_square(pil_mask)
            pil_mask = ImageChops.multiply(pil_mask, pad_mask)
            return pil_img, pil_mask

        raise ValueError(
            "Unsupported `mode`. Use one of: 'center crop (square)', 'keep aspect ratio', 'autocrop with mask'."
        )

    def _center_crop_square(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        return img.crop((left, top, left + s, top + s))

    def _pad_to_square(self, img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        # Pads with black to square; returns (padded_image, padding_exclusion_mask)
        w, h = img.size
        s = max(w, h)
        new_img = Image.new(img.mode, (s, s), color=0)
        left = (s - w) // 2
        top = (s - h) // 2
        new_img.paste(img, (left, top))

        # Mask: 255 in original region, 0 in padding
        pad_mask = Image.new("L", (s, s), color=0)
        region = Image.new("L", (w, h), color=255)
        pad_mask.paste(region, (left, top))
        return new_img, pad_mask

    def _autocrop_with_mask(
        self, img: Image.Image, mask: Image.Image, autocrop_margin: float
    ) -> Tuple[Image.Image, Image.Image]:
        w, h = img.size
        m = np.array(mask, dtype=np.uint8)
        ys, xs = np.where(m > 0)
        if len(xs) == 0 or len(ys) == 0:
            # No mask content; fallback to center-crop square
            return self._center_crop_square(img), self._center_crop_square(mask)

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        # Margin is relative to total image size on each side (per repo description)
        mx = int(round(autocrop_margin * w))
        my = int(round(autocrop_margin * h))

        x0 = max(0, x0 - mx)
        x1 = min(w - 1, x1 + mx)
        y0 = max(0, y0 - my)
        y1 = min(h - 1, y1 + my)

        # PIL crop box is (left, top, right, bottom) with right/bottom exclusive
        crop_box = (int(x0), int(y0), int(x1) + 1, int(y1) + 1)
        return img.crop(crop_box), mask.crop(crop_box)

    # -------------------------
    # Helpers: Token masking + token downsampling strength
    # -------------------------
    def _apply_token_masks(
        self,
        image_embeds: torch.FloatTensor,                # (B, T, D)
        masks: List[Optional[Image.Image]],
        device: torch.device,
    ) -> torch.FloatTensor:
        if all(m is None for m in masks):
            return image_embeds

        B, T, D = image_embeds.shape
        side = int(round(T ** 0.5))
        if side * side != T:
            # If tokens aren't a square grid, fallback: scalar mask per token = 1 (no-op)
            return image_embeds

        # Build mask tensor (B, 1, side, side)
        mask_tensors = []
        for m in masks:
            if m is None:
                mask_tensors.append(torch.ones((1, side, side), device=device, dtype=image_embeds.dtype))
                continue
            arr = np.array(m.convert("L"), dtype=np.float32) / 255.0  # (H,W) in [0,1]
            t = torch.from_numpy(arr).to(device=device, dtype=image_embeds.dtype)[None, None]  # (1,1,H,W)
            # Resize to token grid
            t = F.interpolate(t, size=(side, side), mode="bilinear", align_corners=False)
            mask_tensors.append(t[0])  # (1, side, side)

        token_mask = torch.stack(mask_tensors, dim=0)  # (B, 1, side, side)
        token_mask = token_mask.view(B, 1, T).transpose(1, 2)  # (B, T, 1)

        return image_embeds * token_mask

    def _token_lowpass(
        self,
        image_embeds: torch.FloatTensor,  # (B, T, D)
        downsampling_factor: float,
        downsampling_function: str,
    ) -> torch.FloatTensor:
        if downsampling_factor is None or downsampling_factor <= 1.0:
            return image_embeds

        B, T, D = image_embeds.shape
        side = int(round(T ** 0.5))
        if side * side != T:
            # Can't interpret as a 2D patch grid; no-op fallback
            return image_embeds

        x = image_embeds.view(B, side, side, D).permute(0, 3, 1, 2)  # (B, D, H, W)

        # Compute downsampled spatial size
        ds = max(1, int(round(side / float(downsampling_factor))))

        mode = downsampling_function.strip().lower()
        if mode == "nearest_exact":
            interp_mode = "nearest-exact" if "nearest-exact" in F.interpolate.__doc__ else "nearest"
        elif mode in {"nearest", "bilinear", "bicubic", "area"}:
            interp_mode = mode
        else:
            raise ValueError("Invalid downsampling_function. Use nearest, nearest_exact, bilinear, bicubic, or area.")

        # Downsample
        if interp_mode in {"bilinear", "bicubic"}:
            x_ds = F.interpolate(x, size=(ds, ds), mode=interp_mode, align_corners=False)
        else:
            x_ds = F.interpolate(x, size=(ds, ds), mode=interp_mode)

        # Upsample back (area is not valid for upsampling; use bilinear for reconstruction)
        up_mode = interp_mode if interp_mode in {"nearest", "nearest-exact", "bilinear", "bicubic"} else "bilinear"
        if up_mode in {"bilinear", "bicubic"}:
            x_up = F.interpolate(x_ds, size=(side, side), mode=up_mode, align_corners=False)
        else:
            x_up = F.interpolate(x_ds, size=(side, side), mode=up_mode)

        y = x_up.permute(0, 2, 3, 1).contiguous().view(B, T, D)
        return y
