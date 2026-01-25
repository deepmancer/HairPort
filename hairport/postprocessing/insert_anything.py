import argparse
import gc
import json
import math
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ================================
# Third-Party Libraries
# ================================
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from tqdm import tqdm

# ================================
# Diffusers Imports
# ================================
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline

from utils.sam_mask_extractor import SAMMaskExtractor
from utils.bg_remover import BackgroundRemover
from hairport.utility.uncrop_sdxl import ImageUncropper
from hairport.postprocessing.mask_helpers import (
    get_bbox_from_mask, expand_bbox, pad_to_square, box2squre, crop_back, expand_image_mask,
)


@dataclass
class HairTransferConfig:
    """Configuration for hair transfer pipeline."""
    dtype: torch.dtype = torch.float16
    resolution: Tuple[int, int] = (768, 768)  # Width, Height
    # Model settings
    FLUX_FILL_MODEL: str = "black-forest-labs/FLUX.1-Fill-dev"
    FLUX_REDUX_MODEL: str = "black-forest-labs/FLUX.1-Redux-dev"
    LORA_WEIGHTS_PATH: str = "/workspace/HairPort/Hairdar/flux_kontext_blending.safetensors"
    

pipe = FluxFillPipeline.from_pretrained(
    HairTransferConfig.FLUX_FILL_MODEL,
    torch_dtype=HairTransferConfig.dtype
).to("cuda")


pipe.load_lora_weights(
    HairTransferConfig.LORA_WEIGHTS_PATH,
)


redux = FluxPriorReduxPipeline.from_pretrained(
    HairTransferConfig.FLUX_REDUX_MODEL,
).to(dtype=HairTransferConfig.dtype).to("cuda")


def hair_transfer(    
    tar_image,
    tar_mask,
    ref_image,
    ref_mask,
    seed=42,
    size=(768,768),
):
    tar_image = np.asarray(tar_image)
    tar_mask = np.asarray(tar_mask)
    tar_mask = np.where(tar_mask > 128, 1, 0).astype(np.uint8)

    ref_image = np.asarray(ref_image)
    ref_mask = np.asarray(ref_mask)
    ref_mask = np.where(ref_mask > 128, 1, 0).astype(np.uint8)

    if tar_mask.sum() == 0:
        raise gr.Error('No mask for the background image.Please check mask button!')

    if ref_mask.sum() == 0:
        raise gr.Error('No mask for the reference image.Please check mask button!')

    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) 
    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2] 
    ratio = 1.3
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)


    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False) 

    kernel = np.ones((7, 7), np.uint8)
    iterations = 2
    tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)

    # zome in
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=1.2)

    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=2)    #1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop


    old_tar_image = tar_image.copy()
    tar_image = tar_image[y1:y2,x1:x2,:]
    tar_mask = tar_mask[y1:y2,x1:x2]

    H1, W1 = tar_image.shape[0], tar_image.shape[1]
    # zome in

    tar_mask = pad_to_square(tar_mask, pad_value=0)
    tar_mask = cv2.resize(tar_mask, size)

    masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), size).astype(np.uint8)
    pipe_prior_output = redux(Image.fromarray(masked_ref_image))


    tar_image = pad_to_square(tar_image, pad_value=255)

    H2, W2 = tar_image.shape[0], tar_image.shape[1]

    tar_image = cv2.resize(tar_image, size)
    diptych_ref_tar = np.concatenate([masked_ref_image, tar_image], axis=1)


    tar_mask = np.stack([tar_mask,tar_mask,tar_mask],-1)
    mask_black = np.ones_like(tar_image) * 0
    mask_diptych = np.concatenate([mask_black, tar_mask], axis=1)

    diptych_ref_tar = Image.fromarray(diptych_ref_tar)
    mask_diptych[mask_diptych == 1] = 255
    mask_diptych = Image.fromarray(mask_diptych)



    generator = torch.Generator("cuda").manual_seed(seed)
    edited_image = pipe(
        image=diptych_ref_tar,
        mask_image=mask_diptych,
        height=mask_diptych.size[1],
        width=mask_diptych.size[0],
        max_sequence_length=512,
        generator=generator,
        **pipe_prior_output, 
    ).images[0]



    width, height = edited_image.size
    left = width // 2
    right = width
    top = 0
    bottom = height
    edited_image = edited_image.crop((left, top, right, bottom))


    edited_image = np.array(edited_image)
    edited_image = crop_back(edited_image, old_tar_image, np.array([H1, W1, H2, W2]), np.array(tar_box_yyxx_crop)) 
    edited_image = Image.fromarray(edited_image)
