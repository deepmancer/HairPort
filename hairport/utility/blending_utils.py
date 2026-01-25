from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

def create_soft_blend_mask(
    hair_mask: np.ndarray,
    dilation_kernel: int = 15,
    dilation_iterations: int = 2,
    blur_radius: int = 21
) -> np.ndarray:
    """Create a soft blend mask with smooth transitions."""
    # Ensure binary mask
    mask = (hair_mask > 0.5).astype(np.uint8)
    
    # Dilate to create transition zone
    kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=dilation_iterations)
    
    # Create soft edges with Gaussian blur
    soft_mask = cv2.GaussianBlur(dilated.astype(np.float32), (blur_radius, blur_radius), 0)
    
    # Ensure original hair region stays at 1.0
    soft_mask = np.maximum(soft_mask, mask.astype(np.float32))
    
    return soft_mask


def create_distance_soft_blend_mask(
    hair_mask: np.ndarray,
    *,
    dilation_px: int = 0,
    dilation_iterations: int = 1,
    feather_px: int = 12,
    mask_threshold: float = 0.5,
) -> np.ndarray:
    """Create a distance-based soft blend mask."""
    mask = (hair_mask > mask_threshold).astype(np.uint8)
    if int(mask.sum()) == 0:
        return np.zeros_like(mask, dtype=np.float32)

    if dilation_px > 0 and dilation_iterations > 0:
        k = 2 * int(dilation_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=int(dilation_iterations))

    feather_px = int(max(0, feather_px))
    if feather_px == 0:
        return mask.astype(np.float32)

    # Distance in *outside* region to the nearest hair pixel.
    # OpenCV's distanceTransform returns distance to the nearest zero pixel.
    outside = (1 - mask).astype(np.uint8)
    dist_outside = cv2.distanceTransform(outside, distanceType=cv2.DIST_L2, maskSize=5).astype(np.float32)

    # One-sided feather: inside hair is always 1; outside decays to 0 over `feather_px`.
    outside_alpha = np.clip(1.0 - (dist_outside / float(feather_px)), 0.0, 1.0)
    soft = np.where(mask > 0, 1.0, outside_alpha)
    return soft.astype(np.float32)


def create_hierarchical_blend_mask(
    hair_mask: np.ndarray,
    num_levels: int = 4
) -> List[np.ndarray]:
    """Create multi-scale blend masks for hierarchical blending."""
    masks = []
    current_mask = (hair_mask > 0.5).astype(np.float32)
    
    for level in range(num_levels):
        # Progressively more blur at each level
        blur_size = 5 + level * 8  # 5, 13, 21, 29
        if blur_size % 2 == 0:
            blur_size += 1
        
        # Dilation increases with level
        kernel_size = 3 + level * 4  # 3, 7, 11, 15
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(current_mask.astype(np.uint8), kernel, iterations=1)
        
        # Apply blur
        blurred = cv2.GaussianBlur(dilated.astype(np.float32), (blur_size, blur_size), 0)
        masks.append(blurred)
    
    return masks


def poisson_blend(
    source: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    center: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Apply Poisson blending for natural composition."""
    # Ensure proper types
    source = source.astype(np.uint8)
    target = target.astype(np.uint8)
    mask_255 = (mask * 255).astype(np.uint8)
    
    # Find center if not provided
    if center is None:
        bbox = get_bbox_from_mask(mask)
        y1, y2, x1, x2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
    
    try:
        # NORMAL_CLONE preserves texture better than MIXED_CLONE for hair
        result = cv2.seamlessClone(source, target, mask_255, center, cv2.NORMAL_CLONE)
        return result
    except cv2.error:
        # Fallback to alpha blending if Poisson fails
        alpha = mask[:, :, np.newaxis] if mask.ndim == 2 else mask
        return (source * alpha + target * (1 - alpha)).astype(np.uint8)


def multi_scale_blend(
    generated: np.ndarray,
    source: np.ndarray,
    masks: List[np.ndarray],
    use_laplacian: bool = True
) -> np.ndarray:
    """Multi-scale blending using Laplacian pyramids."""
    if not use_laplacian or len(masks) < 2:
        # Simple alpha blend with first mask
        alpha = masks[0][:, :, np.newaxis]
        return (generated * alpha + source * (1 - alpha)).astype(np.uint8)
    
    num_levels = len(masks)
    
    # Build Laplacian pyramids
    def build_laplacian_pyramid(img, levels):
        pyramid = []
        current = img.astype(np.float32)
        for i in range(levels - 1):
            down = cv2.pyrDown(current)
            up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))
            laplacian = current - up
            pyramid.append(laplacian)
            current = down
        pyramid.append(current)
        return pyramid
    
    # Build pyramids
    gen_pyr = build_laplacian_pyramid(generated.astype(np.float32), num_levels)
    src_pyr = build_laplacian_pyramid(source.astype(np.float32), num_levels)
    
    # Blend at each level
    blended_pyr = []
    for i in range(num_levels):
        h, w = gen_pyr[i].shape[:2]
        mask_resized = cv2.resize(masks[i], (w, h))
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]
        blended = gen_pyr[i] * mask_resized + src_pyr[i] * (1 - mask_resized)
        blended_pyr.append(blended)
    
    # Reconstruct from pyramid
    result = blended_pyr[-1]
    for i in range(num_levels - 2, -1, -1):
        result = cv2.pyrUp(result, dstsize=(blended_pyr[i].shape[1], blended_pyr[i].shape[0]))
        result = result + blended_pyr[i]
    
    return np.clip(result, 0, 255).astype(np.uint8)


