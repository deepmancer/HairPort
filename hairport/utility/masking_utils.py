import numpy as np
import cv2
import math

def f(r, T=0.6, beta=0.1):
    return np.where(r < T, beta + (1 - beta) / T * r, 1)


def f_hair(r, T=0.6, beta=0.1, min_output=0.4):
    """Hair-aware version of f() that enforces a minimum output.
    
    For hair restoration, we need more context even when hair fills most of the image.
    This function ensures the output never exceeds min_output, guaranteeing adequate
    expansion for the generation model.
    
    Args:
        r: Ratio of bbox area to total image area
        T: Threshold below which linear scaling applies
        beta: Minimum value at r=0
        min_output: Maximum output value (lower = more expansion). Default 0.4 means
                    at least sqrt(0.4/r) expansion ratio even for large masks.
    
    Returns:
        Scaling factor, capped at min_output to ensure adequate expansion.
    """
    base = np.where(r < T, beta + (1 - beta) / T * r, 1)
    return np.minimum(base, min_output)


def compute_expansion_ratio(mask, min_ratio=1.3, max_ratio=2.5):
    """Compute an appropriate expansion ratio based on mask coverage.
    
    When mask fills most of the image, returns a larger ratio to ensure
    adequate context for generation.
    
    Args:
        mask: Binary mask array
        min_ratio: Minimum expansion ratio (used for small masks)
        max_ratio: Maximum expansion ratio (used when mask fills image)
    
    Returns:
        Expansion ratio between min_ratio and max_ratio
    """
    H, W = mask.shape[:2]
    bbox = get_bbox_from_mask(mask)
    y1, y2, x1, x2 = bbox
    
    bbox_area = (y2 - y1 + 1) * (x2 - x1 + 1)
    image_area = H * W
    coverage = bbox_area / image_area
    
    # Linear interpolation: more coverage -> higher ratio needed
    # coverage=0 -> min_ratio, coverage=1 -> max_ratio
    ratio = min_ratio + (max_ratio - min_ratio) * coverage
    
    return ratio, coverage


def expand_bbox_for_hair(mask, yyxx, min_ratio=1.4, coverage_threshold=0.5):
    """Expand bounding box with hair-aware logic.
    
    Unlike expand_bbox which reduces expansion for large masks, this function
    increases expansion when hair fills more of the image, ensuring adequate
    context for high-fidelity hair generation.
    
    Args:
        mask: Binary mask array
        yyxx: Bounding box as (y1, y2, x1, x2)
        min_ratio: Minimum expansion ratio
        coverage_threshold: If mask covers more than this fraction, use larger expansion
    
    Returns:
        Tuple of (expanded_bbox, needs_outpaint):
            - expanded_bbox: (y1, y2, x1, x2)
            - needs_outpaint: True if expanded bbox exceeds image bounds significantly
    """
    y1, y2, x1, x2 = yyxx
    H, W = mask.shape[:2]
    
    # Compute coverage
    bbox_area = (y2 - y1 + 1) * (x2 - x1 + 1)
    coverage = bbox_area / (H * W)
    
    # Determine expansion ratio based on coverage
    # Higher coverage = need more expansion for context
    if coverage > coverage_threshold:
        # Scale up ratio as coverage increases
        # At coverage=0.5: ratio=min_ratio, at coverage=1.0: ratio=min_ratio*1.8
        ratio = min_ratio * (1.0 + 0.8 * (coverage - coverage_threshold) / (1.0 - coverage_threshold))
    else:
        # Use standard f() logic for smaller masks
        r2 = f(coverage)
        ratio = math.sqrt(r2 / coverage) if coverage > 0 else min_ratio
        ratio = max(ratio, min_ratio)
    
    # Compute new bbox
    xc = 0.5 * (x1 + x2)
    yc = 0.5 * (y1 + y2)
    h = ratio * (y2 - y1 + 1)
    w = ratio * (x2 - x1 + 1)
    
    new_x1 = int(xc - w * 0.5)
    new_x2 = int(xc + w * 0.5)
    new_y1 = int(yc - h * 0.5)
    new_y2 = int(yc + h * 0.5)
    
    # Check if we need outpainting (bbox significantly exceeds image bounds)
    overflow_x = max(0, -new_x1) + max(0, new_x2 - W)
    overflow_y = max(0, -new_y1) + max(0, new_y2 - H)
    total_overflow = (overflow_x * h + overflow_y * w) / (h * w)
    needs_outpaint = total_overflow > 0.2  # More than 20% overflow
    
    # Clamp to image bounds
    new_x1 = max(0, new_x1)
    new_x2 = min(W, new_x2)
    new_y1 = max(0, new_y1)
    new_y2 = min(H, new_y2)
    
    return (new_y1, new_y2, new_x1, new_x2), needs_outpaint

# Get the bounding box of the mask
def get_bbox_from_mask(mask):
    h,w = mask.shape[0],mask.shape[1]

    if mask.sum() < 10:
        return 0,h,0,w
    rows = np.any(mask,axis=1)
    cols = np.any(mask,axis=0)
    y1,y2 = np.where(rows)[0][[0,-1]]
    x1,x2 = np.where(cols)[0][[0,-1]]
    return (y1,y2,x1,x2)

# Expand the bounding box
def expand_bbox(mask, yyxx, ratio, min_crop=0):
    y1,y2,x1,x2 = yyxx
    H,W = mask.shape[0], mask.shape[1]

    yyxx_area = (y2-y1+1) * (x2-x1+1)
    r1 = yyxx_area / (H * W)
    r2 = f(r1)
    ratio = math.sqrt(r2 / r1)

    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2-y1+1)
    w = ratio * (x2-x1+1)
    h = max(h,min_crop)
    w = max(w,min_crop)

    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)

# Pad the image to a square shape
def pad_to_square(image, pad_value = 255, random = False):
    H,W = image.shape[0], image.shape[1]
    if H == W:
        return image

    padd = abs(H - W)
    if random:
        padd_1 = int(np.random.randint(0,padd))
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1

    if len(image.shape) == 2: 
        if H > W:
            pad_param = ((0, 0), (padd_1, padd_2))
        else:
            pad_param = ((padd_1, padd_2), (0, 0))
    elif len(image.shape) == 3: 
        if H > W:
            pad_param = ((0, 0), (padd_1, padd_2), (0, 0))
        else:
            pad_param = ((padd_1, padd_2), (0, 0), (0, 0))

    image = np.pad(image, pad_param, 'constant', constant_values=pad_value)

    return image

# Expand the image and mask
def expand_image_mask(image, mask, ratio=1.4):
    h,w = image.shape[0], image.shape[1]
    H,W = int(h * ratio), int(w * ratio) 
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W -w) // 2)
    w2 = W -w - w1

    pad_param_image = ((h1,h2),(w1,w2),(0,0))
    pad_param_mask = ((h1,h2),(w1,w2))
    image = np.pad(image, pad_param_image, 'constant', constant_values=255)
    mask = np.pad(mask, pad_param_mask, 'constant', constant_values=0)
    return image, mask

# Convert the bounding box to a square shape
def box2squre(image, box):
    H,W = image.shape[0], image.shape[1]
    y1,y2,x1,x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h,w = y2-y1, x2-x1

    if h >= w:
        x1 = cx - h//2
        x2 = cx + h//2
    else:
        y1 = cy - w//2
        y2 = cy + w//2
    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)

# Crop the predicted image back to the original image
def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop, margin=0):
    """Paste predicted crop back into original image.
    
    Args:
        pred: Predicted/edited crop (H_pred x W_pred x 3)
        tar_image: Original target image to paste into (will be modified)
        extra_sizes: [H1, W1, H2, W2] where H1,W1 = crop size before padding, H2,W2 = after padding
        tar_box_yyxx_crop: [y1, y2, x1, x2] bounding box in tar_image coordinates
        margin: Pixel margin to leave at edges (default 0 for seamless paste)
    """
    H1, W1, H2, W2 = extra_sizes
    y1, y2, x1, x2 = tar_box_yyxx_crop
    pred = cv2.resize(pred, (W2, H2))
    m = margin

    if W1 == H1:
        # Crop was already square, no padding was added
        if m != 0:
            tar_image[y1+m:y2-m, x1+m:x2-m, :] = pred[m:-m, m:-m]
        else:
            tar_image[y1:y2, x1:x2, :] = pred
        return tar_image

    # Remove padding that was added by pad_to_square
    if W1 < W2:
        # Width was padded (original was taller than wide)
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:, pad1:-pad2, :]
    else:
        # Height was padded (original was wider than tall)
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1:-pad2, :, :]

    gen_image = tar_image.copy()
    if m != 0:
        gen_image[y1+m:y2-m, x1+m:x2-m, :] = pred[m:-m, m:-m]
    else:
        gen_image[y1:y2, x1:x2, :] = pred

    return gen_image

