import numpy as np

def stretch_depth_by_median_gamma(depth: np.ndarray, mask: np.ndarray = None, depth_out_value: float = -1) -> np.ndarray:
    """
    Contrast-stretch `depth` under `mask>0` so that:
      - the minimum hair depth → 0
      - the maximum hair depth → 1
      - the median hair depth → 0.5
    Non-masked regions are set to `depth_out_value`.
    """

    norm = depth.copy()
    
    if mask is None:
        valid_pixels = norm
    else:
        valid_pixels = norm[mask > 0]
    
    # Normalize valid pixels to 0-1 range first
    min_val = np.min(valid_pixels)
    max_val = np.max(valid_pixels)
    if max_val > min_val:
        valid_pixels_normalized = (valid_pixels - min_val) / (max_val - min_val)
    else:
        valid_pixels_normalized = valid_pixels
    
    median = np.median(valid_pixels_normalized)
    # avoid log(0)
    epsilon = 1e-4
    gamma = np.log(0.50 + epsilon) / np.log(median + epsilon)
    
    if mask is None:
        norm = (norm - min_val) / (max_val - min_val) if max_val > min_val else norm
        norm = np.power(norm, gamma)
    else:
        # Set non-masked regions to depth_out_value
        norm[mask <= 0] = depth_out_value
        # Apply normalization and gamma correction to masked regions
        norm[mask > 0] = (norm[mask > 0] - min_val) / (max_val - min_val) if max_val > min_val else norm[mask > 0]
        norm[mask > 0] = np.power(norm[mask > 0], gamma)

    return norm