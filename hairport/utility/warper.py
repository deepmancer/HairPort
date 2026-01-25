"""
Hair Alignment and Transfer Script

This script aligns hair from a source image to a target image using:
1. Landmark-based alignment
2. IoU optimization on head masks
3. Hair mask extraction and transfer
"""

# === Standard library ===
import os
import sys
import numpy as np
import cv2
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve

# === Third-party libraries ===
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import minimize

# === Local / project-specific modules ===
from data.preprocess.compute_mask import HairMaskPipeline
from utils.bg_remover import BackgroundRemover



# === Utility Functions ===
def resize_landmarks(landmarks, original_size, target_size):
    """
    Resize landmarks from original image size to target image size.
    
    Args:
        landmarks: (N, 2) array of [x, y] coordinates
        original_size: tuple (height, width) of original image
        target_size: tuple (height, width) of target image
    
    Returns:
        Resized landmarks (N, 2) array
    """
    orig_h, orig_w = original_size
    target_h, target_w = target_size
    
    # Calculate scale factors
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    
    # Apply scaling
    resized_landmarks = landmarks.copy()
    resized_landmarks[:, 0] = landmarks[:, 0] * scale_x  # x coordinates
    resized_landmarks[:, 1] = landmarks[:, 1] * scale_y  # y coordinates
    
    return resized_landmarks


def compute_iou(mask_a, mask_b):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return intersection / union


def transform_landmarks(lmk, scale_x, scale_y, angle, tx, ty, center_x, center_y):
    """Transform landmarks using affine transformation parameters."""
    if lmk is None or len(lmk) == 0:
        return None
    
    # Build transformation matrix
    M_scale = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])
    
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    M_rotate = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    M_translate = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    
    M_to_origin = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ])
    
    M_from_origin = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])
    
    # Combine transformations
    M = M_from_origin @ M_translate @ M_rotate @ M_scale @ M_to_origin
    
    # Transform landmarks: convert to homogeneous coordinates
    lmk_homogeneous = np.hstack([lmk, np.ones((len(lmk), 1))])
    transformed_lmk = (M @ lmk_homogeneous.T).T[:, :2]
    
    return transformed_lmk


def compute_landmark_distance(lmk_a, lmk_b):
    """Compute average Euclidean distance between corresponding landmarks."""
    if lmk_a is None or lmk_b is None:
        return 0.0
    if len(lmk_a) != len(lmk_b):
        return 0.0
    
    # Compute Euclidean distances
    distances = np.sqrt(np.sum((lmk_a - lmk_b) ** 2, axis=1))
    return np.mean(distances)


def transform_image_and_mask(img, mask, head_mask, scale_x, scale_y, angle, tx, ty):
    """Apply affine transformation to image and masks."""
    h, w = img.shape[:2]
    
    # Create transformation matrix
    center_x, center_y = w / 2, h / 2
    
    # Scale matrix
    M_scale = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])
    
    # Rotation matrix (around center)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    M_rotate = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    # Translation matrix
    M_translate = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    
    # Center translation matrices
    M_to_origin = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ])
    
    M_from_origin = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])
    
    # Combine transformations
    M = M_from_origin @ M_translate @ M_rotate @ M_scale @ M_to_origin
    
    # Apply to image and masks using OpenCV (2x3 matrix)
    M_cv = M[:2, :]
    
    transformed_img = cv2.warpAffine(img, M_cv, (w, h), flags=cv2.INTER_LINEAR, 
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    transformed_mask = cv2.warpAffine(mask.astype(np.uint8), M_cv, (w, h), 
                                      flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    transformed_head_mask = cv2.warpAffine(head_mask.astype(np.uint8), M_cv, (w, h), 
                                           flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    return transformed_img, transformed_mask > 0, transformed_head_mask > 0


def _compute_scale_and_translation_no_rotation(source_lmk: np.ndarray, target_lmk: np.ndarray, 
                                                  center_x: float, center_y: float) -> dict:
    """
    Compute optimal uniform scale and translation (NO rotation) to align target landmarks to source.
    
    The transformation model (matching transform_landmarks with angle=0):
        transformed = scale * (target - center) + center + translation
                    = scale * target + (1 - scale) * center + translation
    
    We want to minimize: ||source - transformed||^2
    
    This is a linear least squares problem that can be solved in closed form.
    """
    n = len(source_lmk)
    center = np.array([center_x, center_y])
    
    # Rewrite the transformation:
    # transformed = scale * target + (1 - scale) * center + t
    #             = scale * (target - center) + center + t
    # 
    # Let target_centered = target - center
    # transformed = scale * target_centered + center + t
    #
    # We want: source ≈ scale * target_centered + center + t
    # Or: (source - center) ≈ scale * target_centered + t
    #
    # Let source_shifted = source - center
    # source_shifted ≈ scale * target_centered + t
    
    target_centered = target_lmk - center  # (N, 2)
    source_shifted = source_lmk - center   # (N, 2)
    
    # Build linear system: we want to find [scale, tx, ty] that minimizes
    # sum over i: ||source_shifted[i] - (scale * target_centered[i] + [tx, ty])||^2
    #
    # This separates into x and y components:
    # For x: source_shifted[:,0] = scale * target_centered[:,0] + tx
    # For y: source_shifted[:,1] = scale * target_centered[:,1] + ty
    #
    # Combined system (2N equations, 3 unknowns):
    # [target_centered[:,0], 1, 0] [scale]   [source_shifted[:,0]]
    # [target_centered[:,1], 0, 1] [tx   ] = [source_shifted[:,1]]
    #                              [ty   ]
    
    # Build the design matrix A and target vector b
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    
    # X equations
    A[:n, 0] = target_centered[:, 0]  # scale coefficient for x
    A[:n, 1] = 1.0                     # tx coefficient
    A[:n, 2] = 0.0                     # ty coefficient
    b[:n] = source_shifted[:, 0]
    
    # Y equations  
    A[n:, 0] = target_centered[:, 1]  # scale coefficient for y
    A[n:, 1] = 0.0                     # tx coefficient
    A[n:, 2] = 1.0                     # ty coefficient
    b[n:] = source_shifted[:, 1]
    
    # Solve least squares: min ||Ax - b||^2
    result, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    scale, tx, ty = result
    
    # Compute RMSE
    transformed = scale * target_centered + np.array([tx, ty]) + center
    rmse = np.sqrt(np.mean(np.sum((source_lmk - transformed) ** 2, axis=1)))
    
    return {
        'scale': scale,
        'tx': tx,
        'ty': ty,
        'rmse': rmse
    }


def _compute_scale_rotation_translation(source_lmk: np.ndarray, target_lmk: np.ndarray,
                                         center_x: float, center_y: float) -> dict:
    """
    Compute optimal uniform scale, rotation, and translation to align target landmarks to source.
    
    The transformation model (matching transform_landmarks):
        transformed = R @ (scale * (target - center)) + center + translation
    
    where R is a 2D rotation matrix.
    """
    n = len(source_lmk)
    center = np.array([center_x, center_y])
    
    target_centered = target_lmk - center
    source_shifted = source_lmk - center
    
    # We want: source_shifted ≈ R @ (scale * target_centered) + t
    #        = scale * R @ target_centered + t
    #
    # Let R = [[cos, -sin], [sin, cos]]
    # Parameters: scale, cos, sin (with constraint cos^2 + sin^2 = 1), tx, ty
    #
    # For efficiency, parameterize as: scale * cos = a, scale * sin = b
    # Then: scale = sqrt(a^2 + b^2), angle = atan2(b, a)
    #
    # transformed_x = a * target_x - b * target_y + tx
    # transformed_y = b * target_x + a * target_y + ty
    #
    # This is linear in [a, b, tx, ty]!
    
    A = np.zeros((2 * n, 4))
    b_vec = np.zeros(2 * n)
    
    # X equations: source_shifted_x = a * target_centered_x - b * target_centered_y + tx
    A[:n, 0] = target_centered[:, 0]   # a coefficient
    A[:n, 1] = -target_centered[:, 1]  # b coefficient  
    A[:n, 2] = 1.0                      # tx coefficient
    A[:n, 3] = 0.0                      # ty coefficient
    b_vec[:n] = source_shifted[:, 0]
    
    # Y equations: source_shifted_y = b * target_centered_x + a * target_centered_y + ty
    A[n:, 0] = target_centered[:, 1]   # a coefficient
    A[n:, 1] = target_centered[:, 0]   # b coefficient
    A[n:, 2] = 0.0                      # tx coefficient
    A[n:, 3] = 1.0                      # ty coefficient
    b_vec[n:] = source_shifted[:, 1]
    
    # Solve least squares
    result, residuals, rank, s = np.linalg.lstsq(A, b_vec, rcond=None)
    a, b, tx, ty = result
    
    # Extract scale and angle
    scale = np.sqrt(a**2 + b**2)
    if scale < 1e-8:
        scale = 1.0
        angle = 0.0
    else:
        # a = scale * cos(angle), b = scale * sin(angle)
        angle = np.arctan2(b, a)
    
    # Compute RMSE
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    transformed = (scale * target_centered) @ R.T + np.array([tx, ty]) + center
    rmse = np.sqrt(np.mean(np.sum((source_lmk - transformed) ** 2, axis=1)))
    
    return {
        'scale': scale,
        'angle_rad': angle,
        'tx': tx,
        'ty': ty,
        'rmse': rmse
    }


def align_images_for_min_lmk_diff(
    source_image: np.ndarray,
    target_image: np.ndarray,
    source_lmk: np.ndarray,
    target_lmk: np.ndarray,
    allow_rotation: bool = False,
    allow_nonuniform_scale: bool = False,
    scale_bounds: tuple = (0.2, 5.0),
    rotation_bounds: tuple = (-np.pi/35, np.pi/35),
    max_translation_ratio: float = 0.8,
    aspect_ratio_bounds: tuple = (0.75, 1.33),
) -> tuple:
    """
    Align target_image to source_image by minimizing landmark distance.
    
    Uses closed-form least squares solutions that are compatible with
    the image-center-based transformation used by transform_landmarks and apply_transformation.
    
    Args:
        source_image: Source image (numpy array, HxWx3) - reference image
        target_image: Target image to be aligned (numpy array, HxWx3)
        source_lmk: Source landmarks (Nx2 array of [x, y] coordinates)
        target_lmk: Target landmarks (Nx2 array of [x, y] coordinates)
        allow_rotation: Whether to allow rotation (default: False)
        allow_nonuniform_scale: Whether to allow different x/y scales (default: False)
        scale_bounds: Tuple of (min_scale, max_scale) for scaling bounds
        rotation_bounds: Tuple of (min_angle, max_angle) in radians
        max_translation_ratio: Maximum translation as ratio of image dimension
        aspect_ratio_bounds: Tuple of (min_ratio, max_ratio) for scale_x/scale_y ratio
                            when allow_nonuniform_scale=True (default: (0.75, 1.33))
    
    Returns:
        tuple: (aligned_target_image, best_params_dict, final_landmark_distance)
    """
    if source_lmk is None or target_lmk is None:
        raise ValueError("Both source_lmk and target_lmk must be provided")
    
    if len(source_lmk) != len(target_lmk):
        raise ValueError(f"Landmark count mismatch: source has {len(source_lmk)}, target has {len(target_lmk)}")
    
    if len(source_lmk) == 0:
        raise ValueError("Landmarks cannot be empty")
    
    source_lmk = np.asarray(source_lmk, dtype=np.float64)
    target_lmk = np.asarray(target_lmk, dtype=np.float64)
    
    h, w = target_image.shape[:2]
    center_x, center_y = w / 2, h / 2
    max_translation = max(h, w) * max_translation_ratio
    
    print(f"\n=== Landmark-based Alignment ===")
    print(f"Aligning {len(source_lmk)} landmark pairs")
    print(f"Image size: {w}x{h}, center: ({center_x}, {center_y})")
    print(f"Options: rotation={allow_rotation}, non-uniform_scale={allow_nonuniform_scale}")
    
    # Compute initial landmark distance
    initial_dist = compute_landmark_distance(source_lmk, target_lmk)
    print(f"Initial landmark distance: {initial_dist:.2f} pixels")
    
    # ========================================
    # Compute closed-form solution
    # ========================================
    if allow_rotation:
        print("\nComputing optimal scale + rotation + translation (closed-form)...")
        result = _compute_scale_rotation_translation(source_lmk, target_lmk, center_x, center_y)
        init_scale = result['scale']
        init_angle = result['angle_rad']
        init_tx = result['tx']
        init_ty = result['ty']
        print(f"Closed-form solution: scale={init_scale:.4f}, angle={np.degrees(init_angle):.2f}°, "
              f"translation=({init_tx:.2f}, {init_ty:.2f}), RMSE={result['rmse']:.2f}")
    else:
        print("\nComputing optimal scale + translation (closed-form, no rotation)...")
        result = _compute_scale_and_translation_no_rotation(source_lmk, target_lmk, center_x, center_y)
        init_scale = result['scale']
        init_angle = 0.0
        init_tx = result['tx']
        init_ty = result['ty']
        print(f"Closed-form solution: scale={init_scale:.4f}, "
              f"translation=({init_tx:.2f}, {init_ty:.2f}), RMSE={result['rmse']:.2f}")
    
    # ========================================
    # Apply bounds and refine if needed
    # ========================================
    scale_clamped = np.clip(init_scale, scale_bounds[0], scale_bounds[1])
    angle_clamped = np.clip(init_angle, rotation_bounds[0], rotation_bounds[1]) if allow_rotation else 0.0
    tx_clamped = np.clip(init_tx, -max_translation, max_translation)
    ty_clamped = np.clip(init_ty, -max_translation, max_translation)
    
    # Check if we need refinement (bounds were hit or non-uniform scale requested)
    needs_refinement = (
        allow_nonuniform_scale or
        init_scale != scale_clamped or
        init_angle != angle_clamped or
        init_tx != tx_clamped or
        init_ty != ty_clamped
    )
    
    if needs_refinement:
        print("\nRefining with constrained optimization...")
        
        def landmark_objective(params):
            """Objective: sum of squared landmark distances."""
            if allow_nonuniform_scale:
                if allow_rotation:
                    sx, sy, ang, tx, ty = params
                else:
                    sx, sy, tx, ty = params
                    ang = 0.0
            else:
                if allow_rotation:
                    s, ang, tx, ty = params
                    sx = sy = s
                else:
                    s, tx, ty = params
                    sx = sy = s
                    ang = 0.0
            
            transformed = transform_landmarks(target_lmk, sx, sy, ang, tx, ty, center_x, center_y)
            return np.sum((source_lmk - transformed) ** 2)
        
        # Build initial guess and bounds
        if allow_nonuniform_scale:
            if allow_rotation:
                x0 = np.array([scale_clamped, scale_clamped, angle_clamped, tx_clamped, ty_clamped])
                bounds = [scale_bounds, scale_bounds, rotation_bounds,
                         (-max_translation, max_translation), (-max_translation, max_translation)]
            else:
                x0 = np.array([scale_clamped, scale_clamped, tx_clamped, ty_clamped])
                bounds = [scale_bounds, scale_bounds,
                         (-max_translation, max_translation), (-max_translation, max_translation)]
            
            # Build aspect ratio constraints for SLSQP
            # Constraint 1: scale_x / scale_y >= min_ratio  =>  scale_x - min_ratio * scale_y >= 0
            # Constraint 2: scale_x / scale_y <= max_ratio  =>  max_ratio * scale_y - scale_x >= 0
            min_ratio, max_ratio = aspect_ratio_bounds
            
            if allow_rotation:
                # params = [sx, sy, ang, tx, ty]
                constraints = [
                    {'type': 'ineq', 'fun': lambda p: p[0] - min_ratio * p[1]},  # sx >= min_ratio * sy
                    {'type': 'ineq', 'fun': lambda p: max_ratio * p[1] - p[0]},  # sx <= max_ratio * sy
                ]
            else:
                # params = [sx, sy, tx, ty]
                constraints = [
                    {'type': 'ineq', 'fun': lambda p: p[0] - min_ratio * p[1]},  # sx >= min_ratio * sy
                    {'type': 'ineq', 'fun': lambda p: max_ratio * p[1] - p[0]},  # sx <= max_ratio * sy
                ]
            
            # Use SLSQP for constrained optimization when non-uniform scale is enabled
            opt_result = minimize(
                landmark_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 600, 'ftol': 1e-12}
            )
            print(f"Aspect ratio constraint: {min_ratio:.2f} <= scale_x/scale_y <= {max_ratio:.2f}")
        else:
            if allow_rotation:
                x0 = np.array([scale_clamped, angle_clamped, tx_clamped, ty_clamped])
                bounds = [scale_bounds, rotation_bounds,
                         (-max_translation, max_translation), (-max_translation, max_translation)]
            else:
                x0 = np.array([scale_clamped, tx_clamped, ty_clamped])
                bounds = [scale_bounds,
                         (-max_translation, max_translation), (-max_translation, max_translation)]
            
            # Use Powell for uniform scale (no aspect ratio constraint needed)
            opt_result = minimize(
                landmark_objective,
                x0,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 600, 'ftol': 1e-12}
            )
        
        # Extract parameters
        if allow_nonuniform_scale:
            if allow_rotation:
                scale_x, scale_y, angle, tx, ty = opt_result.x
            else:
                scale_x, scale_y, tx, ty = opt_result.x
                angle = 0.0
            # Log the actual aspect ratio achieved
            actual_ratio = scale_x / scale_y if scale_y != 0 else float('inf')
            print(f"Actual aspect ratio (scale_x/scale_y): {actual_ratio:.4f}")
        else:
            if allow_rotation:
                scale, angle, tx, ty = opt_result.x
                scale_x = scale_y = scale
            else:
                scale, tx, ty = opt_result.x
                scale_x = scale_y = scale
                angle = 0.0
        
        print(f"Refined: scale=({scale_x:.4f}, {scale_y:.4f}), "
              f"angle={np.degrees(angle):.2f}°, translation=({tx:.2f}, {ty:.2f})")
    else:
        # Use closed-form solution directly
        print("\nUsing closed-form solution directly (within bounds)")
        scale_x = scale_y = scale_clamped
        angle = angle_clamped
        tx, ty = tx_clamped, ty_clamped
    
    # ========================================
    # Apply transformation to image
    # ========================================
    print("\nApplying transformation to image...")
    aligned_target_image = apply_transformation(target_image, scale_x, scale_y, angle, tx, ty)
    
    # Compute final landmark distance
    transformed_target_lmk = transform_landmarks(target_lmk, scale_x, scale_y, angle, tx, ty, center_x, center_y)
    final_lmk_dist = np.sqrt(np.mean(np.sum((source_lmk - transformed_target_lmk) ** 2, axis=1)))
    
    print(f"\n=== Alignment Complete ===")
    print(f"Final landmark RMSE: {final_lmk_dist:.2f} pixels (initial: {initial_dist:.2f})")
    improvement = (1 - final_lmk_dist / initial_dist) * 100 if initial_dist > 0 else 0
    print(f"Improvement: {improvement:.1f}%")
    print(f"Parameters:")
    print(f"  Scale: ({scale_x:.4f}, {scale_y:.4f})")
    print(f"  Rotation: {np.degrees(angle):.2f}°")
    print(f"  Translation: ({tx:.2f}, {ty:.2f}) pixels")
    
    best_params_dict = {
        'scale_x': float(scale_x),
        'scale_y': float(scale_y),
        'angle_rad': float(angle),
        'angle_deg': float(np.degrees(angle)),
        'tx': float(tx),
        'ty': float(ty),
        'landmark_distance': float(final_lmk_dist)
    }
    
    return aligned_target_image, best_params_dict, final_lmk_dist


def align_images_for_max_iou(source_image, mask1, target_image, mask2, head_mask1, head_mask2, 
                              source_lmk, target_lmk, 
                              iou_weight=1.0, landmark_weight=1.0):
    """
    Align target_image to source_image by optimizing scale, rotation, and translation
    to maximize IoU between head masks and minimize landmark distances.
  
    Returns:
        aligned_target_image: Transformed target image
        aligned_mask2: Transformed mask
        aligned_head_mask2: Transformed head mask
        best_params: Dictionary of optimal transformation parameters
        best_iou: The maximum IoU achieved
    """
    
    h, w = target_image.shape[:2]
    img_diagonal = np.sqrt(h**2 + w**2)
    center_x, center_y = w / 2, h / 2
    
    def objective(params):
        """Multi-objective function: minimize (-IoU + landmark_distance)"""
        scale_x, scale_y, angle, tx, ty = params
        
        # Apply transformation to masks
        _, _, transformed_head_mask = transform_image_and_mask(
            target_image, mask2, head_mask2, scale_x, scale_y, angle, tx, ty
        )
        
        # Compute IoU
        iou = compute_iou(head_mask1, transformed_head_mask)
        
        # Add boundary alignment penalty: penalize misalignment at mask boundaries
        # Compute boundary overlap to encourage better edge alignment
        if iou > 0:
            # Find boundary pixels
            from scipy.ndimage import binary_erosion
            boundary1 = head_mask1 & ~binary_erosion(head_mask1, iterations=2)
            boundary2 = transformed_head_mask & ~binary_erosion(transformed_head_mask, iterations=2)
            
            # Compute boundary IoU (encourages precise edge alignment)
            boundary_intersection = np.logical_and(boundary1, boundary2).sum()
            boundary_union = np.logical_or(boundary1, boundary2).sum()
            boundary_iou = boundary_intersection / boundary_union if boundary_union > 0 else 0.0
            
            # Enhanced IoU metric that emphasizes boundary alignment
            enhanced_iou = 0.7 * iou + 0.3 * boundary_iou
        else:
            enhanced_iou = iou
        
        # Compute landmark distance if landmarks are provided
        landmark_dist = 0.0
        if source_lmk is not None and target_lmk is not None:
            transformed_target_lmk = transform_landmarks(target_lmk, scale_x, scale_y, angle, tx, ty, center_x, center_y)
            
            # Normalize landmark distance by image diagonal for scale invariance
            landmark_dist = compute_landmark_distance(source_lmk, transformed_target_lmk) / img_diagonal
        
        # Combined objective with enhanced IoU
        loss = -iou_weight * enhanced_iou + landmark_weight * landmark_dist
        
        return loss
    
    # === Step 1: Landmark-only optimization (scale + translation, zero rotation) ===
    # Dynamic bounds based on image size (define early for use in Phase 1)
    max_translation = max(h, w) * 0.25  # Allow up to 50% of image dimension
    
    x0 = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    
    if source_lmk is not None and target_lmk is not None and len(source_lmk) > 0:
        print("\n=== PHASE 1: Landmark-only optimization (scale + translation, zero rotation) ===")
        
        def landmark_only_objective(params):
            """Objective function for landmark alignment only (no rotation)."""
            scale_x, scale_y, tx, ty = params
            angle = 0.0  # Force zero rotation
            
            # Transform target landmarks
            transformed_target_lmk = transform_landmarks(
                target_lmk, scale_x, scale_y, angle, tx, ty, center_x, center_y
            )
            
            # Compute landmark distance (normalized by image diagonal)
            landmark_dist = compute_landmark_distance(source_lmk, transformed_target_lmk) / img_diagonal
            
            return landmark_dist
        
        # Initial guess for scale and translation (no rotation)
        lmk_x0 = np.array([1.0, 1.0, 0.0, 0.0])
        
        # Bounds for scale and translation (no rotation)
        lmk_bounds = [
            (0.7, 1.5),                          # scale_x
            (0.7, 1.5),                          # scale_y
            (-max_translation, max_translation), # tx
            (-max_translation, max_translation)  # ty
        ]

        print(f"Optimizing alignment for {len(source_lmk)} landmark pairs (zero rotation)...")
        
        # Optimize for landmark alignment only
        try:
            lmk_result = minimize(
                landmark_only_objective,
                lmk_x0,
                method='L-BFGS-B',
                bounds=lmk_bounds,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            
            if lmk_result.success:
                lmk_scale_x, lmk_scale_y, lmk_tx, lmk_ty = lmk_result.x
                lmk_angle = 0.0
                
                # Compute landmark distance after optimization
                transformed_lmk = transform_landmarks(
                    target_lmk, lmk_scale_x, lmk_scale_y, lmk_angle, lmk_tx, lmk_ty, center_x, center_y
                )
                final_lmk_dist = compute_landmark_distance(source_lmk, transformed_lmk)
                
                print(f"Landmark-only optimization complete:")
                print(f"  Scale: ({lmk_scale_x:.4f}, {lmk_scale_y:.4f})")
                print(f"  Translation: ({lmk_tx:.2f}, {lmk_ty:.2f}) pixels")
                print(f"  Final landmark distance: {final_lmk_dist:.2f} pixels")
                
                # Use this as initial guess for full optimization
                x0 = np.array([lmk_scale_x, lmk_scale_y, lmk_angle, lmk_tx, lmk_ty])
            else:
                print(f"Landmark-only optimization did not converge, using default initialization")
        except Exception as e:
            print(f"Landmark-only optimization failed: {e}, using default initialization")
    
    print(f"\n=== PHASE 2: Full optimization (IoU + landmarks) ===")
    print(f"Starting with: scale=({x0[0]:.3f}, {x0[1]:.3f}), angle={np.degrees(x0[2]):.1f}°, trans=({x0[3]:.1f}, {x0[4]:.1f})")
    
    bounds = [
        (0.7, 1.5),                          # scale_x: 0.5x to 4x
        (0.7, 1.5),                          # scale_y: 0.5x to 4x
        (-np.pi/15, np.pi/15),                 # angle: -60 to 60 degrees (tighter for heads)
        (-max_translation, max_translation), # tx: dynamic based on image size
        (-max_translation, max_translation)  # ty: dynamic based on image size
    ]
    
    if source_lmk is not None and target_lmk is not None:
        print(f"Using landmarks: {len(source_lmk)} keypoints")
        print(f"Weights: IoU={iou_weight}, Landmarks={landmark_weight}")
    else:
        print("No landmarks provided, optimizing IoU only")
    
    # Compute initial metrics
    initial_loss = objective(x0)
    _, _, initial_head_mask = transform_image_and_mask(target_image, mask2, head_mask2, *x0)
    initial_iou = compute_iou(head_mask1, initial_head_mask)
    
    initial_lmk_dist = 0.0
    if source_lmk is not None and target_lmk is not None:
        transformed_initial_lmk = transform_landmarks(target_lmk, x0[0], x0[1], x0[2], x0[3], x0[4], center_x, center_y)
        initial_lmk_dist = compute_landmark_distance(source_lmk, transformed_initial_lmk) / img_diagonal
    
    print(f"Initial metrics: IoU={initial_iou:.4f}, Landmark Distance={initial_lmk_dist:.4f}")
    
    # === Step 2: Multi-stage optimization with multiple initializations ===
    best_result = None
    best_loss = float('inf')
    best_method = None
    
    # Generate multiple initial guesses for robustness
    initial_guesses = [x0]
    
    # Add perturbations around landmark-based initialization
    if source_lmk is not None and target_lmk is not None:
        perturbations = [
            [0.98, 0.98, -0.02, 0, 0],
            [1.02, 1.02, 0.02, 0, 0],
            [1.0, 1.0, 0, 5, -5],
            [1.0, 1.0, 0, -5, 5],
        ]
        for perturb in perturbations:
            perturbed = x0 + np.array(perturb)
            # Ensure perturbations are within bounds
            perturbed = np.clip(perturbed, [b[0] for b in bounds], [b[1] for b in bounds])
            initial_guesses.append(perturbed)
    
    # Try different optimization methods with different starting points
    optimization_configs = [
        # ('L-BFGS-B', {'maxiter': 2000, 'ftol': 1e-8}),
        ('Powell', {'maxiter': 1000, 'ftol': 1e-8}),
        # ('TNC', {'maxiter': 1500}),
    ]
    
    print("\nRunning multi-start optimization...")
    for init_idx, init_guess in enumerate(initial_guesses):
        for method, options in optimization_configs:
            try:
                result = minimize(
                    objective,
                    init_guess,
                    method=method,
                    bounds=bounds,
                    options=options
                )
                
                current_loss = result.fun
                
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_result = result
                    best_method = method
                    print(f"  Init {init_idx}, Method {method}: Loss = {current_loss:.6f} *** NEW BEST ***")
                else:
                    print(f"  Init {init_idx}, Method {method}: Loss = {current_loss:.6f}")
                    
            except Exception as e:
                print(f"  Init {init_idx}, Method {method} failed: {e}")
                continue
    
    # === Step 3: Fine-tuning stage ===
    if best_result is not None:
        print("\nFine-tuning best result...")
        try:
            refined_result = minimize(
                objective,
                best_result.x,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 600, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if refined_result.fun < best_loss:
                print(f"Fine-tuning improved loss: {best_loss:.6f} -> {refined_result.fun:.6f}")
                best_result = refined_result
                best_loss = refined_result.fun
        except Exception as e:
            print(f"Fine-tuning failed: {e}, using previous best")
    
    if best_result is None:
        print("All optimization methods failed! Returning original images.")
        return target_image, mask2, head_mask2, x0, 0.0
    
    # Extract best parameters
    best_params = best_result.x
    scale_x, scale_y, angle, tx, ty = best_params
    
    # Compute final metrics
    aligned_target_image, aligned_mask2, aligned_head_mask2 = transform_image_and_mask(
        target_image, mask2, head_mask2, scale_x, scale_y, angle, tx, ty
    )
    
    final_iou = compute_iou(head_mask1, aligned_head_mask2)
    
    final_lmk_dist = 0.0
    final_lmk_dist_pixels = 0.0
    if source_lmk is not None and target_lmk is not None:
        transformed_target_lmk = transform_landmarks(target_lmk, scale_x, scale_y, angle, tx, ty, center_x, center_y)
        final_lmk_dist = compute_landmark_distance(source_lmk, transformed_target_lmk) / img_diagonal
        final_lmk_dist_pixels = compute_landmark_distance(source_lmk, transformed_target_lmk)
    
    print(f"\nOptimization complete! (Best method: {best_method})")
    print(f"Final IoU: {final_iou:.4f} (initial: {initial_iou:.4f}, improvement: {(final_iou-initial_iou)*100:.1f}%)")
    if source_lmk is not None and target_lmk is not None:
        print(f"Final Landmark Distance: {final_lmk_dist_pixels:.2f} pixels (initial: {initial_lmk_dist * img_diagonal:.2f} pixels)")
    print(f"Parameters:")
    print(f"  Scale X: {scale_x:.4f}")
    print(f"  Scale Y: {scale_y:.4f}")
    print(f"  Rotation: {np.degrees(angle):.2f}°")
    print(f"  Translation: ({tx:.2f}, {ty:.2f}) pixels")
    
    best_params_dict = {
        'scale_x': scale_x,
        'scale_y': scale_y,
        'angle_rad': angle,
        'angle_deg': np.degrees(angle),
        'tx': tx,
        'ty': ty,
        'iou': final_iou,
        'landmark_distance': final_lmk_dist_pixels if source_lmk is not None else None
    }
    
    return aligned_target_image, aligned_mask2, aligned_head_mask2, best_params_dict, final_iou


def apply_transformation(img, scale_x, scale_y, angle, tx, ty):
    """
    Apply the same transformation as during optimization to any image.
    
    Args:
        img: Input image (numpy array or PIL Image)
        scale_x, scale_y: Scaling factors
        angle: Rotation angle in radians
        tx, ty: Translation in pixels
    
    Returns:
        Transformed image as numpy array
    """
    # Convert to numpy if PIL
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    h, w = img.shape[:2]
    
    # Create transformation matrix
    center_x, center_y = w / 2, h / 2
    
    # Scale matrix
    M_scale = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ])
    
    # Rotation matrix (around center)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    M_rotate = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    # Translation matrix
    M_translate = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    
    # Center translation matrices
    M_to_origin = np.array([
        [1, 0, -center_x],
        [0, 1, -center_y],
        [0, 0, 1]
    ])
    
    M_from_origin = np.array([
        [1, 0, center_x],
        [0, 1, center_y],
        [0, 0, 1]
    ])
    
    # Combine transformations
    M = M_from_origin @ M_translate @ M_rotate @ M_scale @ M_to_origin
    
    # Apply to image using OpenCV (2x3 matrix)
    M_cv = M[:2, :]
    
    # Determine border value based on image channels
    if len(img.shape) == 3 and img.shape[2] == 3:
        border_value = (255, 255, 255)  # White for RGB images
    else:
        border_value = 0  # Black for grayscale masks
    
    # Use appropriate interpolation for masks vs images
    interpolation = cv2.INTER_LINEAR if len(img.shape) == 3 else cv2.INTER_NEAREST
    
    transformed_img = cv2.warpAffine(img, M_cv, (w, h), flags=interpolation, 
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    
    return transformed_img


def _prepare_images_and_mask(source_image, aligned_target_with_hair, aligned_hair_mask):
    """
    Prepare and validate images and mask for blending operations.
    
    Returns:
        tuple: (dst_img, src_img, refined_mask, target_h, target_w) all as uint8
    """
    target_h, target_w = source_image.shape[:2]
    
    # Resize inputs if needed to match source image
    if aligned_target_with_hair.shape[:2] != (target_h, target_w):
        src_img = cv2.resize(
            aligned_target_with_hair, 
            (target_w, target_h), 
            interpolation=cv2.INTER_LINEAR
        )
        src_mask = cv2.resize(
            aligned_hair_mask, 
            (target_w, target_h), 
            interpolation=cv2.INTER_NEAREST
        )
    else:
        src_img = aligned_target_with_hair.copy()
        src_mask = aligned_hair_mask.copy()
    
    # Ensure correct types - must be uint8 for cv2 operations
    if source_image.dtype != np.uint8:
        dst_img = (source_image * 255).clip(0, 255).astype(np.uint8)
    else:
        dst_img = source_image.copy()
        
    if src_img.dtype != np.uint8:
        src_img = (src_img * 255).clip(0, 255).astype(np.uint8)
    
    # Ensure images are 3-channel BGR/RGB
    if len(dst_img.shape) == 2:
        dst_img = cv2.cvtColor(dst_img, cv2.COLOR_GRAY2BGR)
    elif dst_img.shape[2] == 4:
        dst_img = dst_img[:, :, :3]
        
    if len(src_img.shape) == 2:
        src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
    elif src_img.shape[2] == 4:
        src_img = src_img[:, :, :3]
    
    # Ensure contiguous arrays
    dst_img = np.ascontiguousarray(dst_img)
    src_img = np.ascontiguousarray(src_img)

    # Process Mask: Binarize and refine
    # 1. Binarize
    if src_mask.max() > 1.0:
        binary_mask = (src_mask > 127).astype(np.uint8) * 255
    else:
        binary_mask = (src_mask > 0.5).astype(np.uint8) * 255
        
    if len(binary_mask.shape) == 3:
        binary_mask = binary_mask[:, :, 0]

    # 2. Refine mask (remove noise and smooth)
    kernel_size = max(3, int(min(target_h, target_w) * 0.005))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    refined_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
    
    return dst_img, src_img, refined_mask, target_h, target_w


def _alpha_blend(dst_img, src_img, refined_mask, target_h, target_w):
    """
    Perform alpha blending with soft feathered edges.
    
    Args:
        dst_img: Destination image (bald) as uint8
        src_img: Source image (with hair) as uint8
        refined_mask: Refined binary mask as uint8
        target_h, target_w: Target dimensions
    
    Returns:
        Blended result as uint8
    """
    # Create Soft Alpha Matte (Feathering)
    blur_radius = max(3, int(min(target_h, target_w) * 0.01)) 
    if blur_radius % 2 == 0:
        blur_radius += 1  # Must be odd
    
    alpha_mask = cv2.GaussianBlur(refined_mask, (blur_radius, blur_radius), 0)
    
    # Normalize alpha to 0.0 - 1.0
    alpha = alpha_mask.astype(np.float32) / 255.0
    
    # Expand alpha to 3 channels for broadcasting
    alpha_3ch = np.stack([alpha] * 3, axis=-1)
    
    # Alpha Blending: Result = Source_Hair * Alpha + Dest_Bald * (1 - Alpha)
    f_src = src_img.astype(np.float32)
    f_dst = dst_img.astype(np.float32)
    
    blended = f_src * alpha_3ch + f_dst * (1.0 - alpha_3ch)
    
    return blended.clip(0, 255).astype(np.uint8)

def _screened_poisson_blend(
    dst_img: np.ndarray,
    src_img: np.ndarray,
    mask_u8: np.ndarray,
    pad: int = 24,
    lam: float = 1.0,
    mixed_gradients: bool = True,
    max_iter: int = 600,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Screened Poisson blending (identity-preserving) on a cropped ROI around the mask.

    Minimizes:
        sum ||∇I - ∇S||^2 + lam * sum ||I - S||^2   over mask domain Ω
    with destination boundary conditions imposed implicitly via neighbors outside Ω.

    Args:
        dst_img: uint8 HxWx3 destination image
        src_img: uint8 HxWx3 source image (aligned hair donor)
        mask_u8: uint8 HxW mask, 0..255 (hair region)
        pad: padding around mask bbox (and also used to pad full image to avoid border issues)
        lam: screening weight. Higher => preserves source appearance more (less drift).
        mixed_gradients: if True, use stronger gradient per-edge from src vs dst (like MIXED_CLONE)
        max_iter, tol: CG solver settings

    Returns:
        uint8 HxWx3 blended image
    """
    if dst_img.dtype != np.uint8 or src_img.dtype != np.uint8:
        raise ValueError("dst_img and src_img must be uint8")
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)

    # If mask empty, return destination
    if np.count_nonzero(mask_u8) == 0:
        return dst_img

    # Pad entire images to make ROI safe even if hair touches original borders
    dst_pad = cv2.copyMakeBorder(dst_img, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101)
    src_pad = cv2.copyMakeBorder(src_img, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT_101)
    m_pad   = cv2.copyMakeBorder(mask_u8, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=0)

    mask_bool = m_pad > 0
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return dst_img

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    # ROI crop with additional pad so ROI contains boundary neighbors
    y0r = max(0, y0 - pad)
    y1r = min(dst_pad.shape[0] - 1, y1 + pad)
    x0r = max(0, x0 - pad)
    x1r = min(dst_pad.shape[1] - 1, x1 + pad)

    dst_roi = dst_pad[y0r:y1r+1, x0r:x1r+1].astype(np.float32)
    src_roi = src_pad[y0r:y1r+1, x0r:x1r+1].astype(np.float32)
    m_roi   = (m_pad[y0r:y1r+1, x0r:x1r+1] > 0)

    h, w = m_roi.shape
    n = int(np.count_nonzero(m_roi))
    if n < 50:
        # Too small to solve meaningfully; return original dst
        return dst_img

    # Map mask pixels to variable indices
    idx = -np.ones((h, w), dtype=np.int32)
    idx[m_roi] = np.arange(n, dtype=np.int32)

    # Build sparse system A x = b (per channel)
    A = sp.lil_matrix((n, n), dtype=np.float32)
    b = np.zeros((n, 3), dtype=np.float32)

    # 4-neighborhood
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(h):
        for x in range(w):
            if not m_roi[y, x]:
                continue

            p = idx[y, x]
            A[p, p] = 4.0 + lam  # Laplacian degree + screening

            Sp = src_roi[y, x]
            Dp = dst_roi[y, x]

            # screening term pulls solution toward source to preserve identity
            b[p] += lam * Sp

            for dy, dx in nbrs:
                yy, xx = y + dy, x + dx

                # ROI is padded enough that this is usually in-bounds; still guard.
                if yy < 0 or yy >= h or xx < 0 or xx >= w:
                    continue

                Sq = src_roi[yy, xx]
                Dq = dst_roi[yy, xx]

                # Guidance field v_pq (gradient to match)
                if mixed_gradients:
                    grad_s = Sp - Sq
                    grad_d = Dp - Dq
                    v = np.where(np.abs(grad_s) >= np.abs(grad_d), grad_s, grad_d)
                else:
                    v = Sp - Sq

                if m_roi[yy, xx]:
                    q = idx[yy, xx]
                    A[p, q] = -1.0
                    b[p] += v
                else:
                    # Neighbor outside Ω => boundary condition from destination
                    # Equation contributes: + Dq and + v
                    b[p] += v + Dq

    A = A.tocsr()

    out_roi = dst_roi.copy()
    # Solve per channel
    for c in range(3):
        x0 = src_roi[:, :, c][m_roi].reshape(-1).astype(np.float32)

        sol, info = cg(A, b[:, c], x0=x0, maxiter=max_iter, rtol=tol)
        if info != 0:
            # Fallback to direct solve (slower but robust)
            sol = spsolve(A, b[:, c]).astype(np.float32)

        out_roi[:, :, c][m_roi] = sol

    out_roi = np.clip(out_roi, 0, 255).astype(np.uint8)

    # Paste ROI back into padded dst
    blended_pad = dst_pad.copy()
    blended_pad[y0r:y1r+1, x0r:x1r+1] = out_roi

    # Unpad back to original size
    blended = blended_pad[pad:-pad, pad:-pad]
    return blended


def _poisson_blend(
    dst_img, src_img, refined_mask, target_h, target_w,
    clone_type=cv2.NORMAL_CLONE,
    screened_lam: float = 1.0,
    mixed_gradients: bool = True,
):
    """
    Improved Poisson blending:
    - Prefer Screened Poisson (identity-preserving, robust to borders).
    - If anything fails unexpectedly, fall back to cv2.seamlessClone.
    """
    # Screened Poisson is typically the best choice for hair realism + identity
    return _screened_poisson_blend(
        dst_img=dst_img,
        src_img=src_img,
        mask_u8=refined_mask,
        pad=max(16, int(min(target_h, target_w) * 0.02)),
        lam=float(screened_lam),
        mixed_gradients=bool(mixed_gradients),
    )


def transfer_hair(source_image, aligned_target_with_hair, aligned_hair_mask, 
                  method: str = "alpha_blending", poisson_strength: float = 1.0):
    """
    Transfer hair from aligned target image to source image using either soft alpha blending
    or Poisson blending for a seamless look.
    
    Args:
        source_image: Target image (bald) to receive hair
        aligned_target_with_hair: Aligned source image with hair
        aligned_hair_mask: Aligned hair mask
        method: Blending method to use. Options:
            - "alpha_blending": Soft alpha blending with feathered edges (default)
            - "poisson_blending": Seamless cloning using Poisson blending (cv2.seamlessClone)
        poisson_strength: Controls the intensity of Poisson blending (0.0 to 1.0).
            Only used when method="poisson_blending".
            - 1.0: Full Poisson blending (default, most aggressive)
            - 0.5: 50% Poisson + 50% alpha blending
            - 0.0: Equivalent to pure alpha blending
            This allows reducing the strong color/texture changes that Poisson blending can cause.
    
    Returns:
        Result image with transferred hair
    """
    # Validate method parameter
    valid_methods = ["alpha_blending", "poisson_blending"]
    if method not in valid_methods:
        raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}")
    
    # Validate poisson_strength
    poisson_strength = np.clip(poisson_strength, 0.0, 1.0)
    
    # Prepare images and mask
    dst_img, src_img, refined_mask, target_h, target_w = _prepare_images_and_mask(
        source_image, aligned_target_with_hair, aligned_hair_mask
    )
    
    # Check if mask is empty
    if np.sum(refined_mask > 0) == 0:
        return dst_img
    
    # Always compute alpha blending (needed as fallback or for mixing)
    alpha_result = _alpha_blend(dst_img, src_img, refined_mask, target_h, target_w)
    
    if method == "poisson_blending":
        # Try MIXED_CLONE first if strength < 1.0, as it's less aggressive
        clone_type = cv2.MIXED_CLONE if poisson_strength < 0.7 else cv2.NORMAL_CLONE
        
        poisson_result = _poisson_blend(dst_img, src_img, refined_mask, target_h, target_w, clone_type)
        
        if poisson_result is not None:
            if poisson_strength >= 1.0:
                # Full Poisson blending
                return poisson_result
            elif poisson_strength <= 0.0:
                # No Poisson, just alpha
                return alpha_result
            else:
                # Mix Poisson result with alpha blending result
                # This allows controlling how much the Poisson effect is applied
                result = cv2.addWeighted(
                    poisson_result, poisson_strength,
                    alpha_result, 1.0 - poisson_strength,
                    0
                )
                return result
        else:
            # Fall back to alpha blending if Poisson blending fails
            print("Info: Poisson blending not possible for this configuration. Using alpha blending.")
            return alpha_result
    
    # Alpha blending (default)
    return alpha_result


def main(source_id: str, target_id: str, data_dir: str, visualize: bool = False, debug: bool = False):
    """
    Main function to perform hair alignment and transfer.
    
    Args:
        source_id: ID of the source image (bald)
        target_id: ID of the target image (with hair)
        data_dir: Base data directory
        visualize: Whether to show visualization plots
        debug: Whether to save debug images
    """
    # Setup paths
    BALD_DATA_DIR = os.path.join(data_dir, "bald", "wo_seg")
    VIEW_ALIGNED_DATA_DIR = os.path.join(data_dir, f"refined_w_flux/{target_id}_to_{source_id}")
    DEBUG_DIR = os.path.join(VIEW_ALIGNED_DATA_DIR, "debug")
    os.makedirs(VIEW_ALIGNED_DATA_DIR, exist_ok=True)
    
    # Delete and recreate debug directory to ensure clean state (only if debug mode is enabled)
    import shutil
    if os.path.exists(DEBUG_DIR):
        shutil.rmtree(DEBUG_DIR)
    if debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
    
    print(f"Processing: {target_id} -> {source_id}")
    print(f"Data directory: {data_dir}")
    print(f"Debug directory: {DEBUG_DIR}")
    
    # === Load Source (Bald) Data ===
    print("\nLoading source (bald) data...")
    source_image_path = os.path.join(BALD_DATA_DIR, "image", f"{source_id}.png")
    source_image = Image.open(source_image_path).convert('RGB')
    
    source_lmk_path = os.path.join(BALD_DATA_DIR, "lmk", source_id, "landmarks.npy")
    source_lmk_data = np.load(source_lmk_path, allow_pickle=True).item()
    
    source_flame_data_dir = os.path.join(BALD_DATA_DIR, "pixel3dmm_output", source_id)
    source_head_mask_path = os.path.join(source_flame_data_dir, "flame_segmentation.png")
    source_head_mask = Image.open(source_head_mask_path).convert('L')
    
    # Save debug: source images
    if debug:
        source_image.save(os.path.join(DEBUG_DIR, "01_source_image.png"))
        source_head_mask.save(os.path.join(DEBUG_DIR, "02_source_head_mask.png"))
    
    # === Load Target (With Hair) Data ===
    print("Loading target (with hair) data...")
    target_image_path = os.path.join(VIEW_ALIGNED_DATA_DIR, "refined_view_aligned.png")
    target_image = Image.open(target_image_path).convert('RGB')
    
    target_lmk_path = os.path.join(VIEW_ALIGNED_DATA_DIR, "lmk", "refined_view_aligned", "landmarks.npy")
    target_lmk_data = np.load(target_lmk_path, allow_pickle=True).item()
    
    target_flame_data_dir = os.path.join(VIEW_ALIGNED_DATA_DIR, "pixel3dmm_output", "refined_view_aligned")
    target_head_mask_path = os.path.join(target_flame_data_dir, "flame_segmentation.png")
    target_head_mask = Image.open(target_head_mask_path).convert('L')
    
    # Save debug: target images
    if debug:
        target_image.save(os.path.join(DEBUG_DIR, "03_target_image.png"))
        target_head_mask.save(os.path.join(DEBUG_DIR, "04_target_head_mask.png"))
    
    # === Remove Background ===
    print("\nRemoving backgrounds...")
    bg_remover = BackgroundRemover()
    
    source_nobg, mask1 = bg_remover.remove_background(source_image)
    mask1 = np.array(mask1) > 127
    
    target_nobg, mask2 = bg_remover.remove_background(target_image)
    mask2 = np.array(mask2) > 127
    
    # Save debug: background removed images and masks
    if debug:
        source_nobg.save(os.path.join(DEBUG_DIR, "05_source_nobg.png"))
        Image.fromarray((mask1 * 255).astype(np.uint8)).save(os.path.join(DEBUG_DIR, "06_source_mask.png"))
        target_nobg.save(os.path.join(DEBUG_DIR, "07_target_nobg.png"))
        Image.fromarray((mask2 * 255).astype(np.uint8)).save(os.path.join(DEBUG_DIR, "08_target_mask.png"))
    
    # === Prepare Data ===
    target_size = 1024
    # Ensure images are RGB before resizing
    source_nobg_rgb = source_nobg.convert('RGB')
    target_nobg_rgb = target_nobg.convert('RGB')
    
    source_nobg_np = np.array(source_nobg_rgb.resize((target_size, target_size)))
    target_nobg_np = np.array(target_nobg_rgb.resize((target_size, target_size)))
    
    # Resize background masks to match resized images
    mask1_resized = cv2.resize(mask1.astype(np.uint8), (target_size, target_size), interpolation=cv2.INTER_NEAREST) > 0
    mask2_resized = cv2.resize(mask2.astype(np.uint8), (target_size, target_size), interpolation=cv2.INTER_NEAREST) > 0
    
    # Prepare masks (resize to match resized images)
    source_shape = source_nobg_np.shape[:2]
    source_head_mask_np = np.array(source_head_mask.resize((target_size, target_size), Image.NEAREST)) > 127
    source_lmk_size = [source_lmk_data["image_height"], source_lmk_data["image_width"]]
    source_lmk_original = source_lmk_data['ldm478']
    
    target_shape = target_nobg_np.shape[:2]
    target_head_mask_np = np.array(target_head_mask.resize((target_size, target_size), Image.NEAREST)) > 127
    target_lmk_size = [target_lmk_data["image_height"], target_lmk_data["image_width"]]
    target_lmk_original = target_lmk_data['ldm478']
    
    # Resize landmarks
    source_lmk = resize_landmarks(source_lmk_original, source_lmk_size, source_shape)
    target_lmk = resize_landmarks(target_lmk_original, target_lmk_size, target_shape)
    
    print(f"Source shape: {source_shape}, Landmarks: {source_lmk.shape}")
    print(f"Target shape: {target_shape}, Landmarks: {target_lmk.shape}")
    
    # Save debug: resized images and landmarks visualization
    if debug:
        Image.fromarray(source_nobg_np).save(os.path.join(DEBUG_DIR, "09_source_resized.png"))
        Image.fromarray(target_nobg_np).save(os.path.join(DEBUG_DIR, "10_target_resized.png"))
        
        # Save landmarks visualization
        source_lmk_vis = source_nobg_np.copy()
        for x, y in source_lmk:
            cv2.circle(source_lmk_vis, (int(x), int(y)), 3, (255, 0, 0), -1)
        Image.fromarray(source_lmk_vis).save(os.path.join(DEBUG_DIR, "11_source_landmarks.png"))
        
        target_lmk_vis = target_nobg_np.copy()
        for x, y in target_lmk:
            cv2.circle(target_lmk_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
        Image.fromarray(target_lmk_vis).save(os.path.join(DEBUG_DIR, "12_target_landmarks.png"))
    
    # === Align Images ===
    print("\n=== Performing Alignment ===")
    aligned_target_image_np, aligned_mask2, aligned_head_mask2, params, final_iou = align_images_for_max_iou(
        source_nobg_np, mask1_resized, target_nobg_np, mask2_resized, source_head_mask_np, target_head_mask_np,
        source_lmk=source_lmk, target_lmk=target_lmk, iou_weight=1.0, landmark_weight=1.0
    )
    
    aligned_target_image = Image.fromarray(aligned_target_image_np)
    
    # Save debug: aligned images
    if debug:
        aligned_target_image.save(os.path.join(DEBUG_DIR, "13_aligned_target.png"))
        Image.fromarray((aligned_mask2 * 255).astype(np.uint8)).save(os.path.join(DEBUG_DIR, "14_aligned_mask.png"))
        Image.fromarray((aligned_head_mask2 * 255).astype(np.uint8)).save(os.path.join(DEBUG_DIR, "15_aligned_head_mask.png"))
    
    # Visualize aligned landmarks
    h, w = target_nobg_np.shape[:2]
    center_x, center_y = w / 2, h / 2
    aligned_target_lmk = transform_landmarks(target_lmk, params['scale_x'], params['scale_y'], 
                                             params['angle_rad'], params['tx'], params['ty'], 
                                             center_x, center_y)
    if debug:
        aligned_lmk_vis = aligned_target_image_np.copy()
        for x, y in aligned_target_lmk:
            cv2.circle(aligned_lmk_vis, (int(x), int(y)), 3, (0, 255, 0), -1)
        Image.fromarray(aligned_lmk_vis).save(os.path.join(DEBUG_DIR, "16_aligned_landmarks.png"))
        
        # Compare landmarks overlay
        overlay = source_nobg_np.copy()
        for x, y in source_lmk:
            cv2.circle(overlay, (int(x), int(y)), 3, (255, 0, 0), -1)
        for x, y in aligned_target_lmk:
            cv2.circle(overlay, (int(x), int(y)), 3, (0, 255, 0), -1)
        Image.fromarray(overlay).save(os.path.join(DEBUG_DIR, "17_landmarks_comparison.png"))
    
    # === Extract Hair Masks ===
    print("\nExtracting hair masks...")
    hair_mask_pipeline = HairMaskPipeline()
    
    aligned_mask_results = hair_mask_pipeline.preprocess(aligned_target_image)
    aligned_target_hair_mask = aligned_mask_results['hair_mask']
    
    mask_results = hair_mask_pipeline.preprocess(target_image)
    target_hair_mask = mask_results['hair_mask']
    
    # Save debug: hair masks
    if debug:
        Image.fromarray((np.array(aligned_target_hair_mask) * 255).astype(np.uint8)).save(os.path.join(DEBUG_DIR, "18_aligned_hair_mask.png"))
        Image.fromarray((np.array(target_hair_mask) * 255).astype(np.uint8)).save(os.path.join(DEBUG_DIR, "19_target_hair_mask.png"))
    
    # === Apply Transformation to Hair Only ===
    print("\nExtracting and transforming hair only...")
    # Resize target image to match the size used during optimization
    # Force RGB conversion to avoid RGBA issues
    target_image_rgb = target_image.convert('RGB')
    target_image_resized = target_image_rgb.resize((target_size, target_size))
    target_image_with_hair_np = np.array(target_image_resized)
    
    # Double check - ensure target image is RGB (not RGBA)
    if len(target_image_with_hair_np.shape) == 3 and target_image_with_hair_np.shape[2] > 3:
        target_image_with_hair_np = target_image_with_hair_np[:, :, :3]
    
    # Resize hair mask to match (target_hair_mask is already a numpy array)
    target_hair_mask_np = np.array(target_hair_mask)
    target_image_hair_mask_np = cv2.resize(target_hair_mask_np.astype(np.float32), (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    
    # Extract ONLY the hair region from target image (set non-hair to white/transparent)
    hair_mask_binary = target_image_hair_mask_np > 0.5
    hair_mask_3ch = np.stack([hair_mask_binary] * 3, axis=-1)
    
    # Create hair-only image (hair pixels + white background for non-hair areas)
    target_hair_only = np.where(hair_mask_3ch, target_image_with_hair_np, 255).astype(np.uint8)
    
    # Now transform ONLY the hair
    aligned_hair_only = apply_transformation(
        target_hair_only,
        scale_x=params['scale_x'],
        scale_y=params['scale_y'],
        angle=params['angle_rad'],
        tx=params['tx'],
        ty=params['ty']
    )
    
    # Transform the hair mask too
    aligned_hair_mask = apply_transformation(
        (target_image_hair_mask_np * 255).astype(np.uint8),
        scale_x=params['scale_x'],
        scale_y=params['scale_y'],
        angle=params['angle_rad'],
        tx=params['tx'],
        ty=params['ty']
    )
    
    # Save debug: transformed images
    if debug:
        Image.fromarray(aligned_hair_only).save(os.path.join(DEBUG_DIR, "20_aligned_hair_only.png"))
        Image.fromarray(aligned_hair_mask.astype(np.uint8)).save(os.path.join(DEBUG_DIR, "21_aligned_hair_mask_transformed.png"))
    
    # === Transfer Hair ===
    print("\nTransferring hair to source image...")
    result = transfer_hair(source_nobg_np, aligned_hair_only, aligned_hair_mask)
    
    # Save debug: final result and comparison
    if debug:
        Image.fromarray(result).save(os.path.join(DEBUG_DIR, "22_final_result.png"))
        
        # Create side-by-side comparison
        comparison = np.hstack([source_nobg_np, aligned_hair_only, result])
        Image.fromarray(comparison).save(os.path.join(DEBUG_DIR, "23_comparison.png"))
    
    # === Save Result ===
    output_path = os.path.join(VIEW_ALIGNED_DATA_DIR, "final.png")
    Image.fromarray(result).save(output_path)
    print(f"\nResult saved to: {output_path}")
    if debug:
        print(f"Debug images saved to: {DEBUG_DIR}")
    
    # === Print Summary ===
    print("\n" + "="*60)
    print("ALIGNMENT RESULTS SUMMARY")
    print("="*60)
    print(f"Final IoU: {params['iou']:.4f}")
    print(f"Landmark Distance: {params['landmark_distance']:.2f} pixels")
    print(f"Scale X: {params['scale_x']:.4f}x")
    print(f"Scale Y: {params['scale_y']:.4f}x")
    print(f"Rotation: {params['angle_deg']:.2f}°")
    print(f"Translation: ({params['tx']:.1f}, {params['ty']:.1f}) pixels")
    print("="*60)
    
    if visualize:
        # Visualize results
        h, w = target_nobg_np.shape[:2]
        center_x, center_y = w / 2, h / 2
        aligned_target_lmk = transform_landmarks(target_lmk, params['scale_x'], params['scale_y'], 
                                                 params['angle_rad'], params['tx'], params['ty'], 
                                                 center_x, center_y)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].imshow(source_nobg_np)
        axes[0, 0].set_title("Source (Bald)")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(target_nobg_np)
        axes[0, 1].set_title("Target (Original)")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(aligned_target_image_np)
        axes[0, 2].set_title("Target (Aligned)")
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(source_nobg_np)
        axes[1, 0].set_title("Source (Bald)")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(aligned_hair_only)
        axes[1, 1].set_title("Aligned Hair Only")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(result)
        axes[1, 2].set_title("Final Result")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return result, params


if __name__ == "__main__":
    # Process all subdirectories in the data directory
    data_dir = "/localhome/aaa324/Project/MultiView/pixel3dmm/io/outputs/"
    
    # Get all subdirectories in the bald/wo_seg directory
    bald_dir = os.path.join(data_dir, "bald", "wo_seg", "image")
    
    if not os.path.exists(bald_dir):
        print(f"Error: Directory not found: {bald_dir}")
        sys.exit(1)
    
    # Get all bald image IDs (source images)
    bald_image_files = [f for f in os.listdir(bald_dir) if f.endswith('.png')]
    source_ids = [os.path.splitext(f)[0] for f in bald_image_files]
    
    # Get all refined directories (format: refined_w_flux/{target_id}_to_{source_id})
    refined_base_dir = os.path.join(data_dir, "refined_w_flux")
    
    if not os.path.exists(refined_base_dir):
        print(f"Error: Directory not found: {refined_base_dir}")
        sys.exit(1)
    
    refined_subdirs = [d for d in os.listdir(refined_base_dir) 
                      if os.path.isdir(os.path.join(refined_base_dir, d)) and "_to_" in d]
    
    print(f"Found {len(refined_subdirs)} directories to process")
    print(f"Found {len(source_ids)} source (bald) images")
    
    # Track success and failures
    successful = []
    failed = []
    
    # Process each refined directory
    for subdir in refined_subdirs:
        try:
            # Parse target_id and source_id from directory name
            # Format: {target_id}_to_{source_id}
            parts = subdir.split("_to_")
            if len(parts) != 2:
                print(f"Skipping invalid directory name: {subdir}")
                continue
            
            target_id = parts[0]
            source_id = parts[1]
            
            print(f"\n{'='*80}")
            print(f"Processing: {target_id} -> {source_id}")
            print(f"{'='*80}")
            
            # Run the main function (debug=False to disable debug image saving)
            result, params = main(source_id, target_id, data_dir, visualize=False, debug=False)
            
            successful.append((target_id, source_id))
            print(f"✓ Successfully processed: {target_id} -> {source_id}")
            
        except Exception as e:
            failed.append((subdir, str(e)))
            print(f"✗ Failed to process {subdir}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total directories: {len(refined_subdirs)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✓ Successfully processed:")
        for target_id, source_id in successful:
            print(f"  - {target_id} -> {source_id}")
    
    if failed:
        print(f"\n✗ Failed:")
        for subdir, error in failed:
            print(f"  - {subdir}: {error}")
    
    print(f"\n{'='*80}")
    print("Hair transfer batch processing complete!")
