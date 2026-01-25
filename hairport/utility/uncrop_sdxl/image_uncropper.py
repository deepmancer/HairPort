"""
Image Uncropping Utility Module

This module provides a reusable class for cropping and uncropping images
based on resize percentages, with support for dynamic resize computation
using facial landmarks.
"""

import json
import os
from typing import Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.distance import pdist


class ImageUncropper:
    """
    A class for cropping and uncropping images based on resize percentages.
    
    This class handles the logic of placing an image on a canvas with margins
    (uncropping) and extracting the original region from an uncropped image
    (cropping). It also supports computing dynamic resize percentages based
    on facial landmarks to ensure consistent face sizes across images.
    
    Attributes:
        target_width (int): Target output width.
        target_height (int): Target output height.
        alignment (str): Alignment of the image within the canvas.
        overlap_percentage (int): Percentage of overlap for mask blending.
        overlap_left (bool): Whether to blend on the left edge.
        overlap_right (bool): Whether to blend on the right edge.
        overlap_top (bool): Whether to blend on the top edge.
        overlap_bottom (bool): Whether to blend on the bottom edge.
    """
    
    def __init__(
        self,
        target_width: int = 1024,
        target_height: int = 1024,
        alignment: str = "Middle",
        overlap_percentage: int = 5,
        overlap_left: bool = True,
        overlap_right: bool = True,
        overlap_top: bool = True,
        overlap_bottom: bool = True,
    ):
        """
        Initialize the ImageUncropper.
        
        Args:
            target_width: Target output width in pixels.
            target_height: Target output height in pixels.
            alignment: Alignment of image on canvas ("Middle", "Left", "Right", "Top", "Bottom").
            overlap_percentage: Percentage of overlap for seamless blending.
            overlap_left: Whether to allow blending on the left edge.
            overlap_right: Whether to allow blending on the right edge.
            overlap_top: Whether to allow blending on the top edge.
            overlap_bottom: Whether to allow blending on the bottom edge.
        """
        self.target_width = target_width
        self.target_height = target_height
        self.alignment = alignment
        self.overlap_percentage = overlap_percentage
        self.overlap_left = overlap_left
        self.overlap_right = overlap_right
        self.overlap_top = overlap_top
        self.overlap_bottom = overlap_bottom
    
    @staticmethod
    def compute_face_size_from_landmarks(landmarks: np.ndarray) -> float:
        """
        Compute the maximum distance among any two facial landmarks.
        
        This represents the "size" of the face in pixel coordinates.
        
        Args:
            landmarks: Array of shape (68, 2) containing facial landmark coordinates.
            
        Returns:
            Maximum pairwise distance between any two landmarks.
        """
        if landmarks.shape[0] < 2:
            raise ValueError("Need at least 2 landmarks to compute face size")
        
        # Compute all pairwise distances and return the maximum
        distances = pdist(landmarks)
        return float(np.max(distances))
    
    @staticmethod
    def compute_resize_percentage(
        face_size_original: float,
        target_width: int,
        face_to_width_ratio: float,
        min_percentage: float = 10.0,
        max_percentage: float = 100.0,
    ) -> float:
        """
        Compute the resize percentage needed to achieve the desired face-to-width ratio.
        
        Args:
            face_size_original: The face size in the original image (in pixels).
            target_width: The target output width (in pixels).
            face_to_width_ratio: Desired ratio of face size to image width (e.g., 0.2).
            min_percentage: Minimum allowed resize percentage.
            max_percentage: Maximum allowed resize percentage.
            
        Returns:
            The computed resize percentage, clamped to [min_percentage, max_percentage].
        """
        if face_size_original <= 0:
            raise ValueError("Face size must be positive")
        
        # Formula: custom_resize_percentage = (target_width * face_to_width_ratio / face_size_original) * 100
        desired_face_size = target_width * face_to_width_ratio
        resize_percentage = (desired_face_size / face_size_original) * 100
        
        # Clamp to valid range
        resize_percentage = max(min_percentage, min(max_percentage, resize_percentage))
        
        return resize_percentage
    
    def compute_resize_info(
        self,
        image: Image.Image,
        resize_percentage: float,
        landmark_offset_x: Optional[int] = None,
        landmark_offset_y: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute the resize and placement information for an image.
        
        Args:
            image: Input PIL Image.
            resize_percentage: Percentage to resize the image (0-100).
            landmark_offset_x: Optional X offset for landmark-based alignment.
            landmark_offset_y: Optional Y offset for landmark-based alignment.
            
        Returns:
            Dictionary containing:
                - new_width: Width after resizing
                - new_height: Height after resizing
                - margin_x: X offset on the canvas
                - margin_y: Y offset on the canvas
                - scale_factor: Initial scale factor to fit target size
                - resize_factor: Final resize factor from percentage
                - landmark_offset_x: X offset for landmark alignment (or None)
                - landmark_offset_y: Y offset for landmark alignment (or None)
        """
        target_size = (self.target_width, self.target_height)
        
        # Calculate the scaling factor to fit the image within the target size
        scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
        scaled_width = int(image.width * scale_factor)
        scaled_height = int(image.height * scale_factor)
        
        # Apply resize percentage
        resize_factor = resize_percentage / 100
        new_width = int(scaled_width * resize_factor)
        new_height = int(scaled_height * resize_factor)
        
        # Calculate margins based on alignment (or use custom offsets)
        if landmark_offset_x is not None and landmark_offset_y is not None:
            # Use landmark-based placement (asymmetric)
            # Start with centered placement and apply offset
            base_margin_x = (target_size[0] - new_width) // 2
            base_margin_y = (target_size[1] - new_height) // 2
            margin_x = base_margin_x + landmark_offset_x
            margin_y = base_margin_y + landmark_offset_y
        else:
            # Use alignment-based placement (symmetric)
            if self.alignment == "Middle":
                margin_x = (target_size[0] - new_width) // 2
                margin_y = (target_size[1] - new_height) // 2
            elif self.alignment == "Left":
                margin_x = 0
                margin_y = (target_size[1] - new_height) // 2
            elif self.alignment == "Right":
                margin_x = target_size[0] - new_width
                margin_y = (target_size[1] - new_height) // 2
            elif self.alignment == "Top":
                margin_x = (target_size[0] - new_width) // 2
                margin_y = 0
            elif self.alignment == "Bottom":
                margin_x = (target_size[0] - new_width) // 2
                margin_y = target_size[1] - new_height
            else:
                margin_x = (target_size[0] - new_width) // 2
                margin_y = (target_size[1] - new_height) // 2
        
        # Adjust margins to eliminate gaps and ensure image stays within canvas
        margin_x = max(0, min(margin_x, target_size[0] - new_width))
        margin_y = max(0, min(margin_y, target_size[1] - new_height))
        
        return {
            "new_width": new_width,
            "new_height": new_height,
            "margin_x": margin_x,
            "margin_y": margin_y,
            "scale_factor": scale_factor,
            "resize_factor": resize_factor,
            "scaled_width": scaled_width,
            "scaled_height": scaled_height,
            "landmark_offset_x": landmark_offset_x,
            "landmark_offset_y": landmark_offset_y,
        }
    
    def prepare_for_uncrop(
        self,
        image: Image.Image,
        resize_percentage: float,
        landmark_offset_x: Optional[int] = None,
        landmark_offset_y: Optional[int] = None,
    ) -> Tuple[Image.Image, Image.Image, Dict[str, Any]]:
        """
        Prepare an image for uncropping by placing it on a canvas with margins.
        
        Args:
            image: Input PIL Image.
            resize_percentage: Percentage to resize the image (0-100).
            landmark_offset_x: Optional X offset for landmark-based alignment.
            landmark_offset_y: Optional Y offset for landmark-based alignment.
            
        Returns:
            Tuple of (background_image, mask, resize_info):
                - background_image: The image placed on a white canvas
                - mask: The inpainting mask (white = areas to generate, black = keep)
                - resize_info: Dictionary with resize/placement information
        """
        target_size = (self.target_width, self.target_height)
        resize_info = self.compute_resize_info(
            image, resize_percentage, landmark_offset_x, landmark_offset_y
        )
        
        new_width = resize_info["new_width"]
        new_height = resize_info["new_height"]
        margin_x = resize_info["margin_x"]
        margin_y = resize_info["margin_y"]
        
        # First resize to fit target, then apply percentage
        scale_factor = resize_info["scale_factor"]
        scaled = image.resize(
            (resize_info["scaled_width"], resize_info["scaled_height"]),
            Image.LANCZOS
        )
        source = scaled.resize((new_width, new_height), Image.LANCZOS)
        
        # Create background and paste
        background = Image.new('RGB', target_size, (255, 255, 255))
        background.paste(source, (margin_x, margin_y))
        
        # Create the mask
        mask = Image.new('L', target_size, 255)
        mask_draw = ImageDraw.Draw(mask)
        
        # Calculate overlap areas
        overlap_x = int(new_width * (self.overlap_percentage / 100))
        overlap_y = int(new_height * (self.overlap_percentage / 100))
        overlap_x = max(overlap_x, 1)
        overlap_y = max(overlap_y, 1)
        
        white_gaps_patch = 2
        
        left_overlap = margin_x + overlap_x if self.overlap_left else margin_x + white_gaps_patch
        right_overlap = margin_x + new_width - overlap_x if self.overlap_right else margin_x + new_width - white_gaps_patch
        top_overlap = margin_y + overlap_y if self.overlap_top else margin_y + white_gaps_patch
        bottom_overlap = margin_y + new_height - overlap_y if self.overlap_bottom else margin_y + new_height - white_gaps_patch
        
        # Only apply edge-specific overlap adjustments for alignment-based placement
        # (not for landmark-based asymmetric placement)
        if landmark_offset_x is None and landmark_offset_y is None:
            if self.alignment == "Left":
                left_overlap = margin_x + overlap_x if self.overlap_left else margin_x
            elif self.alignment == "Right":
                right_overlap = margin_x + new_width - overlap_x if self.overlap_right else margin_x + new_width
            elif self.alignment == "Top":
                top_overlap = margin_y + overlap_y if self.overlap_top else margin_y
            elif self.alignment == "Bottom":
                bottom_overlap = margin_y + new_height - overlap_y if self.overlap_bottom else margin_y + new_height
        
        # Draw the mask (black rectangle where original image is)
        mask_draw.rectangle([
            (left_overlap, top_overlap),
            (right_overlap, bottom_overlap)
        ], fill=0)
        
        return background, mask, resize_info
    
    def crop_from_uncropped(
        self,
        uncropped_image: Image.Image,
        resize_info: Dict[str, Any],
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """
        Extract the original image region from an uncropped image.
        
        This is the reverse operation of uncropping - given an uncropped image
        and the resize information, extract the region where the original
        image was placed and optionally resize it.
        
        Args:
            uncropped_image: The uncropped PIL Image (with generated margins).
            resize_info: Dictionary with resize/placement info from prepare_for_uncrop.
            output_size: Optional tuple (width, height) to resize the output.
                        If None, returns at the size it was placed on canvas.
            
        Returns:
            The extracted image region, optionally resized.
        """
        margin_x = resize_info["margin_x"]
        margin_y = resize_info["margin_y"]
        new_width = resize_info["new_width"]
        new_height = resize_info["new_height"]
        
        # Extract the region
        cropped = uncropped_image.crop((
            margin_x,
            margin_y,
            margin_x + new_width,
            margin_y + new_height
        ))
        
        # Optionally resize to output size
        if output_size is not None:
            cropped = cropped.resize(output_size, Image.LANCZOS)
        
        return cropped
    
    def save_resize_info(
        self,
        resize_info: Dict[str, Any],
        output_path: str,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save resize information to a JSON file.
        
        Args:
            resize_info: Dictionary with resize/placement information.
            output_path: Path to save the JSON file.
            extra_info: Optional additional information to include.
        """
        save_data = {
            "resize_info": resize_info,
            "uncropper_config": {
                "target_width": self.target_width,
                "target_height": self.target_height,
                "alignment": self.alignment,
                "overlap_percentage": self.overlap_percentage,
            }
        }
        
        if extra_info:
            save_data["extra_info"] = extra_info
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    @classmethod
    def load_resize_info(cls, json_path: str) -> Tuple['ImageUncropper', Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Load resize information and recreate an ImageUncropper instance.
        
        Args:
            json_path: Path to the JSON file saved by save_resize_info.
            
        Returns:
            Tuple of (uncropper_instance, resize_info, extra_info).
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        config = data.get("uncropper_config", {})
        uncropper = cls(
            target_width=config.get("target_width", 1024),
            target_height=config.get("target_height", 1024),
            alignment=config.get("alignment", "Middle"),
            overlap_percentage=config.get("overlap_percentage", 5),
        )
        
        resize_info = data.get("resize_info", {})
        extra_info = data.get("extra_info")
        
        return uncropper, resize_info, extra_info


def load_landmarks(landmark_path: str) -> np.ndarray:
    """
    Load facial landmarks from a numpy file.
    
    Args:
        landmark_path: Path to the .npy file containing landmarks.
        
    Returns:
        Array of shape (68, 2) containing the ldm68 landmarks.
    """
    data = np.load(landmark_path, allow_pickle=True).item()
    landmarks = data.get("ldm68")
    
    if landmarks is None:
        raise ValueError(f"No 'ldm68' key found in {landmark_path}")
    
    return np.array(landmarks)


def compute_landmark_centroid(landmarks: np.ndarray) -> Tuple[float, float]:
    """
    Compute the centroid (mean position) of facial landmarks.
    
    Args:
        landmarks: Array of shape (N, 2) containing landmark coordinates.
        
    Returns:
        Tuple of (centroid_x, centroid_y).
    """
    if landmarks.shape[0] < 1:
        raise ValueError("Need at least 1 landmark to compute centroid")
    
    centroid = np.mean(landmarks, axis=0)
    return float(centroid[0]), float(centroid[1])


def compute_alignment_offset(
    source_landmarks: np.ndarray,
    target_landmarks: np.ndarray,
    target_resize_factor: float,
    source_image_width: int,
    source_image_height: int,
    target_image_width: int,
    target_image_height: int,
    target_canvas_width: int = 1024,
    target_canvas_height: int = 1024,
    source_resize_percentage: float = 100.0,
) -> Tuple[int, int]:
    """
    Compute the offset needed to align target landmarks with source landmarks.
    
    This function assumes:
    - Source landmarks are in the coordinate space of the source image
    - Target landmarks are in the coordinate space of the target image (before resizing)
    - The target image will be resized by target_resize_factor
    - The source image was resized by source_resize_percentage when uncropped
    - Both images should be placed on canvases of the same size
    
    Args:
        source_landmarks: Source image landmarks (N, 2).
        target_landmarks: Target image landmarks (N, 2).
        target_resize_factor: The factor by which the target will be resized (scale_factor * resize_percentage / 100).
        source_image_width: Width of the source image.
        source_image_height: Height of the source image.
        target_image_width: Width of the target image (before resizing).
        target_image_height: Height of the target image (before resizing).
        target_canvas_width: Width of the canvas (default 1024).
        target_canvas_height: Height of the canvas (default 1024).
        source_resize_percentage: The resize percentage used when uncropping the source (default 100.0).
        
    Returns:
        Tuple of (offset_x, offset_y) to be added to the default centered placement.
    """
    # Compute centroids in original image coordinates
    source_centroid_x, source_centroid_y = compute_landmark_centroid(source_landmarks)
    target_centroid_x, target_centroid_y = compute_landmark_centroid(target_landmarks)
    
    # Compute where the source centroid ended up on the canvas
    # Account for both the scale-to-fit and the resize percentage
    source_scale = min(target_canvas_width / source_image_width, target_canvas_height / source_image_height)
    source_resize_factor = source_scale * (source_resize_percentage / 100.0)
    source_scaled_width = int(source_image_width * source_scale)
    source_scaled_height = int(source_image_height * source_scale)
    source_final_width = int(source_scaled_width * (source_resize_percentage / 100.0))
    source_final_height = int(source_scaled_height * (source_resize_percentage / 100.0))
    
    source_margin_x = (target_canvas_width - source_final_width) // 2
    source_margin_y = (target_canvas_height - source_final_height) // 2
    
    # Adjust margins to keep image within canvas (matching _prepare_image_and_mask logic)
    source_margin_x = max(0, min(source_margin_x, target_canvas_width - source_final_width))
    source_margin_y = max(0, min(source_margin_y, target_canvas_height - source_final_height))
    
    # Source centroid position on canvas (landmarks are scaled by the full resize factor)
    source_canvas_centroid_x = source_margin_x + source_centroid_x * source_resize_factor
    source_canvas_centroid_y = source_margin_y + source_centroid_y * source_resize_factor
    
    # Compute where the target centroid will be after resizing
    target_centroid_after_resize_x = target_centroid_x * target_resize_factor
    target_centroid_after_resize_y = target_centroid_y * target_resize_factor
    
    # Compute target image dimensions after resizing
    # IMPORTANT: Match the integer truncation order used in _prepare_image_and_mask and compute_resize_info
    # They do: int(int(width * scale) * resize_factor), not int(width * scale * resize_factor)
    target_scale = min(target_canvas_width / target_image_width, target_canvas_height / target_image_height)
    target_scaled_width = int(target_image_width * target_scale)
    target_scaled_height = int(target_image_height * target_scale)
    # Recover resize_factor from target_resize_factor (which is scale * resize_percentage / 100)
    target_resize_factor_normalized = target_resize_factor / target_scale  # This is resize_percentage / 100
    target_final_width = int(target_scaled_width * target_resize_factor_normalized)
    target_final_height = int(target_scaled_height * target_resize_factor_normalized)
    
    # Default centered placement
    default_margin_x = (target_canvas_width - target_final_width) // 2
    default_margin_y = (target_canvas_height - target_final_height) // 2
    
    # Adjust margins to keep image within canvas (matching _prepare_image_and_mask logic)
    default_margin_x = max(0, min(default_margin_x, target_canvas_width - target_final_width))
    default_margin_y = max(0, min(default_margin_y, target_canvas_height - target_final_height))
    
    # Target centroid position with default centered placement
    default_canvas_centroid_x = default_margin_x + target_centroid_after_resize_x
    default_canvas_centroid_y = default_margin_y + target_centroid_after_resize_y
    
    # Compute offset needed to align centroids
    offset_x = int(source_canvas_centroid_x - default_canvas_centroid_x)
    offset_y = int(source_canvas_centroid_y - default_canvas_centroid_y)
    
    return offset_x, offset_y


def compute_dynamic_resize_percentage(
    landmark_path: str,
    target_width: int = 1024,
    face_to_width_ratio: float = 0.2,
    min_percentage: float = 10.0,
    max_percentage: float = 100.0,
) -> Tuple[float, float]:
    """
    Compute the dynamic resize percentage based on facial landmarks.
    
    Args:
        landmark_path: Path to the landmarks.npy file.
        target_width: Target output width.
        face_to_width_ratio: Desired ratio of face size to image width.
        min_percentage: Minimum allowed resize percentage.
        max_percentage: Maximum allowed resize percentage.
        
    Returns:
        Tuple of (resize_percentage, face_size_original).
    """
    landmarks = load_landmarks(landmark_path)
    face_size = ImageUncropper.compute_face_size_from_landmarks(landmarks)
    resize_percentage = ImageUncropper.compute_resize_percentage(
        face_size,
        target_width,
        face_to_width_ratio,
        min_percentage,
        max_percentage,
    )
    
    return resize_percentage, face_size
