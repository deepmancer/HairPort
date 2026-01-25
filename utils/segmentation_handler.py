"""
Segmentation Handler Module

This module provides efficient operations for working with colored segmentation images,
including extracting binary masks for specific regions and handling class combinations.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List, Tuple, Optional, Dict
import cv2
from config.segmentation_config import (
    SegmentationConfig, ExtendedLIPSegmentationConfig, LIPSegmentationConfig,
    ExtendedLIPClass, LIPClass, ClassGroups
)


class SegmentationHandler:
    """
    Handler class for working with colored segmentation images.
    Provides efficient operations to extract binary masks for specific regions.
    """
    
    def __init__(self, config: SegmentationConfig = None):
        """
        Initialize the segmentation handler.
        
        Args:
            config: Segmentation configuration. Defaults to ExtendedLIPSegmentationConfig.
        """
        self.config = config or ExtendedLIPSegmentationConfig()
        self._colored_seg = None
        self._class_map = None  # Cache for class ID mapping
        
    def load_colored_segmentation(self, 
                                  segmentation: Union[str, np.ndarray, Image.Image]) -> 'SegmentationHandler':
        """
        Load a colored segmentation image.
        
        Args:
            segmentation: Path to image file, PIL Image, or numpy array (H,W,3)
            
        Returns:
            Self for method chaining
        """
        if isinstance(segmentation, str):
            if not Path(segmentation).exists():
                raise FileNotFoundError(f"Segmentation file not found: {segmentation}")
            self._colored_seg = np.array(Image.open(segmentation).convert('RGB'))
        elif isinstance(segmentation, Image.Image):
            self._colored_seg = np.array(segmentation.convert('RGB'))
        elif isinstance(segmentation, np.ndarray):
            if segmentation.ndim == 3 and segmentation.shape[2] == 3:
                self._colored_seg = segmentation.astype(np.uint8)
            elif segmentation.ndim == 2:
                # Handle grayscale by converting to RGB
                self._colored_seg = np.stack([segmentation] * 3, axis=2).astype(np.uint8)
            else:
                raise ValueError("Numpy array must be (H,W,3) RGB format or (H,W) grayscale")
        else:
            raise ValueError("Segmentation must be path, PIL Image, or numpy array")
        
        # Validate the loaded image
        if self._colored_seg.size == 0:
            raise ValueError("Loaded segmentation image is empty")
        
        # Clear cache
        self._class_map = None
        return self
        
    def _ensure_loaded(self):
        """Ensure segmentation is loaded"""
        if self._colored_seg is None:
            raise RuntimeError("No segmentation loaded. Call load_colored_segmentation() first.")
    
    def _get_class_map(self) -> np.ndarray:
        """
        Convert colored segmentation to class ID map (cached).
        Uses nearest color matching to handle compression artifacts and slight color variations.
        
        Returns:
            Class ID array of shape (H, W)
        """
        if self._class_map is None:
            self._ensure_loaded()
            h, w = self._colored_seg.shape[:2]
            
            # Convert to float for distance calculations
            img_flat = self._colored_seg.reshape(-1, 3).astype(np.float32)
            
            # Get all colors and IDs from config
            colors = np.array(list(self.config._id_to_color.values())).astype(np.float32)
            class_ids = np.array(list(self.config._id_to_color.keys()))
            
            # Compute distances from each pixel to all colors
            # Using squared euclidean distance for efficiency (no sqrt needed for argmin)
            distances = np.sum((img_flat[:, None, :] - colors[None, :, :]) ** 2, axis=2)
            
            # Find closest color for each pixel
            closest_indices = np.argmin(distances, axis=1)
            
            # Map to class IDs and reshape
            self._class_map = class_ids[closest_indices].reshape(h, w).astype(np.uint8)
                
        return self._class_map
    
    def get_binary_mask(self, 
                       classes: Union[int, ExtendedLIPClass, LIPClass, List[Union[int, ExtendedLIPClass, LIPClass]]], 
                       output_format: str = 'numpy') -> Union[np.ndarray, Image.Image]:
        """
        Extract binary mask for specified class(es).
        
        Args:
            classes: Single class ID/enum or list of class IDs/enums
            output_format: 'numpy' for numpy array, 'pil' for PIL Image
            
        Returns:
            Binary mask (255 for class pixels, 0 for others)
        """
        self._ensure_loaded()
        class_map = self._get_class_map()
        
        # Convert to list of integers
        if not isinstance(classes, list):
            classes = [classes]
        
        class_ids = []
        for cls in classes:
            if isinstance(cls, (ExtendedLIPClass, LIPClass)):
                class_ids.append(int(cls))
            elif isinstance(cls, int):
                class_ids.append(cls)
            else:
                raise ValueError(f"Invalid class type: {type(cls)}")
        
        # Create binary mask
        mask = np.zeros_like(class_map, dtype=np.uint8)
        for class_id in class_ids:
            mask[class_map == class_id] = 255
            
        if output_format == 'pil':
            return Image.fromarray(mask)
        elif output_format == 'numpy':
            return mask
        else:
            raise ValueError("output_format must be 'numpy' or 'pil'")
    
    def get_hair_mask(self, output_format: str = 'numpy') -> Union[np.ndarray, Image.Image]:
        """Get binary mask for hair region."""
        return self.get_binary_mask(ClassGroups.HAIR, output_format)
    
    def get_face_mask(self, include_features: bool = True, output_format: str = 'numpy') -> Union[np.ndarray, Image.Image]:
        """
        Get binary mask for face region.
        
        Args:
            include_features: If True, includes detailed facial features (eyes, nose, mouth, etc.)
            output_format: 'numpy' or 'pil'
        """
        if include_features:
            classes = ClassGroups.FACE_ALL
        else:
            classes = [ExtendedLIPClass.FACE]
        return self.get_binary_mask(classes, output_format)
    
    def get_body_mask(self, include_face: bool = True, output_format: str = 'numpy') -> Union[np.ndarray, Image.Image]:
        """
        Get binary mask for body region (traditional body mask).
        
        Args:
            include_face: If True, includes face and hair
            output_format: 'numpy' or 'pil'
        """
        if include_face:
            classes = ClassGroups.BODY_AND_HAIR
        else:
            classes = ClassGroups.BODY_LIMBS
        return self.get_binary_mask(classes, output_format)
    
    def get_clothing_mask(self, output_format: str = 'numpy') -> Union[np.ndarray, Image.Image]:
        """Get binary mask for clothing regions."""
        return self.get_binary_mask(ClassGroups.CLOTHING, output_format)
    
    def get_combined_mask(self, 
                         class_groups: List[str], 
                         output_format: str = 'numpy') -> Union[np.ndarray, Image.Image]:
        """
        Get combined binary mask from multiple predefined class groups.
        
        Args:
            class_groups: List of group names ('hair', 'face', 'body', 'clothing', 'accessories')
            output_format: 'numpy' or 'pil'
        """
        group_mapping = {
            'hair': ClassGroups.HAIR,
            'face': ClassGroups.FACE_ALL,
            'facial_features': ClassGroups.FACIAL_FEATURES,
            'body': ClassGroups.BODY_LIMBS,
            'clothing': ClassGroups.CLOTHING,
            'accessories': ClassGroups.ACCESSORIES,
            'human': ClassGroups.HUMAN_ALL,
            'body_generic': ClassGroups.BODY_GENERIC,
        }
        
        combined_classes = []
        for group_name in class_groups:
            if group_name in group_mapping:
                combined_classes.extend(group_mapping[group_name])
            else:
                available_groups = list(group_mapping.keys())
                raise ValueError(f"Unknown group '{group_name}'. Available groups: {available_groups}")
        
        # Remove duplicates
        combined_classes = list(set(combined_classes))
        return self.get_binary_mask(combined_classes, output_format)
    
    def get_class_statistics(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Get statistics about class distribution in the segmentation.
        
        Returns:
            Dictionary with class statistics
        """
        self._ensure_loaded()
        class_map = self._get_class_map()
        
        total_pixels = class_map.size
        unique_classes, counts = np.unique(class_map, return_counts=True)
        
        stats = {}
        for class_id, count in zip(unique_classes, counts):
            class_name = self.config.get_class_name(class_id)
            stats[class_name] = {
                'class_id': int(class_id),
                'pixel_count': int(count),
                'percentage': float(count / total_pixels * 100),
                'color': self.config.get_color(class_id)
            }
        
        return stats
    
    def visualize_classes(self, 
                         classes: Union[int, List[int]] = None, 
                         background_alpha: float = 0.3) -> Image.Image:
        """
        Visualize specific classes by dimming others.
        
        Args:
            classes: Class IDs to highlight (None for all classes)
            background_alpha: Alpha for dimmed regions (0.0 = black, 1.0 = original)
            
        Returns:
            PIL Image with highlighted classes
        """
        self._ensure_loaded()
        
        if classes is None:
            # Show all classes as-is
            return Image.fromarray(self._colored_seg)
        
        if not isinstance(classes, list):
            classes = [classes]
        
        # Create highlight mask
        highlight_mask = self.get_binary_mask(classes, 'numpy') > 0
        
        # Create output image
        output = self._colored_seg.copy().astype(np.float32)
        
        # Dim non-highlighted regions
        output[~highlight_mask] = output[~highlight_mask] * background_alpha
        
        return Image.fromarray(output.astype(np.uint8))
    
    def save_masks(self, 
                   output_dir: str, 
                   prefix: str = '', 
                   save_individual: bool = True,
                   save_combined: bool = True) -> Dict[str, str]:
        """
        Save various masks to disk.
        
        Args:
            output_dir: Directory to save masks
            prefix: Prefix for filenames
            save_individual: Whether to save individual class masks
            save_combined: Whether to save combined group masks
            
        Returns:
            Dictionary mapping mask names to saved file paths
        """
        from pathlib import Path
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        if save_combined:
            # Save common combined masks
            masks_to_save = {
                'hair': self.get_hair_mask(),
                'face_all': self.get_face_mask(include_features=True),
                'face_basic': self.get_face_mask(include_features=False),
                'body_all': self.get_body_mask(include_face=True),
                'body_limbs': self.get_body_mask(include_face=False),
                'clothing': self.get_clothing_mask(),
            }
            
            for mask_name, mask in masks_to_save.items():
                filename = f"{prefix}{mask_name}_mask.png" if prefix else f"{mask_name}_mask.png"
                filepath = output_dir / filename
                Image.fromarray(mask).save(filepath)
                saved_files[mask_name] = str(filepath)
        
        if save_individual:
            # Save individual class masks
            class_map = self._get_class_map()
            unique_classes = np.unique(class_map)
            
            for class_id in unique_classes:
                if class_id == 0:  # Skip background
                    continue
                    
                class_name = self.config.get_class_name(class_id)
                mask = self.get_binary_mask(class_id)
                filename = f"{prefix}{class_name}_mask.png" if prefix else f"{class_name}_mask.png"
                filepath = output_dir / filename
                Image.fromarray(mask).save(filepath)
                saved_files[f"individual_{class_name}"] = str(filepath)
        
        return saved_files
    
    def get_image_shape(self) -> Optional[Tuple[int, int]]:
        """Get shape of loaded segmentation image (H, W)"""
        if self._colored_seg is not None:
            return self._colored_seg.shape[:2]
        return None
    
    def get_available_classes(self) -> List[Tuple[int, str, Tuple[int, int, int]]]:
        """
        Get list of classes present in the loaded segmentation.
        
        Returns:
            List of tuples (class_id, class_name, color)
        """
        if self._colored_seg is None:
            return []
        
        class_map = self._get_class_map()
        unique_classes = np.unique(class_map)
        
        result = []
        for class_id in unique_classes:
            class_name = self.config.get_class_name(class_id)
            color = self.config.get_color(class_id)
            result.append((int(class_id), class_name, color))
        
        return result
    
    def debug_color_matching(self, max_colors: int = 50) -> Dict:
        """
        Debug method to analyze color distribution in the segmentation.
        Helps identify color matching issues.
        
        Args:
            max_colors: Maximum number of unique colors to analyze
            
        Returns:
            Dictionary with color analysis information
        """
        if self._colored_seg is None:
            return {"error": "No segmentation loaded"}
        
        # Get unique colors in the image
        pixels = self._colored_seg.reshape(-1, 3)
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        
        # Sort by frequency
        sort_indices = np.argsort(counts)[::-1]
        unique_colors = unique_colors[sort_indices][:max_colors]
        counts = counts[sort_indices][:max_colors]
        
        # Find closest config color for each unique color
        config_colors = np.array(list(self.config._id_to_color.values())).astype(np.float32)
        config_ids = list(self.config._id_to_color.keys())
        
        analysis = {
            "total_unique_colors": len(unique_colors),
            "image_shape": self._colored_seg.shape,
            "color_analysis": []
        }
        
        for i, (color, count) in enumerate(zip(unique_colors, counts)):
            # Find closest config color
            distances = np.sum((config_colors - color.astype(np.float32)) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            closest_distance = np.sqrt(distances[closest_idx])
            
            analysis["color_analysis"].append({
                "rank": i + 1,
                "color": tuple(map(int, color)),
                "pixel_count": int(count),
                "percentage": float(count / pixels.shape[0] * 100),
                "closest_config_color": self.config.get_color(config_ids[closest_idx]),
                "closest_class_name": self.config.get_class_name(config_ids[closest_idx]),
                "distance": float(closest_distance),
                "exact_match": closest_distance < 1.0
            })
        
        return analysis


def create_segmentation_handler(config_type: str = 'extended_lip') -> SegmentationHandler:
    """
    Convenience function to create a segmentation handler with predefined config.
    
    Args:
        config_type: 'lip' or 'extended_lip'
        
    Returns:
        SegmentationHandler instance
    """
    if config_type == 'lip':
        return SegmentationHandler(LIPSegmentationConfig())
    elif config_type == 'extended_lip':
        return SegmentationHandler(ExtendedLIPSegmentationConfig())
    else:
        raise ValueError("config_type must be 'lip' or 'extended_lip'")


# Example usage functions
def extract_hair_mask(segmentation_path: str, output_path: str = None) -> np.ndarray:
    """
    Quick function to extract hair mask from colored segmentation.
    
    Args:
        segmentation_path: Path to colored segmentation image
        output_path: Optional path to save the mask
        
    Returns:
        Hair mask as numpy array
    """
    handler = create_segmentation_handler('extended_lip')
    handler.load_colored_segmentation(segmentation_path)
    mask = handler.get_hair_mask()
    
    if output_path:
        Image.fromarray(mask).save(output_path)
    
    return mask


def extract_custom_mask(segmentation_path: str, 
                       classes: List[Union[int, str]], 
                       output_path: str = None) -> np.ndarray:
    """
    Quick function to extract custom mask from colored segmentation.
    
    Args:
        segmentation_path: Path to colored segmentation image
        classes: List of class IDs or names
        output_path: Optional path to save the mask
        
    Returns:
        Custom mask as numpy array
    """
    try:
        handler = create_segmentation_handler('extended_lip')
        handler.load_colored_segmentation(segmentation_path)
        
        # Convert class names to IDs if needed
        class_ids = []
        for cls in classes:
            if isinstance(cls, str):
                class_id = handler.config.get_class_id_by_name(cls)
                if class_id is not None:
                    class_ids.append(class_id)
                else:
                    print(f"Warning: Unknown class name '{cls}'. Available classes: {list(handler.config._name_to_id.keys())}")
            elif isinstance(cls, int):
                class_ids.append(cls)
        
        if not class_ids:
            raise ValueError(f"No valid class IDs found from input: {classes}")
        
        mask = handler.get_binary_mask(class_ids)
        
        if output_path:
            Image.fromarray(mask).save(output_path)
        
        return mask
        
    except Exception as e:
        print(f"Error in extract_custom_mask: {e}")
        # Return debug info
        try:
            handler = create_segmentation_handler('extended_lip')
            handler.load_colored_segmentation(segmentation_path)
            debug_info = handler.debug_color_matching(10)
            print("Debug info:", debug_info)
        except:
            pass
        raise


def validate_segmentation_handler(segmentation_path: str) -> Dict:
    """
    Validation function to test the segmentation handler and provide debugging info.
    
    Args:
        segmentation_path: Path to a colored segmentation image
        
    Returns:
        Dictionary with validation results and debugging information
    """
    try:
        handler = create_segmentation_handler('extended_lip')
        handler.load_colored_segmentation(segmentation_path)
        
        # Get basic info
        shape = handler.get_image_shape()
        available_classes = handler.get_available_classes()
        debug_info = handler.debug_color_matching(20)
        
        # Test mask extraction
        test_results = {}
        
        # Test hair mask if hair class is present
        hair_present = any(cls[0] == ExtendedLIPClass.HAIR for cls in available_classes)
        if hair_present:
            hair_mask = handler.get_hair_mask()
            hair_pixels = np.sum(hair_mask > 0)
            test_results['hair_mask'] = {
                'success': True,
                'non_zero_pixels': int(hair_pixels),
                'percentage': float(hair_pixels / (shape[0] * shape[1]) * 100)
            }
        
        # Test face mask if face class is present  
        face_present = any(cls[0] == ExtendedLIPClass.FACE for cls in available_classes)
        if face_present:
            face_mask = handler.get_face_mask()
            face_pixels = np.sum(face_mask > 0)
            test_results['face_mask'] = {
                'success': True,
                'non_zero_pixels': int(face_pixels),
                'percentage': float(face_pixels / (shape[0] * shape[1]) * 100)
            }
        
        return {
            'status': 'success',
            'image_shape': shape,
            'available_classes': available_classes,
            'test_results': test_results,
            'debug_info': debug_info,
            'total_classes_found': len(available_classes)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }