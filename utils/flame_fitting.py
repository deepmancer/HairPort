import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List

import cv2
import numpy as np
import torch
from PIL import Image
from roma import rotvec_to_rotmat
from scipy import ndimage

from sheap import inference_images_list, load_sheap_model, render_mesh
from sheap.flame_segmentation import create_binary_mask_texture
from sheap.tiny_flame import TinyFlame, pose_components_to_rotmats

# Add parent path for utils imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.facial_landmark_detector import FacialLandmarkDetector

os.environ["PYOPENGL_PLATFORM"] = "egl"


def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in binary mask (e.g., open mouth region)."""
    inverted = 255 - mask
    labeled, num_features = ndimage.label(inverted)
    
    if num_features > 1:
        component_sizes = ndimage.sum(inverted, labeled, range(1, num_features + 1))
        largest_component_label = np.argmax(component_sizes) + 1
        largest_component_mask = (labeled == largest_component_label)
        mask = np.where(largest_component_mask, 0, 255).astype(np.uint8)
    
    return mask


def _check_mask_border_touch(mask: np.ndarray, threshold: int = 127) -> Dict[str, bool]:
    """Check if white pixels in the mask touch any image borders.
    
    Returns a dict indicating which borders are touched:
    {'left': bool, 'right': bool, 'top': bool, 'bottom': bool}
    """
    binary_mask = mask > threshold
    h, w = mask.shape[:2]
    
    return {
        'left': np.any(binary_mask[:, 0]),
        'right': np.any(binary_mask[:, w - 1]),
        'top': np.any(binary_mask[0, :]),
        'bottom': np.any(binary_mask[h - 1, :]),
    }


def _any_border_touched(touched_borders: Dict[str, bool]) -> bool:
    """Check if any border is touched."""
    return any(touched_borders.values())


def _rotmat_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to XYZ Euler angles (in radians)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


class FLAMEFitter:
    """FLAME model fitting with face cropping, SHeaP inference, and mesh rendering."""
    
    def __init__(
        self,
        flame_dir: Union[str, Path] = "FLAME2020/",
        device: Optional[torch.device] = None,
        model_type: str = "expressive",
        padding_ratio: float = 0.1,
        min_detection_confidence: float = 0.5,
        expand_factor: float = 2.0,
        max_expansion_iterations: int = 10,
        disable_face_crop: bool = False,
        disable_multi_run: bool = False,
    ):
        self.flame_dir = Path("/workspace/HairPort/SHeaP/FLAME2020")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.padding_ratio = padding_ratio
        self.min_detection_confidence = min_detection_confidence
        self.expand_factor = expand_factor
        self.max_expansion_iterations = 2
        self.disable_face_crop = disable_face_crop
        self.disable_multi_run = disable_multi_run
        
        # Initialize models (lazy loading)
        self._sheap_model = None
        self._flame_model = None
        self._face_detector = None
        
        # Camera-to-world transform (identity with z-offset)
        self._c2w = torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
            dtype=torch.float32
        )
    
    @property
    def sheap_model(self):
        if self._sheap_model is None:
            self._sheap_model = load_sheap_model(model_type=self.model_type).to(self.device)
        return self._sheap_model
    
    @property
    def flame_model(self) -> TinyFlame:
        if self._flame_model is None:
            self._flame_model = TinyFlame(
                self.flame_dir / "generic_model.pt",
                eyelids_ckpt=self.flame_dir / "eyelids.pt"
            )
        return self._flame_model
    
    @property
    def face_detector(self) -> FacialLandmarkDetector:
        if self._face_detector is None:
            self._face_detector = FacialLandmarkDetector(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
            )
        return self._face_detector
    
    def _crop_face_with_padding(
        self,
        image: np.ndarray,
        expansion_factors: Optional[Dict[str, float]] = None,
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Crop face region from image based on facial landmarks.
        
        The crop bounding box is 3 times larger than the maximum distance
        of the facial landmarks, centered on the landmark centroid.
        
        Args:
            image: Input image as numpy array.
            expansion_factors: Optional dict with per-direction expansion factors
                {'left': float, 'right': float, 'top': float, 'bottom': float}.
                These multiply the base size in each direction.
        """
        height, width = image.shape[:2]
        
        # Get facial landmarks (478 MediaPipe landmarks)
        landmarks = self.face_detector.get_lmk_478(image)
        
        if landmarks is None:
            return None
        
        # Calculate the maximum distance of facial landmarks
        # This is the maximum pairwise distance (diameter of the landmark spread)
        min_x, min_y = landmarks.min(axis=0)
        max_x, max_y = landmarks.max(axis=0)
        landmark_width = max_x - min_x
        landmark_height = max_y - min_y
        max_landmark_distance = max(landmark_width, landmark_height)
        
        # Crop box is 3 times the maximum landmark distance
        crop_size = max_landmark_distance * 2.5
        
        # Center of landmarks
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        
        # Apply per-direction expansion factors if provided
        if expansion_factors is None:
            expansion_factors = {'left': 1.0, 'right': 1.0, 'top': 1.0, 'bottom': 1.0}
        
        # Half size in each direction, scaled by expansion factors
        half_size = crop_size / 2.0
        pad_left = half_size * expansion_factors.get('left', 1.0)
        pad_right = half_size * expansion_factors.get('right', 1.0)
        pad_top = half_size * expansion_factors.get('top', 1.0)
        pad_bottom = half_size * expansion_factors.get('bottom', 1.0)
        
        x1 = max(0, int(center_x - pad_left))
        y1 = max(0, int(center_y - pad_top))
        x2 = min(width, int(center_x + pad_right))
        y2 = min(height, int(center_y + pad_bottom))
        
        cropped = image[y1:y2, x1:x2].copy()
        
        # Store original face bbox from landmarks for reference
        face_bbox = [int(min_x), int(min_y), int(landmark_width), int(landmark_height)]
        
        crop_params = {
            'crop_coords': [x1, y1, x2, y2],
            'original_size': [width, height],
            'cropped_size': [cropped.shape[1], cropped.shape[0]],
            'face_bbox': face_bbox,
            'landmark_center': [float(center_x), float(center_y)],
            'max_landmark_distance': float(max_landmark_distance),
            'expansion_factors': expansion_factors.copy(),
        }
        
        return cropped, crop_params
    
    def _uncrop_mask_to_original(
        self,
        mask: np.ndarray,
        crop_params: Dict[str, Any],
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """Map a mask from cropped image space back to original image coordinates."""
        x1, y1, x2, y2 = crop_params['crop_coords']
        orig_width, orig_height = crop_params['original_size']
        crop_width, crop_height = crop_params['cropped_size']
        
        mask_h, mask_w = mask.shape[:2]
        if mask_w != crop_width or mask_h != crop_height:
            mask_resized = cv2.resize(
                mask.astype(np.float32),
                (crop_width, crop_height),
                interpolation=interpolation
            )
            mask_resized = (mask_resized > 127).astype(np.uint8) * 255
        else:
            mask_resized = mask
        
        full_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)
        
        src_x1 = max(0, -x1) if x1 < 0 else 0
        src_y1 = max(0, -y1) if y1 < 0 else 0
        src_x2 = crop_width - max(0, x2 - orig_width) if x2 > orig_width else crop_width
        src_y2 = crop_height - max(0, y2 - orig_height) if y2 > orig_height else crop_height
        
        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(orig_width, x2)
        dst_y2 = min(orig_height, y2)
        
        full_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask_resized[src_y1:src_y2, src_x1:src_x2]
        
        return full_mask
    
    def _extract_head_orientation(self, predictions: Dict, frame_idx: int = 0) -> Dict[str, Any]:
        """Extract head orientation as Euler angles and direction vectors."""
        torso_rotvec = predictions["torso_pose"][frame_idx].cpu()
        neck_rotvec = predictions["neck_pose"][frame_idx].cpu()
        
        torso_rotmat = rotvec_to_rotmat(torso_rotvec.unsqueeze(0))[0]
        neck_rotmat = rotvec_to_rotmat(neck_rotvec.unsqueeze(0))[0]
        
        head_rotmat = (torso_rotmat @ neck_rotmat).numpy()
        euler_xyz = _rotmat_to_euler_xyz(head_rotmat)
        
        return {
            "euler_xyz_radians": euler_xyz.tolist(),
            "forward": (-head_rotmat[:, 2]).tolist(),
            "up": head_rotmat[:, 1].tolist(),
            "right": head_rotmat[:, 0].tolist(),
        }
    
    def _render_flame_mask(
        self,
        verts: torch.Tensor,
        output_size: int = 512,
    ) -> np.ndarray:
        """Render binary FLAME mask with holes filled."""
        mask_verts, mask_faces, mask_colors = create_binary_mask_texture(
            verts, self.flame_model.faces, flame_masks_path=self.flame_dir / "FLAME_masks.pkl"
        )
        mask_render, _ = render_mesh(
            verts=mask_verts,
            faces=mask_faces,
            c2w=self._c2w,
            img_width=output_size,
            img_height=output_size,
            render_normals=False,
            render_segmentation=True,
            vertex_colors=mask_colors,
            black_background=True
        )
        
        mask_gray = mask_render[:, :, 0]
        mask_filled = _fill_mask_holes(mask_gray)
        
        return mask_filled
    
    def _extract_flame_parameters(self, predictions: Dict, frame_idx: int = 0) -> Dict[str, Any]:
        """Extract FLAME parameters from SHeaP predictions."""
        return {
            "shape": predictions["shape_from_facenet"][frame_idx].cpu().numpy().tolist(),
            "expression": predictions["expr"][frame_idx].cpu().numpy().tolist(),
            "eyelids": predictions["eyelids"][frame_idx].cpu().numpy().tolist(),
            "translation": predictions["cam_trans"][frame_idx].cpu().numpy().tolist(),
            "torso_pose": predictions["torso_pose"][frame_idx].cpu().numpy().tolist(),
            "neck_pose": predictions["neck_pose"][frame_idx].cpu().numpy().tolist(),
            "jaw_pose": predictions["jaw_pose"][frame_idx].cpu().numpy().tolist(),
            # "leye_pose": predictions["leye_pose"][frame_idx].cpu().numpy().tolist(),
            # "reye_pose": predictions["reye_pose"][frame_idx].cpu().numpy().tolist(),
        }
    
    def _update_expansion_factors(
        self,
        current_factors: Dict[str, float],
        touched_borders: Dict[str, bool],
    ) -> Dict[str, float]:
        """Update expansion factors for borders that are touched."""
        new_factors = current_factors.copy()
        for border, is_touched in touched_borders.items():
            if is_touched:
                new_factors[border] = current_factors[border] * self.expand_factor
        return new_factors
    
    def _run_single_inference(
        self,
        cropped_image: np.ndarray,
        crop_params: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Dict, torch.Tensor]:
        """Run SHeaP inference on a single cropped image.
        
        Returns:
            mask_cropped: The rendered mask in cropped image space
            flame_mask: The mask in original image coordinates
            predictions: SHeaP predictions dict
            verts: FLAME vertices
        """
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            Image.fromarray(cropped_image).save(tmp_path, quality=95)
        
        try:
            with torch.no_grad():
                predictions = inference_images_list(
                    model=self.sheap_model,
                    device=self.device,
                    image_paths=[tmp_path],
                )
        finally:
            tmp_path.unlink()
        
        verts = self.flame_model(
            shape=predictions["shape_from_facenet"],
            expression=predictions["expr"],
            pose=pose_components_to_rotmats(predictions),
            eyelids=predictions["eyelids"],
            translation=predictions["cam_trans"],
        )
        
        crop_width, crop_height = crop_params['cropped_size']
        render_size = max(crop_width, crop_height)
        
        mask_cropped = self._render_flame_mask(
            verts=verts[0],
            output_size=render_size,
        )
        
        if crop_width != crop_height:
            mask_cropped = cv2.resize(
                mask_cropped,
                (crop_width, crop_height),
                interpolation=cv2.INTER_LINEAR
            )
            mask_cropped = (mask_cropped > 127).astype(np.uint8) * 255
        
        flame_mask = self._uncrop_mask_to_original(mask_cropped, crop_params)
        
        return mask_cropped, flame_mask, predictions, verts
    
    def fit_flame(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Fit FLAME model to an input image. Returns (flame_mask, result_dict) or None.
        
        If the rendered FLAME mask touches any border of the cropped image,
        the inference is repeated with an expanded crop region until the mask
        no longer touches any borders (or max iterations is reached).
        """
        # Load image
        if isinstance(image, (str, Path)):
            image_array = np.array(Image.open(image).convert('RGB'))
        elif isinstance(image, Image.Image):
            image_array = np.array(image.convert('RGB'))
        else:
            image_array = image
        
        # Initialize expansion factors
        expansion_factors = {'left': 1.0, 'right': 1.0, 'top': 1.0, 'bottom': 1.0}
        
        # Determine number of iterations based on disable_multi_run flag
        max_iterations = 1 if self.disable_multi_run else self.max_expansion_iterations
        
        for iteration in range(max_iterations):
            # Step 1: Crop face with current expansion factors (or use full image if disabled)
            if self.disable_face_crop:
                height, width = image_array.shape[:2]
                cropped_image = image_array.copy()
                crop_params = {
                    'crop_coords': [0, 0, width, height],
                    'original_size': [width, height],
                    'cropped_size': [width, height],
                    'face_bbox': [0, 0, width, height],
                    'landmark_center': [width / 2.0, height / 2.0],
                    'max_landmark_distance': float(max(width, height)),
                    'expansion_factors': expansion_factors.copy(),
                }
            else:
                crop_result = self._crop_face_with_padding(image_array, expansion_factors)
                
                if crop_result is None:
                    return None
                
                cropped_image, crop_params = crop_result
            
            # Step 2-4: Run inference and render mask
            mask_cropped, flame_mask, predictions, verts = self._run_single_inference(
                cropped_image, crop_params
            )
            
            # Skip border check if multi-run is disabled or face crop is disabled
            if self.disable_multi_run or self.disable_face_crop:
                break
            
            # Step 5: Check if mask touches any borders
            touched_borders = _check_mask_border_touch(mask_cropped)
            
            if not _any_border_touched(touched_borders):
                # No borders touched, we're done
                break
            
            # Update expansion factors for touched borders
            expansion_factors = self._update_expansion_factors(expansion_factors, touched_borders)
            
            if iteration < max_iterations - 1:
                touched_list = [k for k, v in touched_borders.items() if v]
                print(f"  Iteration {iteration + 1}: mask touches borders {touched_list}, expanding...")
        else:
            # Max iterations reached, warn user
            if not self.disable_multi_run and not self.disable_face_crop:
                touched_list = [k for k, v in touched_borders.items() if v]
                print(f"  Warning: max expansion iterations reached, mask still touches borders {touched_list}")
        
        # Extract parameters and orientation
        flame_parameters = self._extract_flame_parameters(predictions, frame_idx=0)
        head_orientation = self._extract_head_orientation(predictions, frame_idx=0)
        
        result_dict = {
            'cropped_image': cropped_image,
            'cropping_params': crop_params,
            'flame_parameters': flame_parameters,
            'head_orientation': head_orientation,
        }
        
        return flame_mask, result_dict
    
    def fit_flame_batch(
        self,
        image_paths: List[Union[str, Path]],
    ) -> List[Optional[Tuple[np.ndarray, Dict[str, Any]]]]:
        """Fit FLAME model to a batch of images. Returns list of (flame_mask, result_dict) or None.
        
        If the rendered FLAME mask touches any border of the cropped image,
        the inference is repeated with an expanded crop region until the mask
        no longer touches any borders (or max iterations is reached).
        """
        import tempfile
        import shutil
        
        num_images = len(image_paths)
        results = [None] * num_images
        
        # Load all images upfront
        image_arrays = []
        for img_path in image_paths:
            image_arrays.append(np.array(Image.open(img_path).convert('RGB')))
        
        # Track expansion factors per image
        expansion_factors_list = [
            {'left': 1.0, 'right': 1.0, 'top': 1.0, 'bottom': 1.0}
            for _ in range(num_images)
        ]
        
        # Track which images are done (either completed or no face detected)
        done = [False] * num_images
        no_face_detected = [False] * num_images
        
        # Determine number of iterations based on disable_multi_run flag
        max_iterations = 1 if self.disable_multi_run else self.max_expansion_iterations
        
        for iteration in range(max_iterations):
            # Gather images that need processing in this iteration
            indices_to_process = [i for i in range(num_images) if not done[i] and not no_face_detected[i]]
            
            if not indices_to_process:
                break
            
            # Step 1: Crop all faces for images that need processing (or use full image if disabled)
            cropped_images = []
            crop_params_list = []
            batch_to_orig_idx = []  # Maps batch index to original image index
            
            for orig_idx in indices_to_process:
                if self.disable_face_crop:
                    # Use full image without cropping
                    height, width = image_arrays[orig_idx].shape[:2]
                    cropped_image = image_arrays[orig_idx].copy()
                    crop_params = {
                        'crop_coords': [0, 0, width, height],
                        'original_size': [width, height],
                        'cropped_size': [width, height],
                        'face_bbox': [0, 0, width, height],
                        'landmark_center': [width / 2.0, height / 2.0],
                        'max_landmark_distance': float(max(width, height)),
                        'expansion_factors': expansion_factors_list[orig_idx].copy(),
                    }
                else:
                    crop_result = self._crop_face_with_padding(
                        image_arrays[orig_idx],
                        expansion_factors_list[orig_idx]
                    )
                    
                    if crop_result is None:
                        print(f"Warning: No face detected in {image_paths[orig_idx]}, skipping.")
                        no_face_detected[orig_idx] = True
                        continue
                    
                    cropped_image, crop_params = crop_result
                
                cropped_images.append(cropped_image)
                crop_params_list.append(crop_params)
                batch_to_orig_idx.append(orig_idx)
            
            if not cropped_images:
                continue
            
            # Step 2: Save cropped images temporarily for batch inference
            temp_dir = Path(tempfile.mkdtemp())
            temp_paths = []
            
            try:
                for i, cropped_img in enumerate(cropped_images):
                    temp_path = temp_dir / f"crop_{i}.jpg"
                    Image.fromarray(cropped_img).save(temp_path, quality=95)
                    temp_paths.append(temp_path)
                
                # Step 3: Run batch inference
                with torch.no_grad():
                    predictions = inference_images_list(
                        model=self.sheap_model,
                        device=self.device,
                        image_paths=temp_paths,
                    )
                
                # Step 4: Compute FLAME vertices for all
                verts = self.flame_model(
                    shape=predictions["shape_from_facenet"],
                    expression=predictions["expr"],
                    pose=pose_components_to_rotmats(predictions),
                    eyelids=predictions["eyelids"],
                    translation=predictions["cam_trans"],
                )
                
                # Step 5: Process each result and check for border touching
                for batch_idx, orig_idx in enumerate(batch_to_orig_idx):
                    crop_params = crop_params_list[batch_idx]
                    crop_width, crop_height = crop_params['cropped_size']
                    render_size = max(crop_width, crop_height)
                    
                    mask_cropped = self._render_flame_mask(
                        verts=verts[batch_idx],
                        output_size=render_size,
                    )
                    
                    if crop_width != crop_height:
                        mask_cropped = cv2.resize(
                            mask_cropped,
                            (crop_width, crop_height),
                            interpolation=cv2.INTER_LINEAR
                        )
                        mask_cropped = (mask_cropped > 127).astype(np.uint8) * 255
                    
                    # Skip border check if multi-run is disabled or face crop is disabled
                    if self.disable_multi_run or self.disable_face_crop:
                        # Finalize result immediately
                        flame_mask = self._uncrop_mask_to_original(mask_cropped, crop_params)
                        flame_parameters = self._extract_flame_parameters(predictions, frame_idx=batch_idx)
                        head_orientation = self._extract_head_orientation(predictions, frame_idx=batch_idx)
                        
                        result_dict = {
                            'cropped_image': cropped_images[batch_idx],
                            'cropping_params': crop_params,
                            'flame_parameters': flame_parameters,
                            'head_orientation': head_orientation,
                        }
                        
                        results[orig_idx] = (flame_mask, result_dict)
                        done[orig_idx] = True
                        continue
                    
                    # Check if mask touches any borders
                    touched_borders = _check_mask_border_touch(mask_cropped)
                    
                    if not _any_border_touched(touched_borders):
                        # No borders touched, finalize this result
                        flame_mask = self._uncrop_mask_to_original(mask_cropped, crop_params)
                        flame_parameters = self._extract_flame_parameters(predictions, frame_idx=batch_idx)
                        head_orientation = self._extract_head_orientation(predictions, frame_idx=batch_idx)
                        
                        result_dict = {
                            'cropped_image': cropped_images[batch_idx],
                            'cropping_params': crop_params,
                            'flame_parameters': flame_parameters,
                            'head_orientation': head_orientation,
                        }
                        
                        results[orig_idx] = (flame_mask, result_dict)
                        done[orig_idx] = True
                    else:
                        # Update expansion factors for next iteration
                        expansion_factors_list[orig_idx] = self._update_expansion_factors(
                            expansion_factors_list[orig_idx],
                            touched_borders
                        )
                        
                        if iteration < max_iterations - 1:
                            touched_list = [k for k, v in touched_borders.items() if v]
                            print(f"  {image_paths[orig_idx]}: iteration {iteration + 1}, "
                                  f"mask touches borders {touched_list}, expanding...")
                        else:
                            # Max iterations reached, save current result with warning
                            touched_list = [k for k, v in touched_borders.items() if v]
                            print(f"  Warning: {image_paths[orig_idx]}: max expansion iterations reached, "
                                  f"mask still touches borders {touched_list}")
                            
                            flame_mask = self._uncrop_mask_to_original(mask_cropped, crop_params)
                            flame_parameters = self._extract_flame_parameters(predictions, frame_idx=batch_idx)
                            head_orientation = self._extract_head_orientation(predictions, frame_idx=batch_idx)
                            
                            result_dict = {
                                'cropped_image': cropped_images[batch_idx],
                                'cropping_params': crop_params,
                                'flame_parameters': flame_parameters,
                                'head_orientation': head_orientation,
                            }
                            
                            results[orig_idx] = (flame_mask, result_dict)
                            done[orig_idx] = True
                
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        return results
    
    def create_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.4,
        color: Tuple[int, int, int] = (255, 64, 64),
    ) -> np.ndarray:
        """Create an overlay of the FLAME mask on an image."""
        mask_rgb = np.zeros_like(image)
        mask_rgb[:, :, 0] = (mask > 127).astype(np.uint8) * color[0]
        mask_rgb[:, :, 1] = (mask > 127).astype(np.uint8) * color[1]
        mask_rgb[:, :, 2] = (mask > 127).astype(np.uint8) * color[2]
        
        overlay = image.copy().astype(np.float32)
        mask_normalized = (mask > 127).astype(np.float32)[:, :, np.newaxis]
        overlay = overlay * (1 - alpha * mask_normalized) + mask_rgb.astype(np.float32) * alpha * mask_normalized
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay


if __name__ == "__main__":
    # Initialize FLAMEFitter
    fitter = FLAMEFitter(
        flame_dir=Path("FLAME2020/"),
        padding_ratio=0.1,
    )

    # Process example images
    folder_containing_images = Path("example_images/")
    image_paths = list(sorted(folder_containing_images.glob("*.jpg")))
    
    # Use batch processing for efficiency
    results = fitter.fit_flame_batch(image_paths)
    
    # Save results
    for img_path, result in zip(image_paths, results):
        if result is None:
            print(f"Skipping {img_path.name}: no face detected.")
            continue
        
        flame_mask, result_dict = result
        
        # Create output folder
        output_dir = img_path.parent / img_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save flame mask
        Image.fromarray(flame_mask).save(output_dir / "flame_segmentation.png")
        
        # Save cropped image
        Image.fromarray(result_dict['cropped_image']).save(output_dir / "cropped_image.png")
        
        # Create and save overlay
        original_image = np.array(Image.open(img_path).convert('RGB'))
        overlay = fitter.create_overlay(original_image, flame_mask, alpha=0.4)
        Image.fromarray(overlay).save(output_dir / "overlay.png")
        
        # Save head orientation
        with open(output_dir / "head_orientation.json", "w") as f:
            json.dump(result_dict['head_orientation'], f, indent=2)
        
        # Save crop parameters
        with open(output_dir / "crop_params.json", "w") as f:
            json.dump(result_dict['cropping_params'], f, indent=2)
        
        # Save FLAME parameters
        with open(output_dir / "flame_parameters.json", "w") as f:
            json.dump(result_dict['flame_parameters'], f, indent=2)
        
        print(f"Processed: {img_path.name}")
    
    print(f"\nProcessed {sum(1 for r in results if r is not None)} images successfully.")
