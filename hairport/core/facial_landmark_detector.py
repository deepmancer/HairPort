from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np
from PIL import Image
import mediapipe as mp
import cv2


class FacialLandmarkDetector:
    """Robust facial landmark detector using MediaPipe with fallback strategies."""
    
    # MediaPipe 478 to Dlib 68 correspondence mapping
    MP2DLIB_CORRESPONDENCE = [
        ## Face Contour
        [127],       # 1
        [234],       # 2
        [93],        # 3
        [132, 58],   # 4
        [58, 172],   # 5
        [136],       # 6
        [150],       # 7
        [176],       # 8
        [152],       # 9
        [400],       # 10
        [379],       # 11
        [365],       # 12
        [397, 288],  # 13
        [361],       # 14
        [323],       # 15
        [454],       # 16
        [356],       # 17
        
        ## Right Brow 
        [70],        # 18
        [63],        # 19
        [105],       # 20
        [66],        # 21
        [107],       # 22
        
        ## Left Brow
        [336],       # 23
        [296],       # 24
        [334],       # 25
        [293],       # 26
        [300],       # 27
        
        ## Nose
        [168, 6],    # 28
        [197, 195],  # 29
        [5],         # 30
        [4],         # 31
        [75],        # 32
        [97],        # 33
        [2],         # 34
        [326],       # 35
        [305],       # 36
        
        ## Right Eye
        [33],        # 37
        [160],       # 38
        [158],       # 39
        [133],       # 40
        [153],       # 41
        [144],       # 42
        
        ## Left Eye
        [362],       # 43
        [385],       # 44
        [387],       # 45
        [263],       # 46
        [373],       # 47
        [380],       # 48
        
        ## Upper Lip Contour Top
        [61],        # 49
        [39],        # 50
        [37],        # 51
        [0],         # 52
        [267],       # 53
        [269],       # 54
        [291],       # 55
        
        ## Lower Lip Contour Bottom
        [321],   # 56
        [314],   # 57
        [17],    # 58
        [84],    # 59
        [91],    # 60
        
        ## Upper Lip Contour Bottom
        [78],    # 61
        [82],    # 62
        [13],    # 63
        [312],   # 64
        [308],   # 65
        
        ## Lower Lip Contour Top
        [317],   # 66
        [14],    # 67
        [87],    # 68
    ]
    
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float | None = None,
        min_tracking_confidence: float = 0.5,
        mediapipe_flame_embedding_path: str | None = None,
        enable_fallback: bool = True,
        fallback_confidence_thresholds: Tuple[float, ...] | None = None,
        enable_preprocessing: bool = True,
        min_face_size: int | None = None,
        target_face_size: int | None = None,
    ):
        from hairport.config import get_config
        cfg = get_config()

        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = (
            min_detection_confidence if min_detection_confidence is not None
            else cfg.facial_landmarks.detection_confidence
        )
        self.min_tracking_confidence = min_tracking_confidence
        self.mediapipe_flame_embedding_path = (
            mediapipe_flame_embedding_path
            if mediapipe_flame_embedding_path is not None
            else cfg.paths.mediapipe_flame_embedding
        )
        self.enable_fallback = enable_fallback
        self.fallback_confidence_thresholds = (
            fallback_confidence_thresholds
            if fallback_confidence_thresholds is not None
            else tuple(cfg.facial_landmarks.fallback_confidences)
        )
        self.enable_preprocessing = enable_preprocessing
        self.min_face_size = (
            min_face_size if min_face_size is not None
            else cfg.facial_landmarks.min_face_size
        )
        self.target_face_size = (
            target_face_size if target_face_size is not None
            else cfg.facial_landmarks.target_face_size
        )
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # Create fallback face mesh instances with lower confidence thresholds
        self.fallback_face_meshes = []
        if self.enable_fallback:
            for conf in self.fallback_confidence_thresholds:
                fm = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=self.max_num_faces,
                    refine_landmarks=self.refine_landmarks,
                    min_detection_confidence=conf,
                    min_tracking_confidence=conf
                )
                self.fallback_face_meshes.append(fm)
        
        # Initialize MediaPipe Face Detection (both short-range and full-range models)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Full-range model (0-5m)
            min_detection_confidence=self.min_detection_confidence
        )
        self.face_detection_short = self.mp_face_detection.FaceDetection(
            model_selection=0,  # Short-range model (0-2m, better for close-ups)
            min_detection_confidence=self.min_detection_confidence
        )
        
        # Store expected number of landmarks
        self.num_landmarks = 478 if refine_landmarks else 468
        
        # Load MediaPipe-FLAME landmark mapping
        self.mediapipe_flame_mapping = None
        self._load_mediapipe_flame_mapping()
        
        # Normalize MP2DLIB_CORRESPONDENCE for single-index entries
        self.mp2dlib_correspondence_normalized = []
        for indices in self.MP2DLIB_CORRESPONDENCE:
            if len(indices) == 1:
                self.mp2dlib_correspondence_normalized.append([indices[0], indices[0]])
            else:
                self.mp2dlib_correspondence_normalized.append(indices)
        
    def __del__(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        if hasattr(self, 'face_detection_short'):
            self.face_detection_short.close()
        if hasattr(self, 'fallback_face_meshes'):
            for fm in self.fallback_face_meshes:
                fm.close()
    
    def _load_mediapipe_flame_mapping(self):
        try:
            mapping_path = Path(self.mediapipe_flame_embedding_path)
            if not mapping_path.exists():
                print(f"Warning: MediaPipe-FLAME mapping file not found at {mapping_path}")
                print("ldm105_flame will not be available in get_lmk_full()")
                return
            
            mapping_data = np.load(mapping_path)
            self.mediapipe_flame_mapping = {
                'landmark_indices': mapping_data['landmark_indices'],
            }
            print(f"Loaded MediaPipe-FLAME mapping: {len(self.mediapipe_flame_mapping['landmark_indices'])} landmarks")
            
        except Exception as e:
            print(f"Error loading MediaPipe-FLAME mapping: {e}")
            self.mediapipe_flame_mapping = None
    
    def _enhance_image(self, image_array: np.ndarray) -> np.ndarray:
        """Apply adaptive preprocessing to improve detection on difficult images."""
        # Convert to LAB color space for luminance-based enhancement
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(l_channel)
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
    
    def _get_face_crop_params(
        self,
        image_array: np.ndarray,
        padding_ratio: float = 0.3
    ) -> Optional[Tuple[int, int, int, int, float]]:
        """Detect face and compute crop parameters with padding."""
        height, width = image_array.shape[:2]
        
        # Try full-range model first, then short-range
        for detector in [self.face_detection, self.face_detection_short]:
            results = detector.process(image_array)
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                
                # Convert to pixel coordinates with padding
                face_w = int(bbox.width * width)
                face_h = int(bbox.height * height)
                face_x = int(bbox.xmin * width)
                face_y = int(bbox.ymin * height)
                
                # Add padding
                pad_w = int(face_w * padding_ratio)
                pad_h = int(face_h * padding_ratio)
                
                x1 = max(0, face_x - pad_w)
                y1 = max(0, face_y - pad_h)
                x2 = min(width, face_x + face_w + pad_w)
                y2 = min(height, face_y + face_h + pad_h)
                
                # Compute scale factor if face is small
                face_size = max(x2 - x1, y2 - y1)
                scale = max(1.0, self.target_face_size / face_size) if face_size < self.min_face_size else 1.0
                
                return (x1, y1, x2, y2, scale)
        
        return None
    
    def _process_with_fallback(
        self,
        image_array: np.ndarray
    ) -> Optional[any]:
        """Try detection with progressively lower confidence thresholds."""
        # Try primary detector first
        results = self.face_mesh.process(image_array)
        if results.multi_face_landmarks:
            return results
        
        # Try fallback detectors with lower confidence
        for fm in self.fallback_face_meshes:
            results = fm.process(image_array)
            if results.multi_face_landmarks:
                return results
        
        return None
    
    def _detect_landmarks_robust(
        self,
        image_array: np.ndarray,
        original_height: int,
        original_width: int
    ) -> Optional[any]:
        """Robust landmark detection with multiple strategies."""
        height, width = image_array.shape[:2]
        
        # Strategy 1: Direct detection on original image
        results = self._process_with_fallback(image_array)
        if results and results.multi_face_landmarks:
            return results, None
        
        if not self.enable_preprocessing:
            return None, None
        
        # Strategy 2: Enhanced image (CLAHE)
        enhanced = self._enhance_image(image_array)
        results = self._process_with_fallback(enhanced)
        if results and results.multi_face_landmarks:
            return results, None
        
        # Strategy 3: Face-crop based detection for small/distant faces
        crop_params = self._get_face_crop_params(image_array)
        if crop_params is not None:
            x1, y1, x2, y2, scale = crop_params
            face_crop = image_array[y1:y2, x1:x2]
            
            # Upscale if face is small
            if scale > 1.0:
                new_w = int((x2 - x1) * scale)
                new_h = int((y2 - y1) * scale)
                face_crop = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            results = self._process_with_fallback(face_crop)
            if results and results.multi_face_landmarks:
                # Return crop parameters for coordinate transformation
                return results, (x1, y1, x2, y2, scale)
            
            # Try enhanced version of crop
            enhanced_crop = self._enhance_image(face_crop)
            results = self._process_with_fallback(enhanced_crop)
            if results and results.multi_face_landmarks:
                return results, (x1, y1, x2, y2, scale)
        
        # Strategy 4: Multi-scale detection
        for scale_factor in [1.5, 2.0, 0.75]:
            new_w = int(width * scale_factor)
            new_h = int(height * scale_factor)
            if new_w < 64 or new_h < 64:
                continue
            
            scaled = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            results = self._process_with_fallback(scaled)
            if results and results.multi_face_landmarks:
                return results, ('scale', scale_factor)
        
        return None, None
    
    def _transform_landmarks_to_original(
        self,
        landmarks: np.ndarray,
        transform_params: Optional[Tuple],
        original_height: int,
        original_width: int,
        processed_height: int,
        processed_width: int
    ) -> np.ndarray:
        """Transform landmarks back to original image coordinates."""
        if transform_params is None:
            return landmarks
        
        transformed = landmarks.copy()
        
        if isinstance(transform_params[0], str) and transform_params[0] == 'scale':
            # Simple scaling case
            scale_factor = transform_params[1]
            transformed[:, 0] /= scale_factor
            transformed[:, 1] /= scale_factor
            if transformed.shape[1] == 3:
                transformed[:, 2] /= scale_factor
        else:
            # Crop + optional scale case
            x1, y1, x2, y2, scale = transform_params
            crop_w = x2 - x1
            crop_h = y2 - y1
            
            if scale > 1.0:
                # First undo the upscaling
                transformed[:, 0] /= scale
                transformed[:, 1] /= scale
                if transformed.shape[1] == 3:
                    transformed[:, 2] /= scale
            
            # Then add crop offset
            transformed[:, 0] += x1
            transformed[:, 1] += y1
        
        return transformed
    
    def convert_mediapipe_to_dlib68(self, lmks_mp: np.ndarray) -> np.ndarray:
        """Convert MediaPipe 478 landmarks to Dlib 68 format using MP2DLIB_CORRESPONDENCE.
        
        Args:
            lmks_mp: MediaPipe landmarks, shape (478, 2) or (478, 3)
        Returns:
            Dlib 68 landmarks, shape (68, 2) or (68, 3)
        """
        # Convert landmarks by averaging corresponding MediaPipe indices
        # This works for both 2D (N, 2) and 3D (N, 3) landmark arrays
        lmks_dlib = np.array([
            lmks_mp[indices].mean(axis=0) 
            for indices in self.mp2dlib_correspondence_normalized
        ])
        
        return lmks_dlib
    
    def get_mediapipe_flame_subset(self, lmks_mp: np.ndarray) -> Optional[np.ndarray]:
        """Extract FLAME-aligned 105 landmarks from MediaPipe 478 using pre-computed indices.
        
        Args:
            lmks_mp: MediaPipe landmarks, shape (478, 2) or (478, 3)
        Returns:
            FLAME landmarks, shape (105, 2) or (105, 3), or None if mapping unavailable
        """
        if self.mediapipe_flame_mapping is None:
            return None
        
        landmark_indices = self.mediapipe_flame_mapping['landmark_indices']
        return lmks_mp[landmark_indices]
    
    def get_lmk_full(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        face_index: int = 0
    ) -> Optional[dict]:
        """Detect facial landmarks with fallback strategies. Returns all formats in 2D and 3D.
        
        Returns dict with keys:
            2D: ldm478, ldm468, ldm68, ldm105_flame (pixel coords in original image space)
            3D: ldm478_3d, ldm468_3d, ldm68_3d, ldm105_flame_3d (pixel coords + depth)
            Meta: image_height, image_width
        """
        try:
            # Preprocess image
            image_array, height, width = self._preprocess_image(image)
            
            # Robust detection with fallback strategies
            results, transform_params = self._detect_landmarks_robust(image_array, height, width)
            
            # Check if any faces were detected
            if results is None or not results.multi_face_landmarks:
                print("Warning: No faces detected in the image after all fallback strategies")
                return None
            
            # Check if requested face index is valid
            num_faces = len(results.multi_face_landmarks)
            if face_index >= num_faces:
                print(
                    f"Warning: Requested face index {face_index} but only "
                    f"{num_faces} face(s) detected. Using face 0."
                )
                face_index = 0
            
            # Extract landmarks for the specified face
            face_landmarks = results.multi_face_landmarks[face_index]
            
            # Determine processed image dimensions for landmark extraction
            if transform_params is None:
                proc_height, proc_width = height, width
            elif isinstance(transform_params[0], str) and transform_params[0] == 'scale':
                scale_factor = transform_params[1]
                proc_width = int(width * scale_factor)
                proc_height = int(height * scale_factor)
            else:
                x1, y1, x2, y2, scale = transform_params
                proc_width = int((x2 - x1) * scale) if scale > 1.0 else (x2 - x1)
                proc_height = int((y2 - y1) * scale) if scale > 1.0 else (y2 - y1)
            
            # Extract raw landmarks in processed image coordinates
            ldm478 = np.zeros((478, 2), dtype=np.float32)
            ldm478_3d = np.zeros((478, 3), dtype=np.float32)
            
            for idx, landmark in enumerate(face_landmarks.landmark):
                ldm478[idx, 0] = landmark.x * proc_width
                ldm478[idx, 1] = landmark.y * proc_height
                ldm478_3d[idx, 0] = landmark.x * proc_width
                ldm478_3d[idx, 1] = landmark.y * proc_height
                ldm478_3d[idx, 2] = landmark.z * proc_width
            
            # Transform back to original image coordinates
            ldm478 = self._transform_landmarks_to_original(
                ldm478, transform_params, height, width, proc_height, proc_width
            )
            ldm478_3d = self._transform_landmarks_to_original(
                ldm478_3d, transform_params, height, width, proc_height, proc_width
            )
            
            # Derive ldm468 (without iris landmarks)
            ldm468 = ldm478[:468].copy()
            ldm468_3d = ldm478_3d[:468].copy()
            
            # Convert to Dlib 68-point format
            ldm68 = self.convert_mediapipe_to_dlib68(ldm478)
            ldm68_3d = self.convert_mediapipe_to_dlib68(ldm478_3d)
            
            # Extract FLAME-aligned 105 landmarks
            ldm105_flame = self.get_mediapipe_flame_subset(ldm478)
            ldm105_flame_3d = self.get_mediapipe_flame_subset(ldm478_3d)
            
            return {
                # 2D landmarks (pixel coordinates in original image)
                'ldm478': ldm478,
                'ldm468': ldm468,
                'ldm68': ldm68,
                'ldm105_flame': ldm105_flame,
                # 3D landmarks (pixel coordinates with depth)
                'ldm478_3d': ldm478_3d,
                'ldm468_3d': ldm468_3d,
                'ldm68_3d': ldm68_3d,
                'ldm105_flame_3d': ldm105_flame_3d,
                # Image metadata
                'image_height': height,
                'image_width': width
            }
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"Error during landmark detection: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _preprocess_image(
        self, 
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Tuple[np.ndarray, int, int]:
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            pil_image = Image.open(image_path).convert('RGB')
            image_array = np.array(pil_image)
            
        elif isinstance(image, Image.Image):
            image_array = np.array(image.convert('RGB'))
            
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image_array = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image_array = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif image.shape[2] == 3:
                image_array = image.copy()
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        else:
            raise ValueError(
                f"Unsupported image type: {type(image)}. "
                "Expected str, Path, np.ndarray, or PIL.Image.Image"
            )
        
        height, width = image_array.shape[:2]
        return image_array, height, width
    
    def get_lmk_478(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        face_index: int = 0
    ) -> Optional[np.ndarray]:
        """Detect 478 MediaPipe landmarks with fallback strategies."""
        try:
            image_array, height, width = self._preprocess_image(image)
            
            # Robust detection with fallback strategies
            results, transform_params = self._detect_landmarks_robust(image_array, height, width)
            
            if results is None or not results.multi_face_landmarks:
                print("Warning: No faces detected in the image")
                return None
            
            num_faces = len(results.multi_face_landmarks)
            if face_index >= num_faces:
                print(
                    f"Warning: Requested face index {face_index} but only "
                    f"{num_faces} face(s) detected. Using face 0."
                )
                face_index = 0
            
            face_landmarks = results.multi_face_landmarks[face_index]
            
            # Determine processed image dimensions
            if transform_params is None:
                proc_height, proc_width = height, width
            elif isinstance(transform_params[0], str) and transform_params[0] == 'scale':
                scale_factor = transform_params[1]
                proc_width = int(width * scale_factor)
                proc_height = int(height * scale_factor)
            else:
                x1, y1, x2, y2, scale = transform_params
                proc_width = int((x2 - x1) * scale) if scale > 1.0 else (x2 - x1)
                proc_height = int((y2 - y1) * scale) if scale > 1.0 else (y2 - y1)
            
            landmarks_pixel = np.zeros((self.num_landmarks, 2), dtype=np.float32)
            for idx, landmark in enumerate(face_landmarks.landmark):
                landmarks_pixel[idx, 0] = landmark.x * proc_width
                landmarks_pixel[idx, 1] = landmark.y * proc_height
            
            # Transform back to original coordinates
            landmarks_pixel = self._transform_landmarks_to_original(
                landmarks_pixel, transform_params, height, width, proc_height, proc_width
            )
            
            return landmarks_pixel
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"Error during landmark detection: {e}")
            return None
    
    def get_all_faces_lmk_478(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Optional[List[np.ndarray]]:
        try:
            # Preprocess image
            image_array, height, width = self._preprocess_image(image)
            
            # Detect landmarks using MediaPipe
            results = self.face_mesh.process(image_array)
            
            # Check if any faces were detected
            if not results.multi_face_landmarks:
                print("Warning: No faces detected in the image")
                return None
            
            # Extract landmarks for all detected faces
            all_landmarks = []
            for face_landmarks in results.multi_face_landmarks:
                landmarks_pixel = np.zeros((self.num_landmarks, 2), dtype=np.float32)
                
                for idx, landmark in enumerate(face_landmarks.landmark):
                    landmarks_pixel[idx, 0] = landmark.x * width
                    landmarks_pixel[idx, 1] = landmark.y * height
                
                all_landmarks.append(landmarks_pixel)
            
            return all_landmarks
            
        except Exception as e:
            print(f"Error during multi-face landmark detection: {e}")
            return None
    
    def get_lmk_478_with_confidence(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        face_index: int = 0
    ) -> Optional[Tuple[np.ndarray, List[float]]]:
        try:
            image_array, height, width = self._preprocess_image(image)
            results = self.face_mesh.process(image_array)
            
            if not results.multi_face_landmarks:
                return None
            
            num_faces = len(results.multi_face_landmarks)
            if face_index >= num_faces:
                face_index = 0
            
            face_landmarks = results.multi_face_landmarks[face_index]
            
            landmarks_pixel = np.zeros((self.num_landmarks, 2), dtype=np.float32)
            confidence_scores = []
            
            for idx, landmark in enumerate(face_landmarks.landmark):
                landmarks_pixel[idx, 0] = landmark.x * width
                landmarks_pixel[idx, 1] = landmark.y * height
                confidence_scores.append(landmark.visibility if hasattr(landmark, 'visibility') else 1.0)
            
            return landmarks_pixel, confidence_scores
            
        except Exception as e:
            print(f"Error during landmark detection with confidence: {e}")
            return None
    
    def get_lmk_478_3d(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        face_index: int = 0
    ) -> Optional[np.ndarray]:
        """Detect 478 MediaPipe 3D landmarks with fallback strategies."""
        try:
            image_array, height, width = self._preprocess_image(image)
            
            results, transform_params = self._detect_landmarks_robust(image_array, height, width)
            
            if results is None or not results.multi_face_landmarks:
                return None
            
            num_faces = len(results.multi_face_landmarks)
            if face_index >= num_faces:
                face_index = 0
            
            face_landmarks = results.multi_face_landmarks[face_index]
            
            # Determine processed image dimensions
            if transform_params is None:
                proc_height, proc_width = height, width
            elif isinstance(transform_params[0], str) and transform_params[0] == 'scale':
                scale_factor = transform_params[1]
                proc_width = int(width * scale_factor)
                proc_height = int(height * scale_factor)
            else:
                x1, y1, x2, y2, scale = transform_params
                proc_width = int((x2 - x1) * scale) if scale > 1.0 else (x2 - x1)
                proc_height = int((y2 - y1) * scale) if scale > 1.0 else (y2 - y1)
            
            landmarks_3d = np.zeros((self.num_landmarks, 3), dtype=np.float32)
            for idx, landmark in enumerate(face_landmarks.landmark):
                landmarks_3d[idx, 0] = landmark.x * proc_width
                landmarks_3d[idx, 1] = landmark.y * proc_height
                landmarks_3d[idx, 2] = landmark.z * proc_width
            
            # Transform back to original coordinates
            landmarks_3d = self._transform_landmarks_to_original(
                landmarks_3d, transform_params, height, width, proc_height, proc_width
            )
            
            return landmarks_3d
            
        except Exception as e:
            print(f"Error during 3D landmark detection: {e}")
            return None
    
    def crop_face(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        scale: float = 1.8,
        target_size: Optional[int] = None,
        return_params: bool = False,
        face_index: int = 0
    ) -> Optional[Union[np.ndarray, Tuple[np.ndarray, dict]]]:
        """
        Robust face cropping with multiple fallback strategies.
        
        This method detects faces using multiple strategies (including fallback
        confidence thresholds, image enhancement, and multi-scale detection)
        and crops the face region with configurable scaling.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            scale: Scale factor for the crop bounding box (default 1.8)
            target_size: If provided, resize the cropped image to this size (square)
            return_params: If True, also return crop parameters for uncropping
            face_index: Which face to crop if multiple detected (default 0)
            
        Returns:
            If return_params=False: Cropped image as numpy array (RGB), or None if no face
            If return_params=True: Tuple of (cropped_image, crop_params_dict)
                crop_params_dict contains:
                    - 'original_bbox': [x, y, w, h] detected face bbox
                    - 'crop_coords': [x1, y1, x2, y2] actual crop coordinates
                    - 'original_size': [width, height] of input image
                    - 'cropped_size': [width, height] of output crop
                    - 'scale_factor': scale factor used
        """
        try:
            # Preprocess image
            image_array, height, width = self._preprocess_image(image)
            
            # Try to detect face with robust fallback strategies
            bbox_result = self._detect_face_robust(image_array)
            
            if bbox_result is None:
                print("Warning: No face detected after all fallback strategies")
                return None
            
            x, y, w, h = bbox_result
            
            # Compute center and scaled crop region
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Use the larger dimension and apply scale
            size = max(w, h)
            scaled_size = int(size * scale)
            
            # Compute crop coordinates (clamped to image bounds)
            x1 = max(0, center_x - scaled_size // 2)
            y1 = max(0, center_y - scaled_size // 2)
            x2 = min(width, center_x + scaled_size // 2)
            y2 = min(height, center_y + scaled_size // 2)
            
            # Extract crop
            cropped = image_array[y1:y2, x1:x2].copy()
            
            # Make square if not already (pad with zeros)
            crop_h, crop_w = cropped.shape[:2]
            if crop_h != crop_w:
                max_dim = max(crop_h, crop_w)
                square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
                y_offset = (max_dim - crop_h) // 2
                x_offset = (max_dim - crop_w) // 2
                square[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = cropped
                cropped = square
                # Update crop info for padding
                x1 -= x_offset
                y1 -= y_offset
                x2 = x1 + max_dim
                y2 = y1 + max_dim
            
            # Resize if target size specified
            output_size = cropped.shape[:2]
            if target_size is not None:
                cropped = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
                output_size = (target_size, target_size)
            
            if return_params:
                crop_params = {
                    'original_bbox': [x, y, w, h],
                    'crop_coords': [x1, y1, x2, y2],
                    'original_size': [width, height],
                    'cropped_size': list(output_size),
                    'scale_factor': scale
                }
                return cropped, crop_params
            
            return cropped
            
        except Exception as e:
            print(f"Error during face cropping: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _detect_face_robust(
        self,
        image_array: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Robust face detection with multiple fallback strategies.
        
        Returns:
            Tuple of (x, y, w, h) bounding box or None if no face detected
        """
        height, width = image_array.shape[:2]
        
        # Strategy 1: Try both face detection models
        for detector in [self.face_detection, self.face_detection_short]:
            results = detector.process(image_array)
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)
                w = int(bbox.width * width)
                h = int(bbox.height * height)
                return (x, y, w, h)
        
        # Strategy 2: Try with enhanced image (CLAHE)
        if self.enable_preprocessing:
            enhanced = self._enhance_image(image_array)
            for detector in [self.face_detection, self.face_detection_short]:
                results = detector.process(enhanced)
                if results.detections:
                    bbox = results.detections[0].location_data.relative_bounding_box
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)
                    return (x, y, w, h)
        
        # Strategy 3: Try multi-scale detection
        for scale_factor in [1.5, 2.0, 0.75, 0.5]:
            new_w = int(width * scale_factor)
            new_h = int(height * scale_factor)
            if new_w < 64 or new_h < 64 or new_w > 4096 or new_h > 4096:
                continue
            
            scaled = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            for detector in [self.face_detection, self.face_detection_short]:
                results = detector.process(scaled)
                if results.detections:
                    bbox = results.detections[0].location_data.relative_bounding_box
                    # Convert back to original image coordinates
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)
                    return (x, y, w, h)
        
        # Strategy 4: Try with lower confidence thresholds
        for conf in [0.3, 0.2, 0.1]:
            try:
                low_conf_detector = self.mp_face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=conf
                )
                results = low_conf_detector.process(image_array)
                low_conf_detector.close()
                
                if results.detections:
                    bbox = results.detections[0].location_data.relative_bounding_box
                    x = int(bbox.xmin * width)
                    y = int(bbox.ymin * height)
                    w = int(bbox.width * width)
                    h = int(bbox.height * height)
                    return (x, y, w, h)
            except Exception:
                continue
        
        # Strategy 5: Use face mesh landmarks to estimate bbox
        results = self._process_with_fallback(image_array)
        if results and results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            xs = [lm.x * width for lm in face_landmarks.landmark]
            ys = [lm.y * height for lm in face_landmarks.landmark]
            x = int(min(xs))
            y = int(min(ys))
            w = int(max(xs) - min(xs))
            h = int(max(ys) - min(ys))
            # Add some padding
            pad = int(max(w, h) * 0.1)
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(width - x, w + 2 * pad)
            h = min(height - y, h + 2 * pad)
            return (x, y, w, h)
        
        return None
    
    def batch_process(
        self,
        image_paths: List[Union[str, Path]]
    ) -> List[Optional[np.ndarray]]:
        results = []
        for image_path in image_paths:
            landmarks = self.get_lmk_478(image_path)
            results.append(landmarks)
        return results
    
    def get_landmark_subset(
        self,
        landmarks: np.ndarray,
        indices: List[int]
    ) -> np.ndarray:
        return landmarks[indices]
    
    @staticmethod
    def get_face_region_indices() -> dict:
        return {
            # Silhouette / Face oval
            'face_oval': [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ],
            'left_eye': [
                33, 7, 163, 144, 145, 153, 154, 155, 133,
                173, 157, 158, 159, 160, 161, 246
            ],
            'right_eye': [
                362, 382, 381, 380, 374, 373, 390, 249,
                263, 466, 388, 387, 386, 385, 384, 398
            ],
            'left_eyebrow': [
                46, 53, 52, 65, 55, 70, 63, 105, 66, 107
            ],
            'right_eyebrow': [
                276, 283, 282, 295, 285, 300, 293, 334, 296, 336
            ],
            'nose_bridge': [
                168, 6, 197, 195, 5, 4, 1, 19, 94, 2
            ],
            'nose_tip': [
                4, 5, 195, 197
            ],
            'lips_outer': [
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                291, 185, 40, 39, 37, 0, 267, 269, 270, 409
            ],
            'lips_inner': [
                78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                308, 191, 80, 81, 82, 13, 312, 311, 310, 415
            ],
            'left_iris': [
                468, 469, 470, 471, 472
            ],
            'right_iris': [
                473, 474, 475, 476, 477
            ]
        }
    
    def get_face_bounding_box(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        face_index: int = 0,
        return_format: str = 'xyxy'
    ) -> Optional[Union[np.ndarray, dict]]:
        try:
            # Preprocess image
            image_array, height, width = self._preprocess_image(image)
            
            # Detect faces using MediaPipe Face Detection
            results = self.face_detection.process(image_array)
            
            # Check if any faces were detected
            if not results.detections:
                print("Warning: No faces detected in the image")
                return None
            
            # Check if requested face index is valid
            num_faces = len(results.detections)
            if face_index >= num_faces:
                print(
                    f"Warning: Requested face index {face_index} but only "
                    f"{num_faces} face(s) detected. Using face 0."
                )
                face_index = 0
            
            # Get the detection for the specified face
            detection = results.detections[face_index]
            
            # Extract bounding box in normalized coordinates
            bbox = detection.location_data.relative_bounding_box
            xmin_norm = bbox.xmin
            ymin_norm = bbox.ymin
            width_norm = bbox.width
            height_norm = bbox.height
            
            # Convert to pixel coordinates
            xmin_pixel = int(xmin_norm * width)
            ymin_pixel = int(ymin_norm * height)
            width_pixel = int(width_norm * width)
            height_pixel = int(height_norm * height)
            xmax_pixel = xmin_pixel + width_pixel
            ymax_pixel = ymin_pixel + height_pixel
            
            # Extract keypoints if available
            keypoints = {}
            if detection.location_data.relative_keypoints:
                keypoint_names = [
                    'right_eye', 'left_eye', 'nose_tip',
                    'mouth_center', 'right_ear_tragion', 'left_ear_tragion'
                ]
                for idx, kp in enumerate(detection.location_data.relative_keypoints):
                    if idx < len(keypoint_names):
                        keypoints[keypoint_names[idx]] = {
                            'x': int(kp.x * width),
                            'y': int(kp.y * height),
                            'x_normalized': kp.x,
                            'y_normalized': kp.y
                        }
            
            # Get detection confidence
            confidence = detection.score[0] if detection.score else 0.0
            
            # Return in requested format
            if return_format == 'xyxy':
                return np.array([xmin_pixel, ymin_pixel, xmax_pixel, ymax_pixel], dtype=np.int32)
            
            elif return_format == 'xywh':
                return np.array([xmin_pixel, ymin_pixel, width_pixel, height_pixel], dtype=np.int32)
            
            elif return_format == 'normalized':
                return {
                    'xmin': xmin_norm,
                    'ymin': ymin_norm,
                    'width': width_norm,
                    'height': height_norm
                }
            
            elif return_format == 'all':
                return {
                    'xyxy': np.array([xmin_pixel, ymin_pixel, xmax_pixel, ymax_pixel], dtype=np.int32),
                    'xywh': np.array([xmin_pixel, ymin_pixel, width_pixel, height_pixel], dtype=np.int32),
                    'normalized': {
                        'xmin': xmin_norm,
                        'ymin': ymin_norm,
                        'width': width_norm,
                        'height': height_norm
                    },
                    'confidence': confidence,
                    'keypoints': keypoints,
                    'image_width': width,
                    'image_height': height
                }
            
            else:
                raise ValueError(
                    f"Unsupported return_format: {return_format}. "
                    "Expected 'xyxy', 'xywh', 'normalized', or 'all'."
                )
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None
        except Exception as e:
            print(f"Error during face bounding box detection: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_all_faces_bounding_boxes(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        return_format: str = 'xyxy'
    ) -> Optional[List[Union[np.ndarray, dict]]]:
        try:
            # Preprocess image
            image_array, height, width = self._preprocess_image(image)
            
            # Detect faces using MediaPipe Face Detection
            results = self.face_detection.process(image_array)
            
            # Check if any faces were detected
            if not results.detections:
                print("Warning: No faces detected in the image")
                return None
            
            # Extract bounding boxes for all detected faces
            all_bboxes = []
            for face_idx, detection in enumerate(results.detections):
                # Extract bounding box in normalized coordinates
                bbox = detection.location_data.relative_bounding_box
                xmin_norm = bbox.xmin
                ymin_norm = bbox.ymin
                width_norm = bbox.width
                height_norm = bbox.height
                
                # Convert to pixel coordinates
                xmin_pixel = int(xmin_norm * width)
                ymin_pixel = int(ymin_norm * height)
                width_pixel = int(width_norm * width)
                height_pixel = int(height_norm * height)
                xmax_pixel = xmin_pixel + width_pixel
                ymax_pixel = ymin_pixel + height_pixel
                
                # Extract keypoints if available
                keypoints = {}
                if detection.location_data.relative_keypoints:
                    keypoint_names = [
                        'right_eye', 'left_eye', 'nose_tip',
                        'mouth_center', 'right_ear_tragion', 'left_ear_tragion'
                    ]
                    for idx, kp in enumerate(detection.location_data.relative_keypoints):
                        if idx < len(keypoint_names):
                            keypoints[keypoint_names[idx]] = {
                                'x': int(kp.x * width),
                                'y': int(kp.y * height),
                                'x_normalized': kp.x,
                                'y_normalized': kp.y
                            }
                
                # Get detection confidence
                confidence = detection.score[0] if detection.score else 0.0
                
                # Format based on return_format
                if return_format == 'xyxy':
                    all_bboxes.append(np.array([xmin_pixel, ymin_pixel, xmax_pixel, ymax_pixel], dtype=np.int32))
                
                elif return_format == 'xywh':
                    all_bboxes.append(np.array([xmin_pixel, ymin_pixel, width_pixel, height_pixel], dtype=np.int32))
                
                elif return_format == 'normalized':
                    all_bboxes.append({
                        'xmin': xmin_norm,
                        'ymin': ymin_norm,
                        'width': width_norm,
                        'height': height_norm
                    })
                
                elif return_format == 'all':
                    all_bboxes.append({
                        'xyxy': np.array([xmin_pixel, ymin_pixel, xmax_pixel, ymax_pixel], dtype=np.int32),
                        'xywh': np.array([xmin_pixel, ymin_pixel, width_pixel, height_pixel], dtype=np.int32),
                        'normalized': {
                            'xmin': xmin_norm,
                            'ymin': ymin_norm,
                            'width': width_norm,
                            'height': height_norm
                        },
                        'confidence': confidence,
                        'keypoints': keypoints,
                        'image_width': width,
                        'image_height': height
                    })
            
            return all_bboxes
            
        except Exception as e:
            print(f"Error during multi-face bounding box detection: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def visualize_landmarks(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        landmarks: Optional[np.ndarray] = None,
        show_indices: bool = False,
        point_size: int = 1,
        point_color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        try:
            # Preprocess image
            image_array, height, width = self._preprocess_image(image)
            
            # Get landmarks if not provided
            if landmarks is None:
                landmarks = self.get_lmk_478(image)
                if landmarks is None:
                    return image_array
            
            output_image = image_array.copy()
            
            for idx, (x, y) in enumerate(landmarks):
                cv2.circle(
                    output_image,
                    (int(x), int(y)),
                    point_size,
                    point_color,
                    -1
                )
                
                if show_indices:
                    cv2.putText(
                        output_image,
                        str(idx),
                        (int(x) + 2, int(y) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.2,
                        (255, 255, 255),
                        1
                    )
            
            return output_image
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            return image_array if 'image_array' in locals() else None
    
    def visualize_bounding_boxes(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        bboxes: Optional[List[Union[np.ndarray, dict]]] = None,
        bbox_color: Tuple[int, int, int] = (0, 255, 0),
        bbox_thickness: int = 2,
        show_confidence: bool = True,
        show_keypoints: bool = True
    ) -> np.ndarray:
        try:
            # Preprocess image
            image_array, height, width = self._preprocess_image(image)
            
            # Get bounding boxes if not provided
            if bboxes is None:
                bboxes = self.get_all_faces_bounding_boxes(image, return_format='all')
                if bboxes is None:
                    return image_array
            
            output_image = image_array.copy()
            
            for idx, bbox in enumerate(bboxes):
                if isinstance(bbox, dict) and 'xyxy' in bbox:
                    xyxy = bbox['xyxy']
                    confidence = bbox.get('confidence', None)
                    keypoints = bbox.get('keypoints', {})
                elif isinstance(bbox, np.ndarray):
                    if len(bbox) == 4:
                        xyxy = bbox
                        confidence = None
                        keypoints = {}
                    else:
                        continue
                else:
                    continue
                
                # Extract coordinates
                x_min, y_min, x_max, y_max = xyxy
                
                # Draw bounding box
                cv2.rectangle(
                    output_image,
                    (int(x_min), int(y_min)),
                    (int(x_max), int(y_max)),
                    bbox_color,
                    bbox_thickness
                )
                
                if show_confidence and confidence is not None:
                    text = f"Face {idx}: {confidence:.2f}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    
                    cv2.rectangle(
                        output_image,
                        (int(x_min), int(y_min) - text_size[1] - 10),
                        (int(x_min) + text_size[0] + 5, int(y_min)),
                        bbox_color,
                        -1
                    )
                    
                    cv2.putText(
                        output_image,
                        text,
                        (int(x_min) + 2, int(y_min) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                
                if show_keypoints and keypoints:
                    for kp_name, kp_data in keypoints.items():
                        kp_x = kp_data['x']
                        kp_y = kp_data['y']
                        cv2.circle(
                            output_image,
                            (int(kp_x), int(kp_y)),
                            3,
                            (255, 0, 0),
                            -1
                        )
            
            return output_image
            
        except Exception as e:
            print(f"Error during bounding box visualization: {e}")
            import traceback
            traceback.print_exc()
            return image_array if 'image_array' in locals() else None


if __name__ == "__main__":
    import sys
    
    # Initialize detector with refined landmarks (478 points)
    detector = FacialLandmarkDetector(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    # Example 1: Detect all landmark formats with get_lmk_full
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        print(f"Processing image: {image_path}")
        print("="*80)
        
        # Get all landmark formats in one call
        result = detector.get_lmk_full(image_path)
        
        if result is not None:
            print(f"\n✓ Successfully detected landmarks")
            print(f"\nImage dimensions:")
            print(f"  Width: {result['image_width']}")
            print(f"  Height: {result['image_height']}")
            
            print(f"\n2D Landmark formats:")
            print(f"  ldm478 (Full MediaPipe): {result['ldm478'].shape}")
            print(f"  ldm468 (MediaPipe without iris): {result['ldm468'].shape}")
            print(f"  ldm68 (Dlib format): {result['ldm68'].shape}")
            if result['ldm105_flame'] is not None:
                print(f"  ldm105_flame (FLAME subset): {result['ldm105_flame'].shape}")
            else:
                print(f"  ldm105_flame: Not available (mapping file not found)")
            
            print(f"\n3D Landmark formats:")
            print(f"  ldm478_3d (Full MediaPipe 3D): {result['ldm478_3d'].shape}")
            print(f"  ldm468_3d (MediaPipe 3D without iris): {result['ldm468_3d'].shape}")
            print(f"  ldm68_3d (Dlib format 3D): {result['ldm68_3d'].shape}")
            if result['ldm105_flame_3d'] is not None:
                print(f"  ldm105_flame_3d (FLAME subset 3D): {result['ldm105_flame_3d'].shape}")
            else:
                print(f"  ldm105_flame_3d: Not available (mapping file not found)")
            
            print(f"\nSample 2D landmarks (ldm478):")
            print(f"  Nose tip (index 4): {result['ldm478'][4]}")
            print(f"  Left iris center (index 468): {result['ldm478'][468]}")
            print(f"  Right iris center (index 473): {result['ldm478'][473]}")
            
            print(f"\nSample 3D landmarks (ldm478_3d):")
            print(f"  Nose tip (index 4): {result['ldm478_3d'][4]}")
            print(f"  Left iris center (index 468): {result['ldm478_3d'][468]}")
            print(f"  Right iris center (index 473): {result['ldm478_3d'][473]}")
            
            print(f"\nSample Dlib 2D landmarks (ldm68):")
            print(f"  Nose tip (index 30): {result['ldm68'][30]}")
            print(f"  Left eye corner (index 36): {result['ldm68'][36]}")
            print(f"  Right eye corner (index 45): {result['ldm68'][45]}")
            
            print(f"\nSample Dlib 3D landmarks (ldm68_3d):")
            print(f"  Nose tip (index 30): {result['ldm68_3d'][30]}")
            print(f"  Left eye corner (index 36): {result['ldm68_3d'][36]}")
            print(f"  Right eye corner (index 45): {result['ldm68_3d'][45]}")
            
            # Get face region indices
            regions = detector.get_face_region_indices()
            print(f"\nAvailable regions: {list(regions.keys())}")
            
            # Extract nose landmarks
            nose_landmarks = detector.get_landmark_subset(result['ldm478'], regions['nose_tip'])
            print(f"\nNose tip landmarks (from ldm478):\n{nose_landmarks}")
            
            # Get 3D landmarks separately if needed
            landmarks_3d = detector.get_lmk_478_3d(image_path)
            if landmarks_3d is not None:
                print(f"\n✓ 3D landmarks shape: {landmarks_3d.shape}")
            
            # Visualize ldm478 (save to file)
            output_image = detector.visualize_landmarks(image_path, result['ldm478'])
            output_path = "landmarks_visualization_478.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            print(f"\n✓ Visualization (478 landmarks) saved to: {output_path}")
            
            # Visualize ldm68 (save to file)
            output_image_68 = detector.visualize_landmarks(image_path, result['ldm68'], point_size=2, point_color=(255, 0, 0))
            output_path_68 = "landmarks_visualization_68.jpg"
            cv2.imwrite(output_path_68, cv2.cvtColor(output_image_68, cv2.COLOR_RGB2BGR))
            print(f"✓ Visualization (68 Dlib landmarks) saved to: {output_path_68}")
            
            # Example 2: Detect face bounding box
            print(f"\n" + "="*80)
            print("Face Bounding Box Detection:")
            print("="*80)
            
            # Get bounding box in different formats
            bbox_xyxy = detector.get_face_bounding_box(image_path, return_format='xyxy')
            bbox_xywh = detector.get_face_bounding_box(image_path, return_format='xywh')
            bbox_all = detector.get_face_bounding_box(image_path, return_format='all')
            
            if bbox_xyxy is not None:
                print(f"\n✓ Successfully detected face bounding box")
                print(f"\nBounding box formats:")
                print(f"  xyxy: {bbox_xyxy}")
                print(f"  xywh: {bbox_xywh}")
                
                if bbox_all:
                    print(f"  confidence: {bbox_all['confidence']:.4f}")
                    print(f"\nKeypoints detected: {len(bbox_all['keypoints'])}")
                    for kp_name, kp_data in bbox_all['keypoints'].items():
                        print(f"    {kp_name}: ({kp_data['x']}, {kp_data['y']})")
                
                # Visualize bounding box
                output_bbox_image = detector.visualize_bounding_boxes(image_path)
                output_bbox_path = "bounding_box_visualization.jpg"
                cv2.imwrite(output_bbox_path, cv2.cvtColor(output_bbox_image, cv2.COLOR_RGB2BGR))
                print(f"\n✓ Bounding box visualization saved to: {output_bbox_path}")
            
        else:
            print("✗ Failed to detect landmarks")
    else:
        print("Usage: python facial_landmark_detector.py <image_path>")
        print("\nExample:")
        print("  python facial_landmark_detector.py face_image.jpg")
        print("\nFeatures:")
        print("  - Detects 478/468 MediaPipe facial landmarks")
        print("  - Converts to 68 Dlib landmark format")
        print("  - Extracts 105 FLAME-aligned landmarks")
        print("  - Detects face bounding boxes with confidence scores")
        print("  - Provides facial keypoints (eyes, nose, mouth, ears)")
