import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import cv2
from PIL import Image

from .camera_utils import get_3x4_P_matrix_ortho
from .segmentation import FaceSegmenter
from .ray_intersector import RayMeshIntersector
from .enhancer_singleton import CodeFormerEnhancerSingleton
from hairport.core import FacialLandmarkDetector

class MultiViewLandmarkFuser:
    def __init__(
        self,
        mesh_path: str,
        cam_loc: torch.Tensor,
        cam_rot: torch.Tensor,
        ortho_scale: float = 1.1,
        resolution: int = 1024,
        device: str = 'cuda',
        debug: bool = False,
        debug_dir: str = './debug_outputs',
        super_resolution: bool = True
    ):
        self.device = torch.device(device)
        self.mesh_path = mesh_path
        self.resolution = resolution
        self.ortho_scale = ortho_scale
        self.debug = debug
        self.debug_dir = Path(debug_dir)
        self.super_resolution = super_resolution
        
        if self.debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            (self.debug_dir / 'landmarks_overlay').mkdir(exist_ok=True)
            (self.debug_dir / 'segmentation').mkdir(exist_ok=True)
            (self.debug_dir / 'renders').mkdir(exist_ok=True)
            if self.super_resolution:
                (self.debug_dir / 'renders_raw').mkdir(exist_ok=True)
                (self.debug_dir / 'renders_enhanced').mkdir(exist_ok=True)
        
        self.cam_loc = cam_loc.to(self.device) if isinstance(cam_loc, torch.Tensor) else torch.tensor(cam_loc, device=self.device)
        self.cam_rot = cam_rot.to(self.device) if isinstance(cam_rot, torch.Tensor) else torch.tensor(cam_rot, device=self.device)
        
        self.landmark_detector = FacialLandmarkDetector()
        self.segmenter = FaceSegmenter(device=device)
        self.intersector = RayMeshIntersector(mesh_path, device=device)
        
        # Get or create shared CodeFormer enhancer instance if super_resolution is enabled
        self.enhancer = None
        if self.super_resolution:
            self.enhancer = CodeFormerEnhancerSingleton.get_enhancer(device=str(self.device))
            if self.enhancer is None:
                print("Proceeding without super-resolution enhancement")
                self.super_resolution = False
    
    def compute_landmark_confidence(
        self,
        lmk_2d: np.ndarray,
        face_mask: np.ndarray,
        hair_mask: np.ndarray,
        bg_mask: np.ndarray
    ) -> np.ndarray:
        confidences = np.ones(len(lmk_2d), dtype=np.float32)
        
        h, w = face_mask.shape
        
        for i, (x, y) in enumerate(lmk_2d):
            xi, yi = int(round(x)), int(round(y))
            
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                confidences[i] = 0.0
                continue
            
            if bg_mask[yi, xi] > 0:
                confidences[i] = 0.1
                continue
            
            if hair_mask[yi, xi] > 0:
                confidences[i] = 0.3
                continue
            
            if face_mask[yi, xi] > 0:
                confidences[i] = 1.0
            else:
                confidences[i] = 0.5
        
        return confidences
    
    def process_single_view(
        self,
        image_path: Path,
        cam_loc: torch.Tensor,
        cam_rot: torch.Tensor,
        view_idx: int = 0,
        lmk_2d: np.ndarray = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Apply super-resolution if enabled
        enhanced_image_path = image_path
        if self.super_resolution and self.enhancer is not None:
            try:
                # Load raw image
                raw_image = Image.open(image_path)
                
                # Save raw image to debug folder if debug is enabled
                if self.debug:
                    raw_debug_path = self.debug_dir / 'renders_raw' / f'view_{view_idx:03d}.png'
                    raw_image.save(raw_debug_path)
                
                # Enhance image using CodeFormer
                from hairport.config import get_config
                _lmk_cfg = get_config().landmark_3d
                enhanced_image = self.enhancer.enhance(
                    raw_image,
                    face_align=True,
                    background_enhance=True,
                    face_upsample=True,
                    upscale=_lmk_cfg.codeformer_upscale,
                    codeformer_fidelity=_lmk_cfg.codeformer_fidelity
                )
                enhanced_image = enhanced_image.resize(raw_image.size, Image.Resampling.LANCZOS)

                # Save enhanced image
                enhanced_image_path = image_path.parent / f'{image_path.stem}_enhanced{image_path.suffix}'
                enhanced_image.save(enhanced_image_path)
                
                # Save enhanced image to debug folder if debug is enabled
                if self.debug:
                    enhanced_debug_path = self.debug_dir / 'renders_enhanced' / f'view_{view_idx:03d}.png'
                    enhanced_image.save(enhanced_debug_path)
                
                # Use enhanced image for landmark detection
                image_path = enhanced_image_path
                
            except Exception as e:
                print(f"Warning: Super-resolution failed for view {view_idx}: {e}")
                print("Using raw render for landmark detection")
        
        # Detect landmarks if not provided
        if lmk_2d is None:
            lmk_result = self.landmark_detector.get_lmk_full(image_path)
            if lmk_result is not None:
                lmk_2d = lmk_result["ldm478"]
        
        if lmk_2d is None:
            return None, None, None
        
        masks = self.segmenter.extract_masks(image_path)
        
        confidences = self.compute_landmark_confidence(
            lmk_2d,
            masks['face_mask'],
            masks['hair_mask'],
            masks['bg_mask']
        )
        
        if self.debug:
            self._save_debug_visualization(
                enhanced_image_path if self.super_resolution and self.enhancer is not None else image_path, 
                lmk_2d, 
                confidences, 
                masks, 
                view_idx
            )
        
        P, K, RT = get_3x4_P_matrix_ortho(
            cam_loc=cam_loc,
            cam_rot=cam_rot,
            ortho_scale=self.ortho_scale,
            resolution_x=self.resolution,
            resolution_y=self.resolution,
            device=self.device
        )
        
        lmk_2d_torch = torch.tensor(lmk_2d, device=self.device, dtype=torch.float32)
        
        ray_origins, ray_directions = self.intersector.compute_ray_from_pixel(
            lmk_2d_torch, K, RT, ortho=True
        )
        
        hit_points, hit_faces, hit_valid = self.intersector.ray_triangle_intersection_batch(
            ray_origins, ray_directions
        )
        
        if hit_points is None:
            return None, None, None
        
        confidences_torch = torch.tensor(confidences, device=self.device, dtype=torch.float32)
        confidences_torch = confidences_torch * hit_valid.float()
        
        return hit_points, confidences_torch, hit_valid
    
    def _save_debug_visualization(
        self,
        image_path: Path,
        landmarks_2d: np.ndarray,
        confidences: np.ndarray,
        masks: Dict[str, np.ndarray],
        view_idx: int
    ):
        image = cv2.imread(str(image_path))
        if image is None:
            image = np.array(Image.open(image_path).convert('RGB'))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save the original rendered image
        render_output_path = self.debug_dir / 'renders' / f'view_{view_idx:02d}.png'
        cv2.imwrite(str(render_output_path), image)
        
        overlay = image.copy()
        
        for i, (lmk, conf) in enumerate(zip(landmarks_2d, confidences)):
            x, y = int(lmk[0]), int(lmk[1])
            
            if conf > 0.8:
                color = (0, 255, 0)
            elif conf > 0.5:
                color = (0, 255, 255)
            elif conf > 0.2:
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            
            cv2.circle(overlay, (x, y), 3, color, -1)
            cv2.circle(overlay, (x, y), 5, color, 1)
            
            if i % 5 == 0:
                cv2.putText(
                    overlay, str(i), (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1
                )
        
        legend_y = 20
        cv2.putText(overlay, "Confidence:", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(overlay, (120, legend_y - 5), 3, (0, 255, 0), -1)
        cv2.putText(overlay, "High (>0.8)", (130, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        legend_y += 20
        cv2.circle(overlay, (120, legend_y - 5), 3, (0, 255, 255), -1)
        cv2.putText(overlay, "Medium (>0.5)", (130, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        legend_y += 20
        cv2.circle(overlay, (120, legend_y - 5), 3, (0, 165, 255), -1)
        cv2.putText(overlay, "Low (>0.2)", (130, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        legend_y += 20
        cv2.circle(overlay, (120, legend_y - 5), 3, (0, 0, 255), -1)
        cv2.putText(overlay, "Very Low", (130, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        output_path = self.debug_dir / 'landmarks_overlay' / f'view_{view_idx:02d}_landmarks.png'
        cv2.imwrite(str(output_path), overlay)
        
        face_vis = cv2.cvtColor((masks['face_mask'] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        hair_vis = cv2.cvtColor((masks['hair_mask'] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        bg_vis = cv2.cvtColor((masks['bg_mask'] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        seg_combined = np.hstack([face_vis, hair_vis, bg_vis])
        seg_output_path = self.debug_dir / 'segmentation' / f'view_{view_idx:02d}_masks.png'
        cv2.imwrite(str(seg_output_path), seg_combined)
    
    def compute_final_landmark_confidences(
        self,
        landmarks_3d_list: List[torch.Tensor],
        confidences_list: List[torch.Tensor],
        valid_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute robust final confidence values for each landmark based on multi-view fusion.
        
        This algorithm uses a weighted combination of four complementary confidence factors:
        
        1. **Mean Per-View Confidence (35% weight)**: 
           - Average of segmentation-based confidences across all valid views
           - Reflects whether landmarks fall on face (1.0), hair (0.3), background (0.1), etc.
           - Captures local 2D image quality and segmentation reliability
        
        2. **Visibility Score (25% weight)**:
           - Ratio of views where landmark was successfully detected
           - Higher visibility → more reliable triangulation
           - Penalizes landmarks visible in only a few views
        
        3. **Geometric Consistency (30% weight)**:
           - Measures spatial agreement of 3D positions across views
           - Uses Median Absolute Deviation (MAD) as robust spread metric
           - Exponential decay: exp(-MAD / 0.01) where 0.01m = 1cm threshold
           - Low spatial variance → high consistency → reliable 3D position
        
        4. **Confidence Stability (10% weight)**:
           - Standard deviation of per-view confidence scores
           - Exponential decay: exp(-std / 0.3) where 0.3 is stability threshold
           - Low variance → stable detections across views
           - High variance → uncertain/inconsistent detections
        
        Design Rationale:
        - Geometric consistency weighted highest (30%) as it directly measures 3D reliability
        - Per-view confidence (35%) captures 2D detection quality
        - Visibility (25%) ensures sufficient multi-view coverage
        - Stability (10%) acts as a consistency check but lower weight to avoid over-penalization
        
        Args:
            landmarks_3d_list: List of [N, 3] tensors with 3D landmarks per view
            confidences_list: List of [N] tensors with per-view confidence scores
            valid_list: List of [N] boolean tensors indicating valid detections
            
        Returns:
            torch.Tensor of shape [N] with final confidence scores in [0, 1]
        """
        # Dynamically determine number of landmarks from first valid view
        num_landmarks = len(landmarks_3d_list[0]) if landmarks_3d_list else 0
        num_views = len(landmarks_3d_list)
        final_confidences = torch.zeros(num_landmarks, device=self.device, dtype=torch.float32)
        
        for lmk_idx in range(num_landmarks):
            # Collect valid observations for this landmark
            valid_confs = []
            valid_points = []
            
            for view_idx, (lmks, confs, valid) in enumerate(zip(landmarks_3d_list, confidences_list, valid_list)):
                if valid[lmk_idx]:
                    valid_confs.append(confs[lmk_idx])
                    valid_points.append(lmks[lmk_idx])
            
            if len(valid_confs) == 0:
                final_confidences[lmk_idx] = 0.0
                continue
            
            valid_confs = torch.stack(valid_confs)
            valid_points = torch.stack(valid_points)
            
            # Component 1: Average per-view confidence (segmentation quality)
            mean_conf = valid_confs.mean()
            
            # Component 2: Visibility factor (how many views detected this landmark)
            visibility_ratio = len(valid_confs) / num_views
            visibility_score = torch.tensor(visibility_ratio, device=self.device, dtype=torch.float32)
            
            # Component 3: Geometric consistency (spatial agreement across views)
            if len(valid_points) >= 2:
                # Compute pairwise distances between all valid points
                pairwise_dists = torch.cdist(valid_points.unsqueeze(0), valid_points.unsqueeze(0)).squeeze(0)
                
                # Use median absolute deviation (MAD) as a robust measure of spread
                median_point = valid_points.median(dim=0).values
                deviations = torch.norm(valid_points - median_point.unsqueeze(0), dim=1)
                mad = deviations.median()
                
                # Convert spatial variance to a consistency score
                # Use exponential decay: high variance -> low consistency
                # Scale by 0.01 (assuming landmarks within 1cm are consistent)
                consistency_score = torch.exp(-mad / 0.01)
            else:
                # Single view: moderate consistency assumed
                consistency_score = torch.tensor(0.7, device=self.device, dtype=torch.float32)
            
            # Component 4: Confidence variance (stable vs. uncertain detections)
            if len(valid_confs) >= 2:
                conf_std = valid_confs.std()
                # Low variance in confidences -> more reliable
                # High variance -> some views are uncertain
                conf_stability = torch.exp(-conf_std / 0.3)  # 0.3 std threshold
            else:
                conf_stability = torch.tensor(1.0, device=self.device, dtype=torch.float32)
            
            # Weighted combination of all factors
            # Weights reflect relative importance of each component
            w_mean = 0.35        # Per-view quality
            w_visibility = 0.25  # Multi-view coverage
            w_consistency = 0.30 # Geometric agreement
            w_stability = 0.10   # Confidence stability
            
            final_conf = (
                w_mean * mean_conf +
                w_visibility * visibility_score +
                w_consistency * consistency_score +
                w_stability * conf_stability
            )
            
            # Clamp to [0, 1] range
            final_confidences[lmk_idx] = torch.clamp(final_conf, 0.0, 1.0)
        
        return final_confidences
    
    def fuse_multi_view_landmarks(
        self,
        landmarks_3d_list: List[torch.Tensor],
        confidences_list: List[torch.Tensor],
        valid_list: List[torch.Tensor]
    ) -> torch.Tensor:
        # Dynamically determine number of landmarks from first valid view
        num_landmarks = len(landmarks_3d_list[0]) if landmarks_3d_list else 0
        fused_landmarks = torch.zeros((num_landmarks, 3), device=self.device, dtype=torch.float32)
        
        for lmk_idx in range(num_landmarks):
            points = []
            weights = []
            
            for view_idx, (lmks, confs, valid) in enumerate(zip(landmarks_3d_list, confidences_list, valid_list)):
                if valid[lmk_idx]:
                    points.append(lmks[lmk_idx])
                    weights.append(confs[lmk_idx])
            
            if len(points) == 0:
                continue
            
            points = torch.stack(points)
            weights = torch.stack(weights)
            
            weights = weights / (weights.sum() + 1e-8)
            
            fused_point = (points * weights.unsqueeze(1)).sum(dim=0)
            fused_landmarks[lmk_idx] = fused_point
        
        return fused_landmarks
    
    def get_confidence_statistics(
        self,
        confidences: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute summary statistics for landmark confidences.
        
        Args:
            confidences: Tensor of shape [N] with confidence scores (N = number of landmarks)
            
        Returns:
            Dictionary with statistical measures
        """
        confs_np = confidences.cpu().numpy()
        
        return {
            'mean': float(confidences.mean()),
            'std': float(confidences.std()),
            'min': float(confidences.min()),
            'max': float(confidences.max()),
            'median': float(confidences.median()),
            'q25': float(np.percentile(confs_np, 25)),
            'q75': float(np.percentile(confs_np, 75)),
            'num_high': int((confidences > 0.8).sum()),  # High confidence (>0.8)
            'num_medium': int(((confidences > 0.5) & (confidences <= 0.8)).sum()),  # Medium (0.5-0.8)
            'num_low': int((confidences <= 0.5).sum()),  # Low (<0.5)
        }
    
    def optimize_landmarks(
        self,
        initial_landmarks: torch.Tensor,
        num_iterations: int = 50,
        lr: float = 0.001
    ) -> torch.Tensor:
        landmarks = initial_landmarks.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([landmarks], lr=lr)
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            smoothness_loss = 0.0
            for i in range(len(landmarks) - 1):
                dist = torch.norm(landmarks[i + 1] - landmarks[i])
                smoothness_loss += dist
            
            jaw_indices = list(range(0, 17))
            for i in range(len(jaw_indices) - 2):
                p1 = landmarks[jaw_indices[i]]
                p2 = landmarks[jaw_indices[i + 1]]
                p3 = landmarks[jaw_indices[i + 2]]
                
                v1 = p2 - p1
                v2 = p3 - p2
                
                curvature = torch.norm(v2 - v1)
                smoothness_loss += curvature * 0.1
            
            mesh_vertices = self.intersector.vertices
            closest_dists = []
            for lmk in landmarks:
                dists = torch.norm(mesh_vertices - lmk.unsqueeze(0), dim=1)
                min_dist = torch.min(dists)
                closest_dists.append(min_dist)
            
            mesh_proximity_loss = torch.stack(closest_dists).mean()
            
            total_loss = 0.01 * smoothness_loss + 10.0 * mesh_proximity_loss
            
            total_loss.backward()
            optimizer.step()
        
        return landmarks.detach()
    
    def process_multi_view(
        self,
        rendered_paths: List[Path],
        cam_locs: torch.Tensor,
        cam_rots: torch.Tensor,
        optimize: bool = True
    ) -> Dict[str, torch.Tensor]:
        landmarks_3d_list = []
        confidences_list = []
        valid_list = []
        landmarks_2d_list = []
        
        for view_idx, (img_path, cam_loc, cam_rot) in enumerate(zip(rendered_paths, cam_locs, cam_rots)):
            print(f"Processing view {view_idx}...")
            
            lmk_2d = self.landmark_detector.get_lmk_478(img_path)
            if lmk_2d is not None:
                landmarks_2d_list.append(lmk_2d)
            else:
                print(f"  Warning: No face detected in view {view_idx}")
            
            # Pass the detected landmarks to avoid redundant detection
            lmks_3d, confs, valid = self.process_single_view(
                img_path, cam_loc, cam_rot, view_idx, lmk_2d
            )
            
            if lmks_3d is not None:
                landmarks_3d_list.append(lmks_3d)
                confidences_list.append(confs)
                valid_list.append(valid)
            else:
                print(f"  Warning: Failed to process view {view_idx}")
        
        if len(landmarks_3d_list) == 0:
            print("\nERROR: No valid landmarks detected in any view!")
            print("Possible reasons:")
            print("  - No face detected in rendered images")
            print("  - Face detection model failed")
            print("  - Rendered images are corrupted or empty")
            raise RuntimeError("No valid landmarks detected in any view. Check rendered images and face detection.")
        
        # Determine number of landmarks from first valid view
        num_landmarks = len(landmarks_3d_list[0])
        print(f"\nDetected {num_landmarks} landmarks per view")
        print(f"Fusing landmarks from {len(landmarks_3d_list)} views...")
        
        fused_landmarks = self.fuse_multi_view_landmarks(
            landmarks_3d_list, confidences_list, valid_list
        )
        
        # Compute final confidence scores for each landmark
        print("Computing final confidence scores...")
        final_confidences = self.compute_final_landmark_confidences(
            landmarks_3d_list, confidences_list, valid_list
        )
        
        # Print confidence statistics
        conf_stats = self.get_confidence_statistics(final_confidences)
        print(f"Confidence Statistics:")
        print(f"  Mean: {conf_stats['mean']:.3f} ± {conf_stats['std']:.3f}")
        print(f"  Range: [{conf_stats['min']:.3f}, {conf_stats['max']:.3f}]")
        print(f"  Median: {conf_stats['median']:.3f} (Q25={conf_stats['q25']:.3f}, Q75={conf_stats['q75']:.3f})")
        print(f"  Quality: {conf_stats['num_high']} high, {conf_stats['num_medium']} medium, {conf_stats['num_low']} low confidence landmarks")
        
        if optimize:
            print("Optimizing landmarks...")
            fused_landmarks = self.optimize_landmarks(fused_landmarks)
        
        vertex_indices = []
        for lmk in fused_landmarks:
            vertex_idx = self.intersector.find_nearest_vertex(lmk)
            vertex_indices.append(vertex_idx)
        
        return {
            'landmarks_3d': fused_landmarks,
            'vertex_indices': torch.tensor(vertex_indices, device=self.device, dtype=torch.long),
            'per_view_landmarks': landmarks_3d_list,
            'per_view_confidences': confidences_list,
            'confidences': final_confidences
        }
