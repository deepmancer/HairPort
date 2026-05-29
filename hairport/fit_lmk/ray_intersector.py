import torch
import numpy as np
import trimesh
import logging
from typing import Tuple, Optional
from pathlib import Path
from .transforms import blender_import_rotation

logger = logging.getLogger(__name__)


class RayMeshIntersector:
    def __init__(self, mesh_path: str, device='cuda'):
        self.device = torch.device(device)
        
        # Load mesh WITHOUT transformations
        # The mesh stays in its original coordinate system
        mesh_path_obj = Path(mesh_path)
        mesh_ext = mesh_path_obj.suffix.lower()
        self.mesh = trimesh.load(mesh_path, force='mesh')
        
        # Store the transformation that Blender applies for this file type
        # We'll use this to transform rays from render space to mesh space
        self.blender_transform = blender_import_rotation(mesh_ext)
        
        # Compute inverse transform: this takes rays from render space to mesh space
        self.render_to_mesh_transform = np.linalg.inv(self.blender_transform)
        
        self.vertices = torch.tensor(
            self.mesh.vertices, 
            dtype=torch.float32, 
            device=self.device
        )
        self.faces = torch.tensor(
            self.mesh.faces, 
            dtype=torch.long, 
            device=self.device
        )
        
        self.backend = "triangle"
        try:
            from trimesh.ray.ray_pyembree import RayMeshIntersector as EmbreeIntersector
            self.trimesh_intersector = EmbreeIntersector(self.mesh)
            self.backend = "embree"
        except (ImportError, AttributeError):
            logger.warning(
                "Embree ray backend is unavailable; falling back to trimesh triangle "
                "intersection. Install the pinned Embree dependency for production performance."
            )
            self.trimesh_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)

    @staticmethod
    def available_backend() -> str:
        """Report the ray backend that will be used without loading a mesh."""
        try:
            from trimesh.ray.ray_pyembree import RayMeshIntersector as _EmbreeIntersector
            del _EmbreeIntersector
            return "embree"
        except (ImportError, AttributeError):
            return "triangle"

    @staticmethod
    def preflight_backend() -> str:
        """Verify the selected backend can execute a minimal intersection."""
        backend = RayMeshIntersector.available_backend()
        mesh = trimesh.Trimesh(
            vertices=np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]),
            faces=np.array([[0, 1, 2]]),
            process=False,
        )
        try:
            if backend == "embree":
                from trimesh.ray.ray_pyembree import RayMeshIntersector as Intersector
            else:
                from trimesh.ray.ray_triangle import RayMeshIntersector as Intersector
            intersector = Intersector(mesh)
            intersector.intersects_location(
                np.array([[0.0, 0.0, 1.0]]),
                np.array([[0.0, 0.0, -1.0]]),
                multiple_hits=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Landmark3D ray backend '{backend}' cannot execute intersections. "
                "Install the pinned Embree backend (preferred) or the trimesh/rtree "
                "fallback dependencies before inference."
            ) from exc
        return backend
    
    def ray_triangle_intersection_batch(
        self, 
        ray_origins: torch.Tensor, 
        ray_directions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ray_origins_np = ray_origins.cpu().numpy()
        ray_directions_np = ray_directions.cpu().numpy()
        
        locations, index_ray, index_tri = self.trimesh_intersector.intersects_location(
            ray_origins_np,
            ray_directions_np,
            multiple_hits=False
        )
        
        if len(locations) == 0:
            return None, None, None
        
        hit_points = torch.zeros((len(ray_origins), 3), device=self.device, dtype=torch.float32)
        hit_faces = torch.full((len(ray_origins),), -1, device=self.device, dtype=torch.long)
        hit_valid = torch.zeros(len(ray_origins), device=self.device, dtype=torch.bool)
        
        locations_torch = torch.tensor(locations, device=self.device, dtype=torch.float32)
        index_ray_torch = torch.tensor(index_ray, device=self.device, dtype=torch.long)
        index_tri_torch = torch.tensor(index_tri, device=self.device, dtype=torch.long)
        
        hit_points[index_ray_torch] = locations_torch
        hit_faces[index_ray_torch] = index_tri_torch
        hit_valid[index_ray_torch] = True
        
        return hit_points, hit_faces, hit_valid
    
    def compute_ray_from_pixel(
        self,
        pixel_coords: torch.Tensor,
        K: torch.Tensor,
        RT: torch.Tensor,
        ortho: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pixel_coords.dim() == 1:
            pixel_coords = pixel_coords.unsqueeze(0)
        
        N = pixel_coords.shape[0]
        
        u = pixel_coords[:, 0]
        v = pixel_coords[:, 1]
        
        if ortho:
            K_inv = torch.inverse(K)
            
            pixel_homog = torch.stack([u, v, torch.ones_like(u)], dim=1)
            
            cam_coords = (K_inv @ pixel_homog.T).T
            
            cam_coords_3d = torch.cat([
                cam_coords[:, :2],
                torch.zeros((N, 1), device=self.device, dtype=torch.float32)
            ], dim=1)
            
            R = RT[:3, :3]
            t = RT[:3, 3]
            
            R_inv = R.T
            
            # Compute rays in render/world space
            ray_origin_render = R_inv @ (cam_coords_3d.T - t.unsqueeze(1))
            ray_origin_render = ray_origin_render.T
            
            ray_direction_cam = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32)
            ray_direction_render = R_inv @ ray_direction_cam
            ray_direction_render = ray_direction_render / torch.norm(ray_direction_render)
            
            # Transform rays from render space to mesh space
            # This accounts for the coordinate transformation Blender applies
            transform_torch = torch.tensor(
                self.render_to_mesh_transform, 
                device=self.device, 
                dtype=torch.float32
            )
            
            # Transform ray origins
            ray_origin_mesh = (transform_torch @ ray_origin_render.T).T
            
            # Transform ray direction (only rotation, no translation)
            ray_direction_mesh = transform_torch @ ray_direction_render
            ray_direction_mesh = ray_direction_mesh / torch.norm(ray_direction_mesh)
            
            ray_directions_mesh = ray_direction_mesh.unsqueeze(0).repeat(N, 1)
        else:
            raise NotImplementedError("Perspective projection not implemented")
        
        return ray_origin_mesh, ray_directions_mesh
    
    def find_nearest_vertex(self, point: torch.Tensor) -> int:
        distances = torch.norm(self.vertices - point.unsqueeze(0), dim=1)
        return torch.argmin(distances).item()
