"""
Mesh processing utilities using Open3D and trimesh.
"""

import gc
import logging
from typing import Tuple

import numpy as np
import open3d as o3d
import torch
import trimesh

logger = logging.getLogger("mesh_process")


### Mesh Utils using Open3D and trimesh ###

def merge_close_vertices_o3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    threshold: float = 0.0001,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge vertices that are closer than a threshold using trimesh.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        threshold: Distance threshold for merging
        verbose: Whether to print progress
        
    Returns:
        Merged vertices and updated faces
    """
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.merge_vertices(merge_tex=False, merge_norm=False)
        
        return np.array(mesh.vertices), np.array(mesh.faces)
    except Exception as e:
        logger.debug(f"Vertex merging failed: {e}")
        return vertices, faces


def remove_isolated_pieces_o3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    min_component_ratio: float = 0.02,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove small isolated mesh components using trimesh.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        min_component_ratio: Minimum ratio of faces to keep a component
        verbose: Whether to print progress
        
    Returns:
        Cleaned vertices and faces
    """
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        components = mesh.split(only_watertight=False)
        
        if len(components) <= 1:
            return vertices, faces
        
        # Find the largest component or components above threshold
        min_faces = int(len(faces) * min_component_ratio)
        valid_components = [c for c in components if len(c.faces) >= min_faces]
        
        if not valid_components:
            # Keep at least the largest component
            valid_components = [max(components, key=lambda c: len(c.faces))]
        
        result_mesh = valid_components[0] if len(valid_components) == 1 else trimesh.util.concatenate(valid_components)
        return np.array(result_mesh.vertices), np.array(result_mesh.faces)
    except Exception as e:
        logger.debug(f"Isolated piece removal failed: {e}")
        return vertices, faces


def fill_holes_trimesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_hole_size: int = 30,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fill small holes in the mesh using trimesh.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        max_hole_size: Maximum number of edges in hole to fill
        verbose: Whether to print progress
        
    Returns:
        Mesh with holes filled
    """
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.fill_holes()
        return np.array(mesh.vertices), np.array(mesh.faces)
    except Exception as e:
        logger.debug(f"Hole filling failed: {e}")
        return vertices, faces


def repair_mesh_trimesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Repair non-manifold edges and remove degenerate faces using trimesh.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        verbose: Whether to print progress
        
    Returns:
        Repaired mesh
    """
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
        mesh.fix_normals()
        return np.array(mesh.vertices), np.array(mesh.faces)
    except Exception as e:
        logger.debug(f"Mesh repair failed: {e}")
        return vertices, faces


def smooth_mesh_laplacian(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 3,
    lambda_factor: float = 0.5,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Laplacian smoothing using Open3D.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        iterations: Number of smoothing iterations
        lambda_factor: Smoothing strength (0-1)
        verbose: Whether to print progress
        
    Returns:
        Smoothed mesh
    """
    try:
        # Create Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        
        # Apply Laplacian smoothing
        o3d_mesh = o3d_mesh.filter_smooth_laplacian(
            number_of_iterations=iterations,
            lambda_filter=lambda_factor
        )
        
        return np.asarray(o3d_mesh.vertices).astype(np.float32), np.asarray(o3d_mesh.triangles).astype(np.int64)
    except Exception as e:
        logger.debug(f"Laplacian smoothing failed: {e}")
        return vertices, faces


def smooth_mesh_taubin(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 10,
    lambda_factor: float = 0.5,
    mu_factor: float = -0.53,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Taubin smoothing using Open3D.
    Taubin smoothing reduces shrinkage compared to Laplacian smoothing.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        iterations: Number of smoothing iterations
        lambda_factor: Positive smoothing factor
        mu_factor: Negative smoothing factor (should be negative)
        verbose: Whether to print progress
        
    Returns:
        Smoothed mesh
    """
    try:
        # Create Open3D mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        
        # Apply Taubin smoothing
        o3d_mesh = o3d_mesh.filter_smooth_taubin(
            number_of_iterations=iterations,
            lambda_filter=lambda_factor,
            mu=mu_factor
        )
        
        return np.asarray(o3d_mesh.vertices).astype(np.float32), np.asarray(o3d_mesh.triangles).astype(np.int64)
    except Exception as e:
        logger.debug(f"Taubin smoothing failed: {e}")
        return vertices, faces


def decimate_mesh_o3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int = 50000,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decimate mesh using Open3D's quadric decimation.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        target_faces: Target number of faces
        verbose: Whether to print progress
        
    Returns:
        Decimated mesh
    """
    if len(faces) <= target_faces:
        return vertices, faces
    
    try:
        # Use Open3D tensor API for decimation
        device = o3d.core.Device("CPU:0")
        dtype_f = o3d.core.float32
        dtype_i = o3d.core.int64
        
        mesh = o3d.t.geometry.TriangleMesh(device)
        mesh.vertex.positions = o3d.core.Tensor(
            vertices.astype(np.float32), dtype_f, device
        )
        mesh.triangle.indices = o3d.core.Tensor(
            faces.astype(np.int64), dtype_i, device
        )
        
        target_reduction = 1.0 - float(target_faces) / len(faces)
        simplified_mesh = mesh.simplify_quadric_decimation(
            target_reduction=target_reduction
        )
        
        return simplified_mesh.vertex.positions.numpy(), simplified_mesh.triangle.indices.numpy()
    except Exception as e:
        logger.debug(f"Decimation failed: {e}")
        return vertices, faces


def compute_vertex_normals(
    vertices: np.ndarray,
    faces: np.ndarray
) -> np.ndarray:
    """
    Compute vertex normals using trimesh.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        
    Returns:
        Vertex normals (N, 3)
    """
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        return np.array(mesh.vertex_normals).astype(np.float32)
    except Exception as e:
        logger.debug(f"Normal computation failed: {e}")
        return np.zeros_like(vertices, dtype=np.float32)


### Pre-process Mesh (pymeshlab-free version) ###
def process_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    threshold: float = 0.0001,
    mincomponentRatio: float = 0.02,
    targetfacenum: int = 100_000,
    maxholesize: int = 30,
    stepsmoothnum: int = 10,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process mesh with various cleanup and smoothing operations.
    This version uses Open3D and trimesh instead of pymeshlab.
    
    Args:
        vertices: Input vertex positions
        faces: Input face indices
        threshold: Vertex merge threshold
        mincomponentRatio: Minimum component size ratio
        targetfacenum: Target face count for decimation
        maxholesize: Maximum hole size to fill
        stepsmoothnum: Number of smoothing iterations
        verbose: Whether to print progress
        
    Returns:
        Processed vertices, faces, and normals
    """
    try:
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int64)
        
        # Step 1: Merge close vertices
        vertices, faces = merge_close_vertices_o3d(
            vertices, faces, threshold=threshold, verbose=verbose
        )
        
        # Step 2: Remove isolated pieces
        vertices, faces = remove_isolated_pieces_o3d(
            vertices, faces, min_component_ratio=mincomponentRatio, verbose=verbose
        )
        
        # Step 3: Repair mesh
        vertices, faces = repair_mesh_trimesh(vertices, faces, verbose=verbose)
        
        # Step 4: Fill holes
        vertices, faces = fill_holes_trimesh(
            vertices, faces, max_hole_size=maxholesize, verbose=verbose
        )
        
        # Step 5: Taubin smoothing (first pass)
        vertices, faces = smooth_mesh_taubin(
            vertices, faces, iterations=stepsmoothnum, verbose=verbose
        )
        
        # Step 6: Decimate if needed
        if len(faces) > targetfacenum:
            vertices, faces = decimate_mesh_o3d(
                vertices, faces, target_faces=targetfacenum, verbose=verbose
            )
        
        # Step 7: Taubin smoothing (second pass)
        vertices, faces = smooth_mesh_taubin(
            vertices, faces, iterations=stepsmoothnum, verbose=verbose
        )
        
        # Step 8: Final repair
        vertices, faces = repair_mesh_trimesh(vertices, faces, verbose=verbose)
        
        # Step 9: Compute normals
        normals = compute_vertex_normals(vertices, faces)
        return vertices, faces, normals
        
    except Exception as e:
        logger.error(f"Mesh preprocessing failed: {e}")
        # Return original mesh with computed normals
        normals = compute_vertex_normals(vertices, faces)
        return vertices, faces, normals
    finally:
        # Cleanup
        gc.collect()


### UV Un-Warp ###
def uv_parameterize_xatlas(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """
    Compute UV atlas using xatlas directly (more stable than Open3D wrapper).
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        
    Returns:
        Texture UV coordinates (#F, 3, 2)
    """
    import xatlas
    
    try:
        vertices = np.ascontiguousarray(vertices.astype(np.float32))
        faces = np.ascontiguousarray(faces.astype(np.uint32))
        vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
        
        # Build per-face UV coordinates (#F, 3, 2)
        # indices contains the new face indices into the uvs array
        num_faces = len(indices)
        texture_uvs = np.zeros((num_faces, 3, 2), dtype=np.float32)
        
        for i, face in enumerate(indices):
            for j, idx in enumerate(face):
                texture_uvs[i, j] = uvs[idx]
        
        return texture_uvs
        
    except Exception as e:
        logger.error(f"xatlas UV computation failed: {e}")
        raise


def uv_parameterize_uvatlas(
    vertices: np.ndarray,
    faces: np.ndarray,
    size: int = 1024,
    gutter: float = 2.5,
    max_stretch: float = 0.1666666716337204,
    parallel_partitions: int = 1,  # Use single partition for stability
    nthreads: int = 1,  # Single thread for stability
    use_cuda: bool = False,  # Disabled - causes segfaults
) -> np.ndarray:
    """
    Compute UV atlas using Open3D (fallback, may be unstable).
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        size: UV atlas size
        gutter: Gutter size between UV islands
        max_stretch: Maximum texture stretch
        parallel_partitions: Number of parallel partitions
        nthreads: Number of threads
        use_cuda: Whether to use CUDA (disabled by default due to segfaults)
        
    Returns:
        Texture UV coordinates (#F, 3, 2)
    """
    try:
        device = o3d.core.Device("CPU:0")
            
        dtype_f = o3d.core.float32
        dtype_i = o3d.core.int64

        mesh = o3d.t.geometry.TriangleMesh(device)
        mesh.vertex.positions = o3d.core.Tensor(
            vertices.astype(np.float32), dtype_f, device
        )
        mesh.triangle.indices = o3d.core.Tensor(
            faces.astype(np.int64), dtype_i, device
        )

        mesh.compute_uvatlas(
            size=size,
            gutter=gutter,
            max_stretch=max_stretch,
            parallel_partitions=parallel_partitions,
            nthreads=nthreads,
        )

        texture_uvs = mesh.triangle.texture_uvs
        if texture_uvs.device.get_type() != o3d.core.Device.DeviceType.CPU:
            texture_uvs = texture_uvs.cpu()
        
        return texture_uvs.numpy()  # (#F, 3, 2)
        
    except Exception as e:
        logger.error(f"Open3D UV atlas computation failed: {e}")
        raise


### Pack All ###
def process_raw(
    mesh_path: str,
    save_path: str,
    preprocess: bool = True,
    device: str = "cpu"
) -> None:
    """
    Complete mesh processing pipeline: load, process, UV unwrap, and save.
    
    Args:
        mesh_path: Path to input mesh file
        save_path: Path to save processed mesh
        preprocess: Whether to run preprocessing steps
        device: Device for tensor operations
    """
    logger.info(f"Processing mesh: {mesh_path}")
    
    try:
        scene = trimesh.load(mesh_path, force="mesh", process=False)
        
        if isinstance(scene, trimesh.Trimesh):
            mesh = scene
        elif isinstance(scene, trimesh.scene.Scene):
            mesh = trimesh.Trimesh()
            for obj in scene.geometry.values():
                mesh = trimesh.util.concatenate([mesh, obj])
        else:
            raise ValueError(f"Unknown mesh type at {mesh_path}: {type(scene)}")

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)

        mesh_post_process_options = {
            "mincomponentRatio": 0.02,
            "targetfacenum": 80_000,  # Reduced from 150k for faster UV computation
            "maxholesize": 100,
            "stepsmoothnum": 2,
            "verbose": True,
        }

        # Process mesh
        if preprocess:
            v_pos, t_pos_idx, normals = process_mesh(
                vertices=vertices,
                faces=faces,
                **mesh_post_process_options,
            )
        else:
            v_pos = vertices
            t_pos_idx = faces
            normals = compute_vertex_normals(vertices, faces)

        # UV parameterization using xatlas
        v_tex_np = (
            uv_parameterize_xatlas(v_pos, t_pos_idx).reshape(-1, 2).astype(np.float32)
        )

        # Convert to tensors
        v_pos = torch.from_numpy(v_pos).to(device=device, dtype=torch.float32)
        t_pos_idx = torch.from_numpy(t_pos_idx).to(device=device, dtype=torch.long)
        v_tex = torch.from_numpy(v_tex_np).to(device=device, dtype=torch.float32)
        normals = torch.from_numpy(normals).to(device=device, dtype=torch.float32)
        
        assert v_tex.shape[0] == t_pos_idx.shape[0] * 3, \
            f"UV count mismatch: {v_tex.shape[0]} vs {t_pos_idx.shape[0] * 3}"
        
        t_tex_idx = torch.arange(
            t_pos_idx.shape[0] * 3,
            device=device,
            dtype=torch.long,
        ).reshape(-1, 3)
        
        # Super efficient de-duplication
        v_tex_u_uint32 = v_tex_np[..., 0].view(np.uint32)
        v_tex_v_uint32 = v_tex_np[..., 1].view(np.uint32)
        v_hashed = (v_tex_u_uint32.astype(np.uint64) << 32) | v_tex_v_uint32
        v_hashed = torch.from_numpy(v_hashed.view(np.int64)).to(v_pos.device)

        t_pos_idx_f3 = torch.arange(
            t_pos_idx.shape[0] * 3, device=t_pos_idx.device, dtype=torch.long
        ).reshape(-1, 3)
        v_pos_f3 = v_pos[t_pos_idx].reshape(-1, 3)
        normals_f3 = normals[t_pos_idx].reshape(-1, 3)

        v_hashed_dedup, inverse_indices = torch.unique(v_hashed, return_inverse=True)
        dedup_size, full_size = v_hashed_dedup.shape[0], inverse_indices.shape[0]
        indices = torch.scatter_reduce(
            torch.full(
                [dedup_size],
                fill_value=full_size,
                device=inverse_indices.device,
                dtype=torch.long,
            ),
            index=inverse_indices,
            src=torch.arange(full_size, device=inverse_indices.device, dtype=torch.int64),
            dim=0,
            reduce="amin",
        )
        v_tex = v_tex[indices]
        t_tex_idx = inverse_indices.reshape(-1, 3)

        v_pos = v_pos_f3[indices]
        normals = normals_f3[indices]
        normals = normals.to(dtype=torch.float32, device=device)

        # Flip UV (either flip uv or flip texture - here we flip uv)
        uv_to_save = v_tex.clone()
        uv_to_save[:, 1] = 1.0 - uv_to_save[:, 1]

        # Create and save mesh
        visual = trimesh.visual.TextureVisuals(uv=uv_to_save.cpu().numpy())
        tmesh = trimesh.Trimesh(
            vertices=v_pos.cpu().numpy(),
            faces=t_tex_idx.cpu().numpy(),
            vertex_normals=normals.cpu().numpy(),
            visual=visual,
            process=False,
        )
        tmesh.export(save_path)
        logger.info(f"Mesh saved: {save_path}")
        
    except Exception as e:
        logger.error(f"process_raw failed: {e}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Legacy pymeshlab wrapper functions (for backwards compatibility)
# These are no longer used but kept for API compatibility

def read_mesh_from_path(mesh_path: str):
    """Legacy function - use trimesh.load instead."""
    return trimesh.load(mesh_path, force="mesh", process=False)


def mesh_to_meshlab(vertices: np.ndarray, faces: np.ndarray):
    """Legacy function - returns trimesh instead."""
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def meshlab_to_mesh(mesh):
    """Legacy function - extracts data from trimesh."""
    if isinstance(mesh, trimesh.Trimesh):
        return mesh.vertices, mesh.faces, mesh.vertex_normals
    raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh)}")
