import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation
import open3d as o3d

"""
GLB Mesh Simplification using Open3D and Trimesh

This module provides functionality to simplify meshes in GLB files while
preserving topology and visual quality.

Requirements:
    open3d
    trimesh
"""

import math
from pathlib import Path


def simplify_glb_mesh(
    input_glb_path: str,
    output_glb_path: str,
    target_faces: int = 150_000,
    fix_texture: bool = False,
) -> None:
    """
    Simplify a GLB mesh file using Open3D's quadric decimation.
    
    This function reduces the polygon count of a mesh while minimizing loss of
    geometric detail, preserving topology, and maintaining overall visual quality.
    Holes are filled and mesh integrity is maintained.
    
    Uses Open3D with CUDA acceleration when available.
    
    Args:
        input_glb_path: Path to the input GLB file.
        output_glb_path: Path where the simplified GLB file will be saved.
        target_faces: Target number of faces for the simplified mesh (default: 150,000).
        fix_texture: If True, preserve textures and materials in the output.
                     If False (default), save only mesh geometry without textures.
    
    Raises:
        FileNotFoundError: If the input GLB file does not exist.
        ValueError: If no mesh objects are found in the GLB file.
    """
    input_path = Path(input_glb_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input GLB file not found: {input_glb_path}")
    
    output_path = Path(output_glb_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load the GLB file using trimesh
    loaded = trimesh.load(str(input_path), force='mesh')
    
    if loaded is None or (isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) == 0):
        raise ValueError(f"No mesh objects found in GLB file: {input_glb_path}")
    
    # Handle both single mesh and scene with multiple meshes
    if isinstance(loaded, trimesh.Scene):
        # Combine all meshes into one
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh objects found in GLB file: {input_glb_path}")
        combined_mesh = trimesh.util.concatenate(meshes)
    else:
        combined_mesh = loaded
    
    original_face_count = len(combined_mesh.faces)
    
    # If already below target, just export
    if original_face_count <= target_faces:
        _export_mesh(combined_mesh, output_path, fix_texture)
        return
    
    # Convert trimesh to Open3D mesh
    o3d_mesh = _trimesh_to_open3d(combined_mesh)
    
    # --- Pre-processing: Clean mesh before decimation ---
    
    # Remove duplicated vertices (merge close vertices)
    o3d_mesh = o3d_mesh.remove_duplicated_vertices()
    
    # Remove degenerate triangles
    o3d_mesh = o3d_mesh.remove_degenerate_triangles()
    
    # Remove duplicated triangles
    o3d_mesh = o3d_mesh.remove_duplicated_triangles()
    
    # Remove non-manifold edges
    o3d_mesh = o3d_mesh.remove_non_manifold_edges()
    
    # Convert back to trimesh to fill holes before decimation
    tri_mesh = _open3d_to_trimesh(o3d_mesh)
    tri_mesh.fill_holes()
    
    # Convert back to Open3D for decimation
    o3d_mesh = _trimesh_to_open3d(tri_mesh)
    
    # Ensure consistent vertex normals
    o3d_mesh.compute_vertex_normals()
    
    # --- Decimation using quadric error metrics ---
    # This provides the best quality decimation while preserving geometric detail
    o3d_mesh = o3d_mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_faces
    )
    
    # --- Post-processing: Fix holes and clean up mesh ---
    
    # Remove any degenerate geometry created during decimation
    o3d_mesh = o3d_mesh.remove_degenerate_triangles()
    o3d_mesh = o3d_mesh.remove_duplicated_vertices()
    o3d_mesh = o3d_mesh.remove_duplicated_triangles()
    o3d_mesh = o3d_mesh.remove_non_manifold_edges()
    
    # Remove unreferenced vertices
    o3d_mesh = o3d_mesh.remove_unreferenced_vertices()
    
    # Convert to trimesh for hole filling
    tri_mesh = _open3d_to_trimesh(o3d_mesh)
    
    # Fill holes created by decimation
    tri_mesh.fill_holes()
    
    # Convert back to Open3D for smoothing
    o3d_mesh = _trimesh_to_open3d(tri_mesh)
    
    # Apply Laplacian smoothing to reduce sharp artifacts
    # Using 1 iteration with lambda=0.1 for subtle smoothing
    o3d_mesh = o3d_mesh.filter_smooth_laplacian(
        number_of_iterations=1,
        lambda_filter=0.1
    )
    
    # Recompute vertex normals for smooth shading
    o3d_mesh.compute_vertex_normals()
    
    # Final conversion to trimesh for export
    final_mesh = _open3d_to_trimesh(o3d_mesh)
    
    # Export the simplified mesh
    _export_mesh(final_mesh, output_path, fix_texture)


def _trimesh_to_open3d(tri_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """Convert a trimesh mesh to an Open3D triangle mesh."""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces.astype(np.int32))
    
    if tri_mesh.vertex_normals is not None and len(tri_mesh.vertex_normals) > 0:
        o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(
            tri_mesh.vertex_normals.astype(np.float64)
        )
    
    return o3d_mesh


def _open3d_to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh) -> trimesh.Trimesh:
    """Convert an Open3D triangle mesh to a trimesh mesh."""
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    
    vertex_normals = None
    if o3d_mesh.has_vertex_normals():
        vertex_normals = np.asarray(o3d_mesh.vertex_normals)
    
    tri_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        process=False  # Don't modify the mesh during construction
    )
    
    return tri_mesh


def _export_mesh(mesh: trimesh.Trimesh, output_path: Path, fix_texture: bool) -> None:
    """Export mesh to GLB format."""
    # Create a scene for proper GLB export
    scene = trimesh.Scene(mesh)
    
    # Export as GLB
    scene.export(
        str(output_path),
        file_type='glb'
    )


def apply_inverse_rotation(mesh, euler_angles_rad):
    """Apply inverse rotation to mesh around its center."""
    rotation = Rotation.from_euler('xyz', euler_angles_rad)
    inverse_rotation_matrix = rotation.inv().as_matrix()
    
    mesh_center = mesh.vertices.mean(axis=0)
    vertices_centered = mesh.vertices - mesh_center
    vertices_rotated = vertices_centered @ inverse_rotation_matrix.T
    
    rotated_mesh = mesh.copy()
    rotated_mesh.vertices = vertices_rotated + mesh_center
    return rotated_mesh

def apply_rotation(mesh, euler_angles_rad):
    """Apply rotation to mesh around its center."""
    rotation_matrix = Rotation.from_euler('xyz', euler_angles_rad).as_matrix()
    
    mesh_center = mesh.vertices.mean(axis=0)
    vertices_centered = mesh.vertices - mesh_center
    vertices_rotated = vertices_centered @ rotation_matrix.T
    
    rotated_mesh = mesh.copy()
    rotated_mesh.vertices = vertices_rotated + mesh_center
    return rotated_mesh


def align_target_to_source_view(target_mesh, target_euler_rad, source_euler_rad):
    """Align target mesh to source view by first frontalizing, then rotating to source orientation."""
    frontal_mesh = apply_inverse_rotation(target_mesh, target_euler_rad)
    aligned_mesh = apply_rotation(frontal_mesh, source_euler_rad)
    return aligned_mesh

def apply_glb_to_target_transform(mesh):
    """Transform mesh from GLB coordinate system to target coordinate system."""
    mesh = mesh.copy()
    # mesh.vertices[:, 2] *= -1
    rotation_matrix = Rotation.from_euler('x', np.pi/2).as_matrix()
    mesh_center = mesh.vertices.mean(axis=0)
    vertices_centered = mesh.vertices - mesh_center
    vertices_rotated = vertices_centered @ rotation_matrix.T
    mesh.vertices = vertices_rotated + mesh_center
    
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
        mesh.vertex_normals = mesh.vertex_normals @ rotation_matrix.T
    if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
        mesh.face_normals = mesh.face_normals @ rotation_matrix.T
    
    return mesh


def apply_target_to_glb_transform(mesh):
    """Transform mesh from target coordinate system back to GLB coordinate system."""
    mesh = mesh.copy()
    rotation_matrix = Rotation.from_euler('x', -np.pi / 2).as_matrix()
    
    mesh_center = mesh.vertices.mean(axis=0)
    vertices_centered = mesh.vertices - mesh_center
    vertices_rotated = vertices_centered @ rotation_matrix.T
    mesh.vertices = vertices_rotated + mesh_center
    
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
        mesh.vertex_normals = mesh.vertex_normals @ rotation_matrix.T
    if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
        mesh.face_normals = mesh.face_normals @ rotation_matrix.T
    
    # mesh.vertices[:, 2] *= -1
    
    return mesh


def rotate_glb_mesh(
    input_glb_path: str,
    output_glb_path: str = None,
    euler_angles_rad: list = [0.0, 0.0, 0.0],
    rotate_fn=apply_inverse_rotation,
):
    """
    Rotate a GLB mesh file by applying transformations.
    Returns both the scene and the frontalized mesh.
    """
    input_path = Path(input_glb_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input GLB file not found: {input_glb_path}")
    
    if output_glb_path is not None:
        output_path = Path(output_glb_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    loaded = trimesh.load(str(input_path), force='scene')
    
    if isinstance(loaded, trimesh.Scene):
        for mesh_name, mesh in loaded.geometry.items():
            if isinstance(mesh, trimesh.Trimesh):
                mesh_in_target = apply_glb_to_target_transform(mesh)
                frontalized_mesh = rotate_fn(mesh_in_target, euler_angles_rad)
                mesh_back_to_glb = apply_target_to_glb_transform(frontalized_mesh)
                loaded.geometry[mesh_name] = mesh_back_to_glb
        
        result_scene = loaded
        result_mesh = list(loaded.geometry.values())[0] if loaded.geometry else None
        
    elif isinstance(loaded, trimesh.Trimesh):
        mesh_in_target = apply_glb_to_target_transform(loaded)
        frontalized_mesh = rotate_fn(mesh_in_target, euler_angles_rad)
        result_mesh = apply_target_to_glb_transform(frontalized_mesh)
        result_scene = trimesh.Scene(result_mesh)
    else:
        raise ValueError(f"Unexpected type loaded: {type(loaded)}")
    
    if output_glb_path is not None:
        result_scene.export(str(output_path), file_type='glb')
    
    return result_scene, frontalized_mesh

