import bpy
import bmesh
from mathutils import Vector
import struct
import numpy as np
import math
import os
import array
from tqdm.auto import tqdm
import random
import time

def read_difflocks_strands(npz_path, return_stats=False, strands_subsample=1.0, vertex_subsample=1.0, min_strand_length=2):
    """
    Read Difflocks hair strand data from NPZ file and convert to the same format as read_ply_polylines.
    
    Args:
        npz_path (str): Path to the Difflocks NPZ file containing strand positions
        return_stats (bool): Whether to return bounding box statistics
        strands_subsample (float): Fraction of strands to keep (1.0 = all, 0.5 = half)
        vertex_subsample (float): Fraction of vertices per strand to keep (1.0 = all, 0.5 = half)
        min_strand_length (int): Minimum number of points required for a valid strand
        
    Returns:
        list: List of strands, where each strand is a list of mathutils.Vector objects
        dict (optional): Statistics with min/max coordinates if return_stats=True
    """
    try:
        # Load NPZ file
        npz_file = np.load(npz_path)
        if 'positions' not in npz_file:
            raise ValueError(f"NPZ file {npz_path} does not contain 'positions' key")
            
        positions = npz_file['positions']  # Shape: (N, 256, 3)
        print(f"Loaded Difflocks strands: {positions.shape[0]} strands, {positions.shape[1]} points per strand")
        
        # Apply strand subsampling
        if strands_subsample < 1.0:
            num_strands_to_keep = int(positions.shape[0] * strands_subsample)
            strands_to_keep = np.random.choice(positions.shape[0], num_strands_to_keep, replace=False)
            positions = positions[strands_to_keep, :, :].copy()
            print(f"After strand subsampling: {positions.shape[0]} strands")
            
        # Apply vertex subsampling
        if vertex_subsample < 1.0:
            nr_verts_to_skip = int(np.floor(1.0 / vertex_subsample))
            positions = positions[:, ::nr_verts_to_skip, :].copy()
            print(f"After vertex subsampling: {positions.shape[1]} points per strand")
        
        valid_strands = []
        all_points = []
        
        # Process each strand
        for strand_idx in range(positions.shape[0]):
            strand_points = positions[strand_idx]  # Shape: (256, 3) or subsampled

            # Filter out invalid points (all zeros or NaN)
            valid_indices = ~(np.allclose(strand_points, 0, atol=1e-6) | np.isnan(strand_points).any(axis=1))
            if valid_indices.sum() < min_strand_length:
                continue

            valid_points = strand_points[valid_indices]

            # Apply coordinate transformation to match the expected format
            # Difflocks uses standard (x, y, z), but the pipeline expects (x, -z, y)
            # This transformation swaps Y and Z coordinates, then negates the new Y
            transformed_points = []
            for point in valid_points:
                x, y, z = point[0], point[1], point[2]
                # Transform: (x, y, z) -> (x, -z, y)
                transformed_point = Vector((x, -z, y))
                transformed_points.append(transformed_point)
                all_points.append(transformed_point)
                
            if len(transformed_points) >= min_strand_length:
                valid_strands.append(transformed_points)
        
        print(f"Successfully processed {len(valid_strands)} valid strands from Difflocks data")
        
        # Calculate statistics if requested
        if return_stats and all_points:
            min_coords = {
                "x": min(p.x for p in all_points),
                "y": min(p.y for p in all_points), 
                "z": min(p.z for p in all_points),
            }
            max_coords = {
                "x": max(p.x for p in all_points),
                "y": max(p.y for p in all_points),
                "z": max(p.z for p in all_points),
            }
            stats = {"min": min_coords, "max": max_coords}
            print(f"Bounding box: min=({min_coords['x']:.3f}, {min_coords['y']:.3f}, {min_coords['z']:.3f}), "
                  f"max=({max_coords['x']:.3f}, {max_coords['y']:.3f}, {max_coords['z']:.3f})")
            del all_points  # Free memory
            return valid_strands, stats
            
        del all_points  # Free memory
        return valid_strands
        
    except Exception as e:
        print(f"Error reading Difflocks NPZ file {npz_path}: {e}")
        if return_stats:
            return [], {}
        return []

def read_strand_data(data_dir, strand_index, return_stats=False):
    """Read strand data from file and convert it to Blender's format."""
    file_path = f"{data_dir}/hairstyles/strands{strand_index}.data"
    valid_strands = []
    all_points = []

    with open(file_path, "rb") as fin:
        num_strands = struct.unpack("<i", fin.read(4))[0]

        for _ in range(num_strands):
            num_verts = struct.unpack("<i", fin.read(4))[0]
            if num_verts >= 2:
                raw_data = fin.read(4 * 3 * num_verts)
                verts = np.frombuffer(raw_data, dtype=np.float32).reshape(num_verts, 3)
                transformed = np.column_stack((verts[:, 0], -verts[:, 2], verts[:, 1]))
                strand = [Vector(row) for row in transformed]
                valid_strands.append(strand)
                all_points.extend(strand)
            else:
                fin.seek(4 * 3 * num_verts, 1)

    if return_stats and all_points:
        min_coords = {
            "x": min(p.x for p in all_points),
            "y": min(p.y for p in all_points),
            "z": min(p.z for p in all_points),
        }
        max_coords = {
            "x": max(p.x for p in all_points),
            "y": max(p.y for p in all_points),
            "z": max(p.z for p in all_points),
        }
        stats = {"min": min_coords, "max": max_coords}
        del all_points  # Free memory
        return valid_strands, stats

    return valid_strands
    

def read_strand_data_from_path(file_path, return_stats=False, return_blender_vec=True, transform_fn=lambda v: v, sample_ratio=1.0):
    """Read strand data from file and convert it to Blender's format."""
    valid_strands = []
    all_points = []

    with open(file_path, "rb") as fin:
        num_strands = struct.unpack("<i", fin.read(4))[0]

        for _ in tqdm(range(num_strands), desc="Processing strands"):
            num_verts = struct.unpack("<i", fin.read(4))[0]
            if num_verts >= 2:
                raw_data = fin.read(4 * 3 * num_verts)
                if random.random() > sample_ratio:
                    continue
                verts = np.frombuffer(raw_data, dtype=np.float32).reshape(num_verts, 3)                
                verts_transformed = transform_fn(verts)
                transformed = np.column_stack((verts_transformed[:, 0], -verts_transformed[:, 2], verts_transformed[:, 1]))
                if return_blender_vec:
                    strand = [Vector(row) for row in transformed]
                else:
                    strand = transformed
                valid_strands.append(strand)
                all_points.extend(strand)
            else:
                fin.seek(4 * 3 * num_verts, 1)

    if return_stats and all_points:
        if return_blender_vec:
            min_coords = {
                "x": min(p.x for p in all_points),
                "y": min(p.y for p in all_points),
                "z": min(p.z for p in all_points),
            }
            max_coords = {
                "x": max(p.x for p in all_points),
                "y": max(p.y for p in all_points),
                "z": max(p.z for p in all_points),
            }
        else:
            min_coords = {
                "x": min(p[0] for p in all_points),
                "y": min(p[1] for p in all_points),
                "z": min(p[2] for p in all_points),
            }
            max_coords = {
                "x": max(p[0] for p in all_points),
                "y": max(p[1] for p in all_points),
                "z": max(p[2] for p in all_points),
            }
        stats = {"min": min_coords, "max": max_coords}
        del all_points  # Free memory
        return valid_strands, stats

    return valid_strands

def compute_vertex_color(point_prev, point_curr, point_next, camera):
    """Compute vertex color based on tangent in camera space"""
    if point_prev is None:
        tangent = (point_next - point_curr).normalized()
    elif point_next is None:
        tangent = (point_curr - point_prev).normalized()
    else:
        tangent1 = (point_curr - point_prev).normalized()
        tangent2 = (point_next - point_curr).normalized()
        tangent = (tangent1 + tangent2).normalized()

    camera_matrix = camera.matrix_world.inverted()
    camera_tangent = camera_matrix.to_3x3() @ tangent
    camera_tangent.z = 0
    camera_tangent.normalize()
    length = max(1e-6, camera_tangent.length)
    p = -1 * camera_tangent.y / length
    r = -1 * camera_tangent.x / length

    p_mapped = p * 0.5 + 0.5
    r_mapped = r * 0.5 + 0.5

    p_mapped = max(0, min(1, p_mapped))
    r_mapped = max(0, min(1, r_mapped))
    return (1.0, p_mapped, r_mapped, 1.0)

def create_circle_verts(bm, center, frame, radius, segments):
    """Create vertices for a circular cross-section using precomputed frame"""
    right, up = frame
    verts = []
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        offset = math.cos(angle) * right + math.sin(angle) * up
        verts.append(bm.verts.new(center + offset * radius))
    return verts

def compute_rotation_minimizing_frame(prev_frame, prev_tangent, curr_tangent):
    """Compute stable frame using rotation minimizing projection"""
    right_prev, up_prev = prev_frame
    
    # Project previous frame onto new tangent plane
    new_right = right_prev - curr_tangent * curr_tangent.dot(right_prev)
    
    # Handle degenerate cases
    if new_right.length_squared < 1e-5:
        fallback_up = Vector((0, 0, 1))
        if abs(curr_tangent.dot(fallback_up)) > 0.99:
            fallback_up = Vector((0, 1, 0))
        new_right = curr_tangent.cross(fallback_up).normalized()
    
    new_right.normalize()
    new_up = curr_tangent.cross(new_right).normalized()
    return (new_right, new_up)

def create_hair_mesh_from_strands(strands, camera, radius=0.001, segments=12):
    """Create high-quality hair mesh with smooth shading and minimized twisting"""
    mesh = bpy.data.meshes.new("HairMesh")
    bm = bmesh.new()
    color_layer = bm.loops.layers.color.new("Col")

    for strand_points in strands:
        n_points = len(strand_points)
        if n_points < 2:
            continue
            
        # Precompute tangents with central differences
        tangents = []
        for i in range(n_points):
            if i == 0:
                tangents.append((strand_points[1] - strand_points[0]).normalized())
            elif i == n_points - 1:
                tangents.append((strand_points[-1] - strand_points[-2]).normalized())
            else:
                v = (strand_points[i+1] - strand_points[i-1])
                tangents.append(v.normalized() if v.length_squared > 0 else tangents[-1])

        # Precompute vertex colors
        strand_colors = []
        for i in range(n_points):
            prev = strand_points[i-1] if i > 0 else None
            next = strand_points[i+1] if i < n_points - 1 else None
            strand_colors.append(compute_vertex_color(prev, strand_points[i], next, camera))

        # Compute initial frame
        init_up = Vector((0, 0, 1))
        if abs(tangents[0].dot(init_up)) > 0.99:
            init_up = Vector((0, 1, 0))
        init_right = tangents[0].cross(init_up).normalized()
        init_up = init_right.cross(tangents[0]).normalized()
        frames = [(init_right, init_up)]

        # Compute rotation-minimizing frames
        for i in range(1, n_points):
            frames.append(compute_rotation_minimizing_frame(
                frames[i-1], tangents[i-1], tangents[i]
            ))

        # Create vertex rings
        rings = []
        for i in range(n_points):
            rings.append(create_circle_verts(bm, strand_points[i], frames[i], radius, segments))

        # Create faces between rings
        for i in range(n_points - 1):
            ring1 = rings[i]
            ring2 = rings[i+1]
            for j in range(segments):
                j_next = (j + 1) % segments
                v1 = ring1[j]
                v2 = ring2[j]
                v3 = ring2[j_next]
                v4 = ring1[j_next]
                
                face = bm.faces.new((v1, v2, v3, v4))
                face.smooth = True
                
                # Assign vertex colors
                loops = list(face.loops)
                loops[0][color_layer] = strand_colors[i]
                loops[1][color_layer] = strand_colors[i+1]
                loops[2][color_layer] = strand_colors[i+1]
                loops[3][color_layer] = strand_colors[i]

    # Finalize mesh
    bm.to_mesh(mesh)
    bm.free()
    mesh.update(calc_edges=True)
    return mesh


def create_hair_mesh_from_strands_no_camera(strands, radius=0.001, segments=12, color_rgb=(1.0, 1.0, 1.0)):
    """Create high-quality hair mesh with smooth shading and minimized twisting"""
    mesh = bpy.data.meshes.new("HairMesh")
    bm = bmesh.new()
    color_layer = bm.loops.layers.color.new("Col")
    color_rgba = (color_rgb[0], color_rgb[1], color_rgb[2], 1.0)

    for strand_points in strands:
        n_points = len(strand_points)
        if n_points < 2:
            continue
            
        # Precompute tangents with central differences
        tangents = []
        for i in range(n_points):
            if i == 0:
                tangents.append((strand_points[1] - strand_points[0]).normalized())
            elif i == n_points - 1:
                tangents.append((strand_points[-1] - strand_points[-2]).normalized())
            else:
                v = (strand_points[i+1] - strand_points[i-1])
                tangents.append(v.normalized() if v.length_squared > 0 else tangents[-1])

        # Precompute vertex colors
        strand_colors = []
        for i in range(n_points):
            strand_colors.append(color_rgba)

        # Compute initial frame
        init_up = Vector((0, 0, 1))
        if abs(tangents[0].dot(init_up)) > 0.99:
            init_up = Vector((0, 1, 0))
        init_right = tangents[0].cross(init_up).normalized()
        init_up = init_right.cross(tangents[0]).normalized()
        frames = [(init_right, init_up)]

        # Compute rotation-minimizing frames
        for i in range(1, n_points):
            frames.append(compute_rotation_minimizing_frame(
                frames[i-1], tangents[i-1], tangents[i]
            ))

        # Create vertex rings
        rings = []
        for i in range(n_points):
            rings.append(create_circle_verts(bm, strand_points[i], frames[i], radius, segments))

        # Create faces between rings
        for i in range(n_points - 1):
            ring1 = rings[i]
            ring2 = rings[i+1]
            for j in range(segments):
                j_next = (j + 1) % segments
                v1 = ring1[j]
                v2 = ring2[j]
                v3 = ring2[j_next]
                v4 = ring1[j_next]
                
                face = bm.faces.new((v1, v2, v3, v4))
                face.smooth = True
                
                # Assign vertex colors
                loops = list(face.loops)
                loops[0][color_layer] = strand_colors[i]
                loops[1][color_layer] = strand_colors[i+1]
                loops[2][color_layer] = strand_colors[i+1]
                loops[3][color_layer] = strand_colors[i]

    # Finalize mesh
    bm.to_mesh(mesh)
    bm.free()
    mesh.update(calc_edges=True)
    return mesh


def import_difflocks_model():
    """Import the appropriate models based on the specified type"""
    objects = []
    # Import specified body model
    body_path = f"assets/bust/difflocks/body_mesh.obj"
    if os.path.exists(body_path):
        bpy.ops.wm.obj_import(filepath=body_path)
        if bpy.context.selected_objects:
            body_obj = bpy.context.selected_objects[0]
            objects.append(("body", body_obj))
        else:
            print(f"Warning: No objects were imported from {body_path}")
    else:
        print(f"Warning: Body model not found at {body_path}")
    return objects

def import_trimesh_to_blender(trimesh_obj, object_name="BodyMesh", temp_path="assets/temp"):
    objects = []
    # Import specified body model
    random_prefix = f"{int(time.time() * 1000000):06d}"
    body_path = f"{temp_path}/body_mesh_{random_prefix}.obj"
    trimesh_obj.export(body_path)

    if os.path.exists(body_path):
        bpy.ops.wm.obj_import(filepath=body_path)
        if bpy.context.selected_objects:
            body_obj = bpy.context.selected_objects[0]
            body_obj.name = object_name

            # Apply shade smooth to the imported mesh
            bpy.context.view_layer.objects.active = body_obj
            bpy.ops.object.shade_smooth()

            objects.append(("body", body_obj))
        else:
            print(f"Warning: No objects were imported from {body_path}")
    else:
        print(f"Warning: Body model not found at {body_path}")
    return objects

def import_models(model_path):
    """Import the mesh to be rendered"""
    objects = []
    # Import specified body model
    bpy.ops.wm.obj_import(filepath=model_path)
    body_obj = bpy.context.selected_objects[0]

    objects.append(("body", body_obj))
    return objects

def read_ply_polylines(file_path, return_stats=False, blender_format=True):
    """Read polyline data from binary PLY file and format as strands"""
    vertices = []
    polylines = []
    valid_strands = []
    stats = {}
    with open(file_path, "rb") as f:
        # Read header
        header = []
        vertex_count = 0
        edge_count = 0

        while True:
            line = f.readline().decode("ascii").strip()
            if line == "end_header":
                break

            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("element edge"):
                edge_count = int(line.split()[-1])

            header.append(line)

        print(f"Reading {vertex_count} vertices and {edge_count} edges")

        # Read vertices - using double precision (8 bytes per coordinate)
        for i in range(vertex_count):
            data = f.read(24)  # 3 * 8 bytes for double precision coordinates
            x, y, z = struct.unpack("<ddd", data)
            # Store vertex in the same coordinate system as the .data file format
            if blender_format:
                # Convert to Blender's coordinate system (x, -z, y)
                vertices.append(Vector((x, -z, y)))
            else:
                vertices.append(Vector((x, y, z)))

        # Read edges and construct polylines
        strand_points = []
        prev_vertex = None

        # Process edges to build connected strands
        for i in range(edge_count):
            data = f.read(8)  # 2 * 4 bytes for int indices
            v1, v2 = struct.unpack("<ii", data)

            # Start new strand if this edge doesn't connect to previous
            if prev_vertex is not None and v1 != prev_vertex:
                if len(strand_points) >= 2:
                    valid_strands.append(strand_points)
                strand_points = []

            # Add points to current strand
            if not strand_points:
                strand_points.append(vertices[v1])
            strand_points.append(vertices[v2])
            prev_vertex = v2

        # Add final strand
        if strand_points and len(strand_points) >= 2:
            valid_strands.append(strand_points)

        print(f"Successfully constructed {len(valid_strands)} strands")

        # Calculate bounding box for debugging
        if valid_strands:
            all_points = [point for strand in valid_strands for point in strand]
            if blender_format:
                min_coords = Vector(
                    (
                        min(p.x for p in all_points),
                        min(p.y for p in all_points),
                        min(p.z for p in all_points),
                    )
                )
                max_coords = Vector(
                    (
                        max(p.x for p in all_points),
                        max(p.y for p in all_points),
                        max(p.z for p in all_points),
                    )
                )
            else:
                min_coords = np.min(all_points, axis=0)
                max_coords = np.max(all_points, axis=0)
            if blender_format:
                min_coords = {"x": min_coords.x, "y": min_coords.y, "z": min_coords.z}
                max_coords = {"x": max_coords.x, "y": max_coords.y, "z": max_coords.z}
            else:
                min_coords = {"x": min_coords[0], "y": min_coords[1], "z": min_coords[2]}
                max_coords = {"x": max_coords[0], "y": max_coords[1], "z": max_coords[2]}
            if return_stats:
                stats = {
                    "min": min_coords,
                    "max": max_coords,
                }
            print(f"Bounding box: min={min_coords}, max={max_coords}")

    if return_stats:
        return valid_strands, stats
    return valid_strands


def create_curve_material_with_camera_colors(curves_obj, base_color=(0.3, 0.2, 0.1)):
    """
    Create a material for curves that uses the stored camera-based color information.
    
    Args:
        curves_obj: Blender curves object with stored color data
        base_color: Base hair color (default: brown)
    
    Returns:
        bpy.types.Material: Material configured for camera-based coloring
    """
    # Create new material
    mat = bpy.data.materials.new(name="HairCurvesMaterial")
    mat.use_nodes = True
    mat.blend_method = 'HASHED'
    mat.shadow_method = 'HASHED'
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Create basic nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    hair_bsdf = nodes.new('ShaderNodeBsdfHairPrincipled')
    
    # Position nodes
    output.location = (300, 0)
    hair_bsdf.location = (0, 0)
    
    # Configure hair BSDF
    hair_bsdf.parametrization = 'MELANIN'
    hair_bsdf.inputs['Melanin'].default_value = 0.8
    hair_bsdf.inputs['Melanin Redness'].default_value = 0.5
    hair_bsdf.inputs['Roughness'].default_value = 0.3
    hair_bsdf.inputs['Radial Roughness'].default_value = 0.3
    hair_bsdf.inputs['Coat'].default_value = 0.0
    hair_bsdf.inputs['IOR'].default_value = 1.55
    hair_bsdf.inputs['Offset'].default_value = 2.0
    
    # If camera colors are available, we could add nodes to interpolate them
    # For now, use a base color with some variation
    if curves_obj.get("has_camera_colors", False):
        # Add variation based on curve parameter
        tex_coord = nodes.new('ShaderNodeTexCoord')
        color_ramp = nodes.new('ShaderNodeValToRGB')
        
        tex_coord.location = (-400, -200)
        color_ramp.location = (-200, -200)
        
        # Setup color ramp for root to tip variation
        color_ramp.color_ramp.elements[0].color = (*base_color, 1.0)
        color_ramp.color_ramp.elements[1].color = (base_color[0]*0.7, base_color[1]*0.7, base_color[2]*0.7, 1.0)
        
        # Connect parametric coordinate to color variation
        links.new(tex_coord.outputs['Generated'], color_ramp.inputs['Fac'])
        links.new(color_ramp.outputs['Color'], hair_bsdf.inputs['Color'])
    else:
        # Use uniform base color
        hair_bsdf.inputs['Color'].default_value = (*base_color, 1.0)
    
    # Connect BSDF to output
    links.new(hair_bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return mat


def assign_material_to_curves(curves_obj, material):
    """
    Assign a material to a curves object.
    
    Args:
        curves_obj: Blender curves object
        material: Material to assign
    """
    # Clear existing materials
    curves_obj.data.materials.clear()
    
    # Add the new material
    curves_obj.data.materials.append(material)
    
    # Set the object to use the material
    if curves_obj.material_slots:
        curves_obj.material_slots[0].material = material


def convert_curves_to_mesh(curves_obj, apply_modifiers=True):
    """
    Convert a curves object to a mesh for compatibility with existing workflows.
    
    Args:
        curves_obj: Blender curves object to convert
        apply_modifiers: Whether to apply modifiers during conversion
    
    Returns:
        bpy.types.Object: New mesh object
    """
    # Ensure we're in object mode
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Select and make the curves object active
    bpy.context.view_layer.objects.active = curves_obj
    curves_obj.select_set(True)
    
    # Convert to mesh
    bpy.ops.object.convert(target='MESH')
    
    # The conversion operation modifies the original object
    mesh_obj = curves_obj
    mesh_obj.name = mesh_obj.name.replace("Curves", "Mesh")
    
    return mesh_obj


def get_curves_bounding_box(curves_data):
    """
    Calculate bounding box for curves data.
    
    Args:
        curves_data: Blender curves data
    
    Returns:
        tuple: (min_coords, max_coords) as Vector objects
    """
    if not curves_data.splines:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    
    all_points = []
    for spline in curves_data.splines:
        for point in spline.points:
            all_points.append(Vector(point.co[:3]))  # Ignore homogeneous coordinate
    
    if not all_points:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    
    min_coords = Vector((
        min(p.x for p in all_points),
        min(p.y for p in all_points),
        min(p.z for p in all_points)
    ))
    
    max_coords = Vector((
        max(p.x for p in all_points),
        max(p.y for p in all_points),
        max(p.z for p in all_points)
    ))
    
    return min_coords, max_coords