# export_glb_with_textures.py
# Usage (after the -- are script args):
# blender --background --python export_glb_with_textures.py -- \
#   --mesh /path/model.obj \
#   --base_color /path/albedo.png \
#   --roughness /path/roughness.png \
#   --metallic /path/metallic.png \
#   --uv_map UVMap \
#   --out /path/textured.glb

import argparse
import math
import os
from pdb import main
import sys

import bpy



def apply_subdivision_surface(mesh_obj, levels=2, render_levels=2, subdivision_type='CATMULL_CLARK'):
    if mesh_obj.type != 'MESH':
        return

    # Clamp subdivision levels to valid range
    levels = max(0, int(levels))
    render_levels = max(0, int(render_levels))

    # Add subdivision surface modifier
    modifier = mesh_obj.modifiers.new(name="Subdivision", type='SUBSURF')
    modifier.levels = levels
    modifier.render_levels = render_levels
    modifier.subdivision_type = subdivision_type
    
    # Use the limit surface for smoother results
    modifier.show_only_control_edges = True
    
    # Enable "Use Custom Normals" - preserves custom normals during subdivision
    modifier.use_custom_normals = True
    
    print(
        f"Applied Subdivision Surface modifier on {mesh_obj.name} with {levels} viewport levels "
        f"and {render_levels} render levels (custom normals: {modifier.use_custom_normals})"
    )


def apply_triangulate_modifier(mesh_obj, quad_method='BEAUTY', ngon_method='BEAUTY'):
    if mesh_obj.type != 'MESH':
        return

    # Add triangulate modifier
    modifier = mesh_obj.modifiers.new(name="Triangulate", type='TRIANGULATE')
    modifier.quad_method = quad_method
    modifier.ngon_method = ngon_method
    modifier.keep_custom_normals = True  # Preserve custom normals from subdivision
    
    print(
        f"Applied Triangulate modifier on {mesh_obj.name} with quad_method={quad_method}, "
        f"ngon_method={ngon_method}"
    )


def apply_smooth_shading(mesh_obj, auto_smooth_angle=math.pi):
    if mesh_obj.type != 'MESH':
        return
    mesh_data = mesh_obj.data
    if mesh_data is None or not mesh_data.polygons:
        return

    for poly in mesh_data.polygons:
        poly.use_smooth = True

    # Enable auto smooth for better normal calculation
    mesh_data.use_auto_smooth = True
    mesh_data.auto_smooth_angle = auto_smooth_angle
    
    print(
        f"Applied smooth shading with auto smooth angle {math.degrees(auto_smooth_angle):.1f}° on {mesh_obj.name}"
    )

# ----------------------------
# Helpers
# ----------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # Ensure glTF exporter is available (it ships with Blender)
    if 'io_scene_gltf2' not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module='io_scene_gltf2')

def import_mesh(mesh_path: str):
    ext = os.path.splitext(mesh_path)[1].lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=mesh_path)
    elif ext in [".fbx"]:
        bpy.ops.wm.fbx_import(filepath=mesh_path)
    elif ext in [".glb", ".gltf"]:
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif ext in [".ply"]:
        bpy.ops.wm.ply_import(filepath=mesh_path)
    elif ext in [".stl"]:
        bpy.ops.wm.stl_import(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not mesh_objs:
        raise RuntimeError("No mesh objects found after import.")
    return mesh_objs

def ensure_uv_map(obj, uv_map_name=None):
    if not obj.data.uv_layers:
        raise RuntimeError(f"Object '{obj.name}' has no UV layers.")
    if uv_map_name:
        if uv_map_name not in obj.data.uv_layers:
            raise RuntimeError(
                f"UV map '{uv_map_name}' not found on '{obj.name}'. "
                f"Available: {[uv.name for uv in obj.data.uv_layers]}"
            )
        obj.data.uv_layers.active = obj.data.uv_layers[uv_map_name]
    else:
        # Use whatever is already active
        pass
    return obj.data.uv_layers.active.name

def load_image_node(nodes, image_path, label, colorspace="sRGB"):
    node = nodes.new("ShaderNodeTexImage")
    node.label = label
    node.interpolation = 'Smart'
    img = bpy.data.images.load(image_path)
    node.image = img
    # Color space: albedo = sRGB; data maps = Non-Color
    node.image.colorspace_settings.name = colorspace
    return node

def build_pbr_material(mat_name, base_color_path, roughness_path, metallic_path, uv_map_name=None):
    # Create material
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links

    # Clear default nodes except output
    for n in list(nodes):
        if n.type != 'OUTPUT_MATERIAL':
            nodes.remove(n)
    out = next(n for n in nodes if n.type == 'OUTPUT_MATERIAL')

    # Principled BSDF
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (200, 200)

    # UV Map node (optional—if omitted, Image Texture nodes use the active UV)
    uvnode = nodes.new("ShaderNodeUVMap")
    uvnode.from_instancer = False
    if uv_map_name:
        uvnode.uv_map = uv_map_name
    uvnode.location = (-900, 200)

    # Texture nodes
    tex_base = load_image_node(nodes, base_color_path, "BaseColor", colorspace="sRGB")
    tex_base.location = (-650, 300)

    tex_rough = load_image_node(nodes, roughness_path, "Roughness", colorspace="Non-Color")
    tex_rough.location = (-650, 50)

    tex_metal = load_image_node(nodes, metallic_path, "Metallic", colorspace="Non-Color")
    tex_metal.location = (-650, -200)

    # Wire UVs to textures
    links.new(uvnode.outputs["UV"], tex_base.inputs["Vector"])
    links.new(uvnode.outputs["UV"], tex_rough.inputs["Vector"])
    links.new(uvnode.outputs["UV"], tex_metal.inputs["Vector"])

    # Wire textures to Principled
    links.new(tex_base.outputs["Color"], principled.inputs["Base Color"])
    links.new(tex_rough.outputs["Color"], principled.inputs["Roughness"])
    links.new(tex_metal.outputs["Color"], principled.inputs["Metallic"])

    # Alpha handling (optional): if base color has alpha you want to use
    # principled.inputs["Alpha"].default_value = 1.0
    # out->Surface
    links.new(principled.outputs["BSDF"], out.inputs["Surface"])

    # Ensure glTF-compatible settings
    mat.blend_method = 'OPAQUE'  # set 'CLIP'/'HASHED' if you actually use alpha
    return mat

def assign_material(objs, mat):
    for o in objs:
        if o.type != 'MESH':
            continue
        if not o.data.materials:
            o.data.materials.append(mat)
        else:
            # Replace all existing slots with our mat
            for i in range(len(o.data.materials)):
                o.data.materials[i] = mat

def export_glb(filepath, select_objs=None):
    if select_objs:
        # Select only the provided objects
        bpy.ops.object.select_all(action='DESELECT')
        for o in select_objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = select_objs[0]
        use_selection = True
    else:
        use_selection = False

    bpy.ops.export_scene.gltf(
        filepath=filepath,
        export_format='GLB',
        export_materials='EXPORT',
        export_texcoords=True,
        export_normals=True,
        export_cameras=False,
        export_lights=False,
        export_yup=True,            # glTF is Y-up; Blender is Z-up, exporter handles conversion
        export_image_format='AUTO', # keep original; GLB will embed
        use_selection=use_selection,
        export_apply=True
    )

# ----------------------------
# Main
# ----------------------------

def parse_args():
    # Blender passes its own args before '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    ap = argparse.ArgumentParser(description="Apply PBR textures and export GLB")
    ap.add_argument("--mesh", default="/localhome/aha220/Hairdar/modules/Hunyuan3D-2.1/demo_output/demo_textured.ply", help="Path to mesh (OBJ/FBX/STL/PLY/GLB/GLTF)")
    ap.add_argument("--base_color", default="/localhome/aha220/Hairdar/modules/Hunyuan3D-2.1/demo_output/demo_textured.jpg", help="Albedo/BaseColor image (sRGB)")
    ap.add_argument("--metallic", default="/localhome/aha220/Hairdar/modules/Hunyuan3D-2.1/demo_output/demo_textured_metallic.jpg", help="Roughness image (Non-Color)")
    ap.add_argument("--roughness", default="/localhome/aha220/Hairdar/modules/Hunyuan3D-2.1/demo_output/demo_textured_roughness.jpg", help="Metallic image (Non-Color)")
    ap.add_argument("--uv_map", default=None, help="UV map name to use (default: active UV)")
    ap.add_argument("--out", default="./textured.glb", help="Output .glb path")
    ap.add_argument("--subdivision_levels", type=int, default=1, help="Viewport subdivision levels for smoother surfaces")
    ap.add_argument("--subdivision_render_levels", type=int, default=3, help="Render subdivision levels for export")
    ap.add_argument("--subdivision_type", choices=["CATMULL_CLARK", "SIMPLE"], default="SIMPLE", help="Subdivision surface type")
    ap.add_argument("--auto_smooth_angle", type=float, default=30.0, help="Auto smooth angle in degrees")
    ap.add_argument("--triangulate_quad_method", choices=["BEAUTY", "FIXED", "FIXED_ALTERNATE", "SHORTEST_DIAGONAL"], default="BEAUTY", help="Triangulate modifier quad method")
    ap.add_argument("--triangulate_ngon_method", choices=["BEAUTY", "CLIP"], default="BEAUTY", help="Triangulate modifier ngon method")
    return ap.parse_args(argv)

def add_texture_to_mesh(
    mesh_path,
    base_color_path,
    metallic_path,
    roughness_path,
    uv_map_name=None,
    out_path="./textured.glb",
    subdivision_levels=2,
    subdivision_render_levels=3,
    subdivision_type="SIMPLE",
    auto_smooth_angle=30.0,
    triangulate_quad_method="BEAUTY",
    triangulate_ngon_method="BEAUTY"
):
    reset_scene()

    mesh_objs = import_mesh(os.path.abspath(mesh_path))

    # Ensure the chosen UV map is active on each mesh object
    active_uv_name = None
    smooth_angle_deg = max(0.0, min(180.0, auto_smooth_angle))
    smooth_angle_rad = math.radians(smooth_angle_deg)
    for obj in mesh_objs:
        active_uv_name = ensure_uv_map(obj, uv_map_name)  # validates and (optionally) sets active
        apply_subdivision_surface(
            obj,
            levels=subdivision_levels,
            render_levels=subdivision_render_levels,
            subdivision_type=subdivision_type,
        )
        # apply_triangulate_modifier(
        #     obj,
        #     quad_method=triangulate_quad_method,
        #     ngon_method=triangulate_ngon_method,
        # )
        apply_smooth_shading(obj, auto_smooth_angle=smooth_angle_rad)
    # Build material (use the validated/active UV name)
    mat = build_pbr_material(
        mat_name="PBR_Material",
        base_color_path=os.path.abspath(base_color_path),
        roughness_path=os.path.abspath(roughness_path),
        metallic_path=os.path.abspath(metallic_path),
        uv_map_name=active_uv_name
    )
    assign_material(mesh_objs, mat)

    # Export GLB
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export_glb(out_path, select_objs=mesh_objs)

    print(f"[OK] Exported: {out_path}")

if __name__ == "__main__":
    parser = parse_args()
    add_texture_to_mesh(
        mesh_path=parser.mesh,
        base_color_path=parser.base_color,
        metallic_path=parser.metallic,
        roughness_path=parser.roughness,
        uv_map_name=parser.uv_map,
        out_path=parser.out,
        subdivision_levels=parser.subdivision_levels,
        subdivision_render_levels=parser.subdivision_render_levels,
        subdivision_type=parser.subdivision_type,
        auto_smooth_angle=parser.auto_smooth_angle,
        triangulate_quad_method=parser.triangulate_quad_method,
        triangulate_ngon_method=parser.triangulate_ngon_method
    )
