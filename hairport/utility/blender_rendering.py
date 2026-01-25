import bpy
from mathutils import Vector
import math
from pathlib import Path
from typing import Tuple
import bpy
from PIL import Image


def setup_camera(
    location: Tuple[float, float, float] = (0.0, -1.2, 1.82),
    rotation: Tuple[float, float, float] = (math.radians(90.0), 0.0, 0.0),
    ortho_scale: float = 1.0,
):
    """Setup Blender orthographic camera with given parameters."""
    scn = bpy.context.scene
    cam1 = bpy.data.cameras.new("Camera 1")
    cam1_obj = bpy.data.objects.new("Camera 1", cam1)
    cam1_obj.location = location
    cam1_obj.rotation_euler = rotation
    cam1_obj.data.type = 'ORTHO'
    cam1_obj.data.ortho_scale = float(ortho_scale)
    scn.collection.objects.link(cam1_obj)

    return cam1_obj

def enable_gpus(max_gpus=None):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    
    available_types = []
    try:
        for compute_type in ['OPTIX', 'HIP', 'CUDA', 'METAL', 'OPENCL']:
            try:
                cycles_preferences.compute_device_type = compute_type
                cycles_preferences.refresh_devices()
                gpu_devices = [d for d in cycles_preferences.devices if d.type != 'CPU']
                if gpu_devices:
                    available_types.append((compute_type, len(gpu_devices)))
                    print(f"Found {len(gpu_devices)} GPU(s) for {compute_type}")
            except (AttributeError, TypeError):
                continue
    except Exception as e:
        print(f"Warning during device detection: {e}")
    
    if not available_types:
        raise RuntimeError("No GPU compute devices available. Falling back to CPU is not implemented.")
    
    compute_type_priority = {
        'OPTIX': 3,
        'CUDA': 4,
        'HIP': 2,
        'METAL': 1,
        'OPENCL': 0
    }
    
    available_types.sort(key=lambda x: compute_type_priority.get(x[0], -1), reverse=True)
    selected_type = available_types[0][0]
    
    cycles_preferences.compute_device_type = selected_type
    cycles_preferences.refresh_devices()
    
    gpu_devices = [d for d in cycles_preferences.devices if d.type != 'CPU']
    if gpu_devices and selected_type == 'OPTIX':
        compute_gpu_indicators = ['H100', 'A100', 'A40', 'A30', 'A10', 'V100', 'P100', 'Tesla']
        for device in gpu_devices:
            if any(indicator in device.name for indicator in compute_gpu_indicators):
                if 'CUDA' in [t[0] for t in available_types]:
                    selected_type = 'CUDA'
                    print(f"Detected compute GPU ({device.name}), switching from OPTIX to CUDA for better performance")
                    break
    
    cycles_preferences.compute_device_type = selected_type
    cycles_preferences.refresh_devices()
    
    all_devices = cycles_preferences.devices
    gpu_devices = [d for d in all_devices if d.type != 'CPU']
    
    if not gpu_devices:
        raise RuntimeError(f"No GPU devices found for {selected_type}")
    
    devices_to_use = gpu_devices if max_gpus is None else gpu_devices[:max_gpus]
    
    activated_gpus = []
    for device in all_devices:
        if device in devices_to_use:
            device.use = True
            activated_gpus.append(device.name)
            print(f"Activated GPU: {device.name}")
        else:
            device.use = False
    
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.use_persistent_data = True
    
    return {'gpus': activated_gpus, 'compute_type': selected_type}

def render_mesh(camera_location, camera_rotation, output_path, 
                mesh_path, texture_root=None, resolution=1024, ortho_scale=1.0):
    """Render a GLB mesh with embedded textures using Blender Cycles with orthographic camera."""
    if bpy is None:
        raise RuntimeError("Blender Python API (bpy) is unavailable; run inside Blender 4.0+")

    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    mesh_ext = mesh_path.suffix.lower()
    if mesh_ext not in {'.glb', '.gltf'}:
        raise ValueError(f"Expected GLB/GLTF format, got '{mesh_ext}'")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    gpu_info = enable_gpus()
    compute_type = gpu_info['compute_type']

    # Configure rendering
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 512
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.adaptive_min_samples = 64
    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = True
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(output_path)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True

    # Configure denoising
    view_layer = bpy.context.view_layer
    if hasattr(view_layer, "cycles"):
        view_layer.cycles.use_denoising = True
        if hasattr(view_layer.cycles, "denoiser"):
            if compute_type == 'OPTIX':
                view_layer.cycles.denoiser = 'OPTIX'
                print("Using OPTIX denoiser (RTX GPU detected)")
            elif compute_type == 'CUDA':
                view_layer.cycles.denoiser = 'OPENIMAGEDENOISE'
                print("Using OpenImageDenoise denoiser (CUDA compute mode)")
            elif compute_type == 'HIP':
                view_layer.cycles.denoiser = 'OPENIMAGEDENOISE'
                print("Using OpenImageDenoise denoiser (AMD GPU)")
            else:
                view_layer.cycles.denoiser = 'OPENIMAGEDENOISE'
                print(f"Using OpenImageDenoise denoiser ({compute_type})")
    
    # Setup transparent world
    if scene.world:
        if scene.world.use_nodes:
            bg_node = scene.world.node_tree.nodes.get('Background')
            if bg_node is not None:
                bg_node.inputs['Strength'].default_value = 0.0
        else:
            scene.world.use_nodes = True
            scene.world.node_tree.nodes.clear()
    else:
        scene.world = bpy.data.worlds.new("TransparentWorld")
        scene.world.use_nodes = True
        scene.world.node_tree.nodes.clear()

    # Import GLB/GLTF mesh with embedded textures
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    
    # No mesh rotation needed - coordinate systems are handled via camera transformation
    imported_meshes = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not imported_meshes:
        raise RuntimeError(f"No mesh geometry imported from {mesh_path}")

    # Update mesh transforms to match coordinate system conventions
    # bpy.context.view_layer.update()
    # for obj in imported_meshes:
    #     obj.scale.y = -obj.scale.y
    #     obj.scale.z = -obj.scale.z
    # bpy.context.view_layer.update()
    print(f"Imported {len(imported_meshes)} mesh(es) with embedded materials")

    # Verify materials are present
    for obj in imported_meshes:
        if not obj.data.materials:
            print(f"Warning: Mesh '{obj.name}' has no materials")
        else:
            print(f"Mesh '{obj.name}' has {len(obj.data.materials)} material(s)")
            for mat in obj.data.materials:
                if mat and mat.use_nodes:
                    print(f"  - Material '{mat.name}' uses nodes (embedded textures)")

    # Setup orthographic camera and lighting
    print(f"Camera location: {camera_location}")
    print(f"Camera rotation (Euler): {camera_rotation}")
    print(f"Camera type: ORTHO")
    print(f"Camera ortho scale: {ortho_scale}")
    
    camera_obj = setup_camera(
        location=camera_location,
        rotation=camera_rotation,
        ortho_scale=ortho_scale,
    )
    scene.camera = camera_obj

    # Add area light at camera position
    light_data = bpy.data.lights.new(name='HairdarKeyLight', type='AREA')
    light_data.energy = 3500.0
    light_data.size = 20
    light_obj = bpy.data.objects.new('HairdarKeyLight', light_data)
    light_obj.location = camera_obj.location
    light_obj.rotation_euler = camera_obj.rotation_euler
    scene.collection.objects.link(light_obj)
    
    # Add fill light from the left
    fill_light_data = bpy.data.lights.new(name='HairdarFillLight', type='AREA')
    fill_light_data.energy = 2000.0
    fill_light_data.size = 15
    fill_light_obj = bpy.data.objects.new('HairdarFillLight', fill_light_data)
    fill_light_obj.location = (camera_obj.location[0] - 3.0, camera_obj.location[1], camera_obj.location[2])
    # Point light towards center (0, 0, 0)
    fill_light_obj.rotation_euler = (math.radians(90.0), 0.0, math.radians(-30.0))
    scene.collection.objects.link(fill_light_obj)
    
    # Add rim light from the right
    rim_light_data = bpy.data.lights.new(name='HairdarRimLight', type='AREA')
    rim_light_data.energy = 2000.0
    rim_light_data.size = 15
    rim_light_obj = bpy.data.objects.new('HairdarRimLight', rim_light_data)
    rim_light_obj.location = (camera_obj.location[0] + 3.0, camera_obj.location[1], camera_obj.location[2])
    # Point light towards center (0, 0, 0)
    rim_light_obj.rotation_euler = (math.radians(90.0), 0.0, math.radians(30.0))
    scene.collection.objects.link(rim_light_obj)

    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    
    rendered_image = Image.open(output_path)
    return rendered_image
