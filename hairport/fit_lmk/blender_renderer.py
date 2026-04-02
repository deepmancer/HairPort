import bpy
import math
import numpy as np
from pathlib import Path
from mathutils import Matrix, Vector


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
    
    if max_gpus is None:
        devices_to_use = gpu_devices
    else:
        devices_to_use = gpu_devices[:max_gpus]
    
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
    
    return {
        'gpus': activated_gpus,
        'compute_type': selected_type
    }


def setup_scene(resolution=1024, max_gpus=None):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    gpu_info = enable_gpus(max_gpus=max_gpus)
    compute_type = gpu_info['compute_type']
    
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    scene.cycles.samples = 256
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.adaptive_min_samples = 64
    
    if hasattr(scene.cycles, "use_denoising"):
        scene.cycles.use_denoising = True
    
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.film_transparent = True
    
    scene.cycles.tile_size = 256
    
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
    
    if hasattr(scene.cycles, 'preview_samples'):
        scene.cycles.preview_samples = 32
    
    if hasattr(scene.cycles, 'scrambling_distance'):
        scene.cycles.scrambling_distance = 1.0
    
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 12
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 8
    
    if scene.world:
        if scene.world.use_nodes:
            for node in scene.world.node_tree.nodes:
                if node.type == 'BACKGROUND':
                    node.inputs['Strength'].default_value = 0.0
        else:
            scene.world.color = (0, 0, 0)
    
    print(f"Scene configured with {len(gpu_info['gpus'])} GPU(s) using {compute_type}")
    return scene, gpu_info


def load_mesh(mesh_path):
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    mesh_ext = mesh_path.suffix.lower()
    
    if mesh_ext in {'.glb', '.gltf'}:
        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    elif mesh_ext == '.obj':
        bpy.ops.wm.obj_import(filepath=str(mesh_path))
    elif mesh_ext == '.ply':
        bpy.ops.wm.ply_import(filepath=str(mesh_path))
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_ext}")
    
    imported_meshes = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
    if not imported_meshes:
        raise RuntimeError(f"No mesh imported from {mesh_path}")
    
    if mesh_ext in {'.glb', '.gltf'}:
        rotation_matrix = Matrix.Rotation(math.radians(90.0), 4, 'X')
        for obj in imported_meshes:
            obj.matrix_world = rotation_matrix @ obj.matrix_world
        bpy.context.view_layer.update()
    
    if mesh_ext in [".obj", ".ply"]:
        rotation_matrix = Matrix.Rotation(math.radians(180.0), 4, 'Y')
        for obj in imported_meshes:
            obj.matrix_world = rotation_matrix @ obj.matrix_world
        bpy.context.view_layer.update()
    
    return imported_meshes


def setup_camera(scene, cam_loc, cam_rot, ortho_scale=1.0):
    camera_data = bpy.data.cameras.new(name='LandmarkCamera')
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = ortho_scale
    camera_data.clip_start = 0.01
    camera_data.clip_end = 100.0

    camera_obj = bpy.data.objects.new('LandmarkCamera', camera_data)
    camera_obj.location = cam_loc
    camera_obj.rotation_euler = cam_rot
    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj

    return camera_obj


def setup_lighting(scene, camera_obj):
    light_data = bpy.data.lights.new(name='KeyLight', type='AREA')
    light_data.energy = 7000.0
    light_data.size = 30
    light_obj = bpy.data.objects.new('KeyLight', light_data)
    light_obj.location = camera_obj.location
    light_obj.rotation_euler = camera_obj.rotation_euler
    scene.collection.objects.link(light_obj)
    
    return light_obj


def render_view(scene, output_path):
    scene.render.filepath = str(output_path)
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True, use_viewport=False)


def render_multi_view(
    mesh_path,
    cam_loc,
    cam_rot,
    mesh_rotations,
    output_dir,
    ortho_scale=1.1,
    resolution=1024,
    max_gpus=None
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    scene, gpu_info = setup_scene(resolution=resolution, max_gpus=max_gpus)
    
    mesh_path_obj = Path(mesh_path)
    mesh_ext = mesh_path_obj.suffix.lower()
    
    imported_meshes = load_mesh(mesh_path)
    
    if mesh_ext in {'.glb', '.gltf'}:
        rotation_axis = 'Z'
    else:
        rotation_axis = 'Y'
    
    original_transforms = []
    for mesh_obj in imported_meshes:
        original_transforms.append(mesh_obj.matrix_world.copy())
    
    camera_obj = setup_camera(scene, cam_loc, cam_rot, ortho_scale)
    setup_lighting(scene, camera_obj)
    
    rendered_paths = []
    
    for idx, rotation_angle in enumerate(mesh_rotations):
        rotation_matrix = Matrix.Rotation(float(rotation_angle), 4, rotation_axis)
        
        for mesh_obj, original_transform in zip(imported_meshes, original_transforms):
            bbox_center = 0.125 * sum((Vector(corner) for corner in mesh_obj.bound_box), Vector())
            mesh_center_world = mesh_obj.matrix_world @ bbox_center
            
            T_to_origin = Matrix.Translation(-mesh_center_world)
            T_from_origin = Matrix.Translation(mesh_center_world)
            
            mesh_obj.matrix_world = T_from_origin @ rotation_matrix @ T_to_origin @ original_transform
        
        bpy.context.view_layer.update()
        
        output_path = output_dir / f'view_{idx:02d}.png'
        render_view(scene, output_path)
        rendered_paths.append(output_path)
        
        print(f"Rendered view {idx} (Y-axis rotation: {math.degrees(rotation_angle):.1f}°) to {output_path}")
    
    return rendered_paths
