"""Utility functions for rendering."""

import bpy
from mathutils import Vector
import math
from typing import Tuple


def get_selected_camera_views():
    """Returns camera views with absolute parameters"""
    import math
    from mathutils import Vector
    views = [
        {'name': 'original_front', 'location': Vector((0.000000, -1.200000, 1.868000)), 'rotation': (1.363886, -0.000000, 0.001014)},
        {'name': 'original_back', 'location': Vector((0.000000, 1.200000, 1.868000)), 'rotation': (1.366587, -0.000000, 3.140592)},
        {'name': 'view_front_15_positive', 'location': Vector((0.280573, -1.047113, 1.670830)), 'rotation': (1.521580, 0.000000, 0.264834)},
        {'name': 'view_front_15_negative', 'location': Vector((-0.311924, -1.164115, 1.821198)), 'rotation': (1.402489, -0.000000, -0.262577)},
        {'name': 'view_aerial', 'location': Vector((-0.001209, -0.008110, 2.817804)), 'rotation': (0.000000, -0.000000, 0.000000)},
    ]

    return views


def setup_camera(
    location: Tuple[float, float, float] = (0.0, -1.2, 1.82),
    rotation: Tuple[float, float, float] = (math.radians(90.0), 0.0, 0.0),
    focal_length: float = 50.00,
):
    """
    Add a perspective camera and configure its physical intrinsics.

    Parameters
    ----------
    location, rotation
        Pose of the new camera in world space (Blender units, radians).
    focal_length
        Focal length in millimetres (Camera → Lens panel ▸ Focal Length).
    sensor_width, sensor_height
        Physical dimensions of the camera’s imaging sensor in millimetres.
    sensor_fit
        Whether Blender should keep the horizontal FOV fixed ('HORIZONTAL'),
        the vertical FOV fixed ('VERTICAL'), or decide automatically ('AUTO').
    """
    # ----------------------------------------------------------------------
    # 1. add camera object and make it the scene’s active camera
    # ----------------------------------------------------------------------
    bpy.ops.object.camera_add(location=location, rotation=rotation)
    camera = bpy.context.active_object

    # ----------------------------------------------------------------------
    # 2. set perspective parameters
    # ----------------------------------------------------------------------
    camera.data.type = 'PERSP'                         # just to be explicit
    camera.data.clip_start = 0.01
    camera.data.clip_end   = 100.0

    # ------- physical intrinsics ------------------------------------------
    camera.data.lens      = float(focal_length)            # focal length
    bpy.context.scene.camera = camera
    return camera

def setup_camera_ortho(
    location: Tuple[float, float, float] = (0.0, -1.2, 1.82),
    rotation: Tuple[float, float, float] = (math.radians(90.0), 0.0, 0.0),
    ortho_scale: float = 2.5,
):
    """
    Add an orthographic camera and configure its physical intrinsics.

    Parameters
    ----------
    location, rotation
        Pose of the new camera in world space (Blender units, radians).
    ortho_scale
        Orthographic scale (Camera → Lens panel ▸ Orthographic Scale).
    """
    # ----------------------------------------------------------------------
    # 1. add camera object and make it the scene’s active camera
    # ----------------------------------------------------------------------
    bpy.ops.object.camera_add(location=location, rotation=rotation)
    camera = bpy.context.active_object

    # ----------------------------------------------------------------------
    # 2. set orthographic parameters
    # ----------------------------------------------------------------------
    camera.data.type = 'ORTHO'                         # just to be explicit
    camera.data.clip_start = 0.01
    camera.data.clip_end   = 100.0

    # ------- physical intrinsics ------------------------------------------
    camera.data.ortho_scale      = float(ortho_scale)            # orthographic scale
    bpy.context.scene.camera = camera
    return camera

def create_flat_material(color):
    """Create a flat material with the given color. Note that the color should be in linear space."""
    mat = bpy.data.materials.new(name=f"FlatMaterial_{color[0]}_{color[1]}_{color[2]}")
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    mat.shadow_method = "NONE"

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    rgb = nodes.new("ShaderNodeRGB")

    # Add gamma correction for linear space
    gamma = nodes.new("ShaderNodeGamma")
    gamma.inputs[1].default_value = 2.2  # Standard gamma correction

    rgb.outputs[0].default_value = (*color, 1.0)

    links.new(rgb.outputs[0], gamma.inputs[0])
    links.new(gamma.outputs[0], emission.inputs[0])
    links.new(emission.outputs[0], output.inputs[0])

    return mat

def create_realistic_body_material():
    """Creates material for realistic skin rendering"""
    mat = bpy.data.materials.new(name="RealisticBodyMaterial")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")


    # More natural skin shader settings
    principled.inputs["Base Color"].default_value = (0.9, 0.75, 0.68, 1.0)
    principled.inputs["Subsurface Weight"].default_value = 0.15

    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Specular IOR Level"].default_value = 0.3
    principled.inputs["Roughness"].default_value = 1
    principled.inputs["Coat Weight"].default_value = 0.1
    principled.inputs["Coat Roughness"].default_value = 0.3
    links.new(principled.outputs[0], output.inputs[0])
    return mat

def compute_camera_z(x_bounds, y_bounds, z_bounds, y_cam=1.82, focal_length=50.00):
    """
    Computes the minimum (most negative) z-value for the camera location to ensure all points are visible.
    
    Args:
        x_bounds (tuple): (x_min, x_max) bounding box in world coordinates.
        y_bounds (tuple): (y_min, y_max) bounding box in world coordinates.
        z_bounds (tuple): (z_min, z_max) bounding box in world coordinates.
        y_cam (float): Camera's fixed Y-coordinate.
        focal_length (float): Camera focal length in mm (default 50.0).
    
    Returns:
        float: Minimum camera Z-coordinate.
    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    z_min, z_max = z_bounds
    sensor_height = 36.0  # mm, standard full-frame sensor height
    sensor_width = 36.0  # mm, standard full-frame sensor width
    # Half sensor dimensions

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    z_min, _ = z_bounds  # Only z_min constrains visibility
    
    # Compute maximum absolute deviations
    max_x = max(abs(x_min), abs(x_max))
    max_y = max(abs(y_min - y_cam), abs(y_max - y_cam))
    
    # Compute constraint terms
    term_x = (1.275*focal_length * max_x) / sensor_width
    term_y = (1.275*focal_length * max_y) / sensor_height
    
    # Camera Z is constrained by the more restrictive term
    z_cam = min(z_min - term_x, z_min - term_y)
    # Smallest candidate (most negative) ensures all points are visible
    z_min_const = -1.30  * (focal_length / 50.0)  # Adjusted for focal length
    z_max_const = -0.65  * (focal_length / 50.0)  # Adjusted for focal length
    min_relative_z = -0.70 * (focal_length / 50.0)  # Adjusted for focal length
    z_cam = min(z_cam , z_min + min_relative_z, z_max_const)
    z_cam = max(z_cam, z_min_const)
    return z_cam



def compute_camera_z_back(x_bounds, y_bounds, z_bounds, y_cam=1.82, focal_length=50.00):
    """
    Computes the minimum positive z-value for the back view camera location to ensure all points are visible.
    This is for a camera positioned 180 degrees from the front view (behind the object).
    
    Args:
        x_bounds (tuple): (x_min, x_max) bounding box in world coordinates.
        y_bounds (tuple): (y_min, y_max) bounding box in world coordinates.
        z_bounds (tuple): (z_min, z_max) bounding box in world coordinates.
        y_cam (float): Camera's fixed Y-coordinate.
        focal_length (float): Camera focal length in mm (default 50.0).
    
    Returns:
        float: Minimum positive camera Z-coordinate for back view.
    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    z_min, z_max = z_bounds
    sensor_height = 36.0  # mm, standard full-frame sensor height
    sensor_width = 36.0  # mm, standard full-frame sensor width
    
    # For back view, we need to see all points from behind (positive z direction)
    # Use z_max as the reference point since camera is behind the object
    
    # Compute maximum absolute deviations
    max_x = max(abs(x_min), abs(x_max))
    max_y = max(abs(y_min - y_cam), abs(y_max - y_cam))
    
    # Compute constraint terms for field of view coverage
    term_x = (1.275 * focal_length * max_x) / sensor_width
    term_y = (1.275 * focal_length * max_y) / sensor_height
    
    # Camera Z must be far enough behind z_max to see all points
    z_cam = max(z_max + term_x, z_max + term_y)
    
    # Apply constraints for reasonable camera positioning
    z_min_const = 0.70 * (focal_length / 50.0)   # Minimum positive distance
    z_max_const = 1.5 * (focal_length / 50.0)   # Maximum reasonable distance
    min_relative_z = 0.6 * (focal_length / 50.0)  # Minimum relative distance from object
    
    # Ensure camera is positioned with adequate distance
    z_cam = max(z_cam, z_max + min_relative_z, z_min_const)
    z_cam = min(z_cam, z_max_const)
    
    return z_cam

def create_difflocks_realistic_hair_material(metadata: dict):
    """
    Build the Difflocks hair material so it mirrors the handcrafted
    Bgen_Hair_Shader node graph from the reference .blend.

    The ``metadata`` dictionary can override the baked shader controls:
    material_name, root_darkness_start, root_darkness_end, root_darkness_strength,
    material_melanin_amount, bsdf_melanin_redness, bsdf_roughness,
    bsdf_radial_roughness, bsdf_random_roughness, bsdf_coat, bsdf_ior,
    bsdf_offset, wave_variation_scale, wave_variation_phase, melanin_wave_scale,
    melanin_wave_phase, melanin_wave_strength, melanin_wave_bias,
    melanin_wave_divisor, eevee_variation_factor, eevee_specular_mix,
    eevee_specular_tint_mix, color_switch_factor, color_hue_shift,
    color_saturation, color_value, variation_secondary_color, viewport_color,
    viewport_metallic, viewport_roughness, eevee_root_color, eevee_tip_color,
    eevee_root_position, eevee_tip_position.
    """
    defaults = {
        'root_darkness_start': 0.0,
        'root_darkness_end': 0.2920003,
        'root_darkness_strength': 0.5,
        'material_melanin_amount': 0.4,
        'bsdf_melanin_redness': 0.13997258,
        'bsdf_roughness': 0.16827232,
        'bsdf_radial_roughness': 0.69477248,
        'bsdf_random_roughness': 0.23256475,
        'bsdf_coat': 0.48664537,
        'bsdf_ior': 1.5,
        'bsdf_offset': 0.034906585,
        'wave_variation_scale': 9.799999,
        'wave_variation_phase': 2.0999994,
        'melanin_wave_scale': 5.8385777,
        'melanin_wave_phase': 280.45837,
        'melanin_wave_detail': 0.0,
        'melanin_wave_detail_scale': 0.0,
        'melanin_wave_detail_roughness': 0.60000002,
        'melanin_wave_strength': 0.86493331,
        'melanin_wave_bias': 0.5,
        'melanin_wave_divisor': 3.0,
        'eevee_variation_factor': 0.5,
        'eevee_specular_mix': 0.1717868,
        'eevee_specular_tint_mix': 0.57272732,
        'color_switch_factor': 0.0,
        'color_hue_shift': 0.5,
        'color_saturation': 1.0,
        'color_value': 1.0,
        'variation_secondary_color': (0.61107123, 0.48011706, 0.5006873, 1.0),
        'viewport_color': (0.61107123, 0.48011706, 0.5006873, 1.0),
        'viewport_metallic': 0.1717868,
        'viewport_roughness': 0.23604027,
        'eevee_root_color': (0.10291953, 0.03619754, 0.01299444, 1.0),
        'eevee_tip_color': (0.84203988, 0.43962979, 0.2396407, 1.0),
        'eevee_root_position': 0.1468254,
        'eevee_tip_position': 0.57142824,
    }
    params = {key: metadata.get(key, value) for key, value in defaults.items()}
    material_name = metadata.get('material_name', 'OptimizedHairMaterial')

    def _clamp01(value):
        return max(0.0, min(1.0, float(value)))

    def _ensure_color(value, fallback):
        base = value if isinstance(value, (tuple, list)) else fallback
        if len(base) == 3:
            return (float(base[0]), float(base[1]), float(base[2]), 1.0)
        if len(base) >= 4:
            return tuple(float(base[i]) for i in range(4))
        return tuple(float(fallback[i]) for i in range(4))

    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    mat.blend_method = 'HASHED'
    mat.shadow_method = 'HASHED'

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    node_map = {}

    def add_node(name: str, node_type: str, location, label: str = ""):
        node = nodes.new(node_type)
        node.name = name
        node.location = location
        node.label = label
        node_map[name] = node
        return node

    # Node creation ---------------------------------------------------------
    add_node('Material Output', 'ShaderNodeOutputMaterial', (1459.967, 70.054))
    add_node('Material Output.001', 'ShaderNodeOutputMaterial', (2319.364, 1027.087))

    for name, location, attr_name in (
        ('Attribute.001', (-1020.762, 116.592), 'matt'),
        ('Attribute.002', (-1073.616, 667.697), 'matt'),
        ('Attribute.004', (295.534, -327.425), 'surface'),
        ('Attribute.005', (-1073.616, 667.697), 'matt'),
    ):
        node = add_node(name, 'ShaderNodeAttribute', location)
        node.attribute_name = attr_name
        node.attribute_type = 'GEOMETRY'

    for name, location in (
        ('Mapping.001', (-1017.67, 77.723)),
        ('Mapping.002', (-1074.728, 632.277)),
        ('Mapping.004', (-1074.728, 632.277)),
    ):
        node = add_node(name, 'ShaderNodeMapping', location)
        node.vector_type = 'POINT'
        node.inputs['Location'].default_value = (0.0, 0.0, 0.0)
        node.inputs['Rotation'].default_value = (0.0, 0.0, 0.0)
        node.inputs['Scale'].default_value = (1.0, 1.0, 1.0)

    for name, location in (
        ('Separate XYZ', (-1017.815, 39.921)),
        ('Separate XYZ.001', (-1071.53, 597.154)),
        ('Separate XYZ.003', (-1071.53, 597.154)),
        ('Separate XYZ.004', (534.11, 815.401)),
    ):
        add_node(name, 'ShaderNodeSeparateXYZ', location)

    add_node('Separate Color', 'ShaderNodeSeparateColor', (1554.968, 443.487))
    add_node('Curves Info', 'ShaderNodeHairInfo', (1080.338, 327.057))
    add_node('Curves Info.001', 'ShaderNodeHairInfo', (1229.492, 1213.401))
    add_node('Reroute', 'NodeReroute', (999.594, -188.075))

    gradient = add_node('Eevee Gradient', 'ShaderNodeValToRGB', (-828.005, 216.855), 'Gradient Fall_off')
    variation_ramp = add_node('ColorRamp.003', 'ShaderNodeValToRGB', (-678.722, 758.086))
    root_ramp = add_node('root_darkness_color_ramp', 'ShaderNodeValToRGB', (1259.636, 353.396))
    random_ramp = add_node('Color Ramp', 'ShaderNodeValToRGB', (1429.136, 1213.123))

    wave_variation = add_node('Wave Texture.001', 'ShaderNodeTexWave', (-880.31, 748.245))
    wave_cycles = add_node('Wave Texture Cycles', 'ShaderNodeTexWave', (-895.933, 738.616))

    hsv = add_node('Hue Saturation Value', 'ShaderNodeHueSaturation', (296.154, -113.188))
    color_switch = add_node('color_switch_eevee', 'ShaderNodeMix', (547.86, -136.639))
    variation = add_node('Eevee Variation', 'ShaderNodeMix', (759.0, 96.775), 'Variation Intensity')
    mix_spec = add_node('Mix', 'ShaderNodeMix', (839.158, -13.176))
    mix_tint = add_node('Mix.001', 'ShaderNodeMix', (939.158, -13.176))

    math_specs = (
        ('Math.001', (701.295, 802.992), 'ADD', False),
        ('Math.002', (729.024, 756.893), 'ADD', False),
        ('Math.003', (889.519, 777.854), 'DIVIDE', True),
        ('Math.005', (645.991, 934.555), 'SUBTRACT', False),
        ('Wave Strength 2', (839.992, 1022.88), 'MULTIPLY', True),
        ('Melanin3', (1035.07, 1073.032), 'ADD', True),
        ('Math.004', (1214.973, 931.112), 'ADD', False),
        ('root_darkness_strength', (1774.278, 449.255), 'SUBTRACT', True),
    )
    for name, location, operation, use_clamp in math_specs:
        node = add_node(name, 'ShaderNodeMath', location)
        node.operation = operation
        node.use_clamp = use_clamp

    cycles_bsdf = add_node('Cycles bsdf', 'ShaderNodeBsdfHairPrincipled', (1392.31, 1015.661))
    cycles_bsdf_secondary = add_node('Cycles bsdf.001', 'ShaderNodeBsdfHairPrincipled', (1781.581, 1226.548))
    eevee_bsdf = add_node('Eevee bsdf', 'ShaderNodeBsdfPrincipled', (1109.158, 106.824))

    # Helpers ---------------------------------------------------------------
    def configure_color_ramp(node, color_mode, interpolation, hue_interp, stops):
        ramp = node.color_ramp
        ramp.color_mode = color_mode
        ramp.interpolation = interpolation
        ramp.hue_interpolation = hue_interp
        while len(ramp.elements) < len(stops):
            ramp.elements.new(0.5)
        while len(ramp.elements) > len(stops):
            ramp.elements.remove(ramp.elements[-1])
        for element, (position, color) in zip(ramp.elements, stops):
            element.position = position
            element.color = color

    def set_socket(node, socket_name, value):
        if socket_name not in node.inputs:
            return
        socket = node.inputs[socket_name]
        if isinstance(value, (tuple, list)):
            socket.default_value = tuple(float(v) for v in value)
        else:
            socket.default_value = float(value)

    # Configure color ramps -------------------------------------------------
    gradient_root_pos = _clamp01(params['eevee_root_position'])
    gradient_tip_pos = _clamp01(params['eevee_tip_position'])
    if gradient_tip_pos <= gradient_root_pos:
        gradient_tip_pos = min(1.0, gradient_root_pos + 0.01)

    gradient_root = _ensure_color(params['eevee_root_color'], defaults['eevee_root_color'])
    gradient_tip = _ensure_color(params['eevee_tip_color'], defaults['eevee_tip_color'])
    configure_color_ramp(
        gradient,
        'RGB',
        'LINEAR',
        'NEAR',
        [
            (gradient_root_pos, gradient_root),
            (gradient_tip_pos, gradient_tip),
        ],
    )

    configure_color_ramp(
        variation_ramp,
        'RGB',
        'LINEAR',
        'NEAR',
        [
            (0.031818, (0.0, 0.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 1.0, 1.0)),
        ],
    )

    root_start = _clamp01(params['root_darkness_start'])
    root_end = _clamp01(params['root_darkness_end'])
    if root_end <= root_start:
        root_end = min(1.0, root_start + 0.01)
    configure_color_ramp(
        root_ramp,
        'RGB',
        'LINEAR',
        'NEAR',
        [
            (root_start, (0.0, 0.0, 0.0, 1.0)),
            (root_end, (1.0, 1.0, 1.0, 1.0)),
        ],
    )

    random_color_stops = [
        (0.0, (1.0, 0.3611866, 0.0, 1.0)),
        (0.15199997, (0.0, 1.0, 0.97076285, 1.0)),
        (0.29399997, (0.30117694, 1.0, 0.0, 1.0)),
        (0.43299994, (0.0, 0.23712516, 1.0, 1.0)),
        (0.58550036, (0.44585222, 0.0, 1.0, 1.0)),
        (0.71349996, (1.0, 0.32064715, 0.46955317, 1.0)),
        (0.82399988, (1.0, 1.0, 1.0, 1.0)),
    ]
    configure_color_ramp(random_ramp, 'RGB', 'CONSTANT', 'NEAR', random_color_stops)

    # Wave textures ---------------------------------------------------------
    wave_variation.wave_type = 'BANDS'
    wave_variation.wave_profile = 'SIN'
    wave_variation.bands_direction = 'X'
    wave_variation.rings_direction = 'X'
    set_socket(wave_variation, 'Vector', (0.0, 0.0, 0.0))
    set_socket(wave_variation, 'Scale', params['wave_variation_scale'])
    set_socket(wave_variation, 'Distortion', 0.0)
    set_socket(wave_variation, 'Detail', 2.0)
    set_socket(wave_variation, 'Detail Scale', 1.0)
    set_socket(wave_variation, 'Detail Roughness', 0.66153848)
    set_socket(wave_variation, 'Phase Offset', params['wave_variation_phase'])

    wave_cycles.wave_type = 'BANDS'
    wave_cycles.wave_profile = 'SIN'
    wave_cycles.bands_direction = 'X'
    wave_cycles.rings_direction = 'X'
    set_socket(wave_cycles, 'Vector', (0.0, 0.0, 0.0))
    set_socket(wave_cycles, 'Scale', params['melanin_wave_scale'])
    set_socket(wave_cycles, 'Distortion', 0.0)
    set_socket(wave_cycles, 'Detail', params['melanin_wave_detail'])
    set_socket(wave_cycles, 'Detail Scale', params['melanin_wave_detail_scale'])
    set_socket(wave_cycles, 'Detail Roughness', params['melanin_wave_detail_roughness'])
    set_socket(wave_cycles, 'Phase Offset', params['melanin_wave_phase'])

    # Color processing ------------------------------------------------------
    hsv.inputs['Hue'].default_value = float(params['color_hue_shift'])
    hsv.inputs['Saturation'].default_value = float(params['color_saturation'])
    hsv.inputs['Value'].default_value = float(params['color_value'])
    hsv.inputs['Fac'].default_value = 1.0

    color_switch.blend_type = 'MIX'
    color_switch.data_type = 'RGBA'
    color_switch.factor_mode = 'UNIFORM'
    color_switch.clamp_factor = True
    color_switch.clamp_result = False
    color_switch.inputs['Factor'].default_value = _clamp01(params['color_switch_factor'])

    variation.blend_type = 'MULTIPLY'
    variation.data_type = 'RGBA'
    variation.factor_mode = 'UNIFORM'
    variation.clamp_factor = False
    variation.clamp_result = False
    variation.inputs['Factor'].default_value = _clamp01(params['eevee_variation_factor'])
    variation.inputs['B'].default_value = _ensure_color(
        params['variation_secondary_color'],
        defaults['variation_secondary_color'],
    )

    mix_spec.blend_type = 'MIX'
    mix_spec.data_type = 'RGBA'
    mix_spec.factor_mode = 'UNIFORM'
    mix_spec.clamp_factor = True
    mix_spec.clamp_result = False
    mix_spec.inputs['Factor'].default_value = _clamp01(params['eevee_specular_mix'])
    mix_spec.inputs['B'].default_value = (1.0, 1.0, 1.0, 1.0)

    mix_tint.blend_type = 'MIX'
    mix_tint.data_type = 'RGBA'
    mix_tint.factor_mode = 'UNIFORM'
    mix_tint.clamp_factor = True
    mix_tint.clamp_result = False
    mix_tint.inputs['Factor'].default_value = _clamp01(params['eevee_specular_tint_mix'])
    mix_tint.inputs['A'].default_value = (1.0, 1.0, 1.0, 1.0)

    # Math nodes ------------------------------------------------------------
    node_map['root_darkness_strength'].inputs[1].default_value = _clamp01(params['root_darkness_strength'])
    node_map['Melanin3'].inputs[1].default_value = float(params['material_melanin_amount'])
    node_map['Wave Strength 2'].inputs[1].default_value = float(params['melanin_wave_strength'])
    node_map['Math.005'].inputs[1].default_value = float(params['melanin_wave_bias'])
    node_map['Math.003'].inputs[1].default_value = max(0.0001, float(params['melanin_wave_divisor']))

    # BSDF nodes ------------------------------------------------------------
    cycles_bsdf.parametrization = 'MELANIN'
    set_socket(cycles_bsdf, 'Melanin Redness', params['bsdf_melanin_redness'])
    set_socket(cycles_bsdf, 'Roughness', params['bsdf_roughness'])
    set_socket(cycles_bsdf, 'Radial Roughness', params['bsdf_radial_roughness'])
    set_socket(cycles_bsdf, 'Random Roughness', params['bsdf_random_roughness'])
    set_socket(cycles_bsdf, 'Coat', params['bsdf_coat'])
    set_socket(cycles_bsdf, 'IOR', params['bsdf_ior'])
    set_socket(cycles_bsdf, 'Offset', params['bsdf_offset'])
    set_socket(cycles_bsdf, 'Random Color', 0.0)
    set_socket(cycles_bsdf, 'Random', 0.0)

    cycles_bsdf_secondary.parametrization = 'COLOR'
    set_socket(cycles_bsdf_secondary, 'Roughness', 0.65)
    set_socket(cycles_bsdf_secondary, 'Radial Roughness', 1.0)
    set_socket(cycles_bsdf_secondary, 'Random Roughness', params['bsdf_random_roughness'])
    set_socket(cycles_bsdf_secondary, 'Coat', 0.0)
    set_socket(cycles_bsdf_secondary, 'IOR', params['bsdf_ior'])
    set_socket(cycles_bsdf_secondary, 'Offset', params['bsdf_offset'])

    set_socket(eevee_bsdf, 'Metallic', params['viewport_metallic'])
    set_socket(eevee_bsdf, 'Roughness', params['viewport_roughness'])
    set_socket(eevee_bsdf, 'IOR', 1.55)
    set_socket(eevee_bsdf, 'Alpha', 1.0)
    set_socket(eevee_bsdf, 'Subsurface Weight', 0.0)
    set_socket(eevee_bsdf, 'Subsurface Radius', (1.0, 0.2, 0.1))
    set_socket(eevee_bsdf, 'Subsurface Scale', 0.05)
    set_socket(eevee_bsdf, 'Subsurface IOR', 1.01)
    set_socket(eevee_bsdf, 'Specular IOR Level', 0.59545457)
    set_socket(eevee_bsdf, 'Anisotropic', 1.0)
    set_socket(eevee_bsdf, 'Anisotropic Rotation', 1.0)
    set_socket(eevee_bsdf, 'Transmission Weight', 0.70998728)
    set_socket(eevee_bsdf, 'Coat Weight', 0.0)
    set_socket(eevee_bsdf, 'Coat Roughness', 0.0)
    set_socket(eevee_bsdf, 'Coat IOR', 1.5)
    set_socket(eevee_bsdf, 'Sheen Weight', 0.0)
    set_socket(eevee_bsdf, 'Sheen Roughness', 0.5)
    set_socket(eevee_bsdf, 'Emission Strength', 0.0)

    # Links -----------------------------------------------------------------
    link_specs = [
        ('Attribute.001', 'Vector', 'Mapping.001', 'Vector'),
        ('Attribute.002', 'Vector', 'Mapping.002', 'Vector'),
        ('Mapping.001', 'Vector', 'Separate XYZ', 'Vector'),
        ('Mapping.002', 'Vector', 'Separate XYZ.001', 'Vector'),
        ('Separate XYZ.001', 'Y', 'Wave Texture.001', 'Vector'),
        ('Wave Texture.001', 'Color', 'ColorRamp.003', 'Fac'),
        ('Reroute', 'Output', 'Eevee bsdf', 'Base Color'),
        ('Separate XYZ', 'X', 'Eevee Gradient', 'Fac'),
        ('Eevee bsdf', 'BSDF', 'Material Output', 'Surface'),
        ('ColorRamp.003', 'Color', 'Eevee Variation', 'Factor'),
        ('Reroute', 'Output', 'Eevee bsdf', 'Emission Color'),
        ('Attribute.005', 'Vector', 'Mapping.004', 'Vector'),
        ('Mapping.004', 'Vector', 'Separate XYZ.003', 'Vector'),
        ('Separate XYZ.003', 'Y', 'Wave Texture Cycles', 'Vector'),
        ('Attribute.004', 'Vector', 'color_switch_eevee', 'B'),
        ('Hue Saturation Value', 'Color', 'color_switch_eevee', 'A'),
        ('Eevee Gradient', 'Color', 'Hue Saturation Value', 'Color'),
        ('color_switch_eevee', 'Result', 'Eevee Variation', 'A'),
        ('Eevee Variation', 'Result', 'Reroute', 'Input'),
        ('Separate XYZ.004', 'X', 'Math.001', 'Value'),
        ('Separate XYZ.004', 'Y', 'Math.001', 'Value'),
        ('Math.001', 'Value', 'Math.002', 'Value'),
        ('Separate XYZ.004', 'Z', 'Math.002', 'Value'),
        ('Math.002', 'Value', 'Math.003', 'Value'),
        ('Math.003', 'Value', 'Math.005', 'Value'),
        ('Math.005', 'Value', 'Wave Strength 2', 'Value'),
        ('Wave Strength 2', 'Value', 'Melanin3', 'Value'),
        ('Wave Texture Cycles', 'Color', 'Separate XYZ.004', 'Vector'),
        ('Melanin3', 'Value', 'Cycles bsdf', 'Color'),
        ('Curves Info', 'Intercept', 'root_darkness_color_ramp', 'Fac'),
        ('Melanin3', 'Value', 'Math.004', 'Value'),
        ('Math.004', 'Value', 'Cycles bsdf', 'Melanin'),
        ('root_darkness_color_ramp', 'Color', 'Separate Color', 'Color'),
        ('Separate Color', 'Red', 'root_darkness_strength', 'Value'),
        ('root_darkness_strength', 'Value', 'Math.004', 'Value'),
        ('Reroute', 'Output', 'Mix', 'A'),
        ('Mix', 'Result', 'Mix.001', 'B'),
        ('Mix.001', 'Result', 'Eevee bsdf', 'Specular Tint'),
        ('Curves Info.001', 'Random', 'Color Ramp', 'Fac'),
        ('Color Ramp', 'Color', 'Cycles bsdf.001', 'Color'),
        ('Cycles bsdf.001', 'BSDF', 'Material Output.001', 'Surface'),
    ]
    for from_name, from_socket, to_name, to_socket in link_specs:
        links.new(
            node_map[from_name].outputs[from_socket],
            node_map[to_name].inputs[to_socket],
        )

    # Viewport preview ------------------------------------------------------
    viewport_color = _ensure_color(params['viewport_color'], defaults['viewport_color'])
    mat.diffuse_color = viewport_color
    mat.metallic = float(params['viewport_metallic'])
    mat.roughness = float(params['viewport_roughness'])

    return mat

def create_difflocks_realistic_body_material():
    """Creates a Principled-BSDF skin material matching your Blender panel settings."""
    # new material
    mat = bpy.data.materials.new(name="RealisticBodyMaterial")
    mat.use_nodes = True

    # grab nodes & links
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # clear default nodes
    for n in nodes:
        nodes.remove(n)

    # output node
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (300, 0)

    # principled BSDF
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    # link BSDF → surface
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # — Base layer
    bsdf.inputs["Base Color"].default_value       = (0.98, 0.93, 0.93, 1.0)
    bsdf.inputs["Metallic"].default_value         = 0.0
    bsdf.inputs["Roughness"].default_value        = 0.676
    bsdf.inputs["IOR"].default_value              = 1.45
    bsdf.inputs["Alpha"].default_value            = 1.0
    bsdf.inputs["Normal"].default_value           = (0.0, 0.0, 0.0)  # leave unconnected, using your Normal Map node

    # — Subsurface (Random Walk skin)
    bsdf.subsurface_method                         = 'RANDOM_WALK'
    # bsdf.inputs["Subsurface"].default_value        = 1.0
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.2, 0.1)
    bsdf.inputs["Subsurface Scale"].default_value  = 0.04
    bsdf.inputs["IOR"].default_value    = 1.50
    bsdf.inputs["Subsurface Anisotropy"].default_value = 0.0

    # — Specular (GGX)
    # bsdf.inputs["Specular"].default_value          = 0.5
    # bsdf.inputs["Specular Tint"].default_value     = (1.0, 1.0, 1.0)
    bsdf.inputs["Anisotropic"].default_value       = 0.0
    bsdf.inputs["Anisotropic Rotation"].default_value = 0.0
    # Tangent left at “Default”

    # — Transmission
    # bsdf.inputs["Transmission"].default_value      = 0.0

    # — Coat (“Coat”)
    # bsdf.inputs["Coat"].default_value         = 0.0
    # bsdf.inputs["chiang"]["Coat Roughness"].default_value = 0.03
    # Coat IOR and Tint exposed in newer Blender versions:
    bsdf.inputs["Coat IOR"].default_value     = 1.50
    # bsdf.inputs["Coat Tint"].default_value    = (1.0, 1.0, 1.0)

    # — Sheen
    # bsdf.inputs["Sheen"].default_value             = 0.0
    bsdf.inputs["Sheen Roughness"].default_value   = 0.50
    # bsdf.inputs["Sheen Tint"].default_value        = (1.0, 1.0, 1.0)

    # — Emission
    # bsdf.inputs["Emission"].default_value          = (0.0, 0.0, 0.0, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 1.0

    # — Thin Film
    # bsdf.inputs["Thin Film Thickness"].default_value = 0.0
    # bsdf.inputs["Thin Film IOR"].default_value      = 1.33

    return mat


def create_realistic_hair_only_material(metadata={}):
    """Creates optimized hair material with correct color handling"""
    mat = bpy.data.materials.new(name="OptimizedHairMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Enable essential rendering properties
    mat.blend_method = "HASHED"
    mat.shadow_method = "HASHED"
    
    # Create core nodes
    output = nodes.new("ShaderNodeOutputMaterial")
    hair_bsdf = nodes.new("ShaderNodeBsdfHairPrincipled")
    hair_info = nodes.new("ShaderNodeHairInfo")
    
    # Position nodes efficiently
    output.location = (400, 0)
    hair_bsdf.location = (200, 0)
    hair_info.location = (-200, 0)
    
    
    melanin = metadata.get('material_melanin_amount', 0.6)
    roughness = metadata.get('bsdf_roughness', 0.2)
    radial_roughness = metadata.get('bsdf_radial_roughness', 0.25)
    melanin_redness = metadata.get('bsdf_melanin_redness', 0.0)
    # Configure Principled Hair BSDF
    hair_bsdf.parametrization = 'MELANIN'
    hair_bsdf.inputs['Melanin'].default_value = melanin
    hair_bsdf.inputs['Melanin Redness'].default_value = melanin_redness
    hair_bsdf.inputs['Roughness'].default_value = roughness
    hair_bsdf.inputs['Radial Roughness'].default_value = radial_roughness
    hair_bsdf.inputs['Coat'].default_value = 0.7
    hair_bsdf.inputs['IOR'].default_value = 1.55
    
    # Create subtle melanin variation along the strand
    melanin_variation = nodes.new("ShaderNodeMath")
    melanin_variation.location = (-200, -200)
    melanin_variation.operation = 'MULTIPLY_ADD'
    melanin_variation.inputs[1].default_value = 0.08  # Variation amount
    melanin_variation.inputs[2].default_value = melanin * 0.98  # Base melanin
    
    # Create roughness variation
    roughness_variation = nodes.new("ShaderNodeMath")
    roughness_variation.location = (-200, -400)
    roughness_variation.operation = 'MULTIPLY_ADD'
    roughness_variation.inputs[1].default_value = 0.15
    roughness_variation.inputs[2].default_value = roughness * 0.85
    
    # Connect nodes
    links.new(hair_info.outputs['Intercept'], melanin_variation.inputs[0])
    links.new(melanin_variation.outputs[0], hair_bsdf.inputs['Melanin'])
    links.new(hair_info.outputs['Random'], roughness_variation.inputs[0])
    links.new(roughness_variation.outputs[0], hair_bsdf.inputs['Roughness'])
    links.new(hair_bsdf.outputs[0], output.inputs[0])
    
    return mat


def create_strand_material():
    """Creates material that uses vertex colors with emission shader for direction visualization"""
    mat = bpy.data.materials.new(name="HairStrandMaterial")
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    mat.shadow_method = "NONE"

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    vertex_color = nodes.new("ShaderNodeVertexColor")
    vertex_color.layer_name = "Col"

    links.new(vertex_color.outputs[0], emission.inputs[0])
    links.new(emission.outputs[0], output.inputs[0])

    return mat

def create_depth_material():
    """Creates material for depth rendering"""
    mat = bpy.data.materials.new(name="DepthMaterial")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    camera_data = nodes.new("ShaderNodeCameraData")
    gamma = nodes.new("ShaderNodeGamma")  # Add gamma correction
    map_range = nodes.new("ShaderNodeMapRange")
    emission = nodes.new("ShaderNodeEmission")

    map_range.inputs["From Min"].default_value = 0.0
    map_range.inputs["From Max"].default_value = 3.5
    map_range.inputs["To Min"].default_value = 1.0
    map_range.inputs["To Max"].default_value = 0.0

    gamma.inputs[1].default_value = 2.2  # Standard gamma correction

    links.new(camera_data.outputs["View Distance"], map_range.inputs["Value"])
    links.new(map_range.outputs[0], gamma.inputs[0])
    links.new(gamma.outputs[0], emission.inputs["Color"])
    links.new(emission.outputs[0], output.inputs[0])

    return mat


def setup_canny_compositor(low_threshold=0.4, high_threshold=0.8, blur_size=5):
    """
    Setup Blender's compositor for Canny edge detection post-processing
    
    Parameters:
    -----------
    low_threshold : float (0.0-1.0)
        Lower threshold for compositor edge detection (default: 0.4)
    high_threshold : float (0.0-1.0)
        Upper threshold for compositor edge detection (default: 0.8)
    blur_size : int
        Size of Gaussian blur for noise reduction, similar to aperture_size (default: 2)
    """
    # Enable compositor and use nodes
    bpy.context.scene.use_nodes = True
    
    tree = bpy.context.scene.node_tree
    tree.nodes.clear()
    
    # Create basic compositor nodes for edge detection
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    
    # Optional Gaussian blur for noise reduction (similar to cv.Canny preprocessing)
    if blur_size > 0:
        blur_node = tree.nodes.new('CompositorNodeBlur')
        blur_node.size_x = blur_size
        blur_node.size_y = blur_size
        blur_node.filter_type = 'GAUSS'
    
    # Edge detection using filter node (Sobel operator - similar to cv.Canny gradient computation)
    edge_filter = tree.nodes.new('CompositorNodeFilter')
    edge_filter.filter_type = 'SOBEL'
    
    # Configurable threshold for edge detection (similar to cv.Canny minVal/maxVal)
    color_ramp = tree.nodes.new('CompositorNodeValToRGB')
    color_ramp.color_ramp.elements[0].position = low_threshold   # minVal equivalent
    color_ramp.color_ramp.elements[1].position = high_threshold  # maxVal equivalent
    color_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)  # Black (no edge)
    color_ramp.color_ramp.elements[1].color = (1, 1, 1, 1)  # White (edge)
    
    # Final output
    composite = tree.nodes.new('CompositorNodeComposite')
    
    # Position nodes
    if blur_size > 0:
        render_layers.location = (-700, 0)
        blur_node.location = (-500, 0)
        edge_filter.location = (-300, 0)
        color_ramp.location = (0, 0)
        composite.location = (300, 0)
    else:
        render_layers.location = (-600, 0)
        edge_filter.location = (-300, 0)
        color_ramp.location = (0, 0)
        composite.location = (300, 0)
    
    # Connect nodes with configurable edge detection pipeline
    if blur_size > 0:
        tree.links.new(render_layers.outputs[0], blur_node.inputs[0])
        tree.links.new(blur_node.outputs[0], edge_filter.inputs[0])
    else:
        tree.links.new(render_layers.outputs[0], edge_filter.inputs[0])
    
    tree.links.new(edge_filter.outputs[0], color_ramp.inputs[0])
    tree.links.new(color_ramp.outputs[0], composite.inputs[0])
    
    # Store parameters as scene custom properties for reference
    bpy.context.scene["canny_compositor_low_threshold"] = low_threshold
    bpy.context.scene["canny_compositor_high_threshold"] = high_threshold
    bpy.context.scene["canny_compositor_blur_size"] = blur_size


def create_canny_material(low_threshold=0.2, high_threshold=0.4, edge_power=2.0, ior=1.55):
    """
    Creates material for Canny edge detection rendering using Fresnel-based edge detection
    
    Parameters:
    -----------
    low_threshold : float (0.0-1.0)
        Lower threshold for edge detection, similar to cv.Canny minVal (default: 0.3)
    high_threshold : float (0.0-1.0)
        Upper threshold for edge detection, similar to cv.Canny maxVal (default: 0.8)
    edge_power : float
        Power function exponent to enhance edge contrast (default: 2.0)
    ior : float
        Index of refraction for Fresnel effect, affects edge sensitivity (default: 1.5)
    """
    mat = bpy.data.materials.new(name="CannyMaterial")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create nodes for edge detection
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    
    # Method: Fresnel-based edge detection (most reliable)
    fresnel = nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = ior
    
    # Edge enhancement using power function
    edge_power_node = nodes.new("ShaderNodeMath")
    edge_power_node.operation = 'POWER'
    edge_power_node.inputs[1].default_value = edge_power
    
    # Configurable threshold for clean edges (similar to cv.Canny thresholds)
    threshold = nodes.new("ShaderNodeValToRGB")
    threshold.color_ramp.elements[0].position = low_threshold   # Lower threshold (minVal)
    threshold.color_ramp.elements[1].position = high_threshold  # Upper threshold (maxVal)
    threshold.color_ramp.elements[0].color = (0, 0, 0, 1)  # Non-edge (black)
    threshold.color_ramp.elements[1].color = (1, 1, 1, 1)  # Edge (white)
    
    # Invert for proper edge representation (edges should be white)
    invert = nodes.new("ShaderNodeInvert")
    invert.inputs["Fac"].default_value = 1.0

    # Position nodes
    output.location = (600, 0)
    emission.location = (400, 0)
    invert.location = (200, 0)
    threshold.location = (0, 0)
    edge_power_node.location = (-200, 0)
    fresnel.location = (-400, 0)

    # Connect the simplified node network
    links.new(fresnel.outputs["Fac"], edge_power_node.inputs[0])
    links.new(edge_power_node.outputs["Value"], threshold.inputs["Fac"])
    links.new(threshold.outputs["Color"], invert.inputs["Color"])
    links.new(invert.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    # Store parameters as custom properties for debugging/reference
    mat["canny_low_threshold"] = low_threshold
    mat["canny_high_threshold"] = high_threshold
    mat["canny_edge_power"] = edge_power
    mat["canny_ior"] = ior

    return mat


def create_segmentation_materials():
    """Creates materials for segmentation with distinct colors"""
    # Hair segmentation material (red)
    hair_mat = create_flat_material((1, 0, 0))
    
    # Body segmentation material (yellow) - for body
    body_mat = create_flat_material((1, 1, 0))

    return hair_mat, body_mat

def create_body_material():
    """Creates grey material for body"""
    return create_flat_material((0.5, 0.5, 0.5))


def enable_gpus():
    """Activate GPU devices for Cycles (OPTIX or CUDA)."""
    prefs = bpy.context.preferences
    cycles_addon = prefs.addons.get('cycles')
    if not cycles_addon:
        return []
    cycles_prefs = cycles_addon.preferences

    # Choose best device type
    for dev_type in ('OPTIX', 'CUDA'):
        if dev_type in cycles_prefs.get_compute_device_type():
            cycles_prefs.compute_device_type = dev_type
            break
    cycles_prefs.refresh_devices()

    activated = []
    for dev in cycles_prefs.devices:
        if dev.type == 'GPU':
            dev.use = True
            activated.append(dev.name)

    scene = bpy.context.scene
    scene.cycles.device = 'GPU'
    scene.cycles.use_persistent_data = True
    return activated

def setup_render_settings(resolution=512, render_type="direction", samples=256, min_adaptive_samples=64, 
                         canny_low_threshold=0.3, canny_high_threshold=0.8, canny_edge_power=2.0, 
                         canny_ior=1.5, canny_blur_size=2):
    """
    Setup render settings based on render type
    
    Parameters:
    -----------
    resolution : int
        Render resolution
    render_type : str
        Type of render ("direction", "depth", "segmentation", "canny", etc.)
    samples : int
        Number of render samples
    min_adaptive_samples : int
        Minimum adaptive samples
    canny_low_threshold : float (0.0-1.0)
        Lower threshold for Canny edge detection, similar to cv.Canny minVal
    canny_high_threshold : float (0.0-1.0) 
        Upper threshold for Canny edge detection, similar to cv.Canny maxVal
    canny_edge_power : float
        Power function exponent for edge enhancement
    canny_ior : float
        Index of refraction for Fresnel edge detection
    canny_blur_size : int
        Gaussian blur size for noise reduction (0 to disable)
    """
    scene = bpy.context.scene
    prefs = bpy.context.preferences.addons['cycles'].preferences
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    if render_type == "segmentation":
        bpy.context.scene.render.image_settings.file_format = 'WEBP'
        bpy.context.scene.render.image_settings.quality = 100
    else:
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_depth = "16"
    bpy.context.scene.render.image_settings.compression = 0

    prefs.compute_device_type = 'OPTIX'
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True

    if bpy.context.scene.world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    bpy.context.scene.world.use_nodes = True
    background = bpy.context.scene.world.node_tree.nodes["Background"]

    if render_type == "direction":
        background.inputs[0].default_value = (0, 0, 0, 0)
    elif render_type == "segmentation":
        background.inputs[0].default_value = (0.5, 0, 0.5, 1)  # Purple background
    elif render_type == "canny":
        background.inputs[0].default_value = (0, 0, 0, 1)  # Black background for edge detection
        # Setup compositor with configurable parameters
        setup_canny_compositor(
            low_threshold=canny_low_threshold,
            high_threshold=canny_high_threshold, 
            blur_size=canny_blur_size
        )
    else:  # depth
        background.inputs[0].default_value = (0, 0, 0, 1)

    # Enable CPU threading
    bpy.context.scene.render.threads_mode = "AUTO"
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.adaptive_threshold = 0.001
    bpy.context.scene.cycles.adaptive_max_samples = min_adaptive_samples
    bpy.context.scene.cycles.adaptive_min_samples = min_adaptive_samples // 2
    # Simplified render settings for clean output
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = "OPTIX"
    bpy.context.scene.cycles.max_bounces = 0
    bpy.context.scene.cycles.diffuse_bounces = 0
    bpy.context.scene.cycles.glossy_bounces = 0
    bpy.context.scene.cycles.transmission_bounces = 0
    bpy.context.scene.cycles.volume_bounces = 0

    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.look = "None"
    bpy.context.scene.view_settings.exposure = 0
    bpy.context.scene.view_settings.gamma = 1

    # Threading and memory
    bpy.context.scene.render.use_persistent_data = True

def create_realistic_hair_material():
    """Creates material for realistic hair rendering with natural variation"""
    mat = bpy.data.materials.new(name="RealisticHairMaterial")
    mat.use_nodes = True
    mat.blend_method = "HASHED"  # Better transparency handling

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create nodes
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    color_ramp = nodes.new("ShaderNodeValToRGB")
    noise = nodes.new("ShaderNodeTexNoise")
    mapping = nodes.new("ShaderNodeMapping")
    tex_coord = nodes.new("ShaderNodeTexCoord")

    # Setup noise for color variation
    noise.inputs["Scale"].default_value = 2.0
    noise.inputs["Detail"].default_value = 2.0

    # Setup color ramp for natural hair color variation
    color_ramp.color_ramp.elements[0].position = 0.3
    color_ramp.color_ramp.elements[0].color = (0.25, 0.15, 0.05, 1.0)  # Darker brown
    color_ramp.color_ramp.elements[1].position = 0.7
    color_ramp.color_ramp.elements[1].color = (0.4, 0.25, 0.15, 1.0)  # Lighter brown

    # Hair shader settings
    principled.inputs["Metallic"].default_value = 0.2
    principled.inputs["Specular IOR Level"].default_value = 0.4
    principled.inputs["Roughness"].default_value = 0.6
    principled.inputs["Alpha"].default_value = 0.95

    # Connect nodes
    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], principled.inputs["Base Color"])
    links.new(principled.outputs[0], output.inputs[0])

    return mat

def setup_realistic_render_settings(resolution=1024, samples=1024, min_adaptive_samples=128, transparent_background=True):
    """Setup render settings for improved realistic rendering"""
    prefs = bpy.context.preferences.addons['cycles'].preferences

    prefs.compute_device_type = 'OPTIX'
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.film_transparent = transparent_background  # Enable transparent background
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    # bpy.context.scene.render.image_settings.quality = 100
    # Better lighting settings
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.adaptive_threshold = 0.001
    bpy.context.scene.cycles.adaptive_min_samples = min_adaptive_samples
    bpy.context.scene.cycles.adaptive_max_samples = 2 * min_adaptive_samples  # Ensure max samples are at least double min samples
    bpy.context.scene.cycles.samples = samples  # Set high sample count for quality
    bpy.context.scene.cycles.use_fast_gi = True  # Enable fast GI approximation if available
    bpy.context.scene.render.use_persistent_data = True
    
    
    bpy.context.scene.cycles.use_curves             = True
    # bpy.context.scene.cycles.curves_subdivisions    = 5     # per-strand shading accuracy
    bpy.context.scene.cycles.curves_use_camera_cull = True  # cull strands outside FOV
    bpy.context.scene.cycles.curves_use_backfacing_cull = True
    # Improved quality settings
    bpy.context.scene.cycles.max_bounces = 12
    bpy.context.scene.cycles.diffuse_bounces = 4
    bpy.context.scene.cycles.glossy_bounces = 4
    bpy.context.scene.cycles.transmission_bounces = 12
    bpy.context.scene.cycles.volume_bounces = 4
    bpy.context.scene.cycles.transparent_max_bounces = 8

    bpy.context.scene.render.threads_mode = "AUTO"
    # Better denoising
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = "OPTIX"
    bpy.context.scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"

    # # Improved color management
    bpy.context.scene.view_settings.view_transform = "Filmic"
    # bpy.context.scene.view_settings.look = "High Contrast"
    bpy.context.scene.view_settings.exposure = 0.0

def create_three_point_lighting():
    """Create an improved three-point lighting setup"""
    # Key light (main light)
    bpy.ops.object.light_add(type="AREA", location=(2, -2, 3))
    key_light = bpy.context.active_object
    key_light.data.energy = 300
    key_light.data.size = 2
    key_light.data.color = (1, 0.95, 0.9)

    # Fill light (softer, secondary light)
    bpy.ops.object.light_add(type="AREA", location=(-2, -1, 2))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 150
    fill_light.data.size = 3
    fill_light.data.color = (0.9, 0.95, 1)

    # Back light (rim lighting)
    bpy.ops.object.light_add(type="AREA", location=(0, 2, 2.5))
    back_light = bpy.context.active_object
    back_light.data.energy = 200
    back_light.data.size = 2
    back_light.data.color = (1, 1, 1)

    # Add ambient light
    bpy.ops.object.light_add(type="AREA", location=(0, 0, 4))
    ambient_light = bpy.context.active_object
    ambient_light.data.energy = 50
    ambient_light.data.size = 5
    ambient_light.data.color = (0.8, 0.9, 1)

    return [key_light, fill_light, back_light, ambient_light]
