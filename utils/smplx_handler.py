import os
import os.path as osp
import pickle

import numpy as np
import torch
import trimesh
import smplx

from config.types import SMPLXVersions, SMPLXUVTypes


_BODY21 = (
    'L_Hip', 'R_Hip', 'Spine_1', 'L_Knee', 'R_Knee', 'Spine_2', 'L_Ankle', 'R_Ankle', 'Spine_3',
    'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist'
)
_IDX = {n: i for i, n in enumerate(_BODY21)}
_UPPER_LIMB = ('L_Collar', 'R_Collar', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist')


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(arr, device, shape=None):
    if arr is None:
        return None
    t = torch.from_numpy(np.array(arr, dtype=np.float32)).to(device)
    if shape is not None:
        t = t.view(*shape)
    return t


def _zeros_body_pose():
    return np.zeros((21, 3), dtype=np.float32)


def _apose_body_pose(angle_deg=35.0, collar_frac=0.35):
    bp = _zeros_body_pose()
    theta = np.deg2rad(float(angle_deg))
    
    bp[_IDX['L_Shoulder']] = np.array([0.0, 0.0, +theta], dtype=np.float32)
    bp[_IDX['R_Shoulder']] = np.array([0.0, 0.0, -theta], dtype=np.float32)
    
    if collar_frac and collar_frac > 0.0:
        ctheta = theta * float(collar_frac)
        bp[_IDX['L_Collar']] = np.array([0.0, 0.0, +ctheta], dtype=np.float32)
        bp[_IDX['R_Collar']] = np.array([0.0, 0.0, -ctheta], dtype=np.float32)
    
    return bp


def _mask_upper_limb(body_pose_63, keep_names=_UPPER_LIMB):
    bp = np.asarray(body_pose_63, dtype=np.float32).reshape(21, 3)
    masked = np.zeros_like(bp)
    for name in keep_names:
        masked[_IDX[name]] = bp[_IDX[name]]
    return masked.reshape(-1)


def _compute_smooth_vertex_normals(mesh):
    face_normals = mesh.face_normals
    vertex_normals = np.zeros((len(mesh.vertices), 3))
    vertex_face_count = np.zeros(len(mesh.vertices))
    
    for face_idx, face in enumerate(mesh.faces):
        face_normal = face_normals[face_idx]
        for vertex_idx in face:
            vertex_normals[vertex_idx] += face_normal
            vertex_face_count[vertex_idx] += 1
    
    vertex_face_count[vertex_face_count == 0] = 1
    vertex_normals = vertex_normals / vertex_face_count[:, np.newaxis]
    
    norms = np.linalg.norm(vertex_normals, axis=1)
    norms[norms == 0] = 1
    vertex_normals = vertex_normals / norms[:, np.newaxis]
    
    return vertex_normals


def apply_shade_smooth_to_mesh(trimesh_obj):
    if not isinstance(trimesh_obj, trimesh.Trimesh):
        raise ValueError("Input must be a trimesh.Trimesh object")
    
    smooth_mesh = trimesh_obj.copy()
    
    if len(smooth_mesh.faces) == 0:
        print("Warning: Mesh has no faces, cannot apply smooth shading")
        return smooth_mesh
    
    smooth_mesh.fix_normals()
    smooth_mesh.vertex_normals = _compute_smooth_vertex_normals(smooth_mesh)
    smooth_mesh.visual = smooth_mesh.visual.copy()
    
    return smooth_mesh


def _transform_coordinates(vertices, normals=None):
    transformed_vertices = np.column_stack([
        vertices[:, 0],
        vertices[:, 2],
        -vertices[:, 1]
    ])
    
    transformed_normals = None
    if normals is not None:
        transformed_normals = np.column_stack([
            normals[:, 0],
            normals[:, 2],
            -normals[:, 1]
        ])
    
    return transformed_vertices, transformed_normals


def convert_trimesh_to_blender(trimesh_obj):
    if not isinstance(trimesh_obj, trimesh.Trimesh):
        raise ValueError("Input must be a trimesh.Trimesh object")
    
    blender_mesh = trimesh_obj.copy()
    
    normals = blender_mesh.vertex_normals if hasattr(blender_mesh, 'vertex_normals') else None
    transformed_vertices, transformed_normals = _transform_coordinates(blender_mesh.vertices, normals)
    
    blender_mesh.vertices = transformed_vertices
    if transformed_normals is not None:
        blender_mesh.vertex_normals = transformed_normals
    
    return blender_mesh


def _build_uv_mapping(ft, f_geom, n_uv):
    uv2base = np.full(n_uv, -1, dtype=np.int32)
    for face_uv, face_geom in zip(ft, f_geom):
        for uv_idx, base_idx in zip(face_uv, face_geom):
            if uv2base[uv_idx] == -1:
                uv2base[uv_idx] = base_idx
    return uv2base


def _apply_npz_uv_mapping(mesh, npz_path, vertices, blender_format, verbose):
    npz_data = np.load(npz_path)
    vt = npz_data['vt']
    ft = npz_data['ft']
    f_geom = npz_data['f']
    
    uv2base = _build_uv_mapping(ft, f_geom, len(vt))
    expanded_vertices = vertices[uv2base]
    
    mesh = trimesh.Trimesh(vertices=expanded_vertices, faces=ft, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=vt)
    
    if blender_format:
        mesh = convert_trimesh_to_blender(mesh)
    
    if verbose:
        print(f"✓ Applied UV mapping from npz (v1_1/lh built-in UVs)")
        print(f"  UV vertices: {len(vt)}, Base vertices: {len(vertices)}")
    
    return mesh, mesh.vertices, mesh.faces


def _apply_external_uv_mapping(mesh, uv, vertices, blender_format, verbose):
    uv_mesh_path = f"assets/body_models/textures/smplx/uv_maps/{uv}/smplx_uv.ply"
    vertex_map_path = f"assets/body_models/textures/smplx/uv_maps/{uv}/vertex_map.npy"
    
    if not (osp.exists(uv_mesh_path) and osp.exists(vertex_map_path)):
        if verbose:
            print(f"⚠ UV mapping files not found for {uv}, using original mesh")
        return mesh, vertices, mesh.faces
    
    smplx_with_uv = trimesh.load(uv_mesh_path)
    ext2base = np.load(vertex_map_path)
    expanded_vertices = vertices[ext2base]
    smplx_with_uv.vertices = expanded_vertices
    mesh = smplx_with_uv.copy()
    
    if blender_format:
        mesh = convert_trimesh_to_blender(mesh)
    
    if verbose:
        print(f"✓ Applied UV mapping from {uv}")
    
    return mesh, mesh.vertices, mesh.faces


def _apply_uv_mapping(mesh, uv, version, npz_path, vertices, blender_format, verbose):
    if uv is None:
        if verbose:
            print("○ UV mapping skipped (uv=None)")
        return mesh, vertices, mesh.faces
    
    try:
        if False and version in [SMPLXVersions.V1_1.value]:
            return _apply_npz_uv_mapping(mesh, npz_path, vertices, blender_format, verbose)
        else:
            return _apply_external_uv_mapping(mesh, uv, vertices, blender_format, verbose)
    except Exception as e:
        if verbose:
            print(f"⚠ Failed to apply UV mapping: {e}, using original mesh")
        return mesh, vertices, mesh.faces


def _save_mesh(mesh, output_path, verbose):
    if mesh is None or output_path is None:
        return
    
    try:
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        mesh.export(output_path)
        if verbose:
            print(f"✓ Saved mesh to: {output_path}")
    except Exception as e:
        if verbose:
            print(f"✗ Failed to save OBJ: {e}")


def _create_smplx_model(npz_path, gender, num_betas, num_expressions, use_pca=False, num_pca_comps=None):
    device = _get_device()
    return smplx.create(
        model_path=npz_path,
        model_type="smplx",
        gender=gender,
        use_pca=use_pca,
        num_pca_comps=num_pca_comps,
        num_betas=num_betas,
        num_expression_coeffs=num_expressions,
        batch_size=1
    ).to(device).eval()


def _extract_mesh_data(output, smplx_model):
    vertices = output.vertices[0].detach().cpu().numpy()
    faces = smplx_model.faces.astype(np.int32) if hasattr(smplx_model, "faces") else None
    joints = output.joints[0].detach().cpu().numpy()
    return vertices, faces, joints


def _print_mesh_info(verbose, vertices, faces, joints):
    if not verbose:
        return
    print(f"\nGenerated SMPL-X mesh:")
    print(f"  Vertices: {vertices.shape}")
    print(f"  Faces: {faces.shape if faces is not None else 'None'}")
    print(f"  Joints: {joints.shape}")


def _create_mesh_with_processing(vertices, faces, uv, version, npz_path, blender_format, verbose):
    if faces is None:
        return None
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh, vertices, faces = _apply_uv_mapping(mesh, uv, version, npz_path, vertices, blender_format, verbose)
    mesh = apply_shade_smooth_to_mesh(mesh)
    
    return mesh

def create_default_flame(
    model_path = "assets/body_models/base_models/",
    gender = "neutral",
    num_betas = 10,
    num_expressions = 10,
    output_path = None,
    save_obj = False,
    verbose = False,
    return_mesh = False,
    blender_format = False
):
    """
    Create a default FLAME mesh with canonical parameters (neutral expression, zero pose).
    
    Args:
        model_path: Path to FLAME model directory
        gender: Gender for the model ('neutral', 'male', 'female')
        num_betas: Number of shape parameters
        num_expressions: Number of expression parameters
        output_path: Path to save the output mesh (if save_obj=True)
        save_obj: Whether to save the mesh to a file
        verbose: Whether to print detailed information
        return_mesh: Whether to return the trimesh object
        blender_format: Whether to convert coordinates to Blender format
    
    Returns:
        Dictionary containing vertices, faces, joints, flame_output, parameters_used, and optionally mesh
    """
    device = _get_device()
    
    if verbose:
        print("Creating default FLAME mesh:")
        print(f"  Gender: {gender}")
        print(f"  Shape parameters: {num_betas}")
        print(f"  Expression parameters: {num_expressions}")
    
    # Create FLAME model
    flame_model = smplx.create(
        model_path=model_path,
        model_type="flame",
        gender=gender,
        num_betas=num_betas,
        num_expression_coeffs=num_expressions,
        batch_size=1
    ).to(device).eval()
    
    # Default parameters (all zeros - canonical pose and neutral expression)
    default_params = {
        "global_orient": torch.zeros((1, 3), device=device),
        "neck_pose": torch.zeros((1, 3), device=device),
        "jaw_pose": torch.zeros((1, 3), device=device),
        "leye_pose": torch.zeros((1, 3), device=device),
        "reye_pose": torch.zeros((1, 3), device=device),
        "betas": torch.zeros((1, num_betas), device=device),
        "expression": torch.zeros((1, num_expressions), device=device),
        "transl": torch.zeros((1, 3), device=device),
    }
    
    # Generate FLAME mesh
    with torch.no_grad():
        output = flame_model(**default_params, return_verts=True)
    
    vertices = output.vertices[0].detach().cpu().numpy()
    faces = flame_model.faces.astype(np.int32) if hasattr(flame_model, "faces") else None
    joints = output.joints[0].detach().cpu().numpy() if hasattr(output, "joints") else None
    
    if verbose:
        print(f"\nGenerated FLAME mesh:")
        print(f"  Vertices: {vertices.shape}")
        print(f"  Faces: {faces.shape if faces is not None else 'None'}")
        if joints is not None:
            print(f"  Joints: {joints.shape}")
    
    # Create trimesh object
    mesh = None
    if faces is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh = apply_shade_smooth_to_mesh(mesh)
        
        if blender_format:
            mesh = convert_trimesh_to_blender(mesh)
    
    # Save mesh if requested
    if save_obj:
        _save_mesh(mesh, output_path, verbose)
    
    result = {
        "vertices": vertices,
        "faces": faces,
        "joints": joints,
        "flame_output": output,
        "parameters_used": default_params
    }
    
    if return_mesh:
        result["mesh"] = mesh
    
    return result

def create_default_smplx(
    model_path = "assets/body_models/base_models/smplx/parametric_models/",
    version = SMPLXVersions.LOCKED_HEAD.value,
    gender = "neutral",
    num_betas = 10,
    num_expressions = 10,
    output_path = None,
    save_obj = False,
    verbose = False,
    return_mesh = False,
    blender_format = False,
    uv = SMPLXUVTypes.UV_2021.value,
    arm_pose = "A",
    a_angle_deg = -75.0,
    a_collar_frac = -0.5
):
    device = _get_device()
    npz_path = os.path.join(model_path, version, f"SMPLX_{gender.upper()}.npz")
    
    smplx_model = _create_smplx_model(npz_path, gender, num_betas, num_expressions)
    
    body_pose_21x3 = _apose_body_pose(angle_deg=a_angle_deg, collar_frac=a_collar_frac) if arm_pose.upper() == "A" else _zeros_body_pose()
    body_pose = torch.from_numpy(body_pose_21x3.reshape(1, -1)).to(device)
    
    default_params = {
        "global_orient": torch.zeros((1, 3), device=device),
        "body_pose": body_pose,
        "left_hand_pose": torch.zeros((1, 15 * 3), device=device),
        "right_hand_pose": torch.zeros((1, 15 * 3), device=device),
        "jaw_pose": torch.zeros((1, 3), device=device),
        "leye_pose": torch.zeros((1, 3), device=device),
        "reye_pose": torch.zeros((1, 3), device=device),
        "betas": torch.zeros((1, num_betas), device=device),
        "expression": torch.zeros((1, num_expressions), device=device),
        "transl": torch.zeros((1, 3), device=device),
    }
    
    if verbose:
        print("Creating default SMPL-X mesh:")
        print(f"  Gender: {gender}")
        print(f"  Shape parameters: {num_betas}")
        print(f"  Expression parameters: {num_expressions}")
        print(f"  Arm pose: {'A-pose' if arm_pose.upper()=='A' else 'T-pose'} (angle={a_angle_deg}°, collar_frac={a_collar_frac})")
    
    with torch.no_grad():
        output = smplx_model(**default_params, return_verts=True)
    
    vertices, faces, joints = _extract_mesh_data(output, smplx_model)
    _print_mesh_info(verbose, vertices, faces, joints)
    
    mesh = _create_mesh_with_processing(vertices, faces, uv, version, npz_path, blender_format, verbose)
    
    if save_obj:
        _save_mesh(mesh, output_path, verbose)
    
    result = {
        "vertices": vertices,
        "faces": faces,
        "joints": joints,
        "smplx_output": output,
        "parameters_used": default_params
    }
    if return_mesh:
        result["mesh"] = mesh
    return result


def _extract_param(params_dict, *keys):
    for key in keys:
        if key in params_dict:
            return params_dict[key]
    return None


def _infer_hand_mode_and_dim(arr):
    if isinstance(arr, (list, np.ndarray)):
        L = int(len(arr))
        if L == 45:
            return "axis-angle", 45
        elif L > 0:
            return "pca", L
    return "unknown", 0


def _resolve_hand_representation(lhand_raw, rhand_raw):
    left_mode, left_dim = _infer_hand_mode_and_dim(lhand_raw)
    right_mode, right_dim = _infer_hand_mode_and_dim(rhand_raw)
    
    if left_mode == "axis-angle" or right_mode == "axis-angle":
        hand_mode, hand_dim = "axis-angle", 45
    elif left_mode == "pca" or right_mode == "pca":
        hand_dim = max(left_dim, right_dim, 12)
        hand_mode = "pca"
    else:
        hand_mode, hand_dim = "axis-angle", 45
    
    use_pca_model = (hand_mode == "pca")
    return hand_mode, hand_dim, use_pca_model


def _prepare_hand_pose_axis_angle(hand_raw, hand_dim, device):
    hp = np.array(hand_raw, dtype=np.float32) if isinstance(hand_raw, (list, np.ndarray)) else np.zeros(hand_dim, np.float32)
    if hp.size != hand_dim:
        hp = np.pad(hp, (0, max(0, hand_dim - hp.size)), mode='constant')[:hand_dim]
    return torch.from_numpy(hp[None, ...]).to(device)


def _prepare_hand_pose_pca(hand_raw, hand_dim, device):
    arr = list(hand_raw) if isinstance(hand_raw, (list, np.ndarray)) else []
    if len(arr) >= hand_dim:
        arr = arr[:hand_dim]
    else:
        arr = arr + [0.0] * (hand_dim - len(arr))
    hp = np.asarray(arr, dtype=np.float32)
    return torch.from_numpy(hp[None, ...]).to(device)


def _build_hand_poses(use_hand_poses, hand_mode, hand_dim, lhand_raw, rhand_raw, use_pca_model, device, verbose):
    if use_hand_poses:
        if hand_mode == "axis-angle":
            left_hand = _prepare_hand_pose_axis_angle(lhand_raw, 45, device)
            right_hand = _prepare_hand_pose_axis_angle(rhand_raw, 45, device)
        else:
            left_hand = _prepare_hand_pose_pca(lhand_raw, hand_dim, device)
            right_hand = _prepare_hand_pose_pca(rhand_raw, hand_dim, device)
        if verbose:
            print(f"✓ Using fitted hand poses ({'PCA' if use_pca_model else 'axis-angle'})")
    else:
        dim = hand_dim if use_pca_model else 45
        left_hand = torch.zeros((1, dim), device=device)
        right_hand = torch.zeros((1, dim), device=device)
        if verbose:
            print("○ Using canonical hand poses (neutral)")
    
    return left_hand, right_hand


def _build_body_pose(use_body_pose, body_pose, use_hand_poses, preserve_upper_limb_if_hands, device, verbose):
    if use_body_pose and body_pose is not None:
        if verbose:
            print("✓ Using fitted body pose")
        return body_pose
    
    body_zero = torch.zeros((1, 21 * 3), device=device)
    if use_hand_poses and preserve_upper_limb_if_hands and body_pose is not None:
        masked = _mask_upper_limb(body_pose.cpu().numpy().reshape(-1))
        body_zero = torch.from_numpy(masked).to(device).view(1, -1)
        if verbose:
            print("○ Using canonical body pose except upper-limb copied from fitted")
    else:
        if verbose:
            print("○ Using canonical body pose (T/A-pose)")
    
    return body_zero


def _build_shape_param(use_shape, betas, default_betas, num_betas, device, verbose):
    if use_shape and betas is not None:
        if verbose:
            print("✓ Using fitted body shape")
        return betas
    
    if isinstance(default_betas, (list, np.ndarray, torch.Tensor)):
        if verbose:
            print("○ Using provided default body shape")
        return _to_tensor(default_betas, device, (1, -1))
    
    if verbose:
        print("○ Using canonical body shape (average)")
    return torch.zeros((1, num_betas), device=device)


def _build_expression_param(use_expressions, expression, default_expression, num_expressions, device, verbose):
    if use_expressions and expression is not None:
        if verbose:
            print("✓ Using fitted facial expressions")
        return expression
    
    if isinstance(default_expression, (list, np.ndarray, torch.Tensor)):
        if verbose:
            print("○ Using provided default expressions")
        return _to_tensor(default_expression, device, (1, -1))
    
    if verbose:
        print("○ Using canonical expressions (neutral)")
    return torch.zeros((1, num_expressions), device=device)


def _build_selective_params(
    use_global_orient, root_pose,
    use_body_pose, body_pose, use_hand_poses, preserve_upper_limb_if_hands,
    hand_mode, hand_dim, lhand_raw, rhand_raw, use_pca_model,
    use_jaw_pose, jaw_pose,
    use_shape, betas, default_betas, num_betas,
    use_expressions, expression, default_expression, num_expressions,
    use_translation, transl,
    use_eye_poses, leye_pose, reye_pose,
    device, verbose
):
    params = {}
    
    params["global_orient"] = root_pose if (use_global_orient and root_pose is not None) else torch.zeros((1, 3), device=device)
    if verbose:
        print("✓ Using fitted global orientation" if use_global_orient and root_pose is not None else "○ Using canonical global orientation (identity)")
    
    params["body_pose"] = _build_body_pose(use_body_pose, body_pose, use_hand_poses, preserve_upper_limb_if_hands, device, verbose)
    
    left_hand, right_hand = _build_hand_poses(use_hand_poses, hand_mode, hand_dim, lhand_raw, rhand_raw, use_pca_model, device, verbose)
    params["left_hand_pose"] = left_hand
    params["right_hand_pose"] = right_hand
    
    params["jaw_pose"] = jaw_pose if use_jaw_pose and jaw_pose is not None else torch.zeros((1, 3), device=device)
    if verbose:
        print("✓ Using fitted jaw pose" if use_jaw_pose and jaw_pose is not None else "○ Using canonical jaw pose (closed)")
    
    params["betas"] = _build_shape_param(use_shape, betas, default_betas, num_betas, device, verbose)
    params["expression"] = _build_expression_param(use_expressions, expression, default_expression, num_expressions, device, verbose)
    
    params["transl"] = transl if use_translation and transl is not None else torch.zeros((1, 3), device=device)
    if verbose:
        print("✓ Using fitted translation" if use_translation and transl is not None else "○ Using canonical translation (origin)")
    
    if use_eye_poses:
        params["leye_pose"] = leye_pose if leye_pose is not None else torch.zeros((1, 3), device=device)
        params["reye_pose"] = reye_pose if reye_pose is not None else torch.zeros((1, 3), device=device)
        if verbose:
            print("✓ Using fitted eye poses")
    else:
        params["leye_pose"] = torch.zeros((1, 3), device=device)
        params["reye_pose"] = torch.zeros((1, 3), device=device)
        if verbose:
            print("○ Using canonical eye poses (neutral)")
    
    return params


def create_selective_smplx_transfer(
    smplx_params,
    model_path = "assets/body_models/base_models/smplx/parametric_models/",
    version = SMPLXVersions.LOCKED_HEAD.value,
    gender = "neutral",
    output_path = None,
    use_expressions = True,
    use_jaw_pose = True,
    use_eye_poses = True,
    use_hand_poses = False,
    use_body_pose = False,
    use_global_orient = False,
    use_shape = False,
    use_translation = False,
    preserve_upper_limb_if_hands = True,
    default_betas = None,
    default_expression = None,
    save_obj = False,
    verbose = False,
    return_mesh = True,
    blender_format = False,
    uv = SMPLXUVTypes.UV_2021.value,
):
    sp = _extract_param(smplx_params, "smplx_param", "smplx_params") or smplx_params
    device = _get_device()
    
    root_pose = _to_tensor(_extract_param(sp, "smplx_root_pose", "root_pose"), device, (1, 3))
    body_pose = _to_tensor(_extract_param(sp, "smplx_body_pose", "body_pose"), device, (1, 21 * 3))
    lhand_raw = _extract_param(sp, "smplx_lhand_pose", "lhand_pose")
    rhand_raw = _extract_param(sp, "smplx_rhand_pose", "rhand_pose")
    jaw_pose = _to_tensor(_extract_param(sp, "smplx_jaw_pose", "jaw_pose") or [0, 0, 0], device, (1, 3))
    leye_pose = _to_tensor(_extract_param(sp, "smplx_leye_pose", "leye_pose") or [0, 0, 0], device, (1, 3))
    reye_pose = _to_tensor(_extract_param(sp, "smplx_reye_pose", "reye_pose") or [0, 0, 0], device, (1, 3))
    betas_arr = _extract_param(sp, "smplx_shape", "shape")
    expr_arr = _extract_param(sp, "smplx_expr", "expression")
    betas = _to_tensor(betas_arr, device, (1, -1)) if betas_arr is not None else None
    expression = _to_tensor(expr_arr, device, (1, -1)) if expr_arr is not None else None
    transl = _to_tensor(_extract_param(sp, "cam_trans", "trans"), device, (1, 3)) if use_translation else None
    
    num_betas = betas.shape[1] if betas is not None else (default_betas.shape[-1] if default_betas is not None else 10)
    num_expressions = expression.shape[1] if expression is not None else (default_expression.shape[-1] if default_expression is not None else 10)
    
    hand_mode, hand_dim, use_pca_model = _resolve_hand_representation(lhand_raw, rhand_raw)
    
    npz_path = os.path.join(model_path, version, f"SMPLX_{gender.upper()}.npz")
    smplx_model = smplx.create(
        model_path=npz_path,
        model_type="smplx",
        gender=gender,
        use_pca=use_pca_model,
        num_pca_comps=hand_dim if use_pca_model else None,
        num_betas=num_betas,
        num_expression_coeffs=num_expressions,
        batch_size=1
    ).to(device).eval()
    
    params_to_use = _build_selective_params(
        use_global_orient, root_pose,
        use_body_pose, body_pose, use_hand_poses, preserve_upper_limb_if_hands,
        hand_mode, hand_dim, lhand_raw, rhand_raw, use_pca_model,
        use_jaw_pose, jaw_pose,
        use_shape, betas, default_betas, num_betas,
        use_expressions, expression, default_expression, num_expressions,
        use_translation, transl,
        use_eye_poses, leye_pose, reye_pose,
        device, verbose
    )
    
    with torch.no_grad():
        output = smplx_model(**params_to_use, return_verts=True)
    
    vertices, faces, joints = _extract_mesh_data(output, smplx_model)
    _print_mesh_info(verbose, vertices, faces, joints)
    
    mesh = _create_mesh_with_processing(vertices, faces, uv, version, npz_path, blender_format, verbose)
    
    if save_obj:
        _save_mesh(mesh, output_path, verbose)
    
    ret = {
        "vertices": vertices,
        "faces": faces,
        "joints": joints,
        "smplx_output": output,
        "parameters_used": params_to_use
    }
    if return_mesh:
        ret["mesh"] = mesh
    return ret


def get_smplx_ldm68(
    smplx_vertices,
    flame_generic_pkl = "assets/body_models/base_models/flame/parametric_models/generic_model.pkl",
    flame_embed_68 = "assets/body_models/landmarks/flame/flame_static_embedding_68_v4.npz",
    smplx_flame_corr_pkl = "assets/body_models/base_models/smplx/vertex_mappings/smplx_flame_vertex_ids.npy"
):
    smplx_flame_vertex_ids = np.load(smplx_flame_corr_pkl, allow_pickle=True)
    
    with open(flame_generic_pkl, 'rb') as f:
        flame_data = pickle.load(f, encoding='latin1')
    flame_faces = flame_data['f']
    
    embed_data = np.load(flame_embed_68, allow_pickle=True)
    lmk_face_idx = embed_data['lmk_face_idx']
    lmk_b_coords = embed_data['lmk_b_coords']
    
    landmarks_smplx_xyz = np.zeros((68, 3))
    
    for i in range(68):
        face_id = lmk_face_idx[i]
        b_coords = lmk_b_coords[i]
        flame_vertex_ids = flame_faces[face_id]
        smplx_vertex_ids = smplx_flame_vertex_ids[flame_vertex_ids]
        tri_vertices = smplx_vertices[smplx_vertex_ids]
        landmarks_smplx_xyz[i] = np.dot(b_coords, tri_vertices)
    
    return landmarks_smplx_xyz


def compute_flame_landmark_coordinates(vertices, faces, flame_landmarks_data_path="assets/body_models/landmarks/flame/flame_static_embedding_68_v4.npz"):
    """
    Compute 3D coordinates of 68 facial landmarks on FLAME mesh.
    
    Args:
        flame_vertices: (N, 3) array of FLAME mesh vertices
        flame_landmarks_data_path: Path to a npz file containing 'lmk_face_idx' and 'lmk_b_coords'
    
    Returns:
        landmarks_3d: (68, 3) array of 3D landmark coordinates
    """
    flame_landmarks_data = np.load(flame_landmarks_data_path)
    lmk_face_idx = flame_landmarks_data['lmk_face_idx']  # (68,) face indices
    lmk_b_coords = flame_landmarks_data['lmk_b_coords']  # (68, 3) barycentric coords
    
    # For each landmark, compute barycentric interpolation
    landmarks_3d = np.zeros((68, 3), dtype=np.float32)
    
    for i in range(68):
        face_idx = lmk_face_idx[i]
        bary_coords = lmk_b_coords[i]
        
        # Get vertex indices for this face
        v_indices = faces[face_idx]
        
        # Get the 3 vertices of the face
        v0 = vertices[v_indices[0]]
        v1 = vertices[v_indices[1]]
        v2 = vertices[v_indices[2]]

        # Barycentric interpolation
        landmarks_3d[i] = (bary_coords[0] * v0 + 
                          bary_coords[1] * v1 + 
                          bary_coords[2] * v2)
    
    return landmarks_3d

def create_flame_from_smplx(
    smplx_params,
    flame_model_path="assets/body_models/base_models/",
    smplx_flame_corr_path="assets/body_models/base_models/smplx/vertex_mappings/smplx_flame_vertex_ids.npy",
    gender="neutral",
    output_path=None,
    save_obj=False,
    verbose=False,
    return_mesh=True,
    blender_format=False
):
    """
    Create a FLAME mesh from SMPL-X parameters using vertex correspondence mapping.
    
    Args:
        smplx_params: Dictionary containing SMPL-X parameters (shape, expression, pose)
        flame_model_path: Path to FLAME generic_model.pkl directory
        smplx_flame_corr_path: Path to SMPL-X to FLAME vertex correspondence mapping
        gender: Gender for the model
        output_path: Path to save the output mesh (if save_obj=True)
        save_obj: Whether to save the mesh to a file
        verbose: Whether to print detailed information
        return_mesh: Whether to return the trimesh object
        blender_format: Whether to convert coordinates to Blender format
    
    Returns:
        Dictionary containing vertices, faces, and optionally the mesh object
    """
    device = _get_device()
    
    # Load SMPL-X to FLAME vertex correspondence
    smplx_flame_vertex_ids = np.load(smplx_flame_corr_path, allow_pickle=True)
    
    # Extract SMPL-X parameters
    sp = _extract_param(smplx_params, "smplx_param", "smplx_params") or smplx_params
    betas_arr = _extract_param(sp, "smplx_shape", "shape")
    expr_arr = _extract_param(sp, "smplx_expr", "expression")
    jaw_pose_arr = _extract_param(sp, "smplx_jaw_pose", "jaw_pose") or [0, 0, 0]
    leye_pose_arr = _extract_param(sp, "smplx_leye_pose", "leye_pose") or [0, 0, 0]
    reye_pose_arr = _extract_param(sp, "smplx_reye_pose", "reye_pose") or [0, 0, 0]
    
    # Convert to tensors
    betas = _to_tensor(betas_arr, device, (1, -1)) if betas_arr is not None else torch.zeros((1, 10), device=device)
    expression = _to_tensor(expr_arr, device, (1, -1)) if expr_arr is not None else torch.zeros((1, 10), device=device)
    jaw_pose = _to_tensor(jaw_pose_arr, device, (1, 3))
    leye_pose = _to_tensor(leye_pose_arr, device, (1, 3))
    reye_pose = _to_tensor(reye_pose_arr, device, (1, 3))
    
    # Adjust dimensions for FLAME model (FLAME typically uses 10 shape params and 10 expression params)
    num_betas = min(betas.shape[1], 300)  # FLAME supports up to 300 shape params
    num_expressions = min(expression.shape[1], 100)  # FLAME supports up to 100 expression params
    
    if verbose:
        print(f"Creating FLAME mesh from SMPL-X parameters:")
        print(f"  Shape params: {num_betas}")
        print(f"  Expression params: {num_expressions}")
    
    # Load FLAME model
    flame_model = smplx.create(
        model_path=flame_model_path,
        model_type="flame",
        gender=gender,
        num_betas=num_betas,
        num_expression_coeffs=num_expressions,
        batch_size=1
    ).to(device).eval()
    
    # Prepare FLAME parameters
    flame_params = {
        "betas": betas[:, :num_betas],
        "expression": expression[:, :num_expressions],
        "jaw_pose": jaw_pose,
        "leye_pose": leye_pose,
        "reye_pose": reye_pose,
        "global_orient": torch.zeros((1, 3), device=device),
        "transl": torch.zeros((1, 3), device=device),
        "neck_pose": torch.zeros((1, 3), device=device)
    }
    
    if verbose:
        print(f"✓ Using SMPL-X expression and jaw parameters for FLAME")
    
    # Generate FLAME mesh
    with torch.no_grad():
        flame_output = flame_model(**flame_params, return_verts=True)
    
    vertices = flame_output.vertices[0].detach().cpu().numpy()
    faces = flame_model.faces.astype(np.int32) if hasattr(flame_model, "faces") else None
    
    if verbose:
        print(f"\nGenerated FLAME mesh:")
        print(f"  Vertices: {vertices.shape}")
        print(f"  Faces: {faces.shape if faces is not None else 'None'}")
    
    # Create trimesh object
    mesh = None
    if faces is not None:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh = apply_shade_smooth_to_mesh(mesh)
        
        if blender_format:
            mesh = convert_trimesh_to_blender(mesh)
    
    # Save mesh if requested
    if save_obj:
        _save_mesh(mesh, output_path, verbose)
    
    result = {
        "vertices": vertices,
        "faces": faces,
        "flame_output": flame_output,
        "parameters_used": flame_params
    }
    
    if return_mesh:
        result["mesh"] = mesh
    
    return result

def create_default_flame(
    model_path="assets/body_models/base_models/",
    gender="neutral",
    num_betas=100,
    num_expressions=50,
    device=None,
    return_mesh=False
):
    """Instantiate a canonical FLAME mesh with zero pose, neutral shape, and expression."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    flame_model = smplx.create(
        model_path=model_path,
        model_type="flame",
        gender=gender,
        num_betas=num_betas,
        num_expression_coeffs=num_expressions,
        batch_size=1
    ).to(device).eval()

    flame_params = {
        "global_orient": torch.zeros((1, 3), device=device),
        "neck_pose": torch.zeros((1, 3), device=device),
        "jaw_pose": torch.zeros((1, 3), device=device),
        "leye_pose": torch.zeros((1, 3), device=device),
        "reye_pose": torch.zeros((1, 3), device=device),
        "betas": torch.zeros((1, num_betas), device=device),
        "expression": torch.zeros((1, num_expressions), device=device),
        "transl": torch.zeros((1, 3), device=device),
    }

    with torch.no_grad():
        flame_output = flame_model(**flame_params, return_verts=True)

    vertices = flame_output.vertices[0].detach().cpu().numpy()
    faces = flame_model.faces.astype(np.int32) if hasattr(flame_model, "faces") else None
    joints = flame_output.joints[0].detach().cpu().numpy() if hasattr(flame_output, "joints") else None

    result = {
        "model": flame_model,
        "vertices": vertices,
        "faces": faces,
        "joints": joints,
        "output": flame_output,
        "params": flame_params,
    }

    if return_mesh and faces is not None:
        result["mesh"] = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy(), process=False)

    return result