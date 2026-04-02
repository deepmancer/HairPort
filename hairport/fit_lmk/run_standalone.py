import sys
from pathlib import Path
import torch
import numpy as np
from hairport.config import get_config


def estimate_3d_landmarks_standalone(
    mesh_path,
    cam_loc,
    cam_rot,
    ortho_scale=None,
    output_dir='./output_landmarks',
    num_perturbations=None,
    angle_range=None,
    trans_range=None,
    resolution=None,
    optimize=None,
    device=None,
    debug=False,
    debug_dir='./debug_outputs',
    super_resolution=None
):
    cfg = get_config()
    lmk = cfg.landmark_3d
    if ortho_scale is None:
        ortho_scale = lmk.ortho_scale
    if num_perturbations is None:
        num_perturbations = lmk.num_perturbations
    if angle_range is None:
        angle_range = lmk.angle_range
    if trans_range is None:
        trans_range = lmk.trans_range
    if resolution is None:
        resolution = lmk.resolution
    if optimize is None:
        optimize = lmk.optimize
    if device is None:
        device = cfg.device
    if super_resolution is None:
        super_resolution = lmk.super_resolution

    from .camera_utils import generate_mesh_rotations
    from .multi_view_fusion import MultiViewLandmarkFuser
    from .blender_renderer import render_multi_view
    
    device = torch.device(device)
    
    if not isinstance(cam_loc, torch.Tensor):
        cam_loc = torch.tensor(cam_loc, device=device, dtype=torch.float32)
    if not isinstance(cam_rot, torch.Tensor):
        cam_rot = torch.tensor(cam_rot, device=device, dtype=torch.float32)
    
    # Generate mesh rotation angles instead of camera perturbations
    mesh_rotations = generate_mesh_rotations(
        num_views=num_perturbations,
        angle_range=angle_range,
        device=device
    )
    
    # Create repeated camera parameters (same for all views)
    num_views = len(mesh_rotations)
    cam_locs = cam_loc.unsqueeze(0).repeat(num_views, 1)
    cam_rots = cam_rot.unsqueeze(0).repeat(num_views, 1)
    
    output_dir = Path(output_dir)
    render_dir = output_dir / 'rendered_views'
    
    print("Rendering multi-view images with Blender...")
    rendered_paths = render_multi_view(
        mesh_path=mesh_path,
        cam_loc=cam_loc.cpu().numpy().tolist(),
        cam_rot=cam_rot.cpu().numpy().tolist(),
        mesh_rotations=mesh_rotations.cpu().numpy().tolist(),
        output_dir=render_dir,
        ortho_scale=ortho_scale,
        resolution=resolution
    )
    
    print("Processing landmarks from multiple views...")
    fuser = MultiViewLandmarkFuser(
        mesh_path=mesh_path,
        cam_loc=cam_loc,
        cam_rot=cam_rot,
        ortho_scale=ortho_scale,
        resolution=resolution,
        device=device,
        debug=debug,
        debug_dir=debug_dir if debug_dir else str(output_dir / 'debug'),
        super_resolution=super_resolution
    )
    
    results = fuser.process_multi_view(
        rendered_paths=[Path(p) for p in rendered_paths],
        cam_locs=cam_locs,
        cam_rots=cam_rots,
        optimize=optimize
    )
    
    landmarks_path = output_dir / 'landmarks_3d.npy'
    np.save(landmarks_path, results['landmarks_3d'].cpu().numpy())
    print(f"Saved 3D landmarks to {landmarks_path}")
    
    vertex_indices_path = output_dir / 'vertex_indices.npy'
    np.save(vertex_indices_path, results['vertex_indices'].cpu().numpy())
    print(f"Saved vertex indices to {vertex_indices_path}")
    
    confidences_path = output_dir / 'confidences.npy'
    np.save(confidences_path, results['confidences'].cpu().numpy())
    print(f"Saved confidence scores to {confidences_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    cfg = get_config()
    lmk = cfg.landmark_3d

    parser = argparse.ArgumentParser(description='Estimate 3D facial landmarks (standalone)')
    parser.add_argument('--mesh_path', type=str, required=True)
    parser.add_argument('--cam_loc', type=float, nargs=3, default=lmk.default_cam_location)
    parser.add_argument('--cam_rot', type=float, nargs=3, default=lmk.default_cam_rotation)
    parser.add_argument('--ortho_scale', type=float, default=None)
    parser.add_argument('--output_dir', type=str, default='./output_landmarks')
    parser.add_argument('--num_perturbations', type=int, default=None)
    parser.add_argument('--angle_range', type=float, default=None)
    parser.add_argument('--trans_range', type=float, default=None)
    parser.add_argument('--resolution', type=int, default=None)
    parser.add_argument('--no_optimize', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with visualizations')
    parser.add_argument('--debug_dir', type=str, default='./debug_outputs', help='Debug output directory')
    parser.add_argument('--no_super_resolution', action='store_true', help='Disable super-resolution enhancement')
    
    args = parser.parse_args()
    
    results = estimate_3d_landmarks_standalone(
        mesh_path=args.mesh_path,
        cam_loc=args.cam_loc,
        cam_rot=args.cam_rot,
        ortho_scale=args.ortho_scale,
        output_dir=args.output_dir,
        num_perturbations=args.num_perturbations,
        angle_range=args.angle_range,
        trans_range=args.trans_range,
        resolution=args.resolution,
        optimize=False if args.no_optimize else None,
        device=args.device,
        debug=args.debug,
        debug_dir=args.debug_dir,
        super_resolution=False if args.no_super_resolution else None
    )
    
    print(f"\nEstimation complete!")
    print(f"Number of landmarks: {len(results['landmarks_3d'])}")
