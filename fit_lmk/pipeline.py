import torch
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Dict
import argparse

from fit_lmk.camera_utils import generate_mesh_rotations
from fit_lmk.multi_view_fusion import MultiViewLandmarkFuser


def estimate_3d_landmarks(
    mesh_path: Union[str, Path],
    cam_loc: Union[Tuple[float, float, float], torch.Tensor],
    cam_rot: Union[Tuple[float, float, float], torch.Tensor],
    ortho_scale: float = 1.1,
    output_dir: Union[str, Path] = None,
    num_perturbations: int = 4,
    angle_range: float = 0.15,
    trans_range: float = 0.05,
    resolution: int = 1024,
    optimize: bool = True,
    device: str = 'cuda',
    debug: bool = False,
    debug_dir: str = './debug_outputs',
    super_resolution: bool = True,
    render_function = None
) -> Dict[str, torch.Tensor]:
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
    
    if output_dir is None:
        output_dir = Path('./output_landmarks')
    output_dir = Path(output_dir)
    render_dir = output_dir / 'rendered_views'
    render_dir.mkdir(parents=True, exist_ok=True)
    
    if render_function is None:
        from .blender_renderer import render_multi_view
        render_function = render_multi_view
    
    print("Rendering multi-view images...")
    rendered_paths = render_function(
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
        rendered_paths=rendered_paths,
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


def main():
    parser = argparse.ArgumentParser(description='Estimate 3D facial landmarks on mesh')
    parser.add_argument('--mesh_path', type=str, required=True, help='Path to mesh file (.glb, .obj, .ply)')
    parser.add_argument('--cam_loc', type=float, nargs=3, default=[0.0, -1.45, 0.0], help='Camera location (x, y, z)')
    parser.add_argument('--cam_rot', type=float, nargs=3, default=[1.5708, 0.0, 0.0], help='Camera rotation (rx, ry, rz) in radians')
    parser.add_argument('--ortho_scale', type=float, default=1.1, help='Orthographic scale')
    parser.add_argument('--output_dir', type=str, default='./output_landmarks', help='Output directory')
    parser.add_argument('--num_perturbations', type=int, default=4, help='Number of perturbed views')
    parser.add_argument('--angle_range', type=float, default=0.15, help='Angle perturbation range (radians)')
    parser.add_argument('--trans_range', type=float, default=0.05, help='Translation perturbation range')
    parser.add_argument('--resolution', type=int, default=1024, help='Rendering resolution')
    parser.add_argument('--no_optimize', action='store_true', help='Disable optimization')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with visualizations')
    parser.add_argument('--debug_dir', type=str, default='./debug_outputs', help='Debug output directory')
    parser.add_argument('--no_super_resolution', action='store_true', help='Disable super-resolution enhancement')
    
    args = parser.parse_args()
    
    results = estimate_3d_landmarks(
        mesh_path=args.mesh_path,
        cam_loc=args.cam_loc,
        cam_rot=args.cam_rot,
        ortho_scale=args.ortho_scale,
        output_dir=args.output_dir,
        num_perturbations=args.num_perturbations,
        angle_range=args.angle_range,
        trans_range=args.trans_range,
        resolution=args.resolution,
        optimize=not args.no_optimize,
        device=args.device,
        debug=args.debug,
        debug_dir=args.debug_dir,
        super_resolution=not args.no_super_resolution
    )
    
    print(f"\nEstimation complete!")
    print(f"Number of landmarks: {len(results['landmarks_3d'])}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
