import sys
from pathlib import Path
import json
import torch
import numpy as np
from typing import List, Dict


from .run_standalone import estimate_3d_landmarks_standalone


def process_batch(
    config_file: str,
    device: str = 'cuda',
    blender_path: str = 'blender'
):
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    results_summary = []
    
    for idx, item in enumerate(config['meshes']):
        print(f"\n{'='*60}")
        print(f"Processing mesh {idx + 1}/{len(config['meshes'])}")
        print(f"Mesh: {item['mesh_path']}")
        print(f"{'='*60}\n")
        
        try:
            results = estimate_3d_landmarks_standalone(
                mesh_path=item['mesh_path'],
                cam_loc=item.get('cam_loc', [0.0, -1.45, 0.0]),
                cam_rot=item.get('cam_rot', [1.5708, 0.0, 0.0]),
                ortho_scale=item.get('ortho_scale', 1.1),
                output_dir=item.get('output_dir', f'./output_{idx}'),
                num_perturbations=config.get('num_perturbations', 4),
                angle_range=config.get('angle_range', 0.15),
                trans_range=config.get('trans_range', 0.05),
                resolution=config.get('resolution', 1024),
                optimize=config.get('optimize', True),
                device=device,
                blender_path=blender_path
            )
            
            results_summary.append({
                'mesh_path': item['mesh_path'],
                'status': 'success',
                'num_landmarks': len(results['landmarks_3d']),
                'output_dir': item.get('output_dir', f'./output_{idx}')
            })
            
            print(f"✓ Successfully processed {item['mesh_path']}")
            
        except Exception as e:
            print(f"✗ Failed to process {item['mesh_path']}: {e}")
            results_summary.append({
                'mesh_path': item['mesh_path'],
                'status': 'failed',
                'error': str(e)
            })
    
    summary_file = Path(config.get('output_dir', '.')) / 'batch_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Batch Processing Complete")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*60}")
    
    successful = sum(1 for r in results_summary if r['status'] == 'success')
    print(f"\nSuccessful: {successful}/{len(config['meshes'])}")
    print(f"Failed: {len(config['meshes']) - successful}/{len(config['meshes'])}")
    
    return results_summary


def create_example_config(output_path: str = 'batch_config.json'):
    example_config = {
        "num_perturbations": 4,
        "angle_range": 0.15,
        "trans_range": 0.05,
        "resolution": 1024,
        "optimize": True,
        "output_dir": "./batch_output",
        "meshes": [
            {
                "mesh_path": "/path/to/mesh1.glb",
                "cam_loc": [0.0, -1.45, 0.0],
                "cam_rot": [1.5708, 0.0, 0.0],
                "ortho_scale": 1.1,
                "output_dir": "./output_mesh1"
            },
            {
                "mesh_path": "/path/to/mesh2.glb",
                "cam_loc": [0.0, -1.5, 0.0],
                "cam_rot": [1.5708, 0.0, 0.0],
                "ortho_scale": 1.15,
                "output_dir": "./output_mesh2"
            }
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print(f"Example config file created: {output_path}")
    print("Edit this file with your mesh paths and camera parameters")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process multiple meshes')
    parser.add_argument('--config', type=str, help='Path to batch config JSON file')
    parser.add_argument('--create_example', action='store_true', help='Create example config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--blender_path', type=str, default='blender', help='Path to Blender executable')
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_config()
    elif args.config:
        process_batch(args.config, args.device, args.blender_path)
    else:
        parser.print_help()
