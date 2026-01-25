
import argparse
import base64
import json
import logging
import shutil
import struct
from typing import List, Tuple
from argparse import ArgumentParser
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os
import pymeshlab as pml
from PIL import Image  # pillow
import tqdm
from hairport.utility.simplify_glb_mesh import simplify_glb

def get_to_simplify_ids(data_dir: str, min_size_mb: float = 20.0, meshes_dir: str = None) -> List[str]:
    if meshes_dir is None:
        meshes_dir = os.path.join(data_dir, "hi3dgen_copy")
    to_simplify_ids = []
    for folder in os.listdir(meshes_dir):
        shape_mesh_path = os.path.join(meshes_dir, folder, "shape_mesh.glb")
        if os.path.exists(shape_mesh_path):
            file_size_mb = os.path.getsize(shape_mesh_path) / (1024 * 1024)
            if file_size_mb >= min_size_mb:
                to_simplify_ids.append(folder)
    return to_simplify_ids


def get_to_simplify_ids_non_rep(data_dir: str, min_size_mb: float = 20.0, meshes_dir: str = None) -> List[str]:
    if meshes_dir is None:
        meshes_dir = os.path.join(data_dir, "hi3dgen")
    to_simplify_ids = []
    for folder in os.listdir(meshes_dir):
        shape_mesh_path = os.path.join(meshes_dir, folder, "shape_mesh.glb")
        if os.path.exists(shape_mesh_path):
            file_size_mb = os.path.getsize(shape_mesh_path) / (1024 * 1024)
            if file_size_mb <= min_size_mb:
                to_simplify_ids.append(folder)
    return to_simplify_ids


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/workspace/celeba_reduced/")
    parser.add_argument("--target-faces", default=150_000, type=int, help="Target triangle face count")
    args = parser.parse_args()

    data_dir = args.data_dir
    random.seed()
    
    meshes_dir = os.path.join(data_dir, "hi3dgen")
    output_meshes_dir = os.path.join(data_dir, "hi3dgen_new")
    all_ids = os.listdir(meshes_dir)
    # get_to_simplify_ids(data_dir)
    
    # Delete existing folders in output directory if they exist in all_ids
    # for mesh_id in all_ids:
    #     output_folder = os.path.join(output_meshes_dir, mesh_id)
    #     if os.path.exists(output_folder):
    #         shutil.rmtree(output_folder)
        
    #     os.makedirs(output_folder, exist_ok=True)
    
    random.seed()
    random.shuffle(all_ids)
    
    for mesh_id in tqdm.tqdm(all_ids):
        # to_simplify_ids = get_to_simplify_ids_non_rep(data_dir)
        # if mesh_id not in to_simplify_ids:
        #     continue
        # if os.path.exists(f"/workspace/outputs/mvadapter/hi3dgen/{mesh_id}/textured_mesh.glb"):
        #     shutil.rmtree(f"/workspace/outputs/mvadapter/hi3dgen/{mesh_id}")
        #     continue
        # Skip if file is smaller than 10 MB
        try:
            input_glb_path = os.path.join(meshes_dir, mesh_id, "shape_mesh.glb")
            # if os.path.exists(input_glb_path):
            #     file_size_mb = os.path.getsize(input_glb_path) / (1024 * 1024)
            #     if file_size_mb < 10.0:
            #         print(f"Skipping {mesh_id} as it is smaller than 10 MB")
            #         continue
            
            output_glb_path = os.path.join(output_meshes_dir, mesh_id, "shape_mesh.glb")
            simplify_glb(
                input_glb=Path(input_glb_path),
                output_glb=Path(output_glb_path),
                target_faces=args.target_faces,
                qualitythr=0.8,
                extratcoordw=4.0,
                skip_texture=True,
            )
            print(f"Simplified {mesh_id} and saved to {output_glb_path}")
        except Exception as e:
            print(f"Error processing {mesh_id}: {e}")
     