
import argparse
import os
import re
import shutil
import glob
from pathlib import Path
from typing import Dict, List, Set, Tuple
import random
import pandas as pd

to_combine_directories = {
    "/workspace/celeba_subset/": {
        "sub_files": [
            "image",
            "matted_image",
            "matted_image_centered",
            "silh_mask",
            "silh_mask_centered",
            "prompt",
            "balder_input/image",
            "balder_input/combined_hair_body",
            "balder_input/dataset",
            "balder_input/final_mask",
            "balder_input/seg",
        ],
        "sub_folders": [
            "lmk",
            "lmk_3d/shape_hi3dgen__texture_mvadapter",
            "mvadapter/hi3dgen",
            "pixel3dmm_output",
            "view_aligned/shape_hi3dgen__texture_mvadapter",
            "hairfastgan",
            "hi3dgen",
        ],
    },
    "/workspace/celeba_subset/bald/w_seg": {
        "sub_files": [
            "image",
            "matted_image",
            "matted_image_centered",
            "silh_mask",
            "silh_mask_centered",
        ],
        "sub_folders": [
            "lmk",
            "pixel3dmm_output",
        ],
    },
    "/workspace/celeba_subset/bald/wo_seg": {
        "sub_files": [
            "image",
            "matted_image",
            "matted_image_centered",
            "silh_mask",
            "silh_mask_centered",
        ],
        "sub_folders": [
            "lmk",
            "pixel3dmm_output",
        ],
    },
}

dataset_dir = "/workspace/celeba_subset"
old_dataset_dir = "/workspace/celeba_subset_old"


pairs_csv_path = "/workspace/celeba_subset/pairs.csv"


def extract_id_from_name(name: str) -> str:
    """
    Extract ID from filename or foldername.
    
    Args:
        name: Filename (e.g., '12345.jpg', '12345.png', '12345.txt') or folder name (e.g., '12345')
        
    Returns:
        ID without extension
    """
    # Remove extension if present
    return Path(name).stem


def extract_ids_from_transfer_name(name: str) -> Tuple[str, str]:
    """
    Extract target_id and source_id from transfer folder name.
    
    Args:
        name: Folder name in format '{target_id}_to_{source_id}' (e.g., '123_to_456')
        
    Returns:
        Tuple of (target_id, source_id), or (None, None) if pattern doesn't match
    """
    parts = name.split('_to_')
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None


def is_view_aligned_directory(dir_path: Path) -> bool:
    """
    Check if the directory is a view_aligned directory.
    
    Args:
        dir_path: Directory path to check
        
    Returns:
        True if this is a view_aligned directory
    """
    return 'view_aligned' in str(dir_path)


def load_valid_ids(pairs_csv_path: str) -> Set[str]:
    """
    Load valid source and target IDs from pairs.csv.
    
    Args:
        pairs_csv_path: Path to pairs.csv file
        
    Returns:
        Set of valid IDs (both source and target) as strings
    """
    pairs_df = pd.read_csv(pairs_csv_path)
    
    # Convert integer IDs to strings without decimals
    # Use int() to remove any floating point, then str() to convert to string
    source_ids = set(str(int(x)) for x in pairs_df['source_id'])
    target_ids = set(str(int(x)) for x in pairs_df['target_id'])
    valid_ids = source_ids | target_ids
    
    print(f"Loaded {len(source_ids)} unique source IDs and {len(target_ids)} unique target IDs")
    print(f"Total unique valid IDs: {len(valid_ids)}")
    
    return valid_ids


def copy_files_from_directory(old_dir: Path, new_dir: Path, valid_ids: Set[str]) -> int:
    """
    Copy files from old directory to new directory if they don't exist.
    
    Args:
        old_dir: Source directory path
        new_dir: Destination directory path
        valid_ids: Set of valid IDs from pairs.csv (not used, kept for compatibility)
        
    Returns:
        Number of files copied
    """
    if not old_dir.exists():
        print(f"  Source directory does not exist: {old_dir}")
        return 0
    
    # Create destination directory if it doesn't exist
    new_dir.mkdir(parents=True, exist_ok=True)
    
    files_copied = 0
    
    # Copy files that don't exist
    for old_file in old_dir.iterdir():
        if old_file.is_file():
            new_file = new_dir / old_file.name
            
            if not new_file.exists():
                shutil.copy2(old_file, new_file)
                files_copied += 1
    
    return files_copied


def copy_subfolders_from_directory(old_dir: Path, new_dir: Path, valid_ids: Set[str]) -> int:
    """
    Copy direct subfolders from old directory to new directory if they don't exist.
    
    Args:
        old_dir: Source directory path
        new_dir: Destination directory path
        valid_ids: Set of valid IDs from pairs.csv (not used, kept for compatibility)
        
    Returns:
        Number of folders copied
    """
    if not old_dir.exists():
        print(f"  Source directory does not exist: {old_dir}")
        return 0
    
    # Create destination directory if it doesn't exist
    new_dir.mkdir(parents=True, exist_ok=True)
    
    folders_copied = 0
    
    # Copy subfolders that don't exist
    for old_folder in old_dir.iterdir():
        if old_folder.is_dir():
            new_folder = new_dir / old_folder.name
            
            if not new_folder.exists():
                shutil.copytree(old_folder, new_folder)
                folders_copied += 1
    
    return folders_copied


def merge_datasets():
    """
    Merge old dataset into new dataset based on to_combine_directories configuration.
    """
    print("="*80)
    print("DATASET MERGE (COPY ONLY)")
    print("="*80)
    
    # Load valid IDs from pairs.csv
    print(f"\nLoading valid IDs from: {pairs_csv_path}")
    valid_ids = load_valid_ids(pairs_csv_path)
    
    total_files_copied = 0
    total_folders_copied = 0
    
    # Process each base directory
    for base_dir, config in to_combine_directories.items():
        print(f"\n{'='*80}")
        print(f"Processing base directory: {base_dir}")
        print(f"{'='*80}")
        
        # Extract relative path from base_dir
        base_dir_path = Path(base_dir)
        
        # Determine the relative path from dataset_dir
        if base_dir.startswith(dataset_dir):
            relative_base = base_dir[len(dataset_dir):].lstrip('/')
        else:
            print(f"Warning: Base directory {base_dir} is not under {dataset_dir}")
            continue
        
        # Process sub_files
        print(f"\n--- Processing sub_files ---")
        for sub_file_path in config.get("sub_files", []):
            print(f"\nProcessing file directory: {sub_file_path}")
            
            # Construct full paths
            old_path = Path(old_dataset_dir) / relative_base / sub_file_path
            new_path = Path(dataset_dir) / relative_base / sub_file_path
            
            files_copied = copy_files_from_directory(
                old_path, new_path, valid_ids
            )
            
            total_files_copied += files_copied
            
            print(f"  Files copied: {files_copied}")
        
        # Process sub_folders
        print(f"\n--- Processing sub_folders ---")
        for sub_folder_path in config.get("sub_folders", []):
            print(f"\nProcessing folder directory: {sub_folder_path}")
            
            # Construct full paths
            old_path = Path(old_dataset_dir) / relative_base / sub_folder_path
            new_path = Path(dataset_dir) / relative_base / sub_folder_path
            
            folders_copied = copy_subfolders_from_directory(
                old_path, new_path, valid_ids
            )
            
            total_folders_copied += folders_copied
            
            print(f"  Folders copied: {folders_copied}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("MERGE SUMMARY")
    print(f"{'='*80}")
    print(f"Total files copied: {total_files_copied}")
    print(f"Total folders copied: {total_folders_copied}")
    
    print(f"\n{'='*80}")
    print("MERGE COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    merge_datasets()

