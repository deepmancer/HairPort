#!/usr/bin/env python3
"""
Copy images from CelebAMask-HQ dataset based on pairs.csv.

This script reads pairs.csv and copies all images corresponding to the 
source_id and target_id values from the original CelebAMask-HQ dataset 
to the target dataset's image directory.
"""

import argparse
import shutil
from pathlib import Path
from typing import Set
import pandas as pd


def load_image_ids_from_pairs(pairs_csv_path: str) -> Set[str]:
    """
    Load all unique image IDs from pairs.csv (both source and target).
    
    Args:
        pairs_csv_path: Path to pairs.csv file
        
    Returns:
        Set of unique image IDs as strings
    """
    print(f"Loading pairs from: {pairs_csv_path}")
    pairs_df = pd.read_csv(pairs_csv_path)
    
    # Convert integer IDs to strings without decimals
    source_ids = set(str(int(x)) for x in pairs_df['source_id'])
    target_ids = set(str(int(x)) for x in pairs_df['target_id'])
    
    # Combine both sets
    all_ids = source_ids | target_ids
    
    print(f"Found {len(source_ids)} unique source IDs")
    print(f"Found {len(target_ids)} unique target IDs")
    print(f"Total unique IDs to process: {len(all_ids)}")
    
    return all_ids


def copy_images_from_celeba(
    image_ids: Set[str],
    celeba_dir: str,
    target_dataset_dir: str
) -> None:
    """
    Copy images from CelebAMask-HQ to target dataset.
    
    Args:
        image_ids: Set of image IDs to copy
        celeba_dir: Path to CelebAMask-HQ directory
        target_dataset_dir: Path to target dataset directory
    """
    celeba_path = Path(celeba_dir)
    target_path = Path(target_dataset_dir)
    
    # Source and destination image directories
    source_image_dir = celeba_path / "CelebA-HQ-img"
    dest_image_dir = target_path / "image"
    
    # Validate source directory
    if not source_image_dir.exists():
        print(f"Error: Source image directory does not exist: {source_image_dir}")
        return
    
    # Create destination directory if it doesn't exist
    dest_image_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSource directory: {source_image_dir}")
    print(f"Destination directory: {dest_image_dir}")
    print(f"\nStarting image copy process...")
    
    copied_count = 0
    skipped_count = 0
    missing_count = 0
    missing_ids = []
    
    for idx, image_id in enumerate(sorted(image_ids), 1):
        # Construct filenames
        filename = f"{image_id}.jpg"
        source_file = source_image_dir / filename
        dest_file = dest_image_dir / filename
        
        # Check if destination already exists
        if dest_file.exists():
            skipped_count += 1
        elif not source_file.exists():
            missing_count += 1
            missing_ids.append(image_id)
            print(f"Warning: Source image not found: {source_file}")
        else:
            # Copy the file
            shutil.copy2(source_file, dest_file)
            copied_count += 1
        
        # Progress update every 50 images
        if idx % 50 == 0:
            print(f"Progress: {idx}/{len(image_ids)} images processed "
                  f"(copied: {copied_count}, skipped: {skipped_count}, missing: {missing_count})")
    
    # Final summary
    print(f"\n{'='*80}")
    print("COPY SUMMARY")
    print(f"{'='*80}")
    print(f"Total IDs to process: {len(image_ids)}")
    print(f"Images copied: {copied_count}")
    print(f"Images skipped (already exist): {skipped_count}")
    print(f"Images missing from source: {missing_count}")
    
    if missing_ids:
        print(f"\nMissing image IDs ({len(missing_ids)}):")
        for idx, missing_id in enumerate(missing_ids, 1):
            print(f"  {idx}. {missing_id}")
    
    print(f"\n{'='*80}")
    print("COPY COMPLETE")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Copy images from CelebAMask-HQ based on pairs.csv"
    )
    parser.add_argument(
        "--pairs_csv",
        type=str,
        required=True,
        help="Path to pairs.csv file"
    )
    parser.add_argument(
        "--celeba_dir",
        type=str,
        default="/workspace/CelebAMask-HQ",
        help="Path to CelebAMask-HQ directory (default: /workspace/CelebAMask-HQ)"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to target dataset directory"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.pairs_csv).exists():
        print(f"Error: pairs.csv not found: {args.pairs_csv}")
        return 1
    
    if not Path(args.celeba_dir).exists():
        print(f"Error: CelebAMask-HQ directory not found: {args.celeba_dir}")
        return 1
    
    # Load image IDs from pairs.csv
    image_ids = load_image_ids_from_pairs(args.pairs_csv)
    
    # Copy images
    copy_images_from_celeba(image_ids, args.celeba_dir, args.dataset_dir)
    
    return 0


if __name__ == "__main__":
    exit(main())
