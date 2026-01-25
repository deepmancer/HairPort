#!/usr/bin/env python3
"""Create a downsampled subset of the CelebAMask-HQ dataset."""

import argparse
import glob
import os
import random
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Utility Functions
# =============================================================================

def has_hair_mask(image_filename: str, mask_annot_dir: str) -> bool:
    """Check if a hair mask exists for the given image."""
    image_id = image_filename.replace('.jpg', '')
    pattern = os.path.join(mask_annot_dir, "*", f"{image_id}_hair.png")
    return len(glob.glob(pattern)) > 0


def parse_attributes(attr_file: str) -> Dict[str, Dict[str, int]]:
    """Parse the CelebAMask-HQ attribute annotation file."""
    attributes = {}
    with open(attr_file, 'r') as f:
        lines = f.readlines()
    
    attr_names = lines[1].strip().split()
    hat_idx = attr_names.index('Wearing_Hat')
    
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) < len(attr_names) + 1:
            continue
        filename = parts[0]
        attrs = [int(x) for x in parts[1:]]
        attributes[filename] = {
            'Wearing_Hat': attrs[hat_idx],
            'all_attrs': dict(zip(attr_names, attrs))
        }
    return attributes


def parse_pose_annotations(pose_file: str) -> Dict[str, Tuple[float, float, float]]:
    """Parse pose annotation file. Returns dict mapping filename to (yaw, pitch, roll)."""
    poses = {}
    with open(pose_file, 'r') as f:
        lines = f.readlines()
    
    # Skip count line and header
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 4:
            filename = parts[0]
            yaw, pitch, roll = float(parts[1]), float(parts[2]), float(parts[3])
            poses[filename] = (yaw, pitch, roll)
    return poses


# =============================================================================
# Candidate Pool
# =============================================================================

class CandidatePool:
    """Collects and manages eligible sample candidates."""
    
    def __init__(
        self,
        attributes: Dict[str, Dict[str, int]],
        mask_annot_dir: str,
        existing_ids: Optional[Set[str]] = None
    ):
        self.attributes = attributes
        self.mask_annot_dir = mask_annot_dir
        self.existing_ids = existing_ids or set()
        self._candidates: Optional[List[str]] = None
    
    def _is_eligible(self, filename: str) -> bool:
        """Check if image meets all eligibility criteria."""
        # Must not wear hat
        if self.attributes[filename]['Wearing_Hat'] != -1:
            return False
        # Must have hair mask
        if not has_hair_mask(filename, self.mask_annot_dir):
            return False
        # Must not be in existing dataset
        image_id = filename.replace('.jpg', '')
        if image_id in self.existing_ids:
            return False
        return True
    
    def get_candidates(self) -> List[str]:
        """Return list of eligible candidate filenames."""
        if self._candidates is None:
            print(f"Building candidate pool from {len(self.attributes)} images...")
            self._candidates = [
                f for f in tqdm(self.attributes, desc="Filtering candidates")
                if self._is_eligible(f)
            ]
            self._candidates.sort()
            print(f"Found {len(self._candidates)} eligible candidates")
        return self._candidates
    
    def print_stats(self):
        """Print filtering statistics."""
        total = len(self.attributes)
        no_hat = sum(1 for a in self.attributes.values() if a['Wearing_Hat'] == -1)
        candidates = len(self.get_candidates())
        print(f"Total images: {total}")
        print(f"Images without hat: {no_hat}")
        print(f"Already in dataset: {len(self.existing_ids)}")
        print(f"Eligible candidates: {candidates}")


# =============================================================================
# Sampler Abstraction
# =============================================================================

class Sampler(ABC):
    """Abstract base class for sampling strategies."""
    
    @abstractmethod
    def sample(self, candidates: List[str], n: int) -> Tuple[List[str], List[str]]:
        """Sample and pair candidates into source/target sets."""
        pass


class RandomPairSampler(Sampler):
    """Randomly samples and pairs candidates into source/target sets."""
    
    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
    
    def sample(self, candidates: List[str], n: int) -> Tuple[List[str], List[str]]:
        if len(candidates) < n:
            print(f"Warning: Only {len(candidates)} candidates available, requested {n}")
            n = len(candidates)
        
        if n == 0:
            return [], []
        
        random.seed(self.seed)
        selected = random.sample(candidates, n)
        selected.sort()
        
        split = n // 2
        source_images = selected[:split]
        target_images = selected[split:2 * split]
        
        print(f"Sampled {len(selected)} images")
        print(f"  Source set: {len(source_images)}")
        print(f"  Target set: {len(target_images)}")
        
        return source_images, target_images


class PoseDiversitySampler(Sampler):
    """Samples pairs with high pose diversity between source and target."""
    
    def __init__(self, seed: int = 42, poses: Optional[Dict[str, Tuple[float, float, float]]] = None,
                 top_k: int = 40, **kwargs):
        self.seed = seed
        self.poses = poses or {}
        self.top_k = top_k
    
    def _compute_pose_differences(self, source: str, candidates: List[str]) -> np.ndarray:
        """Compute L2 pose differences between source and all candidates (vectorized)."""
        source_pose = np.array(self.poses.get(source, (0, 0, 0)))
        candidate_poses = np.array([self.poses.get(c, (0, 0, 0)) for c in candidates])
        # L2 distance in (yaw, pitch, roll) space
        return np.linalg.norm(candidate_poses - source_pose, axis=1)
    
    def sample(self, candidates: List[str], n: int) -> Tuple[List[str], List[str]]:
        n_pairs = n // 2
        if len(candidates) < n_pairs * 2:
            print(f"Warning: Only {len(candidates)} candidates, need {n_pairs * 2}")
            n_pairs = len(candidates) // 2
        
        if n_pairs == 0:
            return [], []
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        available = set(candidates)
        source_images = []
        target_images = []
        
        for _ in range(n_pairs):
            if len(available) < 2:
                break
            
            # Randomly select source
            source = random.choice(list(available))
            available.remove(source)
            
            # Compute pose differences to remaining candidates
            remaining = list(available)
            diffs = self._compute_pose_differences(source, remaining)
            
            # Get top-k with largest pose differences
            k = min(self.top_k, len(remaining))
            top_indices = np.argsort(diffs)[-k:]
            top_candidates = [remaining[i] for i in top_indices]
            
            # Randomly select target from top-k
            target = random.choice(top_candidates)
            available.remove(target)
            
            source_images.append(source)
            target_images.append(target)
        
        source_images.sort()
        target_images.sort()
        
        print(f"Sampled {len(source_images) + len(target_images)} images (pose-diverse)")
        print(f"  Source set: {len(source_images)}")
        print(f"  Target set: {len(target_images)}")
        
        return source_images, target_images


class SamplerFactory:
    """Factory for creating sampler instances."""
    
    @staticmethod
    def create(sampler_type: str = "random", **kwargs) -> Sampler:
        samplers = {
            "random": RandomPairSampler,
            "pose_diverse": PoseDiversitySampler,
        }
        if sampler_type not in samplers:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
        return samplers[sampler_type](**kwargs)


# =============================================================================
# Existing Dataset Loader
# =============================================================================

class ExistingDataset:
    """Manages loading and tracking of existing dataset state."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.pairs_df: Optional[pd.DataFrame] = None
        self.image_ids: Set[str] = set()
        self.source_ids: Set[str] = set()
        self.target_ids: Set[str] = set()
        self._load()
    
    def _load(self):
        pairs_path = self.output_dir / "pairs.csv"
        if pairs_path.exists():
            print(f"\nFound existing dataset at: {self.output_dir}")
            self.pairs_df = pd.read_csv(pairs_path)
            self.source_ids = set(self.pairs_df['source_id'].astype(str))
            self.target_ids = set(self.pairs_df['target_id'].astype(str))
            self.image_ids = self.source_ids | self.target_ids
            print(f"  {len(self.pairs_df)} pairs, {len(self.image_ids)} unique images")
    
    @property
    def exists(self) -> bool:
        return self.pairs_df is not None


# =============================================================================
# Dataset Writer
# =============================================================================

class DatasetWriter:
    """Handles all file I/O for creating/updating the dataset."""
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_path = Path(source_dir)
        self.output_path = Path(output_dir)
        self.image_output_dir = self.output_path / "image"
        self.hair_mask_dir = self.output_path / "hair_mask"
        self.mask_annot_dir = self.source_path / "CelebAMask-HQ-mask-anno"
    
    def write(
        self,
        source_images: List[str],
        target_images: List[str],
        existing_pairs_df: Optional[pd.DataFrame] = None,
        sampling_method: str = "random"
    ):
        """Write dataset to disk."""
        is_update = existing_pairs_df is not None and len(existing_pairs_df) > 0
        selected_images = source_images + target_images
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.image_output_dir.mkdir(exist_ok=True)
        self.hair_mask_dir.mkdir(exist_ok=True)
        
        self._copy_images(selected_images)
        self._copy_hair_masks(selected_images)
        self._update_attributes(selected_images)
        self._update_mapping(selected_images)
        self._update_pose(selected_images)
        pairs_df = self._update_pairs(source_images, target_images, existing_pairs_df, sampling_method)
        self._update_readme(pairs_df, is_update, len(selected_images))
        
        self._print_summary(pairs_df, is_update, len(selected_images))
    
    def _copy_images(self, selected_images: List[str]):
        if not selected_images:
            print("\nNo new images to copy.")
            return
        
        print(f"\nCopying {len(selected_images)} NEW images...")
        image_source_dir = self.source_path / "CelebA-HQ-img"
        
        for i, filename in enumerate(selected_images, 1):
            src = image_source_dir / filename
            dst = self.image_output_dir / filename
            if not dst.exists():
                if src.exists():
                    shutil.copy2(src, dst)
                    if i % 10 == 0:
                        print(f"  Copied {i}/{len(selected_images)} images...")
                else:
                    print(f"Warning: Image not found: {src}")
        print(f"Completed copying images to {self.image_output_dir}")
    
    def _copy_hair_masks(self, selected_images: List[str]):
        if not selected_images:
            print("\nNo new hair masks to copy.")
            return
        
        print(f"\nCopying hair masks for {len(selected_images)} images...")
        copied = 0
        skipped = 0
        
        for filename in selected_images:
            sample_id = filename.replace('.jpg', '')
            dst = self.hair_mask_dir / f"{sample_id}.png"
            
            if dst.exists():
                skipped += 1
                continue
            
            # Find source hair mask: mask_annot_dir/*/{sample_id}_hair.png
            pattern = str(self.mask_annot_dir / "*" / f"{sample_id}_hair.png")
            matches = glob.glob(pattern)
            
            if matches:
                shutil.copy2(matches[0], dst)
                copied += 1
            else:
                print(f"Warning: Hair mask not found for {sample_id}")
        
        print(f"Copied {copied} hair masks, skipped {skipped} existing")
    
    def _update_attributes(self, selected_images: List[str]):
        print("\nUpdating attribute file...")
        attr_source = self.source_path / "CelebAMask-HQ-attribute-anno.txt"
        attr_dest = self.output_path / "CelebAMask-HQ-attribute-anno.txt"
        
        existing_lines = {}
        if attr_dest.exists():
            with open(attr_dest, 'r') as f:
                lines = f.readlines()
                for line in lines[2:]:
                    parts = line.strip().split()
                    if parts:
                        existing_lines[parts[0]] = line
        
        with open(attr_source, 'r') as f:
            source_lines = f.readlines()
        
        selected_set = set(selected_images)
        all_lines = existing_lines.copy()
        for line in source_lines[2:]:
            filename = line.strip().split()[0]
            if filename in selected_set:
                all_lines[filename] = line
        
        with open(attr_dest, 'w') as f:
            f.write(f"{len(all_lines)}\n")
            f.write(source_lines[1])
            for filename in sorted(all_lines.keys()):
                f.write(all_lines[filename])
        
        print(f"Updated attribute file: {attr_dest} (total: {len(all_lines)} images)")
    
    def _update_mapping(self, selected_images: List[str]):
        print("\nUpdating mapping file...")
        mapping_source = self.source_path / "CelebA-HQ-to-CelebA-mapping.txt"
        mapping_dest = self.output_path / "CelebA-HQ-to-CelebA-mapping.txt"
        
        existing_lines = {}
        header = None
        if mapping_dest.exists():
            with open(mapping_dest, 'r') as f:
                lines = f.readlines()
                if lines:
                    header = lines[0]
                    for line in lines[1:]:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            existing_lines[f"{parts[0]}.jpg"] = line
        
        with open(mapping_source, 'r') as f:
            source_lines = f.readlines()
        
        if header is None:
            header = source_lines[0]
        
        selected_set = set(selected_images)
        all_lines = existing_lines.copy()
        for line in source_lines[1:]:
            parts = line.strip().split()
            if len(parts) >= 3:
                filename = f"{parts[0]}.jpg"
                if filename in selected_set:
                    all_lines[filename] = line
        
        with open(mapping_dest, 'w') as f:
            f.write(header)
            for filename in sorted(all_lines.keys()):
                f.write(all_lines[filename])
        
        print(f"Updated mapping file: {mapping_dest} (total: {len(all_lines)} mappings)")
    
    def _update_pose(self, selected_images: List[str]):
        pose_source = self.source_path / "CelebAMask-HQ-pose-anno.txt"
        if not pose_source.exists():
            return
        
        print("\nUpdating pose annotation file...")
        pose_dest = self.output_path / "CelebAMask-HQ-pose-anno.txt"
        
        existing_lines = {}
        pose_header = None
        if pose_dest.exists():
            with open(pose_dest, 'r') as f:
                lines = f.readlines()
                if lines and not lines[0].strip()[0].isdigit():
                    pose_header = lines[0]
                    start_idx = 1
                else:
                    start_idx = 0
                for line in lines[start_idx:]:
                    parts = line.strip().split()
                    if parts:
                        existing_lines[parts[0]] = line
        
        with open(pose_source, 'r') as f:
            source_lines = f.readlines()
        
        if source_lines and not source_lines[0].strip()[0].isdigit():
            if pose_header is None:
                pose_header = source_lines[0]
            start_idx = 1
        else:
            start_idx = 0
        
        selected_set = set(selected_images)
        all_lines = existing_lines.copy()
        for line in source_lines[start_idx:]:
            parts = line.strip().split()
            if parts and parts[0] in selected_set:
                all_lines[parts[0]] = line
        
        with open(pose_dest, 'w') as f:
            if pose_header:
                f.write(pose_header)
            for filename in sorted(all_lines.keys()):
                f.write(all_lines[filename])
        
        print(f"Updated pose annotation file: {pose_dest} (total: {len(all_lines)} annotations)")
    
    def _update_pairs(
        self,
        source_images: List[str],
        target_images: List[str],
        existing_pairs_df: Optional[pd.DataFrame],
        sampling_method: str = "random"
    ) -> pd.DataFrame:
        print("\nUpdating pairs CSV...")
        
        source_ids = [img.replace('.jpg', '') for img in source_images]
        target_ids = [img.replace('.jpg', '') for img in target_images]
        
        # Map sampler type to label
        method_label = "pose_aware" if sampling_method == "pose_diverse" else "random"
        
        new_pairs_df = pd.DataFrame({
            'source_id': source_ids,
            'target_id': target_ids,
            'sampling_method': method_label
        })
        
        if existing_pairs_df is not None and len(existing_pairs_df) > 0:
            pairs_df = pd.concat([existing_pairs_df, new_pairs_df], ignore_index=True)
            pairs_df = pairs_df.drop_duplicates(subset=['source_id', 'target_id'], keep='first')
            print(f"Merged {len(existing_pairs_df)} existing + {len(new_pairs_df)} new pairs")
        else:
            pairs_df = new_pairs_df
            print(f"Created {len(pairs_df)} new pairs")
        
        pairs_csv_path = self.output_path / "pairs.csv"
        pairs_df.to_csv(pairs_csv_path, index=False)
        print(f"Saved pairs CSV: {pairs_csv_path}")
        
        return pairs_df
    
    def _update_readme(self, pairs_df: pd.DataFrame, is_update: bool, new_image_count: int):
        total_unique = len(set(pairs_df['source_id']) | set(pairs_df['target_id']))
        readme_source = self.source_path / "README.txt"
        readme_dest = self.output_path / "README.txt"
        
        if not is_update or not readme_dest.exists():
            content = ""
            if readme_source.exists():
                with open(readme_source, 'r') as f:
                    content = f.read()
            
            with open(readme_dest, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("CelebAMask-HQ SUBSET\n")
                f.write("=" * 60 + "\n\n")
                f.write("Selection criteria: Wearing_Hat == -1 (no hat)\n")
                f.write("Note: 'CelebA-HQ-img' directory renamed to 'image'.\n\n")
                f.write(f"Dataset Statistics:\n")
                f.write(f"  - Total images: {total_unique}\n")
                f.write(f"  - Total pairs: {len(pairs_df)}\n\n")
                f.write("=" * 60 + "\n")
                if content:
                    f.write("ORIGINAL README\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(content)
        else:
            with open(readme_dest, 'r') as f:
                readme_content = f.read()
            
            stats_pattern = r"Dataset Statistics:.*?(?=\n={60}|\Z)"
            new_stats = f"Dataset Statistics:\n  - Total images: {total_unique}\n  - Total pairs: {len(pairs_df)}\n"
            updated_content = re.sub(stats_pattern, new_stats, readme_content, flags=re.DOTALL)
            
            with open(readme_dest, 'w') as f:
                f.write(updated_content)
    
    def _print_summary(self, pairs_df: pd.DataFrame, is_update: bool, new_image_count: int):
        total_unique = len(set(pairs_df['source_id']) | set(pairs_df['target_id']))
        print(f"\n{'='*60}")
        print("Dataset update complete!" if is_update else "Subset creation complete!")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_path}")
        print(f"Total pairs: {len(pairs_df)}")
        print(f"Total unique images: {total_unique}")
        if is_update and new_image_count > 0:
            print(f"New images added: {new_image_count}")
        print(f"{'='*60}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Create a CelebAMask-HQ subset.")
    parser.add_argument("--source_dir", type=str, default="/workspace/CelebAMask-HQ")
    parser.add_argument("--output_dir", type=str, default="/workspace/celeba_reduced")
    parser.add_argument("--n_set", type=int, default=600)
    parser.add_argument("--seed", type=int, default=423)
    parser.add_argument("--sampler", type=str, default="random",
                        choices=["random", "pose_diverse"])
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k candidates for pose_diverse sampler")
    args = parser.parse_args()
    
    source_path = Path(args.source_dir)
    if not source_path.exists():
        print(f"Error: Source directory does not exist: {args.source_dir}")
        return 1
    
    attr_file = source_path / "CelebAMask-HQ-attribute-anno.txt"
    if not attr_file.exists():
        print(f"Error: Attribute file not found: {attr_file}")
        return 1
    
    mask_annot_dir = source_path / "CelebAMask-HQ-mask-anno"
    if not mask_annot_dir.exists():
        print(f"Error: Mask annotation directory not found: {mask_annot_dir}")
        return 1
    
    # Parse attributes
    print("Parsing attribute file...")
    attributes = parse_attributes(str(attr_file))
    
    # Parse pose annotations if needed
    poses = {}
    pose_file = source_path / "CelebAMask-HQ-pose-anno.txt"
    if args.sampler == "pose_diverse":
        if not pose_file.exists():
            print(f"Error: Pose file not found: {pose_file}")
            return 1
        print("Parsing pose annotations...")
        poses = parse_pose_annotations(str(pose_file))
    
    # Load existing dataset
    existing = ExistingDataset(args.output_dir)
    
    # Build candidate pool
    pool = CandidatePool(attributes, str(mask_annot_dir), existing.image_ids)
    pool.print_stats()
    candidates = pool.get_candidates()
    
    if not candidates and not existing.exists:
        print("\nNo eligible candidates found and no existing dataset.")
        return 1
    
    # Sample source/target pairs
    sampler = SamplerFactory.create(
        args.sampler, seed=args.seed, poses=poses, top_k=args.top_k
    )
    source_images, target_images = sampler.sample(candidates, args.n_set)
    
    # Write dataset
    if source_images or existing.exists:
        writer = DatasetWriter(args.source_dir, args.output_dir)
        writer.write(source_images, target_images, existing.pairs_df, args.sampler)
    else:
        print("\nNo images to process.")
        return 1
    
    return 0


# =============================================================================
# Utility: Filter Pairs by Hair Mask
# =============================================================================

def filter_pairs_by_hair_mask(
    pairs_df: pd.DataFrame,
    mask_annot_dir: str
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """Filter pairs to keep only those where both source and target have hair masks."""
    unique_sources = pairs_df['source_id'].unique()
    unique_targets = pairs_df['target_id'].unique()
    
    print(f"Checking {len(unique_sources)} source IDs for hair masks...")
    sources_missing = [
        str(sid) for sid in unique_sources
        if not has_hair_mask(f"{sid}.jpg", mask_annot_dir)
    ]
    
    print(f"Checking {len(unique_targets)} target IDs for hair masks...")
    targets_missing = [
        str(tid) for tid in unique_targets
        if not has_hair_mask(f"{tid}.jpg", mask_annot_dir)
    ]
    
    sources_set = set(sources_missing)
    targets_set = set(targets_missing)
    
    cleaned_df = pairs_df[
        ~pairs_df['source_id'].astype(str).isin(sources_set) &
        ~pairs_df['target_id'].astype(str).isin(targets_set)
    ].copy()
    
    before_dedup = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(
        subset=['source_id', 'target_id'], keep='first'
    ).reset_index(drop=True)
    
    print(f"\nResults:")
    print(f"  Sources without mask: {len(sources_missing)}")
    print(f"  Targets without mask: {len(targets_missing)}")
    print(f"  Original pairs: {len(pairs_df)}")
    print(f"  Final cleaned pairs: {len(cleaned_df)}")
    print(f"  Duplicates removed: {before_dedup - len(cleaned_df)}")
    
    return sources_missing, targets_missing, cleaned_df


if __name__ == "__main__":
    exit(main())
