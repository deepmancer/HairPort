import os 
import shutil
import random
import itertools
import tqdm
import numpy as np
import pandas as pd
from argparse import ArgumentParser


source_preferred_ids = [
    "ana_2",
    "bale_1", #  nano
    "eren_1", # nano
    "gojo_2", # nano
    "john_snow_1", # nano
    "joker_2", # nano
    "sample_002",
    "sample_041", # nano
    "sample_042",
    "sample_047",
    "sample_048",
    "sample_051",
    "sample_059",
    "sample_066",
    "sample_070",
    "sample_076",
    "sample_086",
    "sample_089",
    "huntington",
    "mave",
    "morty_1",
    "pedro_3",
    "rick_2",
    "sample_095",
    "sample_102",
    "sample_106",
    "sample_110",
    "sample_118",
    "sample_122",
    "sample_128",
    "sample_133",
    "sample_135",
    "sample_136",
    "sample_138",
    # "sample_140",
    "sample_146",
    "sample_164",
    "sample_166",
    "sample_167",
    "sample_168",
    "sample_169",
    "sample_171",
    "superman",
    "taylor_1",
    "taylor_3",
    "thor_2",
    "vanessa_2",
    "vanessa_3",
    "vanessa",
    "wolvorine_1",
]


target_preferred_ids = []
meshes_dir = "/workspace/outputs/hi3dgen_copy"
for folder in os.listdir(meshes_dir):
    shape_mesh_path = os.path.join(meshes_dir, folder, "shape_mesh.glb")
    if os.path.exists(shape_mesh_path):
        file_size_mb = os.path.getsize(shape_mesh_path) / (1024 * 1024)
        if file_size_mb >= 25:
            target_preferred_ids.append(folder)
            
print(len(target_preferred_ids))


def sample_target_id(all_ids, target_preferred_ids, prob_preferred=0.95):
    random.seed()
    if random.random() < prob_preferred:
        return random.choice(target_preferred_ids)
    else:
        non_preferred_ids = [id for id in all_ids if id not in target_preferred_ids]
        return random.choice(non_preferred_ids) if non_preferred_ids else random.choice(all_ids)
    
def sample_source_id(all_ids, source_preferred_ids, prob_preferred=0.45):
    random.seed()
    side_ids = [id for id in all_ids if id.startswith("side")]
    
    rand_val = random.random()
    
    if rand_val < 0.15 and side_ids:
        # 15% chance to select from side files
        return random.choice(side_ids)
    elif rand_val < 0.15 + (prob_preferred * 0.85):
        # prob_preferred (60%) of remaining 85% for preferred ids
        return random.choice(source_preferred_ids)
    else:
        # Remaining chance for all ids (excluding preferred)
        non_preferred_ids = [id for id in all_ids if id not in source_preferred_ids]
        return random.choice(non_preferred_ids) if non_preferred_ids else random.choice(all_ids)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/workspace/outputs/")
    parser.add_argument("--num_pairs", type=int, default=600, help="Number of custom pairs to create")
    parser.add_argument("--weighted_sampling", action="store_true", default=False, 
                        help="Use weighted sampling for preferred source/target IDs (default: False - uniform sampling)")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_path = os.path.join(data_dir, "pairs.csv")
    
    random.seed()
    all_ids = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(data_dir, "image")) if os.path.isfile(os.path.join(data_dir, "image", f))]
    
    # Load existing pairs if the file exists
    existing_pairs = []
    existing_pairs_set = set()
    if os.path.exists(output_path):
        print(f"Loading existing pairs from: {output_path}")
        existing_df = pd.read_csv(output_path)
        existing_pairs = existing_df.to_dict('records')
        existing_pairs_set = set((row['source_id'], row['target_id']) for row in existing_pairs)
        print(f"Found {len(existing_pairs)} existing pairs")
    
    # Sample new pairs
    pairs = []
    attempts = 0
    max_attempts = args.num_pairs * 10  # Prevent infinite loops if pool is exhausted
    
    while len(pairs) < args.num_pairs and attempts < max_attempts:
        attempts += 1
        
        if args.weighted_sampling:
            # Use weighted sampling with preferred IDs
            source_id = sample_source_id(all_ids, source_preferred_ids)
            target_id = sample_target_id(all_ids, target_preferred_ids)
        else:
            # Use uniform sampling (default)
            source_id = random.choice(all_ids)
            target_id = random.choice(all_ids)
        
        # Skip if same ID or if pair already exists
        if source_id == target_id:
            continue
        if (source_id, target_id) in existing_pairs_set:
            continue
        
        # Add to new pairs and update the set to avoid duplicates within new pairs
        pairs.append({"source_id": source_id, "target_id": target_id})
        existing_pairs_set.add((source_id, target_id))
    
    if len(pairs) < args.num_pairs:
        print(f"Warning: Could only generate {len(pairs)} new unique pairs (requested {args.num_pairs})")
    
    # Combine existing and new pairs
    all_pairs = existing_pairs + pairs
    pairs_df = pd.DataFrame(all_pairs)
    
    pairs_df.to_csv(output_path, index=False)
    print(f"Total pairs: {len(all_pairs)} ({len(existing_pairs)} existing + {len(pairs)} new)")
    print(f"Pairs saved to: {output_path}")
