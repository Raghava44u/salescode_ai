"""
Utilities for scene-aware dataset splitting.

Design Decisions:
    - Greedy assignment algorithm based on multi-dimensional bin packing.
    - Balances both total size and class distribution.
    - Verifies zero data leakage (exact and perceptual duplicates).
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from src.config import ProjectConfig
from src.utils.hash_utils import compute_md5, compute_phash, find_exact_duplicates, find_perceptual_duplicates

logger = logging.getLogger("split_utils")

def greedy_scene_splitter(
    scene_mapping: Dict[str, str],
    real_files: List[Path],
    fake_files: List[Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, str]:
    """Split dataset into train/val/test using a greedy scene-aware algorithm.

    Args:
        scene_mapping: Dict mapping filename -> scene_id.
        real_files: Active real images.
        fake_files: Active fake images.
        train_ratio: Target train fraction.
        val_ratio: Target val fraction.
        test_ratio: Target test fraction.

    Returns:
        Dict mapping filename -> 'train', 'val', or 'test'.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # 1. Aggregate stats per scene
    scene_stats = defaultdict(lambda: {"real": 0, "fake": 0})
    
    for p in real_files:
        scene_id = scene_mapping.get(p.name, p.name)
        scene_stats[scene_id]["real"] += 1
        
    for p in fake_files:
        scene_id = scene_mapping.get(p.name, p.name)
        scene_stats[scene_id]["fake"] += 1

    # Total counts
    total_real = len(real_files)
    total_fake = len(fake_files)

    # Targets
    targets = {
        "train": {"real": total_real * train_ratio, "fake": total_fake * train_ratio},
        "val": {"real": total_real * val_ratio, "fake": total_fake * val_ratio},
        "test": {"real": total_real * test_ratio, "fake": total_fake * test_ratio},
    }

    current = {
        "train": {"real": 0, "fake": 0},
        "val": {"real": 0, "fake": 0},
        "test": {"real": 0, "fake": 0},
    }

    # Sort scenes by total size (largest first) to pack large chunks early
    sorted_scenes = sorted(
        scene_stats.items(), 
        key=lambda x: x[1]["real"] + x[1]["fake"], 
        reverse=True
    )

    scene_to_split = {}

    for scene_id, counts in sorted_scenes:
        s_real = counts["real"]
        s_fake = counts["fake"]
        
        best_split = None
        best_score = -float('inf')
        
        for split in ["train", "val", "test"]:
            need_real = targets[split]["real"] - current[split]["real"]
            need_fake = targets[split]["fake"] - current[split]["fake"]
            
            # Weight is the fraction of unmet need relative to target size
            weight_real = need_real / max(1.0, targets[split]["real"])
            weight_fake = need_fake / max(1.0, targets[split]["fake"])
            
            score = (weight_real * s_real) + (weight_fake * s_fake)
            
            # Tie breaker: if scores are equal, prefer the one with highest absolute need
            # We add a tiny fraction of absolute need
            score += 1e-6 * (need_real + need_fake)
            
            if score > best_score:
                best_score = score
                best_split = split
                
        # Assign
        scene_to_split[scene_id] = best_split
        current[best_split]["real"] += s_real
        current[best_split]["fake"] += s_fake

    # Map back to files
    file_to_split = {}
    for p in real_files + fake_files:
        s_id = scene_mapping.get(p.name, p.name)
        file_to_split[p.name] = scene_to_split[s_id]

    return file_to_split

def verify_no_leakage(
    file_to_split: Dict[str, str],
    active_files: List[Path],
    phash_threshold: int = 10,
) -> Tuple[bool, List[str], List[Dict[str, List[str]]]]:
    """Validate that no identical or perceptually similar images cross split boundaries.

    Args:
        file_to_split: Map of filename -> split name.
        active_files: List of all active image paths.
        phash_threshold: Max Hamming distance for pHash duplicate detection.

    Returns:
        (is_valid, list of warning/error strings, list of raw conflict dicts)
    """
    logger.info("Computing hashes for leakage validation...")
    
    file_hashes_md5 = {}
    file_hashes_phash = {}
    
    for p in active_files:
        if p.name not in file_to_split:
            continue
        file_hashes_md5[p] = compute_md5(p)
        ph = compute_phash(p)
        if ph:
            file_hashes_phash[p] = ph

    issues = []
    conflicts = []
    
    
    # 1. Exact duplicates
    exact_groups = find_exact_duplicates(file_hashes_md5)
    for group in exact_groups:
        splits = {file_to_split[p.name] for p in group}
        if len(splits) > 1:
            file_names = [p.name for p in group]
            issues.append(f"LEAKAGE: Exact duplicate group spans splits {splits}: {file_names}")
            conflicts.append({"type": "exact", "files": file_names})

    # 2. Perceptual duplicates
    perceptual_pairs = find_perceptual_duplicates(file_hashes_phash, threshold=phash_threshold)
    for pa, pb, dist in perceptual_pairs:
        sa = file_to_split[pa.name]
        sb = file_to_split[pb.name]
        if sa != sb:
            issues.append(f"LEAKAGE: Perceptual match (dist={dist}) spans {sa}/{sb}: {pa.name} and {pb.name}")
            conflicts.append({"type": "perceptual", "files": [pa.name, pb.name]})

    is_valid = len(issues) == 0
    return is_valid, issues, conflicts

def plot_split_statistics(file_to_split: Dict[str, str], real_files: List[Path], fake_files: List[Path], out_path: Path):
    """Generate a bar chart showing the split distribution."""
    splits = ["train", "val", "test"]
    real_counts = {s: 0 for s in splits}
    fake_counts = {s: 0 for s in splits}
    
    for p in real_files:
        if p.name in file_to_split:
            real_counts[file_to_split[p.name]] += 1
            
    for p in fake_files:
        if p.name in file_to_split:
            fake_counts[file_to_split[p.name]] += 1

    x = range(len(splits))
    width = 0.35

    plt.figure(figsize=(10, 6))
    
    r_vals = [real_counts[s] for s in splits]
    f_vals = [fake_counts[s] for s in splits]
    
    plt.bar([pos - width/2 for pos in x], r_vals, width, label='Real', color='#2ecc71')
    plt.bar([pos + width/2 for pos in x], f_vals, width, label='Fake', color='#e74c3c')
    
    plt.xlabel('Split')
    plt.ylabel('Number of Images')
    plt.title('Dataset Split Distribution')
    plt.xticks(x, [s.upper() for s in splits])
    plt.legend()
    
    # Add values on top of bars
    for i, v in enumerate(r_vals):
        plt.text(i - width/2, v + 1, str(v), ha='center')
    for i, v in enumerate(f_vals):
        plt.text(i + width/2, v + 1, str(v), ha='center')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
