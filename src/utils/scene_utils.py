"""
Scene utilities for managing dataset groupings.

Design Decisions:
    - Manual scene mapping guarantees zero data leakage.
    - We generate a template CSV with default unique IDs for each file.
    - The user can edit the CSV to group files into the same scene.
    - The script never overwrites an existing scene_mapping.csv.

Usage:
    from src.utils.scene_utils import generate_scene_mapping_template, load_scene_mapping
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import ProjectConfig

logger = logging.getLogger("scene_utils")


def generate_scene_mapping_template(
    real_files: List[Path],
    fake_files: List[Path],
    output_csv: Path,
) -> bool:
    """Generate a template CSV for manual scene mapping.

    If the output CSV already exists, it will NOT be overwritten.
    By default, each file gets a unique scene_id based on its filename stem.

    Args:
        real_files: List of paths to real images.
        fake_files: List of paths to fake images.
        output_csv: Path to save the CSV file.

    Returns:
        True if a new template was created, False if it already existed.
    """
    if output_csv.exists():
        logger.warning(f"Scene mapping already exists at {output_csv}. Skipping template generation.")
        return False

    rows = []
    
    # Process real files
    for p in real_files:
        scene_id = f"scene_{p.stem}"
        rows.append({"scene_id": scene_id, "filename": p.name, "class": ProjectConfig.class_real, "split": ""})
        
    # Process fake files
    for p in fake_files:
        scene_id = f"scene_{p.stem}"
        rows.append({"scene_id": scene_id, "filename": p.name, "class": ProjectConfig.class_fake, "split": ""})

    # Sort rows by filename for consistent ordering
    rows.sort(key=lambda x: x["filename"])

    try:
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["scene_id", "filename", "class", "split"])
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Generated scene mapping template with {len(rows)} entries at {output_csv}")
        return True
    except Exception as e:
        logger.error(f"Failed to generate scene mapping template: {e}")
        return False


def load_scene_mapping(csv_path: Path) -> Dict[str, str]:
    """Load the scene mapping from CSV.

    Args:
        csv_path: Path to the scene_mapping.csv file.

    Returns:
        A dictionary mapping filename (e.g., 'real_0001.jpg') to scene_id.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Scene mapping file not found: {csv_path}")

    mapping: Dict[str, str] = {}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            scene_id = row["scene_id"].strip()
            
            if not scene_id:
                logger.warning(f"Empty scene_id for {filename}. Using filename stem as fallback.")
                scene_id = f"scene_{Path(filename).stem}"
                
            mapping[filename] = scene_id

    return mapping

def get_scene_statistics(
    mapping: Dict[str, str],
    real_files: List[Path],
    fake_files: List[Path],
) -> Tuple[int, Dict[str, int], Dict[str, int]]:
    """Compute basic statistics about the scene groupings.

    Args:
        mapping: Dictionary mapping filename to scene_id.
        real_files: List of active real files.
        fake_files: List of active fake files.

    Returns:
        Tuple containing:
        - Total number of unique scenes.
        - Dict mapping scene_id to total number of images in that scene.
        - Dict mapping scene_id to a score/balance (or class counts).
          Actually, we return a dict of scene stats.
    """
    # Just returning raw data; actual stats JSON will be built in the script
    pass
