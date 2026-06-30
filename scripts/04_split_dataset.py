"""
Phase 2 (Step 2-5) - Dataset Splitting and Validation.

This script takes the dataset and scene groupings, applies a greedy split,
constructs the final dataset directory structure (train/val/test),
and validates that no data leakage exists.

Usage:
    python scripts/04_split_dataset.py
"""

import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

from src.config import ProjectConfig
from src.utils.image_utils import get_image_files
from src.utils.logging_utils import setup_logger
from src.utils.scene_utils import load_scene_mapping
from src.utils.split_utils import greedy_scene_splitter, plot_split_statistics, verify_no_leakage


def generate_split_review_html(
    split_stats: dict,
    html_path: Path,
) -> None:
    """Generate a HTML report summarizing the dataset splits."""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dataset Split Review</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
        h1, h2, h3 {{ color: #ffffff; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background-color: #1e1e1e; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #333; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background-color: #1a1a1a; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #333; }}
        .summary-card h2 {{ margin: 0 0 10px 0; font-size: 2.5rem; }}
        .summary-card p {{ margin: 0; color: #aaa; text-transform: uppercase; font-size: 0.9rem; letter-spacing: 1px; }}
        img {{ max-width: 100%; border-radius: 4px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background-color: #2a2a2a; color: #ccc; }}
        tr:hover {{ background-color: #252525; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>✂️ Dataset Split Review Report</h1>
        <p>Review of the stratified, scene-aware train/val/test splits.</p>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h2>{split_stats['train']['total']}</h2>
                <p>Train Images</p>
            </div>
            <div class="summary-card">
                <h2>{split_stats['val']['total']}</h2>
                <p>Validation Images</p>
            </div>
            <div class="summary-card">
                <h2>{split_stats['test']['total']}</h2>
                <p>Test Images</p>
            </div>
        </div>
        
        <div class="card">
            <h3>Class Distribution</h3>
            <table>
                <thead>
                    <tr>
                        <th>Split</th>
                        <th>Real</th>
                        <th>Fake</th>
                        <th>Ratio (Real/Fake)</th>
                    </tr>
                </thead>
                <tbody>
"""
    for split in ["train", "val", "test"]:
        r = split_stats[split]['real']
        f = split_stats[split]['fake']
        ratio = f"{r/f:.2f}" if f > 0 else "N/A"
        html_content += f"""
                    <tr>
                        <td><strong>{split.upper()}</strong></td>
                        <td>{r}</td>
                        <td>{f}</td>
                        <td>{ratio}</td>
                    </tr>
"""
    html_content += """
                </tbody>
            </table>
        </div>

        <div class="card">
            <h3>Split Visualization</h3>
            <img src="split_visualization.png" alt="Split Distribution Chart" />
        </div>
    </div>
</body>
</html>
"""
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        logging.getLogger("split_dataset").error(f"Failed to generate HTML report: {e}")


def copy_or_symlink(src: Path, dst: Path, use_symlink: bool) -> None:
    """Link or copy a file safely."""
    if dst.exists():
        dst.unlink()
        
    if use_symlink:
        try:
            os.symlink(src, dst)
            return
        except OSError:
            # Fallback to copy if symlink fails (e.g. on Windows without admin)
            pass
            
    shutil.copy2(src, dst)

def main() -> None:
    config = ProjectConfig()
    logger = setup_logger("split_dataset", config.logs_dir / "04_split_dataset.log")
    
    logger.info("=" * 60)
    logger.info("PHASE 2 - DATASET SPLITTING")
    logger.info("=" * 60)

    # 1. Discover files
    try:
        real_files = get_image_files(config.real_dir, config.image_extensions)
        fake_files = get_image_files(config.fake_dir, config.image_extensions)
    except FileNotFoundError as e:
        logger.error(f"Dataset directories missing: {e}")
        return

    csv_path = config.outputs_dir / "scene_mapping.csv"
    if not csv_path.exists():
        logger.error(f"Scene mapping missing: {csv_path.name}. Run 03_scene_grouping.py first.")
        return

    try:
        scene_mapping = load_scene_mapping(csv_path)
    except Exception as e:
        logger.error(f"Failed to load scene mapping: {e}")
        return

    # 2. Greedy Scene-Aware Splitting
    logger.info("Running greedy scene-aware splitting...")
    file_to_split = greedy_scene_splitter(
        scene_mapping=scene_mapping,
        real_files=real_files,
        fake_files=fake_files,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    # 3. Create Dataset Directories
    for split in ["train", "validation", "test"]:
        for class_dir in [config.class_real, config.class_fake]:
            (config.dataset_dir / split / class_dir).mkdir(parents=True, exist_ok=True)

    # 4. Populate Directories
    logger.info(f"Populating split directories (use_symlinks={config.use_symlinks})...")
    split_map = {"train": "train", "val": "validation", "test": "test"}
    
    for p in real_files + fake_files:
        if p.name not in file_to_split:
            continue
        
        split = file_to_split[p.name]
        dest_split = split_map[split]
        class_name = config.class_real if p.parent.name == config.class_real else config.class_fake
        
        dest_path = config.dataset_dir / dest_split / class_name / p.name
        copy_or_symlink(p, dest_path, config.use_symlinks)

    logger.info("Split population complete.")

    # 5. Compute and Output Statistics
    split_stats = {
        "train": {"real": 0, "fake": 0, "total": 0},
        "val": {"real": 0, "fake": 0, "total": 0},
        "test": {"real": 0, "fake": 0, "total": 0},
    }
    
    for p in real_files:
        if p.name in file_to_split:
            split_stats[file_to_split[p.name]]["real"] += 1
            split_stats[file_to_split[p.name]]["total"] += 1
            
    for p in fake_files:
        if p.name in file_to_split:
            split_stats[file_to_split[p.name]]["fake"] += 1
            split_stats[file_to_split[p.name]]["total"] += 1

    stats_path = config.outputs_dir / "split_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2)
        
    viz_path = config.outputs_dir / "split_visualization.png"
    plot_split_statistics(file_to_split, real_files, fake_files, viz_path)
    
    html_path = config.outputs_dir / "split_review.html"
    generate_split_review_html(split_stats, html_path)
    
    logger.info(f"Saved split statistics to {stats_path.name}")
    logger.info(f"Saved split visualization to {viz_path.name}")
    logger.info(f"Saved split review report to {html_path.name}")

    # 6. Leakage Validation
    logger.info("Running leakage validation...")
    is_valid, issues, conflicts = verify_no_leakage(file_to_split, real_files + fake_files, config.phash_threshold)
    
    val_report = {
        "is_valid": is_valid,
        "issues": issues,
        "conflicts": conflicts,
    }
    
    val_path = config.outputs_dir / "validation_report.json"
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)
        
    if not is_valid:
        logger.warning(f"⚠️ LEAKAGE DETECTED! Found {len(issues)} issues. Check validation_report.json")
        for iss in issues:
            logger.warning(iss)
    else:
        logger.info("✅ LEAKAGE VALIDATION PASSED: Zero exact or perceptual duplicates across splits.")

    logger.info("=" * 60)
    logger.info("DATASET SPLITTING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
