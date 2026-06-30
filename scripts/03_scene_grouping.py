"""
Phase 2 (Step 1) - Scene Grouping & Template Generation.

This script scans the active dataset (excluding quarantine), generates a 
`scene_mapping.csv` template for manual review, and computes scene-level statistics.
It also generates `dataset_review.html` for visual inspection of the dataset.

Usage:
    python scripts/03_scene_grouping.py
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

from src.config import ProjectConfig
from src.utils.image_utils import get_image_files, get_image_info
from src.utils.logging_utils import setup_logger
from src.utils.scene_utils import generate_scene_mapping_template, load_scene_mapping


def generate_dataset_review_html(
    scene_stats: Dict[str, Any],
    html_path: Path,
) -> None:
    """Generate a simple HTML report summarizing the dataset and scenes."""
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dataset & Scene Review</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
        h1, h2, h3 {{ color: #ffffff; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background-color: #1e1e1e; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #333; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background-color: #1a1a1a; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #333; }}
        .summary-card h2 {{ margin: 0 0 10px 0; font-size: 2.5rem; }}
        .summary-card p {{ margin: 0; color: #aaa; text-transform: uppercase; font-size: 0.9rem; letter-spacing: 1px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background-color: #2a2a2a; color: #ccc; }}
        tr:hover {{ background-color: #252525; }}
        .tag-real {{ background-color: rgba(39, 174, 96, 0.2); color: #2ecc71; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; border: 1px solid rgba(39, 174, 96, 0.5); }}
        .tag-fake {{ background-color: rgba(192, 57, 43, 0.2); color: #e74c3c; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; border: 1px solid rgba(192, 57, 43, 0.5); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Dataset & Scene Review Report</h1>
        <p>Review of the current active dataset and scene groupings.</p>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h2>{scene_stats['total_images']}</h2>
                <p>Total Active Images</p>
            </div>
            <div class="summary-card">
                <h2>{scene_stats['total_scenes']}</h2>
                <p>Total Unique Scenes</p>
            </div>
            <div class="summary-card">
                <h2>{scene_stats['total_real']}</h2>
                <p>Real Images</p>
            </div>
            <div class="summary-card">
                <h2>{scene_stats['total_fake']}</h2>
                <p>Fake Images</p>
            </div>
        </div>

        <div class="card">
            <h3>Scene Distribution (Top 50 Largest Scenes)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Scene ID</th>
                        <th>Total Images</th>
                        <th>Real Count</th>
                        <th>Fake Count</th>
                    </tr>
                </thead>
                <tbody>
"""
    # Sort scenes by size (descending)
    sorted_scenes = sorted(scene_stats['scenes'].items(), key=lambda x: x[1]['total'], reverse=True)
    
    for scene_id, data in sorted_scenes[:50]:
        html_content += f"""
                    <tr>
                        <td>{scene_id}</td>
                        <td>{data['total']}</td>
                        <td><span class="tag-real">{data['real']}</span></td>
                        <td><span class="tag-fake">{data['fake']}</span></td>
                    </tr>
"""

    html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="card" style="background-color: #2c1a1a; border-color: #4a2a2a;">
            <h3>⚠️ Important Next Steps</h3>
            <p>1. Open <code>outputs/scene_mapping.csv</code> in a spreadsheet editor.</p>
            <p>2. Review the automatically generated unique scene IDs.</p>
            <p>3. If any images share the same underlying physical scene, manually edit their <code>scene_id</code> to be identical.</p>
            <p>4. Save the CSV. The next phase will respect your manual mappings to prevent data leakage.</p>
        </div>
    </div>
</body>
</html>
"""
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
    except Exception as e:
        logging.getLogger("scene_grouping").error(f"Failed to generate HTML report: {e}")


def main() -> None:
    config = ProjectConfig()
    logger = setup_logger("scene_grouping", config.logs_dir / "03_scene_grouping.log")
    
    logger.info("=" * 60)
    logger.info("PHASE 2 - SCENE GROUPING & DATASET REVIEW")
    logger.info("=" * 60)

    # 1. Discover files
    try:
        real_files = get_image_files(config.real_dir, config.image_extensions)
        fake_files = get_image_files(config.fake_dir, config.image_extensions)
    except FileNotFoundError as e:
        logger.error(f"Dataset directories missing: {e}")
        return

    logger.info(f"Discovered {len(real_files)} real and {len(fake_files)} fake images.")

    # 2. Generate or Load Scene Mapping
    csv_path = config.outputs_dir / "scene_mapping.csv"
    created = generate_scene_mapping_template(real_files, fake_files, csv_path)
    
    if created:
        logger.info(f"Generated new scene mapping template at {csv_path.name}")
    else:
        logger.info(f"Using existing scene mapping from {csv_path.name}")

    try:
        mapping = load_scene_mapping(csv_path)
    except Exception as e:
        logger.error(f"Failed to load scene mapping: {e}")
        return

    # 3. Compute Statistics
    scene_stats = {
        "total_images": len(real_files) + len(fake_files),
        "total_real": len(real_files),
        "total_fake": len(fake_files),
        "scenes": defaultdict(lambda: {"total": 0, "real": 0, "fake": 0})
    }

    # Verify all current files are in the mapping
    active_filenames = {p.name for p in real_files + fake_files}
    mapped_filenames = set(mapping.keys())
    
    missing_in_mapping = active_filenames - mapped_filenames
    if missing_in_mapping:
        logger.warning(f"⚠️ {len(missing_in_mapping)} active files are missing from scene_mapping.csv!")
        logger.warning("Please delete scene_mapping.csv to regenerate it, or manually add the missing entries.")
        # We will continue on warning as requested
    
    # Process Real
    for p in real_files:
        if p.name in mapping:
            s_id = mapping[p.name]
            scene_stats["scenes"][s_id]["total"] += 1
            scene_stats["scenes"][s_id]["real"] += 1
            
    # Process Fake
    for p in fake_files:
        if p.name in mapping:
            s_id = mapping[p.name]
            scene_stats["scenes"][s_id]["total"] += 1
            scene_stats["scenes"][s_id]["fake"] += 1

    scene_stats["total_scenes"] = len(scene_stats["scenes"])
    
    # Output JSON stats
    stats_path = config.outputs_dir / "scene_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(scene_stats, f, indent=2)
    logger.info(f"Saved scene statistics to {stats_path.name}")

    # 4. Check for anomalous components (Warnings only)
    max_scene_size = max([data["total"] for data in scene_stats["scenes"].values()]) if scene_stats["scenes"] else 0
    if max_scene_size > 10:
        logger.warning(f"⚠️ Large scene detected! The largest scene contains {max_scene_size} images. Please review scene_mapping.csv.")

    # 5. Generate HTML Review
    html_path = config.outputs_dir / "dataset_review.html"
    generate_dataset_review_html(scene_stats, html_path)
    logger.info(f"Saved dataset review report to {html_path.name}")
    
    logger.info("=" * 60)
    logger.info("SCENE GROUPING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
