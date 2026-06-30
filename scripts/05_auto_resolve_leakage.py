"""
Auto-resolve Leakage script.

Iteratively runs the dataset split and resolves perceptual leakage by merging scenes 
in the scene_mapping.csv file. Preserves manual mappings.

Usage:
    python scripts/05_auto_resolve_leakage.py
"""

import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

from src.config import ProjectConfig
from src.utils.logging_utils import setup_logger

def is_default_scene(scene_id: str) -> bool:
    """Check if a scene_id looks like the default generated ones."""
    return scene_id.startswith("scene_real_") or scene_id.startswith("scene_fake_")

def run_split() -> int:
    """Run the split script and return the exit code."""
    result = subprocess.run(
        [sys.executable, "-m", "scripts.04_split_dataset"],
        capture_output=True,
        text=True
    )
    return result.returncode

def generate_final_report(config: ProjectConfig, stats: dict, logger: logging.Logger):
    """Generate outputs/final_split_report.html."""
    html_path = config.outputs_dir / "final_split_report.html"
    
    # Calculate class balance
    total = stats["train"]["total"] + stats["val"]["total"] + stats["test"]["total"]
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Final Leakage-Free Split Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 20px; }}
        h1, h2, h3 {{ color: #ffffff; }}
        .card {{ background-color: #1e1e1e; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #333; }}
        .success {{ color: #2ecc71; font-weight: bold; font-size: 1.2rem; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background-color: #2a2a2a; color: #ccc; }}
    </style>
</head>
<body>
    <h1>🎯 Final Dataset Split Report</h1>
    
    <div class="card">
        <h3>Status: <span class="success">✅ ZERO LEAKAGE ACHIEVED</span></h3>
        <p>All exact and perceptual duplicate pairs have been grouped into identical scenes and remain within single splits.</p>
    </div>

    <div class="card">
        <h3>Split Statistics</h3>
        <table>
            <tr><th>Split</th><th>Total</th><th>Real</th><th>Fake</th></tr>
            <tr>
                <td><strong>TRAIN</strong></td>
                <td>{stats['train']['total']}</td>
                <td>{stats['train']['real']}</td>
                <td>{stats['train']['fake']}</td>
            </tr>
            <tr>
                <td><strong>VAL</strong></td>
                <td>{stats['val']['total']}</td>
                <td>{stats['val']['real']}</td>
                <td>{stats['val']['fake']}</td>
            </tr>
            <tr>
                <td><strong>TEST</strong></td>
                <td>{stats['test']['total']}</td>
                <td>{stats['test']['real']}</td>
                <td>{stats['test']['fake']}</td>
            </tr>
        </table>
    </div>
</body>
</html>
"""
    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Generated final split report at {html_path}")
    except Exception as e:
        logger.error(f"Failed to write final report: {e}")

def main():
    config = ProjectConfig()
    logger = setup_logger("auto_resolve", config.logs_dir / "05_auto_resolve.log")
    
    logger.info("=" * 60)
    logger.info("AUTO-RESOLVING LEAKAGE")
    logger.info("=" * 60)

    csv_path = config.outputs_dir / "scene_mapping.csv"
    val_report_path = config.outputs_dir / "validation_report.json"
    manual_review_path = config.outputs_dir / "manual_review.csv"
    stats_path = config.outputs_dir / "split_statistics.json"
    
    iteration = 1
    max_iterations = 20

    while iteration <= max_iterations:
        logger.info(f"--- Iteration {iteration} ---")
        
        # 1. Run Split
        code = run_split()
        if code != 0:
            logger.error("04_split_dataset failed!")
            return

        # 2. Check Validation Report
        if not val_report_path.exists():
            logger.error("validation_report.json not found!")
            return
            
        with open(val_report_path, "r", encoding="utf-8") as f:
            val_report = json.load(f)

        if val_report.get("is_valid"):
            logger.info("✅ SUCCESS! Leakage count = 0.")
            with open(stats_path, "r") as f:
                stats = json.load(f)
            generate_final_report(config, stats, logger)
            break

        # 3. We have leakage, read conflicts
        conflicts = val_report.get("conflicts", [])
        if not conflicts:
            logger.warning("is_valid is false but no conflicts found. Exiting.")
            break

        logger.info(f"Found {len(conflicts)} cross-split conflicts. Attempting to merge scenes...")

        # 4. Load Scene Mapping (with all rows so we can re-write it exactly)
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                rows.append(row)
                
        # Map filename -> row dict
        file_to_row = {r["filename"]: r for r in rows}
        
        # Quick lookup for current scene assignment
        def get_scene(filename: str) -> str:
            return file_to_row[filename]["scene_id"].strip() or f"scene_{Path(filename).stem}"

        # 5. Resolve Conflicts
        unresolved_conflicts = []
        changes_made = 0
        
        # We need to map old_scene_id -> new_scene_id
        scene_replacements = {}
        
        # To handle chained replacements (A->B, B->C), we define a find root function
        def find_root(s: str) -> str:
            while s in scene_replacements:
                s = scene_replacements[s]
            return s
            
        for conflict in conflicts:
            files = conflict.get("files", [])
            if len(files) < 2:
                continue
                
            # We want to merge all these files into one scene.
            s0 = find_root(get_scene(files[0]))
            
            for f in files[1:]:
                s_current = find_root(get_scene(f))
                if s_current == s0:
                    continue
                    
                # We need to merge s0 and s_current
                is_s0_def = is_default_scene(s0)
                is_sc_def = is_default_scene(s_current)
                
                if is_s0_def and is_sc_def:
                    # both default, pick s0
                    scene_replacements[s_current] = s0
                elif not is_s0_def and is_sc_def:
                    # s0 is custom, keep s0
                    scene_replacements[s_current] = s0
                elif is_s0_def and not is_sc_def:
                    # s_current is custom, keep s_current
                    scene_replacements[s0] = s_current
                    s0 = s_current # update root
                else:
                    # BOTH CUSTOM! Conflict!
                    logger.warning(f"Manual conflict between custom scenes {s0} and {s_current} for files {files[0]} and {f}")
                    unresolved_conflicts.append({
                        "file1": files[0], "scene1": s0,
                        "file2": f, "scene2": s_current,
                        "reason": "Both scenes have custom manual IDs"
                    })

        # Apply replacements to all rows
        for row in rows:
            orig_scene = row["scene_id"].strip()
            if not orig_scene:
                orig_scene = f"scene_{Path(row['filename']).stem}"
                
            new_scene = find_root(orig_scene)
            if new_scene != orig_scene:
                row["scene_id"] = new_scene
                changes_made += 1

        if unresolved_conflicts:
            # Write to manual_review.csv
            logger.warning(f"Writing {len(unresolved_conflicts)} unresolved conflicts to manual_review.csv")
            with open(manual_review_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["file1", "scene1", "file2", "scene2", "reason"])
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerows(unresolved_conflicts)
                
            if changes_made == 0:
                logger.error("No changes could be made automatically due to manual conflicts. Exiting.")
                break

        if changes_made > 0:
            logger.info(f"Merged scenes for {changes_made} files. Updating scene_mapping.csv...")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        else:
            logger.error("No changes made despite finding conflicts. Exiting loop.")
            break
            
        iteration += 1

    if iteration > max_iterations:
        logger.error("Reached maximum iterations without resolving all leakage.")

if __name__ == "__main__":
    main()
