#!/usr/bin/env python3
"""
01_dataset_inspector.py — Dataset Inspection & Organisation Pipeline.

This script performs a comprehensive inspection of the raw image dataset,
renames files with a consistent convention, detects quality issues, and
generates a machine-readable report plus human-readable visualisations.

Pipeline Steps:
    1. Scan raw image files in ``dataset/real/`` and ``dataset/fake/``.
    2. Validate every image (detect corrupted / broken files).
    3. Rename images to ``{class}_{NNNN}.jpg`` with zero-padded indices.
    4. Generate ``outputs/filename_mapping.csv``.
    5. Compute MD5 and perceptual hashes for duplicate detection.
    6. Detect exact duplicates, near-duplicates, and cross-class matches.
    7. Compute per-image statistics: resolution, blur, brightness, etc.
    8. Generate all visualisation plots to ``outputs/plots/``.
    9. Save ``outputs/dataset_report.json``.
    10. Print a human-readable summary to the console.

Design Decisions:
    - Single Responsibility: this script orchestrates; all reusable logic
      lives in ``src/utils/`` modules.
    - Idempotent: detects if renaming was already performed and skips it.
    - All paths resolved via ``ProjectConfig`` — no hardcoded strings.
    - Outputs both JSON (machine) and plots (human) for full traceability.

Usage:
    python scripts/01_dataset_inspector.py
    python scripts/01_dataset_inspector.py --dataset-dir /path/to/dataset
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Add project root to sys.path for standalone execution ────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import ProjectConfig
from src.utils.hash_utils import (
    compute_md5,
    compute_phash,
    find_cross_class_duplicates,
    find_exact_duplicates,
    find_perceptual_duplicates,
)
from src.utils.image_utils import (
    ImageInfo,
    compute_rgb_histogram,
    get_image_files,
    get_image_info,
    is_valid_image,
    load_image_cv2,
)
from src.utils.logging_utils import setup_logger
from src.utils.visualization import (
    plot_aspect_ratio_distribution,
    plot_blur_distribution,
    plot_brightness_distribution,
    plot_class_distribution,
    plot_file_size_distribution,
    plot_resolution_scatter,
    plot_rgb_histograms,
    plot_sample_grid,
)


# ── Naming convention pattern ────────────────────────────────────────
_RENAMED_PATTERN = re.compile(r"^(real|fake)_\d{4}\.jpg$")


class DatasetInspector:
    """Orchestrates the full dataset inspection and organisation pipeline.

    Attributes:
        config: Project configuration object.
        logger: Structured logger for this module.
        image_data: Per-image metadata collected during inspection.
        corrupted_files: Paths of images that failed validation.
        filename_mapping: Records of old → new filename mappings.
        exact_dupes: Groups of byte-identical files.
        perceptual_dupes_real: Near-duplicate pairs within the real class.
        perceptual_dupes_fake: Near-duplicate pairs within the fake class.
        cross_class_matches: Perceptually similar pairs across classes.
    """

    def __init__(self, config: ProjectConfig) -> None:
        """Initialise the inspector with project configuration.

        Args:
            config: ``ProjectConfig`` instance defining paths and thresholds.
        """
        self.config = config
        self.config.ensure_directories()
        self.logger = setup_logger(
            "dataset_inspector",
            log_dir=config.logs_dir,
        )

        # Internal state — populated by pipeline steps
        self.image_data: List[Dict[str, Any]] = []
        self.corrupted_files: List[Dict[str, str]] = []
        self.filename_mapping: List[Dict[str, str]] = []
        self.exact_dupes: List[List[str]] = []
        self.perceptual_dupes_real: List[Tuple[str, str, int]] = []
        self.perceptual_dupes_fake: List[Tuple[str, str, int]] = []
        self.cross_class_matches: List[Tuple[str, str, int]] = []
        self.rgb_histograms_real: List[Dict[str, np.ndarray]] = []
        self.rgb_histograms_fake: List[Dict[str, np.ndarray]] = []

        # Output directories
        self.plots_dir = config.outputs_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    #  PUBLIC API
    # ================================================================

    def run(self) -> Dict[str, Any]:
        """Execute the full inspection pipeline.

        Returns:
            The complete dataset report as a dictionary.
        """
        self.logger.info("=" * 60)
        self.logger.info("DATASET INSPECTION PIPELINE — START")
        self.logger.info("=" * 60)

        # Step 1 & 2: Scan and validate
        real_files, fake_files = self._scan_and_validate()

        # Step 3: Rename images
        real_files, fake_files = self._rename_images(real_files, fake_files)

        # Step 4: Save filename mapping
        self._save_filename_mapping()

        # Step 5 & 6: Hash and find duplicates
        self._detect_duplicates(real_files, fake_files)

        # Step 7: Compute per-image statistics
        self._compute_statistics(real_files, fake_files)

        # Step 8: Generate visualisations
        self._generate_visualisations()

        # Step 9: Build and save report
        report = self._build_report()
        self._save_report(report)

        # Step 10: Print summary
        self._print_summary(report)

        self.logger.info("=" * 60)
        self.logger.info("DATASET INSPECTION PIPELINE — COMPLETE")
        self.logger.info("=" * 60)
        return report

    # ================================================================
    #  PRIVATE PIPELINE STEPS
    # ================================================================

    def _scan_and_validate(self) -> Tuple[List[Path], List[Path]]:
        """Scan dataset directories and validate every image.

        Returns:
            Tuple of (valid_real_files, valid_fake_files).
        """
        self.logger.info("Step 1/8: Scanning and validating images...")

        raw_real = get_image_files(self.config.real_dir, self.config.image_extensions)
        raw_fake = get_image_files(self.config.fake_dir, self.config.image_extensions)
        self.logger.info(
            f"  Found {len(raw_real)} files in real/, {len(raw_fake)} files in fake/"
        )

        valid_real: List[Path] = []
        valid_fake: List[Path] = []

        for path in tqdm(raw_real, desc="Validating real", unit="img"):
            if is_valid_image(path):
                valid_real.append(path)
            else:
                self.corrupted_files.append({
                    "path": str(path),
                    "class": self.config.class_real,
                    "filename": path.name,
                })
                self.logger.warning(f"  CORRUPTED: {path.name}")

        for path in tqdm(raw_fake, desc="Validating fake", unit="img"):
            if is_valid_image(path):
                valid_fake.append(path)
            else:
                self.corrupted_files.append({
                    "path": str(path),
                    "class": self.config.class_fake,
                    "filename": path.name,
                })
                self.logger.warning(f"  CORRUPTED: {path.name}")

        self.logger.info(
            f"  Valid: {len(valid_real)} real, {len(valid_fake)} fake  |  "
            f"Corrupted: {len(self.corrupted_files)}"
        )
        return valid_real, valid_fake

    def _is_already_renamed(self, files: List[Path]) -> bool:
        """Check if all files already follow the naming convention.

        Args:
            files: List of image file paths.

        Returns:
            ``True`` if every file matches ``{class}_NNNN.jpg``.
        """
        if not files:
            return True
        return all(_RENAMED_PATTERN.match(p.name) for p in files)

    def _rename_images(
        self,
        real_files: List[Path],
        fake_files: List[Path],
    ) -> Tuple[List[Path], List[Path]]:
        """Rename images to a consistent ``{class}_{NNNN}.jpg`` convention.

        If images are already renamed, this step is skipped.  The mapping
        CSV from a previous run (if present) is preserved.

        Images are copied (not moved) to the new name to preserve originals
        during development.  In production, use ``rename`` instead.

        Args:
            real_files: Valid real image paths.
            fake_files: Valid fake image paths.

        Returns:
            Tuple of (new_real_paths, new_fake_paths).
        """
        self.logger.info("Step 2/8: Renaming images...")

        mapping_path = self.config.outputs_dir / "filename_mapping.csv"

        # Check if already renamed
        if self._is_already_renamed(real_files) and self._is_already_renamed(fake_files):
            self.logger.info("  Images already follow naming convention — skipping rename.")
            # Load existing mapping if available
            if mapping_path.exists():
                df = pd.read_csv(mapping_path)
                self.filename_mapping = df.to_dict("records")
            return real_files, fake_files

        new_real = self._rename_class(
            real_files, self.config.real_dir, self.config.class_real
        )
        new_fake = self._rename_class(
            fake_files, self.config.fake_dir, self.config.class_fake
        )

        self.logger.info(
            f"  Renamed {len(new_real)} real + {len(new_fake)} fake images."
        )
        return new_real, new_fake

    def _rename_class(
        self,
        files: List[Path],
        directory: Path,
        class_name: str,
    ) -> List[Path]:
        """Rename files for a single class with zero-padded numbering.

        Uses a two-phase approach to avoid naming collisions:
          Phase 1: Rename all to temporary UUIDs.
          Phase 2: Rename from temporary to final ``{class}_{NNNN}.jpg``.

        Args:
            files: Sorted list of current file paths.
            directory: Parent directory for the files.
            class_name: Class label (``"real"`` or ``"fake"``).

        Returns:
            List of new file paths in the same order.
        """
        import uuid

        new_paths: List[Path] = []

        # Phase 1: Move to temp names to avoid collisions
        temp_paths: List[Tuple[Path, Path]] = []
        for path in files:
            temp_name = f"_tmp_{uuid.uuid4().hex}{path.suffix}"
            temp_path = directory / temp_name
            path.rename(temp_path)
            temp_paths.append((temp_path, path))

        # Phase 2: Rename to final names
        for idx, (temp_path, original_path) in enumerate(temp_paths, start=1):
            new_name = f"{class_name}_{idx:04d}.jpg"
            new_path = directory / new_name
            temp_path.rename(new_path)
            new_paths.append(new_path)
            self.filename_mapping.append({
                "old_filename": original_path.name,
                "new_filename": new_name,
                "class": class_name,
            })

        return new_paths

    def _save_filename_mapping(self) -> None:
        """Save the old-to-new filename mapping as CSV."""
        if not self.filename_mapping:
            self.logger.info("  No filename mapping to save (rename was skipped).")
            return

        mapping_path = self.config.outputs_dir / "filename_mapping.csv"
        df = pd.DataFrame(self.filename_mapping)
        df.to_csv(mapping_path, index=False)
        self.logger.info(f"  Saved filename mapping → {mapping_path}")

    def _detect_duplicates(
        self,
        real_files: List[Path],
        fake_files: List[Path],
    ) -> None:
        """Detect exact and perceptual duplicates within and across classes.

        Args:
            real_files: Paths to real images (post-rename).
            fake_files: Paths to fake images (post-rename).
        """
        self.logger.info("Step 3/8: Computing hashes and detecting duplicates...")

        all_files = real_files + fake_files

        # Compute MD5 hashes
        md5_hashes: Dict[Path, str] = {}
        for path in tqdm(all_files, desc="MD5 hashing", unit="img"):
            md5_hashes[path] = compute_md5(path)

        # Compute perceptual hashes
        phash_real: Dict[Path, str] = {}
        phash_fake: Dict[Path, str] = {}

        for path in tqdm(real_files, desc="pHash (real)", unit="img"):
            h = compute_phash(path)
            if h is not None:
                phash_real[path] = h

        for path in tqdm(fake_files, desc="pHash (fake)", unit="img"):
            h = compute_phash(path)
            if h is not None:
                phash_fake[path] = h

        # Find exact duplicates (across all files)
        exact_groups = find_exact_duplicates(md5_hashes)
        self.exact_dupes = [
            [str(p) for p in group] for group in exact_groups
        ]
        if self.exact_dupes:
            self.logger.warning(
                f"  ⚠ Found {len(self.exact_dupes)} group(s) of exact duplicates!"
            )
            for group in self.exact_dupes:
                names = [Path(p).name for p in group]
                self.logger.warning(f"    Exact duplicate group: {names}")

        # Find perceptual duplicates within each class
        threshold = self.config.phash_threshold
        perceptual_real = find_perceptual_duplicates(phash_real, threshold=threshold)
        perceptual_fake = find_perceptual_duplicates(phash_fake, threshold=threshold)
        self.perceptual_dupes_real = [
            (str(a), str(b), d) for a, b, d in perceptual_real
        ]
        self.perceptual_dupes_fake = [
            (str(a), str(b), d) for a, b, d in perceptual_fake
        ]

        if self.perceptual_dupes_real:
            self.logger.info(
                f"  Near-duplicates within real: {len(self.perceptual_dupes_real)} pair(s)"
            )
        if self.perceptual_dupes_fake:
            self.logger.info(
                f"  Near-duplicates within fake: {len(self.perceptual_dupes_fake)} pair(s)"
            )

        # Find cross-class matches (data leakage risk)
        cross_matches = find_cross_class_duplicates(
            phash_real, phash_fake, threshold=threshold
        )
        self.cross_class_matches = [
            (str(a), str(b), d) for a, b, d in cross_matches
        ]

        if self.cross_class_matches:
            self.logger.warning(
                f"  ⚠ DATA LEAKAGE RISK: {len(self.cross_class_matches)} "
                f"cross-class match(es) detected!"
            )
            for ra, fb, dist in self.cross_class_matches:
                self.logger.warning(
                    f"    {Path(ra).name} ↔ {Path(fb).name}  (distance={dist})"
                )
        else:
            self.logger.info("  No cross-class perceptual matches detected.")

    def _compute_statistics(
        self,
        real_files: List[Path],
        fake_files: List[Path],
    ) -> None:
        """Compute per-image metadata and RGB histograms.

        Args:
            real_files: Paths to real images.
            fake_files: Paths to fake images.
        """
        self.logger.info("Step 4/8: Computing image statistics...")

        for path in tqdm(real_files, desc="Analysing real", unit="img"):
            info = get_image_info(path)
            if info is not None:
                self.image_data.append({
                    "path": str(info.path),
                    "filename": info.path.name,
                    "class": self.config.class_real,
                    "width": info.width,
                    "height": info.height,
                    "channels": info.channels,
                    "file_size_bytes": info.file_size_bytes,
                    "aspect_ratio": info.aspect_ratio,
                    "blur_score": info.blur_score,
                    "brightness": info.brightness,
                })
                # RGB histogram
                img = load_image_cv2(path)
                if img is not None:
                    self.rgb_histograms_real.append(compute_rgb_histogram(img))

        for path in tqdm(fake_files, desc="Analysing fake", unit="img"):
            info = get_image_info(path)
            if info is not None:
                self.image_data.append({
                    "path": str(info.path),
                    "filename": info.path.name,
                    "class": self.config.class_fake,
                    "width": info.width,
                    "height": info.height,
                    "channels": info.channels,
                    "file_size_bytes": info.file_size_bytes,
                    "aspect_ratio": info.aspect_ratio,
                    "blur_score": info.blur_score,
                    "brightness": info.brightness,
                })
                img = load_image_cv2(path)
                if img is not None:
                    self.rgb_histograms_fake.append(compute_rgb_histogram(img))

        self.logger.info(f"  Collected statistics for {len(self.image_data)} images.")

    def _generate_visualisations(self) -> None:
        """Generate all visualisation plots and save to outputs/plots/."""
        self.logger.info("Step 5/8: Generating visualisations...")

        if not self.image_data:
            self.logger.warning("  No image data — skipping visualisations.")
            return

        df = pd.DataFrame(self.image_data)

        # 1. Class distribution
        counts = df["class"].value_counts().to_dict()
        plot_class_distribution(counts, self.plots_dir)
        self.logger.info("  ✓ class_distribution.png")

        # 2. Resolution scatter
        plot_resolution_scatter(df, self.plots_dir)
        self.logger.info("  ✓ resolution_scatter.png")

        # 3. Aspect ratio distribution
        plot_aspect_ratio_distribution(df, self.plots_dir)
        self.logger.info("  ✓ aspect_ratio_distribution.png")

        # 4. Blur distribution
        plot_blur_distribution(df, self.plots_dir)
        self.logger.info("  ✓ blur_distribution.png")

        # 5. Brightness distribution
        plot_brightness_distribution(df, self.plots_dir)
        self.logger.info("  ✓ brightness_distribution.png")

        # 6. File size distribution
        plot_file_size_distribution(df, self.plots_dir)
        self.logger.info("  ✓ file_size_distribution.png")

        # 7. RGB histograms
        plot_rgb_histograms(
            self.rgb_histograms_real, self.rgb_histograms_fake, self.plots_dir
        )
        self.logger.info("  ✓ rgb_histograms.png")

        # 8. Sample grid
        image_paths = [(Path(d["path"]), d["class"]) for d in self.image_data]
        plot_sample_grid(image_paths, self.plots_dir, seed=self.config.random_seed)
        self.logger.info("  ✓ sample_grid.png")

    def _build_report(self) -> Dict[str, Any]:
        """Compile all inspection results into a structured report dictionary.

        Returns:
            Comprehensive dataset report.
        """
        self.logger.info("Step 6/8: Building report...")

        df = pd.DataFrame(self.image_data) if self.image_data else pd.DataFrame()

        # Per-class counts
        real_count = len(df[df["class"] == self.config.class_real]) if len(df) > 0 else 0
        fake_count = len(df[df["class"] == self.config.class_fake]) if len(df) > 0 else 0
        total = real_count + fake_count

        # Resolution statistics
        resolution_stats = {}
        if len(df) > 0:
            resolution_stats = {
                "min_width": int(df["width"].min()),
                "max_width": int(df["width"].max()),
                "mean_width": round(float(df["width"].mean()), 1),
                "min_height": int(df["height"].min()),
                "max_height": int(df["height"].max()),
                "mean_height": round(float(df["height"].mean()), 1),
                "unique_resolutions": len(df.groupby(["width", "height"])),
            }

        # File size statistics
        file_size_stats = {}
        if len(df) > 0:
            sizes_kb = df["file_size_bytes"] / 1024
            file_size_stats = {
                "min_kb": round(float(sizes_kb.min()), 1),
                "max_kb": round(float(sizes_kb.max()), 1),
                "mean_kb": round(float(sizes_kb.mean()), 1),
                "median_kb": round(float(sizes_kb.median()), 1),
                "total_mb": round(float(sizes_kb.sum() / 1024), 2),
            }

        # Blur statistics
        blur_stats = {}
        if len(df) > 0:
            blur_stats = {
                "min": round(float(df["blur_score"].min()), 2),
                "max": round(float(df["blur_score"].max()), 2),
                "mean": round(float(df["blur_score"].mean()), 2),
                "median": round(float(df["blur_score"].median()), 2),
                "blurry_count": int(
                    (df["blur_score"] < self.config.blur_threshold).sum()
                ),
                "blurry_threshold": self.config.blur_threshold,
            }

        # Brightness statistics
        brightness_stats = {}
        if len(df) > 0:
            brightness_stats = {
                "min": round(float(df["brightness"].min()), 2),
                "max": round(float(df["brightness"].max()), 2),
                "mean": round(float(df["brightness"].mean()), 2),
                "median": round(float(df["brightness"].median()), 2),
            }
            # Per-class brightness
            for cls in [self.config.class_real, self.config.class_fake]:
                subset = df[df["class"] == cls]
                if len(subset) > 0:
                    brightness_stats[f"mean_{cls}"] = round(
                        float(subset["brightness"].mean()), 2
                    )

        # Aspect ratio statistics
        aspect_stats = {}
        if len(df) > 0:
            aspect_stats = {
                "min": round(float(df["aspect_ratio"].min()), 4),
                "max": round(float(df["aspect_ratio"].max()), 4),
                "mean": round(float(df["aspect_ratio"].mean()), 4),
                "unique_ratios": len(df["aspect_ratio"].unique()),
            }

        # Imbalance report
        imbalance = {}
        if total > 0:
            majority = max(real_count, fake_count)
            minority = min(real_count, fake_count)
            imbalance = {
                "majority_class": self.config.class_fake if fake_count >= real_count else self.config.class_real,
                "minority_class": self.config.class_real if fake_count >= real_count else self.config.class_fake,
                "majority_count": majority,
                "minority_count": minority,
                "imbalance_ratio": round(majority / minority, 3) if minority > 0 else float("inf"),
                "is_balanced": abs(real_count - fake_count) / total < 0.1,
            }

        report: Dict[str, Any] = {
            "summary": {
                "total_images": total,
                "real_count": real_count,
                "fake_count": fake_count,
                "corrupted_count": len(self.corrupted_files),
            },
            "duplicates": {
                "exact_duplicate_groups": len(self.exact_dupes),
                "exact_duplicates": self.exact_dupes,
                "perceptual_duplicates_real": len(self.perceptual_dupes_real),
                "perceptual_duplicates_fake": len(self.perceptual_dupes_fake),
                "cross_class_matches": len(self.cross_class_matches),
                "cross_class_details": [
                    {
                        "real_image": Path(a).name,
                        "fake_image": Path(b).name,
                        "hamming_distance": d,
                    }
                    for a, b, d in self.cross_class_matches
                ],
            },
            "corrupted_files": self.corrupted_files,
            "resolution_statistics": resolution_stats,
            "file_size_statistics": file_size_stats,
            "blur_statistics": blur_stats,
            "brightness_statistics": brightness_stats,
            "aspect_ratio_statistics": aspect_stats,
            "class_imbalance": imbalance,
            "per_image_data": [
                {k: v for k, v in d.items() if k != "path"}
                for d in self.image_data
            ],
        }

        return report

    def _save_report(self, report: Dict[str, Any]) -> None:
        """Save the inspection report as JSON.

        Args:
            report: Complete dataset report dictionary.
        """
        report_path = self.config.outputs_dir / "dataset_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        self.logger.info(f"Step 7/8: Saved report → {report_path}")

    def _print_summary(self, report: Dict[str, Any]) -> None:
        """Print a human-readable summary to the console.

        Args:
            report: Complete dataset report dictionary.
        """
        self.logger.info("Step 8/8: Summary")
        self.logger.info("-" * 50)

        s = report["summary"]
        self.logger.info(f"  Total images:     {s['total_images']}")
        self.logger.info(f"  Real:             {s['real_count']}")
        self.logger.info(f"  Fake:             {s['fake_count']}")
        self.logger.info(f"  Corrupted:        {s['corrupted_count']}")

        d = report["duplicates"]
        self.logger.info(f"  Exact dupes:      {d['exact_duplicate_groups']} group(s)")
        self.logger.info(f"  Near-dupes real:  {d['perceptual_duplicates_real']} pair(s)")
        self.logger.info(f"  Near-dupes fake:  {d['perceptual_duplicates_fake']} pair(s)")
        self.logger.info(f"  Cross-class:      {d['cross_class_matches']} match(es)")

        if report["resolution_statistics"]:
            r = report["resolution_statistics"]
            self.logger.info(
                f"  Resolution range: {r['min_width']}×{r['min_height']} → "
                f"{r['max_width']}×{r['max_height']}"
            )
            self.logger.info(f"  Unique resolutions: {r['unique_resolutions']}")

        if report["file_size_statistics"]:
            fs = report["file_size_statistics"]
            self.logger.info(
                f"  File size:        {fs['min_kb']:.0f} KB → {fs['max_kb']:.0f} KB "
                f"(mean {fs['mean_kb']:.0f} KB)"
            )
            self.logger.info(f"  Total dataset:    {fs['total_mb']:.1f} MB")

        if report["blur_statistics"]:
            b = report["blur_statistics"]
            self.logger.info(
                f"  Blur scores:      {b['min']:.1f} → {b['max']:.1f} "
                f"(mean {b['mean']:.1f})"
            )
            self.logger.info(
                f"  Blurry images:    {b['blurry_count']} "
                f"(threshold < {b['blurry_threshold']})"
            )

        if report["brightness_statistics"]:
            br = report["brightness_statistics"]
            self.logger.info(
                f"  Brightness:       {br['min']:.1f} → {br['max']:.1f} "
                f"(mean {br['mean']:.1f})"
            )

        if report["class_imbalance"]:
            ci = report["class_imbalance"]
            status = "✓ BALANCED" if ci["is_balanced"] else "⚠ IMBALANCED"
            self.logger.info(
                f"  Class balance:    {status} "
                f"(ratio {ci['imbalance_ratio']:.2f}:1)"
            )

        self.logger.info("-" * 50)


# ======================================================================
#  CLI ENTRY POINT
# ======================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Dataset Inspection & Organisation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Override project root directory (default: auto-detected).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the dataset inspector script."""
    args = parse_args()

    if args.project_root is not None:
        config = ProjectConfig(project_root=args.project_root.resolve())
    else:
        config = ProjectConfig()

    inspector = DatasetInspector(config)
    inspector.run()


if __name__ == "__main__":
    main()
