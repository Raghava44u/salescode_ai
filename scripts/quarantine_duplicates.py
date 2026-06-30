#!/usr/bin/env python3
"""
quarantine_duplicates.py — Duplicate Inspection, Quarantine & Report Generator.

Scans the dataset for exact and near-duplicate image pairs, generates
side-by-side montage comparisons, moves clearly suspicious images to a
quarantine directory, and produces a self-contained HTML report for manual
review by the engineer.

Pipeline Steps:
    1. Scan all images and compute MD5 + perceptual hashes.
    2. Identify exact cross-class duplicates (byte-identical).
    3. Identify perceptual cross-class matches (same scene, different bytes).
    4. Identify within-class near-duplicates.
    5. Generate side-by-side montage images for every pair.
    6. Generate a combined contact sheet PNG.
    7. Move exact cross-class duplicate fakes to ``dataset/quarantine/``.
    8. Generate ``outputs/duplicate_report.html`` with embedded images.

Design Decisions:
    - Only exact byte-identical cross-class duplicates are quarantined.
      Perceptual matches may be valid paired data (same scene, real + fake).
    - Montages are created BEFORE any files are moved, so the HTML always
      shows the original state of the data.
    - The HTML embeds images as base64 data URIs for full portability.
    - No image is permanently deleted — quarantine is a reversible staging area.

Usage:
    python scripts/quarantine_duplicates.py
"""

import base64
import io
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ── Add project root to sys.path ─────────────────────────────────────
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
from src.utils.image_utils import get_image_files, get_image_info
from src.utils.logging_utils import setup_logger


# ── Constants ─────────────────────────────────────────────────────────

MONTAGE_HEIGHT = 300           # Pixel height for each image in a montage
SEPARATOR_WIDTH = 10           # Pixel width of the vertical divider
LABEL_HEIGHT = 80              # Pixel height reserved for text labels
CONTACT_SHEET_COLS = 1         # Contact sheet columns (1 = vertical stack)

SEVERITY_COLOURS_BGR = {
    "critical": (71, 29, 231),   # #E71D47 coral-red
    "warning":  (0, 165, 255),   # #FFA500 orange
    "info":     (180, 196, 36),  # #24C4B4 teal
}

SEVERITY_LABELS = {
    "critical": "EXACT DUPLICATE — QUARANTINED",
    "warning":  "PERCEPTUAL MATCH — REVIEW REQUIRED",
    "info":     "WITHIN-CLASS NEAR-DUPLICATE — REVIEW REQUIRED",
}


# ── Data containers ──────────────────────────────────────────────────

@dataclass
class ImageMeta:
    """Lightweight metadata for an image used in the report."""

    path: Path
    filename: str
    class_name: str
    file_size_kb: float
    width: int
    height: int
    blur_score: float
    brightness: float
    md5: str
    phash: str


@dataclass
class DuplicatePair:
    """A pair of images flagged as duplicates or near-duplicates."""

    image_a: ImageMeta
    image_b: ImageMeta
    distance: int                  # Hamming distance (0 = identical hash)
    is_exact_md5: bool             # True if byte-identical
    severity: str                  # "critical" | "warning" | "info"
    category: str                  # Human-readable category label
    action: str                    # Action taken (e.g. "quarantined")
    montage_path: Optional[Path] = None  # Path to saved montage image


# ── Montage creation ─────────────────────────────────────────────────

def _resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """Resize an image to a target height, preserving aspect ratio.

    Args:
        image: HWC numpy array.
        target_height: Desired height in pixels.

    Returns:
        Resized image.
    """
    h, w = image.shape[:2]
    if h == 0:
        return image
    scale = target_height / h
    new_w = max(1, int(w * scale))
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)


def _put_text_bg(
    canvas: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float = 0.45,
    colour: Tuple[int, int, int] = (220, 220, 220),
    bg_colour: Tuple[int, int, int] = (40, 40, 40),
    thickness: int = 1,
) -> None:
    """Draw text with a filled background rectangle for readability.

    Args:
        canvas: Image to draw on (modified in-place).
        text: Text string to render.
        origin: Bottom-left corner of the text baseline (x, y).
        font_scale: Font scale factor.
        colour: Text colour in BGR.
        bg_colour: Background rectangle colour in BGR.
        thickness: Text thickness.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    cv2.rectangle(canvas, (x, y - th - 4), (x + tw + 4, y + baseline + 2), bg_colour, -1)
    cv2.putText(canvas, text, (x + 2, y), font, font_scale, colour, thickness, cv2.LINE_AA)


def create_pair_montage(
    img_a_path: Path,
    img_b_path: Path,
    meta_a: ImageMeta,
    meta_b: ImageMeta,
    severity: str,
    distance: int,
    is_exact: bool,
) -> np.ndarray:
    """Create a side-by-side comparison montage of two images.

    Renders both images at equal height with metadata labels,
    severity-coloured header bar, and a central divider.

    Args:
        img_a_path: Path to the first image.
        img_b_path: Path to the second image.
        meta_a: Metadata for the first image.
        meta_b: Metadata for the second image.
        severity: One of ``"critical"``, ``"warning"``, ``"info"``.
        distance: Hamming distance between perceptual hashes.
        is_exact: Whether the pair is byte-identical.

    Returns:
        HWC numpy array of the montage image (BGR).
    """
    # Load and resize images
    img_a = cv2.imread(str(img_a_path))
    img_b = cv2.imread(str(img_b_path))

    if img_a is None:
        img_a = np.zeros((MONTAGE_HEIGHT, 300, 3), dtype=np.uint8)
        _put_text_bg(img_a, "IMAGE NOT FOUND", (20, MONTAGE_HEIGHT // 2))
    if img_b is None:
        img_b = np.zeros((MONTAGE_HEIGHT, 300, 3), dtype=np.uint8)
        _put_text_bg(img_b, "IMAGE NOT FOUND", (20, MONTAGE_HEIGHT // 2))

    img_a = _resize_to_height(img_a, MONTAGE_HEIGHT)
    img_b = _resize_to_height(img_b, MONTAGE_HEIGHT)

    wa, ha = img_a.shape[1], img_a.shape[0]
    wb, hb = img_b.shape[1], img_b.shape[0]

    # Canvas dimensions
    header_h = 28
    total_w = wa + SEPARATOR_WIDTH + wb
    total_h = header_h + MONTAGE_HEIGHT + LABEL_HEIGHT
    canvas = np.full((total_h, total_w, 3), 30, dtype=np.uint8)

    # Severity header bar
    bar_colour = SEVERITY_COLOURS_BGR.get(severity, (100, 100, 100))
    cv2.rectangle(canvas, (0, 0), (total_w, header_h), bar_colour, -1)

    # Header text
    header_text = SEVERITY_LABELS.get(severity, severity.upper())
    dist_text = f"  |  MD5 identical" if is_exact else f"  |  Hamming distance: {distance}"
    _put_text_bg(
        canvas, header_text + dist_text, (8, header_h - 8),
        font_scale=0.5, colour=(255, 255, 255), bg_colour=bar_colour,
        thickness=1,
    )

    # Place images
    y_img = header_h
    canvas[y_img:y_img + ha, 0:wa] = img_a
    canvas[y_img:y_img + hb, wa + SEPARATOR_WIDTH:wa + SEPARATOR_WIDTH + wb] = img_b

    # Separator line
    sep_x = wa + SEPARATOR_WIDTH // 2
    cv2.line(canvas, (sep_x, header_h), (sep_x, total_h), (80, 80, 80), 2)

    # "VS" badge in separator
    cv2.circle(canvas, (sep_x, y_img + MONTAGE_HEIGHT // 2), 18, (60, 60, 60), -1)
    cv2.circle(canvas, (sep_x, y_img + MONTAGE_HEIGHT // 2), 18, (120, 120, 120), 1)
    _put_text_bg(
        canvas, "VS", (sep_x - 10, y_img + MONTAGE_HEIGHT // 2 + 5),
        font_scale=0.45, colour=(200, 200, 200), bg_colour=(60, 60, 60),
    )

    # Labels for image A (below image area)
    y_label = y_img + MONTAGE_HEIGHT + 5
    lines_a = [
        f"{meta_a.filename}  [{meta_a.class_name.upper()}]",
        f"{meta_a.width}x{meta_a.height}  |  {meta_a.file_size_kb:.0f} KB",
        f"Blur: {meta_a.blur_score:.1f}  |  Brightness: {meta_a.brightness:.1f}",
    ]
    for i, line in enumerate(lines_a):
        _put_text_bg(canvas, line, (5, y_label + i * 22 + 15), font_scale=0.42)

    # Labels for image B
    lines_b = [
        f"{meta_b.filename}  [{meta_b.class_name.upper()}]",
        f"{meta_b.width}x{meta_b.height}  |  {meta_b.file_size_kb:.0f} KB",
        f"Blur: {meta_b.blur_score:.1f}  |  Brightness: {meta_b.brightness:.1f}",
    ]
    for i, line in enumerate(lines_b):
        _put_text_bg(canvas, line, (wa + SEPARATOR_WIDTH + 5, y_label + i * 22 + 15), font_scale=0.42)

    return canvas


def create_contact_sheet(montages: List[np.ndarray], max_width: int = 1200) -> np.ndarray:
    """Stack multiple montage images into a vertical contact sheet.

    Each montage is resized to ``max_width`` if wider, then all are
    stacked vertically with a 6-pixel gap between them.

    Args:
        montages: List of montage images (HWC numpy arrays).
        max_width: Maximum width for each row.

    Returns:
        Combined contact sheet image.
    """
    if not montages:
        blank = np.full((100, max_width, 3), 30, dtype=np.uint8)
        _put_text_bg(blank, "No duplicate pairs found.", (20, 50))
        return blank

    gap = 6
    resized: List[np.ndarray] = []

    for m in montages:
        h, w = m.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_h = int(h * scale)
            m = cv2.resize(m, (max_width, new_h), interpolation=cv2.INTER_AREA)
        resized.append(m)

    # Calculate total height
    total_h = sum(m.shape[0] for m in resized) + gap * (len(resized) - 1)
    sheet_w = max(m.shape[1] for m in resized)
    sheet = np.full((total_h, sheet_w, 3), 20, dtype=np.uint8)

    y = 0
    for m in resized:
        h, w = m.shape[:2]
        sheet[y:y + h, 0:w] = m
        y += h + gap

    return sheet


def image_to_base64(image: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode an OpenCV image as a base64 data URI string.

    Args:
        image: HWC numpy array in BGR.
        fmt: Image format extension (``".jpg"`` or ``".png"``).

    Returns:
        Base64 data URI string suitable for ``<img src="...">``.
    """
    success, buffer = cv2.imencode(fmt, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        return ""
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def file_to_base64(path: Path) -> str:
    """Read an image file and return its base64 data URI.

    Args:
        path: Path to the image file.

    Returns:
        Base64 data URI string.
    """
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


# ── HTML Report Generator ────────────────────────────────────────────

def _generate_html(
    pairs: List[DuplicatePair],
    quarantined: List[Dict[str, str]],
    summary: Dict[str, Any],
    montage_b64: Dict[int, str],
    individual_b64: Dict[str, str],
) -> str:
    """Generate the complete self-contained HTML report.

    Args:
        pairs: All duplicate pairs with metadata.
        quarantined: List of quarantine action records.
        summary: Summary statistics dictionary.
        montage_b64: Pair index → base64-encoded montage image.
        individual_b64: Filename → base64-encoded original image (resized).

    Returns:
        Complete HTML document as a string.
    """
    # Build pair cards
    pair_cards = []
    for idx, pair in enumerate(pairs):
        severity_class = pair.severity
        badge_text = {
            "critical": "EXACT DUPLICATE",
            "warning": "PERCEPTUAL MATCH",
            "info": "NEAR-DUPLICATE",
        }.get(pair.severity, "UNKNOWN")

        action_html = ""
        if pair.action:
            action_html = f"""
            <div class="action-bar">
                <span class="action-icon">📦</span>
                <span class="action-text">{pair.action}</span>
            </div>"""

        distance_text = "MD5 identical" if pair.is_exact_md5 else f"Hamming distance: {pair.distance}"

        img_a_b64 = individual_b64.get(pair.image_a.filename, "")
        img_b_b64 = individual_b64.get(pair.image_b.filename, "")

        card = f"""
        <div class="pair-card {severity_class}">
            <div class="pair-header">
                <span class="badge {severity_class}">{badge_text}</span>
                <span class="pair-id">Pair #{idx + 1}</span>
                <span class="distance">{distance_text}</span>
            </div>
            <div class="pair-category">{pair.category}</div>
            <div class="pair-content">
                <div class="image-panel">
                    <img src="{img_a_b64}" alt="{pair.image_a.filename}" loading="lazy"/>
                    <table class="meta-table">
                        <tr><td class="label">File</td><td>{pair.image_a.filename}</td></tr>
                        <tr><td class="label">Class</td><td><span class="class-tag {pair.image_a.class_name}">{pair.image_a.class_name.upper()}</span></td></tr>
                        <tr><td class="label">Size</td><td>{pair.image_a.file_size_kb:.1f} KB</td></tr>
                        <tr><td class="label">Resolution</td><td>{pair.image_a.width}×{pair.image_a.height}</td></tr>
                        <tr><td class="label">Blur</td><td>{pair.image_a.blur_score:.1f}</td></tr>
                        <tr><td class="label">Brightness</td><td>{pair.image_a.brightness:.1f}</td></tr>
                        <tr><td class="label">MD5</td><td class="mono">{pair.image_a.md5[:12]}…</td></tr>
                        <tr><td class="label">pHash</td><td class="mono">{pair.image_a.phash}</td></tr>
                    </table>
                </div>
                <div class="separator">
                    <div class="vs-badge">VS</div>
                </div>
                <div class="image-panel">
                    <img src="{img_b_b64}" alt="{pair.image_b.filename}" loading="lazy"/>
                    <table class="meta-table">
                        <tr><td class="label">File</td><td>{pair.image_b.filename}</td></tr>
                        <tr><td class="label">Class</td><td><span class="class-tag {pair.image_b.class_name}">{pair.image_b.class_name.upper()}</span></td></tr>
                        <tr><td class="label">Size</td><td>{pair.image_b.file_size_kb:.1f} KB</td></tr>
                        <tr><td class="label">Resolution</td><td>{pair.image_b.width}×{pair.image_b.height}</td></tr>
                        <tr><td class="label">Blur</td><td>{pair.image_b.blur_score:.1f}</td></tr>
                        <tr><td class="label">Brightness</td><td>{pair.image_b.brightness:.1f}</td></tr>
                        <tr><td class="label">MD5</td><td class="mono">{pair.image_b.md5[:12]}…</td></tr>
                        <tr><td class="label">pHash</td><td class="mono">{pair.image_b.phash}</td></tr>
                    </table>
                </div>
            </div>
            {action_html}
        </div>
        """
        pair_cards.append(card)

    # Build quarantine table rows
    quarantine_rows = ""
    for q in quarantined:
        quarantine_rows += f"""
        <tr>
            <td>{q['filename']}</td>
            <td>{q['original_class']}</td>
            <td class="mono">{q['original_path']}</td>
            <td class="mono">{q['quarantine_path']}</td>
            <td>{q['reason']}</td>
        </tr>"""

    # Count by severity
    n_critical = sum(1 for p in pairs if p.severity == "critical")
    n_warning = sum(1 for p in pairs if p.severity == "warning")
    n_info = sum(1 for p in pairs if p.severity == "info")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Duplicate Inspection Report — Screen Recapture Detection</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            line-height: 1.6;
            padding: 2rem;
        }}

        h1 {{
            text-align: center;
            font-size: 2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #2EC4B6, #E71D36);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }}

        /* ── Summary Cards ─────────────────────────────── */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2.5rem;
        }}

        .summary-card {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            backdrop-filter: blur(10px);
        }}

        .summary-card .value {{
            font-size: 2.2rem;
            font-weight: 700;
        }}

        .summary-card .label {{
            color: #999;
            font-size: 0.85rem;
            margin-top: 0.3rem;
        }}

        .summary-card.critical .value {{ color: #E71D36; }}
        .summary-card.warning .value {{ color: #FFA500; }}
        .summary-card.info .value {{ color: #2EC4B6; }}
        .summary-card.neutral .value {{ color: #e0e0e0; }}

        /* ── Section headers ───────────────────────────── */
        .section-header {{
            font-size: 1.4rem;
            font-weight: 600;
            margin: 2rem 0 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }}

        /* ── Pair Cards ────────────────────────────────── */
        .pair-card {{
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            margin-bottom: 1.5rem;
            overflow: hidden;
            transition: transform 0.2s;
        }}

        .pair-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(255,255,255,0.12);
        }}

        .pair-card.critical {{ border-left: 4px solid #E71D36; }}
        .pair-card.warning {{ border-left: 4px solid #FFA500; }}
        .pair-card.info {{ border-left: 4px solid #2EC4B6; }}

        .pair-header {{
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 0.8rem 1.2rem;
            background: rgba(255,255,255,0.03);
        }}

        .badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}

        .badge.critical {{ background: rgba(231,29,54,0.2); color: #ff6b7f; }}
        .badge.warning  {{ background: rgba(255,165,0,0.2);  color: #ffb84d; }}
        .badge.info     {{ background: rgba(46,196,182,0.2); color: #5ae0d3; }}

        .pair-id {{
            color: #888;
            font-size: 0.85rem;
        }}

        .distance {{
            color: #aaa;
            font-size: 0.85rem;
            margin-left: auto;
            font-family: 'Consolas', 'Courier New', monospace;
        }}

        .pair-category {{
            padding: 0 1.2rem 0.5rem;
            color: #777;
            font-size: 0.82rem;
            font-style: italic;
        }}

        .pair-content {{
            display: flex;
            align-items: flex-start;
            padding: 1rem 1.2rem;
            gap: 0;
        }}

        .image-panel {{
            flex: 1;
            min-width: 0;
        }}

        .image-panel img {{
            width: 100%;
            max-height: 320px;
            object-fit: contain;
            border-radius: 8px;
            background: #1a1a2e;
            border: 1px solid rgba(255,255,255,0.06);
        }}

        .separator {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0 0.8rem;
            flex-shrink: 0;
        }}

        .vs-badge {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.75rem;
            color: #999;
        }}

        .meta-table {{
            width: 100%;
            margin-top: 0.6rem;
            font-size: 0.8rem;
            border-collapse: collapse;
        }}

        .meta-table td {{
            padding: 0.2rem 0.5rem;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }}

        .meta-table .label {{
            color: #888;
            width: 90px;
            font-weight: 500;
        }}

        .mono {{
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 0.78rem;
        }}

        .class-tag {{
            padding: 0.1rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        .class-tag.real {{ background: rgba(46,196,182,0.15); color: #2EC4B6; }}
        .class-tag.fake {{ background: rgba(231,29,54,0.15);  color: #E71D36; }}

        .action-bar {{
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.6rem 1.2rem;
            background: rgba(231,29,54,0.08);
            border-top: 1px solid rgba(231,29,54,0.15);
            font-size: 0.85rem;
        }}

        .action-icon {{ font-size: 1.1rem; }}
        .action-text {{ color: #ff9999; font-family: monospace; }}

        /* ── Quarantine Table ──────────────────────────── */
        .quarantine-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            margin-top: 1rem;
        }}

        .quarantine-table th {{
            text-align: left;
            padding: 0.7rem;
            background: rgba(255,255,255,0.05);
            color: #aaa;
            font-weight: 600;
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }}

        .quarantine-table td {{
            padding: 0.5rem 0.7rem;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }}

        /* ── Instructions ─────────────────────────────── */
        .instructions {{
            background: rgba(46,196,182,0.08);
            border: 1px solid rgba(46,196,182,0.2);
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 2rem;
        }}

        .instructions h3 {{ color: #2EC4B6; margin-bottom: 0.8rem; }}

        .instructions code {{
            background: rgba(255,255,255,0.08);
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.85rem;
        }}

        .instructions pre {{
            background: rgba(0,0,0,0.3);
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            overflow-x: auto;
            font-size: 0.82rem;
        }}

        .instructions li {{ margin: 0.3rem 0; margin-left: 1.5rem; }}

        /* ── Footer ───────────────────────────────────── */
        .footer {{
            text-align: center;
            color: #555;
            font-size: 0.8rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255,255,255,0.06);
        }}

        @media (max-width: 768px) {{
            .pair-content {{ flex-direction: column; }}
            .separator {{ padding: 0.5rem 0; }}
        }}
    </style>
</head>
<body>

<h1>🔍 Duplicate Inspection Report</h1>
<p class="subtitle">Screen Recapture Detection — Dataset Quality Review</p>

<!-- Summary -->
<div class="summary-grid">
    <div class="summary-card neutral">
        <div class="value">{summary['total_pairs']}</div>
        <div class="label">Total Flagged Pairs</div>
    </div>
    <div class="summary-card critical">
        <div class="value">{n_critical}</div>
        <div class="label">Exact Duplicates (Quarantined)</div>
    </div>
    <div class="summary-card warning">
        <div class="value">{n_warning}</div>
        <div class="label">Perceptual Matches (Review)</div>
    </div>
    <div class="summary-card info">
        <div class="value">{n_info}</div>
        <div class="label">Within-Class Near-Dupes (Review)</div>
    </div>
    <div class="summary-card neutral">
        <div class="value">{summary['quarantined_count']}</div>
        <div class="label">Images Quarantined</div>
    </div>
</div>

<!-- Pair Cards -->
<div class="section-header">All Flagged Pairs</div>
{''.join(pair_cards)}

<!-- Quarantine Log -->
<div class="section-header">Quarantine Log</div>
<table class="quarantine-table">
    <thead>
        <tr>
            <th>Filename</th>
            <th>Class</th>
            <th>Original Path</th>
            <th>Quarantine Path</th>
            <th>Reason</th>
        </tr>
    </thead>
    <tbody>
        {quarantine_rows if quarantine_rows else '<tr><td colspan="5" style="text-align:center;color:#666">No images quarantined</td></tr>'}
    </tbody>
</table>

<!-- Instructions -->
<div class="instructions">
    <h3>📋 Engineer Review Instructions</h3>
    <ol>
        <li>Review each pair above. Exact duplicates (red) are byte-identical across classes — one is clearly mislabelled.</li>
        <li>Perceptual matches (orange) may be <strong>valid paired data</strong> — the same scene photographed directly (real) and from a screen (fake). These should stay in the dataset but be placed in the <strong>same train/val/test split</strong>.</li>
        <li>Within-class near-duplicates (teal) are similar images within the same class. They may reduce effective dataset diversity.</li>
    </ol>

    <h3 style="margin-top:1rem;">🔄 To Restore a Quarantined Image</h3>
    <pre># Windows (PowerShell)
Move-Item "dataset\\quarantine\\fake_XXXX.jpg" "dataset\\fake\\fake_XXXX.jpg"

# Linux / macOS
mv dataset/quarantine/fake_XXXX.jpg dataset/fake/fake_XXXX.jpg</pre>

    <h3 style="margin-top:1rem;">🗑️ To Permanently Delete a Quarantined Image</h3>
    <pre># Only after manual verification!
# Windows: Remove-Item "dataset\\quarantine\\fake_XXXX.jpg"
# Linux:   rm dataset/quarantine/fake_XXXX.jpg</pre>
</div>

<div class="footer">
    Generated by Screen Recapture Detection Pipeline — quarantine_duplicates.py
</div>

</body>
</html>"""

    return html


# ── Main Orchestrator ─────────────────────────────────────────────────

class DuplicateHandler:
    """Orchestrates duplicate detection, quarantine, and report generation.

    Attributes:
        config: Project configuration.
        logger: Structured logger.
        quarantine_dir: Path to quarantine directory.
        montages_dir: Path to saved montage images.
        image_meta: Mapping of filename → ImageMeta.
        pairs: List of all detected DuplicatePair objects.
        quarantine_log: Records of quarantine actions.
    """

    def __init__(self, config: ProjectConfig) -> None:
        """Initialise the handler.

        Args:
            config: ``ProjectConfig`` instance.
        """
        self.config = config
        self.config.ensure_directories()
        self.logger = setup_logger("quarantine", log_dir=config.logs_dir)

        self.quarantine_dir = config.dataset_dir / "quarantine"
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)

        self.montages_dir = config.outputs_dir / "montages"
        self.montages_dir.mkdir(parents=True, exist_ok=True)

        self.image_meta: Dict[str, ImageMeta] = {}
        self.pairs: List[DuplicatePair] = []
        self.quarantine_log: List[Dict[str, str]] = []

    def run(self) -> None:
        """Execute the full duplicate handling pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("DUPLICATE QUARANTINE PIPELINE — START")
        self.logger.info("=" * 60)

        self._scan_and_hash()
        self._find_all_duplicates()
        montage_images = self._generate_montages()
        self._generate_contact_sheet(montage_images)
        self._quarantine_suspicious()
        self._generate_report(montage_images)

        self.logger.info("=" * 60)
        self.logger.info("DUPLICATE QUARANTINE PIPELINE — COMPLETE")
        self.logger.info(f"  Pairs flagged:     {len(self.pairs)}")
        self.logger.info(f"  Images quarantined: {len(self.quarantine_log)}")
        self.logger.info(f"  Report: outputs/duplicate_report.html")
        self.logger.info(f"  Contact sheet: outputs/montages/contact_sheet.png")
        self.logger.info("=" * 60)

    # ── Step 1: Scan & Hash ───────────────────────────────────────────

    def _scan_and_hash(self) -> None:
        """Scan dataset directories and compute hashes + metadata for all images."""
        self.logger.info("Step 1/5: Scanning images and computing hashes...")

        for class_name, directory in [
            (self.config.class_real, self.config.real_dir),
            (self.config.class_fake, self.config.fake_dir),
        ]:
            files = get_image_files(directory, self.config.image_extensions)
            for path in tqdm(files, desc=f"Hashing {class_name}", unit="img"):
                info = get_image_info(path)
                md5 = compute_md5(path)
                phash = compute_phash(path) or ""

                if info is not None:
                    meta = ImageMeta(
                        path=path,
                        filename=path.name,
                        class_name=class_name,
                        file_size_kb=info.file_size_bytes / 1024,
                        width=info.width,
                        height=info.height,
                        blur_score=info.blur_score,
                        brightness=info.brightness,
                        md5=md5,
                        phash=phash,
                    )
                    self.image_meta[path.name] = meta

        self.logger.info(f"  Processed {len(self.image_meta)} images.")

    # ── Step 2: Find Duplicates ───────────────────────────────────────

    def _find_all_duplicates(self) -> None:
        """Identify all duplicate and near-duplicate pairs."""
        self.logger.info("Step 2/5: Finding duplicates...")

        real_meta = {k: v for k, v in self.image_meta.items() if v.class_name == self.config.class_real}
        fake_meta = {k: v for k, v in self.image_meta.items() if v.class_name == self.config.class_fake}

        # Build hash maps
        all_md5: Dict[Path, str] = {m.path: m.md5 for m in self.image_meta.values()}
        real_phash: Dict[Path, str] = {m.path: m.phash for m in real_meta.values() if m.phash}
        fake_phash: Dict[Path, str] = {m.path: m.phash for m in fake_meta.values() if m.phash}

        # Track which pairs we've already added (avoid duplicates in report)
        seen_pairs = set()

        def _pair_key(a: str, b: str) -> Tuple[str, str]:
            return (min(a, b), max(a, b))

        # --- Exact cross-class duplicates ---
        exact_groups = find_exact_duplicates(all_md5)
        for group in exact_groups:
            classes_in_group = set()
            for p in group:
                name = p.name
                if name in self.image_meta:
                    classes_in_group.add(self.image_meta[name].class_name)

            if len(classes_in_group) > 1:
                # Cross-class exact duplicate
                reals = [p for p in group if self.image_meta.get(p.name, ImageMeta(
                    path=p, filename="", class_name="", file_size_kb=0,
                    width=0, height=0, blur_score=0, brightness=0, md5="", phash="",
                )).class_name == self.config.class_real]
                fakes = [p for p in group if self.image_meta.get(p.name, ImageMeta(
                    path=p, filename="", class_name="", file_size_kb=0,
                    width=0, height=0, blur_score=0, brightness=0, md5="", phash="",
                )).class_name == self.config.class_fake]

                for r in reals:
                    for f in fakes:
                        key = _pair_key(r.name, f.name)
                        if key not in seen_pairs:
                            seen_pairs.add(key)
                            self.pairs.append(DuplicatePair(
                                image_a=self.image_meta[r.name],
                                image_b=self.image_meta[f.name],
                                distance=0,
                                is_exact_md5=True,
                                severity="critical",
                                category="Byte-identical file found in both real/ and fake/ — one is mislabelled.",
                                action=f"{f.name} → dataset/quarantine/",
                            ))

        self.logger.info(f"  Exact cross-class duplicates: {sum(1 for p in self.pairs if p.severity == 'critical')}")

        # --- Perceptual cross-class matches (non-exact) ---
        cross_matches = find_cross_class_duplicates(
            real_phash, fake_phash, threshold=self.config.phash_threshold,
        )
        for pa, pb, dist in cross_matches:
            key = _pair_key(pa.name, pb.name)
            if key not in seen_pairs:
                seen_pairs.add(key)
                self.pairs.append(DuplicatePair(
                    image_a=self.image_meta[pa.name],
                    image_b=self.image_meta[pb.name],
                    distance=dist,
                    is_exact_md5=False,
                    severity="warning",
                    category="Perceptually similar across classes — may be valid paired data (same scene).",
                    action="",
                ))

        self.logger.info(
            f"  Perceptual cross-class matches: "
            f"{sum(1 for p in self.pairs if p.severity == 'warning')}"
        )

        # --- Within-class near-duplicates ---
        for label, phash_map in [("real", real_phash), ("fake", fake_phash)]:
            within_pairs = find_perceptual_duplicates(phash_map, threshold=self.config.phash_threshold)
            for pa, pb, dist in within_pairs:
                key = _pair_key(pa.name, pb.name)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    self.pairs.append(DuplicatePair(
                        image_a=self.image_meta[pa.name],
                        image_b=self.image_meta[pb.name],
                        distance=dist,
                        is_exact_md5=False,
                        severity="info",
                        category=f"Near-duplicate within the {label} class — may reduce dataset diversity.",
                        action="",
                    ))

        self.logger.info(
            f"  Within-class near-duplicates: "
            f"{sum(1 for p in self.pairs if p.severity == 'info')}"
        )

        # Sort: critical first, then warning, then info
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        self.pairs.sort(key=lambda p: (severity_order.get(p.severity, 9), p.distance))

        self.logger.info(f"  Total pairs: {len(self.pairs)}")

    # ── Step 3: Generate Montages ─────────────────────────────────────

    def _generate_montages(self) -> List[np.ndarray]:
        """Create side-by-side montage images for all pairs.

        Returns:
            List of montage images (HWC numpy arrays).
        """
        self.logger.info("Step 3/5: Generating montage images...")

        montages: List[np.ndarray] = []
        for idx, pair in enumerate(self.pairs):
            montage = create_pair_montage(
                pair.image_a.path,
                pair.image_b.path,
                pair.image_a,
                pair.image_b,
                pair.severity,
                pair.distance,
                pair.is_exact_md5,
            )
            montages.append(montage)

            # Save individual montage
            save_path = self.montages_dir / f"pair_{idx + 1:02d}_{pair.severity}.png"
            cv2.imwrite(str(save_path), montage)
            pair.montage_path = save_path

        self.logger.info(f"  Saved {len(montages)} montage images to {self.montages_dir}")
        return montages

    # ── Step 4: Contact Sheet ─────────────────────────────────────────

    def _generate_contact_sheet(self, montages: List[np.ndarray]) -> None:
        """Combine all montages into a single contact sheet image.

        Args:
            montages: List of montage images.
        """
        self.logger.info("Step 4/5: Generating contact sheet...")

        if not montages:
            self.logger.info("  No pairs — skipping contact sheet.")
            return

        sheet = create_contact_sheet(montages, max_width=1400)
        sheet_path = self.montages_dir / "contact_sheet.png"
        cv2.imwrite(str(sheet_path), sheet)
        self.logger.info(f"  Saved contact sheet → {sheet_path} ({sheet.shape[0]}×{sheet.shape[1]})")

    # ── Step 5: Quarantine ────────────────────────────────────────────

    def _quarantine_suspicious(self) -> None:
        """Move exact cross-class duplicate fakes to quarantine.

        Only byte-identical cross-class duplicates are moved. Perceptual
        matches and within-class near-duplicates are flagged for review only.
        """
        self.logger.info("Step 5/5: Quarantining exact cross-class duplicates...")

        quarantined_files = set()

        for pair in self.pairs:
            if pair.severity != "critical":
                continue

            # Quarantine the fake copy (the real one stays)
            target = pair.image_b  # Always the fake in critical pairs
            if target.filename in quarantined_files:
                continue  # Already quarantined

            src = target.path
            dst = self.quarantine_dir / target.filename

            if src.exists():
                shutil.move(str(src), str(dst))
                quarantined_files.add(target.filename)
                self.quarantine_log.append({
                    "filename": target.filename,
                    "original_class": target.class_name,
                    "original_path": str(src),
                    "quarantine_path": str(dst),
                    "reason": f"Byte-identical to {pair.image_a.filename} in real/",
                })
                self.logger.info(f"  📦 {target.filename} → quarantine/")
            else:
                self.logger.warning(f"  File not found (already moved?): {src}")

        self.logger.info(f"  Quarantined {len(self.quarantine_log)} image(s).")

    # ── Step 6: HTML Report ───────────────────────────────────────────

    def _generate_report(self, montages: List[np.ndarray]) -> None:
        """Generate the self-contained HTML inspection report.

        Args:
            montages: List of montage images (used for base64 encoding).
        """
        self.logger.info("Generating HTML report...")

        # Encode montages as base64
        montage_b64: Dict[int, str] = {}
        for idx, m in enumerate(montages):
            montage_b64[idx] = image_to_base64(m)

        # Encode individual images as base64 (resized for display)
        individual_b64: Dict[str, str] = {}
        for pair in self.pairs:
            for meta in [pair.image_a, pair.image_b]:
                if meta.filename not in individual_b64:
                    # Try original path first, then quarantine
                    img_path = meta.path
                    if not img_path.exists():
                        img_path = self.quarantine_dir / meta.filename
                    if img_path.exists():
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            img = _resize_to_height(img, 320)
                            individual_b64[meta.filename] = image_to_base64(img)

        summary = {
            "total_pairs": len(self.pairs),
            "quarantined_count": len(self.quarantine_log),
        }

        html = _generate_html(
            pairs=self.pairs,
            quarantined=self.quarantine_log,
            summary=summary,
            montage_b64=montage_b64,
            individual_b64=individual_b64,
        )

        report_path = self.config.outputs_dir / "duplicate_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)

        self.logger.info(f"  Saved HTML report → {report_path}")


# ── CLI Entry Point ───────────────────────────────────────────────────

def main() -> None:
    """Entry point for the quarantine duplicates script."""
    config = ProjectConfig()
    handler = DuplicateHandler(config)
    handler.run()


if __name__ == "__main__":
    main()
