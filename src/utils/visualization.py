"""
Visualization utilities for dataset inspection and model evaluation.

Design Decisions:
    - All plot functions accept pre-computed data (DataFrames, dicts) and a
      ``save_path`` — they do NOT perform I/O or computation beyond rendering.
    - Every function calls ``plt.close()`` after saving to prevent memory
      leaks in long-running pipelines.
    - Consistent colour scheme: teal for real, coral for fake, matching a
      professional palette that reads well in both light and dark contexts.
    - High DPI (150) for crisp output in reports and presentations.
    - Functions are stateless — no module-level figure state.

Usage:
    from src.utils.visualization import plot_class_distribution
    plot_class_distribution({"real": 66, "fake": 84}, Path("outputs/plots"))
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Consistent colour scheme
COLOUR_REAL = "#2EC4B6"   # teal
COLOUR_FAKE = "#E71D36"   # coral-red
COLOUR_PALETTE = [COLOUR_REAL, COLOUR_FAKE]
CLASS_COLOURS = {"real": COLOUR_REAL, "fake": COLOUR_FAKE}


def _apply_style() -> None:
    """Apply a clean, professional matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "#FAFAFA",
        "axes.facecolor": "#FAFAFA",
        "axes.edgecolor": "#CCCCCC",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.color": "#CCCCCC",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
    })


def plot_class_distribution(
    counts: Dict[str, int],
    save_dir: Path,
) -> Path:
    """Plot class distribution as a horizontal bar chart.

    Args:
        counts: Mapping of class name → image count.
        save_dir: Directory to save the plot.

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    classes = list(counts.keys())
    values = list(counts.values())
    colours = [CLASS_COLOURS.get(c, "#999999") for c in classes]

    bars = ax.barh(classes, values, color=colours, edgecolor="white", height=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val}",
            va="center", fontweight="bold", fontsize=13,
        )

    total = sum(values)
    ratio = max(values) / min(values) if min(values) > 0 else float("inf")
    ax.set_title(f"Class Distribution  (total={total}, ratio={ratio:.2f}:1)")
    ax.set_xlabel("Number of Images")
    ax.set_xlim(0, max(values) * 1.2)
    plt.tight_layout()

    save_path = save_dir / "class_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_resolution_scatter(
    df: pd.DataFrame,
    save_dir: Path,
) -> Path:
    """Scatter plot of image width × height, coloured by class.

    Expects ``df`` to have columns: ``width``, ``height``, ``class``.

    Args:
        df: DataFrame with per-image resolution data.
        save_dir: Directory to save the plot.

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    for cls in df["class"].unique():
        subset = df[df["class"] == cls]
        ax.scatter(
            subset["width"], subset["height"],
            c=CLASS_COLOURS.get(cls, "#999"),
            label=cls, alpha=0.7, edgecolors="white", s=60,
        )

    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image Resolution Distribution")
    ax.legend()
    plt.tight_layout()

    save_path = save_dir / "resolution_scatter.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_aspect_ratio_distribution(
    df: pd.DataFrame,
    save_dir: Path,
    bins: int = 20,
) -> Path:
    """Overlapping histogram of aspect ratios by class.

    Expects ``df`` to have columns: ``aspect_ratio``, ``class``.

    Args:
        df: DataFrame with per-image aspect ratio data.
        save_dir: Directory to save the plot.
        bins: Number of histogram bins.

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for cls in df["class"].unique():
        subset = df[df["class"] == cls]
        ax.hist(
            subset["aspect_ratio"], bins=bins,
            color=CLASS_COLOURS.get(cls, "#999"),
            alpha=0.6, label=cls, edgecolor="white",
        )

    ax.set_xlabel("Aspect Ratio (width / height)")
    ax.set_ylabel("Count")
    ax.set_title("Aspect Ratio Distribution")
    ax.legend()
    plt.tight_layout()

    save_path = save_dir / "aspect_ratio_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_blur_distribution(
    df: pd.DataFrame,
    save_dir: Path,
    bins: int = 25,
) -> Path:
    """Overlapping histogram of blur scores by class.

    Expects ``df`` to have columns: ``blur_score``, ``class``.

    Args:
        df: DataFrame with per-image blur scores.
        save_dir: Directory to save the plot.
        bins: Number of histogram bins.

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for cls in df["class"].unique():
        subset = df[df["class"] == cls]
        ax.hist(
            subset["blur_score"], bins=bins,
            color=CLASS_COLOURS.get(cls, "#999"),
            alpha=0.6, label=cls, edgecolor="white",
        )

    ax.axvline(x=100, color="#FF6B6B", linestyle="--", linewidth=1.5, label="Blur threshold")
    ax.set_xlabel("Blur Score (Laplacian Variance)")
    ax.set_ylabel("Count")
    ax.set_title("Blur Score Distribution")
    ax.legend()
    plt.tight_layout()

    save_path = save_dir / "blur_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_brightness_distribution(
    df: pd.DataFrame,
    save_dir: Path,
    bins: int = 25,
) -> Path:
    """Overlapping histogram of brightness values by class.

    Expects ``df`` to have columns: ``brightness``, ``class``.

    Args:
        df: DataFrame with per-image brightness values.
        save_dir: Directory to save the plot.
        bins: Number of histogram bins.

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for cls in df["class"].unique():
        subset = df[df["class"] == cls]
        ax.hist(
            subset["brightness"], bins=bins,
            color=CLASS_COLOURS.get(cls, "#999"),
            alpha=0.6, label=cls, edgecolor="white",
        )

    ax.set_xlabel("Mean Brightness (0–255)")
    ax.set_ylabel("Count")
    ax.set_title("Brightness Distribution")
    ax.legend()
    plt.tight_layout()

    save_path = save_dir / "brightness_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_file_size_distribution(
    df: pd.DataFrame,
    save_dir: Path,
    bins: int = 20,
) -> Path:
    """Overlapping histogram of file sizes (KB) by class.

    Expects ``df`` to have columns: ``file_size_bytes``, ``class``.

    Args:
        df: DataFrame with per-image file size data.
        save_dir: Directory to save the plot.
        bins: Number of histogram bins.

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for cls in df["class"].unique():
        subset = df[df["class"] == cls]
        sizes_kb = subset["file_size_bytes"] / 1024
        ax.hist(
            sizes_kb, bins=bins,
            color=CLASS_COLOURS.get(cls, "#999"),
            alpha=0.6, label=cls, edgecolor="white",
        )

    ax.set_xlabel("File Size (KB)")
    ax.set_ylabel("Count")
    ax.set_title("File Size Distribution")
    ax.legend()
    plt.tight_layout()

    save_path = save_dir / "file_size_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_rgb_histograms(
    real_histograms: List[Dict[str, np.ndarray]],
    fake_histograms: List[Dict[str, np.ndarray]],
    save_dir: Path,
) -> Path:
    """Plot aggregated (mean) RGB histograms for real vs fake classes.

    Each input list contains per-image histogram dicts with keys
    ``"red"``, ``"green"``, ``"blue"`` mapping to 256-bin arrays.

    Args:
        real_histograms: List of per-image RGB histogram dicts for real class.
        fake_histograms: List of per-image RGB histogram dicts for fake class.
        save_dir: Directory to save the plot.

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    channel_colours = {"red": "#E63946", "green": "#2A9D8F", "blue": "#457B9D"}

    for ax, histograms, title in zip(
        axes,
        [real_histograms, fake_histograms],
        ["Real Images — RGB Histogram", "Fake Images — RGB Histogram"],
    ):
        if not histograms:
            ax.set_title(title + " (no data)")
            continue

        for channel, colour in channel_colours.items():
            stacked = np.stack([h[channel] for h in histograms], axis=0)
            mean_hist = stacked.mean(axis=0)
            ax.plot(mean_hist, color=colour, alpha=0.8, label=channel.capitalize())

        ax.set_title(title)
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Mean Frequency")
        ax.legend()

    plt.tight_layout()

    save_path = save_dir / "rgb_histograms.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_sample_grid(
    image_paths: List[Tuple[Path, str]],
    save_dir: Path,
    n_per_class: int = 8,
    seed: int = 42,
) -> Path:
    """Plot a grid of random sample images, grouped by class.

    Args:
        image_paths: List of ``(path, class_name)`` tuples.
        save_dir: Directory to save the plot.
        n_per_class: Number of samples per class.
        seed: Random seed for reproducible sampling.

    Returns:
        Path to the saved figure.
    """
    _apply_style()
    rng = random.Random(seed)

    # Group by class
    by_class: Dict[str, List[Path]] = {}
    for path, cls in image_paths:
        by_class.setdefault(cls, []).append(path)

    classes = sorted(by_class.keys())
    n_classes = len(classes)
    samples_per_class = min(n_per_class, min(len(v) for v in by_class.values()))

    fig, axes = plt.subplots(
        n_classes, samples_per_class,
        figsize=(samples_per_class * 2.5, n_classes * 2.8),
    )
    if n_classes == 1:
        axes = [axes]

    for row, cls in enumerate(classes):
        sampled = rng.sample(by_class[cls], samples_per_class)
        for col, path in enumerate(sampled):
            ax = axes[row][col] if samples_per_class > 1 else axes[row]
            img = cv2.imread(str(path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize for display
                img = cv2.resize(img, (224, 224))
                ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(
                    cls.upper(),
                    fontsize=13, fontweight="bold",
                    rotation=0, labelpad=50, va="center",
                )

    fig.suptitle("Random Sample Grid", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = save_dir / "sample_grid.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
