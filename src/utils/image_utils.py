"""
Image I/O and analysis utilities for the Screen Recapture Detection project.

Design Decisions:
    - PIL for loading/saving (preserves quality, handles EXIF orientation).
    - OpenCV for numerical analysis (Laplacian, histogram) because cv2 operates
      on numpy arrays and is 2-5× faster than pure-PIL for pixel-level math.
    - Every function that can fail on a corrupted file returns ``Optional``
      instead of raising — the caller decides how to handle bad inputs.
    - ``ImageInfo`` dataclass bundles per-image metadata for downstream
      aggregation without repeated I/O.

Usage:
    from src.utils.image_utils import get_image_files, get_image_info
    files = get_image_files(Path("dataset/real"), extensions=(".jpg",))
    info = get_image_info(files[0])
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ImageInfo:
    """Per-image metadata container.

    Attributes:
        path: Absolute path to the image file.
        width: Image width in pixels.
        height: Image height in pixels.
        channels: Number of colour channels (3 for RGB, 1 for grayscale).
        file_size_bytes: On-disk file size.
        aspect_ratio: width / height.
        blur_score: Laplacian variance — lower values indicate blurrier images.
        brightness: Mean pixel intensity (0–255 scale).
    """

    path: Path
    width: int
    height: int
    channels: int
    file_size_bytes: int
    aspect_ratio: float
    blur_score: float
    brightness: float


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def get_image_files(
    directory: Path,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> List[Path]:
    """Recursively collect image file paths from a directory.

    Only files whose suffix (case-insensitive) matches one of the
    supplied ``extensions`` are returned.  Results are sorted by name
    for deterministic ordering.

    Args:
        directory: Root directory to scan.
        extensions: Tuple of lowercase file extensions including the dot.

    Returns:
        Sorted list of ``Path`` objects for each matching file.

    Raises:
        FileNotFoundError: If ``directory`` does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = [
        p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in extensions
    ]
    return files


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_pil(path: Path) -> Optional[Image.Image]:
    """Load an image file as a PIL Image in RGB mode.

    Args:
        path: Path to the image file.

    Returns:
        PIL ``Image`` in RGB mode, or ``None`` if the file is corrupted
        or unreadable.
    """
    try:
        img = Image.open(path)
        img.verify()  # lightweight structural check
        # Re-open because verify() can consume the file pointer
        img = Image.open(path).convert("RGB")
        return img
    except Exception:
        return None


def load_image_cv2(path: Path) -> Optional[np.ndarray]:
    """Load an image file as a BGR numpy array via OpenCV.

    Args:
        path: Path to the image file.

    Returns:
        HWC numpy array in BGR channel order, or ``None`` if the
        file cannot be decoded.
    """
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return None
        return img
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_valid_image(path: Path) -> bool:
    """Check whether a file is a valid, non-corrupted image.

    Performs both a PIL structural verification and an OpenCV decode
    check.  An image must pass *both* to be considered valid.

    Args:
        path: Path to the image file.

    Returns:
        ``True`` if the image is decodable and structurally sound.
    """
    # PIL check
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception:
        return False

    # OpenCV decode check
    img_cv = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_cv is None or img_cv.size == 0:
        return False

    return True


# ---------------------------------------------------------------------------
# Image analysis
# ---------------------------------------------------------------------------

def compute_blur_score(image: np.ndarray) -> float:
    """Compute image sharpness using the variance of the Laplacian.

    Higher values indicate sharper images.  Typical thresholds:
        - < 50   : very blurry
        - 50–100 : moderately blurry
        - > 100  : acceptably sharp

    Args:
        image: HWC numpy array (BGR or RGB).

    Returns:
        Laplacian variance as a float.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_brightness(image: np.ndarray) -> float:
    """Compute mean brightness of an image (0–255 scale).

    Converts to grayscale first to obtain a single luminance value.

    Args:
        image: HWC numpy array (BGR or RGB).

    Returns:
        Mean pixel intensity.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(np.mean(gray))


def compute_rgb_histogram(
    image: np.ndarray,
    bins: int = 256,
) -> Dict[str, np.ndarray]:
    """Compute per-channel histogram for a BGR image.

    Args:
        image: HWC numpy array in BGR channel order.
        bins: Number of histogram bins per channel.

    Returns:
        Dictionary with keys ``"blue"``, ``"green"``, ``"red"`` mapping
        to 1-D numpy arrays of bin counts.
    """
    channels = {"blue": 0, "green": 1, "red": 2}
    histograms: Dict[str, np.ndarray] = {}
    for name, idx in channels.items():
        hist = cv2.calcHist([image], [idx], None, [bins], [0, 256])
        histograms[name] = hist.flatten()
    return histograms


# ---------------------------------------------------------------------------
# Composite metadata extraction
# ---------------------------------------------------------------------------

def get_image_info(path: Path) -> Optional[ImageInfo]:
    """Extract comprehensive metadata from a single image file.

    Combines file-system stats with pixel-level analysis (blur,
    brightness) into a single ``ImageInfo`` object.

    Args:
        path: Path to the image file.

    Returns:
        ``ImageInfo`` instance, or ``None`` if the image is corrupted.
    """
    img = load_image_cv2(path)
    if img is None:
        return None

    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    file_size = path.stat().st_size

    return ImageInfo(
        path=path,
        width=w,
        height=h,
        channels=channels,
        file_size_bytes=file_size,
        aspect_ratio=round(w / h, 4) if h > 0 else 0.0,
        blur_score=round(compute_blur_score(img), 2),
        brightness=round(compute_brightness(img), 2),
    )
