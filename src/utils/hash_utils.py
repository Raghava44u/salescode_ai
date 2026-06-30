"""
File hashing utilities for duplicate detection.

Design Decisions:
    - MD5 for exact-duplicate detection (byte-identical files).
    - Perceptual hashing (pHash via ``imagehash``) for near-duplicate
      detection — essential because WhatsApp re-encodes images, so the
      same visual content may have different byte streams.
    - Separate within-class and cross-class duplicate finders so the
      inspector can flag data-leakage risks independently.
    - Hamming-distance threshold is configurable (default 10) to
      accommodate WhatsApp compression variance.

Usage:
    from src.utils.hash_utils import compute_md5, compute_phash
    md5 = compute_md5(Path("image.jpg"))
    phash = compute_phash(Path("image.jpg"))
"""

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imagehash
from PIL import Image


def compute_md5(path: Path, chunk_size: int = 8192) -> str:
    """Compute the MD5 hex-digest of a file.

    Reads the file in chunks to keep memory usage constant regardless
    of file size.

    Args:
        path: Path to the file.
        chunk_size: Bytes per read chunk.

    Returns:
        Lowercase hexadecimal MD5 digest string.
    """
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_phash(path: Path, hash_size: int = 8) -> Optional[str]:
    """Compute the perceptual hash (pHash) of an image.

    pHash is robust to minor compression differences, making it ideal
    for detecting near-duplicates produced by WhatsApp re-encoding.

    Args:
        path: Path to the image file.
        hash_size: DCT hash grid size (default 8 → 64-bit hash).

    Returns:
        Hexadecimal hash string, or ``None`` if the image cannot be loaded.
    """
    try:
        img = Image.open(path).convert("RGB")
        return str(imagehash.phash(img, hash_size=hash_size))
    except Exception:
        return None


def find_exact_duplicates(
    file_hashes: Dict[Path, str],
) -> List[List[Path]]:
    """Group files that share an identical MD5 hash.

    Args:
        file_hashes: Mapping of file paths to their MD5 hex-digests.

    Returns:
        List of groups, where each group contains two or more paths
        that are byte-identical.  Singleton groups are excluded.
    """
    hash_to_files: Dict[str, List[Path]] = defaultdict(list)
    for path, md5 in file_hashes.items():
        hash_to_files[md5].append(path)

    return [paths for paths in hash_to_files.values() if len(paths) > 1]


def find_perceptual_duplicates(
    file_hashes: Dict[Path, str],
    threshold: int = 10,
) -> List[Tuple[Path, Path, int]]:
    """Find pairs of images whose perceptual hashes are within a
    Hamming-distance threshold.

    This is an O(n²) comparison but is acceptable for datasets under
    ~10 000 images.

    Args:
        file_hashes: Mapping of file paths to hex pHash strings.
        threshold: Maximum Hamming distance to consider a pair as
            perceptually similar.

    Returns:
        List of ``(path_a, path_b, distance)`` tuples for every
        qualifying pair.
    """
    paths = list(file_hashes.keys())
    hashes = {p: imagehash.hex_to_hash(h) for p, h in file_hashes.items() if h is not None}

    duplicates: List[Tuple[Path, Path, int]] = []
    path_list = list(hashes.keys())

    for i in range(len(path_list)):
        for j in range(i + 1, len(path_list)):
            dist = hashes[path_list[i]] - hashes[path_list[j]]
            if dist <= threshold:
                duplicates.append((path_list[i], path_list[j], dist))

    return duplicates


def find_cross_class_duplicates(
    hashes_a: Dict[Path, str],
    hashes_b: Dict[Path, str],
    threshold: int = 10,
) -> List[Tuple[Path, Path, int]]:
    """Find perceptually similar images across two sets (e.g., real vs fake).

    Critical for detecting data leakage — the same scene appearing in
    both training classes.

    Args:
        hashes_a: pHash mapping for class A (e.g., real images).
        hashes_b: pHash mapping for class B (e.g., fake images).
        threshold: Maximum Hamming distance.

    Returns:
        List of ``(path_a, path_b, distance)`` tuples.
    """
    parsed_a = {p: imagehash.hex_to_hash(h) for p, h in hashes_a.items() if h is not None}
    parsed_b = {p: imagehash.hex_to_hash(h) for p, h in hashes_b.items() if h is not None}

    matches: List[Tuple[Path, Path, int]] = []
    for pa, ha in parsed_a.items():
        for pb, hb in parsed_b.items():
            dist = ha - hb
            if dist <= threshold:
                matches.append((pa, pb, dist))

    return matches
