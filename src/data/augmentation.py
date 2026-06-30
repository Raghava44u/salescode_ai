"""
Data augmentation pipelines using Albumentations.

Design Decisions:
    - Focuses on preserving high-frequency screen artifacts (Moiré, pixel grids).
    - Excludes aggressive augmentations like heavy blur, extreme rotation, 
      or vertical flips that destroy the anti-spoofing signal.
    - Normalizes using standard ImageNet mean and std.

Usage:
    from src.data.augmentation import get_train_transforms, get_eval_transforms
    train_transform = get_train_transforms(config)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import ProjectConfig


def get_train_transforms(config: ProjectConfig) -> A.Compose:
    """Get the conservative training augmentation pipeline.

    Args:
        config: Central project configuration.

    Returns:
        Albumentations Compose object.
    """
    return A.Compose([
        A.Resize(config.input_size, config.input_size),
        
        # Spatial - Preserve high-frequency forensic artifacts
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5, border_mode=0),
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), p=0.3),
        # Avoid Perspective and heavy distortion
        
        # Pixel-level Lighting
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(p=0.1),
        
        # Camera Effects (very small to avoid destroying Moiré)
        A.ImageCompression(quality_range=(70, 100), p=0.3),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.GaussNoise(std_range=(0.01, 0.02), p=0.2),
        A.Sharpen(p=0.1),
        
        # Normalization
        A.Normalize(
            mean=config.imagenet_mean,
            std=config.imagenet_std,
        ),
        ToTensorV2(),
    ])


def get_eval_transforms(config: ProjectConfig) -> A.Compose:
    """Get the evaluation pipeline (Resize & Normalize only).

    Args:
        config: Central project configuration.

    Returns:
        Albumentations Compose object.
    """
    return A.Compose([
        A.Resize(config.input_size, config.input_size),
        A.Normalize(
            mean=config.imagenet_mean,
            std=config.imagenet_std,
        ),
        ToTensorV2(),
    ])
