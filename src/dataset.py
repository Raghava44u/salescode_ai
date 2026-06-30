"""
Dataset module for Screen Recapture Detection.

Design Decisions:
    - Custom PyTorch Dataset to load from train/val/test directories.
    - Class labels hardcoded mapped to integers (0=Real, 1=Fake).
    - Albumentations for transforms (requires numpy RGB arrays).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset

from src.config import ProjectConfig


class SpoofDataset(Dataset):
    """Custom dataset for loading real vs fake images.
    
    Attributes:
        data_dir: Path to the specific split (e.g., dataset/train).
        config: Project configuration.
        transform: Optional Albumentations Compose pipeline.
    """
    
    def __init__(
        self,
        data_dir: Path,
        config: ProjectConfig,
        transform: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.transform = transform
        
        # 0 = Real, 1 = Fake
        self.class_to_idx = {
            config.class_real: 0,
            config.class_fake: 1
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        self.samples: List[Tuple[Path, int]] = []
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
            
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = data_dir / class_name
            if not class_dir.exists():
                continue
                
            for ext in config.image_extensions:
                for img_path in class_dir.glob(f"*{ext}"):
                    self.samples.append((img_path, class_idx))
                    
        # Sort for deterministic behavior
        self.samples.sort(key=lambda x: x[0].name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Read image
        # cv2.imread does not handle pathlib.Path directly in older versions, use str()
        # Convert BGR to RGB for Albumentations
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Failed to read image: {img_path}")
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            augmented = self.transform(image=img_rgb)
            img_tensor = augmented["image"]
        else:
            # Fallback if no transform is provided (e.g., simple testing)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            
        return img_tensor, label

    def get_class_counts(self) -> Dict[int, int]:
        """Return the number of samples per class."""
        counts = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        return counts
