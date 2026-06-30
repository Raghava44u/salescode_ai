"""
Inference module for running predictions.

Design Decisions:
    - Provides a standard interface (load_model, predict) decoupled from training logic.
    - Handles preprocessing strictly identical to the validation pipeline.
    - Designed to be imported by production scripts or Streamlit apps later.
"""

import cv2
import numpy as np
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, List, Dict, Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models import build_model
from src.config import ProjectConfig

class AntiSpoofPredictor:
    """Production predictor for Screen Recapture Detection."""
    
    def __init__(self, model_name: str, checkpoint_path: Path, config: ProjectConfig, device: str = None):
        self.config = config
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.model = self._load_model(model_name, checkpoint_path)
        self.transform = self._build_transforms()
        
    def _load_model(self, model_name: str, checkpoint_path: Path) -> nn.Module:
        """Initialize model architecture and load trained weights."""
        model = build_model(model_name, num_classes=2, pretrained=False)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        import pickle
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except (RuntimeError, FutureWarning, pickle.UnpicklingError):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model
        
    def _build_transforms(self) -> A.Compose:
        """Create exact validation preprocessing pipeline."""
        return A.Compose([
            A.Resize(self.config.input_size, self.config.input_size),
            A.Normalize(
                mean=self.config.imagenet_mean,
                std=self.config.imagenet_std,
            ),
            ToTensorV2(),
        ])
        
    def preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """Convert BGR numpy image to normalized tensor."""
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image_rgb)
        tensor = augmented["image"].unsqueeze(0)  # Add batch dim
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Run inference on a single image.
        
        Returns:
            Dict containing predicted class label, index, and confidence score.
        """
        t0 = time.perf_counter()
        tensor = self.preprocess(image_bgr)
        t1 = time.perf_counter()
        
        outputs = self.model(tensor)
        t2 = time.perf_counter()
        
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        
        # Mapping 0=Real, 1=Fake
        class_name = self.config.class_fake if pred_idx == 1 else self.config.class_real
        t3 = time.perf_counter()
        
        return {
            "prediction": class_name,
            "class_idx": pred_idx,
            "confidence": confidence,
            "is_fake": pred_idx == 1,
            "probs": {"real": probs[0].item(), "fake": probs[1].item()},
            "timings": {
                "preprocess_ms": (t1 - t0) * 1000,
                "forward_ms": (t2 - t1) * 1000,
                "postprocess_ms": (t3 - t2) * 1000,
                "total_ms": (t3 - t0) * 1000
            }
        }

    @torch.no_grad()
    def predict_batch(self, images_bgr: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Run inference on a batch of images."""
        t0 = time.perf_counter()
        tensors = []
        for img in images_bgr:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aug = self.transform(image=img_rgb)["image"]
            tensors.append(aug)
            
        batch_tensor = torch.stack(tensors).to(self.device)
        t1 = time.perf_counter()
        
        outputs = self.model(batch_tensor)
        t2 = time.perf_counter()
        
        probs_batch = torch.softmax(outputs, dim=1)
        preds_batch = torch.argmax(probs_batch, dim=1)
        
        results = []
        for i in range(len(images_bgr)):
            pred_idx = preds_batch[i].item()
            conf = probs_batch[i, pred_idx].item()
            class_name = self.config.class_fake if pred_idx == 1 else self.config.class_real
            results.append({
                "prediction": class_name,
                "class_idx": pred_idx,
                "confidence": conf,
                "is_fake": pred_idx == 1,
                "probs": {"real": probs_batch[i, 0].item(), "fake": probs_batch[i, 1].item()}
            })
            
        t3 = time.perf_counter()
        batch_timings = {
            "preprocess_ms": (t1 - t0) * 1000,
            "forward_ms": (t2 - t1) * 1000,
            "postprocess_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
            "per_image_ms": ((t3 - t0) * 1000) / max(1, len(images_bgr))
        }
        
        for res in results:
            res["timings"] = batch_timings
            
        return results
