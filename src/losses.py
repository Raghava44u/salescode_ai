"""
Loss functions.

Design Decisions:
    - Provides a standard CrossEntropyLoss.
    - Automates inverse class weighting if specified to handle dataset imbalance.
"""

import torch
import torch.nn as nn
from typing import Dict

from src.config import ProjectConfig

def build_loss(config: ProjectConfig, class_counts: Dict[int, int] = None, device: torch.device = None) -> nn.Module:
    """Build the loss function, optionally with class weights.
    
    Args:
        config: Project configuration.
        class_counts: Dictionary mapping class index to number of samples (from training set).
        device: Device to place the weights on.
        
    Returns:
        The loss module.
    """
    if config.use_class_weights and class_counts:
        # Inverse frequency weighting
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        
        weights = []
        for i in range(num_classes):
            count = class_counts.get(i, 0)
            if count == 0:
                weight = 1.0
            else:
                weight = total_samples / (num_classes * count)
            weights.append(weight)
            
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        if device is not None:
            weight_tensor = weight_tensor.to(device)
            
        return nn.CrossEntropyLoss(weight=weight_tensor)
        
    return nn.CrossEntropyLoss()
