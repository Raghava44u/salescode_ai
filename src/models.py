"""
Model factory for pretrained architectures.

Design Decisions:
    - Restricts support strictly to the requested backbones to maintain stability.
    - Standardizes the classifier head replacement across disparate architectures.
    - Provides parameter counting utilities.
"""

import torch
import torch.nn as nn
import torchvision.models as models

def build_model(model_name: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Build a pretrained model and replace its classifier head.
    
    Args:
        model_name: One of 'mobilenet_v3_large', 'efficientnet_b0', 'convnext_tiny'.
        num_classes: Number of output classes (binary = 2).
        pretrained: Whether to download and load ImageNet pretrained weights.
        
    Returns:
        The constructed PyTorch model.
    """
    model_name = model_name.lower().strip()
    
    if model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        # MobileNetV3 uses a Sequential classifier
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        # EfficientNet uses a Sequential classifier
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    elif model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_tiny(weights=weights)
        # ConvNeXt uses a Sequential classifier
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            "Supported: 'mobilenet_v3_large', 'efficientnet_b0', 'convnext_tiny'"
        )
        
    return model

def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze the backbone while keeping the classifier trainable.
    
    Args:
        model: The constructed model.
        trainable: True to unfreeze backbone, False to freeze.
    """
    # 1. Set everything to the requested trainable state
    for param in model.parameters():
        param.requires_grad = trainable
        
    # 2. Force the final classifier to always be trainable
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True

def get_model_summary(model: nn.Module, input_size: int = 224) -> str:
    """Generate a string summarizing the model parameters and shapes."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    # Approx size in MB (assuming 32-bit float = 4 bytes)
    model_size_mb = total_params * 4 / (1024 ** 2)
    
    summary = []
    summary.append("=" * 50)
    summary.append(f"Model Architecture: {model.__class__.__name__}")
    summary.append("=" * 50)
    summary.append(f"Input Shape         : (Batch, 3, {input_size}, {input_size})")
    summary.append(f"Output Shape        : (Batch, 2)")
    summary.append("-" * 50)
    summary.append(f"Total Parameters    : {total_params:,}")
    summary.append(f"Trainable Parameters: {trainable_params:,}")
    summary.append(f"Frozen Parameters   : {frozen_params:,}")
    summary.append(f"Estimated Size (MB) : {model_size_mb:.2f} MB")
    summary.append("=" * 50)
    
    return "\n".join(summary)
