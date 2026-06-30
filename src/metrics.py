"""
Metrics tracking and evaluation module.

Design Decisions:
    - Encapsulates all metric calculations using scikit-learn.
    - Tracks Loss, Accuracy, Precision, Recall, F1, and ROC-AUC.
    - Generates plots directly from the tracked validation results.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay
)


class MetricTracker:
    """Tracks predictions and computes metrics."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.y_true = []
        self.y_prob = []  # Probability of class 1 (Fake)
        self.y_pred = []
        self.losses = []
        
    def update(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray, loss: float = None):
        self.y_true.extend(y_true.tolist())
        self.y_prob.extend(y_prob.tolist())
        self.y_pred.extend(y_pred.tolist())
        if loss is not None:
            self.losses.append(loss)
            
    def compute(self) -> Dict[str, float]:
        if not self.y_true:
            return {}
            
        metrics = {
            "loss": np.mean(self.losses) if self.losses else 0.0,
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(self.y_true, self.y_pred, zero_division=0),
            "recall": recall_score(self.y_true, self.y_pred, zero_division=0),
            "f1": f1_score(self.y_true, self.y_pred, zero_division=0),
        }
        
        try:
            metrics["roc_auc"] = roc_auc_score(self.y_true, self.y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0
            
        return metrics

    def generate_classification_report(self, out_path: Path):
        """Generate and save the scikit-learn classification report."""
        if not self.y_true:
            return
            
        report = classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=["Real", "Fake"],
            zero_division=0
        )
        
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
            
    def save_confusion_matrix(self, out_path: Path):
        if not self.y_true:
            return
            
        cm = confusion_matrix(self.y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
        
        plt.figure(figsize=(8, 6))
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close("all")

    def save_roc_curve(self, out_path: Path):
        if not self.y_true:
            return
            
        try:
            fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close("all")
        except ValueError:
            pass

    def save_pr_curve(self, out_path: Path):
        if not self.y_true:
            return
            
        try:
            precision, recall, _ = precision_recall_curve(self.y_true, self.y_prob)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='purple', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close("all")
        except ValueError:
            pass


def plot_training_curves(history: List[Dict[str, Any]], out_path: Path):
    """Plot Loss and F1 score curves across epochs."""
    if not history:
        return
        
    epochs = [h["epoch"] for h in history]
    train_loss = [h.get("train_loss", 0.0) for h in history]
    val_loss = [h.get("val_loss", 0.0) for h in history]
    
    val_f1 = [h.get("val_f1", 0.0) for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss", marker="o")
    ax1.plot(epochs, val_loss, label="Val Loss", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 Score
    ax2.plot(epochs, val_f1, label="Val F1 Score", marker="s", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.set_title("Validation F1 Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close("all")
