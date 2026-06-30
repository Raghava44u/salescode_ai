"""
Training callbacks.

Design Decisions:
    - EarlyStopping monitors Validation F1.
    - ModelCheckpoint saves the best model based on F1 (falling back to loss on tie).
    - Loggers track learning rate for debugging fine-tuning behavior.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger("callbacks")


class EarlyStopping:
    """Stops training when a monitored metric has stopped improving."""
    
    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False
            
        if self.mode == "max":
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


class ModelCheckpoint:
    """Saves the best and last model checkpoints."""
    
    def __init__(self, exp_dir: Path):
        self.best_path = exp_dir / "best_model.pth"
        self.last_path = exp_dir / "last_model.pth"
        
        self.best_f1 = -1.0
        self.best_loss = float('inf')
        
    def __call__(self, model, optimizer, scheduler, epoch: int, metrics: Dict[str, float]):
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics
        }
        
        # Always save last
        torch.save(state, self.last_path)
        
        # Check if best
        current_f1 = metrics.get("val_f1", 0.0)
        current_loss = metrics.get("val_loss", float('inf'))
        
        is_best = False
        if current_f1 > self.best_f1:
            is_best = True
        elif abs(current_f1 - self.best_f1) < 1e-5:
            # Tie breaker: validation loss
            if current_loss < self.best_loss:
                is_best = True
                
        if is_best:
            self.best_f1 = current_f1
            self.best_loss = current_loss
            torch.save(state, self.best_path)
            logger.info(f"Saved new best model at epoch {epoch} (F1: {current_f1:.4f}, Loss: {current_loss:.4f})")
            
        return is_best


class CSVLogger:
    """Logs epoch metrics to a CSV file."""
    
    def __init__(self, exp_dir: Path):
        self.filepath = exp_dir / "training_history.csv"
        self.fieldnames = []
        self.file = None
        self.writer = None
        
    def log(self, metrics: Dict[str, Any]):
        if not self.fieldnames:
            self.fieldnames = list(metrics.keys())
            self.file = open(self.filepath, "w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
            
        # Ensure all keys exist
        row = {k: metrics.get(k, "") for k in self.fieldnames}
        self.writer.writerow(row)
        self.file.flush()
        
    def close(self):
        if self.file:
            self.file.close()


class TensorBoardLogger:
    """Logs epoch metrics to TensorBoard."""
    
    def __init__(self, exp_dir: Path):
        self.log_dir = exp_dir / "tensorboard"
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
    def log(self, metrics: Dict[str, Any], epoch: int):
        for k, v in metrics.items():
            if k == "epoch":
                continue
            if isinstance(v, (int, float)):
                self.writer.add_scalar(k, v, epoch)
                
    def close(self):
        self.writer.close()
