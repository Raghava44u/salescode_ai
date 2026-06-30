"""
Generic Training Engine.

Design Decisions:
    - Reusable Trainer class completely decoupled from specific model architectures.
    - Uses Mixed Precision (AMP) automatically on CUDA.
    - Implements gradient clipping for fine-tuning stability.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.metrics import MetricTracker
from src.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoardLogger
from src.config import ProjectConfig

logger = logging.getLogger("trainer")

class Trainer:
    """Generic training engine."""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        exp_dir: Path,
        config: ProjectConfig,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.exp_dir = exp_dir
        self.config = config
        self.scheduler = scheduler
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.device.type == "cuda")
        
        self.train_tracker = MetricTracker()
        self.val_tracker = MetricTracker()
        
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience, mode="max")
        self.checkpoint = ModelCheckpoint(exp_dir)
        self.csv_logger = CSVLogger(exp_dir)
        self.tb_logger = TensorBoardLogger(exp_dir)
        
        self.history: List[Dict[str, Any]] = []
        self.current_epoch = 0

    def get_lr(self) -> float:
        """Get the current learning rate from the optimizer."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        self.train_tracker.reset()
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=self.device.type == "cuda"):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
            self.scaler.scale(loss).backward()
            
            if self.config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Track metrics
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            
            self.train_tracker.update(
                y_true=targets.cpu().numpy(),
                y_prob=probs,
                y_pred=preds,
                loss=loss.item()
            )
            
        return self.train_tracker.compute()

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        self.val_tracker.reset()
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            with torch.amp.autocast('cuda', enabled=self.device.type == "cuda"):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            self.val_tracker.update(
                y_true=targets.cpu().numpy(),
                y_prob=probs,
                y_pred=preds,
                loss=loss.item()
            )
            
        return self.val_tracker.compute()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, start_epoch: int = 1):
        for epoch in range(start_epoch, start_epoch + epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            current_lr = self.get_lr()
            logger.info(f"Epoch {epoch}/{start_epoch + epochs - 1} | LR: {current_lr:.2e}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # LR Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("val_loss", 0.0))
                else:
                    self.scheduler.step()
                    
            epoch_time = time.time() - start_time
            
            # Log metrics
            metrics = {"epoch": epoch, "lr": current_lr, "time_sec": round(epoch_time, 1)}
            for k, v in train_metrics.items():
                metrics[f"train_{k}"] = v
            for k, v in val_metrics.items():
                metrics[f"val_{k}"] = v
                
            self.history.append(metrics)
            
            log_str = f"Time: {epoch_time:.1f}s | "
            log_str += f"Train Loss: {metrics['train_loss']:.4f} Acc: {metrics['train_accuracy']:.4f} | "
            log_str += f"Val Loss: {metrics['val_loss']:.4f} Acc: {metrics['val_accuracy']:.4f} F1: {metrics['val_f1']:.4f}"
            logger.info(log_str)
            
            self.csv_logger.log(metrics)
            self.tb_logger.log(metrics, epoch)
            
            # Checkpoint
            is_best = self.checkpoint(self.model, self.optimizer, self.scheduler, epoch, metrics)
            if is_best:
                # Save best plots
                self.val_tracker.save_confusion_matrix(self.exp_dir / "confusion_matrix.png")
                self.val_tracker.save_roc_curve(self.exp_dir / "roc_curve.png")
                self.val_tracker.save_pr_curve(self.exp_dir / "precision_recall_curve.png")
                self.val_tracker.generate_classification_report(self.exp_dir / "classification_report.txt")
            
            # Early Stopping
            if self.early_stopping(metrics.get("val_f1", 0.0)):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    def cleanup(self):
        self.csv_logger.close()
        self.tb_logger.close()
