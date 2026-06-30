"""
Central configuration for the Screen Recapture Detection project.

Design Decisions:
    - Dataclass-based for type safety, IDE autocompletion, and immutability control.
    - All filesystem paths derived from a single `project_root` via @property,
      ensuring portability across Windows and Linux without hardcoding.
    - ImageNet normalization constants centralized here so every pipeline stage
      uses identical values (prevents subtle train/inference mismatches).
    - Class labels stored as constants to avoid magic strings throughout the codebase.

Usage:
    from src.config import ProjectConfig
    config = ProjectConfig()
    print(config.real_dir)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class ProjectConfig:
    """Centralized project configuration.

    All filesystem paths are derived from ``project_root`` using ``pathlib.Path``
    properties, ensuring cross-platform compatibility and a single source of truth.

    Attributes:
        project_root: Absolute path to the project root directory.
        image_extensions: Tuple of supported image file extensions (lowercase).
        input_size: Square input dimension expected by all pretrained models.
        imagenet_mean: Per-channel mean for ImageNet normalization (R, G, B).
        imagenet_std: Per-channel std for ImageNet normalization (R, G, B).
        random_seed: Global seed for reproducibility across all random operations.
        class_real: Label string for genuine camera photos.
        class_fake: Label string for screen-recaptured photos.
        blur_threshold: Laplacian variance below which an image is flagged as blurry.
        phash_threshold: Maximum Hamming distance for perceptual duplicate detection.
        use_symlinks: If True, uses symbolic links to construct dataset splits. If False, copies files.
        batch_size: Number of samples per batch.
        epochs_stage1: Number of epochs to train just the classifier head.
        epochs_stage2: Number of epochs to fine-tune the entire network.
        lr_stage1: Learning rate for stage 1 (classifier).
        lr_stage2: Learning rate for stage 2 (fine-tuning).
        weight_decay: Weight decay for the AdamW optimizer.
        grad_clip_norm: Max norm for gradient clipping.
        early_stopping_patience: Number of epochs to wait for val_f1 improvement before stopping.
        optimizer_name: Name of optimizer (e.g. "AdamW").
        scheduler_name: Name of scheduler (e.g. "CosineAnnealingLR").
        use_class_weights: Whether to use automatically computed class weights for CrossEntropy.
    """

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    image_extensions: Tuple[str, ...] = (
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
    )
    input_size: int = 224
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    random_seed: int = 42
    class_real: str = "real"
    class_fake: str = "fake"
    blur_threshold: float = 100.0
    phash_threshold: int = 10
    use_symlinks: bool = False
    
    # Phase 3 Training Config
    batch_size: int = 32
    epochs_stage1: int = 5
    epochs_stage2: int = 15
    lr_stage1: float = 1e-3
    lr_stage2: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    optimizer_name: str = "AdamW"
    scheduler_name: str = "CosineAnnealingLR"
    use_class_weights: bool = True

    # ------------------------------------------------------------------ paths
    @property
    def dataset_dir(self) -> Path:
        """Root directory containing class sub-folders."""
        return self.project_root / "dataset"

    @property
    def real_dir(self) -> Path:
        """Directory containing genuine camera photos."""
        return self.dataset_dir / self.class_real

    @property
    def fake_dir(self) -> Path:
        """Directory containing screen-recaptured photos."""
        return self.dataset_dir / self.class_fake

    @property
    def models_dir(self) -> Path:
        """Directory for saved model checkpoints."""
        return self.project_root / "models"

    @property
    def outputs_dir(self) -> Path:
        """Directory for reports, plots, and exported artefacts."""
        return self.project_root / "outputs"

    @property
    def experiments_dir(self) -> Path:
        """Directory for training experiment outputs."""
        return self.outputs_dir / "experiments"

    @property
    def logs_dir(self) -> Path:
        """Directory for training and application logs."""
        return self.project_root / "logs"

    @property
    def scripts_dir(self) -> Path:
        """Directory for runnable pipeline scripts."""
        return self.project_root / "scripts"

    # --------------------------------------------------------------- helpers
    def ensure_directories(self) -> None:
        """Create all required project directories if they do not exist."""
        for directory in (
            self.models_dir,
            self.outputs_dir,
            self.experiments_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def get_class_dir(self, class_name: str) -> Path:
        """Return the dataset sub-directory for a given class label.

        Args:
            class_name: One of ``class_real`` or ``class_fake``.

        Returns:
            Absolute path to the class sub-directory.

        Raises:
            ValueError: If ``class_name`` is not a recognized label.
        """
        if class_name == self.class_real:
            return self.real_dir
        if class_name == self.class_fake:
            return self.fake_dir
        raise ValueError(
            f"Unknown class '{class_name}'. Expected '{self.class_real}' "
            f"or '{self.class_fake}'."
        )
