"""
Reproducibility utilities.

Design Decisions:
    - Sets seeds for Python, NumPy, PyTorch, and CUDA.
    - Forces deterministic algorithms in cuDNN to ensure identical runs.
"""

import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.
    
    Args:
        seed: The integer seed to use.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # Force deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
