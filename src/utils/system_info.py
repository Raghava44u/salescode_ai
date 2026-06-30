"""
System information gathering utilities.

Design Decisions:
    - Automatically collects CPU, GPU, RAM, OS, and key package versions.
    - Dumps to a JSON file in the experiment directory for full provenance.
"""

import json
import platform
import sys
from pathlib import Path
from typing import Dict, Any

import psutil
import torch


def get_system_info() -> Dict[str, Any]:
    """Gather hardware, OS, and software environment information."""
    
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
    }

    # CUDA info
    info["cuda_available"] = torch.cuda.is_available()
    if info["cuda_available"]:
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(info["gpu_count"])]
    else:
        info["cuda_version"] = None
        info["cudnn_version"] = None
        info["gpu_count"] = 0
        info["gpus"] = []

    return info

def save_system_info(output_path: Path) -> None:
    """Gather and save system info to a JSON file."""
    info = get_system_info()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
