import sys
import os
import warnings
import logging

# Aggressively suppress all warnings and logging to ensure strict output format
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.CRITICAL)

import cv2
import json
from pathlib import Path

# Important: these imports happen after warnings are suppressed
from src.config import ProjectConfig
from src.inference import AntiSpoofPredictor


def main():
    if len(sys.argv) != 2:
        sys.exit(1)
        
    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        sys.exit(1)
        
    try:
        # Resolve production model paths
        model_dir = Path("models/production")
        ckpt_path = model_dir / "best_model.pth"
        config_path = model_dir / "config.json"
        
        if not ckpt_path.exists() or not config_path.exists():
            sys.exit(1)
            
        # Load configuration
        config = ProjectConfig()
        with open(config_path) as f:
            config_dict = json.load(f)
            
        for k, v in config_dict.items():
            if hasattr(config, k) and k != "project_root":
                if isinstance(v, list): v = tuple(v)
                setattr(config, k, v)
                
        # Initialize predictor on CPU for universal compatibility
        predictor = AntiSpoofPredictor(
            model_name="convnext_tiny", 
            checkpoint_path=ckpt_path, 
            config=config, 
            device="cpu"
        )
        
        # Read image
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            sys.exit(1)
            
        # Run inference
        res = predictor.predict(image_bgr)
        
        # User explicitly requested strict 0 or 1 output
        is_fake = 1 if res['probs']['fake'] >= 0.5 else 0
        print(f"{is_fake}")
        
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
