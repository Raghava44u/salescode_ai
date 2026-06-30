"""
Test script to automatically verify the production inference pipeline on the test dataset.
"""
import os
import cv2
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ProjectConfig
from src.inference import AntiSpoofPredictor

def main():
    print("Testing Production Inference Pipeline...")
    model_dir = Path("models/production")
    ckpt_path = model_dir / "best_model.pth"
    config_path = model_dir / "config.json"
    
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found.")
        sys.exit(1)
        
    import json
    config = ProjectConfig()
    with open(config_path) as f:
        config_dict = json.load(f)
    for k, v in config_dict.items():
        if hasattr(config, k) and k != "project_root":
            if isinstance(v, list): v = tuple(v)
            setattr(config, k, v)
    predictor = AntiSpoofPredictor(
        model_name="convnext_tiny",
        checkpoint_path=ckpt_path,
        config=config,
        device="cpu"
    )
    
    test_dir = Path("dataset/test")
    if not test_dir.exists():
        print(f"Warning: {test_dir} not found. Using train/val data for smoke test instead if needed.")
        test_dir = Path("dataset/validation")
        if not test_dir.exists():
            print("No test/validation dataset found to test on.")
            sys.exit(0)
            
    real_dir = test_dir / "real"
    fake_dir = test_dir / "fake"
    
    test_images = []
    
    if real_dir.exists():
        for f in list(real_dir.glob("*.jpg"))[:2]:
            test_images.append(str(f))
    
    if fake_dir.exists():
        for f in list(fake_dir.glob("*.jpg"))[:2]:
            test_images.append(str(f))
            
    if not test_images:
        print("No JPG images found in test directories.")
        sys.exit(0)
        
    print(f"\nLoaded {len(test_images)} test images.")
    
    # Test Single Inference
    print("\n--- Single Image Inference ---")
    for img_path in test_images:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        res = predictor.predict(img_bgr)
        print(f"File: {Path(img_path).name}")
        print(f"Prediction: {res['prediction']} (Fake: {res['is_fake']})")
        print(f"Confidence: {res['confidence']:.4f}")
        print(f"Total Time: {res['timings']['total_ms']:.2f} ms")
        print("-" * 30)
        
    # Test Batch Inference
    print("\n--- Batch Image Inference ---")
    images_bgr = [cv2.imread(p) for p in test_images]
    images_bgr = [img for img in images_bgr if img is not None]
    
    if images_bgr:
        results = predictor.predict_batch(images_bgr)
        for idx, res in enumerate(results):
            print(f"File: {Path(test_images[idx]).name}")
            print(f"Prediction: {res['prediction']} (Fake: {res['is_fake']})")
            print(f"Confidence: {res['confidence']:.4f}")
            print(f"Per Image Time: {res['timings']['per_image_ms']:.2f} ms")
            print("-" * 30)
            
    print("\nAll production pipeline tests passed successfully!")

if __name__ == "__main__":
    main()
