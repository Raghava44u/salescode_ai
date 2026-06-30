"""
Error Analysis tool for Anti-Spoofing Models.
Generates an HTML report of false positives and false negatives with thumbnails,
and exports a CSV file detailing confidence scores and latencies.
"""

import os
import cv2
import json
import base64
import pandas as pd
from pathlib import Path

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ProjectConfig
from src.inference import AntiSpoofPredictor

def image_to_base64(img_bgr):
    """Convert a cv2 image to a base64 string for HTML embedding."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Resize thumbnail to save space
    h, w = img_rgb.shape[:2]
    max_dim = 256
    scale = max_dim / max(h, w)
    if scale < 1:
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode('utf-8')

def main():
    print("Running Error Analysis...")
    model_dir = Path("models/production")
    ckpt_path = model_dir / "best_model.pth"
    config_path = model_dir / "config.json"
    
    if not ckpt_path.exists():
        print(f"Error: {ckpt_path} not found.")
        sys.exit(1)
        
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
    
    # We will test on validation set as test might not exist or be small
    test_dir = Path("dataset/validation")
    if not test_dir.exists():
        print(f"Error: {test_dir} not found.")
        sys.exit(1)
        
    samples = []
    real_dir = test_dir / "real"
    fake_dir = test_dir / "fake"
    
    if real_dir.exists():
        for f in real_dir.glob("*.jpg"):
            samples.append((str(f), "real"))
    if fake_dir.exists():
        for f in fake_dir.glob("*.jpg"):
            samples.append((str(f), "fake"))
            
    print(f"Loaded {len(samples)} images for analysis.")
    
    errors = []
    
    # Run Inference
    for img_path, true_label in samples:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        res = predictor.predict(img_bgr)
        pred_label = res["prediction"]
        
        if pred_label != true_label:
            confidence = res["confidence"]
            b64_img = image_to_base64(img_bgr)
            error_data = {
                "filename": Path(img_path).name,
                "true label": true_label,
                "predicted label": pred_label,
                "confidence": confidence,
                "inference time": res["timings"]["total_ms"],
                "b64": b64_img
            }
            errors.append(error_data)
            print(f"Error Found: {Path(img_path).name} (True: {true_label}, Pred: {pred_label}, Conf: {confidence:.2f})")
            
    # Group errors
    high_conf_mistakes = [e for e in errors if e["confidence"] > 0.8]
    low_conf_mistakes = [e for e in errors if e["confidence"] <= 0.8]
    
    print(f"\nAnalysis complete. Found {len(errors)} errors.")
    print(f"High-confidence mistakes: {len(high_conf_mistakes)}")
    print(f"Low-confidence mistakes: {len(low_conf_mistakes)}")
    
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    
    # Export CSV
    csv_path = out_dir / "error_analysis.csv"
    df = pd.DataFrame(errors)
    if not df.empty:
        df_csv = df.drop(columns=["b64"])
        df_csv.to_csv(csv_path, index=False)
        print(f"Exported CSV to {csv_path}")
    else:
        # Save empty CSV to maintain contract
        pd.DataFrame(columns=["filename", "true label", "predicted label", "confidence", "inference time"]).to_csv(csv_path, index=False)
        print(f"Exported empty CSV to {csv_path} (No errors found!)")
        
    # Generate HTML
    html_path = out_dir / "error_analysis.html"
    
    html_content = """
    <html>
    <head>
        <title>Error Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px; }
            h1, h2 { color: #333; }
            .grid { display: flex; flex-wrap: wrap; gap: 20px; }
            .card { background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; width: 260px; }
            .card img { max-width: 100%; border-radius: 4px; }
            .high-conf { border-left: 5px solid #d32f2f; }
            .low-conf { border-left: 5px solid #fbc02d; }
            .info { text-align: left; margin-top: 10px; font-size: 14px; }
            .label-true { color: #2e7d32; font-weight: bold; }
            .label-pred { color: #d32f2f; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Error Analysis Report</h1>
    """
    
    if not errors:
        html_content += "<p>No errors found! The model is perfect on this dataset.</p>"
    else:
        def render_section(title, mistakes, css_class):
            section = f"<h2>{title} ({len(mistakes)} errors)</h2><div class='grid'>"
            for m in mistakes:
                section += f"""
                <div class="card {css_class}">
                    <img src="data:image/jpeg;base64,{m['b64']}" />
                    <div class="info">
                        <b>File:</b> {m['filename']}<br>
                        <b>True:</b> <span class="label-true">{m['true label'].upper()}</span><br>
                        <b>Pred:</b> <span class="label-pred">{m['predicted label'].upper()}</span><br>
                        <b>Confidence:</b> {m['confidence']:.4f}<br>
                        <b>Time:</b> {m['inference time']:.1f} ms
                    </div>
                </div>
                """
            section += "</div>"
            return section
            
        html_content += render_section("High-Confidence Mistakes (Confidence > 0.8)", high_conf_mistakes, "high-conf")
        html_content += render_section("Low-Confidence Mistakes (Confidence &le; 0.8)", low_conf_mistakes, "low-conf")
        
    html_content += """
    </body>
    </html>
    """
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Exported HTML report to {html_path}")

if __name__ == "__main__":
    main()
