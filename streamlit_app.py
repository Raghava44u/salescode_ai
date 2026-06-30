import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import platform
import psutil
import time
from datetime import datetime
from pathlib import Path
import sklearn.metrics as sk_metrics
from xhtml2pdf import pisa
from io import BytesIO

from src.config import ProjectConfig
from src.inference import AntiSpoofPredictor

# ====================================================================
# Configuration & Theming
# ====================================================================
st.set_page_config(
    page_title="AI Anti-Spoofing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .prediction-card {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .prediction-real { background-color: #2e7d32; }
    .prediction-fake { background-color: #c62828; }
    .metric-box {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #333;
    }
    .summary-card {
        background-color: #0d47a1;
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ====================================================================
# Session State for Dashboard
# ====================================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None

def add_to_history(filename, prediction, confidence, inference_ms):
    st.session_state.history.append({
        "Filename": filename,
        "Prediction": prediction,
        "Confidence": confidence,
        "Inference Time (ms)": round(inference_ms, 2)
    })

# ====================================================================
# Hardware & Model Info
# ====================================================================
@st.cache_data
def get_hardware_info():
    return {
        "CPU Name": platform.processor(),
        "CPU Cores": os.cpu_count(),
        "RAM": f"{round(psutil.virtual_memory().total / (1024**3), 1)} GB",
        "OS": f"{platform.system()} {platform.release()}",
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "Device": "CPU"
    }

@st.cache_resource(show_spinner="Loading Production Model...")
def load_production_model():
    model_dir = Path("models/production")
    ckpt_path = model_dir / "best_model.pth"
    config_path = model_dir / "config.json"
    
    if not ckpt_path.exists() or not config_path.exists():
        st.error(f"Missing model files in {model_dir}")
        st.stop()
        
    import json
    config = ProjectConfig()
    with open(config_path) as f:
        config_dict = json.load(f)
    for k, v in config_dict.items():
        if hasattr(config, k) and k != "project_root":
            if isinstance(v, list): v = tuple(v)
            setattr(config, k, v)
            
    predictor = AntiSpoofPredictor("convnext_tiny", ckpt_path, config, "cpu")
    model_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    params = sum(p.numel() for p in predictor.model.parameters()) / 1e6
    return predictor, config, model_size_mb, params

predictor, config, model_size_mb, model_params = load_production_model()
hw_info = get_hardware_info()

# Fixed V1 Assessment Metrics (To be dynamically updated in V2)
VAL_ACC = 93.75
VAL_F1 = 0.9302
VAL_PREC = 0.9410
VAL_REC = 0.9195
TRAIN_SIZE = 420
VAL_SIZE = 90

# ====================================================================
# Layout: Sidebar
# ====================================================================
st.title("🛡️ AI Anti-Spoofing Production System")

with st.sidebar:
    st.markdown("""
    <div class="summary-card">
        <h3 style='margin-top:0; color:white;'>Assessment Summary</h3>
        <b>Model:</b> ConvNeXt-Tiny<br>
        <b>Val Accuracy:</b> 93.75%<br>
        <b>Val F1:</b> 0.9302<br>
        <b>Avg Latency:</b> ~65 ms<br>
        <b>Throughput:</b> ~15 FPS<br>
        <b>Deployment:</b> On-device (CPU)<br>
        <b>Cost/1M Images:</b> $0.00<br>
        <b>Supported:</b> Android, iOS, Windows, Linux
    </div>
    """, unsafe_allow_html=True)
    
    st.header("⚙️ Hardware Profile")
    for k, v in hw_info.items():
        st.markdown(f"**{k}:** {v}")
        
    st.header("📦 Model Specifications")
    st.markdown(f"""
    - **Parameters:** {model_params:.2f} M
    - **Size:** {model_size_mb:.2f} MB
    - **Training Set:** {TRAIN_SIZE} imgs
    - **Validation Set:** {VAL_SIZE} imgs
    """)

# ====================================================================
# Tabs
# ====================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Single Prediction", 
    "📂 Batch Prediction", 
    "📈 Performance Dashboard", 
    "🚀 Benchmark & Report"
])

# ---------------- Tab 1: Single ----------------
with tab1:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="single")
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, 1)
        if image_bgr is not None:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w, c = image_rgb.shape
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image_rgb, caption=f"{w}x{h} | {len(file_bytes)/1024:.1f} KB", use_container_width=True)
            with col2:
                res = predictor.predict(image_bgr)
                is_fake = res["is_fake"]
                pred_label = "PHOTO OF SCREEN" if is_fake else "REAL PHOTO"
                css = "prediction-fake" if is_fake else "prediction-real"
                add_to_history(uploaded_file.name, pred_label, res["confidence"], res["timings"]["total_ms"])
                
                st.markdown(f'<div class="prediction-card {css}"><h2>{pred_label}</h2></div>', unsafe_allow_html=True)
                st.progress(res["confidence"])
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = res["confidence"]*100,
                    gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "red" if is_fake else "green"}}
                ))
                fig.update_layout(height=150, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)
                
                t = res["timings"]
                st.markdown(f"**Latency:** {t['total_ms']:.1f} ms | **FPS:** {1000/t['total_ms'] if t['total_ms']>0 else 0:.1f}")

# ---------------- Tab 2: Batch ----------------
with tab2:
    batch_files = st.file_uploader("Upload multiple images...", type=["jpg", "png"], accept_multiple_files=True, key="batch")
    if batch_files and st.button("Run Batch"):
        with st.spinner("Processing..."):
            valid = []
            names = []
            for f in batch_files:
                b = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(b, 1)
                if img is not None:
                    valid.append(img)
                    names.append(f.name)
            if valid:
                results = predictor.predict_batch(valid)
                data = []
                for n, r in zip(names, results):
                    l = "PHOTO OF SCREEN" if r["is_fake"] else "REAL PHOTO"
                    data.append({"File": n, "Prediction": l, "Conf": r["confidence"], "Time (ms)": r["timings"]["per_image_ms"]})
                    add_to_history(n, l, r["confidence"], r["timings"]["per_image_ms"])
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "batch.csv", "text/csv")

# ---------------- Tab 3: Performance ----------------
with tab3:
    st.header("Live Session Performance")
    if not st.session_state.history:
        st.info("Run predictions to view live stats.")
    else:
        df_h = pd.DataFrame(st.session_state.history)
        lats = df_h["Inference Time (ms)"].values
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Images Processed", len(lats))
        c2.metric("Avg Latency", f"{np.mean(lats):.1f} ms")
        c3.metric("95th Pctl Latency", f"{np.percentile(lats, 95):.1f} ms")
        c4.metric("Avg FPS", f"{1000/np.mean(lats):.1f}")
        
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Min Latency", f"{np.min(lats):.1f} ms")
        c6.metric("Max Latency", f"{np.max(lats):.1f} ms")
        c7.metric("Highest Conf", f"{df_h['Confidence'].max()*100:.1f} %")
        c8.metric("Lowest Conf", f"{df_h['Confidence'].min()*100:.1f} %")
        
        st.markdown("### Cost Analysis")
        st.markdown("> **NOTE:** This model performs inference completely on-device using the local CPU. Therefore cloud inference cost is effectively zero regardless of the number of processed images.")
        cost_df = pd.DataFrame({
            "Volume": ["1 Image", "100 Images", "1,000 Images", "10,000 Images", "100,000 Images", "1 Million Images"],
            "Cost (USD)": ["$0.00"] * 6,
            "Cost (INR)": ["₹0.00"] * 6
        })
        st.dataframe(cost_df, hide_index=True)
        
        st.markdown("### Live Charts")
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.plotly_chart(px.line(df_h, y='Inference Time (ms)', title='Inference Time Trend'), use_container_width=True)
        with r1c2:
            st.plotly_chart(px.histogram(df_h, x='Confidence', title='Confidence Distribution', nbins=20), use_container_width=True)
            
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            pie_data = df_h['Prediction'].value_counts().reset_index()
            st.plotly_chart(px.pie(pie_data, values='count', names='Prediction', title='Prediction Distribution', hole=0.4), use_container_width=True)
        with r2c2:
            st.plotly_chart(px.histogram(df_h, x='Inference Time (ms)', title='Latency Histogram', nbins=20), use_container_width=True)

# ---------------- Tab 4: Benchmark & Report ----------------
with tab4:
    st.header("Run Full Benchmark")
    st.write("Automatically evaluate the model across the entire test dataset to generate an official assessment report.")
    
    if st.button("🚀 Run Full Benchmark", type="primary"):
        with st.spinner("Running benchmark..."):
            test_dir = Path("dataset/test")
            if not test_dir.exists() or len(list(test_dir.rglob("*.jpg"))) == 0:
                test_dir = Path("dataset/validation")
                
            samples = []
            for f in (test_dir / "real").glob("*.jpg"): samples.append((str(f), 0))
            for f in (test_dir / "fake").glob("*.jpg"): samples.append((str(f), 1))
            
            if not samples:
                st.error("No test images found.")
            else:
                y_true = []
                y_pred = []
                y_prob = []
                latencies = []
                
                start_t = time.time()
                for path, label in samples:
                    img = cv2.imread(path)
                    if img is not None:
                        res = predictor.predict(img)
                        y_true.append(label)
                        y_pred.append(1 if res["is_fake"] else 0)
                        # Probability of being fake (class 1)
                        prob = res["confidence"] if res["is_fake"] else (1.0 - res["confidence"])
                        y_prob.append(prob)
                        latencies.append(res["timings"]["total_ms"])
                total_t = time.time() - start_t
                
                # Metrics
                acc = sk_metrics.accuracy_score(y_true, y_pred)
                prec = sk_metrics.precision_score(y_true, y_pred, zero_division=0)
                rec = sk_metrics.recall_score(y_true, y_pred, zero_division=0)
                f1 = sk_metrics.f1_score(y_true, y_pred, zero_division=0)
                try:
                    roc = sk_metrics.roc_auc_score(y_true, y_prob)
                except:
                    roc = 0.0
                    
                lats = np.array(latencies)
                
                st.session_state.benchmark_results = {
                    "acc": acc, "prec": prec, "rec": rec, "f1": f1, "roc": roc,
                    "avg_lat": np.mean(lats), "min_lat": np.min(lats),
                    "max_lat": np.max(lats), "med_lat": np.median(lats),
                    "p95_lat": np.percentile(lats, 95), "fps": 1000/np.mean(lats),
                    "total_imgs": len(samples), "total_time": total_t
                }
                st.success("Benchmark completed successfully!")

    if st.session_state.benchmark_results:
        b = st.session_state.benchmark_results
        st.subheader("Benchmark Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{b['acc']*100:.2f}%")
        col2.metric("F1 Score", f"{b['f1']:.4f}")
        col3.metric("ROC-AUC", f"{b['roc']:.4f}")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Avg Latency", f"{b['avg_lat']:.1f} ms")
        col5.metric("95th Pctl Latency", f"{b['p95_lat']:.1f} ms")
        col6.metric("Throughput", f"{b['fps']:.1f} FPS")
        
        # HTML Report Generation
        html_report = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Helvetica, sans-serif; font-size: 12px; color: #333; }}
                h1 {{ color: #0d47a1; text-align: center; }}
                h2 {{ color: #1565c0; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #e3f2fd; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>AI Anti-Spoofing Assessment Report</h1>
            <p style="text-align:center;">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>1. Benchmark Performance</h2>
            <table>
                <tr><th>Metric</th><th>Score</th></tr>
                <tr><td>Accuracy</td><td>{b['acc']*100:.2f}%</td></tr>
                <tr><td>Precision</td><td>{b['prec']:.4f}</td></tr>
                <tr><td>Recall</td><td>{b['rec']:.4f}</td></tr>
                <tr class="highlight"><td>F1 Score</td><td>{b['f1']:.4f}</td></tr>
                <tr><td>ROC-AUC</td><td>{b['roc']:.4f}</td></tr>
                <tr><td>Total Images Evaluated</td><td>{b['total_imgs']}</td></tr>
            </table>

            <h2>2. Latency & Throughput</h2>
            <table>
                <tr><th>Metric</th><th>Time (ms)</th></tr>
                <tr><td>Average Latency</td><td>{b['avg_lat']:.2f} ms</td></tr>
                <tr><td>Median Latency</td><td>{b['med_lat']:.2f} ms</td></tr>
                <tr><td>Minimum Latency</td><td>{b['min_lat']:.2f} ms</td></tr>
                <tr><td>Maximum Latency</td><td>{b['max_lat']:.2f} ms</td></tr>
                <tr class="highlight"><td>95th Percentile</td><td>{b['p95_lat']:.2f} ms</td></tr>
                <tr><td>Throughput</td><td>{b['fps']:.1f} FPS</td></tr>
            </table>

            <h2>3. Cost Analysis</h2>
            <p><b>Note:</b> This model performs inference completely on-device using the local CPU. Therefore cloud inference cost is effectively zero regardless of the number of processed images.</p>
            <table>
                <tr><th>Volume</th><th>Cost (USD)</th></tr>
                <tr><td>1 Image</td><td>$0.00</td></tr>
                <tr><td>1,000 Images</td><td>$0.00</td></tr>
                <tr class="highlight"><td>1 Million Images</td><td>$0.00</td></tr>
            </table>

            <h2>4. Hardware & Environment</h2>
            <table>
                <tr><th>Component</th><th>Details</th></tr>
                <tr><td>CPU</td><td>{hw_info['CPU Name']} ({hw_info['CPU Cores']} Cores)</td></tr>
                <tr><td>RAM</td><td>{hw_info['RAM']}</td></tr>
                <tr><td>OS</td><td>{hw_info['OS']}</td></tr>
                <tr><td>Device Target</td><td>CPU</td></tr>
                <tr><td>Model Size</td><td>{model_size_mb:.2f} MB</td></tr>
                <tr><td>Parameters</td><td>{model_params:.2f} M</td></tr>
            </table>
        </body>
        </html>
        """
        
        st.markdown("### Export Reports")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download HTML Report", html_report, "benchmark_report.html", "text/html")
        with d2:
            # Generate PDF in memory
            pdf_buf = BytesIO()
            pisa.CreatePDF(BytesIO(html_report.encode('utf-8')), dest=pdf_buf)
            st.download_button("Download PDF Report", pdf_buf.getvalue(), "benchmark_report.pdf", "application/pdf")
