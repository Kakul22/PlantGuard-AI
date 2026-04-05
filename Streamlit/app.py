import streamlit as st
import requests
from PIL import Image
import io
import base64

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

FLASK_URL = "http://127.0.0.1:5000/predict"

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500;600&display=swap');

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2017 0%, #1a3a28 50%, #0d1f14 100%);
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(10, 30, 18, 0.95) !important;
    border-right: 1px solid rgba(74, 222, 128, 0.2);
}

/* Main title */
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4ade80, #86efac, #bbf7d0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
}

.sub-title {
    text-align: center;
    color: #86efac;
    font-size: 1rem;
    margin-bottom: 2rem;
    opacity: 0.8;
}

/* Cards */
.result-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(74, 222, 128, 0.25);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}

.healthy-badge {
    background: linear-gradient(135deg, #065f46, #047857);
    color: #d1fae5;
    padding: 0.5rem 1.2rem;
    border-radius: 999px;
    font-size: 1rem;
    font-weight: 600;
    display: inline-block;
    margin: 0.5rem 0;
}

.disease-badge {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    color: #fee2e2;
    padding: 0.5rem 1.2rem;
    border-radius: 999px;
    font-size: 1rem;
    font-weight: 600;
    display: inline-block;
    margin: 0.5rem 0;
}

.advice-box {
    background: rgba(251, 191, 36, 0.1);
    border-left: 4px solid #fbbf24;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #fef3c7;
    margin: 1rem 0;
}

/* Architecture boxes */
.arch-layer {
    background: rgba(74, 222, 128, 0.08);
    border: 1px solid rgba(74, 222, 128, 0.3);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin: 0.5rem 0;
    color: #d1fae5;
}

.arch-layer h4 {
    color: #4ade80;
    margin: 0 0 0.3rem 0;
    font-size: 1rem;
}

.arch-layer p {
    margin: 0;
    font-size: 0.85rem;
    opacity: 0.8;
}

/* Upload area */
.upload-hint {
    text-align: center;
    color: #86efac;
    font-size: 0.9rem;
    opacity: 0.7;
    margin-top: 0.5rem;
}

/* Metric boxes */
.metric-box {
    background: rgba(74, 222, 128, 0.08);
    border: 1px solid rgba(74, 222, 128, 0.2);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    color: #d1fae5;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #4ade80;
}

.metric-label {
    font-size: 0.8rem;
    opacity: 0.7;
    margin-top: 0.2rem;
}

/* Divider */
hr { border-color: rgba(74, 222, 128, 0.15) !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #16a34a, #15803d) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #15803d, #166534) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(74, 222, 128, 0.3) !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    color: #86efac !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    color: #4ade80 !important;
    border-bottom-color: #4ade80 !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #4ade80, #22c55e) !important;
}

/* All text white-ish */
p, li, label { color: #d1fae5 !important; }
h1, h2, h3 { color: #bbf7d0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size: 3rem;'>🌿</div>
        <div style='font-family: Playfair Display, serif; font-size: 1.3rem;
                    color: #4ade80; font-weight: 700;'>PlantGuard AI</div>
        <div style='font-size: 0.75rem; color: #86efac; opacity: 0.7;'>
            Powered by TensorFlow
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        ["🔍  Detect Disease", "🏗️  Model Architecture", "📊  About & Stats"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("""
    <div style='font-size:0.75rem; color:#86efac; opacity:0.6; text-align:center;'>
        Supports: Tomato · Potato · Pepper<br>
        15 disease classes<br>
        ~96% accuracy
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 1 — DETECT DISEASE
# ════════════════════════════════════════════════════════════════
if "🔍" in page:

    st.markdown('<div class="main-title">🌿 Plant Disease Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload or capture a leaf photo — AI will diagnose it instantly</div>', unsafe_allow_html=True)

    # Input method tabs
    tab1, tab2 = st.tabs(["📁  Upload Image", "📷  Take Photo"])

    image_data = None
    source = None

    with tab1:
        uploaded = st.file_uploader(
            "Drop your leaf image here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded:
            image_data = uploaded.read()
            source = "upload"
            st.markdown('<div class="upload-hint">✅ Image loaded — click Detect below</div>', unsafe_allow_html=True)

    with tab2:
        camera_photo = st.camera_input("Point camera at the leaf")
        if camera_photo:
            image_data = camera_photo.read()
            source = "camera"
            st.markdown('<div class="upload-hint">✅ Photo captured — click Detect below</div>', unsafe_allow_html=True)

    st.divider()

    # Show preview + results
    if image_data:
        col_img, col_result = st.columns([1, 1], gap="large")

        with col_img:
            st.markdown("#### 🖼️ Your Image")
            img = Image.open(io.BytesIO(image_data))
            st.image(img, use_container_width=True)

        with col_result:
            st.markdown("#### 🔍 Analysis")
            if st.button("🌿 Detect Disease", use_container_width=True):
                with st.spinner("Analyzing leaf..."):
                    try:
                        response = requests.post(
                            FLASK_URL,
                            files={"file": ("image.jpg", image_data, "image/jpeg")},
                            timeout=30
                        )

                        if response.status_code == 200:
                            result      = response.json()
                            pred_class  = result["predicted_class"]
                            confidence  = result["confidence"]
                            advice      = result["recommendation"]
                            top3        = result["top3"]

                            # Badge
                            is_healthy = "healthy" in pred_class.lower()
                            badge_class = "healthy-badge" if is_healthy else "disease-badge"
                            icon = "✅" if is_healthy else "⚠️"
                            display_name = pred_class.replace("_", " ").replace("  ", " ")

                            st.markdown(f"""
                            <div class="result-card">
                                <div style="font-size:0.85rem; color:#86efac; opacity:0.7;">DETECTED</div>
                                <div class="{badge_class}">{icon} {display_name}</div>
                                <div style="margin-top:1rem; color:#d1fae5; font-size:0.9rem;">Confidence</div>
                            </div>
                            """, unsafe_allow_html=True)

                            st.progress(confidence / 100)
                            st.markdown(f"<div style='text-align:right; color:#4ade80; font-weight:600;'>{confidence:.1f}%</div>", unsafe_allow_html=True)

                            # Advice
                            st.markdown(f"""
                            <div class="advice-box">
                                💊 <strong>Treatment:</strong> {advice}
                            </div>
                            """, unsafe_allow_html=True)

                            # Top 3
                            st.markdown("**📊 Top 3 Predictions**")
                            for i, item in enumerate(top3):
                                name = item["class"].replace("_", " ")
                                conf = item["confidence"]
                                st.markdown(f"<div style='color:#86efac; font-size:0.85rem;'>{i+1}. {name}</div>", unsafe_allow_html=True)
                                st.progress(conf / 100)
                                st.markdown(f"<div style='text-align:right; color:#4ade80; font-size:0.8rem; margin-top:-0.5rem;'>{conf:.1f}%</div>", unsafe_allow_html=True)

                        else:
                            st.error(f"API Error: {response.json().get('error')}")

                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to Flask API. Make sure Flask is running on port 5000!")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.markdown("""
                <div style='text-align:center; padding: 3rem 1rem; color:#86efac; opacity:0.5;'>
                    <div style='font-size:3rem;'>🔬</div>
                    <div>Click Detect Disease to analyze</div>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL ARCHITECTURE
# ════════════════════════════════════════════════════════════════
elif "🏗️" in page:

    st.markdown('<div class="main-title">🏗️ Model Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">How the CNN is structured under the hood</div>', unsafe_allow_html=True)

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-box"><div class="metric-value">4</div><div class="metric-label">Conv Blocks</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-box"><div class="metric-value">458K</div><div class="metric-label">Parameters</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-box"><div class="metric-value">128²</div><div class="metric-label">Input Size</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-box"><div class="metric-value">15</div><div class="metric-label">Output Classes</div></div>', unsafe_allow_html=True)

    st.divider()

    col_arch, col_info = st.columns([1, 1], gap="large")

    with col_arch:
        st.markdown("#### 🔢 Layer Stack")

        layers = [
            ("📥 Input", "Shape: (128, 128, 3) — RGB image"),
            ("⚖️ Rescaling", "Normalizes pixels 0–255 → 0–1 (built into model)"),
            ("🔵 Conv2D (32 filters)", "3×3 kernel, ReLU activation, Same padding"),
            ("⬇️ MaxPooling2D", "2×2 pool — halves spatial size to 64×64"),
            ("🔵 Conv2D (64 filters)", "3×3 kernel, ReLU activation, Same padding"),
            ("⬇️ MaxPooling2D", "2×2 pool — halves spatial size to 32×32"),
            ("🔵 Conv2D (128 filters)", "3×3 kernel, ReLU activation, Same padding"),
            ("⬇️ MaxPooling2D", "2×2 pool — halves spatial size to 16×16"),
            ("🔵 Conv2D (256 filters)", "3×3 kernel, ReLU activation, Same padding"),
            ("⬇️ MaxPooling2D", "2×2 pool — halves spatial size to 8×8"),
            ("🌐 GlobalAveragePooling2D", "Collapses 8×8×256 → 256 vector"),
            ("🟡 Dense (256)", "Fully connected, ReLU activation"),
            ("💧 Dropout (0.3)", "Randomly drops 30% neurons — prevents overfitting"),
            ("📤 Dense (15) + Softmax", "Output: probability for each of 15 classes"),
        ]

        for name, desc in layers:
            st.markdown(f"""
            <div class="arch-layer">
                <h4>{name}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    with col_info:
        st.markdown("#### 🎓 Training Details")
        details = {
            "Optimizer": "RMSprop (lr = 0.0005)",
            "Loss Function": "Sparse Categorical Crossentropy",
            "Epochs": "50 (EarlyStopping, patience=5)",
            "Batch Size": "32",
            "Train / Val / Test Split": "70% / 15% / 15%",
            "Dataset": "PlantVillage (20,063 images)",
            "Image Size": "128 × 128 pixels",
            "Normalization": "Built-in Rescaling layer (÷255)",
            "Regularization": "Dropout 0.3",
        }
        for k, v in details.items():
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between;
                        padding: 0.6rem 0; border-bottom: 1px solid rgba(74,222,128,0.1);'>
                <span style='color:#86efac; font-size:0.9rem;'>{k}</span>
                <span style='color:#4ade80; font-weight:600; font-size:0.9rem;'>{v}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📈 Final Results")
        metrics = [
            ("Train Accuracy", "96.3%", "#4ade80"),
            ("Validation Accuracy", "97.3%", "#4ade80"),
            ("Test Accuracy", "96.2%", "#4ade80"),
            ("Test Loss", "0.131", "#fbbf24"),
        ]
        for label, val, color in metrics:
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; align-items:center;
                        padding: 0.6rem 0; border-bottom: 1px solid rgba(74,222,128,0.1);'>
                <span style='color:#86efac; font-size:0.9rem;'>{label}</span>
                <span style='color:{color}; font-weight:700; font-size:1.1rem;'>{val}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🌿 Supported Classes")
        classes = [
            "Pepper — Bacterial Spot", "Pepper — Healthy",
            "Potato — Early Blight", "Potato — Late Blight", "Potato — Healthy",
            "Tomato — Bacterial Spot", "Tomato — Early Blight", "Tomato — Late Blight",
            "Tomato — Leaf Mold", "Tomato — Septoria Leaf Spot",
            "Tomato — Spider Mites", "Tomato — Target Spot",
            "Tomato — Yellow Leaf Curl Virus", "Tomato — Mosaic Virus",
            "Tomato — Healthy"
        ]
        for c in classes:
            icon = "✅" if "Healthy" in c else "🔴"
            st.markdown(f"<div style='color:#d1fae5; font-size:0.85rem; padding:0.2rem 0;'>{icon} {c}</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT & STATS
# ════════════════════════════════════════════════════════════════
elif "📊" in page:

    st.markdown('<div class="main-title">📊 About & Stats</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">What this app does and how it helps farmers</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🌱 About PlantGuard AI")
        st.markdown("""
        <div class="result-card">
            <p>PlantGuard AI is a deep learning-powered plant disease detection system
            trained on the <strong style='color:#4ade80;'>PlantVillage dataset</strong>
            containing over 20,000 leaf images.</p>
            <p>It uses a <strong style='color:#4ade80;'>Convolutional Neural Network (CNN)</strong>
            to analyze leaf images and identify diseases with ~96% accuracy — helping
            farmers take faster action and reduce crop loss.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🚀 How to Use")
        steps = [
            ("1", "Go to 🔍 Detect Disease page"),
            ("2", "Upload a leaf photo OR take one with your camera"),
            ("3", "Click 'Detect Disease'"),
            ("4", "Get instant diagnosis + treatment advice"),
        ]
        for num, step in steps:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:1rem; padding:0.6rem 0;
                        border-bottom:1px solid rgba(74,222,128,0.1);'>
                <div style='background:#4ade80; color:#0f2017; border-radius:50%;
                            width:28px; height:28px; display:flex; align-items:center;
                            justify-content:center; font-weight:700; flex-shrink:0;'>{num}</div>
                <div style='color:#d1fae5;'>{step}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### ⚡ Tech Stack")
        stack = [
            ("🧠", "TensorFlow / Keras", "Deep learning model"),
            ("🌶️", "Flask", "Backend REST API"),
            ("🎨", "Streamlit", "Frontend UI"),
            ("👁️", "OpenCV", "Image preprocessing"),
            ("🔥", "Grad-CAM", "Visual explainability"),
            ("📦", "PlantVillage", "Training dataset"),
        ]
        for icon, name, desc in stack:
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:1rem; padding:0.7rem;
                        background:rgba(74,222,128,0.05); border-radius:10px; margin:0.3rem 0;
                        border:1px solid rgba(74,222,128,0.15);'>
                <div style='font-size:1.5rem;'>{icon}</div>
                <div>
                    <div style='color:#4ade80; font-weight:600;'>{name}</div>
                    <div style='color:#86efac; font-size:0.8rem; opacity:0.7;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔗 API Status")
        try:
            r = requests.get("http://127.0.0.1:5000/", timeout=3)
            if r.status_code == 200:
                st.markdown("""
                <div style='background:rgba(74,222,128,0.1); border:1px solid #4ade80;
                            border-radius:10px; padding:0.8rem 1rem; color:#4ade80; font-weight:600;'>
                    ✅ Flask API is Online — http://127.0.0.1:5000
                </div>
                """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div style='background:rgba(239,68,68,0.1); border:1px solid #ef4444;
                        border-radius:10px; padding:0.8rem 1rem; color:#ef4444; font-weight:600;'>
                ❌ Flask API is Offline — Start it with: python app.py
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#86efac; opacity:0.4; font-size:0.75rem;'>
    PlantGuard AI · Built with TensorFlow + Flask + Streamlit
</div>
""", unsafe_allow_html=True)
