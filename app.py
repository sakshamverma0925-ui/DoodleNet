"""
DoodleNet: Real-Time Sketch Recognition - Streamlit Web Application

Aesthetic: Artist's Studio / Sketchbook Theme
- Cream/Off-white background: #FDFBF7
- Charcoal typography: #2C2C2C
- Burnt Sienna / Terracotta accent: #D35400
"""

import os
import json
import numpy as np
import streamlit as st
import cv2

# Import canvas
try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("Please install streamlit-drawable-canvas: pip install streamlit-drawable-canvas")
    st.stop()

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="DoodleNet | Real-Time Sketch Recognition",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Artist's Studio / Sketchbook" Aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,600;0,700;1,400&display=swap');

    /* Global App Background & Base Typography */
    .stApp {
        background-color: #FDFBF7;
        color: #2C2C2C;
        font-family: 'Outfit', sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 1080px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F8F4ED !important;
        border-right: 1px solid #EADBCE;
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Center Header */
    .studio-header {
        text-align: center;
        margin-bottom: 1.8rem;
        padding-bottom: 1.2rem;
        border-bottom: 1px solid #EADBCE;
    }
    .studio-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 700;
        color: #2C2C2C;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .studio-title span {
        color: #D35400;
    }
    .studio-subtitle {
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        color: #7D6F64;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .studio-tagline {
        font-size: 0.95rem;
        color: #9C8E82;
        margin-top: 0.4rem;
        font-style: italic;
    }

    /* Sketchbook Card Container */
    .canvas-card {
        background-color: #FFFFFF;
        border: 1px solid #EADDD0;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 10px 30px rgba(60, 45, 35, 0.05);
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .canvas-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #2C2C2C;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }

    /* Live Analysis Card */
    .analysis-card {
        background: #FFFFFF;
        border: 1px solid #EADDD0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 10px 30px rgba(60, 45, 35, 0.05);
        height: 100%;
    }

    .analysis-header {
        font-family: 'Playfair Display', serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: #2C2C2C;
        border-bottom: 2px solid #F4EAE0;
        padding-bottom: 0.6rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Primary Prediction Highlight */
    .winner-box {
        background: linear-gradient(135deg, #FFF9F3 0%, #FDF4EB 100%);
        border: 1.5px solid #F3C9AB;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1.4rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .winner-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #A35322;
        font-weight: 600;
    }
    .winner-name {
        font-family: 'Playfair Display', serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: #D35400;
        text-transform: capitalize;
    }
    .winner-confidence {
        font-size: 1.7rem;
        font-weight: 700;
        color: #2C2C2C;
    }

    /* Prediction Item & Confidence Bar */
    .pred-row {
        margin-bottom: 1.1rem;
    }
    .pred-meta {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.4rem;
    }
    .pred-rank {
        display: inline-block;
        width: 24px;
        height: 24px;
        line-height: 24px;
        text-align: center;
        border-radius: 50%;
        background-color: #F4E8DB;
        color: #7D6F64;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .pred-class {
        font-weight: 600;
        font-size: 1.05rem;
        color: #2C2C2C;
        text-transform: capitalize;
    }
    .pred-pct {
        font-weight: 700;
        font-size: 1rem;
        color: #D35400;
    }
    .bar-bg {
        background-color: #F4ECE2;
        height: 11px;
        border-radius: 8px;
        overflow: hidden;
        position: relative;
    }
    .bar-fill {
        background: linear-gradient(90deg, #E67E22 0%, #D35400 100%);
        height: 100%;
        border-radius: 8px;
        transition: width 0.3s ease-in-out;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3.5rem 1.5rem;
        color: #9C8E82;
    }
    .empty-icon {
        font-size: 3.2rem;
        margin-bottom: 0.8rem;
        opacity: 0.8;
    }

    /* Sidebar Cards */
    .sidebar-card {
        background-color: #FFFFFF;
        border: 1px solid #EADBCE;
        border-radius: 12px;
        padding: 1.1rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(60, 45, 35, 0.03);
    }
    .sidebar-heading {
        font-family: 'Playfair Display', serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #2C2C2C;
        margin-bottom: 0.6rem;
        border-bottom: 1px solid #F0E6DA;
        padding-bottom: 0.3rem;
    }

    /* Class badges */
    .class-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 0.5rem;
    }
    .class-chip {
        background-color: #F6EFE6;
        border: 1px solid #EADDCF;
        color: #5C4F44;
        font-size: 0.8rem;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 500;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #D35400 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        font-weight: 600 !important;
        font-family: 'Outfit', sans-serif !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        background-color: #BA4A00 !important;
        box-shadow: 0 4px 12px rgba(211, 84, 0, 0.25) !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# Target 10 Classes
DEFAULT_CLASSES = [
    "apple", "bus", "cat", "dog", "fish",
    "flower", "house", "key", "pencil", "star"
]

CLASS_ICONS = {
    "apple": "🍎",
    "bus": "🚌",
    "cat": "🐱",
    "dog": "🐶",
    "fish": "🐟",
    "flower": "🌸",
    "house": "🏠",
    "key": "🔑",
    "pencil": "✏️",
    "star": "⭐"
}


@st.cache_resource
def load_trained_model():
    """Loads trained Keras model and classes.json metadata."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.h5")
    classes_path = os.path.join(base_dir, "classes.json")

    classes = DEFAULT_CLASSES
    if os.path.exists(classes_path):
        try:
            with open(classes_path, "r") as f:
                data = json.load(f)
                classes = data.get("classes", DEFAULT_CLASSES)
        except Exception:
            pass

    if not os.path.exists(model_path):
        return None, classes, f"Model file '{model_path}' not found."

    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model, classes, None
    except Exception as e:
        return None, classes, f"Failed to load model: {str(e)}"


def preprocess_canvas_drawing(image_data):
    """
    Transforms the user's canvas drawing to precisely match the Quick, Draw! feature space:
    1. Extracts ink stroke mask from RGBA canvas.
    2. Crops tightly around the bounding box with 15% proportional padding.
    3. Normalizes aspect ratio and scales doodle into a 20x20 box inside a 28x28 grid.
    4. Applies stroke dilation if necessary to match the 1.5-2.5px stroke width of QuickDraw.
    5. Normalizes pixel values [0, 255] -> [0.0, 1.0] with shape (1, 28, 28, 1).
    """
    if image_data is None:
        return None, False, None

    alpha = image_data[:, :, 3]
    rgb = image_data[:, :, :3]

    # Convert to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Detect drawn ink strokes: dark pixels or active alpha
    is_stroke = (alpha > 40) & (gray < 225)
    stroke_count = np.count_nonzero(is_stroke)

    if stroke_count < 25:  # Empty canvas
        return None, False, None

    # Find stroke bounding box
    rows = np.any(is_stroke, axis=1)
    cols = np.any(is_stroke, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop stroke area
    cropped = is_stroke[rmin : rmax + 1, cmin : cmax + 1].astype(np.uint8) * 255
    h, w = cropped.shape

    # Apply gentle dilation if stroke is thin to match QuickDraw contour thickness
    kernel_size = 2 if max(h, w) > 80 else 1
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        cropped = cv2.dilate(cropped, kernel, iterations=1)

    # Scale maintaining aspect ratio so the larger dimension fits inside a 20x20 box
    if h > w:
        new_h = 20
        new_w = max(1, int(round((w * 20.0) / h)))
    else:
        new_w = 20
        new_h = max(1, int(round((h * 20.0) / w)))

    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center within a 28x28 black canvas
    processed_28x28 = np.zeros((28, 28), dtype=np.uint8)
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    processed_28x28[top : top + new_h, left : left + new_w] = resized

    # Normalize contrast so peak stroke pixels reach full intensity
    if processed_28x28.max() > 0:
        processed_28x28 = ((processed_28x28.astype(np.float32) / processed_28x28.max()) * 255.0).astype(np.uint8)

    # Format tensor for CNN: (1, 28, 28, 1) in [0.0, 1.0]
    model_input = (processed_28x28 / 255.0).astype(np.float32).reshape(1, 28, 28, 1)
    return model_input, True, processed_28x28


def main():
    model, classes, error_msg = load_trained_model()

    # ------------------ SIDEBAR ------------------
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-card" style="border-left: 4px solid #D35400;">
            <div style="font-family:'Playfair Display', serif; font-size:1.6rem; font-weight:700; color:#2C2C2C;">DoodleNet</div>
            <div style="font-size:0.88rem; color:#7D6F64; margin-top:2px;">Real-Time Sketch Recognition</div>
        </div>
        """, unsafe_allow_html=True)

        # Drawing Tool Controls
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-heading">Drawing Controls</div>', unsafe_allow_html=True)
        brush_size = st.slider("Stroke Width", min_value=6, max_value=24, value=12, step=1)
        stroke_color = st.color_picker("Stroke Color", value="#2C2C2C")
        st.markdown('</div>', unsafe_allow_html=True)

        # Target Categories Guide
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-heading">Supported Classes (10)</div>', unsafe_allow_html=True)
        chips_html = "".join([f'<span class="class-chip">{CLASS_ICONS.get(c, "✏️")} {c}</span>' for c in classes])
        st.markdown(f'<div class="class-grid">{chips_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ------------------ MAIN STUDIO AREA ------------------
    # Header Front & Center
    st.markdown("""
    <div class="studio-header">
        <div class="studio-title">Doodle<span>Net</span></div>
        <div class="studio-subtitle">Real-Time Sketch Recognition</div>
        <div class="studio-tagline">Draw an object on the easel and watch the neural network classify your sketch in real time.</div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.warning(f"⚠️ **Model Not Found**: {error_msg}")
        st.info("💡 Run `python train.py` in your terminal to train on real Quick, Draw! sketches.")

    col_canvas, col_analysis = st.columns([1.1, 1.0], gap="large")

    with col_canvas:
        st.markdown("""
        <div class="canvas-card">
            <div class="canvas-title">
                <span>🖋️</span> Drawing Easel
            </div>
            <div style="font-size:0.85rem; color:#8C7E74; margin-bottom:10px;">
                Sketch inside the box. Clear anytime using the trash icon.
            </div>
        </div>
        """, unsafe_allow_html=True)

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=brush_size,
            stroke_color=stroke_color,
            background_color="#FFFFFF",
            update_streamlit=True,
            height=320,
            width=320,
            drawing_mode="freedraw",
            return_image_data=True,
            key="doodlenet_canvas"
        )

        st.caption("✨ Real-time inference activates automatically as you sketch.")

    with col_analysis:
        st.markdown("""
        <div class="analysis-card">
            <div class="analysis-header">
                <span>Live Model Analysis</span>
                <span style="font-size:0.8rem; font-family:'Outfit'; color:#A08F83; font-weight:600;">TOP-3 PREDICTIONS</span>
            </div>
        """, unsafe_allow_html=True)

        has_strokes = False
        preprocessed = None
        vis_28x28 = None

        if canvas_result is not None:
            try:
                img_data = canvas_result.image_data
                if img_data is not None:
                    preproc_result = preprocess_canvas_drawing(img_data)
                    if preproc_result[1]:
                        has_strokes = True
                        preprocessed = preproc_result[0]
                        vis_28x28 = preproc_result[2]
            except Exception:
                has_strokes = False

        if not has_strokes:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🎨</div>
                <div style="font-weight:600; font-size:1.1rem; color:#4A4A4A; margin-bottom:4px;">Awaiting Your First Stroke</div>
                <div style="font-size:0.88rem; line-height:1.4;">Pick a subject like a <strong>house</strong>, <strong>apple</strong>, <strong>star</strong>, or <strong>cat</strong> and draw on the easel.</div>
            </div>
            """, unsafe_allow_html=True)
        elif model is None:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">⚙️</div>
                <div style="font-weight:600; font-size:1.1rem; color:#4A4A4A; margin-bottom:4px;">Model Weights Missing</div>
                <div style="font-size:0.85rem;">Please train the model first with <code>python train.py</code>.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Model prediction
            predictions = model.predict(preprocessed, verbose=0)[0]
            top3_indices = np.argsort(predictions)[::-1][:3]

            top_idx = top3_indices[0]
            top_class = classes[top_idx]
            top_conf = predictions[top_idx] * 100
            top_icon = CLASS_ICONS.get(top_class, "🎯")

            # Primary match card
            st.markdown(f"""
            <div class="winner-box">
                <div>
                    <div class="winner-label">Primary Match</div>
                    <div class="winner-name">{top_icon} {top_class}</div>
                </div>
                <div style="text-align:right;">
                    <div class="winner-label">Confidence</div>
                    <div class="winner-confidence">{top_conf:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Top-3 Bars
            for rank, idx in enumerate(top3_indices, start=1):
                c_name = classes[idx]
                c_conf = predictions[idx] * 100
                c_icon = CLASS_ICONS.get(c_name, "✏️")

                st.markdown(f"""
                <div class="pred-row">
                    <div class="pred-meta">
                        <div>
                            <span class="pred-rank">#{rank}</span>
                            <span class="pred-class">{c_icon} {c_name}</span>
                        </div>
                        <span class="pred-pct">{c_conf:.1f}%</span>
                    </div>
                    <div class="bar-bg">
                        <div class="bar-fill" style="width: {max(3.0, c_conf):.1f}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Explainability / Preprocessed visualizer
            with st.expander("🔍 Model Input (28×28 Grayscale)", expanded=False):
                st.markdown(
                    "<div style='font-size:0.82rem; color:#7D6F64; margin-bottom:8px;'>"
                    "The centered, normalized 28×28 representation evaluated by the convolutional feature maps:"
                    "</div>",
                    unsafe_allow_html=True
                )
                if vis_28x28 is not None:
                    col_p1, col_p2 = st.columns([1, 2])
                    with col_p1:
                        st.image(vis_28x28, width=110, caption="28x28 CNN Input")
                    with col_p2:
                        st.markdown(f"""
                        - **Resolution**: 28 × 28 × 1
                        - **Bounding Box**: Extracted & Centered
                        - **Contrast**: Normalized to [0.0, 1.0]
                        - **Active Pixels**: {np.count_nonzero(vis_28x28)} / 784
                        """)

        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
