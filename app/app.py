import streamlit as st
import tensorflow as tf
import numpy as np
import json

from PIL import Image
from pathlib import Path


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "plant_disease_model.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "class_names.json"


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================
# 3. CUSTOM CSS
# ==========================================

st.html("""
<style>
    :root {
        --background: #101713;
        --surface: #18221c;
        --surface-soft: #1d2a22;
        --border: #304438;
        --text: #edf4ee;
        --muted: #aabbb0;
        --green: #91d4a5;
        --green-strong: #477c55;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    .block-container {
        max-width: 1240px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .hero {
        padding: 38px 40px;
        background: linear-gradient(120deg, #1b2b21 0%, #17221c 65%, #1d3024 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        margin-bottom: 28px;
    }

    .hero-label {
        color: var(--green);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2.2px;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #f3f8f4;
        font-size: clamp(30px, 4vw, 42px);
        line-height: 1.15;
        font-weight: 750;
        margin: 13px 0 12px 0;
    }

    .hero p {
        color: var(--muted);
        font-size: 15px;
        line-height: 1.7;
        max-width: 720px;
        margin-bottom: 0;
    }

    .section-heading {
        color: var(--text);
        font-size: 19px;
        font-weight: 650;
        margin: 5px 0 15px 0;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 13px;
        margin-top: -9px;
        margin-bottom: 16px;
    }

    .surface-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 23px;
        min-height: 190px;
    }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 19px 16px;
        text-align: center;
    }

    .metric-value {
        color: var(--green);
        font-size: 27px;
        font-weight: 750;
        letter-spacing: -0.5px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 12px;
        margin-top: 5px;
    }

    .result-label {
        color: #8fbd9b;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.6px;
        text-transform: uppercase;
    }

    .result-name {
        color: #f0f6f1;
        font-size: 22px;
        line-height: 1.4;
        font-weight: 700;
        margin: 10px 0 20px 0;
        overflow-wrap: anywhere;
    }

    .confidence-value {
        color: var(--green);
        font-size: 34px;
        font-weight: 750;
        margin-top: 5px;
    }

    .empty-title {
        color: var(--green);
        font-size: 19px;
        font-weight: 650;
        margin-bottom: 9px;
    }

    .empty-copy {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.65;
    }

    .disclaimer {
        background: #25251b;
        border: 1px solid #5b5430;
        border-radius: 13px;
        padding: 19px 21px;
        color: #d8d1a6;
        font-size: 13px;
        line-height: 1.75;
    }

    .footer {
        text-align: center;
        color: #75867a;
        font-size: 12px;
        margin-top: 38px;
        padding-top: 20px;
        border-top: 1px solid #29372e;
        letter-spacing: 0.3px;
    }

    div[data-testid="stFileUploader"] {
        background: var(--surface);
        border: 1px dashed #54735d;
        border-radius: 12px;
        padding: 13px;
    }

    div.stButton > button {
        background: var(--green-strong);
        color: white;
        border: 1px solid #5d946a;
        border-radius: 9px;
        min-height: 46px;
        font-weight: 650;
        width: 100%;
        transition: background 0.2s ease, border 0.2s ease;
    }

    div.stButton > button:hover {
        background: #589466;
        color: white;
        border-color: #75aa80;
    }

    div[data-testid="stImage"] img {
        border-radius: 11px;
    }

    div[data-testid="stProgressBar"] > div {
        background: #26382c;
    }

    div[data-testid="stProgressBar"] > div > div {
        background: #78b889;
    }

    .stAlert {
        border-radius: 10px;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 27px 23px;
        }
        .hero h1 {
            font-size: 30px;
        }
        .surface-card {
            padding: 18px;
        }
    }
</style>
""")


# ==========================================
# 4. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return model, class_names


try:
    model, class_names = load_model()
except Exception as error:
    st.error(f"Unable to load the model or class names: {error}")
    st.info(
        "Check that plant_disease_model.keras and class_names.json "
        "are available inside the project's models folder."
    )
    st.stop()


# ==========================================
# 5. HEADER
# ==========================================

st.html("""
<div class="hero">
    <div class="hero-label">Artificial Intelligence / Agriculture</div>
    <h1>Plant Disease Detection</h1>
    <p>
        An AI-powered system that analyzes plant leaf images and predicts
        potential diseases using a convolutional neural network.
        Upload a clear leaf image to get a model-generated prediction.
    </p>
</div>
""")


# ==========================================
# 6. MODEL INFORMATION
# ==========================================

metric_col1, metric_col2, metric_col3 = st.columns(3, gap="medium")

with metric_col1:
    st.html("""
    <div class="metric-card">
        <div class="metric-value">CNN</div>
        <div class="metric-label">Deep Learning Architecture</div>
    </div>
    """)

with metric_col2:
    st.html("""
    <div class="metric-card">
        <div class="metric-value">38</div>
        <div class="metric-label">Supported Classes</div>
    </div>
    """)

with metric_col3:
    st.html("""
    <div class="metric-card">
        <div class="metric-value">95.98%</div>
        <div class="metric-label">Test Accuracy</div>
    </div>
    """)

st.write("")
st.divider()


# ==========================================
# 7. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ==========================================
# 8. DISEASE PREDICTION
# ==========================================

def predict_disease(image):
    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    predicted_index = int(np.argmax(predictions[0]))
    predicted_disease = class_names[predicted_index]
    confidence = float(predictions[0][predicted_index]) * 100

    return predicted_disease, confidence


def format_class_name(class_name):
    return class_name.replace("___", " - ").replace("_", " ")


# ==========================================
# 9. MAIN APPLICATION
# ==========================================

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown('<div class="section-heading">01 / Upload Image</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Select a clear photograph of a plant leaf.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a plant leaf image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG.",
        key="leaf_image"
    )

    image = None

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image,
                caption="Uploaded leaf image",
                use_container_width=True
            )
        except Exception:
            st.error("Unable to read this image. Please upload a valid JPG or PNG.")
            st.stop()
    else:
        st.html("""
        <div class="surface-card">
            <div class="empty-title">Image Analysis</div>
            <div class="empty-copy">
                Your selected leaf image will appear here.
                For more useful predictions, upload a clear image where
                the leaf is visible and reasonably well lit.
            </div>
        </div>
        """)

with right_col:
    st.markdown('<div class="section-heading">02 / Analysis Result</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Run the model to view its predicted class.</div>',
        unsafe_allow_html=True
    )

    if image is None:
        st.html("""
        <div class="surface-card">
            <div class="empty-title">Awaiting Image</div>
            <div class="empty-copy">
                The prediction and confidence score will appear in this
                section after you upload an image and select Analyze.
            </div>
        </div>
        """)
    else:
        analyze_clicked = st.button(
            "Analyze Leaf Image",
            type="primary",
            use_container_width=True
        )

        if analyze_clicked:
            try:
                with st.spinner("Analyzing leaf image..."):
                    disease, confidence = predict_disease(image)

                st.session_state["prediction_result"] = {
                    "filename": uploaded_file.name,
                    "disease": disease,
                    "confidence": confidence
                }
            except Exception as error:
                st.error(f"Prediction failed: {error}")
                st.stop()

        result = st.session_state.get("prediction_result")

        # Do not show an old result when the user selects a different image.
        if result and result["filename"] != uploaded_file.name:
            st.session_state.pop("prediction_result", None)
            result = None

        if result:
            readable_name = format_class_name(result["disease"])
            confidence = result["confidence"]

            st.html(f"""
            <div class="surface-card">
                <div class="result-label">Predicted Class</div>
                <div class="result-name">{readable_name}</div>
                <div class="result-label">Model Confidence</div>
                <div class="confidence-value">{confidence:.2f}%</div>
            </div>
            """)

            st.progress(
                min(max(confidence / 100, 0.0), 1.0),
                text="Prediction confidence"
            )

            if confidence < 70:
                st.warning(
                    "The model has relatively low confidence. "
                    "Treat this prediction with caution."
                )
            else:
                st.info(
                    "The model strongly favors this class among its "
                    "supported categories. This does not guarantee that "
                    "the diagnosis is correct."
                )
        else:
            st.html("""
            <div class="surface-card">
                <div class="empty-title">Ready to Analyze</div>
                <div class="empty-copy">
                    Your image is ready. Select the button above to run
                    the trained CNN model.
                </div>
            </div>
            """)


# ==========================================
# 10. DISCLAIMER
# ==========================================

st.write("")
st.divider()

st.html("""
<div class="disclaimer">
    <strong>Important Disclaimer</strong>
    <br><br>
    This model was trained using 38 classes from the PlantVillage dataset.
    It can only predict one of these supported classes.
    <br><br>
    Diseases outside the training categories may receive incorrect
    predictions, even when the displayed confidence is high.
    <br><br>
    This application provides an AI-generated prediction and should not
    replace professional agricultural diagnosis.
</div>
""")


# ==========================================
# 11. FOOTER
# ==========================================

st.html("""
<div class="footer">
    PLANT DISEASE DETECTION
    <br><br>
    Built with TensorFlow, CNN and Streamlit
</div>
""")
