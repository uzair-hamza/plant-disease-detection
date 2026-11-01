"""Plant Disease Detector - live web demo.

Upload a leaf photo and get an instant disease prediction.
Model: custom CNN trained on PlantVillage (54,303 images, 38 classes),
94.57% validation accuracy. Built by Muhammad Uzair Hamza.
"""

import json
import os
import urllib.request

import numpy as np
import streamlit as st
from tensorflow import keras

MODEL_URL = "https://github.com/uzair-hamza/plant-disease-detection/releases/download/v1.0/plant_model.keras"
CACHE_DIR = "model_cache"

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
)

# ---------------------------------------------------------------- styling
st.markdown(
    """
    <style>
    .banner {
        background: linear-gradient(120deg, #1b5e20 0%, #2e7d32 60%, #43a047 100%);
        border-radius: 14px;
        padding: 34px 30px;
        color: white;
        margin-bottom: 22px;
    }
    .banner h1 { color: white !important; margin: 0 0 6px 0; font-size: 2.1rem; }
    .banner p { color: #e8f5e9 !important; margin: 0; font-size: 1.05rem; }
    .stat-box {
        background: #f1f8e9;
        border: 1px solid #c5e1a5;
        border-radius: 12px;
        padding: 16px 10px;
        text-align: center;
    }
    .stat-box b { font-size: 1.5rem; color: #1b5e20; display: block; }
    .stat-box span { color: #555; font-size: 0.9rem; }
    .result-card {
        background: #e8f5e9;
        border-left: 6px solid #2e7d32;
        border-radius: 10px;
        padding: 18px 20px;
        margin: 14px 0;
    }
    .result-card b { color: #1b5e20; font-size: 1.25rem; }
    .section-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .footer {
        text-align: center; color: #888; font-size: 0.85rem;
        margin-top: 30px; padding-top: 14px; border-top: 1px solid #e0e0e0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_classes():
    """Download the trained model once, then reuse it."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    model_path = os.path.join(CACHE_DIR, "plant_model.keras")
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(MODEL_URL, model_path)
    model = keras.models.load_model(model_path)
    with open(os.path.join("results", "class_names.json")) as f:
        class_names = json.load(f)
    return model, class_names


def pretty_name(raw: str) -> str:
    """Tomato___Late_blight -> Tomato - Late blight."""
    return raw.replace("___", " - ").replace("_", " ")


def detector_tab(model, class_names):
    left, right = st.columns([1, 1])

    with left:
        st.subheader("1. Upload a leaf photo")
        uploaded = st.file_uploader("Choose a leaf photo",
                                    type=["jpg", "jpeg", "png"],
                                    label_visibility="collapsed")
        if uploaded is not None:
            st.image(uploaded, caption="Your leaf photo", use_column_width=True)

    with right:
        st.subheader("2. Get the diagnosis")
        if uploaded is None:
            st.info("Upload a photo on the left and the model will diagnose it here.")
            return
        with st.spinner("Analyzing the leaf..."):
            img = keras.utils.load_img(uploaded, target_size=(128, 128))
            x = keras.utils.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            probs = model.predict(x, verbose=0)[0]
        top3 = probs.argsort()[-3:][::-1]
        st.markdown(
            f'<div class="result-card">Predicted: <b>{pretty_name(class_names[top3[0]])}</b>'
            f"<br>Confidence: {probs[top3[0]] * 100:.1f}%</div>",
            unsafe_allow_html=True,
        )
        st.write("Top 3 predictions:")
        for i in top3:
            st.write(f"**{pretty_name(class_names[i])}** - {probs[i] * 100:.1f}%")
            st.progress(float(probs[i]))


def project_tab():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("What this project is")
    st.write(
        "A convolutional neural network that looks at a photo of a plant leaf and "
        "tells which disease it has - or that the leaf is healthy. I trained it on "
        "the public PlantVillage dataset and built this page so anyone can try it."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, "54,303", "Leaf photos"),
        (c2, "38", "Classes (30 diseases + 8 healthy)"),
        (c3, "14", "Crops"),
        (c4, "94.57%", "Validation accuracy"),
    ]:
        col.markdown(
            f'<div class="stat-box"><b>{val}</b><span>{label}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("The model")
    st.write(
        "A custom CNN I designed and trained from scratch: three convolutional blocks "
        "(32 / 64 / 128 filters), data augmentation during training (flip, rotation, zoom), "
        "and a dense classification head with dropout. Input size 128x128, trained for "
        "10 epochs with Adam. Weighted F1 score: 0.95."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Tested honestly")
    st.write(
        "On lab photos (PlantVillage) the model reaches 94.57% accuracy. I also tested it "
        "on real field photos (PlantDoc dataset) - there it only reaches 8.90%. That gap "
        "is the honest finding of my research: lab-trained models struggle in the field, "
        "and that is exactly what my ICECT 2027 paper studies, with Grad-CAM visualizations "
        "showing where the model looks when it decides."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.link_button("View the code on GitHub",
                   "https://github.com/uzair-hamza/plant-disease-detection")


def about_tab():
    c1, c2 = st.columns([1, 3])
    with c1:
        st.image("https://github.com/uzair-hamza.png", width=180)
    with c2:
        st.subheader("Muhammad Uzair Hamza")
        st.write(
            "BSc Computer Systems Engineering student (5th semester) at UET Peshawar. "
            "Undergraduate Research Assistant and NAVTTC Courses Coordinator at the "
            "National Center for Big Data and Cloud Computing."
        )
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("What I do")
    st.write(
        "Machine learning, deep learning, and computer vision - plus graphic design "
        "and UI/UX. This plant disease detector is one of five ML projects on my GitHub, "
        "and the subject of my ICECT 2027 research paper."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    b1.link_button("GitHub", "https://github.com/uzair-hamza")
    b2.link_button("ML Portfolio", "https://ml-portfolio-uzair-hamza.vercel.app")
    b3.link_button("Design Portfolio",
                   "https://graphics-portfolio-uzair-hamza.vercel.app")


def main() -> None:
    st.markdown(
        '<div class="banner"><h1>Plant Disease Detector</h1>'
        "<p>Upload a leaf photo - my AI model tells you its disease in seconds.</p></div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading the trained model (first visit takes a moment)..."):
        model, class_names = load_model_and_classes()

    tab1, tab2, tab3 = st.tabs(["Detector", "About the Project", "About Me"])
    with tab1:
        detector_tab(model, class_names)
    with tab2:
        project_tab()
    with tab3:
        about_tab()

    st.markdown(
        '<div class="footer">Built by Muhammad Uzair Hamza · UET Peshawar · '
        "Model: custom CNN on PlantVillage</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
