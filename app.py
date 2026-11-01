"""Plant Disease Detection - live web demo.

Upload a leaf photo and get an instant disease prediction.
Model: custom CNN trained on PlantVillage (54,303 images, 38 classes),
94.57% validation accuracy.
"""

import json
import os
import urllib.request

import numpy as np
import streamlit as st
from tensorflow import keras

MODEL_URL = "https://github.com/uzair-hamza/plant-disease-detection/releases/download/v1.0/plant_model.keras"
CACHE_DIR = "model_cache"


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


def main() -> None:
    st.title("Plant Disease Detection")
    st.write("Upload a photo of a leaf, and the model will tell you its disease.")

    with st.spinner("Loading the trained model (first visit takes a moment)..."):
        model, class_names = load_model_and_classes()

    uploaded = st.file_uploader("Choose a leaf photo", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    st.image(uploaded, caption="Your leaf photo", width=300)

    img = keras.utils.load_img(uploaded, target_size=(128, 128))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    top3 = probs.argsort()[-3:][::-1]

    st.success(f"Predicted: **{class_names[top3[0]]}** "
               f"({probs[top3[0]] * 100:.1f}% confidence)")
    st.write("Top 3:")
    for i in top3:
        st.write(f"- {class_names[i]}: {probs[i] * 100:.1f}%")


if __name__ == "__main__":
    main()
