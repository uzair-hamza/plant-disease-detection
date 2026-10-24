"""Simple web app for Plant Disease Detection.

Run after training (needs results/plant_model.keras + results/class_names.json):

    pip install streamlit
    streamlit run app.py

Then open the link it shows, upload a leaf photo, and it tells you the
disease (or healthy) with confidence.
"""

import json
import os

import numpy as np
import streamlit as st
from tensorflow import keras

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


@st.cache_resource
def load_model_and_classes():
    """Load the trained model + class names once, then reuse them."""
    model = keras.models.load_model(os.path.join(RESULTS_DIR, "plant_model.keras"))
    with open(os.path.join(RESULTS_DIR, "class_names.json")) as f:
        class_names = json.load(f)
    return model, class_names


def main() -> None:
    st.title("🌿 Plant Disease Detection")
    st.write("Upload a photo of a leaf, and the model will tell you its disease.")

    model_path = os.path.join(RESULTS_DIR, "plant_model.keras")
    if not os.path.exists(model_path):
        st.warning("No trained model found yet. Run `python src/train.py` first.")
        return

    model, class_names = load_model_and_classes()

    uploaded = st.file_uploader("Choose a leaf photo", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        return

    st.image(uploaded, caption="Your leaf photo", width=300)

    # The model expects 128x128 images (see train.py --img-size).
    img = keras.utils.load_img(uploaded, target_size=(128, 128))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # add batch dimension

    probs = model.predict(x, verbose=0)[0]
    top3 = probs.argsort()[-3:][::-1]

    st.success(f"Predicted: **{class_names[top3[0]]}** "
               f"({probs[top3[0]] * 100:.1f}% confidence)")
    st.write("Top 3:")
    for i in top3:
        st.write(f"- {class_names[i]}: {probs[i] * 100:.1f}%")


if __name__ == "__main__":
    main()
