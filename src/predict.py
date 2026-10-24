"""Predict one leaf photo with the trained model.

Usage:
    python src/predict.py path/to/leaf.jpg

Requires results/plant_model.keras + results/class_names.json (from train.py).
"""

import json
import os
import sys

import numpy as np
from tensorflow import keras

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python src/predict.py path/to/leaf.jpg")

    model_path = os.path.join(RESULTS_DIR, "plant_model.keras")
    names_path = os.path.join(RESULTS_DIR, "class_names.json")
    if not os.path.exists(model_path) or not os.path.exists(names_path):
        raise SystemExit("No trained model found. Run: python src/train.py")

    with open(names_path) as f:
        class_names = json.load(f)

    # The model expects 128x128 images (see train.py --img-size).
    img = keras.utils.load_img(sys.argv[1], target_size=(128, 128))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # add batch dimension; Rescaling is inside the model

    model = keras.models.load_model(model_path)
    probs = model.predict(x, verbose=0)[0]
    top3 = probs.argsort()[-3:][::-1]
    print(f"Predicted: {class_names[top3[0]]} ({probs[top3[0]] * 100:.1f}%)")
    print("Top 3:")
    for i in top3:
        print(f"  {class_names[i]}: {probs[i] * 100:.1f}%")


if __name__ == "__main__":
    main()
