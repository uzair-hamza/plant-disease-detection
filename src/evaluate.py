"""Evaluate the trained model: accuracy, report, confusion matrix.

Usage:
    python src/evaluate.py --data-dir data/plantvillage

Requires results/plant_model.keras + results/class_names.json (from train.py).
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/plantvillage")
    parser.add_argument("--img-size", type=int, default=128)
    args = parser.parse_args()

    model_path = os.path.join(RESULTS_DIR, "plant_model.keras")
    names_path = os.path.join(RESULTS_DIR, "class_names.json")
    if not os.path.exists(model_path) or not os.path.exists(names_path):
        raise SystemExit("No trained model found. Run: python src/train.py")

    with open(names_path) as f:
        class_names = json.load(f)

    val_ds = keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=32,
        shuffle=False,
    )

    model = keras.models.load_model(model_path)
    preds = model.predict(val_ds, verbose=0).argmax(axis=1)
    y_true = np.concatenate([y.numpy() for _, y in val_ds])

    acc = (preds == y_true).mean()
    print(f"Validation accuracy: {acc:.4f} ({acc * 100:.2f}%)\n")
    print(classification_report(y_true, preds, target_names=class_names))

    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(12, 12))
    plt.imshow(cm, cmap="Greens")
    plt.title("Confusion matrix (rows = true, columns = predicted)")
    plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
    plt.yticks(range(len(class_names)), class_names, fontsize=7)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    print(f"Saved confusion matrix to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
