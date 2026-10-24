"""Train the CNN on PlantVillage (folder-based dataset).

Usage:
    python src/train.py --data-dir data/plantvillage --epochs 10

Each subfolder of --data-dir is one class (folder name = class name).
The data is split 80% train / 20% validation automatically.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

from model import build_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/plantvillage")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=128)
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise SystemExit(
            f"Data folder not found: {args.data_dir}\n"
            "Download PlantVillage from Kaggle and unzip it into data/plantvillage/ "
            "(see README.md)."
        )

    # --- Load data: folder names become class names automatically ---
    train_ds = keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )
    class_names = train_ds.class_names
    print(f"Found {len(class_names)} classes: {class_names}")

    model = build_model(
        input_shape=(args.img_size, args.img_size, 3),
        num_classes=len(class_names),
    )
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
    )

    # --- Report + save ---
    loss, acc = model.evaluate(val_ds, verbose=0)
    print(f"\nValidation accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    model.save(os.path.join(RESULTS_DIR, "plant_model.keras"))
    with open(os.path.join(RESULTS_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.title("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "history.png"))
    print(f"Saved model + class names + curves to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
