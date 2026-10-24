"""Download PlantVillage via TensorFlow Datasets and export it to folders.

This avoids needing a Kaggle login: TFDS downloads PlantVillage from its
public source. The exported folder layout matches what train.py expects:

    data/plantvillage/<class_name>/image_000001.jpg ...

Usage:
    python scripts/export_plantvillage.py

Requires: tensorflow-datasets (pip install tensorflow-datasets).
"""

import os

import tensorflow_datasets as tfds
from PIL import Image

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "plantvillage")


def main() -> None:
    print("Downloading + preparing PlantVillage (this can take a while)...")
    builder = tfds.builder("plant_village")
    builder.download_and_prepare()
    info = builder.info
    class_names = info.features["label"].names
    print(f"Classes ({len(class_names)}): {class_names}")

    ds = builder.as_dataset(split="train", as_supervised=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    counts = {}
    for i, (image, label) in enumerate(ds):
        name = class_names[int(label)]
        class_dir = os.path.join(OUT_DIR, name)
        os.makedirs(class_dir, exist_ok=True)
        path = os.path.join(class_dir, f"image_{i:06d}.jpg")
        Image.fromarray(image.numpy()).save(path, "JPEG")
        counts[name] = counts.get(name, 0) + 1
        if (i + 1) % 5000 == 0:
            print(f"... exported {i + 1} images")

    total = sum(counts.values())
    print(f"Done. Exported {total} images into {OUT_DIR}")
    for name in class_names:
        print(f"  {name}: {counts.get(name, 0)}")


if __name__ == "__main__":
    main()
