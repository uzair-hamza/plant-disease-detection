"""CNN architecture for plant disease detection.

The model starts with augmentation layers (active only during training),
then three conv blocks, then a dense head. The number of output classes is
passed in, because it comes from the dataset folders (38 for PlantVillage).
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_model(input_shape=(128, 128, 3), num_classes=38) -> keras.Model:
    """Build the CNN. Augmentation runs only while training."""
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ], name="augmentation")

    model = keras.Sequential([
        layers.Input(shape=input_shape),
        augmentation,
        # Scale pixels from 0-255 to 0-1.
        layers.Rescaling(1.0 / 255.0),
        # --- Conv block 1 ---
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        # --- Conv block 2 ---
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        # --- Conv block 3 ---
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # --- Dense head ---
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
