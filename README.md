# Plant Disease Detection (PlantVillage)

A convolutional neural network (CNN) that looks at a photo of a plant leaf
and tells which disease it has — or that the leaf is healthy.

Built by **Muhammad Uzair Hamza** — BSc Computer Systems Engineering student, UET Peshawar.
Project #12 in my machine learning series (after the CIFAR-10 Image Classifier).

## Dataset

**PlantVillage** — ~54,306 leaf photos across 14 crops.
Kaggle page: https://www.kaggle.com/datasets/maishaidaralkhateeb/plantvillage-dataset/data

- **38 classes** in total: 30 diseases + 8 healthy.
- Crops include tomato, apple, corn (maize), grape, potato, pepper (bell),
  cherry, peach, and more.
- Example — tomato has 10 classes: bacterial spot, early blight, late blight,
  leaf mold, septoria leaf spot, spider mites, target spot, tomato yellow
  leaf curl virus, tomato mosaic virus, healthy.
- Licence: free for academic research (CC BY-NC-SA 4.0).

### Get the data

Option A — no Kaggle login needed (used for this project):

```bash
pip install tensorflow-datasets
python scripts/export_plantvillage.py
```

This downloads PlantVillage via TensorFlow Datasets and exports it into
`data/plantvillage/` automatically.

Option B — Kaggle:

1. Download the dataset from the Kaggle link above (you need a free Kaggle
   account) and unzip it.
2. Put the class folders inside `data/plantvillage/`, so it looks like:

```
data/plantvillage/
├── Pepper,_bell___Bacterial_spot/
├── Pepper,_bell___healthy/
├── Potato___Early_blight/
├── Tomato___healthy/
└── ... (one folder per class)
```

The code reads the class names from the folder names automatically — no
list to type by hand.

## Project structure

```
plant-disease-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── model.py      # CNN architecture (with augmentation layers)
│   ├── train.py      # Trains the model, saves it + class names + curves
│   ├── evaluate.py   # Test accuracy + confusion matrix
│   └── predict.py    # Predict one leaf photo from the command line
└── results/          # Created by train.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py --data-dir data/plantvillage --epochs 10
```

This splits the data into train/validation (80/20), trains the CNN, prints
validation accuracy, and saves the model to `results/plant_model.keras`,
the class names to `results/class_names.json`, and training curves to
`results/history.png`.

## Evaluate

```bash
python src/evaluate.py --data-dir data/plantvillage
```

Prints accuracy and a classification report, and saves a confusion matrix
to `results/confusion_matrix.png`.

## Predict one leaf photo

```bash
python src/predict.py path/to/leaf.jpg
```

Prints the predicted disease (or "healthy") and the top-3 probabilities.

## How the model works

- Input: 128×128×3 leaf photo, pixel values scaled to 0–1.
- Augmentation (training only): random flips, rotations and zooms, so the
  model learns the disease pattern, not the exact photo.
- Conv blocks: 32 → 64 → 128 filters (3×3), ReLU, max-pool after each.
- Dense head: 256 units → ReLU → dropout → one output per class (softmax).
- Loss: sparse categorical crossentropy. Optimizer: Adam.

## What I learned

- How folder-based datasets work: each folder name *is* the label.
- Why augmentation matters even more for leaf photos: lighting, angle and
  background change a lot between photos.
- Why a confusion matrix matters here: some diseases look alike (for
  example early blight vs late blight), and the matrix shows exactly which
  ones get mixed up.
- How a model like this could help farmers: a phone photo → instant
  disease check, before the disease spreads.

## App

```bash
pip install streamlit
streamlit run app.py
```

Opens a web page: upload a leaf photo → the disease name + confidence.
Needs `results/plant_model.keras` + `results/class_names.json` (both created
by `train.py`).

## My results

Trained for 10 epochs on PlantVillage (43,443 training images, batch size 32,
Adam optimizer, augmentation on):

- Validation accuracy: **94.57%** (10,860 validation images)
- Weighted precision: 0.95, recall: 0.95, F1-score: 0.95
- Training curves: `results/history.png`
- Confusion matrix: `results/confusion_matrix.png`

Note: an early evaluation script measured 97.40% on a skewed validation
subset (only 23 of 38 classes) — that figure was caught and discarded. The
94.57% above is verified by two independent evaluations on the full random
validation split. The trained model file (`results/plant_model.keras`) is not
committed to GitHub (97 MB); it is included in the project ZIP, and
`train.py` recreates it.

## Next ideas

- Try transfer learning (e.g. MobileNetV2) and compare accuracy.
- Test the model on real photos taken with a phone, not lab photos.
- Add treatment advice for each predicted disease.
