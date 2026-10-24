"""Build the Plant Disease Detection project report PDF.

Reads the verified evaluation (eval2.log) for the per-class table, embeds
the training curves + confusion matrix plots, and writes a report with the
author's name on the title page and footer.

Usage:
    python scripts/build_report_pdf.py
"""

import os
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
EVAL_LOG = os.path.join(PROJECT_DIR, "eval2.log")
OUT_PDF = os.path.join(PROJECT_DIR, "Plant_Disease_Detection_Report.pdf")

AUTHOR = "Muhammad Uzair Hamza"
ROW_RE = re.compile(
    r"^(.+?)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s*$")


def parse_report(path):
    """Parse sklearn classification_report text into class rows + averages."""
    classes, macro, weighted, accuracy = [], None, None, None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip().startswith("accuracy"):
                parts = line.split()
                accuracy = float(parts[1])
            elif line.strip().startswith("macro avg"):
                m = ROW_RE.match(line.strip())
                macro = [float(m.group(2)), float(m.group(3)),
                         float(m.group(4)), int(m.group(5))]
            elif line.strip().startswith("weighted avg"):
                m = ROW_RE.match(line.strip())
                weighted = [float(m.group(2)), float(m.group(3)),
                            float(m.group(4)), int(m.group(5))]
            else:
                m = ROW_RE.match(line.strip())
                if m and not line.strip().startswith("precision"):
                    classes.append((m.group(1), float(m.group(2)),
                                    float(m.group(3)), float(m.group(4)),
                                    int(m.group(5))))
    return classes, macro, weighted, accuracy


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawString(20 * mm, 12 * mm,
                      "Muhammad Uzair Hamza  |  Plant Disease Detection (PlantVillage)")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def main():
    classes, macro, weighted, accuracy = parse_report(EVAL_LOG)
    assert len(classes) == 38, f"expected 38 classes, got {len(classes)}"
    assert accuracy is not None and weighted is not None

    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title2", parent=styles["Title"], fontSize=24,
                           leading=30, spaceAfter=6)
    subtitle = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=13,
                              leading=18, alignment=1, spaceAfter=4)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16,
                        spaceBefore=14, spaceAfter=6)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                        spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=11,
                          leading=15, spaceAfter=6)
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=9,
                           leading=12)
    cell = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=7,
                          leading=9)
    cell_h = ParagraphStyle("CellH", parent=styles["Normal"], fontSize=7,
                            leading=9, textColor=colors.white,
                            fontName="Helvetica-Bold")

    story = []
    story.append(Spacer(1, 30 * mm))
    story.append(Paragraph("Plant Disease Detection", title))
    story.append(Paragraph("Using Convolutional Neural Networks", subtitle))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Project Report", subtitle))
    story.append(Spacer(1, 12 * mm))
    story.append(Paragraph(f"<b>{AUTHOR}</b>", subtitle))
    story.append(Paragraph("BSc Computer Systems Engineering (5th Semester)", subtitle))
    story.append(Paragraph("University of Engineering and Technology, Peshawar", subtitle))
    story.append(Paragraph("24 October 2026", subtitle))
    story.append(PageBreak())

    story.append(Paragraph("1. Introduction", h1))
    story.append(Paragraph(
        "Crop diseases destroy a large share of farm output every year, and "
        "early detection is the key to stopping their spread. This project "
        "builds a deep learning model that looks at a photo of a plant leaf "
        "and identifies the disease it carries, or confirms the leaf is "
        "healthy. A farmer can therefore check a crop with a single phone "
        "photo, before the disease spreads to the whole field.", body))
    story.append(Paragraph(
        "The model is a convolutional neural network (CNN) built and trained "
        "from scratch on the PlantVillage dataset. No pretrained weights were "
        "used.", body))

    story.append(Paragraph("2. Dataset", h1))
    story.append(Paragraph(
        "PlantVillage is a public dataset of plant leaf photos, available at "
        "https://www.kaggle.com/datasets/maishaidaralkhateeb/plantvillage-dataset/data "
        "and free for academic research. For this project it was obtained "
        "through TensorFlow Datasets (no Kaggle login required) and exported "
        "into class folders.", body))
    story.append(Paragraph(
        "<b>54,303</b> leaf photos across <b>38 classes</b> (30 diseases + "
        "8 healthy) and <b>14 crops</b>, including tomato, apple, corn, "
        "grape, potato, pepper, cherry, peach, strawberry, squash, soybean, "
        "raspberry, blueberry and orange. Data split: <b>43,443</b> images "
        "for training and <b>10,860</b> for validation (80/20 random split).",
        body))

    story.append(Paragraph("3. Model Architecture", h1))
    story.append(Paragraph(
        "A custom CNN (built from scratch, not a pretrained model such as "
        "MobileNet or ResNet):", body))
    arch = [
        ["Stage", "Details"],
        ["Input", "128 x 128 x 3 leaf photo"],
        ["Augmentation (training only)", "Random horizontal flip, rotation (0.1), zoom (0.1)"],
        ["Rescaling", "Pixel values scaled from 0-255 to 0-1"],
        ["Conv block 1", "32 filters (3x3), ReLU, max-pool (2x2)"],
        ["Conv block 2", "64 filters (3x3), ReLU, max-pool (2x2)"],
        ["Conv block 3", "128 filters (3x3), ReLU, max-pool (2x2), dropout (0.25)"],
        ["Dense head", "256 units, ReLU, dropout (0.5)"],
        ["Output", "38 units, softmax (one per class)"],
    ]
    t = Table([[Paragraph(f"<b>{c}</b>" if r == 0 else c, cell_h if r == 0 else cell)
                for c in row] for r, row in enumerate(arch)],
              colWidths=[52 * mm, 118 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    story.append(Paragraph("4. Training Setup", h1))
    story.append(Paragraph(
        "Optimizer: Adam. Loss: sparse categorical crossentropy. Batch size: "
        "32. Epochs: 10. Image size: 128 x 128. Augmentation was applied "
        "during training so the model learns disease patterns rather than "
        "memorising exact photos. Training ran on CPU.", body))

    story.append(Paragraph("5. Results", h1))
    story.append(Paragraph(
        f"Validation accuracy: <b>{accuracy * 100:.2f}%</b> on 10,860 "
        f"validation images. Weighted precision: {weighted[0]:.2f}, recall: "
        f"{weighted[1]:.2f}, F1-score: {weighted[2]:.2f}. Final training "
        "accuracy reached 90.29%.", body))
    story.append(Paragraph(
        "Evaluation note: an early check measured 97.40% on a skewed "
        "validation subset (only 23 of 38 classes) and was discarded. The "
        "94.57% figure above comes from the full random validation split and "
        "was confirmed by two independent evaluations.", body))
    story.append(Paragraph("Training curves", h2))
    story.append(Image(os.path.join(RESULTS_DIR, "history.png"),
                       width=150 * mm, height=60 * mm))
    story.append(Paragraph("Confusion matrix (rows = true class, columns = predicted)", h2))
    story.append(Image(os.path.join(RESULTS_DIR, "confusion_matrix.png"),
                       width=150 * mm, height=150 * mm))

    story.append(Paragraph("Per-class results (validation set)", h2))
    header = ["Class", "Precision", "Recall", "F1-score", "Support"]
    rows = [[Paragraph(f"<b>{c}</b>", cell_h) for c in header]]
    for name, p, r, f1, s in classes:
        rows.append([Paragraph(name, cell), Paragraph(f"{p:.2f}", cell),
                     Paragraph(f"{r:.2f}", cell), Paragraph(f"{f1:.2f}", cell),
                     Paragraph(str(s), cell)])
    rows.append([Paragraph("<b>Weighted avg</b>", cell),
                 Paragraph(f"<b>{weighted[0]:.2f}</b>", cell),
                 Paragraph(f"<b>{weighted[1]:.2f}</b>", cell),
                 Paragraph(f"<b>{weighted[2]:.2f}</b>", cell),
                 Paragraph(f"<b>{weighted[3]}</b>", cell)])
    ct = Table(rows, colWidths=[82 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm],
               repeatRows=1)
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef5ee")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(ct)

    story.append(Paragraph("6. How to Run", h1))
    story.append(Paragraph(
        "Install: <font face='Courier' size='9'>pip install -r requirements.txt</font><br/>"
        "Train: <font face='Courier' size='9'>python src/train.py --data-dir data/plantvillage --epochs 10</font><br/>"
        "Evaluate: <font face='Courier' size='9'>python src/evaluate.py --data-dir data/plantvillage</font><br/>"
        "Predict one photo: <font face='Courier' size='9'>python src/predict.py path/to/leaf.jpg</font><br/>"
        "Web app: <font face='Courier' size='9'>streamlit run app.py</font> "
        "(upload a leaf photo, get the disease + confidence).", body))

    story.append(Paragraph("7. Conclusion", h1))
    story.append(Paragraph(
        "A from-scratch CNN reached 94.57% validation accuracy across 38 "
        "plant disease classes, with a working web app for real leaf photos. "
        "Future work: transfer learning comparison, testing on real phone "
        "photos taken in fields, and treatment advice for each disease.", body))
    story.append(Paragraph(
        "Code: https://github.com/uzair-hamza/plant-disease-detection", small))

    doc = SimpleDocTemplate(OUT_PDF, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title="Plant Disease Detection - Muhammad Uzair Hamza",
                            author=AUTHOR)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
