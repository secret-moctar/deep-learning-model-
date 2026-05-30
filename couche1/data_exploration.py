#!/usr/bin/env python3
"""
data_exploration.py — Exploration du dataset (Couche 1)
=======================================================

Analyse la répartition des 11 classes (numéros 1..11) et affiche des
échantillons d'images. Produit :

  * outputs/class_distribution.png  -> nombre d'images par classe (train / test)
  * outputs/sample_images.png       -> exemples d'images par classe

Usage :  python data_exploration.py
"""
import os
import sys
# Bootstrap : permet `python couche1/data_exploration.py` ET `python -m couche1.data_exploration`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from PIL import Image

from config import (IMAGE_DIR, LABEL_FILE, OUTPUT_DIR, LABEL_OFFSET,
                    CLASS_NAMES, NUM_CLASSES, class_label)


def load_labels():
    """Lit le fichier de labels et sépare train / test. Labels en 0-based."""
    train_data, test_data = [], []
    with open(LABEL_FILE, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            fn, label = parts[0], int(parts[1]) - LABEL_OFFSET
            (train_data if fn.startswith("train") else test_data).append((fn, label))
    return train_data, test_data


def plot_distribution(train_data, test_data):
    """Histogramme du nombre d'images par classe (train et test)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (data, title) in zip(axes, [(train_data, "TRAIN"), (test_data, "TEST")]):
        counts = [0] * NUM_CLASSES
        for _, label in data:
            counts[label] += 1
        bars = ax.bar(CLASS_NAMES, counts, color="steelblue", edgecolor="white")
        ax.set_title(f"Distribution — {title} ({len(data)} images)", fontweight="bold")
        ax.set_xlabel("Classe (numéro)")
        ax.set_ylabel("Nombre d'images")
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    str(val), ha="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] {out}")


def plot_samples(train_data, n=3):
    """Affiche n échantillons d'images pour chaque classe."""
    by_class = {}
    for fn, label in train_data:
        by_class.setdefault(label, []).append(fn)
    labels = sorted(by_class.keys())

    fig, axes = plt.subplots(len(labels), n, figsize=(3 * n, 2.4 * len(labels)))
    for row, label in enumerate(labels):
        for col in range(n):
            ax = axes[row, col]
            files = by_class[label]
            if col < len(files):
                path = os.path.join(IMAGE_DIR, files[col])
                if os.path.exists(path):
                    ax.imshow(Image.open(path).convert("RGB"))
            ax.set_title(f"Classe {class_label(label)}", fontsize=8, fontweight="bold")
            ax.axis("off")
    plt.suptitle("Échantillons par classe", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "sample_images.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] {out}")


def main():
    print("=" * 60)
    print("  COUCHE 1 — EXPLORATION DU DATASET")
    print("=" * 60)
    train_data, test_data = load_labels()
    print(f"\n  Total: {len(train_data) + len(test_data)} | "
          f"Train: {len(train_data)} | Test: {len(test_data)} | Classes: {NUM_CLASSES}")

    counts = [0] * NUM_CLASSES
    for _, label in train_data + test_data:
        counts[label] += 1
    print(f"\n  {'Classe':<10s} {'Images':>8s}")
    print(f"  {'-' * 20}")
    for c in range(NUM_CLASSES):
        print(f"  Classe {CLASS_NAMES[c]:<3s} {counts[c]:>8d}")

    print("\n  Génération des graphiques...")
    plot_distribution(train_data, test_data)
    plot_samples(train_data)
    print("\n  ✅ Exploration terminée")


if __name__ == "__main__":
    main()
