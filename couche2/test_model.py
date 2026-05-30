#!/usr/bin/env python3
"""
couche2/test_model.py — Inférence sur des images individuelles (Couche 2)
========================================================================

Charge un classifieur entraîné et prédit le NUMÉRO de classe (1..11) d'une
image, avec une visualisation : image originale, visage recadré, top-3.

Usage :
    python -m couche2.test_model
    python -m couche2.test_model --model resnet50

Importable :
    from couche2 import test_model
    model = test_model.load_classifier("resnet50")
"""
import os
import sys
# Bootstrap : permet `python couche2/test_model.py` ET `python -m couche2.test_model`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import argparse

import torch
import matplotlib.pyplot as plt
from PIL import Image

from config import (LABEL_FILE, LABEL_OFFSET, IMAGE_DIR, OUTPUT_DIR,
                    DEFAULT_MODEL, model_path, ema_path, class_label)
from couche1.face_detection import detect_and_crop
from couche2.dataset import EVAL_TRANSFORM
from couche2.model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_classifier(model_name=DEFAULT_MODEL, prefer_ema=True):
    """Reconstruit l'architecture et charge le meilleur checkpoint disponible."""
    ckpt_ema  = ema_path(model_name)
    ckpt_main = model_path(model_name)
    if prefer_ema and os.path.exists(ckpt_ema):
        ckpt = ckpt_ema
    elif os.path.exists(ckpt_main):
        ckpt = ckpt_main
    else:
        print(f"  ✗ Modèle non trouvé : {ckpt_main}")
        print(f"    Entraîne-le d'abord : python -m couche2.train --model {model_name}")
        sys.exit(1)
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict(model, face):
    """Prédit la classe d'un visage PIL. Retourne (classe, confiance%, top3)."""
    probs = torch.softmax(model(EVAL_TRANSFORM(face).unsqueeze(0).to(device)), dim=1)[0]
    top3_probs, top3_idx = torch.topk(probs, 3)
    top3 = [(class_label(int(i)), p.item() * 100)
            for i, p in zip(top3_idx, top3_probs)]
    return top3[0][0], top3[0][1], top3


def get_true_class(image_path):
    """Numéro de classe réel (1..11) depuis le fichier de labels, ou None."""
    basename = os.path.basename(image_path)
    try:
        with open(LABEL_FILE, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == basename:
                    return int(parts[1])  # déjà 1-based dans le fichier
    except Exception:
        pass
    return None


def show_result(image_path, face, pred_class, conf, top3, true_class, index, show):
    """Visualisation : original | visage recadré | top-3."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(Image.open(image_path).convert("RGB"))
    axes[0].set_title("Image originale", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(face)
    axes[1].set_title("Entrée du modèle (visage recadré)", fontweight="bold")
    axes[1].axis("off")

    correct = (pred_class == true_class)
    color = "#4CAF50" if correct else "#F44336"
    names = [f"Classe {c}" for c, _ in top3][::-1]
    values = [v for _, v in top3][::-1]
    bars = axes[2].barh(names, values, color=color, edgecolor="black")
    axes[2].set_xlim(0, 100)
    axes[2].set_xlabel("Confiance (%)")
    for bar in bars:
        axes[2].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{bar.get_width():.1f}%", va="center")
    status = "✅" if correct else "❌"
    axes[2].set_title(f"{status} Préd: Classe {pred_class} ({conf:.1f}%)\n"
                      f"Réel: Classe {true_class if true_class else '?'}",
                      fontweight="bold", color="green" if correct else "red")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"res_{index}_{'ok' if correct else 'fail'}.png")
    plt.savefig(out, dpi=150)
    print(f"  [→] {os.path.basename(out)}")
    plt.show() if show else plt.close()


def test_one_image(image_path, model, mode="bbox", index=0, show=True):
    """Prédit et visualise une image."""
    face = detect_and_crop(image_path, mode=mode).convert("RGB")
    pred_class, conf, top3 = predict(model, face)
    true_class = get_true_class(image_path)
    print(f"  [{index}] {os.path.basename(image_path)} -> Classe {pred_class} ({conf:.1f}%)")
    show_result(image_path, face, pred_class, conf, top3, true_class, index, show)


def main():
    parser = argparse.ArgumentParser(description="Inférence FER-CE")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args, _ = parser.parse_known_args()

    model = load_classifier(args.model)
    print("\n1. Tester une image précise\n2. Tester des images aléatoires")
    choice = input("Choix: ")
    mode = input("Mode (bbox/auto) [défaut: bbox]: ") or "bbox"

    if choice == "1":
        test_one_image(input("Chemin de l'image: "), model, mode=mode, show=True)
    else:
        num = int(input("Combien d'images ? "))
        all_imgs = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
        for i, name in enumerate(random.sample(all_imgs, min(num, len(all_imgs)))):
            test_one_image(os.path.join(IMAGE_DIR, name), model, mode=mode,
                           index=i, show=False)


if __name__ == "__main__":
    main()
