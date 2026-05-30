#!/usr/bin/env python3
"""
explain.py — Pipeline prédiction + explication (Couche 2)
=========================================================

Combine, pour chaque image :

  1. un classifieur (ResNet/ViT/Swin) -> NUMÉRO de classe (1..11) + confiance
  2. le Vision-LLM (Qwen2-VL)         -> description + EXPLICATION textuelle

Puis sauvegarde une visualisation (visage + classe + explication) et un JSON.

------------------------------------------------------------------------------
À EXÉCUTER SUR GOOGLE COLAB (le Vision-LLM ~4.5 Go ne se télécharge pas en local).
    !pip install -q -U transformers accelerate bitsandbytes pillow
    python explain.py --model resnet50
------------------------------------------------------------------------------

Importable :
    import explain
    explain.explain_images(["a.jpg", "b.jpg"], model_name="resnet50")
"""
import os
import sys
# Bootstrap : permet `python couche2/explain.py` ET `python -m couche2.explain`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import argparse
import textwrap

import torch
import matplotlib.pyplot as plt

from config import (IMAGE_DIR, OUTPUT_DIR, CLASS_NAMES, DEFAULT_MODEL,
                    model_path, ema_path, class_label)
from couche1.face_detection import detect_and_crop
# vision_llm est importé "à l'usage" (transformers/accelerate sont installés
# uniquement sur Google Colab — voir requirements.txt).

N_IMAGES = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Classifieur — fournit le numéro de classe (optionnel)
# ---------------------------------------------------------------------------
def load_classifier(model_name, prefer_ema=True):
    """Charge un classifieur entraîné (EMA si dispo). Retourne (None, None) s'il est absent."""
    ckpt_ema  = ema_path(model_name)
    ckpt_main = model_path(model_name)
    ckpt = ckpt_ema if (prefer_ema and os.path.exists(ckpt_ema)) else ckpt_main
    if not os.path.exists(ckpt):
        print(f"  ⚠ {ckpt} introuvable — explication sans numéro de classe.")
        return None, None
    from couche2.model import build_model
    from couche2.dataset import EVAL_TRANSFORM
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model.to(device).eval(), EVAL_TRANSFORM


@torch.no_grad()
def classify(model, transform, face):
    """Prédit le numéro de classe (1..11) et la confiance d'un visage PIL."""
    probs = torch.softmax(model(transform(face).unsqueeze(0).to(device)), dim=1)[0]
    idx = int(probs.argmax())
    return class_label(idx), probs[idx].item() * 100


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def save_visualization(face, result, index):
    """Sauvegarde : visage analysé | classe + explication générée."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(face); axes[0].axis("off")
    axes[0].set_title("Visage analysé", fontweight="bold")

    lines = []
    if result.get("class_id") is not None:
        lines.append(f"Classe (classifieur) : {result['class_id']}"
                     f"  ({result['confidence']:.1f}%)")
    lines.append(f"Émotion (Vision-LLM) : {result['emotion']}")
    lines += ["", "Explication générée :"]
    lines += textwrap.wrap(result["explanation"] or "", width=52)

    axes[1].axis("off")
    axes[1].text(0.0, 0.98, "\n".join(lines), va="top", ha="left",
                 fontsize=11, family="monospace")
    axes[1].set_title("Vision-LLM — Émotion + Explication", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"explain_{index}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [→] {out}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def explain_images(image_paths, model_name=DEFAULT_MODEL, quantize=False):
    """Applique classifieur (numéro) + Vision-LLM (explication) sur des images."""
    from couche2.vision_llm import load_vision_llm, explain_emotion  # lazy import (Colab)
    print("  Chargement du Vision-LLM (Qwen2-VL)...")
    vlm, processor = load_vision_llm(quantize=quantize)
    clf, transform = load_classifier(model_name)

    results = []
    for index, path in enumerate(image_paths):
        face = detect_and_crop(path, mode="bbox").convert("RGB")

        class_id, conf = (None, None)
        if clf is not None:
            class_id, conf = classify(clf, transform, face)

        res = explain_emotion(vlm, processor, face)
        res.update({"image": os.path.basename(path),
                    "class_id": class_id, "confidence": conf})
        results.append(res)

        print(f"\n  [{index}] {res['image']}")
        print(f"      Classe     : {class_id}")
        print(f"      Vision-LLM : {res['emotion']}")
        print(f"      Explication: {res['explanation']}")
        save_visualization(face, res, index)

    out_json = os.path.join(OUTPUT_DIR, "explanations.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  [✓] {out_json}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Explication FER-CE")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=N_IMAGES)
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("  EXPLICATION DES ÉMOTIONS (classifieur + Vision-LLM)")
    print("=" * 60)
    imgs = [f for f in os.listdir(IMAGE_DIR) if f.startswith("test")] \
        or [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
    random.seed(0)
    selected = [os.path.join(IMAGE_DIR, f)
                for f in random.sample(imgs, min(args.n, len(imgs)))]
    explain_images(selected, model_name=args.model)
    print("\n  ✅ Explications terminées")


if __name__ == "__main__":
    main()
