#!/usr/bin/env python3
"""
interpret.py — Couche 3 : Interprétation Multimodale (XAI)
==========================================================

Relie ce que le MODÈLE regarde à ce que le VISION-LLM dit. Trois étapes :

  1. Grad-CAM        : heatmap des zones du visage qui déclenchent la prédiction
                       du CNN (où le modèle "regarde").
  2. Zones faciales  : la heatmap est découpée en 3 bandes (front/sourcils, yeux,
                       bouche/joues) -> approximation des Action Units (AUs).
  3. Cohérence       : on extrait les indices faciaux cités dans l'explication
                       du Vision-LLM et on vérifie qu'ils correspondent à la
                       zone réellement activée (idée du "Causal Emotion Grounding").

Sortie : outputs/interpret_<i>.png  +  outputs/interpretation.json

------------------------------------------------------------------------------
À EXÉCUTER SUR GOOGLE COLAB si on veut la partie Vision-LLM (--vlm).
Sans Vision-LLM, le Grad-CAM seul fonctionne en local.
    python interpret.py --model resnet50            # Grad-CAM seul
    python interpret.py --model resnet50 --vlm      # + cohérence Vision-LLM
------------------------------------------------------------------------------

Importable :
    import interpret
    interpret.interpret_images(["a.jpg"], model_name="resnet50", use_vlm=False)
"""
import os
import sys
# Bootstrap : permet `python couche3/interpret.py` ET `python -m couche3.interpret`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import argparse
import textwrap

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import (IMAGE_DIR, OUTPUT_DIR, DEFAULT_MODEL,
                    model_path, ema_path, class_label)
from couche1.face_detection import detect_and_crop
from couche2.dataset import EVAL_TRANSFORM
from couche2.model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Les 3 bandes horizontales du visage et les mots-clés faciaux associés.
REGIONS = ["Front / sourcils", "Yeux", "Bouche / joues"]
REGION_CUE_WORDS = {
    0: ["eyebrow", "brow", "forehead", "wrinkle"],
    1: ["eye", "eyes", "eyelid", "gaze"],
    2: ["mouth", "lip", "cheek", "nose", "jaw", "smile", "smiling", "frown", "teeth"],
}


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    """
    Grad-CAM : visualise les régions de l'image qui contribuent le plus à une
    classe. On "branche" deux hooks sur une couche de convolution :
      - forward hook  -> récupère les activations (ce que la couche voit) ;
      - backward hook -> récupère les gradients   (ce qui influence la classe).
    La heatmap = somme pondérée des activations par l'importance des gradients.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """Retourne (heatmap 224x224 normalisée 0-1, indice de classe 0-based)."""
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(1))
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)        # importance
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx


def get_target_layer(model, model_name):
    """Couche cible du Grad-CAM (dernière couche convolutive)."""
    if model_name == "resnet50":
        return model.layer4
    raise ValueError(
        f"Grad-CAM implémenté pour resnet50 (CNN). Pour {model_name} (ViT/Swin), "
        "il faudrait utiliser l'attention rollout — voir la documentation.")


# ---------------------------------------------------------------------------
# Analyse des zones faciales + cohérence linguistique
# ---------------------------------------------------------------------------
def region_scores(cam):
    """Score d'activation moyen des 3 bandes (front, yeux, bouche). Somme = 1."""
    h = cam.shape[0]
    bands = np.array([cam[:h // 3].mean(),
                      cam[h // 3:2 * h // 3].mean(),
                      cam[2 * h // 3:].mean()])
    return bands / (bands.sum() + 1e-8)


def cues_in_text(text):
    """Renvoie l'ensemble des indices de régions (0/1/2) cités dans un texte."""
    text = (text or "").lower()
    return {region for region, words in REGION_CUE_WORDS.items()
            if any(word in text for word in words)}


# ---------------------------------------------------------------------------
# Interprétation d'une image
# ---------------------------------------------------------------------------
def interpret_image(model, gradcam, image_path, vlm=None):
    """Grad-CAM (+ cohérence Vision-LLM si vlm fourni) pour une image."""
    face = detect_and_crop(image_path, mode="bbox").convert("RGB")
    x = EVAL_TRANSFORM(face).unsqueeze(0).to(device)
    cam, class_idx = gradcam.generate(x)
    scores = region_scores(cam)
    focus = int(scores.argmax())

    result = {
        "image": os.path.basename(image_path),
        "class_id": class_label(class_idx),
        "region_scores": {REGIONS[i]: float(scores[i]) for i in range(3)},
        "focus_region": REGIONS[focus],
    }

    if vlm is not None:
        from couche2.vision_llm import explain_emotion
        ex = explain_emotion(vlm[0], vlm[1], face)
        mentioned = cues_in_text(f"{ex['emotion']} {ex['explanation']}")
        result.update({
            "vlm_emotion": ex["emotion"],
            "explanation": ex["explanation"],
            "mentioned_regions": [REGIONS[i] for i in sorted(mentioned)],
            # Cohérence : la zone activée par le CNN est-elle citée par le LLM ?
            "coherent": focus in mentioned,
        })
    return face, cam, result


def save_visualization(face, cam, result, index):
    """3 panneaux : visage | Grad-CAM | scores des zones + texte du Vision-LLM."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    axes[0].imshow(face); axes[0].axis("off")
    axes[0].set_title("Visage", fontweight="bold")

    axes[1].imshow(face)
    axes[1].imshow(cam, cmap="jet", alpha=0.5)
    axes[1].axis("off")
    axes[1].set_title(f"Grad-CAM — Classe {result['class_id']}", fontweight="bold")

    scores = [result["region_scores"][r] for r in REGIONS]
    colors = ["#E8734C" if r == result["focus_region"] else "#4C9BE8" for r in REGIONS]
    axes[2].barh(REGIONS, scores, color=colors, edgecolor="black")
    axes[2].set_xlim(0, 1); axes[2].invert_yaxis()
    axes[2].set_xlabel("Activation Grad-CAM (part)")

    title = f"Zone dominante : {result['focus_region']}"
    if "coherent" in result:
        verdict = "✅ cohérent" if result["coherent"] else "⚠ incohérent"
        caption = (f"Vision-LLM : {result.get('vlm_emotion', '')}\n" +
                   "\n".join(textwrap.wrap(result.get("explanation") or "", 46)) +
                   f"\n\nCohérence image/texte : {verdict}")
        axes[2].text(0.0, -0.30, caption, transform=axes[2].transAxes,
                     va="top", fontsize=8, family="monospace")
    axes[2].set_title(title, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"interpret_{index}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [→] {out}")


def interpret_images(image_paths, model_name=DEFAULT_MODEL, use_vlm=False,
                     quantize=False, prefer_ema=True):
    """Interprète une liste d'images : Grad-CAM (+ cohérence Vision-LLM)."""
    ckpt_ema  = ema_path(model_name)
    ckpt_main = model_path(model_name)
    ckpt = ckpt_ema if (prefer_ema and os.path.exists(ckpt_ema)) else ckpt_main
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Modèle introuvable: {ckpt_main} — entraîne-le d'abord.")
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device).eval()
    gradcam = GradCAM(model, get_target_layer(model, model_name))

    vlm = None
    if use_vlm:
        print("  Chargement du Vision-LLM (Qwen2-VL)...")
        from couche2.vision_llm import load_vision_llm
        vlm = load_vision_llm(quantize=quantize)

    results = []
    for index, path in enumerate(image_paths):
        face, cam, result = interpret_image(model, gradcam, path, vlm=vlm)
        results.append(result)
        print(f"\n  [{index}] {result['image']}  -> Classe {result['class_id']}")
        print(f"      Zone dominante : {result['focus_region']}")
        if "coherent" in result:
            print(f"      Cohérence LLM  : {'OUI' if result['coherent'] else 'NON'}")
        save_visualization(face, cam, result, index)

    out_json = os.path.join(OUTPUT_DIR, "interpretation.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  [✓] {out_json}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Couche 3 — Interprétation FER-CE")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--vlm", action="store_true", help="ajoute la cohérence Vision-LLM")
    parser.add_argument("--n", type=int, default=5)
    args, _ = parser.parse_known_args()

    print("=" * 60)
    print("  COUCHE 3 — INTERPRÉTATION MULTIMODALE (Grad-CAM)")
    print("=" * 60)
    imgs = [f for f in os.listdir(IMAGE_DIR) if f.startswith("test")] \
        or [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
    random.seed(0)
    selected = [os.path.join(IMAGE_DIR, f)
                for f in random.sample(imgs, min(args.n, len(imgs)))]
    interpret_images(selected, model_name=args.model, use_vlm=args.vlm)
    print("\n  ✅ Interprétation terminée")


if __name__ == "__main__":
    main()
