#!/usr/bin/env python3
"""
=============================================================================
normalization.py — Normalisation des Images
=============================================================================
Redimensionne et normalise les images selon le modèle utilisé.

Chaque modèle a ses propres paramètres de normalisation :
  - "resnet" / "vit"  : ImageNet mean/std, 224×224
  - "clip"            : CLIP mean/std, 224×224
  - "blip2" / "llava" : Utiliser le processor du modèle directement
  - "simple"          : Juste [0,1], taille configurable (pour CNN custom)

Usage en script :
    python3 normalization.py Image/original/train_0001.jpg
    python3 normalization.py Image/original/train_0001.jpg --model resnet

Usage en module :
    from normalization import normalize_image
    tensor = normalize_image(pil_image, model="resnet")  # ImageNet norms
    tensor = normalize_image(pil_image, model="simple")  # [0,1] basique
=============================================================================
"""

import os
import sys
import numpy as np
from PIL import Image
from config import IMG_SIZE, OUTPUT_DIR

# ============================================================================
# Paramètres de normalisation par modèle
# ============================================================================
NORM_PARAMS = {
    "resnet": {
        "size": 224,
        "mean": [0.485, 0.456, 0.406],  # ImageNet
        "std":  [0.229, 0.224, 0.225],
    },
    "vit": {
        "size": 224,
        "mean": [0.485, 0.456, 0.406],  # ImageNet (même que ResNet)
        "std":  [0.229, 0.224, 0.225],
    },
    "clip": {
        "size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],  # CLIP
        "std":  [0.26862954, 0.26130258, 0.27577711],
    },
    "simple": {
        "size": IMG_SIZE,  # 64 par défaut (config.py)
        "mean": None,      # Pas de normalisation mean/std
        "std":  None,
    },
}


def normalize_image(pil_image, model="resnet", size=None):
    """
    Normalise une image PIL selon le modèle cible.

    Args:
        pil_image: Image PIL en RGB
        model: "resnet", "vit", "clip", ou "simple"
        size: Taille custom (sinon celle du modèle)

    Returns:
        numpy array (C, H, W) float32 normalisé
    """
    params = NORM_PARAMS.get(model, NORM_PARAMS["resnet"])
    target_size = size or params["size"]

    # 1. Redimensionner
    img = pil_image.resize((target_size, target_size))

    # 2. Convertir en float32 [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # 3. Transposer (H, W, C) → (C, H, W)
    img_array = img_array.transpose(2, 0, 1)

    # 4. Normaliser avec mean/std du modèle
    if params["mean"] is not None and params["std"] is not None:
        mean = np.array(params["mean"], dtype=np.float32).reshape(3, 1, 1)
        std = np.array(params["std"], dtype=np.float32).reshape(3, 1, 1)
        img_array = (img_array - mean) / std

    return img_array


def denormalize_image(tensor_chw, model="resnet"):
    """
    Inverse de normalize_image : (C, H, W) → image affichable.

    Args:
        tensor_chw: numpy array (C, H, W)
        model: Le modèle utilisé pour la normalisation

    Returns:
        numpy array (H, W, C) uint8 [0, 255]
    """
    params = NORM_PARAMS.get(model, NORM_PARAMS["resnet"])
    img = tensor_chw.copy()

    # Dé-normaliser mean/std
    if params["mean"] is not None and params["std"] is not None:
        mean = np.array(params["mean"], dtype=np.float32).reshape(3, 1, 1)
        std = np.array(params["std"], dtype=np.float32).reshape(3, 1, 1)
        img = img * std + mean

    # (C,H,W) → (H,W,C) et [0,255]
    img = img.transpose(1, 2, 0)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


# ============================================================================
# Usage en ligne de commande
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 normalization.py <image_path>                # défaut: resnet")
        print("  python3 normalization.py <image_path> --model clip   # CLIP norms")
        print("  python3 normalization.py <image_path> --model simple # [0,1] basique")
        sys.exit(1)

    image_path = sys.argv[1]

    # Lire --model
    model = "resnet"
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model = sys.argv[idx + 1]

    if not os.path.exists(image_path):
        print(f"  ✗ Image non trouvée: {image_path}")
        sys.exit(1)

    img = Image.open(image_path).convert("RGB")
    params = NORM_PARAMS.get(model, NORM_PARAMS["resnet"])

    print(f"  Image: {image_path}")
    print(f"  Modèle: {model}")
    print(f"  Taille: {params['size']}×{params['size']}")
    print(f"  Mean: {params['mean']}")
    print(f"  Std:  {params['std']}")

    # Normaliser
    normalized = normalize_image(img, model=model)
    print(f"\n  Shape: {normalized.shape}")
    print(f"  Min/Max: {normalized.min():.4f} / {normalized.max():.4f}")
    print(f"  Dtype: {normalized.dtype}")

    # Dé-normaliser et sauvegarder
    back = denormalize_image(normalized, model=model)
    out_path = os.path.join(OUTPUT_DIR, f"normalized_{model}.png")
    Image.fromarray(back).save(out_path)
    print(f"  [✓] Preview: {out_path}")
