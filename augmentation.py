#!/usr/bin/env python3
"""
=============================================================================
augmentation.py — Augmentation de Données
=============================================================================
Applique des augmentations aléatoires aux images pour augmenter
la diversité du dataset d'entraînement.

Augmentations :
  - Flip horizontal aléatoire
  - Rotation aléatoire (±15°)
  - Variation d'éclairage (luminosité/contraste)
  - Occlusions légères (patches noirs aléatoires)

Usage en script :
    python3 augmentation.py Image/original/train_0001.jpg

Usage en module :
    from augmentation import augment_image
    augmented = augment_image(pil_image)
=============================================================================
"""

import os
import sys
import random
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from config import OUTPUT_DIR


def random_flip(img):
    """Flip horizontal aléatoire (p=0.5)."""
    if random.random() > 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def random_rotation(img, max_angle=15):
    """Rotation aléatoire de -max_angle à +max_angle degrés."""
    angle = random.uniform(-max_angle, max_angle)
    return img.rotate(angle, fillcolor=(128, 128, 128))


def random_brightness(img, factor_range=(0.7, 1.3)):
    """Variation aléatoire de la luminosité."""
    factor = random.uniform(*factor_range)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def random_contrast(img, factor_range=(0.7, 1.3)):
    """Variation aléatoire du contraste."""
    factor = random.uniform(*factor_range)
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def random_occlusion(img, max_patches=2, max_size_ratio=0.15):
    """
    Occlusions légères : patches noirs aléatoires sur l'image.
    Simule des obstructions partielles du visage.
    """
    img_array = np.array(img)
    h, w = img_array.shape[:2]

    n_patches = random.randint(0, max_patches)
    for _ in range(n_patches):
        patch_h = random.randint(1, int(h * max_size_ratio))
        patch_w = random.randint(1, int(w * max_size_ratio))
        y = random.randint(0, h - patch_h)
        x = random.randint(0, w - patch_w)
        img_array[y:y+patch_h, x:x+patch_w] = 0  # Patch noir

    return Image.fromarray(img_array)


def augment_image(pil_image):
    """
    Applique une chaîne complète d'augmentations aléatoires.

    Args:
        pil_image: Image PIL en RGB

    Returns:
        Image PIL augmentée
    """
    img = pil_image.copy()

    # 1. Flip horizontal (50% chance)
    img = random_flip(img)

    # 2. Rotation (±15°)
    img = random_rotation(img, max_angle=15)

    # 3. Éclairage : luminosité + contraste
    img = random_brightness(img)
    img = random_contrast(img)

    # 4. Occlusions légères (30% chance)
    if random.random() < 0.3:
        img = random_occlusion(img)

    return img


# ============================================================================
# Usage en ligne de commande
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 augmentation.py <image_path>")
        print("  Ex:  python3 augmentation.py Image/original/train_0001.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"  ✗ Image non trouvée: {image_path}")
        sys.exit(1)

    img = Image.open(image_path).convert("RGB")
    print(f"  Image: {image_path} ({img.size})")

    # Générer 6 augmentations et les afficher
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original", fontweight="bold")
    axes[0, 0].axis("off")

    augment_names = [
        "Flip", "Rotation", "Luminosité",
        "Contraste", "Occlusion", "Augmentée 1", "Augmentée 2"
    ]
    augment_funcs = [
        lambda i: random_flip(i),
        lambda i: random_rotation(i),
        lambda i: random_brightness(i),
        lambda i: random_contrast(i),
        lambda i: random_occlusion(i),
        lambda i: augment_image(i),
        lambda i: augment_image(i),
    ]

    for idx, (name, func) in enumerate(zip(augment_names, augment_funcs)):
        row = (idx + 1) // 4
        col = (idx + 1) % 4
        aug = func(img.copy())
        axes[row, col].imshow(aug)
        axes[row, col].set_title(name, fontweight="bold")
        axes[row, col].axis("off")

    plt.suptitle("Augmentations de données — FER-CE", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "augmentation_demo.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] Démo sauvegardée: {out_path}")
