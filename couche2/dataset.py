#!/usr/bin/env python3
"""
dataset.py — Couche de données pour l'entraînement (Couche 2)

Réutilise la Couche 1 (détection/recadrage du visage via les bounding boxes
RAF-DB) puis :

  1. Recadre chaque visage UNE seule fois et met le résultat en cache
     (outputs/faces_<split>.npz) -> évite de re-décoder/re-cropper à chaque
     époque, l'entraînement devient beaucoup plus rapide.
  2. Expose un Dataset PyTorch qui applique l'augmentation À LA VOLÉE
     (différente à chaque époque) au lieu d'une augmentation figée.

Usage en script (construit/rafraîchit le cache) :
    python3 dataset.py
    python3 dataset.py --rebuild
"""
import os
import sys
# Bootstrap : permet `python couche2/dataset.py` ET `python -m couche2.dataset`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_DIR, LABEL_FILE, LABEL_OFFSET, IMG_SIZE, NUM_CLASSES, OUTPUT_DIR
from couche1.face_detection import detect_and_crop

# Statistiques de normalisation ImageNet (backbone ResNet pré-entraîné).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Lecture des labels
# ---------------------------------------------------------------------------
def read_split(split):
    """Retourne la liste [(filename, label_0based), ...] pour 'train' ou 'test'."""
    items = []
    with open(LABEL_FILE, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            fn, label = parts[0], int(parts[1]) - LABEL_OFFSET
            current = "train" if fn.startswith("train") else "test"
            if current == split and os.path.exists(os.path.join(IMAGE_DIR, fn)):
                items.append((fn, label))
    return items


# ---------------------------------------------------------------------------
# Cache des visages recadrés
# ---------------------------------------------------------------------------
def _cache_path(split):
    return os.path.join(OUTPUT_DIR, f"faces_{split}.npz")


def build_face_cache(split):
    """Recadre tous les visages d'un split et les stocke en uint8 (cache .npz)."""
    items = read_split(split)
    print(f"  [{split}] recadrage de {len(items)} visages...", end=" ", flush=True)
    images = np.zeros((len(items), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    labels = np.zeros(len(items), dtype=np.int64)
    for i, (fn, label) in enumerate(items):
        face = detect_and_crop(os.path.join(IMAGE_DIR, fn), mode="bbox").convert("RGB")
        images[i] = np.asarray(face.resize((IMG_SIZE, IMG_SIZE)))
        labels[i] = label
    np.savez_compressed(_cache_path(split), images=images, labels=labels)
    print(f"OK -> {os.path.basename(_cache_path(split))}")
    return images, labels


def load_face_cache(split, rebuild=False):
    """Charge le cache des visages, le construit si absent ou si rebuild=True."""
    path = _cache_path(split)
    if os.path.exists(path) and not rebuild:
        data = np.load(path)
        return data["images"], data["labels"]
    return build_face_cache(split)


# ---------------------------------------------------------------------------
# Transformations (augmentation à la volée)
# ---------------------------------------------------------------------------
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25),  # occlusions légères
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------
class FERDataset(Dataset):
    """Dataset des expressions composées : visages en cache + augmentation."""

    def __init__(self, split, train=False, rebuild=False):
        self.images, self.labels = load_face_cache(split, rebuild=rebuild)
        self.transform = TRAIN_TRANSFORM if train else EVAL_TRANSFORM

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        return self.transform(img), int(self.labels[idx])

    def class_counts(self):
        """Nombre d'échantillons par classe (vecteur de taille NUM_CLASSES)."""
        return np.bincount(self.labels, minlength=NUM_CLASSES)

    def sample_weights(self):
        """Poids par échantillon pour un WeightedRandomSampler (classes équilibrées)."""
        counts = self.class_counts()
        per_class = 1.0 / np.maximum(counts, 1)
        return torch.as_tensor([per_class[l] for l in self.labels], dtype=torch.double)


if __name__ == "__main__":
    rebuild = "--rebuild" in sys.argv
    print("=" * 60)
    print("  DATASET — Construction du cache des visages")
    print("=" * 60)
    for split in ("train", "test"):
        ds = FERDataset(split, train=False, rebuild=rebuild)
        print(f"  {split:5s}: {len(ds)} images | classes={ds.class_counts().tolist()}")
    print("  ✅ Cache prêt")
