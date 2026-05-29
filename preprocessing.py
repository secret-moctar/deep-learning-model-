#!/usr/bin/env python3
"""
02_preprocessing.py — Pipeline : detection → augmentation → normalisation
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import IMAGE_DIR, LABEL_FILE, IMG_SIZE, LABEL_OFFSET, NUM_CLASSES, BATCH_SIZE
from face_detection import detect_and_crop
from normalization import normalize_image
from augmentation import augment_image


def process_one_image(image_path, augment=False, model="resnet"):
    """Pipeline : face_detection → augmentation → normalization."""
    cropped = detect_and_crop(image_path, mode="bbox")
    if augment:
        cropped = augment_image(cropped)
    return normalize_image(cropped, model=model)


def load_all_images(split="train", augment=False, model="resnet"):
    """Charge toutes les images d'un split."""
    print(f"  Chargement {split}...", end=" ", flush=True)
    images, labels = [], []
    with open(LABEL_FILE, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            fn, label = parts[0], int(parts[1]) - LABEL_OFFSET
            if ("train" if fn.startswith("train") else "test") != split:
                continue
            path = os.path.join(IMAGE_DIR, fn)
            if not os.path.exists(path):
                continue
            images.append(process_one_image(path, augment=augment, model=model))
            labels.append(label)
    x = torch.tensor(np.array(images)).float()
    y = torch.tensor(labels).long()
    print(f"{x.shape[0]} images, shape={x.shape}")
    return x, y


def create_dataloaders(batch_size=BATCH_SIZE, model="resnet"):
    x_train, y_train = load_all_images("train", augment=True, model=model)
    x_test, y_test = load_all_images("test", augment=False, model=model)
    class_counts = np.bincount(y_train.numpy(), minlength=NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size)
    print(f"  [✓] Train={len(x_train)} Test={len(x_test)} Batch={batch_size}")
    return train_loader, test_loader, class_weights, len(x_train), len(x_test)


def main():
    print("=" * 60)
    print("  02 — PRÉTRAITEMENT")
    print("=" * 60)
    train_loader, test_loader, cw, _, _ = create_dataloaders(model="resnet")
    x, y = next(iter(train_loader))
    print(f"  Batch: x={x.shape} y={y.shape} labels={torch.unique(y).tolist()}")
    print("  ✅ OK")


if __name__ == "__main__":
    main()
