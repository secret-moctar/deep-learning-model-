#!/usr/bin/env python3
"""
=============================================================================
check_dataset.py — Vérification de l'intégrité du dataset
=============================================================================
Vérifie toutes les images du dataset avant l'entraînement :
  - Image corrompue (impossible à ouvrir)
  - Image pas en RGB (grayscale, RGBA, etc.)
  - Image trop petite (visage pas exploitable)
  - Image sans bounding box
  - Label manquant ou invalide
  - Fichier référencé dans les labels mais absent

Usage :
    python3 check_dataset.py
=============================================================================
"""

import os
import sys
from PIL import Image
from config import IMAGE_DIR, LABEL_FILE, BBOX_DIR, OUTPUT_DIR, LABEL_OFFSET, NUM_CLASSES

# Seuils
MIN_WIDTH = 28
MIN_HEIGHT = 28


def check_image(image_path):
    """
    Vérifie une image. Retourne une liste de problèmes (vide = OK).
    """
    problems = []

    # 1. Peut-on ouvrir l'image ?
    try:
        img = Image.open(image_path)
    except Exception as e:
        return [f"CORROMPUE — impossible à ouvrir: {e}"]

    # 2. Peut-on la charger complètement ?
    try:
        img.load()  # Force le chargement complet (détecte les fichiers tronqués)
    except Exception as e:
        return [f"CORROMPUE — fichier tronqué: {e}"]

    # 3. Mode couleur
    if img.mode != "RGB":
        problems.append(f"MODE={img.mode} (attendu: RGB)")

    # 4. Peut-on convertir en RGB ?
    try:
        img_rgb = img.convert("RGB")
    except Exception as e:
        problems.append(f"CONVERSION RGB impossible: {e}")

    # 5. Taille trop petite
    w, h = img.size
    if w < MIN_WIDTH or h < MIN_HEIGHT:
        problems.append(f"TROP PETITE: {w}×{h} (min: {MIN_WIDTH}×{MIN_HEIGHT})")

    return problems


def check_bbox(filename):
    """Vérifie si la bounding box existe et est valide."""
    base = filename.replace(".jpg", "")
    bbox_file = os.path.join(BBOX_DIR, f"{base}_boundingbox.txt")

    if not os.path.exists(bbox_file):
        return "BBOX manquante"

    try:
        with open(bbox_file, "r") as f:
            coords = f.readline().strip().split()
            if len(coords) < 4:
                return "BBOX invalide (< 4 coords)"
            x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
            if x2 <= x1 or y2 <= y1:
                return f"BBOX invalide: x1={x1} y1={y1} x2={x2} y2={y2}"
    except Exception as e:
        return f"BBOX erreur: {e}"

    return None  # OK


def main():
    print("=" * 60)
    print("  CHECK DATASET — Vérification de l'intégrité")
    print("=" * 60)

    # --- 1. Charger les labels ---
    label_entries = {}  # filename → label
    with open(LABEL_FILE, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"  ⚠ Ligne {line_num}: format invalide → '{line}'")
                continue
            fn, label_raw = parts[0], int(parts[1])
            label = label_raw - LABEL_OFFSET
            label_entries[fn] = label

    print(f"\n  Labels chargés: {len(label_entries)}")

    # --- 2. Vérifier chaque image ---
    stats = {
        "total": 0,
        "ok": 0,
        "corrupted": 0,
        "bad_mode": 0,
        "too_small": 0,
        "no_bbox": 0,
        "bad_label": 0,
        "missing_file": 0,
    }
    bad_images = []

    for filename, label in label_entries.items():
        stats["total"] += 1
        image_path = os.path.join(IMAGE_DIR, filename)
        issues = []

        # Vérifier le label
        if label < 0 or label >= NUM_CLASSES:
            issues.append(f"LABEL INVALIDE: {label} (max: {NUM_CLASSES-1})")
            stats["bad_label"] += 1

        # Vérifier que le fichier existe
        if not os.path.exists(image_path):
            issues.append("FICHIER MANQUANT")
            stats["missing_file"] += 1
            bad_images.append((filename, issues))
            continue

        # Vérifier l'image
        img_issues = check_image(image_path)
        for issue in img_issues:
            if "CORROMPUE" in issue:
                stats["corrupted"] += 1
            elif "MODE" in issue:
                stats["bad_mode"] += 1
            elif "TROP PETITE" in issue:
                stats["too_small"] += 1
        issues.extend(img_issues)

        # Vérifier la bounding box
        bbox_issue = check_bbox(filename)
        if bbox_issue:
            issues.append(bbox_issue)
            stats["no_bbox"] += 1

        if issues:
            bad_images.append((filename, issues))
        else:
            stats["ok"] += 1

    # --- 3. Rapport ---
    print(f"\n{'='*60}")
    print(f"  RAPPORT")
    print(f"{'='*60}")
    print(f"  Total images:    {stats['total']}")
    print(f"  ✅ OK:           {stats['ok']}")
    print(f"  ✗ Corrompues:    {stats['corrupted']}")
    print(f"  ✗ Mauvais mode:  {stats['bad_mode']}")
    print(f"  ✗ Trop petites:  {stats['too_small']}")
    print(f"  ✗ Sans bbox:     {stats['no_bbox']}")
    print(f"  ✗ Label invalide:{stats['bad_label']}")
    print(f"  ✗ Fichier absent:{stats['missing_file']}")

    if bad_images:
        print(f"\n  IMAGES PROBLÉMATIQUES ({len(bad_images)}):")
        print(f"  {'Fichier':<25s} {'Problèmes'}")
        print(f"  {'-'*55}")
        for fn, issues in bad_images[:50]:  # Max 50
            print(f"  {fn:<25s} {', '.join(issues)}")
        if len(bad_images) > 50:
            print(f"  ... et {len(bad_images)-50} autres")

        # Sauvegarder la liste
        report_path = os.path.join(OUTPUT_DIR, "bad_images.txt")
        with open(report_path, "w") as f:
            for fn, issues in bad_images:
                f.write(f"{fn}\t{', '.join(issues)}\n")
        print(f"\n  [✓] Liste complète: {report_path}")
    else:
        print(f"\n  ✅ Aucune image problématique — dataset OK !")

    # --- 4. Vérifier les images orphelines (dans le dossier mais pas dans les labels) ---
    if os.path.exists(IMAGE_DIR):
        all_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg")]
        orphans = [f for f in all_files if f not in label_entries]
        if orphans:
            print(f"\n  ⚠ {len(orphans)} images dans le dossier sans label:")
            for f in orphans[:10]:
                print(f"    {f}")
            if len(orphans) > 10:
                print(f"    ... et {len(orphans)-10} autres")
        else:
            print(f"\n  ✅ Toutes les images ({len(all_files)}) ont un label")

    print(f"\n  ✅ Vérification terminée")


if __name__ == "__main__":
    main()
