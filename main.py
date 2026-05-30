#!/usr/bin/env python3
"""
main.py — Point d'entrée FER-CE (orchestration du pipeline)
===========================================================

Couche 1 — Préparation et alignement des données :
  --explore     Exploration / distribution des classes
  --check       Vérification de l'intégrité du dataset
  --preprocess  Test du prétraitement (détection -> augmentation -> normalisation)

Couche 2 — Entraînement, évaluation, explication :
  --train       Entraînement d'un modèle  (option --model)
  --evaluate    Évaluation (accuracy, F1, matrice de confusion)
  --explain     Explication Vision-LLM (Colab)

Couche 3 — Interprétation :
  --interpret   Grad-CAM + cohérence Vision-LLM (Colab)

Divers :
  --all         Couches 1 et 2 de base (explore, check, preprocess, train, evaluate)
  --list        Liste les étapes
  --model NAME  Choix du modèle pour --train / --evaluate (défaut: resnet50)

Exemples :
  python main.py --explore --check
  python main.py --train --evaluate --model resnet50
"""
import argparse
import importlib
import os
import sys
import time

from config import DEFAULT_MODEL

# (flag, module dotté, description, fait_partie_de_--all)
STEPS = [
    ("explore",    "couche1.data_exploration", "Couche 1 — Exploration du dataset",   True),
    ("check",      "couche1.check_dataset",    "Couche 1 — Vérification du dataset",  True),
    ("preprocess", "couche1.preprocessing",    "Couche 1 — Test du prétraitement",    True),
    ("train",      "couche2.train",            "Couche 2 — Entraînement",             True),
    ("evaluate",   "couche2.evaluate",         "Couche 2 — Évaluation",               True),
    ("explain",    "couche2.explain",          "Couche 2 — Explication Vision-LLM",   False),
    ("interpret",  "couche3.interpret",        "Couche 3 — Interprétation (Grad-CAM)", False),
]


def run_step(module_name, desc):
    """Importe et exécute la fonction main() d'une étape."""
    print(f"\n{'━' * 60}\n  ▶ {desc}\n{'━' * 60}")
    start = time.time()
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, "main"):
            module.main()
        else:
            os.system(f"{sys.executable} -m {module_name}")
        print(f"  ✅ {desc} ({time.time() - start:.1f}s)")
        return True
    except Exception as exc:
        print(f"  ✗ Erreur: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="FER-CE Pipeline")
    for flag, _, desc, _ in STEPS:
        parser.add_argument(f"--{flag}", action="store_true", help=desc)
    parser.add_argument("--all",  action="store_true", help="Couches 1 et 2 de base")
    parser.add_argument("--list", action="store_true", help="Lister les étapes")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="modèle (train/evaluate)")
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════╗")
    print("║  FER-CE : Reconnaissance d'Émotions Composées  ║")
    print("║  11 classes (1..11) · RAF-DB · 3 couches       ║")
    print("╚════════════════════════════════════════════════╝")

    if args.list:
        for flag, _, desc, _ in STEPS:
            print(f"  --{flag:<12s} {desc}")
        print(f"  --{'all':<12s} Couches 1 et 2 de base")
        return

    if args.all:
        to_run = [s for s in STEPS if s[3]]
    else:
        to_run = [s for s in STEPS if getattr(args, s[0])]
    if not to_run:
        parser.print_help()
        return

    # Les étapes train/evaluate/explain/interpret lisent le modèle via sys.argv.
    results = {}
    for _, module_name, desc, _ in to_run:
        results[desc] = "✅" if run_step(module_name, desc) else "✗"

    print(f"\n{'=' * 60}\n  RÉCAPITULATIF")
    for desc, status in results.items():
        print(f"  {status} {desc}")


if __name__ == "__main__":
    main()
