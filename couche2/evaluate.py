#!/usr/bin/env python3
"""
evaluate.py — Évaluation d'un modèle entraîné (Couche 2)
========================================================

Évalue UN modèle sur les ensembles train ET test, et produit :

  * eval_<model>_overall.png    -> accuracy globale train vs test
  * eval_<model>_per_class.png  -> accuracy de chaque classe (1..11)
  * eval_<model>_confusion.png  -> matrice de confusion (test)
  * report_<model>.txt          -> precision / recall / F1 par classe
  * metrics_<model>.json        -> toutes les métriques (utilisé pour comparer)

Par défaut on évalue le checkpoint EMA s'il existe (plus stable), sinon le
checkpoint standard. On peut forcer un choix avec --raw / --ema.

Usage :
    python evaluate.py --model resnet50
    python evaluate.py --model resnet50 --raw    # ignore l'EMA
    python evaluate.py --model resnet50 --no-tta # désactive le TTA

Importable :
    import evaluate
    metrics = evaluate.evaluate_model("resnet50")
"""
import os
import sys
# Bootstrap : permet `python couche2/evaluate.py` ET `python -m couche2.evaluate`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)

from config import (OUTPUT_DIR, BATCH_SIZE, NUM_CLASSES, CLASS_NAMES,
                    DEFAULT_MODEL, model_path, ema_path, TTA)
from couche2.dataset import FERDataset
from couche2.model import build_model
from couche2.training_utils import tta_logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(model_name, prefer_ema=True, run_name=None):
    """Reconstruit l'architecture (model_name) et charge le checkpoint du run.

    run_name = identifiant des fichiers (défaut = model_name). Sépare l'arch de
    l'identifiant de run pour permettre plusieurs variantes par architecture.
    """
    run_name = run_name or model_name
    ckpt_main = model_path(run_name)
    ckpt_ema  = ema_path(run_name)
    if prefer_ema and os.path.exists(ckpt_ema):
        ckpt = ckpt_ema
        tag  = "EMA"
    elif os.path.exists(ckpt_main):
        ckpt = ckpt_main
        tag  = "raw"
    else:
        raise FileNotFoundError(
            f"Aucun checkpoint pour {run_name}.\n"
            f"  Entraîne-le d'abord : python -m couche2.train --model {model_name}")
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"  [{tag}] {os.path.basename(ckpt)}")
    return model.to(device).eval()


@torch.no_grad()
def predict_split(model, split, use_tta=True):
    """Retourne (y_true, y_pred) pour un split, sans augmentation."""
    loader = DataLoader(FERDataset(split, train=False), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=2)
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = tta_logits(model, x) if use_tta else model(x)
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(y.tolist())
    return np.array(y_true), np.array(y_pred)


def per_class_accuracy(y_true, y_pred):
    """Accuracy (= recall) pour chaque classe."""
    acc = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        mask = y_true == c
        acc[c] = (y_pred[mask] == c).mean() if mask.sum() else 0.0
    return acc


# --- Visualisations --------------------------------------------------------
def _plot_overall(train_acc, test_acc, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Train", "Test"], [train_acc, test_acc],
                  color=["#4C9BE8", "#E8734C"], edgecolor="black")
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy globale — {model_name}", fontweight="bold")
    for bar, val in zip(bars, [train_acc, test_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val * 100:.1f}%", ha="center", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"eval_{model_name}_overall.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [✓] {out}")


def _plot_per_class(train_pc, test_pc, model_name):
    x = np.arange(NUM_CLASSES)
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - w / 2, train_pc, w, label="Train", color="#4C9BE8", edgecolor="black")
    ax.bar(x + w / 2, test_pc,  w, label="Test",  color="#E8734C", edgecolor="black")
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy par classe — {model_name}", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"Classe {n}" for n in CLASS_NAMES],
                                         rotation=45, ha="right", fontsize=9)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"eval_{model_name}_per_class.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [✓] {out}")


def _plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Classe prédite"); ax.set_ylabel("Classe réelle")
    ax.set_title(f"Matrice de confusion (Test) — {model_name}", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"eval_{model_name}_confusion.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [✓] {out}")


def evaluate_model(model_name=DEFAULT_MODEL, prefer_ema=True, use_tta=True, run_name=None):
    """Évalue un modèle, génère les graphiques et retourne le dict des métriques.

    model_name : architecture (passée à build_model).
    run_name   : identifiant pour les fichiers (poids, report, metrics). Défaut = model_name.
    """
    run_name = run_name or model_name
    print("=" * 60)
    print(f"  ÉVALUATION — arch={model_name} run={run_name} · TTA={use_tta} · EMA={prefer_ema}")
    print("=" * 60)

    model = load_trained_model(model_name, prefer_ema=prefer_ema, run_name=run_name)
    yt_tr, yp_tr = predict_split(model, "train", use_tta=use_tta)
    yt_te, yp_te = predict_split(model, "test",  use_tta=use_tta)

    train_acc, test_acc = accuracy_score(yt_tr, yp_tr), accuracy_score(yt_te, yp_te)
    train_f1 = f1_score(yt_tr, yp_tr, average="macro", zero_division=0)
    test_f1  = f1_score(yt_te, yp_te, average="macro", zero_division=0)
    train_pc, test_pc = per_class_accuracy(yt_tr, yp_tr), per_class_accuracy(yt_te, yp_te)

    print(f"\n  Accuracy  — Train: {train_acc*100:.2f}%   Test: {test_acc*100:.2f}%")
    print(f"  Macro-F1  — Train: {train_f1*100:.2f}%   Test: {test_f1*100:.2f}%\n")
    print(f"  {'Classe':<10s} {'Train':>8s} {'Test':>8s}")
    print(f"  {'-' * 28}")
    for c in range(NUM_CLASSES):
        print(f"  Classe {CLASS_NAMES[c]:<3s} {train_pc[c]*100:>7.1f}% {test_pc[c]*100:>7.1f}%")

    report = classification_report(yt_te, yp_te, labels=range(NUM_CLASSES),
                                   target_names=[f"Classe {n}" for n in CLASS_NAMES],
                                   zero_division=0)
    print(f"\n{report}")
    with open(os.path.join(OUTPUT_DIR, f"report_{run_name}.txt"), "w") as f:
        f.write(f"Run: {run_name}   Arch: {model_name}\n")
        f.write(f"Accuracy train/test: {train_acc:.4f} / {test_acc:.4f}\n")
        f.write(f"Macro-F1 train/test: {train_f1:.4f} / {test_f1:.4f}\n\n{report}")

    _plot_overall(train_acc, test_acc, run_name)
    _plot_per_class(train_pc, test_pc, run_name)
    _plot_confusion(yt_te, yp_te, run_name)

    metrics = {
        "name":  run_name,
        "arch":  model_name,
        "accuracy": {"train": float(train_acc), "test": float(test_acc)},
        "macro_f1": {"train": float(train_f1), "test": float(test_f1)},
        "per_class_accuracy": {CLASS_NAMES[c]: {"train": float(train_pc[c]),
                                                "test": float(test_pc[c])}
                               for c in range(NUM_CLASSES)},
        "ema_used": prefer_ema,
        "tta": use_tta,
    }
    with open(os.path.join(OUTPUT_DIR, f"metrics_{run_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [✓] metrics_{run_name}.json\n  ✅ Évaluation terminée")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Évaluation FER-CE")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--raw", action="store_true", help="utilise le checkpoint brut (pas EMA)")
    parser.add_argument("--no-tta", action="store_true", help="désactive le Test-Time Augmentation")
    args, _ = parser.parse_known_args()
    evaluate_model(args.model, prefer_ema=not args.raw,
                   use_tta=(not args.no_tta) and TTA)


if __name__ == "__main__":
    main()
