#!/usr/bin/env python3
"""
run_local.py — Lancement local : entraînement + enregistrement du modèle
========================================================================

Script "tout-en-un" pour faire tourner FER-CE sur une machine locale (sans
Colab). Il fait, dans l'ordre :

  1. Vérifications rapides (Couche 1) — dataset, cache des visages.
  2. Entraînement (Couche 2)          — `train.train_model(...)`.
  3. Évaluation (Couche 2)            — `evaluate.evaluate_model(...)`.
  4. Enregistrement du modèle         — manifeste JSON dans
                                         `outputs/registry.json`.

L'enregistrement (« registry ») = un seul fichier `outputs/registry.json`
qui liste TOUS les modèles entraînés sur la machine, avec leurs métriques,
les chemins de leurs poids, leurs hyper-paramètres et la date d'entraînement.
On peut ensuite recharger n'importe quel modèle enregistré via
`load_registered_model(name)` ou la commande `--load NAME`.

Usage :
    python run_local.py                          # défaut : resnet50, 45 époques
    python run_local.py --model vit_base_patch16_224 --epochs 30
    python run_local.py --all-models             # entraîne config.MODELS un par un
    python run_local.py --skip-train --model resnet50   # juste évaluer + enregistrer
    python run_local.py --list                   # liste les modèles enregistrés
    python run_local.py --load resnet50          # recharge un modèle enregistré

Le notebook `demo_fer_ce.ipynb` reste la voie recommandée sur Google Colab
(GPU gratuit + Vision-LLM). Ce script-ci est la version « machine perso ».
"""
import os
import json
import argparse
import datetime
import platform

import torch

from config import (OUTPUT_DIR, MODELS, DEFAULT_MODEL, EPOCHS, BATCH_SIZE,
                    LR_HEAD, LR_BACKBONE, WEIGHT_DECAY, LABEL_SMOOTHING,
                    WARMUP_EPOCHS, USE_MIXUP, USE_CUTMIX, USE_EMA, USE_FOCAL,
                    TTA, SEED, model_path, ema_path)

REGISTRY_PATH = os.path.join(OUTPUT_DIR, "registry.json")


# ---------------------------------------------------------------------------
# Registry : lecture / écriture du manifeste
# ---------------------------------------------------------------------------
def _load_registry():
    """Charge le manifeste ou en crée un vide."""
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"models": {}}


def _save_registry(reg):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def list_models():
    """Affiche la liste des modèles enregistrés (la registry)."""
    reg = _load_registry()
    if not reg["models"]:
        print("  Aucun modèle enregistré pour l'instant.")
        print(f"  Lance d'abord : python run_local.py --model resnet50")
        return
    print(f"  Modèles enregistrés dans {REGISTRY_PATH}\n")
    print(f"  {'Nom':<32s} {'Test acc':>9s} {'Test F1':>9s} {'Date':>20s}")
    print(f"  {'-' * 75}")
    for name, info in reg["models"].items():
        m = info.get("metrics", {})
        acc = m.get("accuracy", {}).get("test", 0.0)
        f1  = m.get("macro_f1", {}).get("test", 0.0)
        date = info.get("registered_at", "?")[:19]
        print(f"  {name:<32s} {acc*100:>8.2f}% {f1*100:>8.2f}% {date:>20s}")


def register_model(model_name, metrics, epochs_run):
    """Ajoute / met à jour le modèle dans le registry et retourne l'entrée créée."""
    reg = _load_registry()
    ckpt  = model_path(model_name)
    ekpt  = ema_path(model_name)
    entry = {
        "model": model_name,
        "registered_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "weights": {
            "raw": ckpt if os.path.exists(ckpt) else None,
            "ema": ekpt if os.path.exists(ekpt) else None,
        },
        "artifacts": {
            "history":   os.path.join(OUTPUT_DIR, f"history_{model_name}.json"),
            "curves":    os.path.join(OUTPUT_DIR, f"curves_{model_name}.png"),
            "metrics":   os.path.join(OUTPUT_DIR, f"metrics_{model_name}.json"),
            "report":    os.path.join(OUTPUT_DIR, f"report_{model_name}.txt"),
            "overall":   os.path.join(OUTPUT_DIR, f"eval_{model_name}_overall.png"),
            "per_class": os.path.join(OUTPUT_DIR, f"eval_{model_name}_per_class.png"),
            "confusion": os.path.join(OUTPUT_DIR, f"eval_{model_name}_confusion.png"),
        },
        "hyperparams": {
            "epochs_run":      epochs_run,
            "batch_size":      BATCH_SIZE,
            "warmup_epochs":   WARMUP_EPOCHS,
            "lr_head":         LR_HEAD,
            "lr_backbone":     LR_BACKBONE,
            "weight_decay":    WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "seed":            SEED,
            "mixup":           USE_MIXUP,
            "cutmix":          USE_CUTMIX,
            "ema":             USE_EMA,
            "focal":           USE_FOCAL,
            "tta":             TTA,
        },
        "environment": {
            "python":    platform.python_version(),
            "torch":     torch.__version__,
            "cuda":      torch.cuda.is_available(),
            "device":    (torch.cuda.get_device_name(0)
                          if torch.cuda.is_available() else "cpu"),
        },
        "metrics": metrics,
    }
    reg["models"][model_name] = entry
    _save_registry(reg)
    print(f"\n  [✓] Modèle « {model_name} » enregistré dans {REGISTRY_PATH}")
    return entry


def load_registered_model(model_name, prefer_ema=True):
    """Recharge un modèle enregistré (architecture + meilleurs poids)."""
    reg = _load_registry()
    if model_name not in reg["models"]:
        raise KeyError(
            f"Modèle « {model_name} » introuvable dans la registry.\n"
            f"  Liste : python run_local.py --list")
    info  = reg["models"][model_name]
    paths = info["weights"]
    ckpt  = paths["ema"] if (prefer_ema and paths.get("ema")) else paths["raw"]
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(f"Poids manquants pour {model_name}: {ckpt}")

    # Imports tardifs : on évite de charger torch dans des sous-commandes (--list).
    from model import build_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    tag = "EMA" if (prefer_ema and paths.get("ema")) else "raw"
    print(f"  [{tag}] Chargé : {ckpt}")
    return model, info


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------
def _print_header(title):
    line = "═" * 72
    print(f"\n{line}\n  {title}\n{line}")


def run_pipeline(model_name=DEFAULT_MODEL, epochs=EPOCHS,
                 skip_check=False, skip_train=False):
    """Pipeline local : (option) check → (option) train → evaluate → register."""
    _print_header(f"FER-CE — pipeline local pour « {model_name} »")
    print(f"  device   : {'cuda — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"  epochs   : {epochs}")
    print(f"  registry : {REGISTRY_PATH}")

    # --- 1. Vérifications Couche 1 ------------------------------------------
    if not skip_check:
        _print_header("Couche 1 — vérification du dataset")
        import check_dataset
        check_dataset.main()

    # --- 2. Entraînement Couche 2 -------------------------------------------
    if not skip_train:
        _print_header(f"Couche 2 — entraînement ({model_name})")
        import train
        train.train_model(model_name, epochs=epochs)
    else:
        print("\n  --skip-train : on saute l'entraînement")
        if not os.path.exists(model_path(model_name)) and \
           not os.path.exists(ema_path(model_name)):
            raise FileNotFoundError(
                f"Pas de checkpoint pour {model_name} — entraîne d'abord")

    # --- 3. Évaluation Couche 2 ---------------------------------------------
    _print_header(f"Couche 2 — évaluation ({model_name})")
    import evaluate as eval_mod
    metrics = eval_mod.evaluate_model(model_name, prefer_ema=True, use_tta=TTA)

    # --- 4. Enregistrement du modèle ----------------------------------------
    _print_header("Enregistrement du modèle")
    entry = register_model(model_name, metrics, epochs_run=epochs)
    test_acc = metrics["accuracy"]["test"]
    test_f1  = metrics["macro_f1"]["test"]
    print(f"  ✅ {model_name} : test acc {test_acc*100:.2f}%  ·  test F1 {test_f1*100:.2f}%")
    return entry


def main():
    parser = argparse.ArgumentParser(
        description="FER-CE — entraînement local + enregistrement du modèle")
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help=f"nom du modèle (défaut: {DEFAULT_MODEL})")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"nombre d'époques (défaut: {EPOCHS})")
    parser.add_argument("--all-models", action="store_true",
                        help="entraîne tous les modèles de config.MODELS")
    parser.add_argument("--skip-check", action="store_true",
                        help="saute la vérification Couche 1")
    parser.add_argument("--skip-train", action="store_true",
                        help="saute l'entraînement (évalue + enregistre)")
    parser.add_argument("--list", action="store_true",
                        help="affiche les modèles enregistrés et quitte")
    parser.add_argument("--load", metavar="NAME",
                        help="recharge un modèle enregistré et affiche son info")
    args = parser.parse_args()

    if args.list:
        list_models()
        return
    if args.load:
        _, info = load_registered_model(args.load)
        print(json.dumps(info, indent=2))
        return

    targets = MODELS if args.all_models else [args.model]
    for name in targets:
        run_pipeline(name, epochs=args.epochs,
                     skip_check=args.skip_check, skip_train=args.skip_train)

    if len(targets) > 1:
        _print_header("Récapitulatif")
        list_models()


if __name__ == "__main__":
    main()
