#!/usr/bin/env python3
"""
train.py — Entraînement d'un modèle (Couche 2)
==============================================

Entraîne UN modèle (resnet50 / ViT / Swin) sur les 11 classes composées (1..11).

Pipeline "haute efficacité" :
  * augmentation à la volée                      (dataset.py)
  * WeightedRandomSampler                        -> batches équilibrés
  * MixUp / CutMix aléatoires par batch          (training_utils.py)
  * label smoothing OU Focal-Loss                (config.USE_FOCAL)
  * entraînement en 2 phases                     -> warmup tête, puis tout
  * cosine LR + AdamW + mixed precision (AMP)
  * EMA des poids du modèle                      -> évaluation stabilisée
  * sauvegarde du meilleur modèle (EMA si actif) -> outputs/best_<model>.pth

Usage :
    python train.py                        # modèle par défaut (resnet50)
    python train.py --model vit_base_patch16_224
    python train.py --model swin_base_patch4_window7_224 --epochs 30

Importable :
    import train
    train.train_model("resnet50", epochs=45)
"""
import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score

from config import (OUTPUT_DIR, BATCH_SIZE, EPOCHS, WARMUP_EPOCHS, LR_HEAD,
                    LR_BACKBONE, WEIGHT_DECAY, LABEL_SMOOTHING, SEED,
                    DEFAULT_MODEL, model_path, ema_path,
                    USE_MIXUP, MIXUP_ALPHA, USE_CUTMIX, CUTMIX_ALPHA,
                    USE_EMA, EMA_DECAY, USE_FOCAL, FOCAL_GAMMA, TTA)
from dataset import FERDataset
from model import build_model, set_backbone_trainable, param_groups
from training_utils import (mixup_data, cutmix_data, mixup_criterion,
                            FocalLoss, ModelEMA, tta_logits)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
def make_loaders(num_workers=2):
    """DataLoaders : train (échantillonnage équilibré) et test (séquentiel)."""
    train_ds = FERDataset("train", train=True)
    test_ds  = FERDataset("test",  train=False)
    sampler = WeightedRandomSampler(train_ds.sample_weights(),
                                    num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Boucles d'entraînement / d'évaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, loss_fn, optimizer, scaler, ema=None):
    """Une époque d'entraînement avec MixUp/CutMix aléatoires. Retourne (loss, acc, f1)."""
    model.train()
    loss_sum, n, preds, trues = 0.0, 0, [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # On choisit aléatoirement MixUp, CutMix ou rien (1/3 chacun si actifs).
        roll = np.random.rand()
        use_mix = USE_MIXUP and roll < 0.33
        use_cut = USE_CUTMIX and 0.33 <= roll < 0.66

        if use_mix:
            x, y_a, y_b, lam = mixup_data(x, y, alpha=MIXUP_ALPHA)
        elif use_cut:
            x, y_a, y_b, lam = cutmix_data(x, y, alpha=CUTMIX_ALPHA)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(x)
            if use_mix or use_cut:
                loss = mixup_criterion(loss_fn, logits, y_a, y_b, lam)
            else:
                loss = loss_fn(logits, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if ema is not None:
            ema.update(model)

        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        trues.extend(y.cpu().tolist())

    acc = float(np.mean(np.array(preds) == np.array(trues)))
    f1  = f1_score(trues, preds, average="macro", zero_division=0)
    return loss_sum / n, acc, f1


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, use_tta=False):
    """Évaluation (loss, accuracy, macro-F1) avec TTA optionnel."""
    model.eval()
    loss_sum, n, preds, trues = 0.0, 0, [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = tta_logits(model, x) if use_tta else model(x)
        loss = loss_fn(logits, y)
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        preds.extend(logits.argmax(1).cpu().tolist())
        trues.extend(y.cpu().tolist())
    acc = float(np.mean(np.array(preds) == np.array(trues)))
    f1  = f1_score(trues, preds, average="macro", zero_division=0)
    return loss_sum / n, acc, f1


# ---------------------------------------------------------------------------
# Tracé des courbes
# ---------------------------------------------------------------------------
def _plot_curves(history, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, title) in zip(axes, [("loss", "Loss"), ("acc", "Accuracy"),
                                       ("f1", "Macro-F1")]):
        ax.plot(history[f"train_{key}"], "o-", ms=3, label="Train")
        ax.plot(history[f"test_{key}"],  "o-", ms=3, label="Test")
        if f"ema_{key}" in history:
            ax.plot(history[f"ema_{key}"], "o--", ms=3, label="Test (EMA)")
        ax.set_title(f"{title} — {model_name}", fontweight="bold")
        ax.set_xlabel("Époque"); ax.set_ylabel(title)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"curves_{model_name}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] {out}")


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------
def train_model(model_name=DEFAULT_MODEL, epochs=EPOCHS):
    """Entraîne un modèle et sauvegarde le meilleur checkpoint. Retourne l'historique."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("=" * 72)
    print(f"  ENTRAÎNEMENT — {model_name}")
    print(f"  Device: {device} · Epochs: {epochs} · Batch: {BATCH_SIZE}")
    print(f"  MixUp:{USE_MIXUP}  CutMix:{USE_CUTMIX}  EMA:{USE_EMA}  Focal:{USE_FOCAL}  TTA:{TTA}")
    print("=" * 72)

    train_loader, test_loader = make_loaders()
    model = build_model(model_name, pretrained=True).to(device)

    # Loss : soit Focal-Loss, soit CrossEntropy avec label smoothing.
    if USE_FOCAL:
        loss_fn = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # `torch.amp.GradScaler("cuda")` >= 2.3 ; ancien `torch.cuda.amp.GradScaler` sinon.
    if device.type == "cuda":
        scaler = (torch.amp.GradScaler("cuda")
                  if hasattr(torch.amp, "GradScaler")
                  else torch.cuda.amp.GradScaler())
    else:
        scaler = None
    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None

    # --- Phase 1 (warmup) : on n'entraîne que la tête neuve --------------------
    set_backbone_trainable(model, False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = None

    history = {f"{s}_{k}": [] for s in ("train", "test", "ema")
               for k in ("loss", "acc", "f1")} if ema else \
              {f"{s}_{k}": [] for s in ("train", "test")
               for k in ("loss", "acc", "f1")}
    best_f1 = 0.0
    ckpt = model_path(model_name)
    ekpt = ema_path(model_name)

    head = (f"  {'Ep':>3s} | {'phase':<8s} | "
            f"{'T.loss':>7s} {'T.acc':>6s} {'T.f1':>6s} | "
            f"{'V.loss':>7s} {'V.acc':>6s} {'V.f1':>6s}")
    if ema:
        head += f" | {'E.acc':>6s} {'E.f1':>6s}"
    print("\n" + head); print(f"  {'-' * len(head)}")

    for epoch in range(epochs):
        # --- Phase 2 : on dégèle tout le réseau après le warmup --------------
        if epoch == WARMUP_EPOCHS:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(
                param_groups(model, LR_BACKBONE, LR_HEAD),
                weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs - WARMUP_EPOCHS, 1))

        phase = "warmup" if epoch < WARMUP_EPOCHS else "finetune"
        tl, ta, tf = train_one_epoch(model, train_loader, loss_fn, optimizer,
                                     scaler, ema=ema)
        vl, va, vf = eval_epoch(model, test_loader, loss_fn, use_tta=TTA)

        line = (f"  {epoch + 1:>3d} | {phase:<8s} | "
                f"{tl:>7.3f} {ta:>6.3f} {tf:>6.3f} | "
                f"{vl:>7.3f} {va:>6.3f} {vf:>6.3f}")

        if scheduler is not None:
            scheduler.step()

        for key, val in zip(["train_loss", "train_acc", "train_f1",
                             "test_loss", "test_acc", "test_f1"],
                            (tl, ta, tf, vl, va, vf)):
            history[key].append(float(val))

        # Évaluation EMA
        if ema:
            _, ea, ef = eval_epoch(ema.model, test_loader, loss_fn, use_tta=TTA)
            history["ema_loss"].append(0.0)
            history["ema_acc"].append(float(ea))
            history["ema_f1"].append(float(ef))
            line += f" | {ea:>6.3f} {ef:>6.3f}"

        # On retient le meilleur F1 (EMA si actif, sinon brut).
        candidate_f1 = ef if ema else vf
        if candidate_f1 > best_f1:
            best_f1 = candidate_f1
            torch.save(model.state_dict(), ckpt)
            if ema:
                torch.save(ema.state_dict(), ekpt)
            line += " ★"
        print(line)

    _plot_curves(history, model_name)
    with open(os.path.join(OUTPUT_DIR, f"history_{model_name}.json"), "w") as f:
        json.dump(history, f, indent=2)

    best_acc = (max(history["ema_acc"]) if ema else max(history["test_acc"]))
    print(f"\n  ✅ {model_name} terminé — best macro-F1: {best_f1:.4f}"
          f" | best accuracy: {best_acc:.4f}")
    print(f"  [✓] Poids : {ckpt}")
    if ema:
        print(f"  [✓] EMA   : {ekpt}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Entraînement FER-CE")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="nom du modèle")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args, _ = parser.parse_known_args()
    train_model(args.model, args.epochs)


if __name__ == "__main__":
    main()
