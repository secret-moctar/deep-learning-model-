#!/usr/bin/env python3
"""
training_utils.py — Briques modernes pour rendre l'entraînement plus efficace
=============================================================================

Trois ingrédients indépendants, tous activables via `config.py` :

    * MixUp        -> mélange linéairement 2 images et leurs labels.
    * CutMix       -> remplace un rectangle d'une image par un rectangle d'une
                      autre, et mélange les labels au prorata des surfaces.
    * Focal-Loss   -> CrossEntropy pondérée pour mieux apprendre les classes
                      rares (gamma=2 par défaut).
    * EMA          -> moyenne mobile exponentielle des poids du modèle ; on
                      évalue sur cette moyenne, plus stable que le modèle brut.

Toutes les fonctions sont importables individuellement :

    from training_utils import mixup_data, cutmix_data, FocalLoss, ModelEMA
"""
import math
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MixUp & CutMix
# ---------------------------------------------------------------------------
def mixup_data(x, y, alpha=0.2):
    """
    Mélange linéaire : x' = lam * x_a + (1 - lam) * x_b.
    Le label devient un mélange : on renvoie (x', y_a, y_b, lam) et la perte
    est calculée par `mixup_criterion`.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], float(lam)


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix : remplace un rectangle aléatoire de chaque image par le même
    rectangle pris dans une autre image du batch.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    _, _, h, w = x.size()

    # taille du rectangle  (sqrt(1 - lam) en largeur ET en hauteur)
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut_ratio), int(h * cut_ratio)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, x2 = max(0, cx - cut_w // 2), min(w, cx + cut_w // 2)
    y1, y2 = max(0, cy - cut_h // 2), min(h, cy + cut_h // 2)

    x_out = x.clone()
    x_out[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (w * h))   # surface réelle remplacée
    return x_out, y, y[idx], float(lam)


def mixup_criterion(loss_fn, logits, y_a, y_b, lam):
    """Perte pour MixUp/CutMix : combinaison convexe des deux pertes."""
    return lam * loss_fn(logits, y_a) + (1 - lam) * loss_fn(logits, y_b)


# ---------------------------------------------------------------------------
# Focal-Loss (option pour les classes rares)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal-Loss = - (1 - p_t)^gamma * log(p_t).

    Quand un échantillon est facile (p_t -> 1) il est "écrasé" -> les
    classes rares (souvent difficiles) pèsent davantage dans le gradient.
    """

    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            n_classes = logits.size(-1)
            true = torch.zeros_like(log_probs).scatter_(1, target.view(-1, 1), 1.0)
            if self.label_smoothing > 0:
                true = true * (1 - self.label_smoothing) + self.label_smoothing / n_classes
        probs = log_probs.exp()
        focal = (1 - probs) ** self.gamma
        return -(true * focal * log_probs).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) des poids
# ---------------------------------------------------------------------------
class ModelEMA:
    """
    Maintient une copie "lissée" des poids du modèle.
    Après chaque step :  ema_w = decay * ema_w + (1 - decay) * model_w.
    Évaluer sur `ema.model` donne des métriques plus stables.

        ema = ModelEMA(model, decay=0.999)
        ...
        for batch in loader:
            train_step(model, batch)
            ema.update(model)
        accuracy = eval(ema.model, test_loader)
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for ep, p in zip(self.model.parameters(), model.parameters()):
            ep.mul_(d).add_(p.detach(), alpha=1 - d)
        # On copie aussi les buffers (BatchNorm running stats, etc.).
        for eb, b in zip(self.model.buffers(), model.buffers()):
            eb.copy_(b)

    def state_dict(self):
        return self.model.state_dict()


# ---------------------------------------------------------------------------
# Test-Time Augmentation (eval avec moyenne image + image flippée)
# ---------------------------------------------------------------------------
@torch.no_grad()
def tta_logits(model, x):
    """Moyenne des logits sur (image, image flippée horizontalement)."""
    return 0.5 * (model(x) + model(torch.flip(x, dims=[3])))
