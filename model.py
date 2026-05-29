#!/usr/bin/env python3
"""
model.py — Construction des modèles (Couche 2)
==============================================

Constructeur unique partagé par l'entraînement, l'évaluation, l'inférence et
l'interprétation. Garantit que l'architecture est IDENTIQUE entre la sauvegarde
et le rechargement des poids.

Modèles supportés :
  * "resnet50"                     -> CNN (torchvision)
  * tout modèle `timm` (ViT, Swin) -> ex. "vit_base_patch16_224"

Fonctions :
  build_model()            -> crée le modèle
  classifier_module()      -> renvoie la tête de classification
  set_backbone_trainable() -> gèle / dégèle le backbone
  param_groups()           -> 2 groupes de LR (backbone lent / tête rapide)
"""
import torch.nn as nn
from torchvision import models as tvmodels

from config import NUM_CLASSES, DEFAULT_MODEL


def build_model(name=DEFAULT_MODEL, num_classes=NUM_CLASSES, pretrained=True):
    """
    Construit un modèle de classification à `num_classes` sorties.

    Args:
        name: "resnet50" ou un nom de modèle `timm` (ViT, Swin...).
        num_classes: nombre de classes de sortie (11).
        pretrained: True pour les poids pré-entraînés (entraînement),
                    False pour recharger un checkpoint sauvegardé.
    """
    name = name.lower()

    if name == "resnet50":
        weights = tvmodels.ResNet50_Weights.DEFAULT if pretrained else None
        model = tvmodels.resnet50(weights=weights)
        # On remplace la tête ImageNet (1000 classes) par notre tête (11 classes).
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, num_classes),
        )
        return model

    # Tout autre modèle est créé via timm (ViT, Swin, ConvNeXt...).
    import timm
    return timm.create_model(name, pretrained=pretrained,
                             num_classes=num_classes, drop_rate=0.3)


def classifier_module(model):
    """Renvoie le module de tête de classification (compatible timm et torchvision)."""
    if hasattr(model, "get_classifier"):   # modèles timm
        return model.get_classifier()
    return model.fc                        # ResNet torchvision


def set_backbone_trainable(model, trainable):
    """Gèle (False) ou dégèle (True) le backbone, en gardant la tête modifiable."""
    head_ids = {id(p) for p in classifier_module(model).parameters()}
    for param in model.parameters():
        if id(param) not in head_ids:
            param.requires_grad = trainable


def param_groups(model, lr_backbone, lr_head):
    """Deux groupes de paramètres : backbone (LR faible) et tête (LR élevé)."""
    head_params = list(classifier_module(model).parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params,     "lr": lr_head},
    ]
