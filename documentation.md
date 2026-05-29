# Documentation du Projet — FER-CE (état d'avancement)

> Documentation **technique détaillée** (scripts, fonctions, concepts) :
> voir **`explanation.md`**.

## 1. Vue d'ensemble
Pipeline de reconnaissance d'**émotions faciales composées** (dataset RAF-DB,
**11 classes désignées par leur numéro 1..11**, sans nom d'émotion), organisé
en 3 couches.

## 2. Couche 1 — Préparation des données ✅
- Détection / recadrage du visage (`face_detection.py`).
- Augmentation et normalisation (`augmentation.py`, `normalization.py`).
- Vérification d'intégrité et exploration (`check_dataset.py`, `data_exploration.py`).
- Pipeline démontré dans `preprocessing.py`.
- **Sans sémantique** : chaque image est identifiée uniquement par son numéro
  de classe (1..11), dans l'esprit du *self-supervised learning*.

## 3. Couche 2 — Entraînement & explication ✅
- **Données** : `dataset.py` — cache des visages + augmentation à la volée.
- **Modèles** : `model.py` — ResNet-50, ViT, Swin (constructeur unifié).
- **Briques d'entraînement** : `training_utils.py` — MixUp, CutMix, Focal-Loss,
  EMA, TTA.
- **Entraînement** : `train.py` — 2 phases (warmup → fine-tuning),
  `WeightedRandomSampler`, label smoothing, cosine LR, mixed precision,
  MixUp/CutMix, EMA. Multi-modèles via `--model`.
- **Évaluation** : `evaluate.py` — accuracy, macro-F1, matrice de confusion,
  accuracy par classe (train + test), TTA, EMA.
- **Vision-LLM** : `vision_llm.py` + `explain.py` — Qwen2-VL en zero-shot
  génère une explication textuelle des indices faciaux.

> **Important.** L'ancien run ResNet-50 sans MixUp/EMA atteignait ~53.7 %.
> Le nouveau pipeline (MixUp + CutMix + EMA + TTA) doit dépasser ce score
> de plusieurs points sur le test set — à confirmer après réentraînement sur
> Colab (`demo_fer_ce.ipynb`).

## 4. Couche 3 — Interprétation ✅
- `interpret.py` — Grad-CAM (zones du visage activées), découpage en 3 bandes
  (≈ Action Units), et vérification de la **cohérence** entre la zone activée
  et l'explication du Vision-LLM (*Causal Emotion Grounding*).

## 5. Exécution
- **Google Colab (recommandé)** : ouvrir `demo_fer_ce.ipynb`. Le notebook
  monte Drive, dézippe, installe, entraîne plusieurs modèles, évalue/compare,
  exécute le Vision-LLM et la Couche 3 d'un bout à l'autre.
- **Local — étapes manuelles** : `python main.py --explore --check` puis
  `python train.py`, `evaluate.py`, `interpret.py`.
- **Local — tout-en-un + registry** : `python run_local.py --model resnet50`
  fait `check → train → evaluate → register` et enregistre le modèle (poids,
  hyper-paramètres, métriques, date, environnement) dans
  `outputs/registry.json`. Voir `python run_local.py --list` pour la liste
  des modèles enregistrés et `--load NAME` pour en recharger un.

## 6. Pistes restantes (optionnel)
- Vraie pré-formation auto-supervisée (SimCLR / MAE) sur les visages avant la
  tête supervisée — gain potentiel sur les classes rares.
- Attention rollout pour ViT/Swin (équivalent du Grad-CAM CNN).
- Interface Streamlit pour démo interactive.
