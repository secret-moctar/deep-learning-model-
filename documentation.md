# Documentation du Projet — FER-CE (état d'avancement)

> Documentation **technique détaillée** (scripts, fonctions, concepts) :
> voir **`explanation.md`**.
> Pour exécuter en local : **`README.md`**.  Pour Colab : **`README_COLAB.md`**.

## 1. Vue d'ensemble
Pipeline de reconnaissance d'**émotions faciales composées** (dataset RAF-DB,
**11 classes désignées par leur numéro 1..11**, sans nom d'émotion), organisé
en **3 couches** :

```
couche1/   préparation des données (recadrage, augmentation, normalisation, audit)
couche2/   entraînement, évaluation, explication Vision-LLM
couche3/   interprétation Grad-CAM
```

Orchestrateurs à la racine : `main.py`, `run_local.py`, `sweep.py`, `demo_fer_ce.ipynb`.

## 2. Couche 1 — Préparation des données ✅
- Détection / recadrage du visage (`couche1/face_detection.py`).
- Augmentation et normalisation (`couche1/augmentation.py`, `couche1/normalization.py`).
- Vérification d'intégrité et exploration (`couche1/check_dataset.py`, `couche1/data_exploration.py`).
- Pipeline démontré dans `couche1/preprocessing.py`.
- **Sans sémantique** : chaque image est identifiée uniquement par son numéro
  de classe (1..11), dans l'esprit du *self-supervised learning*.

## 3. Couche 2 — Entraînement & explication ✅
- **Données** : `couche2/dataset.py` — cache des visages + augmentation à la volée.
- **Modèles** : `couche2/model.py` — ResNet-50, ViT, Swin (constructeur unifié).
- **Briques d'entraînement** : `couche2/training_utils.py` — MixUp, CutMix,
  Focal-Loss, EMA, TTA.
- **Entraînement** : `couche2/train.py` — 2 phases (warmup → fine-tuning),
  `WeightedRandomSampler`, label smoothing, cosine LR, mixed precision,
  MixUp/CutMix, EMA. Supporte un `run_name` distinct de l'arch (utilisé par `sweep.py`).
- **Évaluation** : `couche2/evaluate.py` — accuracy, macro-F1, matrice de
  confusion, accuracy par classe (train + test), TTA, EMA.
- **Vision-LLM** : `couche2/vision_llm.py` + `couche2/explain.py` — Qwen2-VL
  en zero-shot génère une explication textuelle des indices faciaux.
- **Inférence individuelle** : `couche2/test_model.py` — top-3 sur 1 image.

> **Important.** L'ancien run ResNet-50 sans MixUp/EMA atteignait ~53.7 %.
> Le nouveau pipeline (MixUp + CutMix + EMA + TTA) doit dépasser ce score
> de plusieurs points sur le test set — à confirmer en relançant le sweep
> (`python sweep.py --preset local` ou via le notebook).

## 4. Couche 3 — Interprétation ✅
- `couche3/interpret.py` — Grad-CAM (zones du visage activées), découpage en
  3 bandes (≈ Action Units), et vérification de la **cohérence** entre la
  zone activée et l'explication du Vision-LLM (*Causal Emotion Grounding*).

## 5. Multi-modèles & rapport présentable ✅
- `sweep.py` — **sweep multi-modèles** avec 2 presets :
  - `SWEEP_COLAB` (6 runs, ~1-2 h sur T4 gratuit)
  - `SWEEP_LOCAL` (13 runs, ~3-5 h sur RTX 3090)
- Chaque run varie un hyperparamètre (mixup, focal, EMA, batch_size, LR, …) ;
  les overrides sont appliqués à `config` puis restaurés.
- Sortie : `outputs/sweep_report.md` (markdown présentable),
  `sweep_comparison.png` (bar chart Acc/F1, meilleur run en doré),
  `sweep_curves.png` (courbes test_acc superposées),
  `sweep_results.json` (données brutes).
- Le rapport contient une **analyse run par run** (sur-/sous-apprentissage,
  divergence) et une **synthèse "pourquoi ça dégrade"** — directement
  utilisable pour la présentation à la classe.

## 6. Exécution
- **Google Colab (recommandé)** : ouvrir `demo_fer_ce.ipynb`. Le notebook
  monte Drive, dézippe, installe, lance `sweep.SWEEP_COLAB` (6 runs),
  exécute le Vision-LLM et la Couche 3, puis **affiche le rapport final
  inline** dans la dernière cellule. Voir `README_COLAB.md` pour le détail
  + les commandes de zip.
- **Local — étapes manuelles** : `python main.py --explore --check`
  puis `python -m couche2.train`, `python -m couche2.evaluate`,
  `python -m couche3.interpret`.
- **Local — tout-en-un + registry** : `python run_local.py --model resnet50`
  fait `check → train → evaluate → register` et enregistre le modèle (poids,
  hyper-paramètres, métriques, date, environnement) dans
  `outputs/registry.json`. Voir `python run_local.py --list` pour la liste
  des modèles enregistrés et `--load NAME` pour en recharger un.
- **Local — sweep multi-modèles** : `python sweep.py --preset local` ; ou
  `--preset colab` pour un sweep léger ; `--list` pour voir les presets sans
  rien lancer ; `--report-only` pour régénérer le rapport sans réentraîner.

## 7. Pistes restantes (optionnel)
- Vraie pré-formation auto-supervisée (SimCLR / MAE) sur les visages avant la
  tête supervisée — gain potentiel sur les classes rares.
- Attention rollout pour ViT/Swin (équivalent du Grad-CAM CNN).
- Interface Streamlit pour démo interactive.
- Sweep avancé : ajouter un mode `--n-trials` aléatoire (Optuna / random search).
