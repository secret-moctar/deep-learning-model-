# Documentation complète — Projet FER-CE

> **But.** Référence **fichier par fichier** et **fonction par fonction** du
> projet : ce que chaque morceau fait, comment il s'utilise, comment l'étendre.
>
> Le `README.md` est court et ciblé "lancement local". Ce document-ci est le
> manuel exhaustif.

---

## Table des matières
1. [Vue d'ensemble](#1-vue-densemble)
2. [Architecture & flux de données](#2-architecture--flux-de-données)
3. [Carte des fichiers](#3-carte-des-fichiers)
4. [`config.py` — la configuration centrale](#4-configpy--la-configuration-centrale)
5. [Couche 1 — Préparation des données](#5-couche-1--préparation-des-données)
6. [Couche 2 — Modèles & entraînement](#6-couche-2--modèles--entraînement)
7. [Couche 3 — Interprétation (XAI)](#7-couche-3--interprétation-xai)
8. [Orchestration & lancement](#8-orchestration--lancement)
9. [Notebook Colab — `demo_fer_ce.ipynb`](#9-notebook-colab--demo_fer_ceipynb)
10. [Glossaire des concepts](#10-glossaire-des-concepts)
11. [Tableau "Je veux… → où aller"](#11-tableau-je-veux--où-aller)
12. [Résultats actuels & pistes d'amélioration](#12-résultats-actuels--pistes-damélioration)

---

## 1. Vue d'ensemble

**FER-CE** = *Facial Emotion Recognition — Compound Expressions*. Le projet
classe des **émotions composées** (mélange de 2 émotions de base) sur des
visages, à partir du dataset **RAF-DB**.

- **11 classes**, désignées **uniquement par leur numéro 1..11**. Aucun nom
  d'émotion n'est utilisé côté entraînement : on identifie les images par un
  numéro, dans l'esprit du *self-supervised learning* (les labels servent
  juste d'identifiants).
- Dataset : **3162 images d'entraînement**, **792 de test**, **déséquilibré**
  (de ~18 à ~700 images par classe).
- Deux familles de modèles complémentaires :
  - **Vision-Only** (ResNet-50 / ViT / Swin) → un **numéro de classe**.
  - **Vision-LLM** (Qwen2-VL en zero-shot) → une **explication textuelle**
    des indices faciaux.

Le projet est organisé en **3 couches** :

| Couche | Rôle | Scripts |
|--------|------|---------|
| **1** | Préparer / aligner les données | `face_detection.py`, `augmentation.py`, `normalization.py`, `check_dataset.py`, `data_exploration.py`, `preprocessing.py` |
| **2** | Entraîner et expliquer | `dataset.py`, `model.py`, `training_utils.py`, `train.py`, `evaluate.py`, `vision_llm.py`, `explain.py`, `main_test_model.py` |
| **3** | Interpréter (XAI) | `interpret.py` |
| **Orchestration** | Lanceurs | `main.py`, `run_local.py`, `demo_fer_ce.ipynb` |

---

## 2. Architecture & flux de données

```
                        ┌──────────────── COUCHE 1 ────────────────┐
  Image RAF-DB  ──▶  detect_and_crop (bbox)  ──▶  cache des visages
                        └───────────────────────────────────────────┘
                                          │
                        ┌──────────────── COUCHE 2 ────────────────┐
                        ▼                                          ▼
              FERDataset + augmentation              vision_llm (Qwen2-VL, zero-shot)
                        │                                          │
                        ▼                                          │
          build_model (ResNet / ViT / Swin)                        │
                        │                                          │
                  train.py (2 phases, MixUp/CutMix, EMA, AMP)      │
                        │                                          │
              outputs/best_<model>.pth + ema_<model>.pth           │
                        │                                          │
              ┌─────────┼──────────────┐                           │
              ▼         ▼              ▼                           ▼
        evaluate.py  main_test_model  explain.py ◀──────────────────┘
       (métriques)   (1 image)       (numéro + explication)
                        │
                        ▼  ┌──────────── COUCHE 3 ─────────────┐
                  interpret.py : Grad-CAM + cohérence image/texte
```

**Idée clé.** La Couche 1 transforme les images brutes en visages propres ;
la Couche 2 entraîne des modèles **purement numériques** et génère des
**explications textuelles** via un Vision-LLM zero-shot ; la Couche 3 vérifie
que ce que le modèle *regarde* (Grad-CAM) correspond à ce que le Vision-LLM
*dit*.

---

## 3. Carte des fichiers

```
deep_learning/
├── config.py              Configuration centrale (TOUT se règle ici)
│
│  ── COUCHE 1 ──
├── face_detection.py      Recadrage du visage (bbox RAF-DB / MTCNN)
├── augmentation.py        Augmentations "maison" (flip, rotation, éclairage, occlusion)
├── normalization.py       Resize + normalisation mean/std (multi-modèles)
├── check_dataset.py       Vérification d'intégrité du dataset
├── data_exploration.py    Distribution des classes + échantillons (par numéro)
├── preprocessing.py       Démonstration du pipeline Couche 1
│
│  ── COUCHE 2 ──
├── dataset.py             Dataset PyTorch (cache visages + augmentation à la volée)
├── model.py               Constructeur des modèles (ResNet / ViT / Swin)
├── training_utils.py      MixUp · CutMix · Focal-Loss · EMA · TTA
├── train.py               Entraînement (2 phases, MixUp/CutMix, EMA, cosine LR, AMP)
├── evaluate.py            Évaluation (accuracy, F1, confusion, EMA, TTA)
├── vision_llm.py          Vision-LLM Qwen2-VL (explication, zero-shot)
├── explain.py             Pipeline classifieur + Vision-LLM
├── main_test_model.py     Inférence interactive sur 1 image (top-3)
│
│  ── COUCHE 3 ──
├── interpret.py           Grad-CAM + cohérence image/texte
│
│  ── ORCHESTRATION ──
├── main.py                Orchestrateur ligne de commande (étapes individuelles)
├── run_local.py           Lanceur "tout-en-un" + registry des modèles
├── demo_fer_ce.ipynb      Notebook Colab (démo complète)
│
├── Image/ Annotation/ EmoLabel/   Données RAF-DB (présentes localement
│                                    ou dézippées sur Colab)
└── outputs/               Modèles, graphiques, métriques générés
```

**Tous les scripts sont importables** (noms de modules valides) — utilisables
dans un notebook : `import train`, `from evaluate import evaluate_model`, etc.
`vision_llm.py` et `explain.py` font un import *lazy* de `transformers` pour
fonctionner partout, même sans la librairie installée.

---

## 4. `config.py` — la configuration centrale

**Seul fichier à modifier** pour changer un réglage. Aucune valeur n'est
codée en dur dans les autres scripts.

### Sections

| Bloc | Variables | Rôle |
|------|-----------|------|
| **Chemins** | `BASE_DIR`, `IMAGE_DIR`, `LABEL_FILE`, `BBOX_DIR`, `OUTPUT_DIR` | Localisation des données et des sorties (relatif à `BASE_DIR` → marche local + Colab) |
| **Images** | `IMG_SIZE` (224), `IMG_CHS` (3), `LABEL_OFFSET` (1) | Taille d'entrée ; conversion labels 1-based ↔ 0-based |
| **Classes** | `NUM_CLASSES` (11), `CLASS_IDS` [1..11], `CLASS_NAMES` ["1".."11"] | 11 classes, désignées par numéro |
| **Modèles** | `MODELS`, `DEFAULT_MODEL` | Liste des architectures comparables, modèle par défaut (`resnet50`) |
| **Entraînement** | `BATCH_SIZE` (32), `EPOCHS` (45), `WARMUP_EPOCHS` (3), `LR_HEAD`, `LR_BACKBONE`, `WEIGHT_DECAY`, `LABEL_SMOOTHING`, `SEED` | Hyper-paramètres standard |
| **Haute efficacité** | `USE_MIXUP`, `MIXUP_ALPHA`, `USE_CUTMIX`, `CUTMIX_ALPHA`, `USE_EMA`, `EMA_DECAY`, `USE_FOCAL`, `FOCAL_GAMMA`, `TTA` | Bascules MixUp / CutMix / EMA / Focal-Loss / Test-Time Augmentation |
| **Vision-LLM** | `VLM_MODEL_ID`, `VLM_MAX_NEW_TOKENS` | Identifiant HuggingFace et longueur de réponse |

### Fonctions

| Fonction | Signature | Rôle |
|----------|-----------|------|
| `class_label` | `class_label(index_0based)` | Convertit l'indice interne 0..10 en numéro affiché 1..11. |
| `model_path` | `model_path(name)` | Chemin du checkpoint principal : `outputs/best_<name>.pth`. |
| `ema_path`   | `ema_path(name)` | Chemin du checkpoint **EMA** : `outputs/ema_<name>.pth`. |

> **Règle d'or.** Un réglage = une variable dans `config.py`. Si tu codes une
> valeur en dur ailleurs, c'est qu'il faut la déplacer ici.

---

## 5. Couche 1 — Préparation des données

Transforme une **image brute** en **visage propre normalisé**, identifié
uniquement par son **numéro de classe**.

### 5.1 `face_detection.py`

Découpe le visage de l'image. Deux modes : `"bbox"` (annotations RAF-DB,
utilisé pour l'entraînement) et `"auto"` (MTCNN, utilisé pour des images
hors dataset).

| Fonction | Signature | Rôle |
|----------|-----------|------|
| `load_bbox`        | `load_bbox(filename)` | Lit `Annotation/boundingbox/<base>_boundingbox.txt` et renvoie `[x1, y1, x2, y2]` (ou `None`). |
| `_crop_with_bbox`  | `_crop_with_bbox(image_path, margin=0.1)` | Recadrage via la bbox RAF-DB + marge 10 %. |
| `_get_mtcnn`       | `_get_mtcnn()` | Charge MTCNN une seule fois (cache global). Affiche un avertissement si `facenet-pytorch` n'est pas installé. |
| `_crop_with_mtcnn` | `_crop_with_mtcnn(image_path, margin=0.1)` | Détection MTCNN du visage le plus grand, recadrage avec marge. |
| `detect_and_crop`  | `detect_and_crop(image_path, mode="bbox", margin=0.1)` | **Point d'entrée** : dispatche `bbox` ou `auto`. |
| `main`             | `main()` | CLI : `python face_detection.py <image> [--auto]` → sauve `outputs/face_cropped_<mode>.png`. |

### 5.2 `augmentation.py`

Augmentations aléatoires "maison" (démonstration / réutilisable). En Couche 2,
`dataset.py` utilise plutôt `torchvision.transforms` pour la performance.

| Fonction | Signature | Rôle |
|----------|-----------|------|
| `random_flip`       | `random_flip(img)` | Flip horizontal avec p=0.5. |
| `random_rotation`   | `random_rotation(img, max_angle=15)` | Rotation aléatoire entre ±max_angle (remplit gris). |
| `random_brightness` | `random_brightness(img, factor_range=(0.7, 1.3))` | Variation de luminosité. |
| `random_contrast`   | `random_contrast(img, factor_range=(0.7, 1.3))` | Variation de contraste. |
| `random_occlusion`  | `random_occlusion(img, max_patches=2, max_size_ratio=0.15)` | Patches noirs aléatoires (occlusions). |
| `augment_image`     | `augment_image(pil_image)` | **Chaîne complète** : flip → rotation → brightness → contrast → occlusion (p=0.3). |
| *script `__main__`* | — | CLI : `python augmentation.py <image>` → grille de démos dans `outputs/augmentation_demo.png`. |

### 5.3 `normalization.py`

Resize + normalisation mean/std selon le modèle cible.

| Fonction / variable | Signature | Rôle |
|---------------------|-----------|------|
| `NORM_PARAMS`         | dict | Paramètres par modèle : `resnet`/`vit` (ImageNet 224), `clip`, `simple` (juste `[0,1]`). |
| `normalize_image`     | `normalize_image(pil_image, model="resnet", size=None)` | Resize → `float32 [0,1]` → `(C,H,W)` → (mean, std). Retourne numpy. |
| `denormalize_image`   | `denormalize_image(tensor_chw, model="resnet")` | Inverse de `normalize_image` → `(H,W,C) uint8` pour affichage. |
| *script `__main__`*   | — | CLI : `python normalization.py <image> [--model X]` → preview dans `outputs/normalized_<model>.png`. |

### 5.4 `check_dataset.py`

Vérification d'intégrité avant l'entraînement.

| Fonction | Signature | Rôle |
|----------|-----------|------|
| `check_image`  | `check_image(image_path)` | Renvoie une liste de problèmes (corruption, mode ≠ RGB, taille < 28×28, conversion impossible). Liste vide = OK. |
| `check_bbox`   | `check_bbox(filename)` | Vérifie l'existence et la validité de la bbox associée. Renvoie une str d'erreur ou `None`. |
| `main`         | `main()` | Boucle sur le fichier de labels, agrège les stats (`ok`, `corrupted`, `bad_mode`, `too_small`, `no_bbox`, `bad_label`, `missing_file`), liste les fichiers orphelins, sauve `outputs/bad_images.txt`. |

### 5.5 `data_exploration.py`

Inspection visuelle de la distribution des classes.

| Fonction | Signature | Rôle |
|----------|-----------|------|
| `load_labels`         | `load_labels()` | Lit `EmoLabel/list_patition_label.txt`, sépare `train` / `test`, renvoie `(train_data, test_data)` avec labels 0-based. |
| `plot_distribution`   | `plot_distribution(train_data, test_data)` | Sauve `outputs/class_distribution.png` (histogrammes train + test, étiquettes = numéros). |
| `plot_samples`        | `plot_samples(train_data, n=3)` | Sauve `outputs/sample_images.png` (n exemples par classe ; les titres sont les numéros). |
| `main`                | `main()` | Affiche le résumé en console + génère les 2 graphes. |

### 5.6 `preprocessing.py`

Démonstration "bout en bout" du pipeline Couche 1.

| Fonction | Signature | Rôle |
|----------|-----------|------|
| `process_one_image`   | `process_one_image(image_path, augment=False, model="resnet")` | Pipeline complet : `detect_and_crop` → (optionnel) `augment_image` → `normalize_image`. |
| `load_all_images`     | `load_all_images(split="train", augment=False, model="resnet")` | Charge tout le split en mémoire (tenseurs). |
| `create_dataloaders`  | `create_dataloaders(batch_size, model="resnet")` | Renvoie `(train_loader, test_loader, class_weights, n_train, n_test)`. |
| `main`                | `main()` | Sanity-check : crée les loaders et affiche la shape d'un batch. |

> **Note.** En production, l'entraînement utilise `dataset.py` (cache .npz +
> augmentation à la volée) — beaucoup plus rapide. `preprocessing.py` reste
> la démo pédagogique du flux Couche 1.

---

## 6. Couche 2 — Modèles & entraînement

### 6.1 `dataset.py` — données pour l'entraînement

Cache des visages recadrés + augmentation à la volée via `torchvision`.

| Fonction / classe | Signature | Rôle |
|-------------------|-----------|------|
| `IMAGENET_MEAN/STD`     | constantes | Stats de normalisation ImageNet utilisées par les backbones pré-entraînés. |
| `read_split`            | `read_split(split)` | Liste `[(filename, label_0based), …]` filtrée par `train`/`test`, vérifie que le fichier existe. |
| `_cache_path`           | `_cache_path(split)` | Renvoie `outputs/faces_<split>.npz`. |
| `build_face_cache`      | `build_face_cache(split)` | Recadre **une seule fois** tous les visages, stocke en `uint8 (N, 224, 224, 3)` + labels. |
| `load_face_cache`       | `load_face_cache(split, rebuild=False)` | Charge depuis le cache, ou le construit si absent / `rebuild=True`. |
| `TRAIN_TRANSFORM`       | `transforms.Compose` | `RandomResizedCrop` → `Flip` → `Rotation(15°)` → `ColorJitter` → `ToTensor` → `Normalize` → `RandomErasing(p=0.25)`. |
| `EVAL_TRANSFORM`        | `transforms.Compose` | Resize fixe + ToTensor + Normalize (déterministe). |
| `class FERDataset`      | — | Dataset PyTorch principal. |
| `FERDataset.__init__`   | `(split, train=False, rebuild=False)` | Charge le cache et choisit la transform. |
| `FERDataset.__len__`    | — | Nombre d'échantillons. |
| `FERDataset.__getitem__`| `(idx)` | Renvoie `(tensor, label)` — augmentation à la volée si `train=True`. |
| `FERDataset.class_counts` | — | `np.bincount` des labels. |
| `FERDataset.sample_weights` | — | Poids par image pour `WeightedRandomSampler` (équilibre les classes rares). |
| *script `__main__`* | — | CLI : `python dataset.py [--rebuild]` reconstruit le cache. |

### 6.2 `model.py` — architectures

Constructeur unique partagé par l'entraînement, l'évaluation, l'inférence,
l'interprétation → l'architecture est **identique** entre la sauvegarde et
le rechargement.

| Fonction | Signature | Rôle |
|----------|-----------|------|
| `build_model`            | `build_model(name=DEFAULT_MODEL, num_classes=NUM_CLASSES, pretrained=True)` | Crée un modèle. `resnet50` → torchvision avec tête `Dropout(0.5) + Linear`. Sinon → `timm.create_model(name, num_classes=…, drop_rate=0.3)` (ViT, Swin, ConvNeXt…). |
| `classifier_module`      | `classifier_module(model)` | Renvoie la tête (`model.get_classifier()` pour timm, `model.fc` pour torchvision). |
| `set_backbone_trainable` | `set_backbone_trainable(model, trainable)` | Gèle (False) ou dégèle (True) tout sauf la tête. |
| `param_groups`           | `param_groups(model, lr_backbone, lr_head)` | Renvoie 2 groupes pour AdamW : backbone (LR faible), tête (LR élevé). |

### 6.3 `training_utils.py` — briques "haute efficacité"

| Fonction / classe | Signature | Rôle |
|-------------------|-----------|------|
| `mixup_data`        | `mixup_data(x, y, alpha=0.2)` | Mélange `x' = lam·x_a + (1-lam)·x_b`. Renvoie `(x', y_a, y_b, lam)`. |
| `cutmix_data`       | `cutmix_data(x, y, alpha=1.0)` | Remplace un rectangle aléatoire par celui d'une autre image. `lam` = part conservée. |
| `mixup_criterion`   | `mixup_criterion(loss_fn, logits, y_a, y_b, lam)` | Perte combinée : `lam·L(y_a) + (1-lam)·L(y_b)`. |
| `class FocalLoss`   | `FocalLoss(gamma=2.0, label_smoothing=0.0)` | `−(1−p_t)^γ · log p_t` → pèse plus les exemples difficiles. Supporte label smoothing. |
| `class ModelEMA`    | `ModelEMA(model, decay=0.999)` | Copie figée du modèle, mise à jour avec `ema_w = decay·ema_w + (1-decay)·model_w`. |
| `ModelEMA.update`   | `.update(model)` | Met à jour `ema.model` (params + buffers BN). |
| `ModelEMA.state_dict` | — | Renvoie le state_dict du modèle EMA pour sauvegarde. |
| `tta_logits`        | `tta_logits(model, x)` | Moyenne des logits `(image, image flippée H)` — utilisé à l'éval. |

### 6.4 `train.py` — entraînement

| Fonction | Signature | Rôle |
|----------|-----------|------|
| `make_loaders`     | `make_loaders(num_workers=2)` | DataLoaders : train **équilibré** via `WeightedRandomSampler`, test séquentiel. |
| `train_one_epoch`  | `train_one_epoch(model, loader, loss_fn, optimizer, scaler, ema=None)` | Une époque. Pour chaque batch : choisit aléatoirement (1/3) **MixUp**, **CutMix** ou rien ; AMP via `scaler` ; mise à jour EMA après chaque step. Renvoie `(loss, acc, f1)`. |
| `eval_epoch`       | `eval_epoch(model, loader, loss_fn, use_tta=False)` | Évaluation `no_grad`. Renvoie `(loss, acc, f1)`. TTA optionnel via `tta_logits`. |
| `_plot_curves`     | `_plot_curves(history, model_name)` | Sauve `outputs/curves_<model>.png` : loss / acc / F1 train vs test (+ EMA si actif). |
| `train_model`      | `train_model(model_name=DEFAULT_MODEL, epochs=EPOCHS)` | **Fonction principale**. Construit modèle + EMA + scaler AMP. Choisit `FocalLoss` ou `CrossEntropyLoss(label_smoothing)`. Phase 1 (`< WARMUP_EPOCHS`) : backbone gelé, AdamW sur la tête. Phase 2 : tout dégelé, AdamW à 2 LR + cosine LR. Sauve `best_<model>.pth` et `ema_<model>.pth` à chaque amélioration du **macro-F1** (sur EMA si actif). Écrit `history_<model>.json`. Renvoie l'historique. |
| `main`             | `main()` | CLI : `--model`, `--epochs`. |

### 6.5 `evaluate.py` — évaluation

| Fonction | Signature | Rôle |
|----------|-----------|------|
| `load_trained_model` | `load_trained_model(model_name, prefer_ema=True)` | Reconstruit l'archi via `build_model(..., pretrained=False)` et charge **EMA** si présent, sinon le checkpoint brut. Affiche le tag `[EMA]`/`[raw]`. |
| `predict_split`      | `predict_split(model, split, use_tta=True)` | Inférence sans augmentation sur train ou test. Renvoie `(y_true, y_pred)`. |
| `per_class_accuracy` | `per_class_accuracy(y_true, y_pred)` | Accuracy (= recall) de chaque classe. |
| `_plot_overall`      | `_plot_overall(train_acc, test_acc, model_name)` | Barres train vs test → `eval_<model>_overall.png`. |
| `_plot_per_class`    | `_plot_per_class(train_pc, test_pc, model_name)` | Barres par classe → `eval_<model>_per_class.png`. |
| `_plot_confusion`    | `_plot_confusion(y_true, y_pred, model_name)` | Heatmap normalisée → `eval_<model>_confusion.png`. |
| `evaluate_model`     | `evaluate_model(model_name, prefer_ema=True, use_tta=True)` | **Fonction principale**. Lance les 3 graphes + `report_<model>.txt` + `metrics_<model>.json` (accuracy/F1/per-classe + flags `ema_used`/`tta`). |
| `main`               | `main()` | CLI : `--model`, `--raw`, `--no-tta`. |

### 6.6 `vision_llm.py` — Vision-LLM (Qwen2-VL)

Modèle ~4.5 Go chargé uniquement sur Colab. Aucun entraînement (zero-shot).

| Fonction / variable | Signature | Rôle |
|---------------------|-----------|------|
| `FACIAL_CUES`       | liste | Mots-clés "indices faciaux" partagés avec la Couche 3. |
| `load_vision_llm`   | `load_vision_llm(model_id=VLM_MODEL_ID, quantize=False)` | Charge Qwen2-VL-2B en fp16 (ou 4-bit avec `bitsandbytes`). Renvoie `(model, processor)`. |
| `_build_prompt`     | `_build_prompt(hint=None)` | Prompt zero-shot : demande une émotion composée + une explication, format fixe `Emotion: …\nExplanation: …`. Optionnellement un `hint` (ex. label du classifieur). |
| `_parse_answer`     | `_parse_answer(text)` | Sépare la réponse en `{emotion, explanation, raw}`. |
| `explain_emotion`   | `explain_emotion(model, processor, image, hint=None, max_new_tokens=…)` | **Fonction principale**. Construit le prompt, fait l'inférence, renvoie le dict structuré. |

### 6.7 `explain.py` — classifieur + Vision-LLM

Pipeline complet pour une image : classe (numéro) + explication textuelle.

| Fonction | Signature | Rôle |
|----------|-----------|------|
| `load_classifier`    | `load_classifier(model_name, prefer_ema=True)` | Charge le meilleur checkpoint (EMA prioritaire). Renvoie `(model, EVAL_TRANSFORM)` ou `(None, None)` si introuvable (le pipeline continue alors sans numéro de classe). |
| `classify`           | `classify(model, transform, face)` | Softmax → renvoie `(numéro_classe, confiance%)`. |
| `save_visualization` | `save_visualization(face, result, index)` | Visage + numéro classe + texte du Vision-LLM → `outputs/explain_<i>.png`. |
| `explain_images`     | `explain_images(image_paths, model_name=DEFAULT_MODEL, quantize=False)` | **Fonction principale**. Boucle : recadrage → classifieur (numéro) → `explain_emotion` (texte) → sauvegarde + JSON `outputs/explanations.json`. Import *lazy* de `vision_llm` (Colab only). |
| `main`               | `main()` | CLI : `--model`, `--n`. Échantillonne aléatoirement N images de test. |

### 6.8 `main_test_model.py` — inférence interactive

| Fonction | Signature | Rôle |
|----------|-----------|------|
| `load_classifier`  | `load_classifier(model_name=DEFAULT_MODEL, prefer_ema=True)` | Comme `evaluate.load_trained_model` mais sort proprement avec un message si rien n'est trouvé. |
| `predict`          | `predict(model, face)` | Renvoie `(classe, conf%, top3=[(classe, conf%)*3])`. |
| `get_true_class`   | `get_true_class(image_path)` | Cherche le label réel (1-based) dans le fichier, ou `None`. |
| `show_result`      | `show_result(image_path, face, pred_class, conf, top3, true_class, index, show)` | Visualisation 3 panneaux : original / visage / top-3 → `outputs/res_<i>_(ok|fail).png`. |
| `test_one_image`   | `test_one_image(image_path, model, mode="bbox", index=0, show=True)` | Glue : recadrage → prédiction → visu. |
| `main`             | `main()` | CLI interactive : 1 image précise, ou N images aléatoires. |

---

## 7. Couche 3 — Interprétation (XAI)

### 7.1 `interpret.py`

Relie ce que le **modèle regarde** (Grad-CAM) à ce que le **Vision-LLM dit**
(*Causal Emotion Grounding*).

| Fonction / classe | Signature | Rôle |
|-------------------|-----------|------|
| `REGIONS`            | liste 3 str | "Front / sourcils", "Yeux", "Bouche / joues". |
| `REGION_CUE_WORDS`   | dict | Mots-clés par bande horizontale (eyebrow / eye / mouth…). |
| `class GradCAM`      | — | Grad-CAM : hooks `forward`/`backward` sur une conv. |
| `GradCAM.__init__`   | `(model, target_layer)` | Enregistre les 2 hooks sur la couche cible. |
| `GradCAM._save_activation` | hook | Capture les activations forward. |
| `GradCAM._save_gradient`   | hook | Capture les gradients backward. |
| `GradCAM.generate`   | `generate(input_tensor, class_idx=None)` | Calcule la heatmap : `relu(Σ weight·activation)`, interpole en 224×224, normalise 0..1. Renvoie `(cam, class_idx)`. |
| `get_target_layer`   | `get_target_layer(model, model_name)` | Couche cible (ResNet-50 → `layer4`). Lève une erreur claire pour ViT/Swin (→ attention rollout). |
| `region_scores`      | `region_scores(cam)` | Score moyen des 3 bandes horizontales, somme = 1. |
| `cues_in_text`       | `cues_in_text(text)` | Ensemble des indices `{0, 1, 2}` cités dans le texte (basé sur `REGION_CUE_WORDS`). |
| `interpret_image`    | `interpret_image(model, gradcam, image_path, vlm=None)` | Recadrage → Grad-CAM → scores. Si `vlm` fourni : appelle `explain_emotion` et calcule la cohérence. Renvoie `(face, cam, result)`. |
| `save_visualization` | `save_visualization(face, cam, result, index)` | 3 panneaux : visage / heatmap / barres + caption → `outputs/interpret_<i>.png`. |
| `interpret_images`   | `interpret_images(image_paths, model_name=DEFAULT_MODEL, use_vlm=False, quantize=False, prefer_ema=True)` | **Fonction principale**. Charge le modèle (EMA prioritaire), instancie Grad-CAM, optionnellement charge le VLM, boucle sur les images, JSON `outputs/interpretation.json`. |
| `main`               | `main()` | CLI : `--model`, `--vlm`, `--n`. |

**Cohérence image/texte.** Si la zone activée par le CNN (ex. la bouche) est
citée dans l'explication du Vision-LLM (ex. « smiling mouth ») → explication
**cohérente**. Sinon → incohérente, ce qui signale un possible biais.

> **Limite connue.** Grad-CAM est implémenté pour ResNet-50. Pour ViT/Swin,
> l'équivalent est l'*attention rollout* — non implémenté (piste §12).

---

## 8. Orchestration & lancement

### 8.1 `main.py` — orchestrateur "étape par étape"

CLI minimaliste qui sélectionne et exécute des étapes individuelles.

| Élément | Rôle |
|---------|------|
| `STEPS` (liste) | `(flag, module, description, fait_partie_de_--all)` pour `explore`, `check`, `preprocess`, `train`, `evaluate`, `explain`, `interpret`. |
| `run_step(module_name, desc)` | Importe le module, appelle `main()`, mesure le temps, gère les erreurs. |
| `main()` | Parse les flags, exécute les étapes choisies, affiche un récap. |

Exemples : `python main.py --explore --check`, `python main.py --all`.

### 8.2 `run_local.py` — pipeline tout-en-un + registry des modèles

Lance `check → train → evaluate → register` et tient à jour un **registry**
(`outputs/registry.json`) qui liste tous les modèles entraînés avec leurs
métriques, leurs poids, leurs hyper-paramètres et leur environnement.

| Élément | Signature | Rôle |
|---------|-----------|------|
| `REGISTRY_PATH`        | constante | `outputs/registry.json`. |
| `_load_registry`       | `_load_registry()` | Lit le JSON ou renvoie `{"models": {}}`. |
| `_save_registry`       | `_save_registry(reg)` | Écrit le JSON (`indent=2`). |
| `list_models`          | `list_models()` | Affiche un tableau (nom, test acc, test F1, date). |
| `register_model`       | `register_model(model_name, metrics, epochs_run)` | Ajoute / met à jour l'entrée d'un modèle dans la registry. Contient : `weights {raw, ema}`, `artifacts {history, curves, metrics, report, overall, per_class, confusion}`, `hyperparams {epochs, batch, LR, mixup, ema, focal, tta, …}`, `environment {python, torch, cuda, device}`, `metrics`, `registered_at`. |
| `load_registered_model`| `load_registered_model(model_name, prefer_ema=True)` | Recharge l'architecture (`build_model`) + les meilleurs poids (EMA prioritaire). Renvoie `(model, info_dict)`. |
| `_print_header`        | `_print_header(title)` | Cadre console. |
| `run_pipeline`         | `run_pipeline(model_name=DEFAULT_MODEL, epochs=EPOCHS, skip_check=False, skip_train=False)` | Pipeline complet : `check_dataset.main()` → `train.train_model(...)` → `evaluate.evaluate_model(...)` → `register_model(...)`. |
| `main`                 | `main()` | CLI : `--model`, `--epochs`, `--all-models`, `--skip-check`, `--skip-train`, `--list`, `--load`. |

API Python :
```python
from run_local import (run_pipeline, register_model, list_models,
                       load_registered_model)
model, info = load_registered_model("resnet50")     # EMA si dispo
```

---

## 9. Notebook Colab — `demo_fer_ce.ipynb`

Notebook auto-portant, 8 sections, fait tourner le projet de bout en bout
sur Colab (GPU gratuit). Voir le README pour le déroulé d'usage.

| Section | Contenu |
|---------|---------|
| **1. Préparation** | Détection GPU, mount Drive, dézippage de `deep_learning.zip` + sous-zips `Image.zip`/`Annotation.zip`/`EmoLabel.zip`, install `transformers accelerate bitsandbytes timm seaborn pillow<12`, smoke-test des imports. |
| **2. Audit Couche 1** | `data_exploration.main()`, `check_dataset.main()`, build du cache `FERDataset`, visualisations. |
| **3. Entraînement** | Boucle sur `config.MODELS` (ResNet-50, ViT, Swin) avec `train.train_model(...)`. |
| **4. Évaluation & comparaison** | `evaluate.evaluate_model(...)` pour chaque modèle + bar chart comparatif. |
| **5. Inférence individuelle** | `main_test_model.test_one_image(...)` sur quelques images aléatoires, top-3. |
| **6. Vision-LLM** | `explain.explain_images(...)` (Qwen2-VL zero-shot). |
| **7. Couche 3** | `interpret.interpret_images(..., use_vlm=True)` (Grad-CAM + cohérence). |
| **8. Conclusion** | Résumé + sauvegarde optionnelle vers Drive. |

---

## 10. Glossaire des concepts

| Concept | Explication courte |
|---------|--------------------|
| **Transfer learning** | Réutiliser un réseau pré-entraîné (millions d'images) plutôt que partir de zéro — indispensable avec 3162 images. |
| **Fine-tuning en 2 phases** | Phase 1 : backbone gelé, on entraîne la tête neuve (warmup). Phase 2 : on dégèle tout. Évite que les gradients aléatoires de la tête abîment le backbone. |
| **Augmentation à la volée** | Transformations aléatoires ré-appliquées à chaque époque → le modèle ne voit jamais 2× la même image. |
| **WeightedRandomSampler** | Tire les classes rares plus souvent → batches équilibrés malgré le déséquilibre. |
| **Label smoothing** | La cible n'est pas « 100 % classe X » mais « 90 % X » → moins de sur-confiance. |
| **MixUp** | Mélange linéaire de 2 images et de leurs labels → forte régularisation. |
| **CutMix** | Variante : un rectangle d'une image est collé dans une autre, les labels suivent la surface remplacée. |
| **Focal-Loss** | CrossEntropy pondérée pour mieux apprendre les classes rares. |
| **EMA (poids)** | Moyenne mobile exponentielle des poids → évaluation plus stable. |
| **TTA** | Test-Time Augmentation : on moyenne les logits sur l'image et son flip horizontal → quelques % de mieux pour 0 entraînement. |
| **Cosine LR** | Le learning rate décroît en courbe cosinus → entraînement stable. |
| **Mixed precision (AMP)** | Calculs en 16 bits → plus rapide, tient dans ~4 Go de VRAM. |
| **Accuracy** | % de prédictions correctes (trompeuse si classes déséquilibrées). |
| **Macro-F1** | Moyenne du F1 de **chaque classe** comptée également → métrique honnête ici. |
| **Matrice de confusion** | Quelle classe est confondue avec quelle autre. |
| **Grad-CAM** | Heatmap des zones d'image qui déclenchent une prédiction (XAI). |
| **Vision-LLM** | Modèle qui comprend image **et** texte → peut *expliquer*, pas seulement classer. |
| **Zero-shot** | Utiliser un modèle sans l'entraîner sur la tâche. |
| **Prompt-engineering visuel** | Rédiger une consigne précise pour guider le Vision-LLM vers des indices faciaux concrets. |
| **Self-supervised learning** | Apprendre sans étiquettes sémantiques. Ici, les classes sont des **numéros** : la sémantique n'est utilisée qu'à l'affichage et par le Vision-LLM zero-shot. |
| **Causal Emotion Grounding** | Vérifier que la zone activée par le CNN est bien celle citée par le Vision-LLM. |
| **Registry de modèles** | Manifeste JSON listant les modèles entraînés (poids, métriques, hyper-paramètres, environnement). Cf. `run_local.py`. |

---

## 11. Tableau "Je veux… → où aller"

| Je veux... | Comment faire |
|------------|---------------|
| Changer le nombre d'époques | `EPOCHS` dans `config.py` (ou `--epochs` en CLI) |
| Changer la taille de batch | `BATCH_SIZE` dans `config.py` |
| Activer / désactiver MixUp | `USE_MIXUP = True/False` |
| Activer / désactiver CutMix | `USE_CUTMIX = True/False` |
| Activer / désactiver l'EMA | `USE_EMA = True/False` |
| Passer à Focal-Loss | `USE_FOCAL = True`, `FOCAL_GAMMA = 2.0` |
| Désactiver le TTA | `TTA = False` (ou `--no-tta` côté CLI eval) |
| Ajouter une architecture | Compléter `MODELS` (n'importe quel nom `timm`) |
| Entraîner un autre modèle | `python train.py --model vit_base_patch16_224` |
| Tout-en-un + registry | `python run_local.py --model resnet50` |
| Lister les modèles enregistrés | `python run_local.py --list` |
| Recharger un modèle enregistré | `python run_local.py --load resnet50` |
| Changer le Vision-LLM | `VLM_MODEL_ID` dans `config.py` |
| Réduire le sur-apprentissage | Augmenter `WEIGHT_DECAY` ; renforcer `TRAIN_TRANSFORM` ; activer Focal-Loss |
| Reconstruire le cache des visages | `python dataset.py --rebuild` |
| Changer la tête d'un modèle | Éditer `build_model()` dans `model.py` |
| Quel script fait quoi | Cf. §3 (Carte) + §5–7 (référence) |

> **Règle d'or :** un réglage = une variable dans `config.py`. Ne code jamais
> une valeur en dur dans un script.

---

## 12. Résultats actuels & pistes d'amélioration

### Baseline (avant)
ResNet-50 entraîné 45 époques **sans** MixUp / CutMix / EMA / TTA :

| Métrique | Train | Test |
|----------|-------|------|
| Accuracy | 96.2 % | **53.7 %** |
| Macro-F1 | 97.0 % | 38.9 % |

L'écart train/test révèle du **sur-apprentissage** (peu de données pour
11 classes fines, classes très déséquilibrées).

### Pipeline actuel
`train.py` ajoute, par rapport à la baseline :
- **MixUp + CutMix** (régularisation forte) ;
- **EMA** des poids (évaluation stabilisée) ;
- **Focal-Loss** disponible (`USE_FOCAL=True`) pour les classes rares ;
- **TTA** (image + flip horizontal) à l'évaluation.

Ces 4 leviers doivent faire gagner plusieurs points sur le test. Le score
final dépend du GPU et de la longueur du run (45 époques par défaut) — voir
`metrics_<model>.json` après l'entraînement Colab, ou `registry.json` si le
modèle a été entraîné via `run_local.py`.

### Pistes d'amélioration restantes
- **Vraie pré-formation auto-supervisée** (SimCLR / MAE) sur les visages
  RAF-DB avant la tête supervisée — gain potentiel sur les classes rares.
- **Attention rollout** pour ViT / Swin (équivalent du Grad-CAM côté CNN).
- **Auto-distillation** : entraîner un grand modèle (ViT/Swin) puis distiller
  vers ResNet-50.
- **Interface Streamlit** pour démo interactive.

> **Note — niveau de supervision.** Les *backbones* (ResNet/ViT/Swin)
> proviennent d'un **pré-entraînement à grande échelle** ; le Vision-LLM est
> en **zero-shot**. La tête de classification, elle, est entraînée de façon
> supervisée — mais **uniquement sur des numéros de classe** (1..11) ; aucun
> nom d'émotion n'est utilisé. C'est ce que résume "*self-supervised
> learning*" dans l'esprit du projet : pas de sémantique côté entraînement.
