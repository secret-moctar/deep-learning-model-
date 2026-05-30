# FER-CE — Lancement local

Pipeline d'apprentissage profond pour classer des **expressions faciales
composées** (RAF-DB, **11 classes numérotées 1..11**) avec :

- 3 architectures interchangeables : **ResNet-50**, **ViT**, **Swin**
- pipeline d'entraînement « haute efficacité » (MixUp, CutMix, EMA, Focal-Loss, TTA)
- **sweep multi-modèles** qui entraîne plusieurs variantes d'hyperparamètres et
  produit un **rapport markdown auto-généré** (graphes + analyse) — idéal pour
  comparer et présenter
- **registry** JSON qui enregistre chaque run (poids, hyperparams, métriques, env)

> Ce README couvre uniquement **comment faire tourner le projet sur ta machine
> locale**. Pour Colab, voir **`README_COLAB.md`**. Pour la documentation
> complète fonction par fonction, voir **`explanation.md`**.

---

## 1. Pré-requis

- **Python ≥ 3.9**
- **GPU CUDA fortement recommandé** — sweep complet visé pour une **RTX 3090**
  (ou équivalent ≥ 16 GB VRAM). Sur GPU 6-8 GB tu peux quand même tourner avec
  `--preset colab` ou `BATCH_SIZE=16`.
- Les **3 dossiers de données** présents à la racine :
  `Image/`, `Annotation/`, `EmoLabel/`
  (ou les sous-zips `Image.zip`, `Annotation.zip`, `EmoLabel.zip` dans le dossier
  — ils seront extraits par les scripts).

---

## 2. Installation

```bash
git clone <repo>  &&  cd deep_learning
python3 -m venv env  &&  source env/bin/activate
pip install -r requirements.txt
```

> `transformers` + `accelerate` + `bitsandbytes` (Vision-LLM Qwen2-VL, ~4.5 Go)
> sont listés dans `requirements.txt` mais **ne sont nécessaires que pour
> l'explication textuelle** (`couche2/explain.py`) et la cohérence de la
> Couche 3. Les couches 1 et 2 (training, evaluation, Grad-CAM) tournent
> sans eux.

---

## 3. Structure des scripts

```
deep_learning/
├── config.py             ← Toute la config (chemins, hyperparams)
├── main.py               ← Orchestrateur d'étapes (--explore --check --train …)
├── run_local.py          ← Pipeline 1-modèle complet + registry JSON
├── sweep.py              ← Sweep multi-modèles + rapport auto
│
├── couche1/              ← Préparation des données
│   ├── face_detection.py
│   ├── augmentation.py
│   ├── normalization.py
│   ├── check_dataset.py
│   ├── data_exploration.py
│   └── preprocessing.py
│
├── couche2/              ← Entraînement, évaluation, Vision-LLM
│   ├── dataset.py
│   ├── model.py
│   ├── training_utils.py
│   ├── train.py
│   ├── evaluate.py
│   ├── vision_llm.py
│   ├── explain.py
│   └── test_model.py
│
└── couche3/              ← Interprétation (Grad-CAM)
    └── interpret.py
```

Tous les scripts peuvent être lancés de deux manières équivalentes :
`python -m couche2.train --model resnet50`  **OU**  `python couche2/train.py --model resnet50`.

---

## 4. Quatre façons de lancer

### A. Sweep multi-modèles avec rapport *(recommandé pour explorer)*

C'est le mode « j'essaie plusieurs hyperparamètres et je regarde ce qui marche ».

```bash
# Preset léger (5-7 runs, ~1-2 h sur GPU moyenne)
python sweep.py --preset colab

# Preset complet (12-15 runs, ~3-5 h sur RTX 3090)
python sweep.py --preset local

# Ne lancer que certains runs
python sweep.py --preset local --only resnet50_baseline,resnet50_focal

# Voir la liste sans rien lancer
python sweep.py --list

# Régénérer juste le rapport (si les metrics_*.json sont déjà là)
python sweep.py --preset local --report-only
```

Chaque run est :
1. **entraîné** avec ses overrides (`config.USE_MIXUP=False` etc.) ;
2. **évalué** sur train + test ;
3. **enregistré** dans `outputs/registry.json` sous son `run_name` ;
4. **comparé** aux autres dans le rapport final.

**Sorties du sweep :**

| Fichier | Description |
|---------|-------------|
| `outputs/sweep_report.md` | Rapport markdown complet : classement, analyse run par run, synthèse |
| `outputs/sweep_comparison.png` | Bar chart Acc + F1 par run, meilleur run en doré |
| `outputs/sweep_curves.png` | Courbes test_acc superposées par époque |
| `outputs/sweep_results.json` | Données brutes de tous les runs |

Le rapport contient **pourquoi chaque run dégrade** (sur-apprentissage, sous-
entraînement, divergence, etc.) — directement utilisable pour une présentation.

### B. Un seul modèle, pipeline complet + registry

```bash
python run_local.py --model resnet50               # check → train → evaluate → register
python run_local.py --model vit_base_patch16_224 --epochs 30
python run_local.py --all-models                   # entraîne config.MODELS un par un
python run_local.py --list                         # affiche les runs enregistrés
python run_local.py --load resnet50                # recharge un run enregistré
python run_local.py --skip-train --model resnet50  # juste évaluer + ré-enregistrer
```

La registry (`outputs/registry.json`) stocke pour chaque run :
chemins des poids (raw + EMA), artéfacts (history, courbes, matrice de confusion),
hyperparamètres, environnement (torch/CUDA), date, métriques. Utilisable depuis
n'importe quel script :

```python
from run_local import load_registered_model, list_models
model, info = load_registered_model("resnet50")    # EMA si dispo
```

### C. Étapes individuelles via l'orchestrateur

```bash
python main.py --explore --check                       # Couche 1
python main.py --train --evaluate --model resnet50     # Couche 2
python main.py --interpret --model resnet50            # Couche 3
python main.py --all                                   # Couches 1 + 2 de base
python main.py --list                                  # liste toutes les étapes
```

### D. Chaque script à la main

```bash
python -m couche1.data_exploration                   # distribution des classes
python -m couche1.check_dataset                      # intégrité du dataset
python -m couche2.dataset                            # construit le cache des visages (1×)
python -m couche2.train --model resnet50             # entraînement
python -m couche2.evaluate --model resnet50          # évaluation
python -m couche2.test_model --model resnet50        # inférence interactive (1 image)
python -m couche3.interpret --model resnet50         # Couche 3 — Grad-CAM
```

---

## 5. Sorties générées (`outputs/`)

| Fichier | Description |
|---------|-------------|
| `best_<run>.pth` · `ema_<run>.pth` | Poids du meilleur modèle (raw + EMA) |
| `history_<run>.json` · `curves_<run>.png` | Historique et courbes loss/acc/F1 |
| `metrics_<run>.json` · `report_<run>.txt` | Métriques finales + rapport sklearn |
| `eval_<run>_overall.png` · `_per_class.png` · `_confusion.png` | Graphes d'évaluation |
| `faces_train.npz` · `faces_test.npz` | Cache des visages recadrés |
| `class_distribution.png` · `sample_images.png` | Audit Couche 1 |
| `interpret_<i>.png` · `interpretation.json` | Couche 3 — Grad-CAM |
| `registry.json` | Manifeste des runs enregistrés |
| `sweep_report.md` · `sweep_comparison.png` · `sweep_curves.png` | Sortie du sweep |

`<run>` = `model_name` quand on entraîne un seul modèle, ou le `run_name` du
sweep (ex. `resnet50_focal`).

---

## 6. Réglages rapides

Tout passe par **`config.py`** — aucune valeur n'est codée en dur ailleurs.
Bascules les plus utiles :

```python
BATCH_SIZE      = 32        # 16 si GPU 4-6 Go, 64 sur RTX 3090
EPOCHS          = 45
USE_MIXUP       = True      # forte régularisation
USE_CUTMIX      = True
USE_EMA         = True      # évaluation stabilisée
USE_FOCAL       = False     # True pour les classes rares
TTA             = True      # +1-2 % à l'éval
LR_HEAD         = 3e-4
LR_BACKBONE     = 3e-5
```

Pour la liste complète + l'explication de chaque variable, voir
`explanation.md` § Config.

> Le sweep modifie ces valeurs **temporairement** entre les runs (et les
> restaure ensuite) — pas besoin d'éditer `config.py` à la main pour comparer
> des variantes.

---

## 7. Et sur Colab ?

Pour le pipeline complet avec Vision-LLM (explication textuelle + cohérence
image/texte) sans payer le GPU localement → voir **`README_COLAB.md`** :
commandes de zip à exécuter ici, upload Drive, ouverture du notebook
`demo_fer_ce.ipynb`.

---

## 8. Liens

- **`README_COLAB.md`** — exécuter le projet sur Google Colab (preset sweep léger)
- **`explanation.md`** — documentation complète fichier par fichier, fonction par fonction
- **`documentation.md`** — état d'avancement par couche
- **`demo_fer_ce.ipynb`** — notebook de bout en bout (à ouvrir sur Colab)
