# FER-CE — Lancement local

Pipeline d'apprentissage profond pour classer des **expressions faciales
composées** (RAF-DB, **11 classes numérotées 1..11**) et générer des
**explications textuelles** via un Vision-LLM zero-shot.

> Ce README couvre uniquement **comment faire tourner le projet sur ta
> machine locale**.
> Pour la **documentation complète** (chaque fichier, chaque fonction, les
> concepts, la config, l'architecture) → voir **`explanation.md`**.
> Pour l'**état d'avancement** par couche → voir **`documentation.md`**.

---

## 1. Pré-requis

- Python ≥ 3.9
- (Optionnel mais fortement conseillé) GPU CUDA — `torch.cuda.is_available()` doit renvoyer `True`
- Les 3 dossiers de données présents à la racine du projet :
  `Image/`, `Annotation/`, `EmoLabel/`
  (ou bien `deep_learning.zip` + les 3 sous-zips dézippés sur place).

---

## 2. Installation

```bash
git clone <repo>  &&  cd deep_learning
python -m venv env  &&  source env/bin/activate
pip install -r requirements.txt
```

> Le Vision-LLM (`transformers` + `accelerate` + `bitsandbytes` + Qwen2-VL
> ~4.5 Go) est lourd : en local on **ne l'installe pas** ; il ne sert que
> sur Colab. Les couches 1 et 2 (training, évaluation, Grad-CAM) tournent
> sans lui.

---

## 3. Trois façons de lancer

### A. Pipeline tout-en-un avec enregistrement *(recommandé)*

```bash
python run_local.py --model resnet50            # check → train → evaluate → register
python run_local.py --all-models                # entraîne tous les modèles de config.MODELS
python run_local.py --list                      # liste les modèles enregistrés
python run_local.py --load resnet50             # recharge un modèle enregistré
python run_local.py --skip-train --model resnet50   # juste évaluer + ré-enregistrer
```

Chaque entraînement est consigné dans **`outputs/registry.json`** : chemins
des poids (raw + EMA), artéfacts (history, courbes, matrice de confusion),
hyper-paramètres, environnement (torch/CUDA), date, métriques. Utilisable
depuis n'importe quel script :

```python
from run_local import load_registered_model, list_models
model, info = load_registered_model("resnet50")   # EMA si dispo
```

### B. Étapes individuelles via l'orchestrateur

```bash
python main.py --explore --check                # Couche 1
python main.py --train --evaluate --model resnet50   # Couche 2
python main.py --interpret --model resnet50     # Couche 3
python main.py --all                            # Couches 1+2 de base
python main.py --list                           # liste toutes les étapes
```

### C. Chaque script à la main

```bash
python data_exploration.py                      # distribution des classes
python check_dataset.py                         # intégrité du dataset
python dataset.py                               # construit le cache des visages (1×)
python train.py --model resnet50                # entraînement
python evaluate.py --model resnet50             # évaluation
python main_test_model.py --model resnet50      # inférence interactive (1 image)
python interpret.py --model resnet50            # Couche 3 — Grad-CAM
```

---

## 4. Sorties générées (`outputs/`)

| Fichier | Description |
|---------|-------------|
| `best_<model>.pth` / `ema_<model>.pth` | Poids du meilleur modèle (raw + EMA) |
| `history_<model>.json` · `curves_<model>.png` | Historique et courbes loss/acc/F1 |
| `metrics_<model>.json` · `report_<model>.txt` | Métriques finales + rapport sklearn |
| `eval_<model>_overall.png` · `per_class.png` · `confusion.png` | Graphes d'évaluation |
| `faces_train.npz` / `faces_test.npz` | Cache des visages recadrés |
| `class_distribution.png` · `sample_images.png` | Audit Couche 1 |
| `interpret_<i>.png` · `interpretation.json` | Couche 3 — Grad-CAM |
| `registry.json` | Manifeste des modèles enregistrés via `run_local.py` |

---

## 5. Réglages rapides

Tout passe par **`config.py`** — aucune valeur n'est codée en dur ailleurs.
Bascules les plus utiles :

```python
BATCH_SIZE      = 32        # 16 si GPU 4 Go
EPOCHS          = 45
USE_MIXUP       = True      # forte régularisation
USE_CUTMIX      = True
USE_EMA         = True      # évaluation stabilisée
USE_FOCAL       = False     # True pour les classes rares
TTA             = True      # +1-2 % à l'éval
```

Pour la liste complète + l'explication de chaque variable, voir
`explanation.md` §4.

---

## 6. Et sur Colab ?

Pour le **pipeline complet avec Vision-LLM** (explication textuelle + cohérence
image/texte) → ouvrir **`demo_fer_ce.ipynb`** sur Google Colab. Le notebook
fait tout : mount Drive, dézip, install, train, evaluate, Vision-LLM,
Couche 3.

---

## 7. Liens

- **`explanation.md`** — documentation complète (fichier par fichier, fonction par fonction)
- **`documentation.md`** — état d'avancement par couche
- **`demo_fer_ce.ipynb`** — notebook Colab
