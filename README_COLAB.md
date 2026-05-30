# FER-CE — Lancement sur Google Colab

Ce README explique **comment exécuter tout le projet sur Google Colab**, du
zip local jusqu'au rapport final affiché dans le notebook.

> Pour exécuter en local (machine perso avec GPU type RTX 3090), voir
> **`README.md`**. Pour la documentation fonction par fonction, voir
> **`explanation.md`**.

---

## 1. Pré-requis

- Compte Google avec **Google Drive**
- Un runtime Colab avec GPU (T4 gratuit suffit pour le **preset Colab du sweep**)
- Ce projet cloné/téléchargé localement (le zip va y être préparé)

---

## 2. Préparer le zip à uploader sur Drive

Le notebook attend un fichier `deep_learning.zip` sur Drive qui contient :

- le **code** (`couche1/`, `couche2/`, `couche3/`, `config.py`, `main.py`,
  `run_local.py`, `sweep.py`)
- les **données packagées en sous-zips** (`Image.zip`, `Annotation.zip`,
  `EmoLabel.zip`) — le notebook les dézippe sur Colab à la 1re cellule
- les **docs** + **le notebook lui-même** (utile pour avoir tout au même
  endroit)
- `requirements.txt`

### Commandes à exécuter en local

```bash
# Depuis la racine du projet
cd /chemin/vers/deep_learning

# (Optionnel) Si Image.zip / Annotation.zip / EmoLabel.zip n'existent pas
# encore, les créer à partir des dossiers correspondants :
zip -r Image.zip      Image/
zip -r Annotation.zip Annotation/
zip -r EmoLabel.zip   EmoLabel/

# Supprime un éventuel zip précédent pour éviter de l'ajouter à la nouvelle
rm -f deep_learning.zip

# Zip principal — préserve la structure couche1/ couche2/ couche3/
zip -r deep_learning.zip \
    couche1 couche2 couche3 \
    config.py main.py run_local.py sweep.py \
    requirements.txt \
    README.md README_COLAB.md explanation.md documentation.md \
    demo_fer_ce.ipynb \
    Image.zip Annotation.zip EmoLabel.zip \
    -x '*__pycache__/*' '*.pyc' 'outputs/*' 'env/*' '.git/*'

ls -lh deep_learning.zip   # confirme la taille (~140-150 MB avec les données)
```

### Variante « léger » (sans les images)

Si tu veux uploader uniquement le code (et avoir les données déjà sur Drive ou
ailleurs) :

```bash
zip -r deep_learning_code_only.zip \
    couche1 couche2 couche3 \
    config.py main.py run_local.py sweep.py \
    requirements.txt \
    README.md README_COLAB.md explanation.md documentation.md \
    demo_fer_ce.ipynb \
    -x '*__pycache__/*' '*.pyc'
```

Puis adapte `DRIVE_ZIP_PATH` dans le notebook + dézippe les données depuis un
autre chemin.

---

## 3. Upload sur Google Drive

**Méthode simple — interface web :**

1. Ouvre `https://drive.google.com`
2. Glisse `deep_learning.zip` à la racine de **My Drive**
3. Attends la fin de l'upload (la barre en bas à droite)

> Le chemin Drive attendu par le notebook est `/content/drive/MyDrive/deep_learning.zip`.
> Si tu mets le fichier ailleurs (dans un sous-dossier), ajuste la variable
> `DRIVE_ZIP_PATH` dans la **cellule 1.3** du notebook.

**Méthode CLI (alternative) :** [`rclone`](https://rclone.org) ou `gdrive`
fonctionnent aussi — mais c'est plus simple de glisser le fichier dans Drive.

---

## 4. Ouvrir le notebook dans Colab

Deux options :

- **A.** Sur drive.google.com, clic-droit sur `demo_fer_ce.ipynb` → *Open with*
  → *Google Colaboratory* (ou *Connecter plus d'apps* la 1re fois).
- **B.** Sur colab.research.google.com → *File* → *Upload notebook* → choisir
  `demo_fer_ce.ipynb` depuis ton disque local.

**Important : active le GPU.**
*Runtime* → *Change runtime type* → *Hardware accelerator* = **GPU (T4)**.

---

## 5. Exécution — ce qui se passe cellule par cellule

| Section | Cellules | Ce qui s'exécute | Temps T4 |
|--------|----------|------------------|----------|
| 1 — Prépa env | 1.1 → 1.6 | GPU, Drive, dézip, dépendances, sanity-check imports | ~2-3 min |
| 2 — Audit Couche 1 | `data_exploration`, `check_dataset`, cache faces | distribution classes + intégrité + cache visages | ~1 min |
| 3 — Sweep multi-modèles | `sweep.list_presets()`, `[run_one(s) for s in SWEEP_COLAB]`, `write_report` | **6 runs** : resnet50_baseline / resnet50_no_mix / resnet50_focal / resnet50_no_ema / vit_baseline / swin_baseline | **~1-2 h** |
| 4 — Sélection + inférence | Lit la registry, charge le meilleur run, top-3 sur 6 images | rapide après le sweep | ~30 s |
| 5 — Vision-LLM | `explain.explain_images(...)` | génère 5 explications zero-shot avec Qwen2-VL | ~3-5 min (DL modèle 1re fois) |
| 6 — Grad-CAM + cohérence | `interpret.interpret_images(..., use_vlm=True)` | heatmaps + cohérence image/texte | ~2-3 min |
| 7 — **Rapport final** | Lit `sweep_report.md` et l'affiche inline en Markdown + matrice de confusion du meilleur run | **C'est la cellule à montrer en présentation** | quelques secondes |
| 8 — Sauvegarde Drive | (commentée) copie `outputs/` vers Drive | optionnel | ~1 min |

### Conseil — preset Colab vs local

Le notebook utilise **`SWEEP_COLAB`** (6 runs courts) par défaut, calibré
pour rester sous les 2 h de session T4 gratuite. Si tu as Colab Pro / une A100,
tu peux remplacer `runs = sweep.SWEEP_COLAB` par `runs = sweep.SWEEP_LOCAL`
(13 runs, ~3-5 h).

### Pour ne lancer qu'un sous-ensemble (test rapide)

Dans la cellule **3 — sweep run**, remplace :
```python
runs = sweep.SWEEP_COLAB
```
par :
```python
runs = [s for s in sweep.SWEEP_COLAB if s['run_name'] in {'resnet50_baseline', 'resnet50_focal'}]
```

---

## 6. La cellule « rapport final » (section 7)

Cette cellule affiche **inline dans le notebook** :

- un classement de tous les runs par accuracy test (tableau)
- les graphes `sweep_comparison.png` + `sweep_curves.png`
- une **analyse run par run** (sur-apprentissage, sous-entraînement,
  divergence — auto-générée à partir des métriques)
- une **synthèse** « pourquoi certains runs dégradent » (effet de désactiver
  MixUp/EMA, focal-loss, LR trop élevé, etc.)

C'est le contenu prévu pour la **présentation à la classe et au prof**.
Le rapport est aussi sauvegardé en `outputs/sweep_report.md` — copie-le sur
Drive avec la cellule **8** si tu veux le garder.

---

## 7. Sauvegarder les résultats

À la fin de la session, **Colab efface tout**. Pour conserver les modèles,
graphes et le rapport, décommente la **cellule 8 (sauvegarde Drive)** :

```python
OUT_DRIVE = "/content/drive/MyDrive/fer_ce_outputs"
import shutil, os
os.makedirs(OUT_DRIVE, exist_ok=True)
for f in os.listdir(config.OUTPUT_DIR):
    src = os.path.join(config.OUTPUT_DIR, f)
    if os.path.isfile(src):
        shutil.copy(src, OUT_DRIVE)
print('  [OK] outputs/ copié vers', OUT_DRIVE)
```

---

## 8. Pièges courants Colab

| Problème | Cause typique | Solution |
|----------|---------------|----------|
| `FileNotFoundError: deep_learning.zip` | Chemin Drive faux | Vérifier `DRIVE_ZIP_PATH` (cellule 1.3) |
| `RuntimeError: CUDA out of memory` | Batch trop gros | `config.BATCH_SIZE = 16` avant la cellule sweep |
| Le notebook se déconnecte | Inactivité ou 12 h max | Garde l'onglet ouvert, ou utilise Colab Pro |
| Vision-LLM trop lent / OOM | Modèle non quantizé | Modifie `load_vision_llm(quantize=True)` dans `couche2/explain.py` |
| Imports cassent (`ModuleNotFoundError: couche1`) | `os.chdir` pas fait | Re-lance la cellule 1.4 (dézip + cwd) |

---

## 9. Liens

- **`README.md`** — exécution locale (RTX 3090 / sweep complet)
- **`explanation.md`** — documentation fichier par fichier, fonction par fonction
- **`documentation.md`** — état d'avancement par couche
- **`demo_fer_ce.ipynb`** — ce notebook
