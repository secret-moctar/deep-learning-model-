"""
config.py — Configuration centrale du projet FER-CE
====================================================

TOUT se règle ici. Aucun chemin, classe ou hyper-paramètre n'est codé en dur
ailleurs dans le code : chaque autre script lit ce module.

Sections :
    1. Chemins
    2. Images
    3. Classes
    4. Modèles
    5. Hyper-paramètres d'entraînement (+ Mixup / EMA / Focal-loss)
    6. Vision-LLM
"""
import os

# ===========================================================================
# 1. CHEMINS
# ---------------------------------------------------------------------------
# BASE_DIR = dossier où se trouve ce fichier ; tous les chemins sont
# relatifs à lui -> le projet fonctionne en local ET sur Google Colab.
# ===========================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR  = os.path.join(BASE_DIR, "Image", "original")                   # images RAF-DB
LABEL_FILE = os.path.join(BASE_DIR, "EmoLabel", "list_patition_label.txt") # numéros 1..11
BBOX_DIR   = os.path.join(BASE_DIR, "Annotation", "boundingbox")           # bounding boxes
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")                             # résultats
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===========================================================================
# 2. IMAGES
# ---------------------------------------------------------------------------
# IMG_SIZE     : ResNet / ViT / Swin attendent du 224x224.
# LABEL_OFFSET : le fichier de labels est 1-based ; PyTorch est 0-based.
#                On soustrait 1 à la lecture et on ré-ajoute 1 à l'affichage.
# ===========================================================================
IMG_SIZE     = 224
IMG_CHS      = 3
LABEL_OFFSET = 1

# ===========================================================================
# 3. CLASSES
# ---------------------------------------------------------------------------
# 11 expressions composées RAF-DB désignées UNIQUEMENT par leur NUMÉRO (1..11).
# Aucun nom d'émotion n'est utilisé dans le code : la Couche 1 traite les
# images comme des entités identifiées par un numéro, dans l'esprit du
# « self-supervised learning » (pas de sémantique liée aux labels).
#
#   CLASS_IDS   = [1..11]            -> numéros "humains" (affichage)
#   CLASS_NAMES = ["1", ..., "11"]   -> étiquettes des graphiques
# Indices internes du modèle : 0..10 (= CLASS_IDS - LABEL_OFFSET).
# ===========================================================================
NUM_CLASSES = 11
CLASS_IDS   = list(range(1, NUM_CLASSES + 1))
CLASS_NAMES = [str(i) for i in CLASS_IDS]


def class_label(index_0based):
    """Indice 0-based du modèle  ->  numéro de classe affichable (1-based)."""
    return index_0based + LABEL_OFFSET

# ===========================================================================
# 4. MODÈLES
# ---------------------------------------------------------------------------
# "resnet50" vient de torchvision ; les autres sont créés via `timm`.
# Pour ajouter une architecture : compléter `MODELS` avec n'importe quel
# identifiant timm valide (ConvNeXt, EfficientNet, etc.).
# ===========================================================================
MODELS = [
    "resnet50",                      # CNN (torchvision)
    "vit_base_patch16_224",          # Vision Transformer (timm)
    "swin_base_patch4_window7_224",  # Swin Transformer  (timm)
]
DEFAULT_MODEL = "resnet50"

# ===========================================================================
# 5. HYPER-PARAMÈTRES D'ENTRAÎNEMENT
# ---------------------------------------------------------------------------
# BATCH_SIZE / EPOCHS / WARMUP_EPOCHS / LR / WEIGHT_DECAY : standard.
# LABEL_SMOOTHING : régularisation (le modèle reste moins sur-confiant).
#
# --- Astuces "haute efficacité" ----------------------------------------------
# USE_MIXUP    : mélange 2 images et leurs labels (augmentation très efficace).
# MIXUP_ALPHA  : intensité du mixup ; 0.2 est un bon défaut.
# USE_CUTMIX   : variante de mixup qui colle un patch d'une image dans l'autre.
# CUTMIX_ALPHA : intensité du cutmix.
# USE_EMA      : moyenne mobile des poids du modèle  -> évaluation plus stable.
# EMA_DECAY    : facteur de moyenne (0.999 = lente, fortement stabilisé).
# USE_FOCAL    : passer à Focal-Loss (gamma=2)  -> aide sur les classes rares.
# TTA          : Test-Time Augmentation (eval avec flip horizontal moyenné).
# ===========================================================================
BATCH_SIZE      = 32        # 16 en local 4 Go ; 32 sur GPU Colab
EPOCHS          = 45
WARMUP_EPOCHS   = 3
LR_HEAD         = 3e-4
LR_BACKBONE     = 3e-5
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1
SEED            = 42

# Améliorations "haute efficacité" (activées par défaut)
USE_MIXUP   = True
MIXUP_ALPHA = 0.2
USE_CUTMIX  = True
CUTMIX_ALPHA = 1.0
USE_EMA     = True
EMA_DECAY   = 0.999
USE_FOCAL   = False         # bascule à True si on veut tester la focal-loss
FOCAL_GAMMA = 2.0
TTA         = True          # eval avec moyenne (image, image flippée)


def model_path(name=DEFAULT_MODEL):
    """Chemin du checkpoint d'un modèle  -> outputs/best_<name>.pth."""
    return os.path.join(OUTPUT_DIR, f"best_{name}.pth")


def ema_path(name=DEFAULT_MODEL):
    """Chemin du checkpoint EMA d'un modèle  -> outputs/ema_<name>.pth."""
    return os.path.join(OUTPUT_DIR, f"ema_{name}.pth")

# ===========================================================================
# 6. VISION-LLM (Couches 2 & 3 — à exécuter sur Google Colab)
# ---------------------------------------------------------------------------
# Qwen2-VL-2B-Instruct utilisé en ZERO-SHOT pour générer une explication
# textuelle des indices faciaux ; modèle (~4.5 Go) téléchargé sur Colab.
# ===========================================================================
VLM_MODEL_ID       = "Qwen/Qwen2-VL-2B-Instruct"
VLM_MAX_NEW_TOKENS = 256
