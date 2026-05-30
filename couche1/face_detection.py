import os
import sys
# Bootstrap : permet `python couche1/face_detection.py` ET `python -m couche1.face_detection`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from config import BBOX_DIR, OUTPUT_DIR


# ============================================================================
# MODE BBOX — Pour l'entraînement (RAF-DB)
# ============================================================================

def load_bbox(filename):
    """
    Charge la bounding box depuis les fichiers d'annotation RAF-DB.

    Args:
        filename: Nom du fichier image (ex: "train_0001.jpg")

    Returns:
        Liste [x1, y1, x2, y2] ou None si pas de bbox
    """
    base = os.path.basename(filename).replace(".jpg", "")
    bbox_file = os.path.join(BBOX_DIR, f"{base}_boundingbox.txt")

    if os.path.exists(bbox_file):
        with open(bbox_file, "r") as f:
            coords = f.readline().strip().split()
            if len(coords) >= 4:
                return [float(c) for c in coords[:4]]
    return None


def _crop_with_bbox(image_path, margin=0.1):
    """Recadrage avec les bounding boxes RAF-DB (entraînement)."""
    img = Image.open(image_path).convert("RGB")
    filename = os.path.basename(image_path)

    bbox = load_bbox(filename)
    if bbox is None:
        return img

    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    # Marge de 10% autour du visage
    x1 = max(0, x1 - w * margin)
    y1 = max(0, y1 - h * margin)
    x2 = min(img.width, x2 + w * margin)
    y2 = min(img.height, y2 + h * margin)

    return img.crop((x1, y1, x2, y2))


# ============================================================================
# MODE AUTO — Pour l'inférence (nouvelles images)
# ============================================================================

# Cache du détecteur MTCNN (créé une seule fois)
_mtcnn_detector = None


def _get_mtcnn():
    """Charge MTCNN une seule fois et le réutilise."""
    global _mtcnn_detector
    if _mtcnn_detector is None:
        try:
            from facenet_pytorch import MTCNN
            _mtcnn_detector = MTCNN(
                keep_all=False,      # Un seul visage (le plus grand)
                min_face_size=40,    # Taille minimum du visage
                thresholds=[0.6, 0.7, 0.7],  # Seuils de confiance
                post_process=False,
            )
            print("  [✓] MTCNN chargé")
        except ImportError:
            print("  ✗ facenet-pytorch non installé.")
            print("    Installez-le : pip install facenet-pytorch, 4GB gone ")
            return None
    return _mtcnn_detector


def _crop_with_mtcnn(image_path, margin=0.1):
    """Détection automatique du visage avec MTCNN (inférence)."""
    img = Image.open(image_path).convert("RGB")
    detector = _get_mtcnn()

    if detector is None:
        print("  ⚠ MTCNN non disponible, retour image originale")
        return img

    # Détecter le visage
    boxes, probs = detector.detect(img)

    if boxes is None or len(boxes) == 0:
        print(f"  ⚠ Aucun visage détecté dans {os.path.basename(image_path)}")
        return img

    # Prendre le premier visage (plus haute confiance)
    x1, y1, x2, y2 = boxes[0]
    w = x2 - x1
    h = y2 - y1

    # Marge autour du visage
    x1 = max(0, x1 - w * margin)
    y1 = max(0, y1 - h * margin)
    x2 = min(img.width, x2 + w * margin)
    y2 = min(img.height, y2 + h * margin)

    confidence = probs[0]
    return img.crop((x1, y1, x2, y2))


# ============================================================================
# FONCTION PRINCIPALE — Choisit le mode automatiquement
# ============================================================================

def detect_and_crop(image_path, mode="bbox", margin=0.1):
    """
    Détecte et recadre le visage dans une image.

    Args:
        image_path: Chemin vers l'image
        mode:
          "bbox" → utilise les annotations RAF-DB (entraînement)
          "auto" → utilise MTCNN pour détecter le visage (inférence)
        margin: Marge autour du visage (0.1 = 10%)

    Returns:
        Image PIL recadrée
    """
    if mode == "bbox":
        return _crop_with_bbox(image_path, margin)
    elif mode == "auto":
        return _crop_with_mtcnn(image_path, margin)
    else:
        raise ValueError(f"Mode inconnu: {mode}. Utilisez 'bbox' ou 'auto'.")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 face_detection.py <image_path>          # mode bbox (RAF-DB)")
        print("  python3 face_detection.py <image_path> --auto   # mode auto (MTCNN)")
        sys.exit(1)

    image_path = sys.argv[1]
    mode = "auto" if "--auto" in sys.argv else "bbox"

    if not os.path.exists(image_path):
        print(f"  ✗ Image non trouvée: {image_path}")
        sys.exit(1)

    print(f"  Image: {image_path}")
    print(f"  Mode: {mode}")

    # Charger et recadrer
    original = Image.open(image_path).convert("RGB")
    cropped = detect_and_crop(image_path, mode=mode)

    print(f"  Taille originale: {original.size}")
    if mode == "bbox":
        bbox = load_bbox(os.path.basename(image_path))
        print(f"  Bbox (annotation): {bbox}")
    else:
        print(f"  Détection: MTCNN")
    print(f"  Taille recadrée: {cropped.size}")

    # Sauvegarder
    out_path = os.path.join(OUTPUT_DIR, f"face_cropped_{mode}.png")
    cropped.save(out_path)
    print(f"  [✓] Sauvegardé: {out_path}")

# ============================================================================
# Usage en ligne de commande
# ============================================================================
if __name__ == "__main__":
    main()
