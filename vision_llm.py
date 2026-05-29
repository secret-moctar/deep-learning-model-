#!/usr/bin/env python3
"""
vision_llm.py — Vision-LLM pour l'explication des émotions (Couche 2)
=====================================================================

Utilise Qwen2-VL-2B-Instruct (un Large Vision-Language Model) en ZERO-SHOT —
AUCUN entraînement — pour, à partir d'un visage :

  * décrire l'état émotionnel composé (mélange de 2 émotions de base),
  * GÉNÉRER une explication textuelle des indices faciaux qui le justifient
    (sourcils, yeux, bouche, joues, tension musculaire...).

Le classifieur (ResNet/ViT/Swin) donne, lui, le NUMÉRO de classe (1..11).
Le Vision-LLM apporte la partie "raisonnement / explication".

------------------------------------------------------------------------------
EXÉCUTION SUR GOOGLE COLAB
------------------------------------------------------------------------------
Le modèle (~4.5 Go) n'est pas téléchargé en local. Sur Colab :

    !pip install -q -U transformers accelerate bitsandbytes pillow

    from vision_llm import load_vision_llm, explain_emotion
    model, processor = load_vision_llm()
    res = explain_emotion(model, processor, "visage.jpg")
    print(res["emotion"], "|", res["explanation"])
------------------------------------------------------------------------------
"""
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from config import VLM_MODEL_ID, VLM_MAX_NEW_TOKENS

# Indices faciaux que l'on demande au modèle de commenter (réutilisé en Couche 3).
FACIAL_CUES = ["eyebrow", "eye", "mouth", "lip", "cheek", "nose",
               "forehead", "jaw", "smile", "frown", "wrinkle"]


# ---------------------------------------------------------------------------
# Chargement du modèle
# ---------------------------------------------------------------------------
def load_vision_llm(model_id=VLM_MODEL_ID, quantize=False):
    """
    Charge le Vision-LLM et son processeur.

    Args:
        model_id: identifiant HuggingFace du modèle.
        quantize: True -> chargement 4-bit (~2 Go VRAM). False -> fp16 (GPU T4 OK).

    Returns:
        (model, processor)
    """
    kwargs = {"device_map": "auto"}
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
    else:
        kwargs["torch_dtype"] = torch.float16

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


# ---------------------------------------------------------------------------
# Prompt-engineering visuel
# ---------------------------------------------------------------------------
def _build_prompt(hint=None):
    """
    Construit le prompt. On guide explicitement le modèle ("prompt-engineering
    visuel") pour qu'il s'appuie sur des indices faciaux concrets et réponde
    dans un format fixe, facile à analyser ensuite.
    """
    hint_txt = f"\nHint: a classifier suggests this is «{hint}».\n" if hint else ""
    return (
        "You are an expert in facial expression analysis.\n"
        "Look carefully at the person's face in the image.\n"
        f"{hint_txt}"
        "Describe the COMPOUND emotional state (a blend of two basic emotions, "
        "e.g. 'happily surprised', 'sadly angry') and explain which facial cues "
        "support it (eyebrows, eyes, mouth, cheeks, muscle tension...).\n\n"
        "Answer in EXACTLY this format:\n"
        "Emotion: <short compound-emotion description>\n"
        "Explanation: <2-3 sentences describing the visible facial cues>"
    )


def _parse_answer(text):
    """Sépare l'émotion et l'explication de la réponse formatée du modèle."""
    emotion, explanation = None, text.strip()
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("emotion:"):
            emotion = line.split(":", 1)[1].strip()
        elif low.startswith("explanation:"):
            explanation = line.split(":", 1)[1].strip()
    return {"emotion": emotion, "explanation": explanation, "raw": text.strip()}


# ---------------------------------------------------------------------------
# Inférence
# ---------------------------------------------------------------------------
@torch.no_grad()
def explain_emotion(model, processor, image, hint=None,
                    max_new_tokens=VLM_MAX_NEW_TOKENS):
    """
    Décrit et explique l'émotion composée d'un visage (zero-shot).

    Args:
        model, processor: renvoyés par load_vision_llm().
        image: chemin d'image OU image PIL.
        hint: indice textuel optionnel (ex. label d'un autre modèle).

    Returns:
        dict {"emotion": str, "explanation": str, "raw": str}
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": _build_prompt(hint)},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=[image],
                       return_tensors="pt").to(model.device)

    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    return _parse_answer(processor.decode(new_tokens, skip_special_tokens=True))