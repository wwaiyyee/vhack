# app/audio_model_loader.py
"""
Loads pretrained HuggingFace audio deepfake detection models.

Models:
  - Primary:   Gustking/wav2vec2-large-xlsr-deepfake-audio-classification
               XLS-R 300M fine-tuned on ASVspoof 2019 LA
               Accuracy: 92.86%  |  F1: 0.937  |  EER: 4.01%

  - Secondary: motheecreator/Deepfake-audio-detection
               Wav2Vec2-base fine-tuned on multi-source real/fake audio

Both expose:  (model, feature_extractor)  for chunk-based inference.

Dataset Provenance:
  ASVspoof 2019 Logical Access (LA) — gold-standard anti-spoofing benchmark.
  2580 genuine + 22 800 synthetic/voice-converted utterances, 109 speakers,
  19 different TTS/VC spoofing algorithms.
"""

import os
import torch
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model registry ────────────────────────────────────────────────────────────
# Each entry: (internal_key, hf_repo_id, display_name, role)
_AUDIO_MODEL_REGISTRY = [
    (
        "xlsr",
        "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification",
        "XLS-R Deepfake (ASVspoof2019)",
        "primary",
    ),
    (
        "wav2vec2",
        "motheecreator/Deepfake-audio-detection",
        "Wav2Vec2 Deepfake",
        "secondary",
    ),
]


def _load_hf_audio_model(key: str, repo_id: str, display_name: str, role: str):
    """Download (or load from HF cache) a pretrained audio classification model."""
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)
        model = AutoModelForAudioClassification.from_pretrained(repo_id)
        model.to(device).eval()
        print(f"  ✅ [{role.upper()}] Loaded: {display_name}")
        print(f"     Labels: {model.config.id2label}")
        return model, feature_extractor
    except Exception as e:
        print(f"  ⚠️  Could not load {display_name} ({repo_id}): {e}")
        return None, None


def load_all_audio_models() -> dict:
    """
    Load all registered pretrained audio deepfake detection models.

    Returns:
        dict[str, tuple[model, feature_extractor]]
        e.g. {"xlsr": (model, fe), "wav2vec2": (model, fe)}
    """
    print("🎙️  Loading pretrained audio deepfake detection models...")
    audio_models: dict = {}

    for key, repo_id, display_name, role in _AUDIO_MODEL_REGISTRY:
        model, fe = _load_hf_audio_model(key, repo_id, display_name, role)
        if model is not None and fe is not None:
            audio_models[key] = (model, fe)

    loaded = len(audio_models)
    total = len(_AUDIO_MODEL_REGISTRY)
    print(f"🎙️  Loaded {loaded}/{total} audio models")

    if loaded == 0:
        print("  ❌  No audio models available — /predict-audio will return 503.")

    return audio_models


# ── Load at import time (same pattern as model_loader.py) ─────────────────────
audio_models = load_all_audio_models()
