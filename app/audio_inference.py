# app/audio_inference.py
"""
Pretrained audio deepfake inference using HuggingFace wav2vec2/XLS-R models.

Each model tuple is (AutoModelForAudioClassification, AutoFeatureExtractor).
Audio is loaded ONCE and shared across all models for efficiency.
"""

import os
import torch
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
CLIP_DURATION = 4          # seconds per chunk (longer = more context for transformers)
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION  # 64 000 samples
MAX_CHUNKS = 8             # max chunks analyzed per model per file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Audio I/O helpers (unchanged from original) ───────────────────────────────

def _resample(audio: np.ndarray, src_sr: int) -> np.ndarray:
    """Linear-interpolation resample to 16 kHz."""
    if src_sr == SAMPLE_RATE:
        return audio
    new_len = int(round(len(audio) * SAMPLE_RATE / src_sr))
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio)


def _load_with_av(file_path: str) -> np.ndarray:
    """Decode any format soundfile can't handle (WebM, Opus, MP3…) using PyAV."""
    import av as _av
    samples: list[np.ndarray] = []
    with _av.open(file_path) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = _av.audio.resampler.AudioResampler(
            format="fltp", layout="mono", rate=SAMPLE_RATE
        )
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                arr = resampled.to_ndarray()   # shape (1, N) float32
                samples.append(arr[0])
        for resampled in resampler.resample(None):   # flush
            arr = resampled.to_ndarray()
            samples.append(arr[0])
    if not samples:
        return np.zeros(CLIP_LENGTH, dtype=np.float32)
    return np.concatenate(samples).astype(np.float32)


def _load_audio(file_path: str) -> np.ndarray:
    """Load audio → 16 kHz mono float32 numpy array.

    Tries soundfile first (WAV/FLAC/OGG); falls back to PyAV
    for formats soundfile cannot read (WebM/Opus, MP3, MP4…).
    """
    try:
        audio, sr = sf.read(file_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return _resample(audio.astype(np.float32), sr)
    except Exception:
        pass
    return _load_with_av(file_path)


# ── Label normalisation ───────────────────────────────────────────────────────

_FAKE_LABELS = {
    "fake", "spoof", "0", "1",  # "1" in some repos means Fake
    "deepfake", "synthetic", "generated",
}
_REAL_LABELS = {
    "real", "bonafide", "genuine", "human",
}


def _parse_id2label(id2label: dict) -> dict[int, str]:
    """
    Return a normalised {class_idx: "real"|"fake"} map.

    The label names used on HuggingFace vary (e.g. "FAKE"/"REAL",
    "spoof"/"bonafide", "0"/"1").  We map them all to canonical strings.
    """
    mapping: dict[int, str] = {}
    for idx_raw, label in id2label.items():
        idx = int(idx_raw)
        ll = label.lower().strip()
        if ll in _FAKE_LABELS:
            mapping[idx] = "fake"
        elif ll in _REAL_LABELS:
            mapping[idx] = "real"
        else:
            # Fallback: trust index 0 → real, 1 → fake (common convention)
            mapping[idx] = "real" if idx == 0 else "fake"
    return mapping


# ── Single-model, chunk-based inference ──────────────────────────────────────

def predict_single_model(
    audio: np.ndarray,
    model_key: str,
    model_tuple: tuple,
    chunk_size: int = CLIP_LENGTH,
    max_chunks: int = MAX_CHUNKS,
) -> tuple[int, float, list]:
    """
    Run a single HuggingFace audio-classification model on audio chunks.

    Args:
        audio:       Raw 16 kHz mono float32 waveform.
        model_key:   Internal key (for logging).
        model_tuple: (AutoModelForAudioClassification, AutoFeatureExtractor)
        chunk_size:  Samples per chunk.
        max_chunks:  Max number of evenly-spaced chunks to analyse.

    Returns:
        (final_pred, confidence, chunk_results)
        final_pred: 1 = Fake, 0 = Real
        confidence: float in [0, 1]
    """
    model, feature_extractor = model_tuple

    # normalise label mapping once
    label_map = _parse_id2label(model.config.id2label)

    # ── chunk selection ───────────────────────────────────────────────────────
    total_chunks = max(1, len(audio) // chunk_size)
    if total_chunks <= max_chunks:
        chunk_indices = list(range(total_chunks))
    else:
        chunk_indices = [int(i * total_chunks / max_chunks) for i in range(max_chunks)]

    chunk_results: list[dict] = []
    for ci in chunk_indices:
        start = ci * chunk_size
        chunk = audio[start : start + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

        # ── HuggingFace preprocessing ─────────────────────────────────────────
        inputs = feature_extractor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits           # shape: (1, num_classes)
            probs = torch.softmax(logits, dim=-1)[0]  # shape: (num_classes,)

        p_real = 0.0
        p_fake = 0.0
        for idx, label in label_map.items():
            p = float(probs[idx])
            if label == "real":
                p_real += p
            elif label == "fake":
                p_fake += p

        # re-normalise in case labels didn't cover all classes
        total_p = p_real + p_fake
        if total_p > 0:
            p_real /= total_p
            p_fake /= total_p
        else:
            p_real = p_fake = 0.5

        chunk_results.append(
            {
                "prediction": 1 if p_fake >= 0.5 else 0,   # 1=Fake, 0=Real
                "p_real": round(p_real, 4),
                "p_fake": round(p_fake, 4),
            }
        )

    if not chunk_results:
        return 0, 0.5, []

    fake_count = sum(1 for c in chunk_results if c["prediction"] == 1)
    fake_ratio = fake_count / len(chunk_results)
    avg_fake_prob = float(np.mean([c["p_fake"] for c in chunk_results]))

    final_pred = 1 if fake_ratio >= 0.5 else 0
    confidence = avg_fake_prob if final_pred == 1 else (1.0 - avg_fake_prob)

    return final_pred, float(confidence), chunk_results


# ── Ensemble ──────────────────────────────────────────────────────────────────

def predict_ensemble(file_path: str, audio_models: dict) -> dict:
    """
    Run ALL loaded audio models and aggregate with majority vote.

    audio_models: dict[str, tuple(model, feature_extractor)]

    Returns a dict compatible with the /predict-audio response schema.
    """
    audio = _load_audio(file_path)

    model_results: dict = {}
    ensemble_votes: list[int] = []

    for model_name, model_tuple in audio_models.items():
        try:
            pred, conf, chunks = predict_single_model(audio, model_name, model_tuple)
            fake_chunks = sum(1 for c in chunks if c["prediction"] == 1)

            model_results[model_name] = {
                "prediction": "Fake" if pred == 1 else "Real",
                "confidence": round(conf, 4),
                "fake_probability": round(
                    float(np.mean([c["p_fake"] for c in chunks])) if chunks else 0.5, 4
                ),
                "chunks_analyzed": len(chunks),
                "chunks_fake": fake_chunks,
                "chunks_real": len(chunks) - fake_chunks,
            }
            ensemble_votes.append(pred)

        except Exception as e:
            model_results[model_name] = {
                "prediction": "Error",
                "error": str(e),
            }

    # ── majority vote ─────────────────────────────────────────────────────────
    if ensemble_votes:
        fake_model_count = sum(ensemble_votes)
        total_models = len(ensemble_votes)
        ensemble_pred = 1 if fake_model_count > total_models / 2 else 0

        valid = [r for r in model_results.values() if "confidence" in r]
        avg_confidence = float(np.mean([r["confidence"] for r in valid])) if valid else 0.5
        avg_fake_prob = float(np.mean([r["fake_probability"] for r in valid])) if valid else 0.5
    else:
        ensemble_pred = 0
        avg_confidence = 0.5
        avg_fake_prob = 0.5
        fake_model_count = 0
        total_models = 0

    return {
        "prediction": "Fake" if ensemble_pred == 1 else "Real",
        "confidence": round(avg_confidence, 4),
        "probabilities": {
            "real": round(1.0 - avg_fake_prob, 4),
            "fake": round(avg_fake_prob, 4),
        },
        "ensemble": {
            "models_voted_fake": fake_model_count,
            "models_voted_real": total_models - fake_model_count,
            "total_models": total_models,
            "decision": "majority_vote",
        },
        "model_details": model_results,
    }
