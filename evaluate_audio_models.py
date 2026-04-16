#!/usr/bin/env python3
"""
evaluate_audio_models.py
────────────────────────
Evaluate the pretrained audio deepfake detection models against a folder
of labelled audio files.

Usage:
    python evaluate_audio_models.py --real <dir_of_real_wavs> --fake <dir_of_fake_wavs>

Or against the ASVspoof 2019 eval set (if you have it locally):
    python evaluate_audio_models.py \
        --real  /path/to/ASVspoof2019_LA_eval/bonafide \
        --fake  /path/to/ASVspoof2019_LA_eval/spoof \
        --limit 200

Outputs:
  - Per-model accuracy, precision, recall, F1, EER
  - Ensemble accuracy
  - Confusion matrix
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

# ── add project root to path ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def collect_files(directory: str, extensions=(".wav", ".flac", ".mp3", ".ogg")) -> list[Path]:
    d = Path(directory)
    files = [f for f in sorted(d.iterdir()) if f.suffix.lower() in extensions]
    return files


def compute_eer(real_scores: list[float], fake_scores: list[float]) -> float:
    """Compute Equal Error Rate (EER). Lower = better."""
    all_scores = real_scores + fake_scores
    all_labels = [0] * len(real_scores) + [1] * len(fake_scores)
    thresholds = sorted(set(all_scores))
    best_eer = 1.0
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in all_scores]
        tp = sum(1 for p, l in zip(preds, all_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, all_labels) if p == 1 and l == 0)
        tn = sum(1 for p, l in zip(preds, all_labels) if p == 0 and l == 0)
        fn = sum(1 for p, l in zip(preds, all_labels) if p == 0 and l == 1)
        fpr = fp / (fp + tn + 1e-9)
        fnr = fn / (fn + tp + 1e-9)
        eer = (fpr + fnr) / 2
        if eer < best_eer:
            best_eer = eer
    return round(best_eer, 4)


def evaluate(real_dir: str, fake_dir: str, limit: int = 100):
    # Import here so the script can be run standalone
    from app.audio_model_loader import audio_models
    from app.audio_inference import predict_ensemble

    real_files = collect_files(real_dir)[:limit]
    fake_files = collect_files(fake_dir)[:limit]

    if not real_files:
        print(f"❌ No audio files found in: {real_dir}")
        return
    if not fake_files:
        print(f"❌ No audio files found in: {fake_dir}")
        return

    print(f"\n📁 Real: {len(real_files)} files   Fake: {len(fake_files)} files")
    print(f"🔬 Models: {list(audio_models.keys())}\n")

    # Per-model trackers
    per_model = {k: {"tp": 0, "fp": 0, "tn": 0, "fn": 0,
                     "real_scores": [], "fake_scores": []}
                 for k in audio_models.keys()}
    ens = {"tp": 0, "fp": 0, "tn": 0, "fn": 0,
           "real_scores": [], "fake_scores": []}

    def run_file(path: Path, true_label: int):
        """true_label: 0=real, 1=fake"""
        result = predict_ensemble(str(path), audio_models)
        ens_pred = 1 if result["prediction"] == "Fake" else 0
        ens_score = result["probabilities"]["fake"]

        if true_label == 0:
            ens["real_scores"].append(ens_score)
            if ens_pred == 0: ens["tn"] += 1
            else:             ens["fp"] += 1
        else:
            ens["fake_scores"].append(ens_score)
            if ens_pred == 1: ens["tp"] += 1
            else:             ens["fn"] += 1

        for k, det in result.get("model_details", {}).items():
            if k not in per_model or "error" in det:
                continue
            pred = 1 if det["prediction"] == "Fake" else 0
            score = det.get("fake_probability", 0.5)
            if true_label == 0:
                per_model[k]["real_scores"].append(score)
                if pred == 0: per_model[k]["tn"] += 1
                else:         per_model[k]["fp"] += 1
            else:
                per_model[k]["fake_scores"].append(score)
                if pred == 1: per_model[k]["tp"] += 1
                else:         per_model[k]["fn"] += 1

    total = len(real_files) + len(fake_files)
    for i, f in enumerate(real_files):
        print(f"  [{i+1:3d}/{total}] REAL  {f.name}", end="\r")
        run_file(f, 0)
    for i, f in enumerate(fake_files):
        print(f"  [{len(real_files)+i+1:3d}/{total}] FAKE  {f.name}", end="\r")
        run_file(f, 1)

    print("\n" + "=" * 60)

    def metrics(d: dict, label: str):
        tp, fp, tn, fn = d["tp"], d["fp"], d["tn"], d["fn"]
        acc  = (tp + tn) / (tp + fp + tn + fn + 1e-9)
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        eer  = compute_eer(d["real_scores"], d["fake_scores"])
        print(f"\n  [{label}]")
        print(f"    Accuracy:  {acc*100:.1f}%")
        print(f"    Precision: {prec*100:.1f}%")
        print(f"    Recall:    {rec*100:.1f}%")
        print(f"    F1:        {f1:.4f}")
        print(f"    EER:       {eer:.4f}  ({eer*100:.1f}%)")
        print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")

    for model_key, d in per_model.items():
        metrics(d, model_key)

    metrics(ens, "ENSEMBLE (majority vote)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate audio deepfake detection models")
    parser.add_argument("--real",  required=True,  help="Directory of REAL (genuine) audio files")
    parser.add_argument("--fake",  required=True,  help="Directory of FAKE (spoofed) audio files")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max files per class to evaluate (default: 100)")
    args = parser.parse_args()
    evaluate(args.real, args.fake, limit=args.limit)
