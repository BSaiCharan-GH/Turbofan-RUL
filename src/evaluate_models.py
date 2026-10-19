"""Evaluate subset-specific LSTM and Transformer models on NASA C-MAPSS test sets.

Usage:
    python -m src.evaluate_models --subset FD001
    python -m src.evaluate_models --subset all
"""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.model_lstm import RUL_LSTM
from src.model_transformer import RUL_Transformer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
WINDOW = 30
MAX_RUL = 125.0
FEATURES = [f"setting{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
COLUMNS = ["engine_id", "cycle"] + FEATURES


def artifact(subset, suffix, legacy):
    preferred = MODELS / f"{subset}_{suffix}"
    if preferred.exists():
        return preferred
    old = MODELS / legacy
    if subset == "FD001" and old.exists():
        return old
    raise FileNotFoundError(f"Missing model artifact: {preferred}. Train {subset} first.")


def load_test(subset):
    path = RAW / f"test_{subset}.txt"
    rul_path = RAW / f"RUL_{subset}.txt"
    if not path.exists() or not rul_path.exists():
        raise FileNotFoundError(f"Download NASA C-MAPSS test and RUL files into {RAW}: {path.name}, {rul_path.name}")
    frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    frame = frame.dropna(axis=1, how="all")
    if frame.shape[1] != 26:
        raise ValueError(f"{path.name}: expected 26 columns, found {frame.shape[1]}")
    frame.columns = COLUMNS
    frame["engine_id"] = frame["engine_id"].astype(int)
    truth = np.loadtxt(rul_path, dtype=np.float32).reshape(-1)
    ids = sorted(frame["engine_id"].unique())
    if len(ids) != len(truth):
        raise ValueError(f"{rul_path.name}: {len(truth)} RUL values for {len(ids)} engines")
    return frame, ids, truth


def make_windows(frame, ids, scaler):
    scaled = scaler.transform(frame[FEATURES].to_numpy(dtype=np.float32))
    frame = frame.copy()
    frame.loc[:, FEATURES] = scaled
    windows = []
    for engine_id in ids:
        seq = frame.loc[frame["engine_id"] == engine_id].sort_values("cycle")[FEATURES].to_numpy(dtype=np.float32)
        if len(seq) < WINDOW:
            seq = np.pad(seq, ((WINDOW - len(seq), 0), (0, 0)), mode="edge")
        windows.append(seq[-WINDOW:])
    return np.stack(windows)


def load_weights(model, path, device):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def nasa_score(predicted, actual):
    # NASA scoring convention: d = prediction - ground truth.
    d = np.asarray(predicted) - np.asarray(actual)
    return float(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0).sum())


def evaluate(subset, device):
    frame, ids, truth = load_test(subset)
    scaler = joblib.load(artifact(subset, "data_scaler.pkl", "data_scaler.pkl"))
    X = torch.from_numpy(make_windows(frame, ids, scaler)).float().to(device)
    outputs = {"engine_id": ids, "actual_rul": truth.astype(float)}
    metrics = {"subset": subset, "engines": len(ids)}

    definitions = (
        ("lstm", RUL_LSTM(input_size=24, hidden_size=64, num_layers=2, dropout=0.2), "lstm_weights.pth"),
        ("transformer", RUL_Transformer(input_size=24, d_model=64, nhead=4, num_layers=2, dropout=0.2), "transformer_weights.pth"),
    )
    for name, model, legacy in definitions:
        weights = artifact(subset, f"{name}_weights.pth", legacy)
        model = load_weights(model, weights, device)
        with torch.no_grad():
            predictions = model(X).reshape(-1).cpu().numpy() * MAX_RUL
        predictions = np.maximum(predictions, 0.0)
        outputs[f"{name}_predicted_rul"] = predictions
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean((predictions - truth) ** 2)))
        metrics[f"{name}_mae"] = float(np.mean(np.abs(predictions - truth)))
        metrics[f"{name}_nasa_score"] = nasa_score(predictions, truth)

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outputs).to_csv(RESULTS / f"{subset}_evaluation.csv", index=False)
    print(f"{subset}: LSTM RMSE={metrics['lstm_rmse']:.3f}, Transformer RMSE={metrics['transformer_rmse']:.3f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=[*SUBSETS, "all"], default="FD001")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(device_name)
    selected = SUBSETS if args.subset == "all" else (args.subset,)
    metrics = [evaluate(subset, device) for subset in selected]
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(RESULTS / "evaluation_summary.csv", index=False)
    (RESULTS / "evaluation_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved results to {RESULTS}")


if __name__ == "__main__":
    main()
