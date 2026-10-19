"""
Dataset preparation for all four NASA C-MAPSS subsets.

Run one subset:
    python -m src.data_prep --subset FD001

Run all subsets:
    python -m src.data_prep --subset all

Expected raw files:
    data/raw/train_FD001.txt
    data/raw/train_FD002.txt
    data/raw/train_FD003.txt
    data/raw/train_FD004.txt

Outputs for each subset:
    data/processed/FD001/X_train.npy
    data/processed/FD001/y_train.npy
    models/FD001_data_scaler.pkl

Important:
- Fit each scaler on that subset's TRAIN file only.
- Training RUL labels are derived from each training engine's final observed cycle
  and clipped to 125 cycles, matching the existing project's target convention.
- Test sequences are not generated here because NASA's test RUL labels refer to
  each engine's final, partially observed test sequence. Evaluation handles those.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
WINDOW_SIZE = 30
RUL_CAP = 125.0

FEATURE_COLUMNS = (
    ["setting_1", "setting_2", "setting_3"]
    + [f"s_{i}" for i in range(1, 22)]
)
ALL_COLUMNS = ["unit_number", "time_cycles"] + FEATURE_COLUMNS


def load_cmapss_file(file_path: Path) -> pd.DataFrame:
    """Load an original whitespace-separated NASA C-MAPSS file."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}\n"
            "Download the NASA C-MAPSS dataset and place the matching "
            "train_FD00X.txt file inside data/raw/."
        )

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        engine="python",
    )
    # Some copies have trailing whitespace, which can create empty columns.
    df = df.dropna(axis=1, how="all")

    if df.shape[1] != len(ALL_COLUMNS):
        raise ValueError(
            f"{file_path.name}: expected {len(ALL_COLUMNS)} columns, "
            f"found {df.shape[1]}. Use the original NASA C-MAPSS text file."
        )

    df.columns = ALL_COLUMNS
    for column in ALL_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df.isna().any().any():
        raise ValueError(f"{file_path.name} contains missing or non-numeric values.")
    if df.empty:
        raise ValueError(f"{file_path.name} contains no records.")
    if (df["unit_number"] <= 0).any() or (df["time_cycles"] <= 0).any():
        raise ValueError("Engine IDs and cycle numbers must be positive.")

    df["unit_number"] = df["unit_number"].astype(int)
    df["time_cycles"] = df["time_cycles"].astype(int)
    df = df.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)

    # Every engine should have one record per cycle number.
    if df.duplicated(["unit_number", "time_cycles"]).any():
        raise ValueError(
            f"{file_path.name} has duplicate engine/cycle rows. "
            "Inspect the source file before training."
        )

    return df


def add_training_rul(df: pd.DataFrame, clip_rul: float = RUL_CAP) -> pd.DataFrame:
    """Add capped RUL targets to fully observed training trajectories."""
    result = df.copy()
    last_cycles = result.groupby("unit_number")["time_cycles"].max()
    result["max_cycle"] = result["unit_number"].map(last_cycles)
    result["RUL"] = (result["max_cycle"] - result["time_cycles"]).clip(
        upper=clip_rul
    )
    return result.drop(columns=["max_cycle"])


def fit_and_apply_scaler(
    train_df: pd.DataFrame,
    scaler_path: Path,
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Fit scaler only on this subset's training features and save it."""
    scaler = MinMaxScaler()
    scaled_df = train_df.copy()

    scaled_df.loc[:, FEATURE_COLUMNS] = scaler.fit_transform(
        scaled_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    )

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return scaled_df, scaler


def generate_sequences(
    df: pd.DataFrame,
    sequence_length: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create fixed-length training windows per engine.

    Engines shorter than sequence_length are skipped and reported. Each window's
    target is the RUL at the window's final cycle.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1.")

    sequences: list[np.ndarray] = []
    targets: list[float] = []
    short_engines: list[int] = []

    for engine_id, engine in df.groupby("unit_number", sort=True):
        engine = engine.sort_values("time_cycles")
        features = engine[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        rul_values = engine["RUL"].to_numpy(dtype=np.float32)

        if len(engine) < sequence_length:
            short_engines.append(int(engine_id))
            continue

        for end in range(sequence_length, len(engine) + 1):
            sequences.append(features[end - sequence_length : end])
            targets.append(float(rul_values[end - 1]))

    if not sequences:
        raise ValueError(
            f"No sequences were created. Need engines with at least "
            f"{sequence_length} cycles."
        )

    if short_engines:
        print(
            f"[WARN] Skipped {len(short_engines)} engines shorter than "
            f"{sequence_length} cycles: {short_engines[:10]}"
            + (" ..." if len(short_engines) > 10 else "")
        )

    X = np.stack(sequences).astype(np.float32)
    y = np.asarray(targets, dtype=np.float32)
    return X, y


def prepare_subset(subset: str, sequence_length: int = WINDOW_SIZE) -> dict:
    """Prepare one subset and save scaler plus processed training arrays."""
    subset = subset.upper()
    if subset not in SUBSETS:
        raise ValueError(f"Unsupported subset {subset}. Choose from {SUBSETS}.")

    train_path = RAW_DIR / f"train_{subset}.txt"
    output_dir = PROCESSED_DIR / subset
    scaler_path = MODEL_DIR / f"{subset}_data_scaler.pkl"

    print(f"\n{'=' * 64}")
    print(f"[INFO] Preparing NASA C-MAPSS {subset}")
    print(f"[INFO] Loading: {train_path.relative_to(ROOT)}")

    train_df = load_cmapss_file(train_path)
    train_df = add_training_rul(train_df)

    print(f"[INFO] Training records: {len(train_df):,}")
    print(f"[INFO] Training engines: {train_df['unit_number'].nunique():,}")
    print(f"[INFO] Fitting subset-specific scaler...")

    scaled_df, _ = fit_and_apply_scaler(train_df, scaler_path)
    X_train, y_train = generate_sequences(scaled_df, sequence_length)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train.npy", X_train)
    np.save(output_dir / "y_train.npy", y_train)

    metadata = {
        "subset": subset,
        "sequence_length": int(sequence_length),
        "feature_columns": FEATURE_COLUMNS,
        "rul_cap": float(RUL_CAP),
        "n_engines": int(train_df["unit_number"].nunique()),
        "n_records": int(len(train_df)),
        "n_sequences": int(len(X_train)),
        "n_features": int(X_train.shape[2]),
        "scaler_filename": scaler_path.name,
    }
    joblib.dump(metadata, output_dir / "metadata.pkl")

    print(f"[INFO] Saved scaler: {scaler_path.relative_to(ROOT)}")
    print(f"[INFO] Saved: {(output_dir / 'X_train.npy').relative_to(ROOT)}")
    print(f"[INFO] Saved: {(output_dir / 'y_train.npy').relative_to(ROOT)}")
    print(f"[DONE] {subset}: X={X_train.shape}, y={y_train.shape}")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare NASA C-MAPSS data for FD001-FD004."
    )
    parser.add_argument(
        "--subset",
        default="all",
        choices=["all", *SUBSETS],
        help="Subset to prepare (default: all).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=WINDOW_SIZE,
        help=f"Sequence length (default: {WINDOW_SIZE}).",
    )
    args = parser.parse_args()

    selected = SUBSETS if args.subset == "all" else (args.subset,)
    results = []
    for subset in selected:
        results.append(prepare_subset(subset, args.window))

    print("\nPreparation summary")
    print("-" * 64)
    for item in results:
        print(
            f"{item['subset']}: {item['n_sequences']:,} sequences, "
            f"{item['n_features']} features, window={item['sequence_length']}"
        )
    print("\nNext: update src/pytorch_dataset.py and training scripts to load each subset.")


if __name__ == "__main__":
    main()
