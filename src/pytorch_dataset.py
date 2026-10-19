"""
PyTorch Dataset and DataLoader for NASA C-MAPSS FD001-FD004.

Expected outputs from src/data_prep.py:
    data/processed/FD001/X_train.npy
    data/processed/FD001/y_train.npy
    ... same layout for FD002-FD004.

Usage:
    from src.pytorch_dataset import get_dataloaders
    train_loader = get_dataloaders(subset="FD002", batch_size=64)

For backward compatibility, subset defaults to FD001.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
VALID_SUBSETS = ("FD001", "FD002", "FD003", "FD004")
EXPECTED_FEATURES = 24
DEFAULT_WINDOW_SIZE = 30


class TurbofanDataset(Dataset):
    """Load and validate preprocessed C-MAPSS arrays."""

    def __init__(
        self,
        X_npy_path: str | Path,
        y_npy_path: str | Path,
        expected_window: int | None = DEFAULT_WINDOW_SIZE,
        expected_features: int = EXPECTED_FEATURES,
    ) -> None:
        self.X_path = Path(X_npy_path)
        self.y_path = Path(y_npy_path)

        if not self.X_path.exists():
            raise FileNotFoundError(
                f"Training input array not found: {self.X_path}. "
                "Run: python -m src.data_prep --subset FD001"
            )
        if not self.y_path.exists():
            raise FileNotFoundError(
                f"Training target array not found: {self.y_path}. "
                "Run the matching src.data_prep command first."
            )

        X_data = np.load(self.X_path, allow_pickle=False)
        y_data = np.load(self.y_path, allow_pickle=False)

        if X_data.ndim != 3:
            raise ValueError(
                "Expected X with shape (samples, window, features); "
                f"received {X_data.shape}."
            )
        y_data = y_data.reshape(-1)
        if len(X_data) != len(y_data):
            raise ValueError(
                f"Sample count mismatch: X={len(X_data)}, y={len(y_data)}."
            )
        if not len(X_data):
            raise ValueError("Processed training arrays contain no samples.")
        if X_data.shape[2] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features; got {X_data.shape[2]}."
            )
        if expected_window is not None and X_data.shape[1] != expected_window:
            raise ValueError(
                f"Expected window length {expected_window}; got {X_data.shape[1]}."
            )
        if not np.isfinite(X_data).all() or not np.isfinite(y_data).all():
            raise ValueError("Processed arrays contain NaN or infinite values.")

        self.X = torch.from_numpy(np.asarray(X_data, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y_data, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def get_dataloaders(
    batch_size: int = 64,
    subset: str = "FD001",
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    expected_window: int = DEFAULT_WINDOW_SIZE,
) -> DataLoader:
    """Return a training DataLoader for the selected subset."""
    subset = subset.upper()
    if subset not in VALID_SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; choose from {VALID_SUBSETS}.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    subset_dir = PROCESSED_DIR / subset
    dataset = TurbofanDataset(
        subset_dir / "X_train.npy",
        subset_dir / "y_train.npy",
        expected_window=expected_window,
        expected_features=EXPECTED_FEATURES,
    )

    if drop_last and len(dataset) < batch_size:
        raise ValueError(
            f"{subset} has only {len(dataset)} samples, fewer than batch_size "
            f"{batch_size}; use a smaller batch size or drop_last=False."
        )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a processed NASA C-MAPSS subset."
    )
    parser.add_argument("--subset", choices=VALID_SUBSETS, default="FD001")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    loader = get_dataloaders(
        subset=args.subset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    X_batch, y_batch = next(iter(loader))
    print(f"Subset: {args.subset}")
    print(f"Training windows: {len(loader.dataset):,}")
    print(f"Batch X shape: {tuple(X_batch.shape)}")
    print(f"Batch y shape: {tuple(y_batch.shape)}")
    print(f"Device available: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
