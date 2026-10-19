"""
Train a Transformer model for one NASA C-MAPSS subset.

Examples:
    python -m src.train_transformer --subset FD001
    python -m src.train_transformer --subset FD002
    python -m src.train_transformer --subset all

Run preprocessing first:
    python -m src.data_prep --subset all

Saves:
    models/FD001_transformer_weights.pth
    models/FD002_transformer_weights.pth
    models/FD003_transformer_weights.pth
    models/FD004_transformer_weights.pth

The model predicts RUL normalized by RUL_CAP=125 during training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.model_transformer import RUL_Transformer
from src.pytorch_dataset import get_dataloaders


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
RUL_CAP = 125.0
INPUT_SIZE = 24
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 5e-4


def train_subset(
    subset: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = 42,
) -> Path:
    """Train and save a Transformer for a single subset."""
    subset = subset.upper()
    if subset not in SUBSETS:
        raise ValueError(f"Unsupported subset {subset!r}. Choose from {SUBSETS}.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 64}")
    print(f"[INFO] Training Transformer for {subset} on {device.type.upper()}")
    print(f"[INFO] Epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

    loader = get_dataloaders(
        subset=subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    model = RUL_Transformer(input_size=INPUT_SIZE).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_squared_error = 0.0
        total_samples = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_scaled = (y_batch / RUL_CAP).to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(X_batch).reshape(-1)
            loss = criterion(predictions, y_scaled.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            count = len(y_batch)
            total_squared_error += float(loss.detach().item()) * count
            total_samples += count

        if total_samples == 0:
            raise RuntimeError(
                f"No batches were produced for {subset}. "
                "Run preprocessing first and check the processed arrays."
            )

        normalized_rmse = np.sqrt(total_squared_error / total_samples)
        rmse_cycles = normalized_rmse * RUL_CAP
        print(f"Epoch [{epoch + 1:03d}/{epochs:03d}] | Train RMSE: {rmse_cycles:.2f} cycles")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_DIR / f"{subset}_transformer_weights.pth"
    torch.save(model.to("cpu").state_dict(), output_path)
    print(
        f"[DONE] Saved {subset} Transformer weights to "
        f"{output_path.relative_to(ROOT)}"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Transformer model(s) for NASA C-MAPSS."
    )
    parser.add_argument(
        "--subset",
        choices=["all", *SUBSETS],
        default="FD001",
        help="Subset to train (default: FD001).",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    selected_subsets = SUBSETS if args.subset == "all" else (args.subset,)
    for subset in selected_subsets:
        train_subset(
            subset=subset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
