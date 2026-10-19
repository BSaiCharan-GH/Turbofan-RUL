# Turbofan Engine Remaining Useful Life (RUL) Prediction

### Predictive maintenance with LSTM and Transformer models

This project predicts the **Remaining Useful Life (RUL)** of aircraft engines using the NASA C-MAPSS turbofan degradation datasets. It compares an LSTM model with a Transformer model and provides a Streamlit dashboard for exploring sensor telemetry and model predictions.

**Supported subsets:** FD001, FD002, FD003, and FD004.

> **Important:** Model performance depends on the trained weights and data-preparation workflow in use. Treat any previously reported FD001 RMSE values as historical results unless they are reproduced with the current code and evaluation script.

---

## Features

- Four NASA C-MAPSS subsets supported through subset-specific preprocessing and model artifacts.
- 24 input features: 3 operating settings and 21 sensor measurements.
- Sliding windows of 30 cycles.
- RUL targets capped at 125 cycles and normalized during training.
- LSTM and Transformer model comparison.
- Evaluation metrics: RMSE, MAE, and NASA asymmetric score.
- Streamlit dashboard for dataset exploration and predictions.

## Repository structure

```text
Turbofan-RUL/
├── app.py
├── data/
│   ├── raw/                  # NASA files; keep large data files out of Git
│   └── processed/            # Generated per-subset arrays and metadata
├── models/                   # Per-subset scalers and trained model weights
├── results/                  # Generated evaluation CSV/JSON files
├── src/
│   ├── data_prep.py
│   ├── pytorch_dataset.py
│   ├── model_lstm.py
│   ├── model_transformer.py
│   ├── train_lstm.py
│   ├── train_transformer.py
│   └── evaluate_models.py
└── README.md
```

## 1. Install dependencies

Use Python 3.10 or 3.11 in a virtual environment. From the repository root on Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn joblib torch streamlit plotly
```

If your project includes a `requirements.txt`, you can install from that file instead:

```powershell
pip install -r requirements.txt
```

## 2. Download and place the NASA C-MAPSS data

Download the NASA C-MAPSS dataset from the official NASA Prognostics Center of Excellence repository. Extract the archive and place these eight files in `data/raw/`:

```text
data/raw/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── train_FD002.txt
├── test_FD002.txt
├── RUL_FD002.txt
├── train_FD003.txt
├── test_FD003.txt
└── RUL_FD003.txt
```

Also include `train_FD004.txt`, `test_FD004.txt`, and `RUL_FD004.txt` in the same folder.

Do not commit the raw dataset or generated arrays to Git unless you have checked repository size limits and licensing/distribution requirements.

## 3. Prepare data

Run commands from the repository root:

```powershell
python -m src.data_prep --subset FD001
```

For all four subsets:

```powershell
python -m src.data_prep --subset all
```

The preparation script should create subset-specific processed data and scalers. Each subset must use its own scaler and its own model weights.

## 4. Train the models

Train an LSTM for one subset:

```powershell
python -m src.train_lstm --subset FD001
```

Train a Transformer for one subset:

```powershell
python -m src.train_transformer --subset FD001
```

Repeat with `FD002`, `FD003`, or `FD004`, or train all subsets with:

```powershell
python -m src.train_lstm --subset all
python -m src.train_transformer --subset all
```

Training may take a while. GPU training is used when the training script supports and detects CUDA; otherwise, training can run on the CPU.

Expected subset-specific artifacts include:

```text
models/FD001_data_scaler.pkl
models/FD001_lstm_weights.pth
models/FD001_transformer_weights.pth
```

Equivalent artifacts should exist for FD002, FD003, and FD004 before evaluating or serving predictions for those subsets.

## 5. Evaluate models

Evaluate one subset:

```powershell
python -m src.evaluate_models --subset FD001
```

Evaluate all four subsets:

```powershell
python -m src.evaluate_models --subset all
```

Choose the device explicitly if needed:

```powershell
python -m src.evaluate_models --subset FD001 --device cpu
python -m src.evaluate_models --subset FD001 --device cuda
```

The evaluation script writes per-subset predictions and a summary under `results/`, including RMSE, MAE, and NASA score. Compare models using the same test subset and its matching RUL labels.

## 6. Run the Streamlit dashboard

From the repository root:

```powershell
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

The dashboard needs the correct scaler and model weights for the selected subset. Do not select FD002–FD004 until their corresponding artifacts have been prepared and trained.

## Deployment notes

- Streamlit Community Cloud can host the dashboard, but it cannot access files that exist only on your local computer.
- Ensure `app.py`, source modules, dependencies, and required model artifacts are available to the deployed app.
- Avoid committing raw datasets, processed arrays, virtual environments, or caches.
- If model files exceed Git hosting limits, use an appropriate external artifact store and update the application to load them securely.
- After changing model code, retrain the corresponding weights and re-evaluate before reporting new performance metrics.

## Method overview

1. Parse engine cycles, operating settings, and sensor measurements.
2. Fit a MinMaxScaler using training features for the selected subset.
3. Build 30-cycle sliding windows for sequence learning.
4. Train LSTM and Transformer regressors on capped, normalized RUL targets.
5. Predict the final RUL for each test engine and compare predictions with NASA-provided ground-truth labels.

## Acknowledgment

Dataset: NASA C-MAPSS turbofan engine degradation simulation data, made available through NASA's Prognostics Center of Excellence.
