# Turbofan Engine Remaining Useful Life (RUL) Prediction
### Predictive Maintenance using LSTM and Transformer Architectures

This repository contains a Deep Learning pipeline designed to predict the **Remaining Useful Life (RUL)** of aircraft engines using the **NASA CMAPSS (FD001)** dataset. The project compares traditional Recurrent Neural Networks (**LSTM**) against modern Self-Attention mechanisms (**Transformers**) to determine the most effective architecture for time-series forecasting in aviation safety.

---

## Performance Results

After 50 epochs of training on normalized telemetry, the models achieved the following performance on unseen test engines:

| Architecture | Test RMSE (Cycles) | Result |
| :--- | :--- | :--- |
| **Transformer** | **14.59** | **Winner** |
| **LSTM** | **14.74** | **Baseline** |

### Key Takeaway
The **Transformer** architecture successfully captured long-range temporal dependencies in the sensor data, providing a more accurate failure curve than the sequential LSTM approach. Both models showed significantly higher accuracy as engines approached the end of their life cycles (RUL < 50).

---

## The Pipeline

1.  **Data Engineering**: Raw sensor telemetry is cleaned and transformed into 30-flight sliding windows to provide the models with temporal context.
2.  **Normalization**: All 24 sensor features are scaled using **MinMaxScaler** to ensure stable gradient descent and prevent large-value sensors from dominating the loss function.
3.  **Target Scaling**: RUL targets are scaled to a [0, 1] range during training to prevent the "flatlining" effect where the model defaults to predicting the mean RUL.
4.  **Inference**: A dedicated script allows for real-time RUL diagnosis of specific engine units from the test set.

---

## Folder Structure

```text
RUL_Project/
├── data/
│   ├── raw/             # Original NASA .txt files (FD001)
│   └── processed/       # Pre-processed 3D Numpy arrays
├── models/              # Saved .pth weights and MinMaxScaler object
├── src/                 # Source Code
│   ├── data_prep.py         # Data cleaning & Scaling logic
│   ├── pytorch_dataset.py   # Custom PyTorch Dataset and DataLoader
│   ├── model_lstm.py        # LSTM Neural Network Architecture
│   ├── model_transformer.py # Transformer/Attention Architecture
│   ├── train_lstm.py        # LSTM Training Loop
│   ├── train_transformer.py # Transformer Training Loop
│   ├── evaluate_models.py   # Comparison, Metrics & Plotting
│   └── inference.py         # Single-engine diagnosis tool
└── README.md            # Project Documentation
