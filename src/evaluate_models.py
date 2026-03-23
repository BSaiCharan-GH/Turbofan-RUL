import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from model_lstm import RUL_LSTM
from model_transformer import RUL_Transformer

def evaluate_models():
    print("[INFO] Loading test dataset and ground truth values...")
    
    columns = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
    sensor_cols = [f's_{i}' for i in range(1, 22)]
    columns.extend(sensor_cols)
    feature_cols = ['setting_1', 'setting_2', 'setting_3'] + sensor_cols

    test_df = pd.read_csv(os.path.join('data', 'raw', 'test_FD001.txt'), sep='\s+', header=None, names=columns)
    true_rul = pd.read_csv(os.path.join('data', 'raw', 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

    print("[INFO] Applying saved scaler to normalize test data...")
    scaler_path = os.path.join('models', 'data_scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    else:
        print("[ERROR] Could not find data_scaler.pkl. Did you run data_prep.py?")
        return

    print("[INFO] Extracting final 30-cycle sequences for test engines...")
    sequence_length = 30
    X_test = []
    
    for engine_id, engine_data in test_df.groupby('unit_number'):
        engine_features = engine_data[feature_cols].values[-sequence_length:]
        X_test.append(engine_features)
        
    X_test = np.array(X_test)
    y_test = true_rul['RUL'].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.FloatTensor(X_test).to(device)
    y_tensor = torch.FloatTensor(y_test).to(device)

    print("[INFO] Initializing and loading LSTM model weights...")
    lstm_model = RUL_LSTM(input_size=24, hidden_size=64, num_layers=2).to(device)
    lstm_model.load_state_dict(torch.load('models/lstm_weights.pth'))
    lstm_model.eval() 

    print("[INFO] Initializing and loading Transformer model weights...")
    transformer_model = RUL_Transformer().to(device)
    transformer_model.load_state_dict(torch.load('models/transformer_weights.pth'))
    transformer_model.eval()

    print("[INFO] Generating predictions for both architectures...")
    with torch.no_grad():
        # --- THE FIX: Multiply by 125.0 to un-scale the predictions back to real flights ---
        lstm_preds = lstm_model(X_tensor) * 125.0
        transformer_preds = transformer_model(X_tensor) * 125.0
        
    lstm_rmse = torch.sqrt(torch.nn.functional.mse_loss(lstm_preds, y_tensor)).item()
    transformer_rmse = torch.sqrt(torch.nn.functional.mse_loss(transformer_preds, y_tensor)).item()
    
    print("\n--- Final Model Evaluation Results ---")
    print(f"LSTM Test RMSE:        {lstm_rmse:.2f} cycles")
    print(f"Transformer Test RMSE: {transformer_rmse:.2f} cycles")
    print("--------------------------------------")
    
    if transformer_rmse < lstm_rmse:
        print("[RESULT] The Transformer architecture yielded the lower error rate.")
    else:
        print("[RESULT] The LSTM architecture yielded the lower error rate.")

    print("\n[INFO] Generating comparative visualization...")
    plt.figure(figsize=(14, 7))
    
    indices = np.argsort(y_test)[::-1]
    
    plt.plot(y_test[indices], label='Actual RUL (Ground Truth)', color='black', linewidth=3)
    plt.plot(lstm_preds.cpu().numpy()[indices], label=f'LSTM (RMSE: {lstm_rmse:.2f})', color='red', alpha=0.6, linestyle='--')
    plt.plot(transformer_preds.cpu().numpy()[indices], label=f'Transformer (RMSE: {transformer_rmse:.2f})', color='blue', alpha=0.8, linestyle='-.')
    
    plt.title('Model Comparison: LSTM vs. Transformer for RUL Prediction', fontsize=14, fontweight='bold')
    plt.xlabel('Test Engine ID (Sorted by descending RUL)')
    plt.ylabel('Remaining Useful Life (Flight Cycles)')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('models/final_evaluation_graph.png', bbox_inches='tight')
    print("[INFO] Graph successfully saved to models/final_evaluation_graph.png")
    
    plt.show()

if __name__ == "__main__":
    evaluate_models()