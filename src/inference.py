import torch
import pandas as pd
import numpy as np
import os
import joblib
from model_transformer import RUL_Transformer

def predict_single_engine(engine_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Scaler and Model
    scaler = joblib.load('models/data_scaler.pkl')
    model = RUL_Transformer().to(device)
    model.load_state_dict(torch.load('models/transformer_weights.pth'))
    model.eval()

    # 2. Load Test Data
    columns = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
    feature_cols = ['setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
    
    test_df = pd.read_csv(os.path.join('data', 'raw', 'test_FD001.txt'), sep='\s+', header=None, names=columns)
    true_rul_df = pd.read_csv(os.path.join('data', 'raw', 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

    # 3. Process specific engine
    engine_data = test_df[test_df['unit_number'] == engine_id]
    if len(engine_data) < 30:
        print(f"Engine {engine_id} doesn't have enough flight history (needs 30 cycles).")
        return

    # Grab last 30 cycles and scale them
    input_data = engine_data[feature_cols].values[-30:]
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0).to(device) # Add batch dimension

    # 4. Predict
    with torch.no_grad():
        prediction = model(input_tensor).item() * 125.0
    
    actual_rul = true_rul_df.iloc[engine_id - 1]['RUL']
    
    print(f"\n--- Analysis for Engine Unit #{engine_id} ---")
    print(f"AI Prediction: {prediction:.1f} flights remaining")
    print(f"Actual Answer: {actual_rul} flights remaining")
    print(f"Error Margin:  {abs(prediction - actual_rul):.1f} flights")

if __name__ == "__main__":
    # Change this number to any engine ID from 1 to 100
    predict_single_engine(engine_id=42)