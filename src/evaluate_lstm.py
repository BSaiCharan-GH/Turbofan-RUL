import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import our architecture
from model_lstm import RUL_LSTM

def evaluate_model():
    print("Loading Test Data and Answer Key...")
    
    # 1. Define columns
    columns = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
    sensor_cols = [f's_{i}' for i in range(1, 22)]
    columns.extend(sensor_cols)
    feature_cols = ['setting_1', 'setting_2', 'setting_3'] + sensor_cols

    # 2. Load the test telemetry and the true answers
    test_df = pd.read_csv(os.path.join('data', 'raw', 'test_FD001.txt'), sep='\s+', header=None, names=columns)
    true_rul = pd.read_csv(os.path.join('data', 'raw', 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

    # 3. Extract the last 30 flights for each of the 100 test engines
    sequence_length = 30
    X_test = []
    
    for engine_id, engine_data in test_df.groupby('unit_number'):
        # We only want the very last 30 rows of data before the recording stopped
        engine_features = engine_data[feature_cols].values[-sequence_length:]
        X_test.append(engine_features)
        
    X_test = np.array(X_test)
    y_test = true_rul['RUL'].values

    # 4. Move data to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.FloatTensor(X_test).to(device)
    y_tensor = torch.FloatTensor(y_test).to(device)

    # 5. Load the Trained LSTM Weights
    print("Loading Trained LSTM from disk...")
    model = RUL_LSTM(input_size=24, hidden_size=64, num_layers=2).to(device)
    model.load_state_dict(torch.load('models/lstm_weights.pth'))
    
    # PUT MODEL IN EVALUATION MODE (turns off dropout, locks weights)
    model.eval() 

    # 6. Make Predictions!
    # torch.no_grad() tells the GPU we aren't training, saving massive amounts of VRAM
    with torch.no_grad():
        predictions = model(X_tensor)
        
    # 7. Calculate the official Test RMSE
    mse = torch.nn.functional.mse_loss(predictions, y_tensor)
    rmse = torch.sqrt(mse).item()
    
    print(f"\n=====================================")
    print(f"🏆 OFFICIAL TEST RMSE: {rmse:.2f} flights")
    print(f"=====================================")

    # 8. Plot the Graph
    print("\nGenerating graph...")
    plt.figure(figsize=(12, 6))
    
    # Sort the values so the graph looks like a clean curve instead of chaotic spikes
    indices = np.argsort(y_test)[::-1]
    
    plt.plot(y_test[indices], label='Actual RUL (Ground Truth)', color='blue', linewidth=2)
    plt.plot(predictions.cpu().numpy()[indices], label='LSTM Predictions', color='red', alpha=0.7, linestyle='dashed')
    
    plt.title('LSTM Baseline: Actual vs Predicted Engine Failures')
    plt.xlabel('Test Engine ID (Sorted by descending RUL)')
    plt.ylabel('Remaining Useful Life (Flights)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the graph to your models folder
    plt.savefig('models/lstm_evaluation.png')
    print("Saved prediction graph to models/lstm_evaluation.png")
    
    # Show the graph on your screen
    plt.show()

if __name__ == "__main__":
    evaluate_model()