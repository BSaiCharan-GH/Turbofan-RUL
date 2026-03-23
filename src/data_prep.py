import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_cmapss_data(file_path):
    print(f"[INFO] Loading data from: {file_path}...")
    columns = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
    sensor_cols = [f's_{i}' for i in range(1, 22)]
    columns.extend(sensor_cols)
    return pd.read_csv(file_path, sep='\s+', header=None, names=columns)

def add_rul_column(df, clip_rul=125):
    max_cycles = pd.DataFrame(df.groupby('unit_number')['time_cycles'].max()).reset_index()
    max_cycles.columns = ['unit_number', 'max_cycle']
    df = df.merge(max_cycles, on=['unit_number'], how='left')
    df['RUL'] = df['max_cycle'] - df['time_cycles']
    df['RUL'] = df['RUL'].clip(upper=clip_rul)
    df.drop('max_cycle', axis=1, inplace=True)
    return df

def generate_sequences(df, sequence_length, feature_cols):
    print(f"[INFO] Generating {sequence_length}-flight sliding windows...")
    X, y = [], []
    
    for engine_id, engine_data in df.groupby('unit_number'):
        engine_features = engine_data[feature_cols].values
        engine_rul = engine_data['RUL'].values
        num_windows = len(engine_data) - sequence_length + 1
        
        for i in range(num_windows):
            window = engine_features[i : i + sequence_length]
            target_rul = engine_rul[i + sequence_length - 1]
            X.append(window)
            y.append(target_rul)
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    train_file = os.path.join('data', 'raw', 'train_FD001.txt')
    
    # 1. Load and calculate RUL
    train_df = load_cmapss_data(train_file)
    train_df = add_rul_column(train_df, clip_rul=125)
    
    # 2. Define our inputs
    feature_columns = ['setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
    
    # 3. --- NEW SCALING STEP ---
    print("[INFO] Normalizing sensor data (MinMaxScaler)...")
    scaler = MinMaxScaler()
    
    # Fit the scaler to the training data and transform it
    train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    
    # Save the scaler so we can apply the exact same mathematical transformation to our Test Data later
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, os.path.join('models', 'data_scaler.pkl'))
    print("[INFO] Saved scaler to models/data_scaler.pkl")
    
    # 4. Generate the 3D Tensors
    X_train, y_train = generate_sequences(train_df, sequence_length=30, feature_cols=feature_columns)
    
    print("\n--- Data Engineering Complete ---")
    print(f"X_train Shape: {X_train.shape}")
    
    # 5. Save to disk
    os.makedirs(os.path.join('data', 'processed'), exist_ok=True)
    np.save(os.path.join('data', 'processed', 'X_train.npy'), X_train)
    np.save(os.path.join('data', 'processed', 'y_train.npy'), y_train)
    print("[INFO] Saved normalized 3D arrays to data/processed/")