import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects information about the relative position of the flights in the sequence.
    Without this, the Transformer wouldn't know if a flight happened on day 1 or day 30.
    """
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Shape: (1, max_len, d_model)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        # Add the positional encoding to the input tensor
        device = x.device
        x = x + self.pe[:, :x.size(1), :].to(device)
        return x

class RUL_Transformer(nn.Module):
    def __init__(self, input_size=24, d_model=64, nhead=4, num_layers=2, dropout=0.2):
        super(RUL_Transformer, self).__init__()
        
        # 1. Project the 24 sensors into a larger 64-dimensional space
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 2. Add the Timestamps
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 3. The Core Transformer Blocks
        # batch_first=True keeps our tensors in (Batch, Sequence, Features) format
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=128, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. The Output Layers to guess the RUL
        self.fc1 = nn.Linear(d_model, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Step A: Project and add timestamps
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        
        # Step B: Pass through the Self-Attention blocks
        x = self.transformer_encoder(x)
        
        # Step C: Grab the network's understanding at the very last flight in the window
        last_step_out = x[:, -1, :]
        
        # Step D: Compress down to a single RUL prediction
        out = self.fc1(last_step_out)
        out = self.relu(out)
        predictions = self.fc2(out)
        
        return predictions.squeeze()

if __name__ == "__main__":
    # Sanity Check for the Transformer
    print("Initializing Time-Series Transformer...")
    model = RUL_Transformer()
    
    # Fake batch of 64 engines, 30 flights each, 24 sensors
    dummy_X = torch.rand(64, 30, 24)
    dummy_predictions = model(dummy_X)
    
    print("\n--- Transformer Architecture Test Successful ---")
    print(f"Input Shape: {dummy_X.shape}")
    print(f"Output Shape: {dummy_predictions.shape} -> (64 RUL Predictions)")