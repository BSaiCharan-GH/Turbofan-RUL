import torch
import torch.nn as nn

class RUL_LSTM(nn.Module):
    def __init__(self, input_size=24, hidden_size=64, num_layers=2, dropout=0.2):
        super(RUL_LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The core LSTM layer
        # batch_first=True tells it to expect tensors in (Batch, Sequence, Features) format
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # The fully connected layers to compress the LSTM output down to a single RUL number
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1) # Output is 1 number: the predicted RUL
        
    def forward(self, x):
        # Pass the 3D tensor through the LSTM
        # out shape: (batch_size, sequence_length, hidden_size)
        out, (h_n, c_n) = self.lstm(x)
        
        # We only care about the network's thoughts AFTER looking at all 30 flights
        # So we grab the output from the very last time step: out[:, -1, :]
        last_step_out = out[:, -1, :]
        
        # Pass that final thought through the dense layers
        x = self.fc1(last_step_out)
        x = self.relu(x)
        predictions = self.fc2(x)
        
        # Flatten from shape [64, 1] to [64] to match our target y_batch
        return predictions.squeeze()

if __name__ == "__main__":
    # Let's do a quick sanity check to make sure the model compiles!
    print("Initializing LSTM Architecture...")
    model = RUL_LSTM(input_size=24, hidden_size=64, num_layers=2)
    
    # Create a fake batch of data perfectly matching your DataLoader output
    dummy_X = torch.rand(64, 30, 24)
    
    # Push the fake data through the model
    dummy_predictions = model(dummy_X)
    
    print("\n--- Architecture Test Successful ---")
    print(f"Input Shape: {dummy_X.shape}")
    print(f"Output Shape: {dummy_predictions.shape}")
    print("The model successfully analyzed 64 engines and output 64 RUL predictions!")