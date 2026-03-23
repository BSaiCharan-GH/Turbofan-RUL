import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from pytorch_dataset import get_dataloaders
from model_transformer import RUL_Transformer

def train_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Transformer on Device: {device.type.upper()} ---")

    train_loader = get_dataloaders(batch_size=64) 
    model = RUL_Transformer().to(device)

    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Increased epochs to 50
    num_epochs = 50 
    print("\nStarting Training (Listen for those GPU fans!)...")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            
            # --- THE FIX: Scale target to [0, 1] ---
            y_batch_scaled = y_batch / 125.0
            
            loss = criterion(predictions, y_batch_scaled)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Multiply by 125 here to print real flight numbers
        avg_mse = epoch_loss / len(train_loader)
        rmse = np.sqrt(avg_mse) * 125.0
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | RMSE: {rmse:.2f} flights")

    print("\n--- Transformer Training Complete ---")
    torch.save(model.state_dict(), 'models/transformer_weights.pth')
    print("Saved trained model to models/transformer_weights.pth")

if __name__ == "__main__":
    train_transformer()