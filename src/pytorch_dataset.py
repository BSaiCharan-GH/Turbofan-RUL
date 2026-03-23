import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class TurbofanDataset(Dataset):
    """
    Custom PyTorch Dataset for the CMAPSS Turbofan engine data.
    Converts 3D numpy arrays into PyTorch Tensors.
    """
    def __init__(self, X_npy_path, y_npy_path):
        print(f"Loading data from {X_npy_path} into RAM...")
        
        # Load the numpy arrays from disk
        X_data = np.load(X_npy_path)
        y_data = np.load(y_npy_path)
        
        # Convert Numpy arrays to PyTorch FloatTensors (32-bit float)
        # Deep Learning models require FloatTensors to calculate gradients
        self.X = torch.FloatTensor(X_data)
        self.y = torch.FloatTensor(y_data)
        
    def __len__(self):
        # Tells PyTorch exactly how many samples we have (17,731)
        return len(self.X)
    
    def __getitem__(self, idx):
        # When PyTorch asks for a batch, this grabs the specific rows
        return self.X[idx], self.y[idx]

def get_dataloaders(batch_size=64):
    """
    Builds the conveyor belt that streams data to the GPU.
    A batch size of 64 is perfectly optimized for a 6GB RTX 3050.
    """
    X_train_path = os.path.join('data', 'processed', 'X_train.npy')
    y_train_path = os.path.join('data', 'processed', 'y_train.npy')
    
    # Initialize our custom dataset
    train_dataset = TurbofanDataset(X_train_path, y_train_path)
    
    # Create the DataLoader
    # shuffle=True ensures the model doesn't just memorize the order of the engines
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0, # Keep at 0 for Windows to avoid multiprocess crashing
        drop_last=True # Drops the final uneven batch to keep matrix math clean
    )
    
    return train_loader

if __name__ == "__main__":
    # Let's test the conveyor belt!
    print("Initializing PyTorch DataLoader...")
    train_loader = get_dataloaders(batch_size=64)
    
    # Grab just ONE batch off the conveyor belt to inspect it
    data_iterator = iter(train_loader)
    X_batch, y_batch = next(data_iterator)
    
    print("\n--- Conveyor Belt Test Successful ---")
    print(f"One Batch of X (Inputs): {X_batch.shape} -> (Batch Size, Sequence, Features)")
    print(f"One Batch of y (Targets): {y_batch.shape} -> (Batch Size)")
    print(f"Data Type: {X_batch.dtype}")