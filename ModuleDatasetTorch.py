# source/ModuleDatasetTorch.py
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    """
    Mengubah fitur dan target tabular menjadi Tensor PyTorch 
    yang siap dikonsumsi oleh batch DataLoader (Versi Regresi).
    """
    def __init__(self, X_data, y_data):
        if hasattr(X_data, 'values'):
            X_data = X_data.values
        if hasattr(y_data, 'values'):
            y_data = y_data.values
            
        self.X = torch.tensor(X_data, dtype=torch.float32)
        # 🔥 DIUBAH: target y diubah ke float32 untuk kebutuhan Regresi kecepatan
        self.y = torch.tensor(y_data, dtype=torch.float32) 
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]