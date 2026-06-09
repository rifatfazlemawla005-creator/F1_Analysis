# source/ModuleMLPModelling.py
import torch.nn as nn

class DynamicMLP(nn.Module):
    """
    Arsitektur Multi-Layer Perceptron (MLP) untuk Data Tabular Regresi.
    Ukuran hidden layer dibuat dinamis agar bisa di-tuning oleh Optuna.
    """
    def __init__(self, input_dim, hidden_layers, dropout_rate=0.2):
        super(DynamicMLP, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        # Menyusun Hidden Layers secara otomatis berbasis List input
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim)) 
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate)) 
            in_dim = hidden_dim
            
        # 🔥 DIUBAH: Output layer langsung menembak ke 1 neuron untuk nilai regresi kecepatan
        layers.append(nn.Linear(in_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)