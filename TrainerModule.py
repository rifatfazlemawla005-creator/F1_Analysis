# source/TrainerModule.py
import os
import logging
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
import mlflow

from ModuleDatasetTorch import TabularDataset
from ModuleMLPModelling import DynamicMLP

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0

class PyTorchOptunaTrainer:
    def __init__(self, preprocessor_pipeline, input_dim, log_dir='logs'):
        self.preprocessor = preprocessor_pipeline
        self.input_dim = input_dim
        self.log_dir = log_dir
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"PyTorch Trainer menggunakan Hardware Device: {self.device}")

    def _train_epoch(self, model, dataloader, criterion, optimizer, epoch_idx, total_epochs):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f" 🏃 [Train] Epoch {epoch_idx+1}/{total_epochs}", leave=False)
        
        for batch_X, batch_y in progress_bar:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_X.size(0)
            progress_bar.set_postfix({'batch_mse': f"{loss.item():.4f}"})
            
        return float(total_loss / len(dataloader.dataset))

    def _validate_epoch(self, model, dataloader, criterion, epoch_idx, total_epochs):
        model.eval()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f" 🔍 [Valid] Epoch {epoch_idx+1}/{total_epochs}", leave=False)
        
        with torch.no_grad():
            for batch_X, batch_y in progress_bar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_X.size(0)
                
        return float(total_loss / len(dataloader.dataset))

    def create_optuna_objective(self, X_train_raw, y_train, X_val_raw, y_val, tb_dir=None, epochs=30):
        self.logger.info("Mentransformasikan data mentah via preprocessor pipeline...")
        X_train_clean = self.preprocessor.fit_transform(X_train_raw, y_train)
        X_val_clean = self.preprocessor.transform(X_val_raw)
        
        train_dataset = TabularDataset(X_train_clean, y_train)
        val_dataset = TabularDataset(X_val_clean, y_val)

        def objective(trial):
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
            
            num_layers = trial.suggest_int("num_layers", 1, 3)
            hidden_layers = []
            for i in range(num_layers):
                hidden_layers.append(trial.suggest_int(f"hidden_dim_layer_{i}", 16, 128, step=16))

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = DynamicMLP(self.input_dim, hidden_layers, dropout_rate).to(self.device)
            criterion = nn.MSELoss() 
            optimizer = optim.Adam(model.parameters(), lr=lr)
            early_stopping = EarlyStopping(patience=5, min_delta=0.001)

            writer = None
            if tb_dir is not None:
                trial_tb_path = os.path.join(tb_dir, f"trial_{trial.number}")
                writer = SummaryWriter(log_dir=trial_tb_path)

            print(f"\n▶️ Memulai Trial {trial.number} | Params: {trial.params}")

            with mlflow.start_run(run_name=f"PyTorch_Trial_{trial.number}", nested=True):
                mlflow.log_params(trial.params)
                
                epoch_bar = tqdm(range(epochs), desc=f"🏆 Trial {trial.number} Overall Progress")
                
                for epoch in epoch_bar:
                    train_loss = self._train_epoch(model, train_loader, criterion, optimizer, epoch, epochs)
                    val_loss = self._validate_epoch(model, val_loader, criterion, epoch, epochs)
                    
                    mlflow.log_metric(f"trial_{trial.number}_train_mse", train_loss, step=epoch)
                    mlflow.log_metric(f"trial_{trial.number}_val_mse", val_loss, step=epoch)
                    
                    if writer is not None:
                        writer.add_scalar("Loss/Train_MSE", train_loss, epoch)
                        writer.add_scalar("Loss/Validation_MSE", val_loss, epoch)
                    
                    epoch_bar.set_postfix({
                        'train_mse': f"{train_loss:.4f}", 
                        'val_mse': f"{val_loss:.4f}"
                    })
                    
                    early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                        self.logger.info(f"Trial {trial.number}: Early Stopping aktif di epoch ke-{epoch+1}")
                        break
                
                if writer is not None:
                    writer.close()
                
                # 🔥 PERBAIKAN: Jangan simpan bobot ke user_attr Optuna agar SQLite tidak JSON Error.
                # Kita langsung simpan fisik file bobot terbaik trial ini ke folder lokal 'output/models/'
                if early_stopping.best_model_weights is not None:
                    trial_model_path = f"output/models/trial_{trial.number}_mlp_model.pt"
                    torch.save(early_stopping.best_model_weights, trial_model_path)
                    # Catat path string-nya saja ke Optuna (String aman di-JSON serializable)
                    trial.set_user_attr("model_weight_path", trial_model_path)
                
            return early_stopping.best_loss

        return objective