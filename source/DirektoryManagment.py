# source/DirektoryManagment.py
import os
import logging
import torch
import joblib
import optuna.visualization as vis
import shutil

class WorkspaceManager:
    def __init__(self, base_dir='.', log_dir='logs'):
        """
        OOP Manager untuk otomatisasi folder, penyimpanan model, 
        database Optuna, visualisasi plot, dan log TensorBoard.
        """
        self.base_dir = base_dir
        
        # Definisikan jalur subfolder standar industri
        self.folders = {
            'models': os.path.join(base_dir, 'models'),
            'optuna': os.path.join(base_dir, 'optuna'),
            'visuals': os.path.join(base_dir, 'visuals'),
            'tensorboard': os.path.join(base_dir, 'tensorboard'),
            'logs': os.path.join(base_dir, log_dir)
        }
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._create_subfolders()

    def _create_subfolders(self):
        """Fungsi internal untuk membuat seluruh subfolder jika belum ada"""
        for name, path in self.folders.items():
            if not os.path.exists(path):
                os.makedirs(path)
                self.logger.info(f"📁 Folder baru berhasil digelar: {path}")

    def get_optuna_db_path(self, study_name="f1_mlops_study"):
        """Menghasilkan URL path SQLite lokal untuk database Optuna"""
        db_path = os.path.join(self.folders['optuna'], f"{study_name}.db")
        return f"sqlite:///{os.path.abspath(db_path)}"

    def get_tensorboard_dir(self, experiment_name="mlp_run"):
        """Menghasilkan path khusus untuk menampung log TensorBoard"""
        return os.path.join(self.folders['tensorboard'], experiment_name)

    def save_sklearn_pipeline(self, pipeline_obj, filename="best_pipeline.pkl"):
        """Menyimpan pipeline Scikit-Learn/XGBoost ke folder models/"""
        save_path = os.path.join(self.folders['models'], filename)
        joblib.dump(pipeline_obj, save_path)
        self.logger.info(f"💾 Pipeline model berhasil disimpan di: {save_path}")

    def save_pytorch_weights(self, state_dict, filename="best_mlp_model.pt"):
        """Menyimpan bobot arsitektur PyTorch ke folder models/"""
        save_path = os.path.join(self.folders['models'], filename)
        torch.save(state_dict, save_path)
        self.logger.info(f"💾 Bobot model PyTorch berhasil disimpan di: {save_path}")

    def export_optuna_visuals(self, study, prefix="experiment"):
        """Mengekspor grafik riwayat optimasi Optuna ke bentuk file HTML interaktif"""
        try:
            # 1. Plot Riwayat Optimasi
            fig_hist = vis.plot_optimization_history(study)
            path_hist = os.path.join(self.folders['visuals'], f"{prefix}_optimization_history.html")
            fig_hist.write_html(path_hist)
            
            # 2. Plot Pentingnya Parameter (Hyperparameter Importances)
            fig_imp = vis.plot_param_importances(study)
            path_imp = os.path.join(self.folders['visuals'], f"{prefix}_param_importances.html")
            fig_imp.write_html(path_imp)
            
            self.logger.info(f"📊 Sukses mengekspor grafik visualisasi interaktif ke folder: {self.folders['visuals']}")
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor visualisasi Optuna: {str(e)}")

    def reset_workspace(self):
        """Menghapus seluruh subfolder di dalam base_dir untuk reset eksperimen"""
        self.logger.warning(f"🚨 Memulai pembersihan total direktori di: {self.base_dir}")
        for name, path in self.folders.items():
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                    self.logger.info(f"🗑️ Berhasil menghapus folder: {path}")
                except Exception as e:
                    self.logger.error(f"❌ Gagal menghapus {path}: {e}")
        self._create_subfolders()
