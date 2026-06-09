# source/ModuleXGBRFModelling.py
import os
import logging
from xgboost import XGBRFRegressor
import mlflow
import mlflow.xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

class XGBRFTrainer:
    def __init__(self, preprocessor_pipeline, log_dir='logs'):
        self.preprocessor = preprocessor_pipeline
        self.log_dir = log_dir
        self.best_pipeline = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def tune_and_train_xgbrf(self, X_train, y_train, X_val, y_val, param_grid=None):
        """Fungsi OOP untuk Hyperparameter Tuning + Pelatihan XGBRF Regressor"""
        if param_grid is None:
            param_grid = {
                'regressor__n_estimators': [50, 100],
                'regressor__max_depth': [4, 6, 8]
            }

        base_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', XGBRFRegressor(objective='reg:squarederror', random_state=42, n_jobs=1))
        ])

        mlflow.xgboost.autolog()
        self.logger.info("Memulai proses Grid Search untuk XGBRF Regressor...")
        
        with mlflow.start_run(run_name="GridSearch_XGBRF", nested=True):
            grid_search = GridSearchCV(
                estimator=base_pipeline,
                param_grid=param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                verbose=1,
                n_jobs=-1
            )

            self.logger.info("Mengeksekusi Grid Search pada data training...")
            grid_search.fit(X_train, y_train)

            self.best_pipeline = grid_search.best_estimator_
            best_params = grid_search.best_params_

            self.logger.info(f"🏆 Kombinasi Parameter Terbaik XGBRF: {best_params}")

            # Evaluasi Akhir Regresi
            y_pred = self.best_pipeline.predict(X_val)
            final_mse = mean_squared_error(y_val, y_pred)
            final_r2 = r2_score(y_val, y_pred)
            
            mlflow.log_metric("final_val_xgbrf_mse", final_mse)
            mlflow.log_metric("final_val_xgbrf_r2", final_r2)
            
            self.logger.info(f"📉 Hasil Akhir Data Validasi XGBRF -> MSE: {final_mse:.4f} | R2-Score: {final_r2:.4f}")
            return best_params
