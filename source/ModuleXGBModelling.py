# source/ModuleXGBModelling.py
import os
import logging
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

class ModelTrainer:
    def __init__(self, preprocessor_pipeline, log_dir='logs'):
        self.preprocessor = preprocessor_pipeline
        self.log_dir = log_dir
        self.best_model = None
        self.best_pipeline = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def tune_and_train_xgboost(self, X_train, y_train, X_val, y_val, param_grid=None):
        """Fungsi OOP untuk Hyperparameter Tuning + Pelatihan XGBoost Regressor"""
        if param_grid is None:
            param_grid = {
                'regressor__n_estimators': [50, 100],
                'regressor__learning_rate': [0.05, 0.1],
                'regressor__max_depth': [3, 5, 7]
            }

        # Menggunakan 'regressor' berbasis XGBRegressor
        base_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=1))
        ])

        mlflow.xgboost.autolog()
        self.logger.info("Memulai proses Grid Search untuk mencari hyperparameter XGBRegressor terbaik...")
        
        with mlflow.start_run(run_name="GridSearch_XGBoost", nested=True):
            grid_search = GridSearchCV(
                estimator=base_pipeline,
                param_grid=param_grid,
                cv=3,
                scoring='neg_mean_squared_error',
                verbose=1,
                n_jobs=-1
            )

            self.logger.info("Mengeksekusi Grid Search Cross-Validation pada data training...")
            grid_search.fit(X_train, y_train)

            self.best_pipeline = grid_search.best_estimator_
            self.best_model = grid_search.best_estimator_.named_steps['regressor']
            best_params = grid_search.best_params_

            self.logger.info(f"🏆 Kombinasi Parameter Terbaik XGBoost: {best_params}")

            # Evaluasi Akhir Menggunakan Metrik Regresi
            self.logger.info("Mengeksekusi prediksi akhir model XGBoost terbaik pada data validasi...")
            y_pred = self.best_pipeline.predict(X_val)
            
            final_mse = mean_squared_error(y_val, y_pred)
            final_r2 = r2_score(y_val, y_pred)
            
            mlflow.log_metric("final_val_mse", final_mse)
            mlflow.log_metric("final_val_r2_score", final_r2)
            
            self.logger.info(f"📉 Hasil Akhir Data Validasi -> MSE: {final_mse:.4f} | R2-Score: {final_r2:.4f}")
            return best_params
