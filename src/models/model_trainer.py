import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import joblib
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
from clearml import Task, Dataset

from ..utils.config_loader import ConfigLoader


class ModelTrainer:
    """Train and evaluate ML models for crop price prediction."""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config = config_loader
        self.models = {}
        self.evaluation_results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize ClearML task
        self.clearml_task = None
    
    def initialize_clearml(self):
        """Initialize ClearML task for experiment tracking."""
        try:
            clearml_config = self.config.get_clearml_config()
            self.clearml_task = Task.init(
                project_name=clearml_config.get('project_name', 'Crop Price Prediction'),
                task_name=clearml_config.get('task_name', 'Model Training'),
                task_type=Task.TaskTypes.training,
                reuse_last_task_id=False
            )
            self.logger.info("ClearML task initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize ClearML: {e}")
    
    def get_model_configs(self) -> Dict[str, Dict]:
        """Get model configurations from config."""
        return self.config.get_model_config()
    
    def create_models(self) -> Dict[str, Any]:
        """Create model instances with configurations."""
        model_configs = self.get_model_configs()
        models = {}
        
        # Random Forest
        if 'random_forest' in model_configs:
            rf_config = model_configs['random_forest']
            models['random_forest'] = RandomForestRegressor(**rf_config)
        
        # SVR
        if 'svr' in model_configs:
            svr_config = model_configs['svr']
            models['svr'] = SVR(**svr_config)
        
        # XGBoost
        if 'xgboost' in model_configs:
            xgb_config = model_configs['xgboost']
            models['xgboost'] = xgb.XGBRegressor(**xgb_config)
        
        self.logger.info(f"Created {len(models)} models: {list(models.keys())}")
        return models
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, float]:
        """Evaluate model performance."""
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Cross-validation
        cv_folds = self.config.get('training.cross_validation_folds', 5)
        cv_scores = cross_val_score(model, X_test, y_test, cv=cv_folds, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
        
        self.logger.info(f"{model_name} - R2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        # Log to ClearML if available
        if self.clearml_task:
            for metric, value in results.items():
                self.clearml_task.get_logger().report_scalar(
                    title=f"{model_name} Metrics",
                    series=metric,
                    iteration=0,
                    value=value
                )
        
        return results
    
    def train_model(self, model, X_train: np.ndarray, y_train: np.ndarray, model_name: str):
        """Train a single model."""
        self.logger.info(f"Training {model_name}...")
        
        # Log parameters to ClearML if available
        if self.clearml_task:
            self.clearml_task.connect(model.get_params(), name=f"{model_name}_params")
        
        # Train the model
        model.fit(X_train, y_train)
        
        self.logger.info(f"{model_name} training completed")
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train all models and evaluate them."""
        # Initialize ClearML
        self.initialize_clearml()
        
        # Create models
        models = self.create_models()
        
        # Train and evaluate each model
        for model_name, model in models.items():
            # Train model
            trained_model = self.train_model(model, X_train, y_train, model_name)
            
            # Evaluate model
            results = self.evaluate_model(trained_model, X_test, y_test, model_name)
            
            # Store results
            self.models[model_name] = trained_model
            self.evaluation_results[model_name] = results
        
        # Find best model
        best_model_name = max(self.evaluation_results.keys(), 
                            key=lambda x: self.evaluation_results[x]['r2'])
        
        self.logger.info(f"Best model: {best_model_name} with R2: {self.evaluation_results[best_model_name]['r2']:.4f}")
        
        # Log comparison to ClearML
        if self.clearml_task:
            comparison_data = []
            for model_name, results in self.evaluation_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'R2': results['r2'],
                    'MSE': results['mse'],
                    'MAE': results['mae'],
                    'RMSE': results['rmse']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            self.clearml_task.get_logger().report_table(
                title="Model Comparison",
                series="Performance",
                table_plot=comparison_df
            )
        
        return {
            'models': self.models,
            'evaluation_results': self.evaluation_results,
            'best_model': {
                'name': best_model_name,
                'model': self.models[best_model_name],
                'results': self.evaluation_results[best_model_name]
            }
        }
    
    def save_model(self, model_name: str, model_dir: str = "models"):
        """Save a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model_path = f"{model_dir}/{model_name}.joblib"
        
        joblib.dump(self.models[model_name], model_path)
        self.logger.info(f"Model {model_name} saved to {model_path}")
        
        # Log model artifact to ClearML if available
        if self.clearml_task:
            self.clearml_task.upload_artifact(name=f"{model_name}_model", artifact_object=model_path)
    
    def save_all_models(self, model_dir: str = "models"):
        """Save all trained models."""
        for model_name in self.models.keys():
            self.save_model(model_name, model_dir)
    
    def load_model(self, model_name: str, model_dir: str = "models"):
        """Load a trained model."""
        model_path = f"{model_dir}/{model_name}.joblib"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        self.models[model_name] = model
        self.logger.info(f"Model {model_name} loaded from {model_path}")
        
        return model
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train or load the model first.")
        
        return self.models[model_name].predict(X)
    
    def get_feature_importance(self, model_name: str, feature_names: list = None) -> Dict[str, float]:
        """Get feature importance for tree-based models."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            if feature_names:
                return dict(zip(feature_names, importance))
            else:
                return {f"feature_{i}": imp for i, imp in enumerate(importance)}
        else:
            self.logger.warning(f"Model {model_name} does not support feature importance")
            return {}
    
    def close_clearml_task(self):
        """Close ClearML task."""
        if self.clearml_task:
            self.clearml_task.close()
            self.logger.info("ClearML task closed")
