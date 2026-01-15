import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config_loader import ConfigLoader
from src.models.model_trainer import ModelTrainer


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'models': {
            'random_forest': {
                'n_estimators': 10,
                'max_depth': 5,
                'random_state': 42
            },
            'svr': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale'
            },
            'xgboost': {
                'n_estimators': 10,
                'max_depth': 3,
                'learning_rate': 0.1,
                'random_state': 42
            }
        },
        'training': {
            'cross_validation_folds': 3
        },
        'clearml': {
            'project_name': 'Test Project',
            'task_name': 'Test Task'
        }
    }


@pytest.fixture
def sample_data():
    """Sample training data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 8
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 1000 + 1500  # Price range around 1500
    
    return X, y


@pytest.fixture
def config_loader(sample_config):
    """Create a mock config loader."""
    mock_config = Mock(spec=ConfigLoader)
    mock_config.get.side_effect = lambda key, default=None: sample_config.get(key, default)
    return mock_config


@pytest.fixture
def model_trainer(config_loader):
    """Create a model trainer instance."""
    return ModelTrainer(config_loader)


class TestModelTrainer:
    """Test cases for ModelTrainer class."""
    
    def test_initialization(self, model_trainer):
        """Test model trainer initialization."""
        assert model_trainer.config is not None
        assert model_trainer.models == {}
        assert model_trainer.evaluation_results == {}
    
    def test_get_model_configs(self, model_trainer):
        """Test getting model configurations."""
        configs = model_trainer.get_model_configs()
        
        assert isinstance(configs, dict)
        assert 'random_forest' in configs
        assert 'svr' in configs
        assert 'xgboost' in configs
    
    def test_create_models(self, model_trainer):
        """Test model creation."""
        models = model_trainer.create_models()
        
        assert isinstance(models, dict)
        assert len(models) == 3
        assert 'random_forest' in models
        assert 'svr' in models
        assert 'xgboost' in models
        
        # Check model types
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        import xgboost as xgb
        
        assert isinstance(models['random_forest'], RandomForestRegressor)
        assert isinstance(models['svr'], SVR)
        assert isinstance(models['xgboost'], xgb.XGBRegressor)
    
    @patch('src.models.model_trainer.Task')
    def test_initialize_clearml_success(self, mock_task, model_trainer):
        """Test successful ClearML initialization."""
        mock_task_instance = Mock()
        mock_task.init.return_value = mock_task_instance
        
        model_trainer.initialize_clearml()
        
        mock_task.init.assert_called_once()
        assert model_trainer.clearml_task == mock_task_instance
    
    @patch('src.models.model_trainer.Task')
    def test_initialize_clearml_failure(self, mock_task, model_trainer):
        """Test ClearML initialization failure."""
        mock_task.init.side_effect = Exception("ClearML error")
        
        # Should not raise exception, just log warning
        model_trainer.initialize_clearml()
        
        assert model_trainer.clearml_task is None
    
    def test_evaluate_model(self, model_trainer, sample_data):
        """Test model evaluation."""
        X_train, y_train = sample_data
        X_test, y_test = X_train[:20], y_train[:20]  # Use subset for testing
        
        # Create and train a simple model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        results = model_trainer.evaluate_model(model, X_test, y_test, "test_model")
        
        # Check result structure
        assert isinstance(results, dict)
        assert 'mse' in results
        assert 'mae' in results
        assert 'rmse' in results
        assert 'r2' in results
        assert 'cv_mean' in results
        assert 'cv_std' in results
        
        # Check metric types and ranges
        assert isinstance(results['mse'], (int, float))
        assert isinstance(results['mae'], (int, float))
        assert isinstance(results['r2'], (int, float))
        assert results['mse'] >= 0
        assert results['mae'] >= 0
        assert results['rmse'] >= 0
    
    def test_train_model(self, model_trainer, sample_data):
        """Test model training."""
        X_train, y_train = sample_data
        
        # Create a model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        # Train model
        trained_model = model_trainer.train_model(model, X_train, y_train, "test_model")
        
        # Check that model is trained
        assert hasattr(trained_model, 'predict')
        assert trained_model is model
    
    def test_train_all_models(self, model_trainer, sample_data):
        """Test training all models."""
        X_train, y_train = sample_data
        X_test, y_test = X_train[:20], y_train[:20]  # Use subset for testing
        
        # Train all models
        results = model_trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Check result structure
        assert 'models' in results
        assert 'evaluation_results' in results
        assert 'best_model' in results
        
        # Check that all models were trained
        assert len(results['models']) == 3
        assert len(results['evaluation_results']) == 3
        
        # Check best model structure
        assert 'name' in results['best_model']
        assert 'model' in results['best_model']
        assert 'results' in results['best_model']
        
        # Check that best model has highest R2
        best_r2 = results['best_model']['results']['r2']
        for model_results in results['evaluation_results'].values():
            assert model_results['r2'] <= best_r2
    
    def test_save_and_load_model(self, model_trainer, sample_data, tmp_path):
        """Test saving and loading models."""
        X_train, y_train = sample_data
        
        # Create and train a model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        
        model_trainer.models['test_model'] = model
        
        # Save model
        model_dir = tmp_path / "models"
        model_trainer.save_model('test_model', str(model_dir))
        
        # Check model file exists
        assert (model_dir / "test_model.joblib").exists()
        
        # Create new trainer and load model
        new_trainer = ModelTrainer(model_trainer.config)
        loaded_model = new_trainer.load_model('test_model', str(model_dir))
        
        # Check that loaded model works
        assert hasattr(loaded_model, 'predict')
        predictions = loaded_model.predict(X_train[:5])
        assert len(predictions) == 5
    
    def test_predict(self, model_trainer, sample_data):
        """Test making predictions."""
        X_train, y_train = sample_data
        
        # Create and train a model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        
        model_trainer.models['test_model'] = model
        
        # Make predictions
        predictions = model_trainer.predict('test_model', X_train[:10])
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 10
        assert predictions.dtype in [np.float32, np.float64]
    
    def test_predict_model_not_found(self, model_trainer, sample_data):
        """Test prediction with non-existent model."""
        X, _ = sample_data
        
        with pytest.raises(ValueError, match="Model nonexistent_model not found"):
            model_trainer.predict('nonexistent_model', X)
    
    def test_get_feature_importance(self, model_trainer, sample_data):
        """Test getting feature importance."""
        X_train, y_train = sample_data
        
        # Create and train a tree-based model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        
        model_trainer.models['test_model'] = model
        
        # Get feature importance
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        importance = model_trainer.get_feature_importance('test_model', feature_names)
        
        assert isinstance(importance, dict)
        assert len(importance) == X_train.shape[1]
        assert all(isinstance(v, (int, float)) for v in importance.values())
    
    def test_get_feature_importance_no_support(self, model_trainer, sample_data):
        """Test feature importance with model that doesn't support it."""
        X_train, y_train = sample_data
        
        # Create and train an SVR model (doesn't have feature_importances_)
        from sklearn.svm import SVR
        model = SVR()
        model.fit(X_train, y_train)
        
        model_trainer.models['test_model'] = model
        
        # Get feature importance
        importance = model_trainer.get_feature_importance('test_model')
        
        assert importance == {}
    
    @patch('src.models.model_trainer.Task')
    def test_close_clearml_task(self, mock_task, model_trainer):
        """Test closing ClearML task."""
        mock_task_instance = Mock()
        mock_task.init.return_value = mock_task_instance
        
        # Initialize ClearML
        model_trainer.initialize_clearml()
        
        # Close task
        model_trainer.close_clearml_task()
        
        mock_task_instance.close.assert_called_once()
