import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config_loader import ConfigLoader
from src.models.model_trainer import ModelTrainer


@pytest.fixture
def simple_config():
    """Simple configuration for testing."""
    return {
        'models': {
            'random_forest': {
                'n_estimators': 5,
                'max_depth': 3,
                'random_state': 42
            }
        },
        'training': {
            'cross_validation_folds': 3
        }
    }


@pytest.fixture
def simple_config_loader(simple_config):
    """Create a simple config loader."""
    config = ConfigLoader()
    config.config = simple_config
    return config


@pytest.fixture
def sample_data():
    """Sample training data for testing."""
    np.random.seed(42)
    n_samples = 50
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 1000 + 1500
    
    return X, y


class TestModelTrainerSimple:
    """Simple test cases for ModelTrainer class."""
    
    def test_initialization(self, simple_config_loader):
        """Test model trainer initialization."""
        trainer = ModelTrainer(simple_config_loader)
        assert trainer.config is not None
        assert trainer.models == {}
        assert trainer.evaluation_results == {}
    
    def test_get_model_configs(self, simple_config_loader):
        """Test getting model configurations."""
        trainer = ModelTrainer(simple_config_loader)
        configs = trainer.get_model_configs()
        
        assert isinstance(configs, dict)
        assert 'random_forest' in configs
    
    def test_create_models(self, simple_config_loader):
        """Test model creation."""
        trainer = ModelTrainer(simple_config_loader)
        models = trainer.create_models()
        
        assert isinstance(models, dict)
        assert 'random_forest' in models
        
        # Check model type
        from sklearn.ensemble import RandomForestRegressor
        assert isinstance(models['random_forest'], RandomForestRegressor)
    
    def test_train_model(self, simple_config_loader, sample_data):
        """Test model training."""
        X_train, y_train = sample_data
        trainer = ModelTrainer(simple_config_loader)
        
        # Create a model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        # Train model
        trained_model = trainer.train_model(model, X_train, y_train, "test_model")
        
        # Check that model is trained
        assert hasattr(trained_model, 'predict')
        assert trained_model is model
    
    def test_evaluate_model(self, simple_config_loader, sample_data):
        """Test model evaluation."""
        X_train, y_train = sample_data
        X_test, y_test = X_train[:10], y_train[:10]  # Use subset for testing
        
        trainer = ModelTrainer(simple_config_loader)
        
        # Create and train a simple model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        results = trainer.evaluate_model(model, X_test, y_test, "test_model")
        
        # Check result structure
        assert isinstance(results, dict)
        assert 'mse' in results
        assert 'mae' in results
        assert 'r2' in results
        assert 'rmse' in results
        
        # Check metric types and ranges
        assert isinstance(results['mse'], (int, float))
        assert isinstance(results['mae'], (int, float))
        assert isinstance(results['r2'], (int, float))
        assert results['mse'] >= 0
        assert results['mae'] >= 0
        assert results['rmse'] >= 0
