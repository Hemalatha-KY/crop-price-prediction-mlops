import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config_loader import ConfigLoader
from src.data.preprocessor import DataPreprocessor


@pytest.fixture
def simple_config():
    """Simple configuration for testing."""
    return {
        'data': {
            'target_column': 'Price (₹/ton)',
            'test_size': 0.2,
            'random_state': 42
        },
        'features': {
            'numerical': [
                'Temperature (°C)',
                'Rainfall (mm)'
            ],
            'categorical': [
                'State',
                'City'
            ]
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
    """Sample data for testing."""
    return pd.DataFrame({
        'Date': ['2023-01', '2023-02', '2023-03'],
        'State': ['Maharashtra', 'Gujarat', 'Maharashtra'],
        'City': ['Mumbai', 'Ahmedabad', 'Pune'],
        'Temperature (°C)': [25.5, 30.2, 28.1],
        'Rainfall (mm)': [100.5, 50.2, 75.8],
        'Price (₹/ton)': [1500.0, 1800.0, 1600.0]
    })


@pytest.fixture
def preprocessor(simple_config_loader):
    """Create a preprocessor instance."""
    return DataPreprocessor(simple_config_loader)


class TestDataPreprocessorSimple:
    """Simple test cases for DataPreprocessor class."""
    
    def test_initialization(self, simple_config_loader):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor(simple_config_loader)
        assert preprocessor.config is not None
        assert preprocessor.scaler is not None
        assert preprocessor.label_encoders == {}
        assert preprocessor.feature_columns == []
    
    def test_load_data(self, preprocessor, sample_data, tmp_path):
        """Test data loading."""
        # Save sample data to temporary file
        data_file = tmp_path / "test_data.csv"
        sample_data.to_csv(data_file, index=False)
        
        # Load data
        loaded_data = preprocessor.load_data(str(data_file))
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.shape == (3, 6)
        assert list(loaded_data.columns) == list(sample_data.columns)
    
    def test_clean_data(self, preprocessor, sample_data):
        """Test data cleaning."""
        # Add some missing values
        dirty_data = sample_data.copy()
        dirty_data.loc[0, 'Temperature (°C)'] = np.nan
        dirty_data.loc[1, 'State'] = np.nan
        
        # Clean data
        cleaned_data = preprocessor.clean_data(dirty_data)
        
        # Check no missing values
        assert cleaned_data.isnull().sum().sum() == 0
        
        # Check Date column processing
        assert 'Date' not in cleaned_data.columns
        assert 'Year' in cleaned_data.columns
        assert 'Month' in cleaned_data.columns
        assert 'Day' in cleaned_data.columns
    
    def test_encode_categorical_features(self, preprocessor, sample_data):
        """Test categorical feature encoding."""
        # Clean data first to add date features
        cleaned_data = preprocessor.clean_data(sample_data)
        
        # Encode categorical features
        encoded_data = preprocessor.encode_categorical_features(cleaned_data, fit=True)
        
        # Check categorical columns are encoded
        categorical_features = preprocessor.config.get('features.categorical', [])
        for feature in categorical_features:
            if feature in encoded_data.columns:
                assert encoded_data[feature].dtype in ['int64', 'float64']
    
    def test_scale_numerical_features(self, preprocessor, sample_data):
        """Test numerical feature scaling."""
        # Clean and encode data first
        cleaned_data = preprocessor.clean_data(sample_data)
        encoded_data = preprocessor.encode_categorical_features(cleaned_data, fit=True)
        
        # Scale numerical features
        scaled_data = preprocessor.scale_numerical_features(encoded_data, fit=True)
        
        # Check numerical columns are scaled
        numerical_features = preprocessor.config.get('features.numerical', [])
        for feature in numerical_features:
            if feature in scaled_data.columns:
                assert scaled_data[feature].dtype in ['int64', 'float64']
    
    def test_prepare_features(self, preprocessor, sample_data):
        """Test feature preparation."""
        # Process data through the pipeline
        cleaned_data = preprocessor.clean_data(sample_data)
        encoded_data = preprocessor.encode_categorical_features(cleaned_data, fit=True)
        scaled_data = preprocessor.scale_numerical_features(encoded_data, fit=True)
        
        # Prepare features
        X, y = preprocessor.prepare_features(scaled_data)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y) == 3
        assert 'Price (₹/ton)' in sample_data.columns
    
    def test_split_data(self, preprocessor, sample_data):
        """Test data splitting."""
        # Process data through the pipeline
        cleaned_data = preprocessor.clean_data(sample_data)
        encoded_data = preprocessor.encode_categorical_features(cleaned_data, fit=True)
        scaled_data = preprocessor.scale_numerical_features(encoded_data, fit=True)
        X, y = preprocessor.prepare_features(scaled_data)
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert X_train.shape[1] == X_test.shape[1]
