import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config_loader import ConfigLoader
from src.data.preprocessor import DataPreprocessor


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'data': {
            'target_column': 'Price (₹/ton)',
            'test_size': 0.2,
            'random_state': 42
        },
        'features': {
            'numerical': [
                'Temperature (°C)',
                'Rainfall (mm)',
                'Supply Volume (tons)',
                'Demand Volume (tons)',
                'Transportation Cost (₹/ton)',
                'Fertilizer Usage (kg/hectare)',
                'Pest Infestation (0-1)',
                'Market Competition (0-1)'
            ],
            'categorical': [
                'State',
                'City',
                'Crop Type',
                'Season'
            ]
        }
    }


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return pd.DataFrame({
        'Date': ['2023-01', '2023-02', '2023-03'],
        'State': ['Maharashtra', 'Gujarat', 'Maharashtra'],
        'City': ['Mumbai', 'Ahmedabad', 'Pune'],
        'Crop Type': ['Wheat', 'Rice', 'Wheat'],
        'Season': ['Kharif', 'Rabi', 'Kharif'],
        'Temperature (°C)': [25.5, 30.2, 28.1],
        'Rainfall (mm)': [100.5, 50.2, 75.8],
        'Supply Volume (tons)': [1000.0, 1500.0, 1200.0],
        'Demand Volume (tons)': [800.0, 1200.0, 900.0],
        'Transportation Cost (₹/ton)': [200.0, 150.0, 180.0],
        'Fertilizer Usage (kg/hectare)': [100.0, 80.0, 90.0],
        'Pest Infestation (0-1)': [0.2, 0.1, 0.15],
        'Market Competition (0-1)': [0.7, 0.6, 0.65],
        'Price (₹/ton)': [1500.0, 1800.0, 1600.0]
    })


@pytest.fixture
def config_loader(sample_config):
    """Create a mock config loader."""
    mock_config = Mock(spec=ConfigLoader)
    mock_config.get.side_effect = lambda key, default=None: sample_config.get(key, default)
    return mock_config


@pytest.fixture
def preprocessor(config_loader):
    """Create a preprocessor instance."""
    return DataPreprocessor(config_loader)


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def test_load_data(self, preprocessor, sample_data, tmp_path):
        """Test data loading."""
        # Save sample data to temporary file
        data_file = tmp_path / "test_data.csv"
        sample_data.to_csv(data_file, index=False)
        
        # Load data
        loaded_data = preprocessor.load_data(str(data_file))
        
        assert isinstance(loaded_data, pd.DataFrame)
        assert loaded_data.shape == (3, 14)
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
        
        # Check numerical columns are scaled (approximately zero mean, unit variance)
        numerical_features = preprocessor.config.get('features.numerical', [])
        # Add date features if they exist
        for col in ['Year', 'Month', 'Day']:
            if col in scaled_data.columns and col not in numerical_features:
                numerical_features.append(col)
        
        for feature in numerical_features:
            if feature in scaled_data.columns:
                # Check that scaling was applied (values are not the same as original)
                assert not scaled_data[feature].equals(encoded_data[feature])
    
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
        assert preprocessor.target_column in sample_data.columns
    
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
    
    def test_preprocess_pipeline(self, preprocessor, sample_data, tmp_path):
        """Test complete preprocessing pipeline."""
        # Save sample data to temporary file
        data_file = tmp_path / "test_data.csv"
        sample_data.to_csv(data_file, index=False)
        
        # Save path for processed data
        save_path = tmp_path / "processed_data.joblib"
        
        # Run pipeline
        result = preprocessor.preprocess_pipeline(str(data_file), str(save_path))
        
        # Check result structure
        assert 'X_train' in result
        assert 'X_test' in result
        assert 'y_train' in result
        assert 'y_test' in result
        assert 'feature_columns' in result
        
        # Check processed data file was created
        assert save_path.exists()
    
    def test_save_and_load_artifacts(self, preprocessor, tmp_path):
        """Test saving and loading preprocessing artifacts."""
        # Process some data first to create artifacts
        sample_data = pd.DataFrame({
            'State': ['Maharashtra', 'Gujarat'],
            'Temperature (°C)': [25.5, 30.2],
            'Price (₹/ton)': [1500.0, 1800.0]
        })
        
        cleaned_data = preprocessor.clean_data(sample_data)
        encoded_data = preprocessor.encode_categorical_features(cleaned_data, fit=True)
        scaled_data = preprocessor.scale_numerical_features(encoded_data, fit=True)
        
        # Save artifacts
        artifacts_dir = tmp_path / "artifacts"
        preprocessor.save_artifacts(str(artifacts_dir))
        
        # Check artifacts exist
        assert (artifacts_dir / "scaler.joblib").exists()
        assert (artifacts_dir / "label_encoders.joblib").exists()
        assert (artifacts_dir / "feature_columns.joblib").exists()
        
        # Create new preprocessor and load artifacts
        new_preprocessor = DataPreprocessor(preprocessor.config)
        new_preprocessor.load_artifacts(str(artifacts_dir))
        
        # Check artifacts were loaded
        assert new_preprocessor.feature_columns == preprocessor.feature_columns
        assert hasattr(new_preprocessor, 'scaler')
        assert hasattr(new_preprocessor, 'label_encoders')
