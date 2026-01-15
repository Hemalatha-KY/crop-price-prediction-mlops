import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import logging
from pathlib import Path
import joblib

from ..utils.config_loader import ConfigLoader


class DataPreprocessor:
    """Handle data preprocessing for crop price prediction."""
    
    def __init__(self, config_loader: ConfigLoader):
        self.config = config_loader
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = self.config.get('data.target_column')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load raw data from CSV file."""
        try:
            df = pd.read_csv(file_path)
            # Remove the first column if it's an unnamed index
            if df.columns[0] == 'Unnamed: 0':
                df = df.drop(df.columns[0], axis=1)
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
        
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        # Convert Date column to datetime if exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df = df.drop('Date', axis=1)
        
        self.logger.info(f"Data cleaned. Shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features using Label Encoding."""
        categorical_features = self.config.get('features.categorical', [])
        
        for feature in categorical_features:
            if feature in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[feature] = le.fit_transform(df[feature].astype(str))
                    self.label_encoders[feature] = le
                else:
                    if feature in self.label_encoders:
                        le = self.label_encoders[feature]
                        # Handle unseen labels
                        unique_values = set(df[feature].astype(str).unique())
                        known_values = set(le.classes_)
                        unknown_values = unique_values - known_values
                        
                        if unknown_values:
                            # Add unknown values to the encoder
                            new_classes = list(le.classes_) + list(unknown_values)
                            le.classes_ = np.array(new_classes)
                        
                        df[feature] = le.transform(df[feature].astype(str))
        
        return df
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
        numerical_features = self.config.get('features.numerical', [])
        
        # Add Year, Month, Day if they exist
        for col in ['Year', 'Month', 'Day']:
            if col in df.columns and col not in numerical_features:
                numerical_features.append(col)
        
        # Filter only existing columns
        existing_numerical = [col for col in numerical_features if col in df.columns]
        
        if existing_numerical:
            if fit:
                df[existing_numerical] = self.scaler.fit_transform(df[existing_numerical])
            else:
                df[existing_numerical] = self.scaler.transform(df[existing_numerical])
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variables."""
        # Get all feature columns
        categorical_features = self.config.get('features.categorical', [])
        numerical_features = self.config.get('features.numerical', [])
        
        # Add date features if they exist
        for col in ['Year', 'Month', 'Day']:
            if col in df.columns and col not in numerical_features:
                numerical_features.append(col)
        
        # Combine all features
        all_features = categorical_features + numerical_features
        self.feature_columns = [col for col in all_features if col in df.columns]
        
        # Find the target column dynamically
        target_col = None
        for col in df.columns:
            if 'Price' in col and '₹' in col:
                target_col = col
                break
        
        if target_col is None:
            raise ValueError("Target column (Price) not found in the dataset")
        
        # Separate features and target
        X = df[self.feature_columns]
        y = df[target_col]
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        test_size = self.config.get('data.test_size', 0.2)
        random_state = self.config.get('data.random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, file_path: str, save_path: str = None) -> Dict[str, Any]:
        """Complete preprocessing pipeline."""
        # Load and clean data
        df = self.load_data(file_path)
        df = self.clean_data(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=True)
        
        # Scale numerical features
        df = self.scale_numerical_features(df, fit=True)
        
        # Prepare features and target
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Save processed data if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            processed_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            joblib.dump(processed_data, save_path)
            self.logger.info(f"Processed data saved to {save_path}")
        
        # Save preprocessing artifacts
        self.save_artifacts()
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': self.feature_columns
        }
    
    def save_artifacts(self, artifacts_dir: str = "artifacts"):
        """Save preprocessing artifacts."""
        Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, f"{artifacts_dir}/scaler.joblib")
        
        # Save label encoders
        joblib.dump(self.label_encoders, f"{artifacts_dir}/label_encoders.joblib")
        
        # Save feature columns
        joblib.dump(self.feature_columns, f"{artifacts_dir}/feature_columns.joblib")
        
        self.logger.info(f"Preprocessing artifacts saved to {artifacts_dir}")
    
    def load_artifacts(self, artifacts_dir: str = "artifacts"):
        """Load preprocessing artifacts."""
        self.scaler = joblib.load(f"{artifacts_dir}/scaler.joblib")
        self.label_encoders = joblib.load(f"{artifacts_dir}/label_encoders.joblib")
        self.feature_columns = joblib.load(f"{artifacts_dir}/feature_columns.joblib")
        
        self.logger.info(f"Preprocessing artifacts loaded from {artifacts_dir}")
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessing artifacts."""
        # Clean data
        df = self.clean_data(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=False)
        
        # Scale numerical features
        df = self.scale_numerical_features(df, fit=False)
        
        # Return only the feature columns used during training
        return df[self.feature_columns]
