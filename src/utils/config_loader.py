import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and manage configuration from YAML files."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dots)."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.get('data', {})
    
    def get_model_config(self, model_name: str = None) -> Dict[str, Any]:
        """Get model configuration."""
        models_config = self.get('models', {})
        if model_name:
            return models_config.get(model_name, {})
        return models_config
    
    def get_features(self) -> Dict[str, list]:
        """Get feature configuration."""
        return self.get('features', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.get('training', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.get('api', {})
    
    def get_clearml_config(self) -> Dict[str, Any]:
        """Get ClearML configuration."""
        return self.get('clearml', {})
