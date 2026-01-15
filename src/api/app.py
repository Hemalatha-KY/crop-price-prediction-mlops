from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
import logging
from typing import Dict, Any, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import ConfigLoader
from src.data.preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer


app = Flask(__name__)
CORS(app)

# Global variables
config_loader = None
preprocessor = None
model_trainer = None
best_model_name = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_app():
    """Initialize the Flask app with models and configuration."""
    global config_loader, preprocessor, model_trainer, best_model_name
    
    try:
        # Load configuration
        config_loader = ConfigLoader("config/config.yaml")
        
        # Initialize preprocessor and load artifacts
        preprocessor = DataPreprocessor(config_loader)
        preprocessor.load_artifacts("artifacts")
        
        # Initialize model trainer
        model_trainer = ModelTrainer(config_loader)
        
        # Load the best model (default to random_forest if no best model info)
        available_models = ["random_forest", "svr", "xgboost"]
        for model_name in available_models:
            try:
                model_trainer.load_model(model_name, "models")
                best_model_name = model_name
                logger.info(f"Loaded model: {model_name}")
                break
            except FileNotFoundError:
                continue
        
        if not best_model_name:
            raise ValueError("No trained models found. Please train models first.")
        
        logger.info(f"App initialized successfully with model: {best_model_name}")
        
    except Exception as e:
        logger.error(f"Failed to initialize app: {e}")
        raise


@app.route('/')
def home():
    """Home page with API documentation."""
    return jsonify({
        "message": "Crop Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make price predictions",
            "/predict/batch": "POST - Make batch predictions",
            "/models": "GET - List available models",
            "/health": "GET - Health check"
        },
        "current_model": best_model_name
    })


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": best_model_name is not None,
        "current_model": best_model_name
    })


@app.route('/models')
def list_models():
    """List available models."""
    available_models = []
    model_dir = Path("models")
    
    if model_dir.exists():
        for file_path in model_dir.glob("*.joblib"):
            model_name = file_path.stem
            available_models.append(model_name)
    
    return jsonify({
        "available_models": available_models,
        "current_model": best_model_name
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Make single prediction."""
    try:
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Add date features if they were used during training
        if 'Year' in preprocessor.feature_columns:
            df['Date'] = '2023-01-01'  # Default date
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df = df.drop('Date', axis=1)
        
        # Preprocess the data
        processed_data = preprocessor.transform_new_data(df)
        
        # Make prediction
        prediction = model_trainer.predict(best_model_name, processed_data)
        
        return jsonify({
            "prediction": float(prediction[0]),
            "model_used": best_model_name,
            "input_data": data
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Make batch predictions."""
    try:
        # Get input data
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({"error": "No input data provided. Expected format: {'data': [...]}"}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Add date features if they were used during training
        if 'Year' in preprocessor.feature_columns:
            df['Date'] = '2023-01-01'  # Default date
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df = df.drop('Date', axis=1)
        
        # Preprocess the data
        processed_data = preprocessor.transform_new_data(df)
        
        # Make predictions
        predictions = model_trainer.predict(best_model_name, processed_data)
        
        return jsonify({
            "predictions": predictions.tolist(),
            "model_used": best_model_name,
            "count": len(predictions)
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/feature_importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance for the current model."""
    try:
        feature_importance = model_trainer.get_feature_importance(
            best_model_name, 
            preprocessor.feature_columns
        )
        
        # Sort by importance
        sorted_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return jsonify({
            "model": best_model_name,
            "feature_importance": sorted_importance
        })
        
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Initialize the app
    initialize_app()
    
    # Get API configuration
    api_config = config_loader.get_api_config()
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 5000)
    debug = api_config.get('debug', False)
    
    # Run the app
    app.run(host=host, port=port, debug=debug)
