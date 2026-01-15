#!/usr/bin/env python3
"""
Main training script for crop price prediction models.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config_loader import ConfigLoader
from src.data.preprocessor import DataPreprocessor
from src.models.model_trainer import ModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Train crop price prediction models")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--data", type=str, default="dataset_indian_crop_price.csv",
                       help="Path to training data")
    parser.add_argument("--output", type=str, default="data/processed/processed_data.joblib",
                       help="Path to save processed data")
    
    args = parser.parse_args()
    
    print("🌾 Starting Crop Price Prediction Model Training")
    print("=" * 50)
    
    # Load configuration
    print("📋 Loading configuration...")
    config_loader = ConfigLoader(args.config)
    
    # Initialize preprocessor
    print("🔧 Initializing data preprocessor...")
    preprocessor = DataPreprocessor(config_loader)
    
    # Preprocess data
    print("📊 Preprocessing data...")
    processed_data = preprocessor.preprocess_pipeline(args.data, args.output)
    
    # Initialize model trainer
    print("🤖 Initializing model trainer...")
    trainer = ModelTrainer(config_loader)
    
    # Train models
    print("🚀 Training models...")
    training_results = trainer.train_all_models(
        processed_data['X_train'],
        processed_data['y_train'],
        processed_data['X_test'],
        processed_data['y_test']
    )
    
    # Save models
    print("💾 Saving models...")
    trainer.save_all_models()
    
    # Print results
    print("\n📈 Training Results:")
    print("-" * 30)
    for model_name, results in training_results['evaluation_results'].items():
        print(f"\n{model_name.upper()}:")
        print(f"  R² Score: {results['r2']:.4f}")
        print(f"  MSE: {results['mse']:.4f}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  CV R²: {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
    
    print(f"\n🏆 Best Model: {training_results['best_model']['name']}")
    print(f"   R² Score: {training_results['best_model']['results']['r2']:.4f}")
    
    # Close ClearML task
    trainer.close_clearml_task()
    
    print("\n✅ Training completed successfully!")
    print(f"📁 Models saved in 'models/' directory")
    print(f"📁 Artifacts saved in 'artifacts/' directory")


if __name__ == "__main__":
    main()
