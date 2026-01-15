# Crop Price Prediction MLOps Project

A comprehensive end-to-end MLOps project for predicting crop prices using machine learning models, complete with experiment tracking, data versioning, CI/CD, and deployment capabilities.

## 🌾 Project Overview

This project implements a complete MLOps pipeline for crop price prediction using:
- **Machine Learning Models**: Random Forest, SVR, XGBoost
- **MLOps Tools**: DVC for data versioning, ClearML for experiment tracking
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Deployment**: Docker + Flask API, deployable on Render
- **Dataset**: Indian crop price data with 13 features

## 📊 Dataset

The dataset contains the following columns:
- `Date`: Date of observation
- `State`: Indian state
- `City`: City within the state
- `Crop Type`: Type of crop (Wheat, Rice, etc.)
- `Season`: Growing season (Kharif, Rabi, etc.)
- `Temperature (°C)`: Temperature during the period
- `Rainfall (mm)`: Rainfall amount
- `Supply Volume (tons)`: Total supply volume
- `Demand Volume (tons)`: Total demand volume
- `Transportation Cost (₹/ton)`: Cost per ton for transportation
- `Fertilizer Usage (kg/hectare)`: Fertilizer usage per hectare
- `Pest Infestation (0-1)`: Pest infestation level (normalized)
- `Market Competition (0-1)`: Market competition level (normalized)
- `Price (₹/ton)`: Target variable - price per ton

## 🏗️ Project Structure

```
MLOPS_FINAL/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessor.py          # Data preprocessing pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_trainer.py         # ML model training and evaluation
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py                   # Flask API application
│   └── utils/
│       ├── __init__.py
│       └── config_loader.py         # Configuration management
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py         # Preprocessor tests
│   └── test_models.py               # Model tests
├── config/
│   └── config.yaml                  # Project configuration
├── models/                          # Trained models directory
├── artifacts/                       # Preprocessing artifacts
├── logs/                           # Application logs
├── .github/
│   └── workflows/
│       └── ci-cd.yml               # GitHub Actions workflow
├── dataset_indian_crop_price.csv   # Raw dataset
├── requirements.txt                # Python dependencies
├── setup.py                       # Package setup
├── train.py                       # Training script
├── run_api.py                     # API server script
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose configuration
└── README.md                      # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Docker (optional, for containerized deployment)
- ClearML account (for experiment tracking)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd MLOPS_FINAL
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Initialize DVC:**
   ```bash
   dvc pull  # Pull the dataset from remote storage
   ```

### Training Models

1. **Train all models:**
   ```bash
   python train.py
   ```

2. **Train with custom parameters:**
   ```bash
   python train.py --config config/config.yaml --data dataset_indian_crop_price.csv
   ```

### Running the API

1. **Start the API server:**
   ```bash
   python run_api.py
   ```

2. **Access the API:**
   - Base URL: `http://localhost:5000`
   - Health check: `http://localhost:5000/health`
   - API documentation: `http://localhost:5000`

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessor.py -v
```

## 🐳 Docker Deployment

### Build and Run with Docker

1. **Build the Docker image:**
   ```bash
   docker build -t crop-price-prediction .
   ```

2. **Run the container:**
   ```bash
   docker run -p 5000:5000 crop-price-prediction
   ```

### Use Docker Compose

```bash
docker-compose up --build
```

## 📡 API Endpoints

### Predict Single Price
```http
POST /predict
Content-Type: application/json

{
  "State": "Maharashtra",
  "City": "Mumbai",
  "Crop Type": "Wheat",
  "Season": "Kharif",
  "Temperature (°C)": 25.5,
  "Rainfall (mm)": 100.5,
  "Supply Volume (tons)": 1000.0,
  "Demand Volume (tons)": 800.0,
  "Transportation Cost (₹/ton)": 200.0,
  "Fertilizer Usage (kg/hectare)": 100.0,
  "Pest Infestation (0-1)": 0.2,
  "Market Competition (0-1)": 0.7
}
```

### Batch Prediction
```http
POST /predict/batch
Content-Type: application/json

{
  "data": [
    {
      "State": "Maharashtra",
      "City": "Mumbai",
      "Crop Type": "Wheat",
      "Season": "Kharif",
      "Temperature (°C)": 25.5,
      "Rainfall (mm)": 100.5,
      "Supply Volume (tons)": 1000.0,
      "Demand Volume (tons)": 800.0,
      "Transportation Cost (₹/ton)": 200.0,
      "Fertilizer Usage (kg/hectare)": 100.0,
      "Pest Infestation (0-1)": 0.2,
      "Market Competition (0-1)": 0.7
    }
  ]
}
```

### Other Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `GET /models` - List available models
- `GET /feature_importance` - Get feature importance

## 🔄 CI/CD Pipeline

The project includes a comprehensive CI/CD pipeline using GitHub Actions:

### Triggers
- Push to `main` or `develop` branches
- Pull requests to `main` branch

### Jobs
1. **Test**: Runs tests across Python 3.8, 3.9, 3.10
2. **Train**: Trains models on main branch
3. **Deploy**: Builds and deploys Docker image

### Environment Variables Required
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password
- `RENDER_DEPLOY_WEBHOOK`: Render deployment webhook

## 📊 Experiment Tracking with ClearML

The project uses ClearML for experiment tracking:

1. **Set up ClearML:**
   ```bash
   pip install clearml
   clearml-init
   ```

2. **View experiments:**
   - Access ClearML web UI
   - View model performance metrics
   - Compare different model configurations
   - Track hyperparameters and results

## 📈 Model Performance

The project trains and evaluates three models:
- **Random Forest**: Ensemble method with feature importance
- **SVR**: Support Vector Regression for non-linear relationships
- **XGBoost**: Gradient boosting for high performance

Metrics tracked:
- R² Score
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Cross-validation scores

## 🌐 Deployment on Render

### Manual Deployment
1. Connect your GitHub repository to Render
2. Set environment variables
3. Deploy using the Docker configuration

### Automatic Deployment
The CI/CD pipeline automatically deploys to Render when code is pushed to the main branch.

## 🔧 Configuration

All configuration is managed through `config/config.yaml`:

- Data paths and parameters
- Model hyperparameters
- Feature selections
- Training parameters
- API settings
- ClearML configuration

## 📝 Logging

The application uses Python's logging module:
- Logs are saved to the `logs/` directory
- Different log levels for debugging
- Structured logging for monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Common Issues

1. **DVC data not found:**
   ```bash
   dvc pull
   ```

2. **ClearML connection issues:**
   - Check ClearML configuration
   - Verify network connectivity

3. **Docker build fails:**
   - Check Dockerfile syntax
   - Verify base image availability

4. **API returns 500 error:**
   - Check logs in `logs/` directory
   - Verify model files exist in `models/` directory

### Getting Help

- Check the logs for detailed error messages
- Ensure all dependencies are installed
- Verify configuration files are correct
- Check that models are trained before running the API

## 📚 Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [ClearML Documentation](https://clear.ml/docs)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Render Documentation](https://render.com/docs)
D e p l o y m e n t   t e s t   2 0 2 6 - 0 1 - 1 5   1 8 : 2 9 : 1 4  
 