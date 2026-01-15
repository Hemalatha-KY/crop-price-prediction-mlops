import requests
import json

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test health check endpoint."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_models():
    """Test models endpoint."""
    response = requests.get(f"{BASE_URL}/models")
    print("Available Models:")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_prediction():
    """Test single prediction endpoint."""
    test_data = {
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
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print("Single Prediction:")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_batch_prediction():
    """Test batch prediction endpoint."""
    test_data = {
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
            },
            {
                "State": "Gujarat",
                "City": "Ahmedabad",
                "Crop Type": "Rice",
                "Season": "Rabi",
                "Temperature (°C)": 30.2,
                "Rainfall (mm)": 50.2,
                "Supply Volume (tons)": 1500.0,
                "Demand Volume (tons)": 1200.0,
                "Transportation Cost (₹/ton)": 150.0,
                "Fertilizer Usage (kg/hectare)": 80.0,
                "Pest Infestation (0-1)": 0.1,
                "Market Competition (0-1)": 0.6
            }
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    print("Batch Prediction:")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_feature_importance():
    """Test feature importance endpoint."""
    response = requests.get(f"{BASE_URL}/feature_importance")
    print("Feature Importance:")
    print(f"Status: {response.status_code}")
    response_data = response.json()
    print(f"Model: {response_data.get('model')}")
    print("Top 5 Features:")
    importance = response_data.get('feature_importance', {})
    for i, (feature, score) in enumerate(list(importance.items())[:5]):
        print(f"  {i+1}. {feature}: {score:.4f}")
    print()

if __name__ == "__main__":
    print("🧪 Testing Crop Price Prediction API")
    print("=" * 50)
    
    try:
        test_health_check()
        test_models()
        test_prediction()
        test_batch_prediction()
        test_feature_importance()
        
        print("✅ All API tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API server.")
        print("Make sure the API is running on http://localhost:5000")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
