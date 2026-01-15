#!/usr/bin/env python3
"""
Run the Crop Price Prediction API server.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.api.app import app, initialize_app

if __name__ == '__main__':
    print("🚀 Starting Crop Price Prediction API...")
    
    # Initialize the app
    initialize_app()
    
    # Get API configuration
    from src.utils.config_loader import ConfigLoader
    config_loader = ConfigLoader("config/config.yaml")
    api_config = config_loader.get_api_config()
    
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 5000)
    debug = api_config.get('debug', False)
    
    print(f"🌐 Server running on http://{host}:{port}")
    print("📖 API Documentation: http://localhost:5000")
    print("🔍 Health Check: http://localhost:5000/health")
    print("📊 Available Models: http://localhost:5000/models")
    
    # Run the app
    app.run(host=host, port=port, debug=debug)
