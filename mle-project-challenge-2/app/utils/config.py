import os
import pathlib
from typing import List

# Paths
BASE_DIR = pathlib.Path(__file__).parent.parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

# Model files
MODEL_PATH = MODEL_DIR / "model.pkl"
MODEL_FEATURES_PATH = MODEL_DIR / "model_features.json"

# Data files
ZIPCODE_DEMOGRAPHICS_PATH = DATA_DIR / "zipcode_demographics.csv"
FUTURE_EXAMPLES_PATH = DATA_DIR / "future_unseen_examples.csv"

# API Configuration
API_TITLE = "House Price Prediction API"
API_DESCRIPTION = """
A REST API for predicting house prices in the Seattle area using machine learning.

This API serves a trained K-Nearest Neighbors model that predicts house prices based on:
- House characteristics (bedrooms, bathrooms, square footage, etc.)
- Demographic data from U.S. Census (automatically joined by zipcode)

## Features

- **Main Prediction Endpoint**: `/predict` - Full feature prediction
- **Minimal Prediction Endpoint**: `/predict/minimal` - Prediction with only essential features
- **Health Check**: `/health` - Service health status
- **Model Information**: `/model/info` - Model metadata and performance metrics
"""
API_VERSION = "1.0.0"
API_CONTACT = {
    "name": "Sound Realty ML Team",
    "email": "ml-team@soundrealty.com"
}

# Model Configuration
MODEL_VERSION = "1.0.0"
MODEL_TYPE = "KNeighborsRegressor"

# Essential features for minimal prediction endpoint (based on model evaluation)
# These are the most important features identified from feature importance analysis
ESSENTIAL_FEATURES = [
    "sqft_living",
    "bedrooms", 
    "bathrooms",
    "sqft_above",
    "floors",
    "sqft_lot",
    "sqft_basement"
]

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
WORKERS = int(os.getenv("WORKERS", 1))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# CORS Configuration
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

# Request/Response Configuration
MAX_PREDICTIONS_PER_REQUEST = int(os.getenv("MAX_PREDICTIONS_PER_REQUEST", 100))
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", 30))

# Performance Configuration
ENABLE_PREDICTION_CACHING = os.getenv("ENABLE_PREDICTION_CACHING", "false").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 3600))  # 1 hour

# Logging Configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.getenv("LOG_FILE", None)  # None means console only

# Health Check Configuration
HEALTH_CHECK_INTERVAL_SECONDS = int(os.getenv("HEALTH_CHECK_INTERVAL_SECONDS", 60))

class Config:

    def __init__(self):
        self.validate_paths()
    
    def validate_paths(self):
        required_paths = [
            MODEL_PATH,
            MODEL_FEATURES_PATH,
            ZIPCODE_DEMOGRAPHICS_PATH
        ]
        
        missing_paths = []
        for path in required_paths:
            if not path.exists():
                missing_paths.append(str(path))
        
        if missing_paths:
            raise FileNotFoundError(
                f"Required files not found: {missing_paths}. "
                "Please run 'python create_model.py' first to generate model artifacts."
            )
    
    @property
    def is_development(self) -> bool:
        return os.getenv("ENVIRONMENT", "development") == "development"
    
    @property
    def is_production(self) -> bool:
        return os.getenv("ENVIRONMENT", "development") == "production"

# Global config instance
config = Config()
