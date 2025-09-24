# House Price Prediction API

A FastAPI-based REST service for predicting house prices in the Seattle area using machine learning.

## 🚀 Quick Start to run and see it in action

```bash
git clone <repo-url>
cd mle-project-challenge-2
./scripts/build.sh && ./scripts/deploy.sh
```

API available at: http://localhost:8000  
Documentation: http://localhost:8000/docs

## Quick Start

### 1. Install Dependencies

```bash
# Activate conda environment
conda activate housing

# Install FastAPI dependencies
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Option 1: Using the run script
python run_api.py

# Option 2: Using uvicorn directly
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Option 3: Using the main module
python -m app.main
```

The API will be available at:
- **API Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

### 3. Test the API

```bash
# Run the test script
python test_api.py
```

## API Endpoints

### Core Prediction Endpoints

#### POST `/predict`
Predict house price using full feature set from `future_unseen_examples.csv`.

**Request Body:**
```json
{
  "house_features": {
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_living": 2000,
    "sqft_lot": 8000,
    "floors": 2.0,
    "waterfront": 0,
    "view": 0,
    "condition": 4,
    "grade": 8,
    "sqft_above": 1800,
    "sqft_basement": 200,
    "yr_built": 1990,
    "yr_renovated": 0,
    "zipcode": "98115",
    "lat": 47.6974,
    "long": -122.313,
    "sqft_living15": 1950,
    "sqft_lot15": 8500
  }
}
```

**Response:**
```json
{
  "predicted_price": 685432.50,
  "prediction_metadata": {
    "model_version": "1.0.0",
    "prediction_timestamp": "2024-01-15T10:30:45.123456",
    "features_used": ["bedrooms", "bathrooms", ...],
    "zipcode_demographics_found": true,
    "processing_time_ms": 15.2
  }
}
```

#### POST `/predict/minimal`
Predict house price using only essential features.

**Request Body:**
```json
{
  "house_features": {
    "sqft_living": 2000,
    "bedrooms": 3,
    "bathrooms": 2.5,
    "sqft_above": 1800,
    "floors": 2.0,
    "sqft_lot": 8000,
    "sqft_basement": 200,
    "zipcode": "98115"
  }
}
```

#### POST `/predict/batch`
Predict prices for multiple houses (up to 100 per request).

### Information Endpoints

#### GET `/health`
Service health check.

#### GET `/model/info`
Information about the loaded model.

#### GET `/model/features`
List of features required by the model.

#### GET `/zipcodes`
List of available zipcodes with demographic data.

#### GET `/demographics/summary`
Summary of available demographic data.

#### GET `/stats`
Service usage statistics.

## Example Usage

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "house_features": {
      "bedrooms": 3,
      "bathrooms": 2.0,
      "sqft_living": 1800,
      "sqft_lot": 7000,
      "floors": 1.0,
      "waterfront": 0,
      "view": 0,
      "condition": 3,
      "grade": 7,
      "sqft_above": 1800,
      "sqft_basement": 0,
      "yr_built": 1975,
      "yr_renovated": 0,
      "zipcode": "98115",
      "lat": 47.6974,
      "long": -122.313,
      "sqft_living15": 1560,
      "sqft_lot15": 7000
    }
  }'
```

### Using Python requests

```python
import requests

# API base URL
api_url = "http://localhost:8000"

# Example house data
house_data = {
    "house_features": {
        "bedrooms": 4,
        "bathrooms": 2.5,
        "sqft_living": 2200,
        "sqft_lot": 8500,
        "floors": 2.0,
        "waterfront": 0,
        "view": 0,
        "condition": 4,
        "grade": 8,
        "sqft_above": 2200,
        "sqft_basement": 0,
        "yr_built": 1985,
        "yr_renovated": 0,
        "zipcode": "98115",
        "lat": 47.6974,
        "long": -122.313,
        "sqft_living15": 2000,
        "sqft_lot15": 8000
    }
}

# Make prediction
response = requests.post(f"{api_url}/predict", json=house_data)
result = response.json()

print(f"Predicted price: ${result['predicted_price']:,.2f}")
```

## Configuration

The API can be configured using environment variables:

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `LOG_LEVEL`: Logging level (default: info)
- `ENVIRONMENT`: Environment mode (development/production)
- `MAX_PREDICTIONS_PER_REQUEST`: Max predictions per batch (default: 100)

## Error Handling

The API provides detailed error responses:

```json
{
  "error": "ValidationError",
  "message": "Invalid zipcode format",
  "timestamp": "2024-01-15T10:30:45.123456",
  "details": {...}
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (validation error)
- `422`: Unprocessable Entity (invalid data)
- `500`: Internal Server Error
- `503`: Service Unavailable (model not loaded)

## Model Information

The current model:
- **Type**: K-Nearest Neighbors Regressor with RobustScaler
- **Features**: 33 features (house characteristics + demographics)
- **Performance**: R² ≈ 0.73, MAPE ≈ 18%
- **Training Data**: ~21K house sales in Seattle area

## Architecture

```
├── app/
│   ├── main.py              # FastAPI application
│   ├── models/              # Pydantic models
│   │   └── prediction.py
│   ├── services/            # Business logic
│   │   ├── data_service.py  # Demographics & preprocessing
│   │   └── model_service.py # ML model management
│   └── utils/
│       └── config.py        # Configuration
├── model/                   # Model artifacts
├── data/                    # Data files
└── test_api.py             # Test script
```

## Development

### Running Tests

```bash
# Run the comprehensive test suite
python test_api.py

# For unit tests (if implemented)
pytest tests/
```

### Adding New Features

1. Add new endpoints in `app/main.py`
2. Define request/response models in `app/models/prediction.py`
3. Implement business logic in appropriate service modules
4. Update configuration in `app/utils/config.py`

### Monitoring

The API provides several endpoints for monitoring:
- `/health`: Health status
- `/stats`: Usage statistics
- `/model/info`: Model performance metrics

Logs are structured and include:
- Request/response timing
- Prediction details
- Error information

## Deployment Notes

For production deployment:

1. Use a production WSGI server (e.g., Gunicorn)
2. Set `ENVIRONMENT=production`
3. Configure proper logging
4. Add authentication/authorization if needed
5. Consider using Redis for caching predictions
6. Monitor model performance and retrain as needed
