#!/bin/bash

# Production startup script using Gunicorn

echo "Starting House Price Prediction API in production mode..."

# Check if model files exist
if [ ! -f "model/model.pkl" ]; then
    echo " Model file not found. Running model creation..."
    python create_model.py
fi

# Start with Gunicorn
exec gunicorn app.main:app \
    --config gunicorn.conf.py \
    --worker-class uvicorn.workers.UvicornWorker
