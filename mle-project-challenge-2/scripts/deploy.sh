#!/bin/bash

# Deployment script for House Price Prediction API

set -e  # Exit on any error

ENVIRONMENT=${1:-production}
COMPOSE_FILE="docker-compose.yml"

echo "Deploying House Price Prediction API..."
echo "Environment: $ENVIRONMENT"

# Check if model files exist
if [ ! -f "model/model.pkl" ]; then
    echo " Model file not found. Please run 'python create_model.py' first."
    exit 1
fi

if [ ! -f "model/model_features.json" ]; then
    echo "Model features file not found. Please run 'python create_model.py' first."
    exit 1
fi

# Copy environment file if it exists
if [ -f ".env.$ENVIRONMENT" ]; then
    echo "Using environment file: .env.$ENVIRONMENT"
    cp ".env.$ENVIRONMENT" ".env"
fi

# Build and start services
echo "Building and starting services..."
docker compose -f $COMPOSE_FILE build
docker compose -f $COMPOSE_FILE up -d

# Wait for health check
echo "Waiting for service to be healthy..."
sleep 10

# Check health
for i in {1..30}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "Service is healthy!"
        break
    fi
    echo "Waiting for service... ($i/30)"
    sleep 2
done

# Show status
echo "Service status:"
docker compose -f $COMPOSE_FILE ps

echo "Deployment completed!"
echo "API available at: http://localhost:8000"
echo "Documentation at: http://localhost:8000/docs"
