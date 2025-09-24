#!/bin/bash

# Build script for House Price Prediction API

set -e  # Exit on any error

echo " Building House Price Prediction API..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Please start Docker first."
    exit 1
fi

# Build the Docker image
echo "Building Docker image..."
docker build -t house-price-api:latest .

# Build with specific tag if provided
if [ ! -z "$1" ]; then
    echo "Tagging image as house-price-api:$1"
    docker tag house-price-api:latest house-price-api:$1
fi

echo "Build completed successfully!"
echo "Run with: docker run -p 8000:8000 house-price-api:latest"
echo "Or use docker-compose: docker-compose up"
