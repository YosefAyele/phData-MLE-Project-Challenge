#!/bin/bash

# Scaling script for House Price Prediction API

set -e

REPLICAS=${1:-3}

echo "Scaling House Price Prediction API..."
echo "Target replicas: $REPLICAS"

# Scale the API service
echo "Scaling API service to $REPLICAS replicas..."
docker compose up -d --scale api=$REPLICAS

# Start load balancer
echo "Starting load balancer..."
docker compose --profile scaling up -d nginx

echo "Scaling completed!"
echo "Current status:"
docker compose ps

echo ""
echo "Load-balanced API available at: http://localhost:80"
echo "Documentation at: http://localhost:80/docs"
echo "Individual instances at: http://localhost:8000, etc."
