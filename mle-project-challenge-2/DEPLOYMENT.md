# Deployment Guide

Complete guide for deploying the House Price Prediction API with Docker and scaling options.

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Model trained (run `python create_model.py` if needed)

### Basic Deployment

```bash
# Build and deploy
./scripts/build.sh
./scripts/deploy.sh

# Or manually
docker-compose up -d
```

The API will be available at http://localhost:8000

## Docker Deployment Options

### 1. Single Container

```bash
# Build image
docker build -t house-price-api .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/model:/app/model:ro \
  -v $(pwd)/data:/app/data:ro \
  --name house-price-api \
  house-price-api:latest
```

### 2. Docker Compose (Recommended)

```bash
# Development mode
docker-compose --profile dev up -d

# Production mode
docker-compose up -d

# With load balancing
docker-compose --profile scaling up -d
```

### 3. Scaled Deployment

```bash
# Scale to 3 replicas with load balancer
./scripts/scale.sh 3

# Manual scaling
docker-compose up -d --scale api=3
docker-compose --profile scaling up -d nginx
```

## Configuration

### Environment Variables

Key environment variables for configuration:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | 0.0.0.0 | Server host |
| `PORT` | 8000 | Server port |
| `WORKERS` | 1 | Number of worker processes |
| `LOG_LEVEL` | info | Logging level |
| `ENVIRONMENT` | production | Environment mode |
| `MAX_PREDICTIONS_PER_REQUEST` | 100 | Batch limit |

### Environment Files

- `.env.development` - Development settings
- `.env.production` - Production settings
- `.env.example` - Template file

## Production Deployment

### Using Gunicorn

```bash
# Install production dependencies
pip install gunicorn

# Start with Gunicorn
./start-production.sh

# Or manually
gunicorn app.main:app --config gunicorn.conf.py
```

### Resource Limits

Recommended resource allocation:

- **Development**: 0.5 CPU, 512MB RAM
- **Production**: 1-2 CPU, 1-2GB RAM
- **High Load**: 2-4 CPU, 2-4GB RAM

### Health Checks

Built-in health check endpoint: `/health`

Docker health check included:
```dockerfile
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1
```

## Scaling Strategies

### Horizontal Scaling

1. **Docker Compose Scaling**
   ```bash
   docker-compose up -d --scale api=3
   ```

2. **Load Balancer**
   ```bash
   docker-compose --profile scaling up -d
   ```

3. **Multiple Instances**
   ```bash
   # Instance 1
   docker run -d -p 8001:8000 house-price-api
   # Instance 2
   docker run -d -p 8002:8000 house-price-api
   # Instance 3
   docker run -d -p 8003:8000 house-price-api
   ```

### Load Balancing

Nginx configuration included for:
- Round-robin load balancing
- Health checks
- Failover handling
- Connection pooling

Access load-balanced API at: http://localhost:80

### Auto-scaling (Cloud Deployment)

For cloud deployment, consider:

1. **Kubernetes**
   ```yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: house-price-api-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: house-price-api
     minReplicas: 2
     maxReplicas: 10
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
   ```

2. **Docker Swarm**
   ```bash
   docker service create \
     --name house-price-api \
     --replicas 3 \
     --publish 8000:8000 \
     house-price-api:latest
   ```

## Model Updates

### Zero-Downtime Deployment

1. **Blue-Green Deployment**
   ```bash
   # Deploy new version
   docker-compose -f docker-compose.blue-green.yml up -d green
   
   # Switch traffic
   # Update load balancer config
   
   # Remove old version
   docker-compose -f docker-compose.blue-green.yml down blue
   ```

2. **Rolling Updates**
   ```bash
   # Update one instance at a time
   docker-compose up -d --scale api=3 --no-recreate
   ```

### Model Versioning

1. Build with version tag:
   ```bash
   ./scripts/build.sh v1.1.0
   ```

2. Update model files:
   ```bash
   # Copy new model files
   cp new_model/model.pkl model/
   cp new_model/model_features.json model/
   
   # Restart services
   docker-compose restart api
   ```

## Monitoring

### Health Monitoring

- Health endpoint: `/health`
- Stats endpoint: `/stats`
- Model info: `/model/info`

### Logging

Logs are output to stdout/stderr for container orchestration:

```bash
# View logs
docker-compose logs -f api

# View specific container
docker logs -f <container_id>
```

### Metrics Collection

For production monitoring, integrate with:
- Prometheus + Grafana
- ELK Stack (Elasticsearch, Logstash, Kibana)
- DataDog, New Relic, etc.

## Troubleshooting

### Common Issues

1. **Model files not found**
   ```bash
   # Ensure model is trained
   python create_model.py
   
   # Check volume mounts
   docker-compose config
   ```

2. **Service not healthy**
   ```bash
   # Check logs
   docker-compose logs api
   
   # Test health endpoint
   curl http://localhost:8000/health
   ```

3. **Performance issues**
   ```bash
   # Scale up
   docker-compose up -d --scale api=3
   
   # Check resource usage
   docker stats
   ```

### Performance Tuning

1. **Worker Processes**
   - Adjust `WORKERS` environment variable
   - Rule of thumb: 2 × CPU cores + 1

2. **Memory Settings**
   - Monitor memory usage with `docker stats`
   - Adjust container memory limits

3. **Connection Tuning**
   - Tune Gunicorn worker connections
   - Adjust Nginx upstream settings

## Security Considerations

### Container Security

- Runs as non-root user (`app`)
- Read-only model/data volumes
- No sensitive data in environment variables
- Security headers in Nginx

### Network Security

```bash
# Run on internal network only
docker-compose up -d --scale api=3
# Access only through load balancer on port 80
```

### SSL/TLS

For production with HTTPS:

1. Update `docker-compose.yml` with SSL certificates
2. Configure Nginx with SSL settings
3. Set `SSL_KEYFILE` and `SSL_CERTFILE` environment variables

## Cloud Deployment Examples

### AWS ECS

```json
{
  "family": "house-price-api",
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "house-price-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### Google Cloud Run

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: house-price-api
spec:
  template:
    spec:
      containers:
      - image: gcr.io/project/house-price-api
        ports:
        - containerPort: 8000
        resources:
          limits:
            cpu: 1000m
            memory: 1Gi
```

## Useful Commands

```bash
# Build and deploy
./scripts/build.sh && ./scripts/deploy.sh

# Scale to 5 replicas
./scripts/scale.sh 5

# View logs
docker-compose logs -f

# Check health
curl http://localhost:8000/health

# Check stats
curl http://localhost:8000/stats

# Stop all services
docker-compose down

# Clean up
docker-compose down -v --rmi all
```
