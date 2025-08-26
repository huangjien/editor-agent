# Docker Setup for Editor Agent

This document provides comprehensive instructions for building and running the Editor Agent application using Docker.

## Prerequisites

- Docker Engine 20.10+ installed
- Docker Compose 2.0+ installed
- At least 2GB of available disk space
- At least 1GB of available RAM

## Quick Start

### Using Docker Compose (Recommended)

1. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone <repository-url>
   cd editor-agent
   ```

2. **Create environment file (optional):**
   ```bash
   cp .env.example .env
   # Edit .env with your specific configuration
   ```

3. **Build and run the application:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   - API: http://localhost:8000
   - Health Check: http://localhost:8000/health
   - API Documentation: http://localhost:8000/docs

### Using Docker Commands

1. **Build the Docker image:**
   ```bash
   docker build -t editor-agent:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name editor-agent \
     -p 8000:8000 \
     -v $(pwd)/workspace:/app/workspace \
     -v $(pwd)/logs:/app/logs \
     -e ENVIRONMENT=production \
     -e DEBUG=false \
     editor-agent:latest
   ```

## Configuration

### Environment Variables

The application supports the following environment variables:

#### Application Settings
- `APP_NAME`: Application name (default: "Editor Agent")
- `APP_VERSION`: Application version (default: "0.1.2")
- `ENVIRONMENT`: Environment (development/staging/production)
- `DEBUG`: Enable debug mode (true/false)

#### Server Settings
- `HOST`: Server host (default: "0.0.0.0")
- `PORT`: Server port (default: 8000)
- `WORKERS`: Number of worker processes (default: 1)

#### API Settings
- `API_PREFIX`: API prefix (default: "/api/v1")
- `CORS_ORIGINS`: Allowed CORS origins (comma-separated)
- `CORS_ALLOW_CREDENTIALS`: Allow credentials (true/false)

#### Security Settings
- `SECRET_KEY`: Secret key for JWT tokens (required in production)
- `TRUSTED_HOSTS`: Trusted hosts (comma-separated)
- `REQUIRE_API_KEY`: Require API key authentication (true/false)
- `API_KEYS`: Valid API keys (comma-separated)

#### Model Settings
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `LANGCHAIN_API_KEY`: LangChain API key
- `LANGCHAIN_PROJECT`: LangChain project name
- `LANGCHAIN_TRACING_V2`: Enable LangChain tracing (true/false)
- `DEFAULT_MODEL`: Default model to use
- `MODEL_PROVIDER`: Model provider (openai/anthropic)
- `MODEL_TEMPERATURE`: Model temperature (0.0-1.0)
- `MODEL_MAX_TOKENS`: Maximum tokens per request

#### Agent Settings
- `MAX_ITERATIONS`: Maximum agent iterations
- `MAX_EXECUTION_TIME`: Maximum execution time in seconds
- `ENABLE_MEMORY`: Enable agent memory (true/false)
- `MEMORY_MAX_TOKENS`: Maximum memory tokens

#### File System Settings
- `WORKSPACE_DIR`: Workspace directory path
- `MAX_FILE_SIZE`: Maximum file size in bytes
- `ALLOWED_FILE_EXTENSIONS`: Allowed file extensions (comma-separated)

#### Logging Settings
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `LOG_FILE`: Log file path
- `LOG_ROTATION`: Log rotation interval
- `LOG_RETENTION`: Log retention period

#### Monitoring Settings
- `ENABLE_METRICS`: Enable metrics collection (true/false)
- `METRICS_PORT`: Metrics server port
- `HEALTH_CHECK_INTERVAL`: Health check interval in seconds

#### Rate Limiting Settings
- `RATE_LIMIT_ENABLED`: Enable rate limiting (true/false)
- `RATE_LIMIT_REQUESTS`: Requests per window
- `RATE_LIMIT_WINDOW`: Rate limit window in seconds

### Volume Mounts

The following volumes are recommended:

- `./workspace:/app/workspace` - Persistent workspace for file operations
- `./logs:/app/logs` - Persistent logs directory

## Production Deployment

### Security Considerations

1. **Change default secrets:**
   ```bash
   export SECRET_KEY=$(openssl rand -hex 32)
   ```

2. **Use environment-specific configuration:**
   ```bash
   export ENVIRONMENT=production
   export DEBUG=false
   ```

3. **Set up proper API key authentication:**
   ```bash
   export REQUIRE_API_KEY=true
   export API_KEYS="your-secure-api-key-1,your-secure-api-key-2"
   ```

4. **Configure trusted hosts:**
   ```bash
   export TRUSTED_HOSTS="yourdomain.com,api.yourdomain.com"
   ```

### Performance Optimization

1. **Multi-worker setup:**
   ```bash
   export WORKERS=4  # Adjust based on CPU cores
   ```

2. **Resource limits:**
   ```yaml
   # In docker-compose.yml
   deploy:
     resources:
       limits:
         cpus: '2.0'
         memory: 2G
       reservations:
         cpus: '1.0'
         memory: 1G
   ```

### Health Monitoring

The container includes a health check that verifies:
- Application responsiveness
- API endpoint availability
- System resource status

Monitor health status:
```bash
docker ps  # Check HEALTH status
docker inspect editor-agent | grep Health -A 10
```

## Development

### Development Mode

For development with hot reloading:

```bash
docker run -d \
  --name editor-agent-dev \
  -p 8000:8000 \
  -v $(pwd):/app \
  -v $(pwd)/workspace:/app/workspace \
  -e ENVIRONMENT=development \
  -e DEBUG=true \
  -e RELOAD=true \
  editor-agent:latest \
  python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Building for Different Architectures

```bash
# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 -t editor-agent:latest .

# Build for specific architecture
docker buildx build --platform linux/arm64 -t editor-agent:arm64 .
```

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Change the port mapping
   docker run -p 8001:8000 editor-agent:latest
   ```

2. **Permission issues with volumes:**
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER ./workspace ./logs
   ```

3. **Container fails to start:**
   ```bash
   # Check logs
   docker logs editor-agent
   
   # Run in interactive mode for debugging
   docker run -it --rm editor-agent:latest /bin/bash
   ```

4. **Health check failures:**
   ```bash
   # Check health endpoint manually
   curl http://localhost:8000/health
   
   # Check container health
   docker inspect editor-agent --format='{{.State.Health.Status}}'
   ```

### Performance Issues

1. **High memory usage:**
   - Reduce `WORKERS` count
   - Limit `MODEL_MAX_TOKENS`
   - Disable `ENABLE_MEMORY` if not needed

2. **Slow response times:**
   - Increase `MODEL_TIMEOUT`
   - Check `MAX_EXECUTION_TIME` setting
   - Monitor system resources

### Logs and Debugging

```bash
# View application logs
docker logs -f editor-agent

# View logs from mounted volume
tail -f ./logs/app.log

# Execute commands in running container
docker exec -it editor-agent /bin/bash

# Check container resource usage
docker stats editor-agent
```

## Cleanup

```bash
# Stop and remove container
docker-compose down

# Remove container and volumes
docker-compose down -v

# Remove images
docker rmi editor-agent:latest

# Clean up unused Docker resources
docker system prune -a
```

## Support

For issues and questions:
1. Check the application logs
2. Verify environment configuration
3. Ensure all required API keys are set
4. Check Docker and system requirements
5. Review this documentation for common solutions