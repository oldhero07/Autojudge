# Enhanced ML Web App Deployment Guide

This guide covers the deployment of the enhanced ML Web Application with all advanced features including caching, monitoring, error recovery, and configuration management.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Monitoring and Health Checks](#monitoring-and-health-checks)
6. [Caching Setup](#caching-setup)
7. [Error Recovery Configuration](#error-recovery-configuration)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- Python 3.8 or higher
- Redis (optional, for distributed caching)
- PostgreSQL or MySQL (optional, for production database)
- 4GB RAM minimum (8GB recommended for production)
- 2 CPU cores minimum (4+ recommended for production)

### Dependencies

All dependencies are listed in `requirements.txt`. Key components include:

- Flask 2.3.3 (web framework)
- scikit-learn 1.3.0 (machine learning)
- Redis 4.6.0 (caching)
- Prometheus-client 0.17.1 (monitoring)
- Cryptography 41.0.4 (security)
- Watchdog 3.0.0 (configuration hot-reload)

## Installation

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd flask_app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Create Required Directories

```bash
mkdir -p models cache logs config
```

### 3. Set Up Configuration

Copy the sample configuration:

```bash
cp config/config.yaml config/production.yaml
```

Edit `config/production.yaml` for your environment.

## Configuration

### Environment Variables

The application supports configuration through environment variables:

```bash
# Application settings
export ENVIRONMENT=production
export DEBUG=false
export HOST=0.0.0.0
export PORT=5000
export SECRET_KEY=your-secret-key-here

# Database settings
export DATABASE_URL=postgresql://user:password@localhost/mlapp

# Redis settings (optional)
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_PASSWORD=your-redis-password

# Monitoring settings
export MONITORING_ENABLED=true
export ALERTING_EMAIL_ENABLED=true
export ALERTING_EMAIL_SMTP_SERVER=smtp.gmail.com
export ALERTING_EMAIL_FROM=alerts@yourcompany.com
export ALERTING_EMAIL_TO=admin@yourcompany.com

# ML settings
export ML_MODEL_PATH=models/
export ML_MAX_BATCH_SIZE=32
```

### Configuration File

The application uses YAML configuration files. Example `config/production.yaml`:

```yaml
app:
  environment: "production"
  debug: false
  host: "0.0.0.0"
  port: 5000

cache:
  redis:
    enabled: true
    host: "redis-server"
    port: 6379

monitoring:
  enabled: true
  health_check_interval: 30

alerting:
  enabled: true
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    from_address: "alerts@yourcompany.com"
    to_addresses:
      - "admin@yourcompany.com"
```

### Secrets Management

Sensitive configuration is automatically encrypted:

```bash
# The application will automatically detect and encrypt:
# - Passwords
# - API keys
# - Secret tokens
# - Database URLs with credentials
```

## Running the Application

### Development Mode

```bash
# Using the enhanced application
python enhanced_app.py

# Or using the original application
python app.py
```

### Production Mode

```bash
# Using Gunicorn (recommended)
gunicorn -w 4 -b 0.0.0.0:5000 enhanced_app:app

# With additional options
gunicorn -w 4 -b 0.0.0.0:5000 \
  --timeout 120 \
  --keep-alive 5 \
  --max-requests 1000 \
  --max-requests-jitter 100 \
  enhanced_app:app
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "enhanced_app:app"]
```

Build and run:

```bash
docker build -t ml-webapp-enhanced .
docker run -p 5000:5000 -e ENVIRONMENT=production ml-webapp-enhanced
```

## Monitoring and Health Checks

### Health Check Endpoints

The application provides several health check endpoints:

```bash
# Basic health check
curl http://localhost:5000/health

# Enhanced health check with detailed metrics
curl http://localhost:5000/health/enhanced

# Prometheus metrics
curl http://localhost:5000/metrics
```

### System Monitoring

The application automatically monitors:

- CPU usage
- Memory usage
- Disk usage
- Response times
- Error rates
- Cache hit rates
- Model performance

### Alerting

Configure alerting in your configuration file:

```yaml
alerting:
  enabled: true
  email:
    enabled: true
    smtp_server: "your-smtp-server"
    from_address: "alerts@yourcompany.com"
    to_addresses:
      - "admin@yourcompany.com"
  webhook:
    enabled: true
    url: "https://your-webhook-url"
  slack:
    enabled: true
    webhook_url: "https://hooks.slack.com/your-webhook"
    channel: "#alerts"
```

## Caching Setup

### Local Caching

Local caching is enabled by default:

```yaml
cache:
  local:
    enabled: true
    max_size: 1000
    default_ttl: 3600
```

### Redis Caching

For distributed caching, set up Redis:

```bash
# Install Redis
sudo apt-get install redis-server

# Start Redis
sudo systemctl start redis-server

# Configure in application
cache:
  redis:
    enabled: true
    host: "localhost"
    port: 6379
```

### Cache Management

```bash
# Clear cache via API
curl -X POST http://localhost:5000/cache/clear

# Get cache statistics
curl http://localhost:5000/metrics
```

## Error Recovery Configuration

The application includes comprehensive error recovery:

### Circuit Breaker

```yaml
circuit_breaker:
  failure_threshold: 5
  recovery_timeout: 60
  half_open_max_calls: 3
```

### Error Recovery Strategies

```yaml
error_recovery:
  enable_fallback_models: true
  enable_graceful_degradation: true
  enable_cached_responses: true
  max_retry_attempts: 3
  retry_delay_seconds: 1
```

## Production Deployment

### 1. Database Setup

For production, use a proper database:

```sql
-- PostgreSQL setup
CREATE DATABASE mlapp;
CREATE USER mlapp_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE mlapp TO mlapp_user;
```

### 2. Reverse Proxy Setup

Use Nginx as a reverse proxy:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. SSL/TLS Setup

```bash
# Using Let's Encrypt
sudo certbot --nginx -d your-domain.com
```

### 4. Process Management

Use systemd for process management:

```ini
# /etc/systemd/system/ml-webapp.service
[Unit]
Description=ML Web App Enhanced
After=network.target

[Service]
Type=exec
User=mlapp
Group=mlapp
WorkingDirectory=/opt/ml-webapp
Environment=PATH=/opt/ml-webapp/venv/bin
ExecStart=/opt/ml-webapp/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 enhanced_app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable ml-webapp
sudo systemctl start ml-webapp
```

### 5. Log Management

Configure log rotation:

```bash
# /etc/logrotate.d/ml-webapp
/opt/ml-webapp/logs/*.log {
    daily
    missingok
    rotate 52
    compress
    delaycompress
    notifempty
    create 644 mlapp mlapp
    postrotate
        systemctl reload ml-webapp
    endscript
}
```

## Troubleshooting

### Common Issues

#### 1. Application Won't Start

```bash
# Check logs
tail -f logs/app.log

# Check configuration
python -c "from config.config_manager import ConfigManager; cm = ConfigManager(); print(cm.get_config_info())"

# Validate dependencies
pip check
```

#### 2. High Memory Usage

```bash
# Check cache usage
curl http://localhost:5000/metrics | grep cache

# Reduce cache size in configuration
cache:
  local:
    max_size: 500  # Reduce from default 1000
```

#### 3. Slow Response Times

```bash
# Check performance metrics
curl http://localhost:5000/health/enhanced

# Enable concurrent processing
processing:
  max_workers: 8  # Increase workers
  enable_batching: true
```

#### 4. Redis Connection Issues

```bash
# Test Redis connection
redis-cli ping

# Check Redis logs
sudo journalctl -u redis

# Verify configuration
redis-cli config get "*"
```

### Performance Tuning

#### 1. Gunicorn Configuration

```bash
# Optimal worker count: (2 x CPU cores) + 1
gunicorn -w 9 -b 0.0.0.0:5000 \
  --worker-class gevent \
  --worker-connections 1000 \
  enhanced_app:app
```

#### 2. Cache Optimization

```yaml
cache:
  local:
    max_size: 2000  # Increase for more memory
    default_ttl: 7200  # Longer TTL for stable data
  redis:
    enabled: true
    default_ttl: 14400  # Even longer for distributed cache
```

#### 3. Model Optimization

```yaml
ml:
  max_batch_size: 64  # Increase batch size
  ensemble:
    enable_adaptive_weights: true  # Enable adaptive optimization
```

### Monitoring and Debugging

#### 1. Enable Debug Logging

```yaml
logging:
  level: "DEBUG"
  structured_logging: true
```

#### 2. Health Check Monitoring

```bash
# Continuous health monitoring
watch -n 5 'curl -s http://localhost:5000/health/enhanced | jq .health_data.system_health'
```

#### 3. Performance Profiling

```python
# Add to enhanced_app.py for profiling
import cProfile
import pstats

@app.route('/profile')
def profile():
    pr = cProfile.Profile()
    pr.enable()
    # Your code here
    pr.disable()
    stats = pstats.Stats(pr)
    return stats.print_stats()
```

## Security Considerations

### 1. Environment Variables

Never commit sensitive environment variables:

```bash
# Use .env file (not committed)
echo "SECRET_KEY=your-secret-key" >> .env
echo "DATABASE_URL=postgresql://..." >> .env
```

### 2. Network Security

```bash
# Firewall configuration
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 3. Application Security

```yaml
security:
  enable_cors: false  # Disable in production
  rate_limiting:
    enabled: true
    requests_per_minute: 100
```

## Backup and Recovery

### 1. Database Backup

```bash
# PostgreSQL backup
pg_dump mlapp > backup_$(date +%Y%m%d).sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump mlapp | gzip > $BACKUP_DIR/mlapp_$DATE.sql.gz
find $BACKUP_DIR -name "mlapp_*.sql.gz" -mtime +7 -delete
```

### 2. Configuration Backup

```bash
# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz config/
```

### 3. Model Backup

```bash
# Backup trained models
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/
```

## Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly**: Check logs, update dependencies
2. **Monthly**: Review performance metrics, optimize cache
3. **Quarterly**: Security updates, backup verification

### Getting Help

1. Check application logs: `tail -f logs/app.log`
2. Review health metrics: `curl http://localhost:5000/health/enhanced`
3. Validate configuration: Check config files and environment variables
4. Test components: Run integration tests with `python test_enhanced_integration.py`

For additional support, refer to the component documentation in each module.