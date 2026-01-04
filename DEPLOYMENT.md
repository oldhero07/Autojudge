# AutoJudge Deployment Guide

This guide provides comprehensive instructions for deploying AutoJudge in various environments.

## Quick Start

### Local Development
```bash
# Clone and setup
git clone https://github.com/yourusername/autojudge.git
cd autojudge

# Install dependencies
pip install -r flask_app/requirements.txt

# Start the application
cd flask_app
python app.py
```

Access the application at `http://localhost:5000`

## Production Deployment

### Using Gunicorn (Recommended)

1. **Install Gunicorn**
```bash
pip install gunicorn
```

2. **Create Gunicorn Configuration**
```python
# gunicorn.conf.py
bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
worker_connections = 1000
timeout = 30
keepalive = 2
max_requests = 1000
max_requests_jitter = 100
preload_app = True
```

3. **Start with Gunicorn**
```bash
cd flask_app
gunicorn --config gunicorn.conf.py app:app
```

### Docker Deployment

1. **Create Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY flask_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Set working directory
WORKDIR /app/flask_app

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

2. **Build and Run**
```bash
docker build -t autojudge .
docker run -p 5000:5000 autojudge
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  autojudge:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - SECRET_KEY=your-secret-key-here
    volumes:
      - ./flask_app/models:/app/flask_app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Cloud Deployment

### AWS Elastic Beanstalk

1. **Install EB CLI**
```bash
pip install awsebcli
```

2. **Initialize and Deploy**
```bash
eb init autojudge
eb create production
eb deploy
```

3. **Configuration File** (`.ebextensions/python.config`)
```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: flask_app/app.py
  aws:elasticbeanstalk:application:environment:
    FLASK_ENV: production
```

### Google Cloud Platform

1. **Create app.yaml**
```yaml
runtime: python39

env_variables:
  FLASK_ENV: production
  SECRET_KEY: your-secret-key

automatic_scaling:
  min_instances: 1
  max_instances: 10
  target_cpu_utilization: 0.6
```

2. **Deploy**
```bash
gcloud app deploy
```

### Heroku

1. **Create Procfile**
```
web: cd flask_app && gunicorn app:app
```

2. **Deploy**
```bash
heroku create autojudge-app
git push heroku main
```

## Environment Configuration

### Environment Variables
```bash
# Required
SECRET_KEY=your-secret-key-here
FLASK_ENV=production

# Optional
FLASK_DEBUG=False
MODEL_CACHE_SIZE=100
MAX_CONTENT_LENGTH=1048576
```

### Configuration File
```python
# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    DEBUG = False
    TESTING = False
    MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1MB

class ProductionConfig(Config):
    DEBUG = False

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
```

## Performance Optimization

### Model Caching
```python
# Enable model persistence
MODELS_CACHE_ENABLED = True
MODELS_CACHE_TTL = 3600  # 1 hour
```

### Request Optimization
```python
# Limit request size
MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1MB

# Enable compression
COMPRESS_MIMETYPES = [
    'text/html', 'text/css', 'text/xml',
    'application/json', 'application/javascript'
]
```

### Database Configuration (if needed)
```python
# For future database integration
DATABASE_URL = os.environ.get('DATABASE_URL')
SQLALCHEMY_TRACK_MODIFICATIONS = False
```

## Monitoring and Logging

### Application Logging
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/autojudge.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
```

### Health Check Endpoint
The application includes a `/health` endpoint for monitoring:
```bash
curl http://localhost:5000/health
```

### Metrics Collection
```python
# Example Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest

prediction_counter = Counter('predictions_total', 'Total predictions made')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction processing time')
```

## Security Considerations

### HTTPS Configuration
```nginx
# nginx.conf
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # ... prediction logic
```

### Input Validation
```python
from flask_wtf import FlaskForm
from wtforms import TextAreaField, validators

class PredictionForm(FlaskForm):
    description = TextAreaField('Description', [
        validators.Length(min=10, max=5000),
        validators.DataRequired()
    ])
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure model files exist in `flask_app/models/`
   - Check file permissions
   - Verify Python version compatibility

2. **Memory Issues**
   - Increase container memory limits
   - Enable model compression
   - Use model quantization

3. **Performance Issues**
   - Enable model caching
   - Use connection pooling
   - Implement request queuing

### Debug Mode
```bash
export FLASK_DEBUG=1
export FLASK_ENV=development
python app.py
```

### Log Analysis
```bash
# View application logs
tail -f logs/autojudge.log

# Check system resources
htop
df -h
```

## Scaling Considerations

### Horizontal Scaling
- Use load balancers (nginx, HAProxy)
- Deploy multiple application instances
- Implement session affinity if needed

### Vertical Scaling
- Increase CPU/memory allocation
- Optimize model loading
- Use faster storage (SSD)

### Caching Strategy
- Redis for prediction caching
- CDN for static assets
- Database query caching

## Backup and Recovery

### Model Backup
```bash
# Backup trained models
tar -czf models-backup-$(date +%Y%m%d).tar.gz flask_app/models/
```

### Configuration Backup
```bash
# Backup configuration
cp flask_app/config.py config-backup-$(date +%Y%m%d).py
```

### Automated Backups
```bash
# Cron job for daily backups
0 2 * * * /path/to/backup-script.sh
```

## Support

For deployment issues:
1. Check the troubleshooting section
2. Review application logs
3. Verify system requirements
4. Contact support with error details

## Version History

- **v1.0.0**: Initial production release
- **v1.1.0**: Performance optimizations
- **v1.2.0**: Enhanced security features