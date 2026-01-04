# AutoJudge Project Structure

```
autojudge/
├── README.md                           # Main project documentation
├── DEPLOYMENT.md                       # Deployment guide
├── PROJECT_STRUCTURE.md               # This file
├── LICENSE                            # MIT License
├── Dockerfile                         # Docker container configuration
├── docker-compose.yml                 # Docker Compose setup
├── .gitignore                         # Git ignore rules
├── problems_data.jsonl                # Training dataset (4,112 problems)
├── metadata.json                      # Dataset metadata
│
├── flask_app/                         # Main Flask application
│   ├── app.py                         # Core Flask application (1,393 lines)
│   ├── requirements.txt               # Python dependencies
│   ├── evaluation_models.py           # Model evaluation classes
│   ├── documentation_generator.py     # Auto-documentation generator
│   ├── error_handler.py              # Error handling and monitoring
│   ├── models/                        # Trained ML models
│   │   └── trained_models.pkl         # Serialized models (~15MB)
│   ├── templates/                     # HTML templates
│   │   └── index.html                 # Web interface
│   ├── static/                        # Static assets
│   │   ├── css/
│   │   │   └── style.css             # Application styling
│   │   └── js/
│   │       └── app.js                # Frontend JavaScript
│   └── logs/                          # Application logs
│       └── autojudge.log             # Runtime logs
│
├── components/                        # React components (optional frontend)
│   ├── InputGroup.tsx                # Problem input component
│   └── ResultDisplay.tsx             # Prediction results component
│
├── docs/                             # Documentation
│   ├── AutoJudge_Predicting_Programming_Problem_Difficulty.pdf
│   └── AutoJudge_Predicting_Programming_Problem_Difficulty.txt
│
├── tests/                            # Test files
│   ├── test_api.py                   # API endpoint tests
│   ├── test_model_accuracy.py        # Model performance tests
│   ├── test_integration.py           # Integration tests
│   └── test_property_tests.py        # Property-based tests
│
├── scripts/                          # Utility scripts
│   ├── get_final_metrics.py          # Model evaluation script
│   ├── extract_training_metrics.py   # Training metrics extraction
│   └── test_api_debug.py             # API debugging tool
│
└── deployment/                       # Deployment configurations
    ├── nginx.conf                    # Nginx configuration
    ├── gunicorn.conf.py             # Gunicorn configuration
    └── systemd/                     # Systemd service files
        └── autojudge.service        # Service configuration
```

## Key Components

### Core Application (`flask_app/app.py`)
- **Lines of Code**: 1,393
- **Main Classes**:
  - `ModelEvaluator`: Comprehensive model evaluation
  - `PredictionService`: Prediction logic and validation
- **Key Functions**:
  - `extract_custom_features()`: 15 domain-specific features
  - `train_models()`: ML pipeline training
  - `load_models()`: Model persistence management

### Feature Engineering
- **TF-IDF Features**: 3,000 selected from 4,000 using chi-square
- **Custom Features**: 15 domain-specific indicators
- **Text Processing**: Advanced preprocessing with abbreviation expansion

### Model Architecture
- **Classification**: Voting Classifier ensemble
  - Logistic Regression (C=2.0, balanced weights)
  - Random Forest (400 estimators, max_depth=35)
  - Gradient Boosting (300 estimators, max_depth=12)
- **Regression**: Random Forest (350 estimators, max_depth=30)

### API Endpoints
- `GET /`: Web interface
- `POST /predict`: Problem difficulty prediction
- `POST /predict/structured`: Structured input format
- `GET /health`: System health check
- `POST /generate-docs`: Documentation generation
- `GET /docs/preview`: Documentation preview

### Dataset
- **Size**: 4,112 programming problems
- **Format**: JSONL (JSON Lines)
- **Classes**: Easy (18.6%), Medium (34.2%), Hard (47.2%)
- **Score Range**: 1.1 - 9.7 (mean: 5.11 ± 2.18)

### Performance Metrics
- **Classification Accuracy**: 55.0%
- **Regression MAE**: 1.735 points
- **Model Size**: ~15MB serialized
- **Inference Time**: ~50ms per prediction
- **Memory Usage**: ~200MB loaded

### Dependencies
- **Core**: Flask 2.3.3, scikit-learn 1.3.0
- **ML**: pandas 2.0.3, numpy 1.24.3, scipy 1.11.1
- **Production**: gunicorn 21.2.0
- **Testing**: pytest 7.4.0

### Configuration
- **Environment Variables**: FLASK_ENV, SECRET_KEY
- **Model Persistence**: Automatic save/load
- **Error Handling**: Comprehensive error recovery
- **Logging**: Structured application logging

### Deployment Options
- **Local**: Direct Python execution
- **Docker**: Containerized deployment
- **Cloud**: AWS, GCP, Heroku ready
- **Production**: Gunicorn + Nginx

### Testing
- **Unit Tests**: Core functionality
- **Integration Tests**: API endpoints
- **Property Tests**: Model behavior validation
- **Performance Tests**: Load and stress testing

### Monitoring
- **Health Checks**: System status endpoint
- **Metrics**: Performance monitoring
- **Logging**: Structured error tracking
- **Alerts**: Configurable notifications

## File Sizes
- `app.py`: ~50KB (1,393 lines)
- `trained_models.pkl`: ~15MB
- `problems_data.jsonl`: ~2.5MB
- Total project: ~20MB (excluding node_modules)

## Development Workflow
1. **Setup**: Clone repository, install dependencies
2. **Training**: Models auto-train on first run
3. **Testing**: Run test suite with pytest
4. **Development**: Flask debug mode for iteration
5. **Deployment**: Docker or cloud deployment
6. **Monitoring**: Health checks and logging

## Security Considerations
- **Input Validation**: Request size limits, content validation
- **Rate Limiting**: API endpoint protection
- **HTTPS**: SSL/TLS configuration
- **Secrets**: Environment variable management
- **CORS**: Cross-origin request handling

## Scalability
- **Horizontal**: Load balancer + multiple instances
- **Vertical**: Increased CPU/memory allocation
- **Caching**: Model and prediction caching
- **Database**: Future integration ready

This structure supports both development and production deployments with comprehensive testing, monitoring, and documentation.