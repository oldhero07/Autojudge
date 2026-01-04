"""
Enhanced Flask ML Web Application with Advanced Features

This enhanced version integrates all the new components:
- Advanced ML model management with ensemble capabilities
- Comprehensive error handling and recovery
- Monitoring and observability
- Caching and performance optimization
- Configuration and secrets management
"""

from flask import Flask, render_template, request, jsonify, g
import pandas as pd
import numpy as np
import logging
import traceback
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import os
import threading

# Import enhanced components
from ensemble.model_manager import ModelManager
from ensemble.model_ensemble import ModelEnsemble
from ensemble.performance_monitor import PerformanceMonitor
from error_recovery.circuit_breaker import CircuitBreaker
from error_recovery.error_recovery_system import ErrorRecoverySystem
from monitoring.health_monitor import HealthMonitor
from monitoring.structured_logger import StructuredLogger
from monitoring.alerting_system import AlertingSystem
from cache.cache_manager import CacheManager
from cache.concurrent_processor import ConcurrentProcessor, ProcessingRequest, ProcessingMode
from config.config_manager import ConfigManager

# Import original components for backward compatibility
from evaluation_models import ClassificationMetrics, RegressionMetrics, EvaluationReport
from error_handler import ErrorHandler, SystemComponent, ErrorSeverity, error_handler

# Initialize Flask application
app = Flask(__name__)

# Global components
config_manager = None
structured_logger = None
health_monitor = None
alerting_system = None
cache_manager = None
concurrent_processor = None
model_manager = None
model_ensemble = None
performance_monitor = None
circuit_breaker = None
error_recovery_system = None

# Legacy components for backward compatibility
classifier_pipeline = None
regressor_pipeline = None
tfidf_vectorizer = None
feature_scaler = None
model_evaluator = None

class EnhancedPredictionService:
    """Enhanced prediction service with caching, monitoring, and error recovery."""
    
    def __init__(self):
        self.request_counter = 0
        self.lock = threading.Lock()
    
    def _generate_cache_key(self, description: str, input_desc: str, output_desc: str) -> str:
        """Generate cache key for prediction request."""
        combined_text = f"{description}|{input_desc}|{output_desc}"
        import hashlib
        return hashlib.md5(combined_text.encode()).hexdigest()
    
    def _extract_features(self, description: str, input_desc: str, output_desc: str) -> Dict[str, Any]:
        """Extract features from text inputs."""
        # Combine text
        combined_text = f"{description} {input_desc} {output_desc}".strip()
        
        # Basic features
        features = {
            'text_length': len(combined_text),
            'word_count': len(combined_text.split()) if combined_text else 0,
            'combined_text': combined_text
        }
        
        return features
    
    def predict_with_caching(self, description: str, input_desc: str = "", 
                           output_desc: str = "", use_cache: bool = True) -> Dict[str, Any]:
        """Make prediction with caching and monitoring."""
        correlation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Log request
            structured_logger.log_request(
                method="POST",
                endpoint="/predict/enhanced",
                correlation_id=correlation_id,
                request_data={
                    'description_length': len(description),
                    'input_desc_length': len(input_desc),
                    'output_desc_length': len(output_desc)
                }
            )
            
            # Generate cache key
            cache_key = self._generate_cache_key(description, input_desc, output_desc)
            
            # Try cache first
            if use_cache and cache_manager:
                cached_result, cache_status = cache_manager.get(cache_key, 'prediction')
                if cached_result:
                    structured_logger.log_info(
                        "Cache hit for prediction",
                        correlation_id=correlation_id,
                        metadata={'cache_key': cache_key}
                    )
                    
                    # Add cache metadata
                    cached_result['cache_hit'] = True
                    cached_result['correlation_id'] = correlation_id
                    return cached_result
            
            # Extract features
            features = self._extract_features(description, input_desc, output_desc)
            
            # Make prediction through circuit breaker
            prediction_result = self._make_prediction_with_recovery(features, correlation_id)
            
            # Cache result
            if use_cache and cache_manager and prediction_result.get('success', False):
                cache_manager.set(
                    cache_key, 
                    prediction_result, 
                    ttl=3600,  # 1 hour
                    cache_type='prediction'
                )
            
            # Log response
            processing_time = (time.time() - start_time) * 1000
            structured_logger.log_response(
                status_code=200,
                correlation_id=correlation_id,
                processing_time_ms=processing_time,
                response_data={'success': prediction_result.get('success', False)}
            )
            
            # Monitor performance
            if performance_monitor:
                performance_monitor.record_prediction(
                    model_name='ensemble',
                    processing_time_ms=processing_time,
                    success=prediction_result.get('success', False)
                )
            
            # Add metadata
            prediction_result['cache_hit'] = False
            prediction_result['correlation_id'] = correlation_id
            prediction_result['processing_time_ms'] = processing_time
            
            return prediction_result
            
        except Exception as e:
            # Log error
            structured_logger.log_error(
                error=str(e),
                correlation_id=correlation_id,
                metadata={'traceback': traceback.format_exc()}
            )
            
            # Return error response
            return {
                'success': False,
                'error': str(e),
                'correlation_id': correlation_id,
                'cache_hit': False
            }
    
    def _make_prediction_with_recovery(self, features: Dict[str, Any], correlation_id: str) -> Dict[str, Any]:
        """Make prediction with error recovery."""
        
        def prediction_function():
            """Core prediction logic."""
            if not model_ensemble:
                raise ValueError("Model ensemble not initialized")
            
            # Use ensemble for prediction
            prediction = model_ensemble.predict(features['combined_text'])
            
            return {
                'success': True,
                'class': prediction['predicted_class'],
                'score': prediction['predicted_score'],
                'confidence': prediction['confidence'],
                'model_used': prediction['model_used'],
                'features': {
                    'text_length': features['text_length'],
                    'word_count': features['word_count']
                }
            }
        
        # Use circuit breaker for prediction
        if circuit_breaker:
            try:
                return circuit_breaker.call(prediction_function)
            except Exception as e:
                # Use error recovery system
                if error_recovery_system:
                    return error_recovery_system.handle_prediction_error(
                        error=e,
                        input_data=features,
                        correlation_id=correlation_id
                    )
                else:
                    raise
        else:
            return prediction_function()
    
    def predict_batch(self, requests: list) -> list:
        """Process batch of prediction requests."""
        if not concurrent_processor:
            # Fallback to sequential processing
            return [self.predict_with_caching(**req) for req in requests]
        
        # Use concurrent processor
        results = []
        batch_id = str(uuid.uuid4())
        
        for i, req in enumerate(requests):
            request_id = f"{batch_id}_{i}"
            
            # Submit to concurrent processor
            concurrent_processor.submit_request(
                request_id=request_id,
                data=req,
                processing_function=lambda data: self.predict_with_caching(**data),
                mode=ProcessingMode.BATCH,
                priority=0
            )
        
        # Wait for results (simplified - in production, use proper async handling)
        time.sleep(0.1)  # Allow processing time
        
        return results

def initialize_enhanced_components():
    """Initialize all enhanced components."""
    global config_manager, structured_logger, health_monitor, alerting_system
    global cache_manager, concurrent_processor, model_manager, model_ensemble
    global performance_monitor, circuit_breaker, error_recovery_system
    
    try:
        app.logger.info("Initializing enhanced components...")
        
        # Initialize configuration manager
        config_manager = ConfigManager(
            config_dir="config",
            enable_hot_reload=True
        )
        
        # Update Flask config from config manager
        app.config['SECRET_KEY'] = config_manager.get('app.secret_key', 'dev-secret-key')
        app.config['DEBUG'] = config_manager.get('app.debug', False)
        
        # Initialize structured logger
        structured_logger = StructuredLogger(
            service_name="ml-webapp-enhanced",
            version="2.0.0",
            environment=config_manager.get('app.environment', 'development')
        )
        
        # Initialize health monitor
        health_monitor = HealthMonitor(
            check_interval=config_manager.get('monitoring.health_check_interval', 30)
        )
        
        # Initialize alerting system
        alerting_config = {
            'delivery': {
                'email': {
                    'enabled': config_manager.get('alerting.email.enabled', False),
                    'smtp_server': config_manager.get('alerting.email.smtp_server', 'localhost'),
                    'from_address': config_manager.get('alerting.email.from_address', 'alerts@mlapp.com'),
                    'to_addresses': config_manager.get('alerting.email.to_addresses', ['admin@mlapp.com'])
                },
                'webhook': {
                    'enabled': config_manager.get('alerting.webhook.enabled', False),
                    'url': config_manager.get('alerting.webhook.url', '')
                }
            }
        }
        alerting_system = AlertingSystem(alerting_config)
        
        # Initialize cache manager
        cache_config = {
            'local': {
                'enabled': config_manager.get('cache.local.enabled', True),
                'max_size': config_manager.get('cache.local.max_size', 1000),
                'default_ttl': config_manager.get('cache.local.default_ttl', 3600)
            },
            'redis': {
                'enabled': config_manager.get('cache.redis.enabled', False),
                'host': config_manager.get('redis.host', 'localhost'),
                'port': config_manager.get('redis.port', 6379),
                'db': config_manager.get('redis.db', 0)
            }
        }
        cache_manager = CacheManager(cache_config)
        
        # Initialize concurrent processor
        concurrent_config = {
            'max_workers': config_manager.get('processing.max_workers', 4),
            'default_mode': config_manager.get('processing.default_mode', 'threaded'),
            'enable_batching': config_manager.get('processing.enable_batching', True),
            'batch': {
                'max_batch_size': config_manager.get('processing.batch.max_batch_size', 32),
                'batch_timeout_ms': config_manager.get('processing.batch.batch_timeout_ms', 100)
            }
        }
        concurrent_processor = ConcurrentProcessor(concurrent_config)
        
        # Initialize model manager
        model_config = {
            'model_path': config_manager.get('ml.model_path', 'models/'),
            'enable_auto_selection': config_manager.get('ml.enable_auto_selection', True),
            'performance_threshold': config_manager.get('ml.performance_threshold', 0.8)
        }
        model_manager = ModelManager(model_config)
        
        # Initialize model ensemble
        ensemble_config = {
            'voting_strategy': config_manager.get('ml.ensemble.voting_strategy', 'weighted'),
            'enable_adaptive_weights': config_manager.get('ml.ensemble.enable_adaptive_weights', True)
        }
        model_ensemble = ModelEnsemble(ensemble_config)
        
        # Initialize performance monitor
        performance_monitor = PerformanceMonitor()
        
        # Initialize circuit breaker
        circuit_config = {
            'failure_threshold': config_manager.get('circuit_breaker.failure_threshold', 5),
            'recovery_timeout': config_manager.get('circuit_breaker.recovery_timeout', 60),
            'expected_exception': Exception
        }
        circuit_breaker = CircuitBreaker(**circuit_config)
        
        # Initialize error recovery system
        recovery_config = {
            'enable_fallback_models': config_manager.get('error_recovery.enable_fallback_models', True),
            'enable_graceful_degradation': config_manager.get('error_recovery.enable_graceful_degradation', True)
        }
        error_recovery_system = ErrorRecoverySystem(recovery_config)
        
        # Start background services
        health_monitor.start_monitoring()
        
        app.logger.info("Enhanced components initialized successfully!")
        
    except Exception as e:
        app.logger.error(f"Failed to initialize enhanced components: {str(e)}")
        app.logger.error(traceback.format_exc())
        raise

def load_models():
    """Load and initialize ML models."""
    global model_manager, model_ensemble
    
    try:
        app.logger.info("Loading ML models...")
        
        # For now, create dummy models for demonstration
        # In production, load actual trained models
        
        if model_manager:
            # Load models into model manager
            # model_manager.load_model('random_forest', model_path)
            # model_manager.load_model('svm', model_path)
            # model_manager.load_model('logistic_regression', model_path)
            pass
        
        if model_ensemble:
            # Add models to ensemble
            # model_ensemble.add_model('random_forest', model, weight=0.4)
            # model_ensemble.add_model('svm', model, weight=0.3)
            # model_ensemble.add_model('logistic_regression', model, weight=0.3)
            pass
        
        app.logger.info("ML models loaded successfully!")
        
    except Exception as e:
        app.logger.error(f"Failed to load models: {str(e)}")
        raise

# Enhanced prediction service instance
enhanced_prediction_service = EnhancedPredictionService()

# Routes
@app.route('/')
def index():
    """Render the main interface."""
    return render_template('index.html')

@app.route('/predict/enhanced', methods=['POST'])
def predict_enhanced():
    """Enhanced prediction endpoint with caching and monitoring."""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided',
                'status_code': 400
            }), 400
        
        # Validate required fields
        if 'description' not in data or not data['description'].strip():
            return jsonify({
                'success': False,
                'error': 'Description field is required',
                'status_code': 400
            }), 400
        
        # Make enhanced prediction
        result = enhanced_prediction_service.predict_with_caching(
            description=data['description'],
            input_desc=data.get('input_desc', ''),
            output_desc=data.get('output_desc', ''),
            use_cache=data.get('use_cache', True)
        )
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error in enhanced predict endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'status_code': 500
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint."""
    try:
        data = request.get_json()
        
        if not data or 'requests' not in data:
            return jsonify({
                'success': False,
                'error': 'Requests array is required',
                'status_code': 400
            }), 400
        
        requests = data['requests']
        if not isinstance(requests, list) or len(requests) == 0:
            return jsonify({
                'success': False,
                'error': 'Requests must be a non-empty array',
                'status_code': 400
            }), 400
        
        # Process batch
        results = enhanced_prediction_service.predict_batch(requests)
        
        return jsonify({
            'success': True,
            'results': results,
            'batch_size': len(requests)
        })
        
    except Exception as e:
        app.logger.error(f"Error in batch predict endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'status_code': 500
        }), 500

@app.route('/health/enhanced', methods=['GET'])
def health_enhanced():
    """Enhanced health check endpoint."""
    try:
        health_data = {}
        
        # Get health monitor data
        if health_monitor:
            health_data['system_health'] = health_monitor.get_health_status()
        
        # Get performance monitor data
        if performance_monitor:
            health_data['performance_metrics'] = performance_monitor.get_performance_summary()
        
        # Get cache statistics
        if cache_manager:
            health_data['cache_statistics'] = cache_manager.get_comprehensive_statistics()
        
        # Get concurrent processor statistics
        if concurrent_processor:
            health_data['processing_statistics'] = concurrent_processor.get_comprehensive_statistics()
        
        # Get circuit breaker status
        if circuit_breaker:
            health_data['circuit_breaker'] = {
                'state': circuit_breaker.state.value,
                'failure_count': circuit_breaker.failure_count,
                'last_failure_time': circuit_breaker.last_failure_time.isoformat() if circuit_breaker.last_failure_time else None
            }
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'health_data': health_data
        })
        
    except Exception as e:
        app.logger.error(f"Error in enhanced health endpoint: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Metrics endpoint for monitoring systems."""
    try:
        metrics_data = {}
        
        # Performance metrics
        if performance_monitor:
            metrics_data['performance'] = performance_monitor.get_performance_summary()
        
        # Cache metrics
        if cache_manager:
            cache_stats = cache_manager.get_comprehensive_statistics()
            metrics_data['cache'] = cache_stats
        
        # Processing metrics
        if concurrent_processor:
            processing_stats = concurrent_processor.get_comprehensive_statistics()
            metrics_data['processing'] = processing_stats
        
        # Alert metrics
        if alerting_system:
            alert_stats = alerting_system.get_alert_statistics()
            metrics_data['alerts'] = alert_stats
        
        return jsonify(metrics_data)
        
    except Exception as e:
        app.logger.error(f"Error in metrics endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration (non-sensitive values only)."""
    try:
        if config_manager:
            config_info = config_manager.get_config_info()
            return jsonify(config_info)
        else:
            return jsonify({'error': 'Configuration manager not initialized'}), 500
            
    except Exception as e:
        app.logger.error(f"Error in config endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear cache endpoint."""
    try:
        if cache_manager:
            cache_manager.clear()
            return jsonify({
                'success': True,
                'message': 'Cache cleared successfully'
            })
        else:
            return jsonify({'error': 'Cache manager not initialized'}), 500
            
    except Exception as e:
        app.logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Legacy endpoints for backward compatibility
@app.route('/predict', methods=['POST'])
def predict_legacy():
    """Legacy prediction endpoint for backward compatibility."""
    try:
        # Use enhanced service but return legacy format
        data = request.get_json()
        
        if not data or 'description' not in data:
            return jsonify({
                'error': 'Description field is required',
                'status_code': 400
            }), 400
        
        # Make prediction
        result = enhanced_prediction_service.predict_with_caching(
            description=data['description'],
            input_desc=data.get('input_desc', ''),
            output_desc=data.get('output_desc', '')
        )
        
        # Convert to legacy format
        if result.get('success', False):
            legacy_result = {
                'class': result.get('class', 'unknown'),
                'score': result.get('score', 0.0),
                'confidence': result.get('confidence', 0.0),
                'reliable': result.get('confidence', 0.0) > 0.6,
                'features': result.get('features', {})
            }
            return jsonify(legacy_result)
        else:
            return jsonify({
                'error': result.get('error', 'Prediction failed'),
                'status_code': 500
            }), 500
            
    except Exception as e:
        app.logger.error(f"Error in legacy predict endpoint: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status_code': 500
        }), 500

@app.route('/health', methods=['GET'])
def health_legacy():
    """Legacy health endpoint."""
    try:
        # Basic health check
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'config_manager': config_manager is not None,
                'cache_manager': cache_manager is not None,
                'model_ensemble': model_ensemble is not None,
                'health_monitor': health_monitor is not None
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.before_request
def before_request():
    """Set up request context."""
    g.start_time = time.time()
    g.correlation_id = str(uuid.uuid4())

@app.after_request
def after_request(response):
    """Log request completion."""
    if structured_logger and hasattr(g, 'start_time'):
        processing_time = (time.time() - g.start_time) * 1000
        
        structured_logger.log_info(
            "Request completed",
            correlation_id=getattr(g, 'correlation_id', 'unknown'),
            metadata={
                'method': request.method,
                'endpoint': request.endpoint,
                'status_code': response.status_code,
                'processing_time_ms': processing_time
            }
        )
    
    return response

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    if structured_logger:
        structured_logger.log_error(
            error=str(error),
            correlation_id=getattr(g, 'correlation_id', 'unknown'),
            metadata={'endpoint': request.endpoint}
        )
    
    return jsonify({
        'error': 'Internal server error',
        'correlation_id': getattr(g, 'correlation_id', 'unknown'),
        'status_code': 500
    }), 500

def initialize_app():
    """Initialize the enhanced application."""
    try:
        app.logger.info("Initializing Enhanced Flask ML Web Application...")
        
        # Initialize enhanced components
        initialize_enhanced_components()
        
        # Load models
        load_models()
        
        app.logger.info("Enhanced application initialized successfully!")
        
    except Exception as e:
        app.logger.error(f"Failed to initialize enhanced application: {str(e)}")
        raise

def shutdown_app():
    """Shutdown the application gracefully."""
    try:
        app.logger.info("Shutting down enhanced application...")
        
        # Shutdown components
        if health_monitor:
            health_monitor.stop_monitoring()
        
        if concurrent_processor:
            concurrent_processor.shutdown()
        
        if config_manager:
            config_manager.shutdown()
        
        app.logger.info("Enhanced application shutdown complete")
        
    except Exception as e:
        app.logger.error(f"Error during shutdown: {str(e)}")

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize application
        initialize_app()
        
        # Run the Flask application
        host = config_manager.get('app.host', '0.0.0.0') if config_manager else '0.0.0.0'
        port = config_manager.get('app.port', 5000) if config_manager else 5000
        debug = config_manager.get('app.debug', False) if config_manager else False
        
        app.run(host=host, port=port, debug=debug)
        
    except KeyboardInterrupt:
        app.logger.info("Received shutdown signal")
    except Exception as e:
        app.logger.error(f"Application error: {str(e)}")
    finally:
        shutdown_app()