"""
Integration Tests for Enhanced ML Web App Components

This module provides comprehensive integration tests for all enhanced components
including caching, monitoring, error recovery, and configuration management.
"""

import pytest
import json
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import components to test
from cache.cache_manager import CacheManager, CacheStatus
from cache.concurrent_processor import ConcurrentProcessor, ProcessingRequest, ProcessingMode
from monitoring.health_monitor import HealthMonitor
from monitoring.structured_logger import StructuredLogger
from monitoring.alerting_system import AlertingSystem, AlertSeverity, AlertChannel
from config.config_manager import ConfigManager, SecretManager, ConfigValidator, ConfigSchema
from ensemble.model_manager import ModelManager
from ensemble.model_ensemble import ModelEnsemble
from ensemble.performance_monitor import PerformanceMonitor
from error_recovery.circuit_breaker import CircuitBreaker, CircuitState
from error_recovery.error_recovery_system import ErrorRecoverySystem

class TestCacheManager:
    """Test cache manager functionality."""
    
    def test_cache_manager_initialization(self):
        """Test cache manager initialization."""
        config = {
            'local': {'enabled': True, 'max_size': 100},
            'redis': {'enabled': False}
        }
        cache_manager = CacheManager(config)
        
        assert cache_manager.use_local is True
        assert cache_manager.use_redis is False
        assert cache_manager.local_cache is not None
    
    def test_local_cache_operations(self):
        """Test local cache set/get operations."""
        config = {'local': {'enabled': True, 'max_size': 10}}
        cache_manager = CacheManager(config)
        
        # Test set and get
        success = cache_manager.set('test_key', 'test_value', cache_type='prediction')
        assert success is True
        
        value, status = cache_manager.get('test_key', cache_type='prediction')
        assert value == 'test_value'
        assert status == CacheStatus.HIT
        
        # Test miss
        value, status = cache_manager.get('nonexistent_key', cache_type='prediction')
        assert value is None
        assert status == CacheStatus.MISS
    
    def test_cache_statistics(self):
        """Test cache statistics collection."""
        config = {'local': {'enabled': True, 'max_size': 10}}
        cache_manager = CacheManager(config)
        
        # Perform some operations
        cache_manager.set('key1', 'value1')
        cache_manager.get('key1')
        cache_manager.get('nonexistent')
        
        stats = cache_manager.get_comprehensive_statistics()
        assert 'local_cache' in stats
        assert stats['local_cache']['total_requests'] > 0

class TestConcurrentProcessor:
    """Test concurrent processing functionality."""
    
    def test_concurrent_processor_initialization(self):
        """Test concurrent processor initialization."""
        config = {
            'max_workers': 2,
            'default_mode': 'threaded',
            'enable_batching': True
        }
        processor = ConcurrentProcessor(config)
        
        assert processor.max_workers == 2
        assert processor.default_mode == ProcessingMode.THREADED
        assert processor.enable_batching is True
    
    def test_sequential_processing(self):
        """Test sequential processing mode."""
        processor = ConcurrentProcessor({'max_workers': 1})
        
        def simple_function(data):
            return data * 2
        
        result_container = []
        
        def callback(result):
            result_container.append(result)
        
        status = processor.submit_request(
            request_id='test_1',
            data=5,
            processing_function=simple_function,
            mode=ProcessingMode.SEQUENTIAL,
            callback=callback
        )
        
        assert status == 'completed_sequential'
        # Give time for callback
        time.sleep(0.1)
        assert len(result_container) == 1
        assert result_container[0].result == 10
    
    def test_processing_statistics(self):
        """Test processing statistics collection."""
        processor = ConcurrentProcessor({'max_workers': 2})
        
        def simple_function(data):
            return data
        
        processor.submit_request(
            request_id='test_stats',
            data='test',
            processing_function=simple_function,
            mode=ProcessingMode.SEQUENTIAL
        )
        
        stats = processor.get_comprehensive_statistics()
        assert 'processing' in stats
        assert stats['processing']['total_requests'] > 0

class TestHealthMonitor:
    """Test health monitoring functionality."""
    
    def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        monitor = HealthMonitor(check_interval=1)
        
        assert monitor.check_interval == 1
        assert monitor.monitoring_active is False
    
    def test_system_metrics_collection(self):
        """Test system metrics collection."""
        monitor = HealthMonitor()
        
        metrics = monitor.collect_system_metrics()
        
        assert 'cpu_percent' in metrics
        assert 'memory_percent' in metrics
        assert 'disk_usage' in metrics
        assert isinstance(metrics['cpu_percent'], (int, float))
        assert isinstance(metrics['memory_percent'], (int, float))
    
    def test_health_status(self):
        """Test health status reporting."""
        monitor = HealthMonitor()
        
        status = monitor.get_health_status()
        
        assert 'status' in status
        assert 'timestamp' in status
        assert 'system_metrics' in status
        assert status['status'] in ['healthy', 'degraded', 'unhealthy']

class TestStructuredLogger:
    """Test structured logging functionality."""
    
    def test_structured_logger_initialization(self):
        """Test structured logger initialization."""
        logger = StructuredLogger(
            service_name='test-service',
            version='1.0.0',
            environment='test'
        )
        
        assert logger.service_name == 'test-service'
        assert logger.version == '1.0.0'
        assert logger.environment == 'test'
    
    def test_correlation_id_generation(self):
        """Test correlation ID generation."""
        logger = StructuredLogger('test-service')
        
        correlation_id = logger.generate_correlation_id()
        
        assert isinstance(correlation_id, str)
        assert len(correlation_id) > 0
    
    def test_log_methods(self):
        """Test various log methods."""
        logger = StructuredLogger('test-service')
        
        # Test different log levels
        logger.log_info('Test info message')
        logger.log_warning('Test warning message')
        logger.log_error('Test error message')
        
        # Test request/response logging
        logger.log_request('GET', '/test', 'test-correlation-id')
        logger.log_response(200, 'test-correlation-id', 100.0)
        
        # Should not raise exceptions
        assert True

class TestAlertingSystem:
    """Test alerting system functionality."""
    
    def test_alerting_system_initialization(self):
        """Test alerting system initialization."""
        config = {
            'delivery': {
                'email': {'enabled': False},
                'webhook': {'enabled': False}
            }
        }
        alerting = AlertingSystem(config)
        
        assert alerting.threshold_manager is not None
        assert alerting.delivery_manager is not None
    
    def test_metric_checking(self):
        """Test metric checking against thresholds."""
        alerting = AlertingSystem()
        
        # This should not trigger alerts (normal CPU usage)
        alerting.check_metric('cpu_percent', 50.0)
        
        # This should trigger alerts (high CPU usage)
        alerting.check_metric('cpu_percent', 90.0)
        
        # Check active alerts
        active_alerts = alerting.get_active_alerts()
        assert isinstance(active_alerts, list)
    
    def test_alert_statistics(self):
        """Test alert statistics collection."""
        alerting = AlertingSystem()
        
        stats = alerting.get_alert_statistics()
        
        assert 'active_alerts' in stats
        assert 'configured_rules' in stats
        assert isinstance(stats['active_alerts']['total'], int)

class TestConfigManager:
    """Test configuration management functionality."""
    
    def test_config_manager_initialization(self):
        """Test config manager initialization."""
        with patch('pathlib.Path.exists', return_value=False):
            config_manager = ConfigManager(
                config_dir='test_config',
                enable_hot_reload=False
            )
            
            assert config_manager.config_dir.name == 'test_config'
            assert config_manager.enable_hot_reload is False
    
    def test_environment_variable_parsing(self):
        """Test environment variable parsing."""
        config_manager = ConfigManager(enable_hot_reload=False)
        
        # Test different value types
        assert config_manager._parse_env_value('true') is True
        assert config_manager._parse_env_value('false') is False
        assert config_manager._parse_env_value('123') == 123
        assert config_manager._parse_env_value('12.34') == 12.34
        assert config_manager._parse_env_value('hello') == 'hello'
    
    def test_secret_detection(self):
        """Test secret key detection."""
        config_manager = ConfigManager(enable_hot_reload=False)
        
        assert config_manager._is_secret_key('database_password') is True
        assert config_manager._is_secret_key('api_key') is True
        assert config_manager._is_secret_key('secret_token') is True
        assert config_manager._is_secret_key('normal_setting') is False

class TestSecretManager:
    """Test secret management functionality."""
    
    def test_secret_manager_initialization(self):
        """Test secret manager initialization."""
        secret_manager = SecretManager()
        
        assert secret_manager.cipher is not None
        assert isinstance(secret_manager.secrets, dict)
    
    def test_secret_storage_and_retrieval(self):
        """Test secret storage and retrieval."""
        secret_manager = SecretManager()
        
        # Store a secret
        success = secret_manager.store_secret('test_key', 'test_secret_value')
        assert success is True
        
        # Retrieve the secret
        retrieved_value = secret_manager.get_secret('test_key')
        assert retrieved_value == 'test_secret_value'
        
        # Try to retrieve non-existent secret
        non_existent = secret_manager.get_secret('non_existent_key')
        assert non_existent is None

class TestConfigValidator:
    """Test configuration validation functionality."""
    
    def test_config_validator_initialization(self):
        """Test config validator initialization."""
        validator = ConfigValidator()
        
        assert isinstance(validator.schemas, dict)
        assert isinstance(validator.validation_errors, list)
    
    def test_schema_validation(self):
        """Test configuration schema validation."""
        validator = ConfigValidator()
        
        # Add a schema
        schema = ConfigSchema(
            key='test_setting',
            value_type=int,
            required=True,
            min_value=1,
            max_value=100
        )
        validator.add_schema(schema)
        
        # Test valid configuration
        valid_config = {'test_setting': 50}
        is_valid, errors = validator.validate_config(valid_config)
        assert is_valid is True
        assert len(errors) == 0
        
        # Test invalid configuration (missing required field)
        invalid_config = {}
        is_valid, errors = validator.validate_config(invalid_config)
        assert is_valid is False
        assert len(errors) > 0

class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 10
        assert cb.state == CircuitState.CLOSED
    
    def test_circuit_breaker_success(self):
        """Test circuit breaker with successful calls."""
        cb = CircuitBreaker(failure_threshold=3)
        
        def successful_function():
            return "success"
        
        result = cb.call(successful_function)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_circuit_breaker_failure(self):
        """Test circuit breaker with failing calls."""
        cb = CircuitBreaker(failure_threshold=2)
        
        def failing_function():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.failure_count == 1
        assert cb.state == CircuitState.CLOSED
        
        # Second failure should open circuit
        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.failure_count == 2
        assert cb.state == CircuitState.OPEN

class TestModelManager:
    """Test model manager functionality."""
    
    def test_model_manager_initialization(self):
        """Test model manager initialization."""
        config = {
            'model_path': 'test_models/',
            'enable_auto_selection': True
        }
        manager = ModelManager(config)
        
        assert manager.model_path == 'test_models/'
        assert manager.enable_auto_selection is True
        assert isinstance(manager.models, dict)
    
    def test_model_registration(self):
        """Test model registration."""
        manager = ModelManager()
        
        # Create a mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=['easy'])
        
        # Register the model
        success = manager.register_model('test_model', mock_model)
        assert success is True
        assert 'test_model' in manager.models
    
    def test_model_selection(self):
        """Test automatic model selection."""
        manager = ModelManager({'enable_auto_selection': True})
        
        # Register multiple models with different performance
        mock_model1 = Mock()
        mock_model2 = Mock()
        
        manager.register_model('model1', mock_model1)
        manager.register_model('model2', mock_model2)
        
        # Set performance scores
        manager.model_performance['model1'] = 0.8
        manager.model_performance['model2'] = 0.9
        
        # Select best model
        best_model_name = manager.select_best_model()
        assert best_model_name == 'model2'

class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    def test_performance_monitor_initialization(self):
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor()
        
        assert isinstance(monitor.metrics_history, dict)
        assert isinstance(monitor.performance_thresholds, dict)
    
    def test_prediction_recording(self):
        """Test prediction performance recording."""
        monitor = PerformanceMonitor()
        
        # Record some predictions
        monitor.record_prediction('test_model', 100.0, True)
        monitor.record_prediction('test_model', 150.0, True)
        monitor.record_prediction('test_model', 200.0, False)
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        
        assert 'test_model' in summary
        assert 'total_predictions' in summary['test_model']
        assert summary['test_model']['total_predictions'] == 3

class TestErrorRecoverySystem:
    """Test error recovery system functionality."""
    
    def test_error_recovery_initialization(self):
        """Test error recovery system initialization."""
        config = {
            'enable_fallback_models': True,
            'enable_graceful_degradation': True
        }
        recovery = ErrorRecoverySystem(config)
        
        assert recovery.enable_fallback_models is True
        assert recovery.enable_graceful_degradation is True
    
    def test_prediction_error_handling(self):
        """Test prediction error handling."""
        recovery = ErrorRecoverySystem()
        
        # Mock error
        error = Exception("Test prediction error")
        input_data = {'text': 'test input'}
        correlation_id = 'test-correlation-id'
        
        # Handle the error
        result = recovery.handle_prediction_error(error, input_data, correlation_id)
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'error' in result or 'fallback_used' in result

# Integration test for the complete system
class TestSystemIntegration:
    """Test complete system integration."""
    
    def test_component_integration(self):
        """Test that all components can work together."""
        # Initialize components
        cache_manager = CacheManager({'local': {'enabled': True, 'max_size': 10}})
        health_monitor = HealthMonitor()
        structured_logger = StructuredLogger('integration-test')
        
        # Test basic operations
        cache_manager.set('test_key', 'test_value')
        value, status = cache_manager.get('test_key')
        
        assert value == 'test_value'
        assert status == CacheStatus.HIT
        
        # Test health monitoring
        health_status = health_monitor.get_health_status()
        assert 'status' in health_status
        
        # Test logging
        structured_logger.log_info('Integration test completed successfully')
        
        # All components should work without conflicts
        assert True

# Pytest fixtures
@pytest.fixture
def cache_manager():
    """Fixture for cache manager."""
    config = {'local': {'enabled': True, 'max_size': 100}}
    return CacheManager(config)

@pytest.fixture
def health_monitor():
    """Fixture for health monitor."""
    return HealthMonitor(check_interval=1)

@pytest.fixture
def structured_logger():
    """Fixture for structured logger."""
    return StructuredLogger('test-service', '1.0.0', 'test')

@pytest.fixture
def circuit_breaker():
    """Fixture for circuit breaker."""
    return CircuitBreaker(failure_threshold=3, recovery_timeout=10)

# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])