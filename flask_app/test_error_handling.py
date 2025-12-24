"""
Unit tests for enhanced error handling and monitoring system.

These tests validate error handling, performance monitoring,
and system health tracking functionality.
"""

import pytest
import numpy as np
import pandas as pd
import scipy.sparse
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from error_handler import (
    ErrorHandler, ErrorSeverity, SystemComponent, ErrorEvent, PerformanceAlert,
    error_handler
)
from evaluation_models import ClassificationMetrics, RegressionMetrics
from app import ModelEvaluator


class TestErrorHandler:
    """Unit tests for ErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
        
    def test_initialization(self):
        """Test ErrorHandler initialization."""
        assert self.error_handler.error_history == []
        assert self.error_handler.performance_alerts == []
        assert self.error_handler.system_health['status'] == 'healthy'
        assert len(self.error_handler.performance_thresholds) > 0
        assert 'min_accuracy' in self.error_handler.performance_thresholds
        
    def test_log_error(self):
        """Test error logging functionality."""
        test_error = ValueError("Test error message")
        
        error_event = self.error_handler.log_error(
            component=SystemComponent.MODEL_TRAINING,
            error=test_error,
            severity=ErrorSeverity.HIGH,
            context={'test_key': 'test_value'},
            recovery_action="Used fallback model"
        )
        
        # Check error event properties
        assert isinstance(error_event, ErrorEvent)
        assert error_event.component == SystemComponent.MODEL_TRAINING
        assert error_event.severity == ErrorSeverity.HIGH
        assert error_event.error_type == "ValueError"
        assert error_event.message == "Test error message"
        assert error_event.details['test_key'] == 'test_value'
        assert error_event.recovery_action == "Used fallback model"
        assert error_event.stack_trace is not None
        
        # Check error history
        assert len(self.error_handler.error_history) == 1
        assert self.error_handler.error_history[0] == error_event
        
        # Check system health update
        assert self.error_handler.system_health['component_status'][SystemComponent.MODEL_TRAINING.value] == 'degraded'
    
    def test_monitor_performance_threshold_violation(self):
        """Test performance monitoring with threshold violations."""
        # Test accuracy below threshold
        alert = self.error_handler.monitor_performance('accuracy', 0.4)
        
        assert alert is not None
        assert isinstance(alert, PerformanceAlert)
        assert alert.metric_name == 'accuracy'
        assert alert.actual_value == 0.4
        assert alert.threshold_value == 0.6
        assert alert.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]
        assert len(alert.recommendation) > 0
        
        # Check alert history
        assert len(self.error_handler.performance_alerts) == 1
    
    def test_monitor_performance_no_violation(self):
        """Test performance monitoring without threshold violations."""
        # Test accuracy above threshold
        alert = self.error_handler.monitor_performance('accuracy', 0.8)
        
        assert alert is None
        assert len(self.error_handler.performance_alerts) == 0
    
    def test_monitor_performance_custom_threshold(self):
        """Test performance monitoring with custom thresholds."""
        # Test with custom threshold
        alert = self.error_handler.monitor_performance('accuracy', 0.7, threshold_value=0.8)
        
        assert alert is not None
        assert alert.threshold_value == 0.8
    
    def test_safe_execute_success(self):
        """Test safe execution with successful function."""
        def successful_function():
            return "success"
        
        result = self.error_handler.safe_execute(
            func=successful_function,
            component=SystemComponent.PREDICTION_SERVICE,
            fallback_value="fallback"
        )
        
        assert result == "success"
        assert len(self.error_handler.error_history) == 0
    
    def test_safe_execute_with_error(self):
        """Test safe execution with failing function."""
        def failing_function():
            raise RuntimeError("Test failure")
        
        result = self.error_handler.safe_execute(
            func=failing_function,
            component=SystemComponent.PREDICTION_SERVICE,
            fallback_value="fallback",
            context={'test': 'context'}
        )
        
        assert result == "fallback"
        assert len(self.error_handler.error_history) == 1
        
        error_event = self.error_handler.error_history[0]
        assert error_event.component == SystemComponent.PREDICTION_SERVICE
        assert error_event.error_type == "RuntimeError"
        assert error_event.message == "Test failure"
        assert error_event.details['test'] == 'context'
    
    def test_safe_execute_no_fallback(self):
        """Test safe execution without fallback value."""
        def failing_function():
            raise RuntimeError("Test failure")
        
        result = self.error_handler.safe_execute(
            func=failing_function,
            component=SystemComponent.PREDICTION_SERVICE,
            fallback_value=None
        )
        
        assert result is None
        assert len(self.error_handler.error_history) == 1
        
        error_event = self.error_handler.error_history[0]
        assert error_event.severity == ErrorSeverity.HIGH  # Higher severity without fallback
    
    def test_get_fallback_metrics(self):
        """Test fallback metrics generation."""
        # Test classification fallback
        classification_fallback = self.error_handler.get_fallback_classification_metrics("Test reason")
        
        assert isinstance(classification_fallback, ClassificationMetrics)
        assert classification_fallback.accuracy == 0.33  # Random guess for 3 classes
        assert classification_fallback.classification_report['fallback_reason'] == "Test reason"
        
        # Test regression fallback
        regression_fallback = self.error_handler.get_fallback_regression_metrics("Test reason")
        
        assert isinstance(regression_fallback, RegressionMetrics)
        assert regression_fallback.mae == 5.0
        assert regression_fallback.rmse == 6.0
        assert regression_fallback.r2_score == -1.0
    
    def test_validate_model_performance(self):
        """Test model performance validation."""
        # Create test metrics
        classification_metrics = ClassificationMetrics(
            accuracy=0.45,  # Below threshold
            confusion_matrix=np.array([[10, 5], [3, 12]]),
            classification_report={}
        )
        
        regression_metrics = RegressionMetrics(
            mae=2.5,  # Above threshold
            rmse=3.0,  # Above threshold
            r2_score=0.2
        )
        
        validation_results = self.error_handler.validate_model_performance(
            classification_metrics, regression_metrics
        )
        
        # Check validation structure
        assert 'alerts' in validation_results
        assert 'recommendations' in validation_results
        assert 'overall_status' in validation_results
        
        # Should have alerts for poor performance
        assert len(validation_results['alerts']) > 0
        assert len(validation_results['recommendations']) > 0
        assert validation_results['overall_status'] in ['warning', 'degraded', 'critical']
    
    def test_get_system_health_report(self):
        """Test system health report generation."""
        # Add some test errors and alerts
        self.error_handler.log_error(
            SystemComponent.MODEL_TRAINING,
            ValueError("Test error"),
            ErrorSeverity.MEDIUM
        )
        
        self.error_handler.monitor_performance('accuracy', 0.4)
        
        health_report = self.error_handler.get_system_health_report()
        
        # Check report structure
        assert 'overall_status' in health_report
        assert 'last_check' in health_report
        assert 'component_status' in health_report
        assert 'recent_errors' in health_report
        assert 'recent_alerts' in health_report
        assert 'error_summary' in health_report
        assert 'recommendations' in health_report
        
        # Check content
        assert health_report['recent_errors'] >= 1
        assert health_report['recent_alerts'] >= 1
        assert len(health_report['recommendations']) > 0
    
    def test_reset_health_status(self):
        """Test health status reset."""
        # Add some errors first
        self.error_handler.log_error(
            SystemComponent.MODEL_TRAINING,
            ValueError("Test error"),
            ErrorSeverity.HIGH
        )
        
        # Verify degraded status
        assert self.error_handler.system_health['component_status'][SystemComponent.MODEL_TRAINING.value] == 'degraded'
        
        # Reset health
        self.error_handler.reset_health_status()
        
        # Verify reset
        assert self.error_handler.system_health['status'] == 'healthy'
        assert self.error_handler.system_health['component_status'][SystemComponent.MODEL_TRAINING.value] == 'healthy'


class TestEnhancedModelEvaluator:
    """Unit tests for enhanced ModelEvaluator with error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator(test_size=0.2, random_state=42)
        
        # Create sample data
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 10
        
        self.X = scipy.sparse.random(self.n_samples, self.n_features, density=0.3, random_state=42)
        self.y_class = pd.Series(np.random.choice(['easy', 'medium', 'hard'], self.n_samples))
        self.y_score = pd.Series(np.random.uniform(1.0, 10.0, self.n_samples))
    
    def test_train_test_split_error_handling(self):
        """Test train/test split with error handling."""
        # Test with valid data
        result = self.evaluator.perform_train_test_split(self.X, self.y_class, self.y_score)
        assert result is not None
        assert len(result) == 6  # Should return 6 elements
        
        # Test with invalid data - should raise exception
        with pytest.raises(RuntimeError):
            self.evaluator.perform_train_test_split(None, self.y_class, self.y_score)
    
    def test_classification_evaluation_error_handling(self):
        """Test classification evaluation with error handling."""
        # First perform split
        self.evaluator.perform_train_test_split(self.X, self.y_class, self.y_score)
        
        # Train a simple model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(self.evaluator.X_train, self.evaluator.y_train_class)
        
        # Test successful evaluation
        metrics = self.evaluator.evaluate_classification(model)
        assert isinstance(metrics, ClassificationMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        
        # Test with None model - should return fallback
        fallback_metrics = self.evaluator.evaluate_classification(None)
        assert isinstance(fallback_metrics, ClassificationMetrics)
        assert 'fallback_reason' in fallback_metrics.classification_report
    
    def test_regression_evaluation_error_handling(self):
        """Test regression evaluation with error handling."""
        # First perform split
        self.evaluator.perform_train_test_split(self.X, self.y_class, self.y_score)
        
        # Train a simple model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(self.evaluator.X_train, self.evaluator.y_train_score)
        
        # Test successful evaluation
        metrics = self.evaluator.evaluate_regression(model)
        assert isinstance(metrics, RegressionMetrics)
        assert metrics.mae >= 0.0
        assert metrics.rmse >= 0.0
        
        # Test with None model - should return fallback
        fallback_metrics = self.evaluator.evaluate_regression(None)
        assert isinstance(fallback_metrics, RegressionMetrics)
        assert fallback_metrics.mae == 5.0  # Fallback value
    
    def test_performance_validation_enhanced(self):
        """Test enhanced performance validation."""
        # Create test metrics
        classification_metrics = ClassificationMetrics(
            accuracy=0.75,
            confusion_matrix=np.array([[10, 2], [3, 15]]),
            classification_report={}
        )
        
        regression_metrics = RegressionMetrics(mae=1.5, rmse=1.8, r2_score=0.65)
        
        # Test validation
        validation_results = self.evaluator.validate_performance_thresholds(
            classification_metrics, regression_metrics
        )
        
        # Check enhanced validation structure
        assert 'enhanced_validation' in validation_results
        assert 'system_health' in validation_results
        assert 'accuracy_acceptable' in validation_results  # Traditional results
        
        # Check enhanced validation content
        enhanced = validation_results['enhanced_validation']
        assert 'alerts' in enhanced
        assert 'recommendations' in enhanced
        assert 'overall_status' in enhanced


class TestDocumentationErrorHandling:
    """Unit tests for documentation generation error handling."""
    
    def test_documentation_generator_error_handling(self):
        """Test documentation generator error handling."""
        from documentation_generator import DocumentationGenerator
        from evaluation_models import EvaluationReport, ClassificationMetrics, RegressionMetrics
        
        doc_gen = DocumentationGenerator()
        
        # Test with invalid evaluation report
        doc_gen.set_evaluation_report(None)  # Should handle gracefully
        
        # Test with valid report
        classification_metrics = ClassificationMetrics(
            accuracy=0.75,
            confusion_matrix=np.array([[10, 2], [3, 15]]),
            classification_report={}
        )
        
        regression_metrics = RegressionMetrics(mae=1.5, rmse=1.8, r2_score=0.65)
        
        evaluation_report = EvaluationReport(
            classification_metrics=classification_metrics,
            regression_metrics=regression_metrics,
            dataset_info={'total_samples': 100, 'train_samples': 80, 'test_samples': 20, 'feature_count': 50},
            model_info={'classification_model': 'LogisticRegression', 'regression_model': 'RandomForestRegressor'}
        )
        
        doc_gen.set_evaluation_report(evaluation_report)
        assert doc_gen.evaluation_report is not None
    
    def test_documentation_save_error_handling(self):
        """Test documentation save error handling."""
        from documentation_generator import DocumentationGenerator, DocumentationConfig
        from evaluation_models import EvaluationReport, ClassificationMetrics, RegressionMetrics
        
        # Create valid documentation generator
        config = DocumentationConfig(dataset_size=100)
        doc_gen = DocumentationGenerator(config)
        
        # Add evaluation report
        classification_metrics = ClassificationMetrics(
            accuracy=0.75,
            confusion_matrix=np.array([[10, 2], [3, 15]]),
            classification_report={}
        )
        
        regression_metrics = RegressionMetrics(mae=1.5, rmse=1.8, r2_score=0.65)
        
        evaluation_report = EvaluationReport(
            classification_metrics=classification_metrics,
            regression_metrics=regression_metrics,
            dataset_info={'total_samples': 100, 'train_samples': 80, 'test_samples': 20, 'feature_count': 50},
            model_info={'classification_model': 'LogisticRegression', 'regression_model': 'RandomForestRegressor'}
        )
        
        doc_gen.set_evaluation_report(evaluation_report)
        
        # Test save to invalid path - should return False
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file where we want to create a directory
            invalid_path = f"{temp_dir}/file.txt"
            with open(invalid_path, 'w') as f:
                f.write("test")
            
            # Try to save to conflicting path
            conflicting_path = f"{invalid_path}/README.md"
            success = doc_gen.save_documentation(conflicting_path)
            assert success is False


class TestGlobalErrorHandler:
    """Unit tests for global error handler instance."""
    
    def test_global_error_handler_exists(self):
        """Test that global error handler instance exists."""
        from error_handler import error_handler
        
        assert error_handler is not None
        assert isinstance(error_handler, ErrorHandler)
    
    def test_global_error_handler_functionality(self):
        """Test global error handler basic functionality."""
        from error_handler import error_handler
        
        # Clear any existing history
        error_handler.error_history.clear()
        error_handler.performance_alerts.clear()
        
        # Test logging
        test_error = ValueError("Global test error")
        error_handler.log_error(
            SystemComponent.API_ENDPOINT,
            test_error,
            ErrorSeverity.LOW
        )
        
        assert len(error_handler.error_history) == 1
        
        # Test performance monitoring
        alert = error_handler.monitor_performance('accuracy', 0.3)
        assert alert is not None
        assert len(error_handler.performance_alerts) == 1


if __name__ == '__main__':
    # Run unit tests
    pytest.main([__file__, '-v'])