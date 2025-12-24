"""
Enhanced Error Handling and Monitoring System for AutoJudge

This module provides comprehensive error handling, logging, and monitoring
capabilities for the AutoJudge system with graceful degradation.
"""

import logging
import traceback
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from evaluation_models import ClassificationMetrics, RegressionMetrics


class ErrorSeverity(Enum):
    """Error severity levels for monitoring and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SystemComponent(Enum):
    """System components for error categorization."""
    DATA_LOADING = "data_loading"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    PREDICTION_SERVICE = "prediction_service"
    DOCUMENTATION_GENERATOR = "documentation_generator"
    API_ENDPOINT = "api_endpoint"


@dataclass
class ErrorEvent:
    """Data class for tracking error events."""
    timestamp: datetime
    component: SystemComponent
    severity: ErrorSeverity
    error_type: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None


@dataclass
class PerformanceAlert:
    """Data class for performance monitoring alerts."""
    timestamp: datetime
    metric_name: str
    actual_value: float
    threshold_value: float
    severity: ErrorSeverity
    message: str
    recommendation: str


class ErrorHandler:
    """
    Comprehensive error handling and monitoring system.
    
    Provides graceful error handling, performance monitoring,
    and system health tracking with fallback mechanisms.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize error handler with logging configuration."""
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: list[ErrorEvent] = []
        self.performance_alerts: list[PerformanceAlert] = []
        self.system_health = {
            'status': 'healthy',
            'last_check': datetime.now(),
            'component_status': {component.value: 'healthy' for component in SystemComponent}
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'min_accuracy': 0.6,
            'max_mae': 2.0,
            'max_rmse': 2.5,
            'min_r2': -1.0,  # R² can be negative for very poor models
            'max_training_time': 300,  # 5 minutes
            'max_prediction_time': 5.0  # 5 seconds
        }
        
        # Fallback values for degraded operation
        self.fallback_metrics = {
            'classification': ClassificationMetrics(
                accuracy=0.33,  # Random guess for 3 classes
                confusion_matrix=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
                classification_report={'warning': 'Fallback metrics - model evaluation failed'}
            ),
            'regression': RegressionMetrics(
                mae=5.0,  # High error indicating poor performance
                rmse=6.0,
                r2_score=-1.0  # Worse than baseline
            )
        }
    
    def log_error(self, component: SystemComponent, error: Exception, 
                  severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                  context: Optional[Dict[str, Any]] = None,
                  recovery_action: Optional[str] = None) -> ErrorEvent:
        """
        Log an error event with comprehensive details.
        
        Args:
            component: System component where error occurred
            error: The exception that occurred
            severity: Error severity level
            context: Additional context information
            recovery_action: Description of recovery action taken
            
        Returns:
            ErrorEvent object
        """
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            component=component,
            severity=severity,
            error_type=type(error).__name__,
            message=str(error),
            details=context or {},
            stack_trace=traceback.format_exc(),
            recovery_action=recovery_action
        )
        
        self.error_history.append(error_event)
        
        # Update system health
        self.system_health['component_status'][component.value] = 'degraded' if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else 'warning'
        self.system_health['last_check'] = datetime.now()
        
        # Log based on severity
        log_message = f"[{component.value}] {error_event.error_type}: {error_event.message}"
        if recovery_action:
            log_message += f" | Recovery: {recovery_action}"
            
        if severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
            
        return error_event
    
    def monitor_performance(self, metric_name: str, actual_value: float,
                          threshold_value: Optional[float] = None,
                          comparison: str = 'min') -> Optional[PerformanceAlert]:
        """
        Monitor performance metrics against thresholds.
        
        Args:
            metric_name: Name of the metric being monitored
            actual_value: Actual metric value
            threshold_value: Threshold value (uses default if None)
            comparison: 'min' for minimum threshold, 'max' for maximum
            
        Returns:
            PerformanceAlert if threshold violated, None otherwise
        """
        if threshold_value is None:
            threshold_key = f"{'min' if comparison == 'min' else 'max'}_{metric_name.lower()}"
            threshold_value = self.performance_thresholds.get(threshold_key)
            
        if threshold_value is None:
            self.logger.warning(f"No threshold defined for metric: {metric_name}")
            return None
        
        # Check threshold violation
        threshold_violated = False
        if comparison == 'min':
            threshold_violated = actual_value < threshold_value
        else:  # comparison == 'max'
            threshold_violated = actual_value > threshold_value
            
        if threshold_violated:
            # Determine severity based on how far from threshold
            deviation = abs(actual_value - threshold_value) / abs(threshold_value) if threshold_value != 0 else 1.0
            
            if deviation > 0.5:
                severity = ErrorSeverity.HIGH
            elif deviation > 0.2:
                severity = ErrorSeverity.MEDIUM
            else:
                severity = ErrorSeverity.LOW
                
            # Generate recommendation
            if metric_name.lower() == 'accuracy':
                recommendation = "Consider retraining with more data or different features"
            elif metric_name.lower() in ['mae', 'rmse']:
                recommendation = "Review feature engineering and model hyperparameters"
            elif metric_name.lower() == 'r2':
                recommendation = "Model may be worse than baseline - investigate data quality"
            else:
                recommendation = "Review system performance and consider optimization"
                
            alert = PerformanceAlert(
                timestamp=datetime.now(),
                metric_name=metric_name,
                actual_value=actual_value,
                threshold_value=threshold_value,
                severity=severity,
                message=f"{metric_name} {actual_value:.3f} {'below' if comparison == 'min' else 'above'} threshold {threshold_value:.3f}",
                recommendation=recommendation
            )
            
            self.performance_alerts.append(alert)
            
            # Log the alert
            self.logger.warning(f"Performance Alert: {alert.message} | {alert.recommendation}")
            
            return alert
            
        return None
    
    def safe_execute(self, func: Callable, component: SystemComponent,
                    fallback_value: Any = None, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Safely execute a function with comprehensive error handling.
        
        Args:
            func: Function to execute
            component: System component for error tracking
            fallback_value: Value to return if function fails
            context: Additional context for error logging
            
        Returns:
            Function result or fallback value
        """
        try:
            start_time = time.time()
            result = func()
            execution_time = time.time() - start_time
            
            # Monitor execution time
            if execution_time > self.performance_thresholds.get('max_prediction_time', 5.0):
                self.monitor_performance('execution_time', execution_time, 
                                       self.performance_thresholds.get('max_prediction_time', 5.0), 'max')
            
            return result
            
        except Exception as e:
            recovery_action = f"Returned fallback value: {fallback_value}" if fallback_value is not None else "No fallback available"
            
            self.log_error(
                component=component,
                error=e,
                severity=ErrorSeverity.HIGH if fallback_value is None else ErrorSeverity.MEDIUM,
                context=context,
                recovery_action=recovery_action
            )
            
            return fallback_value
    
    def get_fallback_classification_metrics(self, reason: str = "Evaluation failed") -> ClassificationMetrics:
        """
        Get fallback classification metrics for degraded operation.
        
        Args:
            reason: Reason for using fallback metrics
            
        Returns:
            Fallback ClassificationMetrics
        """
        self.logger.warning(f"Using fallback classification metrics: {reason}")
        
        fallback = self.fallback_metrics['classification']
        fallback.classification_report['fallback_reason'] = reason
        
        return fallback
    
    def get_fallback_regression_metrics(self, reason: str = "Evaluation failed") -> RegressionMetrics:
        """
        Get fallback regression metrics for degraded operation.
        
        Args:
            reason: Reason for using fallback metrics
            
        Returns:
            Fallback RegressionMetrics
        """
        self.logger.warning(f"Using fallback regression metrics: {reason}")
        return self.fallback_metrics['regression']
    
    def validate_model_performance(self, classification_metrics: ClassificationMetrics,
                                 regression_metrics: RegressionMetrics) -> Dict[str, Any]:
        """
        Validate model performance with enhanced monitoring.
        
        Args:
            classification_metrics: Classification evaluation results
            regression_metrics: Regression evaluation results
            
        Returns:
            Validation results with alerts and recommendations
        """
        validation_results = {
            'alerts': [],
            'recommendations': [],
            'overall_status': 'healthy'
        }
        
        # Monitor classification performance
        accuracy_alert = self.monitor_performance('accuracy', classification_metrics.accuracy)
        if accuracy_alert:
            validation_results['alerts'].append(accuracy_alert)
            validation_results['recommendations'].append(accuracy_alert.recommendation)
        
        # Monitor regression performance
        mae_alert = self.monitor_performance('mae', regression_metrics.mae, comparison='max')
        if mae_alert:
            validation_results['alerts'].append(mae_alert)
            validation_results['recommendations'].append(mae_alert.recommendation)
            
        rmse_alert = self.monitor_performance('rmse', regression_metrics.rmse, comparison='max')
        if rmse_alert:
            validation_results['alerts'].append(rmse_alert)
            validation_results['recommendations'].append(rmse_alert.recommendation)
            
        r2_alert = self.monitor_performance('r2', regression_metrics.r2_score)
        if r2_alert:
            validation_results['alerts'].append(r2_alert)
            validation_results['recommendations'].append(r2_alert.recommendation)
        
        # Determine overall status
        if any(alert.severity == ErrorSeverity.CRITICAL for alert in validation_results['alerts']):
            validation_results['overall_status'] = 'critical'
        elif any(alert.severity == ErrorSeverity.HIGH for alert in validation_results['alerts']):
            validation_results['overall_status'] = 'degraded'
        elif validation_results['alerts']:
            validation_results['overall_status'] = 'warning'
        
        return validation_results
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive system health report.
        
        Returns:
            System health report with status and recommendations
        """
        recent_errors = [e for e in self.error_history if (datetime.now() - e.timestamp).seconds < 3600]  # Last hour
        recent_alerts = [a for a in self.performance_alerts if (datetime.now() - a.timestamp).seconds < 3600]
        
        return {
            'overall_status': self.system_health['status'],
            'last_check': self.system_health['last_check'].isoformat(),
            'component_status': self.system_health['component_status'],
            'recent_errors': len(recent_errors),
            'recent_alerts': len(recent_alerts),
            'error_summary': {
                severity.value: len([e for e in recent_errors if e.severity == severity])
                for severity in ErrorSeverity
            },
            'recommendations': list(set([a.recommendation for a in recent_alerts]))
        }
    
    def reset_health_status(self):
        """Reset system health status to healthy."""
        self.system_health['status'] = 'healthy'
        self.system_health['component_status'] = {component.value: 'healthy' for component in SystemComponent}
        self.system_health['last_check'] = datetime.now()
        self.logger.info("System health status reset to healthy")


# Global error handler instance
error_handler = ErrorHandler()