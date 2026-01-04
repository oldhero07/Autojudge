"""
Performance Monitor for ML Web App Enhancements

This module provides comprehensive performance monitoring and tracking
for ML models including metrics collection, degradation detection,
and retraining triggers.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import threading
import time
from statistics import mean, stdev

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class PredictionMetric:
    """Individual prediction metric record"""
    model_name: str
    prediction_time: float
    confidence: float
    timestamp: datetime
    input_hash: Optional[str] = None
    prediction_result: Optional[str] = None

@dataclass
class ModelPerformanceSnapshot:
    """Snapshot of model performance at a point in time"""
    model_name: str
    timestamp: datetime
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2_score: Optional[float] = None
    avg_response_time: Optional[float] = None
    avg_confidence: Optional[float] = None
    error_rate: Optional[float] = None
    prediction_count: int = 0

@dataclass
class PerformanceAlert:
    """Performance alert record"""
    alert_id: str
    model_name: str
    metric_name: str
    current_value: float
    threshold_value: float
    severity: str  # 'warning', 'critical'
    message: str
    timestamp: datetime
    resolved: bool = False

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for ML models
    """
    
    def __init__(self, config=None):
        """Initialize the PerformanceMonitor"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Performance data storage
        self.prediction_metrics = defaultdict(deque)  # model_name -> deque of PredictionMetric
        self.performance_snapshots = defaultdict(list)  # model_name -> list of ModelPerformanceSnapshot
        self.alerts = []
        
        # Configuration
        self.max_metrics_per_model = self.config.get('max_metrics_per_model', 10000)
        self.snapshot_interval = self.config.get('snapshot_interval', 300)  # 5 minutes
        self.time_window = self.config.get('time_window', 3600)  # 1 hour
        
        # Performance thresholds
        self.thresholds = {
            'accuracy': {'warning': 0.7, 'critical': 0.6},
            'f1_score': {'warning': 0.7, 'critical': 0.6},
            'mae': {'warning': 2.0, 'critical': 3.0},
            'rmse': {'warning': 2.5, 'critical': 3.5},
            'response_time': {'warning': 1000, 'critical': 2000},  # milliseconds
            'error_rate': {'warning': 0.05, 'critical': 0.1},
            'confidence': {'warning': 0.6, 'critical': 0.5}
        }
        
        # Degradation detection
        self.degradation_window = self.config.get('degradation_window', 100)  # Number of predictions
        self.degradation_threshold = self.config.get('degradation_threshold', 0.1)  # 10% degradation
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.logger.info("PerformanceMonitor initialized")
    
    def track_prediction(self, model_name: str, prediction_time: float, confidence: float, 
                        input_hash: Optional[str] = None, prediction_result: Optional[str] = None):
        """
        Track a single prediction metric
        
        Args:
            model_name: Name of the model that made the prediction
            prediction_time: Time taken for prediction in milliseconds
            confidence: Confidence score of the prediction
            input_hash: Optional hash of input data
            prediction_result: Optional prediction result
        """
        with self.lock:
            metric = PredictionMetric(
                model_name=model_name,
                prediction_time=prediction_time,
                confidence=confidence,
                timestamp=datetime.now(),
                input_hash=input_hash,
                prediction_result=prediction_result
            )
            
            # Add to metrics deque
            self.prediction_metrics[model_name].append(metric)
            
            # Maintain maximum size
            if len(self.prediction_metrics[model_name]) > self.max_metrics_per_model:
                self.prediction_metrics[model_name].popleft()
            
            # Check for real-time alerts
            self._check_realtime_alerts(model_name, metric)
    
    def get_model_metrics(self, model_name: str, time_window: int = None) -> Dict[str, Any]:
        """
        Get performance metrics for a specific model within a time window
        
        Args:
            model_name: Name of the model
            time_window: Time window in seconds (default: use configured window)
            
        Returns:
            Dictionary containing performance metrics
        """
        if time_window is None:
            time_window = self.time_window
        
        with self.lock:
            if model_name not in self.prediction_metrics:
                return {}
            
            # Filter metrics within time window
            cutoff_time = datetime.now() - timedelta(seconds=time_window)
            recent_metrics = [
                metric for metric in self.prediction_metrics[model_name]
                if metric.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {}
            
            # Calculate metrics
            response_times = [m.prediction_time for m in recent_metrics]
            confidences = [m.confidence for m in recent_metrics]
            
            metrics = {
                'prediction_count': len(recent_metrics),
                'avg_response_time': mean(response_times),
                'max_response_time': max(response_times),
                'min_response_time': min(response_times),
                'response_time_std': stdev(response_times) if len(response_times) > 1 else 0,
                'avg_confidence': mean(confidences),
                'min_confidence': min(confidences),
                'max_confidence': max(confidences),
                'confidence_std': stdev(confidences) if len(confidences) > 1 else 0,
                'time_window_seconds': time_window,
                'latest_prediction': recent_metrics[-1].timestamp.isoformat()
            }
            
            # Add performance snapshot data if available
            if model_name in self.performance_snapshots and self.performance_snapshots[model_name]:
                latest_snapshot = self.performance_snapshots[model_name][-1]
                metrics.update({
                    'accuracy': latest_snapshot.accuracy,
                    'precision': latest_snapshot.precision,
                    'recall': latest_snapshot.recall,
                    'f1_score': latest_snapshot.f1_score,
                    'mae': latest_snapshot.mae,
                    'rmse': latest_snapshot.rmse,
                    'r2_score': latest_snapshot.r2_score,
                    'error_rate': latest_snapshot.error_rate
                })
            
            return metrics
    
    def update_model_performance(self, model_name: str, performance_metrics: Dict[str, float]):
        """
        Update model performance metrics (accuracy, precision, etc.)
        
        Args:
            model_name: Name of the model
            performance_metrics: Dictionary of metric_name -> value
        """
        with self.lock:
            # Get recent prediction metrics for additional context
            recent_metrics = self._get_recent_prediction_metrics(model_name)
            
            snapshot = ModelPerformanceSnapshot(
                model_name=model_name,
                timestamp=datetime.now(),
                accuracy=performance_metrics.get('accuracy'),
                precision=performance_metrics.get('precision'),
                recall=performance_metrics.get('recall'),
                f1_score=performance_metrics.get('f1_score'),
                mae=performance_metrics.get('mae'),
                rmse=performance_metrics.get('rmse'),
                r2_score=performance_metrics.get('r2_score'),
                avg_response_time=recent_metrics.get('avg_response_time'),
                avg_confidence=recent_metrics.get('avg_confidence'),
                error_rate=performance_metrics.get('error_rate', 0.0),
                prediction_count=recent_metrics.get('prediction_count', 0)
            )
            
            self.performance_snapshots[model_name].append(snapshot)
            
            # Check for performance alerts
            self._check_performance_alerts(model_name, snapshot)
            
            self.logger.info(f"Updated performance metrics for {model_name}")
    
    def _get_recent_prediction_metrics(self, model_name: str) -> Dict[str, Any]:
        """Get recent prediction metrics for a model"""
        if model_name not in self.prediction_metrics:
            return {}
        
        cutoff_time = datetime.now() - timedelta(seconds=self.time_window)
        recent_metrics = [
            metric for metric in self.prediction_metrics[model_name]
            if metric.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        response_times = [m.prediction_time for m in recent_metrics]
        confidences = [m.confidence for m in recent_metrics]
        
        return {
            'prediction_count': len(recent_metrics),
            'avg_response_time': mean(response_times),
            'avg_confidence': mean(confidences)
        }
    
    def trigger_retraining(self, model_name: str) -> bool:
        """
        Determine if model needs retraining based on performance degradation
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if retraining is recommended, False otherwise
        """
        with self.lock:
            if model_name not in self.performance_snapshots:
                return False
            
            snapshots = self.performance_snapshots[model_name]
            if len(snapshots) < 2:
                return False
            
            # Check for degradation in key metrics
            latest_snapshot = snapshots[-1]
            
            # Compare with baseline (first snapshot or average of first few)
            if len(snapshots) >= 5:
                baseline_snapshots = snapshots[:5]
            else:
                baseline_snapshots = snapshots[:-1]
            
            degradation_detected = False
            
            # Check accuracy degradation
            if latest_snapshot.accuracy is not None:
                baseline_accuracy = mean([s.accuracy for s in baseline_snapshots if s.accuracy is not None])
                if baseline_accuracy > 0:
                    accuracy_degradation = (baseline_accuracy - latest_snapshot.accuracy) / baseline_accuracy
                    if accuracy_degradation > self.degradation_threshold:
                        degradation_detected = True
                        self.logger.warning(f"Accuracy degradation detected for {model_name}: {accuracy_degradation:.3f}")
            
            # Check F1 score degradation
            if latest_snapshot.f1_score is not None:
                baseline_f1 = mean([s.f1_score for s in baseline_snapshots if s.f1_score is not None])
                if baseline_f1 > 0:
                    f1_degradation = (baseline_f1 - latest_snapshot.f1_score) / baseline_f1
                    if f1_degradation > self.degradation_threshold:
                        degradation_detected = True
                        self.logger.warning(f"F1 score degradation detected for {model_name}: {f1_degradation:.3f}")
            
            # Check MAE degradation (increase is bad)
            if latest_snapshot.mae is not None:
                baseline_mae = mean([s.mae for s in baseline_snapshots if s.mae is not None])
                if baseline_mae > 0:
                    mae_degradation = (latest_snapshot.mae - baseline_mae) / baseline_mae
                    if mae_degradation > self.degradation_threshold:
                        degradation_detected = True
                        self.logger.warning(f"MAE degradation detected for {model_name}: {mae_degradation:.3f}")
            
            # Check confidence degradation
            if latest_snapshot.avg_confidence is not None:
                baseline_confidence = mean([s.avg_confidence for s in baseline_snapshots if s.avg_confidence is not None])
                if baseline_confidence > 0:
                    confidence_degradation = (baseline_confidence - latest_snapshot.avg_confidence) / baseline_confidence
                    if confidence_degradation > self.degradation_threshold:
                        degradation_detected = True
                        self.logger.warning(f"Confidence degradation detected for {model_name}: {confidence_degradation:.3f}")
            
            if degradation_detected:
                # Generate retraining alert
                alert = PerformanceAlert(
                    alert_id=f"retrain_{model_name}_{int(time.time())}",
                    model_name=model_name,
                    metric_name="performance_degradation",
                    current_value=0.0,  # Placeholder
                    threshold_value=self.degradation_threshold,
                    severity="critical",
                    message=f"Model {model_name} shows performance degradation and should be retrained",
                    timestamp=datetime.now()
                )
                self.alerts.append(alert)
                
                self.logger.critical(f"Retraining recommended for {model_name}")
            
            return degradation_detected
    
    def _check_realtime_alerts(self, model_name: str, metric: PredictionMetric):
        """Check for real-time alerts based on individual prediction metrics"""
        
        # Check response time
        if metric.prediction_time > self.thresholds['response_time']['critical']:
            self._generate_alert(
                model_name, 'response_time', metric.prediction_time,
                self.thresholds['response_time']['critical'], 'critical',
                f"Response time {metric.prediction_time:.1f}ms exceeds critical threshold"
            )
        elif metric.prediction_time > self.thresholds['response_time']['warning']:
            self._generate_alert(
                model_name, 'response_time', metric.prediction_time,
                self.thresholds['response_time']['warning'], 'warning',
                f"Response time {metric.prediction_time:.1f}ms exceeds warning threshold"
            )
        
        # Check confidence
        if metric.confidence < self.thresholds['confidence']['critical']:
            self._generate_alert(
                model_name, 'confidence', metric.confidence,
                self.thresholds['confidence']['critical'], 'critical',
                f"Confidence {metric.confidence:.3f} below critical threshold"
            )
        elif metric.confidence < self.thresholds['confidence']['warning']:
            self._generate_alert(
                model_name, 'confidence', metric.confidence,
                self.thresholds['confidence']['warning'], 'warning',
                f"Confidence {metric.confidence:.3f} below warning threshold"
            )
    
    def _check_performance_alerts(self, model_name: str, snapshot: ModelPerformanceSnapshot):
        """Check for performance alerts based on model performance snapshots"""
        
        # Check accuracy
        if snapshot.accuracy is not None:
            if snapshot.accuracy < self.thresholds['accuracy']['critical']:
                self._generate_alert(
                    model_name, 'accuracy', snapshot.accuracy,
                    self.thresholds['accuracy']['critical'], 'critical',
                    f"Accuracy {snapshot.accuracy:.3f} below critical threshold"
                )
            elif snapshot.accuracy < self.thresholds['accuracy']['warning']:
                self._generate_alert(
                    model_name, 'accuracy', snapshot.accuracy,
                    self.thresholds['accuracy']['warning'], 'warning',
                    f"Accuracy {snapshot.accuracy:.3f} below warning threshold"
                )
        
        # Check F1 score
        if snapshot.f1_score is not None:
            if snapshot.f1_score < self.thresholds['f1_score']['critical']:
                self._generate_alert(
                    model_name, 'f1_score', snapshot.f1_score,
                    self.thresholds['f1_score']['critical'], 'critical',
                    f"F1 score {snapshot.f1_score:.3f} below critical threshold"
                )
            elif snapshot.f1_score < self.thresholds['f1_score']['warning']:
                self._generate_alert(
                    model_name, 'f1_score', snapshot.f1_score,
                    self.thresholds['f1_score']['warning'], 'warning',
                    f"F1 score {snapshot.f1_score:.3f} below warning threshold"
                )
        
        # Check MAE (higher is worse)
        if snapshot.mae is not None:
            if snapshot.mae > self.thresholds['mae']['critical']:
                self._generate_alert(
                    model_name, 'mae', snapshot.mae,
                    self.thresholds['mae']['critical'], 'critical',
                    f"MAE {snapshot.mae:.3f} above critical threshold"
                )
            elif snapshot.mae > self.thresholds['mae']['warning']:
                self._generate_alert(
                    model_name, 'mae', snapshot.mae,
                    self.thresholds['mae']['warning'], 'warning',
                    f"MAE {snapshot.mae:.3f} above warning threshold"
                )
        
        # Check RMSE (higher is worse)
        if snapshot.rmse is not None:
            if snapshot.rmse > self.thresholds['rmse']['critical']:
                self._generate_alert(
                    model_name, 'rmse', snapshot.rmse,
                    self.thresholds['rmse']['critical'], 'critical',
                    f"RMSE {snapshot.rmse:.3f} above critical threshold"
                )
            elif snapshot.rmse > self.thresholds['rmse']['warning']:
                self._generate_alert(
                    model_name, 'rmse', snapshot.rmse,
                    self.thresholds['rmse']['warning'], 'warning',
                    f"RMSE {snapshot.rmse:.3f} above warning threshold"
                )
    
    def _generate_alert(self, model_name: str, metric_name: str, current_value: float,
                       threshold_value: float, severity: str, message: str):
        """Generate a performance alert"""
        alert = PerformanceAlert(
            alert_id=f"{model_name}_{metric_name}_{int(time.time())}",
            model_name=model_name,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            severity=severity,
            message=message,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        
        # Log alert
        if severity == 'critical':
            self.logger.critical(f"CRITICAL ALERT: {message}")
        else:
            self.logger.warning(f"WARNING ALERT: {message}")
    
    def get_active_alerts(self, model_name: Optional[str] = None) -> List[PerformanceAlert]:
        """
        Get active (unresolved) alerts
        
        Args:
            model_name: Optional model name to filter alerts
            
        Returns:
            List of active alerts
        """
        with self.lock:
            active_alerts = [alert for alert in self.alerts if not alert.resolved]
            
            if model_name:
                active_alerts = [alert for alert in active_alerts if alert.model_name == model_name]
            
            return active_alerts
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    self.logger.info(f"Alert {alert_id} resolved")
                    break
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for all models"""
        with self.lock:
            summary = {
                'models': {},
                'alerts': {
                    'total': len(self.alerts),
                    'active': len([a for a in self.alerts if not a.resolved]),
                    'critical': len([a for a in self.alerts if a.severity == 'critical' and not a.resolved]),
                    'warning': len([a for a in self.alerts if a.severity == 'warning' and not a.resolved])
                },
                'monitoring_config': {
                    'time_window': self.time_window,
                    'degradation_threshold': self.degradation_threshold,
                    'thresholds': self.thresholds
                }
            }
            
            # Add model-specific summaries
            for model_name in self.prediction_metrics.keys():
                model_metrics = self.get_model_metrics(model_name)
                model_alerts = len([a for a in self.alerts if a.model_name == model_name and not a.resolved])
                needs_retraining = self.trigger_retraining(model_name)
                
                summary['models'][model_name] = {
                    'metrics': model_metrics,
                    'active_alerts': model_alerts,
                    'needs_retraining': needs_retraining,
                    'last_update': datetime.now().isoformat()
                }
            
            return summary
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("Background monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring thread"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Background monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Periodic checks for degradation
                for model_name in list(self.prediction_metrics.keys()):
                    self.trigger_retraining(model_name)
                
                # Clean up old metrics
                self._cleanup_old_metrics()
                
                time.sleep(self.snapshot_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait before retrying
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory bloat"""
        cutoff_time = datetime.now() - timedelta(seconds=self.time_window * 2)  # Keep 2x time window
        
        with self.lock:
            for model_name in list(self.prediction_metrics.keys()):
                # Clean prediction metrics
                original_count = len(self.prediction_metrics[model_name])
                self.prediction_metrics[model_name] = deque([
                    metric for metric in self.prediction_metrics[model_name]
                    if metric.timestamp >= cutoff_time
                ], maxlen=self.max_metrics_per_model)
                
                cleaned_count = original_count - len(self.prediction_metrics[model_name])
                if cleaned_count > 0:
                    self.logger.debug(f"Cleaned {cleaned_count} old metrics for {model_name}")
                
                # Clean performance snapshots (keep last 100)
                if len(self.performance_snapshots[model_name]) > 100:
                    self.performance_snapshots[model_name] = self.performance_snapshots[model_name][-100:]
            
            # Clean old resolved alerts (keep last 1000)
            if len(self.alerts) > 1000:
                # Keep all unresolved alerts and last 500 resolved alerts
                unresolved = [a for a in self.alerts if not a.resolved]
                resolved = [a for a in self.alerts if a.resolved][-500:]
                self.alerts = unresolved + resolved
    
    def export_metrics(self, filepath: str, model_name: Optional[str] = None):
        """Export metrics to JSON file"""
        try:
            with self.lock:
                export_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'models': {}
                }
                
                models_to_export = [model_name] if model_name else list(self.prediction_metrics.keys())
                
                for model in models_to_export:
                    if model in self.prediction_metrics:
                        export_data['models'][model] = {
                            'prediction_metrics': [asdict(m) for m in self.prediction_metrics[model]],
                            'performance_snapshots': [asdict(s) for s in self.performance_snapshots[model]],
                            'alerts': [asdict(a) for a in self.alerts if a.model_name == model]
                        }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.logger.info(f"Metrics exported to {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {str(e)}")
            return False