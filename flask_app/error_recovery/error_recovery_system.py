"""
Error Recovery System for ML Web App Enhancements

This module provides comprehensive error recovery and fallback strategies
for handling model failures, memory pressure, and system degradation.
"""

import logging
import gc
import psutil
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib

from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, get_circuit_breaker

# Configure logging
logger = logging.getLogger(__name__)

class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    FALLBACK_MODEL = "fallback_model"
    CACHED_RESPONSE = "cached_response"
    DEFAULT_RESPONSE = "default_response"
    ENSEMBLE_VOTING = "ensemble_voting"
    LIGHTER_MODEL = "lighter_model"
    MEMORY_CLEANUP = "memory_cleanup"

class SystemHealth(Enum):
    """System health states"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class RecoveryAction:
    """Recovery action record"""
    strategy: RecoveryStrategy
    timestamp: datetime
    success: bool
    execution_time_ms: float
    error_type: str
    recovery_data: Optional[Dict[str, Any]] = None

@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    timestamp: datetime

class ErrorRecoverySystem:
    """
    Comprehensive error recovery system with multiple fallback strategies
    """
    
    def __init__(self, cache_manager=None, fallback_models=None, config=None):
        """
        Initialize the ErrorRecoverySystem
        
        Args:
            cache_manager: Cache manager instance for cached responses
            fallback_models: Dictionary of fallback models
            config: Configuration dictionary
        """
        self.cache_manager = cache_manager
        self.fallback_models = fallback_models or {}
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Recovery history
        self.recovery_actions = []
        self.max_recovery_history = self.config.get('max_recovery_history', 1000)
        
        # System monitoring
        self.system_metrics_history = []
        self.max_metrics_history = self.config.get('max_metrics_history', 100)
        
        # Memory management
        self.memory_warning_threshold = self.config.get('memory_warning_threshold', 80.0)  # 80%
        self.memory_critical_threshold = self.config.get('memory_critical_threshold', 90.0)  # 90%
        self.memory_emergency_threshold = self.config.get('memory_emergency_threshold', 95.0)  # 95%
        
        # Circuit breakers for different components
        self.model_circuit_breakers = {}
        
        # Default responses
        self.default_responses = {
            'classification': 'medium',
            'regression': 5.0,
            'confidence': 0.0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.logger.info("ErrorRecoverySystem initialized")
    
    def handle_model_failure(self, error: Exception, input_data: str, 
                           model_name: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle ML model failure with comprehensive fallback strategy
        
        Args:
            error: The exception that occurred
            input_data: Input data that caused the failure
            model_name: Name of the failed model
            context: Additional context information
            
        Returns:
            Recovery result with prediction or fallback response
        """
        start_time = time.time()
        context = context or {}
        
        self.logger.warning(f"Handling model failure for {model_name}: {str(error)}")
        
        # Generate input hash for caching
        input_hash = self._generate_input_hash(input_data)
        
        # Try recovery strategies in order of preference
        recovery_strategies = [
            (RecoveryStrategy.CACHED_RESPONSE, self._try_cached_response),
            (RecoveryStrategy.FALLBACK_MODEL, self._try_fallback_model),
            (RecoveryStrategy.ENSEMBLE_VOTING, self._try_ensemble_voting),
            (RecoveryStrategy.LIGHTER_MODEL, self._try_lighter_model),
            (RecoveryStrategy.DEFAULT_RESPONSE, self._try_default_response)
        ]
        
        for strategy, recovery_func in recovery_strategies:
            try:
                result = recovery_func(input_data, input_hash, model_name, context)
                if result is not None:
                    execution_time = (time.time() - start_time) * 1000
                    
                    # Record successful recovery
                    self._record_recovery_action(
                        strategy, True, execution_time, type(error).__name__, 
                        {'model_name': model_name, 'result': result}
                    )
                    
                    # Add recovery metadata to result
                    if isinstance(result, dict):
                        result['recovery_used'] = True
                        result['recovery_strategy'] = strategy.value
                        result['original_error'] = str(error)
                    
                    self.logger.info(f"Successfully recovered using {strategy.value}")
                    return result
                    
            except Exception as recovery_error:
                self.logger.warning(f"Recovery strategy {strategy.value} failed: {str(recovery_error)}")
                continue
        
        # All recovery strategies failed
        execution_time = (time.time() - start_time) * 1000
        self._record_recovery_action(
            RecoveryStrategy.DEFAULT_RESPONSE, False, execution_time, type(error).__name__
        )
        
        # Return emergency fallback
        emergency_response = {
            'class': self.default_responses['classification'],
            'score': self.default_responses['regression'],
            'confidence': self.default_responses['confidence'],
            'recovery_used': True,
            'recovery_strategy': 'emergency_fallback',
            'original_error': str(error),
            'error': 'All recovery strategies failed'
        }
        
        self.logger.error("All recovery strategies failed, returning emergency fallback")
        return emergency_response
    
    def _try_cached_response(self, input_data: str, input_hash: str, 
                           model_name: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try to get cached response"""
        if self.cache_manager is None:
            return None
        
        try:
            cached_result = self.cache_manager.get_prediction(input_hash)
            if cached_result:
                self.logger.info("Using cached response for recovery")
                cached_result['cached'] = True
                return cached_result
        except Exception as e:
            self.logger.warning(f"Cache lookup failed: {str(e)}")
        
        return None
    
    def _try_fallback_model(self, input_data: str, input_hash: str, 
                          model_name: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try to use fallback model"""
        if not self.fallback_models:
            return None
        
        # Find a suitable fallback model (different from the failed one)
        for fallback_name, fallback_model in self.fallback_models.items():
            if fallback_name != model_name:
                try:
                    # Get circuit breaker for this fallback model
                    cb = self._get_model_circuit_breaker(fallback_name)
                    
                    # Try prediction with circuit breaker protection
                    result = cb.call(self._make_prediction_with_model, fallback_model, input_data)
                    
                    if result:
                        self.logger.info(f"Using fallback model {fallback_name}")
                        result['fallback_model_used'] = fallback_name
                        return result
                        
                except Exception as e:
                    self.logger.warning(f"Fallback model {fallback_name} failed: {str(e)}")
                    continue
        
        return None
    
    def _try_ensemble_voting(self, input_data: str, input_hash: str, 
                           model_name: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try ensemble voting with available models"""
        if len(self.fallback_models) < 2:
            return None
        
        predictions = []
        working_models = []
        
        for model_name_iter, model in self.fallback_models.items():
            if model_name_iter != model_name:  # Skip the failed model
                try:
                    cb = self._get_model_circuit_breaker(model_name_iter)
                    result = cb.call(self._make_prediction_with_model, model, input_data)
                    
                    if result:
                        predictions.append(result)
                        working_models.append(model_name_iter)
                        
                except Exception as e:
                    self.logger.warning(f"Model {model_name_iter} failed in ensemble: {str(e)}")
                    continue
        
        if len(predictions) >= 2:
            # Perform simple ensemble voting
            ensemble_result = self._ensemble_vote(predictions)
            ensemble_result['ensemble_models_used'] = working_models
            self.logger.info(f"Using ensemble voting with models: {working_models}")
            return ensemble_result
        
        return None
    
    def _try_lighter_model(self, input_data: str, input_hash: str, 
                         model_name: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try to use a lighter/simpler model"""
        # This would require having pre-trained lighter models
        # For now, return None (not implemented)
        return None
    
    def _try_default_response(self, input_data: str, input_hash: str, 
                            model_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Return default response as last resort"""
        return {
            'class': self.default_responses['classification'],
            'score': self.default_responses['regression'],
            'confidence': self.default_responses['confidence'],
            'default_response': True
        }
    
    def _make_prediction_with_model(self, model, input_data: str) -> Dict[str, Any]:
        """Make prediction with a specific model (placeholder implementation)"""
        # This should integrate with the actual model prediction logic
        # For now, return a dummy response
        return {
            'class': 'medium',
            'score': 5.0,
            'confidence': 0.5
        }
    
    def _ensemble_vote(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform ensemble voting on multiple predictions"""
        if not predictions:
            return self._try_default_response("", "", "", {})
        
        # Classification: majority vote
        class_votes = {}
        for pred in predictions:
            class_pred = pred.get('class', 'medium')
            class_votes[class_pred] = class_votes.get(class_pred, 0) + 1
        
        ensemble_class = max(class_votes.items(), key=lambda x: x[1])[0]
        
        # Regression: average
        scores = [pred.get('score', 5.0) for pred in predictions]
        ensemble_score = sum(scores) / len(scores)
        
        # Confidence: average
        confidences = [pred.get('confidence', 0.5) for pred in predictions]
        ensemble_confidence = sum(confidences) / len(confidences)
        
        return {
            'class': ensemble_class,
            'score': ensemble_score,
            'confidence': ensemble_confidence,
            'ensemble_size': len(predictions)
        }
    
    def handle_memory_pressure(self) -> bool:
        """
        Handle memory pressure with various relief strategies
        
        Returns:
            True if memory pressure was relieved, False otherwise
        """
        start_time = time.time()
        
        # Get current memory usage
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        
        self.logger.warning(f"Handling memory pressure: {memory_percent:.1f}% used")
        
        success = False
        strategies_tried = []
        
        try:
            # Strategy 1: Clear prediction cache
            if self.cache_manager and hasattr(self.cache_manager, 'clear_cache'):
                try:
                    self.cache_manager.clear_cache()
                    strategies_tried.append("cache_clear")
                    self.logger.info("Cleared prediction cache")
                except Exception as e:
                    self.logger.warning(f"Failed to clear cache: {str(e)}")
            
            # Strategy 2: Force garbage collection
            try:
                collected = gc.collect()
                strategies_tried.append(f"gc_collect_{collected}")
                self.logger.info(f"Garbage collection freed {collected} objects")
            except Exception as e:
                self.logger.warning(f"Garbage collection failed: {str(e)}")
            
            # Strategy 3: Clear recovery history
            try:
                if len(self.recovery_actions) > 100:
                    self.recovery_actions = self.recovery_actions[-100:]
                    strategies_tried.append("history_trim")
                    self.logger.info("Trimmed recovery history")
            except Exception as e:
                self.logger.warning(f"Failed to trim history: {str(e)}")
            
            # Strategy 4: Clear metrics history
            try:
                if len(self.system_metrics_history) > 50:
                    self.system_metrics_history = self.system_metrics_history[-50:]
                    strategies_tried.append("metrics_trim")
                    self.logger.info("Trimmed metrics history")
            except Exception as e:
                self.logger.warning(f"Failed to trim metrics: {str(e)}")
            
            # Check if memory pressure was relieved
            time.sleep(1)  # Give system time to free memory
            new_memory_info = psutil.virtual_memory()
            new_memory_percent = new_memory_info.percent
            
            memory_freed = memory_percent - new_memory_percent
            success = memory_freed > 1.0  # At least 1% improvement
            
            execution_time = (time.time() - start_time) * 1000
            
            # Record recovery action
            self._record_recovery_action(
                RecoveryStrategy.MEMORY_CLEANUP, success, execution_time, "memory_pressure",
                {
                    'initial_memory_percent': memory_percent,
                    'final_memory_percent': new_memory_percent,
                    'memory_freed_percent': memory_freed,
                    'strategies_tried': strategies_tried
                }
            )
            
            if success:
                self.logger.info(f"Memory pressure relieved: {memory_freed:.1f}% freed")
            else:
                self.logger.warning("Memory pressure relief was not effective")
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._record_recovery_action(
                RecoveryStrategy.MEMORY_CLEANUP, False, execution_time, "memory_pressure_error",
                {'error': str(e), 'strategies_tried': strategies_tried}
            )
            self.logger.error(f"Memory pressure handling failed: {str(e)}")
        
        return success
    
    def get_system_health(self) -> SystemHealth:
        """
        Assess current system health based on various metrics
        
        Returns:
            Current system health state
        """
        try:
            # Get system metrics
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Check memory pressure
            if memory_info.percent >= self.memory_emergency_threshold:
                return SystemHealth.EMERGENCY
            elif memory_info.percent >= self.memory_critical_threshold:
                return SystemHealth.CRITICAL
            elif memory_info.percent >= self.memory_warning_threshold:
                return SystemHealth.DEGRADED
            
            # Check CPU usage
            if cpu_percent >= 95:
                return SystemHealth.CRITICAL
            elif cpu_percent >= 85:
                return SystemHealth.DEGRADED
            
            # Check circuit breaker states
            open_breakers = sum(1 for cb in self.model_circuit_breakers.values() 
                              if cb.get_state().value == 'OPEN')
            
            if open_breakers > 0:
                if open_breakers >= len(self.model_circuit_breakers) * 0.5:
                    return SystemHealth.CRITICAL
                else:
                    return SystemHealth.DEGRADED
            
            return SystemHealth.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Failed to assess system health: {str(e)}")
            return SystemHealth.DEGRADED
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            metrics = SystemMetrics(
                cpu_percent=psutil.cpu_percent(interval=0.1),
                memory_percent=memory_info.percent,
                memory_available_mb=memory_info.available / (1024 * 1024),
                disk_usage_percent=disk_info.percent,
                timestamp=datetime.now()
            )
            
            # Store metrics history
            with self.lock:
                self.system_metrics_history.append(metrics)
                if len(self.system_metrics_history) > self.max_metrics_history:
                    self.system_metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {str(e)}")
            return SystemMetrics(0, 0, 0, 0, datetime.now())
    
    def _get_model_circuit_breaker(self, model_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a model"""
        if model_name not in self.model_circuit_breakers:
            config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                success_threshold=2
            )
            self.model_circuit_breakers[model_name] = get_circuit_breaker(
                f"model_{model_name}", config
            )
        
        return self.model_circuit_breakers[model_name]
    
    def _generate_input_hash(self, input_data: str) -> str:
        """Generate hash for input data"""
        return hashlib.md5(input_data.encode()).hexdigest()
    
    def _record_recovery_action(self, strategy: RecoveryStrategy, success: bool, 
                              execution_time_ms: float, error_type: str, 
                              recovery_data: Optional[Dict[str, Any]] = None):
        """Record a recovery action"""
        with self.lock:
            action = RecoveryAction(
                strategy=strategy,
                timestamp=datetime.now(),
                success=success,
                execution_time_ms=execution_time_ms,
                error_type=error_type,
                recovery_data=recovery_data
            )
            
            self.recovery_actions.append(action)
            
            # Maintain history size
            if len(self.recovery_actions) > self.max_recovery_history:
                self.recovery_actions.pop(0)
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics"""
        with self.lock:
            if not self.recovery_actions:
                return {'total_actions': 0}
            
            # Calculate statistics
            total_actions = len(self.recovery_actions)
            successful_actions = sum(1 for action in self.recovery_actions if action.success)
            success_rate = (successful_actions / total_actions * 100) if total_actions > 0 else 0
            
            # Strategy statistics
            strategy_stats = {}
            for action in self.recovery_actions:
                strategy = action.strategy.value
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'total': 0, 'successful': 0}
                
                strategy_stats[strategy]['total'] += 1
                if action.success:
                    strategy_stats[strategy]['successful'] += 1
            
            # Add success rates
            for strategy, stats in strategy_stats.items():
                stats['success_rate'] = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
            
            # Recent actions (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_actions = [action for action in self.recovery_actions if action.timestamp >= recent_cutoff]
            
            # System health
            current_health = self.get_system_health()
            
            return {
                'total_actions': total_actions,
                'successful_actions': successful_actions,
                'success_rate_percent': round(success_rate, 2),
                'strategy_statistics': strategy_stats,
                'recent_actions_24h': len(recent_actions),
                'current_system_health': current_health.value,
                'circuit_breaker_count': len(self.model_circuit_breakers),
                'last_action_time': self.recovery_actions[-1].timestamp.isoformat() if self.recovery_actions else None
            }
    
    def start_monitoring(self):
        """Start background system monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("Background system monitoring started")
    
    def stop_monitoring(self):
        """Stop background system monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Background system monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self.collect_system_metrics()
                
                # Check for memory pressure
                memory_info = psutil.virtual_memory()
                if memory_info.percent >= self.memory_critical_threshold:
                    self.logger.warning(f"Critical memory usage detected: {memory_info.percent:.1f}%")
                    self.handle_memory_pressure()
                
                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def export_recovery_data(self, filepath: str) -> bool:
        """Export recovery data to file"""
        try:
            with self.lock:
                export_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'recovery_statistics': self.get_recovery_statistics(),
                    'system_metrics_history': [
                        {
                            'cpu_percent': m.cpu_percent,
                            'memory_percent': m.memory_percent,
                            'memory_available_mb': m.memory_available_mb,
                            'disk_usage_percent': m.disk_usage_percent,
                            'timestamp': m.timestamp.isoformat()
                        }
                        for m in self.system_metrics_history
                    ],
                    'recovery_actions': [
                        {
                            'strategy': action.strategy.value,
                            'timestamp': action.timestamp.isoformat(),
                            'success': action.success,
                            'execution_time_ms': action.execution_time_ms,
                            'error_type': action.error_type,
                            'recovery_data': action.recovery_data
                        }
                        for action in self.recovery_actions
                    ]
                }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                self.logger.info(f"Recovery data exported to {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to export recovery data: {str(e)}")
            return False