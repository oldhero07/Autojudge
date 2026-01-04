"""
Circuit Breaker for ML Web App Enhancements

This module provides circuit breaker pattern implementation for protecting
against cascading failures and providing graceful degradation when
ML models or external services fail.
"""

import logging
import time
import threading
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import functools

# Configure logging
logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Circuit is open, calls are failing fast
    HALF_OPEN = "HALF_OPEN"  # Testing if service has recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5          # Number of failures before opening
    recovery_timeout: int = 60          # Seconds before trying half-open
    success_threshold: int = 3          # Successes needed to close from half-open
    timeout: float = 30.0               # Call timeout in seconds
    expected_exception: type = Exception # Exception type to catch

@dataclass
class CallResult:
    """Result of a circuit breaker call"""
    success: bool
    result: Any = None
    exception: Optional[Exception] = None
    execution_time: float = 0.0
    timestamp: datetime = None

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    def __init__(self, circuit_name: str, failure_count: int):
        self.circuit_name = circuit_name
        self.failure_count = failure_count
        super().__init__(f"Circuit breaker '{circuit_name}' is OPEN after {failure_count} failures")

class CircuitBreaker:
    """
    Circuit breaker implementation with automatic recovery and health checking
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker
        
        Args:
            name: Name of the circuit breaker for identification
            config: Configuration object (uses defaults if None)
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        
        # Call history
        self.call_history = deque(maxlen=1000)  # Keep last 1000 calls
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_timeouts = 0
        self.total_circuit_open_calls = 0
        
        self.logger.info(f"Circuit breaker '{name}' initialized with config: {self.config}")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenException: When circuit is open
            Original exception: When function fails and circuit allows it
        """
        with self.lock:
            self.total_calls += 1
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    self.total_circuit_open_calls += 1
                    self.logger.warning(f"Circuit breaker '{self.name}' is OPEN, failing fast")
                    raise CircuitBreakerOpenException(self.name, self.failure_count)
            
            # Execute the function
            start_time = time.time()
            call_result = CallResult(success=False, timestamp=datetime.now())
            
            try:
                # Execute with timeout
                result = self._execute_with_timeout(func, *args, **kwargs)
                
                execution_time = time.time() - start_time
                call_result.success = True
                call_result.result = result
                call_result.execution_time = execution_time
                
                self._record_success(call_result)
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                call_result.exception = e
                call_result.execution_time = execution_time
                
                self._record_failure(call_result)
                raise
    
    def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout protection"""
        # For now, execute directly without timeout
        # In production, you might want to use threading.Timer or asyncio
        return func(*args, **kwargs)
    
    def _record_success(self, call_result: CallResult):
        """Record a successful call"""
        with self.lock:
            self.success_count += 1
            self.total_successes += 1
            self.last_success_time = datetime.now()
            self.call_history.append(call_result)
            
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            
            self.logger.debug(f"Success recorded for '{self.name}' (count: {self.success_count})")
    
    def _record_failure(self, call_result: CallResult):
        """Record a failed call"""
        with self.lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = datetime.now()
            self.call_history.append(call_result)
            
            # Reset success count on any failure
            self.success_count = 0
            
            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                # Any failure in half-open state goes back to open
                self._transition_to_open()
            
            self.logger.warning(f"Failure recorded for '{self.name}' (count: {self.failure_count})")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _transition_to_open(self):
        """Transition circuit breaker to OPEN state"""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.logger.warning(f"Circuit breaker '{self.name}' transitioned from {old_state.value} to OPEN")
    
    def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state"""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0  # Reset success count for half-open testing
        self.logger.info(f"Circuit breaker '{self.name}' transitioned from {old_state.value} to HALF_OPEN")
    
    def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state"""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.failure_count = 0  # Reset failure count
        self.success_count = 0  # Reset success count
        self.logger.info(f"Circuit breaker '{self.name}' transitioned from {old_state.value} to CLOSED")
    
    def reset(self):
        """Manually reset the circuit breaker to CLOSED state"""
        with self.lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.logger.info(f"Circuit breaker '{self.name}' manually reset from {old_state.value} to CLOSED")
    
    def force_open(self):
        """Manually force the circuit breaker to OPEN state"""
        with self.lock:
            old_state = self.state
            self.state = CircuitState.OPEN
            self.logger.warning(f"Circuit breaker '{self.name}' manually forced from {old_state.value} to OPEN")
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state"""
        return self.state
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker statistics"""
        with self.lock:
            # Calculate success/failure rates
            success_rate = (self.total_successes / self.total_calls * 100) if self.total_calls > 0 else 0
            failure_rate = (self.total_failures / self.total_calls * 100) if self.total_calls > 0 else 0
            
            # Recent call statistics (last 100 calls)
            recent_calls = list(self.call_history)[-100:]
            recent_successes = sum(1 for call in recent_calls if call.success)
            recent_failures = len(recent_calls) - recent_successes
            recent_success_rate = (recent_successes / len(recent_calls) * 100) if recent_calls else 0
            
            # Average execution time
            execution_times = [call.execution_time for call in recent_calls if call.execution_time > 0]
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
            
            return {
                'name': self.name,
                'state': self.state.value,
                'current_failure_count': self.failure_count,
                'current_success_count': self.success_count,
                'total_calls': self.total_calls,
                'total_successes': self.total_successes,
                'total_failures': self.total_failures,
                'total_circuit_open_calls': self.total_circuit_open_calls,
                'success_rate_percent': round(success_rate, 2),
                'failure_rate_percent': round(failure_rate, 2),
                'recent_success_rate_percent': round(recent_success_rate, 2),
                'avg_execution_time_ms': round(avg_execution_time * 1000, 2),
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'last_success_time': self.last_success_time.isoformat() if self.last_success_time else None,
                'config': {
                    'failure_threshold': self.config.failure_threshold,
                    'recovery_timeout': self.config.recovery_timeout,
                    'success_threshold': self.config.success_threshold,
                    'timeout': self.config.timeout
                }
            }
    
    def get_recent_calls(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent call history"""
        with self.lock:
            recent_calls = list(self.call_history)[-limit:]
            return [
                {
                    'success': call.success,
                    'execution_time_ms': round(call.execution_time * 1000, 2),
                    'timestamp': call.timestamp.isoformat(),
                    'exception_type': type(call.exception).__name__ if call.exception else None,
                    'exception_message': str(call.exception) if call.exception else None
                }
                for call in recent_calls
            ]

class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers
    """
    
    def __init__(self):
        """Initialize circuit breaker manager"""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """
        Get or create a circuit breaker
        
        Args:
            name: Name of the circuit breaker
            config: Configuration (uses default if None)
            
        Returns:
            CircuitBreaker instance
        """
        with self.lock:
            if name not in self.circuit_breakers:
                self.circuit_breakers[name] = CircuitBreaker(name, config)
                self.logger.info(f"Created new circuit breaker: {name}")
            
            return self.circuit_breakers[name]
    
    def remove_circuit_breaker(self, name: str):
        """Remove a circuit breaker"""
        with self.lock:
            if name in self.circuit_breakers:
                del self.circuit_breakers[name]
                self.logger.info(f"Removed circuit breaker: {name}")
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        with self.lock:
            return {name: cb.get_statistics() for name, cb in self.circuit_breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers"""
        with self.lock:
            for cb in self.circuit_breakers.values():
                cb.reset()
            self.logger.info("Reset all circuit breakers")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary of all circuit breakers"""
        with self.lock:
            total_breakers = len(self.circuit_breakers)
            open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb.get_state() == CircuitState.OPEN)
            half_open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb.get_state() == CircuitState.HALF_OPEN)
            closed_breakers = total_breakers - open_breakers - half_open_breakers
            
            return {
                'total_circuit_breakers': total_breakers,
                'closed': closed_breakers,
                'half_open': half_open_breakers,
                'open': open_breakers,
                'health_status': 'healthy' if open_breakers == 0 else ('degraded' if open_breakers < total_breakers else 'critical'),
                'circuit_breakers': {name: cb.get_state().value for name, cb in self.circuit_breakers.items()}
            }

# Decorator for easy circuit breaker usage
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator to add circuit breaker protection to a function
    
    Args:
        name: Name of the circuit breaker
        config: Optional configuration
    """
    def decorator(func: Callable) -> Callable:
        # Get or create circuit breaker
        cb_manager = CircuitBreakerManager()
        cb = cb_manager.get_circuit_breaker(name, config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return cb.call(func, *args, **kwargs)
        
        # Add circuit breaker reference to function
        wrapper._circuit_breaker = cb
        return wrapper
    
    return decorator

# Global circuit breaker manager instance
_global_cb_manager = CircuitBreakerManager()

def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get a circuit breaker from the global manager"""
    return _global_cb_manager.get_circuit_breaker(name, config)

def get_all_circuit_breaker_statistics() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all circuit breakers"""
    return _global_cb_manager.get_all_statistics()

def get_circuit_breaker_health_summary() -> Dict[str, Any]:
    """Get health summary of all circuit breakers"""
    return _global_cb_manager.get_health_summary()

def reset_all_circuit_breakers():
    """Reset all circuit breakers"""
    _global_cb_manager.reset_all()