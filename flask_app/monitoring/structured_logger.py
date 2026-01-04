"""
Structured Logger for ML Web App Enhancements

This module provides structured logging capabilities with correlation IDs
for distributed tracing and comprehensive event tracking.
"""

import logging
import json
import uuid
import time
import threading
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import traceback
import sys
import os
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class EventType(Enum):
    """Types of events to log"""
    REQUEST = "request"
    RESPONSE = "response"
    PREDICTION = "prediction"
    ERROR = "error"
    PERFORMANCE = "performance"
    SECURITY = "security"
    SYSTEM = "system"
    MODEL = "model"
    CACHE = "cache"
    DATABASE = "database"

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: str
    level: str
    event_type: str
    correlation_id: str
    message: str
    component: str
    metadata: Dict[str, Any]
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class CorrelationIDGenerator:
    """Generator for correlation IDs"""
    
    def __init__(self, prefix: str = "req"):
        """Initialize correlation ID generator"""
        self.prefix = prefix
        self.counter = 0
        self.lock = threading.Lock()
    
    def generate(self) -> str:
        """Generate a new correlation ID"""
        with self.lock:
            self.counter += 1
            timestamp = int(time.time() * 1000)  # milliseconds
            return f"{self.prefix}-{timestamp}-{self.counter:06d}"
    
    def generate_uuid(self) -> str:
        """Generate UUID-based correlation ID"""
        return f"{self.prefix}-{str(uuid.uuid4())}"

class StructuredLogger:
    """
    Structured logger with correlation ID support and distributed tracing
    """
    
    def __init__(self, component_name: str, config: Optional[Dict] = None):
        """
        Initialize structured logger
        
        Args:
            component_name: Name of the component using this logger
            config: Configuration dictionary
        """
        self.component_name = component_name
        self.config = config or {}
        
        # Correlation ID generator
        self.correlation_id_generator = CorrelationIDGenerator(
            prefix=self.config.get('correlation_id_prefix', 'req')
        )
        
        # Thread-local storage for correlation context
        self.local = threading.local()
        
        # Log storage (for debugging and analysis)
        self.log_entries = []
        self.max_log_entries = self.config.get('max_log_entries', 10000)
        
        # Performance tracking
        self.performance_metrics = {}
        
        # Configure Python logger
        self.python_logger = logging.getLogger(f"structured.{component_name}")
        self._configure_python_logger()
        
        # Lock for thread safety
        self.lock = threading.RLock()
        
        self.python_logger.info(f"StructuredLogger initialized for component: {component_name}")
    
    def _configure_python_logger(self):
        """Configure the underlying Python logger"""
        # Set log level
        log_level = self.config.get('log_level', 'INFO')
        self.python_logger.setLevel(getattr(logging, log_level))
        
        # Create formatter for structured output
        if self.config.get('structured_output', True):
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Add console handler if not exists
        if not self.python_logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.python_logger.addHandler(console_handler)
        
        # Add file handler if configured
        log_file = self.config.get('log_file')
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.python_logger.addHandler(file_handler)
            except Exception as e:
                self.python_logger.error(f"Failed to create file handler: {str(e)}")
    
    def set_correlation_context(self, correlation_id: str, trace_id: Optional[str] = None,
                              span_id: Optional[str] = None, user_id: Optional[str] = None,
                              session_id: Optional[str] = None):
        """Set correlation context for current thread"""
        self.local.correlation_id = correlation_id
        self.local.trace_id = trace_id
        self.local.span_id = span_id
        self.local.user_id = user_id
        self.local.session_id = session_id
    
    def get_correlation_context(self) -> Dict[str, Optional[str]]:
        """Get current correlation context"""
        return {
            'correlation_id': getattr(self.local, 'correlation_id', None),
            'trace_id': getattr(self.local, 'trace_id', None),
            'span_id': getattr(self.local, 'span_id', None),
            'user_id': getattr(self.local, 'user_id', None),
            'session_id': getattr(self.local, 'session_id', None)
        }
    
    def generate_correlation_id(self) -> str:
        """Generate and set new correlation ID for current context"""
        correlation_id = self.correlation_id_generator.generate()
        self.set_correlation_context(correlation_id)
        return correlation_id
    
    @contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None, **context_kwargs):
        """Context manager for correlation ID scope"""
        if correlation_id is None:
            correlation_id = self.correlation_id_generator.generate()
        
        # Save current context
        old_context = self.get_correlation_context()
        
        # Set new context
        self.set_correlation_context(correlation_id, **context_kwargs)
        
        try:
            yield correlation_id
        finally:
            # Restore old context
            self.set_correlation_context(**old_context)
    
    def _create_log_entry(self, level: LogLevel, event_type: EventType, 
                         message: str, metadata: Optional[Dict[str, Any]] = None) -> LogEntry:
        """Create a structured log entry"""
        context = self.get_correlation_context()
        
        return LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            event_type=event_type.value,
            correlation_id=context['correlation_id'] or 'no-correlation-id',
            message=message,
            component=self.component_name,
            metadata=metadata or {},
            trace_id=context['trace_id'],
            span_id=context['span_id'],
            user_id=context['user_id'],
            session_id=context['session_id']
        )
    
    def _log_entry(self, log_entry: LogEntry):
        """Process and store log entry"""
        with self.lock:
            # Store in memory (for analysis)
            self.log_entries.append(log_entry)
            
            # Maintain max entries
            if len(self.log_entries) > self.max_log_entries:
                self.log_entries.pop(0)
            
            # Log to Python logger
            log_message = self._format_log_message(log_entry)
            python_level = getattr(logging, log_entry.level)
            self.python_logger.log(python_level, log_message)
    
    def _format_log_message(self, log_entry: LogEntry) -> str:
        """Format log entry for output"""
        if self.config.get('json_output', False):
            return json.dumps(asdict(log_entry), default=str)
        else:
            return (f"[{log_entry.correlation_id}] {log_entry.event_type.upper()}: "
                   f"{log_entry.message} | {json.dumps(log_entry.metadata, default=str)}")
    
    def log_request(self, request_data: Dict[str, Any], correlation_id: Optional[str] = None):
        """Log incoming request with structured format"""
        if correlation_id:
            self.set_correlation_context(correlation_id)
        elif not getattr(self.local, 'correlation_id', None):
            self.generate_correlation_id()
        
        metadata = {
            'request_method': request_data.get('method', 'unknown'),
            'request_path': request_data.get('path', 'unknown'),
            'request_size': len(str(request_data)),
            'client_ip': request_data.get('client_ip'),
            'user_agent': request_data.get('user_agent'),
            'request_headers': request_data.get('headers', {})
        }
        
        log_entry = self._create_log_entry(
            LogLevel.INFO, EventType.REQUEST,
            f"Incoming request: {metadata['request_method']} {metadata['request_path']}",
            metadata
        )
        
        self._log_entry(log_entry)
    
    def log_response(self, response_data: Dict[str, Any], duration_ms: float):
        """Log outgoing response"""
        metadata = {
            'response_status': response_data.get('status_code', 'unknown'),
            'response_size': len(str(response_data)),
            'duration_ms': duration_ms,
            'response_headers': response_data.get('headers', {})
        }
        
        log_entry = self._create_log_entry(
            LogLevel.INFO, EventType.RESPONSE,
            f"Response sent: {metadata['response_status']} in {duration_ms:.2f}ms",
            metadata
        )
        
        self._log_entry(log_entry)
    
    def log_prediction(self, model_name: str, input_hash: str, result: Dict[str, Any], 
                      duration: float, correlation_id: Optional[str] = None):
        """Log prediction event with performance metrics"""
        if correlation_id:
            self.set_correlation_context(correlation_id)
        
        metadata = {
            'model_name': model_name,
            'input_hash': input_hash,
            'prediction_class': result.get('class'),
            'prediction_score': result.get('score'),
            'confidence': result.get('confidence'),
            'duration_ms': duration,
            'cached': result.get('cached', False),
            'ensemble_used': result.get('ensemble_used', False),
            'fallback_used': result.get('fallback_used', False)
        }
        
        # Track performance metrics
        self._track_performance_metric('prediction_duration', duration)
        
        log_entry = self._create_log_entry(
            LogLevel.INFO, EventType.PREDICTION,
            f"Prediction made by {model_name} in {duration:.2f}ms",
            metadata
        )
        
        self._log_entry(log_entry)
    
    def log_error(self, error: Exception, context: Dict[str, Any], 
                 correlation_id: Optional[str] = None):
        """Log error with full context and stack trace"""
        if correlation_id:
            self.set_correlation_context(correlation_id)
        
        # Get stack trace
        exc_type, exc_value, exc_traceback = sys.exc_info()
        stack_trace = traceback.format_exception(exc_type, exc_value, exc_traceback)
        
        metadata = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'stack_trace': ''.join(stack_trace),
            'context': context,
            'file_name': exc_traceback.tb_frame.f_code.co_filename if exc_traceback else None,
            'line_number': exc_traceback.tb_lineno if exc_traceback else None,
            'function_name': exc_traceback.tb_frame.f_code.co_name if exc_traceback else None
        }
        
        log_entry = self._create_log_entry(
            LogLevel.ERROR, EventType.ERROR,
            f"Error occurred: {type(error).__name__}: {str(error)}",
            metadata
        )
        
        self._log_entry(log_entry)
    
    def log_performance(self, operation: str, duration_ms: float, 
                       metadata: Optional[Dict[str, Any]] = None):
        """Log performance metrics"""
        perf_metadata = {
            'operation': operation,
            'duration_ms': duration_ms,
            **(metadata or {})
        }
        
        # Track performance metrics
        self._track_performance_metric(operation, duration_ms)
        
        log_entry = self._create_log_entry(
            LogLevel.INFO, EventType.PERFORMANCE,
            f"Performance: {operation} completed in {duration_ms:.2f}ms",
            perf_metadata
        )
        
        self._log_entry(log_entry)
    
    def log_security_event(self, event_description: str, severity: str, 
                          metadata: Optional[Dict[str, Any]] = None):
        """Log security-related events"""
        security_metadata = {
            'severity': severity,
            'event_description': event_description,
            **(metadata or {})
        }
        
        level = LogLevel.CRITICAL if severity == 'critical' else LogLevel.WARNING
        
        log_entry = self._create_log_entry(
            level, EventType.SECURITY,
            f"Security event: {event_description}",
            security_metadata
        )
        
        self._log_entry(log_entry)
    
    def log_system_event(self, event_description: str, 
                        metadata: Optional[Dict[str, Any]] = None):
        """Log system-level events"""
        log_entry = self._create_log_entry(
            LogLevel.INFO, EventType.SYSTEM,
            f"System event: {event_description}",
            metadata or {}
        )
        
        self._log_entry(log_entry)
    
    def log_model_event(self, model_name: str, event_description: str, 
                       metadata: Optional[Dict[str, Any]] = None):
        """Log model-related events"""
        model_metadata = {
            'model_name': model_name,
            **(metadata or {})
        }
        
        log_entry = self._create_log_entry(
            LogLevel.INFO, EventType.MODEL,
            f"Model event [{model_name}]: {event_description}",
            model_metadata
        )
        
        self._log_entry(log_entry)
    
    def _track_performance_metric(self, operation: str, duration_ms: float):
        """Track performance metrics for analysis"""
        with self.lock:
            if operation not in self.performance_metrics:
                self.performance_metrics[operation] = {
                    'count': 0,
                    'total_duration': 0,
                    'min_duration': float('inf'),
                    'max_duration': 0,
                    'recent_durations': []
                }
            
            metrics = self.performance_metrics[operation]
            metrics['count'] += 1
            metrics['total_duration'] += duration_ms
            metrics['min_duration'] = min(metrics['min_duration'], duration_ms)
            metrics['max_duration'] = max(metrics['max_duration'], duration_ms)
            
            # Keep recent durations for trend analysis
            metrics['recent_durations'].append(duration_ms)
            if len(metrics['recent_durations']) > 100:
                metrics['recent_durations'].pop(0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        with self.lock:
            summary = {}
            
            for operation, metrics in self.performance_metrics.items():
                avg_duration = metrics['total_duration'] / metrics['count'] if metrics['count'] > 0 else 0
                
                # Calculate recent average (last 50 operations)
                recent = metrics['recent_durations'][-50:]
                recent_avg = sum(recent) / len(recent) if recent else 0
                
                summary[operation] = {
                    'count': metrics['count'],
                    'avg_duration_ms': round(avg_duration, 2),
                    'min_duration_ms': round(metrics['min_duration'], 2) if metrics['min_duration'] != float('inf') else 0,
                    'max_duration_ms': round(metrics['max_duration'], 2),
                    'recent_avg_duration_ms': round(recent_avg, 2),
                    'total_duration_ms': round(metrics['total_duration'], 2)
                }
            
            return summary
    
    def search_logs(self, correlation_id: Optional[str] = None, 
                   event_type: Optional[str] = None,
                   level: Optional[str] = None,
                   time_range_minutes: Optional[int] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Search log entries with filters"""
        with self.lock:
            filtered_logs = []
            
            # Time filter
            cutoff_time = None
            if time_range_minutes:
                cutoff_time = datetime.now() - timedelta(minutes=time_range_minutes)
            
            for log_entry in reversed(self.log_entries):  # Most recent first
                # Apply filters
                if correlation_id and log_entry.correlation_id != correlation_id:
                    continue
                
                if event_type and log_entry.event_type != event_type:
                    continue
                
                if level and log_entry.level != level:
                    continue
                
                if cutoff_time:
                    log_time = datetime.fromisoformat(log_entry.timestamp)
                    if log_time < cutoff_time:
                        continue
                
                filtered_logs.append(asdict(log_entry))
                
                if len(filtered_logs) >= limit:
                    break
            
            return filtered_logs
    
    def get_correlation_trace(self, correlation_id: str) -> List[Dict[str, Any]]:
        """Get all log entries for a specific correlation ID"""
        return self.search_logs(correlation_id=correlation_id, limit=1000)
    
    def export_logs(self, filepath: str, format_type: str = 'json') -> bool:
        """Export logs to file"""
        try:
            with self.lock:
                if format_type == 'json':
                    export_data = {
                        'export_timestamp': datetime.now().isoformat(),
                        'component': self.component_name,
                        'log_entries': [asdict(entry) for entry in self.log_entries],
                        'performance_summary': self.get_performance_summary()
                    }
                    
                    with open(filepath, 'w') as f:
                        json.dump(export_data, f, indent=2, default=str)
                
                elif format_type == 'csv':
                    import csv
                    
                    with open(filepath, 'w', newline='') as f:
                        if self.log_entries:
                            fieldnames = list(asdict(self.log_entries[0]).keys())
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            
                            for entry in self.log_entries:
                                row = asdict(entry)
                                # Convert complex fields to JSON strings
                                row['metadata'] = json.dumps(row['metadata'], default=str)
                                writer.writerow(row)
                
                else:
                    raise ValueError(f"Unsupported format: {format_type}")
                
                self.python_logger.info(f"Logs exported to {filepath} in {format_type} format")
                return True
                
        except Exception as e:
            self.python_logger.error(f"Failed to export logs: {str(e)}")
            return False
    
    def clear_logs(self):
        """Clear stored log entries"""
        with self.lock:
            self.log_entries.clear()
            self.performance_metrics.clear()
            self.python_logger.info("Log entries and performance metrics cleared")

# Global structured logger instance
_global_logger = None

def get_structured_logger(component_name: str, config: Optional[Dict] = None) -> StructuredLogger:
    """Get or create a structured logger instance"""
    global _global_logger
    
    if _global_logger is None or _global_logger.component_name != component_name:
        _global_logger = StructuredLogger(component_name, config)
    
    return _global_logger

# Convenience functions for common logging operations
def log_request(request_data: Dict[str, Any], correlation_id: Optional[str] = None, 
               component: str = "app"):
    """Convenience function for request logging"""
    logger = get_structured_logger(component)
    logger.log_request(request_data, correlation_id)

def log_error(error: Exception, context: Dict[str, Any], 
             correlation_id: Optional[str] = None, component: str = "app"):
    """Convenience function for error logging"""
    logger = get_structured_logger(component)
    logger.log_error(error, context, correlation_id)

def log_prediction(model_name: str, input_hash: str, result: Dict[str, Any], 
                  duration: float, correlation_id: Optional[str] = None, 
                  component: str = "ml"):
    """Convenience function for prediction logging"""
    logger = get_structured_logger(component)
    logger.log_prediction(model_name, input_hash, result, duration, correlation_id)