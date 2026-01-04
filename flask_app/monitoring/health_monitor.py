"""
Health Monitor for ML Web App Enhancements

This module provides comprehensive system health monitoring including
component health checks, metrics collection, and alerting capabilities.
"""

import logging
import psutil
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import socket
import requests
from collections import defaultdict, deque

# Configure logging
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

class ComponentType(Enum):
    """Types of system components"""
    ML_MODEL = "ml_model"
    DATABASE = "database"
    CACHE = "cache"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    MEMORY = "memory"
    CPU = "cpu"
    EXTERNAL_SERVICE = "external_service"

@dataclass
class ComponentHealth:
    """Health information for a system component"""
    name: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    last_check: datetime
    response_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SystemMetrics:
    """System-wide metrics"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    network_connections: int
    load_average: Optional[List[float]] = None
    uptime_seconds: Optional[float] = None

@dataclass
class HealthAlert:
    """Health monitoring alert"""
    alert_id: str
    component_name: str
    severity: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class HealthCheck:
    """Base class for health checks"""
    
    def __init__(self, name: str, component_type: ComponentType, 
                 check_interval: int = 60, timeout: float = 30.0):
        """
        Initialize health check
        
        Args:
            name: Name of the component
            component_type: Type of component
            check_interval: Check interval in seconds
            timeout: Timeout for health check in seconds
        """
        self.name = name
        self.component_type = component_type
        self.check_interval = check_interval
        self.timeout = timeout
        self.last_check = None
        self.last_status = HealthStatus.HEALTHY
        
    def check_health(self) -> ComponentHealth:
        """
        Perform health check (to be implemented by subclasses)
        
        Returns:
            ComponentHealth object
        """
        raise NotImplementedError("Subclasses must implement check_health method")

class MLModelHealthCheck(HealthCheck):
    """Health check for ML models"""
    
    def __init__(self, name: str, model_instance, test_input: str = "test"):
        """Initialize ML model health check"""
        super().__init__(name, ComponentType.ML_MODEL)
        self.model_instance = model_instance
        self.test_input = test_input
    
    def check_health(self) -> ComponentHealth:
        """Check ML model health by making a test prediction"""
        start_time = time.time()
        
        try:
            # Try to make a test prediction
            if hasattr(self.model_instance, 'predict'):
                # Assume it's a sklearn-like model
                test_data = [[1.0] * 10]  # Dummy test data
                prediction = self.model_instance.predict(test_data)
                
                response_time = (time.time() - start_time) * 1000
                
                return ComponentHealth(
                    name=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.HEALTHY,
                    message="Model prediction successful",
                    last_check=datetime.now(),
                    response_time_ms=response_time,
                    metadata={'prediction_shape': str(prediction.shape) if hasattr(prediction, 'shape') else None}
                )
            else:
                return ComponentHealth(
                    name=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.UNHEALTHY,
                    message="Model does not have predict method",
                    last_check=datetime.now()
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return ComponentHealth(
                name=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Model prediction failed: {str(e)}",
                last_check=datetime.now(),
                response_time_ms=response_time
            )

class DatabaseHealthCheck(HealthCheck):
    """Health check for database connections"""
    
    def __init__(self, name: str, connection_func: Callable):
        """Initialize database health check"""
        super().__init__(name, ComponentType.DATABASE)
        self.connection_func = connection_func
    
    def check_health(self) -> ComponentHealth:
        """Check database health by testing connection"""
        start_time = time.time()
        
        try:
            # Test database connection
            result = self.connection_func()
            response_time = (time.time() - start_time) * 1000
            
            if result:
                return ComponentHealth(
                    name=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful",
                    last_check=datetime.now(),
                    response_time_ms=response_time
                )
            else:
                return ComponentHealth(
                    name=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.UNHEALTHY,
                    message="Database connection returned False",
                    last_check=datetime.now(),
                    response_time_ms=response_time
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return ComponentHealth(
                name=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}",
                last_check=datetime.now(),
                response_time_ms=response_time
            )

class NetworkHealthCheck(HealthCheck):
    """Health check for network connectivity"""
    
    def __init__(self, name: str, target_url: str = "https://www.google.com"):
        """Initialize network health check"""
        super().__init__(name, ComponentType.NETWORK)
        self.target_url = target_url
    
    def check_health(self) -> ComponentHealth:
        """Check network health by making HTTP request"""
        start_time = time.time()
        
        try:
            response = requests.get(self.target_url, timeout=self.timeout)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                return ComponentHealth(
                    name=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.HEALTHY,
                    message=f"Network connectivity confirmed to {self.target_url}",
                    last_check=datetime.now(),
                    response_time_ms=response_time,
                    metadata={'status_code': response.status_code}
                )
            else:
                return ComponentHealth(
                    name=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.DEGRADED,
                    message=f"Network request returned status {response.status_code}",
                    last_check=datetime.now(),
                    response_time_ms=response_time,
                    metadata={'status_code': response.status_code}
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return ComponentHealth(
                name=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNHEALTHY,
                message=f"Network check failed: {str(e)}",
                last_check=datetime.now(),
                response_time_ms=response_time
            )

class HealthMonitor:
    """
    Comprehensive system health monitoring with metrics collection and alerting
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize health monitor"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Health checks registry
        self.health_checks: Dict[str, HealthCheck] = {}
        
        # Metrics storage
        self.metrics_history = deque(maxlen=self.config.get('max_metrics_history', 1000))
        self.component_health_history = defaultdict(lambda: deque(maxlen=100))
        
        # Alerting
        self.alerts = []
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'cpu_percent': {'warning': 80, 'critical': 95},
            'memory_percent': {'warning': 80, 'critical': 95},
            'disk_usage_percent': {'warning': 85, 'critical': 95},
            'response_time_ms': {'warning': 1000, 'critical': 5000}
        })
        
        # Monitoring configuration
        self.monitoring_interval = self.config.get('monitoring_interval', 30)  # seconds
        self.health_check_interval = self.config.get('health_check_interval', 60)  # seconds
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize default health checks
        self._initialize_default_health_checks()
        
        self.logger.info("HealthMonitor initialized")
    
    def _initialize_default_health_checks(self):
        """Initialize default system health checks"""
        # Network connectivity check
        self.register_health_check(NetworkHealthCheck("internet_connectivity"))
        
        # Add more default checks as needed
        self.logger.info("Default health checks initialized")
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a new health check"""
        with self.lock:
            self.health_checks[health_check.name] = health_check
            self.logger.info(f"Registered health check: {health_check.name}")
    
    def unregister_health_check(self, name: str):
        """Unregister a health check"""
        with self.lock:
            if name in self.health_checks:
                del self.health_checks[name]
                self.logger.info(f"Unregistered health check: {name}")
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive system health check
        
        Returns:
            Dictionary with system health status and component details
        """
        with self.lock:
            health_results = {}
            overall_status = HealthStatus.HEALTHY
            
            # Check all registered components
            for name, health_check in self.health_checks.items():
                try:
                    component_health = health_check.check_health()
                    health_results[name] = asdict(component_health)
                    
                    # Store in history
                    self.component_health_history[name].append(component_health)
                    
                    # Update overall status
                    if component_health.status == HealthStatus.CRITICAL:
                        overall_status = HealthStatus.CRITICAL
                    elif component_health.status == HealthStatus.UNHEALTHY and overall_status != HealthStatus.CRITICAL:
                        overall_status = HealthStatus.UNHEALTHY
                    elif component_health.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED
                    
                    # Check for alerts
                    self._check_component_alerts(component_health)
                    
                except Exception as e:
                    self.logger.error(f"Health check failed for {name}: {str(e)}")
                    health_results[name] = {
                        'name': name,
                        'status': HealthStatus.UNHEALTHY.value,
                        'message': f"Health check error: {str(e)}",
                        'last_check': datetime.now().isoformat()
                    }
                    overall_status = HealthStatus.UNHEALTHY
            
            # Collect system metrics
            system_metrics = self.collect_metrics()
            
            # Check system-level alerts
            self._check_system_alerts(system_metrics)
            
            return {
                'overall_status': overall_status.value,
                'timestamp': datetime.now().isoformat(),
                'components': health_results,
                'system_metrics': asdict(system_metrics),
                'active_alerts': len([a for a in self.alerts if not a.resolved]),
                'total_components': len(self.health_checks)
            }
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent
            memory_available_mb = memory_info.available / (1024 * 1024)
            
            # Disk metrics
            disk_info = psutil.disk_usage('/')
            disk_usage_percent = disk_info.percent
            
            # Network metrics
            network_connections = len(psutil.net_connections())
            
            # Load average (Unix-like systems)
            load_average = None
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                # Windows doesn't have load average
                pass
            
            # System uptime
            uptime_seconds = time.time() - psutil.boot_time()
            
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                network_connections=network_connections,
                load_average=load_average,
                uptime_seconds=uptime_seconds
            )
            
            # Store in history
            with self.lock:
                self.metrics_history.append(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {str(e)}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0,
                memory_percent=0,
                memory_available_mb=0,
                disk_usage_percent=0,
                network_connections=0
            )
    
    def _check_component_alerts(self, component_health: ComponentHealth):
        """Check for component-specific alerts"""
        # Response time alerts
        if component_health.response_time_ms:
            thresholds = self.alert_thresholds.get('response_time_ms', {})
            
            if component_health.response_time_ms > thresholds.get('critical', float('inf')):
                self._generate_alert(
                    component_health.name, 'critical',
                    f"Response time {component_health.response_time_ms:.1f}ms exceeds critical threshold"
                )
            elif component_health.response_time_ms > thresholds.get('warning', float('inf')):
                self._generate_alert(
                    component_health.name, 'warning',
                    f"Response time {component_health.response_time_ms:.1f}ms exceeds warning threshold"
                )
        
        # Status-based alerts
        if component_health.status == HealthStatus.CRITICAL:
            self._generate_alert(
                component_health.name, 'critical',
                f"Component is in critical state: {component_health.message}"
            )
        elif component_health.status == HealthStatus.UNHEALTHY:
            self._generate_alert(
                component_health.name, 'warning',
                f"Component is unhealthy: {component_health.message}"
            )
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check for system-level alerts"""
        # CPU alerts
        cpu_thresholds = self.alert_thresholds.get('cpu_percent', {})
        if metrics.cpu_percent > cpu_thresholds.get('critical', 100):
            self._generate_alert(
                'system_cpu', 'critical',
                f"CPU usage {metrics.cpu_percent:.1f}% exceeds critical threshold"
            )
        elif metrics.cpu_percent > cpu_thresholds.get('warning', 100):
            self._generate_alert(
                'system_cpu', 'warning',
                f"CPU usage {metrics.cpu_percent:.1f}% exceeds warning threshold"
            )
        
        # Memory alerts
        memory_thresholds = self.alert_thresholds.get('memory_percent', {})
        if metrics.memory_percent > memory_thresholds.get('critical', 100):
            self._generate_alert(
                'system_memory', 'critical',
                f"Memory usage {metrics.memory_percent:.1f}% exceeds critical threshold"
            )
        elif metrics.memory_percent > memory_thresholds.get('warning', 100):
            self._generate_alert(
                'system_memory', 'warning',
                f"Memory usage {metrics.memory_percent:.1f}% exceeds warning threshold"
            )
        
        # Disk alerts
        disk_thresholds = self.alert_thresholds.get('disk_usage_percent', {})
        if metrics.disk_usage_percent > disk_thresholds.get('critical', 100):
            self._generate_alert(
                'system_disk', 'critical',
                f"Disk usage {metrics.disk_usage_percent:.1f}% exceeds critical threshold"
            )
        elif metrics.disk_usage_percent > disk_thresholds.get('warning', 100):
            self._generate_alert(
                'system_disk', 'warning',
                f"Disk usage {metrics.disk_usage_percent:.1f}% exceeds warning threshold"
            )
    
    def _generate_alert(self, component_name: str, severity: str, message: str):
        """Generate a health alert"""
        alert_id = f"{component_name}_{severity}_{int(time.time())}"
        
        # Check if similar alert already exists
        existing_alert = None
        for alert in self.alerts:
            if (alert.component_name == component_name and 
                alert.severity == severity and 
                not alert.resolved and
                alert.message == message):
                existing_alert = alert
                break
        
        if not existing_alert:
            alert = HealthAlert(
                alert_id=alert_id,
                component_name=component_name,
                severity=severity,
                message=message,
                timestamp=datetime.now()
            )
            
            self.alerts.append(alert)
            
            # Log alert
            if severity == 'critical':
                self.logger.critical(f"HEALTH ALERT: {message}")
            else:
                self.logger.warning(f"HEALTH ALERT: {message}")
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        with self.lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    self.logger.info(f"Health alert resolved: {alert_id}")
                    break
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active (unresolved) alerts"""
        with self.lock:
            return [
                asdict(alert) for alert in self.alerts 
                if not alert.resolved
            ]
    
    def get_metrics_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for specified time window"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
            
            # Filter metrics within time window
            recent_metrics = [
                m for m in self.metrics_history 
                if m.timestamp >= cutoff_time
            ]
            
            if not recent_metrics:
                return {}
            
            # Calculate averages
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            avg_disk = sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics)
            
            # Get current metrics
            current_metrics = recent_metrics[-1] if recent_metrics else None
            
            return {
                'time_window_minutes': time_window_minutes,
                'sample_count': len(recent_metrics),
                'averages': {
                    'cpu_percent': round(avg_cpu, 2),
                    'memory_percent': round(avg_memory, 2),
                    'disk_usage_percent': round(avg_disk, 2)
                },
                'current': asdict(current_metrics) if current_metrics else None,
                'trends': self._calculate_trends(recent_metrics)
            }
    
    def _calculate_trends(self, metrics_list: List[SystemMetrics]) -> Dict[str, str]:
        """Calculate trends for metrics"""
        if len(metrics_list) < 2:
            return {}
        
        # Simple trend calculation (comparing first half vs second half)
        mid_point = len(metrics_list) // 2
        first_half = metrics_list[:mid_point]
        second_half = metrics_list[mid_point:]
        
        def get_trend(first_avg: float, second_avg: float) -> str:
            diff_percent = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
            if abs(diff_percent) < 5:
                return "stable"
            elif diff_percent > 0:
                return "increasing"
            else:
                return "decreasing"
        
        first_cpu = sum(m.cpu_percent for m in first_half) / len(first_half)
        second_cpu = sum(m.cpu_percent for m in second_half) / len(second_half)
        
        first_memory = sum(m.memory_percent for m in first_half) / len(first_half)
        second_memory = sum(m.memory_percent for m in second_half) / len(second_half)
        
        return {
            'cpu_trend': get_trend(first_cpu, second_cpu),
            'memory_trend': get_trend(first_memory, second_memory)
        }
    
    def start_monitoring(self):
        """Start background health monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("Background health monitoring started")
    
    def stop_monitoring(self):
        """Stop background health monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Background health monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        last_health_check = datetime.now()
        
        while self.monitoring_active:
            try:
                # Collect metrics every monitoring interval
                self.collect_metrics()
                
                # Perform health checks at specified interval
                if (datetime.now() - last_health_check).total_seconds() >= self.health_check_interval:
                    self.check_system_health()
                    last_health_check = datetime.now()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def export_health_data(self, filepath: str) -> bool:
        """Export health monitoring data to file"""
        try:
            with self.lock:
                export_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'system_health': self.check_system_health(),
                    'metrics_summary': self.get_metrics_summary(),
                    'active_alerts': self.get_active_alerts(),
                    'component_health_history': {
                        name: [asdict(h) for h in history]
                        for name, history in self.component_health_history.items()
                    },
                    'configuration': {
                        'monitoring_interval': self.monitoring_interval,
                        'health_check_interval': self.health_check_interval,
                        'alert_thresholds': self.alert_thresholds
                    }
                }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.logger.info(f"Health data exported to {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to export health data: {str(e)}")
            return False