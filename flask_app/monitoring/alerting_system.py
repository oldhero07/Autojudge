"""
Alerting System for ML Web App Enhancements

This module provides comprehensive alerting capabilities for threshold breaches,
critical errors, and automated remediation actions.
"""

import logging
import time
import threading
import smtplib
import json
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import requests
from collections import defaultdict, deque

# Configure logging
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class AlertChannel(Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    LOG = "log"
    SMS = "sms"

@dataclass
class AlertRule:
    """Alert rule definition"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    threshold_value: float
    comparison_operator: str  # '>', '<', '>=', '<=', '==', '!='
    severity: AlertSeverity
    channels: List[AlertChannel]
    cooldown_minutes: int = 5
    enabled: bool = True
    tags: Optional[Dict[str, str]] = None

@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None

@dataclass
class AlertChannel_Config:
    """Configuration for alert channels"""
    channel_type: AlertChannel
    config: Dict[str, Any]
    enabled: bool = True

class AlertThresholdManager:
    """
    Manager for alert thresholds and rules
    """
    
    def __init__(self):
        """Initialize threshold manager"""
        self.rules: Dict[str, AlertRule] = {}
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.ThresholdManager")
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        with self.lock:
            self.rules[rule.rule_id] = rule
            self.logger.info(f"Added alert rule: {rule.name} ({rule.rule_id})")
    
    def remove_rule(self, rule_id: str):
        """Remove an alert rule"""
        with self.lock:
            if rule_id in self.rules:
                rule = self.rules.pop(rule_id)
                self.logger.info(f"Removed alert rule: {rule.name} ({rule_id})")
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]):
        """Update an existing alert rule"""
        with self.lock:
            if rule_id in self.rules:
                rule = self.rules[rule_id]
                for key, value in updates.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                self.logger.info(f"Updated alert rule: {rule_id}")
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get an alert rule by ID"""
        return self.rules.get(rule_id)
    
    def get_rules_for_metric(self, metric_name: str) -> List[AlertRule]:
        """Get all rules for a specific metric"""
        with self.lock:
            return [rule for rule in self.rules.values() 
                   if rule.metric_name == metric_name and rule.enabled]
    
    def evaluate_metric(self, metric_name: str, current_value: float, 
                       metadata: Optional[Dict[str, Any]] = None) -> List[AlertRule]:
        """
        Evaluate metric against all applicable rules
        
        Returns:
            List of rules that should trigger alerts
        """
        triggered_rules = []
        rules = self.get_rules_for_metric(metric_name)
        
        for rule in rules:
            if self._evaluate_threshold(current_value, rule.threshold_value, rule.comparison_operator):
                triggered_rules.append(rule)
        
        return triggered_rules
    
    def _evaluate_threshold(self, current_value: float, threshold: float, operator: str) -> bool:
        """Evaluate threshold condition"""
        if operator == '>':
            return current_value > threshold
        elif operator == '<':
            return current_value < threshold
        elif operator == '>=':
            return current_value >= threshold
        elif operator == '<=':
            return current_value <= threshold
        elif operator == '==':
            return current_value == threshold
        elif operator == '!=':
            return current_value != threshold
        else:
            self.logger.warning(f"Unknown comparison operator: {operator}")
            return False

class AlertDeliveryManager:
    """
    Manager for alert delivery through various channels
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize delivery manager"""
        self.config = config or {}
        self.channels: Dict[AlertChannel, AlertChannel_Config] = {}
        self.delivery_history = deque(maxlen=1000)
        self.logger = logging.getLogger(f"{__name__}.DeliveryManager")
        
        # Initialize default channels
        self._initialize_channels()
    
    def _initialize_channels(self):
        """Initialize alert delivery channels"""
        # Log channel (always available)
        self.channels[AlertChannel.LOG] = AlertChannel_Config(
            channel_type=AlertChannel.LOG,
            config={},
            enabled=True
        )
        
        # Email channel
        email_config = self.config.get('email', {})
        if email_config.get('enabled', False):
            self.channels[AlertChannel.EMAIL] = AlertChannel_Config(
                channel_type=AlertChannel.EMAIL,
                config=email_config,
                enabled=True
            )
        
        # Webhook channel
        webhook_config = self.config.get('webhook', {})
        if webhook_config.get('enabled', False):
            self.channels[AlertChannel.WEBHOOK] = AlertChannel_Config(
                channel_type=AlertChannel.WEBHOOK,
                config=webhook_config,
                enabled=True
            )
        
        # Slack channel
        slack_config = self.config.get('slack', {})
        if slack_config.get('enabled', False):
            self.channels[AlertChannel.SLACK] = AlertChannel_Config(
                channel_type=AlertChannel.SLACK,
                config=slack_config,
                enabled=True
            )
    
    def deliver_alert(self, alert: Alert, channels: List[AlertChannel]) -> Dict[AlertChannel, bool]:
        """
        Deliver alert through specified channels
        
        Returns:
            Dictionary of channel -> success status
        """
        delivery_results = {}
        
        for channel in channels:
            if channel in self.channels and self.channels[channel].enabled:
                try:
                    success = self._deliver_to_channel(alert, channel)
                    delivery_results[channel] = success
                    
                    # Record delivery attempt
                    self.delivery_history.append({
                        'alert_id': alert.alert_id,
                        'channel': channel.value,
                        'success': success,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    self.logger.error(f"Failed to deliver alert {alert.alert_id} to {channel.value}: {str(e)}")
                    delivery_results[channel] = False
            else:
                self.logger.warning(f"Channel {channel.value} not configured or disabled")
                delivery_results[channel] = False
        
        return delivery_results
    
    def _deliver_to_channel(self, alert: Alert, channel: AlertChannel) -> bool:
        """Deliver alert to specific channel"""
        if channel == AlertChannel.LOG:
            return self._deliver_to_log(alert)
        elif channel == AlertChannel.EMAIL:
            return self._deliver_to_email(alert)
        elif channel == AlertChannel.WEBHOOK:
            return self._deliver_to_webhook(alert)
        elif channel == AlertChannel.SLACK:
            return self._deliver_to_slack(alert)
        else:
            self.logger.warning(f"Unsupported channel: {channel.value}")
            return False
    
    def _deliver_to_log(self, alert: Alert) -> bool:
        """Deliver alert to log"""
        try:
            log_level = logging.CRITICAL if alert.severity == AlertSeverity.EMERGENCY else logging.ERROR
            self.logger.log(log_level, f"ALERT [{alert.severity.value.upper()}]: {alert.title} - {alert.message}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to log alert: {str(e)}")
            return False
    
    def _deliver_to_email(self, alert: Alert) -> bool:
        """Deliver alert via email"""
        try:
            email_config = self.channels[AlertChannel.EMAIL].config
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = email_config['from_address']
            msg['To'] = ', '.join(email_config['to_addresses'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create email body
            body = self._create_email_body(alert)
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            with smtplib.SMTP(email_config['smtp_server'], email_config.get('smtp_port', 587)) as server:
                if email_config.get('use_tls', True):
                    server.starttls()
                
                if email_config.get('username') and email_config.get('password'):
                    server.login(email_config['username'], email_config['password'])
                
                server.send_message(msg)
            
            self.logger.info(f"Alert {alert.alert_id} sent via email")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")
            return False
    
    def _deliver_to_webhook(self, alert: Alert) -> bool:
        """Deliver alert via webhook"""
        try:
            webhook_config = self.channels[AlertChannel.WEBHOOK].config
            
            payload = {
                'alert_id': alert.alert_id,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'metric_name': alert.metric_name,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata or {}
            }
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=webhook_config.get('headers', {}),
                timeout=webhook_config.get('timeout', 30)
            )
            
            response.raise_for_status()
            self.logger.info(f"Alert {alert.alert_id} sent via webhook")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {str(e)}")
            return False
    
    def _deliver_to_slack(self, alert: Alert) -> bool:
        """Deliver alert via Slack"""
        try:
            slack_config = self.channels[AlertChannel.SLACK].config
            
            # Create Slack message
            color = {
                AlertSeverity.INFO: 'good',
                AlertSeverity.WARNING: 'warning',
                AlertSeverity.CRITICAL: 'danger',
                AlertSeverity.EMERGENCY: 'danger'
            }.get(alert.severity, 'warning')
            
            payload = {
                'channel': slack_config['channel'],
                'username': slack_config.get('username', 'AlertBot'),
                'attachments': [{
                    'color': color,
                    'title': alert.title,
                    'text': alert.message,
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity.value.upper(), 'short': True},
                        {'title': 'Metric', 'value': alert.metric_name, 'short': True},
                        {'title': 'Current Value', 'value': str(alert.current_value), 'short': True},
                        {'title': 'Threshold', 'value': str(alert.threshold_value), 'short': True}
                    ],
                    'timestamp': int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(
                slack_config['webhook_url'],
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            self.logger.info(f"Alert {alert.alert_id} sent via Slack")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {str(e)}")
            return False
    
    def _create_email_body(self, alert: Alert) -> str:
        """Create HTML email body for alert"""
        return f"""
        <html>
        <body>
            <h2 style="color: {'red' if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] else 'orange'};">
                Alert: {alert.title}
            </h2>
            
            <table border="1" cellpadding="5" cellspacing="0">
                <tr><td><strong>Severity</strong></td><td>{alert.severity.value.upper()}</td></tr>
                <tr><td><strong>Metric</strong></td><td>{alert.metric_name}</td></tr>
                <tr><td><strong>Current Value</strong></td><td>{alert.current_value}</td></tr>
                <tr><td><strong>Threshold</strong></td><td>{alert.threshold_value}</td></tr>
                <tr><td><strong>Time</strong></td><td>{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</td></tr>
                <tr><td><strong>Alert ID</strong></td><td>{alert.alert_id}</td></tr>
            </table>
            
            <h3>Description</h3>
            <p>{alert.message}</p>
            
            {f'<h3>Additional Information</h3><pre>{json.dumps(alert.metadata, indent=2)}</pre>' if alert.metadata else ''}
        </body>
        </html>
        """

class AlertingSystem:
    """
    Comprehensive alerting system with threshold monitoring and automated remediation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize alerting system"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.threshold_manager = AlertThresholdManager()
        self.delivery_manager = AlertDeliveryManager(self.config.get('delivery', {}))
        
        # Alert storage and tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history = deque(maxlen=self.config.get('max_alert_history', 10000))
        self.cooldown_tracker: Dict[str, datetime] = {}
        
        # Remediation actions
        self.remediation_actions: Dict[str, Callable] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background processing
        self.processing_active = False
        self.processing_thread = None
        
        # Initialize default rules
        self._initialize_default_rules()
        
        self.logger.info("AlertingSystem initialized")
    
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="cpu_high",
                name="High CPU Usage",
                description="CPU usage exceeds threshold",
                metric_name="cpu_percent",
                threshold_value=85.0,
                comparison_operator=">",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL],
                cooldown_minutes=5
            ),
            AlertRule(
                rule_id="cpu_critical",
                name="Critical CPU Usage",
                description="CPU usage critically high",
                metric_name="cpu_percent",
                threshold_value=95.0,
                comparison_operator=">",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.WEBHOOK],
                cooldown_minutes=2
            ),
            AlertRule(
                rule_id="memory_high",
                name="High Memory Usage",
                description="Memory usage exceeds threshold",
                metric_name="memory_percent",
                threshold_value=85.0,
                comparison_operator=">",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL],
                cooldown_minutes=5
            ),
            AlertRule(
                rule_id="memory_critical",
                name="Critical Memory Usage",
                description="Memory usage critically high",
                metric_name="memory_percent",
                threshold_value=95.0,
                comparison_operator=">",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.WEBHOOK],
                cooldown_minutes=2
            ),
            AlertRule(
                rule_id="response_time_high",
                name="High Response Time",
                description="Response time exceeds acceptable threshold",
                metric_name="response_time_ms",
                threshold_value=2000.0,
                comparison_operator=">",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                cooldown_minutes=10
            )
        ]
        
        for rule in default_rules:
            self.threshold_manager.add_rule(rule)
    
    def check_metric(self, metric_name: str, current_value: float, 
                    metadata: Optional[Dict[str, Any]] = None,
                    correlation_id: Optional[str] = None):
        """
        Check metric against alert rules and trigger alerts if necessary
        
        Args:
            metric_name: Name of the metric
            current_value: Current value of the metric
            metadata: Additional metadata for the alert
            correlation_id: Correlation ID for tracing
        """
        triggered_rules = self.threshold_manager.evaluate_metric(metric_name, current_value, metadata)
        
        for rule in triggered_rules:
            # Check cooldown
            if self._is_in_cooldown(rule.rule_id):
                continue
            
            # Create alert
            alert = self._create_alert(rule, current_value, metadata, correlation_id)
            
            # Process alert
            self._process_alert(alert)
    
    def _create_alert(self, rule: AlertRule, current_value: float, 
                     metadata: Optional[Dict[str, Any]], 
                     correlation_id: Optional[str]) -> Alert:
        """Create alert from rule and current conditions"""
        alert_id = f"{rule.rule_id}_{int(time.time())}"
        
        return Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            severity=rule.severity,
            title=rule.name,
            message=f"{rule.description}. Current value: {current_value}, Threshold: {rule.threshold_value}",
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold_value=rule.threshold_value,
            timestamp=datetime.now(),
            metadata=metadata,
            correlation_id=correlation_id
        )
    
    def _process_alert(self, alert: Alert):
        """Process and deliver alert"""
        with self.lock:
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Update cooldown
            self.cooldown_tracker[alert.rule_id] = datetime.now()
            
            # Get rule for delivery channels
            rule = self.threshold_manager.get_rule(alert.rule_id)
            if rule:
                # Deliver alert
                delivery_results = self.delivery_manager.deliver_alert(alert, rule.channels)
                
                # Log delivery results
                successful_channels = [ch.value for ch, success in delivery_results.items() if success]
                failed_channels = [ch.value for ch, success in delivery_results.items() if not success]
                
                if successful_channels:
                    self.logger.info(f"Alert {alert.alert_id} delivered to: {', '.join(successful_channels)}")
                
                if failed_channels:
                    self.logger.warning(f"Alert {alert.alert_id} failed to deliver to: {', '.join(failed_channels)}")
                
                # Trigger remediation if configured
                self._trigger_remediation(alert)
    
    def _is_in_cooldown(self, rule_id: str) -> bool:
        """Check if rule is in cooldown period"""
        if rule_id not in self.cooldown_tracker:
            return False
        
        rule = self.threshold_manager.get_rule(rule_id)
        if not rule:
            return False
        
        last_alert_time = self.cooldown_tracker[rule_id]
        cooldown_period = timedelta(minutes=rule.cooldown_minutes)
        
        return datetime.now() - last_alert_time < cooldown_period
    
    def _trigger_remediation(self, alert: Alert):
        """Trigger automated remediation actions"""
        if alert.rule_id in self.remediation_actions:
            try:
                remediation_func = self.remediation_actions[alert.rule_id]
                self.logger.info(f"Triggering remediation for alert {alert.alert_id}")
                
                # Execute remediation in background
                threading.Thread(
                    target=self._execute_remediation,
                    args=(remediation_func, alert),
                    daemon=True
                ).start()
                
            except Exception as e:
                self.logger.error(f"Failed to trigger remediation for {alert.alert_id}: {str(e)}")
    
    def _execute_remediation(self, remediation_func: Callable, alert: Alert):
        """Execute remediation action"""
        try:
            result = remediation_func(alert)
            
            if result:
                self.logger.info(f"Remediation successful for alert {alert.alert_id}")
                # Could automatically resolve alert here
            else:
                self.logger.warning(f"Remediation failed for alert {alert.alert_id}")
                
        except Exception as e:
            self.logger.error(f"Remediation execution failed for {alert.alert_id}: {str(e)}")
    
    def register_remediation_action(self, rule_id: str, action_func: Callable):
        """Register automated remediation action for a rule"""
        self.remediation_actions[rule_id] = action_func
        self.logger.info(f"Registered remediation action for rule: {rule_id}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an active alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now()
                
                self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                self.logger.info(f"Alert {alert_id} resolved")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        with self.lock:
            alerts = list(self.active_alerts.values())
            
            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]
            
            return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics"""
        with self.lock:
            # Count alerts by severity
            severity_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                severity_counts[alert.severity.value] += 1
            
            # Recent alert trends (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_alerts = [alert for alert in self.alert_history if alert.timestamp >= recent_cutoff]
            
            # Most frequent alert rules
            rule_counts = defaultdict(int)
            for alert in recent_alerts:
                rule_counts[alert.rule_id] += 1
            
            return {
                'active_alerts': {
                    'total': len(self.active_alerts),
                    'by_severity': dict(severity_counts)
                },
                'recent_alerts_24h': len(recent_alerts),
                'total_alerts_in_history': len(self.alert_history),
                'most_frequent_rules': dict(sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                'configured_rules': len(self.threshold_manager.rules),
                'delivery_channels': len(self.delivery_manager.channels)
            }
    
    def export_alerts(self, filepath: str) -> bool:
        """Export alert data to file"""
        try:
            with self.lock:
                export_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'active_alerts': [asdict(alert) for alert in self.active_alerts.values()],
                    'alert_history': [asdict(alert) for alert in self.alert_history],
                    'alert_statistics': self.get_alert_statistics(),
                    'configured_rules': [asdict(rule) for rule in self.threshold_manager.rules.values()]
                }
                
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                self.logger.info(f"Alert data exported to {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to export alert data: {str(e)}")
            return False