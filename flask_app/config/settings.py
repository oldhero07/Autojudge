"""
Default Configuration Settings for Enhanced ML Web App

This module provides default configuration values and environment-specific overrides.
"""

import os
from typing import Dict, Any

# Base configuration
BASE_CONFIG = {
    # Application settings
    'app': {
        'name': 'ML Web App Enhanced',
        'version': '2.0.0',
        'environment': os.getenv('ENVIRONMENT', 'development'),
        'debug': os.getenv('DEBUG', 'false').lower() == 'true',
        'host': os.getenv('HOST', '0.0.0.0'),
        'port': int(os.getenv('PORT', 5000)),
        'secret_key': os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    },
    
    # Database settings
    'database': {
        'url': os.getenv('DATABASE_URL', 'sqlite:///ml_webapp.db'),
        'pool_size': int(os.getenv('DB_POOL_SIZE', 10)),
        'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', 20)),
        'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', 30))
    },
    
    # Redis settings
    'redis': {
        'host': os.getenv('REDIS_HOST', 'localhost'),
        'port': int(os.getenv('REDIS_PORT', 6379)),
        'db': int(os.getenv('REDIS_DB', 0)),
        'password': os.getenv('REDIS_PASSWORD'),
        'timeout': int(os.getenv('REDIS_TIMEOUT', 5)),
        'connect_timeout': int(os.getenv('REDIS_CONNECT_TIMEOUT', 5))
    },
    
    # Machine Learning settings
    'ml': {
        'model_path': os.getenv('ML_MODEL_PATH', 'models/'),
        'enable_auto_selection': os.getenv('ML_AUTO_SELECTION', 'true').lower() == 'true',
        'performance_threshold': float(os.getenv('ML_PERFORMANCE_THRESHOLD', 0.8)),
        'max_batch_size': int(os.getenv('ML_MAX_BATCH_SIZE', 32)),
        'prediction_timeout': int(os.getenv('ML_PREDICTION_TIMEOUT', 30)),
        'ensemble': {
            'voting_strategy': os.getenv('ML_ENSEMBLE_VOTING', 'weighted'),
            'enable_adaptive_weights': os.getenv('ML_ENSEMBLE_ADAPTIVE', 'true').lower() == 'true',
            'weight_update_frequency': int(os.getenv('ML_ENSEMBLE_UPDATE_FREQ', 100))
        }
    },
    
    # Caching settings
    'cache': {
        'local': {
            'enabled': os.getenv('CACHE_LOCAL_ENABLED', 'true').lower() == 'true',
            'max_size': int(os.getenv('CACHE_LOCAL_MAX_SIZE', 1000)),
            'default_ttl': int(os.getenv('CACHE_LOCAL_TTL', 3600))
        },
        'redis': {
            'enabled': os.getenv('CACHE_REDIS_ENABLED', 'false').lower() == 'true',
            'key_prefix': os.getenv('CACHE_REDIS_PREFIX', 'mlapp:'),
            'default_ttl': int(os.getenv('CACHE_REDIS_TTL', 3600))
        }
    },
    
    # Processing settings
    'processing': {
        'max_workers': int(os.getenv('PROCESSING_MAX_WORKERS', 4)),
        'default_mode': os.getenv('PROCESSING_DEFAULT_MODE', 'threaded'),
        'enable_batching': os.getenv('PROCESSING_ENABLE_BATCHING', 'true').lower() == 'true',
        'queue_size': int(os.getenv('PROCESSING_QUEUE_SIZE', 1000)),
        'batch': {
            'max_batch_size': int(os.getenv('PROCESSING_BATCH_MAX_SIZE', 32)),
            'min_batch_size': int(os.getenv('PROCESSING_BATCH_MIN_SIZE', 1)),
            'batch_timeout_ms': int(os.getenv('PROCESSING_BATCH_TIMEOUT', 100)),
            'strategy': os.getenv('PROCESSING_BATCH_STRATEGY', 'adaptive')
        }
    },
    
    # Monitoring settings
    'monitoring': {
        'enabled': os.getenv('MONITORING_ENABLED', 'true').lower() == 'true',
        'health_check_interval': int(os.getenv('MONITORING_HEALTH_INTERVAL', 30)),
        'metrics_retention_hours': int(os.getenv('MONITORING_METRICS_RETENTION', 24)),
        'performance_threshold': {
            'response_time_ms': int(os.getenv('MONITORING_RESPONSE_TIME_THRESHOLD', 1000)),
            'error_rate_percent': float(os.getenv('MONITORING_ERROR_RATE_THRESHOLD', 5.0)),
            'cpu_percent': float(os.getenv('MONITORING_CPU_THRESHOLD', 80.0)),
            'memory_percent': float(os.getenv('MONITORING_MEMORY_THRESHOLD', 85.0))
        }
    },
    
    # Alerting settings
    'alerting': {
        'enabled': os.getenv('ALERTING_ENABLED', 'true').lower() == 'true',
        'email': {
            'enabled': os.getenv('ALERTING_EMAIL_ENABLED', 'false').lower() == 'true',
            'smtp_server': os.getenv('ALERTING_EMAIL_SMTP_SERVER', 'localhost'),
            'smtp_port': int(os.getenv('ALERTING_EMAIL_SMTP_PORT', 587)),
            'use_tls': os.getenv('ALERTING_EMAIL_USE_TLS', 'true').lower() == 'true',
            'username': os.getenv('ALERTING_EMAIL_USERNAME'),
            'password': os.getenv('ALERTING_EMAIL_PASSWORD'),
            'from_address': os.getenv('ALERTING_EMAIL_FROM', 'alerts@mlapp.com'),
            'to_addresses': os.getenv('ALERTING_EMAIL_TO', 'admin@mlapp.com').split(',')
        },
        'webhook': {
            'enabled': os.getenv('ALERTING_WEBHOOK_ENABLED', 'false').lower() == 'true',
            'url': os.getenv('ALERTING_WEBHOOK_URL'),
            'timeout': int(os.getenv('ALERTING_WEBHOOK_TIMEOUT', 30)),
            'headers': {}
        },
        'slack': {
            'enabled': os.getenv('ALERTING_SLACK_ENABLED', 'false').lower() == 'true',
            'webhook_url': os.getenv('ALERTING_SLACK_WEBHOOK_URL'),
            'channel': os.getenv('ALERTING_SLACK_CHANNEL', '#alerts'),
            'username': os.getenv('ALERTING_SLACK_USERNAME', 'AlertBot')
        }
    },
    
    # Circuit breaker settings
    'circuit_breaker': {
        'failure_threshold': int(os.getenv('CIRCUIT_BREAKER_FAILURE_THRESHOLD', 5)),
        'recovery_timeout': int(os.getenv('CIRCUIT_BREAKER_RECOVERY_TIMEOUT', 60)),
        'half_open_max_calls': int(os.getenv('CIRCUIT_BREAKER_HALF_OPEN_CALLS', 3))
    },
    
    # Error recovery settings
    'error_recovery': {
        'enable_fallback_models': os.getenv('ERROR_RECOVERY_FALLBACK_MODELS', 'true').lower() == 'true',
        'enable_graceful_degradation': os.getenv('ERROR_RECOVERY_GRACEFUL_DEGRADATION', 'true').lower() == 'true',
        'enable_cached_responses': os.getenv('ERROR_RECOVERY_CACHED_RESPONSES', 'true').lower() == 'true',
        'max_retry_attempts': int(os.getenv('ERROR_RECOVERY_MAX_RETRIES', 3)),
        'retry_delay_seconds': int(os.getenv('ERROR_RECOVERY_RETRY_DELAY', 1))
    },
    
    # Logging settings
    'logging': {
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'format': os.getenv('LOG_FORMAT', 'json'),
        'file_path': os.getenv('LOG_FILE_PATH'),
        'max_file_size_mb': int(os.getenv('LOG_MAX_FILE_SIZE', 100)),
        'backup_count': int(os.getenv('LOG_BACKUP_COUNT', 5)),
        'structured_logging': os.getenv('LOG_STRUCTURED', 'true').lower() == 'true'
    },
    
    # Security settings
    'security': {
        'enable_cors': os.getenv('SECURITY_ENABLE_CORS', 'true').lower() == 'true',
        'cors_origins': os.getenv('SECURITY_CORS_ORIGINS', '*').split(','),
        'rate_limiting': {
            'enabled': os.getenv('SECURITY_RATE_LIMITING', 'false').lower() == 'true',
            'requests_per_minute': int(os.getenv('SECURITY_RATE_LIMIT_RPM', 100)),
            'burst_size': int(os.getenv('SECURITY_RATE_LIMIT_BURST', 20))
        }
    }
}

# Environment-specific configurations
ENVIRONMENT_CONFIGS = {
    'development': {
        'app': {
            'debug': True
        },
        'logging': {
            'level': 'DEBUG'
        },
        'monitoring': {
            'health_check_interval': 10
        }
    },
    
    'testing': {
        'app': {
            'debug': False
        },
        'database': {
            'url': 'sqlite:///:memory:'
        },
        'cache': {
            'local': {
                'max_size': 100
            },
            'redis': {
                'enabled': False
            }
        },
        'logging': {
            'level': 'WARNING'
        }
    },
    
    'staging': {
        'app': {
            'debug': False
        },
        'monitoring': {
            'enabled': True,
            'health_check_interval': 15
        },
        'alerting': {
            'enabled': True
        },
        'cache': {
            'redis': {
                'enabled': True
            }
        }
    },
    
    'production': {
        'app': {
            'debug': False
        },
        'monitoring': {
            'enabled': True,
            'health_check_interval': 30
        },
        'alerting': {
            'enabled': True,
            'email': {
                'enabled': True
            }
        },
        'cache': {
            'redis': {
                'enabled': True
            }
        },
        'processing': {
            'max_workers': 8
        },
        'security': {
            'rate_limiting': {
                'enabled': True
            }
        }
    }
}

def get_config(environment: str = None) -> Dict[str, Any]:
    """
    Get configuration for specified environment.
    
    Args:
        environment: Environment name (development, testing, staging, production)
        
    Returns:
        Merged configuration dictionary
    """
    if environment is None:
        environment = os.getenv('ENVIRONMENT', 'development')
    
    # Start with base config
    config = BASE_CONFIG.copy()
    
    # Apply environment-specific overrides
    if environment in ENVIRONMENT_CONFIGS:
        env_config = ENVIRONMENT_CONFIGS[environment]
        config = _deep_merge(config, env_config)
    
    return config

def _deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries.
    
    Args:
        base_dict: Base dictionary
        override_dict: Dictionary with override values
        
    Returns:
        Merged dictionary
    """
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

# Export default configuration
DEFAULT_CONFIG = get_config()