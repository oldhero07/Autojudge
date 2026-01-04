"""
Configuration and Secrets Management for ML Web App

This module provides comprehensive configuration management with environment-based
loading, secure secret handling, validation, and hot-reloading capabilities.
"""

import os
import json
import yaml
import logging
import threading
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import watchdog.observers
from watchdog.events import FileSystemEventHandler

# Configure logging
logger = logging.getLogger(__name__)

class ConfigFormat(Enum):
    """Configuration file formats"""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"
    TOML = "toml"

class SecretType(Enum):
    """Types of secrets"""
    API_KEY = "api_key"
    DATABASE_PASSWORD = "database_password"
    ENCRYPTION_KEY = "encryption_key"
    JWT_SECRET = "jwt_secret"
    OAUTH_SECRET = "oauth_secret"
    GENERIC = "generic"

class ConfigSource(Enum):
    """Configuration sources"""
    FILE = "file"
    ENVIRONMENT = "environment"
    VAULT = "vault"
    DATABASE = "database"
    REMOTE = "remote"

@dataclass
class ConfigValue:
    """Configuration value with metadata"""
    key: str
    value: Any
    source: ConfigSource
    is_secret: bool = False
    secret_type: Optional[SecretType] = None
    last_updated: datetime = field(default_factory=datetime.now)
    validation_rules: Optional[List[str]] = None
    description: Optional[str] = None

@dataclass
class ConfigSchema:
    """Configuration schema definition"""
    key: str
    value_type: type
    required: bool = True
    default_value: Any = None
    validation_function: Optional[Callable] = None
    is_secret: bool = False
    secret_type: Optional[SecretType] = None
    description: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None

class SecretManager:
    """
    Secure secret management with encryption
    """
    
    def __init__(self, master_key: Optional[str] = None):
        """Initialize secret manager"""
        self.logger = logging.getLogger(f"{__name__}.SecretManager")
        
        # Initialize encryption
        if master_key:
            self.cipher = self._create_cipher_from_key(master_key)
        else:
            # Generate or load master key
            self.cipher = self._initialize_cipher()
        
        # Secret storage
        self.secrets: Dict[str, str] = {}
        self.secret_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger.info("SecretManager initialized")
    
    def _initialize_cipher(self) -> Fernet:
        """Initialize cipher with master key"""
        # Try to load existing key
        key_file = Path(".secrets_key")
        
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    key = f.read()
                return Fernet(key)
            except Exception as e:
                self.logger.warning(f"Failed to load existing key: {str(e)}")
        
        # Generate new key
        key = Fernet.generate_key()
        
        try:
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Set restrictive permissions
            os.chmod(key_file, 0o600)
            self.logger.info("Generated new master key")
            
        except Exception as e:
            self.logger.error(f"Failed to save master key: {str(e)}")
        
        return Fernet(key)
    
    def _create_cipher_from_key(self, master_key: str) -> Fernet:
        """Create cipher from provided master key"""
        # Derive key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'ml_webapp_salt',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        return Fernet(key)
    
    def store_secret(self, key: str, value: str, secret_type: SecretType = SecretType.GENERIC,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store encrypted secret"""
        try:
            with self.lock:
                # Encrypt value
                encrypted_value = self.cipher.encrypt(value.encode()).decode()
                
                # Store secret
                self.secrets[key] = encrypted_value
                
                # Store metadata
                self.secret_metadata[key] = {
                    'secret_type': secret_type.value,
                    'created_at': datetime.now().isoformat(),
                    'last_accessed': None,
                    'access_count': 0,
                    'metadata': metadata or {}
                }
                
                self.logger.info(f"Stored secret: {key}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store secret {key}: {str(e)}")
            return False
    
    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve and decrypt secret"""
        try:
            with self.lock:
                if key not in self.secrets:
                    return None
                
                # Decrypt value
                encrypted_value = self.secrets[key].encode()
                decrypted_value = self.cipher.decrypt(encrypted_value).decode()
                
                # Update access metadata
                if key in self.secret_metadata:
                    self.secret_metadata[key]['last_accessed'] = datetime.now().isoformat()
                    self.secret_metadata[key]['access_count'] += 1
                
                return decrypted_value
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret {key}: {str(e)}")
            return None
    
    def delete_secret(self, key: str) -> bool:
        """Delete secret"""
        try:
            with self.lock:
                if key in self.secrets:
                    del self.secrets[key]
                
                if key in self.secret_metadata:
                    del self.secret_metadata[key]
                
                self.logger.info(f"Deleted secret: {key}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to delete secret {key}: {str(e)}")
            return False
    
    def list_secrets(self) -> List[Dict[str, Any]]:
        """List all secrets (metadata only)"""
        with self.lock:
            secrets_list = []
            
            for key, metadata in self.secret_metadata.items():
                secrets_list.append({
                    'key': key,
                    'secret_type': metadata.get('secret_type'),
                    'created_at': metadata.get('created_at'),
                    'last_accessed': metadata.get('last_accessed'),
                    'access_count': metadata.get('access_count', 0)
                })
            
            return secrets_list
    
    def export_secrets(self, filepath: str, include_values: bool = False) -> bool:
        """Export secrets to file"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'secrets': {}
            }
            
            with self.lock:
                for key, metadata in self.secret_metadata.items():
                    secret_data = {
                        'metadata': metadata
                    }
                    
                    if include_values:
                        secret_data['encrypted_value'] = self.secrets.get(key)
                    
                    export_data['secrets'][key] = secret_data
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported secrets to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export secrets: {str(e)}")
            return False

class ConfigValidator:
    """
    Configuration validation with schema support
    """
    
    def __init__(self):
        """Initialize config validator"""
        self.schemas: Dict[str, ConfigSchema] = {}
        self.validation_errors: List[str] = []
        self.logger = logging.getLogger(f"{__name__}.ConfigValidator")
    
    def add_schema(self, schema: ConfigSchema):
        """Add configuration schema"""
        self.schemas[schema.key] = schema
        self.logger.debug(f"Added schema for: {schema.key}")
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration against schemas"""
        self.validation_errors = []
        
        # Check required fields
        for key, schema in self.schemas.items():
            if schema.required and key not in config:
                self.validation_errors.append(f"Required configuration key missing: {key}")
                continue
            
            if key in config:
                self._validate_value(key, config[key], schema)
        
        # Check for unknown keys
        known_keys = set(self.schemas.keys())
        config_keys = set(config.keys())
        unknown_keys = config_keys - known_keys
        
        for unknown_key in unknown_keys:
            self.validation_errors.append(f"Unknown configuration key: {unknown_key}")
        
        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors.copy()
    
    def _validate_value(self, key: str, value: Any, schema: ConfigSchema):
        """Validate individual configuration value"""
        # Type validation
        if not isinstance(value, schema.value_type):
            self.validation_errors.append(
                f"Configuration key '{key}' has invalid type. Expected {schema.value_type.__name__}, got {type(value).__name__}"
            )
            return
        
        # Allowed values validation
        if schema.allowed_values and value not in schema.allowed_values:
            self.validation_errors.append(
                f"Configuration key '{key}' has invalid value. Allowed values: {schema.allowed_values}"
            )
        
        # Range validation for numeric values
        if isinstance(value, (int, float)):
            if schema.min_value is not None and value < schema.min_value:
                self.validation_errors.append(
                    f"Configuration key '{key}' value {value} is below minimum {schema.min_value}"
                )
            
            if schema.max_value is not None and value > schema.max_value:
                self.validation_errors.append(
                    f"Configuration key '{key}' value {value} is above maximum {schema.max_value}"
                )
        
        # Custom validation function
        if schema.validation_function:
            try:
                if not schema.validation_function(value):
                    self.validation_errors.append(
                        f"Configuration key '{key}' failed custom validation"
                    )
            except Exception as e:
                self.validation_errors.append(
                    f"Configuration key '{key}' validation function error: {str(e)}"
                )

class ConfigFileWatcher(FileSystemEventHandler):
    """
    File system watcher for configuration hot-reloading
    """
    
    def __init__(self, config_manager, config_files: List[str]):
        """Initialize file watcher"""
        self.config_manager = config_manager
        self.config_files = set(config_files)
        self.logger = logging.getLogger(f"{__name__}.ConfigFileWatcher")
    
    def on_modified(self, event):
        """Handle file modification events"""
        if not event.is_directory and event.src_path in self.config_files:
            self.logger.info(f"Configuration file modified: {event.src_path}")
            
            # Trigger reload
            try:
                self.config_manager.reload_configuration()
            except Exception as e:
                self.logger.error(f"Failed to reload configuration: {str(e)}")

class ConfigManager:
    """
    Comprehensive configuration manager with multiple sources and hot-reloading
    """
    
    def __init__(self, config_dir: str = "config", enable_hot_reload: bool = True):
        """Initialize configuration manager"""
        self.config_dir = Path(config_dir)
        self.enable_hot_reload = enable_hot_reload
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.secret_manager = SecretManager()
        self.validator = ConfigValidator()
        
        # Configuration storage
        self.config_values: Dict[str, ConfigValue] = {}
        self.config_files: List[str] = []
        
        # Hot reload
        self.file_watcher = None
        self.observer = None
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Change callbacks
        self.change_callbacks: List[Callable] = []
        
        # Initialize default schemas
        self._initialize_default_schemas()
        
        # Load initial configuration
        self.load_configuration()
        
        # Start file watcher if enabled
        if self.enable_hot_reload:
            self._start_file_watcher()
        
        self.logger.info(f"ConfigManager initialized (hot_reload: {self.enable_hot_reload})")
    
    def _initialize_default_schemas(self):
        """Initialize default configuration schemas"""
        default_schemas = [
            ConfigSchema(
                key="app.debug",
                value_type=bool,
                default_value=False,
                description="Enable debug mode"
            ),
            ConfigSchema(
                key="app.host",
                value_type=str,
                default_value="localhost",
                description="Application host"
            ),
            ConfigSchema(
                key="app.port",
                value_type=int,
                default_value=5000,
                min_value=1,
                max_value=65535,
                description="Application port"
            ),
            ConfigSchema(
                key="database.url",
                value_type=str,
                required=True,
                is_secret=True,
                secret_type=SecretType.DATABASE_PASSWORD,
                description="Database connection URL"
            ),
            ConfigSchema(
                key="redis.host",
                value_type=str,
                default_value="localhost",
                description="Redis host"
            ),
            ConfigSchema(
                key="redis.port",
                value_type=int,
                default_value=6379,
                min_value=1,
                max_value=65535,
                description="Redis port"
            ),
            ConfigSchema(
                key="ml.model_path",
                value_type=str,
                required=True,
                description="Path to ML model files"
            ),
            ConfigSchema(
                key="ml.max_batch_size",
                value_type=int,
                default_value=32,
                min_value=1,
                max_value=1000,
                description="Maximum batch size for ML processing"
            ),
            ConfigSchema(
                key="monitoring.enabled",
                value_type=bool,
                default_value=True,
                description="Enable monitoring"
            ),
            ConfigSchema(
                key="logging.level",
                value_type=str,
                default_value="INFO",
                allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                description="Logging level"
            )
        ]
        
        for schema in default_schemas:
            self.validator.add_schema(schema)
    
    def load_configuration(self):
        """Load configuration from all sources"""
        self.logger.info("Loading configuration...")
        
        with self.lock:
            # Clear existing configuration
            self.config_values.clear()
            
            # Load from files
            self._load_from_files()
            
            # Load from environment variables
            self._load_from_environment()
            
            # Apply defaults for missing required values
            self._apply_defaults()
            
            # Validate configuration
            config_dict = {key: value.value for key, value in self.config_values.items()}
            is_valid, errors = self.validator.validate_config(config_dict)
            
            if not is_valid:
                self.logger.error("Configuration validation failed:")
                for error in errors:
                    self.logger.error(f"  - {error}")
                raise ValueError("Invalid configuration")
            
            self.logger.info(f"Configuration loaded successfully ({len(self.config_values)} values)")
    
    def _load_from_files(self):
        """Load configuration from files"""
        if not self.config_dir.exists():
            self.logger.warning(f"Configuration directory not found: {self.config_dir}")
            return
        
        # Look for configuration files
        config_patterns = [
            "config.json",
            "config.yaml",
            "config.yml",
            "settings.json",
            "settings.yaml",
            "settings.yml"
        ]
        
        for pattern in config_patterns:
            config_file = self.config_dir / pattern
            if config_file.exists():
                self._load_config_file(str(config_file))
    
    def _load_config_file(self, filepath: str):
        """Load configuration from specific file"""
        try:
            file_path = Path(filepath)
            self.config_files.append(filepath)
            
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            # Flatten nested configuration
            flat_config = self._flatten_dict(config_data)
            
            # Store configuration values
            for key, value in flat_config.items():
                config_value = ConfigValue(
                    key=key,
                    value=value,
                    source=ConfigSource.FILE,
                    is_secret=self._is_secret_key(key)
                )
                
                # Store secrets securely
                if config_value.is_secret:
                    secret_type = self._get_secret_type(key)
                    self.secret_manager.store_secret(key, str(value), secret_type)
                    config_value.value = f"<secret:{key}>"
                
                self.config_values[key] = config_value
            
            self.logger.info(f"Loaded configuration from: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration file {filepath}: {str(e)}")
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_prefix = "MLAPP_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Convert environment variable name to config key
                config_key = key[len(env_prefix):].lower().replace('_', '.')
                
                # Parse value type
                parsed_value = self._parse_env_value(value)
                
                config_value = ConfigValue(
                    key=config_key,
                    value=parsed_value,
                    source=ConfigSource.ENVIRONMENT,
                    is_secret=self._is_secret_key(config_key)
                )
                
                # Store secrets securely
                if config_value.is_secret:
                    secret_type = self._get_secret_type(config_key)
                    self.secret_manager.store_secret(config_key, str(parsed_value), secret_type)
                    config_value.value = f"<secret:{config_key}>"
                
                self.config_values[config_key] = config_value
        
        self.logger.debug(f"Loaded {sum(1 for k in os.environ if k.startswith(env_prefix))} environment variables")
    
    def _apply_defaults(self):
        """Apply default values for missing configuration"""
        for key, schema in self.validator.schemas.items():
            if key not in self.config_values and schema.default_value is not None:
                config_value = ConfigValue(
                    key=key,
                    value=schema.default_value,
                    source=ConfigSource.FILE,  # Default source
                    is_secret=schema.is_secret
                )
                
                self.config_values[key] = config_value
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Boolean values
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # JSON values
        if value.startswith('{') or value.startswith('['):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # String value
        return value
    
    def _is_secret_key(self, key: str) -> bool:
        """Check if configuration key contains sensitive data"""
        secret_keywords = [
            'password', 'secret', 'key', 'token', 'credential',
            'auth', 'api_key', 'private', 'cert', 'ssl'
        ]
        
        key_lower = key.lower()
        return any(keyword in key_lower for keyword in secret_keywords)
    
    def _get_secret_type(self, key: str) -> SecretType:
        """Determine secret type from key"""
        key_lower = key.lower()
        
        if 'database' in key_lower or 'db' in key_lower:
            return SecretType.DATABASE_PASSWORD
        elif 'api' in key_lower:
            return SecretType.API_KEY
        elif 'jwt' in key_lower:
            return SecretType.JWT_SECRET
        elif 'oauth' in key_lower:
            return SecretType.OAUTH_SECRET
        elif 'encrypt' in key_lower:
            return SecretType.ENCRYPTION_KEY
        else:
            return SecretType.GENERIC
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        with self.lock:
            if key not in self.config_values:
                return default
            
            config_value = self.config_values[key]
            
            # Retrieve secret value if needed
            if config_value.is_secret and isinstance(config_value.value, str) and config_value.value.startswith('<secret:'):
                secret_key = config_value.value[8:-1]  # Remove <secret: and >
                return self.secret_manager.get_secret(secret_key)
            
            return config_value.value
    
    def set(self, key: str, value: Any, source: ConfigSource = ConfigSource.ENVIRONMENT) -> bool:
        """Set configuration value"""
        try:
            with self.lock:
                is_secret = self._is_secret_key(key)
                
                config_value = ConfigValue(
                    key=key,
                    value=value,
                    source=source,
                    is_secret=is_secret
                )
                
                # Store secrets securely
                if is_secret:
                    secret_type = self._get_secret_type(key)
                    self.secret_manager.store_secret(key, str(value), secret_type)
                    config_value.value = f"<secret:{key}>"
                
                self.config_values[key] = config_value
                
                # Notify callbacks
                self._notify_change_callbacks(key, value)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to set configuration {key}: {str(e)}")
            return False
    
    def reload_configuration(self):
        """Reload configuration from all sources"""
        self.logger.info("Reloading configuration...")
        
        try:
            old_config = {key: value.value for key, value in self.config_values.items()}
            
            # Reload configuration
            self.load_configuration()
            
            # Check for changes
            new_config = {key: value.value for key, value in self.config_values.items()}
            
            changed_keys = []
            for key in set(old_config.keys()) | set(new_config.keys()):
                if old_config.get(key) != new_config.get(key):
                    changed_keys.append(key)
            
            if changed_keys:
                self.logger.info(f"Configuration reloaded with {len(changed_keys)} changes")
                
                # Notify callbacks for changed keys
                for key in changed_keys:
                    self._notify_change_callbacks(key, new_config.get(key))
            else:
                self.logger.info("Configuration reloaded with no changes")
                
        except Exception as e:
            self.logger.error(f"Failed to reload configuration: {str(e)}")
            raise
    
    def _start_file_watcher(self):
        """Start file system watcher for hot reload"""
        if not self.config_files:
            return
        
        try:
            self.file_watcher = ConfigFileWatcher(self, self.config_files)
            self.observer = watchdog.observers.Observer()
            
            # Watch configuration directory
            self.observer.schedule(self.file_watcher, str(self.config_dir), recursive=False)
            self.observer.start()
            
            self.logger.info("Configuration file watcher started")
            
        except Exception as e:
            self.logger.error(f"Failed to start file watcher: {str(e)}")
    
    def add_change_callback(self, callback: Callable[[str, Any], None]):
        """Add callback for configuration changes"""
        self.change_callbacks.append(callback)
    
    def _notify_change_callbacks(self, key: str, value: Any):
        """Notify all change callbacks"""
        for callback in self.change_callbacks:
            try:
                callback(key, value)
            except Exception as e:
                self.logger.error(f"Configuration change callback failed: {str(e)}")
    
    def get_all_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Get all configuration values"""
        with self.lock:
            config = {}
            
            for key, config_value in self.config_values.items():
                if config_value.is_secret and not include_secrets:
                    config[key] = "<hidden>"
                else:
                    config[key] = self.get(key)
            
            return config
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration metadata and statistics"""
        with self.lock:
            source_counts = {}
            secret_count = 0
            
            for config_value in self.config_values.values():
                source = config_value.source.value
                source_counts[source] = source_counts.get(source, 0) + 1
                
                if config_value.is_secret:
                    secret_count += 1
            
            return {
                'total_config_values': len(self.config_values),
                'secret_values': secret_count,
                'sources': source_counts,
                'config_files': self.config_files,
                'hot_reload_enabled': self.enable_hot_reload,
                'validation_schemas': len(self.validator.schemas)
            }
    
    def shutdown(self):
        """Shutdown configuration manager"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
        
        self.logger.info("ConfigManager shutdown complete")