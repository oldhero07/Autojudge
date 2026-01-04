"""
Network and Database Fallback Mechanisms for ML Web App Enhancements

This module provides fallback mechanisms for network connectivity issues
and database connection failures, including local caching and data sources.
"""

import logging
import json
import os
import sqlite3
import pickle
import time
import threading
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class CachedPrediction:
    """Cached prediction data"""
    input_hash: str
    prediction_result: Dict[str, Any]
    timestamp: datetime
    model_name: str
    ttl_seconds: int = 3600

@dataclass
class NetworkStatus:
    """Network connectivity status"""
    is_connected: bool
    last_check: datetime
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None

class LocalDataStore:
    """
    Local SQLite-based data store for fallback scenarios
    """
    
    def __init__(self, db_path: str = "fallback_data.db"):
        """Initialize local data store"""
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.LocalDataStore")
        self.lock = threading.RLock()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Cached predictions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cached_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        input_hash TEXT UNIQUE NOT NULL,
                        prediction_result TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        ttl_seconds INTEGER NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Training data backup table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS training_data_backup (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        data_type TEXT NOT NULL,
                        data_content TEXT NOT NULL,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # System configuration backup
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS config_backup (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        config_key TEXT UNIQUE NOT NULL,
                        config_value TEXT NOT NULL,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Model metadata backup
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        model_data BLOB,
                        metadata TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                self.logger.info(f"Local database initialized at {self.db_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def store_prediction(self, cached_prediction: CachedPrediction):
        """Store a prediction in local cache"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO cached_predictions 
                        (input_hash, prediction_result, model_name, timestamp, ttl_seconds)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        cached_prediction.input_hash,
                        json.dumps(cached_prediction.prediction_result),
                        cached_prediction.model_name,
                        cached_prediction.timestamp.isoformat(),
                        cached_prediction.ttl_seconds
                    ))
                    
                    conn.commit()
                    self.logger.debug(f"Stored prediction for hash {cached_prediction.input_hash}")
                    
        except Exception as e:
            self.logger.error(f"Failed to store prediction: {str(e)}")
    
    def get_prediction(self, input_hash: str) -> Optional[CachedPrediction]:
        """Retrieve a prediction from local cache"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        SELECT input_hash, prediction_result, model_name, timestamp, ttl_seconds
                        FROM cached_predictions 
                        WHERE input_hash = ?
                    ''', (input_hash,))
                    
                    row = cursor.fetchone()
                    if row:
                        input_hash, result_json, model_name, timestamp_str, ttl_seconds = row
                        
                        # Check if cache entry is still valid
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if datetime.now() - timestamp < timedelta(seconds=ttl_seconds):
                            return CachedPrediction(
                                input_hash=input_hash,
                                prediction_result=json.loads(result_json),
                                timestamp=timestamp,
                                model_name=model_name,
                                ttl_seconds=ttl_seconds
                            )
                        else:
                            # Remove expired entry
                            self._remove_expired_predictions()
                    
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to get prediction: {str(e)}")
            return None
    
    def store_training_data(self, data_type: str, data_content: Any, metadata: Optional[Dict] = None):
        """Store training data backup"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Serialize data content
                    if isinstance(data_content, (dict, list)):
                        content_str = json.dumps(data_content)
                    else:
                        content_str = str(data_content)
                    
                    metadata_str = json.dumps(metadata) if metadata else None
                    
                    cursor.execute('''
                        INSERT INTO training_data_backup (data_type, data_content, metadata)
                        VALUES (?, ?, ?)
                    ''', (data_type, content_str, metadata_str))
                    
                    conn.commit()
                    self.logger.info(f"Stored training data backup: {data_type}")
                    
        except Exception as e:
            self.logger.error(f"Failed to store training data: {str(e)}")
    
    def get_training_data(self, data_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve training data backup"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        SELECT data_content, metadata, created_at
                        FROM training_data_backup 
                        WHERE data_type = ?
                        ORDER BY created_at DESC
                        LIMIT 1
                    ''', (data_type,))
                    
                    row = cursor.fetchone()
                    if row:
                        content_str, metadata_str, created_at = row
                        
                        try:
                            data_content = json.loads(content_str)
                        except json.JSONDecodeError:
                            data_content = content_str
                        
                        metadata = json.loads(metadata_str) if metadata_str else {}
                        
                        return {
                            'data_content': data_content,
                            'metadata': metadata,
                            'created_at': created_at
                        }
                    
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to get training data: {str(e)}")
            return None
    
    def _remove_expired_predictions(self):
        """Remove expired predictions from cache"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Remove entries where current time > timestamp + ttl_seconds
                cursor.execute('''
                    DELETE FROM cached_predictions 
                    WHERE datetime('now') > datetime(timestamp, '+' || ttl_seconds || ' seconds')
                ''')
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    self.logger.info(f"Removed {deleted_count} expired predictions")
                    
        except Exception as e:
            self.logger.error(f"Failed to remove expired predictions: {str(e)}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Count total predictions
                    cursor.execute('SELECT COUNT(*) FROM cached_predictions')
                    total_predictions = cursor.fetchone()[0]
                    
                    # Count valid (non-expired) predictions
                    cursor.execute('''
                        SELECT COUNT(*) FROM cached_predictions 
                        WHERE datetime('now') <= datetime(timestamp, '+' || ttl_seconds || ' seconds')
                    ''')
                    valid_predictions = cursor.fetchone()[0]
                    
                    # Count training data backups
                    cursor.execute('SELECT COUNT(*) FROM training_data_backup')
                    training_backups = cursor.fetchone()[0]
                    
                    # Get database size
                    db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                    
                    return {
                        'total_predictions': total_predictions,
                        'valid_predictions': valid_predictions,
                        'expired_predictions': total_predictions - valid_predictions,
                        'training_backups': training_backups,
                        'database_size_mb': round(db_size / (1024 * 1024), 2),
                        'database_path': self.db_path
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get cache statistics: {str(e)}")
            return {}

class NetworkConnectivityManager:
    """
    Manager for network connectivity monitoring and fallback
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize network connectivity manager"""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.NetworkConnectivityManager")
        
        # Configuration
        self.check_urls = self.config.get('check_urls', [
            'https://httpbin.org/status/200',
            'https://www.google.com',
            'https://www.github.com'
        ])
        self.timeout = self.config.get('timeout', 5)
        self.retry_attempts = self.config.get('retry_attempts', 3)
        self.check_interval = self.config.get('check_interval', 60)  # seconds
        
        # Status tracking
        self.network_status = NetworkStatus(
            is_connected=True,
            last_check=datetime.now()
        )
        
        # HTTP session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.retry_attempts,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.logger.info("NetworkConnectivityManager initialized")
    
    def check_connectivity(self) -> NetworkStatus:
        """Check network connectivity"""
        start_time = time.time()
        
        for url in self.check_urls:
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    response_time = (time.time() - start_time) * 1000
                    
                    self.network_status = NetworkStatus(
                        is_connected=True,
                        last_check=datetime.now(),
                        response_time_ms=response_time
                    )
                    
                    self.logger.debug(f"Network connectivity confirmed via {url}")
                    return self.network_status
                    
            except Exception as e:
                self.logger.warning(f"Connectivity check failed for {url}: {str(e)}")
                continue
        
        # All checks failed
        self.network_status = NetworkStatus(
            is_connected=False,
            last_check=datetime.now(),
            error_message="All connectivity checks failed"
        )
        
        self.logger.warning("Network connectivity lost")
        return self.network_status
    
    def is_connected(self) -> bool:
        """Check if network is currently connected"""
        # Check if last check was recent enough
        time_since_check = datetime.now() - self.network_status.last_check
        if time_since_check.total_seconds() > self.check_interval:
            self.check_connectivity()
        
        return self.network_status.is_connected
    
    def start_monitoring(self):
        """Start background connectivity monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("Network connectivity monitoring started")
    
    def stop_monitoring(self):
        """Stop background connectivity monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Network connectivity monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                self.check_connectivity()
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in connectivity monitoring: {str(e)}")
                time.sleep(60)  # Wait longer on error

class NetworkDatabaseFallbackManager:
    """
    Comprehensive fallback manager for network and database issues
    """
    
    def __init__(self, local_store_path: str = "fallback_data.db", config: Optional[Dict] = None):
        """Initialize fallback manager"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.local_store = LocalDataStore(local_store_path)
        self.network_manager = NetworkConnectivityManager(self.config.get('network', {}))
        
        # Fallback configuration
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour
        self.max_cache_size = self.config.get('max_cache_size', 10000)
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.network_fallbacks = 0
        self.database_fallbacks = 0
        
        self.logger.info("NetworkDatabaseFallbackManager initialized")
    
    def get_prediction_with_fallback(self, input_hash: str, 
                                   prediction_func: callable, 
                                   *args, **kwargs) -> Dict[str, Any]:
        """
        Get prediction with comprehensive fallback strategy
        
        Args:
            input_hash: Hash of input data
            prediction_func: Function to call for prediction
            *args, **kwargs: Arguments for prediction function
            
        Returns:
            Prediction result with fallback metadata
        """
        # Try to get from local cache first
        cached_prediction = self.local_store.get_prediction(input_hash)
        if cached_prediction:
            self.cache_hits += 1
            result = cached_prediction.prediction_result.copy()
            result['cached'] = True
            result['cache_source'] = 'local_store'
            result['cache_timestamp'] = cached_prediction.timestamp.isoformat()
            
            self.logger.debug(f"Serving prediction from local cache: {input_hash}")
            return result
        
        self.cache_misses += 1
        
        # Try to get fresh prediction
        try:
            # Check network connectivity if prediction requires network
            if not self.network_manager.is_connected():
                self.logger.warning("Network connectivity lost, using fallback")
                return self._handle_network_fallback(input_hash)
            
            # Attempt fresh prediction
            result = prediction_func(*args, **kwargs)
            
            # Cache the result for future fallback
            if result and isinstance(result, dict):
                cached_pred = CachedPrediction(
                    input_hash=input_hash,
                    prediction_result=result,
                    timestamp=datetime.now(),
                    model_name=result.get('model_used', 'unknown'),
                    ttl_seconds=self.cache_ttl
                )
                self.local_store.store_prediction(cached_pred)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction function failed: {str(e)}")
            return self._handle_prediction_failure(input_hash, e)
    
    def _handle_network_fallback(self, input_hash: str) -> Dict[str, Any]:
        """Handle network connectivity issues"""
        self.network_fallbacks += 1
        
        # Try to get any cached prediction (even if expired)
        try:
            with sqlite3.connect(self.local_store.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT prediction_result, model_name, timestamp
                    FROM cached_predictions 
                    WHERE input_hash = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                ''', (input_hash,))
                
                row = cursor.fetchone()
                if row:
                    result_json, model_name, timestamp_str = row
                    result = json.loads(result_json)
                    
                    result['cached'] = True
                    result['cache_source'] = 'network_fallback'
                    result['cache_timestamp'] = timestamp_str
                    result['fallback_reason'] = 'network_connectivity_lost'
                    
                    self.logger.info(f"Using stale cache for network fallback: {input_hash}")
                    return result
                    
        except Exception as e:
            self.logger.error(f"Failed to get stale cache: {str(e)}")
        
        # Return default response if no cache available
        return {
            'class': 'medium',
            'score': 5.0,
            'confidence': 0.0,
            'cached': False,
            'fallback_reason': 'network_connectivity_lost_no_cache',
            'error': 'Network connectivity lost and no cached data available'
        }
    
    def _handle_prediction_failure(self, input_hash: str, error: Exception) -> Dict[str, Any]:
        """Handle prediction function failures"""
        self.database_fallbacks += 1
        
        # Try local cache (including stale entries)
        cached_prediction = self.local_store.get_prediction(input_hash)
        if cached_prediction:
            result = cached_prediction.prediction_result.copy()
            result['cached'] = True
            result['cache_source'] = 'prediction_failure_fallback'
            result['fallback_reason'] = f'prediction_failed: {str(error)}'
            
            self.logger.info(f"Using cache for prediction failure fallback: {input_hash}")
            return result
        
        # Return error response
        return {
            'class': 'medium',
            'score': 5.0,
            'confidence': 0.0,
            'cached': False,
            'fallback_reason': 'prediction_failed_no_cache',
            'error': f'Prediction failed and no cached data available: {str(error)}'
        }
    
    def backup_training_data(self, training_data: Any, data_type: str = "default"):
        """Backup training data to local store"""
        try:
            metadata = {
                'backup_timestamp': datetime.now().isoformat(),
                'data_type': data_type,
                'data_size': len(training_data) if hasattr(training_data, '__len__') else 0
            }
            
            self.local_store.store_training_data(data_type, training_data, metadata)
            self.logger.info(f"Training data backed up: {data_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to backup training data: {str(e)}")
    
    def restore_training_data(self, data_type: str = "default") -> Optional[Any]:
        """Restore training data from local store"""
        try:
            backup_data = self.local_store.get_training_data(data_type)
            if backup_data:
                self.logger.info(f"Training data restored: {data_type}")
                return backup_data['data_content']
            else:
                self.logger.warning(f"No backup found for data type: {data_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to restore training data: {str(e)}")
            return None
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fallback statistics"""
        cache_stats = self.local_store.get_cache_statistics()
        network_status = self.network_manager.network_status
        
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_statistics': cache_stats,
            'cache_performance': {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate_percent': round(cache_hit_rate, 2),
                'total_requests': total_requests
            },
            'fallback_usage': {
                'network_fallbacks': self.network_fallbacks,
                'database_fallbacks': self.database_fallbacks
            },
            'network_status': {
                'is_connected': network_status.is_connected,
                'last_check': network_status.last_check.isoformat(),
                'response_time_ms': network_status.response_time_ms,
                'error_message': network_status.error_message
            }
        }
    
    def start_monitoring(self):
        """Start all background monitoring"""
        self.network_manager.start_monitoring()
        self.logger.info("Fallback monitoring started")
    
    def stop_monitoring(self):
        """Stop all background monitoring"""
        self.network_manager.stop_monitoring()
        self.logger.info("Fallback monitoring stopped")
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cache entries"""
        try:
            with sqlite3.connect(self.local_store.db_path) as conn:
                cursor = conn.cursor()
                
                # Remove entries older than max_age_hours
                cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
                
                cursor.execute('''
                    DELETE FROM cached_predictions 
                    WHERE datetime(timestamp) < ?
                ''', (cutoff_time.isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted_count} old cache entries")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup cache: {str(e)}")
            return 0