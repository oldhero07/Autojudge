"""
Enhanced Model Manager for ML Web App Enhancements

This module provides advanced model management capabilities including:
- Multiple ML model support (Random Forest, SVM, Gradient Boosting)
- Model performance tracking and selection
- Fallback mechanisms and error recovery
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, mean_squared_error, r2_score
import scipy.sparse
from config.settings import get_settings

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """Metadata for ML models"""
    name: str
    algorithm: str
    version: str
    training_date: datetime
    performance_metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    hyperparameters: Dict[str, Any] = None

@dataclass
class PredictionResult:
    """Result of a model prediction"""
    class_prediction: str
    score_prediction: float
    confidence: float
    model_used: str
    cached: bool = False
    processing_time_ms: float = 0.0
    correlation_id: Optional[str] = None

class ModelManager:
    """
    Enhanced model manager supporting multiple ML algorithms with automatic selection
    """
    
    def __init__(self, config=None):
        """Initialize the ModelManager"""
        self.config = config or get_settings()
        self.models = {}
        self.performance_metrics = {}
        self.fallback_chain = []
        self.tfidf_vectorizer = None
        self.feature_scaler = None
        self.model_metadata = {}
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.prediction_history = []
        self.model_usage_stats = {}
        
        # Initialize model configurations
        self._initialize_model_configs()
    
    def _initialize_model_configs(self):
        """Initialize model configurations"""
        self.model_configs = {
            'logistic_regression': {
                'classifier': LogisticRegression,
                'regressor': None,  # Use RandomForest for regression
                'params': {
                    'random_state': 42,
                    'max_iter': 2000,
                    'class_weight': 'balanced',
                    'C': 1.0,
                    'solver': 'lbfgs'
                }
            },
            'random_forest': {
                'classifier': RandomForestClassifier,
                'regressor': RandomForestRegressor,
                'params': {
                    'n_estimators': 250,
                    'max_depth': 25,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'svm': {
                'classifier': SVC,
                'regressor': SVR,
                'params': {
                    'random_state': 42,
                    'class_weight': 'balanced',
                    'probability': True,  # Enable probability estimates
                    'C': 1.0,
                    'kernel': 'rbf'
                }
            },
            'gradient_boosting': {
                'classifier': GradientBoostingClassifier,
                'regressor': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
            }
        }
    
    def load_models(self, X_train, y_train_class, y_train_score) -> Dict[str, Pipeline]:
        """
        Load and train all model types on the same dataset
        
        Args:
            X_train: Training features
            y_train_class: Training classification targets
            y_train_score: Training regression targets
            
        Returns:
            Dictionary of model_name -> trained_pipeline
        """
        self.logger.info("Loading and training multiple ML models...")
        
        trained_models = {}
        
        for model_name, config in self.model_configs.items():
            try:
                self.logger.info(f"Training {model_name} models...")
                
                # Train classifier
                if config['classifier']:
                    classifier_params = config['params'].copy()
                    # Remove regression-specific params for classifiers
                    if model_name == 'svm':
                        classifier_params.pop('epsilon', None)
                    
                    classifier = config['classifier'](**classifier_params)
                    classifier.fit(X_train, y_train_class)
                    
                    trained_models[f"{model_name}_classifier"] = classifier
                    self.logger.info(f"✓ {model_name} classifier trained")
                
                # Train regressor
                if config['regressor']:
                    regressor_params = config['params'].copy()
                    # Remove classification-specific params for regressors
                    regressor_params.pop('class_weight', None)
                    regressor_params.pop('probability', None)
                    
                    # Add regression-specific params for SVM
                    if model_name == 'svm':
                        regressor_params['epsilon'] = 0.1
                    
                    regressor = config['regressor'](**regressor_params)
                    regressor.fit(X_train, y_train_score)
                    
                    trained_models[f"{model_name}_regressor"] = regressor
                    self.logger.info(f"✓ {model_name} regressor trained")
                
                # Use Random Forest regressor for Logistic Regression
                elif model_name == 'logistic_regression':
                    rf_regressor = RandomForestRegressor(
                        n_estimators=200,
                        max_depth=20,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    )
                    rf_regressor.fit(X_train, y_train_score)
                    trained_models[f"{model_name}_regressor"] = rf_regressor
                    self.logger.info(f"✓ {model_name} regressor (Random Forest) trained")
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        self.models = trained_models
        self.logger.info(f"Successfully trained {len(trained_models)} models")
        
        return trained_models
    
    def evaluate_models(self, X_test, y_test_class, y_test_score) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models and track performance metrics
        
        Args:
            X_test: Test features
            y_test_class: Test classification targets
            y_test_score: Test regression targets
            
        Returns:
            Dictionary of model performance metrics
        """
        self.logger.info("Evaluating model performance...")
        
        performance_results = {}
        
        for model_name, model in self.models.items():
            try:
                if 'classifier' in model_name:
                    # Evaluate classification model
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test_class, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_test_class, y_pred, average='weighted', zero_division=0
                    )
                    
                    # Get prediction probabilities for confidence
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X_test)
                        confidence = np.mean(np.max(probabilities, axis=1))
                    else:
                        confidence = 0.0
                    
                    performance_results[model_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'confidence': confidence
                    }
                    
                elif 'regressor' in model_name:
                    # Evaluate regression model
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_test_score, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test_score, y_pred))
                    r2 = r2_score(y_test_score, y_pred)
                    
                    performance_results[model_name] = {
                        'mae': mae,
                        'rmse': rmse,
                        'r2_score': r2
                    }
                
                self.logger.info(f"✓ Evaluated {model_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                continue
        
        self.performance_metrics = performance_results
        self._update_fallback_chain()
        
        return performance_results
    
    def _update_fallback_chain(self):
        """Update the fallback chain based on model performance"""
        # Sort classifiers by accuracy
        classifiers = {k: v for k, v in self.performance_metrics.items() if 'classifier' in k}
        sorted_classifiers = sorted(
            classifiers.items(),
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        )
        
        # Sort regressors by R² score
        regressors = {k: v for k, v in self.performance_metrics.items() if 'regressor' in k}
        sorted_regressors = sorted(
            regressors.items(),
            key=lambda x: x[1].get('r2_score', -1),
            reverse=True
        )
        
        self.fallback_chain = {
            'classifiers': [name for name, _ in sorted_classifiers],
            'regressors': [name for name, _ in sorted_regressors]
        }
        
        self.logger.info(f"Updated fallback chain - Classifiers: {self.fallback_chain['classifiers']}")
        self.logger.info(f"Updated fallback chain - Regressors: {self.fallback_chain['regressors']}")
    
    def select_best_model(self, task_type: str, confidence_threshold: float = 0.8) -> str:
        """
        Select the best performing model based on historical performance
        
        Args:
            task_type: 'classification' or 'regression'
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Best performing model name
        """
        if task_type == 'classification':
            candidates = self.fallback_chain.get('classifiers', [])
            metric_key = 'accuracy'
        else:
            candidates = self.fallback_chain.get('regressors', [])
            metric_key = 'r2_score'
        
        if not candidates:
            raise ValueError(f"No {task_type} models available")
        
        # Select best model based on performance
        best_model = candidates[0]  # Already sorted by performance
        
        # Check if confidence threshold is met
        if task_type == 'classification':
            model_confidence = self.performance_metrics.get(best_model, {}).get('confidence', 0)
            if model_confidence < confidence_threshold and len(candidates) > 1:
                self.logger.warning(f"Best model {best_model} confidence {model_confidence:.3f} below threshold {confidence_threshold}")
                # Could trigger ensemble mode here
        
        return best_model
    
    def predict_with_fallback(self, text: str, use_ensemble: bool = False) -> PredictionResult:
        """
        Make prediction with automatic fallback on failure
        
        Args:
            text: Input text for prediction
            use_ensemble: Whether to use ensemble prediction
            
        Returns:
            PredictionResult with prediction details
        """
        start_time = datetime.now()
        
        try:
            if use_ensemble:
                return self._predict_ensemble(text)
            else:
                return self._predict_single_model(text)
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            # Return fallback prediction
            return PredictionResult(
                class_prediction="medium",
                score_prediction=5.0,
                confidence=0.0,
                model_used="fallback",
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    def _predict_single_model(self, text: str) -> PredictionResult:
        """Make prediction using single best model"""
        start_time = datetime.now()
        
        # Prepare features (assuming existing feature extraction is available)
        X_features = self._extract_features(text)
        
        # Select best models
        best_classifier = self.select_best_model('classification')
        best_regressor = self.select_best_model('regression')
        
        # Make predictions
        classifier = self.models[best_classifier]
        regressor = self.models[best_regressor]
        
        class_pred = classifier.predict(X_features)[0]
        score_pred = regressor.predict(X_features)[0]
        
        # Get confidence
        if hasattr(classifier, 'predict_proba'):
            probabilities = classifier.predict_proba(X_features)[0]
            confidence = max(probabilities)
        else:
            confidence = 0.5
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Track usage
        self._track_model_usage(best_classifier, best_regressor)
        
        return PredictionResult(
            class_prediction=class_pred,
            score_prediction=float(score_pred),
            confidence=float(confidence),
            model_used=f"{best_classifier}+{best_regressor}",
            processing_time_ms=processing_time
        )
    
    def _predict_ensemble(self, text: str) -> PredictionResult:
        """Make prediction using ensemble of models"""
        start_time = datetime.now()
        
        # Prepare features
        X_features = self._extract_features(text)
        
        # Get predictions from all available models
        class_predictions = []
        score_predictions = []
        confidences = []
        
        for model_name, model in self.models.items():
            try:
                if 'classifier' in model_name:
                    pred = model.predict(X_features)[0]
                    class_predictions.append(pred)
                    
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(X_features)[0]
                        confidences.append(max(prob))
                    else:
                        confidences.append(0.5)
                        
                elif 'regressor' in model_name:
                    pred = model.predict(X_features)[0]
                    score_predictions.append(pred)
                    
            except Exception as e:
                self.logger.warning(f"Model {model_name} failed in ensemble: {str(e)}")
                continue
        
        # Ensemble predictions
        if class_predictions:
            # Majority vote for classification
            class_pred = max(set(class_predictions), key=class_predictions.count)
        else:
            class_pred = "medium"
        
        if score_predictions:
            # Average for regression
            score_pred = np.mean(score_predictions)
        else:
            score_pred = 5.0
        
        if confidences:
            confidence = np.mean(confidences)
        else:
            confidence = 0.0
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PredictionResult(
            class_prediction=class_pred,
            score_prediction=float(score_pred),
            confidence=float(confidence),
            model_used="ensemble",
            processing_time_ms=processing_time
        )
    
    def _extract_features(self, text: str):
        """Extract features from text (placeholder - integrate with existing feature extraction)"""
        # This should integrate with the existing feature extraction from app.py
        # For now, return a dummy feature vector
        if self.tfidf_vectorizer is None:
            # Return dummy features for testing
            return np.array([[1.0] * 100])  # Dummy feature vector
        
        # Use existing TF-IDF vectorizer
        X_tfidf = self.tfidf_vectorizer.transform([text])
        return X_tfidf
    
    def _track_model_usage(self, classifier_name: str, regressor_name: str):
        """Track model usage statistics"""
        if classifier_name not in self.model_usage_stats:
            self.model_usage_stats[classifier_name] = 0
        if regressor_name not in self.model_usage_stats:
            self.model_usage_stats[regressor_name] = 0
            
        self.model_usage_stats[classifier_name] += 1
        self.model_usage_stats[regressor_name] += 1
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive model performance summary"""
        return {
            'performance_metrics': self.performance_metrics,
            'fallback_chain': self.fallback_chain,
            'model_usage_stats': self.model_usage_stats,
            'available_models': list(self.models.keys()),
            'total_predictions': len(self.prediction_history)
        }
    
    def save_models(self, filepath: str):
        """Save trained models to disk"""
        try:
            model_data = {
                'models': self.models,
                'performance_metrics': self.performance_metrics,
                'fallback_chain': self.fallback_chain,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'feature_scaler': self.feature_scaler,
                'model_metadata': self.model_metadata
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Models saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {str(e)}")
            return False
    
    def load_models_from_disk(self, filepath: str):
        """Load trained models from disk"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data.get('models', {})
            self.performance_metrics = model_data.get('performance_metrics', {})
            self.fallback_chain = model_data.get('fallback_chain', [])
            self.tfidf_vectorizer = model_data.get('tfidf_vectorizer')
            self.feature_scaler = model_data.get('feature_scaler')
            self.model_metadata = model_data.get('model_metadata', {})
            
            self.logger.info(f"Models loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {str(e)}")
            return False