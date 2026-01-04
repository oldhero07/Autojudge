"""
Model Ensemble for ML Web App Enhancements

This module provides ensemble prediction capabilities with weighted voting
and adaptive weight updating based on model performance.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import json

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EnsemblePrediction:
    """Result of an ensemble prediction"""
    class_prediction: str
    score_prediction: float
    confidence: float
    individual_predictions: Dict[str, Any]
    weights_used: Dict[str, float]
    ensemble_method: str
    processing_time_ms: float = 0.0

class ModelEnsemble:
    """
    Advanced model ensemble with weighted voting and adaptive weight updating
    """
    
    def __init__(self, models: Dict[str, Any], initial_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the ModelEnsemble
        
        Args:
            models: Dictionary of model_name -> trained_model
            initial_weights: Optional initial weights for models
        """
        self.models = models
        self.logger = logging.getLogger(__name__)
        
        # Initialize weights
        if initial_weights:
            self.weights = initial_weights.copy()
        else:
            # Equal weights initially
            self.weights = {name: 1.0 / len(models) for name in models.keys()}
        
        # Normalize weights to sum to 1.0
        self._normalize_weights()
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.prediction_count = 0
        
        # Ensemble configuration
        self.min_weight = 0.01  # Minimum weight for any model
        self.weight_decay = 0.95  # Decay factor for poor performing models
        self.weight_boost = 1.05  # Boost factor for well performing models
        
        self.logger.info(f"Initialized ensemble with {len(models)} models")
        self.logger.info(f"Initial weights: {self.weights}")
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {name: weight / total_weight for name, weight in self.weights.items()}
        else:
            # Reset to equal weights if all weights are zero
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
    
    def predict_ensemble(self, X_features, method: str = 'weighted_voting') -> EnsemblePrediction:
        """
        Make ensemble prediction using specified method
        
        Args:
            X_features: Input features for prediction
            method: Ensemble method ('weighted_voting', 'majority_vote', 'average')
            
        Returns:
            EnsemblePrediction with ensemble results
        """
        start_time = datetime.now()
        
        # Collect predictions from all models
        individual_predictions = {}
        classification_preds = []
        regression_preds = []
        confidences = []
        
        for model_name, model in self.models.items():
            try:
                if 'classifier' in model_name:
                    # Classification prediction
                    class_pred = model.predict(X_features)[0]
                    classification_preds.append((model_name, class_pred))
                    
                    # Get confidence if available
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X_features)[0]
                        confidence = max(probabilities)
                        confidences.append((model_name, confidence))
                        
                        individual_predictions[model_name] = {
                            'class': class_pred,
                            'confidence': confidence,
                            'probabilities': probabilities.tolist()
                        }
                    else:
                        individual_predictions[model_name] = {
                            'class': class_pred,
                            'confidence': 0.5
                        }
                
                elif 'regressor' in model_name:
                    # Regression prediction
                    score_pred = model.predict(X_features)[0]
                    regression_preds.append((model_name, score_pred))
                    
                    individual_predictions[model_name] = {
                        'score': float(score_pred)
                    }
                
            except Exception as e:
                self.logger.warning(f"Model {model_name} failed in ensemble: {str(e)}")
                continue
        
        # Ensemble predictions based on method
        if method == 'weighted_voting':
            ensemble_class, ensemble_score, ensemble_confidence = self._weighted_voting(
                classification_preds, regression_preds, confidences
            )
        elif method == 'majority_vote':
            ensemble_class, ensemble_score, ensemble_confidence = self._majority_voting(
                classification_preds, regression_preds, confidences
            )
        elif method == 'average':
            ensemble_class, ensemble_score, ensemble_confidence = self._simple_average(
                classification_preds, regression_preds, confidences
            )
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.prediction_count += 1
        
        return EnsemblePrediction(
            class_prediction=ensemble_class,
            score_prediction=ensemble_score,
            confidence=ensemble_confidence,
            individual_predictions=individual_predictions,
            weights_used=self.weights.copy(),
            ensemble_method=method,
            processing_time_ms=processing_time
        )
    
    def _weighted_voting(self, classification_preds: List[Tuple[str, str]], 
                        regression_preds: List[Tuple[str, float]], 
                        confidences: List[Tuple[str, float]]) -> Tuple[str, float, float]:
        """Perform weighted voting ensemble"""
        
        # Weighted classification voting
        class_votes = defaultdict(float)
        total_class_weight = 0.0
        
        for model_name, class_pred in classification_preds:
            weight = self.weights.get(model_name, 0.0)
            class_votes[class_pred] += weight
            total_class_weight += weight
        
        # Select class with highest weighted vote
        if class_votes:
            ensemble_class = max(class_votes.items(), key=lambda x: x[1])[0]
        else:
            ensemble_class = "medium"  # Default fallback
        
        # Weighted regression averaging
        weighted_score_sum = 0.0
        total_reg_weight = 0.0
        
        for model_name, score_pred in regression_preds:
            weight = self.weights.get(model_name, 0.0)
            weighted_score_sum += score_pred * weight
            total_reg_weight += weight
        
        if total_reg_weight > 0:
            ensemble_score = weighted_score_sum / total_reg_weight
        else:
            ensemble_score = 5.0  # Default fallback
        
        # Weighted confidence averaging
        weighted_conf_sum = 0.0
        total_conf_weight = 0.0
        
        for model_name, confidence in confidences:
            weight = self.weights.get(model_name, 0.0)
            weighted_conf_sum += confidence * weight
            total_conf_weight += weight
        
        if total_conf_weight > 0:
            ensemble_confidence = weighted_conf_sum / total_conf_weight
        else:
            ensemble_confidence = 0.5
        
        return ensemble_class, float(ensemble_score), float(ensemble_confidence)
    
    def _majority_voting(self, classification_preds: List[Tuple[str, str]], 
                        regression_preds: List[Tuple[str, float]], 
                        confidences: List[Tuple[str, float]]) -> Tuple[str, float, float]:
        """Perform majority voting ensemble"""
        
        # Majority vote for classification
        class_counts = defaultdict(int)
        for _, class_pred in classification_preds:
            class_counts[class_pred] += 1
        
        if class_counts:
            ensemble_class = max(class_counts.items(), key=lambda x: x[1])[0]
        else:
            ensemble_class = "medium"
        
        # Simple average for regression
        if regression_preds:
            scores = [score for _, score in regression_preds]
            ensemble_score = np.mean(scores)
        else:
            ensemble_score = 5.0
        
        # Simple average for confidence
        if confidences:
            conf_values = [conf for _, conf in confidences]
            ensemble_confidence = np.mean(conf_values)
        else:
            ensemble_confidence = 0.5
        
        return ensemble_class, float(ensemble_score), float(ensemble_confidence)
    
    def _simple_average(self, classification_preds: List[Tuple[str, str]], 
                       regression_preds: List[Tuple[str, float]], 
                       confidences: List[Tuple[str, float]]) -> Tuple[str, float, float]:
        """Perform simple averaging ensemble (same as majority voting for this implementation)"""
        return self._majority_voting(classification_preds, regression_preds, confidences)
    
    def update_weights(self, performance_data: Dict[str, float], metric_type: str = 'accuracy'):
        """
        Update model weights based on recent performance
        
        Args:
            performance_data: Dictionary of model_name -> performance_metric
            metric_type: Type of metric ('accuracy', 'f1_score', 'r2_score', etc.)
        """
        self.logger.info(f"Updating weights based on {metric_type} performance")
        
        # Store performance history
        for model_name, performance in performance_data.items():
            self.performance_history[model_name].append({
                'metric': metric_type,
                'value': performance,
                'timestamp': datetime.now()
            })
        
        # Update weights based on performance
        for model_name in self.weights.keys():
            if model_name in performance_data:
                performance = performance_data[model_name]
                current_weight = self.weights[model_name]
                
                # Determine if performance is good or bad
                # For accuracy, f1_score, r2_score: higher is better
                # For mae, rmse: lower is better
                if metric_type in ['accuracy', 'f1_score', 'r2_score', 'precision', 'recall']:
                    if performance > 0.7:  # Good performance threshold
                        new_weight = current_weight * self.weight_boost
                    elif performance < 0.5:  # Poor performance threshold
                        new_weight = current_weight * self.weight_decay
                    else:
                        new_weight = current_weight  # No change
                else:  # For error metrics (lower is better)
                    if performance < 1.0:  # Good performance (low error)
                        new_weight = current_weight * self.weight_boost
                    elif performance > 2.0:  # Poor performance (high error)
                        new_weight = current_weight * self.weight_decay
                    else:
                        new_weight = current_weight  # No change
                
                # Apply minimum weight constraint
                self.weights[model_name] = max(new_weight, self.min_weight)
        
        # Normalize weights
        self._normalize_weights()
        
        self.logger.info(f"Updated weights: {self.weights}")
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get comprehensive ensemble statistics"""
        
        # Calculate average performance for each model
        avg_performance = {}
        for model_name, history in self.performance_history.items():
            if history:
                values = [entry['value'] for entry in history]
                avg_performance[model_name] = {
                    'average': np.mean(values),
                    'std': np.std(values),
                    'count': len(values),
                    'latest': values[-1] if values else None
                }
        
        return {
            'current_weights': self.weights,
            'model_count': len(self.models),
            'prediction_count': self.prediction_count,
            'performance_history_length': {name: len(history) for name, history in self.performance_history.items()},
            'average_performance': avg_performance,
            'weight_statistics': {
                'min_weight': min(self.weights.values()) if self.weights else 0,
                'max_weight': max(self.weights.values()) if self.weights else 0,
                'weight_std': np.std(list(self.weights.values())) if self.weights else 0
            }
        }
    
    def reset_weights(self, equal_weights: bool = True):
        """
        Reset model weights
        
        Args:
            equal_weights: If True, set equal weights; if False, use performance-based weights
        """
        if equal_weights:
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        else:
            # Use recent performance to set weights
            if self.performance_history:
                performance_data = {}
                for model_name, history in self.performance_history.items():
                    if history:
                        # Use most recent performance
                        performance_data[model_name] = history[-1]['value']
                
                if performance_data:
                    self.update_weights(performance_data, 'accuracy')  # Assume accuracy metric
        
        self.logger.info(f"Reset weights: {self.weights}")
    
    def add_model(self, model_name: str, model: Any, initial_weight: Optional[float] = None):
        """
        Add a new model to the ensemble
        
        Args:
            model_name: Name of the new model
            model: Trained model object
            initial_weight: Optional initial weight (if None, uses equal weight)
        """
        self.models[model_name] = model
        
        if initial_weight is not None:
            self.weights[model_name] = initial_weight
        else:
            # Add with equal weight and renormalize
            self.weights[model_name] = 1.0 / len(self.models)
        
        self._normalize_weights()
        self.logger.info(f"Added model {model_name} to ensemble with weight {self.weights[model_name]:.3f}")
    
    def remove_model(self, model_name: str):
        """
        Remove a model from the ensemble
        
        Args:
            model_name: Name of the model to remove
        """
        if model_name in self.models:
            del self.models[model_name]
            del self.weights[model_name]
            
            # Remove performance history
            if model_name in self.performance_history:
                del self.performance_history[model_name]
            
            self._normalize_weights()
            self.logger.info(f"Removed model {model_name} from ensemble")
        else:
            self.logger.warning(f"Model {model_name} not found in ensemble")
    
    def save_ensemble_state(self, filepath: str):
        """Save ensemble state to file"""
        try:
            state = {
                'weights': self.weights,
                'performance_history': dict(self.performance_history),
                'prediction_count': self.prediction_count,
                'config': {
                    'min_weight': self.min_weight,
                    'weight_decay': self.weight_decay,
                    'weight_boost': self.weight_boost
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            self.logger.info(f"Ensemble state saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save ensemble state: {str(e)}")
            return False
    
    def load_ensemble_state(self, filepath: str):
        """Load ensemble state from file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.weights = state.get('weights', {})
            self.performance_history = defaultdict(list, state.get('performance_history', {}))
            self.prediction_count = state.get('prediction_count', 0)
            
            config = state.get('config', {})
            self.min_weight = config.get('min_weight', 0.01)
            self.weight_decay = config.get('weight_decay', 0.95)
            self.weight_boost = config.get('weight_boost', 1.05)
            
            self.logger.info(f"Ensemble state loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load ensemble state: {str(e)}")
            return False