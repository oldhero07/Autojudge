"""
Unit tests for ModelEvaluator class and related functionality.

These tests validate specific functionality of the evaluation system
with known inputs and expected outputs.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
import scipy.sparse

from app import (
    ModelEvaluator, ClassificationMetrics, RegressionMetrics, EvaluationReport,
    extract_custom_features, PredictionService
)


class TestModelEvaluator:
    """Unit tests for ModelEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator(test_size=0.2, random_state=42)
        
        # Create sample data
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 10
        
        # Create synthetic feature matrix
        self.X = scipy.sparse.random(self.n_samples, self.n_features, density=0.3, random_state=42)
        
        # Create synthetic targets
        self.y_class = pd.Series(np.random.choice(['easy', 'medium', 'hard'], self.n_samples))
        self.y_score = pd.Series(np.random.uniform(1.0, 10.0, self.n_samples))
    
    def test_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(test_size=0.3, random_state=123)
        
        assert evaluator.test_size == 0.3
        assert evaluator.random_state == 123
        assert evaluator.X_train is None
        assert evaluator.X_test is None
        assert evaluator.min_accuracy == 0.6
        assert evaluator.max_mae == 2.0
        assert evaluator.max_rmse == 2.5
    
    def test_train_test_split(self):
        """Test train/test split functionality."""
        X_train, X_test, y_train_class, y_test_class, y_train_score, y_test_score = \
            self.evaluator.perform_train_test_split(self.X, self.y_class, self.y_score)
        
        # Check split sizes
        expected_test_size = int(self.n_samples * 0.2)
        expected_train_size = self.n_samples - expected_test_size
        
        assert X_train.shape[0] == expected_train_size
        assert X_test.shape[0] == expected_test_size
        assert len(y_train_class) == expected_train_size
        assert len(y_test_class) == expected_test_size
        assert len(y_train_score) == expected_train_size
        assert len(y_test_score) == expected_test_size
        
        # Check that features are preserved
        assert X_train.shape[1] == self.n_features
        assert X_test.shape[1] == self.n_features
        
        # Check that split info is stored
        assert self.evaluator.split_info['train_size'] == expected_train_size
        assert self.evaluator.split_info['test_size'] == expected_test_size
        assert self.evaluator.split_info['feature_count'] == self.n_features
    
    def test_classification_evaluation(self):
        """Test classification model evaluation."""
        # Perform split first
        self.evaluator.perform_train_test_split(self.X, self.y_class, self.y_score)
        
        # Train a simple model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(self.evaluator.X_train, self.evaluator.y_train_class)
        
        # Evaluate
        metrics = self.evaluator.evaluate_classification(model)
        
        # Check metric types and bounds
        assert isinstance(metrics, ClassificationMetrics)
        assert 0.0 <= metrics.accuracy <= 1.0
        assert isinstance(metrics.confusion_matrix, np.ndarray)
        assert isinstance(metrics.classification_report, dict)
        
        # Check confusion matrix shape
        unique_classes = len(self.evaluator.y_test_class.unique())
        assert metrics.confusion_matrix.shape == (unique_classes, unique_classes)
        
        # Check that confusion matrix sums to test set size
        assert metrics.confusion_matrix.sum() == len(self.evaluator.y_test_class)
    
    def test_regression_evaluation(self):
        """Test regression model evaluation."""
        # Perform split first
        self.evaluator.perform_train_test_split(self.X, self.y_class, self.y_score)
        
        # Train a simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(self.evaluator.X_train, self.evaluator.y_train_score)
        
        # Evaluate
        metrics = self.evaluator.evaluate_regression(model)
        
        # Check metric types and bounds
        assert isinstance(metrics, RegressionMetrics)
        assert metrics.mae >= 0.0
        assert metrics.rmse >= 0.0
        assert metrics.rmse >= metrics.mae  # RMSE should be >= MAE
        assert isinstance(metrics.r2_score, float)
    
    def test_performance_threshold_validation(self):
        """Test performance threshold validation."""
        # Test with good metrics
        good_classification = ClassificationMetrics(
            accuracy=0.8,
            confusion_matrix=np.array([[10, 2], [1, 7]]),
            classification_report={}
        )
        good_regression = RegressionMetrics(mae=1.0, rmse=1.2, r2_score=0.8)
        
        validation = self.evaluator.validate_performance_thresholds(
            good_classification, good_regression
        )
        
        assert validation['accuracy_acceptable'] is True
        assert validation['mae_acceptable'] is True
        assert validation['rmse_acceptable'] is True
        assert validation['overall_acceptable'] is True
        
        # Test with poor metrics
        poor_classification = ClassificationMetrics(
            accuracy=0.3,
            confusion_matrix=np.array([[5, 10], [8, 2]]),
            classification_report={}
        )
        poor_regression = RegressionMetrics(mae=3.0, rmse=4.0, r2_score=0.1)
        
        validation = self.evaluator.validate_performance_thresholds(
            poor_classification, poor_regression
        )
        
        assert validation['accuracy_acceptable'] is False
        assert validation['mae_acceptable'] is False
        assert validation['rmse_acceptable'] is False
        assert validation['overall_acceptable'] is False
    
    def test_evaluation_report_generation(self):
        """Test comprehensive evaluation report generation."""
        # Set up split info
        self.evaluator.split_info = {
            'train_size': 80,
            'test_size': 20,
            'feature_count': 100,
            'train_class_distribution': {'easy': 20, 'medium': 30, 'hard': 30}
        }
        
        # Create mock metrics
        classification_metrics = ClassificationMetrics(
            accuracy=0.75,
            confusion_matrix=np.array([[8, 2], [1, 9]]),
            classification_report={'macro avg': {'f1-score': 0.74}}
        )
        regression_metrics = RegressionMetrics(mae=1.5, rmse=1.8, r2_score=0.65)
        
        # Generate report
        report = self.evaluator.generate_evaluation_report(
            classification_metrics, regression_metrics
        )
        
        # Check report structure
        assert isinstance(report, EvaluationReport)
        assert report.classification_metrics == classification_metrics
        assert report.regression_metrics == regression_metrics
        
        # Check dataset info
        assert report.dataset_info['total_samples'] == 100
        assert report.dataset_info['train_samples'] == 80
        assert report.dataset_info['test_samples'] == 20
        assert report.dataset_info['feature_count'] == 100
        
        # Check model info
        assert report.model_info['classification_model'] == 'LogisticRegression'
        assert report.model_info['regression_model'] == 'RandomForestRegressor'
        assert report.model_info['test_size'] == 0.2
        assert report.model_info['random_state'] == 42


class TestFeatureExtraction:
    """Unit tests for feature extraction functions."""
    
    def test_extract_custom_features_basic(self):
        """Test basic custom feature extraction."""
        text = "This is a simple algorithm problem with + and * symbols."
        
        text_len, math_count, keyword_count = extract_custom_features(text)
        
        assert text_len == len(text)
        assert math_count >= 2  # At least + and *
        assert keyword_count >= 1  # At least 'algorithm'
    
    def test_extract_custom_features_empty(self):
        """Test feature extraction with empty text."""
        text = ""
        
        text_len, math_count, keyword_count = extract_custom_features(text)
        
        assert text_len == 0
        assert math_count == 0
        assert keyword_count == 0
    
    def test_extract_custom_features_math_heavy(self):
        """Test feature extraction with many math symbols."""
        text = "Calculate: x + y - z * w / 2 = result >= threshold <= maximum"
        
        text_len, math_count, keyword_count = extract_custom_features(text)
        
        assert text_len == len(text)
        assert math_count >= 8  # Many math symbols
        assert keyword_count >= 1  # 'maximum' is a keyword
    
    def test_extract_custom_features_keyword_heavy(self):
        """Test feature extraction with many difficulty keywords."""
        text = "Use dynamic programming with graph traversal, binary search, and optimal sorting algorithm."
        
        text_len, math_count, keyword_count = extract_custom_features(text)
        
        assert text_len == len(text)
        assert keyword_count >= 6  # dynamic, graph, binary, search, optimal, algorithm


class TestPredictionService:
    """Unit tests for PredictionService class."""
    
    def test_combine_text_features_all_provided(self):
        """Test text combination with all fields provided."""
        description = "Problem description"
        input_desc = "Input format"
        output_desc = "Output format"
        
        combined = PredictionService.combine_text_features(description, input_desc, output_desc)
        
        assert "Problem description" in combined
        assert "Input format" in combined
        assert "Output format" in combined
        assert len(combined.strip()) > 0
    
    def test_combine_text_features_partial(self):
        """Test text combination with some fields empty."""
        description = "Problem description"
        input_desc = ""
        output_desc = "Output format"
        
        combined = PredictionService.combine_text_features(description, input_desc, output_desc)
        
        assert "Problem description" in combined
        assert "Output format" in combined
        assert len(combined.strip()) > 0
    
    def test_combine_text_features_all_empty(self):
        """Test text combination with all fields empty."""
        combined = PredictionService.combine_text_features("", "", "")
        
        assert combined.strip() == ""
    
    def test_combine_text_features_none_values(self):
        """Test text combination with None values."""
        combined = PredictionService.combine_text_features(None, "Input", None)
        
        assert "Input" in combined
        assert len(combined.strip()) > 0


class TestErrorHandling:
    """Unit tests for error handling scenarios."""
    
    def test_classification_evaluation_error_handling(self):
        """Test classification evaluation with invalid inputs."""
        evaluator = ModelEvaluator()
        
        # Test with None model (should return fallback metrics)
        metrics = evaluator.evaluate_classification(None, np.array([[1, 2]]), np.array([0]))
        
        assert isinstance(metrics, ClassificationMetrics)
        assert metrics.accuracy == 0.33  # Fallback accuracy for 3-class random guess
        assert metrics.confusion_matrix.shape == (3, 3)  # 3x3 fallback matrix
        assert 'fallback_reason' in metrics.classification_report
    
    def test_regression_evaluation_error_handling(self):
        """Test regression evaluation with invalid inputs."""
        evaluator = ModelEvaluator()
        
        # Test with None model (should return fallback metrics)
        metrics = evaluator.evaluate_regression(None, np.array([[1, 2]]), np.array([1.0]))
        
        assert isinstance(metrics, RegressionMetrics)
        assert metrics.mae == 5.0  # Fallback MAE value
        assert metrics.rmse == 6.0  # Fallback RMSE value
        assert metrics.r2_score == -1.0  # Fallback R² value
    
    def test_train_test_split_error_handling(self):
        """Test train/test split with edge cases."""
        evaluator = ModelEvaluator(test_size=0.1)  # Very small test set
        
        # Create minimal dataset
        X = scipy.sparse.random(10, 5, density=0.5, random_state=42)
        y_class = pd.Series(['easy'] * 5 + ['hard'] * 5)
        y_score = pd.Series([2.0] * 5 + [8.0] * 5)
        
        # Should handle small dataset gracefully
        try:
            X_train, X_test, y_train_class, y_test_class, y_train_score, y_test_score = \
                evaluator.perform_train_test_split(X, y_class, y_score)
            
            # Basic checks
            assert X_train.shape[0] + X_test.shape[0] == 10
            assert len(y_train_class) + len(y_test_class) == 10
            
        except Exception as e:
            # If stratification fails with small dataset, that's expected
            error_msg = str(e).lower()
            assert ("stratify" in error_msg or "sample" in error_msg or 
                    "test_size" in error_msg or "classes" in error_msg or
                    "insufficient" in error_msg or "fallback" in error_msg)


if __name__ == '__main__':
    # Run unit tests
    pytest.main([__file__, '-v'])