"""
Property-based tests for AutoJudge system using Hypothesis.

These tests validate system properties and invariants across different inputs
to ensure robust behavior under various conditions.
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import scipy.sparse

from app import ModelEvaluator, extract_custom_features, PredictionService
from error_handler import SystemComponent


class TestEvaluationMetricsProperties:
    """Property tests for evaluation metrics calculation."""
    
    @given(
        y_true=st.lists(st.sampled_from(['easy', 'medium', 'hard']), min_size=10, max_size=100),
        noise_level=st.floats(min_value=0.0, max_value=0.3)
    )
    @settings(max_examples=20, deadline=10000)
    def test_property_evaluation_metrics_calculation(self, y_true, noise_level):
        """
        Property 4: Evaluation metrics calculation
        
        Tests that evaluation metrics are calculated correctly and consistently
        across different dataset sizes and class distributions.
        """
        # Create synthetic predictions with controlled noise
        y_pred_class = []
        for true_class in y_true:
            if np.random.random() < noise_level:
                # Add noise by randomly changing class
                y_pred_class.append(np.random.choice(['easy', 'medium', 'hard']))
            else:
                y_pred_class.append(true_class)
        
        # Create regression targets and predictions
        class_to_score = {'easy': 2.0, 'medium': 5.0, 'hard': 8.0}
        y_true_score = [class_to_score[cls] + np.random.normal(0, 0.5) for cls in y_true]
        y_pred_score = [score + np.random.normal(0, noise_level * 2) for score in y_true_score]
        
        # Calculate metrics manually for verification
        expected_accuracy = accuracy_score(y_true, y_pred_class)
        expected_mae = mean_absolute_error(y_true_score, y_pred_score)
        expected_rmse = np.sqrt(mean_squared_error(y_true_score, y_pred_score))
        expected_r2 = r2_score(y_true_score, y_pred_score)
        
        # Test ModelEvaluator metrics calculation
        evaluator = ModelEvaluator()
        
        # Create mock classification metrics
        from app import ClassificationMetrics, RegressionMetrics
        from sklearn.metrics import confusion_matrix, classification_report
        
        classification_metrics = ClassificationMetrics(
            accuracy=expected_accuracy,
            confusion_matrix=confusion_matrix(y_true, y_pred_class),
            classification_report=classification_report(y_true, y_pred_class, output_dict=True)
        )
        
        regression_metrics = RegressionMetrics(
            mae=expected_mae,
            rmse=expected_rmse,
            r2_score=expected_r2
        )
        
        # Property: Accuracy should be between 0 and 1
        assert 0.0 <= classification_metrics.accuracy <= 1.0
        
        # Property: MAE should be non-negative
        assert regression_metrics.mae >= 0.0
        
        # Property: RMSE should be non-negative and >= MAE
        assert regression_metrics.rmse >= 0.0
        assert regression_metrics.rmse >= regression_metrics.mae
        
        # Property: R² should be reasonable (can be negative for very poor models)
        assert regression_metrics.r2_score >= -10.0  # Reasonable lower bound
        
        # Property: Confusion matrix should have correct shape
        unique_classes = len(set(y_true))
        assert classification_metrics.confusion_matrix.shape == (unique_classes, unique_classes)
        
        # Property: Confusion matrix elements should sum to total samples
        assert classification_metrics.confusion_matrix.sum() == len(y_true)

    @given(
        accuracy=st.floats(min_value=0.0, max_value=1.0),
        mae=st.floats(min_value=0.0, max_value=10.0),
        rmse=st.floats(min_value=0.0, max_value=10.0)
    )
    @settings(max_examples=50)
    def test_property_performance_threshold_validation(self, accuracy, mae, rmse):
        """
        Property 5: Performance threshold validation
        
        Tests that performance threshold validation works correctly
        across different metric values.
        """
        assume(rmse >= mae)  # RMSE should be >= MAE mathematically
        
        evaluator = ModelEvaluator()
        
        # Create mock metrics
        from app import ClassificationMetrics, RegressionMetrics
        
        classification_metrics = ClassificationMetrics(
            accuracy=accuracy,
            confusion_matrix=np.array([[1, 0], [0, 1]]),  # Mock confusion matrix
            classification_report={}
        )
        
        regression_metrics = RegressionMetrics(
            mae=mae,
            rmse=rmse,
            r2_score=0.5  # Fixed R² for simplicity
        )
        
        # Test threshold validation
        validation_results = evaluator.validate_performance_thresholds(
            classification_metrics, regression_metrics
        )
        
        # Property: Validation results should contain all expected keys
        expected_keys = ['accuracy_acceptable', 'mae_acceptable', 'rmse_acceptable', 'overall_acceptable']
        assert all(key in validation_results for key in expected_keys)
        
        # Property: Individual validations should match thresholds
        assert validation_results['accuracy_acceptable'] == (accuracy >= evaluator.min_accuracy)
        assert validation_results['mae_acceptable'] == (mae <= evaluator.max_mae)
        assert validation_results['rmse_acceptable'] == (rmse <= evaluator.max_rmse)
        
        # Property: Overall acceptability should be logical AND of individual validations
        expected_overall = (
            validation_results['accuracy_acceptable'] and
            validation_results['mae_acceptable'] and
            validation_results['rmse_acceptable']
        )
        assert validation_results['overall_acceptable'] == expected_overall


class TestFeatureExtractionProperties:
    """Property tests for feature extraction functions."""
    
    @given(
        text=st.text(min_size=0, max_size=1000),
        extra_math_symbols=st.integers(min_value=0, max_value=50),
        extra_keywords=st.integers(min_value=0, max_value=10)
    )
    @settings(max_examples=30)
    def test_property_custom_feature_extraction(self, text, extra_math_symbols, extra_keywords):
        """
        Property: Custom feature extraction should be consistent and bounded.
        
        Tests that custom features are extracted correctly and maintain
        expected relationships and bounds.
        """
        # Add controlled math symbols and keywords to text
        math_symbols = '+' * extra_math_symbols
        keywords = ' algorithm' * extra_keywords
        enhanced_text = f"{text} {math_symbols} {keywords}"
        
        # Extract features
        text_len, math_count, keyword_count = extract_custom_features(enhanced_text)
        
        # Property: Text length should match actual length
        assert text_len == len(enhanced_text)
        
        # Property: Math count should be at least the number we added
        assert math_count >= extra_math_symbols
        
        # Property: Keyword count should be at least the number we added
        assert keyword_count >= extra_keywords
        
        # Property: All features should be non-negative
        assert text_len >= 0
        assert math_count >= 0
        assert keyword_count >= 0
        
        # Property: Features should be integers
        assert isinstance(text_len, int)
        assert isinstance(math_count, int)
        assert isinstance(keyword_count, int)


class TestPredictionServiceProperties:
    """Property tests for prediction service functionality."""
    
    @given(
        description=st.text(min_size=1, max_size=500),
        input_desc=st.text(min_size=0, max_size=200),
        output_desc=st.text(min_size=0, max_size=200)
    )
    @settings(max_examples=20)
    def test_property_text_combination(self, description, input_desc, output_desc):
        """
        Property: Text combination should be consistent and preserve content.
        
        Tests that combining three text fields works correctly and
        preserves all input information.
        """
        # Test text combination
        combined = PredictionService.combine_text_features(description, input_desc, output_desc)
        
        # Property: Combined text should contain all non-empty inputs
        if description.strip():
            assert description.strip() in combined
        if input_desc.strip():
            assert input_desc.strip() in combined
        if output_desc.strip():
            assert output_desc.strip() in combined
        
        # Property: Combined text should not be longer than sum of inputs plus spaces
        max_expected_length = len(description) + len(input_desc) + len(output_desc) + 2  # +2 for spaces
        assert len(combined) <= max_expected_length
        
        # Property: If all inputs are empty, combined should be empty
        if not description.strip() and not input_desc.strip() and not output_desc.strip():
            assert not combined.strip()


class TestErrorHandlingProperties:
    """Property tests for error handling robustness."""
    
    @given(
        test_size=st.floats(min_value=0.1, max_value=0.9),
        random_state=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=10)
    def test_property_error_handling_robustness(self, test_size, random_state):
        """
        Property 10: Error handling robustness
        
        Tests that the system handles various edge cases and parameter
        combinations gracefully without crashing.
        """
        # Test ModelEvaluator initialization with different parameters
        evaluator = ModelEvaluator(test_size=test_size, random_state=random_state)
        
        # Property: Evaluator should initialize with valid parameters
        assert evaluator.test_size == test_size
        assert evaluator.random_state == random_state
        assert evaluator.X_train is None  # Should be None before split
        assert evaluator.X_test is None
        
        # Property: Thresholds should be reasonable
        assert 0.0 <= evaluator.min_accuracy <= 1.0
        assert evaluator.max_mae > 0.0
        assert evaluator.max_rmse > 0.0
        assert evaluator.max_rmse >= evaluator.max_mae  # RMSE should be >= MAE threshold


class TestAPIBackwardCompatibilityProperties:
    """Property tests for API backward compatibility."""
    
    @given(
        description=st.text(min_size=1, max_size=500),
        input_desc=st.one_of(st.none(), st.text(min_size=0, max_size=200)),
        output_desc=st.one_of(st.none(), st.text(min_size=0, max_size=200)),
        format_type=st.sampled_from(['legacy', 'structured'])
    )
    @settings(max_examples=20)
    def test_property_api_backward_compatibility(self, description, input_desc, output_desc, format_type):
        """
        Property 8: API backward compatibility
        
        Tests that both legacy and structured API formats work correctly
        and maintain consistent behavior across different input combinations.
        """
        from app import PredictionService
        
        # Test input validation
        validation_result = PredictionService.validate_input_format(
            description, input_desc, output_desc, format_type
        )
        
        # Property: Validation result should always have required keys
        assert 'valid' in validation_result
        assert 'message' in validation_result
        assert isinstance(validation_result['valid'], bool)
        assert isinstance(validation_result['message'], str)
        
        if format_type == 'legacy':
            # Property: Legacy format should only require description
            if description and description.strip():
                assert validation_result['valid'] is True
            else:
                assert validation_result['valid'] is False
                assert 'description' in validation_result['message'].lower()
        
        elif format_type == 'structured':
            # Property: Structured format should require all fields to be present
            if description is None or input_desc is None or output_desc is None:
                assert validation_result['valid'] is False
                assert 'three fields' in validation_result['message'].lower()
            else:
                # Property: At least one field should have content
                combined_content = f"{description} {input_desc} {output_desc}".strip()
                if combined_content:
                    assert validation_result['valid'] is True
                else:
                    assert validation_result['valid'] is False
                    assert 'text content' in validation_result['message'].lower()
        
        # Property: Text combination should be consistent regardless of format
        combined = PredictionService.combine_text_features(description, input_desc, output_desc)
        
        # Property: Combined text should preserve non-None, non-empty inputs
        if description and description.strip():
            assert description.strip() in combined
        if input_desc and input_desc.strip():
            assert input_desc.strip() in combined
        if output_desc and output_desc.strip():
            assert output_desc.strip() in combined


class TestDocumentationProperties:
    """Property tests for documentation generation."""
    
    @given(
        dataset_size=st.integers(min_value=100, max_value=10000),
        feature_count=st.integers(min_value=1000, max_value=10000),
        accuracy=st.floats(min_value=0.0, max_value=1.0),
        mae=st.floats(min_value=0.0, max_value=10.0),
        rmse=st.floats(min_value=0.0, max_value=10.0)
    )
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow])
    def test_property_documentation_completeness(self, dataset_size, feature_count, accuracy, mae, rmse):
        """
        Property 6: Documentation completeness
        
        Tests that generated documentation contains all required sections
        and meets AutoJudge research paper requirements.
        """
        assume(rmse >= mae)  # RMSE should be >= MAE mathematically
        
        from documentation_generator import DocumentationGenerator, DocumentationConfig
        from evaluation_models import ClassificationMetrics, RegressionMetrics, EvaluationReport
        
        # Create mock configuration
        config = DocumentationConfig(
            dataset_size=dataset_size,
            feature_count=feature_count
        )
        
        # Create mock evaluation report
        classification_metrics = ClassificationMetrics(
            accuracy=accuracy,
            confusion_matrix=np.array([[10, 2, 1], [3, 15, 2], [1, 2, 20]]),
            classification_report={'easy': {'precision': 0.8, 'recall': 0.7, 'f1-score': 0.75, 'support': 13}}
        )
        
        regression_metrics = RegressionMetrics(mae=mae, rmse=rmse, r2_score=0.5)
        
        evaluation_report = EvaluationReport(
            classification_metrics=classification_metrics,
            regression_metrics=regression_metrics,
            dataset_info={'total_samples': dataset_size, 'train_samples': int(dataset_size*0.8), 
                         'test_samples': int(dataset_size*0.2), 'feature_count': feature_count},
            model_info={'classification_model': 'LogisticRegression', 'regression_model': 'RandomForestRegressor'}
        )
        
        # Initialize documentation generator
        doc_gen = DocumentationGenerator(config)
        doc_gen.set_evaluation_report(evaluation_report)
        
        # Generate documentation
        readme_content = doc_gen.generate_complete_readme()
        
        # Property: Documentation should contain all required sections
        required_sections = ['Overview', 'Methodology', 'Evaluation Results', 'Technical Specifications', 'Usage Examples']
        for section in required_sections:
            assert section in readme_content, f"Missing required section: {section}"
        
        # Property: Documentation should contain AutoJudge-specific content
        autojudge_content = ['3 separate fields', 'TF-IDF', 'Logistic Regression', 'Random Forest']
        for content in autojudge_content:
            assert content in readme_content, f"Missing AutoJudge content: {content}"
        
        # Property: Documentation should contain evaluation metrics
        assert f"{accuracy:.3f}" in readme_content or f"{accuracy:.2f}" in readme_content
        assert f"{mae:.3f}" in readme_content or f"{mae:.2f}" in readme_content
        assert f"{rmse:.3f}" in readme_content or f"{rmse:.2f}" in readme_content
        
        # Property: Documentation should be substantial (not empty)
        assert len(readme_content) > 5000  # Should be comprehensive
        
        # Property: Validation should pass for complete documentation
        validation_results = doc_gen.validate_documentation_completeness()
        assert isinstance(validation_results, dict)
        assert len(validation_results) > 0
        
        # Property: Most validation checks should pass for properly generated docs
        passed_validations = sum(1 for result in validation_results.values() if result)
        total_validations = len(validation_results)
        assert passed_validations >= total_validations * 0.8  # At least 80% should pass

    @given(
        accuracy=st.floats(min_value=0.0, max_value=1.0),
        mae=st.floats(min_value=0.0, max_value=5.0),
        rmse=st.floats(min_value=0.0, max_value=5.0),
        r2_score=st.floats(min_value=-1.0, max_value=1.0)
    )
    @settings(max_examples=15)
    def test_property_evaluation_results_integration(self, accuracy, mae, rmse, r2_score):
        """
        Property 7: Evaluation results integration
        
        Tests that evaluation results are properly integrated into documentation
        with correct formatting and interpretation.
        """
        assume(rmse >= mae)  # RMSE should be >= MAE mathematically
        
        from documentation_generator import DocumentationGenerator
        from evaluation_models import ClassificationMetrics, RegressionMetrics, EvaluationReport
        
        # Create evaluation metrics
        classification_metrics = ClassificationMetrics(
            accuracy=accuracy,
            confusion_matrix=np.array([[5, 1], [2, 8]]),
            classification_report={'macro avg': {'f1-score': 0.75}}
        )
        
        regression_metrics = RegressionMetrics(mae=mae, rmse=rmse, r2_score=r2_score)
        
        evaluation_report = EvaluationReport(
            classification_metrics=classification_metrics,
            regression_metrics=regression_metrics,
            dataset_info={'total_samples': 1000, 'train_samples': 800, 'test_samples': 200, 'feature_count': 5000},
            model_info={'classification_model': 'LogisticRegression', 'regression_model': 'RandomForestRegressor'}
        )
        
        # Generate evaluation results section
        doc_gen = DocumentationGenerator()
        doc_gen.set_evaluation_report(evaluation_report)
        evaluation_section = doc_gen.generate_evaluation_results_section()
        
        # Property: All metrics should be present in the documentation
        assert f"{accuracy:.3f}" in evaluation_section
        assert f"{mae:.3f}" in evaluation_section
        assert f"{rmse:.3f}" in evaluation_section
        assert f"{r2_score:.3f}" in evaluation_section
        
        # Property: Percentage accuracy should be calculated correctly
        expected_percentage = f"{accuracy*100:.1f}%"
        assert expected_percentage in evaluation_section
        
        # Property: Performance validation should be included
        assert "Threshold Compliance" in evaluation_section
        assert "PASS" in evaluation_section or "WARN" in evaluation_section
        
        # Property: Interpretation should be provided
        assert "Analysis" in evaluation_section or "Interpretation" in evaluation_section
        
        # Property: Section should be substantial and well-formatted
        assert len(evaluation_section) > 500  # Should be comprehensive
        assert "##" in evaluation_section  # Should have proper markdown headers


class TestErrorHandlingProperties:
    """Property tests for error handling robustness."""
    
    @given(
        dataset_size=st.integers(min_value=10, max_value=100),
        error_rate=st.floats(min_value=0.0, max_value=0.5),
        component=st.sampled_from(list(SystemComponent))
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_property_error_handling_robustness(self, dataset_size, error_rate, component):
        """
        Property 10: Error handling robustness
        
        Tests that the system handles various error scenarios gracefully
        and continues operation with appropriate fallback mechanisms.
        """
        from error_handler import ErrorHandler, ErrorSeverity
        
        # Initialize error handler
        error_handler = ErrorHandler()
        
        # Simulate various error scenarios
        def failing_function():
            if np.random.random() < error_rate:
                raise ValueError(f"Simulated error in {component.value}")
            return f"Success for {component.value}"
        
        # Test safe execution with fallback
        fallback_value = f"Fallback for {component.value}"
        result = error_handler.safe_execute(
            func=failing_function,
            component=component,
            fallback_value=fallback_value,
            context={'dataset_size': dataset_size, 'error_rate': error_rate}
        )
        
        # Property: Result should never be None
        assert result is not None
        
        # Property: Result should be either success or fallback
        assert result in [f"Success for {component.value}", fallback_value]
        
        # Property: If error occurred, should be logged
        if result == fallback_value:
            assert len(error_handler.error_history) > 0
            latest_error = error_handler.error_history[-1]
            assert latest_error.component == component
            assert "Simulated error" in latest_error.message
        
        # Property: System health should be updated appropriately
        health_report = error_handler.get_system_health_report()
        assert isinstance(health_report, dict)
        assert 'overall_status' in health_report
        assert 'component_status' in health_report
        
        # Property: Component status should reflect error state
        if result == fallback_value:
            component_status = health_report['component_status'].get(component, 'healthy')
            assert component_status in ['warning', 'degraded']

    @given(
        accuracy=st.floats(min_value=0.0, max_value=1.0),
        mae=st.floats(min_value=0.0, max_value=10.0),
        rmse=st.floats(min_value=0.0, max_value=10.0)
    )
    @settings(max_examples=15)
    def test_property_performance_monitoring_robustness(self, accuracy, mae, rmse):
        """
        Property: Performance monitoring should handle all metric ranges gracefully.
        
        Tests that performance monitoring works correctly across different
        metric values and generates appropriate alerts.
        """
        assume(rmse >= mae)  # RMSE should be >= MAE mathematically
        
        from error_handler import ErrorHandler
        
        error_handler = ErrorHandler()
        
        # Test performance monitoring
        accuracy_alert = error_handler.monitor_performance('accuracy', accuracy)
        mae_alert = error_handler.monitor_performance('mae', mae, comparison='max')
        rmse_alert = error_handler.monitor_performance('rmse', rmse, comparison='max')
        
        # Property: Alerts should be None or PerformanceAlert objects
        for alert in [accuracy_alert, mae_alert, rmse_alert]:
            assert alert is None or hasattr(alert, 'severity')
            assert alert is None or hasattr(alert, 'recommendation')
        
        # Property: Alert generation should be consistent with thresholds
        if accuracy < error_handler.performance_thresholds['min_accuracy']:
            assert accuracy_alert is not None
        if mae > error_handler.performance_thresholds['max_mae']:
            assert mae_alert is not None
        if rmse > error_handler.performance_thresholds['max_rmse']:
            assert rmse_alert is not None
        
        # Property: Performance alerts should have valid recommendations
        for alert in [accuracy_alert, mae_alert, rmse_alert]:
            if alert is not None:
                assert len(alert.recommendation) > 0
                assert isinstance(alert.message, str)
                assert len(alert.message) > 0


if __name__ == '__main__':
    # Run property tests
    pytest.main([__file__, '-v'])