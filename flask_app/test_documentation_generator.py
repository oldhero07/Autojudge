"""
Unit tests for DocumentationGenerator class.

These tests validate the documentation generation functionality
including methodology, evaluation results, and README structure.
"""

import pytest
import os
import tempfile
import numpy as np
from documentation_generator import DocumentationGenerator, DocumentationConfig
from evaluation_models import ClassificationMetrics, RegressionMetrics, EvaluationReport


class TestDocumentationGenerator:
    """Unit tests for DocumentationGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DocumentationConfig(
            project_name="Test AutoJudge",
            dataset_size=1000,
            feature_count=5003,
            tfidf_features=5000,
            custom_features=3
        )
        
        self.doc_gen = DocumentationGenerator(self.config)
        
        # Create mock evaluation report
        self.classification_metrics = ClassificationMetrics(
            accuracy=0.75,
            confusion_matrix=np.array([[10, 2, 1], [3, 15, 2], [1, 2, 20]]),
            classification_report={
                'easy': {'precision': 0.8, 'recall': 0.7, 'f1-score': 0.75, 'support': 13},
                'medium': {'precision': 0.75, 'recall': 0.8, 'f1-score': 0.77, 'support': 20},
                'hard': {'precision': 0.87, 'recall': 0.85, 'f1-score': 0.86, 'support': 23}
            }
        )
        
        self.regression_metrics = RegressionMetrics(mae=1.2, rmse=1.5, r2_score=0.65)
        
        self.evaluation_report = EvaluationReport(
            classification_metrics=self.classification_metrics,
            regression_metrics=self.regression_metrics,
            dataset_info={
                'total_samples': 1000,
                'train_samples': 800,
                'test_samples': 200,
                'feature_count': 5003,
                'class_distribution': {'easy': 200, 'medium': 400, 'hard': 400}
            },
            model_info={
                'classification_model': 'LogisticRegression',
                'regression_model': 'RandomForestRegressor',
                'test_size': 0.2,
                'random_state': 42
            }
        )
        
        self.doc_gen.set_evaluation_report(self.evaluation_report)
    
    def test_initialization(self):
        """Test DocumentationGenerator initialization."""
        # Test with default config
        doc_gen_default = DocumentationGenerator()
        assert doc_gen_default.config.project_name == "AutoJudge"
        assert doc_gen_default.evaluation_report is None
        
        # Test with custom config
        assert self.doc_gen.config.project_name == "Test AutoJudge"
        assert self.doc_gen.config.dataset_size == 1000
    
    def test_set_evaluation_report(self):
        """Test setting evaluation report."""
        new_doc_gen = DocumentationGenerator()
        assert new_doc_gen.evaluation_report is None
        
        new_doc_gen.set_evaluation_report(self.evaluation_report)
        assert new_doc_gen.evaluation_report == self.evaluation_report
    
    def test_generate_methodology_section(self):
        """Test methodology section generation."""
        methodology = self.doc_gen.generate_methodology_section()
        
        # Check required content
        assert "## Methodology" in methodology
        assert "Data Preprocessing" in methodology
        assert "Feature Extraction" in methodology
        assert "TF-IDF Vectorization" in methodology
        assert "Custom Feature Engineering" in methodology
        assert "Model Architecture" in methodology
        assert "Logistic Regression" in methodology
        assert "Random Forest Regressor" in methodology
        
        # Check specific configuration values
        assert "5,000 features" in methodology
        assert "3 domain-specific features" in methodology
        assert "80% training, 20% testing" in methodology
        
        # Check that methodology is substantial
        assert len(methodology) > 2000
    
    def test_generate_evaluation_results_section(self):
        """Test evaluation results section generation."""
        evaluation_section = self.doc_gen.generate_evaluation_results_section()
        
        # Check required content
        assert "## Evaluation Results" in evaluation_section
        assert "Dataset Statistics" in evaluation_section
        assert "Classification Results" in evaluation_section
        assert "Regression Results" in evaluation_section
        assert "Performance Validation" in evaluation_section
        
        # Check specific metrics
        assert "0.750" in evaluation_section  # Accuracy
        assert "1.200" in evaluation_section  # MAE
        assert "1.500" in evaluation_section  # RMSE
        assert "0.650" in evaluation_section  # R²
        
        # Check percentage calculation
        assert "75.0%" in evaluation_section  # Accuracy percentage
        
        # Check class-wise metrics
        assert "Easy Class:" in evaluation_section
        assert "Medium Class:" in evaluation_section
        assert "Hard Class:" in evaluation_section
        
        # Check dataset info
        assert "1,000" in evaluation_section  # Total samples
        assert "800" in evaluation_section   # Train samples
        assert "200" in evaluation_section   # Test samples
    
    def test_generate_evaluation_results_without_report(self):
        """Test evaluation results generation without evaluation report."""
        doc_gen_no_report = DocumentationGenerator()
        evaluation_section = doc_gen_no_report.generate_evaluation_results_section()
        
        assert "Evaluation results not available" in evaluation_section
        assert "Run model training" in evaluation_section
    
    def test_generate_technical_specifications(self):
        """Test technical specifications section generation."""
        tech_specs = self.doc_gen.generate_technical_specifications()
        
        # Check required content
        assert "## Technical Specifications" in tech_specs
        assert "System Architecture" in tech_specs
        assert "Frontend Components" in tech_specs
        assert "Backend Components" in tech_specs
        assert "Feature Engineering Pipeline" in tech_specs
        assert "API Specifications" in tech_specs
        
        # Check specific technologies
        assert "React 19" in tech_specs
        assert "Flask" in tech_specs
        assert "scikit-learn" in tech_specs
        assert "TF-IDF" in tech_specs
        
        # Check API endpoints
        assert "/predict" in tech_specs
        assert "/predict/structured" in tech_specs
        
        # Check request/response formats
        assert "json" in tech_specs.lower()
        assert "description" in tech_specs
        assert "input_desc" in tech_specs
        assert "output_desc" in tech_specs
    
    def test_generate_usage_examples(self):
        """Test usage examples section generation."""
        usage_examples = self.doc_gen.generate_usage_examples()
        
        # Check required content
        assert "## Usage Examples" in usage_examples
        assert "Web Interface Usage" in usage_examples
        assert "API Usage Examples" in usage_examples
        assert "Expected Outputs" in usage_examples
        
        # Check specific examples
        assert "curl" in usage_examples
        assert "requests.post" in usage_examples
        assert "AutoJudge Research Format" in usage_examples
        assert "Legacy Combined Format" in usage_examples
        
        # Check example outputs
        assert '"class": "easy"' in usage_examples
        assert '"class": "hard"' in usage_examples
        assert '"score":' in usage_examples
    
    def test_generate_complete_readme(self):
        """Test complete README generation."""
        readme = self.doc_gen.generate_complete_readme()
        
        # Check main sections
        assert "# Test AutoJudge" in readme
        assert "## Overview" in readme
        assert "## Methodology" in readme
        assert "## Evaluation Results" in readme
        assert "## Technical Specifications" in readme
        assert "## Usage Examples" in readme
        assert "## Installation and Setup" in readme
        assert "## Testing" in readme
        
        # Check AutoJudge objectives
        assert "✅ Predicts problem difficulty class" in readme
        assert "✅ Predicts numerical difficulty score" in readme
        assert "✅ Works using only textual information" in readme
        
        # Check that README is comprehensive
        assert len(readme) > 10000  # Should be very comprehensive
        
        # Check timestamp
        from datetime import datetime
        current_year = datetime.now().year
        assert str(current_year) in readme
    
    def test_save_documentation(self):
        """Test saving documentation to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_README.md")
            
            # Test successful save
            success = self.doc_gen.save_documentation(output_path)
            assert success is True
            assert os.path.exists(output_path)
            
            # Check file content
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert "# Test AutoJudge" in content
                assert "## Methodology" in content
                assert len(content) > 5000
    
    def test_save_documentation_error_handling(self):
        """Test error handling in save_documentation."""
        # Test with a path that would cause permission error on Windows
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file where we want to create a directory
            invalid_path = os.path.join(temp_dir, "file.txt")
            with open(invalid_path, 'w') as f:
                f.write("test")
            
            # Try to save documentation to a path that conflicts with existing file
            conflicting_path = os.path.join(invalid_path, "README.md")  # This should fail
            success = self.doc_gen.save_documentation(conflicting_path)
            assert success is False
    
    def test_validate_documentation_completeness(self):
        """Test documentation completeness validation."""
        validation_results = self.doc_gen.validate_documentation_completeness()
        
        # Check that validation returns a dictionary
        assert isinstance(validation_results, dict)
        assert len(validation_results) > 0
        
        # Check required sections
        required_sections = ['overview', 'methodology', 'evaluation_results', 
                           'technical_specifications', 'usage_examples', 'installation', 'testing']
        for section in required_sections:
            assert section in validation_results
            assert isinstance(validation_results[section], bool)
        
        # Check AutoJudge requirements
        autojudge_requirements = ['three_input_format', 'classification_metrics', 
                                'regression_metrics', 'feature_engineering', 'model_justification']
        for requirement in autojudge_requirements:
            assert requirement in validation_results
            assert isinstance(validation_results[requirement], bool)
        
        # Most validations should pass for properly configured generator
        passed_validations = sum(1 for result in validation_results.values() if result)
        total_validations = len(validation_results)
        assert passed_validations >= total_validations * 0.8  # At least 80% should pass


class TestDocumentationConfig:
    """Unit tests for DocumentationConfig class."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = DocumentationConfig()
        
        assert config.project_name == "AutoJudge"
        assert config.project_description == "Programming Problem Difficulty Predictor"
        assert config.version == "1.0.0"
        assert config.author == "AutoJudge Team"
        assert config.dataset_size == 4112
        assert config.feature_count == 5003
        assert config.tfidf_features == 5000
        assert config.custom_features == 3
    
    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = DocumentationConfig(
            project_name="Custom AutoJudge",
            version="2.0.0",
            dataset_size=2000,
            feature_count=3000
        )
        
        assert config.project_name == "Custom AutoJudge"
        assert config.version == "2.0.0"
        assert config.dataset_size == 2000
        assert config.feature_count == 3000
        # Other values should remain default
        assert config.project_description == "Programming Problem Difficulty Predictor"
        assert config.tfidf_features == 5000


class TestDocumentationIntegration:
    """Integration tests for documentation generation with real evaluation data."""
    
    def test_documentation_with_various_metrics(self):
        """Test documentation generation with various metric combinations."""
        test_cases = [
            # Good performance
            {'accuracy': 0.85, 'mae': 1.0, 'rmse': 1.3, 'r2': 0.8},
            # Poor performance
            {'accuracy': 0.45, 'mae': 3.0, 'rmse': 3.5, 'r2': 0.1},
            # Mixed performance
            {'accuracy': 0.65, 'mae': 1.8, 'rmse': 2.2, 'r2': 0.4}
        ]
        
        for i, metrics in enumerate(test_cases):
            classification_metrics = ClassificationMetrics(
                accuracy=metrics['accuracy'],
                confusion_matrix=np.array([[10, 2], [3, 15]]),
                classification_report={'macro avg': {'f1-score': 0.7}}
            )
            
            regression_metrics = RegressionMetrics(
                mae=metrics['mae'],
                rmse=metrics['rmse'],
                r2_score=metrics['r2']
            )
            
            evaluation_report = EvaluationReport(
                classification_metrics=classification_metrics,
                regression_metrics=regression_metrics,
                dataset_info={'total_samples': 1000, 'train_samples': 800, 
                            'test_samples': 200, 'feature_count': 5000},
                model_info={'classification_model': 'LogisticRegression', 
                          'regression_model': 'RandomForestRegressor'}
            )
            
            doc_gen = DocumentationGenerator()
            doc_gen.set_evaluation_report(evaluation_report)
            
            # Generate documentation
            readme = doc_gen.generate_complete_readme()
            
            # Check that metrics are properly included
            assert f"{metrics['accuracy']:.3f}" in readme
            assert f"{metrics['mae']:.3f}" in readme
            assert f"{metrics['rmse']:.3f}" in readme
            
            # Check validation
            validation_results = doc_gen.validate_documentation_completeness()
            assert validation_results['evaluation_results'] is True
            assert validation_results['classification_metrics'] is True
            assert validation_results['regression_metrics'] is True


if __name__ == '__main__':
    # Run unit tests
    pytest.main([__file__, '-v'])