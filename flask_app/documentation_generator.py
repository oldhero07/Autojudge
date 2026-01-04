"""
Documentation Generator for AutoJudge System

This module generates comprehensive documentation following the AutoJudge research paper
structure, including methodology, evaluation results, and technical specifications.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from evaluation_models import EvaluationReport, ClassificationMetrics, RegressionMetrics
from error_handler import ErrorHandler, SystemComponent, ErrorSeverity


@dataclass
class DocumentationConfig:
    """Configuration for documentation generation."""
    project_name: str = "AutoJudge"
    project_description: str = "Programming Problem Difficulty Predictor"
    version: str = "1.0.0"
    author: str = "AutoJudge Team"
    dataset_size: int = 4112
    feature_count: int = 5003
    tfidf_features: int = 5000
    custom_features: int = 3


class DocumentationGenerator:
    """
    Comprehensive documentation generator for AutoJudge system.
    
    Generates documentation following the AutoJudge research paper structure
    with methodology, evaluation results, and technical specifications.
    """
    
    def __init__(self, config: Optional[DocumentationConfig] = None):
        """Initialize documentation generator with configuration and error handling."""
        self.config = config or DocumentationConfig()
        self.evaluation_report: Optional[EvaluationReport] = None
        self.error_handler = ErrorHandler()
        
    def set_evaluation_report(self, report: EvaluationReport):
        """Set the evaluation report for documentation generation with validation."""
        try:
            if report is None:
                raise ValueError("Evaluation report cannot be None")
            if not isinstance(report, EvaluationReport):
                raise TypeError(f"Expected EvaluationReport, got {type(report)}")
            
            # Validate report contents
            if report.classification_metrics is None or report.regression_metrics is None:
                raise ValueError("Evaluation report missing required metrics")
                
            self.evaluation_report = report
            
        except Exception as e:
            self.error_handler.log_error(
                component=SystemComponent.DOCUMENTATION_GENERATOR,
                error=e,
                severity=ErrorSeverity.MEDIUM,
                context={'report_type': type(report).__name__ if report else None}
            )
        
    def set_evaluation_report(self, report: EvaluationReport):
        """Set the evaluation report for documentation generation."""
        self.evaluation_report = report
        
    def generate_methodology_section(self) -> str:
        """
        Generate comprehensive methodology section.
        
        Returns:
            Formatted methodology documentation
        """
        methodology = f"""## Methodology

### Data Preprocessing

The system processes programming problem descriptions through a comprehensive preprocessing pipeline:

1. **Text Field Combination**: Three separate input fields (problem description, input description, output description) are combined into a unified text representation for analysis.

2. **Data Cleaning**: Text normalization and cleaning to handle missing values and ensure consistent formatting.

3. **Feature Engineering Pipeline**: Multi-stage feature extraction combining traditional NLP techniques with domain-specific analysis.

### Feature Extraction

The AutoJudge system employs a hybrid feature extraction approach combining multiple text analysis techniques:

#### TF-IDF Vectorization
- **Vocabulary Size**: {self.config.tfidf_features:,} features
- **N-gram Range**: 1-2 grams for capturing both individual terms and phrase patterns
- **Stop Words**: English stop words removed to focus on meaningful content
- **Normalization**: L2 normalization applied for consistent feature scaling

#### Custom Feature Engineering
The system extracts {self.config.custom_features} domain-specific features designed for programming problem analysis:

1. **Text Length Feature**
   - Character count of combined problem description
   - Indicator of problem complexity and specification detail
   - Range: 0 - 7,571 characters in training data

2. **Mathematical Symbols Count**
   - Frequency of mathematical operators and symbols: `+`, `-`, `*`, `/`, `=`, `<`, `>`, `∑`, `∏`, `∫`, `∂`, `∆`, `√`, `π`, `∞`, etc.
   - Enhanced pattern matching for Unicode mathematical symbols
   - Range: 0 - 262 symbols in training data

3. **Difficulty Keywords Frequency**
   - Frequency analysis of algorithm and data structure keywords
   - Keywords include: `algorithm`, `graph`, `tree`, `recursion`, `dynamic`, `greedy`, `binary`, `search`, `sort`, `heap`, `stack`, `queue`, `optimal`, etc.
   - Range: 0 - 37 keywords in training data

#### Feature Scaling and Combination
- **Custom Feature Scaling**: StandardScaler applied to numerical features for normalization
- **Feature Matrix Combination**: TF-IDF sparse matrix horizontally stacked with scaled custom features
- **Final Feature Space**: {self.config.feature_count:,} dimensional feature vectors

### Model Architecture

#### Classification Model: Problem Difficulty Class Prediction
- **Algorithm**: Logistic Regression
- **Justification**: Proven effectiveness for text classification with interpretable results
- **Configuration**: 
  - Maximum iterations: 1,000
  - Random state: 42 for reproducibility
  - Multi-class strategy: One-vs-Rest (OvR)

#### Regression Model: Difficulty Score Prediction  
- **Algorithm**: Random Forest Regressor
- **Justification**: Robust ensemble method handling non-linear relationships and feature interactions
- **Configuration**:
  - Number of estimators: 300 trees
  - Maximum depth: Unlimited (None) for capturing complex patterns
  - Minimum samples per leaf: 1 for detailed fitting
  - Parallel processing: All available CPU cores (-1)
  - Random state: 42 for reproducibility

### Training and Evaluation Strategy

#### Train/Test Split
- **Split Ratio**: 80% training, 20% testing
- **Stratification**: Stratified split maintaining class distribution
- **Training Samples**: 3,289
- **Test Samples**: 823
- **Random State**: 42 for reproducible results

#### Performance Thresholds
- **Classification Accuracy**: Minimum 60% acceptable
- **Regression MAE**: Maximum 2.0 acceptable  
- **Regression RMSE**: Maximum 2.5 acceptable
"""
        return methodology
    
    def generate_evaluation_results_section(self) -> str:
        """
        Generate comprehensive evaluation results section.
        
        Returns:
            Formatted evaluation results documentation
        """
        if not self.evaluation_report:
            return "## Evaluation Results\n\n*Evaluation results not available. Run model training to generate results.*\n"
            
        classification = self.evaluation_report.classification_metrics
        regression = self.evaluation_report.regression_metrics
        dataset_info = self.evaluation_report.dataset_info
        
        # Format confusion matrix
        cm_shape = classification.confusion_matrix.shape
        cm_total = classification.confusion_matrix.sum()
        
        # Calculate class-wise metrics from classification report
        class_metrics = ""
        if classification.classification_report:
            for class_name in ['easy', 'medium', 'hard']:
                if class_name in classification.classification_report:
                    metrics = classification.classification_report[class_name]
                    precision = metrics.get('precision', 0.0)
                    recall = metrics.get('recall', 0.0)
                    f1_score = metrics.get('f1-score', 0.0)
                    support = int(metrics.get('support', 0))
                    
                    class_metrics += f"""
**{class_name.title()} Class:**
- Precision: {precision:.3f}
- Recall: {recall:.3f}
- F1-Score: {f1_score:.3f}
- Support: {support} samples
"""
        
        evaluation_results = f"""## Evaluation Results

### Dataset Statistics
- **Total Samples**: {dataset_info['total_samples']:,}
- **Training Samples**: {dataset_info['train_samples']:,} ({dataset_info['train_samples']/dataset_info['total_samples']*100:.1f}%)
- **Test Samples**: {dataset_info['test_samples']:,} ({dataset_info['test_samples']/dataset_info['total_samples']*100:.1f}%)
- **Feature Dimensions**: {dataset_info['feature_count']:,}

### Classification Results

#### Overall Performance
- **Accuracy**: {classification.accuracy:.3f} ({classification.accuracy*100:.1f}%)
- **Confusion Matrix**: {cm_shape[0]}×{cm_shape[1]} matrix with {cm_total} total predictions

#### Class-wise Performance{class_metrics}

### Regression Results

#### Performance Metrics
- **Mean Absolute Error (MAE)**: {regression.mae:.3f}
- **Root Mean Square Error (RMSE)**: {regression.rmse:.3f}
- **R² Score**: {regression.r2_score:.3f}

#### Interpretation
- **MAE Analysis**: On average, score predictions deviate by {regression.mae:.2f} points from actual values
- **RMSE Analysis**: Root mean square error of {regression.rmse:.2f} indicates prediction variance
- **R² Analysis**: Model explains {regression.r2_score*100:.1f}% of variance in difficulty scores

### Performance Validation

#### Threshold Compliance
- **Classification Accuracy**: {'✓ PASS' if classification.accuracy >= 0.6 else '✗ WARN'} (threshold: 60%)
- **Regression MAE**: {'✓ PASS' if regression.mae <= 2.0 else '✗ WARN'} (threshold: ≤2.0)
- **Regression RMSE**: {'✓ PASS' if regression.rmse <= 2.5 else '✗ WARN'} (threshold: ≤2.5)

#### Model Quality Assessment
The model demonstrates {'acceptable' if classification.accuracy >= 0.6 and regression.mae <= 2.0 and regression.rmse <= 2.5 else 'mixed'} performance across evaluation metrics. 
{'All performance thresholds are met, indicating robust model quality suitable for production deployment.' if classification.accuracy >= 0.6 and regression.mae <= 2.0 and regression.rmse <= 2.5 else 'Some metrics fall below optimal thresholds but remain within reasonable bounds for continued operation with monitoring.'}
"""
        return evaluation_results
    
    def generate_technical_specifications(self) -> str:
        """
        Generate technical specifications section.
        
        Returns:
            Formatted technical specifications
        """
        specs = f"""## Technical Specifications

### System Architecture

#### Frontend Components
- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite for fast development and optimized builds
- **Styling**: Tailwind CSS for responsive design
- **Icons**: Lucide React for consistent iconography
- **Input Modes**: 
  - AutoJudge Research Format (3 separate fields)
  - Legacy Combined Format (backward compatibility)

#### Backend Components
- **Framework**: Flask web framework
- **ML Libraries**: scikit-learn, pandas, numpy
- **Text Processing**: TF-IDF vectorization with scipy sparse matrices
- **Model Persistence**: In-memory model storage with global state management
- **API Endpoints**:
  - `/predict` - Legacy single-input format
  - `/predict/structured` - AutoJudge three-input format

### Feature Engineering Pipeline

#### Text Processing
1. **Input Validation**: Format-specific validation for legacy and structured inputs
2. **Text Combination**: Intelligent merging of three text fields
3. **Feature Extraction**: Parallel processing of TF-IDF and custom features
4. **Feature Scaling**: StandardScaler normalization for numerical features
5. **Matrix Combination**: Sparse matrix operations for memory efficiency

#### Performance Optimizations
- **Sparse Matrix Operations**: Memory-efficient handling of high-dimensional TF-IDF features
- **Vectorized Computations**: NumPy vectorization for custom feature extraction
- **Parallel Processing**: Multi-core utilization for Random Forest training
- **Feature Caching**: TF-IDF vectorizer and scaler persistence

### Model Training Pipeline

#### Data Loading and Preprocessing
- **Dataset Format**: JSONL (JSON Lines) for efficient streaming
- **Column Mapping**: Automatic mapping of input/output description fields
- **Missing Value Handling**: Graceful handling of empty or None values
- **Text Normalization**: Whitespace trimming and cleaning

#### Training Process
1. **Feature Matrix Construction**: {self.config.feature_count:,} dimensional sparse matrices
2. **Stratified Splitting**: Maintaining class distribution across train/test sets
3. **Model Training**: Parallel training of classification and regression models
4. **Evaluation**: Comprehensive metric calculation on held-out test set
5. **Validation**: Automated threshold checking with warning system

### API Specifications

#### Request Formats

**Legacy Format (`/predict`)**:
```json
{{
  "description": "Combined problem description text"
}}
```

**Structured Format (`/predict/structured`)**:
```json
{{
  "description": "Problem statement and requirements",
  "input_desc": "Input format specification", 
  "output_desc": "Expected output format"
}}
```

#### Response Format
```json
{{
  "class": "easy|medium|hard",
  "score": 5.2,
  "features": {{
    "textLength": 150,
    "mathSymbols": 5,
    "keywords": 2,
    "tfidfFeatures": 5000
  }},
  "format": "structured",  // Only in structured endpoint
  "input_fields": {{        // Only in structured endpoint
    "description_length": 50,
    "input_desc_length": 20,
    "output_desc_length": 15
  }}
}}
```

### Deployment Configuration

#### Development Environment
- **Frontend**: `npm run dev` on port 3000
- **Backend**: `python app.py` on port 5000
- **Hot Reload**: Automatic reloading for both frontend and backend

#### Production Considerations
- **Model Loading**: Automatic model training on application startup
- **Error Handling**: Comprehensive error handling with fallback responses
- **Logging**: Structured logging for monitoring and debugging
- **Performance Monitoring**: Threshold validation with warning system
"""
        return specs
    
    def generate_usage_examples(self) -> str:
        """
        Generate usage examples section.
        
        Returns:
            Formatted usage examples
        """
        examples = """## Usage Examples

### Web Interface Usage

#### AutoJudge Research Format (Recommended)
1. **Problem Description**: Enter the main problem statement
   ```
   Given an array of integers, find the maximum sum of any contiguous subarray.
   ```

2. **Input Description**: Specify the input format
   ```
   First line contains n (1 ≤ n ≤ 10^5). 
   Second line contains n integers (-10^9 ≤ ai ≤ 10^9).
   ```

3. **Output Description**: Define the expected output
   ```
   Output a single integer representing the maximum subarray sum.
   ```

#### Legacy Combined Format
Enter all information in a single text field:
```
Given an array of integers, find the maximum sum of any contiguous subarray.
First line contains n (1 ≤ n ≤ 10^5). Second line contains n integers (-10^9 ≤ ai ≤ 10^9).
Output a single integer representing the maximum subarray sum.
```

### API Usage Examples

#### Using cURL

**Legacy Endpoint**:
```bash
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "description": "Find the shortest path in a weighted graph using Dijkstra algorithm."
  }'
```

**Structured Endpoint**:
```bash
curl -X POST http://localhost:5000/predict/structured \\
  -H "Content-Type: application/json" \\
  -d '{
    "description": "Find the shortest path in a weighted graph",
    "input_desc": "Graph with n vertices and m edges",
    "output_desc": "Shortest distance from source to target"
  }'
```

#### Using Python Requests

```python
import requests

# Structured format example
data = {
    "description": "Implement a binary search tree with insert and search operations",
    "input_desc": "Sequence of operations: INSERT x or SEARCH x",
    "output_desc": "For each SEARCH, output YES or NO"
}

response = requests.post('http://localhost:5000/predict/structured', json=data)
result = response.json()

print(f"Predicted Class: {result['class']}")
print(f"Difficulty Score: {result['score']}/10")
print(f"Text Features: {result['features']}")
```

### Expected Outputs

#### Easy Problem Example
```json
{
  "class": "easy",
  "score": 2.3,
  "features": {
    "textLength": 120,
    "mathSymbols": 3,
    "keywords": 1,
    "tfidfFeatures": 5000
  }
}
```

#### Hard Problem Example  
```json
{
  "class": "hard", 
  "score": 8.7,
  "features": {
    "textLength": 450,
    "mathSymbols": 15,
    "keywords": 8,
    "tfidfFeatures": 5000
  }
}
```
"""
        return examples
    
    def generate_complete_readme(self) -> str:
        """
        Generate complete README documentation following AutoJudge structure.
        
        Returns:
            Complete README content
        """
        timestamp = datetime.now().strftime("%Y-%m-%d")
        
        readme_content = f"""# {self.config.project_name}

{self.config.project_description}

*Generated on {timestamp} - Version {self.config.version}*

## Overview

AutoJudge is an intelligent system that automatically predicts programming problem difficulty using advanced machine learning and natural language processing techniques. The system analyzes textual problem descriptions to provide both classification (Easy/Medium/Hard) and regression (numerical difficulty score) predictions.

### Key Features

- **Dual Input Modes**: AutoJudge research format (3 separate fields) and legacy combined format
- **Advanced Feature Engineering**: TF-IDF vectorization combined with domain-specific features
- **Comprehensive Evaluation**: Accuracy, confusion matrix, MAE, and RMSE metrics
- **Real-time Analysis**: Live feature extraction and prediction
- **Backward Compatibility**: Support for both structured and legacy API formats
- **Production Ready**: Robust error handling and performance monitoring

### Project Objectives

By the end of this implementation, the AutoJudge system:
✅ Predicts problem difficulty class (Easy/Medium/Hard)
✅ Predicts numerical difficulty score (1-10 scale)
✅ Works using only textual information from problem descriptions
✅ Provides results through a modern web interface
✅ Supports both structured three-input and legacy single-input formats
✅ Includes comprehensive evaluation metrics and documentation

{self.generate_methodology_section()}

{self.generate_evaluation_results_section()}

{self.generate_technical_specifications()}

{self.generate_usage_examples()}

## Installation and Setup

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- Git

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd autojudge
   ```

2. **Install frontend dependencies**:
   ```bash
   npm install
   ```

3. **Install backend dependencies**:
   ```bash
   cd flask_app
   pip install -r requirements.txt
   cd ..
   ```

4. **Start the development servers**:
   
   **Frontend** (Terminal 1):
   ```bash
   npm run dev
   ```
   
   **Backend** (Terminal 2):
   ```bash
   cd flask_app
   python app.py
   ```

5. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

### Production Build

```bash
npm run build
```

## Testing

The system includes comprehensive testing coverage:

### Property-Based Tests
```bash
cd flask_app
python -m pytest test_property_tests.py -v
```

### Unit Tests
```bash
cd flask_app
python -m pytest test_model_evaluator.py -v
python -m pytest test_api_endpoints.py -v
```

### All Tests
```bash
cd flask_app
python -m pytest -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- AutoJudge research paper for system specifications
- scikit-learn community for machine learning tools
- React and Flask communities for web framework support

---

*AutoJudge System - Intelligent Programming Problem Difficulty Prediction*
"""
        return readme_content
    
    def save_documentation(self, output_path: str = "../README.md") -> bool:
        """
        Save generated documentation to file with enhanced error handling.
        
        Args:
            output_path: Path to save the README file
            
        Returns:
            True if successful, False otherwise
        """
        def _save_documentation():
            readme_content = self.generate_complete_readme()
            
            if not readme_content or len(readme_content) < 1000:
                raise ValueError("Generated documentation is too short or empty")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
                
            # Verify file was written correctly
            if not os.path.exists(output_path):
                raise IOError("File was not created successfully")
                
            # Verify file size
            file_size = os.path.getsize(output_path)
            if file_size < 1000:
                raise IOError("Generated file is too small")
                
            return True
        
        # Use error handler for safe execution
        context = {
            'output_path': output_path,
            'has_evaluation_report': self.evaluation_report is not None
        }
        
        result = self.error_handler.safe_execute(
            func=_save_documentation,
            component=SystemComponent.DOCUMENTATION_GENERATOR,
            fallback_value=False,
            context=context
        )
        
        return result if result is not None else False
    
    def validate_documentation_completeness(self) -> Dict[str, bool]:
        """
        Validate that all required documentation sections are present.
        
        Returns:
            Dictionary with validation results for each section
        """
        readme_content = self.generate_complete_readme()
        
        required_sections = {
            'overview': 'Overview' in readme_content,
            'methodology': 'Methodology' in readme_content,
            'evaluation_results': 'Evaluation Results' in readme_content,
            'technical_specifications': 'Technical Specifications' in readme_content,
            'usage_examples': 'Usage Examples' in readme_content,
            'installation': 'Installation and Setup' in readme_content,
            'testing': 'Testing' in readme_content
        }
        
        # Check for specific AutoJudge requirements
        autojudge_requirements = {
            'three_input_format': 'three separate fields' in readme_content.lower(),
            'classification_metrics': 'accuracy' in readme_content.lower() and 'confusion matrix' in readme_content.lower(),
            'regression_metrics': 'mae' in readme_content.lower() and 'rmse' in readme_content.lower(),
            'feature_engineering': 'tf-idf' in readme_content.lower() and 'custom features' in readme_content.lower(),
            'model_justification': 'logistic regression' in readme_content.lower() and 'random forest' in readme_content.lower()
        }
        
        return {**required_sections, **autojudge_requirements}