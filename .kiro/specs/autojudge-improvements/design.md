# AutoJudge Improvements Design Document

## Overview

This design document outlines the enhancements needed to align the existing AutoJudge system with the research paper specifications. The improvements focus on three critical areas: UI restructuring to match the three-input format, comprehensive evaluation metrics implementation, and enhanced documentation generation.

The enhanced system will maintain backward compatibility while adding the required AutoJudge research paper features, ensuring both existing users and new research-compliant workflows are supported.

## Architecture

The enhanced architecture builds upon the existing layered structure with new evaluation and documentation components:

```
┌─────────────────────────────────────┐
│           Web Layer                 │
│  (Enhanced UI with 3 inputs)       │
├─────────────────────────────────────┤
│         Business Logic              │
│    (Backward Compatible API)       │
├─────────────────────────────────────┤
│        ML Service Layer             │
│  (Enhanced with Evaluation)        │
├─────────────────────────────────────┤
│      Evaluation Module              │
│  (Metrics & Performance Analysis)  │
├─────────────────────────────────────┤
│    Documentation Generator          │
│  (Automated README & Reports)      │
├─────────────────────────────────────┤
│         Data Layer                  │
│    (Train/Test Split Management)   │
└─────────────────────────────────────┘
```

### Key Architectural Decisions

1. **Backward Compatibility**: Maintain existing single-input API while adding three-input support
2. **Evaluation Integration**: Add comprehensive metrics calculation during model training
3. **Documentation Automation**: Generate research-compliant documentation automatically
4. **Flexible UI**: Support both single combined input and three separate inputs

## Components and Interfaces

### Enhanced Web Components

**Updated React Frontend**
```typescript
interface ThreeInputForm {
  problemDescription: string;
  inputDescription: string;
  outputDescription: string;
}

interface CompatibilityMode {
  combinedText: string;
  useThreeInputs: boolean;
}
```

**Enhanced API Endpoints**
```python
# Backward compatible endpoint
POST /predict
{
  "description": "string",
  "input_desc": "string",
  "output_desc": "string"
}

# New three-input endpoint (same format, enhanced processing)
POST /predict/structured
{
  "problem_description": "string",
  "input_description": "string", 
  "output_description": "string"
}
```

### New Evaluation Components

**Model Evaluator**
```python
class ModelEvaluator:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
    
    def evaluate_classification(self, model, X_test, y_test) -> ClassificationMetrics:
        # Returns: accuracy, confusion_matrix, classification_report
        
    def evaluate_regression(self, model, X_test, y_test) -> RegressionMetrics:
        # Returns: mae, rmse, r2_score
        
    def generate_evaluation_report(self) -> EvaluationReport:
        # Returns: comprehensive evaluation summary
```

**Evaluation Metrics Data Models**
```python
@dataclass
class ClassificationMetrics:
    accuracy: float
    confusion_matrix: np.ndarray
    classification_report: dict
    
@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2_score: float
    
@dataclass
class EvaluationReport:
    classification_metrics: ClassificationMetrics
    regression_metrics: RegressionMetrics
    dataset_info: dict
    model_info: dict
```

### Documentation Generator

**Documentation Service**
```python
class DocumentationGenerator:
    def generate_methodology_section(self) -> str:
        # Generates detailed methodology explanation
        
    def generate_evaluation_section(self, metrics: EvaluationReport) -> str:
        # Generates evaluation results with interpretation
        
    def generate_feature_engineering_section(self) -> str:
        # Explains TF-IDF and custom features
        
    def update_readme(self, sections: dict) -> None:
        # Updates README with new sections
```

## Data Models

### Enhanced Input Processing

```python
@dataclass
class StructuredProblemInput:
    problem_description: str
    input_description: str
    output_description: str
    
    def to_combined_text(self) -> str:
        # Combines three fields with proper spacing
        return f"{self.problem_description} {self.input_description} {self.output_description}".strip()
    
    @classmethod
    def from_legacy_input(cls, description: str, input_desc: str = "", output_desc: str = ""):
        # Maintains backward compatibility
        return cls(
            problem_description=description,
            input_description=input_desc,
            output_description=output_desc
        )
```

### Evaluation Data Models

```python
@dataclass
class TrainTestSplit:
    X_train: scipy.sparse.csr_matrix
    X_test: scipy.sparse.csr_matrix
    y_train_class: pd.Series
    y_test_class: pd.Series
    y_train_score: pd.Series
    y_test_score: pd.Series
    split_info: dict
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

After reviewing the prework analysis, I identified several areas where properties can be consolidated:

- Properties 1.2, 1.3, and 1.4 all test text processing with different input combinations and can be combined into a comprehensive text processing property
- Properties 2.2 and 2.3 both test evaluation metric calculation and can be combined into a single evaluation metrics property
- Properties 4.1, 4.2, and 4.5 all test API compatibility and response consistency and can be combined
- Properties 3.1, 3.3, and 3.4 all test documentation content and can be combined into a documentation completeness property

The remaining properties provide unique validation value and should be kept separate.

### Correctness Properties

**Property 1: Three-input UI rendering**
*For any* application load, the user interface should display exactly three separate text input areas with correct labels for problem description, input description, and output description
**Validates: Requirements 1.1**

**Property 2: Dynamic text processing**
*For any* combination of text inputs across the three fields, the system should correctly combine them and update feature analysis in real-time
**Validates: Requirements 1.2, 1.3, 1.4**

**Property 3: Prediction result completeness**
*For any* valid prediction request, the system should return both a predicted class (Easy/Medium/Hard) and a numerical score (1-10)
**Validates: Requirements 1.5**

**Property 4: Evaluation metrics calculation**
*For any* trained model, the evaluation module should calculate and return valid accuracy, confusion matrix, MAE, and RMSE metrics
**Validates: Requirements 2.2, 2.3**

**Property 5: Performance threshold validation**
*For any* model evaluation, the system should validate metrics against acceptable thresholds and log appropriate warnings or confirmations
**Validates: Requirements 2.5**

**Property 6: Documentation completeness**
*For any* documentation generation, the output should include methodology, evaluation results, feature engineering explanation, and model selection justification
**Validates: Requirements 3.1, 3.3, 3.4**

**Property 7: Evaluation results integration**
*For any* calculated evaluation metrics, the documentation should include the results with proper interpretation and formatting
**Validates: Requirements 3.2**

**Property 8: API backward compatibility**
*For any* API request using the legacy single-input format or new three-input format, the system should process it correctly and return consistent response formats
**Validates: Requirements 4.1, 4.2, 4.5**

**Property 9: UI input flexibility**
*For any* user interaction, the interface should support both three separate inputs and combined text input methods
**Validates: Requirements 4.4**

**Property 10: Error handling robustness**
*For any* evaluation-related error condition, the system should provide appropriate error messages, logging, and graceful degradation
**Validates: Requirements 5.1, 5.2, 5.5**

## Error Handling

### Enhanced Error Handling for Evaluation

**Train/Test Split Errors**
- **Dataset Loading Failures**: Log detailed error and use fallback dummy dataset
- **Insufficient Data**: Warn user and adjust split ratio automatically
- **Memory Issues**: Implement chunked processing for large datasets

**Evaluation Metric Errors**
- **Calculation Failures**: Provide fallback metrics (e.g., basic accuracy if confusion matrix fails)
- **Invalid Predictions**: Log warnings and exclude invalid samples from metrics
- **Threshold Validation**: Continue operation with warnings if performance is below thresholds

**Documentation Generation Errors**
- **Template Errors**: Use fallback templates and log specific issues
- **File Write Permissions**: Provide in-memory documentation if file writing fails
- **Missing Data**: Generate partial documentation with clear indicators of missing sections

### Error Response Enhancements

```json
{
  "error": "Evaluation error",
  "message": "Failed to calculate confusion matrix",
  "details": {
    "component": "ModelEvaluator",
    "fallback_used": true,
    "fallback_metrics": {
      "accuracy": 0.85
    }
  },
  "status_code": 200
}
```

## Testing Strategy

### Dual Testing Approach

The enhanced application will use both unit testing and property-based testing:

- **Unit tests** verify specific examples, edge cases, and error conditions
- **Property tests** verify universal properties that should hold across all inputs
- Together they provide comprehensive coverage for the new evaluation and documentation features

### Unit Testing

Unit tests will cover:
- Three-input UI component rendering with correct labels and IDs
- Train/test split functionality with various dataset sizes
- Evaluation metrics calculation with known test cases
- Documentation generation with template validation
- API backward compatibility with legacy request formats

### Property-Based Testing

Property-based testing will use **Hypothesis** for Python and **@fast-check/jest** for TypeScript. Each property-based test will:
- Run a minimum of 100 iterations with randomly generated inputs
- Be tagged with comments explicitly referencing the correctness property from this design document
- Use the format: '**Feature: autojudge-improvements, Property {number}: {property_text}**'

Key property test areas:
- Text processing with various combinations of empty and filled input fields
- Evaluation metrics calculation across different model performance levels
- Documentation generation with various metric combinations
- API compatibility with different request formats
- Error handling with various failure scenarios

### Test Data Generation

For property-based tests, generators will create:
- **Three-Input Generators**: Various combinations of problem, input, and output descriptions
- **Evaluation Data Generators**: Different train/test splits and model performance scenarios
- **Documentation Generators**: Various metric combinations and template scenarios
- **API Request Generators**: Both legacy and new format requests with edge cases

### Testing Framework Requirements

- **Unit Testing**: pytest for Python, Jest for TypeScript
- **Property-Based Testing**: Hypothesis for Python, fast-check for TypeScript
- **UI Testing**: React Testing Library for component testing
- **Integration Testing**: Flask test client for API endpoint testing
- **Coverage**: pytest-cov and Jest coverage for comprehensive test coverage reporting

Each property-based test must implement only the numbered properties from this design document and focus on real functionality validation without excessive mocking.