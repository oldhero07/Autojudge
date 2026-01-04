# Flask ML Web App Design Document

## Overview

The Flask ML Web App is a web-based machine learning application that classifies programming problems and predicts their difficulty scores. The system combines a Flask web framework with scikit-learn machine learning models to provide both a user-friendly web interface and a RESTful API for problem analysis.

The application processes programming problem descriptions, input/output format specifications, and uses TF-IDF vectorization combined with logistic and linear regression models to provide automated classification (Easy/Medium/Hard) and numeric scoring (0-100).

## Architecture

The application follows a layered architecture pattern:

```
┌─────────────────────────────────────┐
│           Web Layer                 │
│  (Flask Routes, Templates, Static)  │
├─────────────────────────────────────┤
│         Business Logic              │
│    (Data Processing, Validation)    │
├─────────────────────────────────────┤
│        ML Service Layer             │
│  (Model Training, Prediction API)   │
├─────────────────────────────────────┤
│         Data Layer                  │
│    (Dummy Dataset, Preprocessing)   │
└─────────────────────────────────────┘
```

### Key Architectural Decisions

1. **Monolithic Structure**: Single Flask application for simplicity and ease of deployment
2. **In-Memory Models**: Models loaded at startup and kept in memory for fast predictions
3. **Stateless Design**: No session management required, each request is independent
4. **RESTful API**: Clean separation between web UI and programmatic access

## Components and Interfaces

### Web Components

**Flask Application (`app.py`)**
- Main application entry point
- Route definitions and request handling
- Model initialization and management

**Templates (`templates/`)**
- `index.html`: Main user interface with form for problem submission
- Uses Jinja2 templating with Flask url_for for asset linking

**Static Assets (`static/`)**
- CSS files for styling
- JavaScript for client-side interaction and AJAX requests

### ML Components

**Data Generator**
```python
def generate_dummy_data() -> pd.DataFrame:
    # Creates 20-row dataset with required columns
    # Returns: DataFrame with description, input_desc, output_desc, problem_class, problem_score
```

**Text Processor**
```python
def combine_text_features(description: str, input_desc: str, output_desc: str) -> str:
    # Combines three text fields into single feature string
    # Returns: Combined text for vectorization
```

**Model Trainer**
```python
class MLModelTrainer:
    def train_models(self, data: pd.DataFrame) -> Tuple[Pipeline, Pipeline]:
        # Trains both classification and regression models
        # Returns: (classifier_pipeline, regressor_pipeline)
```

**Prediction Service**
```python
class PredictionService:
    def predict_class_and_score(self, text: str) -> Dict[str, Union[str, float]]:
        # Makes predictions using trained models
        # Returns: {"class": str, "score": float}
```

### API Interfaces

**Web Routes**
- `GET /`: Renders main interface
- `POST /predict`: Accepts JSON, returns predictions

**Request/Response Schemas**
```json
// POST /predict request
{
    "description": "string",
    "input_desc": "string", 
    "output_desc": "string"
}

// POST /predict response
{
    "class": "Easy|Medium|Hard",
    "score": 0-100
}
```

## Data Models

### Input Data Structure
```python
@dataclass
class ProblemInput:
    description: str      # Programming problem description
    input_desc: str      # Input format specification
    output_desc: str     # Output format specification
```

### Training Data Schema
```python
# Dummy dataset columns:
- description: str     # Problem text (e.g., "Find the maximum element in array")
- input_desc: str     # Input format (e.g., "First line contains n, second line contains n integers")
- output_desc: str    # Output format (e.g., "Single integer representing maximum value")
- problem_class: str  # Target classification ("Easy", "Medium", "Hard")
- problem_score: int  # Target regression score (0-100)
```

### Model Artifacts
```python
# Trained model components stored in memory:
- tfidf_vectorizer: TfidfVectorizer  # Text to feature conversion
- classifier_model: LogisticRegression  # Class prediction
- regressor_model: LinearRegression     # Score prediction
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*
### Property Reflection

After reviewing the prework analysis, I identified several areas where properties can be consolidated:

- Properties 3.3 and 3.4 (classifier and regressor output constraints) are both testing model output validity and can be combined
- Properties 4.2 and 4.5 (API response format and headers) both test API response correctness and can be combined  
- Properties 1.2 and 1.3 (processing input and returning results) test the same core prediction workflow and can be combined
- Properties 3.1 and 3.2 (text combination and vectorization) are sequential steps in the same pipeline and can be combined

The remaining properties provide unique validation value and should be kept separate.

### Correctness Properties

**Property 1: Prediction workflow completeness**
*For any* valid problem input (description, input_desc, output_desc), the prediction workflow should return both a valid class (Easy/Medium/Hard) and a numeric score (0-100)
**Validates: Requirements 1.2, 1.3**

**Property 2: Error handling consistency**  
*For any* invalid or malformed input, the application should return appropriate error responses without crashing
**Validates: Requirements 1.4, 4.4**

**Property 3: Static file URL generation**
*For any* static file reference in templates, the generated URLs should be valid and accessible
**Validates: Requirements 2.1**

**Property 4: ML pipeline text processing**
*For any* text input, the ML pipeline should successfully combine text fields and convert them to numerical feature vectors
**Validates: Requirements 3.1, 3.2**

**Property 5: Model output constraints**
*For any* input to the trained models, the classifier should output one of {Easy, Medium, Hard} and the regressor should output a value in [0, 100]
**Validates: Requirements 3.3, 3.4**

**Property 6: API request processing**
*For any* valid JSON request to /predict endpoint, the response should be valid JSON containing class and score fields with proper content-type headers
**Validates: Requirements 4.1, 4.2, 4.5**

**Property 7: Text preprocessing robustness**
*For any* text input including special characters and various encodings, the preprocessing should handle it without errors
**Validates: Requirements 5.2**

**Property 8: Input validation consistency**
*For any* input to ML models, the system should validate format and feature space compatibility before processing
**Validates: Requirements 5.4**

**Property 9: Error logging completeness**
*For any* error condition that occurs, the application should log appropriate debugging information
**Validates: Requirements 5.5**

## Error Handling

### Input Validation
- **Empty/Null Inputs**: Return HTTP 400 with descriptive error message
- **Malformed JSON**: Return HTTP 400 with JSON parsing error details
- **Missing Required Fields**: Return HTTP 400 listing missing fields
- **Invalid Data Types**: Return HTTP 400 with type validation errors

### ML Model Errors
- **Vectorization Failures**: Log error and return HTTP 500 with generic error message
- **Prediction Failures**: Log model state and return HTTP 500
- **Model Loading Errors**: Fail application startup with clear error message

### System Errors
- **Memory Issues**: Implement graceful degradation and logging
- **File System Errors**: Log detailed error information for debugging
- **Network Errors**: Return appropriate HTTP status codes

### Error Response Format
```json
{
    "error": "Brief error description",
    "message": "Detailed error message for debugging",
    "status_code": 400
}
```

## Testing Strategy

### Dual Testing Approach

The application will use both unit testing and property-based testing to ensure comprehensive coverage:

- **Unit tests** verify specific examples, edge cases, and error conditions
- **Property tests** verify universal properties that should hold across all inputs
- Together they provide comprehensive coverage: unit tests catch concrete bugs, property tests verify general correctness

### Unit Testing

Unit tests will cover:
- Specific examples of successful predictions with known inputs/outputs
- Edge cases like empty strings, very long text, special characters
- Error conditions like malformed JSON, missing fields
- Integration points between Flask routes and ML components
- Template rendering with specific data sets

### Property-Based Testing

Property-based testing will use **Hypothesis** for Python to verify the correctness properties defined above. Each property-based test will:
- Run a minimum of 100 iterations with randomly generated inputs
- Be tagged with comments explicitly referencing the correctness property from this design document
- Use the format: '**Feature: flask-ml-webapp, Property {number}: {property_text}**'
- Generate smart test data that constrains to valid input spaces

Key property test areas:
- Text processing pipeline with various input combinations
- Model output validation across diverse text inputs  
- API endpoint behavior with different JSON payloads
- Error handling with various types of invalid input
- Static file URL generation with different asset names

### Test Data Generation

For property-based tests, generators will create:
- **Text Generators**: Various lengths, character sets, encoding types
- **JSON Generators**: Valid and invalid JSON structures for API testing
- **Problem Data Generators**: Realistic programming problem descriptions
- **Edge Case Generators**: Empty strings, very long text, special characters

### Testing Framework Requirements

- **Unit Testing**: pytest for test discovery and execution
- **Property-Based Testing**: Hypothesis for generating test cases
- **Web Testing**: Flask test client for route testing
- **ML Testing**: scikit-learn test utilities for model validation
- **Coverage**: pytest-cov for test coverage reporting

Each property-based test must be tagged with the exact format specified and implement only the numbered properties from this design document. Tests will focus on core logic without mocking to maintain simplicity and real functionality validation.