# Requirements Document

## Introduction

A Flask web application that uses machine learning to classify programming problems and predict their difficulty scores. The system will accept problem descriptions and provide automated classification and scoring based on trained models.

## Glossary

- **Flask_App**: The web application built using the Flask framework
- **ML_Backend**: The machine learning component containing trained models for classification and regression
- **Problem_Classifier**: The logistic regression model that predicts problem difficulty class
- **Score_Predictor**: The linear regression model that predicts numeric problem scores
- **TfidfVectorizer**: The text vectorization component that converts text to numerical features
- **Problem_Data**: The dataset containing programming problem information with descriptions, input/output formats, classes, and scores

## Requirements

### Requirement 1

**User Story:** As a developer, I want to submit programming problem details through a web interface, so that I can get automated classification and difficulty scoring.

#### Acceptance Criteria

1. WHEN a user accesses the root URL THEN the Flask_App SHALL display an HTML form interface for problem submission
2. WHEN a user submits problem details via the form THEN the Flask_App SHALL process the input and return predictions
3. WHEN the prediction results are returned THEN the Flask_App SHALL display both the problem class and numeric score
4. WHEN invalid or empty data is submitted THEN the Flask_App SHALL handle the error gracefully and provide user feedback
5. WHEN the application starts THEN the Flask_App SHALL load pre-trained models and be ready to serve predictions

### Requirement 2

**User Story:** As a system administrator, I want the application to have a proper file structure, so that static assets are served correctly and the codebase is maintainable.

#### Acceptance Criteria

1. WHEN the Flask_App serves static files THEN the system SHALL use Flask's static file serving with correct URL routing
2. WHEN the Flask_App renders templates THEN the system SHALL use Flask's template engine with proper template inheritance
3. WHEN CSS and JavaScript files are referenced THEN the HTML SHALL use Flask's url_for syntax for asset linking
4. WHEN the application structure is examined THEN the system SHALL follow Flask conventions with templates and static folders
5. WHEN the application is deployed THEN the Flask_App SHALL serve on port 5000 by default

### Requirement 3

**User Story:** As a data scientist, I want the system to use machine learning models for predictions, so that problem classification and scoring are automated and consistent.

#### Acceptance Criteria

1. WHEN text input is processed THEN the ML_Backend SHALL combine description, input_desc, and output_desc into a single feature vector
2. WHEN feature extraction occurs THEN the TfidfVectorizer SHALL convert text data to numerical representations
3. WHEN classification is requested THEN the Problem_Classifier SHALL predict one of three classes: Easy, Medium, or Hard
4. WHEN score prediction is requested THEN the Score_Predictor SHALL return a numeric value between 0 and 100
5. WHEN models are trained THEN the ML_Backend SHALL use a dataset with at least 20 representative samples

### Requirement 4

**User Story:** As an API consumer, I want to interact with the prediction service programmatically, so that I can integrate the functionality into other applications.

#### Acceptance Criteria

1. WHEN a POST request is made to /predict endpoint THEN the Flask_App SHALL accept JSON payload with problem details
2. WHEN valid JSON is received THEN the Flask_App SHALL return predictions in JSON format with class and score fields
3. WHEN the prediction API is called THEN the Flask_App SHALL process the request within reasonable time limits
4. WHEN malformed JSON is sent THEN the Flask_App SHALL return appropriate HTTP error codes and error messages
5. WHEN the API response is generated THEN the Flask_App SHALL include proper content-type headers

### Requirement 5

**User Story:** As a developer, I want the system to handle data processing reliably, so that predictions are accurate and the application is robust.

#### Acceptance Criteria

1. WHEN training data is created THEN the Problem_Data SHALL contain exactly 20 rows with all required columns
2. WHEN text preprocessing occurs THEN the ML_Backend SHALL handle special characters and encoding issues appropriately
3. WHEN model training completes THEN the ML_Backend SHALL validate model performance before deployment
4. WHEN predictions are made THEN the ML_Backend SHALL ensure input data matches the expected format and feature space
5. WHEN the application encounters errors THEN the Flask_App SHALL log appropriate error information for debugging