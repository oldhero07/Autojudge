# Requirements Document

## Introduction

Enhancement of the existing AutoJudge system to fully comply with the AutoJudge research paper specifications. The system currently provides programming problem difficulty prediction but needs modifications to match the exact UI structure, evaluation metrics, and documentation requirements specified in the research paper.

## Glossary

- **AutoJudge_System**: The enhanced web application that predicts programming problem difficulty
- **Three_Input_UI**: The user interface with separate text boxes for problem description, input description, and output description
- **Evaluation_Module**: The component that calculates and displays model performance metrics
- **Model_Evaluator**: The service that computes accuracy, confusion matrix, MAE, and RMSE metrics
- **Documentation_Generator**: The component that generates comprehensive methodology documentation

## Requirements

### Requirement 1

**User Story:** As a user following the AutoJudge research paper, I want to input problem details in three separate text boxes, so that I can provide structured input as specified in the research requirements.

#### Acceptance Criteria

1. WHEN a user accesses the main interface THEN the AutoJudge_System SHALL display three separate text input areas for problem description, input description, and output description
2. WHEN a user enters text in any of the three input fields THEN the AutoJudge_System SHALL update real-time feature analysis for the combined text
3. WHEN a user submits the form THEN the AutoJudge_System SHALL combine all three text fields and process them through the ML pipeline
4. WHEN any of the three input fields are empty THEN the AutoJudge_System SHALL still process the prediction using available text
5. WHEN the prediction is made THEN the AutoJudge_System SHALL display both the predicted class and numerical score

### Requirement 2

**User Story:** As a researcher validating the AutoJudge methodology, I want to see comprehensive evaluation metrics, so that I can assess model performance using standard ML evaluation criteria.

#### Acceptance Criteria

1. WHEN the application initializes THEN the Model_Evaluator SHALL perform proper train/test split on the dataset
2. WHEN model training completes THEN the Evaluation_Module SHALL calculate classification accuracy and generate confusion matrix
3. WHEN regression model evaluation occurs THEN the Evaluation_Module SHALL compute Mean Absolute Error (MAE) and Root Mean Square Error (RMSE)
4. WHEN evaluation metrics are calculated THEN the AutoJudge_System SHALL display or log the results for verification
5. WHEN the application starts THEN the AutoJudge_System SHALL validate that evaluation metrics meet acceptable thresholds

### Requirement 3

**User Story:** As a developer or researcher, I want comprehensive documentation explaining the methodology and results, so that I can understand and reproduce the AutoJudge approach.

#### Acceptance Criteria

1. WHEN documentation is generated THEN the Documentation_Generator SHALL include a detailed methodology section explaining the approach
2. WHEN evaluation results are available THEN the documentation SHALL include performance metrics with interpretation
3. WHEN feature engineering is documented THEN the documentation SHALL explain TF-IDF vectorization and custom feature extraction
4. WHEN model selection is documented THEN the documentation SHALL justify the choice of Logistic Regression and Random Forest
5. WHEN the README is updated THEN it SHALL follow the AutoJudge research paper structure and requirements

### Requirement 4

**User Story:** As a user, I want backward compatibility with the existing single-input approach, so that current functionality remains available while new features are added.

#### Acceptance Criteria

1. WHEN the API receives a request with only description field THEN the AutoJudge_System SHALL process it using the existing logic
2. WHEN the API receives a request with three separate fields THEN the AutoJudge_System SHALL combine them appropriately
3. WHEN the frontend is updated THEN existing API endpoints SHALL continue to function without breaking changes
4. WHEN users access the application THEN they SHALL be able to use either the new three-input format or paste combined text
5. WHEN predictions are made THEN the response format SHALL remain consistent regardless of input method

### Requirement 5

**User Story:** As a system administrator, I want enhanced error handling and logging for the improved evaluation system, so that model performance issues can be diagnosed and resolved.

#### Acceptance Criteria

1. WHEN train/test split fails THEN the AutoJudge_System SHALL log detailed error information and gracefully degrade
2. WHEN evaluation metrics calculation fails THEN the Evaluation_Module SHALL provide fallback metrics and error reporting
3. WHEN model performance is below thresholds THEN the AutoJudge_System SHALL log warnings and continue operation
4. WHEN documentation generation encounters errors THEN the Documentation_Generator SHALL report specific issues
5. WHEN the application encounters any evaluation-related errors THEN appropriate error messages SHALL be displayed to users