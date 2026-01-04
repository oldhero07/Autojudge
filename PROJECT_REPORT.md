# AutoJudge: Programming Problem Difficulty Prediction System
## Comprehensive Project Report

### Executive Summary

AutoJudge is a machine learning system designed to automatically classify programming problems into difficulty categories (Easy, Medium, Hard) and predict numerical difficulty scores on a 1-10 scale. The system combines natural language processing with domain-specific feature engineering to achieve 55.0% classification accuracy on a dataset of 4,112 programming problems.

### 1. Problem Statement

**Objective**: Develop an automated system to predict the difficulty of programming problems based on their textual descriptions.

**Motivation**: 
- Manual difficulty assessment is time-consuming and subjective
- Competitive programming platforms need consistent difficulty ratings
- Educational institutions require automated problem categorization
- Coding interview platforms need difficulty calibration

**Success Criteria**:
- Classification accuracy > 50%
- Mean Absolute Error (MAE) < 2.0 for score prediction
- Real-time prediction capability (< 100ms response time)
- Production-ready web interface

### 2. Dataset Analysis

**Dataset Characteristics**:
- **Total Samples**: 4,112 programming problems
- **Source**: Competitive programming platforms and educational resources
- **Format**: JSONL (JSON Lines) with structured problem descriptions

**Class Distribution**:
- Hard: 1,941 problems (47.2%)
- Medium: 1,405 problems (34.2%) 
- Easy: 766 problems (18.6%)

**Score Distribution**:
- Range: 1.1 - 9.7 (normalized to 1-10 scale)
- Mean: 5.11 ± 2.18
- Distribution shows slight bias toward higher difficulty scores

**Data Quality Assessment**:
- No missing values in critical fields
- Consistent formatting across all samples
- Balanced representation of different problem domains
- Sufficient sample size for statistical significance

### 3. Methodology

#### 3.1 Feature Engineering

**Text Processing Pipeline**:
1. **Preprocessing**: Lowercasing, whitespace normalization
2. **Abbreviation Expansion**: Programming-specific terms (dfs → depth first search)
3. **TF-IDF Vectorization**: 4,000 features with n-gram range (1,3)
4. **Feature Selection**: Chi-square test reducing to 3,000 features

**Custom Domain Features (15 features)**:
1. **Text Metrics**: Length, word count, vocabulary richness
2. **Algorithm Indicators**: Graph algorithms, dynamic programming, data structures
3. **Complexity Markers**: Sorting/searching, string processing, mathematical content
4. **Problem Characteristics**: Constraints, optimization keywords, test case patterns

**Feature Combination**:
- Total Features: 3,015 (3,000 TF-IDF + 15 custom)
- Sparse matrix concatenation for memory efficiency
- StandardScaler normalization for custom features

#### 3.2 Model Architecture

**Classification System**:
- **Type**: Voting Classifier (Ensemble)
- **Components**:
  - Logistic Regression (C=2.0, balanced class weights)
  - Random Forest Classifier (400 estimators, max_depth=35)
  - Gradient Boosting Classifier (300 estimators, max_depth=12)
- **Voting Strategy**: Soft voting for probability-based decisions

**Regression System**:
- **Type**: Random Forest Regressor
- **Parameters**:
  - n_estimators: 350
  - max_depth: 30
  - min_samples_split: 5
  - min_samples_leaf: 2
  - max_features: sqrt

#### 3.3 Training Process

**Data Split**:
- Training: 3,289 samples (80%)
- Testing: 823 samples (20%)
- Stratified split to maintain class distribution

**Class Balancing**:
- Computed class weights for imbalanced dataset
- Applied balanced class weights in Logistic Regression
- Ensemble approach to mitigate individual model biases

**Model Validation**:
- Cross-validation during hyperparameter tuning
- Hold-out test set for final evaluation
- Performance monitoring and threshold validation

### 4. Results and Evaluation

#### 4.1 Classification Performance

**Overall Metrics**:
- **Accuracy**: 55.0%
- **Weighted Precision**: 0.540
- **Weighted Recall**: 0.550
- **Weighted F1-Score**: 0.542

**Confusion Matrix**:
```
                 Predicted
Actual      Easy  Medium  Hard   Total
Easy         69     49     35     153
Medium       36    107    138     281
Hard         25     87    277     389
Total       130    243    450     823
```

**Per-Class Analysis**:
- **Easy Problems**: Precision=0.531, Recall=0.451, F1=0.488
- **Medium Problems**: Precision=0.440, Recall=0.381, F1=0.408
- **Hard Problems**: Precision=0.616, Recall=0.712, F1=0.660

**Key Observations**:
- Best performance on Hard problems (highest F1-score: 0.660)
- Moderate performance on Easy problems
- Challenging classification for Medium problems (lowest F1-score: 0.408)
- Class imbalance affects overall performance

#### 4.2 Regression Performance

**Score Prediction Metrics**:
- **Mean Absolute Error (MAE)**: 1.735 points
- **Root Mean Square Error (RMSE)**: 2.071 points
- **R² Score**: 0.116

**Per-Class Score Prediction**:
- **Easy**: MAE=2.267, Actual=1.99±0.43, Predicted=4.25±0.54
- **Medium**: MAE=0.817, Actual=4.13±0.75, Predicted=4.71±0.52
- **Hard**: MAE=2.189, Actual=7.11±1.13, Predicted=4.92±0.48

**Analysis**:
- Best score prediction for Medium difficulty problems
- Tendency to predict scores toward the middle range (4-5)
- Difficulty in extreme score prediction (very easy or very hard)

### 5. System Implementation

#### 5.1 Web Application Architecture

**Backend (Flask)**:
- RESTful API with JSON responses
- Model persistence and caching
- Comprehensive error handling
- Health monitoring endpoints

**Frontend (HTML/JavaScript)**:
- Responsive web interface
- Real-time prediction display
- Input validation and user feedback
- Mobile-friendly design

**API Endpoints**:
- `POST /predict`: Legacy format prediction
- `POST /predict/structured`: Three-field input format
- `GET /health`: System status monitoring
- `GET /`: Web interface access

#### 5.2 Production Features

**Performance Optimization**:
- Model persistence (111MB trained models)
- Fast startup with pre-trained components
- Response time: ~50ms per prediction
- Memory usage: ~200MB loaded

**Deployment Ready**:
- Docker containerization
- Docker Compose for multi-service setup
- Environment variable configuration
- Production WSGI server (Gunicorn)

**Monitoring and Reliability**:
- Health check endpoints
- Error logging and tracking
- Input validation and sanitization
- Graceful error handling

### 6. Technical Challenges and Solutions

#### 6.1 Class Imbalance
**Challenge**: Dataset heavily skewed toward Hard problems (47.2%)
**Solution**: 
- Balanced class weights in ensemble components
- Soft voting to leverage probability distributions
- Ensemble approach to reduce individual model bias

#### 6.2 Feature Engineering
**Challenge**: Converting text to meaningful numerical features
**Solution**:
- Domain-specific feature extraction
- TF-IDF with programming-specific preprocessing
- Chi-square feature selection for dimensionality reduction

#### 6.3 Model Generalization
**Challenge**: Avoiding overfitting on training data
**Solution**:
- Ensemble methods for robust predictions
- Cross-validation during development
- Conservative hyperparameter tuning

#### 6.4 Production Deployment
**Challenge**: Real-time prediction requirements
**Solution**:
- Model persistence for fast startup
- Efficient sparse matrix operations
- Caching and optimization strategies

### 7. Limitations and Future Work

#### 7.1 Current Limitations

**Model Performance**:
- 55% accuracy leaves room for improvement
- Limited R² score (0.116) for regression
- Bias toward middle-range score predictions

**Data Limitations**:
- English-only problem descriptions
- Competitive programming domain focus
- Class imbalance affects minority classes

**Technical Constraints**:
- Memory requirements for large feature matrices
- Processing time for complex text preprocessing
- Dependency on specific problem description formats

#### 7.2 Future Enhancements

**Model Improvements**:
- Advanced NLP models (BERT, transformers)
- Expanded feature engineering with semantic embeddings
- Deep learning approaches for text classification
- Active learning for continuous improvement

**Data Expansion**:
- Multi-language support
- Broader problem domain coverage
- Synthetic data generation for class balancing
- User feedback integration for model refinement

**System Enhancements**:
- Real-time model retraining
- A/B testing framework for model versions
- Advanced caching and performance optimization
- Comprehensive analytics dashboard

### 8. Conclusion

AutoJudge successfully demonstrates the feasibility of automated programming problem difficulty prediction using machine learning techniques. The system achieves 55.0% classification accuracy and provides a production-ready web interface for real-time predictions.

**Key Achievements**:
- ✅ Exceeded 50% accuracy threshold
- ✅ Achieved MAE < 2.0 for score prediction
- ✅ Implemented production-ready system
- ✅ Created comprehensive documentation
- ✅ Established deployment pipeline

**Impact and Applications**:
- Automated problem categorization for educational platforms
- Difficulty calibration for competitive programming
- Assessment tool for coding interviews
- Research foundation for educational technology

The project provides a solid foundation for future enhancements and demonstrates practical application of machine learning in educational technology. The modular architecture and comprehensive documentation enable continued development and improvement.

### 9. References and Resources

**Technical Documentation**:
- scikit-learn: Machine Learning in Python
- Flask: Web Development Framework
- TF-IDF: Term Frequency-Inverse Document Frequency
- Ensemble Methods: Voting Classifiers and Random Forests

**Dataset Sources**:
- Competitive programming platforms
- Educational problem repositories
- Open source problem collections

**Development Tools**:
- Python 3.8+ for backend development
- Docker for containerization
- Git for version control
- GitHub for repository hosting

---

**Project Repository**: https://github.com/oldhero07/Autojudge
**Documentation**: Complete README and deployment guides included
**Live Demo**: Available via Docker deployment