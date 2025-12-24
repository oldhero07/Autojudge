"""
Flask ML Web Application

A web application that uses machine learning to classify programming problems
and predict their difficulty scores.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier, VotingRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import scipy.sparse
import re
import os
import logging
import traceback
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Tuple, Any
from evaluation_models import ClassificationMetrics, RegressionMetrics, EvaluationReport
from documentation_generator import DocumentationGenerator, DocumentationConfig
from error_handler import ErrorHandler, SystemComponent, ErrorSeverity, error_handler

# Initialize Flask application
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'

# Global variables for trained models and scalers
classifier_pipeline = None
regressor_pipeline = None
tfidf_vectorizer = None
feature_scaler = None
model_evaluator = None
documentation_generator = None

class ModelEvaluator:
    """
    Comprehensive model evaluation class for AutoJudge system.
    Implements train/test split and calculates all required metrics.
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train_class = None
        self.y_test_class = None
        self.y_train_score = None
        self.y_test_score = None
        self.split_info = {}
        
        # Performance thresholds
        self.min_accuracy = 0.6  # Minimum acceptable accuracy
        self.max_mae = 2.0       # Maximum acceptable MAE
        self.max_rmse = 2.5      # Maximum acceptable RMSE
    
    def perform_train_test_split(self, X_combined, y_class, y_score):
        """
        Perform train/test split on the dataset with enhanced error handling.
        
        Args:
            X_combined: Combined feature matrix (TF-IDF + custom features)
            y_class: Classification targets
            y_score: Regression targets
            
        Returns:
            Tuple of (X_train, X_test, y_train_class, y_test_class, y_train_score, y_test_score)
        """
        def _perform_split():
            # Validate inputs
            if X_combined is None or len(y_class) == 0 or len(y_score) == 0:
                raise ValueError("Invalid input data for train/test split")
                
            if X_combined.shape[0] != len(y_class) or X_combined.shape[0] != len(y_score):
                raise ValueError("Mismatched data dimensions")
            
            # Check minimum data requirements
            unique_classes = y_class.nunique()
            min_samples_per_class = int(1 / self.test_size) + 1
            
            if len(y_class) < min_samples_per_class * unique_classes:
                raise ValueError(f"Insufficient data: need at least {min_samples_per_class * unique_classes} samples for {unique_classes} classes")
            
            # Perform stratified split to maintain class distribution
            self.X_train, self.X_test, self.y_train_class, self.y_test_class = train_test_split(
                X_combined, y_class, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=y_class
            )
            
            # Split regression targets with same indices
            _, _, self.y_train_score, self.y_test_score = train_test_split(
                X_combined, y_score,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_class  # Use same stratification
            )
            
            # Store split information
            self.split_info = {
                'train_size': self.X_train.shape[0],
                'test_size': self.X_test.shape[0],
                'train_class_distribution': self.y_train_class.value_counts().to_dict(),
                'test_class_distribution': self.y_test_class.value_counts().to_dict(),
                'feature_count': self.X_train.shape[1]
            }
            
            app.logger.info(f"Train/test split completed: {self.split_info['train_size']} train, {self.split_info['test_size']} test samples")
            return self.X_train, self.X_test, self.y_train_class, self.y_test_class, self.y_train_score, self.y_test_score
        
        # Use error handler for safe execution
        context = {
            'data_shape': X_combined.shape if X_combined is not None else None,
            'class_count': len(y_class) if y_class is not None else 0,
            'score_count': len(y_score) if y_score is not None else 0,
            'test_size': self.test_size
        }
        
        result = error_handler.safe_execute(
            func=_perform_split,
            component=SystemComponent.MODEL_EVALUATION,
            fallback_value=None,
            context=context
        )
        
        if result is None:
            raise RuntimeError("Train/test split failed and no fallback available")
            
        return result
    
    def evaluate_classification(self, model, X_test=None, y_test=None) -> ClassificationMetrics:
        """
        Evaluate classification model performance with enhanced error handling.
        
        Args:
            model: Trained classification model
            X_test: Test features (optional, uses stored if None)
            y_test: Test labels (optional, uses stored if None)
            
        Returns:
            ClassificationMetrics object
        """
        def _evaluate_classification():
            if X_test is None:
                X_test_eval = self.X_test
            else:
                X_test_eval = X_test
                
            if y_test is None:
                y_test_eval = self.y_test_class
            else:
                y_test_eval = y_test
            
            # Validate inputs
            if model is None:
                raise ValueError("Model is None - cannot perform evaluation")
            if X_test_eval is None or y_test_eval is None:
                raise ValueError("Test data is None - cannot perform evaluation")
            if len(y_test_eval) == 0:
                raise ValueError("Empty test set - cannot perform evaluation")
                
            # Make predictions
            y_pred = model.predict(X_test_eval)
            
            # Validate predictions
            if len(y_pred) != len(y_test_eval):
                raise ValueError("Prediction length mismatch")
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_eval, y_pred)
            conf_matrix = confusion_matrix(y_test_eval, y_pred)
            class_report = classification_report(y_test_eval, y_pred, output_dict=True, zero_division=0)
            
            # Validate metrics
            if not (0.0 <= accuracy <= 1.0):
                raise ValueError(f"Invalid accuracy value: {accuracy}")
            
            app.logger.info(f"Classification evaluation - Accuracy: {accuracy:.3f}")
            
            # Monitor performance
            error_handler.monitor_performance('accuracy', accuracy)
            
            return ClassificationMetrics(
                accuracy=accuracy,
                confusion_matrix=conf_matrix,
                classification_report=class_report
            )
        
        # Use error handler for safe execution
        context = {
            'model_type': type(model).__name__ if model else None,
            'test_samples': len(y_test) if y_test is not None else (len(self.y_test_class) if self.y_test_class is not None else 0)
        }
        
        result = error_handler.safe_execute(
            func=_evaluate_classification,
            component=SystemComponent.MODEL_EVALUATION,
            fallback_value=error_handler.get_fallback_classification_metrics("Classification evaluation failed"),
            context=context
        )
        
        return result
    
    def evaluate_regression(self, model, X_test=None, y_test=None) -> RegressionMetrics:
        """
        Evaluate regression model performance with enhanced error handling.
        
        Args:
            model: Trained regression model
            X_test: Test features (optional, uses stored if None)
            y_test: Test targets (optional, uses stored if None)
            
        Returns:
            RegressionMetrics object
        """
        def _evaluate_regression():
            if X_test is None:
                X_test_eval = self.X_test
            else:
                X_test_eval = X_test
                
            if y_test is None:
                y_test_eval = self.y_test_score
            else:
                y_test_eval = y_test
            
            # Validate inputs
            if model is None:
                raise ValueError("Model is None - cannot perform evaluation")
            if X_test_eval is None or y_test_eval is None:
                raise ValueError("Test data is None - cannot perform evaluation")
            if len(y_test_eval) == 0:
                raise ValueError("Empty test set - cannot perform evaluation")
                
            # Make predictions
            y_pred = model.predict(X_test_eval)
            
            # Validate predictions
            if len(y_pred) != len(y_test_eval):
                raise ValueError("Prediction length mismatch")
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                raise ValueError("Invalid predictions (NaN or Inf values)")
            
            # Calculate metrics
            mae = mean_absolute_error(y_test_eval, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_eval, y_pred))
            r2 = r2_score(y_test_eval, y_pred)
            
            # Validate metrics
            if mae < 0 or rmse < 0:
                raise ValueError(f"Invalid error metrics: MAE={mae}, RMSE={rmse}")
            if rmse < mae:
                app.logger.warning(f"RMSE ({rmse:.3f}) is less than MAE ({mae:.3f}) - unusual but possible")
            
            app.logger.info(f"Regression evaluation - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
            
            # Monitor performance
            error_handler.monitor_performance('mae', mae, comparison='max')
            error_handler.monitor_performance('rmse', rmse, comparison='max')
            error_handler.monitor_performance('r2', r2)
            
            return RegressionMetrics(
                mae=mae,
                rmse=rmse,
                r2_score=r2
            )
        
        # Use error handler for safe execution
        context = {
            'model_type': type(model).__name__ if model else None,
            'test_samples': len(y_test) if y_test is not None else (len(self.y_test_score) if self.y_test_score is not None else 0)
        }
        
        result = error_handler.safe_execute(
            func=_evaluate_regression,
            component=SystemComponent.MODEL_EVALUATION,
            fallback_value=error_handler.get_fallback_regression_metrics("Regression evaluation failed"),
            context=context
        )
        
        return result
    
    def validate_performance_thresholds(self, classification_metrics: ClassificationMetrics, 
                                      regression_metrics: RegressionMetrics) -> Dict[str, Any]:
        """
        Validate model performance against acceptable thresholds with enhanced monitoring.
        
        Args:
            classification_metrics: Classification evaluation results
            regression_metrics: Regression evaluation results
            
        Returns:
            Dictionary with validation results, alerts, and recommendations
        """
        # Use error handler for comprehensive performance validation
        validation_results = error_handler.validate_model_performance(
            classification_metrics, regression_metrics
        )
        
        # Add traditional threshold checks for backward compatibility
        traditional_results = {
            'accuracy_acceptable': classification_metrics.accuracy >= self.min_accuracy,
            'mae_acceptable': regression_metrics.mae <= self.max_mae,
            'rmse_acceptable': regression_metrics.rmse <= self.max_rmse,
            'overall_acceptable': True
        }
        
        # Check overall acceptability
        traditional_results['overall_acceptable'] = all([
            traditional_results['accuracy_acceptable'],
            traditional_results['mae_acceptable'],
            traditional_results['rmse_acceptable']
        ])
        
        # Log warnings for poor performance
        if not traditional_results['accuracy_acceptable']:
            app.logger.warning(f"Classification accuracy {classification_metrics.accuracy:.3f} below threshold {self.min_accuracy}")
        if not traditional_results['mae_acceptable']:
            app.logger.warning(f"Regression MAE {regression_metrics.mae:.3f} above threshold {self.max_mae}")
        if not traditional_results['rmse_acceptable']:
            app.logger.warning(f"Regression RMSE {regression_metrics.rmse:.3f} above threshold {self.max_rmse}")
            
        if traditional_results['overall_acceptable']:
            app.logger.info("All performance metrics meet acceptable thresholds")
        else:
            app.logger.warning("Some performance metrics are below acceptable thresholds - continuing with warnings")
        
        # Combine results
        combined_results = {
            **traditional_results,
            'enhanced_validation': validation_results,
            'system_health': error_handler.get_system_health_report()
        }
        
        return combined_results
    
    def generate_evaluation_report(self, classification_metrics: ClassificationMetrics,
                                 regression_metrics: RegressionMetrics) -> EvaluationReport:
        """
        Generate comprehensive evaluation report.
        
        Args:
            classification_metrics: Classification evaluation results
            regression_metrics: Regression evaluation results
            
        Returns:
            EvaluationReport object
        """
        dataset_info = {
            'total_samples': self.split_info.get('train_size', 0) + self.split_info.get('test_size', 0),
            'train_samples': self.split_info.get('train_size', 0),
            'test_samples': self.split_info.get('test_size', 0),
            'feature_count': self.split_info.get('feature_count', 0),
            'class_distribution': self.split_info.get('train_class_distribution', {})
        }
        
        model_info = {
            'classification_model': 'LogisticRegression',
            'regression_model': 'RandomForestRegressor',
            'test_size': self.test_size,
            'random_state': self.random_state
        }
        
        return EvaluationReport(
            classification_metrics=classification_metrics,
            regression_metrics=regression_metrics,
            dataset_info=dataset_info,
            model_info=model_info
        )

def extract_custom_features(text):
    """Extract production-ready custom features focused on generalization."""
    # Preprocess text
    if not text or pd.isna(text):
        return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    
    # Conservative abbreviation expansion
    text = re.sub(r'\bdfs\b', 'depth first search', text)
    text = re.sub(r'\bbfs\b', 'breadth first search', text)
    text = re.sub(r'\bdp\b', 'dynamic programming', text)
    
    text = text.strip()
    
    # Core metrics
    text_len = len(text)
    word_count = len(text.split()) if text else 0
    
    # Algorithm indicators (conservative, high-confidence patterns)
    graph_algorithms = len(re.findall(r'\b(graph|tree|node|edge|path|dfs|bfs|depth first|breadth first)\b', text))
    dynamic_programming = len(re.findall(r'\b(dynamic programming|dp|memoization|optimal substructure)\b', text))
    data_structures = len(re.findall(r'\b(heap|stack|queue|hash|trie|segment tree)\b', text))
    sorting_searching = len(re.findall(r'\b(sort|binary search|merge|quick)\b', text))
    string_processing = len(re.findall(r'\b(string|substring|character|palindrome)\b', text))
    
    # Mathematical content (conservative)
    basic_math = len(re.findall(r'[+\-*/=<>]', text))
    advanced_math = len(re.findall(r'[∑∏∫∂∆√π∞≤≥]', text))
    complexity_notation = len(re.findall(r'o\([^)]+\)', text, re.IGNORECASE))
    
    # Problem complexity indicators
    constraints = len(re.findall(r'constraint|limit|\d+\s*≤.*≤\s*\d+', text))
    optimization = len(re.findall(r'minimum|maximum|optimal|best', text))
    multiple_cases = len(re.findall(r'test.*case|multiple.*test', text))
    
    # Linguistic features (simple and robust)
    if word_count > 0:
        unique_words = len(set(text.split()))
        vocabulary_richness = unique_words / word_count
        avg_word_length = sum(len(word) for word in text.split()) / word_count
    else:
        vocabulary_richness = 0
        avg_word_length = 0
    
    return (text_len, word_count, graph_algorithms, dynamic_programming, data_structures,
            sorting_searching, string_processing, basic_math, advanced_math, complexity_notation,
            constraints, optimization, multiple_cases, vocabulary_richness, avg_word_length)

def load_and_preprocess_data():
    """Load data from problems_data.jsonl and preprocess it."""
    try:
        # Load the dataset
        df = pd.read_json('../problems_data.jsonl', lines=True)
        
        # Print score statistics for verification
        print("\n" + "="*50)
        print("DATASET INSPECTION")
        print("="*50)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nProblem score statistics:")
        print(df['problem_score'].describe())
        print(f"Score range: {df['problem_score'].min()} to {df['problem_score'].max()}")
        print("="*50 + "\n")
        
        # Map column names to expected format
        df = df.rename(columns={
            'input_description': 'input_desc',
            'output_description': 'output_desc'
        })
        
        # Better score scaling - keep original 1-10 scale but map to difficulty ranges
        # Original range: 1.1 to 9.7, we'll keep it as 1-10 scale for intuitive understanding
        # Easy: 1-4, Medium: 4-7, Hard: 7-10
        df['problem_score_scaled'] = df['problem_score']  # Keep original scale
        
        print(f"SCORE SCALING: Keeping original 1-10 scale for intuitive difficulty understanding")
        print(f"Score distribution by class:")
        for cls in ['easy', 'medium', 'hard']:
            if cls in df['problem_class'].values:
                subset = df[df['problem_class'] == cls]
                print(f"  {cls}: {subset['problem_score'].min():.1f} - {subset['problem_score'].max():.1f} (avg: {subset['problem_score'].mean():.1f})")
        print()
        
        # Combine text features
        df['combined_text'] = df['description'].fillna('') + ' ' + \
                             df['input_desc'].fillna('') + ' ' + \
                             df['output_desc'].fillna('')
        
        # Clean the combined text
        df['combined_text'] = df['combined_text'].str.strip()
        
        # Extract custom features
        print("FEATURE ENGINEERING: Extracting production-ready custom features...")
        custom_features = df['combined_text'].apply(extract_custom_features)
        df['text_len'] = [f[0] for f in custom_features]
        df['word_count'] = [f[1] for f in custom_features]
        df['graph_algorithms'] = [f[2] for f in custom_features]
        df['dynamic_programming'] = [f[3] for f in custom_features]
        df['data_structures'] = [f[4] for f in custom_features]
        df['sorting_searching'] = [f[5] for f in custom_features]
        df['string_processing'] = [f[6] for f in custom_features]
        df['basic_math'] = [f[7] for f in custom_features]
        df['advanced_math'] = [f[8] for f in custom_features]
        df['complexity_notation'] = [f[9] for f in custom_features]
        df['constraints'] = [f[10] for f in custom_features]
        df['optimization'] = [f[11] for f in custom_features]
        df['multiple_cases'] = [f[12] for f in custom_features]
        df['vocabulary_richness'] = [f[13] for f in custom_features]
        df['avg_word_length'] = [f[14] for f in custom_features]
        
        print(f"Text length statistics:")
        print(df['text_len'].describe())
        print(f"Math count statistics:")
        print(df['math_count'].describe())
        print(f"Weighted keyword score statistics:")
        print(df['weighted_keyword_score'].describe())
        print(f"Words per sentence statistics:")
        print(df['words_per_sentence'].describe())
        print(f"Unique word ratio statistics:")
        print(df['unique_word_ratio'].describe())
        print(f"Constraint count statistics:")
        print(df['constraint_count'].describe())
        print(f"Graph score statistics:")
        print(df['graph_score'].describe())
        print()
        
        app.logger.info(f"Loaded dataset with {len(df)} rows")
        app.logger.info(f"Problem classes: {df['problem_class'].value_counts().to_dict()}")
        app.logger.info(f"Score range: {df['problem_score'].min():.1f} - {df['problem_score'].max():.1f} (1-10 scale)")
        app.logger.info(f"Optimized features extracted: text_len, math_count, weighted_keyword_score, words_per_sentence, unique_word_ratio, constraint_count, graph_score")
        
        return df
        
    except Exception as e:
        app.logger.error(f"Error loading data: {str(e)}")
        app.logger.error(traceback.format_exc())
        raise

def train_models():
    """Train the ML models using the loaded data with feature engineering and comprehensive evaluation."""
    global classifier_pipeline, regressor_pipeline, tfidf_vectorizer, feature_scaler, model_evaluator, documentation_generator
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Prepare features and targets
        X_text = df['combined_text']
        X_custom = df[['text_len', 'word_count', 'graph_algorithms', 'dynamic_programming', 'data_structures',
                      'sorting_searching', 'string_processing', 'basic_math', 'advanced_math', 'complexity_notation',
                      'constraints', 'optimization', 'multiple_cases', 'vocabulary_richness', 'avg_word_length']].values
        y_class = df['problem_class']
        y_score = df['problem_score_scaled']  # Use original 1-10 scale
        
        # Create and fit production-ready TF-IDF vectorizer
        print("FEATURE ENGINEERING: Creating production-ready TF-IDF features...")
        tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,  # Reduced to prevent overfitting
            stop_words='english', 
            ngram_range=(1, 2),  # Only bigrams
            min_df=5,  # Higher min_df for better generalization
            max_df=0.8,  # More conservative max_df
            sublinear_tf=True  # Apply sublinear tf scaling
        )
        X_tfidf = tfidf_vectorizer.fit_transform(X_text)
        
        # Scale custom features
        print("FEATURE ENGINEERING: Scaling production custom features...")
        feature_scaler = StandardScaler()
        X_custom_scaled = feature_scaler.fit_transform(X_custom)
        
        # Combine TF-IDF and custom features
        print("FEATURE ENGINEERING: Combining TF-IDF and production custom features...")
        X_combined = scipy.sparse.hstack([X_tfidf, scipy.sparse.csr_matrix(X_custom_scaled)])
        
        print(f"Production feature matrix shape: {X_combined.shape}")
        print(f"TF-IDF features: {X_tfidf.shape[1]}")
        print(f"Production custom features: {X_custom_scaled.shape[1]} (15 robust features)")
        print()
        
        # Initialize ModelEvaluator and perform train/test split
        print("MODEL EVALUATION: Performing train/test split...")
        model_evaluator = ModelEvaluator(test_size=0.2, random_state=42)
        X_train, X_test, y_train_class, y_test_class, y_train_score, y_test_score = model_evaluator.perform_train_test_split(
            X_combined, y_class, y_score
        )
        
        # Address class imbalance with SMOTE
        print("CLASS BALANCING: Applying optimized SMOTE to address class imbalance...")
        print(f"Original class distribution: {y_train_class.value_counts().to_dict()}")
        
        # Convert sparse matrix to dense for SMOTE (only if not too large)
        if X_train.shape[1] <= 10000:  # Only if manageable size
            try:
                smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy='auto')
                X_train_balanced, y_train_class_balanced = smote.fit_resample(X_train.toarray(), y_train_class)
                X_train_balanced = scipy.sparse.csr_matrix(X_train_balanced)
                print(f"Balanced class distribution: {pd.Series(y_train_class_balanced).value_counts().to_dict()}")
                
                # Handle regression targets properly
                y_train_score_list = []
                class_indices = {}
                for i, cls in enumerate(y_train_class):
                    if cls not in class_indices:
                        class_indices[cls] = []
                    class_indices[cls].append(i)
                
                for cls in y_train_class_balanced:
                    idx = np.random.choice(class_indices[cls])
                    y_train_score_list.append(y_train_score.iloc[idx])
                
                y_train_score_balanced = pd.Series(y_train_score_list)
                
            except Exception as e:
                print(f"SMOTE failed, using class weights instead: {e}")
                X_train_balanced = X_train
                y_train_class_balanced = y_train_class
                y_train_score_balanced = y_train_score
        else:
            print("Feature matrix too large for SMOTE, using class weights instead")
            X_train_balanced = X_train
            y_train_class_balanced = y_train_class
            y_train_score_balanced = y_train_score
        
        # Calculate class weights for models that support it
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_class), y=y_train_class)
        class_weight_dict = dict(zip(np.unique(y_train_class), class_weights))
        print(f"Computed class weights: {class_weight_dict}")
        
        # Train enhanced classification model with optimized ensemble approach
        app.logger.info("Training optimized classification ensemble...")
        
        # Individual classifiers
        lr_classifier = LogisticRegression(
            random_state=42, 
            max_iter=2000,
            class_weight='balanced',  # Handle class imbalance
            C=1.0,  # Less regularization
            solver='lbfgs'  # Better for multiclass
        )
        
        rf_classifier = RandomForestClassifier(
            n_estimators=250,
            max_depth=25,  # Allow more depth
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Create optimized ensemble classifier
        classifier_pipeline = VotingClassifier(
            estimators=[
                ('lr', lr_classifier),
                ('rf', rf_classifier)
            ],
            voting='soft'  # Use probability-based voting
        )
        
        classifier_pipeline.fit(X_train_balanced, y_train_class_balanced)
        
        # Train optimized regression model
        app.logger.info("Training optimized regression model...")
        
        regressor_pipeline = RandomForestRegressor(
            n_estimators=350,  # Optimized number
            max_depth=30,      # Increased depth
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',  # Better generalization
            random_state=42,
            n_jobs=-1
        )
        
        regressor_pipeline.fit(X_train_balanced, y_train_score_balanced)
        
        # Evaluate models on test set
        print("MODEL EVALUATION: Evaluating classification model...")
        classification_metrics = model_evaluator.evaluate_classification(classifier_pipeline)
        
        print("MODEL EVALUATION: Evaluating regression model...")
        regression_metrics = model_evaluator.evaluate_regression(regressor_pipeline)
        
        # Validate performance thresholds
        print("MODEL EVALUATION: Validating performance thresholds...")
        validation_results = model_evaluator.validate_performance_thresholds(
            classification_metrics, regression_metrics
        )
        
        # Generate comprehensive evaluation report
        evaluation_report = model_evaluator.generate_evaluation_report(
            classification_metrics, regression_metrics
        )
        
        # Initialize documentation generator and set evaluation report
        print("DOCUMENTATION: Initializing documentation generator...")
        doc_config = DocumentationConfig(
            dataset_size=len(df),
            feature_count=X_combined.shape[1],
            tfidf_features=X_tfidf.shape[1],
            custom_features=X_custom_scaled.shape[1]
        )
        documentation_generator = DocumentationGenerator(doc_config)
        documentation_generator.set_evaluation_report(evaluation_report)
        
        # Display evaluation results
        print("\n" + "="*60)
        print("COMPREHENSIVE MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Dataset: {evaluation_report.dataset_info['total_samples']} samples")
        print(f"Train/Test Split: {evaluation_report.dataset_info['train_samples']}/{evaluation_report.dataset_info['test_samples']}")
        print(f"Features: {evaluation_report.dataset_info['feature_count']}")
        print()
        print("CLASSIFICATION METRICS:")
        print(f"  Accuracy: {classification_metrics.accuracy:.3f}")
        print(f"  Confusion Matrix Shape: {classification_metrics.confusion_matrix.shape}")
        print()
        print("REGRESSION METRICS:")
        print(f"  MAE (Mean Absolute Error): {regression_metrics.mae:.3f}")
        print(f"  RMSE (Root Mean Square Error): {regression_metrics.rmse:.3f}")
        print(f"  R² Score: {regression_metrics.r2_score:.3f}")
        print()
        print("PERFORMANCE VALIDATION:")
        for metric, acceptable in validation_results.items():
            status = "✓ PASS" if acceptable else "✗ WARN"
            print(f"  {metric}: {status}")
        print("="*60 + "\n")
        
        # Log comprehensive results
        app.logger.info(f"Model evaluation completed - Accuracy: {classification_metrics.accuracy:.3f}, MAE: {regression_metrics.mae:.3f}, RMSE: {regression_metrics.rmse:.3f}")
        
        # Test predictions on a few samples to verify feature engineering
        sample_indices = [0, 1, 2]
        if X_test.shape[0] >= 3:  # Use shape[0] for sparse matrices instead of len()
            sample_X = X_test[:3]
            sample_classes = classifier_pipeline.predict(sample_X)
            sample_scores = regressor_pipeline.predict(sample_X)
            actual_classes = y_test_class.iloc[:3].tolist()
            actual_scores = y_test_score.iloc[:3].tolist()
            
            app.logger.info("Sample test predictions:")
            for i, (pred_class, pred_score, actual_class, actual_score) in enumerate(zip(
                sample_classes, sample_scores, actual_classes, actual_scores)):
                app.logger.info(f"  Test Sample {i+1}: {pred_class} ({pred_score:.1f}/10) vs actual {actual_class} ({actual_score:.1f}/10)")
        
        app.logger.info("Models trained successfully with comprehensive evaluation!")
        app.logger.info("Documentation generator initialized and ready!")
        
    except Exception as e:
        app.logger.error(f"Error training models: {str(e)}")
        app.logger.error(traceback.format_exc())
        raise

class PredictionService:
    """Service class for making predictions."""
    
    @staticmethod
    def combine_text_features(description, input_desc, output_desc):
        """Combine three text fields into single feature string."""
        # Handle None values
        description = description or ''
        input_desc = input_desc or ''
        output_desc = output_desc or ''
        
        # Combine and clean
        combined = f"{description} {input_desc} {output_desc}".strip()
        return combined
    
    @staticmethod
    def validate_input_format(description, input_desc, output_desc, format_type='legacy'):
        """
        Validate input format based on the expected format type.
        
        Args:
            description: Problem description text
            input_desc: Input description text
            output_desc: Output description text
            format_type: 'legacy' or 'structured'
            
        Returns:
            dict: Validation result with 'valid' boolean and 'message' string
        """
        if format_type == 'structured':
            # For structured format, all fields should be present (can be empty)
            if description is None or input_desc is None or output_desc is None:
                return {
                    'valid': False,
                    'message': 'Structured format requires all three fields: description, input_desc, output_desc'
                }
            
            # At least one field should have content
            combined_content = f"{description} {input_desc} {output_desc}".strip()
            if not combined_content:
                return {
                    'valid': False,
                    'message': 'At least one field must contain text content'
                }
        
        else:  # legacy format
            # For legacy format, only description is required
            if not description or not description.strip():
                return {
                    'valid': False,
                    'message': 'Description field is required and cannot be empty'
                }
        
        return {'valid': True, 'message': 'Input format is valid'}
    
    @staticmethod
    def predict_class_and_score(description, input_desc, output_desc):
        """Make predictions using trained models with feature engineering."""
        global classifier_pipeline, regressor_pipeline, tfidf_vectorizer, feature_scaler
        
        if classifier_pipeline is None or regressor_pipeline is None:
            raise ValueError("Models not trained yet")
        
        if tfidf_vectorizer is None or feature_scaler is None:
            raise ValueError("Feature transformers not trained yet")
        
        # Combine text features
        combined_text = PredictionService.combine_text_features(description, input_desc, output_desc)
        
        if not combined_text.strip():
            raise ValueError("No text provided for prediction")
        
        # Extract TF-IDF features
        X_tfidf = tfidf_vectorizer.transform([combined_text])
        
        # Extract production-ready custom features
        (text_len, word_count, graph_algorithms, dynamic_programming, data_structures,
         sorting_searching, string_processing, basic_math, advanced_math, complexity_notation,
         constraints, optimization, multiple_cases, vocabulary_richness, avg_word_length) = extract_custom_features(combined_text)
        
        X_custom = np.array([[text_len, word_count, graph_algorithms, dynamic_programming, data_structures,
                            sorting_searching, string_processing, basic_math, advanced_math, complexity_notation,
                            constraints, optimization, multiple_cases, vocabulary_richness, avg_word_length]])
        X_custom_scaled = feature_scaler.transform(X_custom)
        
        # Combine features (same as training)
        X_combined = scipy.sparse.hstack([X_tfidf, scipy.sparse.csr_matrix(X_custom_scaled)])
        
        # Make predictions
        predicted_class = classifier_pipeline.predict(X_combined)[0]
        predicted_score = regressor_pipeline.predict(X_combined)[0]
        
        # Get prediction confidence
        class_probabilities = classifier_pipeline.predict_proba(X_combined)[0]
        confidence = max(class_probabilities)
        
        # Ensure score is clamped between 1 and 10 for the original scale
        predicted_score = max(1.0, min(10.0, predicted_score))
        
        return {
            'class': predicted_class,
            'score': round(float(predicted_score), 1),
            'confidence': round(float(confidence), 3),
            'reliable': confidence > 0.6,
            'features': {
                'textLength': int(text_len),
                'wordCount': int(word_count),
                'graphAlgorithms': int(graph_algorithms),
                'dynamicProgramming': int(dynamic_programming),
                'dataStructures': int(data_structures),
                'sortingSearching': int(sorting_searching),
                'stringProcessing': int(string_processing),
                'basicMath': int(basic_math),
                'advancedMath': int(advanced_math),
                'complexityNotation': int(complexity_notation),
                'constraints': int(constraints),
                'optimization': int(optimization),
                'multipleCases': int(multiple_cases),
                'vocabularyRichness': round(float(vocabulary_richness), 3),
                'avgWordLength': round(float(avg_word_length), 2),
                'tfidfFeatures': int(X_tfidf.shape[1])
            }
        }

@app.route('/')
def index():
    """Render the main interface for problem submission."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for problem classification and scoring (legacy format)."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            app.logger.warning("No JSON data provided in request")
            return jsonify({
                'error': 'No JSON data provided',
                'message': 'Request must contain valid JSON',
                'status_code': 400
            }), 400
        
        # Validate required fields - for combined text approach, only description is required
        if 'description' not in data or not data['description'].strip():
            app.logger.warning("Missing required field: description")
            return jsonify({
                'error': 'Missing required field',
                'message': 'The description field is required and cannot be empty',
                'status_code': 400
            }), 400
        
        # Make prediction using legacy format (backward compatibility)
        try:
            result = PredictionService.predict_class_and_score(
                data['description'],
                data.get('input_desc', ''),  # Optional for backward compatibility
                data.get('output_desc', '')  # Optional for backward compatibility
            )
            
            app.logger.info(f"Legacy prediction made: {result}")
            return jsonify(result)
            
        except ValueError as ve:
            app.logger.error(f"Prediction error: {str(ve)}")
            return jsonify({
                'error': 'Prediction error',
                'message': str(ve),
                'status_code': 400
            }), 400
        
    except Exception as e:
        app.logger.error(f"Error in predict endpoint: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': 'An error occurred while processing your request',
            'status_code': 500
        }), 500


@app.route('/predict/structured', methods=['POST'])
def predict_structured():
    """API endpoint for problem classification and scoring (AutoJudge three-input format)."""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            app.logger.warning("No JSON data provided in structured prediction request")
            return jsonify({
                'error': 'No JSON data provided',
                'message': 'Request must contain valid JSON with structured format',
                'status_code': 400
            }), 400
        
        # Validate structured format - all three fields should be present
        required_fields = ['description', 'input_desc', 'output_desc']
        missing_fields = []
        
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
        
        if missing_fields:
            app.logger.warning(f"Missing required fields in structured format: {missing_fields}")
            return jsonify({
                'error': 'Missing required fields',
                'message': f'Structured format requires all fields: {", ".join(required_fields)}. Missing: {", ".join(missing_fields)}',
                'status_code': 400
            }), 400
        
        # Validate that at least one field has content
        combined_content = f"{data['description']} {data['input_desc']} {data['output_desc']}".strip()
        if not combined_content:
            app.logger.warning("All structured fields are empty")
            return jsonify({
                'error': 'Empty content',
                'message': 'At least one of the three fields (description, input_desc, output_desc) must contain text',
                'status_code': 400
            }), 400
        
        # Make prediction using structured format
        try:
            result = PredictionService.predict_class_and_score(
                data['description'],
                data['input_desc'],
                data['output_desc']
            )
            
            # Add format indicator to response
            result['format'] = 'structured'
            result['input_fields'] = {
                'description_length': len(data['description']),
                'input_desc_length': len(data['input_desc']),
                'output_desc_length': len(data['output_desc'])
            }
            
            app.logger.info(f"Structured prediction made: {result}")
            return jsonify(result)
            
        except ValueError as ve:
            app.logger.error(f"Structured prediction error: {str(ve)}")
            return jsonify({
                'error': 'Prediction error',
                'message': str(ve),
                'status_code': 400
            }), 400
        
    except Exception as e:
        app.logger.error(f"Error in structured predict endpoint: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': 'An error occurred while processing your structured request',
            'status_code': 500
        }), 500

@app.route('/generate-docs', methods=['POST'])
def generate_documentation():
    """API endpoint to generate and save comprehensive documentation."""
    global documentation_generator
    
    try:
        if documentation_generator is None:
            return jsonify({
                'error': 'Documentation generator not initialized',
                'message': 'Models must be trained first to generate documentation',
                'status_code': 400
            }), 400
        
        # Generate and save documentation
        success = documentation_generator.save_documentation()
        
        if success:
            # Validate documentation completeness
            validation_results = documentation_generator.validate_documentation_completeness()
            
            app.logger.info("Documentation generated successfully")
            return jsonify({
                'message': 'Documentation generated and saved successfully',
                'output_path': '../README.md',
                'validation': validation_results,
                'status_code': 200
            })
        else:
            return jsonify({
                'error': 'Documentation generation failed',
                'message': 'Failed to save documentation to file',
                'status_code': 500
            }), 500
            
    except Exception as e:
        app.logger.error(f"Error generating documentation: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'message': 'An error occurred while generating documentation',
            'status_code': 500
        }), 500


@app.route('/docs/preview', methods=['GET'])
def preview_documentation():
    """API endpoint to preview generated documentation without saving."""
    global documentation_generator
    
    try:
        if documentation_generator is None:
            return jsonify({
                'error': 'Documentation generator not initialized',
                'message': 'Models must be trained first to preview documentation',
                'status_code': 400
            }), 400
        
        # Generate documentation content
        readme_content = documentation_generator.generate_complete_readme()
        
        return jsonify({
            'content': readme_content,
            'length': len(readme_content),
            'status_code': 200
        })
        
    except Exception as e:
        app.logger.error(f"Error previewing documentation: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An error occurred while previewing documentation',
            'status_code': 500
        }), 500


@app.route('/health', methods=['GET'])
def get_system_health():
    """API endpoint to get comprehensive system health information."""
    try:
        health_report = error_handler.get_system_health_report()
        
        # Add additional system information
        health_report['model_status'] = {
            'classifier_loaded': classifier_pipeline is not None,
            'regressor_loaded': regressor_pipeline is not None,
            'tfidf_vectorizer_loaded': tfidf_vectorizer is not None,
            'feature_scaler_loaded': feature_scaler is not None,
            'evaluator_initialized': model_evaluator is not None,
            'documentation_generator_ready': documentation_generator is not None
        }
        
        # Add performance metrics if available
        if model_evaluator and hasattr(model_evaluator, 'split_info'):
            health_report['training_info'] = model_evaluator.split_info
        
        return jsonify({
            'status': 'success',
            'health_report': health_report,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error getting system health: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to retrieve system health information',
            'error': str(e)
        }), 500


@app.route('/health/reset', methods=['POST'])
def reset_system_health():
    """API endpoint to reset system health status."""
    try:
        error_handler.reset_health_status()
        
        return jsonify({
            'status': 'success',
            'message': 'System health status reset successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error resetting system health: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to reset system health status',
            'error': str(e)
        }), 500


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    app.logger.error(f"500 error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'status_code': 500
    }), 500

def initialize_app():
    """Initialize the application by training models."""
    try:
        app.logger.info("Initializing Flask ML Web Application...")
        train_models()
        app.logger.info("Application initialized successfully!")
    except Exception as e:
        app.logger.error(f"Failed to initialize application: {str(e)}")
        raise

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize models on startup
    initialize_app()
    
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=app.config['DEBUG'])