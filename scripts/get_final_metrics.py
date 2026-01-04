#!/usr/bin/env python3
"""
Get final model metrics by recreating the training process
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
import scipy.sparse
import re
import os

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

def get_comprehensive_metrics():
    """Get comprehensive model performance metrics"""
    
    print("="*70)
    print("AUTOJUDGE MODEL PERFORMANCE EVALUATION")
    print("="*70)
    
    # Load dataset
    try:
        df = pd.read_json('problems_data.jsonl', lines=True)
        print(f"Dataset loaded: {len(df)} programming problems")
        
        # Map column names
        df = df.rename(columns={
            'input_description': 'input_desc',
            'output_description': 'output_desc'
        })
        
        # Combine text features
        df['combined_text'] = df['description'].fillna('') + ' ' + \
                             df['input_desc'].fillna('') + ' ' + \
                             df['output_desc'].fillna('')
        
        print(f"\nDataset Statistics:")
        class_dist = df['problem_class'].value_counts()
        for cls, count in class_dist.items():
            print(f"  {cls.capitalize()}: {count:,} problems ({count/len(df)*100:.1f}%)")
        
        print(f"\nDifficulty Score Distribution:")
        print(f"  Range: {df['problem_score'].min():.1f} - {df['problem_score'].max():.1f}")
        print(f"  Mean: {df['problem_score'].mean():.2f} ± {df['problem_score'].std():.2f}")
        
        # Extract features
        print(f"\nFeature Engineering:")
        custom_features = df['combined_text'].apply(extract_custom_features)
        X_custom = np.array([list(f) for f in custom_features])
        
        # Create TF-IDF features
        tfidf_vectorizer = TfidfVectorizer(
            max_features=4000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            analyzer='word'
        )
        X_tfidf = tfidf_vectorizer.fit_transform(df['combined_text'])
        
        # Feature selection
        selector = SelectKBest(chi2, k=3000)
        X_tfidf_selected = selector.fit_transform(X_tfidf, df['problem_class'])
        
        # Scale custom features
        feature_scaler = StandardScaler()
        X_custom_scaled = feature_scaler.fit_transform(X_custom)
        
        # Combine features
        X_combined = scipy.sparse.hstack([X_tfidf_selected, scipy.sparse.csr_matrix(X_custom_scaled)])
        
        print(f"  Total features: {X_combined.shape[1]:,}")
        print(f"  TF-IDF features: {X_tfidf_selected.shape[1]:,}")
        print(f"  Custom features: {X_custom_scaled.shape[1]}")
        
        # Train/test split
        X_train, X_test, y_train_class, y_test_class, y_train_score, y_test_score = train_test_split(
            X_combined, df['problem_class'], df['problem_score'], 
            test_size=0.2, random_state=42, stratify=df['problem_class']
        )
        
        print(f"\nTrain/Test Split:")
        print(f"  Training: {X_train.shape[0]:,} samples")
        print(f"  Testing: {X_test.shape[0]:,} samples")
        
        # Load trained models
        models_file = 'flask_app/models/trained_models.pkl'
        if os.path.exists(models_file):
            with open(models_file, 'rb') as f:
                model_data = pickle.load(f)
            
            classifier = model_data['classifier_pipeline']
            regressor = model_data['regressor_pipeline']
            
            print(f"\nModel Architecture:")
            print(f"  Classification: {type(classifier).__name__}")
            print(f"  Regression: {type(regressor).__name__}")
            if hasattr(regressor, 'n_estimators'):
                print(f"    n_estimators: {regressor.n_estimators}")
            if hasattr(regressor, 'max_depth'):
                print(f"    max_depth: {regressor.max_depth}")
            
            # Evaluate classification
            print(f"\n" + "="*70)
            print("CLASSIFICATION PERFORMANCE")
            print("="*70)
            
            y_pred_class = classifier.predict(X_test)
            accuracy = accuracy_score(y_test_class, y_pred_class)
            cm = confusion_matrix(y_test_class, y_pred_class, labels=['easy', 'medium', 'hard'])
            report = classification_report(y_test_class, y_pred_class, output_dict=True, zero_division=0)
            
            print(f"Overall Accuracy: {accuracy:.1%}")
            
            print(f"\nConfusion Matrix:")
            print("                 Predicted")
            print("Actual      Easy  Medium  Hard   Total")
            for i, actual_class in enumerate(['Easy', 'Medium', 'Hard']):
                row_total = sum(cm[i])
                row = f"{actual_class:8s}"
                for j in range(3):
                    row += f"{cm[i][j]:7d}"
                row += f"{row_total:8d}"
                print(row)
            
            col_totals = cm.sum(axis=0)
            print(f"Total   {col_totals[0]:7d}{col_totals[1]:7d}{col_totals[2]:7d}{col_totals.sum():8d}")
            
            print(f"\nPer-Class Performance:")
            for class_name in ['easy', 'medium', 'hard']:
                if class_name in report:
                    metrics = report[class_name]
                    print(f"  {class_name.capitalize():6s}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1-Score={metrics['f1-score']:.3f} (n={int(metrics['support'])})")
            
            print(f"\nAverage Metrics:")
            print(f"  Macro Avg:    Precision={report['macro avg']['precision']:.3f}, Recall={report['macro avg']['recall']:.3f}, F1-Score={report['macro avg']['f1-score']:.3f}")
            print(f"  Weighted Avg: Precision={report['weighted avg']['precision']:.3f}, Recall={report['weighted avg']['recall']:.3f}, F1-Score={report['weighted avg']['f1-score']:.3f}")
            
            # Evaluate regression
            print(f"\n" + "="*70)
            print("REGRESSION PERFORMANCE")
            print("="*70)
            
            y_pred_score = regressor.predict(X_test)
            mae = mean_absolute_error(y_test_score, y_pred_score)
            rmse = np.sqrt(mean_squared_error(y_test_score, y_pred_score))
            r2 = r2_score(y_test_score, y_pred_score)
            
            print(f"Mean Absolute Error (MAE): {mae:.3f} points")
            print(f"Root Mean Square Error (RMSE): {rmse:.3f} points")
            print(f"R² Score (Coefficient of Determination): {r2:.3f}")
            
            # Score prediction accuracy by class
            print(f"\nScore Prediction by Class:")
            for class_name in ['easy', 'medium', 'hard']:
                mask = y_test_class == class_name
                if mask.sum() > 0:
                    class_mae = mean_absolute_error(y_test_score[mask], y_pred_score[mask])
                    class_mean_actual = y_test_score[mask].mean()
                    class_mean_pred = y_pred_score[mask].mean()
                    print(f"  {class_name.capitalize():6s}: MAE={class_mae:.3f}, Actual={class_mean_actual:.2f}±{y_test_score[mask].std():.2f}, Predicted={class_mean_pred:.2f}±{y_pred_score[mask].std():.2f}")
            
            return {
                'dataset_size': len(df),
                'train_size': X_train.shape[0],
                'test_size': X_test.shape[0],
                'feature_count': X_combined.shape[1],
                'accuracy': accuracy,
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2,
                'class_distribution': class_dist.to_dict()
            }
        else:
            print("No trained models found!")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    metrics = get_comprehensive_metrics()
    if metrics:
        print(f"\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Dataset: {metrics['dataset_size']:,} programming problems")
        print(f"Classification Accuracy: {metrics['accuracy']:.1%}")
        print(f"Score Prediction MAE: {metrics['mae']:.3f} points")
        print(f"Model Features: {metrics['feature_count']:,}")
    else:
        print("Evaluation failed!")