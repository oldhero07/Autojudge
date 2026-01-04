#!/usr/bin/env python3
"""
Direct Test of Improved Model Performance

This script directly tests the improved model without relying on global variables.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import scipy.sparse
import re
import warnings
warnings.filterwarnings('ignore')

def extract_enhanced_features(text):
    """Extract enhanced custom numerical features from text."""
    # Text length feature
    text_len = len(text)
    
    # Enhanced mathematical symbols count
    math_symbols = r'[+\-*/=<>$^∑∏∫∂∆√π∞≤≥≠≈∈∉∪∩⊂⊃∅%&|~!@#()[\]{}]'
    math_count = len(re.findall(math_symbols, text))
    
    # Enhanced keyword difficulty indicators count
    difficulty_keywords = [
        # Algorithm types
        'dp', 'graph', 'tree', 'recursion', 'constraint', 'query',
        'algorithm', 'optimize', 'complexity', 'binary', 'search',
        'sort', 'heap', 'stack', 'queue', 'dynamic', 'greedy',
        'backtrack', 'dfs', 'bfs', 'shortest', 'path', 'minimum',
        'maximum', 'optimal', 'efficient',
        # Data structures
        'array', 'list', 'matrix', 'string', 'hash', 'map', 'set',
        'trie', 'segment', 'fenwick', 'union', 'find', 'disjoint',
        # Advanced concepts
        'memoization', 'tabulation', 'divide', 'conquer', 'sliding',
        'window', 'two', 'pointer', 'prefix', 'suffix', 'subsequence',
        'substring', 'palindrome', 'anagram', 'permutation', 'combination'
    ]
    text_lower = text.lower()
    keyword_count = sum(len(re.findall(rf'\b{keyword}\b', text_lower)) for keyword in difficulty_keywords)
    
    # NEW FEATURES for better discrimination
    
    # 1. Sentence complexity (average words per sentence)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        words_per_sentence = sum(len(s.split()) for s in sentences) / len(sentences)
    else:
        words_per_sentence = 0
    
    # 2. Unique word ratio (vocabulary richness)
    words = re.findall(r'\b\w+\b', text_lower)
    if words:
        unique_word_ratio = len(set(words)) / len(words)
    else:
        unique_word_ratio = 0
    
    # 3. Constraint indicators (time/space complexity mentions)
    constraint_patterns = [
        r'o\(.*\)', r'time.*complexity', r'space.*complexity',
        r'constraint', r'limit', r'bound', r'≤', r'<=', r'≥', r'>=',
        r'\d+\s*≤.*≤\s*\d+', r'\d+\s*<=.*<=\s*\d+'
    ]
    constraint_count = sum(len(re.findall(pattern, text_lower)) for pattern in constraint_patterns)
    
    # 4. Input/Output complexity indicators
    io_patterns = [
        r'input.*format', r'output.*format', r'first.*line', r'second.*line',
        r'multiple.*test.*case', r'test.*case', r'sample.*input', r'sample.*output',
        r'example.*input', r'example.*output'
    ]
    io_complexity = sum(len(re.findall(pattern, text_lower)) for pattern in io_patterns)
    
    # 5. Number patterns (often indicate mathematical problems)
    number_patterns = [
        r'\d+', r'integer', r'number', r'digit', r'prime', r'factorial',
        r'fibonacci', r'gcd', r'lcm', r'modulo', r'remainder'
    ]
    number_count = sum(len(re.findall(pattern, text_lower)) for pattern in number_patterns)
    
    return (text_len, math_count, keyword_count, words_per_sentence, 
            unique_word_ratio, constraint_count, io_complexity, number_count)

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    try:
        # Load the dataset
        df = pd.read_json('problems_data.jsonl', lines=True)
        
        # Map column names to expected format
        df = df.rename(columns={
            'input_description': 'input_desc',
            'output_description': 'output_desc'
        })
        
        # Keep original score scale
        df['problem_score_scaled'] = df['problem_score']
        
        # Combine text features
        df['combined_text'] = df['description'].fillna('') + ' ' + \
                             df['input_desc'].fillna('') + ' ' + \
                             df['output_desc'].fillna('')
        
        # Clean the combined text
        df['combined_text'] = df['combined_text'].str.strip()
        
        # Extract enhanced custom features
        print("Extracting enhanced features...")
        custom_features = df['combined_text'].apply(extract_enhanced_features)
        df['text_len'] = [f[0] for f in custom_features]
        df['math_count'] = [f[1] for f in custom_features]
        df['keyword_count'] = [f[2] for f in custom_features]
        df['words_per_sentence'] = [f[3] for f in custom_features]
        df['unique_word_ratio'] = [f[4] for f in custom_features]
        df['constraint_count'] = [f[5] for f in custom_features]
        df['io_complexity'] = [f[6] for f in custom_features]
        df['number_count'] = [f[7] for f in custom_features]
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def test_improved_model():
    """Test the improved model with all enhancements."""
    
    print("="*80)
    print("TESTING IMPROVED AUTOJUDGE MODEL")
    print("="*80)
    
    # Load data
    print("1. Loading and preprocessing data...")
    df = load_and_preprocess_data()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Classes: {df['problem_class'].value_counts().to_dict()}")
    
    # Prepare features
    print("\n2. Preparing enhanced features...")
    X_text = df['combined_text']
    X_custom = df[['text_len', 'math_count', 'keyword_count', 'words_per_sentence', 
                  'unique_word_ratio', 'constraint_count', 'io_complexity', 'number_count']].values
    y_class = df['problem_class']
    y_score = df['problem_score_scaled']
    
    # Enhanced TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        max_features=8000,
        stop_words='english', 
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_tfidf = tfidf_vectorizer.fit_transform(X_text)
    
    # Scale custom features
    feature_scaler = StandardScaler()
    X_custom_scaled = feature_scaler.fit_transform(X_custom)
    
    # Combine features
    X_combined = scipy.sparse.hstack([X_tfidf, scipy.sparse.csr_matrix(X_custom_scaled)])
    print(f"   Enhanced feature matrix: {X_combined.shape}")
    print(f"   TF-IDF features: {X_tfidf.shape[1]} (vs 5000 original)")
    print(f"   Custom features: {X_custom_scaled.shape[1]} (vs 3 original)")
    
    # Train/test split
    print("\n3. Performing train/test split...")
    X_train, X_test, y_train_class, y_test_class, y_train_score, y_test_score = train_test_split(
        X_combined, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
    )
    
    print(f"   Train samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Original class distribution: {y_train_class.value_counts().to_dict()}")
    
    # Apply SMOTE for class balancing
    print("\n4. Applying SMOTE for class balancing...")
    try:
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_class_balanced = smote.fit_resample(X_train.toarray(), y_train_class)
        X_train_balanced = scipy.sparse.csr_matrix(X_train_balanced)
        print(f"   ✓ SMOTE applied successfully")
        print(f"   Balanced class distribution: {pd.Series(y_train_class_balanced).value_counts().to_dict()}")
        
        # Handle regression targets - create balanced version
        y_train_score_balanced = []
        class_indices = {}
        for i, cls in enumerate(y_train_class):
            if cls not in class_indices:
                class_indices[cls] = []
            class_indices[cls].append(i)
        
        # For each balanced sample, find corresponding score
        for cls in y_train_class_balanced:
            # Randomly select a score from the original samples of this class
            idx = np.random.choice(class_indices[cls])
            y_train_score_balanced.append(y_train_score.iloc[idx])
        
        y_train_score_balanced = pd.Series(y_train_score_balanced)
        
    except Exception as e:
        print(f"   ⚠ SMOTE failed: {e}")
        X_train_balanced = X_train
        y_train_class_balanced = y_train_class
        y_train_score_balanced = y_train_score
    
    # Train enhanced models
    print("\n5. Training enhanced models...")
    
    # Enhanced classification ensemble
    lr_classifier = LogisticRegression(
        random_state=42, 
        max_iter=2000,
        class_weight='balanced',
        C=0.1
    )
    
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    classifier = VotingClassifier(
        estimators=[('lr', lr_classifier), ('rf', rf_classifier)],
        voting='soft'
    )
    
    classifier.fit(X_train_balanced, y_train_class_balanced)
    print("   ✓ Enhanced classification ensemble trained")
    
    # Enhanced regression model
    regressor = RandomForestRegressor(
        n_estimators=400,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    
    regressor.fit(X_train_balanced, y_train_score_balanced)
    print("   ✓ Enhanced regression model trained")
    
    # Make predictions
    print("\n6. Making predictions...")
    y_pred_class = classifier.predict(X_test)
    y_pred_score = regressor.predict(X_test)
    y_pred_score = np.clip(y_pred_score, 1.0, 10.0)
    
    # Calculate metrics
    print("\n7. Calculating metrics...")
    accuracy = accuracy_score(y_test_class, y_pred_class)
    conf_matrix = confusion_matrix(y_test_class, y_pred_class)
    class_report = classification_report(y_test_class, y_pred_class, output_dict=True)
    
    mae = mean_absolute_error(y_test_score, y_pred_score)
    rmse = np.sqrt(mean_squared_error(y_test_score, y_pred_score))
    r2 = r2_score(y_test_score, y_pred_score)
    
    # Display results
    print("\n" + "="*80)
    print("IMPROVED MODEL RESULTS")
    print("="*80)
    
    print(f"\nIMPROVEMENTS IMPLEMENTED:")
    print(f"  ✓ Enhanced TF-IDF: 8000 features (vs 5000), trigrams, better filtering")
    print(f"  ✓ Enhanced Custom Features: 8 features (vs 3)")
    print(f"  ✓ Class Balancing: SMOTE applied")
    print(f"  ✓ Ensemble Classification: Voting classifier")
    print(f"  ✓ Enhanced Regression: Tuned RandomForest")
    
    print(f"\nCLASSIFICATION RESULTS:")
    print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print(f"\nCONFUSION MATRIX:")
    classes = sorted(y_test_class.unique())
    conf_df = pd.DataFrame(conf_matrix, index=classes, columns=classes)
    print(conf_df.to_string())
    
    print(f"\nCLASS-WISE PERFORMANCE:")
    for class_name in classes:
        if class_name in class_report:
            metrics = class_report[class_name]
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    F1-Score: {metrics['f1-score']:.3f}")
            print(f"    Support: {int(metrics['support'])}")
    
    print(f"\nREGRESSION RESULTS:")
    print(f"  MAE: {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R² Score: {r2:.3f}")
    
    # Performance analysis
    print(f"\nPERFORMANCE ANALYSIS:")
    
    if accuracy >= 0.8:
        class_perf = "Excellent"
    elif accuracy >= 0.7:
        class_perf = "Good"
    elif accuracy >= 0.6:
        class_perf = "Acceptable"
    else:
        class_perf = "Needs Further Improvement"
    
    if mae <= 1.0 and rmse <= 1.5:
        reg_perf = "Excellent"
    elif mae <= 1.5 and rmse <= 2.0:
        reg_perf = "Good"
    elif mae <= 2.0 and rmse <= 2.5:
        reg_perf = "Acceptable"
    else:
        reg_perf = "Needs Further Improvement"
    
    print(f"  Classification Performance: {class_perf}")
    print(f"  Regression Performance: {reg_perf}")
    
    # Threshold validation
    print(f"\nTHRESHOLD VALIDATION:")
    thresholds_met = []
    if accuracy >= 0.6:
        thresholds_met.append("Classification accuracy ≥ 60%")
    if mae <= 2.0:
        thresholds_met.append("MAE ≤ 2.0")
    if rmse <= 2.5:
        thresholds_met.append("RMSE ≤ 2.5")
    
    if thresholds_met:
        print(f"  ✓ Thresholds met: {', '.join(thresholds_met)}")
    else:
        print("  ⚠ Some thresholds not met")
    
    # Sample predictions
    print(f"\nSAMPLE PREDICTIONS (first 10):")
    print("  Actual | Predicted | Actual Score | Predicted Score | Match")
    print("  " + "-" * 60)
    
    for i in range(min(10, len(y_test_class))):
        actual_class = y_test_class.iloc[i]
        pred_class = y_pred_class[i]
        actual_score = y_test_score.iloc[i]
        pred_score = y_pred_score[i]
        
        match = "✓" if actual_class == pred_class else "✗"
        print(f"  {actual_class:>6} | {pred_class:>9} | {actual_score:>11.1f} | {pred_score:>14.1f} | {match}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✓ Enhanced model tested on {len(y_test_class)} samples")
    print(f"✓ Classification accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"✓ Regression MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    print(f"✓ All improvements successfully implemented")
    
    return {
        'accuracy': accuracy,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

if __name__ == "__main__":
    print("Starting improved model testing...")
    results = test_improved_model()
    print("\nImproved model testing completed!")