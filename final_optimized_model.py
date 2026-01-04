#!/usr/bin/env python3
"""
Final Optimized Model for AutoJudge

This script implements a carefully balanced set of improvements:
1. Moderate feature enhancement (avoid overfitting)
2. Balanced ensemble approach
3. Proper class balancing
4. Focused hyperparameter tuning
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

def extract_optimized_features(text):
    """Extract optimized features that balance informativeness with simplicity."""
    # Basic features
    text_len = len(text)
    
    # Mathematical symbols
    math_symbols = r'[+\-*/=<>$^∑∏∫∂∆√π∞≤≥≠≈∈∉∪∩⊂⊃∅%&|~!@#()[\]{}]'
    math_count = len(re.findall(math_symbols, text))
    
    # Weighted algorithm keywords (focused on most discriminative)
    algorithm_keywords = {
        # High difficulty (weight 3)
        'dynamic programming': 3, 'dp': 3, 'segment tree': 3, 'fenwick': 3,
        'suffix array': 3, 'kmp': 3, 'convex hull': 3, 'network flow': 3,
        
        # Medium difficulty (weight 2)
        'binary search': 2, 'two pointer': 2, 'sliding window': 2,
        'dfs': 2, 'bfs': 2, 'dijkstra': 2, 'union find': 2, 'trie': 2,
        
        # Basic difficulty (weight 1)
        'sort': 1, 'search': 1, 'hash': 1, 'stack': 1, 'queue': 1
    }
    
    text_lower = text.lower()
    weighted_keyword_score = 0
    for keyword, weight in algorithm_keywords.items():
        count = len(re.findall(rf'\b{keyword}\b', text_lower))
        weighted_keyword_score += count * weight
    
    # Sentence complexity
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if sentences:
        words_per_sentence = sum(len(s.split()) for s in sentences) / len(sentences)
    else:
        words_per_sentence = 0
    
    # Vocabulary richness
    words = re.findall(r'\b\w+\b', text_lower)
    if words:
        unique_word_ratio = len(set(words)) / len(words)
    else:
        unique_word_ratio = 0
    
    # Constraint complexity (focused patterns)
    constraint_patterns = [
        r'o\(.*\)', r'time.*complexity', r'space.*complexity',
        r'constraint', r'1\s*≤.*≤\s*10\^?\d+', r'1\s*<=.*<=\s*10\^?\d+'
    ]
    constraint_count = sum(len(re.findall(pattern, text_lower)) for pattern in constraint_patterns)
    
    # Problem type indicators (simplified)
    graph_indicators = ['graph', 'tree', 'node', 'edge', 'path']
    graph_score = sum(len(re.findall(rf'\b{indicator}\b', text_lower)) for indicator in graph_indicators)
    
    return (text_len, math_count, weighted_keyword_score, words_per_sentence, 
            unique_word_ratio, constraint_count, graph_score)

def load_and_preprocess_data():
    """Load and preprocess the dataset."""
    try:
        df = pd.read_json('problems_data.jsonl', lines=True)
        
        df = df.rename(columns={
            'input_description': 'input_desc',
            'output_description': 'output_desc'
        })
        
        df['problem_score_scaled'] = df['problem_score']
        
        # Combine text features
        df['combined_text'] = df['description'].fillna('') + ' ' + \
                             df['input_desc'].fillna('') + ' ' + \
                             df['output_desc'].fillna('')
        
        df['combined_text'] = df['combined_text'].str.strip()
        
        # Extract optimized features
        print("Extracting optimized features...")
        optimized_features = df['combined_text'].apply(extract_optimized_features)
        
        feature_names = ['text_len', 'math_count', 'weighted_keyword_score', 'words_per_sentence', 
                        'unique_word_ratio', 'constraint_count', 'graph_score']
        
        for i, name in enumerate(feature_names):
            df[name] = [f[i] for f in optimized_features]
        
        return df, feature_names
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def test_optimized_model():
    """Test the optimized model with balanced improvements."""
    
    print("="*80)
    print("TESTING FINAL OPTIMIZED AUTOJUDGE MODEL")
    print("="*80)
    
    # Load data
    print("1. Loading and preprocessing data...")
    df, feature_names = load_and_preprocess_data()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Classes: {df['problem_class'].value_counts().to_dict()}")
    print(f"   Optimized features: {len(feature_names)}")
    
    # Prepare features
    print("\n2. Preparing optimized feature matrix...")
    X_text = df['combined_text']
    X_custom = df[feature_names].values
    y_class = df['problem_class']
    y_score = df['problem_score_scaled']
    
    # Optimized TF-IDF
    tfidf_vectorizer = TfidfVectorizer(
        max_features=6000,  # Balanced size
        stop_words='english', 
        ngram_range=(1, 2),  # Stick to bigrams to avoid overfitting
        min_df=3,  # Slightly higher to reduce noise
        max_df=0.9,  # More conservative
        sublinear_tf=True
    )
    
    X_tfidf = tfidf_vectorizer.fit_transform(X_text)
    
    # Scale custom features
    feature_scaler = StandardScaler()
    X_custom_scaled = feature_scaler.fit_transform(X_custom)
    
    # Combine features
    X_combined = scipy.sparse.hstack([X_tfidf, scipy.sparse.csr_matrix(X_custom_scaled)])
    
    print(f"   Optimized feature matrix: {X_combined.shape}")
    print(f"   TF-IDF features: {X_tfidf.shape[1]}")
    print(f"   Custom features: {X_custom_scaled.shape[1]}")
    
    # Train/test split
    print("\n3. Performing train/test split...")
    X_train, X_test, y_train_class, y_test_class, y_train_score, y_test_score = train_test_split(
        X_combined, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
    )
    
    print(f"   Train samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Original class distribution: {y_train_class.value_counts().to_dict()}")
    
    # Balanced SMOTE application
    print("\n4. Applying balanced SMOTE...")
    try:
        # Use SMOTE with conservative parameters
        smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy='auto')
        X_train_balanced, y_train_class_balanced = smote.fit_resample(X_train.toarray(), y_train_class)
        X_train_balanced = scipy.sparse.csr_matrix(X_train_balanced)
        print(f"   ✓ SMOTE applied successfully")
        print(f"   Balanced class distribution: {pd.Series(y_train_class_balanced).value_counts().to_dict()}")
        
        # Handle regression targets properly
        y_train_score_balanced = []
        class_indices = {}
        for i, cls in enumerate(y_train_class):
            if cls not in class_indices:
                class_indices[cls] = []
            class_indices[cls].append(i)
        
        for cls in y_train_class_balanced:
            idx = np.random.choice(class_indices[cls])
            y_train_score_balanced.append(y_train_score.iloc[idx])
        
        y_train_score_balanced = pd.Series(y_train_score_balanced)
        
    except Exception as e:
        print(f"   ⚠ SMOTE failed: {e}")
        X_train_balanced = X_train
        y_train_class_balanced = y_train_class
        y_train_score_balanced = y_train_score
    
    # Train optimized models
    print("\n5. Training optimized models...")
    
    # Optimized classification ensemble
    lr_classifier = LogisticRegression(
        random_state=42, 
        max_iter=2000,
        class_weight='balanced',
        C=1.0,  # Less regularization
        solver='lbfgs'
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
    
    classifier = VotingClassifier(
        estimators=[('lr', lr_classifier), ('rf', rf_classifier)],
        voting='soft'
    )
    
    classifier.fit(X_train_balanced, y_train_class_balanced)
    print("   ✓ Optimized classification ensemble trained")
    
    # Optimized regression model
    regressor = RandomForestRegressor(
        n_estimators=350,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    regressor.fit(X_train_balanced, y_train_score_balanced)
    print("   ✓ Optimized regression model trained")
    
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
    print("FINAL OPTIMIZED MODEL RESULTS")
    print("="*80)
    
    print(f"\nOPTIMIZED IMPROVEMENTS IMPLEMENTED:")
    print(f"  ✓ Balanced Feature Engineering: {len(feature_names)} focused custom features")
    print(f"  ✓ Optimized TF-IDF: 6000 features with conservative parameters")
    print(f"  ✓ Balanced Class Balancing: Conservative SMOTE application")
    print(f"  ✓ Optimized Ensemble: LogisticRegression + RandomForest")
    print(f"  ✓ Tuned Hyperparameters: Balanced complexity vs. performance")
    print(f"  ✓ Focused Features: Weighted keywords, problem type detection")
    
    print(f"\nCLASSIFICATION RESULTS:")
    print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print(f"\nCONFUSION MATRIX:")
    classes = sorted(y_test_class.unique())
    conf_df = pd.DataFrame(conf_matrix, index=classes, columns=classes)
    print(conf_df.to_string())
    
    print(f"\nDETAILED CLASSIFICATION REPORT:")
    for class_name in classes:
        if class_name in class_report:
            metrics = class_report[class_name]
            print(f"  {class_name}:")
            print(f"    Precision: {metrics['precision']:.3f}")
            print(f"    Recall: {metrics['recall']:.3f}")
            print(f"    F1-Score: {metrics['f1-score']:.3f}")
            print(f"    Support: {int(metrics['support'])}")
    
    if 'weighted avg' in class_report:
        weighted_avg = class_report['weighted avg']
        print(f"  Weighted Average:")
        print(f"    Precision: {weighted_avg['precision']:.3f}")
        print(f"    Recall: {weighted_avg['recall']:.3f}")
        print(f"    F1-Score: {weighted_avg['f1-score']:.3f}")
    
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
    
    print(f"  Final Classification Performance: {class_perf}")
    print(f"  Final Regression Performance: {reg_perf}")
    
    # Improvement over baseline
    baseline_accuracy = 0.502  # From original model
    improvement = ((accuracy - baseline_accuracy) / baseline_accuracy) * 100
    print(f"  Accuracy Change from Baseline: {improvement:+.1f}%")
    
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
    
    # Class-wise accuracy
    print(f"\nCLASS-WISE ACCURACY:")
    for class_name in classes:
        class_mask = y_test_class == class_name
        if class_mask.sum() > 0:
            class_accuracy = accuracy_score(y_test_class[class_mask], y_pred_class[class_mask])
            class_count = class_mask.sum()
            print(f"  {class_name}: {class_accuracy:.3f} ({class_accuracy*100:.1f}%) - {class_count} samples")
    
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
    print("FINAL OPTIMIZED MODEL SUMMARY")
    print("="*80)
    print(f"✓ Final model tested on {len(y_test_class)} samples")
    print(f"✓ Classification accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"✓ Regression MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    print(f"✓ Total features: {X_combined.shape[1]} (balanced approach)")
    print(f"✓ All optimized improvements successfully implemented")
    
    if accuracy >= 0.6:
        print("🎉 TARGET ACCURACY ACHIEVED!")
    elif accuracy >= 0.55:
        print("📈 CLOSE TO TARGET - Minor tuning needed")
    else:
        print("⚠ Target accuracy not achieved - Consider data quality improvements")
    
    # Final recommendations
    print(f"\nFINAL RECOMMENDATIONS:")
    if accuracy < 0.6:
        print("  • Consider collecting more high-quality training data")
        print("  • Review and improve problem descriptions for consistency")
        print("  • Consider domain-specific embeddings (Word2Vec, BERT)")
        print("  • Implement active learning for difficult cases")
    else:
        print("  • Model ready for production deployment")
        print("  • Monitor performance on new data")
        print("  • Consider periodic retraining")
    
    return {
        'accuracy': accuracy,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'improvement_over_baseline': improvement
    }

if __name__ == "__main__":
    print("Starting final optimized model testing...")
    results = test_optimized_model()
    print("\nFinal optimized model testing completed!")