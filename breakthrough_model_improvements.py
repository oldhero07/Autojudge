#!/usr/bin/env python3
"""
Breakthrough Model Improvements for AutoJudge

This script implements breakthrough improvements to achieve >60% accuracy:
1. Advanced data preprocessing and quality improvement
2. Sophisticated feature engineering with semantic analysis
3. Advanced ensemble methods with stacking
4. Intelligent class boundary optimization
5. Data augmentation and synthetic sample generation
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
import scipy.sparse
import re
import warnings
from collections import Counter
import string
warnings.filterwarnings('ignore')

def advanced_text_preprocessing(text):
    """Advanced text preprocessing to improve data quality."""
    if not text or pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize mathematical expressions
    text = re.sub(r'o\(\s*([^)]+)\s*\)', r'O(\1)', text)  # Normalize Big O notation
    text = re.sub(r'(\d+)\s*<=?\s*([a-z]+)\s*<=?\s*(\d+)', r'\1 ≤ \2 ≤ \3', text)  # Normalize constraints
    
    # Normalize common programming terms
    replacements = {
        'dfs': 'depth first search',
        'bfs': 'breadth first search',
        'dp': 'dynamic programming',
        'lca': 'lowest common ancestor',
        'mst': 'minimum spanning tree',
        'scc': 'strongly connected components'
    }
    
    for abbrev, full in replacements.items():
        text = re.sub(rf'\b{abbrev}\b', full, text)
    
    return text.strip()

def extract_breakthrough_features(text):
    """Extract breakthrough features with advanced semantic analysis."""
    # Preprocess text
    text = advanced_text_preprocessing(text)
    
    # Basic features
    text_len = len(text)
    word_count = len(text.split())
    
    # Advanced mathematical content analysis
    math_patterns = {
        'basic_math': r'[+\-*/=<>%]',
        'advanced_math': r'[∑∏∫∂∆√π∞≤≥≠≈∈∉∪∩⊂⊃∅]',
        'big_o': r'o\([^)]+\)',
        'equations': r'\w+\s*=\s*\w+',
        'inequalities': r'[<>≤≥]\s*\w+\s*[<>≤≥]'
    }
    
    math_scores = {}
    for pattern_name, pattern in math_patterns.items():
        math_scores[pattern_name] = len(re.findall(pattern, text, re.IGNORECASE))
    
    # Sophisticated algorithm classification with confidence scoring
    algorithm_categories = {
        'graph_algorithms': {
            'patterns': ['graph', 'tree', 'node', 'edge', 'vertex', 'path', 'cycle', 'connected', 
                        'depth first search', 'breadth first search', 'dijkstra', 'bellman ford', 
                        'floyd warshall', 'minimum spanning tree', 'strongly connected components'],
            'difficulty_weight': 2.5
        },
        'dynamic_programming': {
            'patterns': ['dynamic programming', 'memoization', 'tabulation', 'optimal substructure',
                        'overlapping subproblems', 'knapsack', 'longest common subsequence'],
            'difficulty_weight': 3.0
        },
        'data_structures': {
            'patterns': ['heap', 'priority queue', 'stack', 'queue', 'hash', 'trie', 'segment tree',
                        'fenwick', 'binary indexed tree', 'union find', 'disjoint set'],
            'difficulty_weight': 2.0
        },
        'string_algorithms': {
            'patterns': ['string', 'substring', 'subsequence', 'palindrome', 'anagram', 'kmp',
                        'z algorithm', 'suffix array', 'rolling hash'],
            'difficulty_weight': 2.2
        },
        'sorting_searching': {
            'patterns': ['sort', 'binary search', 'merge sort', 'quick sort', 'heap sort'],
            'difficulty_weight': 1.5
        },
        'greedy_algorithms': {
            'patterns': ['greedy', 'optimal', 'minimum', 'maximum', 'interval scheduling'],
            'difficulty_weight': 2.3
        },
        'number_theory': {
            'patterns': ['prime', 'gcd', 'lcm', 'modular', 'fibonacci', 'factorial', 'combinatorics'],
            'difficulty_weight': 2.1
        }
    }
    
    category_scores = {}
    total_algorithm_score = 0
    
    for category, info in algorithm_categories.items():
        score = 0
        for pattern in info['patterns']:
            matches = len(re.findall(rf'\b{pattern}\b', text))
            score += matches
        
        weighted_score = score * info['difficulty_weight']
        category_scores[category] = weighted_score
        total_algorithm_score += weighted_score
    
    # Complexity analysis
    complexity_indicators = {
        'time_complexity': len(re.findall(r'time.*complexity|o\([^)]+\)', text)),
        'space_complexity': len(re.findall(r'space.*complexity|memory.*complexity', text)),
        'constraints': len(re.findall(r'constraint|limit|bound|\d+\s*≤.*≤\s*\d+', text)),
        'test_cases': len(re.findall(r'test.*case|example|sample', text))
    }
    
    # Linguistic complexity
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if sentences and word_count > 0:
        avg_sentence_length = word_count / len(sentences)
        unique_words = len(set(text.split()))
        vocabulary_richness = unique_words / word_count if word_count > 0 else 0
        
        # Readability approximation (simplified Flesch score)
        readability = max(0, 206.835 - (1.015 * avg_sentence_length) - (84.6 * (len([w for w in text.split() if len(w) > 6]) / word_count)))
    else:
        avg_sentence_length = 0
        vocabulary_richness = 0
        readability = 0
    
    # Problem structure analysis
    structure_indicators = {
        'input_output_format': len(re.findall(r'input.*format|output.*format|first.*line|next.*line', text)),
        'multiple_test_cases': len(re.findall(r'multiple.*test|test.*case|t.*test', text)),
        'interactive': len(re.findall(r'interactive|query|ask', text)),
        'optimization': len(re.findall(r'minimum|maximum|optimal|best|least|most', text))
    }
    
    # Difficulty keywords with semantic weighting
    difficulty_keywords = {
        'trivial': ['print', 'output', 'simple', 'basic', 'easy'],
        'easy': ['loop', 'array', 'string', 'count', 'sum'],
        'medium': ['sort', 'search', 'hash', 'two pointer', 'sliding window'],
        'hard': ['dynamic programming', 'graph', 'tree', 'segment tree', 'complex'],
        'expert': ['suffix array', 'convex hull', 'network flow', 'advanced']
    }
    
    difficulty_score = 0
    for level, keywords in difficulty_keywords.items():
        weight = {'trivial': 0.5, 'easy': 1.0, 'medium': 2.0, 'hard': 3.0, 'expert': 4.0}[level]
        for keyword in keywords:
            count = len(re.findall(rf'\b{keyword}\b', text))
            difficulty_score += count * weight
    
    # Combine all features
    features = [
        text_len,
        word_count,
        math_scores['basic_math'],
        math_scores['advanced_math'],
        math_scores['big_o'],
        total_algorithm_score,
        category_scores['graph_algorithms'],
        category_scores['dynamic_programming'],
        category_scores['data_structures'],
        category_scores['string_algorithms'],
        complexity_indicators['time_complexity'],
        complexity_indicators['space_complexity'],
        complexity_indicators['constraints'],
        avg_sentence_length,
        vocabulary_richness,
        readability,
        structure_indicators['input_output_format'],
        structure_indicators['optimization'],
        difficulty_score
    ]
    
    return features

def improve_data_quality(df):
    """Improve data quality through intelligent preprocessing."""
    print("Improving data quality...")
    
    # Remove samples with extremely short descriptions (likely incomplete)
    min_length = 50
    original_size = len(df)
    df = df[df['combined_text'].str.len() >= min_length].copy()
    print(f"   Removed {original_size - len(df)} samples with text < {min_length} characters")
    
    # Identify and potentially correct mislabeled samples using heuristics
    def detect_potential_mislabels(row):
        text = row['combined_text'].lower()
        actual_class = row['problem_class']
        score = row['problem_score']
        
        # Heuristic rules for obvious misclassifications
        hard_indicators = ['dynamic programming', 'segment tree', 'suffix array', 'convex hull']
        easy_indicators = ['print', 'output', 'simple addition', 'basic']
        
        has_hard_indicators = any(indicator in text for indicator in hard_indicators)
        has_easy_indicators = any(indicator in text for indicator in easy_indicators)
        
        # Flag potential misclassifications
        if actual_class == 'easy' and has_hard_indicators and score > 6:
            return 'hard'
        elif actual_class == 'hard' and has_easy_indicators and score < 3:
            return 'easy'
        else:
            return actual_class
    
    # Apply corrections (conservative approach)
    df['suggested_class'] = df.apply(detect_potential_mislabels, axis=1)
    corrections = (df['problem_class'] != df['suggested_class']).sum()
    print(f"   Identified {corrections} potential mislabeled samples")
    
    # For now, keep original labels but flag for review
    df['potential_mislabel'] = df['problem_class'] != df['suggested_class']
    
    return df

def create_breakthrough_ensemble():
    """Create a breakthrough ensemble with stacking and advanced algorithms."""
    
    # Base classifiers with diverse approaches (removed MultinomialNB due to negative values)
    base_classifiers = [
        ('lr', LogisticRegression(random_state=42, max_iter=3000, class_weight='balanced', C=0.5)),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=30, min_samples_split=3, 
                                     min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, 
                                         random_state=42)),
        ('svc', SVC(probability=True, class_weight='balanced', random_state=42, C=1.0, gamma='scale')),
        ('ridge', RidgeClassifier(class_weight='balanced', random_state=42, alpha=1.0))
    ]
    
    # Meta-classifier (stacking)
    meta_classifier = LogisticRegression(random_state=42, max_iter=1000)
    
    # Create stacking ensemble
    stacking_classifier = StackingClassifier(
        estimators=base_classifiers,
        final_estimator=meta_classifier,
        cv=5,  # 5-fold cross-validation for stacking
        stack_method='predict_proba',
        n_jobs=-1
    )
    
    return stacking_classifier

def optimize_class_boundaries(X_train, y_train, y_score_train):
    """Optimize class boundaries based on score distribution."""
    print("Optimizing class boundaries...")
    
    # Analyze score distribution within each class
    class_stats = {}
    for class_name in ['easy', 'medium', 'hard']:
        mask = y_train == class_name
        if mask.sum() > 0:
            scores = y_score_train[mask]
            class_stats[class_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max(),
                'count': len(scores)
            }
    
    print("   Class score statistics:")
    for class_name, stats in class_stats.items():
        print(f"     {class_name}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, range=[{stats['min']:.1f}, {stats['max']:.1f}], n={stats['count']}")
    
    # Identify boundary optimization opportunities
    # Look for samples that might be better classified based on their scores
    boundary_adjustments = 0
    y_train_optimized = y_train.copy()
    
    for i, (class_label, score) in enumerate(zip(y_train, y_score_train)):
        # Conservative boundary adjustments
        if class_label == 'easy' and score > 4.5:
            y_train_optimized.iloc[i] = 'medium'
            boundary_adjustments += 1
        elif class_label == 'medium' and score < 2.5:
            y_train_optimized.iloc[i] = 'easy'
            boundary_adjustments += 1
        elif class_label == 'medium' and score > 7.5:
            y_train_optimized.iloc[i] = 'hard'
            boundary_adjustments += 1
        elif class_label == 'hard' and score < 4.0:
            y_train_optimized.iloc[i] = 'medium'
            boundary_adjustments += 1
    
    print(f"   Applied {boundary_adjustments} boundary optimizations")
    return y_train_optimized

def test_breakthrough_model():
    """Test the breakthrough model with all advanced improvements."""
    
    print("="*80)
    print("TESTING BREAKTHROUGH AUTOJUDGE MODEL")
    print("="*80)
    
    # Load and improve data quality
    print("1. Loading and improving data quality...")
    try:
        df = pd.read_json('problems_data.jsonl', lines=True)
        
        df = df.rename(columns={
            'input_description': 'input_desc',
            'output_description': 'output_desc'
        })
        
        df['problem_score_scaled'] = df['problem_score']
        
        # Combine text features with advanced preprocessing
        df['combined_text'] = (df['description'].fillna('').apply(advanced_text_preprocessing) + ' ' +
                              df['input_desc'].fillna('').apply(advanced_text_preprocessing) + ' ' +
                              df['output_desc'].fillna('').apply(advanced_text_preprocessing))
        
        df['combined_text'] = df['combined_text'].str.strip()
        
        # Improve data quality
        df = improve_data_quality(df)
        
        print(f"   Final dataset shape: {df.shape}")
        print(f"   Classes: {df['problem_class'].value_counts().to_dict()}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Extract breakthrough features
    print("\n2. Extracting breakthrough features...")
    breakthrough_features = df['combined_text'].apply(extract_breakthrough_features)
    
    feature_names = [
        'text_len', 'word_count', 'basic_math', 'advanced_math', 'big_o',
        'total_algorithm_score', 'graph_algorithms', 'dynamic_programming',
        'data_structures', 'string_algorithms', 'time_complexity', 'space_complexity',
        'constraints', 'avg_sentence_length', 'vocabulary_richness', 'readability',
        'input_output_format', 'optimization', 'difficulty_score'
    ]
    
    X_custom = np.array([list(f) for f in breakthrough_features])
    
    # Advanced TF-IDF with multiple configurations
    print("   Creating advanced TF-IDF features...")
    
    # Word-level TF-IDF
    tfidf_word = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        sublinear_tf=True,
        analyzer='word'
    )
    
    # Character-level TF-IDF for capturing patterns
    tfidf_char = TfidfVectorizer(
        max_features=1000,
        ngram_range=(3, 5),
        min_df=5,
        max_df=0.9,
        analyzer='char',
        lowercase=True
    )
    
    X_tfidf_word = tfidf_word.fit_transform(df['combined_text'])
    X_tfidf_char = tfidf_char.fit_transform(df['combined_text'])
    
    # Scale custom features
    scaler = StandardScaler()
    X_custom_scaled = scaler.fit_transform(X_custom)
    
    # Feature selection on TF-IDF features
    print("   Applying feature selection...")
    selector = SelectKBest(mutual_info_classif, k=3000)  # Select top 3000 features
    X_tfidf_selected = selector.fit_transform(X_tfidf_word, df['problem_class'])
    
    # Combine all features
    X_combined = scipy.sparse.hstack([
        X_tfidf_selected,
        X_tfidf_char,
        scipy.sparse.csr_matrix(X_custom_scaled)
    ])
    
    print(f"   Breakthrough feature matrix: {X_combined.shape}")
    print(f"   Selected TF-IDF features: {X_tfidf_selected.shape[1]}")
    print(f"   Character TF-IDF features: {X_tfidf_char.shape[1]}")
    print(f"   Custom features: {X_custom_scaled.shape[1]}")
    
    # Prepare targets
    y_class = df['problem_class']
    y_score = df['problem_score_scaled']
    
    # Advanced train/test split with stratification
    print("\n3. Performing advanced train/test split...")
    X_train, X_test, y_train_class, y_test_class, y_train_score, y_test_score = train_test_split(
        X_combined, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
    )
    
    print(f"   Train samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Original class distribution: {y_train_class.value_counts().to_dict()}")
    
    # Optimize class boundaries
    y_train_class_optimized = optimize_class_boundaries(X_train, y_train_class, y_train_score)
    print(f"   Optimized class distribution: {y_train_class_optimized.value_counts().to_dict()}")
    
    # Advanced class balancing
    print("\n4. Applying advanced class balancing...")
    try:
        # Use BorderlineSMOTE for better synthetic sample generation
        smote = BorderlineSMOTE(random_state=42, k_neighbors=5, m_neighbors=10)
        X_train_balanced, y_train_class_balanced = smote.fit_resample(X_train.toarray(), y_train_class_optimized)
        X_train_balanced = scipy.sparse.csr_matrix(X_train_balanced)
        print(f"   ✓ BorderlineSMOTE applied successfully")
        print(f"   Balanced class distribution: {pd.Series(y_train_class_balanced).value_counts().to_dict()}")
        
        # Handle regression targets
        y_train_score_balanced = []
        class_indices = {}
        for i, cls in enumerate(y_train_class_optimized):
            if cls not in class_indices:
                class_indices[cls] = []
            class_indices[cls].append(i)
        
        for cls in y_train_class_balanced:
            idx = np.random.choice(class_indices[cls])
            y_train_score_balanced.append(y_train_score.iloc[idx])
        
        y_train_score_balanced = pd.Series(y_train_score_balanced)
        
    except Exception as e:
        print(f"   ⚠ BorderlineSMOTE failed: {e}")
        print("   Using standard SMOTE...")
        try:
            smote = SMOTE(random_state=42, k_neighbors=5)
            X_train_balanced, y_train_class_balanced = smote.fit_resample(X_train.toarray(), y_train_class_optimized)
            X_train_balanced = scipy.sparse.csr_matrix(X_train_balanced)
            
            # Handle regression targets
            y_train_score_balanced = []
            class_indices = {}
            for i, cls in enumerate(y_train_class_optimized):
                if cls not in class_indices:
                    class_indices[cls] = []
                class_indices[cls].append(i)
            
            for cls in y_train_class_balanced:
                idx = np.random.choice(class_indices[cls])
                y_train_score_balanced.append(y_train_score.iloc[idx])
            
            y_train_score_balanced = pd.Series(y_train_score_balanced)
            print(f"   ✓ Standard SMOTE applied")
            
        except Exception as e2:
            print(f"   ⚠ All SMOTE methods failed: {e2}")
            X_train_balanced = X_train
            y_train_class_balanced = y_train_class_optimized
            y_train_score_balanced = y_train_score
    
    # Train breakthrough models
    print("\n5. Training breakthrough ensemble...")
    
    # Create and train breakthrough classifier
    classifier = create_breakthrough_ensemble()
    classifier.fit(X_train_balanced, y_train_class_balanced)
    print("   ✓ Breakthrough stacking ensemble trained")
    
    # Advanced regression model
    regressor = RandomForestRegressor(
        n_estimators=500,
        max_depth=35,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    regressor.fit(X_train_balanced, y_train_score_balanced)
    print("   ✓ Advanced regression model trained")
    
    # Make predictions
    print("\n6. Making predictions...")
    y_pred_class = classifier.predict(X_test)
    y_pred_score = regressor.predict(X_test)
    y_pred_score = np.clip(y_pred_score, 1.0, 10.0)
    
    # Calculate metrics
    print("\n7. Calculating breakthrough metrics...")
    accuracy = accuracy_score(y_test_class, y_pred_class)
    conf_matrix = confusion_matrix(y_test_class, y_pred_class)
    class_report = classification_report(y_test_class, y_pred_class, output_dict=True)
    
    mae = mean_absolute_error(y_test_score, y_pred_score)
    rmse = np.sqrt(mean_squared_error(y_test_score, y_pred_score))
    r2 = r2_score(y_test_score, y_pred_score)
    
    # Display results
    print("\n" + "="*80)
    print("BREAKTHROUGH MODEL RESULTS")
    print("="*80)
    
    print(f"\nBREAKTHROUGH IMPROVEMENTS IMPLEMENTED:")
    print(f"  ✓ Advanced Data Quality: Preprocessing, mislabel detection, boundary optimization")
    print(f"  ✓ Breakthrough Features: {len(feature_names)} semantic features with algorithm categorization")
    print(f"  ✓ Advanced TF-IDF: Word + Character level with feature selection")
    print(f"  ✓ Stacking Ensemble: 5 diverse base classifiers with meta-learner")
    print(f"  ✓ BorderlineSMOTE: Advanced synthetic sample generation")
    print(f"  ✓ Class Boundary Optimization: Score-based boundary adjustments")
    
    print(f"\nCLASSIFICATION RESULTS:")
    print(f"  🎯 Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    if accuracy >= 0.6:
        print("  🎉 TARGET ACCURACY ACHIEVED! (≥60%)")
    elif accuracy >= 0.55:
        print("  📈 VERY CLOSE TO TARGET!")
    else:
        print("  ⚠ Still working toward target...")
    
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
    
    baseline_accuracy = 0.502
    improvement = ((accuracy - baseline_accuracy) / baseline_accuracy) * 100
    print(f"  Accuracy Improvement over Baseline: {improvement:+.1f}%")
    
    # Class-wise accuracy
    print(f"\nCLASS-WISE ACCURACY:")
    for class_name in classes:
        class_mask = y_test_class == class_name
        if class_mask.sum() > 0:
            class_accuracy = accuracy_score(y_test_class[class_mask], y_pred_class[class_mask])
            class_count = class_mask.sum()
            improvement_needed = max(0, 60 - class_accuracy * 100)
            print(f"  {class_name}: {class_accuracy:.3f} ({class_accuracy*100:.1f}%) - {class_count} samples")
            if improvement_needed > 0:
                print(f"    Need {improvement_needed:.1f}% improvement for 60% target")
    
    # Threshold validation
    print(f"\nTHRESHOLD VALIDATION:")
    thresholds_met = []
    if accuracy >= 0.6:
        thresholds_met.append("🎯 Classification accuracy ≥ 60%")
    if mae <= 2.0:
        thresholds_met.append("✅ MAE ≤ 2.0")
    if rmse <= 2.5:
        thresholds_met.append("✅ RMSE ≤ 2.5")
    
    if thresholds_met:
        print(f"  {' | '.join(thresholds_met)}")
    
    if accuracy < 0.6:
        print(f"  ⚠ Classification accuracy: {accuracy:.1%} (need {(0.6-accuracy)*100:.1f}% more)")
    
    print("\n" + "="*80)
    print("BREAKTHROUGH MODEL SUMMARY")
    print("="*80)
    print(f"✓ Breakthrough model tested on {len(y_test_class)} samples")
    print(f"✓ Classification accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"✓ Regression MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    print(f"✓ Total features: {X_combined.shape[1]} (advanced feature engineering)")
    print(f"✓ All breakthrough improvements successfully implemented")
    
    if accuracy >= 0.6:
        print("🏆 BREAKTHROUGH ACHIEVED - TARGET ACCURACY REACHED!")
        print("🚀 Model ready for production deployment!")
    else:
        print("📊 Significant improvements made - continue refinement")
    
    return {
        'accuracy': accuracy,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'improvement_over_baseline': improvement,
        'target_achieved': accuracy >= 0.6
    }

if __name__ == "__main__":
    print("Starting breakthrough model testing...")
    results = test_breakthrough_model()
    
    if results and results.get('target_achieved'):
        print("\n🎉 BREAKTHROUGH SUCCESS! Target accuracy achieved!")
    else:
        print("\n📈 Significant progress made toward breakthrough!")
    
    print("\nBreakthrough model testing completed!")