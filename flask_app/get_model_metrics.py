"""
Script to extract detailed model metrics including confusion matrix
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import scipy.sparse
import re

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
    # Load the dataset
    df = pd.read_json('../problems_data.jsonl', lines=True)
    
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
    
    # Extract custom features
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
    
    return df

def train_and_evaluate():
    """Train models and get detailed metrics."""
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    
    # Prepare features and targets
    X_text = df['combined_text']
    X_custom = df[['text_len', 'word_count', 'graph_algorithms', 'dynamic_programming', 'data_structures',
                  'sorting_searching', 'string_processing', 'basic_math', 'advanced_math', 'complexity_notation',
                  'constraints', 'optimization', 'multiple_cases', 'vocabulary_richness', 'avg_word_length']].values
    y_class = df['problem_class']
    y_score = df['problem_score_scaled']
    
    print("Creating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words='english', 
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        sublinear_tf=True
    )
    X_tfidf = tfidf_vectorizer.fit_transform(X_text)
    
    print("Scaling custom features...")
    feature_scaler = StandardScaler()
    X_custom_scaled = feature_scaler.fit_transform(X_custom)
    
    print("Combining features...")
    X_combined = scipy.sparse.hstack([X_tfidf, scipy.sparse.csr_matrix(X_custom_scaled)])
    
    print("Performing train/test split...")
    X_train, X_test, y_train_class, y_test_class, y_train_score, y_test_score = train_test_split(
        X_combined, y_class, y_score,
        test_size=0.2, 
        random_state=42,
        stratify=y_class
    )
    
    print("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy='auto')
    X_train_balanced, y_train_class_balanced = smote.fit_resample(X_train.toarray(), y_train_class)
    X_train_balanced = scipy.sparse.csr_matrix(X_train_balanced)
    
    print("Training classification model...")
    lr_classifier = LogisticRegression(
        random_state=42, 
        max_iter=2000,
        class_weight='balanced',
        C=1.0,
        solver='lbfgs'
    )
    
    rf_classifier = RandomForestClassifier(
        n_estimators=250,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    classifier_pipeline = VotingClassifier(
        estimators=[
            ('lr', lr_classifier),
            ('rf', rf_classifier)
        ],
        voting='soft'
    )
    
    classifier_pipeline.fit(X_train_balanced, y_train_class_balanced)
    
    print("Making predictions...")
    y_pred_class = classifier_pipeline.predict(X_test)
    
    # Calculate detailed metrics
    accuracy = accuracy_score(y_test_class, y_pred_class)
    conf_matrix = confusion_matrix(y_test_class, y_pred_class)
    class_report = classification_report(y_test_class, y_pred_class, output_dict=True)
    
    return accuracy, conf_matrix, class_report, y_test_class, y_pred_class

def print_detailed_metrics():
    """Print detailed model metrics."""
    accuracy, conf_matrix, class_report, y_test, y_pred = train_and_evaluate()
    
    print("\n" + "="*80)
    print("ENHANCED ML MODEL - DETAILED PERFORMANCE METRICS")
    print("="*80)
    
    print(f"\n📊 OVERALL ACCURACY: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    print(f"\n🎯 CONFUSION MATRIX:")
    print("    Predicted:")
    print("         easy  medium  hard")
    print("Actual:")
    
    class_names = ['easy', 'medium', 'hard']
    for i, actual_class in enumerate(class_names):
        row_str = f"{actual_class:>6} "
        for j in range(len(class_names)):
            row_str += f"{conf_matrix[i][j]:>6} "
        print(row_str)
    
    print(f"\n📈 DETAILED CLASSIFICATION REPORT:")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 50)
    
    for class_name in class_names:
        if class_name in class_report:
            metrics = class_report[class_name]
            print(f"{class_name:<10} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} {metrics['f1-score']:<10.3f} {int(metrics['support']):<10}")
    
    # Macro and weighted averages
    print("-" * 50)
    macro_avg = class_report['macro avg']
    weighted_avg = class_report['weighted avg']
    print(f"{'macro avg':<10} {macro_avg['precision']:<10.3f} {macro_avg['recall']:<10.3f} {macro_avg['f1-score']:<10.3f} {int(macro_avg['support']):<10}")
    print(f"{'weighted avg':<10} {weighted_avg['precision']:<10.3f} {weighted_avg['recall']:<10.3f} {weighted_avg['f1-score']:<10.3f} {int(weighted_avg['support']):<10}")
    
    print(f"\n🔍 CLASS DISTRIBUTION IN TEST SET:")
    test_distribution = pd.Series(y_test).value_counts().sort_index()
    for class_name, count in test_distribution.items():
        percentage = (count / len(y_test)) * 100
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n⚡ MODEL PERFORMANCE ANALYSIS:")
    
    # Calculate per-class accuracy
    for i, class_name in enumerate(class_names):
        class_correct = conf_matrix[i][i]
        class_total = conf_matrix[i].sum()
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        print(f"  {class_name.capitalize()} class accuracy: {class_accuracy:.3f} ({class_accuracy*100:.1f}%)")
    
    # Most confused classes
    print(f"\n🤔 MOST COMMON MISCLASSIFICATIONS:")
    total_samples = conf_matrix.sum()
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and conf_matrix[i][j] > 0:
                error_rate = conf_matrix[i][j] / total_samples * 100
                print(f"  {class_names[i]} → {class_names[j]}: {conf_matrix[i][j]} samples ({error_rate:.1f}%)")
    
    print("="*80)

if __name__ == "__main__":
    print_detailed_metrics()