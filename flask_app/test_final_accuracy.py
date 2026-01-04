#!/usr/bin/env python3
"""
Test the final accuracy using the breakthrough model approach
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
import scipy.sparse
import re

def advanced_text_preprocessing(text):
    """Advanced text preprocessing with domain knowledge."""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'\\s+', ' ', text)
    
    # Comprehensive abbreviation expansion
    expansions = {
        r'\\bdfs\\b': 'depth first search',
        r'\\bbfs\\b': 'breadth first search',
        r'\\bdp\\b': 'dynamic programming',
        r'\\blca\\b': 'lowest common ancestor',
        r'\\bmst\\b': 'minimum spanning tree',
        r'\\bscc\\b': 'strongly connected components',
        r'\\bgcd\\b': 'greatest common divisor',
        r'\\blcm\\b': 'least common multiple',
        r'\\bbst\\b': 'binary search tree',
        r'\\bavl\\b': 'avl tree',
        r'\\brb\\b': 'red black tree',
        r'\\bkmp\\b': 'knuth morris pratt',
        r'\\bbit\\b': 'binary indexed tree',
        r'\\bfft\\b': 'fast fourier transform'
    }
    
    for pattern, replacement in expansions.items():
        text = re.sub(pattern, replacement, text)
    
    return text.strip()

def extract_ultimate_features(text):
    """Extract ultimate feature set for maximum accuracy."""
    text = advanced_text_preprocessing(text)
    
    # Basic metrics
    text_len = len(text)
    word_count = len(text.split()) if text else 0
    
    # Ultra-specific algorithm detection (5 difficulty levels)
    ultra_hard_patterns = [
        'suffix array', 'convex hull', 'network flow', 'maximum flow', 'minimum cut',
        'bipartite matching', 'heavy light decomposition', 'centroid decomposition',
        'link cut tree', 'persistent segment tree', 'sqrt decomposition',
        'mo algorithm', 'aho corasick', 'z algorithm', 'manacher'
    ]
    
    hard_patterns = [
        'dynamic programming', 'segment tree', 'fenwick tree', 'binary indexed tree',
        'lowest common ancestor', 'strongly connected components', 'articulation points',
        'bridges', 'tarjan', 'kosaraju', 'suffix tree', 'trie', 'kmp algorithm',
        'rabin karp', 'rolling hash'
    ]
    
    medium_hard_patterns = [
        'depth first search', 'breadth first search', 'dijkstra', 'bellman ford',
        'floyd warshall', 'minimum spanning tree', 'kruskal', 'prim', 'topological sort',
        'binary search tree', 'avl tree', 'red black tree', 'heap', 'priority queue'
    ]
    
    medium_patterns = [
        'binary search', 'two pointer', 'sliding window', 'hash table', 'hash map',
        'union find', 'disjoint set', 'merge sort', 'quick sort', 'counting sort'
    ]
    
    basic_patterns = [
        'linear search', 'bubble sort', 'selection sort', 'insertion sort',
        'array', 'string', 'stack', 'queue', 'linked list'
    ]
    
    # Calculate weighted algorithm scores
    ultra_hard_score = sum(len(re.findall(rf'\\b{pattern}\\b', text)) for pattern in ultra_hard_patterns) * 5.0
    hard_score = sum(len(re.findall(rf'\\b{pattern}\\b', text)) for pattern in hard_patterns) * 4.0
    medium_hard_score = sum(len(re.findall(rf'\\b{pattern}\\b', text)) for pattern in medium_hard_patterns) * 3.0
    medium_score = sum(len(re.findall(rf'\\b{pattern}\\b', text)) for pattern in medium_patterns) * 2.0
    basic_score = sum(len(re.findall(rf'\\b{pattern}\\b', text)) for pattern in basic_patterns) * 1.0
    
    total_algorithm_score = ultra_hard_score + hard_score + medium_hard_score + medium_score + basic_score
    
    # Advanced mathematical content analysis
    advanced_math_symbols = len(re.findall(r'[∑∏∫∂∆√π∞≤≥≠≈∈∉∪∩⊂⊃∅⊆⊇∧∨¬→↔∀∃]', text))
    mathematical_functions = len(re.findall(r'\\b(sin|cos|tan|log|ln|exp|sqrt|abs|floor|ceil|mod|gcd|lcm|factorial|fibonacci|prime|composite)\\b', text))
    big_o_notation = len(re.findall(r'o\\([^)]+\\)', text, re.IGNORECASE)) * 3.0
    
    math_complexity_score = advanced_math_symbols * 3.0 + mathematical_functions * 2.0 + big_o_notation
    
    # Problem complexity and constraint analysis
    time_complexity_mentions = len(re.findall(r'time.*complexity|running.*time|time.*limit', text)) * 4.0
    space_complexity_mentions = len(re.findall(r'space.*complexity|memory.*limit|space.*limit', text)) * 3.0
    constraint_patterns = len(re.findall(r'constraint|limit|bound|\\d+\\s*≤.*≤\\s*\\d+|1\\s*≤.*≤\\s*10\\^\\d+', text)) * 2.0
    
    complexity_score = time_complexity_mentions + space_complexity_mentions + constraint_patterns
    
    # Problem domain classification
    graph_indicators = len(re.findall(r'\\b(graph|tree|node|edge|vertex|path|cycle|connected|component|forest|dag|directed|undirected|weighted|unweighted)\\b', text))
    string_indicators = len(re.findall(r'\\b(string|substring|subsequence|character|palindrome|anagram|pattern|text|word|sentence|lexicographic)\\b', text))
    array_indicators = len(re.findall(r'\\b(array|list|element|index|position|subarray|subsequence|permutation|combination)\\b', text))
    number_theory_indicators = len(re.findall(r'\\b(prime|composite|divisor|multiple|modular|arithmetic|geometric|sequence|series)\\b', text))
    geometry_indicators = len(re.findall(r'\\b(point|line|circle|polygon|triangle|rectangle|coordinate|distance|angle|area|perimeter)\\b', text))
    
    # Optimization and difficulty keywords
    optimization_keywords = len(re.findall(r'\\b(minimum|maximum|optimal|best|least|most|minimize|maximize|efficient|optimize)\\b', text))
    difficulty_easy = len(re.findall(r'\\b(print|output|simple|basic|count|sum|find|easy|straightforward)\\b', text))
    difficulty_hard = len(re.findall(r'\\b(complex|advanced|sophisticated|difficult|challenging|tricky|non.?trivial)\\b', text))
    
    # Text structure and linguistic features
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)
    
    if word_count > 0:
        unique_words = len(set(text.split()))
        vocabulary_richness = unique_words / word_count
        avg_word_length = sum(len(word) for word in text.split()) / word_count
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    else:
        vocabulary_richness = 0
        avg_word_length = 0
        avg_sentence_length = 0
    
    # Input/output format complexity
    io_complexity = (
        len(re.findall(r'input.*format|output.*format', text)) * 1.5 +
        len(re.findall(r'test.*case|multiple.*test|t.*test.*case', text)) * 1.0 +
        len(re.findall(r'example|sample', text)) * 0.5
    )
    
    # Competitive programming specific terms
    competitive_terms = len(re.findall(r'\\b(contest|competition|judge|verdict|accepted|wrong.*answer|time.*limit.*exceeded|memory.*limit.*exceeded)\\b', text))
    
    return [
        text_len, word_count, total_algorithm_score, ultra_hard_score, hard_score, 
        medium_hard_score, medium_score, basic_score, math_complexity_score,
        advanced_math_symbols, mathematical_functions, big_o_notation, complexity_score,
        time_complexity_mentions, space_complexity_mentions, constraint_patterns,
        graph_indicators, string_indicators, array_indicators, number_theory_indicators,
        geometry_indicators, optimization_keywords, difficulty_easy, difficulty_hard,
        sentence_count, vocabulary_richness, avg_word_length, avg_sentence_length,
        io_complexity, competitive_terms
    ]

def test_final_accuracy():
    """Test the final accuracy with the updated app.py approach."""
    
    print("🚀 TESTING FINAL ACCURACY WITH UPDATED APP.PY APPROACH")
    print("=" * 70)
    
    # Load data
    print("Loading data...")
    df = pd.read_json('../problems_data.jsonl', lines=True)
    
    df = df.rename(columns={
        'input_description': 'input_desc',
        'output_description': 'output_desc'
    })
    
    # Advanced preprocessing
    print("Advanced preprocessing...")
    df['combined_text'] = (df['description'].fillna('').apply(advanced_text_preprocessing) + ' ' +
                          df['input_desc'].fillna('').apply(advanced_text_preprocessing) + ' ' +
                          df['output_desc'].fillna('').apply(advanced_text_preprocessing))
    
    df['combined_text'] = df['combined_text'].str.strip()
    
    # Quality filtering (same as app.py)
    original_size = len(df)
    df = df[df['combined_text'].str.len() >= 80].copy()
    print(f"Filtered from {original_size} to {len(df)} samples")
    
    # Extract ultimate features
    print("Extracting ultimate feature set...")
    ultimate_features = df['combined_text'].apply(extract_ultimate_features)
    X_custom = np.array([list(f) for f in ultimate_features])
    
    # Optimized TF-IDF with feature selection
    print("Creating optimized TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=4000,
        stop_words='english',
        ngram_range=(1, 3),  # Include trigrams
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        analyzer='word'
    )
    
    X_tfidf = tfidf_vectorizer.fit_transform(df['combined_text'])
    
    # Feature selection for TF-IDF
    print("Applying feature selection...")
    selector = SelectKBest(chi2, k=3000)
    X_tfidf_selected = selector.fit_transform(X_tfidf, df['problem_class'])
    
    # Scale custom features
    scaler = StandardScaler()
    X_custom_scaled = scaler.fit_transform(X_custom)
    
    # Combine features
    X_combined = scipy.sparse.hstack([
        X_tfidf_selected,
        scipy.sparse.csr_matrix(X_custom_scaled)
    ])
    
    print(f"Final feature matrix: {X_combined.shape}")
    
    # Prepare targets
    y_class = df['problem_class']
    
    # Stratified split
    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X_combined, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Enhanced SMOTE
    print("Applying enhanced SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=7, sampling_strategy='auto')
    X_train_balanced, y_train_class_balanced = smote.fit_resample(X_train.toarray(), y_train_class)
    X_train_balanced = scipy.sparse.csr_matrix(X_train_balanced)
    
    # Ultimate ensemble with three strong classifiers (same as app.py)
    print("Training ultimate ensemble...")
    
    lr_classifier = LogisticRegression(
        random_state=42, 
        max_iter=3000,
        class_weight='balanced',
        C=2.0,
        solver='lbfgs'
    )
    
    rf_classifier = RandomForestClassifier(
        n_estimators=400,
        max_depth=35,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    gb_classifier = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=12,
        learning_rate=0.1,
        random_state=42,
        subsample=0.8
    )
    
    ultimate_ensemble = VotingClassifier(
        estimators=[
            ('lr', lr_classifier),
            ('rf', rf_classifier),
            ('gb', gb_classifier)
        ],
        voting='soft'
    )
    
    ultimate_ensemble.fit(X_train_balanced, y_train_class_balanced)
    
    # Predictions
    print("Making final predictions...")
    y_pred_class = ultimate_ensemble.predict(X_test)
    
    # Calculate final results
    accuracy = accuracy_score(y_test_class, y_pred_class)
    conf_matrix = confusion_matrix(y_test_class, y_pred_class)
    class_report = classification_report(y_test_class, y_pred_class, output_dict=True)
    
    print("\\n" + "=" * 70)
    print("🏆 FINAL UPDATED APP.PY ACCURACY RESULTS")
    print("=" * 70)
    
    print(f"\\n🎯 FINAL ACCURACY: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    if accuracy >= 0.6:
        print("🏆🏆🏆 BREAKTHROUGH ACHIEVED! 60%+ ACCURACY! 🏆🏆🏆")
    elif accuracy >= 0.58:
        print("🔥🔥 EXCELLENT! Very close to breakthrough! 🔥🔥")
    elif accuracy >= 0.55:
        print("📈📈 Great improvement! Getting close! 📈📈")
    else:
        print("✅ Solid progress made")
    
    improvement = accuracy - 0.495  # From original 49.5%
    print(f"📊 Total improvement: +{improvement:.1%} ({improvement*100:.1f} percentage points)")
    
    print(f"\\n🎯 CONFUSION MATRIX:")
    print("         Predicted")
    print("         easy  medium  hard")
    print("Actual:")
    
    class_names = ['easy', 'medium', 'hard']
    for i, actual_class in enumerate(class_names):
        row_str = f"{actual_class:>6} "
        for j in range(len(class_names)):
            row_str += f"{conf_matrix[i][j]:>6} "
        print(row_str)
    
    print(f"\\n📈 PER-CLASS PERFORMANCE:")
    for i, class_name in enumerate(class_names):
        class_correct = conf_matrix[i][i]
        class_total = conf_matrix[i].sum()
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        precision = class_report[class_name]['precision']
        recall = class_report[class_name]['recall']
        f1 = class_report[class_name]['f1-score']
        
        print(f"  {class_name.upper():>6}: Acc={class_accuracy:.3f} Prec={precision:.3f} Rec={recall:.3f} F1={f1:.3f}")
    
    return accuracy

if __name__ == "__main__":
    test_final_accuracy()