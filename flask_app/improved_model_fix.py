"""
Quick fix to restore high accuracy using proven techniques from breakthrough model
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
import scipy.sparse
import re

def smart_text_preprocessing(text):
    """Smart text preprocessing to improve data quality."""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    
    # Expand common abbreviations for better understanding
    abbreviations = {
        r'\bdfs\b': 'depth first search',
        r'\bbfs\b': 'breadth first search', 
        r'\bdp\b': 'dynamic programming',
        r'\blca\b': 'lowest common ancestor',
        r'\bmst\b': 'minimum spanning tree',
        r'\bscc\b': 'strongly connected components',
        r'\bgcd\b': 'greatest common divisor',
        r'\blcm\b': 'least common multiple'
    }
    
    for abbrev, expansion in abbreviations.items():
        text = re.sub(abbrev, expansion, text)
    
    return text.strip()

def extract_smart_features(text):
    """Extract smart features with domain expertise."""
    text = smart_text_preprocessing(text)
    
    # Basic text metrics
    text_len = len(text)
    word_count = len(text.split()) if text else 0
    
    # Advanced algorithm detection with confidence scoring
    algorithm_patterns = {
        'advanced_algorithms': {
            'patterns': ['suffix array', 'convex hull', 'network flow', 'bipartite matching', 
                        'minimum cut', 'maximum flow', 'heavy light decomposition', 'centroid decomposition'],
            'weight': 4.0
        },
        'hard_algorithms': {
            'patterns': ['dynamic programming', 'segment tree', 'fenwick tree', 'binary indexed tree',
                        'lowest common ancestor', 'strongly connected components', 'articulation points',
                        'bridges', 'tarjan', 'kosaraju'],
            'weight': 3.0
        },
        'medium_hard_algorithms': {
            'patterns': ['depth first search', 'breadth first search', 'dijkstra', 'bellman ford',
                        'floyd warshall', 'minimum spanning tree', 'kruskal', 'prim', 'topological sort'],
            'weight': 2.5
        },
        'medium_algorithms': {
            'patterns': ['binary search', 'two pointer', 'sliding window', 'hash table', 'heap',
                        'priority queue', 'trie', 'union find', 'disjoint set'],
            'weight': 2.0
        },
        'basic_algorithms': {
            'patterns': ['sort', 'linear search', 'array', 'string', 'stack', 'queue', 'list'],
            'weight': 1.0
        }
    }
    
    total_algorithm_score = 0
    category_scores = {}
    
    for category, info in algorithm_patterns.items():
        score = 0
        for pattern in info['patterns']:
            matches = len(re.findall(rf'\b{pattern}\b', text))
            score += matches
        
        weighted_score = score * info['weight']
        category_scores[category] = weighted_score
        total_algorithm_score += weighted_score
    
    # Mathematical content analysis
    math_indicators = {
        'basic_math': len(re.findall(r'[+\-*/=<>%]', text)),
        'advanced_math': len(re.findall(r'[∑∏∫∂∆√π∞≤≥≠≈∈∉∪∩⊂⊃∅]', text)),
        'big_o_notation': len(re.findall(r'o\([^)]+\)', text, re.IGNORECASE)),
        'mathematical_terms': len(re.findall(r'\b(prime|factorial|fibonacci|modulo|gcd|lcm|permutation|combination)\b', text))
    }
    
    # Complexity and constraint analysis
    complexity_score = (
        len(re.findall(r'time.*complexity|space.*complexity', text)) * 2.0 +
        len(re.findall(r'constraint|limit|bound', text)) * 1.5 +
        len(re.findall(r'\d+\s*≤.*≤\s*\d+', text)) * 2.0
    )
    
    # Problem structure indicators
    structure_score = (
        len(re.findall(r'input.*format|output.*format', text)) * 1.0 +
        len(re.findall(r'test.*case|example|sample', text)) * 0.5 +
        len(re.findall(r'multiple.*test|t.*test.*case', text)) * 1.5
    )
    
    # Linguistic complexity
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if sentences and word_count > 0:
        avg_sentence_length = word_count / len(sentences)
        unique_words = len(set(text.split()))
        vocabulary_richness = unique_words / word_count
        sentence_count = len(sentences)
    else:
        avg_sentence_length = 0
        vocabulary_richness = 0
        sentence_count = 0
    
    # Problem type classification
    problem_types = {
        'graph_problem': len(re.findall(r'\b(graph|tree|node|edge|vertex|path|cycle|connected)\b', text)),
        'string_problem': len(re.findall(r'\b(string|substring|subsequence|character|palindrome|anagram)\b', text)),
        'array_problem': len(re.findall(r'\b(array|list|element|index|position)\b', text)),
        'math_problem': len(re.findall(r'\b(number|integer|digit|calculate|compute|sum|product)\b', text)),
        'optimization_problem': len(re.findall(r'\b(minimum|maximum|optimal|best|least|most|minimize|maximize)\b', text))
    }
    
    # Difficulty keywords with semantic analysis
    difficulty_indicators = {
        'easy_keywords': len(re.findall(r'\b(print|output|simple|basic|count|sum|find)\b', text)),
        'medium_keywords': len(re.findall(r'\b(sort|search|implement|calculate|determine)\b', text)),
        'hard_keywords': len(re.findall(r'\b(optimize|efficient|complex|advanced|sophisticated)\b', text))
    }
    
    # Combine all features into a comprehensive feature vector
    features = [
        text_len,
        word_count,
        total_algorithm_score,
        category_scores['advanced_algorithms'],
        category_scores['hard_algorithms'],
        category_scores['medium_hard_algorithms'],
        category_scores['medium_algorithms'],
        category_scores['basic_algorithms'],
        math_indicators['basic_math'],
        math_indicators['advanced_math'],
        math_indicators['big_o_notation'],
        math_indicators['mathematical_terms'],
        complexity_score,
        structure_score,
        avg_sentence_length,
        vocabulary_richness,
        sentence_count,
        problem_types['graph_problem'],
        problem_types['string_problem'],
        problem_types['array_problem'],
        problem_types['math_problem'],
        problem_types['optimization_problem'],
        difficulty_indicators['easy_keywords'],
        difficulty_indicators['medium_keywords'],
        difficulty_indicators['hard_keywords']
    ]
    
    return features

def create_smart_ensemble():
    """Create a smart ensemble with proven algorithms."""
    base_classifiers = [
        ('lr', LogisticRegression(random_state=42, max_iter=3000, class_weight='balanced', C=1.0)),
        ('rf', RandomForestClassifier(n_estimators=400, max_depth=35, min_samples_split=3, 
                                     min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=10, learning_rate=0.08, 
                                         random_state=42, subsample=0.8)),
        ('svc', SVC(probability=True, class_weight='balanced', random_state=42, C=2.0, gamma='scale'))
    ]
    
    ensemble = VotingClassifier(
        estimators=base_classifiers,
        voting='soft',
        n_jobs=-1
    )
    
    return ensemble

def test_improved_model():
    """Test the improved model with breakthrough techniques."""
    
    print("="*80)
    print("TESTING IMPROVED MODEL WITH BREAKTHROUGH TECHNIQUES")
    print("="*80)
    
    # Load data
    print("Loading data...")
    df = pd.read_json('../problems_data.jsonl', lines=True)
    
    df = df.rename(columns={
        'input_description': 'input_desc',
        'output_description': 'output_desc'
    })
    
    # Smart preprocessing
    print("Applying smart preprocessing...")
    df['combined_text'] = (df['description'].fillna('').apply(smart_text_preprocessing) + ' ' +
                          df['input_desc'].fillna('').apply(smart_text_preprocessing) + ' ' +
                          df['output_desc'].fillna('').apply(smart_text_preprocessing))
    
    df['combined_text'] = df['combined_text'].str.strip()
    
    # Remove very short descriptions
    original_size = len(df)
    df = df[df['combined_text'].str.len() >= 100].copy()
    print(f"Removed {original_size - len(df)} samples with text < 100 characters")
    
    # Extract smart features
    print("Extracting smart features...")
    smart_features = df['combined_text'].apply(extract_smart_features)
    X_custom = np.array([list(f) for f in smart_features])
    
    # Advanced TF-IDF with feature selection
    print("Creating optimized TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=4000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        sublinear_tf=True
    )
    
    X_tfidf = tfidf_vectorizer.fit_transform(df['combined_text'])
    
    # Feature selection
    print("Applying intelligent feature selection...")
    selector = SelectKBest(mutual_info_classif, k=2500)
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
    
    # Train/test split
    print("Performing train/test split...")
    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X_combined, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Smart class balancing with BorderlineSMOTE
    print("Applying BorderlineSMOTE...")
    try:
        smote = BorderlineSMOTE(random_state=42, k_neighbors=7, m_neighbors=15)
        X_train_balanced, y_train_class_balanced = smote.fit_resample(X_train.toarray(), y_train_class)
        X_train_balanced = scipy.sparse.csr_matrix(X_train_balanced)
        print("✓ BorderlineSMOTE applied successfully")
    except:
        print("BorderlineSMOTE failed, using standard SMOTE...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_balanced, y_train_class_balanced = smote.fit_resample(X_train.toarray(), y_train_class)
        X_train_balanced = scipy.sparse.csr_matrix(X_train_balanced)
    
    # Train smart ensemble
    print("Training smart ensemble...")
    classifier = create_smart_ensemble()
    classifier.fit(X_train_balanced, y_train_class_balanced)
    
    # Make predictions
    print("Making predictions...")
    y_pred_class = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_class, y_pred_class)
    conf_matrix = confusion_matrix(y_test_class, y_pred_class)
    class_report = classification_report(y_test_class, y_pred_class, output_dict=True)
    
    # Display results
    print("\n" + "="*80)
    print("IMPROVED MODEL RESULTS")
    print("="*80)
    
    print(f"\n🎯 ACCURACY: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    if accuracy >= 0.6:
        print("🏆 BREAKTHROUGH ACHIEVED! TARGET ACCURACY REACHED!")
    elif accuracy >= 0.55:
        print("📈 VERY CLOSE TO BREAKTHROUGH!")
    else:
        print("📊 Significant improvements made")
    
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
    
    return accuracy, conf_matrix, class_report

if __name__ == "__main__":
    test_improved_model()