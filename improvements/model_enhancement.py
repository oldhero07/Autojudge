#!/usr/bin/env python3
"""
Model Enhancement Strategies for AutoJudge
Current: 55.0% accuracy -> Target: 70%+
"""

# 1. Advanced Feature Engineering
def extract_enhanced_features(text):
    """Enhanced feature extraction with domain expertise"""
    
    # Algorithm complexity indicators
    complexity_keywords = {
        'easy': ['print', 'input', 'output', 'simple', 'basic'],
        'medium': ['sort', 'search', 'loop', 'array', 'string'],
        'hard': ['dynamic programming', 'graph', 'tree', 'optimization', 'complexity']
    }
    
    # Mathematical complexity
    math_indicators = [
        'O(n)', 'O(log n)', 'O(n^2)', 'O(n log n)',
        'time complexity', 'space complexity',
        'constraint', 'limit', '10^6', '10^9'
    ]
    
    # Problem domain classification
    domains = {
        'graph': ['graph', 'tree', 'node', 'edge', 'path', 'cycle'],
        'dp': ['dynamic', 'programming', 'memoization', 'optimal'],
        'string': ['string', 'substring', 'character', 'palindrome'],
        'math': ['number', 'prime', 'factorial', 'modulo', 'gcd']
    }
    
    return extract_features(text, complexity_keywords, math_indicators, domains)

# 2. Advanced Model Architecture
def create_enhanced_model():
    """Enhanced ensemble with better algorithms"""
    from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
    from xgboost import XGBClassifier
    
    models = [
        ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)),
        ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.1)),
        ('et', ExtraTreesClassifier(n_estimators=200, max_depth=20))
    ]
    
    return VotingClassifier(estimators=models, voting='soft')

# 3. Better Text Processing
def advanced_text_preprocessing(text):
    """Enhanced text preprocessing"""
    import re
    
    # Expand programming abbreviations
    abbreviations = {
        'dfs': 'depth first search',
        'bfs': 'breadth first search', 
        'dp': 'dynamic programming',
        'lca': 'lowest common ancestor',
        'mst': 'minimum spanning tree'
    }
    
    # Clean and normalize
    text = text.lower()
    for abbr, full in abbreviations.items():
        text = re.sub(rf'\b{abbr}\b', full, text)
    
    return text