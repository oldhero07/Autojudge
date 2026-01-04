#!/usr/bin/env python3
"""
Test the updated app.py with ultimate 59.8% accuracy model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import train_models, classifier_pipeline, PredictionService
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def test_updated_model():
    """Test the updated model and get accuracy metrics."""
    
    print("🚀 TESTING UPDATED APP.PY WITH ULTIMATE MODEL")
    print("=" * 60)
    
    # Train the models
    print("Training models...")
    try:
        train_models()
        print("✅ Models trained successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test predictions
    print("\n🧪 Testing predictions...")
    
    # Test sample problems
    test_cases = [
        {
            'description': 'Given an array of integers, find the maximum sum of any contiguous subarray.',
            'input_desc': 'First line contains n, the size of array. Second line contains n integers.',
            'output_desc': 'Print the maximum sum.',
            'expected_class': 'medium'
        },
        {
            'description': 'Print "Hello World"',
            'input_desc': 'No input',
            'output_desc': 'Print Hello World',
            'expected_class': 'easy'
        },
        {
            'description': 'Implement a segment tree with range minimum query and point updates. Handle multiple queries efficiently.',
            'input_desc': 'First line contains n and q. Next line contains n integers. Next q lines contain queries.',
            'output_desc': 'For each query, print the result.',
            'expected_class': 'hard'
        }
    ]
    
    print("\nSample predictions:")
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = PredictionService.predict_class_and_score(
                test_case['description'],
                test_case['input_desc'],
                test_case['output_desc']
            )
            
            print(f"\n{i}. Expected: {test_case['expected_class']}")
            print(f"   Predicted: {result['class']} (confidence: {result['confidence']:.3f})")
            print(f"   Score: {result['score']}")
            print(f"   Reliable: {result['reliable']}")
            
        except Exception as e:
            print(f"❌ Prediction {i} failed: {e}")
    
    # Load test data and evaluate
    print("\n📊 EVALUATING ON TEST DATA...")
    try:
        # Load the data
        df = pd.read_json('../problems_data.jsonl', lines=True)
        
        # Rename columns to match app.py expectations
        df = df.rename(columns={
            'input_description': 'input_desc',
            'output_description': 'output_desc'
        })
        
        # Filter for quality (same as training)
        df = df[df['description'].str.len() >= 80].copy()
        
        print(f"Evaluating on {len(df)} samples...")
        
        # Make predictions on all samples
        predictions = []
        actual_classes = []
        
        for idx, row in df.iterrows():
            try:
                result = PredictionService.predict_class_and_score(
                    row['description'],
                    row.get('input_desc', ''),
                    row.get('output_desc', '')
                )
                predictions.append(result['class'])
                actual_classes.append(row['problem_class'])
                
                if len(predictions) % 100 == 0:
                    print(f"Processed {len(predictions)} samples...")
                    
            except Exception as e:
                print(f"Skipping sample {idx}: {e}")
                continue
        
        # Calculate metrics
        if len(predictions) > 0:
            accuracy = accuracy_score(actual_classes, predictions)
            conf_matrix = confusion_matrix(actual_classes, predictions)
            class_report = classification_report(actual_classes, predictions, output_dict=True)
            
            print("\n" + "=" * 70)
            print("🏆 UPDATED MODEL PERFORMANCE RESULTS")
            print("=" * 70)
            
            print(f"\n🎯 ACCURACY: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            if accuracy >= 0.6:
                print("🏆🏆🏆 BREAKTHROUGH ACHIEVED! 60%+ ACCURACY! 🏆🏆🏆")
            elif accuracy >= 0.58:
                print("🔥🔥 EXCELLENT! Very close to breakthrough! 🔥🔥")
            elif accuracy >= 0.55:
                print("📈📈 Great improvement! Getting close! 📈📈")
            else:
                print("✅ Model is working")
            
            print(f"\n🎯 CONFUSION MATRIX:")
            print("         Predicted")
            print("         easy  medium  hard")
            print("Actual:")
            
            class_names = ['easy', 'medium', 'hard']
            for i, actual_class in enumerate(class_names):
                row_str = f"{actual_class:>6} "
                for j in range(len(class_names)):
                    row_str += f"{conf_matrix[i][j]:>6} "
                print(row_str)
            
            print(f"\n📈 PER-CLASS PERFORMANCE:")
            for class_name in class_names:
                if class_name in class_report:
                    precision = class_report[class_name]['precision']
                    recall = class_report[class_name]['recall']
                    f1 = class_report[class_name]['f1-score']
                    support = class_report[class_name]['support']
                    
                    print(f"  {class_name.upper():>6}: Prec={precision:.3f} Rec={recall:.3f} F1={f1:.3f} Support={support}")
            
            return accuracy
            
        else:
            print("❌ No successful predictions made")
            return 0.0
            
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

if __name__ == "__main__":
    test_updated_model()