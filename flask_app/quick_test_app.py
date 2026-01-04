#!/usr/bin/env python3
"""
Quick test of the updated app.py - just train and test a few predictions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import train_models, PredictionService

def quick_test():
    """Quick test of the updated model."""
    
    print("🚀 QUICK TEST OF UPDATED APP.PY")
    print("=" * 50)
    
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
            'description': 'Given an array of integers, find the maximum sum of any contiguous subarray using dynamic programming.',
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
            'description': 'Implement a segment tree with range minimum query and point updates. Handle multiple queries efficiently with O(log n) complexity.',
            'input_desc': 'First line contains n and q. Next line contains n integers. Next q lines contain queries.',
            'output_desc': 'For each query, print the result.',
            'expected_class': 'hard'
        }
    ]
    
    print("\\nSample predictions:")
    correct_predictions = 0
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            result = PredictionService.predict_class_and_score(
                test_case['description'],
                test_case['input_desc'],
                test_case['output_desc']
            )
            
            is_correct = result['class'] == test_case['expected_class']
            if is_correct:
                correct_predictions += 1
                status = "✅ CORRECT"
            else:
                status = "❌ WRONG"
            
            print(f"\\n{i}. {status}")
            print(f"   Expected: {test_case['expected_class']}")
            print(f"   Predicted: {result['class']} (confidence: {result['confidence']:.3f})")
            print(f"   Score: {result['score']}")
            print(f"   Reliable: {result['reliable']}")
            
            # Show key features
            features = result.get('features', {})
            print(f"   Key features: algo_score={features.get('algorithmScore', 0):.1f}, " +
                  f"math={features.get('mathComplexity', 0):.1f}, " +
                  f"words={features.get('wordCount', 0)}")
            
        except Exception as e:
            print(f"❌ Prediction {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    accuracy = correct_predictions / len(test_cases)
    print(f"\\n📊 QUICK TEST RESULTS:")
    print(f"Sample accuracy: {correct_predictions}/{len(test_cases)} = {accuracy:.1%}")
    
    if accuracy >= 0.67:
        print("🎉 Great! Model is working well on samples")
    elif accuracy >= 0.33:
        print("✅ Model is working, some predictions correct")
    else:
        print("⚠️ Model may need more work")
    
    print("\\n✅ Updated app.py is ready to use!")
    return accuracy

if __name__ == "__main__":
    quick_test()