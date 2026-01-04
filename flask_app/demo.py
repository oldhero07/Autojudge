#!/usr/bin/env python3
"""
Demo script for the Flask ML Web Application

This script demonstrates the functionality of the programming problem classifier.
"""

import sys
import json
from app import initialize_app, PredictionService

def main():
    print("🚀 Flask ML Web Application Demo")
    print("=" * 50)
    
    # Initialize the application
    print("Initializing models...")
    initialize_app()
    print("✅ Models loaded and ready!")
    print()
    
    # Test cases
    test_cases = [
        {
            "name": "Simple Array Problem",
            "description": "Find the maximum element in an array",
            "input_desc": "First line contains n, second line contains n integers",
            "output_desc": "Single integer representing the maximum value"
        },
        {
            "name": "String Processing",
            "description": "Count the number of palindromic substrings in a given string",
            "input_desc": "Single line containing a string of lowercase letters",
            "output_desc": "Integer representing the count of palindromic substrings"
        },
        {
            "name": "Complex Graph Algorithm",
            "description": "Find the shortest path in a weighted directed graph with negative edge weights using Bellman-Ford algorithm",
            "input_desc": "First line contains n and m (vertices and edges), followed by m lines with edge information (u, v, weight)",
            "output_desc": "Shortest distances from source to all vertices, or 'NEGATIVE CYCLE' if detected"
        },
        {
            "name": "Basic Math",
            "description": "Calculate the sum of two integers",
            "input_desc": "Two integers a and b on a single line",
            "output_desc": "Single integer representing a + b"
        }
    ]
    
    # Run predictions
    for i, test_case in enumerate(test_cases, 1):
        print(f"🔍 Test Case {i}: {test_case['name']}")
        print(f"Description: {test_case['description'][:60]}...")
        
        try:
            result = PredictionService.predict_class_and_score(
                test_case['description'],
                test_case['input_desc'],
                test_case['output_desc']
            )
            
            # Color coding for difficulty
            class_colors = {
                'easy': '🟢',
                'medium': '🟡', 
                'hard': '🔴'
            }
            
            color = class_colors.get(result['class'].lower(), '⚪')
            print(f"Result: {color} {result['class'].upper()} (Score: {result['score']}/100)")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 50)
    
    print("\n🎉 Demo completed!")
    print("\nTo run the web application:")
    print("1. cd flask_app")
    print("2. python app.py")
    print("3. Open http://localhost:5000 in your browser")

if __name__ == "__main__":
    main()