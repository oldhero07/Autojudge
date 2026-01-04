"""
Comprehensive Integration Tests for AutoJudge System

Tests end-to-end functionality including UI integration with backend API,
evaluation pipeline with real dataset, and documentation generation.
"""

import pytest
import json
import requests
import time
from flask import Flask
from app import app, initialize_app
import threading
import subprocess
import os
import sys
from typing import Dict, Any


class TestIntegration:
    """Integration tests for complete AutoJudge system."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment with running Flask app."""
        # Initialize the app
        with app.app_context():
            initialize_app()
        
        # Start Flask app in a separate thread for testing
        cls.server_thread = threading.Thread(
            target=lambda: app.run(host='localhost', port=5001, debug=False, use_reloader=False)
        )
        cls.server_thread.daemon = True
        cls.server_thread.start()
        
        # Wait for server to start
        time.sleep(3)
        
        cls.base_url = "http://localhost:5001"
    
    def test_health_endpoint_integration(self):
        """Test system health endpoint returns comprehensive status."""
        response = requests.get(f"{self.base_url}/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert 'status' in data
        assert 'health_report' in data
        assert 'timestamp' in data
        
        health_report = data['health_report']
        
        # Verify health report components
        assert 'overall_status' in health_report
        assert 'component_status' in health_report
        assert 'model_status' in health_report
        
        # Verify all models are loaded
        model_status = health_report['model_status']
        assert model_status['classifier_loaded'] is True
        assert model_status['regressor_loaded'] is True
        assert model_status['tfidf_vectorizer_loaded'] is True
        assert model_status['feature_scaler_loaded'] is True
        assert model_status['evaluator_initialized'] is True
        assert model_status['documentation_generator_ready'] is True
    
    def test_legacy_api_integration(self):
        """Test legacy API endpoint with real prediction workflow."""
        # Test data representing different difficulty levels
        test_cases = [
            {
                'description': 'Print "Hello World" to the console.',
                'expected_class': 'easy'
            },
            {
                'description': 'Given an array of integers, find the maximum sum of any contiguous subarray using dynamic programming.',
                'expected_class': 'medium'
            },
            {
                'description': 'Implement a suffix array construction algorithm with O(n log n) complexity for string matching in competitive programming.',
                'expected_class': 'hard'
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            response = requests.post(
                f"{self.base_url}/predict",
                json={'description': test_case['description']},
                headers={'Content-Type': 'application/json'}
            )
            
            assert response.status_code == 200, f"Test case {i+1} failed with status {response.status_code}"
            
            data = response.json()
            
            # Verify response structure
            assert 'class' in data
            assert 'score' in data
            assert 'features' in data
            
            # Verify data types and ranges
            assert data['class'] in ['easy', 'medium', 'hard']
            assert 1.0 <= data['score'] <= 10.0
            assert isinstance(data['features']['textLength'], int)
            assert isinstance(data['features']['mathSymbols'], int)
            assert isinstance(data['features']['keywords'], int)
            assert isinstance(data['features']['tfidfFeatures'], int)
            
            print(f"Legacy API Test {i+1}: {test_case['expected_class']} -> {data['class']} ({data['score']}/10)")
    
    def test_structured_api_integration(self):
        """Test structured API endpoint with AutoJudge three-input format."""
        # Test data with separate fields
        test_cases = [
            {
                'description': 'Find the shortest path between two nodes in a weighted graph.',
                'input_desc': 'First line contains n (number of nodes) and m (number of edges). Next m lines contain u, v, w representing edge from u to v with weight w.',
                'output_desc': 'Output the shortest distance from source to destination, or -1 if no path exists.',
                'expected_class': 'medium'
            },
            {
                'description': 'Implement a data structure that supports range minimum queries.',
                'input_desc': 'First line contains n (array size) and q (number of queries). Second line contains n integers. Next q lines contain l, r for range queries.',
                'output_desc': 'For each query, output the minimum value in range [l, r].',
                'expected_class': 'hard'
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            response = requests.post(
                f"{self.base_url}/predict/structured",
                json={
                    'description': test_case['description'],
                    'input_desc': test_case['input_desc'],
                    'output_desc': test_case['output_desc']
                },
                headers={'Content-Type': 'application/json'}
            )
            
            assert response.status_code == 200, f"Structured test case {i+1} failed with status {response.status_code}"
            
            data = response.json()
            
            # Verify response structure (structured format includes additional fields)
            assert 'class' in data
            assert 'score' in data
            assert 'features' in data
            assert 'format' in data
            assert 'input_fields' in data
            
            # Verify structured format specific fields
            assert data['format'] == 'structured'
            assert 'description_length' in data['input_fields']
            assert 'input_desc_length' in data['input_fields']
            assert 'output_desc_length' in data['input_fields']
            
            # Verify data consistency
            assert data['input_fields']['description_length'] == len(test_case['description'])
            assert data['input_fields']['input_desc_length'] == len(test_case['input_desc'])
            assert data['input_fields']['output_desc_length'] == len(test_case['output_desc'])
            
            print(f"Structured API Test {i+1}: {test_case['expected_class']} -> {data['class']} ({data['score']}/10)")
    
    def test_api_error_handling_integration(self):
        """Test API error handling with various invalid inputs."""
        # Test cases for error scenarios
        error_test_cases = [
            {
                'endpoint': '/predict',
                'payload': {},
                'expected_status': 400,
                'description': 'Empty payload'
            },
            {
                'endpoint': '/predict',
                'payload': {'description': ''},
                'expected_status': 400,
                'description': 'Empty description'
            },
            {
                'endpoint': '/predict/structured',
                'payload': {'description': 'test'},
                'expected_status': 400,
                'description': 'Missing structured fields'
            },
            {
                'endpoint': '/predict/structured',
                'payload': {'description': '', 'input_desc': '', 'output_desc': ''},
                'expected_status': 400,
                'description': 'All empty structured fields'
            }
        ]
        
        for test_case in error_test_cases:
            response = requests.post(
                f"{self.base_url}{test_case['endpoint']}",
                json=test_case['payload'],
                headers={'Content-Type': 'application/json'}
            )
            
            assert response.status_code == test_case['expected_status'], \
                f"Error test '{test_case['description']}' expected {test_case['expected_status']}, got {response.status_code}"
            
            data = response.json()
            assert 'error' in data
            assert 'message' in data
            
            print(f"Error handling test passed: {test_case['description']}")
    
    def test_documentation_generation_integration(self):
        """Test documentation generation with actual evaluation metrics."""
        # Test documentation preview
        response = requests.get(f"{self.base_url}/docs/preview")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'content' in data
        assert 'length' in data
        assert data['length'] > 1000  # Should be substantial documentation
        
        # Verify documentation contains key sections
        content = data['content']
        required_sections = [
            '# AutoJudge',
            '## Overview',
            '## Methodology',
            '## Evaluation Results',
            '## Technical Specifications',
            '## Usage Examples',
            '## Installation and Setup'
        ]
        
        for section in required_sections:
            assert section in content, f"Missing required section: {section}"
        
        # Test documentation generation (save)
        response = requests.post(f"{self.base_url}/generate-docs")
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'message' in data
        assert 'validation' in data
        
        # Calculate completeness score from validation results
        validation_results = data['validation']
        total_checks = len(validation_results)
        passed_checks = sum(1 for v in validation_results.values() if v is True)
        completeness_score = passed_checks / total_checks if total_checks > 0 else 0
        
        assert completeness_score > 0.8  # Should be mostly complete
        
        print(f"Documentation generation test passed: {completeness_score:.1%} complete ({passed_checks}/{total_checks} checks)")
    
    def test_feature_extraction_consistency(self):
        """Test that feature extraction is consistent between UI and API."""
        test_text = "Implement a binary search algorithm with O(log n) complexity. Input: sorted array and target value. Output: index of target or -1."
        
        # Test legacy format
        legacy_response = requests.post(
            f"{self.base_url}/predict",
            json={'description': test_text},
            headers={'Content-Type': 'application/json'}
        )
        
        # Test structured format with same combined text
        structured_response = requests.post(
            f"{self.base_url}/predict/structured",
            json={
                'description': test_text,
                'input_desc': '',
                'output_desc': ''
            },
            headers={'Content-Type': 'application/json'}
        )
        
        assert legacy_response.status_code == 200
        assert structured_response.status_code == 200
        
        legacy_data = legacy_response.json()
        structured_data = structured_response.json()
        
        # Features should be identical for same text
        assert legacy_data['features']['textLength'] == structured_data['features']['textLength']
        assert legacy_data['features']['mathSymbols'] == structured_data['features']['mathSymbols']
        assert legacy_data['features']['keywords'] == structured_data['features']['keywords']
        assert legacy_data['features']['tfidfFeatures'] == structured_data['features']['tfidfFeatures']
        
        # Predictions should be identical
        assert legacy_data['class'] == structured_data['class']
        assert legacy_data['score'] == structured_data['score']
        
        print("Feature extraction consistency test passed")
    
    def test_performance_monitoring_integration(self):
        """Test that performance monitoring is working during predictions."""
        # Make several predictions to trigger monitoring
        test_descriptions = [
            "Simple addition problem",
            "Complex graph algorithm with dynamic programming",
            "Advanced data structure implementation",
            "Basic string manipulation",
            "Optimization problem with constraints"
        ]
        
        for desc in test_descriptions:
            response = requests.post(
                f"{self.base_url}/predict",
                json={'description': desc},
                headers={'Content-Type': 'application/json'}
            )
            assert response.status_code == 200
        
        # Check system health after predictions
        health_response = requests.get(f"{self.base_url}/health")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        health_report = health_data['health_report']
        
        # Should have some performance data
        assert 'recent_errors' in health_report
        assert 'recent_alerts' in health_report
        assert isinstance(health_report['recent_errors'], int)
        assert isinstance(health_report['recent_alerts'], int)
        
        print("Performance monitoring integration test passed")
    
    def test_autojudge_compliance_validation(self):
        """Test that the system complies with AutoJudge research paper specifications."""
        # Test three-input format compliance
        autojudge_test = {
            'description': 'Given a binary tree, determine if it is a valid binary search tree.',
            'input_desc': 'Root of a binary tree where each node has a value.',
            'output_desc': 'Return true if the tree is a valid BST, false otherwise.'
        }
        
        response = requests.post(
            f"{self.base_url}/predict/structured",
            json=autojudge_test,
            headers={'Content-Type': 'application/json'}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify AutoJudge compliance requirements
        compliance_checks = {
            'three_input_support': 'format' in data and data['format'] == 'structured',
            'classification_prediction': 'class' in data and data['class'] in ['easy', 'medium', 'hard'],
            'regression_prediction': 'score' in data and 1.0 <= data['score'] <= 10.0,
            'feature_analysis': 'features' in data and len(data['features']) >= 4,
            'input_field_tracking': 'input_fields' in data and len(data['input_fields']) == 3
        }
        
        for check_name, passed in compliance_checks.items():
            assert passed, f"AutoJudge compliance check failed: {check_name}"
            print(f"✓ AutoJudge compliance: {check_name}")
        
        # Test evaluation metrics are available through health endpoint
        health_response = requests.get(f"{self.base_url}/health")
        health_data = health_response.json()
        
        # Should have training info with evaluation metrics
        assert 'training_info' in health_data['health_report']
        training_info = health_data['health_report']['training_info']
        
        required_training_metrics = ['train_size', 'test_size', 'feature_count']
        for metric in required_training_metrics:
            assert metric in training_info, f"Missing training metric: {metric}"
        
        print("AutoJudge compliance validation passed")


if __name__ == '__main__':
    # Run integration tests
    pytest.main([__file__, '-v', '--tb=short'])