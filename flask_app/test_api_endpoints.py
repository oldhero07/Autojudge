"""
Unit tests for API endpoints and backward compatibility.

These tests validate both legacy and structured API endpoints
to ensure consistent behavior and proper error handling.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from app import app, PredictionService


class TestAPIEndpoints:
    """Unit tests for API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.app = app.test_client()
        self.app.testing = True
        
        # Mock prediction result
        self.mock_prediction = {
            'class': 'medium',
            'score': 5.2,
            'features': {
                'textLength': 150,
                'mathSymbols': 5,
                'keywords': 2,
                'tfidfFeatures': 5000
            }
        }
    
    def test_legacy_predict_endpoint_success(self):
        """Test legacy /predict endpoint with valid input."""
        with patch.object(PredictionService, 'predict_class_and_score', return_value=self.mock_prediction):
            response = self.app.post('/predict', 
                json={
                    'description': 'This is a test problem about sorting algorithms.'
                },
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['class'] == 'medium'
            assert data['score'] == 5.2
            assert 'features' in data
    
    def test_legacy_predict_endpoint_with_optional_fields(self):
        """Test legacy /predict endpoint with optional input_desc and output_desc."""
        with patch.object(PredictionService, 'predict_class_and_score', return_value=self.mock_prediction):
            response = self.app.post('/predict', 
                json={
                    'description': 'Test problem',
                    'input_desc': 'Input format',
                    'output_desc': 'Output format'
                },
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['class'] == 'medium'
    
    def test_legacy_predict_endpoint_missing_description(self):
        """Test legacy /predict endpoint with missing description."""
        response = self.app.post('/predict', 
            json={
                'input_desc': 'Input only'
            },
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'description' in data['message'].lower()
    
    def test_legacy_predict_endpoint_empty_description(self):
        """Test legacy /predict endpoint with empty description."""
        response = self.app.post('/predict', 
            json={
                'description': '   '  # Only whitespace
            },
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'empty' in data['message'].lower()
    
    def test_structured_predict_endpoint_success(self):
        """Test structured /predict/structured endpoint with valid input."""
        enhanced_prediction = self.mock_prediction.copy()
        enhanced_prediction['format'] = 'structured'
        enhanced_prediction['input_fields'] = {
            'description_length': 50,
            'input_desc_length': 20,
            'output_desc_length': 15
        }
        
        with patch.object(PredictionService, 'predict_class_and_score', return_value=self.mock_prediction):
            response = self.app.post('/predict/structured', 
                json={
                    'description': 'This is a test problem about graph algorithms.',
                    'input_desc': 'Input: graph edges',
                    'output_desc': 'Output: result'
                },
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['class'] == 'medium'
            assert data['score'] == 5.2
            assert data['format'] == 'structured'
            assert 'input_fields' in data
    
    def test_structured_predict_endpoint_missing_fields(self):
        """Test structured /predict/structured endpoint with missing fields."""
        response = self.app.post('/predict/structured', 
            json={
                'description': 'Test problem',
                'input_desc': 'Input format'
                # Missing output_desc
            },
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'missing' in data['message'].lower()
        assert 'output_desc' in data['message']
    
    def test_structured_predict_endpoint_all_empty(self):
        """Test structured /predict/structured endpoint with all empty fields."""
        response = self.app.post('/predict/structured', 
            json={
                'description': '',
                'input_desc': '',
                'output_desc': ''
            },
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'text' in data['message'].lower()
    
    def test_structured_predict_endpoint_partial_content(self):
        """Test structured /predict/structured endpoint with partial content."""
        with patch.object(PredictionService, 'predict_class_and_score', return_value=self.mock_prediction):
            response = self.app.post('/predict/structured', 
                json={
                    'description': 'Test problem',
                    'input_desc': '',  # Empty but present
                    'output_desc': 'Output format'
                },
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['class'] == 'medium'
    
    def test_no_json_data(self):
        """Test endpoints with no JSON data."""
        # Test legacy endpoint - Flask returns 500 for JSON decode errors
        response = self.app.post('/predict', content_type='application/json')
        assert response.status_code == 500  # Flask JSON decode error
        
        # Test structured endpoint
        response = self.app.post('/predict/structured', content_type='application/json')
        assert response.status_code == 500  # Flask JSON decode error
    
    def test_prediction_service_error_handling(self):
        """Test error handling when prediction service fails."""
        with patch.object(PredictionService, 'predict_class_and_score', side_effect=ValueError("Test error")):
            response = self.app.post('/predict', 
                json={
                    'description': 'Test problem'
                },
                content_type='application/json'
            )
            
            assert response.status_code == 400
            data = json.loads(response.data)
            assert 'error' in data
            assert 'Test error' in data['message']
    
    def test_internal_server_error_handling(self):
        """Test internal server error handling."""
        with patch.object(PredictionService, 'predict_class_and_score', side_effect=Exception("Internal error")):
            response = self.app.post('/predict', 
                json={
                    'description': 'Test problem'
                },
                content_type='application/json'
            )
            
            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data
            assert 'error occurred' in data['message'].lower()


class TestPredictionServiceValidation:
    """Unit tests for PredictionService validation methods."""
    
    def test_validate_legacy_format_valid(self):
        """Test validation of valid legacy format."""
        result = PredictionService.validate_input_format(
            'Test problem', None, None, 'legacy'
        )
        
        assert result['valid'] is True
        assert 'valid' in result['message'].lower()
    
    def test_validate_legacy_format_invalid(self):
        """Test validation of invalid legacy format."""
        result = PredictionService.validate_input_format(
            '', None, None, 'legacy'
        )
        
        assert result['valid'] is False
        assert 'description' in result['message'].lower()
    
    def test_validate_structured_format_valid(self):
        """Test validation of valid structured format."""
        result = PredictionService.validate_input_format(
            'Problem', 'Input', 'Output', 'structured'
        )
        
        assert result['valid'] is True
        assert 'valid' in result['message'].lower()
    
    def test_validate_structured_format_missing_fields(self):
        """Test validation of structured format with missing fields."""
        result = PredictionService.validate_input_format(
            'Problem', None, 'Output', 'structured'
        )
        
        assert result['valid'] is False
        assert 'three fields' in result['message'].lower()
    
    def test_validate_structured_format_empty_content(self):
        """Test validation of structured format with empty content."""
        result = PredictionService.validate_input_format(
            '', '', '', 'structured'
        )
        
        assert result['valid'] is False
        assert 'text content' in result['message'].lower()
    
    def test_validate_structured_format_partial_content(self):
        """Test validation of structured format with partial content."""
        result = PredictionService.validate_input_format(
            'Problem', '', '', 'structured'
        )
        
        assert result['valid'] is True
        assert 'valid' in result['message'].lower()


class TestResponseFormatConsistency:
    """Unit tests for response format consistency."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.app = app.test_client()
        self.app.testing = True
        
        self.mock_prediction = {
            'class': 'hard',
            'score': 7.8,
            'features': {
                'textLength': 200,
                'mathSymbols': 10,
                'keywords': 5,
                'tfidfFeatures': 5000
            }
        }
    
    def test_response_format_consistency(self):
        """Test that both endpoints return consistent base format."""
        with patch.object(PredictionService, 'predict_class_and_score', return_value=self.mock_prediction):
            # Test legacy endpoint
            legacy_response = self.app.post('/predict', 
                json={'description': 'Test problem'},
                content_type='application/json'
            )
            
            # Test structured endpoint
            structured_response = self.app.post('/predict/structured', 
                json={
                    'description': 'Test problem',
                    'input_desc': 'Input',
                    'output_desc': 'Output'
                },
                content_type='application/json'
            )
            
            assert legacy_response.status_code == 200
            assert structured_response.status_code == 200
            
            legacy_data = json.loads(legacy_response.data)
            structured_data = json.loads(structured_response.data)
            
            # Both should have core prediction fields
            core_fields = ['class', 'score', 'features']
            for field in core_fields:
                assert field in legacy_data
                assert field in structured_data
                assert legacy_data[field] == structured_data[field]
            
            # Structured should have additional fields
            assert 'format' in structured_data
            assert 'input_fields' in structured_data
            assert structured_data['format'] == 'structured'
    
    def test_error_response_format_consistency(self):
        """Test that error responses have consistent format."""
        # Test legacy endpoint error
        legacy_response = self.app.post('/predict', 
            json={},  # Missing description
            content_type='application/json'
        )
        
        # Test structured endpoint error
        structured_response = self.app.post('/predict/structured', 
            json={'description': 'Test'},  # Missing fields
            content_type='application/json'
        )
        
        assert legacy_response.status_code == 400
        assert structured_response.status_code == 400
        
        legacy_data = json.loads(legacy_response.data)
        structured_data = json.loads(structured_response.data)
        
        # Both should have consistent error format
        error_fields = ['error', 'message', 'status_code']
        for field in error_fields:
            assert field in legacy_data
            assert field in structured_data
            assert legacy_data['status_code'] == structured_data['status_code']


if __name__ == '__main__':
    # Run unit tests
    pytest.main([__file__, '-v'])