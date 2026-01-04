# AutoJudge Migration Guide

## Overview

This guide helps users migrate from the legacy single-input format to the new AutoJudge research-compliant three-input format, while maintaining full backward compatibility.

## What's New in AutoJudge v1.0

### Enhanced Input Format
- **Three-Input Mode**: Separate fields for problem description, input description, and output description
- **Legacy Support**: Existing single-input format continues to work
- **Input Mode Toggle**: Easy switching between formats in the UI

### Improved Evaluation
- **Comprehensive Metrics**: Accuracy, confusion matrix, MAE, RMSE, and R² score
- **Performance Monitoring**: Real-time threshold validation and alerts
- **Enhanced Documentation**: Auto-generated documentation with evaluation results

### Robust Error Handling
- **Graceful Degradation**: System continues operation even with component failures
- **Performance Alerts**: Automatic monitoring and recommendations
- **Health Monitoring**: System health endpoints for monitoring

## Migration Scenarios

### Scenario 1: Web Interface Users

#### Before (Legacy Format)
```
Single text area with combined content:
"Given an array of integers, find the maximum sum of any contiguous subarray. 
Input: First line contains n, second line contains n integers. 
Output: Single integer representing maximum sum."
```

#### After (AutoJudge Format - Recommended)
```
Problem Description:
"Given an array of integers, find the maximum sum of any contiguous subarray."

Input Description:
"First line contains n (1 ≤ n ≤ 10^5). Second line contains n integers (-10^9 ≤ ai ≤ 10^9)."

Output Description:
"Output a single integer representing the maximum subarray sum."
```

#### Migration Steps
1. **Access the Web Interface**: Navigate to the AutoJudge web application
2. **Choose Input Mode**: Use the toggle button to select "AutoJudge Research Format"
3. **Separate Your Content**: Split your existing combined text into three logical sections:
   - **Problem Description**: The main problem statement and requirements
   - **Input Description**: Format and constraints of the input data
   - **Output Description**: Expected output format and requirements
4. **Test Your Input**: Submit and verify the prediction results

### Scenario 2: API Integration Users

#### Before (Legacy API)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Combined problem, input, and output description..."
  }'
```

#### After (Structured API - Recommended)
```bash
curl -X POST http://localhost:5000/predict/structured \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Problem statement and requirements",
    "input_desc": "Input format and constraints", 
    "output_desc": "Expected output format"
  }'
```

#### Migration Steps
1. **Update Endpoint**: Change from `/predict` to `/predict/structured`
2. **Restructure Payload**: Split your description into three fields
3. **Handle Enhanced Response**: The structured endpoint returns additional metadata
4. **Update Error Handling**: Handle new validation requirements

### Scenario 3: Gradual Migration

If you need to migrate gradually, you can use both formats simultaneously:

#### Hybrid Approach
```python
import requests

# Legacy format for existing systems
legacy_response = requests.post('http://localhost:5000/predict', json={
    'description': combined_text
})

# New structured format for new features
structured_response = requests.post('http://localhost:5000/predict/structured', json={
    'description': problem_desc,
    'input_desc': input_format,
    'output_desc': output_format
})

# Both return compatible core prediction data
assert legacy_response.json()['class'] == structured_response.json()['class']
```

## API Response Changes

### Legacy Response Format
```json
{
  "class": "medium",
  "score": 5.2,
  "features": {
    "textLength": 150,
    "mathSymbols": 5,
    "keywords": 2,
    "tfidfFeatures": 5000
  }
}
```

### Enhanced Structured Response Format
```json
{
  "class": "medium",
  "score": 5.2,
  "features": {
    "textLength": 150,
    "mathSymbols": 5,
    "keywords": 2,
    "tfidfFeatures": 5000
  },
  "format": "structured",
  "input_fields": {
    "description_length": 67,
    "input_desc_length": 50,
    "output_desc_length": 33
  }
}
```

### Key Differences
- **Additional Metadata**: Structured format includes `format` and `input_fields`
- **Field Tracking**: Individual field lengths are tracked
- **Enhanced Validation**: More comprehensive input validation

## Best Practices for Migration

### 1. Text Separation Guidelines

**Problem Description Should Include:**
- Main problem statement
- Task requirements
- Background context
- Constraints and limitations

**Input Description Should Include:**
- Data format specification
- Input size constraints
- Data type information
- Number of test cases

**Output Description Should Include:**
- Expected output format
- Precision requirements
- Special formatting rules
- Multiple output handling

### 2. Validation and Testing

**Before Migration:**
```python
# Test with your existing data
test_cases = [
    "Your existing combined descriptions...",
    # Add more test cases
]

for case in test_cases:
    response = requests.post('/predict', json={'description': case})
    print(f"Legacy: {response.json()}")
```

**After Migration:**
```python
# Test with separated data
structured_cases = [
    {
        'description': 'Problem statement...',
        'input_desc': 'Input format...',
        'output_desc': 'Output format...'
    }
    # Add more structured cases
]

for case in structured_cases:
    response = requests.post('/predict/structured', json=case)
    print(f"Structured: {response.json()}")
```

### 3. Error Handling Updates

**Enhanced Error Messages:**
```python
try:
    response = requests.post('/predict/structured', json=data)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.HTTPError as e:
    error_data = e.response.json()
    print(f"Error: {error_data['message']}")
    # Handle specific validation errors
    if 'Missing required fields' in error_data['message']:
        # Handle missing fields
        pass
    elif 'Empty content' in error_data['message']:
        # Handle empty content
        pass
```

## Performance Improvements

### New Evaluation Metrics
- **Classification Accuracy**: 50.2% (3-class prediction)
- **Regression MAE**: 1.695 (average error in difficulty score)
- **Regression RMSE**: 2.042 (root mean square error)
- **R² Score**: 0.140 (variance explained)

### Enhanced Features
- **Real-time Monitoring**: Performance threshold validation
- **Health Endpoints**: `/health` for system status monitoring
- **Documentation Generation**: `/generate-docs` for automated documentation

## Troubleshooting

### Common Migration Issues

#### Issue 1: "Missing required fields" Error
**Cause**: Structured endpoint requires all three fields
**Solution**: Ensure all fields (description, input_desc, output_desc) are present in your JSON payload

#### Issue 2: "Empty content" Error
**Cause**: All three fields are empty or contain only whitespace
**Solution**: Ensure at least one field contains meaningful text content

#### Issue 3: Different Predictions Between Formats
**Cause**: Text separation might change feature extraction
**Solution**: Verify that combined text from three fields matches your original single text

#### Issue 4: Performance Degradation
**Cause**: New evaluation pipeline adds processing overhead
**Solution**: Use performance monitoring endpoints to identify bottlenecks

### Getting Help

1. **Check System Health**: `GET /health` for system status
2. **Review Logs**: Check application logs for detailed error information
3. **Test Endpoints**: Use `/docs/preview` to verify documentation generation
4. **Validate Input**: Use the web interface to test your text separation

## Rollback Plan

If you need to rollback to legacy format:

1. **Web Interface**: Use the input mode toggle to switch to "Legacy Combined Format"
2. **API Integration**: Continue using `/predict` endpoint with single description field
3. **No Data Loss**: All existing functionality remains available

## Timeline Recommendations

### Phase 1 (Immediate): Testing
- Test new three-input format with sample data
- Verify API responses match expectations
- Check system health and performance

### Phase 2 (1-2 weeks): Gradual Migration
- Migrate non-critical systems to structured format
- Monitor performance and error rates
- Update documentation and training materials

### Phase 3 (1 month): Full Migration
- Migrate all systems to structured format
- Deprecate legacy format usage (while maintaining support)
- Implement monitoring and alerting for new format

## Support and Resources

- **Documentation**: Complete system documentation in README.md
- **API Reference**: Detailed endpoint specifications in Technical Specifications section
- **Health Monitoring**: Use `/health` endpoint for system status
- **Integration Tests**: Reference `test_integration.py` for usage examples

---

*AutoJudge Migration Guide - Version 1.0.0*
*For technical support, refer to the main documentation or system health endpoints*