# AutoJudge: Programming Problem Difficulty Prediction

A machine learning system that automatically classifies programming problems into difficulty categories (Easy, Medium, Hard) and predicts numerical difficulty scores on a 1-10 scale.

## Overview

AutoJudge analyzes programming problem descriptions and predicts their difficulty using advanced natural language processing and machine learning techniques. The system is designed to assist competitive programming platforms, educational institutions, and coding interview preparation services in automatically categorizing problems.

## Model Performance

### Classification Performance
- **Overall Accuracy**: 55.0%
- **Dataset Size**: 4,112 programming problems
- **Training/Test Split**: 3,289 / 823 samples

#### Confusion Matrix
```
                 Predicted
Actual      Easy  Medium  Hard   Total
Easy         69     49     35     153
Medium       36    107    138     281
Hard         25     87    277     389
Total       130    243    450     823
```

#### Per-Class Metrics
| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Easy   | 0.531     | 0.451  | 0.488    | 153     |
| Medium | 0.440     | 0.381  | 0.408    | 281     |
| Hard   | 0.616     | 0.712  | 0.660    | 389     |

**Weighted Average**: Precision=0.540, Recall=0.550, F1-Score=0.542

### Regression Performance
- **Mean Absolute Error (MAE)**: 1.735 points
- **Root Mean Square Error (RMSE)**: 2.071 points
- **R² Score**: 0.116

#### Score Prediction by Class
| Class  | MAE   | Actual Score | Predicted Score |
|--------|-------|--------------|-----------------|
| Easy   | 2.267 | 1.99 ± 0.43  | 4.25 ± 0.54     |
| Medium | 0.817 | 4.13 ± 0.75  | 4.71 ± 0.52     |
| Hard   | 2.189 | 7.11 ± 1.13  | 4.92 ± 0.48     |

## Dataset Statistics

- **Total Problems**: 4,112
- **Class Distribution**:
  - Hard: 1,941 problems (47.2%)
  - Medium: 1,405 problems (34.2%)
  - Easy: 766 problems (18.6%)
- **Score Range**: 1.1 - 9.7
- **Mean Score**: 5.11 ± 2.18

## Technical Architecture

### Feature Engineering
- **Total Features**: 3,015
- **TF-IDF Features**: 3,000 (selected from 4,000 using chi-square feature selection)
- **Custom Features**: 15 domain-specific features

#### Custom Features
1. **Text Metrics**: Length, word count, vocabulary richness
2. **Algorithm Indicators**: Graph algorithms, dynamic programming, data structures
3. **Complexity Markers**: Sorting/searching, string processing, mathematical content
4. **Problem Characteristics**: Constraints, optimization keywords, test case patterns

### Model Architecture

#### Classification Model
- **Type**: Voting Classifier (Ensemble)
- **Components**: 
  - Logistic Regression (C=2.0, balanced class weights)
  - Random Forest Classifier (400 estimators, max_depth=35)
  - Gradient Boosting Classifier (300 estimators, max_depth=12)
- **Voting Strategy**: Soft voting

#### Regression Model
- **Type**: Random Forest Regressor
- **Parameters**:
  - n_estimators: 350
  - max_depth: 30
  - min_samples_split: 5
  - min_samples_leaf: 2
  - max_features: sqrt

### Text Processing Pipeline
1. **Preprocessing**: Lowercasing, whitespace normalization, abbreviation expansion
2. **TF-IDF Vectorization**: 
   - Max features: 4,000
   - N-gram range: (1, 3)
   - Stop words: English
   - Min/Max document frequency: 2 / 0.85
3. **Feature Selection**: Chi-square test (k=3,000)
4. **Feature Scaling**: StandardScaler for custom features
5. **Feature Combination**: Sparse matrix concatenation

## API Endpoints

### Prediction Endpoint
```http
POST /predict
Content-Type: application/json

{
  "description": "Find the maximum sum of a subarray using dynamic programming"
}
```

**Response:**
```json
{
  "class": "medium",
  "score": 5.2,
  "confidence": 0.678,
  "reliable": true,
  "features": {
    "textLength": 65,
    "wordCount": 11,
    "dynamicProgramming": 1.0,
    "graphAlgorithms": 0.0,
    "dataStructures": 0.0
  }
}
```

### Health Check
```http
GET /health
```

Returns system status and model loading information.

## Installation and Setup

### Prerequisites
- Python 3.8+
- Flask 2.0+
- scikit-learn 1.0+
- pandas, numpy, scipy

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/autojudge.git
cd autojudge

# Install dependencies
pip install -r flask_app/requirements.txt

# Run the application
cd flask_app
python app.py
```

The application will start on `http://localhost:5000`

### Docker Deployment
```bash
# Build the image
docker build -t autojudge .

# Run the container
docker run -p 5000:5000 autojudge
```

## Usage Examples

### Web Interface
Navigate to `http://localhost:5000` to access the web interface for problem submission and prediction.

### API Usage
```python
import requests

# Predict problem difficulty
response = requests.post('http://localhost:5000/predict', json={
    'description': 'Implement a binary search algorithm to find an element in a sorted array'
})

result = response.json()
print(f"Class: {result['class']}")
print(f"Score: {result['score']}/10")
print(f"Confidence: {result['confidence']:.3f}")
```

### Structured Input Format
For problems with separate input/output descriptions:
```python
response = requests.post('http://localhost:5000/predict/structured', json={
    'description': 'Find the shortest path in a weighted graph',
    'input_desc': 'Graph represented as adjacency matrix with weights',
    'output_desc': 'Array of distances from source to all vertices'
})
```

## Model Limitations

1. **Class Imbalance**: The model shows bias toward predicting "Hard" problems due to dataset imbalance (47.2% hard problems)
2. **Score Regression**: Limited R² score (0.116) indicates difficulty in precise numerical score prediction
3. **Domain Specificity**: Trained primarily on competitive programming problems
4. **Language Dependency**: Optimized for English problem descriptions

## Performance Considerations

- **Inference Time**: ~50ms per prediction
- **Memory Usage**: ~200MB for loaded models
- **Throughput**: ~20 requests/second on standard hardware
- **Model Size**: ~15MB serialized

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use AutoJudge in your research or applications, please cite:

```bibtex
@software{autojudge2026,
  title={AutoJudge: Programming Problem Difficulty Prediction},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/autojudge}
}
```

## Acknowledgments

- Dataset sourced from competitive programming platforms
- Built with scikit-learn, Flask, and modern NLP techniques
- Inspired by the need for automated problem categorization in educational technology