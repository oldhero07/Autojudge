# AutoJudge

**Programming Problem Difficulty Predictor**

A machine learning system that automatically predicts programming problem difficulty using natural language processing and advanced feature engineering techniques.

## Overview

AutoJudge analyzes textual problem descriptions to provide both classification (Easy/Medium/Hard) and regression (numerical difficulty score) predictions. The system supports dual input modes and provides comprehensive evaluation metrics.

### Key Features

- **Dual Input Modes**: Structured format (3 separate fields) and legacy combined format
- **Advanced Feature Engineering**: TF-IDF vectorization with domain-specific features
- **Comprehensive Evaluation**: Accuracy, confusion matrix, MAE, and RMSE metrics
- **Real-time Analysis**: Live feature extraction and prediction
- **Production Ready**: Robust error handling and performance monitoring
- **Modern Web Interface**: React frontend with responsive design

## Quick Start

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/oldhero07/Autojudge.git
   cd Autojudge
   ```

2. **Install frontend dependencies**:
   ```bash
   npm install
   ```

3. **Install backend dependencies**:
   ```bash
   cd flask_app
   pip install -r requirements.txt
   cd ..
   ```

4. **Start the development servers**:
   
   **Frontend** (Terminal 1):
   ```bash
   npm run dev
   ```
   
   **Backend** (Terminal 2):
   ```bash
   cd flask_app
   python app.py
   ```

5. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

## Usage

### Web Interface

#### Structured Format (Recommended)
1. **Problem Description**: Enter the main problem statement
2. **Input Description**: Specify the input format
3. **Output Description**: Define the expected output

#### Legacy Combined Format
Enter all information in a single text field.

### API Endpoints

#### Structured Format
```bash
curl -X POST http://localhost:5000/predict/structured \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Find the shortest path in a weighted graph",
    "input_desc": "Graph with n vertices and m edges",
    "output_desc": "Shortest distance from source to target"
  }'
```

#### Legacy Format
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Find the shortest path in a weighted graph using Dijkstra algorithm."
  }'
```

## Technical Architecture

### Machine Learning Pipeline

#### Feature Engineering
- **TF-IDF Vectorization**: 5,000 features with 1-2 gram analysis
- **Custom Features**: Text length, mathematical symbols count, difficulty keywords frequency
- **Feature Scaling**: StandardScaler normalization for numerical features

#### Models
- **Classification**: Logistic Regression for difficulty class prediction
- **Regression**: Random Forest Regressor for numerical difficulty scores

#### Training Data
- **Total Samples**: 4,112
- **Training Split**: 80% (3,289 samples)
- **Test Split**: 20% (823 samples)
- **Feature Dimensions**: 5,003

### Performance Metrics

#### Classification Results
- **Overall Accuracy**: 50.2%
- **Easy Class**: Precision: 0.505, Recall: 0.301, F1: 0.377
- **Medium Class**: Precision: 0.367, Recall: 0.260, F1: 0.304
- **Hard Class**: Precision: 0.552, Recall: 0.756, F1: 0.638

#### Regression Results
- **Mean Absolute Error (MAE)**: 1.695
- **Root Mean Square Error (RMSE)**: 2.042
- **R² Score**: 0.140

### Technology Stack

#### Frontend
- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Icons**: Lucide React

#### Backend
- **Framework**: Flask
- **ML Libraries**: scikit-learn, pandas, numpy
- **Text Processing**: TF-IDF with scipy sparse matrices

## Testing

### API Tests
```bash
cd flask_app
python -m pytest test_api_endpoints.py -v
```

### Error Handling Tests
```bash
cd flask_app
python -m pytest test_error_handling.py -v
```

### All Tests
```bash
cd flask_app
python -m pytest -v
```

### Test Results Summary

#### API Endpoint Tests
- ✅ **19/19 tests passed** - All API endpoints working correctly
- ✅ Legacy and structured format validation
- ✅ Error handling and response format consistency
- ✅ Input validation and edge cases

#### Error Handling Tests
- ✅ Comprehensive error handling coverage
- ✅ Graceful degradation for invalid inputs
- ✅ Proper HTTP status codes and error messages

## Production Build

```bash
npm run build
```

The build artifacts will be stored in the `dist/` directory.

## API Response Format

```json
{
  "class": "easy|medium|hard",
  "score": 5.2,
  "features": {
    "textLength": 150,
    "mathSymbols": 5,
    "keywords": 2,
    "tfidfFeatures": 5000
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- scikit-learn community for machine learning tools
- React and Flask communities for web framework support
- Open source contributors and maintainers

---

**AutoJudge** - Intelligent Programming Problem Difficulty Prediction