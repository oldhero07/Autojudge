# AutoJudge

**Programming Problem Difficulty Predictor**

A machine learning system that automatically predicts programming problem difficulty using natural language processing and feature engineering techniques.

## Overview

AutoJudge analyzes programming problem descriptions to classify difficulty levels (Easy/Medium/Hard) and provide numerical difficulty scores (1-10 scale). The system features a modern React frontend with a Flask ML backend.

### Key Features

- **Dual Input Modes**: Structured format (description, input, output) and legacy combined format
- **Machine Learning Pipeline**: TF-IDF vectorization with custom domain features
- **Real-time Predictions**: Instant difficulty classification and scoring
- **Comprehensive API**: RESTful endpoints with detailed response format
- **Modern Interface**: Responsive React frontend with TypeScript
- **Production Ready**: Error handling, validation, and monitoring

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

2. **Install dependencies**:
   ```bash
   # Frontend
   npm install
   
   # Backend
   cd flask_app
   pip install -r requirements.txt
   cd ..
   ```

3. **Start the application**:
   
   **Frontend** (Terminal 1):
   ```bash
   npm run dev
   ```
   
   **Backend** (Terminal 2):
   ```bash
   cd flask_app
   python app.py
   ```

4. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

## Usage

### Web Interface

1. **Problem Description**: Enter the main problem statement
2. **Input Description**: Specify the input format (optional)
3. **Output Description**: Define the expected output (optional)
4. Click "Predict Difficulty" to get classification and score

### API Usage

#### Structured Format (Recommended)
```bash
curl -X POST http://localhost:5000/predict/structured \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Find the shortest path in a weighted graph using Dijkstra algorithm",
    "input_desc": "Graph with n vertices and m edges, source vertex",
    "output_desc": "Shortest distances from source to all vertices"
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

### Response Format
```json
{
  "class": "medium",
  "score": 6.2,
  "confidence": 0.847,
  "reliable": true,
  "features": {
    "textLength": 156,
    "wordCount": 24,
    "algorithmScore": 8.0,
    "mathComplexity": 2.0
  }
}
```

## Technical Architecture

### Machine Learning Pipeline

- **Feature Engineering**: TF-IDF vectorization (4000 features) + custom domain features (15 features)
- **Classification Model**: Ensemble of Logistic Regression, Random Forest, and Gradient Boosting
- **Regression Model**: Optimized Random Forest for difficulty scoring
- **Data Processing**: Advanced text preprocessing with algorithm-specific expansions

### Technology Stack

**Frontend**
- React 19 with TypeScript
- Vite build system
- Tailwind CSS styling
- Lucide React icons

**Backend**
- Flask web framework
- scikit-learn ML pipeline
- pandas data processing
- numpy numerical computing

### Dataset
- **Total Samples**: 4,112 programming problems
- **Classes**: Easy (766), Medium (1,405), Hard (1,941)
- **Score Range**: 1.1 - 9.7 (normalized to 1-10 scale)
- **Features**: 3,015 total (3,000 TF-IDF + 15 custom)

## Model Performance

### Test Results (20% holdout set, 823 samples)

**Classification Metrics**
- Overall Accuracy: **59.8%**
- Easy Problems: Precision 0.68, Recall 0.45, F1-Score 0.54
- Medium Problems: Precision 0.52, Recall 0.58, F1-Score 0.55  
- Hard Problems: Precision 0.63, Recall 0.72, F1-Score 0.67

**Regression Metrics**
- Mean Absolute Error (MAE): **1.42**
- Root Mean Square Error (RMSE): **1.89**
- R² Score: **0.31**

**Performance Validation**
- ✅ Classification accuracy exceeds 55% threshold
- ✅ Regression MAE below 2.0 target
- ✅ Model reliability confirmed on test set

## Testing

Run the test suite to verify functionality:

```bash
cd flask_app
python -m pytest test_api_endpoints.py -v
```

**Test Coverage**
- ✅ API endpoint validation (19/19 tests passed)
- ✅ Input format handling (structured & legacy)
- ✅ Error handling and edge cases
- ✅ Response format consistency
- ✅ Model prediction accuracy

## Production Build

Build the frontend for production:

```bash
npm run build
```

The optimized build will be available in the `dist/` directory.

## Project Structure

```
Autojudge/
├── src/                    # React frontend source
├── flask_app/             # Flask backend
│   ├── app.py            # Main application
│   ├── requirements.txt  # Python dependencies
│   └── templates/        # HTML templates
├── public/               # Static assets
├── dist/                # Production build
└── package.json         # Node.js configuration
```

## API Response Format

```json
{
  "class": "easy|medium|hard",
  "score": 5.2,
  "confidence": 0.847,
  "reliable": true,
  "features": {
    "textLength": 150,
    "wordCount": 28,
    "algorithmScore": 4.0,
    "mathComplexity": 1.0
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- scikit-learn for machine learning capabilities
- React and Flask communities for framework support
- Open source contributors

---

**AutoJudge** - Intelligent Programming Problem Difficulty Prediction