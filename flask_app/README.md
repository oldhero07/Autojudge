# Flask ML Web Application

A web application that uses machine learning to classify programming problems and predict their difficulty scores.

## Project Structure

```
flask_app/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── templates/         # Jinja2 templates
│   └── index.html     # Main web interface
└── static/           # Static assets
    ├── css/
    │   └── style.css  # Application styling
    └── js/
        └── main.js    # Client-side JavaScript
```

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000`

## API Endpoints

- `GET /` - Main web interface
- `POST /predict` - JSON API for problem classification and scoring

## Development Status

This is the initial project structure. ML models and prediction functionality will be implemented in subsequent tasks.