# Model Persistence - Quick Reference

## What Changed

The Flask server now saves trained models to disk and loads them on startup, eliminating the need to retrain models every time.

## How It Works

1. **First Startup**: Trains models (~4-5 minutes) → Saves to `models/trained_models.pkl`
2. **Subsequent Startups**: Loads saved models (~5-10 seconds) → No training needed!

## File Location

- Models are saved to: `flask_app/models/trained_models.pkl`
- The `models/` directory is created automatically

## Usage

### Normal Startup
```bash
cd flask_app
python app.py
```

The server will:
- Check for saved models
- Load them if they exist (fast)
- Train new ones only if not found (slow, first time only)

### Force Retrain
If you want to retrain models from scratch:
```bash
cd flask_app
rm models/trained_models.pkl  # or delete the file manually
python app.py
```

### Check Status
```bash
# Check if models file exists
ls flask_app/models/trained_models.pkl

# Check file size (should be several MB)
```

## Benefits

- ✅ Fast startup (seconds instead of minutes)
- ✅ No unnecessary retraining
- ✅ Automatic fallback if loading fails
- ✅ Models saved automatically after training

## Technical Details

- Uses Python `pickle` for serialization
- Saves: classifier, regressor, TF-IDF vectorizer, feature scaler, feature selector
- Includes timestamp of when models were saved
- Validates all components before using loaded models

