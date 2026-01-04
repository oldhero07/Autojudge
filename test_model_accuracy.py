#!/usr/bin/env python3
"""
Comprehensive Model Accuracy and Confusion Matrix Testing

This script tests the AutoJudge model's accuracy and generates detailed
evaluation metrics including confusion matrix, classification report,
and regression metrics.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Add flask_app to path
sys.path.append('flask_app')

def test_model_accuracy():
    """Test the model accuracy and generate comprehensive evaluation metrics."""
    
    print("="*70)
    print("AUTOJUDGE MODEL ACCURACY AND CONFUSION MATRIX TESTING")
    print("="*70)
    
    try:
        # Import the Flask app components
        from app import (
            load_and_preprocess_data, 
            ModelEvaluator, 
            extract_custom_features,
            tfidf_vectorizer,
            feature_scaler,
            classifier_pipeline,
            regressor_pipeline
        )
        import scipy.sparse
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestRegressor
        
        print("✓ Successfully imported model components")
        
        # Load and preprocess the data
        print("\n1. Loading and preprocessing data...")
        df = load_and_preprocess_data()
        print(f"   Dataset shape: {df.shape}")
        print(f"   Classes: {df['problem_class'].value_counts().to_dict()}")
        
        # Prepare features and targets
        X_text = df['combined_text']
        X_custom = df[['text_len', 'math_count', 'keyword_count']].values
        y_class = df['problem_class']
        y_score = df['problem_score_scaled']
        
        print(f"   Score range: {y_score.min():.1f} - {y_score.max():.1f}")
        
        # Create feature transformers
        print("\n2. Creating feature transformers...")
        tfidf_vectorizer_local = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        X_tfidf = tfidf_vectorizer_local.fit_transform(X_text)
        
        feature_scaler_local = StandardScaler()
        X_custom_scaled = feature_scaler_local.fit_transform(X_custom)
        
        # Combine features
        X_combined = scipy.sparse.hstack([X_tfidf, scipy.sparse.csr_matrix(X_custom_scaled)])
        print(f"   Combined feature matrix shape: {X_combined.shape}")
        
        # Initialize ModelEvaluator and perform train/test split
        print("\n3. Performing train/test split...")
        evaluator = ModelEvaluator(test_size=0.2, random_state=42)
        X_train, X_test, y_train_class, y_test_class, y_train_score, y_test_score = evaluator.perform_train_test_split(
            X_combined, y_class, y_score
        )
        
        print(f"   Train samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        
        # Train models
        print("\n4. Training models...")
        
        # Classification model
        classifier = LogisticRegression(random_state=42, max_iter=1000)
        classifier.fit(X_train, y_train_class)
        print("   ✓ Classification model trained")
        
        # Regression model
        regressor = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        regressor.fit(X_train, y_train_score)
        print("   ✓ Regression model trained")
        
        # Make predictions
        print("\n5. Making predictions on test set...")
        y_pred_class = classifier.predict(X_test)
        y_pred_score = regressor.predict(X_test)
        
        # Clamp regression predictions to valid range
        y_pred_score = np.clip(y_pred_score, 1.0, 10.0)
        
        # Calculate classification metrics
        print("\n6. Calculating classification metrics...")
        accuracy = accuracy_score(y_test_class, y_pred_class)
        conf_matrix = confusion_matrix(y_test_class, y_pred_class)
        class_report = classification_report(y_test_class, y_pred_class, output_dict=True)
        
        print(f"   Classification Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Calculate regression metrics
        print("\n7. Calculating regression metrics...")
        mae = mean_absolute_error(y_test_score, y_pred_score)
        rmse = np.sqrt(mean_squared_error(y_test_score, y_pred_score))
        r2 = r2_score(y_test_score, y_pred_score)
        
        print(f"   Mean Absolute Error (MAE): {mae:.3f}")
        print(f"   Root Mean Square Error (RMSE): {rmse:.3f}")
        print(f"   R² Score: {r2:.3f}")
        
        # Display detailed results
        print("\n" + "="*70)
        print("DETAILED EVALUATION RESULTS")
        print("="*70)
        
        print(f"\nCLASSIFICATION METRICS:")
        print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"  Number of test samples: {len(y_test_class)}")
        
        print(f"\nCONFUSION MATRIX:")
        classes = sorted(y_test_class.unique())
        print(f"  Classes: {classes}")
        print("  Matrix (rows=actual, cols=predicted):")
        
        # Print confusion matrix with labels
        conf_df = pd.DataFrame(conf_matrix, index=classes, columns=classes)
        print(conf_df.to_string())
        
        print(f"\nCLASSIFICATION REPORT:")
        for class_name in classes:
            if class_name in class_report:
                metrics = class_report[class_name]
                print(f"  {class_name}:")
                print(f"    Precision: {metrics['precision']:.3f}")
                print(f"    Recall: {metrics['recall']:.3f}")
                print(f"    F1-Score: {metrics['f1-score']:.3f}")
                print(f"    Support: {int(metrics['support'])}")
        
        # Macro and weighted averages
        if 'macro avg' in class_report:
            macro_avg = class_report['macro avg']
            print(f"  Macro Average:")
            print(f"    Precision: {macro_avg['precision']:.3f}")
            print(f"    Recall: {macro_avg['recall']:.3f}")
            print(f"    F1-Score: {macro_avg['f1-score']:.3f}")
        
        if 'weighted avg' in class_report:
            weighted_avg = class_report['weighted avg']
            print(f"  Weighted Average:")
            print(f"    Precision: {weighted_avg['precision']:.3f}")
            print(f"    Recall: {weighted_avg['recall']:.3f}")
            print(f"    F1-Score: {weighted_avg['f1-score']:.3f}")
        
        print(f"\nREGRESSION METRICS:")
        print(f"  Mean Absolute Error (MAE): {mae:.3f}")
        print(f"  Root Mean Square Error (RMSE): {rmse:.3f}")
        print(f"  R² Score: {r2:.3f}")
        print(f"  Score range: {y_test_score.min():.1f} - {y_test_score.max():.1f}")
        print(f"  Predicted range: {y_pred_score.min():.1f} - {y_pred_score.max():.1f}")
        
        # Performance analysis
        print(f"\nPERFORMANCE ANALYSIS:")
        
        # Classification performance
        if accuracy >= 0.8:
            class_performance = "Excellent"
        elif accuracy >= 0.7:
            class_performance = "Good"
        elif accuracy >= 0.6:
            class_performance = "Acceptable"
        else:
            class_performance = "Needs Improvement"
        
        print(f"  Classification Performance: {class_performance}")
        
        # Regression performance
        if mae <= 1.0 and rmse <= 1.5:
            reg_performance = "Excellent"
        elif mae <= 1.5 and rmse <= 2.0:
            reg_performance = "Good"
        elif mae <= 2.0 and rmse <= 2.5:
            reg_performance = "Acceptable"
        else:
            reg_performance = "Needs Improvement"
        
        print(f"  Regression Performance: {reg_performance}")
        
        # Sample predictions
        print(f"\nSAMPLE PREDICTIONS (first 10 test cases):")
        print("  Actual Class | Predicted Class | Actual Score | Predicted Score")
        print("  " + "-" * 65)
        
        for i in range(min(10, len(y_test_class))):
            actual_class = y_test_class.iloc[i]
            pred_class = y_pred_class[i]
            actual_score = y_test_score.iloc[i]
            pred_score = y_pred_score[i]
            
            match_symbol = "✓" if actual_class == pred_class else "✗"
            print(f"  {actual_class:>11} | {pred_class:>14} | {actual_score:>11.1f} | {pred_score:>14.1f} {match_symbol}")
        
        # Class-wise accuracy
        print(f"\nCLASS-WISE ACCURACY:")
        for class_name in classes:
            class_mask = y_test_class == class_name
            if class_mask.sum() > 0:
                class_accuracy = accuracy_score(y_test_class[class_mask], y_pred_class[class_mask])
                class_count = class_mask.sum()
                print(f"  {class_name}: {class_accuracy:.3f} ({class_accuracy*100:.1f}%) - {class_count} samples")
        
        # Feature importance (for Random Forest regressor)
        print(f"\nFEATURE IMPORTANCE (Top 10 regression features):")
        if hasattr(regressor, 'feature_importances_'):
            feature_names = [f'tfidf_{i}' for i in range(X_tfidf.shape[1])] + ['text_length', 'math_symbols', 'keywords']
            feature_importance = regressor.feature_importances_
            
            # Get top 10 features
            top_indices = np.argsort(feature_importance)[-10:][::-1]
            for i, idx in enumerate(top_indices):
                importance = feature_importance[idx]
                feature_name = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
                print(f"  {i+1:2d}. {feature_name}: {importance:.4f}")
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"✓ Model tested on {len(y_test_class)} samples")
        print(f"✓ Classification accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"✓ Regression MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
        print(f"✓ Overall performance: Classification={class_performance}, Regression={reg_performance}")
        
        # Performance thresholds check
        thresholds_met = []
        if accuracy >= 0.6:
            thresholds_met.append("Classification accuracy ≥ 60%")
        if mae <= 2.0:
            thresholds_met.append("MAE ≤ 2.0")
        if rmse <= 2.5:
            thresholds_met.append("RMSE ≤ 2.5")
        
        if thresholds_met:
            print(f"✓ Performance thresholds met: {', '.join(thresholds_met)}")
        else:
            print("⚠ Some performance thresholds not met - model may need tuning")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'classes': classes,
            'test_samples': len(y_test_class)
        }
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the Flask app is properly set up and dependencies are installed.")
        return None
    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_confusion_matrix_visualization(results):
    """Create a visual confusion matrix plot."""
    if not results or not VISUALIZATION_AVAILABLE:
        if not VISUALIZATION_AVAILABLE:
            print("\n⚠ matplotlib/seaborn not available - skipping visualization")
        return
    
    try:
        plt.figure(figsize=(8, 6))
        
        # Create confusion matrix heatmap
        conf_matrix = results['confusion_matrix']
        classes = results['classes']
        
        sns.heatmap(conf_matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=classes,
                   yticklabels=classes,
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix\nAccuracy: {results["accuracy"]:.3f} ({results["accuracy"]*100:.1f}%)')
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Confusion matrix visualization saved as 'confusion_matrix.png'")
        plt.close()
        
    except Exception as e:
        print(f"\n⚠ Error creating visualization: {e}")

if __name__ == "__main__":
    print("Starting comprehensive model accuracy testing...")
    
    # Run the accuracy test
    results = test_model_accuracy()
    
    # Create visualization if possible
    if results:
        create_confusion_matrix_visualization(results)
    
    print("\n" + "="*70)
    print("Model accuracy testing completed!")
    print("="*70)