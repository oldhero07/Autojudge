"""
Data models for evaluation metrics and reports.

This module contains the data classes used for storing evaluation results
to avoid circular imports between app.py and documentation_generator.py.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ClassificationMetrics:
    """Data class for classification evaluation metrics."""
    accuracy: float
    confusion_matrix: np.ndarray
    classification_report: Dict[str, Any]


@dataclass
class RegressionMetrics:
    """Data class for regression evaluation metrics."""
    mae: float
    rmse: float
    r2_score: float


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report containing all metrics."""
    classification_metrics: ClassificationMetrics
    regression_metrics: RegressionMetrics
    dataset_info: Dict[str, Any]
    model_info: Dict[str, Any]