"""Model inference module.

Provides inference, signal generation, and evaluation utilities.
"""
from src.inference.predictor import Predictor, InferenceConfig, format_predictions
from src.inference.signals import generate_signals, SignalGenerator
from src.inference.evaluation import (
    compute_prediction_variance,
    check_convergence,
    compute_mae,
    compute_directional_accuracy,
    generate_report,
    compute_metrics_on_test_set
)

__all__ = [
    'Predictor',
    'InferenceConfig',
    'format_predictions',
    'generate_signals',
    'SignalGenerator',
    'compute_prediction_variance',
    'check_convergence',
    'compute_mae',
    'compute_directional_accuracy',
    'generate_report',
    'compute_metrics_on_test_set'
]
