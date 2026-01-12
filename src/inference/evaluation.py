"""Evaluation utilities for model performance assessment.

Provides functions for computing metrics, checking convergence, and generating reports.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


def compute_prediction_variance(
    predictions_list: List[Dict[str, Any]],
    key: str = 'price_h0'
) -> float:
    """Compute variance across multiple prediction runs.

    Used for stability evaluation of model predictions.

    Args:
        predictions_list: List of prediction dictionaries
        key: Key to compute variance for

    Returns:
        Variance across predictions
    """
    values = []
    for pred in predictions_list:
        value = pred.get(key, 0.0)
        if isinstance(value, np.ndarray):
            value = float(value.flatten()[0])
        values.append(value)

    return float(np.var(values))


def check_convergence(
    log_file: Path,
    metric: str = 'val_loss',
    window: int = 5,
    threshold: float = 0.001
) -> bool:
    """Check if training has converged based on training logs.

    Args:
        log_file: Path to training log CSV
        metric: Metric to check for convergence
        window: Window size for computing variance
        threshold: Variance threshold for convergence

    Returns:
        True if converged, False otherwise
    """
    if not Path(log_file).exists():
        return False

    # Load training log
    df = pd.read_csv(log_file)

    if metric not in df.columns:
        return False

    # Get last window values
    values = df[metric].tail(window).values

    if len(values) < window:
        return False

    # Check if variance is below threshold
    variance = np.var(values)
    return variance < threshold


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_directional_accuracy(
    current_prices: np.ndarray,
    predicted_prices: np.ndarray,
    actual_prices: np.ndarray
) -> float:
    """Compute directional accuracy.

    Measures how often the model correctly predicts the direction of price movement.

    Args:
        current_prices: Current prices
        predicted_prices: Predicted future prices
        actual_prices: Actual future prices

    Returns:
        Directional accuracy (0-1)
    """
    # Predicted direction
    pred_direction = predicted_prices > current_prices

    # Actual direction
    actual_direction = actual_prices > current_prices

    # Compute accuracy
    correct = pred_direction == actual_direction
    accuracy = np.mean(correct)

    return float(accuracy)


def generate_report(
    metrics: Dict[str, float],
    report_file: Path
) -> None:
    """Generate evaluation report.

    Args:
        metrics: Dictionary of metric names and values
        report_file: Path to save report
    """
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        # Price prediction metrics
        f.write("Price Prediction Metrics:\n")
        f.write("-" * 60 + "\n")
        for horizon in ['h0', 'h1', 'h2']:
            mae_key = f'mae_{horizon}'
            if mae_key in metrics:
                f.write(f"  MAE {horizon.upper()}: {metrics[mae_key]:.2f}\n")
        f.write("\n")

        # Direction prediction metrics
        f.write("Direction Prediction Metrics:\n")
        f.write("-" * 60 + "\n")
        for horizon in ['h0', 'h1', 'h2']:
            acc_key = f'dir_acc_{horizon}'
            if acc_key in metrics:
                f.write(f"  Direction Accuracy {horizon.upper()}: {metrics[acc_key]:.2%}\n")
        f.write("\n")

        # Additional metrics
        f.write("Additional Metrics:\n")
        f.write("-" * 60 + "\n")
        for key, value in metrics.items():
            if not key.startswith('mae_') and not key.startswith('dir_acc_'):
                f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("=" * 60 + "\n")


def compute_metrics_on_test_set(
    y_true: Dict[str, np.ndarray],
    y_pred: Dict[str, np.ndarray],
    current_prices: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Compute comprehensive metrics on test set.

    Args:
        y_true: Dictionary of true values for each output
        y_pred: Dictionary of predicted values for each output
        current_prices: Current prices for directional accuracy (optional)

    Returns:
        Dictionary of computed metrics
    """
    metrics = {}

    # Compute MAE for each horizon
    for h in [0, 1, 2]:
        price_key = f'price_h{h}'
        if price_key in y_true and price_key in y_pred:
            mae = compute_mae(y_true[price_key], y_pred[price_key])
            metrics[f'mae_h{h}'] = mae

    # Compute directional accuracy if current prices provided
    if current_prices is not None:
        for h in [0, 1, 2]:
            price_key = f'price_h{h}'
            if price_key in y_true and price_key in y_pred:
                dir_acc = compute_directional_accuracy(
                    current_prices,
                    y_pred[price_key],
                    y_true[price_key]
                )
                metrics[f'dir_acc_h{h}'] = dir_acc

    return metrics
