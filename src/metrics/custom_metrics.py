"""Custom metrics for neural trading pipeline.

This module provides custom metrics for multi-output evaluation:
- DirectionAccuracy for binary direction prediction
- DirectionF1Score for direction prediction quality
- DirectionMCC (Matthews Correlation Coefficient) - primary metric
- PriceMAE for price prediction error
- PriceMAPE for price prediction percentage error
- MultiHorizonMetric for aggregated multi-horizon evaluation

As per SRS Section 3.5.1 and 7.1.3.
"""
import tensorflow as tf
from typing import Optional, Type, Dict
from src.metrics.metric_registry import register_metric


@register_metric('direction_accuracy')
class DirectionAccuracy(tf.keras.metrics.Metric):
    """Direction prediction accuracy for binary classification.

    Computes accuracy by comparing binary predictions (>0.5) with ground truth.

    Args:
        threshold: Classification threshold (default: 0.5).
        name: Metric name.

    Examples:
        >>> metric = DirectionAccuracy()
        >>> y_true = tf.constant([[1.0], [0.0], [1.0]])
        >>> y_pred = tf.constant([[0.9], [0.1], [0.8]])
        >>> metric.update_state(y_true, y_pred)
        >>> metric.result().numpy()
        1.0
    """

    def __init__(self, threshold: float = 0.5, name: str = 'direction_accuracy', **kwargs):
        """Initialize Direction Accuracy metric.

        Args:
            threshold: Classification threshold.
            name: Metric name.
        """
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state.

        Args:
            y_true: Ground truth labels of shape (batch_size, 1).
            y_pred: Predicted probabilities of shape (batch_size, 1).
            sample_weight: Optional sample weights.
        """
        # Convert predictions to binary (0 or 1)
        y_pred_binary = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true_binary = tf.cast(y_true >= self.threshold, tf.float32)

        # Count correct predictions
        matches = tf.cast(tf.equal(y_pred_binary, y_true_binary), tf.float32)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            matches = matches * sample_weight

        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        """Compute accuracy.

        Returns:
            Accuracy value between 0 and 1.
        """
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_state(self):
        """Reset metric state."""
        self.correct.assign(0)
        self.total.assign(0)


@register_metric('direction_f1')
class DirectionF1Score(tf.keras.metrics.Metric):
    """Direction prediction F1 score.

    Computes F1 score (harmonic mean of precision and recall) for binary direction prediction.

    Args:
        threshold: Classification threshold (default: 0.5).
        name: Metric name.

    Examples:
        >>> metric = DirectionF1Score()
        >>> y_true = tf.constant([[1.0], [0.0], [1.0], [0.0]])
        >>> y_pred = tf.constant([[0.9], [0.1], [0.8], [0.2]])
        >>> metric.update_state(y_true, y_pred)
        >>> metric.result().numpy()
        1.0
    """

    def __init__(self, threshold: float = 0.5, name: str = 'direction_f1', **kwargs):
        """Initialize Direction F1 Score metric.

        Args:
            threshold: Classification threshold.
            name: Metric name.
        """
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state.

        Args:
            y_true: Ground truth labels of shape (batch_size, 1).
            y_pred: Predicted probabilities of shape (batch_size, 1).
            sample_weight: Optional sample weights.
        """
        # Convert to binary
        y_pred_binary = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true_binary = tf.cast(y_true >= self.threshold, tf.float32)

        # Calculate confusion matrix elements
        tp = tf.reduce_sum(y_true_binary * y_pred_binary)
        fp = tf.reduce_sum((1 - y_true_binary) * y_pred_binary)
        fn = tf.reduce_sum(y_true_binary * (1 - y_pred_binary))

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            tp = tf.reduce_sum(y_true_binary * y_pred_binary * sample_weight)
            fp = tf.reduce_sum((1 - y_true_binary) * y_pred_binary * sample_weight)
            fn = tf.reduce_sum(y_true_binary * (1 - y_pred_binary) * sample_weight)

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        """Compute F1 score.

        Returns:
            F1 score value between 0 and 1.
        """
        precision = tf.math.divide_no_nan(
            self.true_positives,
            self.true_positives + self.false_positives
        )
        recall = tf.math.divide_no_nan(
            self.true_positives,
            self.true_positives + self.false_negatives
        )

        f1 = tf.math.divide_no_nan(
            2 * precision * recall,
            precision + recall
        )

        return f1

    def reset_state(self):
        """Reset metric state."""
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


@register_metric('direction_mcc')
class DirectionMCC(tf.keras.metrics.Metric):
    """Matthews Correlation Coefficient for direction prediction.

    Primary monitoring metric as per SRS (val_dir_mcc_h1).
    MCC ranges from -1 (total disagreement) to +1 (perfect prediction).

    Args:
        threshold: Classification threshold (default: 0.5).
        name: Metric name.

    Examples:
        >>> metric = DirectionMCC(name='dir_mcc_h1')
        >>> y_true = tf.constant([[1.0], [0.0], [1.0], [0.0]])
        >>> y_pred = tf.constant([[0.9], [0.1], [0.8], [0.2]])
        >>> metric.update_state(y_true, y_pred)
        >>> metric.result().numpy()
        1.0
    """

    def __init__(self, threshold: float = 0.5, name: str = 'direction_mcc', **kwargs):
        """Initialize Direction MCC metric.

        Args:
            threshold: Classification threshold.
            name: Metric name.
        """
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.true_negatives = self.add_weight(name='true_negatives', initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state.

        Args:
            y_true: Ground truth labels of shape (batch_size, 1).
            y_pred: Predicted probabilities of shape (batch_size, 1).
            sample_weight: Optional sample weights.
        """
        # Convert to binary
        y_pred_binary = tf.cast(y_pred >= self.threshold, tf.float32)
        y_true_binary = tf.cast(y_true >= self.threshold, tf.float32)

        # Calculate confusion matrix elements
        tp = tf.reduce_sum(y_true_binary * y_pred_binary)
        tn = tf.reduce_sum((1 - y_true_binary) * (1 - y_pred_binary))
        fp = tf.reduce_sum((1 - y_true_binary) * y_pred_binary)
        fn = tf.reduce_sum(y_true_binary * (1 - y_pred_binary))

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            tp = tf.reduce_sum(y_true_binary * y_pred_binary * sample_weight)
            tn = tf.reduce_sum((1 - y_true_binary) * (1 - y_pred_binary) * sample_weight)
            fp = tf.reduce_sum((1 - y_true_binary) * y_pred_binary * sample_weight)
            fn = tf.reduce_sum(y_true_binary * (1 - y_pred_binary) * sample_weight)

        self.true_positives.assign_add(tp)
        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        """Compute Matthews Correlation Coefficient.

        Returns:
            MCC value between -1 and 1.
        """
        tp = self.true_positives
        tn = self.true_negatives
        fp = self.false_positives
        fn = self.false_negatives

        # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        numerator = tp * tn - fp * fn
        denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        mcc = tf.math.divide_no_nan(numerator, denominator)

        return mcc

    def reset_state(self):
        """Reset metric state."""
        self.true_positives.assign(0)
        self.true_negatives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


@register_metric('price_mae')
class PriceMAE(tf.keras.metrics.Metric):
    """Price prediction Mean Absolute Error.

    Computes average absolute error for price predictions.

    Args:
        name: Metric name.

    Examples:
        >>> metric = PriceMAE(name='price_mae_h0')
        >>> y_true = tf.constant([[100.0], [200.0], [300.0]])
        >>> y_pred = tf.constant([[110.0], [190.0], [310.0]])
        >>> metric.update_state(y_true, y_pred)
        >>> metric.result().numpy()
        10.0
    """

    def __init__(self, name: str = 'price_mae', **kwargs):
        """Initialize Price MAE metric.

        Args:
            name: Metric name.
        """
        super().__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name='total_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state.

        Args:
            y_true: Ground truth prices of shape (batch_size, 1).
            y_pred: Predicted prices of shape (batch_size, 1).
            sample_weight: Optional sample weights.
        """
        # Compute absolute error
        abs_error = tf.abs(y_true - y_pred)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            abs_error = abs_error * sample_weight

        self.total_error.assign_add(tf.reduce_sum(abs_error))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        """Compute Mean Absolute Error.

        Returns:
            MAE value.
        """
        return tf.math.divide_no_nan(self.total_error, self.count)

    def reset_state(self):
        """Reset metric state."""
        self.total_error.assign(0)
        self.count.assign(0)


@register_metric('price_mape')
class PriceMAPE(tf.keras.metrics.Metric):
    """Price prediction Mean Absolute Percentage Error.

    Computes average percentage error for price predictions.

    Args:
        epsilon: Small value to prevent division by zero (default: 1e-7).
        name: Metric name.

    Examples:
        >>> metric = PriceMAPE(name='price_mape_h0')
        >>> y_true = tf.constant([[100.0], [200.0]])
        >>> y_pred = tf.constant([[110.0], [210.0]])
        >>> metric.update_state(y_true, y_pred)
        >>> metric.result().numpy()
        7.5
    """

    def __init__(self, epsilon: float = 1e-7, name: str = 'price_mape', **kwargs):
        """Initialize Price MAPE metric.

        Args:
            epsilon: Small value to prevent division by zero.
            name: Metric name.
        """
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon
        self.total_percentage_error = self.add_weight(name='total_percentage_error', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state.

        Args:
            y_true: Ground truth prices of shape (batch_size, 1).
            y_pred: Predicted prices of shape (batch_size, 1).
            sample_weight: Optional sample weights.
        """
        # Compute percentage error with epsilon to avoid division by zero
        abs_error = tf.abs(y_true - y_pred)
        percentage_error = 100.0 * tf.math.divide_no_nan(
            abs_error,
            tf.abs(y_true) + self.epsilon
        )

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            percentage_error = percentage_error * sample_weight

        self.total_percentage_error.assign_add(tf.reduce_sum(percentage_error))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        """Compute Mean Absolute Percentage Error.

        Returns:
            MAPE value in percentage.
        """
        return tf.math.divide_no_nan(self.total_percentage_error, self.count)

    def reset_state(self):
        """Reset metric state."""
        self.total_percentage_error.assign(0)
        self.count.assign(0)


@register_metric('multi_horizon')
class MultiHorizonMetric(tf.keras.metrics.Metric):
    """Aggregated metric across multiple horizons.

    Wraps a base metric and tracks it separately for each horizon (h0, h1, h2).
    Can aggregate results across horizons.

    Args:
        base_metric: Metric class to instantiate for each horizon.
        horizons: List of horizon indices (default: [0, 1, 2]).
        name: Metric name.

    Examples:
        >>> metric = MultiHorizonMetric(
        ...     base_metric=DirectionAccuracy,
        ...     horizons=[0, 1, 2]
        ... )
        >>> y_true = {
        ...     'h0': tf.constant([[1.0]]),
        ...     'h1': tf.constant([[0.0]]),
        ...     'h2': tf.constant([[1.0]])
        ... }
        >>> y_pred = {
        ...     'h0': tf.constant([[0.9]]),
        ...     'h1': tf.constant([[0.1]]),
        ...     'h2': tf.constant([[0.8]])
        ... }
        >>> metric.update_state(y_true, y_pred)
        >>> metric.result(horizon=0).numpy()
        1.0
    """

    def __init__(
        self,
        base_metric: Optional[Type[tf.keras.metrics.Metric]] = None,
        horizons: Optional[list] = None,
        name: str = 'multi_horizon',
        **kwargs
    ):
        """Initialize Multi-Horizon Metric.

        Args:
            base_metric: Metric class to instantiate for each horizon.
            horizons: List of horizon indices.
            name: Metric name.
        """
        super().__init__(name=name, **kwargs)
        self.base_metric = base_metric or DirectionAccuracy
        self.horizons = horizons or [0, 1, 2]

        # Create metric instance for each horizon
        self.horizon_metrics = {}
        for h in self.horizons:
            metric_name = f'{name}_h{h}'
            self.horizon_metrics[h] = self.base_metric(name=metric_name)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update metric state for all horizons.

        Args:
            y_true: Dictionary of ground truth values for each horizon.
            y_pred: Dictionary of predicted values for each horizon.
            sample_weight: Optional sample weights.
        """
        # Handle dict inputs
        if isinstance(y_true, dict) and isinstance(y_pred, dict):
            for h in self.horizons:
                key = f'h{h}'
                if key in y_true and key in y_pred:
                    self.horizon_metrics[h].update_state(
                        y_true[key],
                        y_pred[key],
                        sample_weight
                    )

    def result(self, horizon: Optional[int] = None):
        """Get result for a specific horizon.

        Args:
            horizon: Horizon index. If None, returns average across all horizons.

        Returns:
            Metric value for the specified horizon.
        """
        if horizon is not None:
            return self.horizon_metrics[horizon].result()
        else:
            # Return average across all horizons
            return self.result_aggregate()

    def result_aggregate(self):
        """Aggregate results across all horizons.

        Returns:
            Average metric value across all horizons.
        """
        results = [metric.result() for metric in self.horizon_metrics.values()]
        return tf.reduce_mean(results)

    def reset_state(self):
        """Reset metric state for all horizons."""
        for metric in self.horizon_metrics.values():
            metric.reset_state()
