"""Custom metrics for neural trading pipeline.

This module provides custom metrics for multi-output evaluation:
- DirectionAccuracy for binary direction prediction
- DirectionF1Score for direction prediction quality
- DirectionMCC (Matthews Correlation Coefficient) - primary metric
- PriceMAE for price prediction error
- PriceMAPE for price prediction percentage error
- MultiHorizonMetric for aggregated multi-horizon evaluation

Full implementations will be added in Phase 6: Losses & Metrics.
These are stub implementations to satisfy imports and registry tests.
"""
import tensorflow as tf
from src.metrics.metric_registry import register_metric


@register_metric('direction_accuracy')
class DirectionAccuracy(tf.keras.metrics.Metric):
    """Direction prediction accuracy (stub - Phase 6)."""

    def __init__(self, name='direction_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Stub implementation."""
        self.count.assign_add(1)

    def result(self):
        """Stub implementation."""
        return tf.constant(0.0)

    def reset_state(self):
        """Reset metric state."""
        self.total.assign(0)
        self.count.assign(0)


@register_metric('direction_f1')
class DirectionF1Score(tf.keras.metrics.Metric):
    """Direction prediction F1 score (stub - Phase 6)."""

    def __init__(self, name='direction_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Stub implementation."""
        pass

    def result(self):
        """Stub implementation."""
        return tf.constant(0.0)

    def reset_state(self):
        """Reset metric state."""
        self.total.assign(0)


@register_metric('direction_mcc')
class DirectionMCC(tf.keras.metrics.Metric):
    """Matthews Correlation Coefficient for direction (stub - Phase 6).

    Primary monitoring metric as per SRS.
    """

    def __init__(self, name='direction_mcc', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Stub implementation."""
        pass

    def result(self):
        """Stub implementation."""
        return tf.constant(0.0)

    def reset_state(self):
        """Reset metric state."""
        self.total.assign(0)


@register_metric('price_mae')
class PriceMAE(tf.keras.metrics.Metric):
    """Price prediction Mean Absolute Error (stub - Phase 6)."""

    def __init__(self, name='price_mae', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Stub implementation."""
        pass

    def result(self):
        """Stub implementation."""
        return tf.constant(0.0)

    def reset_state(self):
        """Reset metric state."""
        self.total.assign(0)


@register_metric('price_mape')
class PriceMAPE(tf.keras.metrics.Metric):
    """Price prediction Mean Absolute Percentage Error (stub - Phase 6)."""

    def __init__(self, name='price_mape', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Stub implementation."""
        pass

    def result(self):
        """Stub implementation."""
        return tf.constant(0.0)

    def reset_state(self):
        """Reset metric state."""
        self.total.assign(0)


@register_metric('multi_horizon')
class MultiHorizonMetric(tf.keras.metrics.Metric):
    """Aggregated metric across multiple horizons (stub - Phase 6)."""

    def __init__(self, base_metric=None, horizons=None, name='multi_horizon', **kwargs):
        super().__init__(name=name, **kwargs)
        self.base_metric = base_metric
        self.horizons = horizons or [0, 1, 2]
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Stub implementation."""
        pass

    def result(self):
        """Stub implementation."""
        return tf.constant(0.0)

    def reset_state(self):
        """Reset metric state."""
        self.total.assign(0)
