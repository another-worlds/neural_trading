"""Phase 2.6: HIGH priority tests for registry edge cases.

Target: 3 uncovered statements in loss_registry.py and metric_registry.py
Lines: loss_registry 62, 71; metric_registry 62, 71

Coverage goal: 70% → 100%
"""
import pytest
import tensorflow as tf
from src.losses.loss_registry import LossRegistry, LOSS_REGISTRY, register_loss
from src.metrics.metric_registry import MetricRegistry, METRIC_REGISTRY, register_metric


class TestLossRegistryEdgeCases:
    """Test LossRegistry edge cases (lines 62, 71)."""

    def test_get_unknown_loss_raises_keyerror(self):
        """Should raise KeyError for unknown loss (line 62)."""
        registry = LossRegistry()

        # Try to get a loss that doesn't exist
        with pytest.raises(KeyError, match="Loss 'unknown_loss' not found in registry"):
            registry.get('unknown_loss')

    def test_list_losses(self):
        """Should list all registered losses (line 71)."""
        registry = LossRegistry()

        # Register some losses
        @registry.register('test_loss_1')
        class TestLoss1(tf.keras.losses.Loss):
            def call(self, y_true, y_pred):
                return tf.reduce_mean(tf.square(y_true - y_pred))

        @registry.register('test_loss_2')
        class TestLoss2(tf.keras.losses.Loss):
            def call(self, y_true, y_pred):
                return tf.reduce_mean(tf.abs(y_true - y_pred))

        # Should list all registered losses (line 71)
        loss_names = registry.list_losses()

        assert 'test_loss_1' in loss_names
        assert 'test_loss_2' in loss_names
        assert len(loss_names) == 2

    def test_global_loss_registry_has_losses(self):
        """Should have losses registered from custom_losses module."""
        # LOSS_REGISTRY should have losses from custom_losses.py
        loss_names = LOSS_REGISTRY.list_losses()

        # Check that common losses are registered
        assert 'focal' in loss_names or 'huber' in loss_names or len(loss_names) > 0

    def test_register_loss_decorator(self):
        """Should register loss using convenience function."""
        @register_loss('test_custom_loss')
        class TestCustomLoss(tf.keras.losses.Loss):
            def call(self, y_true, y_pred):
                return tf.reduce_mean(tf.square(y_true - y_pred))

        # Should be registered
        loss_class = LOSS_REGISTRY.get('test_custom_loss')
        assert loss_class == TestCustomLoss

        # Clean up
        del LOSS_REGISTRY.losses['test_custom_loss']


class TestMetricRegistryEdgeCases:
    """Test MetricRegistry edge cases (lines 62, 71)."""

    def test_get_unknown_metric_raises_keyerror(self):
        """Should raise KeyError for unknown metric (line 62)."""
        registry = MetricRegistry()

        # Try to get a metric that doesn't exist
        with pytest.raises(KeyError, match="Metric 'unknown_metric' not found in registry"):
            registry.get('unknown_metric')

    def test_list_metrics(self):
        """Should list all registered metrics (line 71)."""
        registry = MetricRegistry()

        # Register some metrics
        @registry.register('test_metric_1')
        class TestMetric1(tf.keras.metrics.Metric):
            def update_state(self, y_true, y_pred, sample_weight=None):
                pass

            def result(self):
                return tf.constant(0.0)

        @registry.register('test_metric_2')
        class TestMetric2(tf.keras.metrics.Metric):
            def update_state(self, y_true, y_pred, sample_weight=None):
                pass

            def result(self):
                return tf.constant(0.0)

        # Should list all registered metrics (line 71)
        metric_names = registry.list_metrics()

        assert 'test_metric_1' in metric_names
        assert 'test_metric_2' in metric_names
        assert len(metric_names) == 2

    def test_global_metric_registry_has_metrics(self):
        """Should have metrics registered from custom_metrics module."""
        # METRIC_REGISTRY should have metrics from custom_metrics.py
        metric_names = METRIC_REGISTRY.list_metrics()

        # Check that common metrics are registered
        assert 'direction_accuracy' in metric_names or 'direction_mcc' in metric_names or len(metric_names) > 0

    def test_register_metric_decorator(self):
        """Should register metric using convenience function."""
        @register_metric('test_custom_metric')
        class TestCustomMetric(tf.keras.metrics.Metric):
            def update_state(self, y_true, y_pred, sample_weight=None):
                pass

            def result(self):
                return tf.constant(1.0)

            def reset_state(self):
                pass

        # Should be registered
        metric_class = METRIC_REGISTRY.get('test_custom_metric')
        assert metric_class == TestCustomMetric

        # Clean up
        del METRIC_REGISTRY.metrics['test_custom_metric']


class TestRegistryIntegration:
    """Integration tests for registries."""

    def test_loss_registry_get_and_instantiate(self):
        """Should be able to get and instantiate registered losses."""
        registry = LossRegistry()

        @registry.register('mse')
        class MSELoss(tf.keras.losses.Loss):
            def call(self, y_true, y_pred):
                return tf.reduce_mean(tf.square(y_true - y_pred))

        # Get the class
        loss_class = registry.get('mse')

        # Instantiate it
        loss_fn = loss_class()

        # Use it
        y_true = tf.constant([[1.0], [2.0]])
        y_pred = tf.constant([[1.1], [1.9]])
        loss = loss_fn(y_true, y_pred)

        assert tf.math.is_finite(loss)

    def test_metric_registry_get_and_instantiate(self):
        """Should be able to get and instantiate registered metrics."""
        registry = MetricRegistry()

        @registry.register('accuracy')
        class AccuracyMetric(tf.keras.metrics.Metric):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.correct = self.add_weight(name='correct', initializer='zeros')
                self.total = self.add_weight(name='total', initializer='zeros')

            def update_state(self, y_true, y_pred, sample_weight=None):
                matches = tf.cast(tf.equal(y_true, y_pred), tf.float32)
                self.correct.assign_add(tf.reduce_sum(matches))
                self.total.assign_add(tf.cast(tf.size(y_true), tf.float32))

            def result(self):
                return tf.math.divide_no_nan(self.correct, self.total)

            def reset_state(self):
                self.correct.assign(0)
                self.total.assign(0)

        # Get the class
        metric_class = registry.get('accuracy')

        # Instantiate it
        metric = metric_class()

        # Use it
        y_true = tf.constant([[1.0], [0.0], [1.0]])
        y_pred = tf.constant([[1.0], [0.0], [1.0]])
        metric.update_state(y_true, y_pred)
        result = metric.result()

        assert result == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
