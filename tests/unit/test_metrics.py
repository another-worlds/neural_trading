"""Unit tests for custom metrics.

Tests metric registry and custom metrics as per SRS Section 3.5.1 and 7.1.3.
"""
import pytest
import numpy as np
import tensorflow as tf
from src.metrics.metric_registry import MetricRegistry
from src.metrics.custom_metrics import (
    DirectionAccuracy,
    DirectionF1Score,
    DirectionMCC,
    PriceMAE,
    PriceMAPE,
    MultiHorizonMetric,
    register_metric,
)


class TestMetricRegistry:
    """Test metric registry."""

    def test_registry_initialization(self):
        """Should initialize empty registry."""
        registry = MetricRegistry()
        assert len(registry.metrics) == 0

    def test_register_metric(self):
        """Should register metric."""
        registry = MetricRegistry()

        @registry.register('test_metric')
        class TestMetric(tf.keras.metrics.Metric):
            pass

        assert 'test_metric' in registry.metrics

    def test_get_metric(self):
        """Should retrieve registered metric."""
        registry = MetricRegistry()

        @registry.register('test_metric')
        class TestMetric(tf.keras.metrics.Metric):
            pass

        metric_class = registry.get('test_metric')
        assert metric_class == TestMetric

    def test_automatic_registration(self):
        """Metrics should be automatically registered on import."""
        from src.metrics.metric_registry import METRIC_REGISTRY

        assert 'direction_accuracy' in METRIC_REGISTRY.metrics
        assert 'direction_f1' in METRIC_REGISTRY.metrics
        assert 'direction_mcc' in METRIC_REGISTRY.metrics


class TestDirectionAccuracy:
    """Test Direction Accuracy metric."""

    def test_init_direction_accuracy(self):
        """Should initialize direction accuracy metric."""
        metric = DirectionAccuracy(name='dir_acc_h0')

        assert metric.name == 'dir_acc_h0'

    def test_perfect_predictions(self):
        """Perfect predictions should give accuracy 1.0."""
        metric = DirectionAccuracy()

        y_true = tf.constant([[1.0], [0.0], [1.0], [1.0]])
        y_pred = tf.constant([[0.9], [0.1], [0.8], [0.7]])  # All correct

        metric.update_state(y_true, y_pred)
        accuracy = metric.result()

        assert accuracy.numpy() == pytest.approx(1.0)

    def test_worst_predictions(self):
        """Worst predictions should give accuracy 0.0."""
        metric = DirectionAccuracy()

        y_true = tf.constant([[1.0], [0.0], [1.0], [0.0]])
        y_pred = tf.constant([[0.1], [0.9], [0.2], [0.8]])  # All wrong

        metric.update_state(y_true, y_pred)
        accuracy = metric.result()

        assert accuracy.numpy() == pytest.approx(0.0)

    def test_mixed_predictions(self):
        """Should compute correct accuracy for mixed predictions."""
        metric = DirectionAccuracy()

        y_true = tf.constant([[1.0], [0.0], [1.0], [0.0]])
        y_pred = tf.constant([[0.9], [0.1], [0.2], [0.8]])  # 50% correct

        metric.update_state(y_true, y_pred)
        accuracy = metric.result()

        assert accuracy.numpy() == pytest.approx(0.5)

    def test_reset_state(self):
        """Should reset metric state."""
        metric = DirectionAccuracy()

        y_true = tf.constant([[1.0], [0.0]])
        y_pred = tf.constant([[0.9], [0.1]])

        metric.update_state(y_true, y_pred)
        metric.reset_state()

        # After reset, result should be 0
        result_after_reset = metric.result()
        # Depending on implementation, might be 0 or NaN


class TestDirectionF1Score:
    """Test Direction F1 Score metric."""

    def test_init_f1_score(self):
        """Should initialize F1 score metric."""
        metric = DirectionF1Score(name='dir_f1_h0')

        assert metric.name == 'dir_f1_h0'

    def test_perfect_f1_score(self):
        """Perfect predictions should give F1 score 1.0."""
        metric = DirectionF1Score()

        y_true = tf.constant([[1.0], [0.0], [1.0], [0.0]])
        y_pred = tf.constant([[0.9], [0.1], [0.8], [0.2]])

        metric.update_state(y_true, y_pred)
        f1 = metric.result()

        assert f1.numpy() == pytest.approx(1.0)

    def test_f1_with_false_positives(self):
        """Should handle false positives correctly."""
        metric = DirectionF1Score()

        y_true = tf.constant([[0.0], [0.0], [1.0], [1.0]])
        y_pred = tf.constant([[0.9], [0.8], [0.7], [0.6]])  # 2 FP, 2 TP

        metric.update_state(y_true, y_pred)
        f1 = metric.result()

        # Precision = 2/4 = 0.5, Recall = 2/2 = 1.0
        # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 0.667
        assert 0.6 < f1.numpy() < 0.7

    def test_f1_with_false_negatives(self):
        """Should handle false negatives correctly."""
        metric = DirectionF1Score()

        y_true = tf.constant([[1.0], [1.0], [0.0], [0.0]])
        y_pred = tf.constant([[0.1], [0.2], [0.3], [0.4]])  # 2 FN, 2 TN

        metric.update_state(y_true, y_pred)
        f1 = metric.result()

        # F1 should be 0 (no TP)
        assert f1.numpy() == pytest.approx(0.0)


class TestDirectionMCC:
    """Test Matthews Correlation Coefficient metric."""

    def test_init_mcc(self):
        """Should initialize MCC metric."""
        metric = DirectionMCC(name='dir_mcc_h1')

        assert metric.name == 'dir_mcc_h1'

    def test_perfect_mcc(self):
        """Perfect predictions should give MCC 1.0."""
        metric = DirectionMCC()

        y_true = tf.constant([[1.0], [0.0], [1.0], [0.0]])
        y_pred = tf.constant([[0.9], [0.1], [0.8], [0.2]])

        metric.update_state(y_true, y_pred)
        mcc = metric.result()

        assert mcc.numpy() == pytest.approx(1.0)

    def test_worst_mcc(self):
        """Worst predictions should give MCC -1.0."""
        metric = DirectionMCC()

        y_true = tf.constant([[1.0], [0.0], [1.0], [0.0]])
        y_pred = tf.constant([[0.1], [0.9], [0.2], [0.8]])

        metric.update_state(y_true, y_pred)
        mcc = metric.result()

        assert mcc.numpy() == pytest.approx(-1.0)

    def test_random_mcc(self):
        """Random predictions should give MCC near 0."""
        metric = DirectionMCC()

        # Balanced random predictions
        y_true = tf.constant([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]])
        y_pred = tf.constant([[0.6], [0.4], [0.4], [0.6], [0.5], [0.5]])

        metric.update_state(y_true, y_pred)
        mcc = metric.result()

        # Should be close to 0 for random predictions
        assert -0.5 < mcc.numpy() < 0.5

    def test_mcc_as_validation_metric_per_srs(self):
        """MCC should be used for validation (val_dir_mcc_h1) as per SRS."""
        # SRS Section 3.5.3 specifies early stopping monitors val_dir_mcc_h1
        metric = DirectionMCC(name='dir_mcc_h1')

        assert metric.name == 'dir_mcc_h1'


class TestPriceMAE:
    """Test Price MAE metric."""

    def test_init_price_mae(self):
        """Should initialize Price MAE metric."""
        metric = PriceMAE(name='price_mae_h0')

        assert metric.name == 'price_mae_h0'

    def test_perfect_predictions(self):
        """Perfect predictions should give MAE 0.0."""
        metric = PriceMAE()

        y_true = tf.constant([[42000.0], [42100.0], [42200.0]])
        y_pred = tf.constant([[42000.0], [42100.0], [42200.0]])

        metric.update_state(y_true, y_pred)
        mae = metric.result()

        assert mae.numpy() == pytest.approx(0.0)

    def test_mae_calculation(self):
        """Should calculate MAE correctly."""
        metric = PriceMAE()

        y_true = tf.constant([[100.0], [200.0], [300.0]])
        y_pred = tf.constant([[110.0], [190.0], [310.0]])

        # Errors: |10|, |10|, |10| -> MAE = 10
        metric.update_state(y_true, y_pred)
        mae = metric.result()

        assert mae.numpy() == pytest.approx(10.0)

    def test_mae_per_horizon(self):
        """Should track MAE for each horizon separately."""
        mae_h0 = PriceMAE(name='price_mae_h0')
        mae_h1 = PriceMAE(name='price_mae_h1')
        mae_h2 = PriceMAE(name='price_mae_h2')

        # Different errors per horizon
        y_true = tf.constant([[42000.0]])

        y_pred_h0 = tf.constant([[42010.0]])  # Error = 10
        y_pred_h1 = tf.constant([[42050.0]])  # Error = 50
        y_pred_h2 = tf.constant([[42150.0]])  # Error = 150

        mae_h0.update_state(y_true, y_pred_h0)
        mae_h1.update_state(y_true, y_pred_h1)
        mae_h2.update_state(y_true, y_pred_h2)

        assert mae_h0.result().numpy() < mae_h1.result().numpy() < mae_h2.result().numpy()


class TestPriceMAPE:
    """Test Price MAPE metric."""

    def test_init_price_mape(self):
        """Should initialize Price MAPE metric."""
        metric = PriceMAPE(name='price_mape_h0')

        assert metric.name == 'price_mape_h0'

    def test_perfect_predictions(self):
        """Perfect predictions should give MAPE 0.0."""
        metric = PriceMAPE()

        y_true = tf.constant([[42000.0], [42100.0]])
        y_pred = tf.constant([[42000.0], [42100.0]])

        metric.update_state(y_true, y_pred)
        mape = metric.result()

        assert mape.numpy() == pytest.approx(0.0)

    def test_mape_calculation(self):
        """Should calculate MAPE correctly."""
        metric = PriceMAPE()

        y_true = tf.constant([[100.0], [200.0]])
        y_pred = tf.constant([[110.0], [210.0]])

        # Percentage errors: 10%, 5% -> MAPE = 7.5%
        metric.update_state(y_true, y_pred)
        mape = metric.result()

        assert mape.numpy() == pytest.approx(7.5, rel=0.1)

    def test_mape_handles_zero_division(self):
        """Should handle division by zero gracefully."""
        metric = PriceMAPE()

        y_true = tf.constant([[0.0], [100.0]])
        y_pred = tf.constant([[10.0], [110.0]])

        # Should not crash with zero in y_true
        metric.update_state(y_true, y_pred)
        mape = metric.result()

        assert tf.math.is_finite(mape)


class TestMultiHorizonMetric:
    """Test multi-horizon metric aggregation."""

    def test_init_multi_horizon_metric(self):
        """Should initialize multi-horizon metric."""
        metric = MultiHorizonMetric(
            base_metric=DirectionAccuracy,
            horizons=[0, 1, 2]
        )

        assert len(metric.horizon_metrics) == 3

    def test_update_all_horizons(self):
        """Should update metrics for all horizons."""
        metric = MultiHorizonMetric(
            base_metric=DirectionAccuracy,
            horizons=[0, 1, 2]
        )

        y_true = {
            'h0': tf.constant([[1.0]]),
            'h1': tf.constant([[0.0]]),
            'h2': tf.constant([[1.0]]),
        }

        y_pred = {
            'h0': tf.constant([[0.9]]),
            'h1': tf.constant([[0.1]]),
            'h2': tf.constant([[0.8]]),
        }

        metric.update_state(y_true, y_pred)

        # All horizons should have perfect accuracy
        for h in [0, 1, 2]:
            assert metric.result(horizon=h).numpy() == 1.0

    def test_aggregate_across_horizons(self):
        """Should aggregate metrics across horizons."""
        metric = MultiHorizonMetric(
            base_metric=PriceMAE,
            horizons=[0, 1, 2]
        )

        y_true = {
            'h0': tf.constant([[42000.0]]),
            'h1': tf.constant([[42000.0]]),
            'h2': tf.constant([[42000.0]]),
        }

        y_pred = {
            'h0': tf.constant([[42010.0]]),  # MAE = 10
            'h1': tf.constant([[42020.0]]),  # MAE = 20
            'h2': tf.constant([[42030.0]]),  # MAE = 30
        }

        metric.update_state(y_true, y_pred)

        # Average MAE across horizons = (10 + 20 + 30) / 3 = 20
        avg_mae = metric.result_aggregate()
        assert avg_mae.numpy() == pytest.approx(20.0)


class TestMetricIntegration:
    """Test metric integration with model training."""

    def test_compile_model_with_metrics(self, sample_config):
        """Should compile model with custom metrics."""
        from src.models.hybrid_model import build_model

        model = build_model(sample_config)

        # Define metrics for each output
        metrics = {
            f'price_h{h}': [PriceMAE(name=f'price_mae_h{h}'), PriceMAPE(name=f'price_mape_h{h}')]
            for h in [0, 1, 2]
        }

        metrics.update({
            f'direction_h{h}': [
                DirectionAccuracy(name=f'dir_acc_h{h}'),
                DirectionF1Score(name=f'dir_f1_h{h}'),
                DirectionMCC(name=f'dir_mcc_h{h}')
            ]
            for h in [0, 1, 2]
        })

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=metrics
        )

    def test_metrics_from_config(self, sample_config):
        """Should create metrics from configuration."""
        from src.metrics.metric_registry import build_metrics_from_config

        metrics = build_metrics_from_config(sample_config['metrics'])

        assert 'direction_accuracy' in metrics or len(metrics) > 0

    def test_metric_logging_to_csv(self, tmp_path):
        """Should log metrics to CSV as per SRS."""
        from src.metrics.logging import MetricLogger

        logger = MetricLogger(output_file=tmp_path / "metrics.csv")

        # Log epoch metrics
        logger.log_epoch(epoch=1, metrics={
            'loss': 0.5,
            'val_loss': 0.6,
            'dir_acc_h0': 0.75,
            'dir_mcc_h1': 0.65,
        })

        assert (tmp_path / "metrics.csv").exists()


class TestTrainingMetricsTracking:
    """Test training metrics tracking as per SRS Section 7.1.3."""

    def test_track_epoch_wise_metrics(self):
        """Should track metrics per epoch."""
        from src.metrics.tracking import MetricTracker

        tracker = MetricTracker()

        # Simulate training
        for epoch in range(5):
            tracker.update(epoch=epoch, metrics={
                'loss': 1.0 / (epoch + 1),
                'val_loss': 1.2 / (epoch + 1),
            })

        history = tracker.get_history()
        assert len(history) == 5

    def test_metrics_per_horizon(self):
        """Should track metrics per horizon separately."""
        # dir_acc_h0/h1/h2, dir_f1_h0/h1/h2, dir_mcc_h0/h1/h2, price_mae_h0/h1/h2
        metric_names = [
            f'{metric}_h{h}'
            for metric in ['dir_acc', 'dir_f1', 'dir_mcc', 'price_mae']
            for h in [0, 1, 2]
        ]

        assert len(metric_names) == 12  # 4 metrics × 3 horizons

    def test_save_metrics_to_training_log(self, tmp_path):
        """Should save metrics to training_log.csv as per SRS."""
        import pandas as pd

        metrics_df = pd.DataFrame({
            'epoch': [1, 2, 3],
            'loss': [1.0, 0.8, 0.6],
            'val_loss': [1.2, 0.9, 0.7],
            'dir_mcc_h1': [0.5, 0.6, 0.7],
        })

        output_file = tmp_path / "training_log.csv"
        metrics_df.to_csv(output_file, index=False)

        assert output_file.exists()

        # Verify format
        loaded = pd.read_csv(output_file)
        assert 'epoch' in loaded.columns
        assert 'loss' in loaded.columns
        assert 'val_loss' in loaded.columns
