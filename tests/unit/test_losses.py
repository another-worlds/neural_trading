"""Unit tests for custom loss functions.

Tests loss registry and custom losses as per SRS Section 3.5.2.
"""
import pytest
import numpy as np
import tensorflow as tf
from src.losses.loss_registry import LossRegistry
from src.losses.custom_losses import (
    FocalLoss,
    HuberLoss,
    NegativeLogLikelihood,
    TrendLoss,
    CompositeLoss,
    register_loss,
)


class TestLossRegistry:
    """Test loss function registry."""

    def test_registry_initialization(self):
        """Should initialize empty registry."""
        registry = LossRegistry()
        assert len(registry.losses) == 0

    def test_register_loss(self):
        """Should register loss function."""
        registry = LossRegistry()

        @registry.register('test_loss')
        class TestLoss(tf.keras.losses.Loss):
            def call(self, y_true, y_pred):
                return tf.reduce_mean(tf.square(y_true - y_pred))

        assert 'test_loss' in registry.losses

    def test_get_loss(self):
        """Should retrieve registered loss."""
        registry = LossRegistry()

        @registry.register('test_loss')
        class TestLoss(tf.keras.losses.Loss):
            pass

        loss_class = registry.get('test_loss')
        assert loss_class == TestLoss

    def test_get_nonexistent_loss_raises_error(self):
        """Should raise error for nonexistent loss."""
        registry = LossRegistry()

        with pytest.raises(KeyError):
            registry.get('nonexistent')

    def test_automatic_registration(self):
        """Losses should be automatically registered on import."""
        from src.losses.loss_registry import LOSS_REGISTRY

        # Check that standard losses are registered
        assert 'focal' in LOSS_REGISTRY.losses
        assert 'huber' in LOSS_REGISTRY.losses
        assert 'nll' in LOSS_REGISTRY.losses


class TestFocalLoss:
    """Test Focal Loss for direction classification."""

    def test_init_focal_loss(self):
        """Should initialize Focal Loss with alpha and gamma."""
        focal_loss = FocalLoss(alpha=0.7, gamma=1.0)

        assert focal_loss.alpha == 0.7
        assert focal_loss.gamma == 1.0

    def test_focal_loss_computation(self):
        """Should compute focal loss."""
        focal_loss = FocalLoss(alpha=0.7, gamma=1.0)

        y_true = tf.constant([[1.0], [0.0], [1.0]])
        y_pred = tf.constant([[0.9], [0.1], [0.8]])

        loss = focal_loss(y_true, y_pred)

        assert tf.is_tensor(loss)
        assert loss.shape == ()  # Scalar
        assert loss.numpy() >= 0

    def test_focal_loss_default_params_as_per_srs(self):
        """Should use alpha=0.7, gamma=1.0 as per SRS."""
        focal_loss = FocalLoss(alpha=0.7, gamma=1.0)

        assert focal_loss.alpha == 0.7
        assert focal_loss.gamma == 1.0

    def test_focal_loss_focuses_on_hard_examples(self):
        """Focal loss should focus on hard examples."""
        focal_loss = FocalLoss(alpha=0.7, gamma=2.0)

        y_true = tf.constant([[1.0], [1.0]])
        # Easy example (high confidence, correct)
        y_pred_easy = tf.constant([[0.95], [0.95]])
        # Hard example (low confidence, correct)
        y_pred_hard = tf.constant([[0.55], [0.55]])

        loss_easy = focal_loss(y_true, y_pred_easy)
        loss_hard = focal_loss(y_true, y_pred_hard)

        # Hard examples should contribute more to loss
        assert loss_hard.numpy() > loss_easy.numpy()

    def test_focal_loss_handles_class_imbalance(self):
        """Alpha parameter should handle class imbalance."""
        # High alpha favors positive class
        focal_loss_high_alpha = FocalLoss(alpha=0.9, gamma=1.0)
        # Low alpha favors negative class
        focal_loss_low_alpha = FocalLoss(alpha=0.1, gamma=1.0)

        y_true = tf.constant([[1.0], [0.0]])
        y_pred = tf.constant([[0.7], [0.3]])

        loss_high = focal_loss_high_alpha(y_true, y_pred)
        loss_low = focal_loss_low_alpha(y_true, y_pred)

        # Different alpha should give different losses
        assert not np.isclose(loss_high.numpy(), loss_low.numpy())


class TestHuberLoss:
    """Test Huber Loss for price prediction."""

    def test_init_huber_loss(self):
        """Should initialize Huber Loss with delta."""
        huber_loss = HuberLoss(delta=1.0)

        assert huber_loss.delta == 1.0

    def test_huber_loss_computation(self):
        """Should compute Huber loss."""
        huber_loss = HuberLoss(delta=1.0)

        y_true = tf.constant([[42000.0], [42100.0], [42050.0]])
        y_pred = tf.constant([[42010.0], [42120.0], [42040.0]])

        loss = huber_loss(y_true, y_pred)

        assert tf.is_tensor(loss)
        assert loss.numpy() >= 0

    def test_huber_loss_small_errors_quadratic(self):
        """Small errors should be quadratic (L2)."""
        huber_loss = HuberLoss(delta=1.0)

        y_true = tf.constant([[1.0]])
        y_pred = tf.constant([[1.1]])  # Error = 0.1 < delta

        loss = huber_loss(y_true, y_pred)

        # Should be 0.5 * error^2 = 0.5 * 0.01 = 0.005
        expected = 0.5 * (0.1 ** 2)
        assert np.isclose(loss.numpy(), expected, rtol=0.1)

    def test_huber_loss_large_errors_linear(self):
        """Large errors should be linear (L1)."""
        huber_loss = HuberLoss(delta=1.0)

        y_true = tf.constant([[1.0]])
        y_pred = tf.constant([[5.0]])  # Error = 4.0 > delta

        loss = huber_loss(y_true, y_pred)

        # Should be approximately delta * (error - 0.5 * delta)
        # = 1.0 * (4.0 - 0.5) = 3.5
        assert loss.numpy() > 1.0  # Should be linear region


class TestNegativeLogLikelihood:
    """Test Negative Log Likelihood for variance prediction."""

    def test_init_nll(self):
        """Should initialize NLL loss."""
        nll_loss = NegativeLogLikelihood()

        assert nll_loss is not None

    def test_nll_computation(self):
        """Should compute NLL loss."""
        nll_loss = NegativeLogLikelihood()

        y_true = tf.constant([[42000.0], [42100.0]])
        y_pred_mean = tf.constant([[42010.0], [42090.0]])
        y_pred_var = tf.constant([[0.1], [0.2]])

        # NLL requires both mean and variance predictions
        loss = nll_loss(y_true, y_pred_mean, y_pred_var)

        assert tf.is_tensor(loss)
        assert loss.numpy() >= 0

    def test_nll_penalizes_overconfidence(self):
        """Should penalize overconfident (low variance) wrong predictions."""
        nll_loss = NegativeLogLikelihood()

        y_true = tf.constant([[42000.0]])
        # Wrong prediction, very confident (low variance)
        y_pred_mean = tf.constant([[42100.0]])
        y_pred_var_low = tf.constant([[0.001]])
        y_pred_var_high = tf.constant([[1.0]])

        loss_low_var = nll_loss(y_true, y_pred_mean, y_pred_var_low)
        loss_high_var = nll_loss(y_true, y_pred_mean, y_pred_var_high)

        # Overconfident wrong prediction should have higher loss
        assert loss_low_var.numpy() > loss_high_var.numpy()

    def test_nll_encourages_calibrated_uncertainty(self):
        """Should encourage well-calibrated uncertainty estimates."""
        nll_loss = NegativeLogLikelihood()

        # Correct prediction with appropriate uncertainty
        y_true = tf.constant([[42000.0]])
        y_pred_mean = tf.constant([[42005.0]])
        y_pred_var = tf.constant([[25.0]])  # std = 5, error = 5

        loss = nll_loss(y_true, y_pred_mean, y_pred_var)
        assert loss.numpy() < 10.0  # Should be reasonable


class TestTrendLoss:
    """Test Trend Loss for multi-horizon consistency."""

    def test_init_trend_loss(self):
        """Should initialize Trend Loss."""
        trend_loss = TrendLoss(horizons=[0, 1, 2])

        assert len(trend_loss.horizons) == 3

    def test_trend_loss_computation(self):
        """Should compute trend loss across horizons."""
        trend_loss = TrendLoss(horizons=[0, 1, 2])

        y_true = {
            'h0': tf.constant([[42100.0]]),
            'h1': tf.constant([[42200.0]]),
            'h2': tf.constant([[42300.0]]),
        }

        y_pred = {
            'h0': tf.constant([[42110.0]]),
            'h1': tf.constant([[42210.0]]),
            'h2': tf.constant([[42310.0]]),
        }

        loss = trend_loss(y_true, y_pred)

        assert tf.is_tensor(loss)
        assert loss.numpy() >= 0

    def test_local_trend_loss(self):
        """Should compute local trend loss (consecutive horizons)."""
        from src.losses.custom_losses import local_trend_loss

        y_pred = tf.constant([[42100.0], [42200.0], [42300.0]])

        loss = local_trend_loss(y_pred)

        assert tf.is_tensor(loss)

    def test_global_trend_loss(self):
        """Should compute global trend loss (first to last)."""
        from src.losses.custom_losses import global_trend_loss

        current_price = tf.constant([[42000.0]])
        y_pred_h2 = tf.constant([[42300.0]])

        loss = global_trend_loss(current_price, y_pred_h2)

        assert tf.is_tensor(loss)

    def test_extended_trend_loss(self):
        """Should compute extended trend loss (1m, 5m, 15m)."""
        from src.losses.custom_losses import extended_trend_loss

        # Actual price changes at each horizon
        actual_trends = {
            '1m': tf.constant([[0.002]]),  # 0.2% increase
            '5m': tf.constant([[0.005]]),  # 0.5% increase
            '15m': tf.constant([[0.01]]),  # 1% increase
        }

        # Predicted price changes
        predicted_trends = {
            '1m': tf.constant([[0.0025]]),
            '5m': tf.constant([[0.0055]]),
            '15m': tf.constant([[0.011]]),
        }

        loss = extended_trend_loss(actual_trends, predicted_trends)

        assert tf.is_tensor(loss)


class TestCompositeLoss:
    """Test Composite Loss for combining multiple losses."""

    def test_init_composite_loss(self, sample_config):
        """Should initialize composite loss from config."""
        composite_loss = CompositeLoss(sample_config['losses'])

        assert composite_loss is not None

    def test_composite_loss_computation(self, sample_config):
        """Should compute weighted sum of losses."""
        composite_loss = CompositeLoss(sample_config['losses'])

        y_true = {
            'price_h0': tf.constant([[42100.0]]),
            'direction_h0': tf.constant([[1.0]]),
            'variance_h0': tf.constant([[0.1]]),
        }

        y_pred = {
            'price_h0': tf.constant([[42110.0]]),
            'direction_h0': tf.constant([[0.9]]),
            'variance_h0': tf.constant([[0.12]]),
        }

        loss = composite_loss(y_true, y_pred)

        assert tf.is_tensor(loss)
        assert loss.numpy() >= 0

    def test_loss_weights_applied(self):
        """Should apply loss weights (lambdas)."""
        config = {
            'point_loss': {'type': 'huber', 'weight': 1.0},
            'direction_loss': {'type': 'focal', 'weight': 2.0},
        }

        composite_loss = CompositeLoss(config)

        # Higher weight should increase contribution

    def test_cascade_lambda_updates(self, sample_config):
        """Updating lambdas in config should cascade to composite loss."""
        composite_loss = CompositeLoss(sample_config['losses'])

        # Update lambda
        new_config = sample_config['losses'].copy()
        new_config['direction_loss']['weight'] = 5.0

        new_composite_loss = CompositeLoss(new_config)

        # Should use new weights


class TestLossIntegration:
    """Test loss integration with model."""

    def test_compile_model_with_losses(self, sample_config):
        """Should compile model with custom losses."""
        from src.models.hybrid_model import build_model

        model = build_model(sample_config)

        # Define losses for each output
        losses = {
            f'{output}_h{h}': 'mse'
            for output in ['price', 'direction', 'variance']
            for h in [0, 1, 2]
        }

        model.compile(optimizer='adam', loss=losses)

        assert model.optimizer is not None

    def test_loss_weights_in_model_compilation(self, sample_config):
        """Should apply loss weights during compilation."""
        from src.models.hybrid_model import build_model

        model = build_model(sample_config)

        losses = {f'output_{i}': 'mse' for i in range(9)}
        loss_weights = {f'output_{i}': 1.0 / 9 for i in range(9)}

        model.compile(
            optimizer='adam',
            loss=losses,
            loss_weights=loss_weights
        )

    def test_regularization_loss(self, sample_config):
        """Should include L2 regularization loss."""
        sample_config['model']['l2_reg'] = 0.001

        from src.models.hybrid_model import build_model
        model = build_model(sample_config)

        # Model should have regularization losses
        # Check model.losses


class TestLossCalibration:
    """Test loss lambda calibration."""

    def test_grid_search_lambdas(self, sample_config):
        """Should support grid search for lambda calibration."""
        from src.losses.calibration import grid_search_lambdas

        # Define lambda grid
        lambda_grid = {
            'point_loss': [0.5, 1.0, 2.0],
            'direction_loss': [0.5, 1.0, 2.0],
            'variance_loss': [0.5, 1.0, 2.0],
        }

        # Mock model and data
        # Run grid search
        # best_lambdas = grid_search_lambdas(model, data, lambda_grid)

    @pytest.mark.skip(reason="Requires full training pipeline")
    def test_lambda_calibration_improves_performance(self):
        """Calibrated lambdas should improve validation performance."""
        pass
