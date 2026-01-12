"""Additional targeted tests to reach 100% coverage.

Focuses on specific uncovered lines identified through coverage analysis.
"""
import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import tempfile
import yaml
import json


# ============================================================================
# ConfigParser Additional Coverage
# ============================================================================

class TestConfigParserAdditional:
    """Additional tests for ConfigParser uncovered lines."""

    def test_get_section_missing(self):
        """Should raise KeyError for missing section."""
        from src.config.config_parser import ConfigParser

        parser = ConfigParser()
        parser.config = {'data': {}}

        # Line 102: raise KeyError for missing section
        with pytest.raises(KeyError, match="Config section not found"):
            parser.get_section('missing_section')

    def test_set_nested_creates_parents(self):
        """Should create parent keys when setting nested value."""
        from src.config.config_parser import ConfigParser

        parser = ConfigParser()
        parser.config = {}

        # Line 148: creates intermediate dicts if missing
        parser.set('deeply.nested.key.value', 42)

        assert parser.config['deeply']['nested']['key']['value'] == 42

    def test_update_section_new(self):
        """Should create section if it doesn't exist when updating."""
        from src.config.config_parser import ConfigParser

        parser = ConfigParser()
        parser.config = {}

        # Line 162: creates section if missing
        parser.update_section('new_section', {'key': 'value'})

        assert 'new_section' in parser.config
        assert parser.config['new_section']['key'] == 'value'

    def test_save_json_format(self, tmp_path):
        """Should save config in JSON format."""
        from src.config.config_parser import ConfigParser

        parser = ConfigParser()
        parser.config = {'data': {'lookback': 60}}

        json_path = tmp_path / "config.json"
        # Lines 182-184: save as JSON
        parser.save(json_path)

        assert json_path.exists()
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded['data']['lookback'] == 60

    def test_apply_env_overrides_all_types(self, monkeypatch):
        """Should apply environment variable overrides."""
        from src.config.config_parser import ConfigParser

        # Set environment variables
        monkeypatch.setenv('NEURAL_TRADE_EPOCHS', '100')
        monkeypatch.setenv('NEURAL_TRADE_LEARNING_RATE', '0.001')
        monkeypatch.setenv('NEURAL_TRADE_BATCH_SIZE', '144')

        parser = ConfigParser()
        parser.config = {'training': {}, 'model': {}}

        # Lines 212-218: parse and set env vars
        parser.apply_env_overrides()

        assert parser.config['training']['epochs'] == 100
        assert parser.config['training']['learning_rate'] == 0.001
        assert parser.config['training']['batch_size'] == 144

    def test_validate_config_negative_batch_size(self):
        """Should detect negative batch_size validation error."""
        from src.config.config_parser import validate_config_schema

        config = {
            'data': {},
            'training': {'batch_size': -10}
        }

        # Line 262: check batch_size <= 0
        is_valid, errors = validate_config_schema(config)

        assert not is_valid
        assert any('batch_size' in err for err in errors)

    def test_validate_config_negative_epochs(self):
        """Should detect negative epochs validation error."""
        from src.config.config_parser import validate_config_schema

        config = {
            'data': {},
            'training': {'epochs': -5}
        }

        # Line 266-267: check epochs < 0
        is_valid, errors = validate_config_schema(config)

        assert not is_valid
        assert any('epochs' in err for err in errors)

    def test_validate_config_negative_learning_rate(self):
        """Should detect negative learning_rate validation error."""
        from src.config.config_parser import validate_config_schema

        config = {
            'data': {},
            'training': {'learning_rate': -0.1}
        }

        # Line 270-271: check learning_rate <= 0
        is_valid, errors = validate_config_schema(config)

        assert not is_valid
        assert any('learning_rate' in err for err in errors)

    def test_validate_config_negative_alpha(self):
        """Should detect negative alpha in losses."""
        from src.config.config_parser import validate_config_schema

        config = {
            'data': {},
            'training': {},
            'losses': {
                'focal_loss': {'alpha': -0.5}
            }
        }

        # Lines 290-294: check alpha < 0
        is_valid, errors = validate_config_schema(config)

        assert not is_valid
        assert any('alpha' in err for err in errors)

    def test_validate_config_negative_gamma(self):
        """Should detect negative gamma in losses."""
        from src.config.config_parser import validate_config_schema

        config = {
            'data': {},
            'training': {},
            'losses': {
                'focal_loss': {'gamma': -2.0}
            }
        }

        # Lines 297-301: check gamma < 0
        is_valid, errors = validate_config_schema(config)

        assert not is_valid
        assert any('gamma' in err for err in errors)


# ============================================================================
# Dataset Additional Coverage
# ============================================================================

class TestDatasetAdditional:
    """Additional tests for dataset uncovered lines."""

    def test_add_noise_to_dataset_map_function(self):
        """Should apply noise using TF map function."""
        from src.data.dataset import add_noise_to_dataset, create_tf_dataset

        features = np.ones((20, 10), dtype=np.float32)
        targets = np.ones((20, 1), dtype=np.float32)

        dataset = create_tf_dataset(features, targets, batch_size=5, shuffle=False)

        # Lines 161-171: add_noise_fn applied via map
        noisy_dataset = add_noise_to_dataset(dataset, noise_std=0.5, seed=42)

        # Extract batch and check noise was added
        for batch_x, batch_y in noisy_dataset.take(1):
            # Should have noise (not all 1.0)
            assert not np.allclose(batch_x.numpy(), 1.0, atol=0.1)
            # Targets should be unchanged
            assert np.allclose(batch_y.numpy(), 1.0)


# ============================================================================
# Inference Evaluation Additional Coverage
# ============================================================================

class TestInferenceEvaluationAdditional:
    """Additional tests for inference evaluation uncovered lines."""

    def test_compute_metrics_on_test_set_with_dict(self):
        """Should compute metrics when predictions are dict."""
        from src.inference.evaluation import compute_metrics_on_test_set

        y_true = {
            'price_h0': np.array([100.0, 101.0, 102.0]),
            'price_h1': np.array([100.5, 101.5, 102.5]),
            'price_h2': np.array([101.0, 102.0, 103.0])
        }

        y_pred = {
            'price_h0': np.array([100.1, 100.9, 102.1]),
            'price_h1': np.array([100.6, 101.4, 102.4]),
            'price_h2': np.array([101.1, 101.9, 103.1])
        }

        current_prices = np.array([99.0, 100.0, 101.0])

        # Lines 177-198: compute metrics with dicts
        metrics = compute_metrics_on_test_set(y_true, y_pred, current_prices)

        # Should have MAE for each horizon
        assert 'mae_h0' in metrics
        assert 'mae_h1' in metrics
        assert 'mae_h2' in metrics

        # Should have directional accuracy for each horizon
        assert 'dir_acc_h0' in metrics
        assert 'dir_acc_h1' in metrics
        assert 'dir_acc_h2' in metrics


# ============================================================================
# Losses Additional Coverage
# ============================================================================

class TestLossesAdditional:
    """Additional tests for losses uncovered lines."""

    def test_huber_loss_config(self):
        """Should handle get_config for HuberLoss."""
        from src.losses.custom_losses import HuberLoss

        loss_fn = HuberLoss(delta=2.0)

        # Lines 78-83: get_config
        config = loss_fn.get_config()

        assert 'delta' in config
        assert config['delta'] == 2.0

    def test_focal_loss_config(self):
        """Should handle get_config for FocalLoss."""
        from src.losses.custom_losses import FocalLoss

        loss_fn = FocalLoss(alpha=0.8, gamma=2.5)

        # Lines 138-140: get_config
        config = loss_fn.get_config()

        assert 'alpha' in config
        assert 'gamma' in config
        assert config['alpha'] == 0.8
        assert config['gamma'] == 2.5

    def test_nll_loss_config(self):
        """Should handle get_config for NLL."""
        from src.losses.custom_losses import NegativeLogLikelihood

        loss_fn = NegativeLogLikelihood()

        # Lines 178-187: get_config (may be empty or with epsilon)
        config = loss_fn.get_config()

        assert config is not None

    def test_composite_loss_default_mse(self):
        """Should use MSE as default loss type."""
        from src.losses.custom_losses import CompositeLoss

        # Configure with unknown loss type -> should default to MSE
        loss_config = {
            'unknown_loss': {'type': 'unknown_type', 'weight': 1.0}
        }

        # Lines 324-327: default to MSE for unknown type
        loss_fn = CompositeLoss(loss_config=loss_config)

        y_true = tf.constant([[1.0], [2.0]], dtype=tf.float32)
        y_pred = tf.constant([[1.1], [1.9]], dtype=tf.float32)

        loss = loss_fn(y_true, y_pred)
        assert tf.math.is_finite(loss)


# ============================================================================
# Metrics Additional Coverage
# ============================================================================

class TestMetricsAdditional:
    """Additional tests for metrics uncovered lines."""

    def test_price_mae_config(self):
        """Should handle get_config for PriceMAE."""
        from src.metrics.custom_metrics import PriceMAE

        metric = PriceMAE()

        # Lines 304-305: get_config
        config = metric.get_config()

        assert config is not None

    def test_price_mape_config(self):
        """Should handle get_config for PriceMAPE."""
        from src.metrics.custom_metrics import PriceMAPE

        metric = PriceMAPE()

        # Lines 371-372: get_config
        config = metric.get_config()

        assert config is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
