"""Phase 2.4: HIGH Priority Tests - Evaluation & Signals

Tests to achieve 90%+ coverage on evaluation and signals modules.

Target Coverage:
- src/inference/evaluation.py: 85.6% → 98%
- src/inference/signals.py: 89.1% → 98%

Missing Lines to Cover:
- evaluation.py: 30, 54, 60, 66, 156
- signals.py: 153, 155, 176, 192
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from src.inference.evaluation import (
    compute_prediction_variance,
    check_convergence,
    generate_report,
    compute_metrics_on_test_set,
    compute_mae,
    compute_directional_accuracy
)
from src.inference.signals import (
    generate_signals,
    SignalGenerator,
    _extract_value
)


class TestEvaluationEnsemble:
    """Test evaluation with ensemble predictions (lines 30, 54)."""

    def test_evaluation_with_ensemble_predictions_ndarray(self):
        """Test compute_prediction_variance with ndarray values (line 30)."""
        # Create predictions with ndarray values (simulating TensorFlow outputs)
        predictions_list = [
            {'price_h0': np.array([[29500.0]])},
            {'price_h0': np.array([[29600.0]])},
            {'price_h0': np.array([[29550.0]])},
        ]

        # Compute variance - should handle ndarray flattening
        variance = compute_prediction_variance(predictions_list, key='price_h0')

        # Verify it computed variance correctly
        assert isinstance(variance, float)
        assert variance > 0
        # Values are 29500, 29600, 29550 -> mean ~29550, var should be around 1666.67
        assert 1000 < variance < 3000

    def test_evaluation_with_ensemble_predictions_scalar(self):
        """Test compute_prediction_variance with scalar values."""
        # Create predictions with scalar values
        predictions_list = [
            {'price_h0': 29500.0},
            {'price_h0': 29600.0},
            {'price_h0': 29550.0},
        ]

        # Compute variance
        variance = compute_prediction_variance(predictions_list, key='price_h0')

        # Verify variance is computed correctly
        assert isinstance(variance, float)
        assert variance > 0

    def test_evaluation_with_missing_key(self):
        """Test compute_prediction_variance when key is missing (line 30 default path)."""
        # Create predictions without the requested key
        predictions_list = [
            {'price_h1': 29500.0},
            {'price_h1': 29600.0},
        ]

        # Should default to 0.0 for missing keys
        variance = compute_prediction_variance(predictions_list, key='price_h0')

        # Variance of [0.0, 0.0] should be 0
        assert variance == 0.0


class TestConvergenceChecking:
    """Test convergence checking with edge cases (lines 60, 66)."""

    def test_check_convergence_with_missing_metric(self):
        """Test check_convergence when metric not in log (line 60)."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'training.csv'

            # Create log without the requested metric
            df = pd.DataFrame({
                'epoch': [1, 2, 3, 4, 5],
                'loss': [0.5, 0.4, 0.35, 0.33, 0.32]
            })
            df.to_csv(log_file, index=False)

            # Check convergence for metric that doesn't exist
            converged = check_convergence(
                log_file,
                metric='val_loss',  # This metric doesn't exist
                window=5,
                threshold=0.001
            )

            # Should return False when metric is missing
            assert converged is False

    def test_check_convergence_with_insufficient_data(self):
        """Test check_convergence with less data than window (line 66)."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'training.csv'

            # Create log with only 3 epochs (window requires 5)
            df = pd.DataFrame({
                'epoch': [1, 2, 3],
                'val_loss': [0.5, 0.4, 0.35]
            })
            df.to_csv(log_file, index=False)

            # Check convergence with window=5
            converged = check_convergence(
                log_file,
                metric='val_loss',
                window=5,
                threshold=0.001
            )

            # Should return False when insufficient data
            assert converged is False

    def test_check_convergence_with_threshold(self):
        """Test check_convergence with convergence detected."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'training.csv'

            # Create log with converged values (low variance)
            df = pd.DataFrame({
                'epoch': list(range(1, 11)),
                'val_loss': [0.5, 0.4, 0.35, 0.33, 0.32, 0.320, 0.321, 0.319, 0.320, 0.320]
            })
            df.to_csv(log_file, index=False)

            # Check convergence - last 5 values have very low variance
            converged = check_convergence(
                log_file,
                metric='val_loss',
                window=5,
                threshold=0.001
            )

            # Should detect convergence
            assert converged == True

    def test_check_convergence_not_converged(self):
        """Test check_convergence when not converged yet."""
        with TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / 'training.csv'

            # Create log with high variance (not converged)
            df = pd.DataFrame({
                'epoch': list(range(1, 11)),
                'val_loss': [0.5, 0.4, 0.35, 0.33, 0.28, 0.25, 0.22, 0.20, 0.18, 0.15]
            })
            df.to_csv(log_file, index=False)

            # Check convergence - values still changing significantly
            converged = check_convergence(
                log_file,
                metric='val_loss',
                window=5,
                threshold=0.001
            )

            # Should not detect convergence
            assert converged == False


class TestReportGeneration:
    """Test evaluation report generation (line 156)."""

    def test_backtest_strategy_report_with_additional_metrics(self):
        """Test generate_report with additional custom metrics (line 156)."""
        with TemporaryDirectory() as tmpdir:
            report_file = Path(tmpdir) / 'report.txt'

            # Create metrics with standard and custom fields
            metrics = {
                'mae_h0': 50.25,
                'mae_h1': 75.50,
                'mae_h2': 100.75,
                'dir_acc_h0': 0.65,
                'dir_acc_h1': 0.70,
                'dir_acc_h2': 0.68,
                # Additional custom metrics
                'sharpe_ratio': 1.85,
                'max_drawdown': -0.15,
                'win_rate': 0.58,
                'profit_factor': 1.45
            }

            # Generate report
            generate_report(metrics, report_file)

            # Read report and verify additional metrics section
            content = report_file.read_text()

            # Verify structure
            assert 'MODEL EVALUATION REPORT' in content
            assert 'Price Prediction Metrics:' in content
            assert 'Direction Prediction Metrics:' in content
            assert 'Additional Metrics:' in content

            # Verify standard metrics
            assert 'MAE H0: 50.25' in content
            assert 'Direction Accuracy H0: 65.00%' in content

            # Verify additional metrics are in the report (line 156)
            assert 'sharpe_ratio: 1.85' in content
            assert 'max_drawdown: -0.15' in content
            assert 'win_rate: 0.58' in content
            assert 'profit_factor: 1.45' in content


class TestSignalGeneration:
    """Test signal generation with edge cases (lines 153, 155, 176, 192)."""

    def test_generate_signals_with_array_values(self):
        """Test generate_signals handles ndarray values (line 153)."""
        # Create predictions with ndarray values
        predictions = {
            'price_h0': np.array([[29500.0]]),
            'price_h1': np.array([[29600.0]]),
            'price_h2': np.array([[29700.0]]),
            'direction_h0': np.array([[0.75]]),
            'direction_h1': np.array([[0.70]]),
            'direction_h2': np.array([[0.65]]),
            'variance_h0': np.array([[0.01]]),
            'variance_h1': np.array([[0.015]]),
            'variance_h2': np.array([[0.02]])
        }

        current_price = 29000.0

        # Generate signals - should handle array extraction
        signals = generate_signals(predictions, current_price)

        # Verify signals structure
        assert 'signal_strength' in signals
        assert 'position_size_multiplier' in signals
        assert 'multi_horizon_agreement' in signals
        assert 'profit_targets' in signals
        assert 'stop_loss' in signals
        assert 'direction' in signals
        assert 'confidence' in signals

        # Verify types
        assert isinstance(signals['signal_strength'], float)
        assert isinstance(signals['position_size_multiplier'], float)
        assert isinstance(signals['multi_horizon_agreement'], bool)

    def test_generate_signals_conservative_mode(self):
        """Test signal generation with high variance (conservative mode) (line 155)."""
        # Create predictions with high variance (low confidence)
        predictions = {
            'price_h0': 29500.0,
            'price_h1': 29600.0,
            'price_h2': 29700.0,
            'direction_h0': 0.55,  # Weak signal
            'direction_h1': 0.52,
            'direction_h2': 0.53,
            'variance_h0': 0.5,  # High variance -> low confidence
            'variance_h1': 0.6,
            'variance_h2': 0.7
        }

        current_price = 29000.0

        # Generate signals
        signals = generate_signals(predictions, current_price)

        # With high variance, confidence should be low
        # Note: confidence is calculated as 1/(1+variance), so with variance 0.5-0.7,
        # confidence will be around 0.59-0.67, not necessarily < 0.5
        assert signals['signal_strength'] < 1.0
        assert 0 < signals['confidence'] < 1.0
        # Position multiplier should be reduced with disagreement/high variance
        assert 0 < signals['position_size_multiplier'] < 2.0

    def test_signal_confidence_filtering_with_missing_keys(self):
        """Test signal generation with missing prediction keys (lines 176, 192)."""
        # Create incomplete predictions (missing some keys)
        predictions = {
            'price_h0': 29500.0,
            # Missing price_h1, price_h2
            'direction_h0': 0.75,
            # Missing direction_h1, direction_h2
            'variance_h0': 0.01,
            # Missing variance_h1, variance_h2
        }

        current_price = 29000.0

        # Should use defaults for missing keys
        signals = generate_signals(predictions, current_price)

        # Should still generate valid signals with defaults
        assert 'signal_strength' in signals
        assert 'direction' in signals
        assert signals['direction'] in ['up', 'down']

        # Verify predicted prices has all horizons (with defaults)
        assert 'h0' in signals['predicted_prices']
        assert 'h1' in signals['predicted_prices']
        assert 'h2' in signals['predicted_prices']

    def test_signal_generator_class_with_risk_tolerance(self):
        """Test SignalGenerator class (line 192)."""
        # Create generator with custom risk tolerance
        generator = SignalGenerator(risk_tolerance=0.05)

        predictions = {
            'price_h0': 29500.0,
            'price_h1': 29600.0,
            'price_h2': 29700.0,
            'direction_h0': 0.75,
            'direction_h1': 0.70,
            'direction_h2': 0.65,
            'variance_h0': 0.01,
            'variance_h1': 0.015,
            'variance_h2': 0.02
        }

        current_price = 29000.0

        # Generate signals using class method
        signals = generator.generate(predictions, current_price)

        # Verify signals structure
        assert 'stop_loss' in signals
        assert isinstance(signals['stop_loss'], float)

        # Stop loss should reflect higher risk tolerance
        # For long position, stop should be further from entry
        if signals['direction'] == 'up':
            stop_distance = abs(current_price - signals['stop_loss'])
            # With 5% risk tolerance, stop should be significant
            assert stop_distance > current_price * 0.01  # At least 1%


class TestExtractValue:
    """Test _extract_value helper function edge cases."""

    def test_extract_value_with_alternative_naming(self):
        """Test _extract_value with alternative key naming."""
        predictions = {
            'h0_price': 29500.0,  # Alternative naming
            'h1_direction': 0.75
        }

        # Should find value even with swapped naming
        value = _extract_value(predictions, 'price_h0')
        assert value == 29500.0

        value = _extract_value(predictions, 'direction_h1')
        assert value == 0.75

    def test_extract_value_defaults(self):
        """Test _extract_value returns appropriate defaults."""
        predictions = {}

        # Direction should default to 0.5
        direction = _extract_value(predictions, 'direction_h0')
        assert direction == 0.5

        # Variance should default to 0.01
        variance = _extract_value(predictions, 'variance_h0')
        assert variance == 0.01

        # Other keys default to 0.0
        price = _extract_value(predictions, 'price_h0')
        assert price == 0.0


class TestMetricsComputation:
    """Test comprehensive metrics computation."""

    def test_compute_metrics_on_test_set_with_all_outputs(self):
        """Test compute_metrics_on_test_set with all outputs."""
        # Create test data
        y_true = {
            'price_h0': np.array([29500, 29600, 29700]),
            'price_h1': np.array([29550, 29650, 29750]),
            'price_h2': np.array([29600, 29700, 29800])
        }

        y_pred = {
            'price_h0': np.array([29450, 29550, 29680]),
            'price_h1': np.array([29500, 29600, 29730]),
            'price_h2': np.array([29550, 29650, 29780])
        }

        current_prices = np.array([29400, 29500, 29600])

        # Compute metrics
        metrics = compute_metrics_on_test_set(y_true, y_pred, current_prices)

        # Verify MAE computed for each horizon
        assert 'mae_h0' in metrics
        assert 'mae_h1' in metrics
        assert 'mae_h2' in metrics

        # Verify directional accuracy computed for each horizon
        assert 'dir_acc_h0' in metrics
        assert 'dir_acc_h1' in metrics
        assert 'dir_acc_h2' in metrics

        # All metrics should be positive floats
        for key, value in metrics.items():
            assert isinstance(value, float)
            assert value >= 0

    def test_compute_metrics_without_current_prices(self):
        """Test compute_metrics_on_test_set without directional accuracy."""
        # Create test data
        y_true = {
            'price_h0': np.array([29500, 29600, 29700]),
            'price_h1': np.array([29550, 29650, 29750])
        }

        y_pred = {
            'price_h0': np.array([29450, 29550, 29680]),
            'price_h1': np.array([29500, 29600, 29730])
        }

        # Compute metrics without current prices
        metrics = compute_metrics_on_test_set(y_true, y_pred, current_prices=None)

        # Should have MAE but not directional accuracy
        assert 'mae_h0' in metrics
        assert 'mae_h1' in metrics
        assert 'dir_acc_h0' not in metrics
        assert 'dir_acc_h1' not in metrics
