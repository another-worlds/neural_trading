"""Unit tests for data preprocessing module.

Tests preprocessing, windowing, scaling as specified in SRS Section 3.2.
"""
import pytest
import numpy as np
import pandas as pd
from src.data.preprocessor import (
    Preprocessor,
    create_windows,
    scale_features,
    generate_targets,
    StandardScalerWrapper,
)


class TestPreprocessor:
    """Test Preprocessor class for data transformation."""

    def test_init_with_config(self, sample_config):
        """Should initialize with configuration."""
        preprocessor = Preprocessor(sample_config['data'])
        assert preprocessor.lookback == 60
        assert preprocessor.window_step == 1

    def test_create_windows_basic(self, sample_ohlcv_data):
        """Should create sliding windows from data."""
        preprocessor = Preprocessor({'lookback': 10, 'window_step': 1})
        windows = preprocessor.create_windows(sample_ohlcv_data)

        assert isinstance(windows, np.ndarray)
        assert windows.shape[1] == 10  # lookback
        assert windows.shape[0] <= len(sample_ohlcv_data) - 10

    def test_create_windows_with_step(self, sample_ohlcv_data):
        """Should respect window step parameter."""
        preprocessor = Preprocessor({'lookback': 10, 'window_step': 5})
        windows = preprocessor.create_windows(sample_ohlcv_data)

        # With step=5, should have fewer windows
        expected_count = (len(sample_ohlcv_data) - 10) // 5
        assert windows.shape[0] <= expected_count + 1

    def test_create_windows_60_minute_lookback(self, sample_ohlcv_data):
        """Should create 60-minute lookback windows as per SRS."""
        preprocessor = Preprocessor({'lookback': 60, 'window_step': 1})
        windows = preprocessor.create_windows(sample_ohlcv_data)

        assert windows.shape[1] == 60

    def test_sequence_limit(self):
        """Should limit sequences to 2880 as per SRS."""
        # Generate large dataset
        large_data = pd.DataFrame({
            'close': np.random.randn(5000) + 42000
        })

        preprocessor = Preprocessor({
            'lookback': 60,
            'window_step': 1,
            'sequence_limit': 2880
        })
        windows = preprocessor.create_windows(large_data)

        assert windows.shape[0] <= 2880

    def test_fit_scaler(self, sample_ohlcv_data):
        """Should fit scaler on training data."""
        preprocessor = Preprocessor({'lookback': 10, 'window_step': 1})
        features = sample_ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values

        preprocessor.fit_scaler(features)
        assert preprocessor.scaler is not None
        assert hasattr(preprocessor.scaler, 'mean_')

    def test_transform_features(self, sample_ohlcv_data):
        """Should transform features using fitted scaler."""
        preprocessor = Preprocessor({'lookback': 10, 'window_step': 1})
        features = sample_ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values

        preprocessor.fit_scaler(features)
        scaled = preprocessor.transform(features)

        assert scaled.shape == features.shape
        # Scaled data should have approximately zero mean
        assert np.abs(scaled.mean()) < 0.5

    def test_inverse_transform(self, sample_ohlcv_data):
        """Should inverse transform scaled features."""
        preprocessor = Preprocessor({'lookback': 10, 'window_step': 1})
        features = sample_ohlcv_data[['open', 'high', 'low', 'close', 'volume']].values

        preprocessor.fit_scaler(features)
        scaled = preprocessor.transform(features)
        inversed = preprocessor.inverse_transform(scaled)

        # Should approximately recover original data
        np.testing.assert_array_almost_equal(inversed, features, decimal=4)

    def test_separate_scalers_for_input_output(self):
        """Should maintain separate scalers for input and output as per SRS."""
        preprocessor = Preprocessor({'lookback': 10, 'window_step': 1})

        input_features = np.random.randn(100, 10)
        output_targets = np.random.randn(100, 3)

        preprocessor.fit_input_scaler(input_features)
        preprocessor.fit_output_scaler(output_targets)

        assert preprocessor.input_scaler is not None
        assert preprocessor.output_scaler is not None
        assert preprocessor.input_scaler != preprocessor.output_scaler

    def test_save_scaler(self, tmp_path):
        """Should save scaler to file."""
        preprocessor = Preprocessor({'lookback': 10, 'window_step': 1})
        features = np.random.randn(100, 5)
        preprocessor.fit_scaler(features)

        scaler_file = tmp_path / "test_scaler.joblib"
        preprocessor.save_scaler(scaler_file)

        assert scaler_file.exists()

    def test_load_scaler(self, tmp_path):
        """Should load scaler from file."""
        preprocessor = Preprocessor({'lookback': 10, 'window_step': 1})
        features = np.random.randn(100, 5)
        preprocessor.fit_scaler(features)

        scaler_file = tmp_path / "test_scaler.joblib"
        preprocessor.save_scaler(scaler_file)

        # Create new preprocessor and load
        new_preprocessor = Preprocessor({'lookback': 10, 'window_step': 1})
        new_preprocessor.load_scaler(scaler_file)

        # Should produce same transformation
        scaled1 = preprocessor.transform(features)
        scaled2 = new_preprocessor.transform(features)
        np.testing.assert_array_almost_equal(scaled1, scaled2)


class TestGenerateTargets:
    """Test target generation for multi-horizon predictions."""

    def test_generate_h0_targets(self, sample_ohlcv_data):
        """Should generate h0 (1-minute) targets."""
        targets = generate_targets(sample_ohlcv_data, horizon=0)

        assert len(targets) == len(sample_ohlcv_data) - 1
        # h0 = next minute's close
        np.testing.assert_array_equal(
            targets,
            sample_ohlcv_data['close'].values[1:]
        )

    def test_generate_h1_targets(self, sample_ohlcv_data):
        """Should generate h1 (5-minute) targets."""
        targets = generate_targets(sample_ohlcv_data, horizon=1)

        assert len(targets) == len(sample_ohlcv_data) - 5

    def test_generate_h2_targets(self, sample_ohlcv_data):
        """Should generate h2 (15-minute) targets."""
        targets = generate_targets(sample_ohlcv_data, horizon=2)

        assert len(targets) == len(sample_ohlcv_data) - 15

    def test_generate_direction_targets(self, sample_ohlcv_data):
        """Should generate binary direction targets."""
        direction_targets = generate_targets(
            sample_ohlcv_data,
            horizon=0,
            target_type='direction'
        )

        # Should be binary (0 or 1)
        assert all(x in [0, 1] for x in direction_targets)

    def test_direction_up_when_price_increases(self):
        """Direction should be 1 when price increases."""
        data = pd.DataFrame({
            'close': [100, 110, 120, 130]  # Increasing prices
        })

        directions = generate_targets(data, horizon=0, target_type='direction')
        # All should be 1 (up)
        assert all(d == 1 for d in directions)

    def test_direction_down_when_price_decreases(self):
        """Direction should be 0 when price decreases."""
        data = pd.DataFrame({
            'close': [130, 120, 110, 100]  # Decreasing prices
        })

        directions = generate_targets(data, horizon=0, target_type='direction')
        # All should be 0 (down)
        assert all(d == 0 for d in directions)

    def test_generate_multi_horizon_targets(self, sample_ohlcv_data):
        """Should generate targets for all three horizons."""
        from src.data.preprocessor import generate_multi_horizon_targets

        targets = generate_multi_horizon_targets(sample_ohlcv_data)

        assert 'h0' in targets
        assert 'h1' in targets
        assert 'h2' in targets
        assert len(targets['h0']) > len(targets['h2'])  # h2 has fewer due to longer horizon


class TestFeatureEngineering:
    """Test feature engineering functions."""

    def test_calculate_returns(self, sample_ohlcv_data):
        """Should calculate price returns."""
        from src.data.preprocessor import calculate_returns

        returns = calculate_returns(sample_ohlcv_data['close'])

        assert len(returns) == len(sample_ohlcv_data) - 1
        assert np.isfinite(returns).all()

    def test_calculate_log_returns(self, sample_ohlcv_data):
        """Should calculate log returns."""
        from src.data.preprocessor import calculate_log_returns

        log_returns = calculate_log_returns(sample_ohlcv_data['close'])

        assert len(log_returns) == len(sample_ohlcv_data) - 1
        assert np.isfinite(log_returns).all()

    def test_calculate_volatility(self, sample_ohlcv_data):
        """Should calculate rolling volatility."""
        from src.data.preprocessor import calculate_volatility

        volatility = calculate_volatility(sample_ohlcv_data['close'], window=10)

        assert len(volatility) == len(sample_ohlcv_data)
        assert np.all(volatility[~np.isnan(volatility)] >= 0)  # Volatility is non-negative

    def test_add_extended_trend_features(self, sample_ohlcv_data):
        """Should add 1m, 5m, 15m percent change features as per SRS."""
        from src.data.preprocessor import add_extended_trend_features

        data_with_trends = add_extended_trend_features(sample_ohlcv_data)

        assert 'pct_change_1m' in data_with_trends.columns
        assert 'pct_change_5m' in data_with_trends.columns
        assert 'pct_change_15m' in data_with_trends.columns

    def test_add_volume_statistics(self, sample_ohlcv_data):
        """Should add volume statistics."""
        from src.data.preprocessor import add_volume_statistics

        data_with_volume = add_volume_statistics(sample_ohlcv_data)

        assert 'volume_ma' in data_with_volume.columns or 'volume_ratio' in data_with_volume.columns


class TestDataSplitting:
    """Test train/val/test splitting."""

    def test_split_data_basic(self, sample_ohlcv_data):
        """Should split data into train/val/test."""
        from src.data.preprocessor import split_data

        train, val, test = split_data(
            sample_ohlcv_data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )

        assert len(train) + len(val) + len(test) == len(sample_ohlcv_data)

    def test_split_ratios_respected(self, sample_ohlcv_data):
        """Should respect specified split ratios."""
        from src.data.preprocessor import split_data

        train, val, test = split_data(
            sample_ohlcv_data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )

        total_len = len(sample_ohlcv_data)
        assert len(train) == pytest.approx(total_len * 0.7, abs=1)
        assert len(val) == pytest.approx(total_len * 0.15, abs=1)

    def test_split_maintains_temporal_order(self, sample_ohlcv_data):
        """Should maintain temporal order in splits."""
        from src.data.preprocessor import split_data

        train, val, test = split_data(
            sample_ohlcv_data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )

        # Validate temporal order
        assert train['datetime'].iloc[-1] < val['datetime'].iloc[0]
        assert val['datetime'].iloc[-1] < test['datetime'].iloc[0]

    def test_visualize_split(self, sample_ohlcv_data, tmp_path):
        """Should visualize dataset split."""
        from src.data.preprocessor import visualize_split

        train, val, test = split_data(
            sample_ohlcv_data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )

        output_file = tmp_path / "split_viz.png"
        visualize_split(train, val, test, output_file)

        assert output_file.exists()
