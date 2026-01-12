"""Phase 3.2: MEDIUM Priority Tests - Preprocessor Utilities

Tests to achieve 95%+ coverage on data preprocessing utilities.

Target Coverage:
- src/data/preprocessor.py: 84.9% → 95%

Missing Lines to Cover:
- 61, 150, 166, 197, 214, 260-266, 307, 375, 457, 487
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from sklearn.preprocessing import StandardScaler

from src.data.preprocessor import (
    Preprocessor,
    StandardScalerWrapper,
    create_windows,
    scale_features,
    generate_targets,
    generate_multi_horizon_targets,
    calculate_returns,
    calculate_log_returns,
    calculate_volatility,
    split_data
)


class TestStandardScalerWrapper:
    """Test StandardScalerWrapper class (line 61)."""

    def test_scaler_wrapper_fit_transform(self):
        """Test fit_transform method (line 61)."""
        wrapper = StandardScalerWrapper()

        # Create test data
        X = np.random.randn(100, 5)

        # Fit and transform
        X_scaled = wrapper.fit_transform(X)

        # Verify scaling worked
        assert X_scaled.shape == X.shape
        assert wrapper.mean_ is not None
        assert wrapper.scale_ is not None

        # Verify mean is close to 0 and std close to 1
        np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros(5), decimal=10)
        np.testing.assert_array_almost_equal(X_scaled.std(axis=0), np.ones(5), decimal=10)

    def test_scaler_wrapper_inverse_transform(self):
        """Test inverse_transform method."""
        wrapper = StandardScalerWrapper()

        X = np.random.randn(100, 5)
        X_scaled = wrapper.fit_transform(X)

        # Inverse transform
        X_recovered = wrapper.inverse_transform(X_scaled)

        # Should recover original values
        np.testing.assert_array_almost_equal(X, X_recovered, decimal=5)


class TestPreprocessorTransform:
    """Test Preprocessor transform methods (lines 150, 166)."""

    def test_preprocessor_transform_without_fit_raises_error(self):
        """Test transform without fitting raises error (line 150)."""
        config = {
            'window_size': 60,
            'horizons': {'h0': 1, 'h1': 5, 'h2': 15}
        }
        preprocessor = Preprocessor(config)

        # Create test data
        features = np.random.randn(100, 5)

        # Transform without fitting should raise AttributeError
        with pytest.raises(AttributeError):
            preprocessor.transform(features)

    def test_preprocessor_inverse_transform(self):
        """Test inverse_transform method (line 166)."""
        config = {
            'window_size': 60,
            'horizons': {'h0': 1, 'h1': 5, 'h2': 15}
        }
        preprocessor = Preprocessor(config)

        # Create and fit data
        features = np.random.randn(200, 5)
        preprocessor.fit_scaler(features)

        # Transform
        scaled_features = preprocessor.transform(features)

        # Inverse transform (line 166)
        recovered_features = preprocessor.inverse_transform(scaled_features)

        # Should recover original values
        np.testing.assert_array_almost_equal(features, recovered_features, decimal=5)


class TestPreprocessorScalerManagement:
    """Test preprocessor scaler save/load (lines 197, 214)."""

    def test_preprocessor_fit_input_scaler(self):
        """Test fit_input_scaler method (line 197)."""
        config = {
            'window_size': 60,
            'horizons': {'h0': 1, 'h1': 5, 'h2': 15}
        }
        preprocessor = Preprocessor(config)

        # Fit input scaler
        features = np.random.randn(200, 5)
        preprocessor.fit_input_scaler(features)

        # Verify scaler is fitted
        assert preprocessor.feature_scaler is not None
        assert preprocessor.feature_scaler.mean_ is not None

    def test_preprocessor_fit_output_scaler(self):
        """Test fit_output_scaler method (line 214)."""
        config = {
            'window_size': 60,
            'horizons': {'h0': 1, 'h1': 5, 'h2': 15}
        }
        preprocessor = Preprocessor(config)

        # Fit output scaler
        targets = np.random.randn(200, 9)  # 3 horizons × 3 outputs each
        preprocessor.fit_output_scaler(targets)

        # Verify output scaler is fitted
        assert hasattr(preprocessor, 'target_scaler')

    def test_preprocessor_save_load_scaler(self):
        """Test save_scaler and load_scaler methods (lines 197, 214)."""
        with TemporaryDirectory() as tmpdir:
            scaler_path = Path(tmpdir) / 'scaler.pkl'

            config = {
                'window_size': 60,
                'horizons': {'h0': 1, 'h1': 5, 'h2': 15}
            }
            preprocessor = Preprocessor(config)

            # Fit scaler
            features = np.random.randn(200, 5)
            preprocessor.fit_scaler(features)

            # Save scaler
            preprocessor.save_scaler(scaler_path)

            # Verify file was created
            assert scaler_path.exists()

            # Create new preprocessor and load scaler
            preprocessor2 = Preprocessor(config)
            preprocessor2.load_scaler(scaler_path)

            # Transform with both should give same results
            test_features = np.random.randn(50, 5)
            scaled1 = preprocessor.transform(test_features)
            scaled2 = preprocessor2.transform(test_features)

            np.testing.assert_array_almost_equal(scaled1, scaled2)


class TestScaleFeaturesFunction:
    """Test scale_features function (lines 260-266)."""

    def test_scale_features_with_new_scaler(self):
        """Test scale_features creates new scaler when none provided (lines 260-266)."""
        # Create features
        features = np.random.randn(100, 5)

        # Scale without providing scaler
        scaled_features, scaler = scale_features(features, scaler=None)

        # Verify scaler was created (lines 260-266)
        assert scaler is not None
        assert isinstance(scaler, StandardScaler)
        assert scaled_features.shape == features.shape

        # Verify scaling worked
        np.testing.assert_array_almost_equal(scaled_features.mean(axis=0), np.zeros(5), decimal=10)

    def test_scale_features_with_existing_scaler(self):
        """Test scale_features with existing scaler."""
        features = np.random.randn(100, 5)

        # Create and fit scaler
        scaler = StandardScaler()
        scaler.fit(features)

        # Scale with existing scaler
        scaled_features, returned_scaler = scale_features(features, scaler=scaler)

        # Should return the same scaler
        assert returned_scaler is scaler
        assert scaled_features.shape == features.shape


class TestGenerateTargets:
    """Test generate_targets function (line 307)."""

    def test_generate_targets_multi_horizon(self):
        """Test generate_targets with multiple horizons (line 307)."""
        # Create sample price data
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
        current_prices = np.array([100.0, 101.0, 102.0, 103.0])

        horizons = {'h0': 1, 'h1': 2, 'h2': 3}

        # Generate targets
        targets = generate_targets(prices, current_prices, horizons)

        # Verify structure (line 307 - multi-horizon target generation)
        assert 'price_h0' in targets
        assert 'price_h1' in targets
        assert 'price_h2' in targets
        assert 'direction_h0' in targets
        assert 'direction_h1' in targets
        assert 'direction_h2' in targets

        # Verify shapes
        assert targets['price_h0'].shape[0] == len(current_prices)

    def test_generate_targets_directional_accuracy(self):
        """Test that direction targets are computed correctly."""
        prices = np.array([100.0, 102.0, 101.0, 103.0, 104.0])
        current_prices = np.array([100.0, 102.0])

        horizons = {'h0': 1, 'h1': 2}

        targets = generate_targets(prices, current_prices, horizons)

        # For h0 (1 step ahead):
        # current=100, future=102 -> direction should be 1 (up)
        assert targets['direction_h0'][0] == 1

        # For current=102, future=101 -> direction should be 0 (down)
        assert targets['direction_h0'][1] == 0


class TestVolatilityCalculation:
    """Test calculate_volatility function (lines 375, 457)."""

    def test_calculate_volatility_with_window(self):
        """Test volatility calculation with window parameter (line 375)."""
        # Create price series
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])

        # Calculate volatility with window
        volatility = calculate_volatility(prices, window=3)

        # Verify output
        assert len(volatility) == len(prices)
        assert not np.isnan(volatility[-1])  # Last value should be valid

    def test_calculate_volatility_default_window(self):
        """Test volatility calculation with default window (line 457)."""
        prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])

        # Calculate with default window (20)
        volatility = calculate_volatility(prices)

        # Verify output has correct length
        assert len(volatility) == len(prices)

    def test_calculate_volatility_fillna(self):
        """Test that volatility fills NaN values (line 487)."""
        # Small price series (less than default window)
        prices = pd.Series([100, 101, 102, 103, 104])

        # Calculate volatility
        volatility = calculate_volatility(prices, window=20)

        # Should not have NaN values (filled)
        assert not np.any(np.isnan(volatility))


class TestCreateWindows:
    """Test window creation edge cases."""

    def test_create_windows_with_dataframe(self):
        """Test create_windows with DataFrame input."""
        # Create DataFrame
        df = pd.DataFrame({
            'open': np.random.randn(100),
            'high': np.random.randn(100),
            'low': np.random.randn(100),
            'close': np.random.randn(100),
            'volume': np.random.randn(100)
        })

        # Create windows
        windows = create_windows(df, window_size=10)

        # Verify shape
        expected_samples = len(df) - 10 + 1
        assert windows.shape[0] == expected_samples
        assert windows.shape[1] == 10
        assert windows.shape[2] == 5

    def test_create_windows_with_numpy(self):
        """Test create_windows with numpy array input."""
        # Create numpy array
        data = np.random.randn(100, 5)

        # Create windows
        windows = create_windows(data, window_size=10)

        # Verify shape
        expected_samples = len(data) - 10 + 1
        assert windows.shape[0] == expected_samples
        assert windows.shape[1] == 10
        assert windows.shape[2] == 5


class TestCalculateReturns:
    """Test return calculation functions."""

    def test_calculate_returns_series(self):
        """Test returns calculation with pandas Series."""
        prices = pd.Series([100, 102, 101, 103, 105])

        returns = calculate_returns(prices)

        # Verify first return is NaN
        assert np.isnan(returns[0])

        # Verify subsequent returns
        assert returns[1] == 0.02  # (102-100)/100
        np.testing.assert_almost_equal(returns[2], -0.00980392, decimal=6)  # (101-102)/102

    def test_calculate_log_returns_array(self):
        """Test log returns calculation with numpy array."""
        prices = np.array([100, 102, 101, 103, 105])

        log_returns = calculate_log_returns(prices)

        # Verify first return is NaN
        assert np.isnan(log_returns[0])

        # Verify log returns
        expected_first = np.log(102 / 100)
        np.testing.assert_almost_equal(log_returns[1], expected_first, decimal=6)


class TestSplitData:
    """Test data splitting functionality."""

    def test_split_data_with_validation(self):
        """Test split_data creates train/val/test splits."""
        # Create sample data
        X = np.random.randn(1000, 60, 5)
        y = np.random.randn(1000, 9)

        # Split data
        splits = split_data(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        # Verify splits
        assert 'X_train' in splits
        assert 'y_train' in splits
        assert 'X_val' in splits
        assert 'y_val' in splits
        assert 'X_test' in splits
        assert 'y_test' in splits

        # Verify sizes
        assert len(splits['X_train']) == 700
        assert len(splits['X_val']) == 150
        assert len(splits['X_test']) == 150

    def test_split_data_no_validation(self):
        """Test split_data with only train/test split."""
        X = np.random.randn(1000, 60, 5)
        y = np.random.randn(1000, 9)

        # Split without validation set
        splits = split_data(X, y, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2)

        # Verify no validation set
        assert splits['X_val'] is None
        assert splits['y_val'] is None

        # Verify train/test sizes
        assert len(splits['X_train']) == 800
        assert len(splits['X_test']) == 200


class TestMultiHorizonTargets:
    """Test multi-horizon target generation."""

    def test_generate_multi_horizon_targets(self):
        """Test generate_multi_horizon_targets function."""
        # Create sample DataFrame
        data = pd.DataFrame({
            'close': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
        })

        # Generate multi-horizon targets
        targets = generate_multi_horizon_targets(data)

        # Verify all horizons present
        assert 'price_h0' in targets
        assert 'price_h1' in targets
        assert 'price_h2' in targets

        # Verify shapes
        assert isinstance(targets['price_h0'], np.ndarray)
        assert len(targets['price_h0']) < len(data)  # Some samples lost due to horizons


class TestPreprocessorIntegration:
    """Test full preprocessor workflow."""

    def test_preprocessor_end_to_end(self):
        """Test complete preprocessing pipeline."""
        config = {
            'window_size': 10,
            'horizons': {'h0': 1, 'h1': 2, 'h2': 3}
        }

        # Create sample data
        data = pd.DataFrame({
            'open': np.random.randn(100) + 100,
            'high': np.random.randn(100) + 101,
            'low': np.random.randn(100) + 99,
            'close': np.random.randn(100) + 100,
            'volume': np.random.randn(100) * 1000 + 10000
        })

        preprocessor = Preprocessor(config)

        # Create windows
        windows = preprocessor.create_windows(data)

        # Fit and transform
        preprocessor.fit_scaler(windows.reshape(-1, windows.shape[-1]))
        scaled_windows = preprocessor.transform(windows.reshape(-1, windows.shape[-1]))

        # Verify output
        assert scaled_windows.shape[0] == windows.shape[0] * windows.shape[1]
        assert not np.any(np.isnan(scaled_windows))
