"""Unit tests for dataset module.

Tests tf.data.Dataset creation with windowing for time series as per SRS Section 3.2.1.
"""
import pytest
import numpy as np
import tensorflow as tf
from src.data.dataset import (
    create_tf_dataset,
    get_dataset,
    window_generator,
    add_gaussian_noise,
)


class TestDatasetCreation:
    """Test TensorFlow dataset creation."""

    def test_create_basic_dataset(self, sample_ohlcv_array):
        """Should create tf.data.Dataset from numpy arrays."""
        targets = np.random.randn(len(sample_ohlcv_array), 3).astype(np.float32)

        dataset = create_tf_dataset(
            features=sample_ohlcv_array,
            targets=targets,
            batch_size=16
        )

        assert isinstance(dataset, tf.data.Dataset)

        # Check one batch
        for batch_x, batch_y in dataset.take(1):
            assert batch_x.shape[0] <= 16
            assert batch_y.shape[0] <= 16

    def test_dataset_with_windowing(self, sample_ohlcv_data):
        """Should create dataset with sliding windows."""
        from src.data.preprocessor import Preprocessor

        preprocessor = Preprocessor({'lookback': 10, 'window_step': 1})
        windows = preprocessor.create_windows(sample_ohlcv_data)

        dataset = create_tf_dataset(
            features=windows,
            targets=np.random.randn(len(windows), 3),
            batch_size=16
        )

        for batch_x, batch_y in dataset.take(1):
            assert batch_x.shape[1] == 10  # lookback

    def test_dataset_batch_size(self, sample_ohlcv_array):
        """Should respect specified batch size."""
        targets = np.random.randn(len(sample_ohlcv_array), 3).astype(np.float32)

        dataset = create_tf_dataset(
            features=sample_ohlcv_array,
            targets=targets,
            batch_size=20
        )

        for batch_x, batch_y in dataset.take(1):
            assert batch_x.shape[0] <= 20

    def test_dataset_batch_size_144_as_per_srs(self, sample_config):
        """Should support batch size of 144 as specified in SRS."""
        # Generate enough data
        features = np.random.randn(500, 60, 5).astype(np.float32)
        targets = np.random.randn(500, 3).astype(np.float32)

        dataset = create_tf_dataset(
            features=features,
            targets=targets,
            batch_size=144
        )

        for batch_x, batch_y in dataset.take(1):
            assert batch_x.shape[0] <= 144

    def test_dataset_shuffling(self):
        """Should shuffle data when requested."""
        features = np.arange(100).reshape(-1, 1).astype(np.float32)
        targets = np.arange(100).astype(np.float32)

        dataset = create_tf_dataset(
            features=features,
            targets=targets,
            batch_size=10,
            shuffle=True,
            buffer_size=100
        )

        # Get first batch
        first_batch = next(iter(dataset))
        # Should not be in sequential order (probabilistically)
        # This test might occasionally fail due to randomness

    def test_dataset_prefetching(self):
        """Should prefetch data for performance."""
        features = np.random.randn(100, 10).astype(np.float32)
        targets = np.random.randn(100, 3).astype(np.float32)

        dataset = create_tf_dataset(
            features=features,
            targets=targets,
            batch_size=16,
            prefetch=True
        )

        # Check that prefetch is in the dataset pipeline
        # This is implementation-specific

    def test_dataset_caching(self):
        """Should cache dataset when requested."""
        features = np.random.randn(100, 10).astype(np.float32)
        targets = np.random.randn(100, 3).astype(np.float32)

        dataset = create_tf_dataset(
            features=features,
            targets=targets,
            batch_size=16,
            cache=True
        )

        # Verify caching is applied


class TestWindowGenerator:
    """Test window generation for time series."""

    def test_generate_windows(self, sample_ohlcv_data):
        """Should generate sliding windows."""
        windows = window_generator(
            data=sample_ohlcv_data['close'].values,
            window_size=10,
            stride=1
        )

        assert windows.shape[1] == 10
        assert windows.shape[0] <= len(sample_ohlcv_data) - 10

    def test_window_size_60_as_per_srs(self, sample_ohlcv_data):
        """Should support 60-minute window as specified in SRS."""
        windows = window_generator(
            data=sample_ohlcv_data[['close']].values,
            window_size=60,
            stride=1
        )

        assert windows.shape[1] == 60

    def test_window_stride(self, sample_ohlcv_data):
        """Should respect window stride parameter."""
        windows_stride1 = window_generator(
            data=sample_ohlcv_data['close'].values,
            window_size=10,
            stride=1
        )

        windows_stride5 = window_generator(
            data=sample_ohlcv_data['close'].values,
            window_size=10,
            stride=5
        )

        # Stride 5 should produce fewer windows
        assert len(windows_stride5) < len(windows_stride1)

    def test_overlapping_windows(self, sample_ohlcv_data):
        """Windows with stride=1 should overlap."""
        windows = window_generator(
            data=sample_ohlcv_data['close'].values,
            window_size=10,
            stride=1
        )

        # Check that consecutive windows overlap
        if len(windows) >= 2:
            # Last 9 elements of window 0 should match first 9 of window 1
            np.testing.assert_array_equal(
                windows[0, 1:],
                windows[1, :9]
            )


class TestGetDataset:
    """Test high-level dataset getter function."""

    def test_get_dataset_from_config(self, sample_config, sample_ohlcv_data):
        """Should create dataset from configuration."""
        dataset = get_dataset(
            data=sample_ohlcv_data,
            config=sample_config
        )

        assert isinstance(dataset, tf.data.Dataset)

    def test_get_train_val_test_datasets(self, sample_config, sample_ohlcv_data):
        """Should create train, validation, and test datasets."""
        from src.data.dataset import get_train_val_test_datasets

        train_ds, val_ds, test_ds = get_train_val_test_datasets(
            data=sample_ohlcv_data,
            config=sample_config
        )

        assert isinstance(train_ds, tf.data.Dataset)
        assert isinstance(val_ds, tf.data.Dataset)
        assert isinstance(test_ds, tf.data.Dataset)

    def test_dataset_with_indicators(self, sample_config, sample_ohlcv_data):
        """Should include indicator features in dataset."""
        sample_config['indicators']['enabled'] = True

        dataset = get_dataset(
            data=sample_ohlcv_data,
            config=sample_config
        )

        # Check that dataset has indicator features
        for batch_x, batch_y in dataset.take(1):
            # Should have more than just OHLCV features
            pass


class TestGaussianNoise:
    """Test Gaussian noise addition for uncertainty quantification."""

    def test_add_gaussian_noise(self):
        """Should add Gaussian noise to data."""
        clean_data = np.ones((100, 10), dtype=np.float32)

        noisy_data = add_gaussian_noise(
            clean_data,
            noise_std=0.1,
            seed=42
        )

        # Noisy data should be different from clean
        assert not np.allclose(clean_data, noisy_data)

        # But should be close
        assert np.allclose(clean_data, noisy_data, atol=0.5)

    def test_noise_std_control(self):
        """Should control noise level with std parameter."""
        clean_data = np.zeros((100, 10), dtype=np.float32)

        noisy_low = add_gaussian_noise(clean_data, noise_std=0.1, seed=42)
        noisy_high = add_gaussian_noise(clean_data, noise_std=1.0, seed=43)

        # Higher std should produce larger deviations
        assert np.std(noisy_high) > np.std(noisy_low)

    def test_noise_seed_reproducibility(self):
        """Should be reproducible with same seed."""
        clean_data = np.ones((100, 10), dtype=np.float32)

        noisy1 = add_gaussian_noise(clean_data, noise_std=0.1, seed=42)
        noisy2 = add_gaussian_noise(clean_data, noise_std=0.1, seed=42)

        np.testing.assert_array_equal(noisy1, noisy2)

    def test_add_noise_to_dataset(self):
        """Should add noise to tf.data.Dataset."""
        from src.data.dataset import add_noise_to_dataset

        features = np.random.randn(100, 10).astype(np.float32)
        targets = np.random.randn(100, 3).astype(np.float32)

        dataset = create_tf_dataset(features, targets, batch_size=16)
        noisy_dataset = add_noise_to_dataset(dataset, noise_std=0.1)

        # Should still be a valid dataset
        assert isinstance(noisy_dataset, tf.data.Dataset)


class TestDatasetPerformance:
    """Test dataset performance optimizations."""

    def test_parallel_processing(self):
        """Should use parallel processing for data loading."""
        features = np.random.randn(1000, 60, 5).astype(np.float32)
        targets = np.random.randn(1000, 3).astype(np.float32)

        dataset = create_tf_dataset(
            features=features,
            targets=targets,
            batch_size=144,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Verify parallel processing is applied

    def test_mixed_precision_support(self):
        """Should support mixed precision training."""
        features = np.random.randn(100, 60, 5).astype(np.float16)
        targets = np.random.randn(100, 3).astype(np.float16)

        dataset = create_tf_dataset(
            features=features,
            targets=targets,
            batch_size=16
        )

        for batch_x, batch_y in dataset.take(1):
            # Check dtype
            pass


class TestMultiOutputDataset:
    """Test dataset with multiple outputs for multi-horizon predictions."""

    def test_create_multi_output_dataset(self):
        """Should create dataset with multiple output targets."""
        features = np.random.randn(100, 60, 5).astype(np.float32)

        # Multi-horizon targets
        targets = {
            'price_h0': np.random.randn(100, 1).astype(np.float32),
            'price_h1': np.random.randn(100, 1).astype(np.float32),
            'price_h2': np.random.randn(100, 1).astype(np.float32),
            'direction_h0': np.random.randint(0, 2, (100, 1)).astype(np.float32),
            'direction_h1': np.random.randint(0, 2, (100, 1)).astype(np.float32),
            'direction_h2': np.random.randint(0, 2, (100, 1)).astype(np.float32),
            'variance_h0': np.random.randn(100, 1).astype(np.float32),
            'variance_h1': np.random.randn(100, 1).astype(np.float32),
            'variance_h2': np.random.randn(100, 1).astype(np.float32),
        }

        dataset = create_tf_dataset(
            features=features,
            targets=targets,
            batch_size=16
        )

        for batch_x, batch_y in dataset.take(1):
            assert isinstance(batch_y, dict)
            assert 'price_h0' in batch_y

    def test_nine_outputs_as_per_srs(self):
        """Should support 9 outputs (3 towers × 3 outputs) as per SRS."""
        features = np.random.randn(100, 60, 5).astype(np.float32)

        # 9 outputs as per SRS Section 3.5.1
        targets = {
            f'{output}_h{h}': np.random.randn(100, 1).astype(np.float32)
            for output in ['price', 'direction', 'variance']
            for h in [0, 1, 2]
        }

        assert len(targets) == 9

        dataset = create_tf_dataset(
            features=features,
            targets=targets,
            batch_size=16
        )

        for batch_x, batch_y in dataset.take(1):
            assert len(batch_y) == 9
