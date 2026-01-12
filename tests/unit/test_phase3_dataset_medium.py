"""Phase 3.3: MEDIUM Priority Tests - Dataset Edge Cases

Tests to achieve 98%+ coverage on dataset module.

Target Coverage:
- src/data/dataset.py: 88.4% → 98%

Missing Lines to Cover:
- 105, 161-171
"""
import pytest
import tensorflow as tf
import numpy as np

from src.data.dataset import (
    window_generator,
    create_tf_dataset,
    add_noise_to_dataset
)


class TestWindowGenerator1D:
    """Test window_generator with 1D data (line 105)."""

    def test_window_generator_1d_data(self):
        """Test window_generator handles 1D input data (line 105)."""
        # Create 1D data
        data_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Generate windows
        windows = window_generator(data_1d, window_size=3, stride=1)

        # Verify windows were generated
        # Line 105 - 1D arrays get expanded to 2D then squeezed, result is 2D
        assert len(windows) == 8  # 10 - 3 + 1 = 8 windows

        # Verify first window
        assert windows[0][0] == 1
        assert windows[0][1] == 2
        assert windows[0][2] == 3

    def test_window_generator_1d_vs_2d(self):
        """Test that 1D data is handled correctly vs 2D."""
        # Create 1D and 2D versions
        data_1d = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        data_2d = data_1d.reshape(-1, 1)

        # Create generators
        windows_1d = window_generator(data_1d, window_size=3)
        windows_2d = window_generator(data_2d, window_size=3)

        # Both should have same number of windows
        assert len(windows_1d) == len(windows_2d)


class TestAddNoiseDetailed:
    """Test add_noise_to_dataset internals (lines 161-171)."""

    def test_add_noise_to_dataset_detailed(self):
        """Test add_noise_to_dataset adds noise to features (lines 161-171)."""
        # Create sample dataset
        X = np.random.randn(100, 60, 5).astype(np.float32)
        y = np.random.randn(100, 9).astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(8)

        # Add noise with specific parameters (lines 161-171)
        noisy_dataset = add_noise_to_dataset(dataset, noise_std=0.05)

        # Get batch from original and noisy
        original_batch = next(iter(dataset))
        noisy_batch = next(iter(noisy_dataset))

        # Features should be different (noise added)
        assert not np.allclose(original_batch[0].numpy(), noisy_batch[0].numpy())

        # Targets should be same (noise only added to features)
        np.testing.assert_array_almost_equal(
            original_batch[1].numpy(),
            noisy_batch[1].numpy()
        )

    def test_add_noise_with_seed(self):
        """Test adding noise with seed for reproducibility (lines 161-171)."""
        X = np.random.randn(100, 60, 5).astype(np.float32)
        y = np.random.randn(100, 9).astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(8)

        # Add noise with seed
        noisy_dataset1 = add_noise_to_dataset(dataset, noise_std=0.05, seed=42)
        noisy_dataset2 = add_noise_to_dataset(dataset, noise_std=0.05, seed=42)

        batch1 = next(iter(noisy_dataset1))
        batch2 = next(iter(noisy_dataset2))

        # With same seed, noise should be same
        np.testing.assert_array_almost_equal(
            batch1[0].numpy(),
            batch2[0].numpy(),
            decimal=5
        )

    def test_add_noise_stddev_effect(self):
        """Test that noise_std affects magnitude of noise (lines 161-171)."""
        X = np.random.randn(50, 60, 5).astype(np.float32)
        y = np.random.randn(50, 9).astype(np.float32)

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.batch(8)

        # Get original
        original_batch = next(iter(dataset))

        # Add small noise
        small_noise_dataset = add_noise_to_dataset(dataset, noise_std=0.01)
        small_noise_batch = next(iter(small_noise_dataset))

        # Add large noise
        large_noise_dataset = add_noise_to_dataset(dataset, noise_std=0.5)
        large_noise_batch = next(iter(large_noise_dataset))

        # Calculate differences
        small_diff = np.abs(original_batch[0].numpy() - small_noise_batch[0].numpy()).mean()
        large_diff = np.abs(original_batch[0].numpy() - large_noise_batch[0].numpy()).mean()

        # Large noise should create larger differences
        assert large_diff > small_diff


class TestWindowGeneratorEdgeCases:
    """Test window_generator edge cases."""

    def test_window_generator_with_stride(self):
        """Test window_generator with different strides."""
        data = np.arange(100)

        # Stride of 1
        windows_1 = window_generator(data, window_size=10, stride=1)
        assert len(windows_1) == 91  # 100 - 10 + 1

        # Stride of 5
        windows_5 = window_generator(data, window_size=10, stride=5)
        assert len(windows_5) == 19  # (100 - 10) // 5 + 1

    def test_window_generator_2d_data(self):
        """Test window_generator with 2D data."""
        data = np.random.randn(100, 5)

        windows = window_generator(data, window_size=10, stride=1)

        assert windows.shape == (91, 10, 5)

    def test_window_generator_small_dataset(self):
        """Test window_generator with dataset smaller than window."""
        # Very small dataset
        data = np.array([1, 2, 3])

        # Window size larger than data
        windows = window_generator(data, window_size=5, stride=1)

        # Should return empty or no windows
        assert len(windows) == 0


class TestCreateTFDataset:
    """Test create_tf_dataset function."""

    def test_create_tf_dataset_basic(self):
        """Test basic tf.data.Dataset creation."""
        X = np.random.randn(100, 60, 5).astype(np.float32)
        y = np.random.randn(100, 9).astype(np.float32)

        dataset = create_tf_dataset(X, y, batch_size=16, shuffle=False)

        # Verify it's a Dataset
        assert isinstance(dataset, tf.data.Dataset)

        # Get a batch
        batch = next(iter(dataset))
        assert len(batch) == 2  # (features, targets)
        assert batch[0].shape[0] == 16  # batch size
        assert batch[0].shape[1:] == (60, 5)  # feature shape
        assert batch[1].shape == (16, 9)  # target shape

    def test_create_tf_dataset_with_shuffle(self):
        """Test dataset creation with shuffle."""
        X = np.random.randn(100, 60, 5).astype(np.float32)
        y = np.random.randn(100, 9).astype(np.float32)

        dataset = create_tf_dataset(X, y, batch_size=16, shuffle=True, buffer_size=50)

        # Should still be valid
        batch = next(iter(dataset))
        assert batch[0].shape[0] == 16

    def test_create_tf_dataset_prefetch(self):
        """Test dataset creation with prefetch."""
        X = np.random.randn(100, 60, 5).astype(np.float32)
        y = np.random.randn(100, 9).astype(np.float32)

        dataset = create_tf_dataset(X, y, batch_size=16, prefetch=True)

        # Should still work
        batch = next(iter(dataset))
        assert batch[0].shape[0] == 16


class TestDatasetIntegration:
    """Test dataset integration scenarios."""

    def test_dataset_with_augmentation_pipeline(self):
        """Test dataset with full augmentation pipeline."""
        X = np.random.randn(100, 60, 5).astype(np.float32)
        y = np.random.randn(100, 9).astype(np.float32)

        # Create dataset with all features
        dataset = create_tf_dataset(X, y, batch_size=8, shuffle=True)

        # Add noise
        dataset = add_noise_to_dataset(dataset, noise_std=0.05)

        # Should be iterable
        batches = []
        for batch in dataset.take(5):
            batches.append(batch)

        assert len(batches) == 5

    def test_dataset_pipeline_performance(self):
        """Test that dataset pipeline is efficient."""
        X = np.random.randn(1000, 60, 5).astype(np.float32)
        y = np.random.randn(1000, 9).astype(np.float32)

        # Create optimized dataset
        dataset = create_tf_dataset(
            X, y,
            batch_size=32,
            shuffle=True,
            buffer_size=100,
            prefetch=True
        )

        # Should be able to iterate quickly
        import time
        start = time.time()
        for i, batch in enumerate(dataset):
            if i >= 10:  # Just test 10 batches
                break
        elapsed = time.time() - start

        # Should complete in reasonable time (< 5 seconds for 10 batches)
        assert elapsed < 5.0

    def test_window_generator_and_dataset_integration(self):
        """Test window_generator integration with tf.data.Dataset."""
        # Generate time series data
        data = np.random.randn(200, 5).astype(np.float32)

        # Create windows
        windows = window_generator(data, window_size=60, stride=1)

        # Generate targets (simple example)
        targets = np.random.randn(len(windows), 9).astype(np.float32)

        # Create dataset
        dataset = create_tf_dataset(windows, targets, batch_size=16)

        # Verify pipeline works
        batch = next(iter(dataset))
        assert batch[0].shape == (16, 60, 5)
        assert batch[1].shape == (16, 9)
