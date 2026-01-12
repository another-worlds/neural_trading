"""TensorFlow dataset creation for neural trading pipeline.

Provides tf.data.Dataset creation with:
- Batching and shuffling
- Prefetching for performance
- Windowing for time series
- Train/val/test splitting
- Data augmentation
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Tuple, Optional, Union
from src.data.preprocessor import Preprocessor, split_data, generate_multi_horizon_targets


def create_tf_dataset(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = False,
    buffer_size: int = 1000,
    prefetch: bool = True,
    cache: bool = False,
    num_parallel_calls: Optional[int] = None
) -> tf.data.Dataset:
    """Create TensorFlow dataset from numpy arrays.

    Args:
        features: Feature array.
        targets: Target array.
        batch_size: Batch size for training.
        shuffle: Whether to shuffle the dataset.
        buffer_size: Buffer size for shuffling.
        prefetch: Whether to prefetch data for performance.
        cache: Whether to cache dataset in memory.
        num_parallel_calls: Number of parallel calls for transformations.

    Returns:
        tf.data.Dataset ready for training.

    Examples:
        >>> features = np.random.randn(1000, 60, 5)
        >>> targets = np.random.randn(1000, 3)
        >>> dataset = create_tf_dataset(features, targets, batch_size=144)
    """
    # Create dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((features, targets))

    # Apply caching if requested
    if cache:
        dataset = dataset.cache()

    # Apply shuffling if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Apply prefetching if requested
    if prefetch:
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def window_generator(
    data: np.ndarray,
    window_size: int = 60,
    stride: int = 1
) -> np.ndarray:
    """Generate sliding windows from time series data.

    Args:
        data: Input time series data (1D or 2D array).
        window_size: Size of each window.
        stride: Step size between windows.

    Returns:
        Array of windows with shape (n_windows, window_size) or
        (n_windows, window_size, n_features).

    Examples:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> windows = window_generator(data, window_size=3, stride=1)
        >>> windows.shape
        (3, 3)
    """
    # Handle 1D arrays
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_samples = len(data)
    windows = []

    for i in range(0, n_samples - window_size + 1, stride):
        window = data[i:i + window_size]
        windows.append(window)

    windows = np.array(windows)

    # If original data was 1D, squeeze the last dimension
    if windows.shape[-1] == 1 and len(data.shape) == 1:
        windows = windows.squeeze(-1)

    return windows


def add_gaussian_noise(
    data: np.ndarray,
    noise_std: float = 0.01,
    seed: Optional[int] = None
) -> np.ndarray:
    """Add Gaussian noise to data for augmentation.

    Args:
        data: Input data.
        noise_std: Standard deviation of noise (absolute value, not relative).
        seed: Random seed for reproducibility.

    Returns:
        Noisy data.

    Examples:
        >>> data = np.random.randn(100, 10)
        >>> noisy = add_gaussian_noise(data, noise_std=0.01)
    """
    if seed is not None:
        np.random.seed(seed)

    # Use absolute noise_std, not relative to data std
    # This ensures noise is added even for constant data
    noise = np.random.normal(0, noise_std, data.shape)
    noisy_data = data + noise

    return noisy_data.astype(data.dtype)


def add_noise_to_dataset(
    dataset: tf.data.Dataset,
    noise_std: float = 0.01,
    seed: Optional[int] = None
) -> tf.data.Dataset:
    """Add Gaussian noise to dataset features.

    Args:
        dataset: Input TensorFlow dataset.
        noise_std: Standard deviation of noise.
        seed: Random seed for reproducibility.

    Returns:
        Dataset with noise added to features.

    Examples:
        >>> dataset = create_tf_dataset(features, targets, batch_size=32)
        >>> noisy_dataset = add_noise_to_dataset(dataset, noise_std=0.01)
    """
    def add_noise_fn(features, labels):
        """Add noise to features."""
        if seed is not None:
            tf.random.set_seed(seed)

        noise = tf.random.normal(
            shape=tf.shape(features),
            mean=0.0,
            stddev=noise_std,
            dtype=features.dtype
        )
        noisy_features = features + noise
        return noisy_features, labels

    return dataset.map(add_noise_fn, num_parallel_calls=tf.data.AUTOTUNE)


def get_dataset(
    data: pd.DataFrame,
    config: Dict,
    mode: str = 'train'
) -> tf.data.Dataset:
    """Create TensorFlow dataset from configuration.

    Args:
        data: Input OHLCV DataFrame.
        config: Configuration dictionary with data and training settings.
        mode: Dataset mode ('train', 'val', or 'test').

    Returns:
        tf.data.Dataset ready for training.

    Examples:
        >>> config = {
        ...     'data': {'lookback': 60, 'window_step': 1},
        ...     'training': {'batch_size': 144}
        ... }
        >>> dataset = get_dataset(data, config, mode='train')
    """
    # Extract config
    data_config = config.get('data', {})
    training_config = config.get('training', {})

    lookback = data_config.get('lookback', 60)
    window_step = data_config.get('window_step', 1)
    batch_size = training_config.get('batch_size', 144)

    # Create preprocessor
    preprocessor = Preprocessor(data_config)

    # Extract features (OHLCV)
    feature_cols = ['open', 'high', 'low', 'close', 'volume']
    features_df = data[feature_cols]

    # Generate targets (for now, use h0 - 1 minute ahead close price)
    targets_dict = generate_multi_horizon_targets(data)
    targets = targets_dict['h0']  # Use h0 for simplicity

    # Create windows
    windows = preprocessor.create_windows(features_df)

    # Align targets with windows
    # Windows end at index i+lookback-1, so target for window i is targets[i+lookback-1]
    aligned_targets = []
    for i in range(len(windows)):
        target_idx = i + lookback - 1
        if target_idx < len(targets):
            aligned_targets.append(targets[target_idx])

    aligned_targets = np.array(aligned_targets)

    # Ensure we have matching lengths
    min_len = min(len(windows), len(aligned_targets))
    windows = windows[:min_len]
    aligned_targets = aligned_targets[:min_len]

    # Create TensorFlow dataset
    shuffle = (mode == 'train')
    dataset = create_tf_dataset(
        features=windows.astype(np.float32),
        targets=aligned_targets.reshape(-1, 1).astype(np.float32),
        batch_size=batch_size,
        shuffle=shuffle,
        prefetch=True,
        cache=(mode != 'train')  # Cache validation and test sets
    )

    return dataset


def get_train_val_test_datasets(
    data: pd.DataFrame,
    config: Dict
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create train, validation, and test datasets.

    Args:
        data: Input OHLCV DataFrame.
        config: Configuration dictionary.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).

    Examples:
        >>> train_ds, val_ds, test_ds = get_train_val_test_datasets(data, config)
    """
    # Get split ratios from config
    data_config = config.get('data', {})
    train_ratio = data_config.get('train_split', 0.7)
    val_ratio = data_config.get('val_split', 0.15)
    test_ratio = data_config.get('test_split', 0.15)

    # Split data maintaining temporal order
    train_data, val_data, test_data = split_data(
        data,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )

    # Create datasets for each split
    train_dataset = get_dataset(train_data, config, mode='train')
    val_dataset = get_dataset(val_data, config, mode='val')
    test_dataset = get_dataset(test_data, config, mode='test')

    return train_dataset, val_dataset, test_dataset
