"""Data preprocessing module for feature engineering and scaling.

Provides:
- Sliding window creation (60-minute lookback)
- Feature scaling with StandardScaler
- Multi-horizon target generation (h0=1min, h1=5min, h2=15min)
- Direction and price target calculation
- Feature engineering (returns, volatility)
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler


class StandardScalerWrapper:
    """Wrapper around sklearn's StandardScaler for consistent interface."""

    def __init__(self):
        """Initialize wrapper with StandardScaler."""
        self.scaler = StandardScaler()
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray) -> 'StandardScalerWrapper':
        """Fit scaler on data.

        Args:
            X: Input data to fit.

        Returns:
            Self for method chaining.
        """
        self.scaler.fit(X)
        self.mean_ = self.scaler.mean_
        self.scale_ = self.scaler.scale_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler.

        Args:
            X: Input data to transform.

        Returns:
            Scaled data.
        """
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data.

        Args:
            X: Input data.

        Returns:
            Scaled data.
        """
        return self.scaler.fit_transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data.

        Args:
            X: Scaled data.

        Returns:
            Original scale data.
        """
        return self.scaler.inverse_transform(X)


class Preprocessor:
    """Preprocess OHLCV data for neural network training.

    Handles windowing, scaling, and target generation according to SRS
    specifications (60-minute lookback, multi-horizon predictions).

    Examples:
        >>> config = {'lookback': 60, 'window_step': 1}
        >>> preprocessor = Preprocessor(config)
        >>> windows = preprocessor.create_windows(data)
        >>> preprocessor.fit_scaler(features)
        >>> scaled = preprocessor.transform(features)
    """

    def __init__(self, config: Dict):
        """Initialize preprocessor with configuration.

        Args:
            config: Configuration dictionary with keys:
                - lookback: Number of timesteps in window (default: 60)
                - window_step: Step size for sliding window (default: 1)
                - sequence_limit: Maximum number of sequences (default: 2880)
        """
        self.lookback = config.get('lookback', 60)
        self.window_step = config.get('window_step', 1)
        self.sequence_limit = config.get('sequence_limit', 2880)

        self.scaler = None
        self.input_scaler = None
        self.output_scaler = None

    def create_windows(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Create sliding windows from time series data.

        Args:
            data: Input time series data.

        Returns:
            Array of shape (n_windows, lookback, n_features).
        """
        if isinstance(data, pd.DataFrame):
            # Convert to numpy array, excluding datetime column if present
            if 'datetime' in data.columns:
                data = data.drop('datetime', axis=1)
            data = data.values

        return create_windows(
            data,
            lookback=self.lookback,
            step=self.window_step,
            limit=self.sequence_limit
        )

    def fit_scaler(self, features: np.ndarray) -> None:
        """Fit scaler on training features.

        Args:
            features: Training features to fit scaler on.
        """
        self.scaler = StandardScalerWrapper()
        self.scaler.fit(features)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler.

        Args:
            features: Features to transform.

        Returns:
            Scaled features.

        Raises:
            ValueError: If scaler not fitted.
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        return self.scaler.transform(features)

    def inverse_transform(self, scaled_features: np.ndarray) -> np.ndarray:
        """Inverse transform scaled features.

        Args:
            scaled_features: Scaled features.

        Returns:
            Original scale features.

        Raises:
            ValueError: If scaler not fitted.
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        return self.scaler.inverse_transform(scaled_features)

    def fit_input_scaler(self, features: np.ndarray) -> None:
        """Fit separate scaler for input features.

        Args:
            features: Input features to fit scaler on.
        """
        self.input_scaler = StandardScalerWrapper()
        self.input_scaler.fit(features)

    def fit_output_scaler(self, targets: np.ndarray) -> None:
        """Fit separate scaler for output targets.

        Args:
            targets: Output targets to fit scaler on.
        """
        self.output_scaler = StandardScalerWrapper()
        self.output_scaler.fit(targets)

    def save_scaler(self, filepath: Union[str, Path]) -> None:
        """Save scaler to file.

        Args:
            filepath: Path where scaler should be saved.

        Raises:
            ValueError: If scaler not fitted.
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Cannot save.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, filepath)

    def load_scaler(self, filepath: Union[str, Path]) -> None:
        """Load scaler from file.

        Args:
            filepath: Path to saved scaler file.

        Raises:
            FileNotFoundError: If scaler file doesn't exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Scaler file not found: {filepath}")

        self.scaler = joblib.load(filepath)


def create_windows(
    data: np.ndarray,
    lookback: int = 60,
    step: int = 1,
    limit: Optional[int] = None
) -> np.ndarray:
    """Create sliding windows from time series data.

    Args:
        data: Input array of shape (n_timesteps, n_features).
        lookback: Number of timesteps in each window.
        step: Step size for sliding window.
        limit: Maximum number of windows to create (default: None = no limit).

    Returns:
        Array of shape (n_windows, lookback, n_features).
    """
    if len(data) < lookback:
        return np.array([])

    windows = []
    for i in range(0, len(data) - lookback + 1, step):
        window = data[i:i + lookback]
        windows.append(window)

        if limit and len(windows) >= limit:
            break

    return np.array(windows)


def scale_features(features: np.ndarray, scaler=None) -> Tuple[np.ndarray, StandardScaler]:
    """Scale features using StandardScaler.

    Args:
        features: Features to scale.
        scaler: Optional pre-fitted scaler. If None, fits new scaler.

    Returns:
        Tuple of (scaled_features, scaler).
    """
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
    else:
        scaled = scaler.transform(features)

    return scaled, scaler


def generate_targets(
    data: pd.DataFrame,
    horizon: int = 0,
    target_type: str = 'price'
) -> np.ndarray:
    """Generate prediction targets for given horizon.

    Args:
        data: DataFrame with 'close' column.
        horizon: Prediction horizon:
                 0 = h0 (1 minute ahead)
                 1 = h1 (5 minutes ahead)
                 2 = h2 (15 minutes ahead)
        target_type: Type of target:
                     'price' = future price
                     'direction' = binary direction (0=down, 1=up)

    Returns:
        Array of target values.
    """
    horizon_map = {
        0: 1,   # h0 = 1 minute ahead
        1: 5,   # h1 = 5 minutes ahead
        2: 15,  # h2 = 15 minutes ahead
    }

    steps_ahead = horizon_map.get(horizon, 1)
    close_prices = data['close'].values

    if target_type == 'price':
        # Return future prices
        targets = close_prices[steps_ahead:]
    elif target_type == 'direction':
        # Return direction (1 if price goes up, 0 if down)
        current_prices = close_prices[:-steps_ahead]
        future_prices = close_prices[steps_ahead:]
        targets = (future_prices > current_prices).astype(int)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return targets


def generate_multi_horizon_targets(data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Generate targets for all three horizons.

    Args:
        data: DataFrame with 'close' column.

    Returns:
        Dictionary with keys 'h0', 'h1', 'h2' containing target arrays.
    """
    targets = {}
    for horizon in [0, 1, 2]:
        horizon_name = f'h{horizon}'
        targets[horizon_name] = generate_targets(data, horizon=horizon, target_type='price')

    return targets


def calculate_returns(prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Calculate simple returns.

    Args:
        prices: Price series.

    Returns:
        Array of returns (length = len(prices) - 1).
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    returns = np.diff(prices) / prices[:-1]
    return returns


def calculate_log_returns(prices: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Calculate log returns.

    Args:
        prices: Price series.

    Returns:
        Array of log returns (length = len(prices) - 1).
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    log_returns = np.diff(np.log(prices))
    return log_returns


def calculate_volatility(
    prices: Union[pd.Series, np.ndarray],
    window: int = 10
) -> np.ndarray:
    """Calculate rolling volatility (standard deviation of returns).

    Args:
        prices: Price series.
        window: Rolling window size.

    Returns:
        Array of volatility values.
    """
    if isinstance(prices, np.ndarray):
        prices = pd.Series(prices)

    returns = prices.pct_change()
    volatility = returns.rolling(window=window).std()

    return volatility.values


def add_extended_trend_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add extended trend features to data.

    Adds percent change features for 1m, 5m, 15m intervals as per SRS.

    Args:
        data: DataFrame with OHLCV data.

    Returns:
        DataFrame with additional trend feature columns.
    """
    df = data.copy()

    if 'close' in df.columns:
        # Add percent change features for different intervals
        df['pct_change_1m'] = df['close'].pct_change(periods=1)
        df['pct_change_5m'] = df['close'].pct_change(periods=5)
        df['pct_change_15m'] = df['close'].pct_change(periods=15)

        # Add simple moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_15'] = df['close'].rolling(window=15).mean()
        df['sma_30'] = df['close'].rolling(window=30).mean()

        # Add trend indicators
        df['trend_short'] = (df['sma_5'] > df['sma_15']).astype(int)
        df['trend_long'] = (df['sma_15'] > df['sma_30']).astype(int)

    return df


def add_volume_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based statistical features.

    Args:
        data: DataFrame with volume column.

    Returns:
        DataFrame with additional volume feature columns.
    """
    df = data.copy()

    if 'volume' in df.columns:
        # Rolling volume statistics
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_std_10'] = df['volume'].rolling(window=10).std()

        # Volume relative to moving average
        df['volume_ratio'] = df['volume'] / (df['volume_sma_10'] + 1e-8)

    return df


def split_data(
    data: Union[np.ndarray, pd.DataFrame],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple:
    """Split data into train/val/test sets maintaining temporal order.

    Args:
        data: Input data to split.
        train_ratio: Ratio for training set.
        val_ratio: Ratio for validation set.
        test_ratio: Ratio for test set.

    Returns:
        Tuple of (train, val, test) splits.
    """
    # Verify ratios sum to 1.0 (with small tolerance)
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    n_samples = len(data)

    # Calculate split indices
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    # Split data maintaining temporal order
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    return train, val, test


def visualize_split(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray
) -> None:
    """Visualize data splits (placeholder - full visualization in Phase 8).

    Args:
        train: Training data.
        val: Validation data.
        test: Test data.
    """
    # Placeholder implementation for tests
    # Full visualization implementation would use matplotlib
    print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")
