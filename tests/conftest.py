"""Pytest configuration and shared fixtures."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    np.random.seed(42)

    # Generate realistic price data
    close_prices = 42000 + np.cumsum(np.random.randn(100) * 10)

    data = pd.DataFrame({
        'datetime': dates,
        'open': close_prices + np.random.randn(100) * 5,
        'high': close_prices + np.abs(np.random.randn(100) * 10),
        'low': close_prices - np.abs(np.random.randn(100) * 10),
        'close': close_prices,
        'volume': np.abs(np.random.randn(100) * 100)
    })

    # Ensure OHLC logic is valid
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data


@pytest.fixture
def sample_ohlcv_array():
    """Generate sample OHLCV numpy array for testing."""
    np.random.seed(42)
    return np.random.randn(60, 5).astype(np.float32)


@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        'data': {
            'csv_path': 'binance_btcusdt_1min_ccxt.csv',
            'lookback': 60,
            'window_step': 1,
            'sequence_limit': 2880,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15,
        },
        'indicators': {
            'ma_periods': [5, 15, 30],
            'rsi_periods': [9, 21, 30],
            'bb_periods': [10, 20, 30],
            'macd_settings': [[12, 26, 9], [5, 35, 5], [19, 39, 9]],
            'momentum_periods': [5, 10, 15],
        },
        'model': {
            'lstm_units': 128,
            'lstm_layers': 2,
            'transformer_heads': 4,
            'transformer_dim': 128,
            'dropout': 0.2,
            'l2_reg': 0.001,
        },
        'training': {
            'batch_size': 144,
            'epochs': 40,
            'learning_rate': 0.001,
            'patience': 40,
            'gradient_clip_norm': 5.0,
        },
        'losses': {
            'point_loss': {'type': 'huber', 'weight': 1.0},
            'direction_loss': {'type': 'focal', 'weight': 1.0, 'alpha': 0.7, 'gamma': 1.0},
            'variance_loss': {'type': 'nll', 'weight': 1.0},
            'trend_loss': {'type': 'mse', 'weight': 0.5},
        },
        'metrics': {
            'direction_accuracy': True,
            'direction_f1': True,
            'direction_mcc': True,
            'price_mae': True,
            'price_mape': True,
        },
    }


@pytest.fixture
def sample_predictions():
    """Sample model predictions for testing."""
    return {
        'price_h0': 42150.5,
        'price_h1': 42200.3,
        'price_h2': 42350.1,
        'direction_h0': 0.85,
        'direction_h1': 0.78,
        'direction_h2': 0.65,
        'variance_h0': 0.0023,
        'variance_h1': 0.0045,
        'variance_h2': 0.0098,
    }


@pytest.fixture
def sample_multi_horizon_predictions():
    """Sample multi-horizon price predictions."""
    return np.array([42150.5, 42200.3, 42350.1])


@pytest.fixture
def sample_trade():
    """Sample trade dictionary for testing."""
    return {
        'entry_time': datetime(2024, 1, 1, 12, 0),
        'entry_price': 42000.0,
        'position_type': 'LONG',
        'size': 1.0,
        'stop_loss': 41580.0,
        'take_profit_1': 42100.0,
        'take_profit_2': 42200.0,
        'take_profit_3': 42300.0,
        'confidence': 0.85,
        'variance': 0.0023,
        'status': 'OPEN',
    }


@pytest.fixture
def tmp_config_file(tmp_path, sample_config):
    """Create a temporary config file."""
    import yaml
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f)
    return config_file


@pytest.fixture
def tmp_csv_file(tmp_path, sample_ohlcv_data):
    """Create a temporary CSV file with OHLCV data."""
    csv_file = tmp_path / "test_data.csv"
    sample_ohlcv_data.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def mock_model_weights(tmp_path):
    """Create mock model weight files."""
    weights_file = tmp_path / "model.weights.h5"
    weights_file.touch()
    return weights_file


@pytest.fixture
def mock_scaler(tmp_path):
    """Create mock scaler files."""
    import joblib
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.mean_ = np.array([0.0])
    scaler.scale_ = np.array([1.0])

    scaler_file = tmp_path / "scaler.joblib"
    joblib.dump(scaler, scaler_file)
    return scaler_file


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test."""
    np.random.seed(42)
    import random
    random.seed(42)
    try:
        import tensorflow as tf
        tf.random.set_seed(42)
    except ImportError:
        pass
