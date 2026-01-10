"""Unit tests for data loading module.

Tests data loading from CCXT, CSV files, and data validation as specified in SRS Section 3.1.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.data_loader import (
    DataLoader,
    load_from_csv,
    fetch_from_ccxt,
    validate_ohlcv_data,
    DataValidationError,
)


class TestDataLoader:
    """Test DataLoader class for loading market data."""

    def test_load_from_csv(self, tmp_csv_file):
        """Should load OHLCV data from CSV file."""
        loader = DataLoader(tmp_csv_file)
        data = loader.load()

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_load_nonexistent_csv_raises_error(self):
        """Should raise error for nonexistent CSV file."""
        loader = DataLoader('nonexistent.csv')
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_csv_datetime_parsing(self, tmp_csv_file):
        """Should parse datetime column correctly."""
        loader = DataLoader(tmp_csv_file)
        data = loader.load()

        assert 'datetime' in data.columns
        assert pd.api.types.is_datetime64_any_dtype(data['datetime'])

    def test_csv_with_timezone(self, tmp_path):
        """Should handle timezone-aware datetime."""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='1min', tz='UTC')
        data = pd.DataFrame({
            'datetime': dates,
            'open': np.random.randn(10) + 42000,
            'high': np.random.randn(10) + 42010,
            'low': np.random.randn(10) + 41990,
            'close': np.random.randn(10) + 42000,
            'volume': np.abs(np.random.randn(10))
        })
        csv_file = tmp_path / "tz_data.csv"
        data.to_csv(csv_file, index=False)

        loader = DataLoader(csv_file)
        loaded_data = loader.load()

        assert loaded_data['datetime'].dt.tz is not None

    def test_validate_columns(self, sample_ohlcv_data):
        """Should validate required columns are present."""
        loader = DataLoader()
        is_valid, errors = loader.validate_columns(sample_ohlcv_data)

        assert is_valid is True
        assert len(errors) == 0

    def test_missing_columns_detected(self):
        """Should detect missing required columns."""
        incomplete_data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'close': np.random.randn(10) + 42000,
            # Missing open, high, low, volume
        })

        loader = DataLoader()
        is_valid, errors = loader.validate_columns(incomplete_data)

        assert is_valid is False
        assert len(errors) > 0


class TestFetchFromCCXT:
    """Test fetching data from CCXT (Binance)."""

    @pytest.mark.api
    @pytest.mark.slow
    def test_fetch_small_date_range(self):
        """Should fetch small date range successfully."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 1, 1, 0)  # 1 hour

        data = fetch_from_ccxt(
            symbol='BTC/USDT',
            timeframe='1m',
            start_date=start_date,
            end_date=end_date,
            exchange='binance'
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 60  # 60 minutes
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    @pytest.mark.api
    def test_fetch_with_rate_limiting(self):
        """Should respect rate limiting."""
        # Test that rate limiting is applied
        # This would need mocking or actual API calls
        pass

    @pytest.mark.api
    def test_fetch_with_pagination(self):
        """Should handle pagination for large date ranges."""
        # Test pagination logic
        # Fetch > 500 candles to test pagination
        pass

    def test_fetch_with_retry_on_network_error(self, mocker):
        """Should retry on network errors."""
        # Mock CCXT to raise NetworkError
        mock_exchange = mocker.Mock()
        mock_exchange.fetch_ohlcv.side_effect = [
            Exception("NetworkError"),
            Exception("NetworkError"),
            [[1609459200000, 42000, 42100, 41900, 42050, 1.5]]
        ]

        # Test retry logic
        # Should succeed after retries

    def test_fetch_with_retry_on_exchange_error(self, mocker):
        """Should retry on exchange errors."""
        # Mock CCXT to raise ExchangeError
        pass

    def test_fetch_max_retries_exceeded(self, mocker):
        """Should raise error after max retries exceeded."""
        # Mock to always fail
        # Should raise error after max attempts
        pass


class TestValidateOHLCVData:
    """Test OHLCV data validation functions."""

    def test_validate_valid_data(self, sample_ohlcv_data):
        """Should validate correct OHLCV data."""
        is_valid, errors = validate_ohlcv_data(sample_ohlcv_data)

        assert is_valid is True
        assert len(errors) == 0

    def test_detect_missing_values(self):
        """Should detect missing values (NaN)."""
        data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'open': [42000, np.nan, 42050, 42070, 42100],
            'high': [42100, 42150, np.nan, 42200, 42250],
            'low': [41900, 41950, 42000, np.nan, 42050],
            'close': [42050, 42100, 42150, 42200, np.nan],
            'volume': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        is_valid, errors = validate_ohlcv_data(data)
        assert is_valid is False
        assert any('missing' in err.lower() or 'nan' in err.lower() for err in errors)

    def test_detect_ohlc_logic_violations(self):
        """Should detect OHLC logic violations (high < low, etc.)."""
        data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=3, freq='1min'),
            'open': [42000, 42050, 42100],
            'high': [42100, 42000, 42150],  # [1] high < open - violation
            'low': [41900, 42100, 42000],   # [1] low > high - violation
            'close': [42050, 42070, 42120],
            'volume': [1.0, 2.0, 3.0]
        })

        is_valid, errors = validate_ohlcv_data(data)
        assert is_valid is False
        assert any('ohlc' in err.lower() or 'logic' in err.lower() for err in errors)

    def test_detect_negative_prices(self):
        """Should detect negative prices."""
        data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=3, freq='1min'),
            'open': [42000, -42050, 42100],  # Negative price
            'high': [42100, 42150, 42200],
            'low': [41900, 41950, 42000],
            'close': [42050, 42100, 42150],
            'volume': [1.0, 2.0, 3.0]
        })

        is_valid, errors = validate_ohlcv_data(data)
        assert is_valid is False
        assert any('negative' in err.lower() or 'price' in err.lower() for err in errors)

    def test_detect_negative_volume(self):
        """Should detect negative volume."""
        data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=3, freq='1min'),
            'open': [42000, 42050, 42100],
            'high': [42100, 42150, 42200],
            'low': [41900, 41950, 42000],
            'close': [42050, 42100, 42150],
            'volume': [1.0, -2.0, 3.0]  # Negative volume
        })

        is_valid, errors = validate_ohlcv_data(data)
        assert is_valid is False
        assert any('volume' in err.lower() for err in errors)

    def test_detect_time_gaps(self):
        """Should detect missing time periods (gaps)."""
        dates = [
            datetime(2024, 1, 1, 12, 0),
            datetime(2024, 1, 1, 12, 1),
            # Gap: missing 12:02
            datetime(2024, 1, 1, 12, 3),
            datetime(2024, 1, 1, 12, 4),
        ]
        data = pd.DataFrame({
            'datetime': dates,
            'open': [42000, 42050, 42100, 42150],
            'high': [42100, 42150, 42200, 42250],
            'low': [41900, 41950, 42000, 42050],
            'close': [42050, 42100, 42150, 42200],
            'volume': [1.0, 2.0, 3.0, 4.0]
        })

        is_valid, errors = validate_ohlcv_data(data, timeframe='1min')
        assert is_valid is False
        assert any('gap' in err.lower() for err in errors)

    def test_calculate_data_quality_metrics(self, sample_ohlcv_data):
        """Should calculate data quality metrics."""
        from src.data.data_loader import calculate_quality_metrics

        metrics = calculate_quality_metrics(sample_ohlcv_data)

        assert 'total_candles' in metrics
        assert 'missing_values' in metrics
        assert 'ohlc_violations' in metrics
        assert 'time_gaps' in metrics
        assert metrics['total_candles'] == len(sample_ohlcv_data)

    def test_quality_metrics_with_issues(self):
        """Should count quality issues correctly."""
        from src.data.data_loader import calculate_quality_metrics

        data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'open': [42000, np.nan, 42050, 42070, 42100],  # 1 missing
            'high': [42100, 42150, 42000, 42200, 42250],   # 1 OHLC violation
            'low': [41900, 41950, 42100, 42050, 42050],
            'close': [42050, 42100, 42150, 42200, 42250],
            'volume': [1.0, 2.0, 3.0, 4.0, 5.0]
        })

        metrics = calculate_quality_metrics(data)
        assert metrics['missing_values'] >= 1
        assert metrics['ohlc_violations'] >= 1


class TestDataCaching:
    """Test data caching functionality."""

    def test_save_to_csv(self, tmp_path, sample_ohlcv_data):
        """Should save data to CSV file."""
        output_file = tmp_path / "cached_data.csv"
        loader = DataLoader()
        loader.save_to_csv(sample_ohlcv_data, output_file)

        assert output_file.exists()

        # Verify saved data
        loaded = pd.read_csv(output_file)
        assert len(loaded) == len(sample_ohlcv_data)

    def test_csv_format_matches_spec(self, tmp_path, sample_ohlcv_data):
        """CSV format should match SRS specification."""
        output_file = tmp_path / "cached_data.csv"
        loader = DataLoader()
        loader.save_to_csv(sample_ohlcv_data, output_file)

        # Check format
        loaded = pd.read_csv(output_file)
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        assert all(col in loaded.columns for col in required_columns)

    def test_append_to_existing_csv(self, tmp_path, sample_ohlcv_data):
        """Should append new data to existing CSV."""
        output_file = tmp_path / "cached_data.csv"

        # Save initial data
        initial_data = sample_ohlcv_data.iloc[:50]
        loader = DataLoader()
        loader.save_to_csv(initial_data, output_file)

        # Append new data
        new_data = sample_ohlcv_data.iloc[50:]
        loader.append_to_csv(new_data, output_file)

        # Verify
        loaded = pd.read_csv(output_file)
        assert len(loaded) == len(sample_ohlcv_data)
