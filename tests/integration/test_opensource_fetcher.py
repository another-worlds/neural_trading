"""Integration tests for open-source data fetcher.

These tests verify the data fetcher can successfully fetch real data.
Network-dependent tests are marked and can be skipped.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data.opensource_fetcher import (
    OpenSourceDataFetcher,
    quick_fetch_data
)


class TestOpenSourceDataFetcher:
    """Test OpenSourceDataFetcher class."""

    def test_fetcher_initialization(self):
        """Test fetcher can be initialized."""
        fetcher = OpenSourceDataFetcher()
        assert fetcher is not None
        assert hasattr(fetcher, 'session')

    @pytest.mark.slow
    @pytest.mark.network
    def test_fetch_binance_spot_real_data(self):
        """Test fetching real data from Binance API."""
        fetcher = OpenSourceDataFetcher()

        # Fetch 1 day of data (small amount for quick test)
        data = fetcher.fetch_binance_spot(
            symbol='BTCUSDT',
            interval='1h',  # Use hourly to reduce data size
            days=1
        )

        # Verify data structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

        # Verify required columns
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            assert col in data.columns

        # Verify data types
        assert pd.api.types.is_datetime64_any_dtype(data['datetime'])
        assert pd.api.types.is_numeric_dtype(data['open'])
        assert pd.api.types.is_numeric_dtype(data['close'])

        # Verify data sanity
        assert data['high'].min() > 0
        assert data['low'].min() > 0
        assert (data['high'] >= data['low']).all()
        assert (data['high'] >= data['open']).all()
        assert (data['high'] >= data['close']).all()

    def test_fetch_sample_data_generation(self):
        """Test synthetic data generation."""
        fetcher = OpenSourceDataFetcher()

        # Generate small synthetic dataset
        data = fetcher.fetch_sample_data(
            symbol='BTCUSDT',
            interval='1m',
            num_candles=1000
        )

        # Verify structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 1000

        # Verify columns
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            assert col in data.columns

        # Verify OHLC relationships
        assert (data['high'] >= data['low']).all()
        assert (data['high'] >= data['open']).all()
        assert (data['high'] >= data['close']).all()
        assert (data['low'] <= data['open']).all()
        assert (data['low'] <= data['close']).all()

        # Verify timestamps are sorted
        assert data['datetime'].is_monotonic_increasing

    def test_fetch_sample_data_with_save(self):
        """Test synthetic data generation with CSV save."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_data.csv'

            fetcher = OpenSourceDataFetcher()
            data = fetcher.fetch_sample_data(
                num_candles=500,
                save_path=output_path
            )

            # Verify file was created
            assert output_path.exists()

            # Verify can be loaded
            loaded_data = pd.read_csv(output_path)
            assert len(loaded_data) == 500
            assert 'datetime' in loaded_data.columns

    @pytest.mark.slow
    @pytest.mark.network
    def test_fetch_binance_with_save(self):
        """Test Binance fetch with CSV save."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'btc_data.csv'

            fetcher = OpenSourceDataFetcher()
            data = fetcher.fetch_binance_spot(
                symbol='BTCUSDT',
                interval='1h',
                days=1,
                save_path=output_path
            )

            # Verify file was created
            assert output_path.exists()

            # Verify can be loaded
            loaded_data = pd.read_csv(output_path)
            assert len(loaded_data) > 0


class TestQuickFetchData:
    """Test quick_fetch_data convenience function."""

    def test_quick_fetch_synthetic(self):
        """Test quick fetch with synthetic data."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'data.csv'

            data = quick_fetch_data(
                source='synthetic',
                symbol='BTCUSDT',
                interval='1m',
                days=1,
                output_path=str(output_path)
            )

            # Verify data
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0

            # Verify file created
            assert output_path.exists()

    @pytest.mark.slow
    @pytest.mark.network
    def test_quick_fetch_binance(self):
        """Test quick fetch from Binance."""
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'btc.csv'

            data = quick_fetch_data(
                source='binance',
                symbol='BTCUSDT',
                interval='1h',
                days=1,
                output_path=str(output_path)
            )

            # Verify data
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            assert 'datetime' in data.columns

            # Verify file created
            assert output_path.exists()

    def test_quick_fetch_invalid_source(self):
        """Test quick fetch with invalid source raises error."""
        with pytest.raises(ValueError) as exc_info:
            quick_fetch_data(source='invalid_source')

        assert 'Unknown source' in str(exc_info.value)


class TestDataQuality:
    """Test data quality and validation."""

    def test_synthetic_data_has_no_nulls(self):
        """Test that synthetic data has no null values."""
        fetcher = OpenSourceDataFetcher()
        data = fetcher.fetch_sample_data(num_candles=1000)

        # Check for nulls
        assert data.isnull().sum().sum() == 0

    def test_synthetic_data_has_no_duplicates(self):
        """Test that synthetic data has no duplicate timestamps."""
        fetcher = OpenSourceDataFetcher()
        data = fetcher.fetch_sample_data(num_candles=1000)

        # Check for duplicate timestamps
        assert data['datetime'].duplicated().sum() == 0

    def test_synthetic_data_price_continuity(self):
        """Test that synthetic data has reasonable price continuity."""
        fetcher = OpenSourceDataFetcher()
        data = fetcher.fetch_sample_data(num_candles=1000)

        # Calculate price changes
        price_changes = data['close'].pct_change().abs()

        # No single change should be > 10% (reasonable for 1-minute data)
        assert (price_changes[1:] < 0.1).all()

    @pytest.mark.slow
    @pytest.mark.network
    def test_binance_data_quality(self):
        """Test that Binance data meets quality standards."""
        fetcher = OpenSourceDataFetcher()

        data = fetcher.fetch_binance_spot(
            symbol='BTCUSDT',
            interval='1h',
            days=1
        )

        # Check for nulls
        null_count = data.isnull().sum().sum()
        assert null_count == 0, f"Found {null_count} null values"

        # Check for duplicates
        dup_count = data['datetime'].duplicated().sum()
        assert dup_count == 0, f"Found {dup_count} duplicate timestamps"

        # Verify data is sorted
        assert data['datetime'].is_monotonic_increasing


class TestDifferentIntervals:
    """Test fetching different time intervals."""

    @pytest.mark.parametrize('interval', ['1m', '5m', '15m', '1h'])
    def test_synthetic_data_different_intervals(self, interval):
        """Test synthetic data generation for different intervals."""
        fetcher = OpenSourceDataFetcher()

        data = fetcher.fetch_sample_data(
            interval=interval,
            num_candles=100
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100

    @pytest.mark.slow
    @pytest.mark.network
    @pytest.mark.parametrize('interval', ['1h', '4h'])
    def test_binance_different_intervals(self, interval):
        """Test Binance fetch for different intervals."""
        fetcher = OpenSourceDataFetcher()

        data = fetcher.fetch_binance_spot(
            symbol='BTCUSDT',
            interval=interval,
            days=1
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.network
    def test_binance_invalid_symbol(self):
        """Test handling of invalid symbol."""
        fetcher = OpenSourceDataFetcher()

        # Invalid symbol should raise error or return empty
        with pytest.raises((ValueError, Exception)):
            fetcher.fetch_binance_spot(
                symbol='INVALID_SYMBOL',
                interval='1h',
                days=1
            )

    def test_synthetic_zero_candles(self):
        """Test synthetic data with zero candles."""
        fetcher = OpenSourceDataFetcher()

        with pytest.raises((ValueError, Exception)):
            fetcher.fetch_sample_data(num_candles=0)

    def test_synthetic_negative_candles(self):
        """Test synthetic data with negative candles."""
        fetcher = OpenSourceDataFetcher()

        # Should handle gracefully or raise error
        try:
            data = fetcher.fetch_sample_data(num_candles=-100)
            # If it doesn't raise, should return empty or handle gracefully
            assert len(data) == 0
        except (ValueError, Exception):
            # Raising an error is also acceptable
            pass
