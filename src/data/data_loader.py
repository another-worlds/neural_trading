"""Data loading module for OHLCV market data.

Supports loading from:
- CSV files (binance_btcusdt_1min_ccxt.csv format)
- CCXT exchange APIs (Binance, etc.)

Includes validation, quality metrics, and caching functionality.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from datetime import datetime, timedelta
import time


class DataValidationError(Exception):
    """Exception raised when data validation fails."""
    pass


class DataLoader:
    """Load and validate OHLCV market data from various sources.

    Supports CSV files and CCXT exchange APIs. Includes comprehensive
    data validation and quality checks.

    Examples:
        >>> loader = DataLoader('data.csv')
        >>> data = loader.load()
        >>> is_valid, errors = loader.validate_columns(data)
    """

    def __init__(self, csv_path: Optional[Union[str, Path]] = None):
        """Initialize DataLoader.

        Args:
            csv_path: Path to CSV file. If None, loader can be used
                     for validation and saving operations.
        """
        self.csv_path = Path(csv_path) if csv_path else None

    def load(self) -> pd.DataFrame:
        """Load OHLCV data from CSV file.

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume.

        Raises:
            FileNotFoundError: If CSV file doesn't exist.
            ValueError: If CSV format is invalid.
        """
        if self.csv_path is None:
            raise ValueError("CSV path not specified")

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # Load CSV with datetime parsing
        data = pd.read_csv(self.csv_path)

        # Parse datetime column
        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'], utc=True)

        return data

    def validate_columns(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate that required columns are present.

        Args:
            data: DataFrame to validate.

        Returns:
            Tuple of (is_valid, error_messages).
        """
        errors = []
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

        for col in required_columns:
            if col not in data.columns:
                errors.append(f"Missing required column: {col}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def save_to_csv(self, data: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """Save OHLCV data to CSV file.

        Args:
            data: DataFrame to save.
            output_path: Path where CSV should be saved.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with proper datetime format
        data.to_csv(output_path, index=False)

    def append_to_csv(self, data: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """Append new data to existing CSV file.

        Args:
            data: DataFrame to append.
            output_path: Path to existing CSV file.
        """
        output_path = Path(output_path)

        if output_path.exists():
            # Load existing data
            existing = pd.read_csv(output_path)
            # Append new data
            combined = pd.concat([existing, data], ignore_index=True)
            combined.to_csv(output_path, index=False)
        else:
            # If file doesn't exist, create it
            self.save_to_csv(data, output_path)


def load_from_csv(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Convenience function to load data from CSV.

    Args:
        csv_path: Path to CSV file.

    Returns:
        DataFrame with OHLCV data.
    """
    loader = DataLoader(csv_path)
    return loader.load()


def fetch_from_ccxt(
    symbol: str = 'BTC/USDT',
    timeframe: str = '1m',
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    exchange: str = 'binance',
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> pd.DataFrame:
    """Fetch OHLCV data from CCXT exchange API.

    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT').
        timeframe: Candle timeframe (e.g., '1m', '5m', '1h').
        start_date: Start date for data fetch.
        end_date: End date for data fetch.
        exchange: Exchange name (default: 'binance').
        max_retries: Maximum number of retry attempts on error.
        retry_delay: Delay between retries in seconds.

    Returns:
        DataFrame with OHLCV data.

    Raises:
        ImportError: If ccxt is not installed.
        Exception: If fetch fails after max retries.
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError(
            "ccxt library is required for fetching exchange data. "
            "Install with: pip install ccxt"
        )

    # Initialize exchange
    exchange_class = getattr(ccxt, exchange)
    exchange_obj = exchange_class()

    # Convert dates to timestamps
    since = int(start_date.timestamp() * 1000) if start_date else None
    until = int(end_date.timestamp() * 1000) if end_date else None

    all_candles = []
    current_since = since

    # Fetch data with pagination and retry logic
    for attempt in range(max_retries):
        try:
            while True:
                # Fetch batch of candles
                candles = exchange_obj.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current_since,
                    limit=500  # CCXT typical limit
                )

                if not candles:
                    break

                all_candles.extend(candles)

                # Check if we've reached the end date
                last_timestamp = candles[-1][0]
                if until and last_timestamp >= until:
                    break

                # Update since for next batch
                current_since = last_timestamp + 1

                # Rate limiting
                time.sleep(exchange_obj.rateLimit / 1000)

            # Success - break retry loop
            break

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue
            else:
                raise Exception(f"Failed to fetch data after {max_retries} attempts: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(
        all_candles,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )

    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop('timestamp', axis=1)

    # Reorder columns
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    # Filter by end date if specified
    if until:
        df = df[df['datetime'] <= pd.to_datetime(until, unit='ms', utc=True)]

    return df


def validate_ohlcv_data(
    data: pd.DataFrame,
    timeframe: Optional[str] = None
) -> Tuple[bool, List[str]]:
    """Validate OHLCV data for quality issues.

    Checks for:
    - Missing values (NaN)
    - OHLC logic violations (high < low, etc.)
    - Negative prices or volume
    - Time gaps (if timeframe specified)

    Args:
        data: DataFrame with OHLCV data.
        timeframe: Expected timeframe (e.g., '1min') for gap detection.

    Returns:
        Tuple of (is_valid, error_messages).
    """
    errors = []

    # Check for missing values
    if data.isna().any().any():
        missing_cols = data.columns[data.isna().any()].tolist()
        errors.append(f"Missing values (NaN) in columns: {missing_cols}")

    # Check for OHLC logic violations
    # High should be >= low, open, close
    # Low should be <= open, close
    if not data.empty:
        ohlc_violations = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )

        if ohlc_violations.any():
            violation_count = ohlc_violations.sum()
            errors.append(
                f"OHLC logic violations detected ({violation_count} candles)"
            )

    # Check for negative prices
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in data.columns:
            if (data[col] < 0).any():
                errors.append(f"Negative prices detected in column: {col}")

    # Check for negative volume
    if 'volume' in data.columns:
        if (data['volume'] < 0).any():
            errors.append("Negative volume detected")

    # Check for time gaps
    if timeframe and 'datetime' in data.columns and len(data) > 1:
        # Convert timeframe to timedelta
        timeframe_map = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1),
        }

        expected_delta = timeframe_map.get(timeframe)
        if expected_delta:
            time_diffs = data['datetime'].diff()[1:]  # Skip first NaT
            expected_diff_seconds = expected_delta.total_seconds()

            # Allow small tolerance for timestamp variations
            tolerance = timedelta(seconds=1)
            gaps = time_diffs > (expected_delta + tolerance)

            if gaps.any():
                gap_count = gaps.sum()
                errors.append(f"Time gaps detected ({gap_count} gaps)")

    is_valid = len(errors) == 0
    return is_valid, errors


def calculate_quality_metrics(data: pd.DataFrame) -> Dict[str, int]:
    """Calculate data quality metrics.

    Args:
        data: DataFrame with OHLCV data.

    Returns:
        Dictionary with quality metrics:
        - total_candles: Total number of candles
        - missing_values: Number of missing values
        - ohlc_violations: Number of OHLC logic violations
        - time_gaps: Number of time gaps (if datetime available)
    """
    metrics = {
        'total_candles': len(data),
        'missing_values': 0,
        'ohlc_violations': 0,
        'time_gaps': 0,
    }

    # Count missing values
    metrics['missing_values'] = data.isna().sum().sum()

    # Count OHLC violations
    if not data.empty and all(col in data.columns for col in ['open', 'high', 'low', 'close']):
        ohlc_violations = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        metrics['ohlc_violations'] = int(ohlc_violations.sum())

    # Note: time_gaps calculation would require timeframe information
    # For now, leave as 0 unless specific timeframe validation is run

    return metrics
