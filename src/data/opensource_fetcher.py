"""Open-source data fetcher for cryptocurrency market data.

Fetches historical OHLCV data from free public sources:
- Yahoo Finance (via yfinance)
- Binance public API (no auth required for historical data)
- Pre-packaged sample datasets

No API keys or authentication required.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
from datetime import datetime, timedelta
import time
import requests


class OpenSourceDataFetcher:
    """Fetch cryptocurrency data from open-source/free APIs.

    Examples:
        >>> fetcher = OpenSourceDataFetcher()
        >>> data = fetcher.fetch_binance_spot('BTCUSDT', days=30)
        >>> fetcher.save_to_csv(data, 'data_raw/btcusdt.csv')
    """

    def __init__(self):
        """Initialize the data fetcher."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; neural-trading/1.0)'
        })

    def fetch_binance_spot(
        self,
        symbol: str = 'BTCUSDT',
        interval: str = '1m',
        days: int = 7,
        save_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """Fetch historical data from Binance public API (no auth required).

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: Candlestick interval ('1m', '5m', '15m', '1h', '1d')
            days: Number of days of historical data to fetch
            save_path: Optional path to save CSV

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume

        Examples:
            >>> fetcher = OpenSourceDataFetcher()
            >>> data = fetcher.fetch_binance_spot('BTCUSDT', interval='1m', days=7)
            >>> print(f"Fetched {len(data)} candles")
        """
        print(f"Fetching {symbol} {interval} data from Binance (last {days} days)...")

        # Binance API endpoint (public, no auth)
        base_url = "https://api.binance.com/api/v3/klines"

        # Calculate time range
        end_time = int(time.time() * 1000)  # Current time in milliseconds
        start_time = end_time - (days * 24 * 60 * 60 * 1000)  # days ago

        # Fetch data in chunks (Binance limits to 1000 candles per request)
        all_data = []
        current_start = start_time

        while current_start < end_time:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_time,
                'limit': 1000  # Max limit
            }

            try:
                response = self.session.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                klines = response.json()

                if not klines:
                    break

                all_data.extend(klines)

                # Update start time for next batch
                current_start = klines[-1][0] + 1

                # Rate limiting
                time.sleep(0.2)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching data: {e}")
                break

        if not all_data:
            raise ValueError(f"No data fetched for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Select and convert required columns
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()

        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)

        # Remove duplicates
        df = df.drop_duplicates(subset=['datetime'], keep='last')

        print(f"✓ Fetched {len(df)} candles from {df['datetime'].min()} to {df['datetime'].max()}")

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"✓ Saved to {save_path}")

        return df

    def fetch_sample_data(
        self,
        symbol: str = 'BTCUSDT',
        interval: str = '1m',
        num_candles: int = 10000,
        save_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """Generate synthetic sample data for testing (when API is unavailable).

        Args:
            symbol: Symbol name (for reference)
            interval: Interval name (for reference)
            num_candles: Number of candles to generate
            save_path: Optional path to save CSV

        Returns:
            DataFrame with synthetic OHLCV data
        """
        print(f"Generating {num_candles} synthetic {interval} candles for {symbol}...")

        # Generate timestamps
        end_time = datetime.utcnow()

        # Interval to minutes mapping
        interval_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        minutes = interval_minutes.get(interval, 1)

        timestamps = [
            end_time - timedelta(minutes=minutes * i)
            for i in range(num_candles - 1, -1, -1)
        ]

        # Generate realistic-looking price data (random walk)
        np.random.seed(42)
        base_price = 30000.0  # Starting BTC price
        returns = np.random.normal(0, 0.001, num_candles)  # Small random returns
        close_prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLCV
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, close_prices)):
            volatility = close * 0.002  # 0.2% volatility
            high = close + abs(np.random.normal(0, volatility))
            low = close - abs(np.random.normal(0, volatility))
            open_price = close + np.random.normal(0, volatility / 2)
            volume = np.random.uniform(100, 1000)

            data.append({
                'datetime': ts,
                'open': open_price,
                'high': max(open_price, high, close),
                'low': min(open_price, low, close),
                'close': close,
                'volume': volume
            })

        df = pd.DataFrame(data)

        print(f"✓ Generated {len(df)} synthetic candles")

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"✓ Saved to {save_path}")

        return df

    def fetch_yfinance_crypto(
        self,
        symbol: str = 'BTC-USD',
        interval: str = '1m',
        period: str = '7d',
        save_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """Fetch crypto data from Yahoo Finance (requires yfinance package).

        Args:
            symbol: Yahoo Finance symbol (e.g., 'BTC-USD', 'ETH-USD')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
            period: Time period ('1d', '5d', '1mo', '3mo', '1y', 'max')
            save_path: Optional path to save CSV

        Returns:
            DataFrame with OHLCV data

        Note:
            Requires yfinance: pip install yfinance
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance package required. Install with: pip install yfinance"
            )

        print(f"Fetching {symbol} {interval} data from Yahoo Finance (period: {period})...")

        # Fetch data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            raise ValueError(f"No data fetched for {symbol}")

        # Rename columns to standard format
        df = df.reset_index()
        df = df.rename(columns={
            'Datetime': 'datetime',
            'Date': 'datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Select required columns
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()

        # Ensure datetime is UTC
        if df['datetime'].dt.tz is None:
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        else:
            df['datetime'] = df['datetime'].dt.tz_convert('UTC')

        print(f"✓ Fetched {len(df)} candles from {df['datetime'].min()} to {df['datetime'].max()}")

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"✓ Saved to {save_path}")

        return df


def quick_fetch_data(
    source: str = 'binance',
    symbol: str = 'BTCUSDT',
    interval: str = '1m',
    days: int = 7,
    output_path: str = 'data_raw/crypto_data.csv'
) -> pd.DataFrame:
    """Quick function to fetch crypto data from various sources.

    Args:
        source: Data source ('binance', 'yfinance', 'synthetic')
        symbol: Trading symbol
        interval: Candlestick interval
        days: Number of days (or num_candles for synthetic)
        output_path: Path to save CSV

    Returns:
        DataFrame with OHLCV data

    Examples:
        >>> # Fetch from Binance (no auth required)
        >>> data = quick_fetch_data('binance', 'BTCUSDT', '1m', days=7)

        >>> # Generate synthetic data
        >>> data = quick_fetch_data('synthetic', num_candles=10000)
    """
    fetcher = OpenSourceDataFetcher()

    if source == 'binance':
        return fetcher.fetch_binance_spot(symbol, interval, days, output_path)
    elif source == 'yfinance':
        # Convert symbol format (BTCUSDT -> BTC-USD)
        if symbol == 'BTCUSDT':
            yf_symbol = 'BTC-USD'
        elif symbol == 'ETHUSDT':
            yf_symbol = 'ETH-USD'
        else:
            yf_symbol = symbol
        period = f"{days}d"
        return fetcher.fetch_yfinance_crypto(yf_symbol, interval, period, output_path)
    elif source == 'synthetic':
        return fetcher.fetch_sample_data(symbol, interval, days * 1440, output_path)  # days * minutes_per_day
    else:
        raise ValueError(f"Unknown source: {source}. Use 'binance', 'yfinance', or 'synthetic'")
