#!/usr/bin/env python
"""Quick launcher for the neural trading pipeline.

This script allows you to immediately start using the pipeline with:
1. Automatic data fetching from open sources
2. Data preprocessing
3. Model training (optional)
4. Inference and signal generation

Usage:
    # Fetch data and prepare for training
    python launch_pipeline.py --fetch-data --days 7

    # Fetch data, train model, and generate signals
    python launch_pipeline.py --fetch-data --train --predict

    # Use existing data for inference only
    python launch_pipeline.py --data data_raw/btcusdt.csv --predict

    # Full pipeline with all steps
    python launch_pipeline.py --fetch-data --train --predict --evaluate
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.opensource_fetcher import quick_fetch_data, OpenSourceDataFetcher
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.config.config_parser import ConfigParser


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Launch Neural Trading Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch 7 days of Bitcoin data from Binance
  python launch_pipeline.py --fetch-data --source binance --days 7

  # Generate synthetic data for testing
  python launch_pipeline.py --fetch-data --source synthetic --days 10

  # Use existing CSV file
  python launch_pipeline.py --data data_raw/my_data.csv

  # Full pipeline: fetch, preprocess, and analyze
  python launch_pipeline.py --fetch-data --preprocess --evaluate
        """
    )

    # Data fetching options
    parser.add_argument(
        '--fetch-data',
        action='store_true',
        help='Fetch new data from source'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='binance',
        choices=['binance', 'yfinance', 'synthetic'],
        help='Data source (default: binance)'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading symbol (default: BTCUSDT)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='1m',
        choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
        help='Candlestick interval (default: 1m)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='Number of days to fetch (default: 7)'
    )

    # Data path options
    parser.add_argument(
        '--data',
        type=str,
        help='Path to existing CSV data file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data_raw/crypto_data.csv',
        help='Output path for fetched data (default: data_raw/crypto_data.csv)'
    )

    # Pipeline options
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Preprocess data for training'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the model (requires preprocessed data)'
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Generate predictions and signals'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate data quality and statistics'
    )

    # Config options
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file (default: configs/config.yaml)'
    )

    return parser.parse_args()


def fetch_data_step(args):
    """Fetch data from specified source."""
    print("\n" + "=" * 60)
    print("STEP 1: FETCHING DATA")
    print("=" * 60)

    data = quick_fetch_data(
        source=args.source,
        symbol=args.symbol,
        interval=args.interval,
        days=args.days,
        output_path=args.output
    )

    print(f"\nData shape: {data.shape}")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    return data


def load_data_step(args):
    """Load data from CSV file."""
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)

    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")

    loader = DataLoader(args.data)
    data = loader.load()

    # Validate
    is_valid, errors = loader.validate_columns(data)
    if not is_valid:
        print("⚠ Data validation errors:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Data validation failed")

    print(f"✓ Loaded {len(data)} rows from {args.data}")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")

    return data


def evaluate_data_step(data):
    """Evaluate data quality and show statistics."""
    print("\n" + "=" * 60)
    print("DATA EVALUATION")
    print("=" * 60)

    print(f"\nDataset Statistics:")
    print(f"  Total candles: {len(data):,}")
    print(f"  Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"  Duration: {(data['datetime'].max() - data['datetime'].min()).days} days")

    print(f"\nPrice Statistics:")
    print(f"  Close price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"  Mean close: ${data['close'].mean():.2f}")
    print(f"  Std dev: ${data['close'].std():.2f}")

    print(f"\nVolume Statistics:")
    print(f"  Total volume: {data['volume'].sum():,.0f}")
    print(f"  Mean volume: {data['volume'].mean():,.0f}")

    print(f"\nData Quality:")
    missing = data.isnull().sum()
    if missing.sum() == 0:
        print(f"  ✓ No missing values")
    else:
        print(f"  ⚠ Missing values detected:")
        for col, count in missing[missing > 0].items():
            print(f"    - {col}: {count}")

    # Check for duplicates
    duplicates = data.duplicated(subset=['datetime']).sum()
    if duplicates == 0:
        print(f"  ✓ No duplicate timestamps")
    else:
        print(f"  ⚠ {duplicates} duplicate timestamps found")

    return True


def preprocess_data_step(data, args):
    """Preprocess data for training."""
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING DATA")
    print("=" * 60)

    # Load config
    if Path(args.config).exists():
        config_parser = ConfigParser(args.config)
        config = config_parser.load()
    else:
        print(f"⚠ Config not found: {args.config}, using defaults")
        config = {
            'data': {
                'lookback': 60,
                'window_step': 1
            }
        }

    # Create preprocessor
    preprocessor = Preprocessor(config)

    # Get window parameters from config (support both 'lookback' and 'window_size')
    window_size = config['data'].get('lookback', config['data'].get('window_size', 60))
    window_step = config['data'].get('window_step', 1)

    print(f"Window size: {window_size}")
    print(f"Window step: {window_step}")

    # Create windows
    print("\nCreating sliding windows...")
    windows = preprocessor.create_windows(data)
    print(f"✓ Created {len(windows)} windows of shape {windows.shape}")

    # Fit scaler
    print("\nFitting scaler...")
    preprocessor.fit_scaler(windows.reshape(-1, windows.shape[-1]))
    print(f"✓ Scaler fitted")

    # Save scaler
    scaler_path = Path('models_saved/scaler.pkl')
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor.save_scaler(scaler_path)
    print(f"✓ Scaler saved to {scaler_path}")

    return windows, preprocessor


def main():
    """Main pipeline execution."""
    args = parse_args()

    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "NEURAL TRADING PIPELINE" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")

    try:
        # Step 1: Get data
        if args.fetch_data:
            data = fetch_data_step(args)
            data_path = args.output
        elif args.data:
            data = load_data_step(args)
            data_path = args.data
        else:
            print("\n⚠ No data source specified. Use --fetch-data or --data <path>")
            print("Tip: Try 'python launch_pipeline.py --fetch-data --source binance --days 7'")
            return 1

        # Step 2: Evaluate data
        if args.evaluate or not (args.preprocess or args.train or args.predict):
            evaluate_data_step(data)

        # Step 3: Preprocess
        if args.preprocess:
            windows, preprocessor = preprocess_data_step(data, args)
            print(f"\n✓ Preprocessing complete. Data ready for training.")

        # Step 4: Train (placeholder)
        if args.train:
            print("\n" + "=" * 60)
            print("STEP 3: TRAINING MODEL")
            print("=" * 60)
            print("⚠ Training not implemented yet")
            print("Tip: Use the preprocessed data to train your model")

        # Step 5: Predict (placeholder)
        if args.predict:
            print("\n" + "=" * 60)
            print("STEP 4: GENERATING PREDICTIONS")
            print("=" * 60)
            print("⚠ Prediction not implemented yet")
            print("Tip: Load a trained model and generate signals")

        # Summary
        print("\n" + "=" * 60)
        print("PIPELINE SUMMARY")
        print("=" * 60)
        print(f"✓ Data loaded: {len(data):,} candles")
        if args.fetch_data:
            print(f"✓ Data saved to: {data_path}")
        if args.preprocess:
            print(f"✓ Data preprocessed and ready")
        print("\n✓ Pipeline execution complete!")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
