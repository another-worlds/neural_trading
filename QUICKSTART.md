# Quick Start Guide - Neural Trading Pipeline

Get started with the neural trading pipeline in **5 minutes**!

## ⚡ Instant Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Fetch Data & Launch Pipeline

```bash
# Fetch 7 days of Bitcoin data from Binance (no auth required!)
python launch_pipeline.py --fetch-data --source binance --days 7 --evaluate

# Or generate synthetic data for testing
python launch_pipeline.py --fetch-data --source synthetic --days 10 --evaluate
```

**That's it!** You now have data ready for the trading pipeline.

---

## 📊 Data Sources

### Option 1: Binance Public API (Recommended)
**No API keys required!** Fetches real Bitcoin/crypto data.

```bash
python launch_pipeline.py --fetch-data --source binance --symbol BTCUSDT --interval 1m --days 7
```

**Pros:**
- ✅ Real market data
- ✅ No authentication needed
- ✅ Multiple symbols (BTCUSDT, ETHUSDT, etc.)
- ✅ Multiple intervals (1m, 5m, 15m, 1h, 4h, 1d)

**Cons:**
- ⚠️ Requires internet connection
- ⚠️ Rate limited (handled automatically)

### Option 2: Synthetic Data (Testing)
Generate realistic-looking synthetic data for testing.

```bash
python launch_pipeline.py --fetch-data --source synthetic --days 10
```

**Pros:**
- ✅ Works offline
- ✅ Fast generation
- ✅ Customizable size

**Cons:**
- ⚠️ Not real market data
- ⚠️ Only for testing/development

### Option 3: Yahoo Finance (Alternative)
Requires `yfinance` package (install: `pip install yfinance`)

```bash
python launch_pipeline.py --fetch-data --source yfinance --symbol BTC-USD --days 7
```

---

## 🚀 Pipeline Workflows

### Workflow 1: Data Exploration
```bash
# Fetch and analyze data quality
python launch_pipeline.py --fetch-data --evaluate
```

**Output:**
- Data statistics (price range, volume, etc.)
- Quality metrics (missing values, duplicates)
- Date range information

### Workflow 2: Data Preparation
```bash
# Fetch data and preprocess for training
python launch_pipeline.py --fetch-data --preprocess
```

**Output:**
- Sliding windows created (60-minute lookback)
- Feature scaling applied
- Scaler saved to `models_saved/scaler.pkl`

### Workflow 3: Full Pipeline (Coming Soon)
```bash
# Complete pipeline: fetch, train, predict
python launch_pipeline.py --fetch-data --train --predict
```

---

## 📁 File Outputs

### Data Files
```
data_raw/
├── crypto_data.csv          # Default output location
└── btcusdt_7d.csv           # Custom naming example
```

### Model Files
```
models_saved/
├── scaler.pkl               # Feature scaler
└── model.keras              # Trained model (after training)
```

### CSV Format
All data is saved in standard OHLCV format:
```csv
datetime,open,high,low,close,volume
2024-01-01 00:00:00+00:00,42000.0,42100.0,41900.0,42050.0,150.5
2024-01-01 00:01:00+00:00,42050.0,42080.0,42000.0,42030.0,120.3
...
```

---

## 🔧 Advanced Usage

### Custom Output Paths
```bash
python launch_pipeline.py \
    --fetch-data \
    --source binance \
    --output data_raw/my_custom_data.csv
```

### Use Existing Data
```bash
python launch_pipeline.py \
    --data data_raw/my_data.csv \
    --evaluate \
    --preprocess
```

### Different Symbols & Intervals
```bash
# Ethereum, 5-minute candles, 14 days
python launch_pipeline.py \
    --fetch-data \
    --source binance \
    --symbol ETHUSDT \
    --interval 5m \
    --days 14
```

### Custom Config
```bash
python launch_pipeline.py \
    --fetch-data \
    --config my_config.yaml \
    --preprocess
```

---

## 🧪 Testing Data Fetch

Run integration tests (requires internet):

```bash
# Run all data fetcher tests
pytest tests/integration/test_opensource_fetcher.py -v

# Skip network-dependent tests
pytest tests/integration/test_opensource_fetcher.py -v -m "not network"

# Only synthetic data tests (no network)
pytest tests/integration/test_opensource_fetcher.py::TestOpenSourceDataFetcher::test_fetch_sample_data_generation -v
```

---

## 🐍 Python API Usage

### Direct Python Usage
```python
from src.data.opensource_fetcher import quick_fetch_data

# Fetch data programmatically
data = quick_fetch_data(
    source='binance',
    symbol='BTCUSDT',
    interval='1m',
    days=7,
    output_path='data_raw/btc_7d.csv'
)

print(f"Fetched {len(data)} candles")
print(data.head())
```

### Advanced Fetcher Usage
```python
from src.data.opensource_fetcher import OpenSourceDataFetcher

fetcher = OpenSourceDataFetcher()

# Binance data
binance_data = fetcher.fetch_binance_spot(
    symbol='BTCUSDT',
    interval='1m',
    days=7
)

# Synthetic data for testing
test_data = fetcher.fetch_sample_data(
    num_candles=10000,
    interval='1m'
)

# Save to CSV
fetcher.save_to_csv(test_data, 'test_data.csv')  # If you add this helper
```

### Load with DataLoader
```python
from src.data.data_loader import DataLoader

# Load CSV
loader = DataLoader('data_raw/crypto_data.csv')
data = loader.load()

# Validate
is_valid, errors = loader.validate_columns(data)
if not is_valid:
    print("Validation errors:", errors)
```

---

## 🎯 Next Steps

### After Data Fetch:

1. **Explore the Data**
   ```bash
   python launch_pipeline.py --data data_raw/crypto_data.csv --evaluate
   ```

2. **Preprocess for Training**
   ```bash
   python launch_pipeline.py --data data_raw/crypto_data.csv --preprocess
   ```

3. **Train Model** (Coming Soon)
   ```python
   # Will be available soon
   python launch_pipeline.py --data data_raw/crypto_data.csv --train
   ```

4. **Generate Predictions** (Coming Soon)
   ```python
   # Will be available soon
   python launch_pipeline.py --data data_raw/crypto_data.csv --predict
   ```

---

## 💡 Tips & Tricks

### Tip 1: Start Small
For initial testing, use **1-2 days** of data or **synthetic data**:
```bash
python launch_pipeline.py --fetch-data --source synthetic --days 2 --evaluate
```

### Tip 2: Save Different Datasets
```bash
# 1-minute data
python launch_pipeline.py --fetch-data --interval 1m --days 7 --output data_raw/btc_1m_7d.csv

# 5-minute data
python launch_pipeline.py --fetch-data --interval 5m --days 14 --output data_raw/btc_5m_14d.csv

# Hourly data
python launch_pipeline.py --fetch-data --interval 1h --days 30 --output data_raw/btc_1h_30d.csv
```

### Tip 3: Monitor Data Quality
Always use `--evaluate` to check data quality:
```bash
python launch_pipeline.py --data data_raw/crypto_data.csv --evaluate
```

### Tip 4: Offline Development
Use synthetic data when developing without internet:
```bash
python launch_pipeline.py --fetch-data --source synthetic --days 10
```

---

## ❓ Troubleshooting

### Issue: Network errors when fetching from Binance
**Solution:**
- Check internet connection
- Try again in a few minutes (rate limiting)
- Use synthetic data as fallback:
  ```bash
  python launch_pipeline.py --fetch-data --source synthetic
  ```

### Issue: "No module named 'yfinance'"
**Solution:**
- Install yfinance: `pip install yfinance`
- Or use Binance/synthetic instead

### Issue: "Data validation failed"
**Solution:**
- Check CSV format matches OHLCV structure
- Ensure datetime column exists and is parseable
- Run `--evaluate` to see specific errors

---

## 📚 Related Documentation

- **[README.md](README.md)** - Project overview and architecture
- **[SRS.md](SRS.md)** - Complete specifications
- **[TESTING_PLAN.md](TESTING_PLAN.md)** - Testing strategy and progress
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Development roadmap

---

## 🎉 Success!

You're now ready to use the neural trading pipeline with real or synthetic data. Start experimenting and building your trading strategies!

**Have questions?** Check the documentation or run with `-h` for help:
```bash
python launch_pipeline.py -h
```
