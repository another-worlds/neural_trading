# Neural Trading Pipeline - TDD Implementation

**A Test-Driven Development (TDD) implementation of a deep learning-based cryptocurrency trading system with learnable technical indicators and multi-horizon predictions.**

## Overview

This project implements the neural trading pipeline as specified in the SRS (Software Requirements Specification), following a strict Test-Driven Development approach. All tests are written first, ensuring comprehensive coverage before implementation.

### Key Features

- **Multi-Horizon Predictions**: 3 independent prediction horizons (h0: 1min, h1: 5min, h2: 15min)
- **Learnable Technical Indicators**: 30+ trainable indicator parameters
- **Hybrid Architecture**: Transformer + LSTM + Indicator Subnet
- **Uncertainty Quantification**: Variance estimates for each prediction
- **Modular Design**: Registry-based component system with automatic integration
- **Configuration-Driven**: YAML-based configuration with cascading updates

## Project Structure

```
neural_trading/
├── src/                          # Source code (to be implemented)
│   ├── config/                   # Configuration management
│   ├── data/                     # Data processing modules
│   ├── models/                   # Neural network architectures
│   ├── losses/                   # Custom loss functions
│   ├── metrics/                  # Custom metrics
│   ├── training/                 # Training orchestrator
│   ├── inference/                # Model inference
│   └── utils/                    # Utility functions
├── tests/                        # Test suite (TDD - written first!)
│   ├── unit/                     # Unit tests for each module
│   ├── integration/              # Integration tests
│   ├── conftest.py              # Pytest fixtures
│   └── __init__.py
├── configs/                      # Configuration files
│   └── config.yaml              # Main configuration
├── data_raw/                     # Raw market data
├── models_saved/                 # Saved model artifacts
├── logs/                         # Training and inference logs
├── notebooks/                    # Jupyter notebooks
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Pytest configuration
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Test Suite

### Comprehensive Test Coverage

Following TDD principles, **all tests are written before implementation**:

#### Unit Tests

1. **Helper Functions** (`test_helper_functions.py`)
   - ✅ 8 functions from SRS Section 3.3.1
   - ✅ 50+ test cases covering edge cases

2. **Configuration** (`test_config.py`)
   - ✅ YAML/JSON parsing and validation
   - ✅ Cascading configuration updates
   - ✅ Environment variable overrides

3. **Data Processing** (`test_data_loader.py`, `test_preprocessor.py`, `test_indicators.py`, `test_dataset.py`)
   - ✅ Data loading and validation
   - ✅ Windowing and scaling
   - ✅ Learnable indicators (30+ parameters)
   - ✅ tf.data.Dataset creation

4. **Model Components** (`test_model_components.py`)
   - ✅ Transformer block
   - ✅ LSTM block (Bidirectional)
   - ✅ Indicator subnet
   - ✅ Hybrid model with 3 independent towers

5. **Losses** (`test_losses.py`)
   - ✅ Focal Loss (α=0.7, γ=1.0)
   - ✅ Huber Loss
   - ✅ Negative Log Likelihood
   - ✅ Trend losses
   - ✅ Composite loss with lambda weighting

6. **Metrics** (`test_metrics.py`)
   - ✅ Direction accuracy, F1, MCC
   - ✅ Price MAE, MAPE
   - ✅ Multi-horizon metric aggregation

7. **Training** (`test_training.py`)
   - ✅ Training orchestrator
   - ✅ Callbacks (EarlyStopping, ModelCheckpoint, TensorBoard)
   - ✅ Indicator parameter logging
   - ✅ Lambda calibration

8. **Inference** (`test_inference.py`)
   - ✅ Model loading and prediction
   - ✅ Signal generation
   - ✅ Stability evaluation
   - ✅ Performance metrics

#### Integration Tests

- ✅ End-to-end data pipeline
- ✅ Full training pipeline
- ✅ Inference pipeline
- ✅ Backtesting pipeline
- ✅ Configuration cascading

### Running Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_helper_functions.py

# Run integration tests only
pytest -m integration

# Run unit tests only
pytest -m unit

# Skip slow tests
pytest -m "not slow"

# Skip GPU tests
pytest -m "not gpu"
```

## Configuration

The system is fully configurable via `configs/config.yaml`:

```yaml
# Example: Update training parameters
training:
  batch_size: 144
  epochs: 40
  learning_rate: 0.001
  patience: 40

# Example: Modify loss weights
losses:
  direction_loss:
    weight: 2.0  # Increase direction loss weight

# Example: Add new indicator periods
indicators:
  ma_periods: [5, 15, 30, 60]  # Add 60-period MA
```

Changes cascade automatically through the pipeline via the registry system.

## Pipeline Doctrine

This implementation follows the modular pipeline doctrine:

### 1. **Static Interfaces**
- Abstract classes and protocols for each module
- Fixed interfaces ensure stability

### 2. **Automatic Integration**
- Registry-based component registration
- Decorators for automatic discovery
- New components integrate on import

### 3. **Cascading Updates**
- Configuration changes propagate automatically
- No code rewriting required
- Registry system handles integration

### 4. **Isolated Logic**
- Each component in separate file/class
- Clear separation of concerns
- Composable via registry

## Key Components

### Learnable Indicators (30+ parameters)

```python
# Automatically registered via decorator
@INDICATOR_REGISTRY.register('ma')
class LearnableMA:
    """3 learnable MA periods"""
    pass

@INDICATOR_REGISTRY.register('macd')
class LearnableMacd:
    """9 learnable MACD parameters (3 settings × 3)"""
    pass

# Total: MA(3) + MACD(9) + CustomMACD(9) + RSI(3) + BB(3) + Momentum(3) = 30
```

### Multi-Output Model (9 outputs)

```
Input (60-min window)
    ↓
[Transformer Block] → [LSTM Block] → [Indicator Subnet]
    ↓
┌─────────────┬─────────────┬─────────────┐
│  Tower h0   │  Tower h1   │  Tower h2   │
│  (1 min)    │  (5 min)    │  (15 min)   │
└─────────────┴─────────────┴─────────────┘
    ↓              ↓              ↓
[Price]        [Price]        [Price]
[Direction]    [Direction]    [Direction]
[Variance]     [Variance]     [Variance]
```

### Helper Functions (All Tested)

1. `calculate_confidence(variance)` → Confidence [0, 1]
2. `calculate_signal_strength(direction, confidence)` → Signal
3. `normalize_variance(variance, mean, std)` → Normalized variance
4. `calculate_profit_targets(entry, predictions)` → TP1, TP2, TP3
5. `calculate_dynamic_stop_loss(entry, type, variance)` → Stop loss
6. `calculate_position_size_multiplier(confidence)` → Position size
7. `check_multi_horizon_agreement(predictions, current)` → Agreement
8. `detect_variance_spike(variance, mean, std)` → Spike detection

## Development Workflow

### TDD Cycle

1. **Write Test** (RED)
   ```bash
   # Create test for new feature
   vim tests/unit/test_new_feature.py
   pytest tests/unit/test_new_feature.py  # Should fail
   ```

2. **Implement Code** (GREEN)
   ```bash
   # Implement minimal code to pass test
   vim src/module/new_feature.py
   pytest tests/unit/test_new_feature.py  # Should pass
   ```

3. **Refactor** (REFACTOR)
   ```bash
   # Improve code quality
   pytest  # All tests should still pass
   ```

## SRS Compliance

This test suite ensures compliance with all SRS requirements:

- ✅ **FR-1**: Data Acquisition (Section 3.1)
- ✅ **FR-2**: Model Training (Section 3.5)
- ✅ **FR-3**: Prediction and Inference (Section 3.6)
- ✅ **FR-4**: Trading Signal Generation (Section 3.3)
- ✅ **FR-5**: Backtesting (Section 3.4)
- ✅ **NFR-1**: Performance (<1s inference)
- ✅ **NFR-2**: Reliability (error handling, validation)
- ✅ **NFR-3**: Maintainability (PEP 8, docstrings)

## Next Steps

After completing TDD test suite:

1. **Implement Source Code**: Follow tests as specification
2. **Run Tests Iteratively**: Ensure each module passes tests
3. **Integration**: Verify end-to-end pipeline
4. **Training**: Train model on real data
5. **Evaluation**: Assess performance metrics
6. **Backtesting**: Validate trading strategy
7. **Deployment**: Deploy inference service

## Testing Philosophy

> "Tests are not just verification - they are the specification."

- Tests define expected behavior
- Tests document API contracts
- Tests enable confident refactoring
- Tests catch regressions early
- Tests guide implementation

## License

See LICENSE file.

## References

- SRS.md - Software Requirements Specification
- Pipeline Doctrine (provided in Russian)
- TensorFlow 2.15+ Documentation
- pytest Documentation
