# Getting Started with Implementation

This guide helps you start implementing the neural trading pipeline using the TDD approach.

## 📋 Prerequisites

You now have:
- ✅ 340+ tests written (TDD approach)
- ✅ Complete project structure
- ✅ Detailed implementation plan
- ✅ Progress tracker
- ✅ Configuration examples

## 🚀 Quick Start

### 1. Review the Planning Documents

```bash
# Read the implementation plan (21-day roadmap)
cat IMPLEMENTATION_PLAN.md

# Check current progress
cat IMPLEMENTATION_PROGRESS.md
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (when requirements.txt is ready)
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 3. Verify Test Suite

```bash
# Run all tests (they will fail - that's expected!)
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_helper_functions.py -v
```

## 📅 Day 1: Start with Helper Functions

### Step 1: Create the Module

```bash
# Create the file
touch src/utils/helper_functions.py
```

### Step 2: Implement First Function

Open `src/utils/helper_functions.py` and implement:

```python
"""Helper functions for trading signal generation."""
import numpy as np


def calculate_confidence(variance, eps=1e-7):
    """
    Convert variance to confidence score [0, 1].
    
    Args:
        variance: Model variance (float or array)
        eps: Small epsilon to prevent division by zero
        
    Returns:
        Confidence score(s) in range [0, 1]
    """
    return 1.0 / (1.0 + np.asarray(variance) + eps)
```

### Step 3: Run Tests

```bash
# Test just this function
pytest tests/unit/test_helper_functions.py::TestCalculateConfidence -v

# Expected output:
# tests/unit/test_helper_functions.py::TestCalculateConfidence::test_zero_variance_returns_max_confidence PASSED
# tests/unit/test_helper_functions.py::TestCalculateConfidence::test_unit_variance_returns_half_confidence PASSED
# ... (8 tests for this function)
```

### Step 4: Implement Remaining Functions

Continue with the other 7 functions:
1. calculate_signal_strength
2. normalize_variance
3. calculate_profit_targets
4. calculate_dynamic_stop_loss
5. calculate_position_size_multiplier
6. check_multi_horizon_agreement
7. detect_variance_spike

### Step 5: Verify All Helper Function Tests Pass

```bash
# Run all helper function tests
pytest tests/unit/test_helper_functions.py -v

# Target: All 50+ tests passing ✓
```

### Step 6: Update Progress

Edit `IMPLEMENTATION_PROGRESS.md`:
```markdown
### Module 1: utils/helper_functions.py
- **Status**: ✅ COMPLETED
- **Tests Passing**: 50/50
- **Completion Date**: 2026-01-XX
```

## 📊 Implementation Order

Follow this sequence (see IMPLEMENTATION_PLAN.md for details):

**Week 1:**
- Day 1: helper_functions.py ✓
- Day 2: config_parser.py
- Day 3: All 3 registries
- Days 4-6: Data processing

**Week 2:**
- Days 7-8: Learnable indicators
- Days 9-11: Model components
- Day 12: Losses & metrics

**Week 3:**
- Days 13-15: Training infrastructure
- Days 16-18: Inference & backtesting

**Week 4:**
- Days 19-21: Integration testing & validation

## 🔍 Key Implementation Tips

### 1. Always Run Tests After Each Change

```bash
# Quick feedback loop
pytest tests/unit/test_helper_functions.py -v --tb=short
```

### 2. Check Shape Compatibility

```python
# Add assertions during development
print(f"Input shape: {x.shape}")  # Debug tensor shapes
assert x.shape == expected_shape, f"Shape mismatch: {x.shape} vs {expected_shape}"
```

### 3. Use Test-Driven Development

1. Read the test to understand requirements
2. Implement minimal code to pass the test
3. Refactor while keeping tests green
4. Move to next test

### 4. Monitor Progress Daily

```bash
# Update IMPLEMENTATION_PROGRESS.md daily
# Track tests passing, blockers, completion dates
```

## 🎯 Success Criteria Per Module

Each module is complete when:
- ✅ All tests passing
- ✅ Code coverage >70%
- ✅ No import errors
- ✅ Docstrings added
- ✅ Progress tracker updated

## 🔧 Development Commands

```bash
# Run tests with markers
pytest -m "not slow" -v          # Skip slow tests
pytest -m integration -v          # Run integration tests only

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term

# Run in parallel (faster)
pip install pytest-xdist
pytest -n auto

# Debug mode
pytest --pdb -x  # Stop at first failure and enter debugger

# Show print statements
pytest -s
```

## 📚 Reference Documents

- **IMPLEMENTATION_PLAN.md**: Complete 21-day roadmap with code examples
- **IMPLEMENTATION_PROGRESS.md**: Daily progress tracker
- **SRS.md**: Software requirements specification
- **configs/config.yaml**: Configuration example
- **tests/**: All test specifications

## 🚨 Common Pitfalls

1. **Tensor Shape Mismatches**: Always verify shapes at integration points
2. **Forgetting to Run Tests**: Run after each change
3. **Skipping Phases**: Follow dependency order (don't skip ahead)
4. **Not Updating Progress**: Keep IMPLEMENTATION_PROGRESS.md current

## 🎓 Learning Resources

### TensorFlow/Keras
- [TensorFlow Docs](https://www.tensorflow.org/)
- [Keras Functional API](https://keras.io/guides/functional_api/)
- [Custom Layers Guide](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)

### Testing
- [Pytest Documentation](https://docs.pytest.org/)
- [TDD by Example (Book)](https://www.amazon.com/Test-Driven-Development-Kent-Beck/dp/0321146530)

## 💬 Questions?

Refer to:
1. Test specifications in `tests/`
2. Implementation plan in `IMPLEMENTATION_PLAN.md`
3. SRS for requirements in `SRS.md`

## 🏁 Ready to Start?

```bash
# Create first module
touch src/utils/helper_functions.py

# Open in editor and start implementing!
# Remember: Tests are already written, just make them pass!
```

Good luck! 🚀
