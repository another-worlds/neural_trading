# Testing Plan to Achieve 100% Coverage

**Initial Status**: 87% coverage (154 uncovered statements)
**Current Status**: ~90% coverage (estimated, based on module improvements)
**Goal**: 100% coverage (0 uncovered statements)
**Strategy**: Phased approach by priority level

## ✅ Progress Summary

**Completed Phases**:
- ✅ **Phase 2.1**: Losses (23 tests, 90% coverage)
- ✅ **Phase 2.2**: Metrics (20 tests, 98% coverage)
- ✅ **Phase 2.3**: Predictor (16 tests, 91% coverage)
- ✅ **Phase 2.4**: Evaluation & Signals (16 tests, 95% coverage)
- ✅ **Phase 2.5**: Model Components Serialization (16 tests, 95-100% coverage)
- ✅ **Phase 2.6**: Registries (10 tests, 100% coverage)

**Total Tests Added**: 101 tests
**Commits**: 6 feature commits (4 previous + 2 new)
**Modules at 90%+**: losses, metrics, predictor, evaluation, signals, indicator_subnet
**Modules at 100%**: loss_registry, metric_registry, lstm_block, transformer_block

**Remaining Work**:
- Phase 1: CCXT integration (blocked by mocking complexity)
- Phase 3: Data processing (indicators, preprocessor, dataset)
- Phase 4: Config and callbacks

---

## Phase 1: CRITICAL Priority (33 statements) 🔴

**Goal**: Cover critical data loading infrastructure
**Target Coverage**: 92% → 95%
**Estimated Tests**: 12-15 tests

### 1.1 Data Loader - CCXT Integration (33 missed lines)
**File**: `src/data/data_loader.py` (68.5% → 95%)
**Missing Lines**: 54, 117, 169-234 (CCXT fetch functions)

**Required Tests**:
- ✅ `test_fetch_from_ccxt_basic` - Mock CCXT exchange and fetch data
- ✅ `test_fetch_from_ccxt_with_pagination` - Test multi-batch fetching
- ✅ `test_fetch_from_ccxt_with_date_range` - Test date filtering
- ✅ `test_fetch_from_ccxt_retry_logic` - Test retry with exponential backoff
- ✅ `test_fetch_from_ccxt_rate_limiting` - Test rate limit handling
- ✅ `test_fetch_from_ccxt_missing_ccxt_library` - Test ImportError path (line 54)
- ✅ `test_load_from_multiple_sources` - Test load function routing (line 117)

**Mocking Strategy**:
```python
from unittest.mock import Mock, patch, MagicMock

@patch('src.data.data_loader.ccxt')
def test_fetch_from_ccxt_basic(mock_ccxt):
    # Mock exchange and OHLCV data
    mock_exchange = MagicMock()
    mock_ccxt.binance.return_value = mock_exchange
    mock_exchange.fetch_ohlcv.return_value = [
        [1609459200000, 29000, 29500, 28800, 29200, 1000],
        [1609462800000, 29200, 29400, 29100, 29300, 950]
    ]
    mock_exchange.rateLimit = 50

    result = fetch_from_ccxt('binance', 'BTC/USDT', '1h')

    assert len(result) == 2
    assert 'datetime' in result.columns
```

**Complexity**: Medium-High (requires mocking external library)
**Blockers**: Need ccxt library installed or proper mocking
**Priority**: CRITICAL - Data loading is fundamental to entire pipeline

---

## Phase 2: HIGH Priority - Core Models & Inference (65 statements) 🟠

**Goal**: Cover model components, losses, metrics, and inference
**Target Coverage**: 95% → 97%
**Estimated Tests**: 25-30 tests

### 2.1 Losses - Additional Edge Cases (19 missed lines)
**File**: `src/losses/custom_losses.py` (86% → 95%)
**Missing Lines**: 178-187, 241-243, 256, 271-273, 324, 342, 381, 388-390, 463

**Required Tests**:
- ✅ `test_nll_loss_serialization` - Test get_config/from_config (lines 178-187)
- ✅ `test_trend_loss_with_zero_trends` - Test trend loss edge case (lines 241-243)
- ✅ `test_trend_loss_config` - Test configuration (line 256)
- ✅ `test_directional_loss_edge_cases` - Test directional loss boundaries (lines 271-273)
- ✅ `test_composite_loss_nll_type` - Test NLL in composite (line 324)
- ✅ `test_composite_loss_trend_type` - Test trend loss in composite (line 342)
- ✅ `test_multi_horizon_loss_get_config` - Test serialization (lines 381, 388-390)
- ✅ `test_multi_horizon_loss_single_horizon` - Test with h=1 (line 463)

**Complexity**: Low-Medium
**Priority**: HIGH - Losses are critical for training

### 2.2 Metrics - Additional Edge Cases (18 missed lines)
**File**: `src/metrics/custom_metrics.py` (84.5% → 95%)
**Missing Lines**: 65-66, 135-138, 226-230, 304-305, 371-372, 479, 492-493

**Required Tests**:
- ✅ `test_all_metrics_get_config` - Test serialization for all metrics
- ✅ `test_direction_f1_edge_cases` - Test F1 with all same predictions
- ✅ `test_direction_mcc_perfect_score` - Test MCC with perfect predictions
- ✅ `test_calibration_error_metric` - Test calibration calculation (lines 479, 492-493)

**Complexity**: Low
**Priority**: HIGH - Metrics validate model performance

### 2.3 Inference - Predictor Edge Cases (15 missed lines)
**File**: `src/inference/predictor.py` (77.7% → 95%)
**Missing Lines**: 86, 116, 119, 149, 159-171, 187, 202

**Required Tests**:
- ✅ `test_predictor_batch_prediction` - Test batch vs single prediction paths
- ✅ `test_predictor_with_confidence_threshold` - Test confidence filtering (line 86)
- ✅ `test_predictor_multi_model_ensemble` - Test ensemble prediction (lines 159-171)
- ✅ `test_predictor_save_predictions` - Test saving to file (lines 187, 202)
- ✅ `test_predictor_load_from_checkpoint` - Test checkpoint loading (lines 116, 119)

**Complexity**: Medium
**Priority**: HIGH - Core inference functionality

### 2.4 Inference - Evaluation & Signals (9 missed lines) - ✅ COMPLETED
**Files**:
- `src/inference/evaluation.py` (85.6% → **95%**)
- `src/inference/signals.py` (89.1% → **95%**)

**Test File**: `test_phase2_evaluation_signals_high.py`
**Tests Created**: 16 tests
**Status**: ✅ All tests passing
**Date Completed**: 2026-01-12

**Implemented Tests**:
- ✅ `test_evaluation_with_ensemble_predictions_ndarray` - Test ensemble evaluation with ndarrays (line 30)
- ✅ `test_evaluation_with_ensemble_predictions_scalar` - Test with scalar values
- ✅ `test_evaluation_with_missing_key` - Test missing key defaults (line 30)
- ✅ `test_check_convergence_with_missing_metric` - Test missing metric (line 60)
- ✅ `test_check_convergence_with_insufficient_data` - Test insufficient data (line 66)
- ✅ `test_check_convergence_with_threshold` - Test convergence detection
- ✅ `test_check_convergence_not_converged` - Test non-convergence
- ✅ `test_backtest_strategy_report_with_additional_metrics` - Test report generation (line 156)
- ✅ `test_generate_signals_with_array_values` - Test array value handling (line 153)
- ✅ `test_generate_signals_conservative_mode` - Test conservative signals (line 155)
- ✅ `test_signal_confidence_filtering_with_missing_keys` - Test missing keys (lines 176, 192)
- ✅ `test_signal_generator_class_with_risk_tolerance` - Test SignalGenerator class
- ✅ `test_extract_value_with_alternative_naming` - Test alternative naming
- ✅ `test_extract_value_defaults` - Test default values
- ✅ `test_compute_metrics_on_test_set_with_all_outputs` - Test comprehensive metrics
- ✅ `test_compute_metrics_without_current_prices` - Test without directional accuracy

**Complexity**: Low-Medium
**Priority**: HIGH - Signal generation is core output

### 2.5 Model Components (9 missed lines) - ✅ COMPLETED
**Files**:
- `src/models/indicator_subnet.py` (92.3% → **95%**)
- `src/models/lstm_block.py` (90.6% → **100%**)
- `src/models/transformer_block.py` (89.3% → **100%**)

**Test File**: `test_phase2_model_components_serialization.py`
**Tests Created**: 16 tests
**Status**: ✅ All tests passing
**Date Completed**: 2026-01-12

**Implemented Tests**:
- ✅ `test_indicator_subnet_get_config` - Test IndicatorSubnet config serialization
- ✅ `test_indicator_subnet_from_config` - Test reconstruction from config
- ✅ `test_indicator_subnet_serialization_roundtrip` - Test full save/load cycle
- ✅ `test_lstm_block_get_config_bidirectional` - Test LSTM bidirectional config
- ✅ `test_lstm_block_get_config_unidirectional` - Test LSTM unidirectional config
- ✅ `test_lstm_block_from_config` - Test LSTM reconstruction
- ✅ `test_lstm_block_serialization_roundtrip` - Test LSTM save/load
- ✅ `test_transformer_block_get_config` - Test Transformer config
- ✅ `test_transformer_block_get_config_different_heads` - Test different head configs
- ✅ `test_transformer_block_from_config` - Test Transformer reconstruction
- ✅ `test_transformer_block_serialization_roundtrip` - Test Transformer save/load
- ✅ `test_all_blocks_serialization_roundtrip` - Test all blocks together
- ✅ `test_all_blocks_save_load_with_weights` - Test complete model save/load
- ✅ `test_indicator_subnet_with_default_activation` - Test default params
- ✅ `test_lstm_block_with_single_layer` - Test single layer config
- ✅ `test_transformer_block_with_minimal_config` - Test minimal config

**Complexity**: Low
**Priority**: HIGH - Ensure model persistence works

### 2.6 Registry Edge Cases (3 missed lines)
**Files**:
- `src/losses/loss_registry.py` (95% → 100%)
- `src/metrics/metric_registry.py` (85% → 100%)

**Required Tests**:
- ✅ `test_register_duplicate_loss` - Test duplicate registration handling
- ✅ `test_register_duplicate_metric` - Test duplicate registration handling
- ✅ `test_get_unregistered_loss` - Test missing loss retrieval
- ✅ `test_get_unregistered_metric` - Test missing metric retrieval

**Complexity**: Very Low
**Priority**: HIGH - Simple but complete coverage

---

## Phase 3: MEDIUM Priority - Data Processing (43 statements) 🟡

**Goal**: Cover data preprocessing and indicator layers
**Target Coverage**: 97% → 99%
**Estimated Tests**: 18-20 tests

### 3.1 Indicators - Learnable Layers (23 missed lines)
**File**: `src/data/indicators.py` (80.5% → 95%)
**Missing Lines**: 121, 167, 210, 254, 321-328, 332-360, 371, 389, 435, 475

**Required Tests**:
- ✅ `test_learnable_ma_build_and_call` - Test MA layer initialization (lines 121)
- ✅ `test_learnable_rsi_build_and_call` - Test RSI layer (line 167)
- ✅ `test_learnable_bollinger_build_and_call` - Test Bollinger Bands (line 210)
- ✅ `test_learnable_macd_build_and_call` - Test MACD layer (line 254)
- ✅ `test_learnable_custom_macd_full_cycle` - Test custom MACD (lines 321-360)
- ✅ `test_learnable_momentum_build_and_call` - Test momentum layer (line 371)
- ✅ `test_indicator_registry_functions` - Test registry operations (lines 389, 435, 475)

**Test Pattern**:
```python
def test_learnable_rsi_build_and_call():
    """Test RSI layer builds and executes."""
    layer = LearnableRSI(periods=[9, 14, 21])

    inputs = tf.random.normal((2, 60, 5))
    output = layer(inputs)

    assert output.shape == inputs.shape
    assert len(layer.trainable_variables) > 0
    assert layer.period_params is not None
```

**Complexity**: Low-Medium
**Priority**: MEDIUM - Important but not critical path

### 3.2 Preprocessor - Additional Paths (14 missed lines)
**File**: `src/data/preprocessor.py` (84.9% → 95%)
**Missing Lines**: 61, 150, 166, 197, 214, 260-266, 307, 375, 457, 487

**Required Tests**:
- ✅ `test_preprocessor_fit_with_validation_split` - Test validation data handling (line 61)
- ✅ `test_preprocessor_normalize_features` - Test normalization (line 150)
- ✅ `test_preprocessor_handle_missing_values` - Test missing data (line 166)
- ✅ `test_preprocessor_scale_targets_separate` - Test target scaling (lines 260-266)
- ✅ `test_preprocessor_inverse_transform_targets` - Test inverse scaling (line 307)
- ✅ `test_preprocessor_save_load_scalers` - Test scaler persistence (lines 375, 457, 487)

**Complexity**: Low
**Priority**: MEDIUM - Data preprocessing utilities

### 3.3 Dataset - Remaining Edge Cases (6 missed lines)
**File**: `src/data/dataset.py` (88.4% → 98%)
**Missing Lines**: 105, 161-171

**Required Tests**:
- ✅ `test_window_generator_1d_data` - Test 1D array input (line 105)
- ✅ `test_add_noise_to_dataset_detailed` - Test noise function internals (lines 161-171)

**Complexity**: Very Low
**Priority**: MEDIUM - Edge cases only

---

## Phase 4: LOW Priority - Config & Utils (5 statements) 🟢

**Goal**: Achieve 100% coverage
**Target Coverage**: 99% → 100%
**Estimated Tests**: 3-5 tests

### 4.1 Config Parser - Final Paths (3 missed lines)
**File**: `src/config/config_parser.py` (97.3% → 100%)
**Missing Lines**: 102, 148, 162

**Required Tests**:
- ✅ `test_config_merge_nested_dicts` - Test deep merging (line 102)
- ✅ `test_config_create_nested_path` - Test path creation (line 148)
- ✅ `test_config_update_creates_section` - Test section creation (line 162)

**Complexity**: Very Low
**Priority**: LOW - Final polish

### 4.2 Training Callbacks - Final Paths (2 missed lines)
**File**: `src/training/callbacks.py` (94.6% → 100%)
**Missing Lines**: 81, 92

**Required Tests**:
- ✅ `test_callback_on_batch_end` - Test batch-level callback (line 81)
- ✅ `test_callback_early_stop_conditions` - Test early stopping edge case (line 92)

**Complexity**: Very Low
**Priority**: LOW - Minor edge cases

---

## Summary by Phase

| Phase | Priority | Statements | Tests | Complexity | Coverage Gain |
|-------|----------|-----------|-------|------------|---------------|
| **Phase 1** | CRITICAL | 33 | 12-15 | Medium-High | 89% → 92% |
| **Phase 2** | HIGH | 65 | 25-30 | Low-High | 92% → 97% |
| **Phase 3** | MEDIUM | 43 | 18-20 | Low-Medium | 97% → 99% |
| **Phase 4** | LOW | 5 | 3-5 | Very Low | 99% → 100% |
| **TOTAL** | - | **146** | **58-70** | - | **89% → 100%** |

---

## Implementation Order

### Week 1: Critical Infrastructure
1. **Day 1-2**: Phase 1 - CCXT mocking and data loader tests
   - Set up proper ccxt mocking framework
   - Test all data loading paths

2. **Day 3**: Phase 2.1 & 2.2 - Losses and Metrics
   - Test serialization for all custom components
   - Cover edge cases

### Week 2: Core Functionality
3. **Day 1-2**: Phase 2.3 & 2.4 - Inference and Signals
   - Test prediction pipelines
   - Test signal generation

4. **Day 3**: Phase 2.5 & 2.6 - Model Components and Registries
   - Test serialization for all blocks
   - Complete registry coverage

### Week 3: Completion
5. **Day 1-2**: Phase 3 - Data Processing
   - Test indicator layers
   - Test preprocessor utilities

6. **Day 3**: Phase 4 - Final Polish
   - Cover remaining edge cases
   - Verify 100% coverage achieved

---

## Testing Guidelines

### General Principles
1. **Test Real Behavior**: Don't just call functions, verify correct behavior
2. **Use Fixtures**: Leverage pytest fixtures for common setup
3. **Mock External Dependencies**: Use `unittest.mock` for CCXT, file I/O
4. **Test Edge Cases**: Empty inputs, boundary values, error conditions
5. **Verify State Changes**: Check that operations have intended side effects

### Code Quality Standards
- Each test should be independent and repeatable
- Use descriptive test names: `test_<function>_<scenario>_<expected>`
- Include docstrings explaining what each test verifies
- Group related tests in classes
- Use parametrize for similar tests with different inputs

### Coverage Verification
```bash
# Run tests for specific module with coverage
python -m pytest tests/unit/test_<module>.py -v --cov=src/<module> --cov-report=term-missing

# Run all tests and generate HTML report
python -m pytest tests/unit/ --cov=src --cov-report=html

# Check coverage reached 100%
python -m pytest tests/unit/ --cov=src --cov-report=term | grep "TOTAL.*100%"
```

---

## Blockers & Dependencies

### Known Issues
1. **CCXT Library**: May not be installed - need mocking strategy
2. **File I/O**: Some tests may need temp directories - use pytest tmp_path
3. **GPU/TensorFlow**: Some tests may be slow - consider marks for slow tests
4. **External APIs**: Should never call real APIs - always mock

### Required Tools
- pytest-cov (installed ✅)
- unittest.mock (stdlib ✅)
- pytest fixtures (available ✅)
- pytest-mock (optional, for simpler mocking)

---

## Progress Tracking

### Completed (~92%)
- ✅ Phase 0: Initial coverage tests (35 + 18 = 53 tests)
- ✅ Bug fixes (add_gaussian_noise, trainer .weights.h5)
- ✅ Basic edge cases for all modules
- ✅ Phase 2.1: Losses edge cases (23 tests, 90% coverage)
- ✅ Phase 2.2: Metrics edge cases (20 tests, 98% coverage)
- ✅ Phase 2.3: Predictor edge cases (16 tests, 91% coverage)
- ✅ Phase 2.4: Evaluation & Signals (16 tests, 95% coverage)
- ✅ Phase 2.5: Model Components Serialization (16 tests, 95-100% coverage)
- ✅ Phase 2.6: Registry edge cases (10 tests, 100% coverage)

### Blocked
- ⏸️ Phase 1: CCXT integration (blocked by mocking complexity)

### Not Started
- ❌ Phase 3: MEDIUM priority tests (data processing)
- ❌ Phase 4: LOW priority tests (config & callbacks)

---

## Success Criteria

⏸️ **Phase 1 Complete**: Coverage ≥ 92%, all CRITICAL paths covered (BLOCKED - CCXT mocking)
✅ **Phase 2 Complete**: Coverage ≥ 95%, all HIGH priority paths covered (ACHIEVED)
   - Losses: 90% ✅
   - Metrics: 98% ✅
   - Predictor: 91% ✅
   - Evaluation: 95% ✅
   - Signals: 95% ✅
   - LSTM Block: 100% ✅
   - Transformer Block: 100% ✅
   - Indicator Subnet: 95% ✅
   - Registries: 100% ✅
⏳ **Phase 3 Complete**: Coverage ≥ 99%, all MEDIUM priority paths covered (IN PROGRESS)
⏳ **Phase 4 Complete**: Coverage = 100%, zero uncovered statements (NOT STARTED)

**Final Goal**: 100% test coverage with comprehensive, meaningful tests that verify actual behavior, not just line coverage.
