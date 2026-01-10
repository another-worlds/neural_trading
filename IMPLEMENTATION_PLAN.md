# Neural Trading Pipeline - Implementation Plan

**Version**: 1.0
**Created**: 2026-01-10
**Based on**: Comprehensive TDD test suite (340+ tests)

---

## Executive Summary

This document provides a complete implementation roadmap for the neural trading pipeline. All tests are already written (TDD approach). Implementation follows a dependency-driven order over ~18-21 days.

**Total Scope**: 20 modules, ~6,000-8,000 lines of code, 340+ tests

---

## Quick Reference

### Implementation Order (By Dependency Layer)

```
Layer 0 (Days 1-2): Foundation
├── utils/helper_functions.py          ✓ 8 functions
└── config/config_parser.py            ✓ Config management

Layer 1 (Day 3): Registries
├── data/indicators.py (registry)      ✓ Auto-registration
├── losses/loss_registry.py            ✓ Loss registry
└── metrics/metric_registry.py         ✓ Metric registry

Layer 2 (Days 4-6): Data Processing
├── data/data_loader.py                ✓ OHLCV loading
├── data/preprocessor.py               ✓ Windowing, scaling
├── data/dataset.py                    ✓ tf.data.Dataset
└── data/indicators.py (full)          ✓ 30+ learnable params

Layer 3 (Days 7-10): Model Components
├── models/transformer_block.py        ✓ Attention mechanism
├── models/lstm_block.py               ✓ Bidirectional LSTM
├── models/indicator_subnet.py         ✓ Indicator network
└── models/hybrid_model.py             ✓ 3-tower architecture

Layer 4 (Days 11-12): Losses & Metrics
├── losses/custom_losses.py            ✓ 5 loss functions
└── metrics/custom_metrics.py          ✓ 6 metrics

Layer 5 (Days 13-15): Training
├── training/callbacks.py              ✓ Training callbacks
└── training/trainer.py                ✓ Orchestrator

Layer 6 (Days 16-18): Inference
├── inference/predictor.py             ✓ Model inference
├── inference/signals.py               ✓ Signal generation
└── inference/backtesting.py           ✓ Backtesting engine
```

---

## Phase-by-Phase Implementation

### Phase 1: Foundation (Days 1-2) - SIMPLE

#### Day 1: Helper Functions
**File**: `src/utils/helper_functions.py`
**Lines**: 150-200
**Tests**: 540 lines, 50+ cases

**8 Functions to Implement**:
1. `calculate_confidence(variance, eps=1e-7)` → float
2. `calculate_signal_strength(direction_prob, confidence)` → float
3. `normalize_variance(variance, mean, std, eps)` → float
4. `calculate_profit_targets(entry_price, predictions)` → dict
5. `calculate_dynamic_stop_loss(entry, type, variance, mean, ...)` → float
6. `calculate_position_size_multiplier(confidence, ...)` → float
7. `check_multi_horizon_agreement(predictions, current, thresh)` → tuple
8. `detect_variance_spike(variance, mean, std, thresh)` → bool

**Run Tests**:
```bash
pytest tests/unit/test_helper_functions.py -v
```

**Success Criteria**: All 50+ tests passing

---

#### Day 2: Configuration Management
**File**: `src/config/config_parser.py`
**Lines**: 200-250
**Tests**: 405 lines, 35+ cases

**Classes to Implement**:
- `ConfigParser`: Load YAML/JSON, validate, nested access
- `ConfigValidationError`: Custom exception
- `validate_config_schema(config)`: Validation function
- `merge_with_defaults(config, defaults)`: Config merging

**Key Features**:
- Nested config access: `parser.get('data.lookback')`
- Environment variable overrides: `NEURAL_TRADE_EPOCHS`
- Schema validation: positive values, valid ranges

**Run Tests**:
```bash
pytest tests/unit/test_config.py -v
```

**Success Criteria**: All 35+ tests passing

---

### Phase 2: Registry Systems (Day 3) - SIMPLE

#### Implement 3 Registry Modules (Same Pattern)

**Files**:
1. `src/data/indicators.py` (IndicatorRegistry only)
2. `src/losses/loss_registry.py`
3. `src/metrics/metric_registry.py`

**Lines**: ~80-100 each
**Pattern**:
```python
class IndicatorRegistry:
    def __init__(self):
        self.indicators = {}

    def register(self, name):
        def decorator(cls):
            self.indicators[name] = cls
            return cls
        return decorator

    def get(self, name):
        return self.indicators[name]

    def list_indicators(self):
        return list(self.indicators.keys())

# Global instance
INDICATOR_REGISTRY = IndicatorRegistry()
```

**Run Tests**:
```bash
pytest tests/unit/test_indicators.py::TestIndicatorRegistry -v
pytest tests/unit/test_losses.py::TestLossRegistry -v
pytest tests/unit/test_metrics.py::TestMetricRegistry -v
```

**Success Criteria**: All registry tests passing

---

### Phase 3: Data Processing (Days 4-6) - MEDIUM

#### Day 4: Data Loading
**File**: `src/data/data_loader.py`
**Lines**: 300-400
**Tests**: 310 lines, 25+ cases

**Classes & Functions**:
- `DataLoader`: Main class
- `load_from_csv(path)` → DataFrame
- `fetch_from_ccxt(symbol, timeframe, start, end, exchange)` → DataFrame
- `validate_ohlcv_data(df, timeframe)` → (bool, list[errors])
- `calculate_quality_metrics(df)` → dict

**Key Features**:
- OHLC validation: high ≥ max(open, close), low ≤ min(open, close)
- Time gap detection
- CCXT pagination (500 candles per request)
- Retry logic with exponential backoff

**Run Tests**:
```bash
pytest tests/unit/test_data_loader.py -v
```

---

#### Day 5: Preprocessing
**File**: `src/data/preprocessor.py`
**Lines**: 250-350
**Tests**: 322 lines, 28+ cases

**Classes & Functions**:
- `Preprocessor`: Main class
- `create_windows(data, lookback=60, step=1)` → array
- `fit_input_scaler(data)`, `fit_output_scaler(data)`
- `transform(data)`, `inverse_transform(data)`
- `generate_targets(data, horizons=[1, 5, 15])` → dict
- `split_data(data, train=0.7, val=0.15, test=0.15)` → tuple

**Key Features**:
- Sliding windows with 60-minute lookback
- StandardScaler (separate for input/output)
- Multi-horizon target generation (h0, h1, h2)
- Temporal train/val/test split

**Run Tests**:
```bash
pytest tests/unit/test_preprocessor.py -v
```

---

#### Day 6: Dataset Creation
**File**: `src/data/dataset.py`
**Lines**: 200-300
**Tests**: 361 lines, 30+ cases

**Functions**:
- `create_tf_dataset(features, targets, batch_size, shuffle)` → tf.data.Dataset
- `get_train_val_test_datasets(data, config)` → tuple
- `add_gaussian_noise(data, noise_std, seed)` → array
- `window_generator(data, window_size, stride)` → array

**Key Features**:
- tf.data.Dataset with prefetching
- Batch size 144 (as per SRS)
- Multi-output support (9 outputs)
- Gaussian noise for uncertainty

**Run Tests**:
```bash
pytest tests/unit/test_dataset.py -v
```

---

### Phase 4: Learnable Indicators (Days 7-8) - MEDIUM

**File**: `src/data/indicators.py` (full implementation)
**Lines**: 400-600
**Tests**: 397 lines, 35+ cases

**5 Indicator Classes (Keras Layers)**:

1. **LearnableMA** (3 learnable params)
   - 3 MA periods as trainable tf.Variable
   - Compute moving averages with learnable windows

2. **LearnableMacd** (9 learnable params)
   - 3 MACD settings × 3 params (fast, slow, signal)
   - Standard MACD: [12, 26, 9], [5, 35, 5], [19, 39, 9]

3. **LearnableCustomMacd** (9 learnable params)
   - 3 custom pairs × 3 params
   - Custom pairs: [8, 17, 9], [10, 20, 5], [15, 30, 10]

4. **LearnableRSI** (3 learnable params)
   - 3 RSI periods: [9, 21, 30]
   - RSI calculation with learnable windows

5. **LearnableBollingerBands** (3 learnable params)
   - 3 BB periods: [10, 20, 30]
   - Upper, middle, lower bands

6. **LearnableMomentum** (3 learnable params)
   - 3 momentum periods: [5, 10, 15]

**Total**: 30 learnable parameters

**Key Features**:
- Each indicator is a Keras Layer
- Parameters as `tf.Variable(trainable=True)`
- L2 regularization on learnable params
- Auto-registration via `@INDICATOR_REGISTRY.register()`

**Run Tests**:
```bash
pytest tests/unit/test_indicators.py -v
```

**Verify**: `len(model.trainable_variables)` includes indicator params

---

### Phase 5: Model Components (Days 9-11) - COMPLEX

#### Day 9: Transformer & LSTM

**File 1**: `src/models/transformer_block.py`
**Lines**: 250-350

**TransformerBlock Class** (Keras Layer):
- Multi-head attention (4 heads)
- Feed-forward network (FFN)
- Layer normalization (×2)
- Residual connections (×2)
- Dropout

```python
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model=128, num_heads=4, dff=512, dropout=0.2):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads, d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        # Multi-head attention + residual
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # FFN + residual
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
```

**File 2**: `src/models/lstm_block.py`
**Lines**: 150-250

**LSTMBlock Class** (Keras Layer):
- Bidirectional LSTM (as per SRS)
- Multiple layers (2 by default)
- Dropout

```python
class LSTMBlock(tf.keras.layers.Layer):
    def __init__(self, units=128, num_layers=2, dropout=0.2, bidirectional=True):
        super().__init__()
        self.lstm_layers = []
        for i in range(num_layers):
            lstm = tf.keras.layers.LSTM(
                units,
                return_sequences=True,
                dropout=dropout
            )
            if bidirectional:
                lstm = tf.keras.layers.Bidirectional(lstm)
            self.lstm_layers.append(lstm)

    def call(self, x, training=False):
        for lstm in self.lstm_layers:
            x = lstm(x, training=training)
        return x
```

**Run Tests**:
```bash
pytest tests/unit/test_model_components.py::TestTransformerBlock -v
pytest tests/unit/test_model_components.py::TestLSTMBlock -v
```

---

#### Day 10: Indicator Subnet

**File**: `src/models/indicator_subnet.py`
**Lines**: 150-200

**IndicatorSubnet Class** (Keras Layer):
- Integrates all learnable indicators
- MLP: [64, 32] → output_dim=20

```python
class IndicatorSubnet(tf.keras.layers.Layer):
    def __init__(self, indicators, hidden_units=[64, 32], output_dim=20):
        super().__init__()
        self.indicators = indicators  # List of learnable indicator layers
        self.dense1 = tf.keras.layers.Dense(hidden_units[0], activation='relu')
        self.dense2 = tf.keras.layers.Dense(hidden_units[1], activation='relu')
        self.output_layer = tf.keras.layers.Dense(output_dim)

    def call(self, x, training=False):
        # Apply all indicators
        indicator_outputs = [ind(x) for ind in self.indicators]
        # Concatenate all indicator features
        combined = tf.concat(indicator_outputs, axis=-1)
        # MLP
        x = self.dense1(combined)
        x = self.dense2(x)
        return self.output_layer(x)
```

**Run Tests**:
```bash
pytest tests/unit/test_model_components.py::TestIndicatorSubnet -v
```

---

#### Day 11: Hybrid Model (Most Critical!)

**File**: `src/models/hybrid_model.py`
**Lines**: 600-800
**Complexity**: COMPLEX

**Architecture**:
```
Input (batch, 60, features)
    ↓
[Transformer Block] → Global dependencies
    ↓
[LSTM Block] → Sequential patterns
    ↓
[Indicator Subnet] → Learnable indicators
    ↓
Concatenate all features
    ↓
┌─────────────┬─────────────┬─────────────┐
│  Tower h0   │  Tower h1   │  Tower h2   │
│  Dense(128) │  Dense(128) │  Dense(128) │
│  Dense(64)  │  Dense(64)  │  Dense(64)  │
│  Dropout    │  Dropout    │  Dropout    │
└─────────────┴─────────────┴─────────────┘
    ↓              ↓              ↓
  price_h0       price_h1       price_h2      (linear)
  direction_h0   direction_h1   direction_h2  (sigmoid)
  variance_h0    variance_h1    variance_h2   (softplus)
```

**Total Outputs**: 9 (3 towers × 3 outputs each)

**HybridModel Class**:
```python
class HybridModel(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        # Backbone
        self.transformer = TransformerBlock(...)
        self.lstm = LSTMBlock(...)
        self.indicator_subnet = IndicatorSubnet(...)

        # 3 independent towers
        self.towers = []
        for h in range(3):  # h0, h1, h2
            tower = {
                'dense1': tf.keras.layers.Dense(128, activation='relu'),
                'dense2': tf.keras.layers.Dense(64, activation='relu'),
                'dropout': tf.keras.layers.Dropout(0.2),
                'price': tf.keras.layers.Dense(1, name=f'price_h{h}'),
                'direction': tf.keras.layers.Dense(1, activation='sigmoid', name=f'direction_h{h}'),
                'variance': tf.keras.layers.Dense(1, activation='softplus', name=f'variance_h{h}')
            }
            self.towers.append(tower)

    def call(self, inputs, training=False):
        # Backbone
        x = self.transformer(inputs, training=training)
        x = self.lstm(x, training=training)
        indicator_features = self.indicator_subnet(inputs, training=training)

        # Combine features
        combined = tf.concat([x, indicator_features], axis=-1)

        # 3 independent towers
        outputs = {}
        for h, tower in enumerate(self.towers):
            # Tower processing
            tower_out = tower['dense1'](combined)
            tower_out = tower['dense2'](tower_out)
            tower_out = tower['dropout'](tower_out, training=training)

            # 3 outputs per tower
            outputs[f'price_h{h}'] = tower['price'](tower_out)
            outputs[f'direction_h{h}'] = tower['direction'](tower_out)
            outputs[f'variance_h{h}'] = tower['variance'](tower_out)

        return outputs

def build_model(config):
    """Factory function to build model from config."""
    model = HybridModel(config)
    return model
```

**Run Tests**:
```bash
pytest tests/unit/test_model_components.py::TestHybridModel -v
pytest tests/unit/test_model_components.py::TestBuildModel -v
```

**Critical Validation**:
- Output shape: 9 tensors
- Trainable params include indicators
- `model.summary()` shows architecture

---

### Phase 6: Losses & Metrics (Day 12) - MEDIUM

#### Custom Losses

**File**: `src/losses/custom_losses.py`
**Lines**: 400-500
**Tests**: 424 lines, 40+ cases

**5 Loss Classes**:

1. **FocalLoss** (α=0.7, γ=1.0)
```python
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.7, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # Focal loss formula
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        focal_weight = self.alpha * tf.pow(1 - pt, self.gamma)
        ce = -tf.math.log(pt + 1e-7)
        return tf.reduce_mean(focal_weight * ce)
```

2. **HuberLoss** (δ=1.0)
3. **NegativeLogLikelihood** (for variance)
4. **TrendLoss** (MSE on trends)
5. **CompositeLoss** (weighted combination)

**CompositeLoss** (Critical):
```python
class CompositeLoss(tf.keras.losses.Loss):
    def __init__(self, loss_config):
        super().__init__()
        self.losses = {}
        self.weights = {}

        # Build from config
        for name, cfg in loss_config.items():
            loss_type = cfg['type']
            weight = cfg['weight']

            if loss_type == 'focal':
                self.losses[name] = FocalLoss(cfg['alpha'], cfg['gamma'])
            elif loss_type == 'huber':
                self.losses[name] = HuberLoss(cfg['delta'])
            # ... etc

            self.weights[name] = weight

    def call(self, y_true, y_pred):
        total_loss = 0.0
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(y_true[name], y_pred[name])
            total_loss += self.weights[name] * loss_value
        return total_loss
```

**Run Tests**:
```bash
pytest tests/unit/test_losses.py -v
```

---

#### Custom Metrics

**File**: `src/metrics/custom_metrics.py`
**Lines**: 400-500
**Tests**: 492 lines, 45+ cases

**6 Metric Classes**:

1. **DirectionAccuracy**
```python
class DirectionAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='direction_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        correct = tf.reduce_sum(tf.cast(y_true == y_pred_binary, tf.float32))
        self.correct.assign_add(correct)
        self.total.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.correct / self.total

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)
```

2. **DirectionF1Score**
3. **DirectionMCC** (Matthews Correlation Coefficient)
4. **PriceMAE**
5. **PriceMAPE**
6. **MultiHorizonMetric** (aggregator)

**Run Tests**:
```bash
pytest tests/unit/test_metrics.py -v
```

---

### Phase 7: Training Infrastructure (Days 13-15) - MEDIUM-COMPLEX

#### Day 13: Callbacks

**File**: `src/training/callbacks.py`
**Lines**: 250-350

**Key Callback**: IndicatorParamsLogger
```python
class IndicatorParamsLogger(tf.keras.callbacks.Callback):
    def __init__(self, output_file):
        super().__init__()
        self.output_file = output_file
        self.param_history = []

    def on_epoch_end(self, epoch, logs=None):
        # Extract indicator parameters
        params = {'epoch': epoch}
        for var in self.model.trainable_variables:
            if 'indicator' in var.name:
                params[var.name] = float(var.numpy())

        self.param_history.append(params)

        # Save to CSV
        df = pd.DataFrame(self.param_history)
        df.to_csv(self.output_file, index=False)
```

**create_callbacks() Function**:
```python
def create_callbacks(config, log_dir='logs'):
    callbacks = []

    # Early stopping (monitors val_dir_mcc_h1)
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_dir_mcc_h1',
        patience=40,
        mode='max',
        restore_best_weights=True
    ))

    # Model checkpoint
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        filepath='models_saved/model_{epoch:02d}_{val_dir_mcc_h1:.4f}.h5',
        monitor='val_dir_mcc_h1',
        save_best_only=True,
        mode='max'
    ))

    # TensorBoard
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    ))

    # Indicator params logger
    callbacks.append(IndicatorParamsLogger(
        output_file='logs/indicator_params_history.csv'
    ))

    return callbacks
```

**Run Tests**:
```bash
pytest tests/unit/test_training.py::TestCallbacks -v
```

---

#### Days 14-15: Trainer (Orchestrator)

**File**: `src/training/trainer.py`
**Lines**: 500-700
**Complexity**: COMPLEX

**Trainer Class**:
```python
class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.input_scaler = None
        self.output_scaler = None

    def load_datasets(self, data):
        """Load and prepare train/val/test datasets."""
        from src.data.dataset import get_train_val_test_datasets
        return get_train_val_test_datasets(data, self.config)

    def build_model(self):
        """Build model from configuration."""
        from src.models.hybrid_model import build_model
        self.model = build_model(self.config)
        return self.model

    def compile_model(self, model):
        """Compile model with losses and metrics."""
        from src.losses.custom_losses import CompositeLoss
        from src.metrics.custom_metrics import DirectionAccuracy, DirectionMCC, PriceMAE

        # Build losses
        loss = CompositeLoss(self.config['losses'])

        # Build metrics per output
        metrics = {}
        for h in [0, 1, 2]:
            metrics[f'price_h{h}'] = [PriceMAE(name=f'price_mae_h{h}')]
            metrics[f'direction_h{h}'] = [
                DirectionAccuracy(name=f'dir_acc_h{h}'),
                DirectionMCC(name=f'dir_mcc_h{h}')
            ]

        # Optimizer with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config['training']['learning_rate'],
            clipnorm=self.config['training']['gradient_clip_norm']
        )

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def fit(self, model, train_ds, val_ds):
        """Train model."""
        from src.training.callbacks import create_callbacks

        callbacks = create_callbacks(self.config)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config['training']['epochs'],
            callbacks=callbacks,
            verbose=1
        )

        # Save training log
        df = pd.DataFrame(history.history)
        df.to_csv('logs/training_log.csv', index=False)

        return history

    def save_weights(self, model, path):
        """Save model weights."""
        model.save_weights(path)

    def save_scalers(self, input_scaler, output_scaler, output_dir):
        """Save scalers."""
        import joblib
        joblib.dump(input_scaler, f'{output_dir}/scaler_input.joblib')
        joblib.dump(output_scaler, f'{output_dir}/scaler.joblib')
```

**Run Tests**:
```bash
pytest tests/unit/test_training.py::TestTrainer -v
```

**Integration Test**:
```bash
pytest tests/integration/test_end_to_end.py::TestTrainingPipeline -v
```

---

### Phase 8: Inference & Signals (Days 16-18) - MEDIUM

#### Day 16: Predictor

**File**: `src/inference/predictor.py`
**Lines**: 300-400

**Predictor Class**:
```python
class Predictor:
    def __init__(self, model_path, scaler_input_path, scaler_output_path, config):
        self.model_path = model_path
        self.scaler_input_path = scaler_input_path
        self.scaler_output_path = scaler_output_path
        self.config = config
        self.model = None
        self.input_scaler = None
        self.output_scaler = None

    def load_model(self):
        """Load trained model."""
        from src.models.hybrid_model import build_model

        self.model = build_model(self.config)
        self.model.load_weights(self.model_path)
        return self.model

    def load_scalers(self):
        """Load scalers."""
        import joblib

        self.input_scaler = joblib.load(self.scaler_input_path)
        self.output_scaler = joblib.load(self.scaler_output_path)
        return self.input_scaler, self.output_scaler

    def predict(self, data):
        """Generate predictions."""
        # Scale input
        if self.input_scaler:
            data = self.input_scaler.transform(data)

        # Predict
        predictions = self.model.predict(data)

        # Inverse scale outputs
        if self.output_scaler:
            for key in ['price_h0', 'price_h1', 'price_h2']:
                predictions[key] = self.output_scaler.inverse_transform(predictions[key])

        return predictions
```

**Run Tests**:
```bash
pytest tests/unit/test_inference.py::TestPredictor -v
```

---

#### Day 17: Signal Generation

**File**: `src/inference/signals.py`
**Lines**: 250-350

**generate_signals() Function**:
```python
def generate_signals(predictions, current_price, config):
    """Generate trading signals from predictions."""
    from src.utils.helper_functions import (
        calculate_confidence,
        calculate_signal_strength,
        check_multi_horizon_agreement,
        detect_variance_spike,
        calculate_profit_targets,
        calculate_dynamic_stop_loss,
        calculate_position_size_multiplier
    )

    # Extract predictions for h0, h1, h2
    price_predictions = [
        predictions['price_h0'][0],
        predictions['price_h1'][0],
        predictions['price_h2'][0]
    ]

    direction_probs = [
        predictions['direction_h0'][0],
        predictions['direction_h1'][0],
        predictions['direction_h2'][0]
    ]

    variances = [
        predictions['variance_h0'][0],
        predictions['variance_h1'][0],
        predictions['variance_h2'][0]
    ]

    # Calculate confidence from variance
    confidences = [calculate_confidence(v) for v in variances]

    # Check multi-horizon agreement
    is_agreed, agreement = check_multi_horizon_agreement(
        price_predictions,
        current_price,
        config['signals']['agreement_threshold']
    )

    # Calculate signal strength
    signal_strength = calculate_signal_strength(
        direction_probs[1],  # Use h1 (5-min) as primary
        confidences[1]
    )

    # Position sizing
    position_size_multiplier = calculate_position_size_multiplier(
        confidences[1],
        config['signals']['size_high'],
        config['signals']['size_normal'],
        config['signals']['size_low'],
        config['signals']['conf_high_thresh'],
        config['signals']['conf_low_thresh']
    )

    # Profit targets (3-tier)
    profit_targets = calculate_profit_targets(
        current_price,
        price_predictions
    )

    # Dynamic stop loss
    variance_rolling_mean = np.mean(variances)  # Simplified
    position_type = 'LONG' if price_predictions[1] > current_price else 'SHORT'

    stop_loss = calculate_dynamic_stop_loss(
        current_price,
        position_type,
        variances[1],
        variance_rolling_mean,
        config['signals']['base_stop_pct'],
        config['signals']['max_variance_multiplier']
    )

    # Variance spike detection
    variance_rolling_std = np.std(variances)  # Simplified
    variance_spike = detect_variance_spike(
        variances[1],
        variance_rolling_mean,
        variance_rolling_std,
        config['signals']['spike_threshold']
    )

    return {
        'signal_strength': signal_strength,
        'position_size_multiplier': position_size_multiplier,
        'multi_horizon_agreement': is_agreed,
        'agreement_score': agreement,
        'variance_spike_detected': variance_spike,
        'position_type': position_type,
        'stop_loss': stop_loss,
        'take_profit_1': profit_targets['tp1'],
        'take_profit_2': profit_targets['tp2'],
        'take_profit_3': profit_targets['tp3'],
        'tp1_pct': profit_targets['tp1_pct'],
        'tp2_pct': profit_targets['tp2_pct'],
        'tp3_pct': profit_targets['tp3_pct'],
        'confidences': confidences,
        'predicted_prices': price_predictions
    }
```

**Run Tests**:
```bash
pytest tests/unit/test_inference.py::TestSignalGeneration -v
```

---

#### Day 18: Backtesting

**File**: `src/inference/backtesting.py`
**Lines**: 400-600

**Trade Dataclass**:
```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    position_type: str  # 'LONG' or 'SHORT'
    size: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    confidence: float
    variance: float
    status: str = 'OPEN'  # 'OPEN', 'CLOSED_WIN', 'CLOSED_LOSS'
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
```

**BacktestEngine Class**:
```python
class BacktestEngine:
    def __init__(self, initial_capital=10000.0, commission=0.001, slippage=0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.trades = []

    def simulate(self, historical_data, predictions):
        """Simulate trades based on predictions."""
        for i, row in historical_data.iterrows():
            # Check for entry signals
            if self._should_enter(predictions[i]):
                trade = self._open_trade(row, predictions[i])
                self.trades.append(trade)

            # Check open trades for exits
            for trade in self.trades:
                if trade.status == 'OPEN':
                    self._check_exit(trade, row)

        return self.trades

    def _should_enter(self, signals):
        """Entry logic."""
        return (
            signals['multi_horizon_agreement'] and
            signals['signal_strength'] > 0.6 and
            not signals['variance_spike_detected']
        )

    def _open_trade(self, row, signals):
        """Open new trade."""
        return Trade(
            entry_time=row['datetime'],
            entry_price=row['close'],
            position_type=signals['position_type'],
            size=1.0 * signals['position_size_multiplier'],
            stop_loss=signals['stop_loss'],
            take_profit_1=signals['take_profit_1'],
            take_profit_2=signals['take_profit_2'],
            take_profit_3=signals['take_profit_3'],
            confidence=signals['confidences'][1],
            variance=signals['variances'][1]
        )

    def _check_exit(self, trade, row):
        """Check for exit conditions."""
        current_price = row['close']

        # Check stop loss
        if trade.position_type == 'LONG' and current_price <= trade.stop_loss:
            self._close_trade(trade, row, 'STOP_LOSS')
        elif trade.position_type == 'SHORT' and current_price >= trade.stop_loss:
            self._close_trade(trade, row, 'STOP_LOSS')

        # Check take profits (3-tier)
        elif trade.position_type == 'LONG':
            if current_price >= trade.take_profit_3:
                self._close_trade(trade, row, 'TP3')
            elif current_price >= trade.take_profit_2:
                self._partial_close(trade, row, 0.67, 'TP2')
            elif current_price >= trade.take_profit_1:
                self._partial_close(trade, row, 0.33, 'TP1')

    def _close_trade(self, trade, row, reason):
        """Close trade and calculate PnL."""
        trade.exit_time = row['datetime']
        trade.exit_price = row['close']

        if trade.position_type == 'LONG':
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.size
        else:
            trade.pnl = (trade.entry_price - trade.exit_price) * trade.size

        # Apply commission
        trade.pnl -= (trade.entry_price + trade.exit_price) * trade.size * self.commission

        trade.pnl_pct = (trade.pnl / (trade.entry_price * trade.size)) * 100
        trade.status = 'CLOSED_WIN' if trade.pnl > 0 else 'CLOSED_LOSS'

    def _partial_close(self, trade, row, close_pct, reason):
        """Close partial position at profit target."""
        # Reduce size
        trade.size *= (1 - close_pct)

def calculate_performance(trades):
    """Calculate backtest performance metrics."""
    closed_trades = [t for t in trades if t.status != 'OPEN']

    if not closed_trades:
        return {}

    total_pnl = sum(t.pnl for t in closed_trades)
    winning_trades = [t for t in closed_trades if t.status == 'CLOSED_WIN']
    losing_trades = [t for t in closed_trades if t.status == 'CLOSED_LOSS']

    win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0.0
    avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
    avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0

    return {
        'total_pnl': total_pnl,
        'total_pnl_pct': (total_pnl / 10000.0) * 100,  # Assuming 10k initial capital
        'total_trades': len(closed_trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else float('inf')
    }
```

**Run Tests**:
```bash
pytest tests/unit/test_inference.py::TestBacktesting -v
pytest tests/integration/test_end_to_end.py::TestBacktestingPipeline -v
```

---

## Final Integration Testing (Days 19-21)

### Day 19: Full Unit Test Suite
```bash
# Run all unit tests
pytest tests/unit/ -v --tb=short

# Generate coverage report
pytest tests/unit/ --cov=src --cov-report=html --cov-report=term

# Target: 340+ tests passing, >80% coverage
```

### Day 20: Integration Tests
```bash
# Run all integration tests
pytest tests/integration/ -v

# Run slow tests (training, backtesting)
pytest -m slow -v

# Target: All integration tests passing
```

### Day 21: End-to-End Validation
```bash
# Run complete pipeline
pytest tests/integration/test_end_to_end.py::TestFullSystemIntegration -v

# Final validation
pytest --tb=short --maxfail=1
```

---

## Critical Success Factors

### 1. Shape Validation
Always verify tensor shapes at integration points:
```python
# Example
print(f"Input shape: {input_data.shape}")  # (batch, 60, features)
print(f"Output shape: {predictions['price_h0'].shape}")  # (batch, 1)
```

### 2. Trainable Parameters
Verify indicator parameters are trainable:
```python
indicator_params = [v for v in model.trainable_variables if 'indicator' in v.name]
print(f"Learnable indicator params: {len(indicator_params)}")  # Should be 30+
```

### 3. Loss & Metric Integration
Test each loss/metric independently before integration:
```python
# Test individual loss
focal_loss = FocalLoss(alpha=0.7, gamma=1.0)
loss_value = focal_loss(y_true, y_pred)
assert tf.math.is_finite(loss_value)
```

### 4. Incremental Testing
Run tests after each module:
```bash
# After implementing helper_functions.py
pytest tests/unit/test_helper_functions.py -v

# After implementing config_parser.py
pytest tests/unit/test_config.py -v

# And so on...
```

---

## 5 Most Critical Files

Based on complexity and integration dependencies:

1. **`src/models/hybrid_model.py`** (600-800 lines)
   - Integrates all components
   - 3 towers, 9 outputs
   - Most complex architecture

2. **`src/training/trainer.py`** (500-700 lines)
   - Orchestrates entire training pipeline
   - Critical for end-to-end functionality

3. **`src/data/indicators.py`** (400-600 lines)
   - 30+ learnable parameters
   - Core innovation of system
   - Must integrate with model

4. **`src/losses/custom_losses.py`** (400-500 lines)
   - CompositeLoss handles multi-output
   - Critical for model convergence

5. **`src/data/dataset.py`** (200-300 lines)
   - Feeds entire training pipeline
   - Shape mismatches cascade through system

---

## Progress Tracking Template

```markdown
# Implementation Progress

## Phase 1: Foundation (Days 1-2)
- [ ] helper_functions.py (0/50 tests passing)
- [ ] config_parser.py (0/35 tests passing)

## Phase 2: Registries (Day 3)
- [ ] indicator_registry (0/10 tests passing)
- [ ] loss_registry (0/10 tests passing)
- [ ] metric_registry (0/10 tests passing)

## Phase 3: Data Processing (Days 4-6)
- [ ] data_loader.py (0/25 tests passing)
- [ ] preprocessor.py (0/28 tests passing)
- [ ] dataset.py (0/30 tests passing)

## Phase 4: Learnable Indicators (Days 7-8)
- [ ] indicators.py (0/35 tests passing)

## Phase 5: Model Components (Days 9-11)
- [ ] transformer_block.py (0/15 tests passing)
- [ ] lstm_block.py (0/10 tests passing)
- [ ] indicator_subnet.py (0/10 tests passing)
- [ ] hybrid_model.py (0/20 tests passing)

## Phase 6: Losses & Metrics (Day 12)
- [ ] custom_losses.py (0/40 tests passing)
- [ ] custom_metrics.py (0/45 tests passing)

## Phase 7: Training (Days 13-15)
- [ ] callbacks.py (0/15 tests passing)
- [ ] trainer.py (0/30 tests passing)

## Phase 8: Inference (Days 16-18)
- [ ] predictor.py (0/20 tests passing)
- [ ] signals.py (0/15 tests passing)
- [ ] backtesting.py (0/20 tests passing)

## Overall
- Modules Completed: 0/20
- Tests Passing: 0/340+
- Coverage: 0%
```

---

## Commands Reference

```bash
# Run specific test file
pytest tests/unit/test_helper_functions.py -v

# Run specific test class
pytest tests/unit/test_helper_functions.py::TestCalculateConfidence -v

# Run specific test
pytest tests/unit/test_helper_functions.py::TestCalculateConfidence::test_zero_variance_returns_max_confidence -v

# Run with markers
pytest -m "not slow" -v  # Skip slow tests
pytest -m integration -v  # Run only integration tests

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term

# Run with debugging
pytest --pdb -x  # Stop at first failure and enter debugger

# Run in parallel (if pytest-xdist installed)
pytest -n auto

# Show print statements
pytest -s

# Verbose output
pytest -vv
```

---

## Estimated Timeline

**With 2-3 developers working in parallel**:
- **Week 1**: Foundation + Data Processing (Days 1-6)
- **Week 2**: Indicators + Model Components + Losses/Metrics (Days 7-12)
- **Week 3**: Training + Inference (Days 13-18)
- **Week 4**: Integration & Validation (Days 19-21)

**Total**: ~21 working days (~4-5 weeks)

**With 1 developer**:
- ~30-35 working days (~6-7 weeks)

---

This implementation plan provides a complete roadmap from TDD tests to working system. Follow the dependency order, run tests incrementally, and ensure each phase passes before moving to the next.
