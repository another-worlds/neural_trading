"""Phase 4.2: LOW Priority Tests - Training Callbacks Final Paths

Tests to achieve 100% coverage on training callbacks module.

Target Coverage:
- src/training/callbacks.py: 94.6% → 100%

Missing Lines to Cover:
- 81, 92
"""
import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tempfile import TemporaryDirectory

from src.training.callbacks import (
    IndicatorParamsLogger,
    create_callbacks
)


class TestIndicatorParamsLoggerScalarHandling:
    """Test IndicatorParamsLogger with scalar values (line 81)."""

    def test_indicator_logger_scalar_weight(self):
        """Test logging scalar indicator weight (line 81)."""
        with TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / 'indicator_params.csv'

            # Create callback
            callback = IndicatorParamsLogger(output_file=output_file)

            # Create a simple model with a scalar weight that looks like an indicator
            inputs = tf.keras.Input(shape=(10,))
            # Create a layer with a scalar weight
            layer = tf.keras.layers.Dense(5, name='period_layer')
            outputs = layer(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Manually add a scalar indicator weight for testing
            scalar_weight = model.add_weight(
                name='ma_period_scalar',
                shape=(),  # Scalar shape
                initializer=tf.keras.initializers.Constant(15.0),
                trainable=True
            )

            # Set model
            callback.set_model(model)

            # Extract params - should handle scalar (line 81)
            params = callback._extract_indicator_params()

            # Verify scalar was extracted
            assert 'ma_period_scalar' in params
            assert isinstance(params['ma_period_scalar'], float)
            assert params['ma_period_scalar'] == 15.0

    def test_indicator_logger_array_and_scalar_mixed(self):
        """Test logging with both array and scalar indicator weights."""
        with TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / 'indicator_params.csv'

            callback = IndicatorParamsLogger(output_file=output_file)

            # Create model
            inputs = tf.keras.Input(shape=(10,))
            outputs = tf.keras.layers.Dense(5)(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Add scalar weight (line 81 path)
            scalar_weight = model.add_weight(
                name='rsi_period',
                shape=(),
                initializer=tf.keras.initializers.Constant(14.0),
                trainable=True
            )

            # Add array weight
            array_weight = model.add_weight(
                name='ma_periods',
                shape=(3,),
                initializer=tf.keras.initializers.Constant([5.0, 15.0, 30.0]),
                trainable=True
            )

            callback.set_model(model)

            # Extract params
            params = callback._extract_indicator_params()

            # Verify scalar (line 81)
            assert 'rsi_period' in params
            assert params['rsi_period'] == 14.0

            # Verify array elements
            assert 'ma_periods_0' in params
            assert 'ma_periods_1' in params
            assert 'ma_periods_2' in params


class TestIndicatorParamsLoggerEmptyHistory:
    """Test IndicatorParamsLogger with empty history (line 92)."""

    def test_indicator_logger_save_empty_history(self):
        """Test _save_to_csv with empty history does nothing (line 92)."""
        with TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / 'indicator_params.csv'

            # Create callback
            callback = IndicatorParamsLogger(output_file=output_file)

            # Don't log any epochs - history is empty
            assert len(callback.params_history) == 0

            # Call save - should return early (line 92)
            callback._save_to_csv()

            # File should not be created
            assert not output_file.exists()

    def test_indicator_logger_empty_then_log(self):
        """Test that saving after logging creates file."""
        with TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / 'indicator_params.csv'

            callback = IndicatorParamsLogger(output_file=output_file)

            # Initially empty
            assert len(callback.params_history) == 0
            callback._save_to_csv()
            assert not output_file.exists()

            # Add some data
            callback.log_epoch(epoch=0, params={'param1': 1.0, 'param2': 2.0})

            # Now save should work
            callback._save_to_csv()
            assert output_file.exists()

            # Verify content
            df = pd.read_csv(output_file)
            assert len(df) == 1
            assert 'epoch' in df.columns


class TestIndicatorParamsLoggerOnEpochEnd:
    """Test on_epoch_end callback method."""

    def test_indicator_logger_on_epoch_end(self):
        """Test on_epoch_end extracts and logs parameters."""
        with TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / 'indicator_params.csv'

            callback = IndicatorParamsLogger(output_file=output_file)

            # Create model with indicator-like weights
            inputs = tf.keras.Input(shape=(10,))
            outputs = tf.keras.layers.Dense(5)(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Add indicator weights
            period_weight = model.add_weight(
                name='ma_period',
                shape=(),
                initializer=tf.keras.initializers.Constant(15.0),
                trainable=True
            )

            callback.set_model(model)

            # Simulate epoch end
            callback.on_epoch_end(epoch=0, logs={})
            callback.on_epoch_end(epoch=1, logs={})

            # Verify history
            assert len(callback.params_history) == 2
            assert callback.params_history[0]['epoch'] == 0
            assert callback.params_history[1]['epoch'] == 1

    def test_indicator_logger_on_train_end(self):
        """Test on_train_end saves to CSV."""
        with TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / 'indicator_params.csv'

            callback = IndicatorParamsLogger(output_file=output_file)

            # Log some epochs
            callback.log_epoch(epoch=0, params={'param1': 1.0})
            callback.log_epoch(epoch=1, params={'param1': 1.5})
            callback.log_epoch(epoch=2, params={'param1': 2.0})

            # Trigger train end
            callback.on_train_end(logs={})

            # File should be created
            assert output_file.exists()

            # Verify content
            df = pd.read_csv(output_file)
            assert len(df) == 3
            assert list(df['epoch']) == [0, 1, 2]


class TestSetupCallbacks:
    """Test create_callbacks function."""

    def test_create_callbacks_basic(self):
        """Test create_callbacks returns list of callbacks."""
        with TemporaryDirectory() as tmpdir:
            config = {
                'training': {
                    'patience': 10,
                    'monitor': 'val_loss'
                },
                'paths': {
                    'models_dir': str(tmpdir),
                    'logs_dir': str(tmpdir)
                }
            }

            callbacks = create_callbacks(config)

            # Should return a list
            assert isinstance(callbacks, list)
            assert len(callbacks) > 0

            # Should contain various callback types
            callback_types = [type(cb).__name__ for cb in callbacks]

            # Common callbacks
            assert any('EarlyStopping' in name or 'early' in name.lower() for name in callback_types)

    def test_create_callbacks_with_tensorboard(self):
        """Test create_callbacks includes TensorBoard callback."""
        with TemporaryDirectory() as tmpdir:
            config = {
                'training': {
                    'patience': 10,
                    'monitor': 'val_loss',
                    'use_tensorboard': True
                },
                'paths': {
                    'models_dir': str(tmpdir),
                    'logs_dir': str(tmpdir)
                }
            }

            callbacks = create_callbacks(config)

            callback_types = [type(cb).__name__ for cb in callbacks]

            # Should have TensorBoard callback
            assert any('TensorBoard' in name for name in callback_types)


class TestCallbackIntegration:
    """Test callbacks in training context."""

    def test_indicator_logger_full_training_cycle(self):
        """Test IndicatorParamsLogger through full training."""
        with TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / 'indicator_params.csv'

            # Create simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, input_shape=(5,)),
                tf.keras.layers.Dense(1)
            ])

            # Add indicator-like weight
            indicator_weight = model.layers[0].add_weight(
                name='custom_period',
                shape=(),
                initializer=tf.keras.initializers.Constant(20.0),
                trainable=True
            )

            model.compile(optimizer='adam', loss='mse')

            # Create callback
            callback = IndicatorParamsLogger(output_file=output_file)

            # Generate dummy data
            X_train = np.random.randn(100, 5).astype(np.float32)
            y_train = np.random.randn(100, 1).astype(np.float32)

            # Train with callback
            model.fit(
                X_train, y_train,
                epochs=3,
                batch_size=16,
                verbose=0,
                callbacks=[callback]
            )

            # Verify file was created
            assert output_file.exists()

            # Verify content
            df = pd.read_csv(output_file)
            assert len(df) == 3  # 3 epochs
            assert 'epoch' in df.columns
            assert 'custom_period' in df.columns

    def test_callbacks_work_together(self):
        """Test multiple callbacks work together."""
        with TemporaryDirectory() as tmpdir:
            # Create model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, input_shape=(5,)),
                tf.keras.layers.Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')

            # Create multiple callbacks
            checkpoint_path = Path(tmpdir) / 'checkpoint.keras'
            log_path = Path(tmpdir) / 'indicator_log.csv'

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    str(checkpoint_path),
                    save_best_only=False
                ),
                IndicatorParamsLogger(output_file=log_path),
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss',
                    patience=100  # Won't trigger
                )
            ]

            # Generate data
            X_train = np.random.randn(50, 5).astype(np.float32)
            y_train = np.random.randn(50, 1).astype(np.float32)

            # Train
            model.fit(
                X_train, y_train,
                epochs=2,
                batch_size=16,
                verbose=0,
                callbacks=callbacks
            )

            # Verify both callbacks worked
            assert checkpoint_path.exists()  # ModelCheckpoint created file
            # Note: log_path might not exist if no indicator weights present


class TestCallbackEdgeCases:
    """Test callback edge cases."""

    def test_indicator_logger_no_indicator_weights(self):
        """Test IndicatorParamsLogger when model has no indicator weights."""
        with TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / 'indicator_params.csv'

            callback = IndicatorParamsLogger(output_file=output_file)

            # Create model with no indicator-like weights
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, input_shape=(5,)),
                tf.keras.layers.Dense(1)
            ])

            callback.set_model(model)

            # Extract params - should be empty or have no indicator params
            params = callback._extract_indicator_params()

            # No indicator parameters should be found
            assert len(params) == 0 or all('period' not in k.lower() for k in params.keys())

    def test_indicator_logger_log_epoch_manually(self):
        """Test manual epoch logging."""
        with TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / 'indicator_params.csv'

            callback = IndicatorParamsLogger(output_file=output_file)

            # Manually log epochs
            callback.log_epoch(0, {'param_a': 1.0, 'param_b': 2.0})
            callback.log_epoch(1, {'param_a': 1.5, 'param_b': 2.5})

            # Verify history
            assert len(callback.params_history) == 2
            assert callback.params_history[0]['param_a'] == 1.0
            assert callback.params_history[1]['param_b'] == 2.5

            # Save
            callback._save_to_csv()

            # Verify file
            df = pd.read_csv(output_file)
            assert len(df) == 2
            assert 'param_a' in df.columns
            assert 'param_b' in df.columns
