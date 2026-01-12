"""Phase 2.5: HIGH Priority Tests - Model Component Serialization

Tests to achieve 95%+ coverage on model component get_config methods.

Target Coverage:
- src/models/indicator_subnet.py: 92.3% → 98%
- src/models/lstm_block.py: 90.6% → 98%
- src/models/transformer_block.py: 89.3% → 98%

Missing Lines to Cover:
- indicator_subnet.py: 111-119 (get_config)
- lstm_block.py: 112-120 (get_config)
- transformer_block.py: 117-124 (get_config)
"""
import pytest
import tensorflow as tf
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path

from src.models.indicator_subnet import IndicatorSubnet
from src.models.lstm_block import LSTMBlock
from src.models.transformer_block import TransformerBlock


class TestIndicatorSubnetSerialization:
    """Test IndicatorSubnet get_config and serialization (lines 111-119)."""

    def test_indicator_subnet_get_config(self):
        """Test IndicatorSubnet get_config returns correct configuration."""
        # Create layer with specific config
        layer = IndicatorSubnet(
            num_indicators=30,
            hidden_units=[64, 32],
            output_dim=20,
            activation='relu',
            dropout_rate=0.3
        )

        # Build the layer
        layer.build(input_shape=(None, 30))

        # Get configuration
        config = layer.get_config()

        # Verify all parameters are in config
        assert 'num_indicators' in config
        assert 'hidden_units' in config
        assert 'output_dim' in config
        assert 'activation' in config
        assert 'dropout_rate' in config

        # Verify values match
        assert config['num_indicators'] == 30
        assert config['hidden_units'] == [64, 32]
        assert config['output_dim'] == 20
        assert config['activation'] == 'relu'
        assert config['dropout_rate'] == 0.3

    def test_indicator_subnet_from_config(self):
        """Test IndicatorSubnet can be reconstructed from config."""
        # Create original layer
        original = IndicatorSubnet(
            num_indicators=30,
            hidden_units=[128, 64],
            output_dim=25,
            activation='tanh',
            dropout_rate=0.4
        )
        original.build(input_shape=(None, 30))

        # Get config
        config = original.get_config()

        # Recreate layer from config
        reconstructed = IndicatorSubnet.from_config(config)
        reconstructed.build(input_shape=(None, 30))

        # Verify reconstructed layer has same config
        assert reconstructed.num_indicators == original.num_indicators
        assert reconstructed.hidden_units == original.hidden_units
        assert reconstructed.output_dim == original.output_dim
        assert reconstructed.activation == original.activation
        assert reconstructed.dropout_rate == original.dropout_rate

    def test_indicator_subnet_serialization_roundtrip(self):
        """Test IndicatorSubnet serialization and deserialization roundtrip."""
        # Create and build layer
        original = IndicatorSubnet(
            num_indicators=30,
            hidden_units=[64, 32],
            output_dim=20
        )

        # Create a simple model with the layer
        inputs = tf.keras.Input(shape=(30,))
        outputs = original(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Test with some data
        test_input = np.random.randn(2, 30).astype(np.float32)
        original_output = model.predict(test_input, verbose=0)

        # Serialize to JSON
        config = model.get_config()

        # Recreate model from config
        reconstructed_model = tf.keras.Model.from_config(
            config,
            custom_objects={'IndicatorSubnet': IndicatorSubnet}
        )

        # Copy weights
        reconstructed_model.set_weights(model.get_weights())

        # Test with same data
        reconstructed_output = reconstructed_model.predict(test_input, verbose=0)

        # Outputs should be identical
        np.testing.assert_allclose(original_output, reconstructed_output, rtol=1e-5)


class TestLSTMBlockSerialization:
    """Test LSTMBlock get_config and serialization (lines 112-120)."""

    def test_lstm_block_get_config_bidirectional(self):
        """Test LSTMBlock get_config with bidirectional LSTM."""
        # Create bidirectional LSTM block
        layer = LSTMBlock(
            units=128,
            num_layers=3,
            dropout=0.3,
            bidirectional=True,
            return_sequences=True
        )

        # Build the layer
        layer.build(input_shape=(None, 60, 50))

        # Get configuration
        config = layer.get_config()

        # Verify all parameters are in config
        assert 'units' in config
        assert 'num_layers' in config
        assert 'dropout' in config
        assert 'bidirectional' in config
        assert 'return_sequences' in config

        # Verify values match
        assert config['units'] == 128
        assert config['num_layers'] == 3
        assert config['dropout'] == 0.3
        assert config['bidirectional'] is True
        assert config['return_sequences'] is True

    def test_lstm_block_get_config_unidirectional(self):
        """Test LSTMBlock get_config with unidirectional LSTM."""
        # Create unidirectional LSTM block
        layer = LSTMBlock(
            units=64,
            num_layers=2,
            dropout=0.2,
            bidirectional=False,
            return_sequences=False
        )

        # Build the layer
        layer.build(input_shape=(None, 60, 50))

        # Get configuration
        config = layer.get_config()

        # Verify bidirectional is False
        assert config['bidirectional'] is False
        assert config['return_sequences'] is False

    def test_lstm_block_from_config(self):
        """Test LSTMBlock can be reconstructed from config."""
        # Create original layer
        original = LSTMBlock(
            units=256,
            num_layers=4,
            dropout=0.5,
            bidirectional=True,
            return_sequences=False
        )
        original.build(input_shape=(None, 60, 50))

        # Get config
        config = original.get_config()

        # Recreate layer from config
        reconstructed = LSTMBlock.from_config(config)
        reconstructed.build(input_shape=(None, 60, 50))

        # Verify reconstructed layer has same config
        assert reconstructed.units == original.units
        assert reconstructed.num_layers == original.num_layers
        assert reconstructed.dropout == original.dropout
        assert reconstructed.bidirectional == original.bidirectional
        assert reconstructed.return_sequences == original.return_sequences

    def test_lstm_block_serialization_roundtrip(self):
        """Test LSTMBlock serialization and deserialization roundtrip."""
        # Create and build layer
        original = LSTMBlock(
            units=128,
            num_layers=2,
            dropout=0.3,
            bidirectional=True
        )

        # Create a simple model with the layer
        inputs = tf.keras.Input(shape=(60, 50))
        outputs = original(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Test with some data
        test_input = np.random.randn(2, 60, 50).astype(np.float32)
        original_output = model.predict(test_input, verbose=0)

        # Serialize to JSON
        config = model.get_config()

        # Recreate model from config
        reconstructed_model = tf.keras.Model.from_config(
            config,
            custom_objects={'LSTMBlock': LSTMBlock}
        )

        # Copy weights
        reconstructed_model.set_weights(model.get_weights())

        # Test with same data
        reconstructed_output = reconstructed_model.predict(test_input, verbose=0)

        # Outputs should be identical
        np.testing.assert_allclose(original_output, reconstructed_output, rtol=1e-5)


class TestTransformerBlockSerialization:
    """Test TransformerBlock get_config and serialization (lines 117-124)."""

    def test_transformer_block_get_config(self):
        """Test TransformerBlock get_config returns correct configuration."""
        # Create layer with specific config
        layer = TransformerBlock(
            d_model=128,
            num_heads=8,
            dff=512,
            dropout_rate=0.1
        )

        # Build the layer
        layer.build(input_shape=(None, 60, 128))

        # Get configuration
        config = layer.get_config()

        # Verify all parameters are in config
        assert 'd_model' in config
        assert 'num_heads' in config
        assert 'dff' in config
        assert 'dropout_rate' in config

        # Verify values match
        assert config['d_model'] == 128
        assert config['num_heads'] == 8
        assert config['dff'] == 512
        assert config['dropout_rate'] == 0.1

    def test_transformer_block_get_config_different_heads(self):
        """Test TransformerBlock get_config with different head configurations."""
        # Test with 4 heads
        layer_4_heads = TransformerBlock(
            d_model=64,
            num_heads=4,
            dff=256,
            dropout_rate=0.2
        )
        layer_4_heads.build(input_shape=(None, 60, 64))
        config_4 = layer_4_heads.get_config()
        assert config_4['num_heads'] == 4

        # Test with 16 heads
        layer_16_heads = TransformerBlock(
            d_model=256,
            num_heads=16,
            dff=1024,
            dropout_rate=0.15
        )
        layer_16_heads.build(input_shape=(None, 60, 256))
        config_16 = layer_16_heads.get_config()
        assert config_16['num_heads'] == 16

    def test_transformer_block_from_config(self):
        """Test TransformerBlock can be reconstructed from config."""
        # Create original layer
        original = TransformerBlock(
            d_model=256,
            num_heads=16,
            dff=1024,
            dropout_rate=0.2
        )
        original.build(input_shape=(None, 60, 256))

        # Get config
        config = original.get_config()

        # Recreate layer from config
        reconstructed = TransformerBlock.from_config(config)
        reconstructed.build(input_shape=(None, 60, 256))

        # Verify reconstructed layer has same config
        assert reconstructed.d_model == original.d_model
        assert reconstructed.num_heads == original.num_heads
        assert reconstructed.dff == original.dff
        assert reconstructed.dropout_rate == original.dropout_rate

    def test_transformer_block_serialization_roundtrip(self):
        """Test TransformerBlock serialization and deserialization roundtrip."""
        # Create and build layer
        original = TransformerBlock(
            d_model=128,
            num_heads=8,
            dff=512,
            dropout_rate=0.1
        )

        # Create a simple model with the layer
        inputs = tf.keras.Input(shape=(60, 128))
        outputs = original(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Test with some data
        test_input = np.random.randn(2, 60, 128).astype(np.float32)
        original_output = model.predict(test_input, verbose=0)

        # Serialize to JSON
        config = model.get_config()

        # Recreate model from config
        reconstructed_model = tf.keras.Model.from_config(
            config,
            custom_objects={'TransformerBlock': TransformerBlock}
        )

        # Copy weights
        reconstructed_model.set_weights(model.get_weights())

        # Test with same data
        reconstructed_output = reconstructed_model.predict(test_input, verbose=0)

        # Outputs should be identical
        np.testing.assert_allclose(original_output, reconstructed_output, rtol=1e-5)


class TestAllBlocksSerialization:
    """Test serialization roundtrip for all blocks together."""

    def test_all_blocks_serialization_roundtrip(self):
        """Test complete model with all three blocks can be serialized."""
        # Create a model using all three custom blocks
        inputs = tf.keras.Input(shape=(60, 50), name='input')

        # Add transformer block
        x = TransformerBlock(
            d_model=50,
            num_heads=5,
            dff=200,
            dropout_rate=0.1
        )(inputs)

        # Add LSTM block
        x = LSTMBlock(
            units=64,
            num_layers=2,
            dropout=0.2,
            bidirectional=True,
            return_sequences=False
        )(x)

        # Add indicator subnet
        # First flatten to 1D if needed
        if len(x.shape) > 2:
            x = tf.keras.layers.Flatten()(x)

        # Expand to 30 features for indicator subnet
        x = tf.keras.layers.Dense(30)(x)

        outputs = IndicatorSubnet(
            num_indicators=30,
            hidden_units=[64, 32],
            output_dim=20
        )(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Test with some data
        test_input = np.random.randn(2, 60, 50).astype(np.float32)
        original_output = model.predict(test_input, verbose=0)

        # Serialize model config
        config = model.get_config()

        # Recreate model from config
        custom_objects = {
            'TransformerBlock': TransformerBlock,
            'LSTMBlock': LSTMBlock,
            'IndicatorSubnet': IndicatorSubnet
        }
        reconstructed_model = tf.keras.Model.from_config(config, custom_objects=custom_objects)

        # Copy weights
        reconstructed_model.set_weights(model.get_weights())

        # Test with same data
        reconstructed_output = reconstructed_model.predict(test_input, verbose=0)

        # Outputs should be identical
        np.testing.assert_allclose(original_output, reconstructed_output, rtol=1e-5)

    def test_all_blocks_save_load_with_weights(self):
        """Test saving and loading model with all custom blocks."""
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / 'test_model.keras'

            # Create model with all blocks
            inputs = tf.keras.Input(shape=(60, 50))

            x = TransformerBlock(d_model=50, num_heads=5, dff=200)(inputs)
            x = LSTMBlock(units=64, num_layers=2, bidirectional=True, return_sequences=False)(x)

            # Flatten and prepare for indicator subnet
            if len(x.shape) > 2:
                x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(30)(x)

            outputs = IndicatorSubnet(num_indicators=30, hidden_units=[64, 32], output_dim=20)(x)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Generate test data
            test_input = np.random.randn(2, 60, 50).astype(np.float32)
            original_output = model.predict(test_input, verbose=0)

            # Save model
            model.save(model_path)

            # Load model
            custom_objects = {
                'TransformerBlock': TransformerBlock,
                'LSTMBlock': LSTMBlock,
                'IndicatorSubnet': IndicatorSubnet
            }
            loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

            # Test with same data
            loaded_output = loaded_model.predict(test_input, verbose=0)

            # Outputs should be identical
            np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5)


class TestSerializationEdgeCases:
    """Test edge cases in serialization."""

    def test_indicator_subnet_with_default_activation(self):
        """Test IndicatorSubnet serialization with default activation."""
        layer = IndicatorSubnet(
            num_indicators=30,
            hidden_units=[64, 32],
            output_dim=20
        )
        layer.build(input_shape=(None, 30))

        config = layer.get_config()

        # Should have default activation
        assert config['activation'] == 'relu'

        # Should be able to recreate
        reconstructed = IndicatorSubnet.from_config(config)
        assert reconstructed.activation == 'relu'

    def test_lstm_block_with_single_layer(self):
        """Test LSTMBlock serialization with single layer."""
        layer = LSTMBlock(units=128, num_layers=1)
        layer.build(input_shape=(None, 60, 50))

        config = layer.get_config()

        # Should work with single layer
        assert config['num_layers'] == 1

        reconstructed = LSTMBlock.from_config(config)
        assert reconstructed.num_layers == 1

    def test_transformer_block_with_minimal_config(self):
        """Test TransformerBlock serialization with minimal config."""
        layer = TransformerBlock(d_model=64, num_heads=4, dff=256)
        layer.build(input_shape=(None, 60, 64))

        config = layer.get_config()

        # Should have default dropout
        assert 'dropout_rate' in config

        reconstructed = TransformerBlock.from_config(config)
        assert reconstructed.d_model == 64
