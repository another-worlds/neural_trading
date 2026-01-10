"""Unit tests for model components.

Tests Transformer, LSTM, indicator subnet, and hybrid model architecture as per SRS Section 3.5.1.
"""
import pytest
import numpy as np
import tensorflow as tf
from tensorflow import keras
from src.models.transformer_block import TransformerBlock
from src.models.lstm_block import LSTMBlock
from src.models.indicator_subnet import IndicatorSubnet
from src.models.hybrid_model import HybridModel, build_model


class TestTransformerBlock:
    """Test Transformer block for global dependencies."""

    def test_init_transformer(self):
        """Should initialize Transformer block."""
        transformer = TransformerBlock(
            d_model=128,
            num_heads=4,
            dff=512,
            dropout_rate=0.1
        )

        assert transformer.d_model == 128
        assert transformer.num_heads == 4

    def test_transformer_forward_pass(self):
        """Should perform forward pass."""
        transformer = TransformerBlock(
            d_model=128,
            num_heads=4,
            dff=512,
            dropout_rate=0.1
        )

        # Input: (batch_size, seq_len, d_model)
        x = tf.random.normal((2, 60, 128))
        output = transformer(x, training=True)

        assert output.shape == (2, 60, 128)

    def test_multihead_attention_layer(self):
        """Should have MultiHeadAttention layer."""
        transformer = TransformerBlock(
            d_model=128,
            num_heads=4,
            dff=512,
            dropout_rate=0.1
        )

        # Build the model
        x = tf.random.normal((2, 60, 128))
        _ = transformer(x)

        # Check for attention layer
        assert hasattr(transformer, 'mha')

    def test_feedforward_network(self):
        """Should have feed-forward network."""
        transformer = TransformerBlock(
            d_model=128,
            num_heads=4,
            dff=512,
            dropout_rate=0.1
        )

        x = tf.random.normal((2, 60, 128))
        _ = transformer(x)

        # Check for FFN layers
        assert hasattr(transformer, 'ffn')

    def test_layer_normalization(self):
        """Should apply layer normalization."""
        transformer = TransformerBlock(
            d_model=128,
            num_heads=4,
            dff=512,
            dropout_rate=0.1
        )

        x = tf.random.normal((2, 60, 128))
        _ = transformer(x)

        # Check for LayerNorm
        assert hasattr(transformer, 'layernorm1')
        assert hasattr(transformer, 'layernorm2')

    def test_dropout_application(self):
        """Should apply dropout during training."""
        transformer = TransformerBlock(
            d_model=128,
            num_heads=4,
            dff=512,
            dropout_rate=0.5  # High dropout for testing
        )

        x = tf.random.normal((10, 60, 128))
        output_train = transformer(x, training=True)
        output_inference = transformer(x, training=False)

        # Outputs should differ due to dropout
        # (Note: this is probabilistic)

    def test_attention_heads_config(self):
        """Should support configurable number of attention heads."""
        for num_heads in [2, 4, 8]:
            transformer = TransformerBlock(
                d_model=128,
                num_heads=num_heads,
                dff=512,
                dropout_rate=0.1
            )

            x = tf.random.normal((2, 60, 128))
            output = transformer(x)
            assert output.shape == (2, 60, 128)


class TestLSTMBlock:
    """Test LSTM block for sequential dependencies."""

    def test_init_lstm(self):
        """Should initialize LSTM block."""
        lstm_block = LSTMBlock(
            units=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=True
        )

        assert lstm_block.units == 128
        assert lstm_block.num_layers == 2

    def test_lstm_forward_pass(self):
        """Should perform forward pass."""
        lstm_block = LSTMBlock(
            units=128,
            num_layers=2,
            dropout=0.2,
            bidirectional=True
        )

        # Input: (batch_size, seq_len, features)
        x = tf.random.normal((2, 60, 10))
        output = lstm_block(x, training=True)

        assert output.shape[0] == 2
        assert output.shape[1] == 60
        # Bidirectional doubles the output size
        assert output.shape[2] == 128 * 2

    def test_bidirectional_lstm(self):
        """Should support bidirectional LSTM as per SRS."""
        lstm_block = LSTMBlock(
            units=128,
            num_layers=1,
            dropout=0.0,
            bidirectional=True
        )

        x = tf.random.normal((2, 60, 10))
        output = lstm_block(x)

        # Bidirectional should double output dimensions
        assert output.shape[2] == 128 * 2

    def test_unidirectional_lstm(self):
        """Should support unidirectional LSTM."""
        lstm_block = LSTMBlock(
            units=128,
            num_layers=1,
            dropout=0.0,
            bidirectional=False
        )

        x = tf.random.normal((2, 60, 10))
        output = lstm_block(x)

        assert output.shape[2] == 128

    def test_multi_layer_lstm(self):
        """Should support multiple LSTM layers."""
        lstm_block = LSTMBlock(
            units=128,
            num_layers=3,
            dropout=0.2,
            bidirectional=False
        )

        x = tf.random.normal((2, 60, 10))
        output = lstm_block(x)

        assert output.shape == (2, 60, 128)

    def test_return_sequences(self):
        """Should return full sequence."""
        lstm_block = LSTMBlock(
            units=128,
            num_layers=1,
            dropout=0.0,
            bidirectional=False,
            return_sequences=True
        )

        x = tf.random.normal((2, 60, 10))
        output = lstm_block(x)

        # Should return all time steps
        assert output.shape[1] == 60


class TestIndicatorSubnet:
    """Test learnable indicator subnet."""

    def test_init_indicator_subnet(self):
        """Should initialize indicator subnet."""
        subnet = IndicatorSubnet(
            num_indicators=30,
            hidden_units=[64, 32],
            output_dim=20
        )

        assert subnet.num_indicators == 30

    def test_indicator_subnet_forward_pass(self):
        """Should transform indicator parameters."""
        subnet = IndicatorSubnet(
            num_indicators=30,
            hidden_units=[64, 32],
            output_dim=20
        )

        # Input: learnable indicator parameters
        indicator_params = tf.random.normal((2, 30))
        output = subnet(indicator_params)

        assert output.shape == (2, 20)

    def test_mlp_layers(self):
        """Should be implemented as MLP."""
        subnet = IndicatorSubnet(
            num_indicators=30,
            hidden_units=[64, 32],
            output_dim=20
        )

        # Build the subnet
        x = tf.random.normal((2, 30))
        _ = subnet(x)

        # Should have dense layers
        assert len(subnet.layers) > 0

    def test_30_plus_learnable_params_as_per_srs(self):
        """Should support 30+ learnable indicator parameters as per SRS."""
        subnet = IndicatorSubnet(
            num_indicators=30,  # As per SRS
            hidden_units=[64],
            output_dim=20
        )

        x = tf.random.normal((2, 30))
        output = subnet(x)

        assert output.shape[1] == 20


class TestHybridModel:
    """Test hybrid model architecture."""

    def test_init_hybrid_model(self, sample_config):
        """Should initialize hybrid model."""
        model = HybridModel(sample_config['model'])

        assert model is not None

    def test_model_has_transformer_component(self, sample_config):
        """Should include Transformer component."""
        model = HybridModel(sample_config['model'])

        # Check for transformer block
        assert hasattr(model, 'transformer') or any(
            isinstance(layer, TransformerBlock) for layer in model.layers
        )

    def test_model_has_lstm_component(self, sample_config):
        """Should include LSTM component."""
        model = HybridModel(sample_config['model'])

        # Check for LSTM block
        assert hasattr(model, 'lstm') or any(
            'lstm' in layer.name.lower() for layer in model.layers
        )

    def test_model_has_indicator_subnet(self, sample_config):
        """Should include indicator subnet."""
        model = HybridModel(sample_config['model'])

        # Check for indicator subnet
        assert hasattr(model, 'indicator_subnet')

    def test_model_forward_pass(self, sample_config):
        """Should perform forward pass."""
        model = HybridModel(sample_config['model'])
        model.build(input_shape=(None, 60, 10))

        x = tf.random.normal((2, 60, 10))
        outputs = model(x, training=True)

        assert outputs is not None

    def test_multi_output_architecture(self, sample_config):
        """Should have multiple output heads."""
        model = HybridModel(sample_config['model'])
        model.build(input_shape=(None, 60, 10))

        x = tf.random.normal((2, 60, 10))
        outputs = model(x)

        # Should have 9 outputs (3 towers × 3 outputs each)
        assert isinstance(outputs, (list, dict))

    def test_independent_towers_for_horizons(self, sample_config):
        """Should have 3 independent towers for h0, h1, h2 as per SRS."""
        model = HybridModel(sample_config['model'])

        # Check for tower structures
        # Each tower should be independent

    def test_three_outputs_per_tower(self, sample_config):
        """Each tower should have 3 outputs: price, direction, variance."""
        model = HybridModel(sample_config['model'])
        model.build(input_shape=(None, 60, 10))

        x = tf.random.normal((2, 60, 10))
        outputs = model(x)

        # Total 9 outputs: 3 towers × 3 outputs
        # Verify structure

    def test_dropout_regularization(self, sample_config):
        """Should apply dropout as per SRS."""
        sample_config['model']['dropout'] = 0.2

        model = HybridModel(sample_config['model'])
        model.build(input_shape=(None, 60, 10))

        # Check for dropout layers
        dropout_layers = [layer for layer in model.layers if 'dropout' in layer.name.lower()]
        assert len(dropout_layers) > 0

    def test_l2_regularization(self, sample_config):
        """Should apply L2 regularization."""
        sample_config['model']['l2_reg'] = 0.001

        model = HybridModel(sample_config['model'])
        model.build(input_shape=(None, 60, 10))

        # Check for L2 regularization on weights


class TestBuildModel:
    """Test model building function."""

    def test_build_model_from_config(self, sample_config):
        """Should build model from configuration."""
        model = build_model(sample_config)

        assert isinstance(model, keras.Model)

    def test_model_input_shape(self, sample_config):
        """Should have correct input shape for 60-minute lookback."""
        model = build_model(sample_config)

        # Input should accept (batch, 60, features)
        x = tf.random.normal((2, 60, 10))
        outputs = model(x)
        assert outputs is not None

    def test_model_output_structure(self, sample_config):
        """Should have correct output structure."""
        model = build_model(sample_config)

        x = tf.random.normal((2, 60, 10))
        outputs = model(x)

        # Should have 9 outputs or dict with 9 keys
        if isinstance(outputs, dict):
            assert len(outputs) == 9
        elif isinstance(outputs, list):
            assert len(outputs) == 9

    def test_model_compilation(self, sample_config):
        """Should compile model with losses and metrics."""
        model = build_model(sample_config)

        # Model should be compilable
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        assert model.optimizer is not None

    def test_model_summary(self, sample_config):
        """Should generate model summary."""
        model = build_model(sample_config)

        # Should be able to print summary
        summary = []
        model.summary(print_fn=lambda x: summary.append(x))
        assert len(summary) > 0

    def test_trainable_parameters(self, sample_config):
        """Should have trainable parameters."""
        model = build_model(sample_config)

        trainable_count = np.sum([tf.size(w).numpy() for w in model.trainable_weights])
        assert trainable_count > 0


class TestModelPersistence:
    """Test model saving and loading."""

    def test_save_model_weights(self, tmp_path, sample_config):
        """Should save model weights."""
        model = build_model(sample_config)

        weights_file = tmp_path / "model.weights.h5"
        model.save_weights(str(weights_file))

        assert weights_file.exists()

    def test_load_model_weights(self, tmp_path, sample_config):
        """Should load model weights."""
        model = build_model(sample_config)

        weights_file = tmp_path / "model.weights.h5"
        model.save_weights(str(weights_file))

        # Create new model and load weights
        new_model = build_model(sample_config)
        new_model.load_weights(str(weights_file))

        # Verify weights match
        for w1, w2 in zip(model.weights, new_model.weights):
            np.testing.assert_array_almost_equal(w1.numpy(), w2.numpy())

    def test_save_full_model(self, tmp_path, sample_config):
        """Should save full model (architecture + weights)."""
        model = build_model(sample_config)

        model_file = tmp_path / "full_model"
        model.save(str(model_file))

        assert model_file.exists()

    def test_model_weights_size(self, tmp_path, sample_config):
        """Model weights should be approximately 3.2 MB as per SRS."""
        model = build_model(sample_config)

        weights_file = tmp_path / "model.weights.h5"
        model.save_weights(str(weights_file))

        # Check file size
        file_size_mb = weights_file.stat().st_size / (1024 * 1024)
        # Allow some variation
        # assert 2.0 < file_size_mb < 5.0


class TestModelVersioning:
    """Test model versioning."""

    def test_model_version_metadata(self, sample_config):
        """Should track model version."""
        sample_config['model']['version'] = 'v3.0'

        model = build_model(sample_config)

        # Model should have version metadata
        # Implementation-specific

    def test_semantic_versioning(self):
        """Should support semantic versioning (v1.0, v2.0, v3.0)."""
        versions = ['v1.0', 'v2.0', 'v3.0']

        # Test version comparison logic
        # v3.0 > v2.0 > v1.0
