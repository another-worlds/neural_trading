"""Hybrid model architecture combining Transformer, LSTM, and indicators.

Implements the full neural trading model with 3 independent towers for
multi-horizon prediction as per SRS Section 3.5.1.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Optional
from src.models.transformer_block import TransformerBlock
from src.models.lstm_block import LSTMBlock
from src.models.indicator_subnet import IndicatorSubnet


class HybridModel(keras.Model):
    """Hybrid model with Transformer, LSTM, and indicator subnet.

    Combines multiple architectures for robust time series prediction:
    - Transformer for global dependencies
    - LSTM for sequential patterns
    - Indicator subnet for learnable technical indicators

    Has 3 independent towers (h0, h1, h2) for multi-horizon prediction.
    Each tower outputs 3 values: price, direction, variance.

    Args:
        config: Model configuration dictionary with keys:
            - lstm_units: LSTM units per layer
            - lstm_layers: Number of LSTM layers
            - transformer_heads: Number of attention heads
            - transformer_dim: Transformer embedding dimension
            - dropout: Dropout rate
            - l2_reg: L2 regularization factor
        name: Model name.

    Examples:
        >>> config = {
        ...     'lstm_units': 128,
        ...     'lstm_layers': 2,
        ...     'transformer_heads': 4,
        ...     'transformer_dim': 128,
        ...     'dropout': 0.2,
        ...     'l2_reg': 0.001
        ... }
        >>> model = HybridModel(config)
        >>> x = tf.random.normal((2, 60, 10))
        >>> outputs = model(x, training=True)
    """

    def __init__(self, config: Dict, **kwargs):
        """Initialize hybrid model.

        Args:
            config: Model configuration dictionary.
        """
        super().__init__(**kwargs)

        self.config = config

        # Extract config parameters
        lstm_units = config.get('lstm_units', 128)
        lstm_layers = config.get('lstm_layers', 2)
        transformer_heads = config.get('transformer_heads', 4)
        transformer_dim = config.get('transformer_dim', 128)
        dropout = config.get('dropout', 0.2)
        l2_reg = config.get('l2_reg', 0.001)

        # Input projection to transformer dimension
        self.input_projection = layers.Dense(
            transformer_dim,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )

        # Transformer block for global dependencies
        self.transformer = TransformerBlock(
            d_model=transformer_dim,
            num_heads=transformer_heads,
            dff=transformer_dim * 4,
            dropout_rate=dropout
        )

        # LSTM block for sequential dependencies
        self.lstm = LSTMBlock(
            units=lstm_units,
            num_layers=lstm_layers,
            dropout=dropout,
            bidirectional=True,
            return_sequences=True
        )

        # Indicator subnet (processes 30 learnable indicator params)
        self.indicator_subnet = IndicatorSubnet(
            num_indicators=30,
            hidden_units=[64, 32],
            output_dim=20,
            dropout_rate=dropout
        )

        # Global pooling to get single vector representation
        self.global_pool = layers.GlobalAveragePooling1D()

        # Dropout for regularization
        self.dropout = layers.Dropout(dropout)

        # Feature fusion layer
        # Combines: transformer features + LSTM features + indicator features
        lstm_output_dim = lstm_units * 2  # Bidirectional
        combined_dim = transformer_dim + lstm_output_dim + 20  # 20 from indicator subnet

        self.fusion = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg)
        )

        # 3 independent towers for h0, h1, h2
        self.towers = {}
        for horizon in ['h0', 'h1', 'h2']:
            self.towers[horizon] = self._build_tower(
                horizon, dropout, l2_reg
            )

    def _build_tower(self, horizon: str, dropout: float, l2_reg: float):
        """Build prediction tower for a specific horizon.

        Each tower has its own hidden layers and 3 output heads:
        - price: Point prediction
        - direction: Binary direction (up/down)
        - variance: Uncertainty estimate

        Args:
            horizon: Horizon identifier ('h0', 'h1', 'h2').
            dropout: Dropout rate.
            l2_reg: L2 regularization factor.

        Returns:
            Dictionary with tower layers.
        """
        tower = {
            'hidden1': layers.Dense(
                128,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                name=f'{horizon}_hidden1'
            ),
            'dropout1': layers.Dropout(dropout, name=f'{horizon}_dropout1'),
            'hidden2': layers.Dense(
                64,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                name=f'{horizon}_hidden2'
            ),
            'dropout2': layers.Dropout(dropout, name=f'{horizon}_dropout2'),
            'price': layers.Dense(1, name=f'{horizon}_price'),
            'direction': layers.Dense(1, activation='sigmoid', name=f'{horizon}_direction'),
            'variance': layers.Dense(1, activation='softplus', name=f'{horizon}_variance')
        }
        return tower

    def call(self, inputs, training=None):
        """Forward pass through hybrid model.

        Args:
            inputs: Input tensor of shape (batch, seq_len, features).
            training: Whether in training mode.

        Returns:
            Dictionary with 9 outputs (3 towers × 3 outputs each):
            {
                'h0_price': ...,
                'h0_direction': ...,
                'h0_variance': ...,
                'h1_price': ...,
                'h1_direction': ...,
                'h1_variance': ...,
                'h2_price': ...,
                'h2_direction': ...,
                'h2_variance': ...
            }
        """
        # Project input to transformer dimension
        x_proj = self.input_projection(inputs)

        # Transformer path - global dependencies
        transformer_out = self.transformer(x_proj, training=training)

        # LSTM path - sequential dependencies
        lstm_out = self.lstm(inputs, training=training)

        # Create dummy indicator parameters for now
        # In full implementation, these would come from learnable indicator layers
        batch_size = tf.shape(inputs)[0]
        indicator_params = tf.ones((batch_size, 30))

        # Indicator subnet path
        indicator_out = self.indicator_subnet(indicator_params, training=training)

        # Pool temporal dimensions to get fixed-size representations
        transformer_pooled = self.global_pool(transformer_out)
        lstm_pooled = self.global_pool(lstm_out)

        # Concatenate all features
        combined = tf.concat([
            transformer_pooled,
            lstm_pooled,
            indicator_out
        ], axis=-1)

        # Fusion layer
        fused = self.fusion(combined)
        fused = self.dropout(fused, training=training)

        # Pass through each tower independently
        outputs = {}
        for horizon, tower in self.towers.items():
            # Hidden layers
            h = tower['hidden1'](fused)
            h = tower['dropout1'](h, training=training)
            h = tower['hidden2'](h)
            h = tower['dropout2'](h, training=training)

            # Output heads
            outputs[f'{horizon}_price'] = tower['price'](h)
            outputs[f'{horizon}_direction'] = tower['direction'](h)
            outputs[f'{horizon}_variance'] = tower['variance'](h)

        return outputs

    def get_config(self):
        """Get model configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        return {
            'config': self.config
        }


def build_model(config: Dict) -> keras.Model:
    """Build hybrid model from configuration.

    Factory function to create and initialize the hybrid model.

    Args:
        config: Full configuration dictionary with 'model' key.

    Returns:
        Compiled HybridModel instance.

    Examples:
        >>> config = {
        ...     'model': {
        ...         'lstm_units': 128,
        ...         'lstm_layers': 2,
        ...         'transformer_heads': 4,
        ...         'transformer_dim': 128,
        ...         'dropout': 0.2,
        ...         'l2_reg': 0.001
        ...     }
        ... }
        >>> model = build_model(config)
    """
    model_config = config.get('model', config)
    model = HybridModel(model_config)

    # Build the model with a sample input
    sample_input = tf.keras.Input(shape=(60, 10))  # (seq_len=60, features=10)
    _ = model(sample_input)

    return model
