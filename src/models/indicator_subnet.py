"""Indicator subnet for processing learnable indicator parameters.

Implements MLP subnet that transforms 30+ learnable indicator parameters
into features as per SRS Section 3.5.1.
"""
import tensorflow as tf
from tensorflow.keras import layers
from typing import List


class IndicatorSubnet(layers.Layer):
    """Subnet for processing learnable indicator parameters.

    Transforms learnable indicator parameters through MLP layers to produce
    features that are combined with main model representations.

    Args:
        num_indicators: Number of learnable indicator parameters (30+ per SRS).
        hidden_units: List of hidden layer sizes for MLP.
        output_dim: Output dimension.
        activation: Activation function for hidden layers.
        dropout_rate: Dropout rate for regularization.
        name: Layer name.

    Examples:
        >>> subnet = IndicatorSubnet(
        ...     num_indicators=30,
        ...     hidden_units=[64, 32],
        ...     output_dim=20
        ... )
        >>> indicator_params = tf.random.normal((2, 30))
        >>> output = subnet(indicator_params)
        >>> output.shape
        TensorShape([2, 20])
    """

    def __init__(
        self,
        num_indicators: int,
        hidden_units: List[int],
        output_dim: int,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        **kwargs
    ):
        """Initialize indicator subnet.

        Args:
            num_indicators: Number of learnable indicator parameters.
            hidden_units: List of hidden layer sizes.
            output_dim: Output dimension.
            activation: Activation function for hidden layers.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__(**kwargs)

        self.num_indicators = num_indicators
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.activation = activation
        self.dropout_rate = dropout_rate

        # Build MLP layers
        self.mlp_layers = []

        # Hidden layers
        for units in hidden_units:
            self.mlp_layers.append(layers.Dense(units, activation=activation))
            if dropout_rate > 0:
                self.mlp_layers.append(layers.Dropout(dropout_rate))

        # Output layer
        self.mlp_layers.append(layers.Dense(output_dim))

    @property
    def layers(self):
        """Get list of MLP layers.

        Returns:
            List of layer objects.
        """
        return self.mlp_layers

    def call(self, inputs, training=None):
        """Forward pass through indicator subnet.

        Args:
            inputs: Learnable indicator parameters of shape (batch, num_indicators).
            training: Whether in training mode (for dropout).

        Returns:
            Transformed features of shape (batch, output_dim).
        """
        x = inputs

        # Pass through MLP layers
        for layer in self.mlp_layers:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)

        return x

    def get_config(self):
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'num_indicators': self.num_indicators,
            'hidden_units': self.hidden_units,
            'output_dim': self.output_dim,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate
        })
        return config
