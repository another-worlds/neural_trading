"""LSTM block for capturing sequential dependencies.

Implements bidirectional LSTM layers with dropout for time series modeling
as per SRS Section 3.5.1.
"""
import tensorflow as tf
from tensorflow.keras import layers


class LSTMBlock(layers.Layer):
    """LSTM block with optional bidirectional processing.

    Captures sequential dependencies in time series data using LSTM cells.
    Supports multi-layer stacking and bidirectional processing.

    Args:
        units: Number of LSTM units per layer.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate for regularization.
        bidirectional: Whether to use bidirectional LSTM.
        return_sequences: Whether to return full sequence or last output.
        name: Layer name.

    Examples:
        >>> lstm_block = LSTMBlock(
        ...     units=128,
        ...     num_layers=2,
        ...     dropout=0.2,
        ...     bidirectional=True
        ... )
        >>> x = tf.random.normal((2, 60, 10))
        >>> output = lstm_block(x, training=True)
        >>> output.shape
        TensorShape([2, 60, 256])  # 128 * 2 for bidirectional
    """

    def __init__(
        self,
        units: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
        return_sequences: bool = True,
        **kwargs
    ):
        """Initialize LSTM block.

        Args:
            units: Number of LSTM units per layer.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout rate for regularization.
            bidirectional: Whether to use bidirectional LSTM.
            return_sequences: Whether to return full sequence or last output.
        """
        super().__init__(**kwargs)

        self.units = units
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.return_sequences = return_sequences

        # Build LSTM layers
        self.lstm_layers = []
        for i in range(num_layers):
            # Determine if this layer should return sequences
            # Last layer respects return_sequences parameter
            # All other layers must return sequences for stacking
            return_seq = return_sequences if i == num_layers - 1 else True

            # Create LSTM layer
            lstm_layer = layers.LSTM(
                units=units,
                return_sequences=return_seq,
                dropout=dropout,
                recurrent_dropout=0.0  # Avoid recurrent dropout for stability
            )

            # Wrap in bidirectional if needed
            if bidirectional:
                lstm_layer = layers.Bidirectional(lstm_layer)

            self.lstm_layers.append(lstm_layer)

    def call(self, inputs, training=None, mask=None):
        """Forward pass through LSTM block.

        Args:
            inputs: Input tensor of shape (batch, seq_len, features).
            training: Whether in training mode (for dropout).
            mask: Optional mask tensor.

        Returns:
            Output tensor. Shape depends on return_sequences:
            - If True: (batch, seq_len, units) or (batch, seq_len, units*2) if bidirectional
            - If False: (batch, units) or (batch, units*2) if bidirectional
        """
        x = inputs

        # Pass through stacked LSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training, mask=mask)

        return x

    def get_config(self):
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'return_sequences': self.return_sequences
        })
        return config
