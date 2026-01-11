"""Transformer block for capturing global dependencies.

Implements multi-head attention mechanism with feed-forward network,
layer normalization, and residual connections as per SRS Section 3.5.1.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TransformerBlock(layers.Layer):
    """Transformer block with multi-head attention.

    Captures global dependencies in time series data using self-attention
    mechanism. Includes feed-forward network, layer normalization, and
    residual connections.

    Args:
        d_model: Dimension of model (embedding size).
        num_heads: Number of attention heads.
        dff: Dimension of feed-forward network.
        dropout_rate: Dropout rate for regularization.
        name: Layer name.

    Examples:
        >>> transformer = TransformerBlock(
        ...     d_model=128,
        ...     num_heads=4,
        ...     dff=512,
        ...     dropout_rate=0.1
        ... )
        >>> x = tf.random.normal((2, 60, 128))
        >>> output = transformer(x, training=True)
        >>> output.shape
        TensorShape([2, 60, 128])
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """Initialize Transformer block.

        Args:
            d_model: Dimension of model (embedding size).
            num_heads: Number of attention heads.
            dff: Dimension of feed-forward network.
            dropout_rate: Dropout rate for regularization.
        """
        super().__init__(**kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        # Multi-head attention layer
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )

        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])

        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None, mask=None):
        """Forward pass through transformer block.

        Args:
            inputs: Input tensor of shape (batch, seq_len, d_model).
            training: Whether in training mode (for dropout).
            mask: Optional attention mask.

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        # Multi-head self-attention with residual connection
        attn_output = self.mha(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=mask,
            training=training
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        output = self.layernorm2(out1 + ffn_output)

        return output

    def get_config(self):
        """Get layer configuration for serialization.

        Returns:
            Configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config
