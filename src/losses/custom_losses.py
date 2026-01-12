"""Custom loss functions for neural trading pipeline.

This module provides custom loss functions for multi-output prediction:
- FocalLoss for direction classification
- HuberLoss for price prediction
- NegativeLogLikelihood for variance prediction
- TrendLoss for multi-horizon consistency
- CompositeLoss for weighted combination

As per SRS Section 3.5.2.
"""
import tensorflow as tf
from typing import Dict, Optional
from src.losses.loss_registry import register_loss


@register_loss('focal')
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for handling class imbalance in direction classification.

    Focuses on hard-to-classify examples by down-weighting easy examples.
    Uses alpha for class balance and gamma for hard example focus.

    Args:
        alpha: Weighting factor for positive class (default: 0.7 per SRS).
        gamma: Focusing parameter for hard examples (default: 1.0 per SRS).

    Examples:
        >>> focal_loss = FocalLoss(alpha=0.7, gamma=1.0)
        >>> y_true = tf.constant([[1.0], [0.0], [1.0]])
        >>> y_pred = tf.constant([[0.9], [0.1], [0.8]])
        >>> loss = focal_loss(y_true, y_pred)
    """

    def __init__(self, alpha: float = 0.7, gamma: float = 1.0, **kwargs):
        """Initialize Focal Loss.

        Args:
            alpha: Weighting factor for positive class.
            gamma: Focusing parameter for hard examples.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        """Compute focal loss.

        Args:
            y_true: Ground truth labels of shape (batch_size, 1).
            y_pred: Predicted probabilities of shape (batch_size, 1).

        Returns:
            Scalar loss value.
        """
        # Clip predictions to prevent log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Compute binary cross entropy
        bce = y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)

        # Compute focal weight
        # For positive class: (1-p)^gamma, for negative class: p^gamma
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1 - p_t, self.gamma)

        # Apply alpha weighting
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        # Compute focal loss
        loss = -alpha_t * focal_weight * bce

        return tf.reduce_mean(loss)

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config


@register_loss('huber')
class HuberLoss(tf.keras.losses.Loss):
    """Huber Loss for robust price prediction.

    Combines L2 loss for small errors (quadratic) and L1 loss for large errors (linear).
    Less sensitive to outliers than MSE.

    Args:
        delta: Threshold for switching from L2 to L1 (default: 1.0 per SRS).

    Examples:
        >>> huber_loss = HuberLoss(delta=1.0)
        >>> y_true = tf.constant([[42000.0], [42100.0]])
        >>> y_pred = tf.constant([[42010.0], [42090.0]])
        >>> loss = huber_loss(y_true, y_pred)
    """

    def __init__(self, delta: float = 1.0, **kwargs):
        """Initialize Huber Loss.

        Args:
            delta: Threshold for L2 to L1 transition.
        """
        super().__init__(**kwargs)
        self.delta = delta

    def call(self, y_true, y_pred):
        """Compute Huber loss.

        Args:
            y_true: Ground truth values of shape (batch_size, 1).
            y_pred: Predicted values of shape (batch_size, 1).

        Returns:
            Scalar loss value.
        """
        error = y_true - y_pred
        abs_error = tf.abs(error)

        # Quadratic region: |error| <= delta
        quadratic = 0.5 * tf.square(error)

        # Linear region: |error| > delta
        linear = self.delta * abs_error - 0.5 * tf.square(self.delta)

        # Choose based on threshold
        loss = tf.where(abs_error <= self.delta, quadratic, linear)

        return tf.reduce_mean(loss)

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({'delta': self.delta})
        return config


@register_loss('nll')
class NegativeLogLikelihood(tf.keras.losses.Loss):
    """Negative Log Likelihood for variance prediction with uncertainty.

    Encourages well-calibrated uncertainty estimates by penalizing
    overconfident wrong predictions and underconfident correct predictions.

    Examples:
        >>> nll_loss = NegativeLogLikelihood()
        >>> y_true = tf.constant([[42000.0], [42100.0]])
        >>> y_pred_mean = tf.constant([[42010.0], [42090.0]])
        >>> y_pred_var = tf.constant([[0.1], [0.2]])
        >>> loss = nll_loss(y_true, y_pred_mean, y_pred_var)
    """

    def __init__(self, **kwargs):
        """Initialize Negative Log Likelihood loss."""
        super().__init__(**kwargs)

    def call(self, y_true, y_pred_mean, y_pred_var=None):
        """Compute negative log likelihood.

        Args:
            y_true: Ground truth values of shape (batch_size, 1).
            y_pred_mean: Predicted mean values of shape (batch_size, 1).
            y_pred_var: Predicted variance of shape (batch_size, 1).

        Returns:
            Scalar loss value.
        """
        if y_pred_var is None:
            # Fallback to MSE if variance not provided
            return tf.reduce_mean(tf.square(y_true - y_pred_mean))

        # Add small epsilon to prevent log(0) and division by zero
        epsilon = 1e-6
        y_pred_var = tf.maximum(y_pred_var, epsilon)

        # Compute negative log likelihood for Gaussian distribution
        # NLL = 0.5 * (log(2π*var) + (y_true - y_pred_mean)^2 / var)
        # We omit the constant log(2π) term as it doesn't affect optimization
        squared_error = tf.square(y_true - y_pred_mean)
        nll = 0.5 * (tf.math.log(y_pred_var) + squared_error / y_pred_var)

        return tf.reduce_mean(nll)

    def get_config(self):
        """Get configuration for serialization."""
        return super().get_config()


@register_loss('trend')
class TrendLoss(tf.keras.losses.Loss):
    """Trend Loss for multi-horizon consistency.

    Ensures predictions maintain consistent trends across multiple time horizons.
    Combines local trends (consecutive horizons) and global trends (overall direction).

    Args:
        horizons: List of horizon indices (default: [0, 1, 2]).

    Examples:
        >>> trend_loss = TrendLoss(horizons=[0, 1, 2])
        >>> y_true = {
        ...     'h0': tf.constant([[42100.0]]),
        ...     'h1': tf.constant([[42200.0]]),
        ...     'h2': tf.constant([[42300.0]])
        ... }
        >>> y_pred = {
        ...     'h0': tf.constant([[42110.0]]),
        ...     'h1': tf.constant([[42210.0]]),
        ...     'h2': tf.constant([[42310.0]])
        ... }
        >>> loss = trend_loss(y_true, y_pred)
    """

    def __init__(self, horizons: Optional[list] = None, **kwargs):
        """Initialize Trend Loss.

        Args:
            horizons: List of horizon indices.
        """
        super().__init__(**kwargs)
        self.horizons = horizons or [0, 1, 2]

    def call(self, y_true, y_pred):
        """Compute trend loss.

        Args:
            y_true: Dictionary of true values for each horizon.
            y_pred: Dictionary of predicted values for each horizon.

        Returns:
            Scalar loss value.
        """
        # Handle dict inputs for multi-horizon
        if not isinstance(y_true, dict) or not isinstance(y_pred, dict):
            # Fallback for non-dict inputs
            if isinstance(y_true, dict):
                return tf.constant(0.0)
            return tf.reduce_mean(tf.square(y_true - y_pred))

        # Extract predictions for each horizon
        horizons = [f'h{h}' for h in self.horizons]
        pred_values = []
        true_values = []

        for h in horizons:
            if h in y_pred and h in y_true:
                pred_values.append(y_pred[h])
                true_values.append(y_true[h])

        if len(pred_values) < 2:
            return tf.constant(0.0)

        # Stack predictions
        pred_stacked = tf.concat(pred_values, axis=-1)  # (batch, num_horizons)
        true_stacked = tf.concat(true_values, axis=-1)

        # Compute local trend loss (consecutive differences)
        pred_local_trends = pred_stacked[:, 1:] - pred_stacked[:, :-1]
        true_local_trends = true_stacked[:, 1:] - true_stacked[:, :-1]
        local_loss = tf.reduce_mean(tf.square(pred_local_trends - true_local_trends))

        return local_loss

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({'horizons': self.horizons})
        return config


@register_loss('composite')
class CompositeLoss(tf.keras.losses.Loss):
    """Composite Loss for weighted combination of multiple losses.

    Combines point loss, direction loss, variance loss, and trend loss
    with configurable weights (lambdas) as per SRS.

    Args:
        loss_config: Dictionary mapping loss types to their configuration.

    Examples:
        >>> config = {
        ...     'point_loss': {'type': 'huber', 'weight': 1.0},
        ...     'direction_loss': {'type': 'focal', 'weight': 1.0},
        ...     'variance_loss': {'type': 'nll', 'weight': 1.0}
        ... }
        >>> composite_loss = CompositeLoss(config)
    """

    def __init__(self, loss_config: Optional[Dict] = None, **kwargs):
        """Initialize Composite Loss.

        Args:
            loss_config: Dictionary of loss configurations.
        """
        super().__init__(**kwargs)
        self.loss_config = loss_config or {}

        # Initialize component losses
        self.losses = {}
        self.weights = {}

        for loss_name, config in self.loss_config.items():
            loss_type = config.get('type', 'mse')
            weight = config.get('weight', 1.0)
            self.weights[loss_name] = weight

            # Create loss instance based on type
            if loss_type == 'huber':
                delta = config.get('delta', 1.0)
                self.losses[loss_name] = HuberLoss(delta=delta)
            elif loss_type == 'focal':
                alpha = config.get('alpha', 0.7)
                gamma = config.get('gamma', 1.0)
                self.losses[loss_name] = FocalLoss(alpha=alpha, gamma=gamma)
            elif loss_type == 'nll':
                self.losses[loss_name] = NegativeLogLikelihood()
            elif loss_type == 'trend':
                self.losses[loss_name] = TrendLoss()
            else:
                # Default to MSE
                self.losses[loss_name] = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        """Compute weighted composite loss.

        Args:
            y_true: Dictionary of true values for each output.
            y_pred: Dictionary of predicted values for each output.

        Returns:
            Scalar loss value.
        """
        # Handle dict inputs
        if not isinstance(y_true, dict) or not isinstance(y_pred, dict):
            if isinstance(y_true, dict):
                return tf.constant(0.0)
            return tf.reduce_mean(tf.square(y_true - y_pred))

        total_loss = 0.0

        # Compute each component loss
        for loss_name, loss_fn in self.losses.items():
            weight = self.weights.get(loss_name, 1.0)

            # Extract relevant predictions based on loss type
            if 'point' in loss_name or 'price' in loss_name:
                # Price prediction loss
                for h in [0, 1, 2]:
                    key = f'price_h{h}'
                    if key in y_true and key in y_pred:
                        component_loss = loss_fn(y_true[key], y_pred[key])
                        total_loss += weight * component_loss

            elif 'direction' in loss_name:
                # Direction classification loss
                for h in [0, 1, 2]:
                    key = f'direction_h{h}'
                    if key in y_true and key in y_pred:
                        component_loss = loss_fn(y_true[key], y_pred[key])
                        total_loss += weight * component_loss

            elif 'variance' in loss_name:
                # Variance prediction loss (NLL)
                for h in [0, 1, 2]:
                    price_key = f'price_h{h}'
                    var_key = f'variance_h{h}'
                    if price_key in y_true and price_key in y_pred and var_key in y_pred:
                        if isinstance(loss_fn, NegativeLogLikelihood):
                            component_loss = loss_fn(
                                y_true[price_key],
                                y_pred[price_key],
                                y_pred[var_key]
                            )
                        else:
                            component_loss = loss_fn(y_true[var_key], y_pred[var_key])
                        total_loss += weight * component_loss

        return total_loss if isinstance(total_loss, tf.Tensor) else tf.constant(total_loss, dtype=tf.float32)

    def get_config(self):
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({'loss_config': self.loss_config})
        return config


def local_trend_loss(y_pred):
    """Compute local trend loss for consecutive horizons.

    Measures consistency of predicted trends between adjacent time horizons.

    Args:
        y_pred: Stacked predictions of shape (batch, num_horizons).

    Returns:
        Scalar loss value.
    """
    # Compute differences between consecutive predictions
    trend_diffs = y_pred[:, 1:] - y_pred[:, :-1]

    # Penalize large variations in trend
    # We want smooth transitions between horizons
    second_order_diffs = trend_diffs[:, 1:] - trend_diffs[:, :-1]

    loss = tf.reduce_mean(tf.square(second_order_diffs))
    return loss


def global_trend_loss(current_price, y_pred_h2):
    """Compute global trend loss from current price to final horizon.

    Ensures long-term prediction consistency.

    Args:
        current_price: Current price of shape (batch, 1).
        y_pred_h2: Predicted price at final horizon of shape (batch, 1).

    Returns:
        Scalar loss value.
    """
    # Compute overall trend
    global_trend = y_pred_h2 - current_price

    # Penalize extreme predictions (both very large gains and losses)
    # Use a soft threshold
    threshold = 0.05  # 5% change threshold
    normalized_trend = global_trend / (current_price + 1e-8)

    # Penalize trends beyond reasonable bounds
    loss = tf.reduce_mean(tf.square(tf.maximum(tf.abs(normalized_trend) - threshold, 0.0)))

    return loss


def extended_trend_loss(actual_trends: Dict, predicted_trends: Dict):
    """Compute extended trend loss across multiple timeframes (1m, 5m, 15m).

    Ensures consistency across different time scales.

    Args:
        actual_trends: Dictionary of actual percentage changes.
        predicted_trends: Dictionary of predicted percentage changes.

    Returns:
        Scalar loss value.
    """
    total_loss = 0.0
    count = 0

    for timeframe in ['1m', '5m', '15m']:
        if timeframe in actual_trends and timeframe in predicted_trends:
            diff = tf.square(actual_trends[timeframe] - predicted_trends[timeframe])
            total_loss += tf.reduce_mean(diff)
            count += 1

    if count == 0:
        return tf.constant(0.0)

    return total_loss / count
