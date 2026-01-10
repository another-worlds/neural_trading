"""Helper functions for trading signal generation.

This module provides utility functions for:
- Confidence calculation from variance
- Signal strength computation
- Variance normalization
- Profit target calculation
- Dynamic stop loss calculation
- Position sizing
- Multi-horizon agreement checking
- Variance spike detection

All functions are designed to work with both scalar and array inputs.
"""
import numpy as np
from typing import Union, Tuple, Dict


def calculate_confidence(variance: Union[float, np.ndarray], eps: float = 1e-7) -> Union[float, np.ndarray]:
    """
    Convert variance to confidence score [0, 1].

    Higher variance = lower confidence (inverse relationship).
    Formula: confidence = 1 / (1 + variance + eps)

    Args:
        variance: Model variance (float or array)
        eps: Small epsilon to prevent division by zero (default: 1e-7)

    Returns:
        Confidence score(s) in range [0, 1]

    Examples:
        >>> calculate_confidence(0.0)
        1.0
        >>> calculate_confidence(1.0)
        0.5
        >>> calculate_confidence(np.array([0.0, 1.0, 9.0]))
        array([1.0, 0.5, 0.1])
    """
    return 1.0 / (1.0 + np.asarray(variance) + eps)


def calculate_signal_strength(
    direction_prob: Union[float, np.ndarray],
    confidence: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Combine direction probability and confidence into unified signal strength.

    Signal strength is the product of directional prediction and confidence.

    Args:
        direction_prob: Direction probability (0=down, 1=up)
        confidence: Confidence score [0, 1]

    Returns:
        Signal strength value (product of inputs)

    Examples:
        >>> calculate_signal_strength(1.0, 1.0)
        1.0
        >>> calculate_signal_strength(0.8, 0.5)
        0.4
    """
    return np.asarray(direction_prob) * np.asarray(confidence)


def normalize_variance(
    variance: Union[float, np.ndarray],
    variance_rolling_mean: Union[float, np.ndarray],
    variance_rolling_std: Union[float, np.ndarray],
    eps: float = 1e-7
) -> Union[float, np.ndarray]:
    """
    Normalize variance relative to rolling statistics (z-score).

    Computes: (variance - mean) / (std + eps)
    When std is very small (< eps), returns 0.0 to avoid division issues.

    Args:
        variance: Current variance value(s)
        variance_rolling_mean: Rolling mean of variance
        variance_rolling_std: Rolling standard deviation of variance
        eps: Small epsilon for numerical stability (default: 1e-7)

    Returns:
        Normalized variance (z-score)

    Examples:
        >>> normalize_variance(3.0, 2.0, 1.0)
        1.0  # One std above mean
        >>> normalize_variance(1.0, 2.0, 1.0)
        -1.0  # One std below mean
    """
    variance_array = np.asarray(variance)
    mean_array = np.asarray(variance_rolling_mean)
    std_array = np.asarray(variance_rolling_std)

    # Avoid division by very small std
    result = np.where(
        std_array < eps,
        0.0,
        (variance_array - mean_array) / (std_array + eps)
    )

    return result


def calculate_profit_targets(
    entry_price: float,
    price_predictions: list
) -> Dict[str, float]:
    """
    Calculate three-tier profit targets from multi-horizon predictions.

    Uses predicted prices at different horizons (h0, h1, h2) as profit targets.

    Args:
        entry_price: Trade entry price
        price_predictions: List of 3 price predictions [h0, h1, h2]

    Returns:
        Dictionary with keys:
            - tp1, tp2, tp3: Absolute profit target prices
            - tp1_pct, tp2_pct, tp3_pct: Percentage from entry

    Examples:
        >>> calculate_profit_targets(100.0, [102.0, 105.0, 110.0])
        {'tp1': 102.0, 'tp2': 105.0, 'tp3': 110.0,
         'tp1_pct': 2.0, 'tp2_pct': 5.0, 'tp3_pct': 10.0}
    """
    entry_price = float(entry_price)
    tp1 = float(price_predictions[0])
    tp2 = float(price_predictions[1])
    tp3 = float(price_predictions[2])

    # Calculate percentage gains
    tp1_pct = (tp1 - entry_price) / entry_price * 100
    tp2_pct = (tp2 - entry_price) / entry_price * 100
    tp3_pct = (tp3 - entry_price) / entry_price * 100

    return {
        'tp1': tp1,
        'tp2': tp2,
        'tp3': tp3,
        'tp1_pct': tp1_pct,
        'tp2_pct': tp2_pct,
        'tp3_pct': tp3_pct,
    }


def calculate_dynamic_stop_loss(
    entry_price: float,
    position_type: str,
    variance: float,
    variance_rolling_mean: float,
    base_stop_pct: float = 0.02,
    max_variance_multiplier: float = 2.0
) -> float:
    """
    Calculate variance-adjusted stop loss.

    Stop loss distance is adjusted based on model uncertainty (variance).
    Higher variance = wider stop loss to account for uncertainty.

    Formula:
        variance_ratio = min(variance / mean, max_multiplier)
        adjustment_factor = 1 + variance_ratio
        stop_distance = entry_price * base_stop_pct * adjustment_factor

    Args:
        entry_price: Trade entry price
        position_type: 'LONG' or 'SHORT'
        variance: Current model variance
        variance_rolling_mean: Rolling mean of variance
        base_stop_pct: Base stop loss percentage (default: 0.02 = 2%)
        max_variance_multiplier: Maximum variance ratio cap (default: 2.0)

    Returns:
        Stop loss price level

    Examples:
        >>> calculate_dynamic_stop_loss(100.0, 'LONG', 1.0, 1.0)
        96.0  # 4% stop (2% base * (1 + 1.0))
        >>> calculate_dynamic_stop_loss(100.0, 'SHORT', 1.0, 1.0)
        104.0  # 4% stop above entry
    """
    entry_price = float(entry_price)
    variance = float(variance)
    variance_rolling_mean = float(variance_rolling_mean)
    eps = 1e-7

    # Calculate variance ratio, capped at max multiplier
    variance_ratio = min(variance / (variance_rolling_mean + eps), max_variance_multiplier)

    # Adjustment factor: 1 + variance_ratio
    adjustment_factor = 1.0 + variance_ratio

    # Calculate stop distance
    stop_distance = entry_price * base_stop_pct * adjustment_factor

    # Apply based on position type
    if position_type == 'LONG':
        stop_loss = entry_price - stop_distance
    else:  # SHORT
        stop_loss = entry_price + stop_distance

    return stop_loss


def calculate_position_size_multiplier(
    confidence: float,
    size_high: float = 1.2,
    size_normal: float = 1.0,
    size_low: float = 0.6,
    conf_high_thresh: float = 0.7,
    conf_low_thresh: float = 0.5
) -> float:
    """
    Calculate position size multiplier based on confidence.

    Three tiers:
    - High confidence (> conf_high_thresh): size_high
    - Normal confidence (> conf_low_thresh): size_normal
    - Low confidence (≤ conf_low_thresh): size_low

    Args:
        confidence: Confidence score [0, 1]
        size_high: Multiplier for high confidence (default: 1.2)
        size_normal: Multiplier for normal confidence (default: 1.0)
        size_low: Multiplier for low confidence (default: 0.6)
        conf_high_thresh: High confidence threshold (default: 0.7)
        conf_low_thresh: Low confidence threshold (default: 0.5)

    Returns:
        Position size multiplier

    Examples:
        >>> calculate_position_size_multiplier(0.8)
        1.2  # High confidence
        >>> calculate_position_size_multiplier(0.6)
        1.0  # Normal confidence
        >>> calculate_position_size_multiplier(0.3)
        0.6  # Low confidence
    """
    confidence = float(confidence)

    if confidence > conf_high_thresh:
        return size_high
    elif confidence > conf_low_thresh:
        return size_normal
    else:
        return size_low


def check_multi_horizon_agreement(
    price_predictions: Union[list, np.ndarray],
    current_price: float,
    agreement_threshold: float = 0.67
) -> Tuple[bool, float]:
    """
    Check if multiple prediction horizons agree on market direction.

    Calculates what percentage of predictions agree on direction (up/down).
    Agreement is achieved if percentage >= threshold (default: 67% = 2 out of 3).

    Args:
        price_predictions: Array of price predictions for different horizons
        current_price: Current market price
        agreement_threshold: Required agreement percentage (default: 0.67)

    Returns:
        Tuple of (is_agreed: bool, agreement: float)
        - is_agreed: True if agreement >= threshold
        - agreement: Percentage of horizons agreeing (0.0 to 1.0)

    Examples:
        >>> check_multi_horizon_agreement([105, 110, 115], 100)
        (True, 1.0)  # All 3 predict up
        >>> check_multi_horizon_agreement([105, 110, 95], 100)
        (True, 0.667)  # 2 out of 3 predict up
        >>> check_multi_horizon_agreement([105, 95], 100, 0.67)
        (False, 0.5)  # Only 50% agreement
    """
    price_predictions = np.asarray(price_predictions)
    current_price = float(current_price)

    # Count predictions above and below current price
    up_count = np.sum(price_predictions > current_price)
    down_count = np.sum(price_predictions < current_price)

    # Agreement is the max of up/down as fraction of total
    total_predictions = len(price_predictions)
    agreement = max(up_count, down_count) / total_predictions

    # Check if agreement meets threshold (with tolerance for floating point)
    # This allows 2/3 (0.6666...) to pass threshold of 0.67
    # Using relative tolerance of 0.5% to handle fraction comparisons
    tolerance = max(1e-9, agreement_threshold * 0.005)
    is_agreed = bool(agreement >= (agreement_threshold - tolerance))

    return is_agreed, float(agreement)


def detect_variance_spike(
    variance: float,
    variance_rolling_mean: float,
    variance_rolling_std: float,
    spike_threshold: float = 2.0,
    eps: float = 1e-7
) -> bool:
    """
    Detect variance spikes indicating high model uncertainty or regime changes.

    A spike is detected when variance exceeds:
        spike_threshold * (rolling_mean + eps)

    Note: variance_rolling_std parameter is kept for interface compatibility
    but not used in the current spike detection logic.

    Args:
        variance: Current variance value
        variance_rolling_mean: Rolling mean of variance
        variance_rolling_std: Rolling std of variance (not used in calculation)
        spike_threshold: Multiplier for spike detection (default: 2.0)
        eps: Small epsilon for numerical stability (default: 1e-7)

    Returns:
        True if variance spike detected, False otherwise

    Examples:
        >>> detect_variance_spike(10.0, 2.0, 1.0, spike_threshold=2.0)
        True  # 10.0 > 2.0 * 2.0 = 4.0
        >>> detect_variance_spike(3.0, 2.0, 1.0, spike_threshold=2.0)
        False  # 3.0 < 2.0 * 2.0 = 4.0
    """
    variance = float(variance)
    variance_rolling_mean = float(variance_rolling_mean)

    # Calculate spike level
    spike_level = spike_threshold * (variance_rolling_mean + eps)

    # Detect if variance exceeds spike level
    is_spike = variance > spike_level

    return is_spike
