"""Trading signal generation from predictions.

Generates actionable trading signals using helper functions from SRS Section 3.2.
"""
from typing import Dict, Any
import numpy as np

from src.utils.helper_functions import (
    calculate_signal_strength,
    calculate_position_size_multiplier,
    check_multi_horizon_agreement,
    calculate_profit_targets,
    calculate_dynamic_stop_loss
)


def generate_signals(
    predictions: Dict[str, Any],
    current_price: float,
    risk_tolerance: float = 0.02
) -> Dict[str, Any]:
    """Generate trading signals from model predictions.

    Integrates all helper functions to produce actionable trading signals.

    Args:
        predictions: Model predictions with keys like 'price_h0', 'direction_h0', 'variance_h0'
        current_price: Current market price
        risk_tolerance: Risk tolerance parameter (default: 0.02)

    Returns:
        Dictionary containing:
        - signal_strength: Overall signal strength (0-1)
        - position_size_multiplier: Position sizing multiplier
        - multi_horizon_agreement: Whether horizons agree on direction
        - profit_targets: Three-tier profit targets
        - stop_loss: Dynamic stop loss level
        - direction: Overall predicted direction (up/down)
        - confidence: Weighted confidence across horizons
    """
    # Extract predictions for each horizon
    price_h0 = _extract_value(predictions, 'price_h0')
    price_h1 = _extract_value(predictions, 'price_h1')
    price_h2 = _extract_value(predictions, 'price_h2')

    direction_h0 = _extract_value(predictions, 'direction_h0')
    direction_h1 = _extract_value(predictions, 'direction_h1')
    direction_h2 = _extract_value(predictions, 'direction_h2')

    variance_h0 = _extract_value(predictions, 'variance_h0')
    variance_h1 = _extract_value(predictions, 'variance_h1')
    variance_h2 = _extract_value(predictions, 'variance_h2')

    # Calculate confidences from variances
    from src.utils.helper_functions import calculate_confidence
    confidence_h0 = calculate_confidence(variance_h0)
    confidence_h1 = calculate_confidence(variance_h1)
    confidence_h2 = calculate_confidence(variance_h2)

    # Determine overall direction (weighted average of direction signals)
    overall_direction = (
        direction_h0 * 0.5 +  # 50% weight on h0
        direction_h1 * 0.3 +  # 30% weight on h1
        direction_h2 * 0.2    # 20% weight on h2
    )

    # Weighted confidence
    weighted_confidence = (
        confidence_h0 * 0.5 +
        confidence_h1 * 0.3 +
        confidence_h2 * 0.2
    )

    # Calculate signal strength using helper function
    signal_strength = calculate_signal_strength(overall_direction, weighted_confidence)

    # Check multi-horizon agreement
    price_predictions = np.array([price_h0, price_h1, price_h2])
    agreement, agreement_fraction = check_multi_horizon_agreement(price_predictions, current_price)

    # Calculate position size multiplier
    position_multiplier = calculate_position_size_multiplier(
        confidence_h0, confidence_h1, confidence_h2,
        agreement
    )

    # Calculate profit targets
    profit_targets = calculate_profit_targets(
        entry_price=current_price,
        price_predictions=[price_h0, price_h1, price_h2]
    )

    # Calculate dynamic stop loss
    position_type = 'long' if overall_direction > 0.5 else 'short'
    stop_loss = calculate_dynamic_stop_loss(
        entry_price=current_price,
        position_type=position_type,
        variance=variance_h0,
        variance_rolling_mean=variance_h0,  # Use current variance as approximation
        base_stop_pct=risk_tolerance
    )

    return {
        'signal_strength': float(signal_strength),
        'position_size_multiplier': float(position_multiplier),
        'multi_horizon_agreement': bool(agreement),
        'profit_targets': profit_targets,
        'stop_loss': float(stop_loss),
        'direction': 'up' if overall_direction > 0.5 else 'down',
        'direction_probability': float(overall_direction),
        'confidence': float(weighted_confidence),
        'predicted_prices': {
            'h0': float(price_h0),
            'h1': float(price_h1),
            'h2': float(price_h2)
        }
    }


def _extract_value(predictions: Dict[str, Any], key: str) -> float:
    """Extract scalar value from predictions dictionary.

    Handles both direct keys and alternative naming (e.g., 'price_h0' vs 'h0_price').

    Args:
        predictions: Predictions dictionary
        key: Key to extract

    Returns:
        Scalar float value
    """
    # Try direct key
    value = predictions.get(key)

    # Try alternative naming (swap order)
    if value is None:
        parts = key.split('_')
        if len(parts) == 2:
            alt_key = f'{parts[1]}_{parts[0]}'  # e.g., 'h0_price' -> 'price_h0'
            value = predictions.get(alt_key)

    # Default to 0.5 for missing values
    if value is None:
        if 'direction' in key:
            value = 0.5
        elif 'variance' in key:
            value = 0.01
        else:
            value = 0.0

    # Convert to scalar if array
    if isinstance(value, np.ndarray):
        value = float(value.flatten()[0])
    elif not isinstance(value, (int, float)):
        value = float(value)

    return value


class SignalGenerator:
    """Signal generator class for managing signal generation.

    Provides a stateful interface for signal generation with configurable parameters.

    Examples:
        >>> generator = SignalGenerator(risk_tolerance=0.02)
        >>> signals = generator.generate(predictions, current_price=42000.0)
    """

    def __init__(self, risk_tolerance: float = 0.02):
        """Initialize signal generator.

        Args:
            risk_tolerance: Risk tolerance parameter for stop loss calculation
        """
        self.risk_tolerance = risk_tolerance

    def generate(
        self,
        predictions: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """Generate trading signals.

        Args:
            predictions: Model predictions
            current_price: Current market price

        Returns:
            Dictionary of trading signals
        """
        return generate_signals(predictions, current_price, self.risk_tolerance)
