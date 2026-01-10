"""Learnable technical indicators and indicator registry.

This module provides:
1. IndicatorRegistry for automatic component registration
2. Learnable indicator implementations (to be added in Phase 4)

Implements the pipeline doctrine of automatic component integration via decorators.
All indicators are registered automatically on import.
"""
from typing import Dict, Type


class IndicatorRegistry:
    """Registry for learnable technical indicators.

    Enables automatic indicator registration via decorator pattern.
    Supports configuration-driven indicator selection.

    Examples:
        >>> registry = IndicatorRegistry()
        >>> @registry.register('moving_average')
        ... class LearnableMA:
        ...     def __init__(self, **kwargs):
        ...         pass
        >>> indicator_class = registry.get('moving_average')
    """

    def __init__(self):
        """Initialize empty indicator registry."""
        self.indicators: Dict[str, Type] = {}

    def register(self, name: str):
        """Decorator to register an indicator.

        Args:
            name: Name to register the indicator under.

        Returns:
            Decorator function that registers the indicator class.

        Examples:
            >>> @INDICATOR_REGISTRY.register('learnable_rsi')
            ... class LearnableRSI:
            ...     pass
        """
        def decorator(indicator_class: Type) -> Type:
            """Register indicator class and return it unchanged."""
            self.indicators[name] = indicator_class
            return indicator_class
        return decorator

    def get(self, name: str) -> Type:
        """Retrieve registered indicator by name.

        Args:
            name: Name of the registered indicator.

        Returns:
            Indicator class.

        Raises:
            KeyError: If indicator name not found in registry.
        """
        if name not in self.indicators:
            raise KeyError(f"Indicator '{name}' not found in registry")
        return self.indicators[name]

    def list_indicators(self):
        """List all registered indicator names.

        Returns:
            List of registered indicator names.
        """
        return list(self.indicators.keys())


# Global indicator registry instance
INDICATOR_REGISTRY = IndicatorRegistry()


def register_indicator(name: str):
    """Convenience function to register indicator with global registry.

    Args:
        name: Name to register the indicator under.

    Returns:
        Decorator function.

    Examples:
        >>> @register_indicator('learnable_macd')
        ... class LearnableMacd:
        ...     pass
    """
    return INDICATOR_REGISTRY.register(name)


# ============================================================================
# Stub classes for learnable indicators
# These are placeholder implementations to satisfy imports.
# Full implementations will be added in Phase 4: Learnable Indicators
# ============================================================================

@register_indicator('ma')
class LearnableMA:
    """Learnable Moving Average (stub - to be implemented in Phase 4)."""
    pass


@register_indicator('rsi')
class LearnableRSI:
    """Learnable RSI (stub - to be implemented in Phase 4)."""
    pass


@register_indicator('bollinger_bands')
class LearnableBollingerBands:
    """Learnable Bollinger Bands (stub - to be implemented in Phase 4)."""
    pass


@register_indicator('macd')
class LearnableMacd:
    """Learnable MACD (stub - to be implemented in Phase 4)."""
    pass


@register_indicator('custom_macd')
class LearnableCustomMacd:
    """Learnable Custom MACD (stub - to be implemented in Phase 4)."""
    pass


@register_indicator('momentum')
class LearnableMomentum:
    """Learnable Momentum (stub - to be implemented in Phase 4)."""
    pass
