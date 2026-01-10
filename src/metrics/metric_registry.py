"""Metric registry for automatic component registration.

Implements the pipeline doctrine of automatic component integration via decorators.
All custom metrics are registered automatically on import, enabling configuration-driven
metric selection.
"""
from typing import Dict, Type


class MetricRegistry:
    """Registry for custom metrics.

    Enables automatic metric registration via decorator pattern.
    Supports configuration-driven metric selection.

    Examples:
        >>> registry = MetricRegistry()
        >>> @registry.register('my_metric')
        ... class MyMetric(tf.keras.metrics.Metric):
        ...     def update_state(self, y_true, y_pred, sample_weight=None):
        ...         pass
        >>> metric_class = registry.get('my_metric')
    """

    def __init__(self):
        """Initialize empty metric registry."""
        self.metrics: Dict[str, Type] = {}

    def register(self, name: str):
        """Decorator to register a metric.

        Args:
            name: Name to register the metric under.

        Returns:
            Decorator function that registers the metric class.

        Examples:
            >>> @METRIC_REGISTRY.register('direction_accuracy')
            ... class DirectionAccuracy(tf.keras.metrics.Metric):
            ...     pass
        """
        def decorator(metric_class: Type) -> Type:
            """Register metric class and return it unchanged."""
            self.metrics[name] = metric_class
            return metric_class
        return decorator

    def get(self, name: str) -> Type:
        """Retrieve registered metric by name.

        Args:
            name: Name of the registered metric.

        Returns:
            Metric class.

        Raises:
            KeyError: If metric name not found in registry.
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found in registry")
        return self.metrics[name]

    def list_metrics(self):
        """List all registered metric names.

        Returns:
            List of registered metric names.
        """
        return list(self.metrics.keys())


# Global metric registry instance
METRIC_REGISTRY = MetricRegistry()


def register_metric(name: str):
    """Convenience function to register metric with global registry.

    Args:
        name: Name to register the metric under.

    Returns:
        Decorator function.

    Examples:
        >>> @register_metric('direction_mcc')
        ... class DirectionMCC(tf.keras.metrics.Metric):
        ...     pass
    """
    return METRIC_REGISTRY.register(name)
