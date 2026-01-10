"""Loss function registry for automatic component registration.

Implements the pipeline doctrine of automatic component integration via decorators.
All custom losses are registered automatically on import, enabling configuration-driven
loss selection and composition.
"""
from typing import Dict, Type, Any


class LossRegistry:
    """Registry for loss functions.

    Enables automatic loss registration via decorator pattern.
    Supports configuration-driven loss selection and composition.

    Examples:
        >>> registry = LossRegistry()
        >>> @registry.register('my_loss')
        ... class MyLoss(tf.keras.losses.Loss):
        ...     def call(self, y_true, y_pred):
        ...         return tf.reduce_mean(tf.square(y_true - y_pred))
        >>> loss_class = registry.get('my_loss')
    """

    def __init__(self):
        """Initialize empty loss registry."""
        self.losses: Dict[str, Type] = {}

    def register(self, name: str):
        """Decorator to register a loss function.

        Args:
            name: Name to register the loss under.

        Returns:
            Decorator function that registers the loss class.

        Examples:
            >>> @LOSS_REGISTRY.register('focal')
            ... class FocalLoss(tf.keras.losses.Loss):
            ...     pass
        """
        def decorator(loss_class: Type) -> Type:
            """Register loss class and return it unchanged."""
            self.losses[name] = loss_class
            return loss_class
        return decorator

    def get(self, name: str) -> Type:
        """Retrieve registered loss by name.

        Args:
            name: Name of the registered loss.

        Returns:
            Loss class.

        Raises:
            KeyError: If loss name not found in registry.
        """
        if name not in self.losses:
            raise KeyError(f"Loss '{name}' not found in registry")
        return self.losses[name]

    def list_losses(self):
        """List all registered loss names.

        Returns:
            List of registered loss names.
        """
        return list(self.losses.keys())


# Global loss registry instance
LOSS_REGISTRY = LossRegistry()


def register_loss(name: str):
    """Convenience function to register loss with global registry.

    Args:
        name: Name to register the loss under.

    Returns:
        Decorator function.

    Examples:
        >>> @register_loss('huber')
        ... class HuberLoss(tf.keras.losses.Loss):
        ...     pass
    """
    return LOSS_REGISTRY.register(name)
