"""Training infrastructure for neural trading pipeline.

Provides training orchestration, callbacks, and configuration.
"""
from src.training.trainer import Trainer, TrainingConfig
from src.training.callbacks import (
    IndicatorParamsLogger,
    GradientClippingCallback,
    create_callbacks
)

__all__ = [
    'Trainer',
    'TrainingConfig',
    'IndicatorParamsLogger',
    'GradientClippingCallback',
    'create_callbacks'
]
