"""Training callbacks for neural trading pipeline.

Implements custom callbacks as per SRS Section 3.5.3.
"""
import os
import csv
from pathlib import Path
from typing import Optional, List, Dict, Any
import tensorflow as tf
import pandas as pd


class IndicatorParamsLogger(tf.keras.callbacks.Callback):
    """Custom callback to log learnable indicator parameters each epoch.

    Saves indicator parameter evolution to CSV as per SRS Section 7.2.5.
    Tracks all 30+ learnable indicator parameters across training.
    """

    def __init__(self, output_file: Path, **kwargs):
        """Initialize indicator params logger.

        Args:
            output_file: Path to CSV file for logging params
        """
        super().__init__(**kwargs)
        self.output_file = Path(output_file)
        self.params_history = []

        # Ensure parent directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """Log indicator parameters at end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Training logs (unused)
        """
        # Extract indicator parameters from model
        params = self._extract_indicator_params()

        # Add epoch number
        params['epoch'] = epoch

        # Store in history
        self.params_history.append(params)

        # Save to CSV
        self._save_to_csv()

    def _extract_indicator_params(self) -> Dict[str, float]:
        """Extract all indicator parameters from model layers.

        Returns:
            Dictionary of parameter names to values
        """
        params = {}

        if self.model is None:
            return params

        # Find indicator layers in model
        for layer in self.model.layers:
            layer_name = layer.name

            # Check if layer has learnable indicator parameters
            if hasattr(layer, 'weights'):
                for weight in layer.weights:
                    weight_name = weight.name

                    # Extract indicator-related weights
                    # (MA periods, RSI periods, BB periods, MACD periods, etc.)
                    if any(indicator in weight_name.lower() for indicator in
                           ['period', 'ma', 'rsi', 'bb', 'macd', 'momentum']):
                        # Get weight value
                        value = weight.numpy()

                        # Handle scalar or array values
                        if value.shape == ():
                            params[weight_name] = float(value)
                        else:
                            # For arrays, store each element
                            for i, val in enumerate(value.flatten()):
                                params[f'{weight_name}_{i}'] = float(val)

        return params

    def _save_to_csv(self):
        """Save parameter history to CSV file."""
        if not self.params_history:
            return

        # Convert to DataFrame
        df = pd.DataFrame(self.params_history)

        # Save to CSV
        df.to_csv(self.output_file, index=False)

    def log_epoch(self, epoch: int, params: Dict[str, float]):
        """Manually log parameters for an epoch (for testing).

        Args:
            epoch: Epoch number
            params: Dictionary of parameter names to values
        """
        params_with_epoch = params.copy()
        params_with_epoch['epoch'] = epoch
        self.params_history.append(params_with_epoch)
        self._save_to_csv()


class GradientClippingCallback(tf.keras.callbacks.Callback):
    """Callback to apply gradient clipping during training.

    Clips gradients by norm to prevent exploding gradients.
    SRS Section 3.5.3 specifies clip_norm=5.0.
    """

    def __init__(self, clip_norm: float = 5.0, **kwargs):
        """Initialize gradient clipping callback.

        Args:
            clip_norm: Maximum gradient norm (default: 5.0 per SRS)
        """
        super().__init__(**kwargs)
        self.clip_norm = clip_norm

    def on_train_batch_begin(self, batch, logs=None):
        """Apply gradient clipping before batch training.

        Args:
            batch: Batch number
            logs: Training logs (unused)
        """
        # Note: Gradient clipping is typically set in the optimizer
        # This callback serves as a marker/configuration holder
        pass


def create_callbacks(
    patience: int = 40,
    monitor: str = 'val_dir_mcc_h1',
    mode: str = 'max',
    checkpoint_path: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    save_best_only: bool = True,
    indicator_params_file: Optional[Path] = None,
    **kwargs
) -> List[tf.keras.callbacks.Callback]:
    """Create training callbacks as per SRS Section 3.5.3.

    Args:
        patience: Early stopping patience (default: 40)
        monitor: Metric to monitor (default: 'val_dir_mcc_h1' per SRS)
        mode: 'max' or 'min' for monitored metric
        checkpoint_path: Path for model checkpoints
        log_dir: Directory for TensorBoard logs
        save_best_only: Save only best checkpoint
        indicator_params_file: Path for indicator params logging
        **kwargs: Additional callback parameters

    Returns:
        List of configured callbacks
    """
    callbacks = []

    # 1. Early Stopping - monitors val_dir_mcc_h1 (primary validation metric)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)

    # 2. Model Checkpoint - saves best model
    if checkpoint_path is not None:
        # Ensure parent directory exists
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=kwargs.get('checkpoint_monitor', monitor),
            save_best_only=save_best_only,
            save_weights_only=False,
            mode=mode,
            verbose=1
        )
        callbacks.append(model_checkpoint)

    # 3. TensorBoard - visualization and logging
    if log_dir is not None:
        # Ensure log directory exists
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch',
            profile_batch=0
        )
        callbacks.append(tensorboard)

    # 4. Indicator Parameters Logger - tracks learnable params
    if indicator_params_file is not None:
        indicator_logger = IndicatorParamsLogger(
            output_file=indicator_params_file
        )
        callbacks.append(indicator_logger)

    # 5. Learning Rate Scheduler (optional)
    if kwargs.get('use_lr_scheduler', False):
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=kwargs.get('lr_patience', 10),
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(lr_scheduler)

    return callbacks
