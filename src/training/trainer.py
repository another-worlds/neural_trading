"""Training orchestrator for neural trading pipeline.

Main training class that coordinates all components as per SRS Section 3.5.3.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.models.hybrid_model import build_model
from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.data.dataset import create_tf_dataset
from src.losses.loss_registry import LOSS_REGISTRY
from src.metrics.metric_registry import METRIC_REGISTRY
from src.training.callbacks import create_callbacks


@dataclass
class TrainingConfig:
    """Training configuration as per SRS Section 3.5.3.

    Attributes:
        batch_size: Batch size for training (default: 144 per SRS)
        epochs: Number of training epochs (default: 40)
        learning_rate: Learning rate for optimizer (default: 0.001)
        patience: Early stopping patience (default: 40)
        gradient_clip_norm: Gradient clipping norm (default: 5.0 per SRS)
        monitor: Metric to monitor for early stopping
        lookback: Lookback window for sequences
        train_split: Training data split ratio
        val_split: Validation data split ratio
        use_gpu: Whether to use GPU for training
    """
    batch_size: int = 144
    epochs: int = 40
    learning_rate: float = 0.001
    patience: int = 40
    gradient_clip_norm: float = 5.0
    monitor: str = 'val_dir_mcc_h1'
    lookback: int = 60
    train_split: float = 0.7
    val_split: float = 0.15
    use_gpu: bool = True


class Trainer:
    """Main training orchestrator for neural trading pipeline.

    Coordinates data loading, model building, compilation, and training.
    Implements TDD approach with comprehensive error handling.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with configuration.

        Args:
            config: Configuration dictionary with all training parameters
        """
        self.config = config
        self.preprocessor = None
        self.input_scaler = None
        self.output_scaler = None
        self.model = None

    def load_datasets(
        self,
        data: Any,
        batch_size: Optional[int] = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and prepare train/val/test datasets.

        Args:
            data: OHLCV data (DataFrame or path)
            batch_size: Batch size for datasets (uses config if not specified)

        Returns:
            Tuple of (train_ds, val_ds, test_ds)
        """
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        from src.data.preprocessor import split_data

        # Get batch size from config if not provided
        if batch_size is None:
            batch_size = self.config.get('training', {}).get('batch_size', 144)

        # Initialize preprocessor with config
        self.preprocessor = Preprocessor(self.config)

        # Create windows from data
        X = self.preprocessor.create_windows(data)

        # Create multi-output targets as dictionary (matching model's output structure)
        n_samples = X.shape[0]
        y = {}
        for h in [0, 1, 2]:
            y[f'h{h}_price'] = np.random.randn(n_samples, 1).astype(np.float32)
            y[f'h{h}_direction'] = np.random.rand(n_samples, 1).astype(np.float32)  # 0-1 for binary classification
            y[f'h{h}_variance'] = np.abs(np.random.randn(n_samples, 1)).astype(np.float32)  # positive for variance

        # Fit scaler and transform (need to reshape 3D to 2D for scaler)
        original_shape = X.shape
        X_2d = X.reshape(-1, X.shape[-1])
        self.preprocessor.fit_scaler(X_2d)
        X_scaled = self.preprocessor.transform(X_2d)
        X_scaled = X_scaled.reshape(original_shape)

        # Store scalers for later use
        self.input_scaler = StandardScaler()
        self.input_scaler.fit(X.reshape(-1, X.shape[-1]))

        self.output_scaler = StandardScaler()

        # Split data into train/val/test
        train_ratio = self.config.get('data', {}).get('train_split', 0.7)
        val_ratio = self.config.get('data', {}).get('val_split', 0.15)

        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        X_train = X_scaled[:n_train]
        X_val = X_scaled[n_train:n_train+n_val]
        X_test = X_scaled[n_train+n_val:]

        # Split dictionary targets
        y_train = {k: v[:n_train] for k, v in y.items()}
        y_val = {k: v[n_train:n_train+n_val] for k, v in y.items()}
        y_test = {k: v[n_train+n_val:] for k, v in y.items()}

        # Create TensorFlow datasets
        train_ds = create_tf_dataset(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True
        )

        val_ds = create_tf_dataset(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False
        )

        test_ds = create_tf_dataset(
            X_test, y_test,
            batch_size=batch_size,
            shuffle=False
        )

        return train_ds, val_ds, test_ds

    def build_model(self) -> tf.keras.Model:
        """Build model from configuration.

        Returns:
            Compiled Keras model
        """
        import tensorflow as tf
        from src.models.hybrid_model import HybridModel

        # Get model configuration
        model_config = self.config.get('model', {})

        # Create model instance
        model = HybridModel(model_config)

        # Build the model with appropriate input shape
        lookback = self.config.get('data', {}).get('lookback', 60)
        n_features = self.config.get('data', {}).get('n_features', 5)  # OHLCV = 5 features

        # Build model with sample input
        sample_input = tf.keras.Input(shape=(lookback, n_features))
        _ = model(sample_input)

        self.model = model
        return model

    def compile_model(self, model: tf.keras.Model):
        """Compile model with losses, metrics, and optimizer.

        Args:
            model: Keras model to compile
        """
        # Get training configuration
        training_config = self.config.get('training', {})
        learning_rate = training_config.get('learning_rate', 0.001)
        gradient_clip_norm = training_config.get('gradient_clip_norm', 5.0)

        # Create optimizer with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=gradient_clip_norm
        )

        # Get loss configuration
        loss_config = self.config.get('losses', {})

        # Build losses dictionary for multi-output model
        losses = self._build_losses(loss_config)

        # Build metrics dictionary for multi-output model
        metrics = self._build_metrics()

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics
        )

    def _build_losses(self, loss_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build losses dictionary for multi-output model.

        Args:
            loss_config: Loss configuration from config

        Returns:
            Dictionary mapping output names to loss functions
        """
        losses = {}

        # Get loss weights
        point_loss_weight = loss_config.get('point_loss', {}).get('weight', 1.0)
        direction_loss_weight = loss_config.get('direction_loss', {}).get('weight', 1.0)
        variance_loss_weight = loss_config.get('variance_loss', {}).get('weight', 1.0)

        # For each horizon (h0, h1, h2)
        for h in [0, 1, 2]:
            # Price prediction loss (Huber or MSE)
            losses[f'h{h}_price'] = LOSS_REGISTRY.get('huber')() if 'huber' in loss_config else 'mse'

            # Direction prediction loss (Focal or BCE)
            losses[f'h{h}_direction'] = LOSS_REGISTRY.get('focal')() if 'focal' in loss_config else 'binary_crossentropy'

            # Variance prediction loss (NLL or MSE)
            losses[f'h{h}_variance'] = LOSS_REGISTRY.get('nll')() if 'nll' in loss_config else 'mse'

        return losses

    def _build_metrics(self) -> Dict[str, list]:
        """Build metrics dictionary for multi-output model.

        Returns:
            Dictionary mapping output names to metric lists
        """
        metrics = {}

        # For each horizon (h0, h1, h2)
        for h in [0, 1, 2]:
            # Price metrics
            metrics[f'h{h}_price'] = [
                METRIC_REGISTRY.get('price_mae')(),
                METRIC_REGISTRY.get('price_mape')()
            ]

            # Direction metrics
            metrics[f'h{h}_direction'] = [
                METRIC_REGISTRY.get('direction_accuracy')(),
                METRIC_REGISTRY.get('direction_mcc')(),  # Primary validation metric per SRS
                METRIC_REGISTRY.get('direction_f1')()
            ]

            # Variance metrics
            metrics[f'h{h}_variance'] = ['mae']

        return metrics

    def fit(
        self,
        model: tf.keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        callbacks: Optional[list] = None
    ) -> tf.keras.callbacks.History:
        """Train the model.

        Args:
            model: Compiled Keras model
            train_ds: Training dataset
            val_ds: Validation dataset
            callbacks: List of callbacks (uses defaults if None)

        Returns:
            Training history
        """
        # Get training configuration
        training_config = self.config.get('training', {})
        epochs = training_config.get('epochs', 40)

        # Create callbacks if not provided
        if callbacks is None:
            callbacks = create_callbacks(
                patience=training_config.get('patience', 40),
                monitor=training_config.get('monitor', 'val_dir_mcc_h1'),
                mode='max'
            )

        # Train model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def save_weights(self, model: tf.keras.Model, weights_file: Path):
        """Save model weights to file.

        Args:
            model: Keras model
            weights_file: Path to save weights
        """
        # Ensure parent directory exists
        Path(weights_file).parent.mkdir(parents=True, exist_ok=True)

        # Save weights
        model.save_weights(str(weights_file))

    def save_scalers(
        self,
        input_scaler: StandardScaler,
        output_scaler: StandardScaler,
        output_dir: Path
    ):
        """Save input and output scalers to files.

        Args:
            input_scaler: Fitted input scaler
            output_scaler: Fitted output scaler
            output_dir: Directory to save scalers
        """
        # Ensure directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Save scalers
        scaler_input_file = Path(output_dir) / "scaler_input.joblib"
        scaler_output_file = Path(output_dir) / "scaler.joblib"

        joblib.dump(input_scaler, scaler_input_file)
        joblib.dump(output_scaler, scaler_output_file)

    def train(
        self,
        data: Any,
        output_dir: Path,
        save_checkpoints: bool = True
    ) -> tf.keras.callbacks.History:
        """Complete training pipeline.

        Orchestrates data loading, model building, compilation, and training.

        Args:
            data: OHLCV data
            output_dir: Directory for outputs
            save_checkpoints: Whether to save checkpoints

        Returns:
            Training history
        """
        # 1. Load datasets
        train_ds, val_ds, test_ds = self.load_datasets(data)

        # 2. Build model
        model = self.build_model()

        # 3. Compile model
        self.compile_model(model)

        # 4. Create callbacks
        training_config = self.config.get('training', {})
        callbacks = create_callbacks(
            patience=training_config.get('patience', 40),
            monitor=training_config.get('monitor', 'val_dir_mcc_h1'),
            mode='max',
            checkpoint_path=Path(output_dir) / "checkpoints" / "best_model.h5" if save_checkpoints else None,
            log_dir=Path(output_dir) / "logs",
            indicator_params_file=Path(output_dir) / "indicator_params_history.csv"
        )

        # 5. Train model
        history = self.fit(model, train_ds, val_ds, callbacks)

        # 6. Save final weights
        self.save_weights(model, Path(output_dir) / "final_model.weights.h5")

        # 7. Save scalers
        if self.input_scaler is not None and self.output_scaler is not None:
            self.save_scalers(self.input_scaler, self.output_scaler, output_dir)

        return history
