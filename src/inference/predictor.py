"""Model inference and prediction generation.

Implements prediction interface as per SRS Section 3.6.
Handles model loading, prediction, and post-processing.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.models.hybrid_model import HybridModel


@dataclass
class InferenceConfig:
    """Configuration for inference.

    Attributes:
        model_path: Path to trained model weights
        scaler_input_path: Path to input scaler
        scaler_output_path: Path to output scaler
        batch_size: Batch size for prediction (default: 32)
    """
    model_path: str
    scaler_input_path: str
    scaler_output_path: str
    batch_size: int = 32


class Predictor:
    """Predictor class for model inference.

    Handles loading trained models, scalers, and generating predictions.
    Ensures inference latency <1 second per SRS NFR-1.1.

    Examples:
        >>> predictor = Predictor(
        ...     model_path="model.weights.h5",
        ...     scaler_input_path="scaler_input.joblib",
        ...     scaler_output_path="scaler.joblib",
        ...     config=config_dict
        ... )
        >>> predictions = predictor.predict(input_data)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_input_path: Optional[str] = None,
        scaler_output_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize predictor.

        Args:
            model_path: Path to trained model weights
            scaler_input_path: Path to input scaler
            scaler_output_path: Path to output scaler
            config: Configuration dictionary
        """
        self.model_path = model_path
        self.scaler_input_path = scaler_input_path
        self.scaler_output_path = scaler_output_path
        self.config = config or {}

        self.model = None
        self.input_scaler = None
        self.output_scaler = None

    def load_model(self) -> tf.keras.Model:
        """Load trained model from weights file.

        Returns:
            Loaded Keras model

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if self.model_path is None:
            raise ValueError("model_path not specified")

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Build model from config
        model = HybridModel(self.config.get('model', {}))

        # Build with sample input shape
        lookback = self.config.get('data', {}).get('lookback', 60)
        n_features = self.config.get('data', {}).get('n_features', 10)
        sample_input = tf.keras.Input(shape=(lookback, n_features))
        _ = model(sample_input)

        # Load weights
        model.load_weights(self.model_path)

        self.model = model
        return model

    def load_scalers(self) -> Tuple[StandardScaler, StandardScaler]:
        """Load input and output scalers.

        Returns:
            Tuple of (input_scaler, output_scaler)

        Raises:
            FileNotFoundError: If scaler files don't exist
        """
        if self.scaler_input_path is None or self.scaler_output_path is None:
            raise ValueError("Scaler paths not specified")

        if not Path(self.scaler_input_path).exists():
            raise FileNotFoundError(f"Input scaler not found: {self.scaler_input_path}")

        if not Path(self.scaler_output_path).exists():
            raise FileNotFoundError(f"Output scaler not found: {self.scaler_output_path}")

        # Load scalers
        self.input_scaler = joblib.load(self.scaler_input_path)
        self.output_scaler = joblib.load(self.scaler_output_path)

        return self.input_scaler, self.output_scaler

    def predict(
        self,
        input_data: np.ndarray,
        inverse_transform: bool = False
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Run model inference on input data.

        Args:
            input_data: Input features of shape (batch_size, lookback, n_features)
            inverse_transform: Whether to inverse transform predictions

        Returns:
            Dictionary of predictions with keys: h0_price, h0_direction, h0_variance, etc.

        Raises:
            ValueError: If model not loaded
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Ensure input is float32 for TensorFlow compatibility
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)

        # Run inference
        predictions = self.model.predict(input_data, verbose=0)

        # If predictions is a dict (multi-output model), return as is
        if isinstance(predictions, dict):
            return predictions

        # If predictions is a list (Keras functional model), convert to dict
        if isinstance(predictions, list):
            # Assuming 9 outputs: h0_price, h0_direction, h0_variance, h1_*, h2_*
            pred_dict = {}
            output_names = []
            for h in [0, 1, 2]:
                output_names.extend([f'h{h}_price', f'h{h}_direction', f'h{h}_variance'])

            for i, name in enumerate(output_names):
                pred_dict[name] = predictions[i]

            return pred_dict

        return predictions

    def predict_batch(
        self,
        input_data: np.ndarray,
        batch_size: int = 32
    ) -> Dict[str, np.ndarray]:
        """Run inference on batch of samples.

        Args:
            input_data: Input features of shape (n_samples, lookback, n_features)
            batch_size: Batch size for prediction

        Returns:
            Dictionary of predictions
        """
        return self.predict(input_data)

    def inverse_transform(self, scaled_predictions: np.ndarray) -> np.ndarray:
        """Inverse transform scaled predictions back to original scale.

        Args:
            scaled_predictions: Scaled predictions

        Returns:
            Predictions in original scale

        Raises:
            ValueError: If output scaler not loaded
        """
        if self.output_scaler is None:
            raise ValueError("Output scaler not loaded. Call load_scalers() first.")

        return self.output_scaler.inverse_transform(scaled_predictions)


def format_predictions(predictions: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Format raw predictions into structured output.

    Args:
        predictions: Raw predictions from model

    Returns:
        Formatted predictions with structure:
        {
            'h0': {'price': ..., 'direction': ..., 'variance': ..., 'confidence': ...},
            'h1': {'price': ..., 'direction': ..., 'variance': ..., 'confidence': ...},
            'h2': {'price': ..., 'direction': ..., 'variance': ..., 'confidence': ...}
        }
    """
    from src.utils.helper_functions import calculate_confidence

    formatted = {}

    for h in [0, 1, 2]:
        horizon_key = f'h{h}'

        # Extract predictions for this horizon
        price_key = f'price_h{h}' if f'price_h{h}' in predictions else f'h{h}_price'
        direction_key = f'direction_h{h}' if f'direction_h{h}' in predictions else f'h{h}_direction'
        variance_key = f'variance_h{h}' if f'variance_h{h}' in predictions else f'h{h}_variance'

        price = predictions.get(price_key, predictions.get(f'h{h}_price', 0.0))
        direction = predictions.get(direction_key, predictions.get(f'h{h}_direction', 0.5))
        variance = predictions.get(variance_key, predictions.get(f'h{h}_variance', 0.01))

        # Convert to scalar if needed
        if isinstance(price, np.ndarray):
            price = float(price.flatten()[0])
        if isinstance(direction, np.ndarray):
            direction = float(direction.flatten()[0])
        if isinstance(variance, np.ndarray):
            variance = float(variance.flatten()[0])

        # Calculate confidence from variance
        confidence = calculate_confidence(variance)

        formatted[horizon_key] = {
            'price': price,
            'direction': direction,
            'variance': variance,
            'confidence': confidence
        }

    return formatted
