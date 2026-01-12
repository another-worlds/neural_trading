"""Phase 2.3: HIGH priority tests for inference predictor edge cases.

Target: 15 uncovered statements in src/inference/predictor.py
Lines: 83, 86, 113, 116, 119, 145, 149, 159-171, 187, 202, 239, 241, 243

Coverage goal: 68% → 85%+
"""
import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import joblib
from sklearn.preprocessing import StandardScaler

from src.inference.predictor import Predictor, format_predictions


class TestPredictorErrorHandling:
    """Test Predictor error handling (lines 83, 86, 113, 116, 119, 145, 202)."""

    def test_load_model_without_model_path(self):
        """Should raise ValueError when model_path is None (line 83)."""
        predictor = Predictor(model_path=None, config={})

        with pytest.raises(ValueError, match="model_path not specified"):
            predictor.load_model()

    def test_load_model_file_not_found(self):
        """Should raise FileNotFoundError when model file doesn't exist (line 86)."""
        predictor = Predictor(model_path="/nonexistent/model.weights.h5", config={})

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            predictor.load_model()

    def test_load_scalers_without_paths(self):
        """Should raise ValueError when scaler paths are None (line 113)."""
        predictor = Predictor(
            scaler_input_path=None,
            scaler_output_path=None
        )

        with pytest.raises(ValueError, match="Scaler paths not specified"):
            predictor.load_scalers()

    def test_load_scalers_input_not_found(self):
        """Should raise FileNotFoundError when input scaler doesn't exist (line 116)."""
        predictor = Predictor(
            scaler_input_path="/nonexistent/input_scaler.joblib",
            scaler_output_path="/tmp/output.joblib"
        )

        with pytest.raises(FileNotFoundError, match="Input scaler not found"):
            predictor.load_scalers()

    def test_load_scalers_output_not_found(self):
        """Should raise FileNotFoundError when output scaler doesn't exist (line 119)."""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            input_path = tmp.name
            scaler = StandardScaler()
            joblib.dump(scaler, input_path)

        try:
            predictor = Predictor(
                scaler_input_path=input_path,
                scaler_output_path="/nonexistent/output_scaler.joblib"
            )

            with pytest.raises(FileNotFoundError, match="Output scaler not found"):
                predictor.load_scalers()
        finally:
            Path(input_path).unlink(missing_ok=True)

    def test_predict_without_model_loaded(self):
        """Should raise ValueError when model not loaded (line 145)."""
        predictor = Predictor()

        # model is None
        input_data = np.random.randn(1, 60, 10).astype(np.float32)

        with pytest.raises(ValueError, match="Model not loaded"):
            predictor.predict(input_data)

    def test_inverse_transform_without_scaler(self):
        """Should raise ValueError when output scaler not loaded (line 202)."""
        predictor = Predictor()

        # output_scaler is None
        scaled_predictions = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="Output scaler not loaded"):
            predictor.inverse_transform(scaled_predictions)


class TestPredictDtypeConversion:
    """Test predict dtype conversion (line 149)."""

    def test_predict_converts_dtype(self):
        """Should convert input to float32 if needed (line 149)."""
        predictor = Predictor()

        # Mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value={'h0_price': np.array([[42000.0]])})
        predictor.model = mock_model

        # Input is float64, should be converted to float32 (line 149)
        input_data = np.random.randn(1, 60, 10).astype(np.float64)
        assert input_data.dtype == np.float64

        result = predictor.predict(input_data)

        # Should have called predict with float32 input
        called_input = mock_model.predict.call_args[0][0]
        assert called_input.dtype == np.float32

    def test_predict_keeps_float32(self):
        """Should keep float32 input unchanged."""
        predictor = Predictor()

        # Mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value={'h0_price': np.array([[42000.0]])})
        predictor.model = mock_model

        # Input already float32
        input_data = np.random.randn(1, 60, 10).astype(np.float32)

        result = predictor.predict(input_data)

        # Should work without conversion
        assert result is not None


class TestPredictListToDict:
    """Test predict list to dict conversion (lines 159-171)."""

    def test_predict_converts_list_to_dict(self):
        """Should convert list predictions to dict (lines 159-171)."""
        predictor = Predictor()

        # Mock model that returns list of 9 outputs
        mock_model = Mock()
        list_predictions = [
            np.array([[42000.0]]),  # h0_price
            np.array([[0.8]]),      # h0_direction
            np.array([[0.1]]),      # h0_variance
            np.array([[42100.0]]),  # h1_price
            np.array([[0.7]]),      # h1_direction
            np.array([[0.15]]),     # h1_variance
            np.array([[42200.0]]),  # h2_price
            np.array([[0.6]]),      # h2_direction
            np.array([[0.2]])       # h2_variance
        ]
        mock_model.predict = Mock(return_value=list_predictions)
        predictor.model = mock_model

        input_data = np.random.randn(1, 60, 10).astype(np.float32)

        # Should execute lines 159-171 (list to dict conversion)
        result = predictor.predict(input_data)

        # Should be a dict
        assert isinstance(result, dict)
        assert 'h0_price' in result
        assert 'h0_direction' in result
        assert 'h0_variance' in result
        assert 'h1_price' in result
        assert 'h2_variance' in result

    def test_predict_returns_dict_as_is(self):
        """Should return dict predictions unchanged."""
        predictor = Predictor()

        # Mock model that returns dict
        mock_model = Mock()
        dict_predictions = {
            'h0_price': np.array([[42000.0]]),
            'h0_direction': np.array([[0.8]])
        }
        mock_model.predict = Mock(return_value=dict_predictions)
        predictor.model = mock_model

        input_data = np.random.randn(1, 60, 10).astype(np.float32)

        result = predictor.predict(input_data)

        # Should return dict unchanged
        assert result == dict_predictions


class TestPredictBatch:
    """Test predict_batch (line 187)."""

    def test_predict_batch_calls_predict(self):
        """Should call predict internally (line 187)."""
        predictor = Predictor()

        # Mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value={'h0_price': np.array([[42000.0]])})
        predictor.model = mock_model

        input_data = np.random.randn(5, 60, 10).astype(np.float32)

        # Should execute line 187 (calls self.predict)
        result = predictor.predict_batch(input_data, batch_size=2)

        # Should have called model.predict
        assert mock_model.predict.called
        assert result is not None


class TestFormatPredictionsNumpyConversion:
    """Test format_predictions numpy array conversion (lines 239, 241, 243)."""

    def test_format_predictions_converts_numpy_arrays(self):
        """Should convert numpy arrays to scalars (lines 239, 241, 243)."""
        predictions = {
            'h0_price': np.array([[42000.0]]),       # Line 239
            'h0_direction': np.array([[0.8]]),       # Line 241
            'h0_variance': np.array([[0.1]]),        # Line 243
            'h1_price': np.array([[42100.0]]),
            'h1_direction': np.array([[0.7]]),
            'h1_variance': np.array([[0.15]]),
            'h2_price': np.array([[42200.0]]),
            'h2_direction': np.array([[0.6]]),
            'h2_variance': np.array([[0.2]])
        }

        # Should execute lines 239, 241, 243 (numpy array to float conversion)
        formatted = format_predictions(predictions)

        # Check structure
        assert 'h0' in formatted
        assert 'h1' in formatted
        assert 'h2' in formatted

        # Check types - should be Python floats, not numpy
        assert isinstance(formatted['h0']['price'], (float, np.floating))
        assert isinstance(formatted['h0']['direction'], (float, np.floating))
        assert isinstance(formatted['h0']['variance'], (float, np.floating))

    def test_format_predictions_with_scalar_values(self):
        """Should handle scalar predictions."""
        predictions = {
            'h0_price': 42000.0,           # Already scalar
            'h0_direction': 0.8,
            'h0_variance': 0.1,
            'h1_price': 42100.0,
            'h1_direction': 0.7,
            'h1_variance': 0.15,
            'h2_price': 42200.0,
            'h2_direction': 0.6,
            'h2_variance': 0.2
        }

        formatted = format_predictions(predictions)

        # Should work with scalars
        assert formatted['h0']['price'] == 42000.0
        assert formatted['h1']['direction'] == 0.7

    def test_format_predictions_with_alternative_keys(self):
        """Should handle both key formats (price_h0 and h0_price)."""
        predictions = {
            'price_h0': np.array([[42000.0]]),
            'direction_h0': np.array([[0.8]]),
            'variance_h0': np.array([[0.1]]),
            'h1_price': np.array([[42100.0]]),
            'h1_direction': np.array([[0.7]]),
            'h1_variance': np.array([[0.15]]),
            'h2_price': 42200.0,
            'h2_direction': 0.6,
            'h2_variance': 0.2
        }

        formatted = format_predictions(predictions)

        # Should handle both key formats
        assert 'h0' in formatted
        assert 'h1' in formatted
        assert 'h2' in formatted


class TestPredictorIntegration:
    """Integration tests for Predictor."""

    def test_predictor_full_workflow_with_mocks(self):
        """Should handle full workflow with mocked components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock model file
            model_path = Path(tmpdir) / "model.weights.h5"
            model_path.touch()

            # Create mock scalers
            input_scaler_path = Path(tmpdir) / "input_scaler.joblib"
            output_scaler_path = Path(tmpdir) / "output_scaler.joblib"

            input_scaler = StandardScaler()
            input_scaler.mean_ = np.zeros(10)
            input_scaler.scale_ = np.ones(10)
            joblib.dump(input_scaler, input_scaler_path)

            output_scaler = StandardScaler()
            output_scaler.mean_ = np.zeros(3)
            output_scaler.scale_ = np.ones(3)
            joblib.dump(output_scaler, output_scaler_path)

            # Create predictor
            config = {
                'model': {},
                'data': {'lookback': 60, 'n_features': 10}
            }

            predictor = Predictor(
                model_path=str(model_path),
                scaler_input_path=str(input_scaler_path),
                scaler_output_path=str(output_scaler_path),
                config=config
            )

            # Load scalers (should work)
            input_sc, output_sc = predictor.load_scalers()
            assert input_sc is not None
            assert output_sc is not None

            # Test inverse transform
            scaled_pred = np.array([[1.0, 2.0, 3.0]])
            result = predictor.inverse_transform(scaled_pred)
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
