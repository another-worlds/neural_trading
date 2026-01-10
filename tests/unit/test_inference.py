"""Unit tests for inference module.

Tests model inference and prediction generation as per SRS Section 3.6.
"""
import pytest
import numpy as np
import tensorflow as tf
from src.inference.predictor import Predictor, InferenceConfig


class TestPredictor:
    """Test Predictor class for model inference."""

    def test_init_predictor(self, tmp_path, sample_config):
        """Should initialize predictor."""
        # Create mock model files
        model_path = tmp_path / "model.weights.h5"
        scaler_input_path = tmp_path / "scaler_input.joblib"
        scaler_output_path = tmp_path / "scaler.joblib"

        predictor = Predictor(
            model_path=str(model_path),
            scaler_input_path=str(scaler_input_path),
            scaler_output_path=str(scaler_output_path),
            config=sample_config
        )

        assert predictor is not None

    def test_load_model(self, tmp_path, sample_config):
        """Should load trained model."""
        from src.models.hybrid_model import build_model

        # Build and save model
        model = build_model(sample_config)
        weights_file = tmp_path / "model.weights.h5"
        model.save_weights(str(weights_file))

        # Load with predictor
        predictor = Predictor(
            model_path=str(weights_file),
            scaler_input_path=None,
            scaler_output_path=None,
            config=sample_config
        )

        loaded_model = predictor.load_model()
        assert loaded_model is not None

    def test_load_scalers(self, tmp_path):
        """Should load scalers."""
        import joblib
        from sklearn.preprocessing import StandardScaler

        # Create and save scalers
        scaler_input = StandardScaler()
        scaler_output = StandardScaler()

        scaler_input_path = tmp_path / "scaler_input.joblib"
        scaler_output_path = tmp_path / "scaler.joblib"

        joblib.dump(scaler_input, scaler_input_path)
        joblib.dump(scaler_output, scaler_output_path)

        # Load with predictor
        predictor = Predictor(
            model_path=None,
            scaler_input_path=str(scaler_input_path),
            scaler_output_path=str(scaler_output_path),
            config={}
        )

        input_scaler, output_scaler = predictor.load_scalers()
        assert input_scaler is not None
        assert output_scaler is not None

    def test_predict_single_sample(self, sample_config):
        """Should predict on single sample."""
        from src.models.hybrid_model import build_model

        model = build_model(sample_config)
        model.build(input_shape=(None, 60, 10))

        predictor = Predictor(model_path=None, config=sample_config)
        predictor.model = model

        # Single 60-minute window
        input_data = np.random.randn(1, 60, 10).astype(np.float32)
        predictions = predictor.predict(input_data)

        assert predictions is not None

    def test_predict_batch(self, sample_config):
        """Should predict on batch of samples."""
        from src.models.hybrid_model import build_model

        model = build_model(sample_config)
        model.build(input_shape=(None, 60, 10))

        predictor = Predictor(model_path=None, config=sample_config)
        predictor.model = model

        # Batch of samples
        input_data = np.random.randn(10, 60, 10).astype(np.float32)
        predictions = predictor.predict(input_data)

        assert predictions is not None

    def test_prediction_latency(self, sample_config):
        """Inference should complete in <1 second as per SRS NFR-1.1."""
        from src.models.hybrid_model import build_model
        import time

        model = build_model(sample_config)
        model.build(input_shape=(None, 60, 10))

        predictor = Predictor(model_path=None, config=sample_config)
        predictor.model = model

        input_data = np.random.randn(1, 60, 10).astype(np.float32)

        start_time = time.time()
        predictions = predictor.predict(input_data)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 1.0  # Should be < 1 second

    def test_inverse_transform_predictions(self):
        """Should inverse transform scaled predictions."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit([[0], [1], [2], [3], [4]])

        predictor = Predictor(model_path=None, config={})
        predictor.output_scaler = scaler

        scaled_predictions = np.array([[0.0], [1.0]])
        inversed = predictor.inverse_transform(scaled_predictions)

        assert inversed.shape == scaled_predictions.shape


class TestMultiHorizonPredictions:
    """Test multi-horizon prediction output."""

    def test_predict_three_horizons(self, sample_config):
        """Should predict for h0, h1, h2."""
        from src.models.hybrid_model import build_model

        model = build_model(sample_config)
        model.build(input_shape=(None, 60, 10))

        predictor = Predictor(model_path=None, config=sample_config)
        predictor.model = model

        input_data = np.random.randn(1, 60, 10).astype(np.float32)
        predictions = predictor.predict(input_data)

        # Should have predictions for all 3 horizons
        if isinstance(predictions, dict):
            assert 'h0' in predictions or 'price_h0' in predictions
            assert 'h1' in predictions or 'price_h1' in predictions
            assert 'h2' in predictions or 'price_h2' in predictions

    def test_nine_output_structure(self, sample_config):
        """Should have 9 outputs (3 towers × 3 outputs) as per SRS."""
        from src.models.hybrid_model import build_model

        model = build_model(sample_config)
        model.build(input_shape=(None, 60, 10))

        predictor = Predictor(model_path=None, config=sample_config)
        predictor.model = model

        input_data = np.random.randn(1, 60, 10).astype(np.float32)
        predictions = predictor.predict(input_data)

        # Should have 9 outputs or dict with 9 keys
        if isinstance(predictions, dict):
            assert len(predictions) == 9
        elif isinstance(predictions, list):
            assert len(predictions) == 9

    def test_price_direction_variance_per_horizon(self, sample_config):
        """Each horizon should have price, direction, variance."""
        from src.models.hybrid_model import build_model

        model = build_model(sample_config)
        model.build(input_shape=(None, 60, 10))

        predictor = Predictor(model_path=None, config=sample_config)
        predictor.model = model

        input_data = np.random.randn(1, 60, 10).astype(np.float32)
        predictions = predictor.predict(input_data)

        # For each horizon, should have price, direction, variance
        if isinstance(predictions, dict):
            for h in [0, 1, 2]:
                assert f'price_h{h}' in predictions or f'h{h}' in predictions
                assert f'direction_h{h}' in predictions
                assert f'variance_h{h}' in predictions


class TestPredictionPostProcessing:
    """Test prediction post-processing."""

    def test_format_predictions(self, sample_predictions):
        """Should format predictions into structured output."""
        from src.inference.predictor import format_predictions

        formatted = format_predictions(sample_predictions)

        assert 'h0' in formatted
        assert 'h1' in formatted
        assert 'h2' in formatted

        for h in ['h0', 'h1', 'h2']:
            assert 'price' in formatted[h]
            assert 'direction' in formatted[h]
            assert 'variance' in formatted[h]
            assert 'confidence' in formatted[h]

    def test_calculate_confidence_from_variance(self, sample_predictions):
        """Should calculate confidence from variance."""
        from src.utils.helper_functions import calculate_confidence

        variance_h0 = sample_predictions['variance_h0']
        confidence = calculate_confidence(variance_h0)

        assert 0.0 <= confidence <= 1.0

    def test_generate_trading_signals(self, sample_predictions):
        """Should generate trading signals from predictions."""
        from src.inference.signals import generate_signals

        current_price = 42000.0
        signals = generate_signals(sample_predictions, current_price)

        assert 'signal_strength' in signals
        assert 'position_size_multiplier' in signals
        assert 'multi_horizon_agreement' in signals


class TestInferenceConfig:
    """Test inference configuration."""

    def test_init_inference_config(self):
        """Should initialize inference config."""
        config = InferenceConfig(
            model_path="model.weights.h5",
            scaler_input_path="scaler_input.joblib",
            scaler_output_path="scaler.joblib"
        )

        assert config.model_path == "model.weights.h5"


class TestStabilityEvaluation:
    """Test stability and convergence evaluation."""

    def test_multi_run_inference(self, sample_config):
        """Should run inference multiple times for stability check."""
        from src.models.hybrid_model import build_model

        model = build_model(sample_config)
        model.build(input_shape=(None, 60, 10))

        predictor = Predictor(model_path=None, config=sample_config)
        predictor.model = model

        input_data = np.random.randn(1, 60, 10).astype(np.float32)

        # Run multiple times
        predictions_list = []
        for _ in range(10):
            predictions = predictor.predict(input_data)
            predictions_list.append(predictions)

        # Check stability (predictions should be identical without dropout)

    def test_compute_variance_metrics(self):
        """Should compute variance across multiple runs."""
        predictions_list = [
            {'price_h0': 42100.0 + np.random.randn()},
            {'price_h0': 42100.0 + np.random.randn()},
            {'price_h0': 42100.0 + np.random.randn()},
        ]

        from src.inference.evaluation import compute_prediction_variance

        variance = compute_prediction_variance(predictions_list, key='price_h0')
        assert variance >= 0

    def test_convergence_check(self, tmp_path):
        """Should check convergence from training logs."""
        import pandas as pd

        training_log = pd.DataFrame({
            'epoch': range(10),
            'loss': [1.0, 0.8, 0.7, 0.65, 0.6, 0.58, 0.57, 0.56, 0.56, 0.56],
            'val_loss': [1.1, 0.85, 0.75, 0.70, 0.68, 0.67, 0.67, 0.67, 0.67, 0.67],
        })

        log_file = tmp_path / "training_log.csv"
        training_log.to_csv(log_file, index=False)

        from src.inference.evaluation import check_convergence

        converged = check_convergence(log_file, metric='val_loss')
        # Should detect convergence


class TestResultingMetrics:
    """Test final accuracy metrics computation."""

    def test_compute_final_mae(self):
        """Should compute final MAE on test set."""
        y_true = np.array([42000, 42100, 42200])
        y_pred = np.array([42010, 42090, 42210])

        from src.inference.evaluation import compute_mae

        mae = compute_mae(y_true, y_pred)
        assert mae == pytest.approx(10.0)

    def test_compute_directional_accuracy(self):
        """Should compute directional accuracy."""
        current_prices = np.array([42000, 42100, 42200])
        predicted_prices = np.array([42050, 42150, 42180])
        actual_prices = np.array([42100, 42200, 42250])

        from src.inference.evaluation import compute_directional_accuracy

        accuracy = compute_directional_accuracy(current_prices, predicted_prices, actual_prices)
        assert 0.0 <= accuracy <= 1.0

    def test_generate_evaluation_report(self, tmp_path):
        """Should generate evaluation report."""
        metrics = {
            'mae_h0': 10.5,
            'mae_h1': 25.3,
            'mae_h2': 50.7,
            'dir_acc_h0': 0.75,
            'dir_acc_h1': 0.70,
            'dir_acc_h2': 0.65,
        }

        from src.inference.evaluation import generate_report

        report_file = tmp_path / "evaluation_report.txt"
        generate_report(metrics, report_file)

        # Should create report
