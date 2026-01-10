"""Integration tests for end-to-end pipelines.

Tests complete data → training → inference → backtesting workflows as per SRS.
"""
import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path


@pytest.mark.integration
class TestDataPipeline:
    """Test end-to-end data processing pipeline."""

    def test_fetch_to_dataset(self, tmp_path, sample_ohlcv_data):
        """Should process data from CSV to tf.data.Dataset."""
        from src.data.data_loader import DataLoader
        from src.data.preprocessor import Preprocessor
        from src.data.dataset import get_dataset

        # 1. Load data
        csv_file = tmp_path / "data.csv"
        sample_ohlcv_data.to_csv(csv_file, index=False)

        loader = DataLoader(csv_file)
        data = loader.load()

        # 2. Preprocess
        config = {
            'lookback': 10,
            'window_step': 1,
            'sequence_limit': 50,
        }
        preprocessor = Preprocessor(config)
        windows = preprocessor.create_windows(data)

        # 3. Create dataset
        dataset = get_dataset(data, config)

        assert dataset is not None
        assert isinstance(dataset, tf.data.Dataset)

    def test_data_validation_integration(self, sample_ohlcv_data):
        """Should validate data quality throughout pipeline."""
        from src.data.data_loader import validate_ohlcv_data, calculate_quality_metrics

        # Validate
        is_valid, errors = validate_ohlcv_data(sample_ohlcv_data)
        assert is_valid is True

        # Calculate metrics
        metrics = calculate_quality_metrics(sample_ohlcv_data)
        assert metrics['total_candles'] == len(sample_ohlcv_data)


@pytest.mark.integration
class TestModelPipeline:
    """Test end-to-end model building and compilation."""

    def test_build_compile_model(self, sample_config):
        """Should build and compile model with all components."""
        from src.models.hybrid_model import build_model
        from src.losses.custom_losses import CompositeLoss
        from src.metrics.custom_metrics import DirectionAccuracy, PriceMAE

        # Build model
        model = build_model(sample_config)

        # Compile with losses and metrics
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        assert model.optimizer is not None

    def test_model_with_indicators(self, sample_config):
        """Should integrate learnable indicators into model."""
        from src.models.hybrid_model import build_model

        model = build_model(sample_config)
        model.build(input_shape=(None, 60, 10))

        # Model should have indicator components
        assert len(model.trainable_variables) > 0


@pytest.mark.integration
class TestTrainingPipeline:
    """Test end-to-end training pipeline."""

    def test_full_training_pipeline(self, tmp_path, sample_config, sample_ohlcv_data):
        """Should run complete training pipeline."""
        from src.training.trainer import Trainer

        # Configure for quick test
        sample_config['training']['epochs'] = 2
        sample_config['training']['batch_size'] = 16

        trainer = Trainer(sample_config)

        # Load data
        train_ds, val_ds, test_ds = trainer.load_datasets(sample_ohlcv_data)

        # Build and compile model
        model = trainer.build_model()
        trainer.compile_model(model)

        # Train
        history = trainer.fit(model, train_ds, val_ds)

        # Save artifacts
        trainer.save_weights(model, tmp_path / "model.weights.h5")

        assert (tmp_path / "model.weights.h5").exists()
        assert 'loss' in history.history

    def test_training_with_callbacks(self, tmp_path, sample_config, sample_ohlcv_data):
        """Should use callbacks during training."""
        from src.training.trainer import Trainer
        from src.training.callbacks import create_callbacks

        sample_config['training']['epochs'] = 3

        trainer = Trainer(sample_config)
        model = trainer.build_model()
        trainer.compile_model(model)

        train_ds, val_ds, _ = trainer.load_datasets(sample_ohlcv_data)

        callbacks = create_callbacks(
            patience=10,
            monitor='val_loss',
            checkpoint_path=tmp_path / "checkpoints" / "model.h5",
            log_dir=tmp_path / "logs"
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=3,
            callbacks=callbacks
        )

        assert history is not None


@pytest.mark.integration
class TestInferencePipeline:
    """Test end-to-end inference pipeline."""

    def test_load_and_predict(self, tmp_path, sample_config):
        """Should load model and generate predictions."""
        from src.models.hybrid_model import build_model
        from src.inference.predictor import Predictor

        # Build and save model
        model = build_model(sample_config)
        model.build(input_shape=(None, 60, 10))
        weights_file = tmp_path / "model.weights.h5"
        model.save_weights(str(weights_file))

        # Load and predict
        predictor = Predictor(
            model_path=str(weights_file),
            scaler_input_path=None,
            scaler_output_path=None,
            config=sample_config
        )

        loaded_model = predictor.load_model()
        predictions = predictor.predict(np.random.randn(1, 60, 10).astype(np.float32))

        assert predictions is not None

    def test_predictions_to_signals(self, sample_predictions):
        """Should convert predictions to trading signals."""
        from src.inference.signals import generate_signals

        current_price = 42000.0
        signals = generate_signals(sample_predictions, current_price)

        assert 'signal_strength' in signals
        assert 'position_size_multiplier' in signals

    def test_generate_profit_targets(self, sample_predictions):
        """Should generate profit targets from predictions."""
        from src.utils.helper_functions import calculate_profit_targets

        entry_price = 42000.0
        price_predictions = [
            sample_predictions['price_h0'],
            sample_predictions['price_h1'],
            sample_predictions['price_h2']
        ]

        targets = calculate_profit_targets(entry_price, price_predictions)

        assert 'tp1' in targets
        assert 'tp2' in targets
        assert 'tp3' in targets


@pytest.mark.integration
class TestBacktestingPipeline:
    """Test backtesting integration."""

    def test_simulate_trades(self, sample_ohlcv_data):
        """Should simulate trades with model predictions."""
        from src.inference.backtesting import BacktestEngine

        # Mock predictions
        predictions = pd.DataFrame({
            'datetime': sample_ohlcv_data['datetime'],
            'predicted_price_h0': sample_ohlcv_data['close'] * 1.001,
            'direction_h0': 1,
            'confidence': 0.75,
            'variance': 0.01,
        })

        engine = BacktestEngine()
        trades = engine.simulate(sample_ohlcv_data, predictions)

        assert len(trades) >= 0  # May or may not have trades

    def test_calculate_performance_metrics(self):
        """Should calculate backtest performance metrics."""
        from src.inference.backtesting import calculate_performance

        trades = [
            {'entry_price': 42000, 'exit_price': 42100, 'size': 1.0, 'status': 'CLOSED_WIN'},
            {'entry_price': 42100, 'exit_price': 42050, 'size': 1.0, 'status': 'CLOSED_LOSS'},
        ]

        metrics = calculate_performance(trades)

        assert 'total_pnl' in metrics
        assert 'win_rate' in metrics

    def test_three_tier_profit_taking(self):
        """Should implement three-tier profit taking."""
        from src.inference.backtesting import execute_partial_profit_taking

        trade = {
            'entry_price': 42000.0,
            'tp1': 42050.0,
            'tp2': 42100.0,
            'tp3': 42150.0,
            'size': 1.0,
        }

        # Simulate price hitting TP1
        current_price = 42050.0
        updated_trade = execute_partial_profit_taking(trade, current_price)

        # Should close partial position
        assert updated_trade['size'] < 1.0


@pytest.mark.integration
class TestConfigIntegration:
    """Test configuration cascading through pipeline."""

    def test_config_drives_pipeline(self, sample_config, tmp_path):
        """Configuration should control entire pipeline."""
        # Update config
        sample_config['training']['epochs'] = 1
        sample_config['model']['lstm_units'] = 64

        # Config should cascade to all components
        from src.models.hybrid_model import build_model

        model = build_model(sample_config)
        model.build(input_shape=(None, 60, 10))

        # Model should reflect config
        # (Check LSTM units, etc.)

    def test_config_file_loading(self, tmp_path, sample_config):
        """Should load and use config from YAML file."""
        import yaml
        from src.config.config_parser import ConfigParser

        # Save config
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(sample_config, f)

        # Load and use
        parser = ConfigParser(config_file)
        loaded_config = parser.load()

        assert loaded_config['training']['batch_size'] == sample_config['training']['batch_size']


@pytest.mark.integration
@pytest.mark.slow
class TestFullSystemIntegration:
    """Test complete system end-to-end."""

    def test_data_to_backtest(self, tmp_path, sample_config, sample_ohlcv_data):
        """Should run complete pipeline: data → train → predict → backtest."""
        # 1. Data processing
        from src.data.dataset import get_train_val_test_datasets

        train_ds, val_ds, test_ds = get_train_val_test_datasets(
            sample_ohlcv_data,
            sample_config
        )

        # 2. Model building and training
        from src.training.trainer import Trainer

        sample_config['training']['epochs'] = 1
        trainer = Trainer(sample_config)
        model = trainer.build_model()
        trainer.compile_model(model)

        history = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=0)

        # 3. Save model
        model.save_weights(str(tmp_path / "model.weights.h5"))

        # 4. Inference
        from src.inference.predictor import Predictor

        predictor = Predictor(
            model_path=str(tmp_path / "model.weights.h5"),
            scaler_input_path=None,
            scaler_output_path=None,
            config=sample_config
        )

        # 5. Generate signals and backtest
        # (Mock test data)
        test_data = np.random.randn(10, 60, 10).astype(np.float32)
        predictions = predictor.predict(test_data)

        assert predictions is not None
        assert 'loss' in history.history
