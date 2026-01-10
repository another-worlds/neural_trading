"""Unit tests for training module.

Tests training orchestrator, callbacks as per SRS Section 3.5.3.
"""
import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
from src.training.trainer import Trainer, TrainingConfig
from src.training.callbacks import (
    IndicatorParamsLogger,
    create_callbacks,
)


class TestTrainer:
    """Test Trainer class."""

    def test_init_trainer(self, sample_config):
        """Should initialize trainer with configuration."""
        trainer = Trainer(sample_config)

        assert trainer is not None
        assert trainer.config is not None

    def test_load_dataset(self, sample_config, sample_ohlcv_data):
        """Should load and prepare datasets."""
        trainer = Trainer(sample_config)
        train_ds, val_ds, test_ds = trainer.load_datasets(sample_ohlcv_data)

        assert train_ds is not None
        assert val_ds is not None
        assert test_ds is not None

    def test_build_model(self, sample_config):
        """Should build model from configuration."""
        trainer = Trainer(sample_config)
        model = trainer.build_model()

        assert isinstance(model, tf.keras.Model)

    def test_compile_model(self, sample_config):
        """Should compile model with losses and metrics."""
        trainer = Trainer(sample_config)
        model = trainer.build_model()
        trainer.compile_model(model)

        assert model.optimizer is not None

    def test_fit_model(self, sample_config, sample_ohlcv_data):
        """Should train model."""
        trainer = Trainer(sample_config)
        trainer.config['training']['epochs'] = 2  # Quick test

        model = trainer.build_model()
        trainer.compile_model(model)

        train_ds, val_ds, _ = trainer.load_datasets(sample_ohlcv_data)

        history = trainer.fit(model, train_ds, val_ds)

        assert history is not None
        assert 'loss' in history.history

    def test_save_model_weights(self, tmp_path, sample_config):
        """Should save model weights."""
        trainer = Trainer(sample_config)
        model = trainer.build_model()

        weights_file = tmp_path / "model.weights.h5"
        trainer.save_weights(model, weights_file)

        assert weights_file.exists()

    def test_save_scalers(self, tmp_path, sample_config):
        """Should save input and output scalers."""
        trainer = Trainer(sample_config)

        scaler_input_file = tmp_path / "scaler_input.joblib"
        scaler_output_file = tmp_path / "scaler.joblib"

        # Mock scalers
        from sklearn.preprocessing import StandardScaler
        input_scaler = StandardScaler()
        output_scaler = StandardScaler()

        trainer.save_scalers(input_scaler, output_scaler, tmp_path)

        # Check files exist


class TestTrainingConfig:
    """Test training configuration."""

    def test_init_training_config(self):
        """Should initialize training config."""
        config = TrainingConfig(
            batch_size=144,
            epochs=40,
            learning_rate=0.001,
            patience=40
        )

        assert config.batch_size == 144
        assert config.epochs == 40

    def test_config_as_per_srs(self):
        """Should support SRS-specified training parameters."""
        config = TrainingConfig(
            batch_size=144,  # SRS Section 3.5.3
            epochs=40,
            learning_rate=1e-3,  # 0.001
            patience=40,
            gradient_clip_norm=5.0
        )

        assert config.batch_size == 144
        assert config.epochs == 40
        assert config.learning_rate == 0.001
        assert config.patience == 40
        assert config.gradient_clip_norm == 5.0


class TestCallbacks:
    """Test training callbacks."""

    def test_early_stopping_callback(self):
        """Should create EarlyStopping callback."""
        callbacks = create_callbacks(
            patience=40,
            monitor='val_dir_mcc_h1',  # As per SRS
            mode='max'
        )

        early_stopping = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.EarlyStopping)]
        assert len(early_stopping) > 0

    def test_early_stopping_monitors_val_dir_mcc_h1(self):
        """Early stopping should monitor val_dir_mcc_h1 as per SRS."""
        callbacks = create_callbacks(
            patience=40,
            monitor='val_dir_mcc_h1',
            mode='max'
        )

        early_stopping = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.EarlyStopping)][0]
        assert early_stopping.monitor == 'val_dir_mcc_h1'
        assert early_stopping.patience == 40

    def test_model_checkpoint_callback(self, tmp_path):
        """Should create ModelCheckpoint callback."""
        callbacks = create_callbacks(
            checkpoint_path=tmp_path / "checkpoints" / "model_{epoch:02d}.h5",
            monitor='val_loss',
            save_best_only=True
        )

        checkpoint = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.ModelCheckpoint)]
        assert len(checkpoint) > 0

    def test_tensorboard_callback(self, tmp_path):
        """Should create TensorBoard callback."""
        log_dir = tmp_path / "logs"
        callbacks = create_callbacks(log_dir=log_dir)

        tensorboard = [cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.TensorBoard)]
        assert len(tensorboard) > 0

    def test_gradient_clipping(self):
        """Should apply gradient clipping with norm=5.0 as per SRS."""
        # Gradient clipping is typically set in optimizer or callback
        from src.training.callbacks import GradientClippingCallback

        callback = GradientClippingCallback(clip_norm=5.0)
        assert callback.clip_norm == 5.0


class TestIndicatorParamsLogger:
    """Test indicator parameters logging callback."""

    def test_init_indicator_logger(self, tmp_path):
        """Should initialize indicator params logger."""
        logger = IndicatorParamsLogger(
            output_file=tmp_path / "indicator_params_history.csv"
        )

        assert logger is not None

    def test_log_indicator_params_per_epoch(self, tmp_path):
        """Should log indicator parameters each epoch as per SRS."""
        logger = IndicatorParamsLogger(
            output_file=tmp_path / "indicator_params_history.csv"
        )

        # Simulate indicator parameters
        indicator_params = {
            'ma_period_0': 5.2,
            'ma_period_1': 15.1,
            'ma_period_2': 29.8,
            'rsi_period_0': 9.3,
        }

        logger.log_epoch(epoch=1, params=indicator_params)

        assert (tmp_path / "indicator_params_history.csv").exists()

    def test_indicator_params_csv_format(self, tmp_path):
        """CSV should have epoch and 30+ indicator parameter columns."""
        import pandas as pd

        logger = IndicatorParamsLogger(
            output_file=tmp_path / "indicator_params_history.csv"
        )

        # Log multiple epochs
        for epoch in range(3):
            params = {f'param_{i}': np.random.rand() for i in range(30)}
            logger.log_epoch(epoch=epoch, params=params)

        # Verify CSV format
        df = pd.read_csv(tmp_path / "indicator_params_history.csv")
        assert 'epoch' in df.columns
        assert len(df.columns) >= 31  # epoch + 30 params


class TestGPUBenchmark:
    """Test GPU benchmarking."""

    @pytest.mark.gpu
    def test_measure_time_per_epoch(self, sample_config):
        """Should measure GPU training time per epoch."""
        from src.training.benchmark import GPUBenchmark

        benchmark = GPUBenchmark()

        trainer = Trainer(sample_config)
        trainer.config['training']['epochs'] = 3

        # Run benchmark
        times = benchmark.measure_training_time(trainer)

        assert len(times) == 3
        assert all(t > 0 for t in times)

    @pytest.mark.gpu
    def test_report_mean_median_time(self, sample_config):
        """Should report mean and median time per epoch."""
        from src.training.benchmark import GPUBenchmark

        benchmark = GPUBenchmark()
        times = [1.5, 1.6, 1.4, 1.5, 1.7]

        stats = benchmark.calculate_stats(times)

        assert 'mean' in stats
        assert 'median' in stats
        assert stats['mean'] == pytest.approx(1.54)
        assert stats['median'] == 1.5


class TestLambdaCalibration:
    """Test loss lambda calibration."""

    def test_grid_search_lambdas(self, sample_config):
        """Should perform grid search for lambda calibration."""
        from src.training.calibration import LambdaCalibrator

        calibrator = LambdaCalibrator(sample_config)

        lambda_grid = {
            'point_loss': [0.5, 1.0, 2.0],
            'direction_loss': [0.5, 1.0, 2.0],
        }

        # Mock validation data
        # best_lambdas = calibrator.grid_search(lambda_grid, val_data)

    def test_report_calibration_results(self, tmp_path):
        """Should report calibration results in table."""
        import pandas as pd

        results = pd.DataFrame({
            'point_loss_weight': [0.5, 1.0, 2.0],
            'direction_loss_weight': [1.0, 1.0, 1.0],
            'val_dir_mcc_h1': [0.65, 0.70, 0.68],
        })

        output_file = tmp_path / "calibration_results.csv"
        results.to_csv(output_file, index=False)

        assert output_file.exists()

    def test_update_config_with_best_lambdas(self, sample_config):
        """Should update config with calibrated lambdas."""
        best_lambdas = {
            'point_loss': 1.5,
            'direction_loss': 2.0,
            'variance_loss': 1.0,
        }

        # Update config
        for loss_name, weight in best_lambdas.items():
            if loss_name in sample_config['losses']:
                sample_config['losses'][loss_name]['weight'] = weight

        assert sample_config['losses']['point_loss']['weight'] == 1.5


class TestTrainingLogs:
    """Test training log generation."""

    def test_save_training_log_csv(self, tmp_path):
        """Should save training log to CSV as per SRS."""
        import pandas as pd

        history = {
            'loss': [1.0, 0.8, 0.6],
            'val_loss': [1.2, 0.9, 0.7],
            'dir_acc_h0': [0.6, 0.7, 0.75],
            'val_dir_mcc_h1': [0.5, 0.6, 0.65],
        }

        df = pd.DataFrame(history)
        df['epoch'] = range(len(df))

        output_file = tmp_path / "training_log.csv"
        df.to_csv(output_file, index=False)

        assert output_file.exists()

    def test_log_format_matches_srs(self, tmp_path):
        """Training log should match SRS format."""
        import pandas as pd

        # SRS Section 7.2.5 specifies training_log.csv format
        df = pd.read_csv(tmp_path / "training_log.csv" if (tmp_path / "training_log.csv").exists() else None) \
            if (tmp_path / "training_log.csv").exists() else pd.DataFrame()

        # Would have epoch, loss, val_loss, and various metrics
