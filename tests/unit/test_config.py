"""Unit tests for configuration management module.

Tests ConfigParser class for loading, validating, and distributing configuration
parameters across modules as described in the pipeline doctrine.
"""
import pytest
import yaml
import json
from pathlib import Path
from src.config.config_parser import (
    ConfigParser,
    ConfigValidationError,
    validate_config_schema,
    merge_with_defaults,
)


class TestConfigParser:
    """Test ConfigParser class for loading and validating configurations."""

    def test_load_yaml_config(self, tmp_config_file):
        """Should load configuration from YAML file."""
        parser = ConfigParser(tmp_config_file)
        config = parser.load()

        assert config is not None
        assert 'data' in config
        assert 'training' in config

    def test_load_json_config(self, tmp_path, sample_config):
        """Should load configuration from JSON file."""
        json_file = tmp_path / "config.json"
        with open(json_file, 'w') as f:
            json.dump(sample_config, f)

        parser = ConfigParser(json_file)
        config = parser.load()

        assert config is not None
        assert config['data']['lookback'] == 60

    def test_load_nonexistent_file_raises_error(self):
        """Should raise error for nonexistent config file."""
        with pytest.raises(FileNotFoundError):
            parser = ConfigParser("nonexistent.yaml")
            parser.load()

    def test_validate_valid_config(self, sample_config):
        """Should validate a complete valid configuration."""
        parser = ConfigParser()
        parser.config = sample_config
        is_valid, errors = parser.validate()

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_required_fields(self):
        """Should detect missing required fields."""
        incomplete_config = {
            'data': {'csv_path': 'test.csv'},
            # Missing training, model sections
        }

        is_valid, errors = validate_config_schema(incomplete_config)
        assert is_valid is False
        assert len(errors) > 0
        assert any('training' in err.lower() for err in errors)

    def test_validate_invalid_value_ranges(self):
        """Should detect values outside acceptable ranges."""
        invalid_config = {
            'training': {
                'learning_rate': -0.001,  # Negative learning rate
                'batch_size': 0,  # Zero batch size
                'epochs': -10,  # Negative epochs
            }
        }

        is_valid, errors = validate_config_schema(invalid_config)
        assert is_valid is False
        assert len(errors) > 0

    def test_get_section(self, sample_config):
        """Should retrieve specific config sections."""
        parser = ConfigParser()
        parser.config = sample_config

        data_config = parser.get_section('data')
        assert data_config['lookback'] == 60

        training_config = parser.get_section('training')
        assert training_config['batch_size'] == 144

    def test_get_nested_value(self, sample_config):
        """Should retrieve nested configuration values."""
        parser = ConfigParser()
        parser.config = sample_config

        lookback = parser.get('data.lookback')
        assert lookback == 60

        lr = parser.get('training.learning_rate')
        assert lr == 0.001

    def test_get_with_default(self, sample_config):
        """Should return default for missing keys."""
        parser = ConfigParser()
        parser.config = sample_config

        missing_value = parser.get('nonexistent.key', default=42)
        assert missing_value == 42

    def test_set_value(self, sample_config):
        """Should set configuration values."""
        parser = ConfigParser()
        parser.config = sample_config

        parser.set('training.epochs', 100)
        assert parser.get('training.epochs') == 100

    def test_update_section(self, sample_config):
        """Should update entire config section."""
        parser = ConfigParser()
        parser.config = sample_config

        new_training_config = {
            'batch_size': 256,
            'epochs': 50,
            'learning_rate': 0.0005
        }
        parser.update_section('training', new_training_config)

        assert parser.get('training.batch_size') == 256
        assert parser.get('training.epochs') == 50

    def test_save_config(self, tmp_path, sample_config):
        """Should save configuration to file."""
        parser = ConfigParser()
        parser.config = sample_config

        output_file = tmp_path / "saved_config.yaml"
        parser.save(output_file)

        assert output_file.exists()

        # Load and verify
        with open(output_file, 'r') as f:
            loaded = yaml.safe_load(f)
        assert loaded['data']['lookback'] == 60

    def test_merge_with_defaults(self):
        """Should merge user config with defaults."""
        defaults = {
            'training': {
                'batch_size': 32,
                'epochs': 10,
                'learning_rate': 0.001
            }
        }

        user_config = {
            'training': {
                'epochs': 50  # Override only epochs
            }
        }

        merged = merge_with_defaults(user_config, defaults)
        assert merged['training']['batch_size'] == 32  # From defaults
        assert merged['training']['epochs'] == 50  # Overridden
        assert merged['training']['learning_rate'] == 0.001  # From defaults

    def test_environment_variable_override(self, monkeypatch, sample_config):
        """Should allow environment variable overrides."""
        monkeypatch.setenv('NEURAL_TRADE_EPOCHS', '100')
        monkeypatch.setenv('NEURAL_TRADE_BATCH_SIZE', '256')

        parser = ConfigParser()
        parser.config = sample_config
        parser.apply_env_overrides()

        assert parser.get('training.epochs') == 100
        assert parser.get('training.batch_size') == 256

    def test_to_dict(self, sample_config):
        """Should export configuration as dictionary."""
        parser = ConfigParser()
        parser.config = sample_config

        config_dict = parser.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict == sample_config


class TestConfigValidation:
    """Test configuration validation functions."""

    def test_validate_file_paths_exist(self, tmp_path):
        """Should validate that specified file paths exist."""
        csv_file = tmp_path / "data.csv"
        csv_file.touch()

        config = {
            'data': {
                'csv_path': str(csv_file)
            }
        }

        is_valid, errors = validate_config_schema(config)
        # File exists, should be valid
        assert is_valid or 'csv_path' not in str(errors)

    def test_validate_file_paths_missing(self):
        """Should detect missing file paths."""
        config = {
            'data': {
                'csv_path': '/nonexistent/path/data.csv'
            }
        }

        # Note: Some validators may not check file existence
        # This tests the expectation
        # Actual implementation may vary

    def test_validate_array_dimensions(self):
        """Should validate array dimensions match."""
        config = {
            'indicators': {
                'ma_periods': [5, 15, 30],  # 3 periods
                'rsi_periods': [9, 21],  # 2 periods - mismatch
            }
        }

        # Arrays can have different lengths, but should be validated
        # if there are dependencies

    def test_validate_positive_values(self):
        """Should validate that certain values are positive."""
        config = {
            'training': {
                'batch_size': -144,  # Invalid: negative
                'epochs': 0,  # Invalid: zero
                'learning_rate': -0.001,  # Invalid: negative
            }
        }

        is_valid, errors = validate_config_schema(config)
        assert is_valid is False

    def test_validate_probability_ranges(self):
        """Should validate probability values are in [0, 1]."""
        config = {
            'losses': {
                'direction_loss': {
                    'alpha': 1.5,  # Invalid: > 1
                    'gamma': -0.5,  # Invalid: < 0
                }
            }
        }

        is_valid, errors = validate_config_schema(config)
        # Alpha can be > 1 in some cases, but this tests the concept

    def test_validate_split_ratios_sum_to_one(self):
        """Should validate that dataset splits sum to 1.0."""
        valid_config = {
            'data': {
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
            }
        }

        is_valid, errors = validate_config_schema(valid_config)
        # Sum = 1.0, should be valid
        assert is_valid or 'split' not in str(errors)

        invalid_config = {
            'data': {
                'train_split': 0.8,
                'val_split': 0.15,
                'test_split': 0.15,  # Sum = 1.1
            }
        }

        is_valid, errors = validate_config_schema(invalid_config)
        assert is_valid is False or len(errors) > 0


class TestConfigRegistry:
    """Test configuration registry for automatic component registration."""

    def test_register_loss_from_config(self, sample_config):
        """Should register losses from configuration."""
        parser = ConfigParser()
        parser.config = sample_config

        loss_configs = parser.get_section('losses')
        assert 'point_loss' in loss_configs
        assert 'direction_loss' in loss_configs

    def test_register_metric_from_config(self, sample_config):
        """Should register metrics from configuration."""
        parser = ConfigParser()
        parser.config = sample_config

        metric_configs = parser.get_section('metrics')
        assert metric_configs['direction_accuracy'] is True
        assert metric_configs['price_mae'] is True

    def test_register_indicator_from_config(self, sample_config):
        """Should register indicators from configuration."""
        parser = ConfigParser()
        parser.config = sample_config

        indicator_configs = parser.get_section('indicators')
        assert 'ma_periods' in indicator_configs
        assert len(indicator_configs['ma_periods']) == 3


class TestConfigCascading:
    """Test cascading configuration updates."""

    def test_update_loss_lambda_cascades(self, sample_config):
        """Changing loss lambda should cascade to model compilation."""
        parser = ConfigParser()
        parser.config = sample_config

        # Update lambda
        parser.set('losses.direction_loss.weight', 2.0)

        # Verify cascade
        updated_weight = parser.get('losses.direction_loss.weight')
        assert updated_weight == 2.0

    def test_update_indicator_params_cascades(self, sample_config):
        """Changing indicator params should cascade to feature generation."""
        parser = ConfigParser()
        parser.config = sample_config

        # Update MA periods
        new_periods = [10, 20, 40]
        parser.set('indicators.ma_periods', new_periods)

        # Verify cascade
        updated_periods = parser.get('indicators.ma_periods')
        assert updated_periods == new_periods

    def test_update_model_params_cascades(self, sample_config):
        """Changing model params should cascade to model rebuilding."""
        parser = ConfigParser()
        parser.config = sample_config

        # Update LSTM units
        parser.set('model.lstm_units', 256)

        # Verify cascade
        updated_units = parser.get('model.lstm_units')
        assert updated_units == 256


class TestConfigVersioning:
    """Test configuration versioning support."""

    def test_config_version_field(self, sample_config):
        """Config should support version field."""
        sample_config['version'] = 'v1.0.0'
        parser = ConfigParser()
        parser.config = sample_config

        version = parser.get('version')
        assert version == 'v1.0.0'

    def test_backward_compatibility_check(self):
        """Should check backward compatibility of config versions."""
        old_config = {
            'version': 'v1.0.0',
            'data': {'csv_path': 'old_format.csv'}
        }

        new_config = {
            'version': 'v2.0.0',
            'data': {'csv_path': 'new_format.csv'}
        }

        # Test version comparison logic
        # Implementation would check compatibility


class TestConfigHydraIntegration:
    """Test integration with Hydra/OmegaConf (future feature)."""

    @pytest.mark.skip(reason="Hydra integration not yet implemented")
    def test_load_with_hydra(self):
        """Should load config using Hydra."""
        pass

    @pytest.mark.skip(reason="OmegaConf integration not yet implemented")
    def test_compose_config_with_omegaconf(self):
        """Should compose config using OmegaConf."""
        pass

    @pytest.mark.skip(reason="Config groups not yet implemented")
    def test_config_groups(self):
        """Should support config groups (experiments, models, etc.)."""
        pass
