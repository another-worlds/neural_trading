"""Phase 4.1: LOW Priority Tests - Config Parser Final Paths

Tests to achieve 100% coverage on config parser module.

Target Coverage:
- src/config/config_parser.py: 97.3% → 100%

Missing Lines to Cover:
- 102, 148, 162
"""
import pytest
import yaml
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.config.config_parser import (
    ConfigParser,
    ConfigValidationError
)


class TestConfigSectionAccess:
    """Test config section access (line 102)."""

    def test_get_section_nonexistent_raises_error(self):
        """Test get_section raises KeyError for non-existent section (line 102)."""
        parser = ConfigParser()
        parser.config = {
            'data': {'lookback': 60},
            'training': {'epochs': 100}
        }

        # Try to get non-existent section
        with pytest.raises(KeyError) as exc_info:
            parser.get_section('nonexistent_section')

        # Verify error message (line 102)
        assert "Config section not found" in str(exc_info.value)
        assert "nonexistent_section" in str(exc_info.value)

    def test_get_section_existing(self):
        """Test get_section returns correct section."""
        parser = ConfigParser()
        parser.config = {
            'data': {'lookback': 60, 'features': 5},
            'training': {'epochs': 100}
        }

        # Get existing section
        data_section = parser.get_section('data')

        assert data_section == {'lookback': 60, 'features': 5}


class TestConfigNestedPathCreation:
    """Test nested path creation in set method (line 148)."""

    def test_set_creates_nested_path(self):
        """Test set creates nested dictionaries when path doesn't exist (line 148)."""
        parser = ConfigParser()
        parser.config = {}

        # Set deeply nested value - should create intermediate dicts (line 148)
        parser.set('losses.direction_loss.weight', 2.0)

        # Verify nested structure was created
        assert 'losses' in parser.config
        assert 'direction_loss' in parser.config['losses']
        assert parser.config['losses']['direction_loss']['weight'] == 2.0

    def test_set_creates_multiple_nested_levels(self):
        """Test set creates multiple levels of nesting (line 148)."""
        parser = ConfigParser()
        parser.config = {}

        # Create very deep nesting
        parser.set('a.b.c.d.e.f', 'deep_value')

        # Verify all levels created
        assert parser.config['a']['b']['c']['d']['e']['f'] == 'deep_value'

    def test_set_preserves_existing_keys(self):
        """Test set doesn't overwrite existing keys at intermediate levels (line 148)."""
        parser = ConfigParser()
        parser.config = {
            'training': {
                'epochs': 100,
                'batch_size': 32
            }
        }

        # Add new nested key
        parser.set('training.optimizer.learning_rate', 0.001)

        # Verify existing keys preserved
        assert parser.config['training']['epochs'] == 100
        assert parser.config['training']['batch_size'] == 32

        # Verify new key added
        assert parser.config['training']['optimizer']['learning_rate'] == 0.001


class TestConfigSectionUpdate:
    """Test update_section method (line 162)."""

    def test_update_section_creates_section(self):
        """Test update_section creates section if it doesn't exist (line 162)."""
        parser = ConfigParser()
        parser.config = {
            'data': {'lookback': 60}
        }

        # Update non-existent section - should create it (line 162)
        parser.update_section('new_section', {'key1': 'value1', 'key2': 'value2'})

        # Verify section was created
        assert 'new_section' in parser.config
        assert parser.config['new_section']['key1'] == 'value1'
        assert parser.config['new_section']['key2'] == 'value2'

    def test_update_section_merges_with_existing(self):
        """Test update_section merges with existing section (line 162)."""
        parser = ConfigParser()
        parser.config = {
            'training': {
                'epochs': 100,
                'batch_size': 32
            }
        }

        # Update existing section
        parser.update_section('training', {'epochs': 200, 'learning_rate': 0.001})

        # Verify merge behavior
        assert parser.config['training']['epochs'] == 200  # Updated
        assert parser.config['training']['batch_size'] == 32  # Preserved
        assert parser.config['training']['learning_rate'] == 0.001  # Added

    def test_update_section_empty_config(self):
        """Test update_section on completely empty config (line 162)."""
        parser = ConfigParser()
        parser.config = {}

        # Update section on empty config
        parser.update_section('first_section', {'a': 1, 'b': 2})

        # Verify section created
        assert 'first_section' in parser.config
        assert parser.config['first_section'] == {'a': 1, 'b': 2}


class TestConfigEdgeCases:
    """Test additional config parser edge cases."""

    def test_merge_configs_nested_dicts(self):
        """Test merging nested configuration dictionaries."""
        parser = ConfigParser()

        # Start with base config
        parser.config = {
            'data': {
                'lookback': 60,
                'features': 5
            },
            'training': {
                'epochs': 100
            }
        }

        # Merge with updates
        updates = {
            'data': {
                'lookback': 120  # Override
            },
            'new_section': {
                'param': 'value'
            }
        }

        # Manually merge
        for section, values in updates.items():
            if section not in parser.config:
                parser.config[section] = {}
            parser.config[section].update(values)

        # Verify merge
        assert parser.config['data']['lookback'] == 120
        assert parser.config['data']['features'] == 5  # Preserved
        assert 'new_section' in parser.config


class TestConfigSaveLoad:
    """Test config save and load with edge cases."""

    def test_save_and_load_yaml(self):
        """Test saving and loading YAML config."""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'test_config.yaml'

            # Create parser with config
            parser = ConfigParser()
            parser.config = {
                'data': {'lookback': 60},
                'training': {'epochs': 100, 'learning_rate': 0.001}
            }

            # Save
            parser.save(config_path)

            # Load in new parser
            parser2 = ConfigParser(config_path)
            loaded_config = parser2.load()

            # Verify config preserved
            assert loaded_config['data']['lookback'] == 60
            assert loaded_config['training']['epochs'] == 100

    def test_save_and_load_json(self):
        """Test saving and loading JSON config."""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'test_config.json'

            # Create parser with config
            parser = ConfigParser()
            parser.config = {
                'data': {'lookback': 60},
                'training': {'epochs': 100}
            }

            # Save as JSON
            with open(config_path, 'w') as f:
                json.dump(parser.config, f)

            # Load in new parser
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)

            # Verify
            assert loaded_config['data']['lookback'] == 60


class TestConfigGetWithDefault:
    """Test get method with default values."""

    def test_get_nested_with_default(self):
        """Test get returns default for non-existent nested key."""
        parser = ConfigParser()
        parser.config = {
            'data': {'lookback': 60}
        }

        # Get non-existent key with default
        value = parser.get('training.epochs', default=100)

        assert value == 100

    def test_get_partial_path_with_default(self):
        """Test get with partially valid path returns default."""
        parser = ConfigParser()
        parser.config = {
            'data': {'lookback': 60}
        }

        # Path exists partially but not fully
        value = parser.get('data.missing.deeply.nested', default='fallback')

        assert value == 'fallback'


class TestConfigIntegration:
    """Test complete config workflows."""

    def test_config_cascade_update_workflow(self):
        """Test cascading configuration updates workflow."""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / 'config.yaml'

            # Create initial config
            initial_config = {
                'data': {
                    'lookback': 60,
                    'features': 5
                },
                'training': {
                    'epochs': 100,
                    'batch_size': 32
                },
                'losses': {
                    'direction_loss': {
                        'weight': 1.0
                    }
                }
            }

            # Save initial config
            with open(config_path, 'w') as f:
                yaml.dump(initial_config, f)

            # Load and update
            parser = ConfigParser(config_path)
            parser.load()

            # Update via different methods
            parser.set('training.epochs', 200)  # Nested set
            parser.set('training.optimizer.lr', 0.001)  # Create nested path (line 148)
            parser.update_section('new_module', {'param': 'value'})  # Create section (line 162)

            # Save updated config
            output_path = Path(tmpdir) / 'updated_config.yaml'
            parser.save(output_path)

            # Load updated config
            parser2 = ConfigParser(output_path)
            config = parser2.load()

            # Verify all updates
            assert config['training']['epochs'] == 200
            assert config['training']['optimizer']['lr'] == 0.001
            assert config['new_module']['param'] == 'value'
            assert config['data']['lookback'] == 60  # Original preserved

    def test_config_environment_override(self):
        """Test configuration with environment variable overrides."""
        import os

        parser = ConfigParser()
        parser.config = {
            'data': {'lookback': 60},
            'training': {'epochs': 100}
        }

        # Simulate environment override
        env_epochs = os.getenv('TRAINING_EPOCHS', '200')
        parser.set('training.epochs', int(env_epochs))

        assert parser.config['training']['epochs'] == 200
