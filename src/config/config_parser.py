"""Configuration parser for neural trading pipeline.

This module provides configuration management with:
- YAML/JSON file loading
- Schema validation
- Nested value access with dot notation
- Environment variable overrides
- Configuration merging with defaults
- Cascading configuration updates

Supports the pipeline doctrine of centralized configuration management.
"""
import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union
from copy import deepcopy


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass


class ConfigParser:
    """Configuration parser with validation and nested access support.

    Provides centralized configuration management for the trading pipeline.
    Supports loading from YAML/JSON, validation, nested access, and
    environment variable overrides.

    Examples:
        >>> parser = ConfigParser('config.yaml')
        >>> config = parser.load()
        >>> lookback = parser.get('data.lookback')
        >>> parser.set('training.epochs', 100)
        >>> parser.save('updated_config.yaml')
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize ConfigParser.

        Args:
            config_path: Path to configuration file (YAML or JSON).
                        If None, starts with empty config.
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        """Load configuration from file.

        Returns:
            Loaded configuration dictionary.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If file format is unsupported.
        """
        if self.config_path is None:
            raise ValueError("Config path not specified")

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        # Load based on file extension
        if self.config_path.suffix in ['.yaml', '.yml']:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif self.config_path.suffix == '.json':
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {self.config_path.suffix}")

        return self.config

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate current configuration.

        Returns:
            Tuple of (is_valid, error_messages).
            is_valid is True if config passes all validation checks.
            error_messages is a list of validation errors.
        """
        return validate_config_schema(self.config)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Retrieve entire configuration section.

        Args:
            section: Top-level section name (e.g., 'data', 'training').

        Returns:
            Configuration section dictionary.

        Raises:
            KeyError: If section doesn't exist.
        """
        if section not in self.config:
            raise KeyError(f"Config section not found: {section}")
        return self.config[section]

    def get(self, key: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation.

        Args:
            key: Dot-separated key path (e.g., 'data.lookback').
            default: Default value if key not found.

        Returns:
            Configuration value or default.

        Examples:
            >>> parser.get('data.lookback')
            60
            >>> parser.get('nonexistent.key', default=42)
            42
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        """Set nested configuration value using dot notation.

        Args:
            key: Dot-separated key path (e.g., 'training.epochs').
            value: Value to set.

        Examples:
            >>> parser.set('training.epochs', 100)
            >>> parser.set('losses.direction_loss.weight', 2.0)
        """
        keys = key.split('.')
        config = self.config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the final key
        config[keys[-1]] = value

    def update_section(self, section: str, updates: Dict[str, Any]) -> None:
        """Update entire configuration section.

        Args:
            section: Section name to update.
            updates: Dictionary with new values for the section.
        """
        if section not in self.config:
            self.config[section] = {}

        self.config[section].update(updates)

    def save(self, output_path: Union[str, Path]) -> None:
        """Save configuration to file.

        Args:
            output_path: Path where config should be saved.
                        Format determined by file extension (.yaml or .json).
        """
        output_path = Path(output_path)

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save based on extension
        if output_path.suffix in ['.yaml', '.yml']:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        elif output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")

    def apply_env_overrides(self, prefix: str = 'NEURAL_TRADE') -> None:
        """Apply environment variable overrides to configuration.

        Environment variables should be named as: {PREFIX}_{SECTION}_{KEY}
        For example: NEURAL_TRADE_EPOCHS=100

        Args:
            prefix: Prefix for environment variables (default: 'NEURAL_TRADE').
        """
        env_mappings = {
            f'{prefix}_EPOCHS': 'training.epochs',
            f'{prefix}_BATCH_SIZE': 'training.batch_size',
            f'{prefix}_LEARNING_RATE': 'training.learning_rate',
            f'{prefix}_LOOKBACK': 'data.lookback',
            f'{prefix}_LSTM_UNITS': 'model.lstm_units',
        }

        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Try to convert to appropriate type
                try:
                    # Try int first
                    value = int(value)
                except ValueError:
                    try:
                        # Try float
                        value = float(value)
                    except ValueError:
                        # Keep as string
                        pass

                self.set(config_key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary.

        Returns:
            Deep copy of configuration dictionary.
        """
        return deepcopy(self.config)


def validate_config_schema(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate configuration against schema requirements.

    Checks for:
    - Required sections present
    - Required fields within sections
    - Value ranges (positive values, probabilities, etc.)
    - Data split ratios sum to 1.0

    Args:
        config: Configuration dictionary to validate.

    Returns:
        Tuple of (is_valid, error_messages).
    """
    errors = []

    # Define required sections
    required_sections = ['data', 'training']

    # Check required sections
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")

    # Validate training section
    if 'training' in config:
        training = config['training']

        # Check for negative or zero values
        if 'batch_size' in training:
            if training['batch_size'] <= 0:
                errors.append("training.batch_size must be positive")

        if 'epochs' in training:
            if training['epochs'] < 0:
                errors.append("training.epochs must be non-negative")

        if 'learning_rate' in training:
            if training['learning_rate'] <= 0:
                errors.append("training.learning_rate must be positive")

    # Validate data splits sum to 1.0
    if 'data' in config:
        data = config['data']

        if all(k in data for k in ['train_split', 'val_split', 'test_split']):
            split_sum = data['train_split'] + data['val_split'] + data['test_split']
            # Allow small floating point tolerance
            if abs(split_sum - 1.0) > 1e-6:
                errors.append(
                    f"Data splits must sum to 1.0 (got {split_sum:.4f})"
                )

    # Validate loss configuration
    if 'losses' in config:
        for loss_name, loss_config in config['losses'].items():
            if isinstance(loss_config, dict):
                # Check alpha parameter is non-negative
                if 'alpha' in loss_config:
                    if loss_config['alpha'] < 0:
                        errors.append(
                            f"losses.{loss_name}.alpha must be non-negative"
                        )

                # Check gamma parameter is non-negative
                if 'gamma' in loss_config:
                    if loss_config['gamma'] < 0:
                        errors.append(
                            f"losses.{loss_name}.gamma must be non-negative"
                        )

    is_valid = len(errors) == 0
    return is_valid, errors


def merge_with_defaults(
    user_config: Dict[str, Any],
    defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge user configuration with defaults.

    User values override defaults. Performs deep merge for nested dictionaries.

    Args:
        user_config: User-provided configuration.
        defaults: Default configuration values.

    Returns:
        Merged configuration dictionary.

    Examples:
        >>> defaults = {'training': {'batch_size': 32, 'epochs': 10}}
        >>> user = {'training': {'epochs': 50}}
        >>> merged = merge_with_defaults(user, defaults)
        >>> merged['training']
        {'batch_size': 32, 'epochs': 50}
    """
    # Start with a deep copy of defaults
    merged = deepcopy(defaults)

    # Recursively merge user config
    def deep_merge(target: Dict, source: Dict) -> Dict:
        """Recursively merge source into target."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge nested dicts
                deep_merge(target[key], value)
            else:
                # Override with user value
                target[key] = deepcopy(value)
        return target

    return deep_merge(merged, user_config)
