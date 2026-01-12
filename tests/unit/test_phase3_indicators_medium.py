"""Phase 3.1: MEDIUM Priority Tests - Learnable Indicators

Tests to achieve 95%+ coverage on learnable indicator layers.

Target Coverage:
- src/data/indicators.py: 80.5% → 95%

Missing Lines to Cover:
- 121, 167, 210, 254, 321-328, 332-360, 371, 389, 435, 475
"""
import pytest
import tensorflow as tf
import numpy as np

from src.data.indicators import (
    LearnableMA,
    LearnableRSI,
    LearnableBollingerBands,
    LearnableMacd,
    LearnableCustomMacd,
    LearnableMomentum,
    IndicatorRegistry,
    INDICATOR_REGISTRY,
    register_indicator
)


class TestLearnableMA:
    """Test LearnableMA layer (line 121)."""

    def test_learnable_ma_build_and_call(self):
        """Test MA layer initialization and execution (line 121)."""
        # Create layer with custom periods
        layer = LearnableMA(periods=[5, 15, 30])

        # Build the layer
        inputs = tf.random.normal((2, 60, 5))
        output = layer(inputs)

        # Verify output shape matches input
        assert output.shape == inputs.shape

        # Verify trainable parameters were created (line 121 - period_params initialization)
        assert len(layer.trainable_variables) > 0
        assert layer.period_params is not None

        # Verify period parameters have correct shape
        assert layer.period_params.shape == (3,)

        # Verify periods are initialized correctly
        period_values = layer.period_params.numpy()
        np.testing.assert_array_almost_equal(period_values, [5, 15, 30], decimal=1)

    def test_learnable_ma_with_default_periods(self):
        """Test MA layer with default periods."""
        # Create layer without specifying periods
        layer = LearnableMA()

        # Build and call
        inputs = tf.random.normal((2, 60, 5))
        output = layer(inputs)

        # Should use default periods [5, 15, 30]
        assert layer.num_periods == 3
        assert output.shape == inputs.shape

    def test_learnable_ma_custom_periods(self):
        """Test MA layer with custom period count."""
        # Create layer with 4 periods
        layer = LearnableMA(periods=[5, 10, 20, 50])

        inputs = tf.random.normal((2, 60, 5))
        output = layer(inputs)

        # Should have 4 trainable parameters
        assert layer.num_periods == 4
        assert layer.period_params.shape == (4,)


class TestLearnableRSI:
    """Test LearnableRSI layer (line 167)."""

    def test_learnable_rsi_build_and_call(self):
        """Test RSI layer initialization and execution (line 167)."""
        # Create RSI layer
        layer = LearnableRSI(periods=[9, 21, 30])

        # Build and call
        inputs = tf.random.normal((2, 60, 5))
        output = layer(inputs)

        # Verify output
        assert output.shape == inputs.shape

        # Verify trainable parameters (line 167 - default periods path)
        assert len(layer.trainable_variables) > 0
        assert layer.period_params is not None
        assert layer.period_params.shape == (3,)

    def test_learnable_rsi_default_periods(self):
        """Test RSI with default periods (line 167)."""
        # Create without specifying periods
        layer = LearnableRSI()

        inputs = tf.random.normal((2, 60, 5))
        output = layer(inputs)

        # Should use default [9, 21, 30]
        assert layer.num_periods == 3
        period_values = layer.period_params.numpy()
        np.testing.assert_array_almost_equal(period_values, [9, 21, 30], decimal=1)


class TestLearnableBollingerBands:
    """Test LearnableBollingerBands layer (line 210)."""

    def test_learnable_bollinger_build_and_call(self):
        """Test Bollinger Bands layer (line 210)."""
        # Create BB layer
        layer = LearnableBollingerBands(periods=[10, 20, 30])

        # Build and call
        inputs = tf.random.normal((2, 60, 5))
        output = layer(inputs)

        # Verify output
        assert output.shape == inputs.shape
        assert len(layer.trainable_variables) > 0
        assert layer.period_params is not None

    def test_learnable_bollinger_default_periods(self):
        """Test BB with default periods (line 210)."""
        # Create without specifying periods
        layer = LearnableBollingerBands()

        inputs = tf.random.normal((2, 60, 5))
        output = layer(inputs)

        # Should use default [10, 20, 30]
        assert layer.num_periods == 3
        period_values = layer.period_params.numpy()
        np.testing.assert_array_almost_equal(period_values, [10, 20, 30], decimal=1)


class TestLearnableMacd:
    """Test LearnableMacd layer (line 254)."""

    def test_learnable_macd_build_and_call(self):
        """Test MACD layer (line 254)."""
        # Create MACD layer with custom settings
        layer = LearnableMacd(settings=[[12, 26, 9], [5, 35, 5], [19, 39, 9]])

        # Build and call
        inputs = tf.random.normal((2, 60, 5))
        output = layer(inputs)

        # Verify output
        assert output.shape == inputs.shape
        assert len(layer.trainable_variables) > 0

        # MACD has separate fast, slow, signal params (not period_params)
        assert layer.fast_params is not None
        assert layer.slow_params is not None
        assert layer.signal_params is not None

        # Each has 3 settings
        assert layer.fast_params.shape == (3,)
        assert layer.slow_params.shape == (3,)
        assert layer.signal_params.shape == (3,)

    def test_learnable_macd_default_settings(self):
        """Test MACD with default settings (line 254)."""
        # Create without specifying settings
        layer = LearnableMacd()

        inputs = tf.random.normal((2, 60, 5))
        output = layer(inputs)

        # Should use default settings
        assert layer.num_settings == 3
        assert layer.fast_params.shape == (3,)
        assert layer.slow_params.shape == (3,)
        assert layer.signal_params.shape == (3,)


class TestLearnableCustomMacd:
    """Test LearnableCustomMacd layer (lines 321-360)."""

    def test_learnable_custom_macd_full_cycle(self):
        """Test custom MACD full initialization and execution (lines 321-360)."""
        # Create custom MACD layer
        layer = LearnableCustomMacd(settings=[[12, 26, 9], [5, 35, 5], [19, 39, 9]])

        # Build and call
        inputs = tf.random.normal((2, 60, 5))
        output = layer(inputs)

        # Verify output
        assert output.shape == inputs.shape

        # Verify trainable parameters created (lines 321-328)
        assert len(layer.trainable_variables) > 0

        # Custom MACD also has separate fast, slow, signal params
        assert layer.fast_params is not None
        assert layer.slow_params is not None
        assert layer.signal_params is not None

        # Each has 3 settings
        assert layer.fast_params.shape == (3,)
        assert layer.slow_params.shape == (3,)
        assert layer.signal_params.shape == (3,)

    def test_learnable_custom_macd_default_settings(self):
        """Test custom MACD with default settings (lines 332-360)."""
        # Create without settings
        layer = LearnableCustomMacd()

        inputs = tf.random.normal((2, 60, 5))
        output = layer(inputs)

        # Should use defaults
        assert layer.num_settings == 3
        assert output.shape == inputs.shape
        assert layer.fast_params is not None

    def test_learnable_custom_macd_trainable_constraint(self):
        """Test that custom MACD parameters have non-negative constraint."""
        layer = LearnableCustomMacd()

        inputs = tf.random.normal((2, 60, 5))
        layer(inputs)  # Build layer

        # All period parameters should be non-negative
        fast_params = layer.fast_params.numpy()
        slow_params = layer.slow_params.numpy()
        signal_params = layer.signal_params.numpy()

        assert np.all(fast_params >= 0)
        assert np.all(slow_params >= 0)
        assert np.all(signal_params >= 0)


class TestLearnableMomentum:
    """Test LearnableMomentum layer (line 371)."""

    def test_learnable_momentum_build_and_call(self):
        """Test Momentum layer (line 371)."""
        # Create momentum layer
        layer = LearnableMomentum(periods=[10, 20, 30])

        # Build and call
        inputs = tf.random.normal((2, 60, 5))
        output = layer(inputs)

        # Verify output
        assert output.shape == inputs.shape
        assert len(layer.trainable_variables) > 0
        assert layer.period_params is not None

    def test_learnable_momentum_default_periods(self):
        """Test Momentum with default periods (line 371)."""
        # Create without periods
        layer = LearnableMomentum()

        inputs = tf.random.normal((2, 60, 5))
        output = layer(inputs)

        # Should use default periods
        assert layer.num_periods == 3
        assert layer.period_params.shape == (3,)


class TestIndicatorRegistry:
    """Test indicator registry functions (lines 389, 435, 475)."""

    def test_indicator_registry_register_and_get(self):
        """Test registering and retrieving indicators (line 389)."""
        # Create a new registry for testing
        test_registry = IndicatorRegistry()

        # Register a test indicator
        @test_registry.register('test_indicator')
        class TestIndicator:
            pass

        # Verify it was registered
        assert 'test_indicator' in test_registry.indicators

        # Retrieve it
        retrieved = test_registry.get('test_indicator')
        assert retrieved is TestIndicator

    def test_indicator_registry_get_nonexistent(self):
        """Test retrieving non-existent indicator raises KeyError (line 435)."""
        test_registry = IndicatorRegistry()

        # Try to get indicator that doesn't exist
        with pytest.raises(KeyError) as exc_info:
            test_registry.get('nonexistent_indicator')

        assert "not found in registry" in str(exc_info.value)

    def test_indicator_registry_list_indicators(self):
        """Test listing all registered indicators (line 475)."""
        # Create registry with some indicators
        test_registry = IndicatorRegistry()

        @test_registry.register('indicator1')
        class Ind1:
            pass

        @test_registry.register('indicator2')
        class Ind2:
            pass

        # List indicators
        indicators = test_registry.list_indicators()

        assert 'indicator1' in indicators
        assert 'indicator2' in indicators
        assert len(indicators) == 2

    def test_global_indicator_registry_has_all_indicators(self):
        """Test that global registry contains all standard indicators."""
        # Verify all standard indicators are registered
        indicators = INDICATOR_REGISTRY.list_indicators()

        assert 'ma' in indicators
        assert 'rsi' in indicators
        assert 'bollinger_bands' in indicators
        assert 'macd' in indicators
        assert 'custom_macd' in indicators
        assert 'momentum' in indicators

    def test_register_indicator_convenience_function(self):
        """Test register_indicator convenience function."""
        # Create a test indicator using convenience function
        @register_indicator('test_convenience_indicator')
        class TestConvenienceIndicator:
            pass

        # Should be in global registry
        assert 'test_convenience_indicator' in INDICATOR_REGISTRY.list_indicators()

        # Clean up
        del INDICATOR_REGISTRY.indicators['test_convenience_indicator']


class TestIndicatorIntegration:
    """Test indicator integration and usage."""

    def test_all_indicators_can_be_instantiated(self):
        """Test that all registered indicators can be created."""
        # Get all registered indicators
        indicator_names = INDICATOR_REGISTRY.list_indicators()

        for name in indicator_names:
            # Get indicator class
            indicator_class = INDICATOR_REGISTRY.get(name)

            # Create instance
            indicator = indicator_class()

            # Verify it's a Keras layer
            assert isinstance(indicator, tf.keras.layers.Layer)

    def test_indicators_in_sequential_model(self):
        """Test that indicators can be used in a Sequential model."""
        # Create a simple model with indicator layers
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(60, 5)),
            LearnableMA(periods=[5, 15]),
            LearnableRSI(periods=[9, 21]),
            tf.keras.layers.Flatten(),  # Need to flatten before Dense
            tf.keras.layers.Dense(10)
        ])

        # Test prediction
        test_input = np.random.randn(2, 60, 5).astype(np.float32)
        output = model.predict(test_input, verbose=0)

        assert output.shape == (2, 10)

    def test_indicators_have_trainable_parameters(self):
        """Test that all indicators have trainable parameters."""
        inputs = tf.random.normal((2, 60, 5))

        indicators = [
            LearnableMA(),
            LearnableRSI(),
            LearnableBollingerBands(),
            LearnableMacd(),
            LearnableCustomMacd(),
            LearnableMomentum()
        ]

        for indicator in indicators:
            # Call indicator to build it
            indicator(inputs)

            # Verify it has trainable parameters
            assert len(indicator.trainable_variables) > 0

            # Check for appropriate parameter attributes
            # MACD and CustomMacd have fast/slow/signal params, others have period_params
            if isinstance(indicator, (LearnableMacd, LearnableCustomMacd)):
                assert indicator.fast_params is not None
                assert indicator.slow_params is not None
                assert indicator.signal_params is not None
            else:
                assert indicator.period_params is not None

    def test_indicator_parameters_are_positive(self):
        """Test that indicator parameters are constrained to be positive."""
        inputs = tf.random.normal((2, 60, 5))

        indicators = [
            LearnableMA(periods=[5, 15, 30]),
            LearnableRSI(periods=[9, 21, 30]),
            LearnableBollingerBands(periods=[10, 20, 30]),
        ]

        for indicator in indicators:
            # Build indicator
            indicator(inputs)

            # All parameters should be >= 0 due to NonNeg constraint
            params = indicator.period_params.numpy()
            assert np.all(params >= 0)
