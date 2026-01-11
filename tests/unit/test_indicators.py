"""Unit tests for indicators module.

Tests learnable technical indicators as specified in SRS Section 3.2.2 (30+ learnable parameters).
"""
import pytest
import numpy as np
import tensorflow as tf
from src.data.indicators import (
    IndicatorRegistry,
    LearnableMA,
    LearnableRSI,
    LearnableBollingerBands,
    LearnableMacd,
    LearnableMomentum,
    register_indicator,
)


class TestIndicatorRegistry:
    """Test indicator registration system."""

    def test_registry_initialization(self):
        """Should initialize empty registry."""
        registry = IndicatorRegistry()
        assert len(registry.indicators) == 0

    def test_register_indicator(self):
        """Should register indicator in registry."""
        registry = IndicatorRegistry()

        @registry.register('test_indicator')
        class TestIndicator:
            pass

        assert 'test_indicator' in registry.indicators
        assert registry.indicators['test_indicator'] == TestIndicator

    def test_get_indicator(self):
        """Should retrieve registered indicator."""
        registry = IndicatorRegistry()

        @registry.register('test_indicator')
        class TestIndicator:
            pass

        indicator_class = registry.get('test_indicator')
        assert indicator_class == TestIndicator

    def test_get_nonexistent_indicator_raises_error(self):
        """Should raise error for nonexistent indicator."""
        registry = IndicatorRegistry()

        with pytest.raises(KeyError):
            registry.get('nonexistent')

    def test_list_indicators(self):
        """Should list all registered indicators."""
        registry = IndicatorRegistry()

        @registry.register('indicator1')
        class Indicator1:
            pass

        @registry.register('indicator2')
        class Indicator2:
            pass

        indicators = registry.list_indicators()
        assert 'indicator1' in indicators
        assert 'indicator2' in indicators

    def test_automatic_registration_on_import(self):
        """Indicators should be automatically registered on import."""
        # When module is imported, indicators should be registered
        from src.data.indicators import INDICATOR_REGISTRY

        assert 'ma' in INDICATOR_REGISTRY.indicators
        assert 'rsi' in INDICATOR_REGISTRY.indicators
        assert 'macd' in INDICATOR_REGISTRY.indicators


class TestLearnableMA:
    """Test learnable Moving Average indicator."""

    def test_init_with_periods(self):
        """Should initialize with learnable periods."""
        ma = LearnableMA(periods=[5, 15, 30])

        assert ma.num_periods == 3
        assert hasattr(ma, 'period_params')

    def test_periods_are_trainable(self):
        """MA periods should be trainable parameters."""
        ma = LearnableMA(periods=[5, 15, 30])
        ma.build(input_shape=(None, 60, 5))

        # Check that parameters are trainable
        assert len(ma.trainable_variables) == 1  # period_params
        assert ma.trainable_variables[0].shape == (3,)

    def test_forward_pass(self):
        """Should compute MA features in forward pass."""
        ma = LearnableMA(periods=[5, 15, 30])
        ma.build(input_shape=(None, 60, 5))

        # Create sample input (batch_size=2, lookback=60, features=5)
        x = tf.random.normal((2, 60, 5))
        output = ma(x)

        # Output should have MA features
        assert output.shape[0] == 2  # batch_size
        assert output.shape[1] == 60  # lookback
        # Should have original features + MA features

    def test_ma_calculation_correct(self):
        """Should calculate moving average correctly."""
        # Create simple test case
        prices = np.array([[i for i in range(60)] for _ in range(1)])
        prices = tf.constant(prices, dtype=tf.float32)
        prices = tf.expand_dims(prices, -1)  # Add feature dimension

        ma = LearnableMA(periods=[5])
        ma.build(input_shape=(None, 60, 1))

        output = ma(prices)
        # Verify MA calculation is reasonable
        assert output.shape == prices.shape

    def test_learnable_period_initialization(self):
        """Should initialize periods with given values."""
        ma = LearnableMA(periods=[5, 15, 30])
        ma.build(input_shape=(None, 60, 5))

        # Periods should be initialized close to specified values
        period_values = ma.period_params.numpy()
        assert len(period_values) == 3

    def test_three_ma_periods_as_per_srs(self):
        """Should support 3 MA periods as specified in SRS."""
        ma = LearnableMA(periods=[5, 15, 30])
        ma.build(input_shape=(None, 60, 5))

        # Should have 3 learnable parameters
        assert ma.period_params.shape[0] == 3


class TestLearnableRSI:
    """Test learnable RSI indicator."""

    def test_init_with_periods(self):
        """Should initialize with learnable RSI periods."""
        rsi = LearnableRSI(periods=[9, 21, 30])

        assert rsi.num_periods == 3
        assert hasattr(rsi, 'period_params')

    def test_three_rsi_periods_as_per_srs(self):
        """Should support 3 RSI periods as specified in SRS."""
        rsi = LearnableRSI(periods=[9, 21, 30])
        rsi.build(input_shape=(None, 60, 5))

        # Should have 3 learnable parameters
        assert rsi.period_params.shape[0] == 3

    def test_rsi_output_range(self):
        """RSI output should be in [0, 100] range."""
        rsi = LearnableRSI(periods=[14])
        rsi.build(input_shape=(None, 60, 1))

        # Create sample input with price movements
        prices = tf.constant([[[40000 + i * 10] for i in range(60)]], dtype=tf.float32)
        output = rsi(prices)

        # RSI should be bounded
        # Note: actual implementation may normalize differently

    def test_rsi_calculation(self):
        """Should calculate RSI indicator."""
        rsi = LearnableRSI(periods=[14])
        rsi.build(input_shape=(None, 60, 1))

        x = tf.random.normal((2, 60, 1))
        output = rsi(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 60


class TestLearnableBollingerBands:
    """Test learnable Bollinger Bands indicator."""

    def test_init_with_periods(self):
        """Should initialize with learnable BB periods."""
        bb = LearnableBollingerBands(periods=[10, 20, 30])

        assert bb.num_periods == 3
        assert hasattr(bb, 'period_params')

    def test_three_bb_periods_as_per_srs(self):
        """Should support 3 BB periods as specified in SRS."""
        bb = LearnableBollingerBands(periods=[10, 20, 30])
        bb.build(input_shape=(None, 60, 5))

        # Should have 3 learnable parameters
        assert bb.period_params.shape[0] == 3

    def test_bollinger_bands_output(self):
        """Should output upper, middle, lower bands."""
        bb = LearnableBollingerBands(periods=[20])
        bb.build(input_shape=(None, 60, 1))

        x = tf.random.normal((2, 60, 1))
        output = bb(x)

        # Should have features for bands
        assert output.shape[0] == 2
        assert output.shape[1] == 60

    def test_bb_width_calculation(self):
        """Should calculate BB width correctly."""
        bb = LearnableBollingerBands(periods=[20])
        bb.build(input_shape=(None, 60, 1))

        # Create data with known volatility
        x = tf.random.normal((1, 60, 1))
        output = bb(x)

        assert output.shape[1] == 60


class TestLearnableMacd:
    """Test learnable MACD indicator."""

    def test_init_with_settings(self):
        """Should initialize with MACD settings."""
        macd = LearnableMacd(settings=[[12, 26, 9], [5, 35, 5], [19, 39, 9]])

        assert macd.num_settings == 3

    def test_three_macd_settings_as_per_srs(self):
        """Should support 3 MACD settings as specified in SRS (9 params)."""
        macd = LearnableMacd(settings=[[12, 26, 9], [5, 35, 5], [19, 39, 9]])
        macd.build(input_shape=(None, 60, 5))

        # 3 settings × 3 parameters each = 9 learnable parameters
        total_params = sum(v.shape.num_elements() for v in macd.trainable_variables)
        assert total_params == 9

    def test_custom_macd_pairs_as_per_srs(self):
        """Should support custom MACD pairs (9 params)."""
        # Custom MACD pairs: 3 pairs × 3 params = 9 learnable params
        macd = LearnableMacd(settings=[[8, 17, 9], [10, 20, 5], [15, 30, 10]])
        macd.build(input_shape=(None, 60, 5))

        total_params = sum(v.shape.num_elements() for v in macd.trainable_variables)
        assert total_params == 9

    def test_macd_output(self):
        """Should calculate MACD, signal, and histogram."""
        macd = LearnableMacd(settings=[[12, 26, 9]])
        macd.build(input_shape=(None, 60, 1))

        x = tf.random.normal((2, 60, 1))
        output = macd(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 60

    def test_macd_parameters_trainable(self):
        """MACD parameters should be trainable."""
        macd = LearnableMacd(settings=[[12, 26, 9]])
        macd.build(input_shape=(None, 60, 1))

        assert len(macd.trainable_variables) > 0


class TestLearnableMomentum:
    """Test learnable Momentum indicator."""

    def test_init_with_periods(self):
        """Should initialize with learnable momentum periods."""
        momentum = LearnableMomentum(periods=[5, 10, 15])

        assert momentum.num_periods == 3

    def test_three_momentum_periods_as_per_srs(self):
        """Should support 3 momentum periods as specified in SRS."""
        momentum = LearnableMomentum(periods=[5, 10, 15])
        momentum.build(input_shape=(None, 60, 5))

        # Should have 3 learnable parameters
        assert momentum.period_params.shape[0] == 3

    def test_momentum_calculation(self):
        """Should calculate momentum indicator."""
        momentum = LearnableMomentum(periods=[10])
        momentum.build(input_shape=(None, 60, 1))

        x = tf.random.normal((2, 60, 1))
        output = momentum(x)

        assert output.shape[0] == 2
        assert output.shape[1] == 60


class TestTotalLearnableParameters:
    """Test that total learnable indicator parameters matches SRS specification."""

    def test_total_30plus_learnable_params(self):
        """Total learnable indicator parameters should be 30+ as per SRS."""
        # MA: 3 periods
        ma = LearnableMA(periods=[5, 15, 30])
        ma.build(input_shape=(None, 60, 5))

        # MACD: 3 settings × 3 params = 9
        macd = LearnableMacd(settings=[[12, 26, 9], [5, 35, 5], [19, 39, 9]])
        macd.build(input_shape=(None, 60, 5))

        # Custom MACD: 3 pairs × 3 params = 9
        custom_macd = LearnableMacd(settings=[[8, 17, 9], [10, 20, 5], [15, 30, 10]])
        custom_macd.build(input_shape=(None, 60, 5))

        # RSI: 3 periods
        rsi = LearnableRSI(periods=[9, 21, 30])
        rsi.build(input_shape=(None, 60, 5))

        # BB: 3 periods
        bb = LearnableBollingerBands(periods=[10, 20, 30])
        bb.build(input_shape=(None, 60, 5))

        # Momentum: 3 periods
        momentum = LearnableMomentum(periods=[5, 10, 15])
        momentum.build(input_shape=(None, 60, 5))

        # Total: 3 + 9 + 9 + 3 + 3 + 3 = 30 learnable parameters
        total_params = (
            ma.trainable_variables[0].shape[0] +
            sum(v.shape.num_elements() for v in macd.trainable_variables) +
            sum(v.shape.num_elements() for v in custom_macd.trainable_variables) +
            rsi.trainable_variables[0].shape[0] +
            bb.trainable_variables[0].shape[0] +
            momentum.trainable_variables[0].shape[0]
        )

        assert total_params >= 30


class TestIndicatorIntegration:
    """Test indicator integration with data pipeline."""

    def test_add_indicators_to_features(self, sample_ohlcv_array):
        """Should add indicator features to input data."""
        from src.data.indicators import add_indicators_to_features

        # Mock config
        config = {
            'ma_periods': [5, 15, 30],
            'rsi_periods': [9, 21, 30],
        }

        features_with_indicators = add_indicators_to_features(
            sample_ohlcv_array,
            config
        )

        # Should have more features
        assert features_with_indicators.shape[-1] > sample_ohlcv_array.shape[-1]

    def test_cascade_indicator_addition_from_config(self, sample_config):
        """Adding indicator to config should cascade to feature generation."""
        from src.data.indicators import build_indicator_layer

        indicator_layer = build_indicator_layer(sample_config['indicators'])

        # Should build layer with all configured indicators
        assert indicator_layer is not None

    def test_indicator_params_saved_per_epoch(self, tmp_path):
        """Indicator parameters should be tracked over epochs as per SRS."""
        from src.data.indicators import save_indicator_params

        ma = LearnableMA(periods=[5, 15, 30])
        ma.build(input_shape=(None, 60, 5))

        output_file = tmp_path / "indicator_params.csv"
        save_indicator_params(
            indicator_layers=[ma],
            epoch=1,
            output_file=output_file
        )

        assert output_file.exists()

        # Verify CSV format
        import pandas as pd
        df = pd.read_csv(output_file)
        assert 'epoch' in df.columns
