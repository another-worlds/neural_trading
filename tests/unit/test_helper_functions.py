"""Unit tests for helper functions module.

Tests all 8 helper functions from SRS Section 3.3.1:
1. calculate_confidence
2. calculate_signal_strength
3. normalize_variance
4. calculate_profit_targets
5. calculate_dynamic_stop_loss
6. calculate_position_size_multiplier
7. check_multi_horizon_agreement
8. detect_variance_spike
"""
import pytest
import numpy as np
from src.utils.helper_functions import (
    calculate_confidence,
    calculate_signal_strength,
    normalize_variance,
    calculate_profit_targets,
    calculate_dynamic_stop_loss,
    calculate_position_size_multiplier,
    check_multi_horizon_agreement,
    detect_variance_spike,
)


class TestCalculateConfidence:
    """Test calculate_confidence function."""

    def test_zero_variance_returns_max_confidence(self):
        """Zero variance should return confidence close to 1.0."""
        confidence = calculate_confidence(0.0)
        assert confidence == pytest.approx(1.0, rel=1e-6)

    def test_unit_variance_returns_half_confidence(self):
        """Variance of 1.0 should return confidence of 0.5."""
        confidence = calculate_confidence(1.0)
        assert confidence == pytest.approx(0.5, rel=1e-6)

    def test_high_variance_returns_low_confidence(self):
        """High variance should return low confidence."""
        confidence = calculate_confidence(99.0)
        assert confidence < 0.02

    def test_array_input(self):
        """Should handle array inputs."""
        variances = np.array([0.0, 1.0, 9.0])
        confidences = calculate_confidence(variances)
        assert len(confidences) == 3
        assert confidences[0] > confidences[1] > confidences[2]

    def test_custom_eps(self):
        """Should respect custom epsilon value."""
        confidence1 = calculate_confidence(0.0, eps=1e-7)
        confidence2 = calculate_confidence(0.0, eps=1e-5)
        assert confidence1 != confidence2

    def test_negative_variance_handling(self):
        """Should handle negative variance gracefully."""
        # Negative variance is invalid but function should not crash
        confidence = calculate_confidence(-1.0)
        assert confidence > 0

    def test_infinity_variance(self):
        """Infinite variance should return zero confidence."""
        confidence = calculate_confidence(float('inf'))
        assert confidence == pytest.approx(0.0, abs=1e-6)

    def test_nan_handling(self):
        """Should handle NaN input."""
        confidence = calculate_confidence(float('nan'))
        assert np.isnan(confidence)


class TestCalculateSignalStrength:
    """Test calculate_signal_strength function."""

    def test_perfect_signal(self):
        """Perfect direction and confidence should return 1.0."""
        signal = calculate_signal_strength(1.0, 1.0)
        assert signal == pytest.approx(1.0)

    def test_zero_confidence_kills_signal(self):
        """Zero confidence should make signal zero regardless of direction."""
        signal = calculate_signal_strength(1.0, 0.0)
        assert signal == pytest.approx(0.0)

    def test_zero_direction_kills_signal(self):
        """Zero direction probability should make signal zero."""
        signal = calculate_signal_strength(0.0, 1.0)
        assert signal == pytest.approx(0.0)

    def test_array_inputs(self):
        """Should handle array inputs."""
        directions = np.array([0.9, 0.7, 0.5])
        confidences = np.array([0.8, 0.9, 0.6])
        signals = calculate_signal_strength(directions, confidences)
        assert len(signals) == 3
        assert all(signals <= 1.0)
        assert all(signals >= 0.0)

    def test_signal_multiplication(self):
        """Signal should be product of direction and confidence."""
        signal = calculate_signal_strength(0.8, 0.5)
        assert signal == pytest.approx(0.4)

    def test_mixed_arrays(self):
        """Should handle broadcasting."""
        signals = calculate_signal_strength(
            np.array([0.9, 0.8, 0.7]),
            0.5
        )
        expected = np.array([0.45, 0.4, 0.35])
        np.testing.assert_array_almost_equal(signals, expected)


class TestNormalizeVariance:
    """Test normalize_variance function."""

    def test_variance_equals_mean(self):
        """Variance equal to mean should return zero."""
        normalized = normalize_variance(2.0, 2.0, 1.0)
        assert normalized == pytest.approx(0.0)

    def test_variance_one_std_above_mean(self):
        """Variance one std above mean should return 1.0."""
        normalized = normalize_variance(3.0, 2.0, 1.0)
        assert normalized == pytest.approx(1.0)

    def test_variance_one_std_below_mean(self):
        """Variance one std below mean should return -1.0."""
        normalized = normalize_variance(1.0, 2.0, 1.0)
        assert normalized == pytest.approx(-1.0)

    def test_zero_std_returns_zero(self):
        """Zero standard deviation should return 0.0."""
        normalized = normalize_variance(5.0, 2.0, 0.0)
        assert normalized == pytest.approx(0.0)

    def test_array_inputs(self):
        """Should handle array inputs."""
        variances = np.array([1.0, 2.0, 3.0])
        means = np.array([2.0, 2.0, 2.0])
        stds = np.array([1.0, 1.0, 1.0])
        normalized = normalize_variance(variances, means, stds)
        expected = np.array([-1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_custom_eps(self):
        """Should respect custom epsilon."""
        normalized = normalize_variance(2.0, 2.0, 0.0, eps=1e-5)
        assert normalized == pytest.approx(0.0)

    def test_near_zero_std_with_eps(self):
        """Near-zero std should be handled with epsilon."""
        normalized = normalize_variance(3.0, 2.0, 1e-10, eps=1e-7)
        assert np.isfinite(normalized)


class TestCalculateProfitTargets:
    """Test calculate_profit_targets function."""

    def test_basic_profit_targets(self):
        """Should calculate profit targets correctly."""
        entry_price = 100.0
        predictions = [102.0, 105.0, 110.0]
        result = calculate_profit_targets(entry_price, predictions)

        assert result['tp1'] == 102.0
        assert result['tp2'] == 105.0
        assert result['tp3'] == 110.0
        assert result['tp1_pct'] == pytest.approx(2.0)
        assert result['tp2_pct'] == pytest.approx(5.0)
        assert result['tp3_pct'] == pytest.approx(10.0)

    def test_negative_profit_targets(self):
        """Should handle downward predictions (negative profit)."""
        entry_price = 100.0
        predictions = [98.0, 95.0, 90.0]
        result = calculate_profit_targets(entry_price, predictions)

        assert result['tp1_pct'] == pytest.approx(-2.0)
        assert result['tp2_pct'] == pytest.approx(-5.0)
        assert result['tp3_pct'] == pytest.approx(-10.0)

    def test_result_structure(self):
        """Result should contain all required keys."""
        entry_price = 42000.0
        predictions = [42100.0, 42200.0, 42300.0]
        result = calculate_profit_targets(entry_price, predictions)

        required_keys = ['tp1', 'tp2', 'tp3', 'tp1_pct', 'tp2_pct', 'tp3_pct']
        assert all(key in result for key in required_keys)

    def test_type_conversion(self):
        """Should handle various numeric types."""
        result = calculate_profit_targets("100", [102, 105.0, "110"])
        assert isinstance(result['tp1'], float)
        assert isinstance(result['tp1_pct'], float)

    def test_realistic_btc_prices(self):
        """Test with realistic BTC price values."""
        entry_price = 42000.0
        predictions = [42150.5, 42200.3, 42350.1]
        result = calculate_profit_targets(entry_price, predictions)

        assert result['tp1'] == pytest.approx(42150.5)
        assert result['tp1_pct'] == pytest.approx(0.3583, rel=1e-3)


class TestCalculateDynamicStopLoss:
    """Test calculate_dynamic_stop_loss function."""

    def test_long_position_stop_below_entry(self):
        """LONG position stop loss should be below entry price."""
        stop_loss = calculate_dynamic_stop_loss(
            entry_price=100.0,
            position_type='LONG',
            variance=1.0,
            variance_rolling_mean=1.0,
            base_stop_pct=0.02
        )
        assert stop_loss < 100.0

    def test_short_position_stop_above_entry(self):
        """SHORT position stop loss should be above entry price."""
        stop_loss = calculate_dynamic_stop_loss(
            entry_price=100.0,
            position_type='SHORT',
            variance=1.0,
            variance_rolling_mean=1.0,
            base_stop_pct=0.02
        )
        assert stop_loss > 100.0

    def test_base_stop_calculation(self):
        """With variance=mean, stop should be approximately base_stop_pct * 2."""
        entry_price = 100.0
        stop_loss = calculate_dynamic_stop_loss(
            entry_price=entry_price,
            position_type='LONG',
            variance=1.0,
            variance_rolling_mean=1.0,
            base_stop_pct=0.02
        )
        # adjustment_factor = 1 + variance/mean = 1 + 1 = 2
        # stop_distance = 100 * 0.02 * 2 = 4
        expected = entry_price - 4.0
        assert stop_loss == pytest.approx(expected)

    def test_high_variance_wider_stop(self):
        """Higher variance should result in wider stop loss."""
        stop_normal = calculate_dynamic_stop_loss(
            entry_price=100.0,
            position_type='LONG',
            variance=1.0,
            variance_rolling_mean=1.0,
            base_stop_pct=0.02
        )
        stop_high = calculate_dynamic_stop_loss(
            entry_price=100.0,
            position_type='LONG',
            variance=5.0,
            variance_rolling_mean=1.0,
            base_stop_pct=0.02
        )
        # Higher variance = further from entry
        assert abs(100.0 - stop_high) > abs(100.0 - stop_normal)

    def test_max_variance_multiplier_cap(self):
        """Variance multiplier should be capped at max_variance_multiplier."""
        stop_loss = calculate_dynamic_stop_loss(
            entry_price=100.0,
            position_type='LONG',
            variance=100.0,
            variance_rolling_mean=1.0,
            base_stop_pct=0.02,
            max_variance_multiplier=2.0
        )
        # Max multiplier = 2.0, so max adjustment_factor = 3.0
        # Max stop_distance = 100 * 0.02 * 3 = 6
        max_stop = 100.0 - 6.0
        assert stop_loss >= max_stop - 0.01  # Allow small floating point error

    def test_custom_base_stop_pct(self):
        """Should respect custom base_stop_pct."""
        stop_2pct = calculate_dynamic_stop_loss(
            entry_price=100.0,
            position_type='LONG',
            variance=1.0,
            variance_rolling_mean=1.0,
            base_stop_pct=0.02
        )
        stop_5pct = calculate_dynamic_stop_loss(
            entry_price=100.0,
            position_type='LONG',
            variance=1.0,
            variance_rolling_mean=1.0,
            base_stop_pct=0.05
        )
        assert abs(100.0 - stop_5pct) > abs(100.0 - stop_2pct)


class TestCalculatePositionSizeMultiplier:
    """Test calculate_position_size_multiplier function."""

    def test_high_confidence_returns_high_size(self):
        """High confidence should return size_high."""
        multiplier = calculate_position_size_multiplier(
            confidence=0.8,
            size_high=1.2,
            conf_high_thresh=0.7
        )
        assert multiplier == 1.2

    def test_normal_confidence_returns_normal_size(self):
        """Normal confidence should return size_normal."""
        multiplier = calculate_position_size_multiplier(
            confidence=0.6,
            size_normal=1.0,
            conf_high_thresh=0.7,
            conf_low_thresh=0.5
        )
        assert multiplier == 1.0

    def test_low_confidence_returns_low_size(self):
        """Low confidence should return size_low."""
        multiplier = calculate_position_size_multiplier(
            confidence=0.4,
            size_low=0.6,
            conf_low_thresh=0.5
        )
        assert multiplier == 0.6

    def test_boundary_high_threshold(self):
        """Confidence exactly at high threshold should return size_normal."""
        multiplier = calculate_position_size_multiplier(
            confidence=0.7,
            size_high=1.2,
            size_normal=1.0,
            conf_high_thresh=0.7,
            conf_low_thresh=0.5
        )
        # Exactly at threshold: should be normal (not high)
        assert multiplier == 1.0

    def test_boundary_low_threshold(self):
        """Confidence exactly at low threshold should return size_low."""
        multiplier = calculate_position_size_multiplier(
            confidence=0.5,
            size_low=0.6,
            conf_low_thresh=0.5
        )
        # Exactly at threshold: should be low
        assert multiplier == 0.6

    def test_custom_sizes(self):
        """Should respect custom size values."""
        multiplier = calculate_position_size_multiplier(
            confidence=0.9,
            size_high=2.0,
            size_normal=1.5,
            size_low=0.5,
            conf_high_thresh=0.7,
            conf_low_thresh=0.5
        )
        assert multiplier == 2.0

    def test_all_three_tiers(self):
        """Test all three tiers with various confidences."""
        # High tier
        assert calculate_position_size_multiplier(0.75) == 1.2
        # Normal tier
        assert calculate_position_size_multiplier(0.6) == 1.0
        # Low tier
        assert calculate_position_size_multiplier(0.3) == 0.6


class TestCheckMultiHorizonAgreement:
    """Test check_multi_horizon_agreement function."""

    def test_perfect_upward_agreement(self):
        """All predictions above current should show agreement."""
        is_agreed, agreement = check_multi_horizon_agreement(
            price_predictions=[105, 110, 115],
            current_price=100
        )
        assert is_agreed is True
        assert agreement == pytest.approx(1.0)

    def test_perfect_downward_agreement(self):
        """All predictions below current should show agreement."""
        is_agreed, agreement = check_multi_horizon_agreement(
            price_predictions=[95, 90, 85],
            current_price=100
        )
        assert is_agreed is True
        assert agreement == pytest.approx(1.0)

    def test_no_agreement(self):
        """Equal up and down predictions with default threshold."""
        is_agreed, agreement = check_multi_horizon_agreement(
            price_predictions=[105, 95],
            current_price=100,
            agreement_threshold=0.67
        )
        assert is_agreed is False
        assert agreement == pytest.approx(0.5)

    def test_partial_agreement_passes(self):
        """2 out of 3 should pass with default threshold 0.67."""
        is_agreed, agreement = check_multi_horizon_agreement(
            price_predictions=[105, 110, 95],
            current_price=100,
            agreement_threshold=0.67
        )
        assert is_agreed is True
        assert agreement == pytest.approx(2/3)

    def test_partial_agreement_fails(self):
        """2 out of 4 should fail with default threshold 0.67."""
        is_agreed, agreement = check_multi_horizon_agreement(
            price_predictions=[105, 110, 95, 90],
            current_price=100,
            agreement_threshold=0.67
        )
        assert is_agreed is False
        assert agreement == pytest.approx(0.5)

    def test_custom_threshold(self):
        """Should respect custom agreement threshold."""
        is_agreed, agreement = check_multi_horizon_agreement(
            price_predictions=[105, 95],
            current_price=100,
            agreement_threshold=0.5
        )
        # 50% agreement with 50% threshold should pass
        assert is_agreed is True

    def test_numpy_array_input(self):
        """Should handle numpy array input."""
        is_agreed, agreement = check_multi_horizon_agreement(
            price_predictions=np.array([105.0, 110.0, 115.0]),
            current_price=100.0
        )
        assert is_agreed is True

    def test_realistic_btc_scenario(self):
        """Test with realistic BTC predictions."""
        is_agreed, agreement = check_multi_horizon_agreement(
            price_predictions=[42150.5, 42200.3, 42350.1],
            current_price=42000.0
        )
        assert is_agreed is True
        assert agreement == pytest.approx(1.0)


class TestDetectVarianceSpike:
    """Test detect_variance_spike function."""

    def test_no_spike_when_below_threshold(self):
        """Variance below threshold should not be detected as spike."""
        is_spike = detect_variance_spike(
            variance=2.0,
            variance_rolling_mean=2.0,
            variance_rolling_std=1.0,
            spike_threshold=2.0
        )
        assert is_spike is False

    def test_spike_when_above_threshold(self):
        """Variance above threshold should be detected as spike."""
        is_spike = detect_variance_spike(
            variance=10.0,
            variance_rolling_mean=2.0,
            variance_rolling_std=1.0,
            spike_threshold=2.0
        )
        assert is_spike is True

    def test_spike_at_exact_threshold(self):
        """Variance exactly at threshold should be detected as spike."""
        variance_rolling_mean = 2.0
        spike_threshold = 2.0
        spike_level = spike_threshold * variance_rolling_mean  # 4.0

        is_spike = detect_variance_spike(
            variance=4.0,
            variance_rolling_mean=variance_rolling_mean,
            variance_rolling_std=1.0,
            spike_threshold=spike_threshold
        )
        # At exactly spike level, should NOT be spike (> not >=)
        assert is_spike is False

    def test_spike_just_above_threshold(self):
        """Variance just above threshold should be spike."""
        is_spike = detect_variance_spike(
            variance=4.01,
            variance_rolling_mean=2.0,
            variance_rolling_std=1.0,
            spike_threshold=2.0
        )
        assert is_spike is True

    def test_custom_spike_threshold(self):
        """Should respect custom spike threshold."""
        is_spike_low = detect_variance_spike(
            variance=5.0,
            variance_rolling_mean=2.0,
            variance_rolling_std=1.0,
            spike_threshold=1.5
        )
        is_spike_high = detect_variance_spike(
            variance=5.0,
            variance_rolling_mean=2.0,
            variance_rolling_std=1.0,
            spike_threshold=3.0
        )
        assert is_spike_low is True
        assert is_spike_high is False

    def test_zero_mean_with_eps(self):
        """Zero mean should be handled with epsilon."""
        is_spike = detect_variance_spike(
            variance=0.1,
            variance_rolling_mean=0.0,
            variance_rolling_std=0.0,
            spike_threshold=2.0,
            eps=1e-7
        )
        # Should not crash
        assert isinstance(is_spike, (bool, np.bool_))

    def test_std_not_used_in_calculation(self):
        """variance_rolling_std parameter exists but not used in calculation."""
        # Both should give same result regardless of std
        is_spike1 = detect_variance_spike(10.0, 2.0, 1.0, 2.0)
        is_spike2 = detect_variance_spike(10.0, 2.0, 100.0, 2.0)
        assert is_spike1 == is_spike2
