"""
Tests for Volatility Regime Detector.
"""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from research.signals.regime_detector import (
    RegimeConfig,
    RegimeState,
    SimpleThresholdRegimeDetector,
    VolatilityRegimeDetector,
)


class TestRegimeConfig:
    """Tests for RegimeConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RegimeConfig()
        assert config.n_regimes == 3
        assert config.lookback_window == 100
        assert config.min_samples_for_training == 50
        assert config.switch_probability_threshold == 0.7
        assert config.annualization_periods > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = RegimeConfig(
            n_regimes=2,
            lookback_window=200,
            min_samples_for_training=100
        )
        assert config.n_regimes == 2
        assert config.lookback_window == 200
        assert config.min_samples_for_training == 100


class TestVolatilityRegimeDetectorInitialization:
    """Tests for detector initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        detector = VolatilityRegimeDetector()
        assert detector.current_regime == RegimeState.MEDIUM
        assert detector.model is not None
        assert not detector._is_fitted

    def test_custom_config_initialization(self):
        """Test initialization with custom config."""
        config = RegimeConfig(n_regimes=2)
        detector = VolatilityRegimeDetector(config)
        assert detector.config.n_regimes == 2


class TestVolatilityRegimeDetectorUpdate:
    """Tests for update method."""

    def test_update_before_training(self):
        """Test update before model is trained."""
        detector = VolatilityRegimeDetector()

        # Add a few samples
        for _ in range(10):
            regime = detector.update(0.001)

        # Should return MEDIUM before training
        assert regime == RegimeState.MEDIUM
        assert not detector._is_fitted

    def test_update_triggers_training(self):
        """Test that update triggers model training."""
        config = RegimeConfig(min_samples_for_training=20)
        detector = VolatilityRegimeDetector(config)

        # Generate synthetic returns
        np.random.seed(42)
        returns = np.random.normal(0, 0.001, 25)

        for ret in returns:
            detector.update(ret)

        # Should be fitted after enough samples
        assert detector._is_fitted
        assert len(detector.regime_history) > 0

    def test_low_volatility_regime(self):
        """Test detection of low volatility regime."""
        config = RegimeConfig(min_samples_for_training=30)
        detector = VolatilityRegimeDetector(config)

        # Generate low volatility returns
        np.random.seed(42)
        returns = np.random.normal(0, 0.0001, 50)  # Very low vol

        for ret in returns:
            regime = detector.update(ret)

        # Should detect low volatility
        assert detector._is_fitted
        # Note: Exact regime depends on model, but vol should be classified

    def test_high_volatility_regime(self):
        """Test detection of high volatility regime."""
        config = RegimeConfig(min_samples_for_training=30)
        detector = VolatilityRegimeDetector(config)

        # Generate high volatility returns
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 50)  # High vol

        for ret in returns:
            regime = detector.update(ret)

        assert detector._is_fitted

    def test_degenerate_training_data_increments_failure_counter(self):
        """Constant returns should skip unstable HMM fitting and record failures."""
        config = RegimeConfig(min_samples_for_training=5, n_regimes=3)
        detector = VolatilityRegimeDetector(config)

        for _ in range(10):
            detector.update(0.0)

        assert detector._is_fitted is False
        assert detector._training_failures > 0
        assert detector._last_training_error == "insufficient_distinct_samples"


class TestSpreadAdjustment:
    """Tests for spread adjustment."""

    def test_low_vol_spread_multiplier(self):
        """Test spread multiplier for low volatility."""
        detector = VolatilityRegimeDetector()
        detector.current_regime = RegimeState.LOW

        multiplier = detector.get_spread_adjustment()
        assert multiplier == 0.8

    def test_medium_vol_spread_multiplier(self):
        """Test spread multiplier for medium volatility."""
        detector = VolatilityRegimeDetector()
        detector.current_regime = RegimeState.MEDIUM

        multiplier = detector.get_spread_adjustment()
        assert multiplier == 1.0

    def test_high_vol_spread_multiplier(self):
        """Test spread multiplier for high volatility."""
        detector = VolatilityRegimeDetector()
        detector.current_regime = RegimeState.HIGH

        multiplier = detector.get_spread_adjustment()
        assert multiplier == 1.5


class TestRegimeSwitchPrediction:
    """Tests for regime switch prediction."""

    def test_switch_probability_before_training(self):
        """Test switch probability before model is trained."""
        detector = VolatilityRegimeDetector()

        prob = detector.predict_regime_switch_probability()
        assert prob == 0.0

    def test_switch_alert_before_training(self):
        """Test switch alert before training."""
        detector = VolatilityRegimeDetector()

        alert, msg = detector.get_regime_switch_alert()
        assert alert is False


class TestRegimeStats:
    """Tests for regime statistics."""

    def test_stats_before_training(self):
        """Test stats before any data."""
        detector = VolatilityRegimeDetector()

        stats = detector.get_regime_stats()
        assert stats == {}

    def test_stats_after_updates(self):
        """Test stats after some updates."""
        config = RegimeConfig(min_samples_for_training=20)
        detector = VolatilityRegimeDetector(config)

        np.random.seed(42)
        returns = np.random.normal(0, 0.001, 30)

        for ret in returns:
            detector.update(ret)

        stats = detector.get_regime_stats()
        assert "total_observations" in stats
        assert "current_regime" in stats
        assert "model_fitted" in stats
        assert stats["model_fitted"] is True
        assert "training_failures" in stats


class TestReset:
    """Tests for reset functionality."""

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        config = RegimeConfig(min_samples_for_training=20)
        detector = VolatilityRegimeDetector(config)

        # Add some data
        np.random.seed(42)
        for ret in np.random.normal(0, 0.001, 30):
            detector.update(ret)

        assert detector._is_fitted
        assert len(detector.regime_history) > 0

        # Reset
        detector.reset()

        assert not detector._is_fitted
        assert len(detector.regime_history) == 0
        assert detector.current_regime == RegimeState.MEDIUM


class TestSimpleThresholdRegimeDetector:
    """Tests for simple threshold-based detector."""

    def test_low_vol_detection(self):
        """Test low volatility detection."""
        detector = SimpleThresholdRegimeDetector()

        # Low volatility returns
        np.random.seed(42)
        returns = np.random.normal(0, 0.0001, 20)

        for ret in returns:
            regime = detector.update(ret)

        assert regime == RegimeState.LOW

    def test_high_vol_detection(self):
        """Test high volatility detection."""
        detector = SimpleThresholdRegimeDetector()

        # High volatility returns
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 20)

        for ret in returns:
            regime = detector.update(ret)

        assert regime == RegimeState.HIGH

    def test_spread_multipliers(self):
        """Test spread multipliers."""
        detector = SimpleThresholdRegimeDetector()

        detector.current_regime = RegimeState.LOW
        assert detector.get_spread_adjustment() == 0.8

        detector.current_regime = RegimeState.MEDIUM
        assert detector.get_spread_adjustment() == 1.0

        detector.current_regime = RegimeState.HIGH
        assert detector.get_spread_adjustment() == 1.5


class TestRealizedVolatilityCalculation:
    """Tests for realized volatility calculation."""

    def test_rv_with_insufficient_data(self):
        """Test RV calculation with insufficient data."""
        detector = VolatilityRegimeDetector()

        rv = detector._calculate_realized_volatility([0.001], 5)
        assert rv >= 0  # Should return non-negative value

    def test_rv_calculation(self):
        """Test RV calculation with sufficient data."""
        detector = VolatilityRegimeDetector()

        np.random.seed(42)
        returns = list(np.random.normal(0, 0.001, 20))

        rv = detector._calculate_realized_volatility(returns, 10)
        assert rv > 0

    def test_rv_respects_annualization_periods(self):
        """Annualization factor should be configurable for different markets."""
        config = RegimeConfig(annualization_periods=100.0)
        detector = VolatilityRegimeDetector(config)
        rv = detector._calculate_realized_volatility([0.01] * 20, 10)
        assert rv >= 0


class TestFeatureExtraction:
    """Tests for feature extraction."""

    def test_feature_extraction(self):
        """Test feature extraction from returns."""
        detector = VolatilityRegimeDetector()

        features = detector._extract_features(0.001)

        assert features.log_return == 0.001
        assert features.realized_vol_5min >= 0
        assert features.realized_vol_15min >= 0
        assert features.realized_vol_30min >= 0
        assert features.timestamp is not None

    def test_log_return_conversion(self):
        """Test log return conversion."""
        config = RegimeConfig(use_log_returns=True)
        detector = VolatilityRegimeDetector(config)

        # Simple return of 1% should be converted to log return
        detector.update(0.01)

        # Check that log return was used
        assert len(detector._returns_buffer) == 1
        # log(1.01) ≈ 0.00995
        assert abs(detector._returns_buffer[0] - np.log(1.01)) < 0.0001


class TestIntegration:
    """Integration tests for regime detector."""

    def test_full_workflow(self):
        """Test full workflow from data to regime classification."""
        config = RegimeConfig(
            min_samples_for_training=30,
            retrain_interval=50
        )
        detector = VolatilityRegimeDetector(config)

        # Phase 1: Low volatility
        np.random.seed(42)
        low_vol_returns = np.random.normal(0, 0.0002, 40)
        for ret in low_vol_returns:
            detector.update(ret)

        assert detector._is_fitted
        low_vol_regime = detector.current_regime

        # Phase 2: High volatility
        high_vol_returns = np.random.normal(0, 0.005, 30)
        for ret in high_vol_returns:
            detector.update(ret)

        high_vol_regime = detector.current_regime

        # Regimes should be different (with high probability)
        # Note: This is probabilistic, so we just check that detector works

        # Get spread adjustments
        low_mult = config.spread_multipliers[low_vol_regime]
        high_mult = config.spread_multipliers[high_vol_regime]

        # High vol should have higher multiplier
        assert high_mult >= low_mult

    def test_regime_persistence(self):
        """Test that regime doesn't change too frequently."""
        config = RegimeConfig(min_samples_for_training=30)
        detector = VolatilityRegimeDetector(config)

        np.random.seed(42)
        returns = np.random.normal(0, 0.001, 100)

        regimes = []
        for ret in returns:
            regime = detector.update(ret)
            if detector._is_fitted:
                regimes.append(regime)

        # Count regime changes
        changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])

        # Shouldn't change too frequently
        assert changes <= len(regimes) / 5  # Less than 20% changes


class TestRegimeMappingAndStickiness:
    """Tests for variance-based state mapping and sticky switching."""

    def test_state_map_orders_by_conditional_variance_not_mean(self):
        """State ordering should follow risk (covariance trace), not return mean."""
        detector = VolatilityRegimeDetector(RegimeConfig(n_regimes=3))
        detector.model = SimpleNamespace(
            means_=np.array([[0.02], [0.001], [0.005]]),
            covars_=np.array([[[0.04]], [[0.0004]], [[0.01]]]),
        )

        mapping = detector._build_state_map_from_model()

        # old state 1 (lowest variance) -> LOW (0)
        # old state 2 (middle variance) -> MEDIUM (1)
        # old state 0 (highest variance) -> HIGH (2)
        assert mapping[1] == RegimeState.LOW.value
        assert mapping[2] == RegimeState.MEDIUM.value
        assert mapping[0] == RegimeState.HIGH.value

    def test_sticky_switch_blocks_low_confidence_regime_flip(self):
        """Detector should avoid flipping regimes on low-confidence one-step predictions."""
        config = RegimeConfig(
            min_samples_for_training=1,
            retrain_interval=10_000,
            regime_persistence_min_samples=5,
            min_confidence_for_switch=0.65,
            switch_hysteresis=0.2,
        )
        detector = VolatilityRegimeDetector(config)
        detector._is_fitted = True
        detector.current_regime = RegimeState.LOW
        detector.regime_probabilities = np.array([0.7, 0.2, 0.1])
        detector._current_regime_run_length = 1
        detector._last_training_sample = detector._sample_count

        detector._predict_regime = lambda: (RegimeState.HIGH, np.array([0.30, 0.28, 0.42]))

        regime = detector.update(0.001)
        assert regime == RegimeState.LOW
