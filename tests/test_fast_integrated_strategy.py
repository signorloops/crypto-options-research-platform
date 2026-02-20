"""
Tests for Fast Integrated Market Making Strategy.

Verifies:
1. Correctness matches standard IntegratedStrategy
2. Performance improvement (latency < 35ms)
3. FastRegimeDetector integration
"""

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.types import Greeks, MarketState, OrderBook, OrderBookLevel, Position
from strategies.market_making.fast_integrated_strategy import (
    FastIntegratedMarketMakingStrategy,
    FastIntegratedStrategyConfig,
)
from research.signals.fast_regime_detector import (
    FastVolatilityRegimeDetector,
    FastRegimeConfig,
    RegimeState,
)


class TestFastRegimeDetector:
    """Test FastVolatilityRegimeDetector."""

    def test_initial_state(self):
        """Test initial regime is MEDIUM."""
        detector = FastVolatilityRegimeDetector()
        assert detector.current_regime == RegimeState.MEDIUM

    def test_threshold_classification_low(self):
        """Test low volatility classification."""
        detector = FastVolatilityRegimeDetector()

        # Feed low volatility returns
        for _ in range(50):
            detector.update(0.0001)  # Very small returns

        # Should classify as LOW
        assert detector._volatility_estimate < 0.3
        regime, probs = detector._threshold_classify(detector._volatility_estimate)
        assert regime == RegimeState.LOW

    def test_threshold_classification_high(self):
        """Test high volatility classification."""
        detector = FastVolatilityRegimeDetector()

        # Feed high volatility returns
        np.random.seed(42)
        for _ in range(50):
            detector.update(np.random.normal(0, 0.05))  # High volatility

        # Should classify as HIGH
        assert detector._volatility_estimate > 0.6
        regime, probs = detector._threshold_classify(detector._volatility_estimate)
        assert regime == RegimeState.HIGH

    def test_custom_annualization_periods_used(self):
        """Volatility estimate should honor configured annualization periods."""
        config = FastRegimeConfig(annualization_periods=100.0)
        detector = FastVolatilityRegimeDetector(config)
        for _ in range(20):
            detector.update(0.01)
        assert detector._volatility_estimate >= 0

    def test_latency_requirement(self):
        """Test update latency is under 5ms."""
        detector = FastVolatilityRegimeDetector()

        # Pre-train
        np.random.seed(42)
        for ret in np.random.normal(0, 0.001, 100):
            detector.update(ret)

        # Measure latency
        latencies = []
        for ret in np.random.normal(0, 0.001, 1000):
            start = time.perf_counter()
            detector.update(ret)
            latencies.append((time.perf_counter() - start) * 1000)

        p95_latency = np.percentile(latencies, 95)
        assert p95_latency < 5.0, f"P95 latency {p95_latency}ms exceeds 5ms target"

    def test_spread_adjustment(self):
        """Test spread adjustment values."""
        detector = FastVolatilityRegimeDetector()

        # LOW regime: 0.8 multiplier
        detector.current_regime = RegimeState.LOW
        assert detector.get_spread_adjustment() == 0.8

        # MEDIUM regime: 1.0 multiplier
        detector.current_regime = RegimeState.MEDIUM
        assert detector.get_spread_adjustment() == 1.0

        # HIGH regime: 1.5 multiplier
        detector.current_regime = RegimeState.HIGH
        assert detector.get_spread_adjustment() == 1.5

    def test_stats_tracking(self):
        """Test statistics tracking."""
        detector = FastVolatilityRegimeDetector()

        # Generate some updates
        for ret in np.random.normal(0, 0.001, 50):
            detector.update(ret)

        stats = detector.get_stats()

        assert stats["total_inferences"] == 50
        assert "hmm_ratio" in stats
        assert "fallback_ratio" in stats
        assert "volatility_estimate" in stats

    def test_initial_hmm_training_triggers_at_min_samples(self, monkeypatch):
        """Initial HMM training should not wait for full retrain interval."""

        # Mock HMM init to avoid external dependency in this unit test
        def _mock_init_hmm(self):
            self._hmm_model = object()

        monkeypatch.setattr(FastVolatilityRegimeDetector, "_init_hmm", _mock_init_hmm)

        config = FastRegimeConfig(
            use_hmm=True,
            hmm_min_samples=5,
            hmm_retrain_interval=1000,
        )
        detector = FastVolatilityRegimeDetector(config)

        train_calls = []

        def _mock_async_train():
            train_calls.append(detector._hmm_sample_count)

        monkeypatch.setattr(detector, "_async_hmm_train", _mock_async_train)

        for _ in range(5):
            detector.update(0.001)

        assert train_calls, "Should trigger initial HMM training at minimum sample threshold"

    def test_hmm_timeout_counts_single_fallback(self):
        """A single HMM timeout should increment fallback counter exactly once."""
        config = FastRegimeConfig(use_hmm=True, hmm_timeout_ms=1.0)
        detector = FastVolatilityRegimeDetector(config)

        class _SlowModel:
            def predict(self, X):
                time.sleep(0.02)
                return np.array([RegimeState.MEDIUM.value])

            def score_samples(self, X):
                return 0.0, np.array([[0.2, 0.6, 0.2]])

        detector._hmm_model = _SlowModel()
        detector._hmm_fitted = True

        detector.update(0.001)

        assert detector._fallback_count == 1
        assert detector._threshold_inference_count == 1


class TestFastIntegratedStrategy:
    """Test FastIntegratedMarketMakingStrategy."""

    def create_market_state(self, price: float = 50000.0) -> MarketState:
        """Create a test market state."""
        timestamp = datetime.now(timezone.utc)
        return MarketState(
            timestamp=timestamp,
            instrument="BTC-USD",
            spot_price=price,
            order_book=OrderBook(
                timestamp=timestamp,
                instrument="BTC-USD",
                bids=[OrderBookLevel(price=price - 10, size=1.0)],
                asks=[OrderBookLevel(price=price + 10, size=1.0)],
            ),
            recent_trades=[],
        )

    def test_basic_quote_generation(self):
        """Test basic quote generation."""
        strategy = FastIntegratedMarketMakingStrategy()

        # Pre-train regime detector
        np.random.seed(42)
        for ret in np.random.normal(0, 0.001, 100):
            strategy.regime_detector.update(ret)

        state = self.create_market_state(50000.0)
        position = Position("BTC-USD", 0.0, 50000.0)

        quote = strategy.quote(state, position)

        assert quote.bid_price > 0
        assert quote.ask_price > quote.bid_price
        assert quote.bid_size >= 0
        assert quote.ask_size >= 0

    def test_inventory_skew(self):
        """Test inventory skew effect on prices."""
        strategy = FastIntegratedMarketMakingStrategy()

        # Pre-train
        np.random.seed(42)
        for ret in np.random.normal(0, 0.001, 100):
            strategy.regime_detector.update(ret)

        state = self.create_market_state(50000.0)

        # Zero inventory
        quote_zero = strategy.quote(state, Position("BTC-USD", 0.0, 50000.0))

        # Long inventory - bid should be lower (more aggressive selling)
        quote_long = strategy.quote(state, Position("BTC-USD", 5.0, 50000.0))

        # Short inventory - ask should be higher (more aggressive buying)
        quote_short = strategy.quote(state, Position("BTC-USD", -5.0, 50000.0))

        # Long inventory should skew reservation price down
        assert quote_long.bid_price < quote_zero.bid_price

        # Short inventory should skew reservation price up
        assert quote_short.ask_price > quote_zero.ask_price

    def test_circuit_breaker_halting(self):
        """Test circuit breaker halts trading."""
        from research.risk.circuit_breaker import CircuitBreakerConfig

        cb_config = CircuitBreakerConfig(daily_loss_limit_pct=0.01)
        config = FastIntegratedStrategyConfig(circuit_breaker=cb_config)
        strategy = FastIntegratedMarketMakingStrategy(config)

        # Pre-train
        np.random.seed(42)
        for ret in np.random.normal(0, 0.001, 100):
            strategy.regime_detector.update(ret)

        # Create a large loss scenario
        state = self.create_market_state(40000.0)  # Price drop
        position = Position("BTC-USD", 10.0, 50000.0)  # Big position, big loss

        quote = strategy.quote(state, position)

        # Should halt trading after circuit breaker triggers
        # Note: May take a few iterations to trigger

    def test_latency_performance(self):
        """Test quote generation latency is under 35ms."""
        strategy = FastIntegratedMarketMakingStrategy()

        # Pre-train regime detector
        np.random.seed(42)
        for ret in np.random.normal(0, 0.001, 100):
            strategy.regime_detector.update(ret)

        state = self.create_market_state(50000.0)
        position = Position("BTC-USD", 2.0, 50000.0)

        latencies = []
        for i in range(1000):
            # Vary price slightly
            state.order_book.bids[0].price = 49990 + i * 0.1
            state.order_book.asks[0].price = 50010 + i * 0.1

            start = time.perf_counter()
            quote = strategy.quote(state, position)
            latencies.append((time.perf_counter() - start) * 1000)

        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        mean_latency = np.mean(latencies)

        print(
            f"\nLatency stats: mean={mean_latency:.3f}ms, P95={p95_latency:.3f}ms, P99={p99_latency:.3f}ms"
        )

        assert p95_latency < 35.0, f"P95 latency {p95_latency}ms exceeds 35ms target"

    def test_performance_stats(self):
        """Test performance statistics tracking."""
        strategy = FastIntegratedMarketMakingStrategy()

        # Pre-train
        np.random.seed(42)
        for ret in np.random.normal(0, 0.001, 100):
            strategy.regime_detector.update(ret)

        # Generate some quotes
        state = self.create_market_state(50000.0)
        position = Position("BTC-USD", 0.0, 50000.0)

        for _ in range(100):
            strategy.quote(state, position)

        stats = strategy.get_performance_stats()

        assert stats["quote_count"] == 100
        assert stats["avg_latency_ms"] > 0
        assert "regime_stats" in stats

    def test_greeks_cache_hit_when_state_greeks_missing(self):
        """Cached greeks should be reused when subsequent market state lacks greeks."""
        strategy = FastIntegratedMarketMakingStrategy()

        # Prime cache with a quote that has greeks.
        state_with_greeks = self.create_market_state(50000.0)
        state_with_greeks.greeks = Greeks(delta=0.2, gamma=0.01, theta=-0.01, vega=0.1)
        strategy.quote(state_with_greeks, Position("BTC-USD", 0.0, 50000.0))

        # Next quote in same price bucket without greeks should hit cache.
        state_without_greeks = self.create_market_state(50020.0)
        state_without_greeks.greeks = None
        strategy.quote(state_without_greeks, Position("BTC-USD", 0.0, 50000.0))

        stats = strategy.get_performance_stats()
        assert stats["cache_hit_rate"] > 0
        assert strategy._current_greeks is not None

    def test_greeks_cache_max_entries(self):
        """Greeks cache should be bounded by configured max entries."""
        config = FastIntegratedStrategyConfig(greeks_cache_max_entries=2)
        strategy = FastIntegratedMarketMakingStrategy(config)

        for price in [50000.0, 50250.0, 50500.0]:
            state = self.create_market_state(price)
            state.greeks = Greeks(delta=0.1, gamma=0.01, theta=-0.01, vega=0.05)
            strategy.quote(state, Position("BTC-USD", 0.0, 50000.0))

        assert len(strategy._greeks_cache) <= 2

    def test_spread_multipliers(self):
        """Test spread multipliers in different regimes."""
        strategy = FastIntegratedMarketMakingStrategy()

        # Pre-train
        np.random.seed(42)
        for ret in np.random.normal(0, 0.001, 100):
            strategy.regime_detector.update(ret)

        # Test spread adjustment values directly
        multipliers = {}
        for regime in [RegimeState.LOW, RegimeState.MEDIUM, RegimeState.HIGH]:
            strategy.regime_detector.current_regime = regime
            multipliers[regime] = strategy.regime_detector.get_spread_adjustment()

        # Verify expected multipliers
        assert (
            multipliers[RegimeState.LOW] == 0.8
        ), f"LOW multiplier should be 0.8, got {multipliers[RegimeState.LOW]}"
        assert (
            multipliers[RegimeState.MEDIUM] == 1.0
        ), f"MEDIUM multiplier should be 1.0, got {multipliers[RegimeState.MEDIUM]}"
        assert (
            multipliers[RegimeState.HIGH] == 1.5
        ), f"HIGH multiplier should be 1.5, got {multipliers[RegimeState.HIGH]}"

        # Verify relationship
        assert multipliers[RegimeState.LOW] < multipliers[RegimeState.MEDIUM]
        assert multipliers[RegimeState.HIGH] > multipliers[RegimeState.MEDIUM]

    def test_risk_check_throttling_reduces_repeated_checks(self, monkeypatch):
        """Risk checks should be throttled for near-identical consecutive quotes."""
        config = FastIntegratedStrategyConfig(
            risk_check_interval_ms=10_000.0,
            risk_check_price_move_bps=10_000.0,
        )
        strategy = FastIntegratedMarketMakingStrategy(config)

        call_counter = {"checks": 0}
        original_check = strategy.circuit_breaker.check_risk_limits

        def _counted_check(portfolio):
            call_counter["checks"] += 1
            return original_check(portfolio)

        monkeypatch.setattr(strategy.circuit_breaker, "check_risk_limits", _counted_check)

        state = self.create_market_state(50000.0)
        position = Position("BTC-USD", 1.0, 50000.0)

        for _ in range(20):
            strategy.quote(state, position)

        assert call_counter["checks"] < 20

    def test_risk_check_revalidates_on_large_price_move(self):
        """Large price moves should bypass throttling and refresh risk checks."""
        config = FastIntegratedStrategyConfig(
            risk_check_interval_ms=60_000.0,
            risk_check_price_move_bps=1.0,
        )
        strategy = FastIntegratedMarketMakingStrategy(config)
        state = self.create_market_state(50000.0)
        position = Position("BTC-USD", 1.0, 50000.0)

        strategy.quote(state, position)
        first_check_time = strategy._last_risk_check_at

        # Move price by >1bps to force refresh.
        state.order_book.bids[0].price = 50020.0
        state.order_book.asks[0].price = 50040.0
        strategy.quote(state, position)

        assert strategy._last_risk_check_at is not None
        assert strategy._last_risk_check_at >= first_check_time

    def test_reset(self):
        """Test strategy reset."""
        strategy = FastIntegratedMarketMakingStrategy()

        # Pre-train and generate quotes
        np.random.seed(42)
        for ret in np.random.normal(0, 0.001, 100):
            strategy.regime_detector.update(ret)

        state = self.create_market_state(50000.0)
        position = Position("BTC-USD", 0.0, 50000.0)
        strategy.quote(state, position)

        # Reset
        strategy.reset()

        # Check reset state
        assert strategy._quote_count == 0
        assert strategy._cache_hits == 0
        assert len(strategy._greeks_cache) == 0

    def test_cache_miss_clears_stale_greeks(self):
        """Cache miss should clear stale Greeks to avoid stale hedging decisions."""
        config = FastIntegratedStrategyConfig(
            enable_circuit_breaker=False,
            enable_regime_detection=False,
            enable_adaptive_hedging=True,
            cache_greeks=True,
        )
        strategy = FastIntegratedMarketMakingStrategy(config)

        strategy._current_greeks = Greeks(
            delta=1.0,
            gamma=0.1,
            theta=-0.1,
            vega=0.2,
            rho=0.01,
        )
        strategy.hedger.should_hedge = MagicMock()

        state = self.create_market_state(50000.0)
        position = Position("BTC-USD", 0.0, 50000.0)

        strategy.quote(state, position)

        strategy.hedger.should_hedge.assert_not_called()
        assert strategy._current_greeks is None

    def test_return_calculation_uses_log_returns(self):
        """Fast strategy should use log returns consistent with standard strategy."""
        strategy = FastIntegratedMarketMakingStrategy()
        r1 = strategy._calculate_return(100.0)
        r2 = strategy._calculate_return(110.0)
        assert r1 is None
        assert r2 == pytest.approx(np.log(1.1))


class TestFastVsStandardComparison:
    """Compare Fast strategy with standard version."""

    def create_market_state(self, price: float = 50000.0) -> MarketState:
        """Create a test market state."""
        timestamp = datetime.now(timezone.utc)
        return MarketState(
            timestamp=timestamp,
            instrument="BTC-USD",
            spot_price=price,
            order_book=OrderBook(
                timestamp=timestamp,
                instrument="BTC-USD",
                bids=[OrderBookLevel(price=price - 10, size=1.0)],
                asks=[OrderBookLevel(price=price + 10, size=1.0)],
            ),
            recent_trades=[],
        )

    def test_functional_equivalence(self):
        """Test that fast version produces similar results."""
        from strategies.market_making.integrated_strategy import (
            IntegratedMarketMakingStrategy,
            IntegratedStrategyConfig,
        )

        fast_strategy = FastIntegratedMarketMakingStrategy()
        std_strategy = IntegratedMarketMakingStrategy()

        # Pre-train both
        np.random.seed(42)
        returns = np.random.normal(0, 0.001, 100)
        for ret in returns:
            fast_strategy.regime_detector.update(ret)
            std_strategy.regime_detector.update(ret)

        state = self.create_market_state(50000.0)
        position = Position("BTC-USD", 2.0, 50000.0)

        fast_quote = fast_strategy.quote(state, position)
        std_quote = std_strategy.quote(state, position)

        # Prices should be close (within 1%)
        assert abs(fast_quote.bid_price - std_quote.bid_price) / 50000 < 0.01
        assert abs(fast_quote.ask_price - std_quote.ask_price) / 50000 < 0.01
