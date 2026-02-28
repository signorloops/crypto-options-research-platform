"""
Tests for rough-volatility pricer with jump extensions.
"""
import numpy as np

from research.pricing.rough_volatility import RoughVolConfig, RoughVolatilityPricer


class TestRoughVolatilityWithJumps:
    """Test rough-volatility jump extensions and diagnostics."""

    def test_cojump_mode_records_jump_events_and_keeps_positive_paths(self):
        """Co-jump mode should generate jump diagnostics and valid paths."""
        pricer = RoughVolatilityPricer(
            RoughVolConfig(
                spot=100.0,
                maturity=0.5,
                n_steps=32,
                n_paths=256,
                seed=42,
                jump_mode="cojump",
                jump_intensity=6.0,
                jump_mean=-0.01,
                jump_std=0.08,
                variance_jump_scale=0.35,
            )
        )
        spots, vars_path = pricer.simulate_paths()
        stats = pricer.get_last_simulation_stats()

        assert np.all(spots > 0.0)
        assert np.all(vars_path > 0.0)
        assert stats["jump_mode"] == "cojump"
        assert stats["avg_jump_events_per_path"] > 0.0

    def test_clustered_mode_reports_nonzero_intensity_dispersion(self):
        """Clustered jump mode should show varying jump intensity."""
        pricer = RoughVolatilityPricer(
            RoughVolConfig(
                spot=100.0,
                maturity=0.75,
                n_steps=40,
                n_paths=200,
                seed=123,
                jump_mode="clustered",
                jump_intensity=3.0,
                jump_excitation=1.8,
                jump_decay=6.0,
                jump_std=0.06,
            )
        )
        pricer.simulate_paths()
        stats = pricer.get_last_simulation_stats()

        assert stats["jump_mode"] == "clustered"
        assert stats["avg_jump_intensity"] > 0.0
        assert stats["jump_intensity_std"] > 0.0

    def test_price_ci_includes_runtime_and_jump_diagnostics(self):
        """Pricing output should include runtime and jump diagnostics."""
        pricer = RoughVolatilityPricer(
            RoughVolConfig(
                spot=100.0,
                maturity=0.4,
                n_steps=28,
                n_paths=180,
                seed=7,
                jump_mode="cojump",
                jump_intensity=4.0,
                jump_std=0.05,
            )
        )
        out = pricer.price_with_confidence_interval(strike=100.0, option_type="call", confidence=0.9)

        assert out["ci_low"] <= out["price"] <= out["ci_high"]
        assert out["simulation_time_sec"] >= 0.0
        assert out["total_time_sec"] >= out["simulation_time_sec"]
        assert out["avg_jump_events_per_path"] >= 0.0
        assert out["jump_mode"] == "cojump"
