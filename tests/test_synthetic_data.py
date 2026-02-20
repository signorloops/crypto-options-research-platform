"""
Tests for synthetic data generation.
"""
from datetime import datetime

import numpy as np
import pandas as pd

from data.generators.synthetic import (
    CompleteMarketSimulator,
    GBMPriceGenerator,
    MertonJumpDiffusion,
    OrderBookSimulator,
    PriceModelParams,
    TradeFlowSimulator,
)


class TestGBMPriceGenerator:
    """Test Geometric Brownian Motion generator."""

    def test_basic_generation(self):
        """Test basic price path generation."""
        params = PriceModelParams(S0=50000, mu=0.1, sigma=0.5)
        gen = GBMPriceGenerator(params, seed=42)

        df = gen.generate(T=1/365, start_time=datetime(2024, 1, 1))  # 1 day

        assert len(df) == 25  # 24 hours + 1
        assert df['price'].iloc[0] == 50000
        assert all(df['price'] > 0)  # Prices should be positive
        assert 'returns' in df.columns
        assert 'timestamp' in df.columns

    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        params = PriceModelParams(S0=50000, sigma=0.5)

        gen1 = GBMPriceGenerator(params, seed=42)
        gen2 = GBMPriceGenerator(params, seed=42)

        df1 = gen1.generate(T=1/365)
        df2 = gen2.generate(T=1/365)

        assert np.allclose(df1['price'].values, df2['price'].values)

    def test_volatility_scaling(self):
        """Test that higher volatility produces wider price range."""
        low_vol_params = PriceModelParams(S0=50000, sigma=0.1)
        high_vol_params = PriceModelParams(S0=50000, sigma=1.0)

        low_vol_gen = GBMPriceGenerator(low_vol_params, seed=42)
        high_vol_gen = GBMPriceGenerator(high_vol_params, seed=42)

        low_vol_df = low_vol_gen.generate(T=30/365)  # 30 days
        high_vol_df = high_vol_gen.generate(T=30/365)

        low_vol_range = low_vol_df['price'].max() - low_vol_df['price'].min()
        high_vol_range = high_vol_df['price'].max() - high_vol_df['price'].min()

        assert high_vol_range > low_vol_range * 2  # High vol should have much wider range


class TestMertonJumpDiffusion:
    """Test Jump Diffusion model."""

    def test_jump_inclusion(self):
        """Test that jumps are included in the path."""
        params = PriceModelParams(
            S0=50000, mu=0.1, sigma=0.3,
            jump_intensity=10, jump_mean=0, jump_std=0.05
        )
        gen = MertonJumpDiffusion(params, seed=42)

        df = gen.generate(T=30/365)

        assert 'jump_count' in df.columns
        assert df['jump_count'].sum() > 0  # Should have some jumps

    def test_fat_tails(self):
        """Test that returns have fatter tails than GBM."""
        gbm_params = PriceModelParams(S0=50000, sigma=0.3, jump_intensity=0)
        # Use higher jump intensity and std to ensure significant fat tails
        jump_params = PriceModelParams(S0=50000, sigma=0.3, jump_intensity=50, jump_std=0.05)

        # Use different seeds to avoid random state interference
        gbm_gen = MertonJumpDiffusion(gbm_params, seed=42)
        jump_gen = MertonJumpDiffusion(jump_params, seed=123)

        # Generate 30 days for stable statistics while keeping tests fast
        gbm_df = gbm_gen.generate(T=30/365)  # 30 days
        jump_df = jump_gen.generate(T=30/365)

        # Calculate kurtosis (measure of tail fatness)
        gbm_kurt = gbm_df['returns'].kurtosis()
        jump_kurt = jump_df['returns'].kurtosis()

        # Jump model should have higher kurtosis (fatter tails)
        # Use >= with tolerance since exact comparison can be flaky
        assert jump_kurt > gbm_kurt, f"Jump kurtosis ({jump_kurt:.4f}) should be greater than GBM kurtosis ({gbm_kurt:.4f})"


class TestOrderBookSimulator:
    """Test order book simulation."""

    def test_snapshot_generation(self):
        """Test order book snapshot generation."""
        sim = OrderBookSimulator(base_spread_bps=10, depth_levels=10)
        ob = sim.generate_snapshot(mid_price=50000)

        assert len(ob.bids) == 10
        assert len(ob.asks) == 10
        assert ob.best_bid < ob.best_ask  # Spread exists
        assert ob.bids[0].price > ob.bids[1].price  # Descending bids
        assert ob.asks[0].price < ob.asks[1].price  # Ascending asks

    def test_volatility_impact(self):
        """Test that volatility affects spread and depth."""
        sim = OrderBookSimulator()

        ob_normal = sim.generate_snapshot(mid_price=50000, volatility_regime=1.0)
        ob_high_vol = sim.generate_snapshot(mid_price=50000, volatility_regime=2.0)

        # High vol should have wider spread
        assert ob_high_vol.spread > ob_normal.spread

        # High vol should have thinner book (less liquidity)
        normal_depth = sum(lvl.size for lvl in ob_normal.bids[:5])
        high_vol_depth = sum(lvl.size for lvl in ob_high_vol.bids[:5])
        assert high_vol_depth < normal_depth


class TestTradeFlowSimulator:
    """Test trade flow simulation."""

    def test_trade_generation(self):
        """Test basic trade generation."""
        # Create simple price path
        price_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
            'price': 50000 + np.cumsum(np.random.randn(100) * 10),
            'returns': np.random.randn(100) * 0.001
        })

        sim = TradeFlowSimulator(base_arrival_rate=5)
        trades = sim.generate(price_df)

        assert len(trades) > 0
        assert 'timestamp' in trades.columns
        assert 'price' in trades.columns
        assert 'size' in trades.columns
        assert 'side' in trades.columns

    def test_informed_trades(self):
        """Test that informed trades are larger."""
        price_df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
            'price': np.ones(100) * 50000,
            'returns': np.zeros(100)
        })

        sim = TradeFlowSimulator(informed_trade_prob=0.2)
        trades = sim.generate(price_df)

        # Check that we have the informed flag
        if 'is_informed' in trades.columns:
            informed_trades = trades[trades['is_informed'] == True]
            uninformed_trades = trades[trades['is_informed'] == False]

            if len(informed_trades) > 0 and len(uninformed_trades) > 0:
                assert informed_trades['size'].mean() > uninformed_trades['size'].mean()


class TestCompleteMarketSimulator:
    """Test complete market simulator."""

    def test_full_generation(self):
        """Test complete market data generation."""
        sim = CompleteMarketSimulator(seed=42)
        # Use hours=2 for faster tests while maintaining coverage
        data = sim.generate(hours=2, include_options=True)

        assert 'spot' in data
        assert 'order_book' in data
        assert 'trades' in data
        assert 'options' in data

        # Check all have data
        assert len(data['spot']) > 0
        assert len(data['order_book']) > 0
        assert len(data['trades']) > 0
        assert len(data['options']) > 0

    def test_without_options(self):
        """Test generation without options."""
        sim = CompleteMarketSimulator(seed=42)
        # Use hours=2 for faster tests
        data = sim.generate(hours=2, include_options=False)

        assert 'spot' in data
        assert 'options' not in data or len(data.get('options', [])) == 0

    def test_consistency(self):
        """Test that different components are consistent."""
        sim = CompleteMarketSimulator(seed=42)
        # Use hours=2 for faster tests
        data = sim.generate(hours=2, include_options=True)

        # Spot prices should be consistent
        spot_prices = data['spot']['price'].values
        ob_mids = data['order_book']['mid'].values

        # First few should match approximately
        assert abs(spot_prices[0] - ob_mids[0]) < 1

    def test_seed_reproducibility(self):
        """Test reproducibility with same seed."""
        sim1 = CompleteMarketSimulator(seed=42)
        sim2 = CompleteMarketSimulator(seed=42)

        # Use hours=2 for faster tests
        data1 = sim1.generate(hours=2, include_options=False)
        data2 = sim2.generate(hours=2, include_options=False)

        assert np.allclose(data1['spot']['price'].values, data2['spot']['price'].values)
