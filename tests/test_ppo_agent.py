"""
Tests for PPO market making environment safeguards.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from strategies.market_making.ppo_agent import MarketMakingEnv, PPOConfig, PPOMarketMaker


def _sample_market_data(n: int = 160) -> pd.DataFrame:
    prices = np.linspace(50_000.0, 50_200.0, n)
    volumes = np.full(n, 20.0)
    return pd.DataFrame({"price": prices, "volume": volumes})


def test_market_making_env_sanitizes_extreme_actions():
    """Environment should clip invalid/extreme action values to safe bounds."""
    env = MarketMakingEnv(_sample_market_data(), episode_length=40)
    bid_offset, ask_offset, size_scale = env._sanitize_action(np.array([-100.0, 10_000.0, -3.0]))

    assert env.min_offset_bps <= bid_offset <= env.max_offset_bps
    assert env.min_offset_bps <= ask_offset <= env.max_offset_bps
    assert env.min_size_scale <= size_scale <= env.max_size_scale


def test_market_making_env_clips_fill_probabilities():
    """Fill probabilities should be capped below 1, preventing guaranteed fills."""
    env = MarketMakingEnv(_sample_market_data(), episode_length=40)
    env.episode_start = 0
    env.current_step = 12  # ensure recent window path is used

    class _StubRng:
        def random(self):
            return 0.995

        def normal(self, loc, scale):
            return 0.0

    env.rng = _StubRng()

    _, _, done, info = env.step(np.array([5.0, 5.0, 1.0]))
    assert done is False
    assert info["fills"] == 0


def test_market_making_env_uses_dynamic_state_features():
    """State vector should consume provided imbalance/depth/greeks features."""
    data = _sample_market_data().copy()
    data["spread_bps"] = np.linspace(8.0, 18.0, len(data))
    data["imbalance"] = np.linspace(-0.3, 0.3, len(data))
    data["bid_volume_5"] = np.linspace(20.0, 40.0, len(data))
    data["ask_volume_5"] = np.linspace(40.0, 20.0, len(data))
    data["delta"] = np.linspace(0.1, 0.2, len(data))
    data["vega"] = np.linspace(0.4, 0.6, len(data))

    env = MarketMakingEnv(data, episode_length=40, random_seed=7)
    env.episode_start = 0
    env.current_step = 20
    state = env._get_state()

    assert state[4] != 0.0  # imbalance
    assert state[12] != pytest.approx(0.5)  # bid volume norm
    assert state[13] != pytest.approx(0.5)  # ask volume norm
    assert state[20] != 0.0  # delta
    assert state[21] != 0.0  # vega


def test_ppo_train_uses_configured_seed_for_environment_reset():
    """Training should pass random_seed through to environment for deterministic episode start."""
    data = _sample_market_data(n=1500)
    config = PPOConfig(total_timesteps=0, random_seed=123, use_lstm=False)
    agent = PPOMarketMaker(config=config)

    agent.train(data)

    expected_start = int(np.random.default_rng(123).integers(0, len(data) - 1000 - 100))
    assert agent.env is not None
    assert agent.env.episode_start == expected_start
