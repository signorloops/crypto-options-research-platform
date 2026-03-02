"""
Tests for PPO market making environment safeguards.
"""
import numpy as np
import pandas as pd

from strategies.market_making.ppo_agent import MarketMakingEnv


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


def test_market_making_env_clips_fill_probabilities(monkeypatch):
    """Fill probabilities should be capped below 1, preventing guaranteed fills."""
    env = MarketMakingEnv(_sample_market_data(), episode_length=40)
    env.episode_start = 0
    env.current_step = 12  # ensure recent window path is used

    monkeypatch.setattr(np.random, "random", lambda: 0.995)
    monkeypatch.setattr(np.random, "normal", lambda loc, scale: 0.0)

    _, _, done, info = env.step(np.array([5.0, 5.0, 1.0]))
    assert done is False
    assert info["fills"] == 0
