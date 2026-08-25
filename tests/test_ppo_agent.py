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


def test_market_making_env_reset_survives_short_market_data():
    """episode_length larger than the data frame must be clamped, not crash.

    Previously reset() indexed past the end of the frame (IndexError) when
    len(market_data) < episode_length + 100.
    """
    short_data = _sample_market_data(n=50)
    env = MarketMakingEnv(short_data, episode_length=1000, random_seed=1)

    assert env.episode_length == len(short_data) - 1

    state = env.reset()
    assert np.all(np.isfinite(state))

    # A full episode runs to completion without leaving the frame.
    done, steps = False, 0
    while not done:
        _, _, done, _ = env.step(np.array([5.0, 5.0, 1.0]))
        steps += 1
    assert steps == env.episode_length


def test_market_making_env_handles_constant_price_data():
    """Zero std (constant prices) must not produce inf/NaN states."""
    constant_data = pd.DataFrame(
        {"price": np.full(80, 50000.0), "volume": np.full(80, 20.0)}
    )
    env = MarketMakingEnv(constant_data, episode_length=40, random_seed=1)

    assert env.price_std > 0.0

    state = env.reset()
    assert np.all(np.isfinite(state))

    next_state, reward, done, _ = env.step(np.array([5.0, 5.0, 1.0]))
    assert np.all(np.isfinite(next_state))
    assert np.isfinite(reward)


def test_env_action_bounds_match_serving_time_clip_bounds():
    """Env sanitization and live-strategy clipping must share PPOConfig bounds.

    The policy previously trained on [1, 200] bps while serving clipped to
    [2.5, 50] bps, so it learned actions it could never express at quote time.
    """
    config = PPOConfig()
    env = MarketMakingEnv(_sample_market_data(), episode_length=40)

    assert env.min_offset_bps == config.min_spread_bps / 2
    assert env.max_offset_bps == config.max_spread_bps / 2
    assert env.min_size_scale == config.min_size_scale
    assert env.max_size_scale == config.max_size_scale

    from strategies.market_making.ppo_agent import _ppo_spread_offsets_and_size_scale

    extreme = np.array([-1e6, 1e6, 1e6])
    serving_bid, serving_ask, serving_size = _ppo_spread_offsets_and_size_scale(
        action=extreme, config=config
    )
    env_bid, env_ask, env_size = env._sanitize_action(extreme)

    assert (serving_bid, serving_ask, serving_size) == (env_bid, env_ask, env_size)
