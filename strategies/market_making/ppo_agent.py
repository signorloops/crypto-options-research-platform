"""
PPO-based reinforcement learning market making strategy.
Uses Proximal Policy Optimization to learn optimal quoting policy.
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal

from core.types import MarketState, Position, QuoteAction
from strategies.base import MarketMakingStrategy
from utils.logging_config import get_logger, log_extra

logger = get_logger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO strategy."""
    # Network architecture
    hidden_dim: int = 256
    use_lstm: bool = True
    lstm_hidden_dim: int = 256
    sequence_length: int = 16

    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clip parameter
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # Training
    batch_size: int = 64
    epochs: int = 10
    total_timesteps: int = 1_000_000

    # Action bounds (spread in bps)
    min_spread_bps: float = 5.0
    max_spread_bps: float = 100.0
    max_skew_bps: float = 50.0  # Inventory skew

    quote_size: float = 1.0
    inventory_limit: float = 10.0


class MarketMakingActorCritic(nn.Module):
    """
    Actor-Critic network for market making.

    State: [mid_price_norm, inventory_norm, volatility, imbalance, ...]
    Action: [bid_offset, ask_offset, size_scale] - continuous
    Value: expected cumulative reward
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        use_lstm: bool = True,
        lstm_hidden_dim: int = 256
    ):
        super().__init__()
        self.use_lstm = use_lstm

        # Per-timestep feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        if self.use_lstm:
            self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
            head_dim = lstm_hidden_dim
        else:
            head_dim = hidden_dim

        # Actor (policy) head
        self.actor_mean = nn.Linear(head_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic (value) head
        self.critic = nn.Linear(head_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """Forward pass returning action distribution and value estimate."""
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(1)  # [1, 1, dim]
        elif state.dim() == 2:
            state = state.unsqueeze(1)  # [batch, 1, dim]
        # state: [batch, seq, dim]

        features = self.feature(state)
        if self.use_lstm:
            features, _ = self.lstm(features)

        features = features[:, -1, :]  # last step

        # Actor output
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std)
        dist = Normal(mean, std)

        # Critic output
        value = self.critic(features)

        return dist, value

    def get_action(self, state: torch.Tensor) -> Tuple[np.ndarray, float, float]:
        """Get action and value for a single state."""
        with torch.no_grad():
            dist, value = self.forward(state)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum()

        return action.squeeze(0).cpu().numpy(), value.squeeze().item(), log_prob.item()


class MarketMakingEnv:
    """
    Gym-like environment for market making RL training.

    State space:
    - Normalized mid price
    - Normalized inventory
    - Realized volatility
    - Order book imbalance
    - Recent return
    - Time to end of episode

    Action space (continuous):
    - bid_offset: how far below mid to place bid (bps)
    - ask_offset: how far above mid to place ask (bps)
    - size_scale: multiplier on base quote size

    Reward:
    - PnL from trades
    - Penalty for inventory deviation
    - Penalty for extreme spreads
    """

    def __init__(self, market_data: pd.DataFrame, episode_length: int = 1000):
        self.market_data = market_data.reset_index(drop=True)
        self.episode_length = episode_length
        self.current_step = 0
        self.episode_start = 0

        # State normalization parameters
        self.price_mean = market_data['price'].mean()
        self.price_std = market_data['price'].std()

        # Episode tracking
        self.position = 0.0
        self.cash = 0.0
        self.trades: List[Dict] = []

    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        self.episode_start = np.random.randint(0, len(self.market_data) - self.episode_length - 100)
        self.current_step = 0
        self.position = 0.0
        self.cash = 0.0
        self.trades = []

        return self._get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state, reward, done, info.

        Action: [bid_offset_bps, ask_offset_bps, size_scale]
        """
        bid_offset, ask_offset, size_scale = action

        # Convert offsets to actual prices
        current_data = self.market_data.iloc[self.episode_start + self.current_step]
        mid_price = current_data['price']

        bid_price = mid_price * (1 - bid_offset / 10000)
        ask_price = mid_price * (1 + ask_offset / 10000)
        quote_size = max(0.1, size_scale)

        # Simulate fills based on market data
        # Use historical trades or price movements to determine fill probability
        # without looking ahead to the next price
        next_step = self.current_step + 1
        done = next_step >= self.episode_length

        reward = 0.0
        info = {'fills': 0, 'pnl': 0.0}

        if not done:
            # Instead of using next_price (look-ahead bias),
            # we use historical patterns and current order book to simulate fills
            # This simulates what would happen if we quoted at current time

            # Get recent market conditions for fill simulation
            if self.current_step >= 10:
                recent_data = self.market_data.iloc[self.episode_start + self.current_step - 10:
                                                   self.episode_start + self.current_step]
                recent_volatility = recent_data['price'].pct_change().std() * np.sqrt(365 * 24)
                recent_volume = recent_data.get('volume', pd.Series([1.0])).mean()
            else:
                recent_volatility = 0.5
                recent_volume = 1.0

            # Simulate fills based on spread width and market conditions
            # Wider spreads = lower fill probability but better PnL when filled
            # Narrower spreads = higher fill probability but worse PnL when filled (adverse selection)

            # Fill probability decreases with spread width
            fill_prob_bid = max(0.05, min(0.6, 0.3 * (20 / max(bid_offset, 5))))
            fill_prob_ask = max(0.05, min(0.6, 0.3 * (20 / max(ask_offset, 5))))

            # Adjust for market conditions
            fill_prob_bid *= min(2.0, recent_volume)
            fill_prob_ask *= min(2.0, recent_volume)

            # Simulate bid fill (someone sells to our bid)
            if np.random.random() < fill_prob_bid:
                # Adverse selection: when we get filled on bid, price often moves down
                adverse_move = np.random.normal(-recent_volatility * 0.01, recent_volatility * 0.02)
                effective_price = mid_price * (1 + adverse_move)
                fill_pnl = (effective_price - bid_price) * quote_size
                self.cash -= bid_price * quote_size
                self.position += quote_size
                reward += fill_pnl
                info['fills'] += 1
                info['pnl'] += fill_pnl
                info['bid_fill'] = True

            # Simulate ask fill (someone buys from our ask)
            if np.random.random() < fill_prob_ask:
                # Adverse selection: when we get filled on ask, price often moves up
                adverse_move = np.random.normal(recent_volatility * 0.01, recent_volatility * 0.02)
                effective_price = mid_price * (1 + adverse_move)
                fill_pnl = (ask_price - effective_price) * quote_size
                self.cash += ask_price * quote_size
                self.position -= quote_size
                reward += fill_pnl
                info['fills'] += 1
                info['pnl'] += fill_pnl
                info['ask_fill'] = True

        # Inventory penalty (risk management)
        inventory_penalty = 0.01 * (self.position ** 2)
        reward -= inventory_penalty

        # Spread penalty (too wide = opportunity cost)
        spread = bid_offset + ask_offset
        if spread > 80:
            reward -= 0.001 * (spread - 80)

        self.current_step += 1
        next_state = self._get_state()

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """Compute current state vector."""
        idx = self.episode_start + self.current_step
        current = self.market_data.iloc[idx]

        # Normalized features
        price_norm = (current['price'] - self.price_mean) / self.price_std
        inventory_norm = self.position / 10.0  # Normalize to [-1, 1] roughly

        # Volatility (recent)
        if idx >= 20:
            recent = self.market_data.iloc[idx-20:idx]
            returns = recent['price'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(365 * 24)
        else:
            volatility = 0.5

        # Recent return
        if idx >= 5:
            recent_return = (current['price'] / self.market_data.iloc[idx-5]['price'] - 1) * 100
        else:
            recent_return = 0.0

        # Time to end (normalized)
        time_left = 1.0 - (self.current_step / self.episode_length)

        # Build richer state space (22 features)
        if idx >= 10:
            returns_10 = self.market_data.iloc[idx-10:idx]['price'].pct_change().dropna()
            ret_1 = returns_10.iloc[-1] if len(returns_10) > 0 else 0.0
            ret_5 = self.market_data.iloc[idx]['price'] / self.market_data.iloc[idx-5]['price'] - 1.0
            ret_10 = self.market_data.iloc[idx]['price'] / self.market_data.iloc[idx-10]['price'] - 1.0
            momentum = returns_10.mean() if len(returns_10) > 0 else 0.0
            realized_skew = returns_10.skew() if len(returns_10) > 2 else 0.0
            realized_kurt = returns_10.kurt() if len(returns_10) > 3 else 0.0
        else:
            ret_1 = ret_5 = ret_10 = momentum = realized_skew = realized_kurt = 0.0

        volume_series = self.market_data.get('volume', pd.Series(np.ones(len(self.market_data))))
        vol_now = float(volume_series.iloc[idx]) if idx < len(volume_series) else 1.0
        vol_win = volume_series.iloc[max(0, idx-20):idx+1]
        vol_mean = float(vol_win.mean()) if len(vol_win) > 0 else 1.0
        vol_std = float(vol_win.std()) if len(vol_win) > 1 else 1.0
        volume_z = (vol_now - vol_mean) / (vol_std + 1e-8)

        state = np.array([
            price_norm,                       # 1
            inventory_norm,                   # 2
            volatility,                       # 3
            float(volatility),                # 4 volatility_20 proxy
            0.0,                              # 5 imbalance placeholder
            20.0 / 100.0,                     # 6 spread_bps normalized
            float(ret_1),                     # 7
            float(ret_5),                     # 8
            float(ret_10),                    # 9
            float(momentum),                  # 10
            vol_now / (vol_mean + 1e-8),      # 11 volume ratio
            float(volume_z),                  # 12
            0.5,                              # 13 bid volume norm placeholder
            0.5,                              # 14 ask volume norm placeholder
            np.sign(self.position),           # 15
            abs(self.position) / 10.0,        # 16
            min(1.0, abs(self.position) / 10.0),  # 17 inventory utilization
            time_left,                        # 18
            float(realized_skew),             # 19
            float(realized_kurt),             # 20
            0.0,                              # 21 delta placeholder
            0.0,                              # 22 vega placeholder
        ], dtype=np.float32)

        return state


class PPOMarketMaker(MarketMakingStrategy):
    """
    Market making strategy using PPO (Proximal Policy Optimization).

    Learns to quote based on:
    - Current market conditions (volatility, trend)
    - Inventory position
    - Time remaining in episode
    """

    def __init__(self, config: PPOConfig = None):
        self.config = config or PPOConfig()
        self.name = "PPO-MarketMaker"

        # Network (initialized on first use)
        self.network: Optional[MarketMakingActorCritic] = None
        self.optimizer: Optional[torch.optim.Adam] = None

        # Training buffers
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []

        # Environment (set during training)
        self.env: Optional[MarketMakingEnv] = None

        # Running state for inference
        self.current_state: Optional[np.ndarray] = None
        self._state_sequence: deque = deque(maxlen=self.config.sequence_length)

    def _init_network(self, state_dim: int):
        """Initialize network if not already done."""
        if self.network is None:
            action_dim = 3  # [bid_offset, ask_offset, size_scale]
            self.network = MarketMakingActorCritic(
                state_dim,
                action_dim,
                self.config.hidden_dim,
                use_lstm=self.config.use_lstm,
                lstm_hidden_dim=self.config.lstm_hidden_dim
            )
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.config.learning_rate
            )

    def quote(self, state: MarketState, position: Position) -> QuoteAction:
        """Generate quote using PPO policy."""
        mid = state.order_book.mid_price
        if mid is None:
            raise ValueError("Cannot quote without valid order book")

        # Build state vector from market state
        state_vec = self._market_state_to_vector(state, position)
        self.current_state = state_vec
        self._state_sequence.append(state_vec)

        # Get action from policy
        if self.config.use_lstm:
            seq = list(self._state_sequence)
            if len(seq) < self.config.sequence_length:
                pad = [np.zeros_like(state_vec) for _ in range(self.config.sequence_length - len(seq))]
                seq = pad + seq
            state_tensor = torch.FloatTensor(np.array(seq, dtype=np.float32)).unsqueeze(0)
        else:
            state_tensor = torch.FloatTensor(state_vec)
        self._init_network(len(state_vec))

        action, value, log_prob = self.network.get_action(state_tensor)

        # Parse action
        bid_offset_bps = np.clip(action[0], self.config.min_spread_bps/2, self.config.max_spread_bps/2)
        ask_offset_bps = np.clip(action[1], self.config.min_spread_bps/2, self.config.max_spread_bps/2)
        size_scale = np.clip(action[2], 0.1, 2.0)

        # Add inventory skew
        inventory_skew = np.clip(
            -position.size * self.config.max_skew_bps / self.config.inventory_limit,
            -self.config.max_skew_bps,
            self.config.max_skew_bps
        )

        bid_price = mid * (1 - bid_offset_bps / 10000 + inventory_skew / 10000)
        ask_price = mid * (1 + ask_offset_bps / 10000 + inventory_skew / 10000)

        quote_size = self.config.quote_size * size_scale

        # Inventory limits
        bid_size = quote_size if position.size < self.config.inventory_limit else 0
        ask_size = quote_size if position.size > -self.config.inventory_limit else 0

        return QuoteAction(
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
            metadata={
                "strategy": self.name,
                "bid_offset_bps": bid_offset_bps,
                "ask_offset_bps": ask_offset_bps,
                "inventory_skew_bps": inventory_skew,
                "size_scale": size_scale,
                "value_estimate": value,
                "trained": self.network is not None and self.optimizer is not None
            }
        )

    def _market_state_to_vector(self, state: MarketState, position: Position) -> np.ndarray:
        """Convert MarketState to numpy vector for network."""
        mid = state.order_book.mid_price or 1.0

        # Log price normalization (asset-agnostic)
        # Uses log scale to handle different price magnitudes (BTC vs ETH vs altcoins)
        price_norm = np.log(mid) / 10.0 - 1.0 if mid > 0 else 0.0
        inventory_norm = position.size / self.config.inventory_limit

        # Extract or compute features
        volatility_5 = state.features.get('volatility_5', 0.5)
        volatility_20 = state.features.get('volatility_20', volatility_5)
        imbalance = state.order_book.imbalance() if state.order_book else 0.0
        spread_bps = (state.order_book.spread / mid * 10000) if state.order_book and state.order_book.spread and mid > 0 else 20.0
        ret_1 = state.features.get('return_1', 0.0)
        ret_5 = state.features.get('return_5', 0.0)
        ret_10 = state.features.get('return_10', 0.0)
        momentum = state.features.get('momentum', 0.0)
        volume_ratio = state.features.get('volume_ratio', 1.0)
        volume_z = state.features.get('volume_zscore', 0.0)

        bid_vol = sum(lvl.size for lvl in state.order_book.bids[:5]) if state.order_book else 1.0
        ask_vol = sum(lvl.size for lvl in state.order_book.asks[:5]) if state.order_book else 1.0
        bid_norm = bid_vol / (bid_vol + ask_vol + 1e-8)
        ask_norm = ask_vol / (bid_vol + ask_vol + 1e-8)

        inventory_abs = abs(position.size) / max(self.config.inventory_limit, 1e-8)
        inv_util = min(1.0, inventory_abs)
        time_left = state.features.get('time_left', 0.5)
        realized_skew = state.features.get('realized_skew', 0.0)
        realized_kurt = state.features.get('realized_kurt', 0.0)
        delta = state.greeks.delta if state.greeks is not None else 0.0
        vega = state.greeks.vega if state.greeks is not None else 0.0

        return np.array([
            price_norm,             # 1
            inventory_norm,         # 2
            volatility_5,           # 3
            volatility_20,          # 4
            imbalance,              # 5
            spread_bps / 100.0,     # 6
            ret_1,                  # 7
            ret_5,                  # 8
            ret_10,                 # 9
            momentum,               # 10
            volume_ratio,           # 11
            volume_z,               # 12
            bid_norm,               # 13
            ask_norm,               # 14
            np.sign(position.size), # 15
            inventory_abs,          # 16
            inv_util,               # 17
            time_left,              # 18
            realized_skew,          # 19
            realized_kurt,          # 20
            delta,                  # 21
            vega,                   # 22
        ], dtype=np.float32)

    def train(self, historical_data: pd.DataFrame) -> None:
        """Train PPO agent on historical data."""
        logger.info("Training PPO agent", extra=log_extra(samples=len(historical_data)))

        # Create environment
        self.env = MarketMakingEnv(historical_data, episode_length=1000)

        # Initialize network
        sample_state = self.env.reset()
        self._init_network(len(sample_state))

        # Training loop
        total_timesteps = self.config.total_timesteps
        timestep = 0
        episodes = 0

        while timestep < total_timesteps:
            state = self.env.reset()
            episode_reward = 0
            done = False

            # Collect trajectory
            while not done and timestep < total_timesteps:
                state_tensor = torch.FloatTensor(state)
                action, value, log_prob = self.network.get_action(state_tensor)

                next_state, reward, done, info = self.env.step(action)

                # Store transition
                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.values.append(value)
                self.log_probs.append(log_prob)
                self.dones.append(done)

                state = next_state
                episode_reward += reward
                timestep += 1

                # Update every batch_size steps
                if len(self.states) >= self.config.batch_size:
                    self._update()

            episodes += 1
            if episodes % 10 == 0:
                logger.info("Training progress", extra=log_extra(episode=episodes, timestep=timestep, reward=episode_reward))

        logger.info("Training complete", extra=log_extra(total_timesteps=timestep))

    def _update(self) -> None:
        """Perform PPO update on collected experiences."""
        if len(self.states) < self.config.batch_size:
            return

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(self.log_probs)
        rewards = torch.FloatTensor(self.rewards)
        values = torch.FloatTensor(self.values)
        dones = torch.FloatTensor(self.dones)

        # Compute advantages using GAE
        # Estimate bootstrap value for last state if episode continues
        last_state = self.states[-1] if self.states else None
        next_value = None
        if last_state is not None and self.dones[-1] == 0:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(last_state).unsqueeze(0)
                _, last_value = self.network(state_tensor)
                next_value = last_value.squeeze().item()

        advantages = self._compute_gae(rewards, values, dones, next_value)
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO epochs
        for _ in range(self.config.epochs):
            # Get current action distribution and values
            dist, current_values = self.network(states)
            current_log_probs = dist.log_prob(actions).sum(dim=-1)

            # Policy loss (PPO clip)
            ratio = torch.exp(current_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.MSELoss()(current_values.squeeze(), returns)

            # Entropy bonus
            entropy = dist.entropy().mean()

            # Total loss
            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, next_value: Optional[float] = None) -> torch.Tensor:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: Tensor of rewards
            values: Tensor of value estimates
            dones: Tensor of done flags (1 if episode ended, 0 otherwise)
            next_value: Bootstrap value for the state after the last experience.
                       If None and episode not done, will use 0 (underestimates advantage).
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        num_steps = len(rewards)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                # Last timestep: use bootstrap value if episode not done
                next_non_terminal = 1.0 - dones[t]
                # If next_value provided and episode continues, use it; otherwise 0
                next_val = next_value if next_value is not None else 0.0
                delta = rewards[t] + self.config.gamma * next_val * next_non_terminal - values[t]
            else:
                next_val = values[t + 1].item()
                next_non_terminal = 1.0 - dones[t]
                delta = rewards[t] + self.config.gamma * next_val * next_non_terminal - values[t]

            advantages[t] = last_advantage = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_advantage

        return advantages

    def get_internal_state(self) -> Dict:
        """Return training status."""
        return {
            "network_initialized": self.network is not None,
            "buffer_size": len(self.states),
            "config": {
                "hidden_dim": self.config.hidden_dim,
                "learning_rate": self.config.learning_rate,
                "gamma": self.config.gamma,
            }
        }

    def reset(self) -> None:
        """Reset agent state."""
        self.current_state = None
        self._state_sequence.clear()
