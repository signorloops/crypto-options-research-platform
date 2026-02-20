"""
XGBoost-based spread prediction strategy.
Uses gradient boosting to predict optimal spread width based on market features.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from core.types import MarketState, Position, QuoteAction
from strategies.base import MarketMakingStrategy
from utils.logging_config import get_logger, log_extra

logger = get_logger(__name__)


@dataclass
class XGBoostSpreadConfig:
    """Configuration for XGBoost spread strategy."""
    # Model parameters
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1

    # Strategy parameters
    quote_size: float = 1.0
    inventory_limit: float = 10.0
    min_spread_bps: float = 5.0
    max_spread_bps: float = 100.0

    # Feature window
    feature_window: int = 20


class FeatureEngineer:
    """Extract features from market state for ML models."""

    def __init__(self, window: int = 20):
        self.window = window
        self._history: List[Dict] = []

    def update(self, state: MarketState) -> None:
        """Update history with new market state."""
        features = {
            'timestamp': state.timestamp,
            'mid_price': state.order_book.mid_price,
            'spread_bps': state.order_book.spread / state.order_book.mid_price * 10000 if state.order_book.mid_price else 0,
            'imbalance': state.order_book.imbalance(),
            'bid_volume_5': sum(lvl.size for lvl in state.order_book.bids[:5]),
            'ask_volume_5': sum(lvl.size for lvl in state.order_book.asks[:5]),
        }

        # Add any pre-computed features
        features.update(state.features)

        self._history.append(features)

        # Keep only recent history
        if len(self._history) > self.window * 2:
            self._history.pop(0)

    def get_features(self) -> Dict[str, float]:
        """Compute features from history."""
        if len(self._history) < 5:
            return self._default_features()

        df = pd.DataFrame(self._history)

        # Price-based features
        returns = df['mid_price'].pct_change().dropna()

        features = {
            # Current state
            'current_spread_bps': df['spread_bps'].iloc[-1],
            'current_imbalance': df['imbalance'].iloc[-1],
            'volume_ratio': df['bid_volume_5'].iloc[-1] / (df['ask_volume_5'].iloc[-1] + 1e-8),

            # Recent volatility
            'volatility_5': returns.tail(5).std() * np.sqrt(365 * 24) if len(returns) >= 5 else 0.5,
            'volatility_20': returns.tail(20).std() * np.sqrt(365 * 24) if len(returns) >= 20 else 0.5,

            # Trend
            'price_change_5': (df['mid_price'].iloc[-1] / df['mid_price'].iloc[-5] - 1) if len(df) >= 5 else 0,
            'price_change_10': (df['mid_price'].iloc[-1] / df['mid_price'].iloc[-10] - 1) if len(df) >= 10 else 0,

            # Order book dynamics
            'imbalance_change': df['imbalance'].iloc[-1] - df['imbalance'].iloc[-5] if len(df) >= 5 else 0,
            'spread_percentile': (df['spread_bps'].iloc[-1] - df['spread_bps'].min()) / (df['spread_bps'].max() - df['spread_bps'].min() + 1e-8),

            # Microstructure
            'volume_imbalance': (df['bid_volume_5'].iloc[-1] - df['ask_volume_5'].iloc[-1]) / (df['bid_volume_5'].iloc[-1] + df['ask_volume_5'].iloc[-1] + 1e-8),
        }

        return features

    def _default_features(self) -> Dict[str, float]:
        """Return default features when not enough history."""
        return {
            'current_spread_bps': 20.0,
            'current_imbalance': 0.0,
            'volume_ratio': 1.0,
            'volatility_5': 0.5,
            'volatility_20': 0.5,
            'price_change_5': 0.0,
            'price_change_10': 0.0,
            'imbalance_change': 0.0,
            'spread_percentile': 0.5,
            'volume_imbalance': 0.0,
        }

    def reset(self) -> None:
        """Clear history."""
        self._history = []


class XGBoostSpreadStrategy(MarketMakingStrategy):
    """
    Market making strategy using XGBoost to predict optimal spread.

    The model is trained to minimize realized cost (adverse selection + inventory holding).
    Features include volatility, order book imbalance, and recent price action.
    """

    def __init__(self, config: XGBoostSpreadConfig = None):
        self.config = config or XGBoostSpreadConfig()
        self.name = "XGBoostSpread"

        self.model: Optional[xgb.XGBRegressor] = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer(window=self.config.feature_window)
        self._feature_columns: List[str] = []

        # Track for training data generation
        self._training_mode = False
        self._training_samples: List[Dict] = []

    def quote(self, state: MarketState, position: Position) -> QuoteAction:
        """Generate quote based on XGBoost spread prediction."""
        mid = state.order_book.mid_price
        if mid is None:
            raise ValueError("Cannot quote without valid order book")

        # Update features
        self.feature_engineer.update(state)
        features = self.feature_engineer.get_features()

        # Predict optimal spread
        if self.model is not None:
            spread_bps = self._predict_spread(features)
        else:
            # Fallback to heuristic if not trained
            spread_bps = self._heuristic_spread(features)

        # Apply inventory skew (keep AS intuition)
        q = position.size
        gamma = 0.1  # Risk aversion
        sigma = features.get('volatility_20', 0.5)

        reservation_price = mid - q * gamma * sigma**2 * 0.1  # Small time horizon
        half_spread = mid * spread_bps / 10000 / 2

        bid_price = reservation_price - half_spread
        ask_price = reservation_price + half_spread

        # Inventory limits
        bid_size = self.config.quote_size if q < self.config.inventory_limit else 0
        ask_size = self.config.quote_size if q > -self.config.inventory_limit else 0

        return QuoteAction(
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
            metadata={
                "strategy": self.name,
                "predicted_spread_bps": spread_bps,
                "features": features,
                "inventory": q,
                "model_trained": self.model is not None
            }
        )

    def _predict_spread(self, features: Dict[str, float]) -> float:
        """Predict optimal spread directly using trained regressor."""
        if not self._feature_columns:
            self._feature_columns = list(features.keys())

        x_row = np.array([features.get(col, 0.0) for col in self._feature_columns], dtype=float).reshape(1, -1)
        x_scaled = self.scaler.transform(x_row)
        pred = float(self.model.predict(x_scaled)[0])
        return float(np.clip(pred, self.config.min_spread_bps, self.config.max_spread_bps))

    def _heuristic_spread(self, features: Dict[str, float]) -> float:
        """Heuristic spread when model not trained."""
        base_spread = 20.0
        vol_factor = features.get('volatility_20', 0.5) / 0.5  # Normalize to 0.5
        return np.clip(base_spread * vol_factor, self.config.min_spread_bps, self.config.max_spread_bps)

    def train(self, historical_data: pd.DataFrame) -> None:
        """
        Train XGBoost model on historical data.

        Training target: minimize realized cost per trade
        Cost = adverse_selection_loss + inventory_holding_cost
        """
        logger.info("Training XGBoost model", extra=log_extra(samples=len(historical_data)))

        # Generate training samples by simulation
        training_data = self._generate_training_data(historical_data)

        if len(training_data) < 100:
            logger.warning("Insufficient training data", extra=log_extra(samples=len(training_data)))
            return

        df = pd.DataFrame(training_data)

        # Features and target
        feature_cols = [c for c in df.columns if c not in ['target_spread_bps', 'timestamp']]
        X = df[feature_cols]
        y = df['target_spread_bps']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self._feature_columns = list(feature_cols)

        # Train model
        self.model = xgb.XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            objective='reg:squarederror'
        )

        self.model.fit(X_scaled, y)
        logger.info("Model trained", extra=log_extra(feature_importance=dict(zip(feature_cols, self.model.feature_importances_))))

    def _generate_training_data(self, historical_data: pd.DataFrame) -> List[Dict]:
        """Generate training samples with direct target = argmin_spread(cost)."""
        samples = []

        # Use walk-forward approach to avoid look-ahead bias
        # Only use past data that would be available at decision time
        for idx in range(self.config.feature_window + 20, len(historical_data) - 20):
            # Historical window for feature computation (past data only)
            feature_window = historical_data.iloc[idx - self.config.feature_window:idx]

            # Compute features from past data only
            features = self._compute_features_from_window(feature_window)

            # Outcome window (next 20 periods) - this simulates what happens after we quote
            # In real-time, we would wait for these periods to pass
            outcome_window = historical_data.iloc[idx:idx+20]

            spread_candidates = np.linspace(
                self.config.min_spread_bps,
                self.config.max_spread_bps,
                12
            )
            costs = np.array([
                self._simulate_cost(feature_window, spread, outcome_window)
                for spread in spread_candidates
            ])
            best_spread = float(spread_candidates[int(np.argmin(costs))])
            sample = {**features, 'target_spread_bps': best_spread, 'timestamp': feature_window.index[-1]}
            samples.append(sample)

        return samples

    def _compute_features_from_window(self, window: pd.DataFrame) -> Dict[str, float]:
        """Compute train-time features aligned with runtime FeatureEngineer schema."""
        returns = window['price'].pct_change().dropna()
        recent_spread_bps = float(window.get('spread_bps', pd.Series([20.0])).iloc[-1]) if len(window) > 0 else 20.0
        current_imbalance = float(window.get('imbalance', pd.Series([0.0])).iloc[-1]) if len(window) > 0 else 0.0

        bid_vol = float(window.get('bid_volume_5', pd.Series([1.0])).iloc[-1]) if len(window) > 0 else 1.0
        ask_vol = float(window.get('ask_volume_5', pd.Series([1.0])).iloc[-1]) if len(window) > 0 else 1.0

        spread_series = window.get('spread_bps', pd.Series(np.full(len(window), recent_spread_bps)))
        spread_min = float(spread_series.min()) if len(spread_series) > 0 else recent_spread_bps
        spread_max = float(spread_series.max()) if len(spread_series) > 0 else recent_spread_bps

        return {
            'current_spread_bps': recent_spread_bps,
            'current_imbalance': current_imbalance,
            'volume_ratio': bid_vol / (ask_vol + 1e-8),
            'volatility_5': returns.tail(5).std() * np.sqrt(365 * 24) if len(returns) >= 5 else 0.5,
            'volatility_20': returns.tail(20).std() * np.sqrt(365 * 24) if len(returns) >= 20 else 0.5,
            'price_change_5': (window['price'].iloc[-1] / window['price'].iloc[-5] - 1) if len(window) >= 5 else 0.0,
            'price_change_10': (window['price'].iloc[-1] / window['price'].iloc[-10] - 1) if len(window) >= 10 else 0.0,
            'imbalance_change': (
                float(window.get('imbalance', pd.Series([0.0] * len(window))).iloc[-1]) -
                float(window.get('imbalance', pd.Series([0.0] * len(window))).iloc[-5])
            ) if len(window) >= 5 else 0.0,
            'spread_percentile': (recent_spread_bps - spread_min) / (spread_max - spread_min + 1e-8),
            'volume_imbalance': (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8),
        }

    def _simulate_cost(self, historical_window: pd.DataFrame, spread: float, outcome_window: pd.DataFrame) -> float:
        """
        Simulate cost of quoting with given spread.

        Args:
            historical_window: Past data used for decision making (features)
            spread: Spread in basis points
            outcome_window: Future data simulating what happens after quote
                           In real-time this would be actual future periods
        """
        # Simplified: cost = adverse selection + missed opportunity
        mid = historical_window['price'].mean()
        half_spread = mid * spread / 10000 / 2

        # Adverse selection: if price moves against us after quote
        # This uses outcome_window which simulates the future after our quote
        # In training, we have this data; in inference, we predict the cost
        future_returns = outcome_window['price'].pct_change().dropna()
        adverse_move = abs(future_returns.mean()) * 100  # Basis points

        # Cost is higher if spread < adverse selection
        cost = max(0, adverse_move - half_spread)

        # Add penalty for very wide spreads (opportunity cost)
        if spread > 50:
            cost += (spread - 50) * 0.1

        return cost

    def get_internal_state(self) -> Dict:
        """Return model status."""
        return {
            "model_trained": self.model is not None,
            "feature_history_len": len(self.feature_engineer._history),
            "config": {
                "n_estimators": self.config.n_estimators,
                "min_spread_bps": self.config.min_spread_bps,
                "max_spread_bps": self.config.max_spread_bps,
            }
        }

    def reset(self) -> None:
        """Reset feature history."""
        self.feature_engineer.reset()
