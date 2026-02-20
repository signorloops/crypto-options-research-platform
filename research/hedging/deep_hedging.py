"""
Deep-hedging style policy learner for option hedge ratios.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm

try:
    from sklearn.neural_network import MLPRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class DeepHedgingConfig:
    """Configuration for deep hedging policy training."""
    hidden_layer_sizes: Tuple[int, ...] = (32, 32)
    learning_rate_init: float = 1e-3
    max_iter: int = 300
    l2_penalty: float = 1e-5
    max_abs_hedge: float = 3.0
    transaction_cost_bps: float = 1.0
    seed: int = 42


class DeepHedgingPolicy:
    """
    Hedge policy approximator.

    Uses MLP when sklearn is available, otherwise ridge-regularized linear model.
    """

    def __init__(self, config: Optional[DeepHedgingConfig] = None):
        self.config = config or DeepHedgingConfig()
        self._mlp: Optional[MLPRegressor] = None
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_scale: Optional[np.ndarray] = None
        self._is_fitted: bool = False

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self._feature_mean is None or self._feature_scale is None:
            return X
        return (X - self._feature_mean) / self._feature_scale

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if y.ndim != 1:
            raise ValueError("y must be 1D")
        if len(X) != len(y):
            raise ValueError("X and y lengths mismatch")

        self._feature_mean = np.mean(X, axis=0)
        self._feature_scale = np.std(X, axis=0)
        self._feature_scale = np.where(self._feature_scale > 1e-12, self._feature_scale, 1.0)
        Xn = self._normalize(X)

        if HAS_SKLEARN:
            self._mlp = MLPRegressor(
                hidden_layer_sizes=self.config.hidden_layer_sizes,
                learning_rate_init=self.config.learning_rate_init,
                alpha=self.config.l2_penalty,
                max_iter=self.config.max_iter,
                random_state=self.config.seed,
            )
            self._mlp.fit(Xn, y)
            self._coef = None
            self._intercept = 0.0
            self._is_fitted = True
            return

        # Ridge fallback.
        n_features = Xn.shape[1]
        lam = self.config.l2_penalty
        X_aug = np.column_stack([np.ones(len(Xn)), Xn])
        reg = lam * np.eye(n_features + 1)
        reg[0, 0] = 0.0
        beta = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
        self._intercept = float(beta[0])
        self._coef = beta[1:]
        self._mlp = None
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Policy is not fitted")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xn = self._normalize(X)
        if self._mlp is not None:
            pred = self._mlp.predict(Xn)
        elif self._coef is not None:
            pred = self._intercept + Xn @ self._coef
        else:
            raise RuntimeError("Policy is not fitted")
        return np.clip(np.asarray(pred, dtype=float), -self.config.max_abs_hedge, self.config.max_abs_hedge)

    def diagnostics(self) -> Dict[str, float]:
        """Return training/inference diagnostics."""
        n_features = 0 if self._feature_mean is None else int(len(self._feature_mean))
        return {
            "is_fitted": float(self._is_fitted),
            "uses_mlp_backend": float(self._mlp is not None),
            "n_features": float(n_features),
        }


class DeepHedger:
    """
    End-to-end deep-hedging helper.

    Training target is one-step-ahead BS delta, a practical approximation that
    captures non-linear dependence on state while remaining stable in small samples.
    """

    def __init__(self, config: Optional[DeepHedgingConfig] = None):
        self.config = config or DeepHedgingConfig()
        self.policy = DeepHedgingPolicy(self.config)

    @staticmethod
    def _bs_delta(spot: np.ndarray, strike: float, tau: np.ndarray, vol: float, rate: float) -> np.ndarray:
        tau_safe = np.maximum(tau, 1e-8)
        d1 = (np.log(np.maximum(spot, 1e-12) / strike) + (rate + 0.5 * vol * vol) * tau_safe) / (vol * np.sqrt(tau_safe))
        return norm.cdf(d1)

    def build_training_dataset(
        self,
        spot_paths: np.ndarray,
        strike: float,
        maturity: float,
        rate: float,
        vol_proxy: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create supervised features/targets from simulated paths."""
        if spot_paths.ndim != 2:
            raise ValueError("spot_paths must be 2D")
        n_paths, n_cols = spot_paths.shape
        n_steps = n_cols - 1
        if n_steps <= 0:
            raise ValueError("spot_paths needs at least two time points")

        dt = maturity / n_steps
        X_rows = []
        y_rows = []
        for t in range(n_steps):
            tau = max(maturity - t * dt, 1e-8)
            tau_next = max(maturity - (t + 1) * dt, 1e-8)
            s_t = spot_paths[:, t]
            s_tp1 = spot_paths[:, t + 1]
            moneyness = np.log(np.maximum(s_t, 1e-12) / strike)
            state_delta = self._bs_delta(s_t, strike, np.full(n_paths, tau), vol_proxy, rate)
            features = np.column_stack(
                [
                    moneyness,
                    np.full(n_paths, tau),
                    state_delta,
                    np.sqrt(np.maximum(np.abs(np.log(np.maximum(s_tp1, 1e-12) / np.maximum(s_t, 1e-12))), 0.0)),
                ]
            )
            target = self._bs_delta(s_tp1, strike, np.full(n_paths, tau_next), vol_proxy, rate)
            X_rows.append(features)
            y_rows.append(target)

        X = np.vstack(X_rows)
        y = np.hstack(y_rows)
        return X, y

    def fit(
        self,
        spot_paths: np.ndarray,
        strike: float,
        maturity: float,
        rate: float = 0.0,
        vol_proxy: float = 0.2,
    ) -> None:
        """Train hedge policy from simulated path data."""
        X, y = self.build_training_dataset(spot_paths, strike, maturity, rate, vol_proxy)
        self.policy.fit(X, y)

    def evaluate_hedging_error(
        self,
        spot_paths: np.ndarray,
        strike: float,
        maturity: float,
        option_type: str = "call",
        rate: float = 0.0,
        vol_proxy: float = 0.2,
    ) -> Dict[str, float]:
        """
        Evaluate terminal replication error with transaction costs.
        """
        if option_type not in {"call", "put"}:
            raise ValueError("option_type must be call or put")
        if spot_paths.ndim != 2:
            raise ValueError("spot_paths must be 2D")
        n_paths, n_cols = spot_paths.shape
        n_steps = n_cols - 1
        dt = maturity / max(n_steps, 1)
        fee = self.config.transaction_cost_bps / 10_000.0

        hedge = np.zeros(n_paths, dtype=float)
        cash = np.zeros(n_paths, dtype=float)
        for t in range(n_steps):
            s_t = spot_paths[:, t]
            s_tp1 = spot_paths[:, t + 1]
            tau = max(maturity - t * dt, 1e-8)
            moneyness = np.log(np.maximum(s_t, 1e-12) / strike)
            realized_move = np.sqrt(np.maximum(np.abs(np.log(np.maximum(s_tp1, 1e-12) / np.maximum(s_t, 1e-12))), 0.0))
            state_delta = self._bs_delta(s_t, strike, np.full(n_paths, tau), vol_proxy, rate)
            X = np.column_stack([moneyness, np.full(n_paths, tau), state_delta, realized_move])
            target_hedge = self.policy.predict(X)

            trade = target_hedge - hedge
            cash -= trade * s_t
            cash -= np.abs(trade) * s_t * fee
            hedge = target_hedge

        terminal_spot = spot_paths[:, -1]
        cash += hedge * terminal_spot
        if option_type == "call":
            payoff = np.maximum(terminal_spot - strike, 0.0)
        else:
            payoff = np.maximum(strike - terminal_spot, 0.0)

        hedging_error = cash - payoff
        return {
            "mean_error": float(np.mean(hedging_error)),
            "std_error": float(np.std(hedging_error, ddof=1) if len(hedging_error) > 1 else 0.0),
            "mae_error": float(np.mean(np.abs(hedging_error))),
        }

    def diagnostics(self) -> Dict[str, float]:
        """Expose policy diagnostics at hedger level."""
        return self.policy.diagnostics()
