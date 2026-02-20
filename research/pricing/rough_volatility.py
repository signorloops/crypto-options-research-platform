"""
Rough-volatility Monte Carlo pricer.
"""
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from scipy.stats import norm


@dataclass
class RoughVolConfig:
    """Configuration for rough-vol path simulation."""
    spot: float = 100.0
    rate: float = 0.0
    maturity: float = 1.0
    n_steps: int = 64
    n_paths: int = 2000
    hurst: float = 0.1
    vol_of_vol: float = 1.2
    initial_variance: float = 0.04
    correlation: float = -0.7
    seed: Optional[int] = 42
    antithetic_sampling: bool = True


class RoughVolatilityPricer:
    """
    Simplified rough Bergomi-style volatility model pricer.

    This implementation prioritizes numerical robustness and fast experimentation.
    """

    def __init__(self, config: Optional[RoughVolConfig] = None):
        config = config or RoughVolConfig()
        if config.spot <= 0:
            raise ValueError("spot must be positive")
        if config.maturity <= 0:
            raise ValueError("maturity must be positive")
        if config.n_steps <= 1:
            raise ValueError("n_steps must be > 1")
        if config.n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if not (0 < config.hurst < 0.5):
            raise ValueError("hurst must be in (0, 0.5)")
        if not (-0.999 <= config.correlation <= 0.999):
            raise ValueError("correlation must be in [-0.999, 0.999]")
        if config.initial_variance <= 0:
            raise ValueError("initial_variance must be positive")
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def _kernel_weights(self) -> np.ndarray:
        """Volterra kernel weights for rough process increments."""
        h = self.config.hurst
        dt = self.config.maturity / self.config.n_steps
        idx = np.arange(1, self.config.n_steps + 1, dtype=float)
        weights = (idx * dt) ** (h - 0.5)
        # Keep scale bounded for finite-step simulation.
        weights /= np.sqrt(np.sum(weights ** 2) + 1e-12)
        return weights

    def _draw_correlated_brownians(
        self,
        n_paths: int,
        n_steps: int,
        sqrt_dt: float,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate correlated Brownian increments with optional antithetic sampling."""
        cfg = self.config
        if cfg.antithetic_sampling and n_paths > 1:
            half = n_paths // 2
            z1_half = rng.normal(size=(half, n_steps))
            z2_half = rng.normal(size=(half, n_steps))
            z1 = np.vstack([z1_half, -z1_half])
            z2 = np.vstack([z2_half, -z2_half])
            if z1.shape[0] < n_paths:
                z1 = np.vstack([z1, rng.normal(size=(1, n_steps))])
                z2 = np.vstack([z2, rng.normal(size=(1, n_steps))])
            z1 = z1[:n_paths]
            z2 = z2[:n_paths]
        else:
            z1 = rng.normal(size=(n_paths, n_steps))
            z2 = rng.normal(size=(n_paths, n_steps))
        dW1 = z1 * sqrt_dt
        dW2 = (cfg.correlation * z1 + np.sqrt(1.0 - cfg.correlation ** 2) * z2) * sqrt_dt
        return dW1, dW2

    def simulate_paths(
        self,
        n_paths: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate spot and variance paths.

        Returns:
            (spots, variances) with shapes [(n_paths, n_steps+1), (n_paths, n_steps+1)]
        """
        cfg = self.config
        dt = cfg.maturity / cfg.n_steps
        sqrt_dt = np.sqrt(dt)
        n_paths = int(n_paths if n_paths is not None else cfg.n_paths)
        n_steps = cfg.n_steps

        gen = rng or self._rng
        dW1, dW2 = self._draw_correlated_brownians(
            n_paths=n_paths, n_steps=n_steps, sqrt_dt=sqrt_dt, rng=gen
        )

        weights = self._kernel_weights()
        volterra = np.zeros((n_paths, n_steps), dtype=float)
        for i in range(n_steps):
            segment = dW1[:, : i + 1]
            kernel = weights[: i + 1][::-1]
            volterra[:, i] = segment @ kernel

        times = np.arange(1, n_steps + 1, dtype=float) * dt
        var_t = cfg.initial_variance * np.exp(
            cfg.vol_of_vol * volterra - 0.5 * (cfg.vol_of_vol ** 2) * (times ** (2.0 * cfg.hurst))
        )
        var_t = np.maximum(var_t, 1e-10)

        spots = np.zeros((n_paths, n_steps + 1), dtype=float)
        vars_path = np.zeros((n_paths, n_steps + 1), dtype=float)
        spots[:, 0] = cfg.spot
        vars_path[:, 0] = cfg.initial_variance
        vars_path[:, 1:] = var_t

        drift = (cfg.rate - 0.5 * var_t) * dt
        diffusion = np.sqrt(var_t) * dW2
        log_increments = drift + diffusion
        spots[:, 1:] = cfg.spot * np.exp(np.cumsum(log_increments, axis=1))
        spots = np.maximum(spots, 1e-8)
        return spots, vars_path

    def price_european_option(
        self,
        strike: float,
        option_type: Literal["call", "put"] = "call",
    ) -> float:
        """Monte Carlo price of a European option under rough volatility."""
        if strike <= 0:
            raise ValueError("strike must be positive")
        if option_type not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")

        spots, _ = self.simulate_paths()
        terminal = spots[:, -1]
        if option_type == "call":
            payoff = np.maximum(terminal - strike, 0.0)
        else:
            payoff = np.maximum(strike - terminal, 0.0)
        discount = np.exp(-self.config.rate * self.config.maturity)
        return float(discount * np.mean(payoff))

    def price_with_confidence_interval(
        self,
        strike: float,
        option_type: Literal["call", "put"] = "call",
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """Price option and report Monte Carlo confidence interval."""
        if not (0.5 < confidence < 1.0):
            raise ValueError("confidence must be in (0.5, 1.0)")
        spots, _ = self.simulate_paths()
        terminal = spots[:, -1]
        if option_type == "call":
            payoff = np.maximum(terminal - strike, 0.0)
        elif option_type == "put":
            payoff = np.maximum(strike - terminal, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        discount = np.exp(-self.config.rate * self.config.maturity)
        discounted = discount * payoff
        price = float(np.mean(discounted))
        n = len(discounted)
        std = float(np.std(discounted, ddof=1) if n > 1 else 0.0)
        std_error = std / np.sqrt(max(n, 1))

        # Normal approximation CI.
        z = float(norm.ppf(0.5 + confidence / 2.0))
        ci_low = price - z * std_error
        ci_high = price + z * std_error
        return {
            "price": price,
            "std_error": float(std_error),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n_paths": float(n),
        }
