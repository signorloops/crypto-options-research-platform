"""
Almgren-Chriss optimal execution model.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class AlmgrenChrissConfig:
    """Configuration of a single parent order execution problem."""
    total_quantity: float
    horizon: float = 1.0
    n_steps: int = 20
    temporary_impact_eta: float = 0.1
    permanent_impact_gamma: float = 0.01
    volatility: float = 0.2
    risk_aversion_lambda: float = 1e-6
    initial_price: float = 1.0
    max_participation_rate: Optional[float] = None
    expected_step_market_volume: Optional[np.ndarray] = None


@dataclass
class ExecutionReport:
    """Expected and simulated execution metrics."""
    inventory_path: np.ndarray
    trading_schedule: np.ndarray
    expected_cost: float
    expected_variance: float
    objective: float
    simulated_mean_cost: float
    simulated_cost_std: float


class AlmgrenChrissExecutor:
    """
    Optimal execution under linear temporary/permanent market impact.

    References:
    - Almgren, R. and Chriss, N. (2000), Optimal Execution of Portfolio Transactions
    """

    def __init__(self, config: AlmgrenChrissConfig):
        if config.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if config.horizon <= 0:
            raise ValueError("horizon must be positive")
        if config.temporary_impact_eta <= 0:
            raise ValueError("temporary_impact_eta must be positive")
        if config.volatility < 0:
            raise ValueError("volatility must be non-negative")
        if config.total_quantity < 0:
            raise ValueError("total_quantity must be non-negative")
        if config.max_participation_rate is not None and not (0 < config.max_participation_rate <= 1.0):
            raise ValueError("max_participation_rate must be in (0, 1]")
        self.config = config
        self._tau = float(config.horizon / config.n_steps)
        if config.expected_step_market_volume is not None:
            volume = np.asarray(config.expected_step_market_volume, dtype=float)
            if volume.ndim != 1 or len(volume) != config.n_steps:
                raise ValueError("expected_step_market_volume must be 1D with length n_steps")
            if np.any(volume <= 0):
                raise ValueError("expected_step_market_volume must be strictly positive")

    def _kappa(self) -> float:
        """Discrete-time decay parameter."""
        eta = self.config.temporary_impact_eta
        lam = max(self.config.risk_aversion_lambda, 0.0)
        sigma = self.config.volatility
        if lam == 0.0 or sigma == 0.0:
            return 0.0

        inside = 1.0 + 0.5 * lam * sigma * sigma * self._tau * self._tau / eta
        inside = max(inside, 1.0)
        return float(np.arccosh(inside))

    def optimal_inventory_path(self) -> np.ndarray:
        """
        Optimal inventory path x_k (k=0..N), with x_0=X and x_N=0.
        """
        n = self.config.n_steps
        x0 = self.config.total_quantity
        if x0 == 0:
            return np.zeros(n + 1, dtype=float)

        kappa = self._kappa()
        if abs(kappa) < 1e-12:
            # Risk-neutral: linear schedule.
            return np.linspace(x0, 0.0, n + 1, dtype=float)

        idx = np.arange(0, n + 1, dtype=float)
        numer = np.sinh(kappa * (n - idx))
        denom = np.sinh(kappa * n)
        path = x0 * numer / max(denom, 1e-12)
        path[-1] = 0.0
        return path

    def optimal_trading_schedule(self, enforce_participation: bool = False) -> np.ndarray:
        """
        Child order sizes u_k (k=1..N), where sum(u_k)=total_quantity.
        """
        inventory = self.optimal_inventory_path()
        schedule = inventory[:-1] - inventory[1:]
        schedule = np.maximum(schedule, 0.0)
        total = float(schedule.sum())
        if total > 0:
            schedule *= self.config.total_quantity / total
        if enforce_participation:
            schedule = self.enforce_participation_cap(schedule)
        return schedule

    def expected_shortfall_cost(self, schedule: Optional[np.ndarray] = None) -> float:
        """
        Expected implementation shortfall under AC linear impact.
        """
        if schedule is None:
            schedule = self.optimal_trading_schedule()
        x = self._inventory_from_schedule(schedule)

        temp = self.config.temporary_impact_eta * np.sum((schedule / self._tau) ** 2) * self._tau
        perm = 0.5 * self.config.permanent_impact_gamma * (self.config.total_quantity ** 2)
        risk_term = 0.0 * np.sum(x[1:] ** 2)  # kept for decomposition symmetry
        return float(temp + perm + risk_term)

    def execution_variance(self, inventory_path: Optional[np.ndarray] = None) -> float:
        """Variance of execution cost induced by price uncertainty."""
        if inventory_path is None:
            inventory_path = self.optimal_inventory_path()
        sigma = self.config.volatility
        return float((sigma ** 2) * self._tau * np.sum(inventory_path[1:] ** 2))

    def objective_value(self, schedule: Optional[np.ndarray] = None) -> float:
        """Mean-variance objective E[cost] + lambda * Var[cost]."""
        if schedule is not None:
            variance = self.execution_variance(inventory_path=self._inventory_from_schedule(schedule))
        else:
            variance = self.execution_variance()
        expected = self.expected_shortfall_cost(schedule=schedule)
        return float(expected + self.config.risk_aversion_lambda * variance)

    def cost_decomposition(self, schedule: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Break down expected objective into temporary/permanent/risk components."""
        if schedule is None:
            schedule = self.optimal_trading_schedule()
        x = self._inventory_from_schedule(schedule)
        temp = self.config.temporary_impact_eta * np.sum((schedule / self._tau) ** 2) * self._tau
        perm = 0.5 * self.config.permanent_impact_gamma * (self.config.total_quantity ** 2)
        variance = self.execution_variance(inventory_path=x)
        risk_penalty = self.config.risk_aversion_lambda * variance
        total = temp + perm + risk_penalty
        return {
            "temporary_impact_cost": float(temp),
            "permanent_impact_cost": float(perm),
            "risk_penalty": float(risk_penalty),
            "total_objective": float(total),
        }

    def enforce_participation_cap(self, schedule: np.ndarray) -> np.ndarray:
        """
        Enforce max participation per step and redistribute residual across free capacity.
        """
        rate = self.config.max_participation_rate
        market_volume = self.config.expected_step_market_volume
        if rate is None or market_volume is None:
            return np.asarray(schedule, dtype=float)

        schedule = np.asarray(schedule, dtype=float)
        if len(schedule) != self.config.n_steps:
            raise ValueError("schedule length must be n_steps")
        cap = rate * np.asarray(market_volume, dtype=float)
        clipped = np.minimum(np.maximum(schedule, 0.0), cap)
        residual = float(np.sum(schedule) - np.sum(clipped))
        if residual <= 1e-12:
            return clipped

        slack = cap - clipped
        slack_total = float(np.sum(slack))
        if slack_total + 1e-12 < residual:
            raise ValueError("Participation cap too tight: cannot execute parent order within horizon")
        if slack_total > 1e-12:
            clipped += residual * (slack / slack_total)

        # Final normalization for numerical drift.
        target = float(np.sum(schedule))
        total = float(np.sum(clipped))
        if total > 0:
            clipped *= target / total
        return clipped

    def _inventory_from_schedule(self, schedule: np.ndarray) -> np.ndarray:
        """Convert schedule to inventory path."""
        schedule = np.asarray(schedule, dtype=float)
        if len(schedule) != self.config.n_steps:
            raise ValueError("schedule length must equal n_steps")
        inv = np.zeros(self.config.n_steps + 1, dtype=float)
        inv[0] = self.config.total_quantity
        inv[1:] = self.config.total_quantity - np.cumsum(schedule)
        return inv

    def simulate_execution_costs(
        self,
        n_paths: int = 1000,
        rng: Optional[np.random.Generator] = None,
        schedule: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Monte Carlo simulation of realized execution costs for the optimal schedule.
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        gen = rng or np.random.default_rng(42)

        if schedule is None:
            schedule = self.optimal_trading_schedule()
        else:
            schedule = np.asarray(schedule, dtype=float)
            if len(schedule) != self.config.n_steps:
                raise ValueError("schedule length must equal n_steps")
        costs = np.zeros(n_paths, dtype=float)
        sigma = self.config.volatility
        eta = self.config.temporary_impact_eta
        gamma = self.config.permanent_impact_gamma
        s0 = self.config.initial_price
        tau = self._tau

        for i in range(n_paths):
            mid = s0
            cost = 0.0
            for u in schedule:
                noise = sigma * np.sqrt(tau) * float(gen.normal())
                # Execution price penalized by temporary impact.
                exec_price = mid - eta * (u / tau)
                cost += u * (s0 - exec_price)
                # Mid-price evolves with noise and permanent impact.
                mid = mid + noise - gamma * u
            costs[i] = cost
        return costs

    def build_report(
        self,
        n_paths: int = 1000,
        rng: Optional[np.random.Generator] = None,
        enforce_participation: bool = False,
    ) -> ExecutionReport:
        """Create a full expected/simulated execution report."""
        inventory = self.optimal_inventory_path()
        schedule = self.optimal_trading_schedule(enforce_participation=enforce_participation)
        if enforce_participation and self.config.max_participation_rate is not None:
            inventory = self._inventory_from_schedule(schedule)
        expected = self.expected_shortfall_cost(schedule=schedule)
        variance = self.execution_variance(inventory_path=inventory)
        objective = expected + self.config.risk_aversion_lambda * variance
        simulated_costs = self.simulate_execution_costs(n_paths=n_paths, rng=rng, schedule=schedule)

        return ExecutionReport(
            inventory_path=inventory,
            trading_schedule=schedule,
            expected_cost=float(expected),
            expected_variance=float(variance),
            objective=float(objective),
            simulated_mean_cost=float(np.mean(simulated_costs)),
            simulated_cost_std=float(np.std(simulated_costs, ddof=1) if len(simulated_costs) > 1 else 0.0),
        )
