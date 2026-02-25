"""
Value at Risk (VaR) and Expected Shortfall (CVaR) calculations.
Includes parametric, historical, and Monte Carlo methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class VaRResult:
    """Result of VaR calculation."""

    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # Expected shortfall at 95%
    cvar_99: float  # Expected shortfall at 99%
    method: str  # Calculation method used

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""
        return {
            "var_95": self.var_95,
            "var_99": self.var_99,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
            "method": self.method,
        }


class VaRCalculator:
    """
    Calculate Value at Risk for portfolios.

    Supports three methods:
    1. Parametric (Variance-Covariance): Assumes normal distribution
    2. Historical: Uses empirical distribution of historical returns
    3. Monte Carlo: Simulates future paths
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def _prepare_portfolio_inputs(
        self, positions: pd.DataFrame, returns: pd.DataFrame
    ) -> tuple[float, np.ndarray, pd.DataFrame]:
        """Validate inputs and align returns columns to positions index."""
        if "value" not in positions.columns:
            raise ValueError("positions must contain 'value' column")
        if positions.empty:
            raise ValueError("positions must not be empty")
        if returns.empty:
            raise ValueError("returns must not be empty")
        if positions.index.has_duplicates:
            raise ValueError("positions index must be unique")

        position_values = positions["value"].astype(float)
        if not np.all(np.isfinite(position_values.values)):
            raise ValueError("positions values must be finite")

        # Use gross exposure as risk scaling base so long/short books are supported.
        total_value = float(np.abs(position_values.values).sum())
        if abs(total_value) <= 1e-12:
            raise ValueError("total portfolio value must be non-zero")

        assets = list(positions.index)
        missing_assets = [asset for asset in assets if asset not in returns.columns]
        if missing_assets:
            missing_str = ", ".join(map(str, missing_assets))
            raise ValueError(f"missing return series for assets: {missing_str}")

        aligned_returns = returns.loc[:, assets]
        if not np.isfinite(aligned_returns.to_numpy()).all():
            raise ValueError("returns must be finite")

        weights = position_values.values / total_value
        return total_value, weights, aligned_returns

    @staticmethod
    def _portfolio_return_series(aligned_returns: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
        """Compute portfolio return series from aligned asset returns and weights."""
        return aligned_returns.to_numpy() @ weights

    @staticmethod
    def _normalize_option_type(raw_option_type: object) -> str | None:
        """Normalize option type representation to {'call', 'put'}."""
        if raw_option_type is None:
            return None
        text = str(raw_option_type).strip().lower()
        if text in {"call", "c"}:
            return "call"
        if text in {"put", "p"}:
            return "put"
        return None

    def parametric_var(
        self, positions: pd.DataFrame, returns: pd.DataFrame, holding_period: int = 1
    ) -> VaRResult:
        """
        Calculate parametric VaR assuming normal distribution.

        Args:
            positions: DataFrame with 'value' column for each position
            returns: DataFrame of historical returns for each position
            holding_period: Days to hold positions

        Returns:
            VaRResult with VaR and CVaR metrics
        """
        if holding_period <= 0:
            raise ValueError("holding_period must be positive")

        total_value, weights_arr, aligned_returns = self._prepare_portfolio_inputs(
            positions, returns
        )

        # Portfolio mean and variance
        returns_mean_arr = aligned_returns.mean().values
        mean_return = returns_mean_arr @ weights_arr
        cov_matrix = aligned_returns.cov()
        portfolio_var = weights_arr @ cov_matrix.values @ weights_arr
        portfolio_std = np.sqrt(portfolio_var)

        # Scale by holding period
        adjusted_std = portfolio_std * np.sqrt(holding_period)

        # VaR calculations
        # VaR_alpha = -mu + z_alpha * sigma (positive value represents risk)
        z_95 = stats.norm.ppf(0.95)
        z_99 = stats.norm.ppf(0.99)

        var_95 = total_value * (-mean_return * holding_period + z_95 * adjusted_std)
        var_99 = total_value * (-mean_return * holding_period + z_99 * adjusted_std)

        # CVaR (Expected Shortfall) for normal distribution
        # CVaR_alpha = -mu + sigma * phi(z_alpha) / (1 - alpha)
        # where phi is the standard normal PDF
        phi_95 = stats.norm.pdf(z_95)
        phi_99 = stats.norm.pdf(z_99)

        cvar_95 = total_value * (-mean_return * holding_period + adjusted_std * phi_95 / 0.05)
        cvar_99 = total_value * (-mean_return * holding_period + adjusted_std * phi_99 / 0.01)

        return VaRResult(
            var_95=var_95, var_99=var_99, cvar_95=cvar_95, cvar_99=cvar_99, method="parametric"
        )

    def cornish_fisher_var(
        self, positions: pd.DataFrame, returns: pd.DataFrame, holding_period: int = 1
    ) -> VaRResult:
        """
        Cornish-Fisher VaR: 在正态 VaR 基础上加入偏度和峰度修正。
        """
        if holding_period <= 0:
            raise ValueError("holding_period must be positive")

        total_value, weights_arr, aligned_returns = self._prepare_portfolio_inputs(
            positions, returns
        )
        portfolio_returns = self._portfolio_return_series(aligned_returns, weights_arr)
        if len(portfolio_returns) < 10:
            return self.parametric_var(positions, returns, holding_period)

        mu = np.mean(portfolio_returns) * holding_period
        sigma = np.std(portfolio_returns, ddof=1) * np.sqrt(holding_period)
        if sigma <= 1e-12:
            return VaRResult(0.0, 0.0, 0.0, 0.0, method="cornish_fisher")

        skew = stats.skew(portfolio_returns, bias=False)
        kurt = stats.kurtosis(portfolio_returns, fisher=False, bias=False)

        def cf_quantile(alpha: float) -> float:
            z = stats.norm.ppf(alpha)
            z_cf = (
                z
                + (z**2 - 1) * skew / 6.0
                + (z**3 - 3 * z) * (kurt - 3.0) / 24.0
                - (2 * z**3 - 5 * z) * (skew**2) / 36.0
            )
            return z_cf

        z95 = cf_quantile(0.95)
        z99 = cf_quantile(0.99)

        var_95 = total_value * max(0.0, -mu + z95 * sigma)
        var_99 = total_value * max(0.0, -mu + z99 * sigma)

        # 近似 CVaR：采用修正分位数对应的尾部正态近似
        cvar_95 = total_value * max(0.0, -mu + sigma * stats.norm.pdf(z95) / 0.05)
        cvar_99 = total_value * max(0.0, -mu + sigma * stats.norm.pdf(z99) / 0.01)

        return VaRResult(
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            cvar_99=float(cvar_99),
            method="cornish_fisher",
        )

    def historical_var(
        self, positions: pd.DataFrame, returns: pd.DataFrame, holding_period: int = 1
    ) -> VaRResult:
        """
        Calculate historical VaR using empirical distribution.

        Args:
            positions: DataFrame with position values
            returns: DataFrame of historical returns
            holding_period: Days to hold positions

        Returns:
            VaRResult
        """
        if holding_period <= 0:
            raise ValueError("holding_period must be positive")

        total_value, weights_arr, aligned_returns = self._prepare_portfolio_inputs(
            positions, returns
        )

        # Calculate historical portfolio returns
        portfolio_returns = pd.Series(
            aligned_returns.to_numpy() @ weights_arr, index=aligned_returns.index
        )

        # Scale by holding period using non-overlapping windows to maintain independence
        if holding_period > 1:
            n_periods = len(portfolio_returns) // holding_period
            if n_periods > 0:
                # Reshape into non-overlapping periods
                reshaped = portfolio_returns.iloc[: n_periods * holding_period].values.reshape(
                    n_periods, holding_period
                )
                portfolio_returns = pd.Series(reshaped.sum(axis=1))
            else:
                # Not enough data, use simple scaling
                portfolio_returns = portfolio_returns * np.sqrt(holding_period)

        # VaR from empirical quantiles
        var_95 = -np.percentile(portfolio_returns, 5) * total_value
        var_99 = -np.percentile(portfolio_returns, 1) * total_value

        # CVaR (average of returns beyond VaR threshold)
        cvar_95 = (
            -portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()
            * total_value
        )
        cvar_99 = (
            -portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 1)].mean()
            * total_value
        )

        return VaRResult(
            var_95=var_95, var_99=var_99, cvar_95=cvar_95, cvar_99=cvar_99, method="historical"
        )

    def filtered_historical_var(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        holding_period: int = 1,
        lambda_param: float = 0.94,
    ) -> VaRResult:
        """
        Filtered Historical Simulation (FHS):
        1) EWMA 估计条件波动率
        2) 标准化残差重采样
        3) 用最新波动率重构未来损益分布
        """
        if holding_period <= 0:
            raise ValueError("holding_period must be positive")

        total_value, weights_arr, aligned_returns = self._prepare_portfolio_inputs(
            positions, returns
        )
        portfolio_returns = self._portfolio_return_series(aligned_returns, weights_arr)
        if len(portfolio_returns) < 30:
            return self.historical_var(positions, returns, holding_period)

        eps = 1e-12
        ewma_var = np.zeros_like(portfolio_returns, dtype=float)
        ewma_var[0] = np.var(portfolio_returns)
        for i in range(1, len(portfolio_returns)):
            ewma_var[i] = (
                lambda_param * ewma_var[i - 1] + (1 - lambda_param) * portfolio_returns[i - 1] ** 2
            )

        cond_vol = np.sqrt(np.maximum(ewma_var, eps))
        standardized = portfolio_returns / cond_vol
        latest_vol = cond_vol[-1]

        rng = np.random.default_rng(42)
        n_sim = max(5000, len(portfolio_returns) * 20)
        sampled_resid = rng.choice(standardized, size=n_sim, replace=True)
        simulated_returns = sampled_resid * latest_vol * np.sqrt(holding_period)
        pnl = simulated_returns * total_value

        var_95 = -np.percentile(pnl, 5)
        var_99 = -np.percentile(pnl, 1)
        cvar_95 = -pnl[pnl <= np.percentile(pnl, 5)].mean()
        cvar_99 = -pnl[pnl <= np.percentile(pnl, 1)].mean()

        return VaRResult(
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            cvar_99=float(cvar_99),
            method="fhs",
        )

    def evt_var(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        holding_period: int = 1,
        threshold_quantile: float = 0.9,
    ) -> VaRResult:
        """
        Peaks-over-threshold EVT VaR (GPD tail fit on losses).
        """
        if holding_period <= 0:
            raise ValueError("holding_period must be positive")

        total_value, weights_arr, aligned_returns = self._prepare_portfolio_inputs(
            positions, returns
        )
        portfolio_returns = self._portfolio_return_series(aligned_returns, weights_arr) * np.sqrt(
            holding_period
        )
        losses = -portfolio_returns
        if len(losses) < 50:
            return self.historical_var(positions, returns, holding_period)

        threshold = float(np.quantile(losses, threshold_quantile))
        tail_losses = losses[losses > threshold]
        if len(tail_losses) < 20:
            return self.historical_var(positions, returns, holding_period)

        excess = tail_losses - threshold
        shape, loc, scale = stats.genpareto.fit(excess, floc=0.0)
        pu = len(tail_losses) / len(losses)

        def evt_quantile(alpha: float) -> float:
            tail_prob = (1 - alpha) / max(pu, 1e-12)
            tail_prob = max(tail_prob, 1e-12)
            if abs(shape) < 1e-8:
                q = threshold + scale * np.log(1.0 / tail_prob)
            else:
                q = threshold + scale / shape * (tail_prob ** (-shape) - 1.0)
            return float(max(q, 0.0))

        q95 = evt_quantile(0.95)
        q99 = evt_quantile(0.99)
        var_95 = q95 * total_value
        var_99 = q99 * total_value

        # GPD ES formula when shape < 1
        def evt_es(q_alpha: float, alpha: float) -> float:
            if shape >= 1:
                return float(q_alpha * total_value)
            es_loss = q_alpha + (scale - shape * threshold) / (1 - shape)
            return float(max(es_loss, q_alpha) * total_value)

        cvar_95 = evt_es(q95, 0.95)
        cvar_99 = evt_es(q99, 0.99)

        return VaRResult(
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            cvar_99=float(cvar_99),
            method="evt",
        )

    @staticmethod
    def _regularize_covariance(cov: np.ndarray) -> np.ndarray:
        """Ensure covariance matrix is positive definite for simulation stability."""
        cov_matrix = np.array(cov, dtype=float, copy=True)
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        if np.any(eigenvalues <= 1e-10):
            min_eigenvalue = float(np.min(eigenvalues))
            regularization = max(1e-6, -min_eigenvalue + 1e-6)
            cov_matrix += np.eye(len(cov_matrix)) * regularization
        return cov_matrix

    @staticmethod
    def _simulate_correlated_returns(
        mean: np.ndarray,
        cov: np.ndarray,
        weights: np.ndarray,
        n_simulations: int,
        holding_period: int,
        leverage_correlation: float,
        rng: Any,
    ) -> np.ndarray:
        """Generate correlated returns with optional leverage-effect volatility scaling."""
        simulated_returns = rng.multivariate_normal(
            mean * holding_period, cov * holding_period, n_simulations
        )
        if abs(leverage_correlation) <= 1e-8:
            return simulated_returns

        portfolio_shock = simulated_returns @ weights
        shock_std = np.std(portfolio_shock) + 1e-12
        normalized_shock = portfolio_shock / shock_std
        vol_multiplier = np.exp(-leverage_correlation * normalized_shock)
        vol_multiplier = np.clip(vol_multiplier, 0.5, 3.0)
        return simulated_returns * vol_multiplier[:, None]

    @staticmethod
    def _greeks_component_pnl(
        greek_row: pd.Series,
        position_value: float,
        shocks: np.ndarray,
        n_simulations: int,
        rng: Any,
    ) -> np.ndarray:
        """Delta-gamma-vega PnL approximation component for one position."""
        delta_pnl = greek_row["delta"] * shocks * position_value
        gamma_pnl = 0.5 * greek_row.get("gamma", 0) * (shocks**2) * position_value
        vega_pnl = greek_row.get("vega", 0) * rng.normal(0, 0.05, n_simulations) * position_value
        return delta_pnl + gamma_pnl + vega_pnl

    def _single_position_full_revaluation_pnl(
        self,
        idx: object,
        row: pd.Series,
        default_asset_idx: int,
        simulated_returns: np.ndarray,
        column_index: dict[object, int],
        greeks: pd.DataFrame | None,
        n_simulations: int,
        holding_period: int,
        leverage_correlation: float,
        rng: Any,
    ) -> np.ndarray:
        """
        Compute simulated PnL component for one position under option-aware revaluation path.

        Falls back to Greeks approximation or linear approximation when option metadata
        is invalid/incomplete.
        """
        position_value = float(row["value"])
        linear_component = simulated_returns[:, default_asset_idx] * position_value

        option_type = self._normalize_option_type(row.get("option_type"))
        if option_type is None:
            if greeks is not None and idx in greeks.index:
                return self._greeks_component_pnl(
                    greek_row=greeks.loc[idx],
                    position_value=position_value,
                    shocks=simulated_returns[:, default_asset_idx],
                    n_simulations=n_simulations,
                    rng=rng,
                )
            return linear_component

        underlying_asset = row.get("underlying_asset", idx)
        underlying_idx = column_index.get(underlying_asset, default_asset_idx)

        try:
            spot_0 = float(row["spot"])
            strike = float(row["strike"])
            time_to_expiry = float(row["time_to_expiry"])
            implied_vol = float(row["implied_vol"])
            risk_free_rate = float(row.get("risk_free_rate", 0.0))
            vol_of_vol = float(row.get("vol_of_vol", 0.20))
        except (TypeError, ValueError):
            return linear_component

        if spot_0 <= 0 or strike <= 0 or implied_vol <= 0:
            return linear_component

        underlying_returns = simulated_returns[:, underlying_idx]
        shocked_spot = np.clip(spot_0 * np.exp(underlying_returns), 1e-8, None)

        vol_shock_scale = max(vol_of_vol, 1e-6) * np.sqrt(holding_period / 365.25)
        vol_shock = rng.normal(0.0, vol_shock_scale, n_simulations)
        if abs(leverage_correlation) > 1e-8:
            vol_shock += -leverage_correlation * underlying_returns
        shocked_vol = np.clip(implied_vol * (1.0 + vol_shock), 0.01, 5.0)
        shocked_tte = max(1e-8, time_to_expiry - holding_period / 365.25)

        from research.pricing.inverse_options import InverseOptionPricer

        try:
            base_price_btc = InverseOptionPricer.calculate_price(
                S=spot_0,
                K=strike,
                T=max(time_to_expiry, 1e-8),
                r=risk_free_rate,
                sigma=implied_vol,
                option_type=option_type,
            )
        except Exception:
            return linear_component

        base_price_usd = base_price_btc * spot_0
        if base_price_usd <= 1e-12:
            return linear_component

        quantity = position_value / base_price_usd
        revalued_price_btc = np.array(
            [
                InverseOptionPricer.calculate_price(
                    S=float(s),
                    K=strike,
                    T=shocked_tte,
                    r=risk_free_rate,
                    sigma=float(v),
                    option_type=option_type,
                )
                for s, v in zip(shocked_spot, shocked_vol)
            ],
            dtype=float,
        )
        revalued_price_usd = revalued_price_btc * shocked_spot
        return quantity * (revalued_price_usd - base_price_usd)

    def _option_schema_pnl(
        self,
        aligned_positions: pd.DataFrame,
        aligned_columns: list[str],
        simulated_returns: np.ndarray,
        greeks: pd.DataFrame | None,
        n_simulations: int,
        holding_period: int,
        leverage_correlation: float,
        rng: Any,
    ) -> np.ndarray:
        """Compute portfolio PnL when option contract schema is available."""
        pnl = np.zeros(n_simulations, dtype=float)
        column_index = {name: idx for idx, name in enumerate(aligned_columns)}

        for i, (idx, row) in enumerate(aligned_positions.iterrows()):
            pnl += self._single_position_full_revaluation_pnl(
                idx=idx,
                row=row,
                default_asset_idx=i,
                simulated_returns=simulated_returns,
                column_index=column_index,
                greeks=greeks,
                n_simulations=n_simulations,
                holding_period=holding_period,
                leverage_correlation=leverage_correlation,
                rng=rng,
            )
        return pnl

    def _greeks_approximation_pnl(
        self,
        aligned_positions: pd.DataFrame,
        simulated_returns: np.ndarray,
        greeks: pd.DataFrame,
        n_simulations: int,
        rng: Any,
    ) -> np.ndarray:
        """Compute portfolio PnL using delta-gamma-vega approximation only."""
        pnl = np.zeros(n_simulations, dtype=float)
        for i, (idx, row) in enumerate(aligned_positions.iterrows()):
            position_value = float(row["value"])
            if idx in greeks.index:
                pnl += self._greeks_component_pnl(
                    greek_row=greeks.loc[idx],
                    position_value=position_value,
                    shocks=simulated_returns[:, i],
                    n_simulations=n_simulations,
                    rng=rng,
                )
            else:
                pnl += simulated_returns[:, i] * position_value
        return pnl

    @staticmethod
    def _tail_risk_from_pnl(pnl: np.ndarray) -> tuple[float, float, float, float]:
        """Calculate VaR/CVaR metrics from simulated PnL distribution."""
        p5 = float(np.percentile(pnl, 5))
        p1 = float(np.percentile(pnl, 1))
        var_95 = -p5
        var_99 = -p1
        cvar_95 = -float(pnl[pnl <= p5].mean())
        cvar_99 = -float(pnl[pnl <= p1].mean())
        return var_95, var_99, cvar_95, cvar_99

    def monte_carlo_var(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        greeks: pd.DataFrame | None = None,
        n_simulations: int = 10000,
        holding_period: int = 1,
        leverage_correlation: float = -0.35,
        random_seed: int | None = None,
    ) -> VaRResult:
        """
        Calculate VaR using Monte Carlo simulation.

        Can incorporate Greeks for options positions.

        Args:
            positions: Position values
            returns: Historical returns for correlation structure
            greeks: Optional DataFrame with 'delta', 'gamma', 'vega' columns
            n_simulations: Number of Monte Carlo paths
            holding_period: Days to hold
            random_seed: Optional local RNG seed for reproducible simulation

        Returns:
            VaRResult
        """
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if holding_period <= 0:
            raise ValueError("holding_period must be positive")

        total_value, weights, aligned_returns = self._prepare_portfolio_inputs(positions, returns)
        aligned_positions = positions.loc[aligned_returns.columns]

        # Estimate parameters from historical returns
        mean = aligned_returns.mean().values
        cov = self._regularize_covariance(aligned_returns.cov().to_numpy(copy=True))

        # Optional local RNG improves reproducibility without mutating global RNG state.
        rng = np.random.default_rng(random_seed) if random_seed is not None else np.random

        simulated_returns = self._simulate_correlated_returns(
            mean=mean,
            cov=cov,
            weights=weights,
            n_simulations=n_simulations,
            holding_period=holding_period,
            leverage_correlation=leverage_correlation,
            rng=rng,
        )

        # Calculate PnL for each simulation.
        option_fields = {"spot", "strike", "time_to_expiry", "option_type", "implied_vol"}
        option_schema_available = option_fields.issubset(set(aligned_positions.columns))

        if option_schema_available:
            pnl = self._option_schema_pnl(
                aligned_positions=aligned_positions,
                aligned_columns=list(aligned_returns.columns),
                simulated_returns=simulated_returns,
                greeks=greeks,
                n_simulations=n_simulations,
                holding_period=holding_period,
                leverage_correlation=leverage_correlation,
                rng=rng,
            )
        elif greeks is not None:
            pnl = self._greeks_approximation_pnl(
                aligned_positions=aligned_positions,
                simulated_returns=simulated_returns,
                greeks=greeks,
                n_simulations=n_simulations,
                rng=rng,
            )
        else:
            # Linear approximation
            pnl = simulated_returns @ weights * total_value

        var_95, var_99, cvar_95, cvar_99 = self._tail_risk_from_pnl(pnl)

        return VaRResult(
            var_95=var_95, var_99=var_99, cvar_95=cvar_95, cvar_99=cvar_99, method="monte_carlo"
        )


class StressTest:
    """
    Stress testing framework for extreme scenarios.
    """

    # Predefined stress scenarios
    SCENARIOS = {
        "market_crash": {
            "description": "1987-style market crash",
            "spot_shock": -0.20,
            "vol_shock": 0.50,
            "correlation_spike": True,
        },
        "vol_spike": {
            "description": "Sudden volatility explosion",
            "spot_shock": -0.05,
            "vol_shock": 1.00,
            "correlation_spike": False,
        },
        "liquidity_crisis": {
            "description": "Liquidity drought",
            "spot_shock": -0.10,
            "vol_shock": 0.30,
            "bid_ask_widening": 3.0,
        },
        "flash_crash": {
            "description": "2010-style flash crash",
            "spot_shock": -0.10,
            "recovery": 0.08,
            "duration_minutes": 15,
        },
    }

    def run_stress_test(
        self, positions: pd.DataFrame, greeks: pd.DataFrame, scenario: str
    ) -> dict[str, object]:
        """
        Run stress test for a given scenario.

        Args:
            positions: Position data
            greeks: Greeks for each position
            scenario: Name of predefined scenario or custom dict

        Returns:
            Dictionary with scenario results
        """
        if isinstance(scenario, str):
            scenario = self.SCENARIOS.get(scenario, {})

        spot_shock = scenario.get("spot_shock", 0)
        vol_shock = scenario.get("vol_shock", 0)

        total_pnl = 0

        for idx, pos in positions.iterrows():
            if idx in greeks.index:
                g = greeks.loc[idx]

                # Delta PnL
                delta_pnl = g["delta"] * spot_shock * pos["value"]

                # Gamma PnL
                gamma_pnl = 0.5 * g.get("gamma", 0) * (spot_shock**2) * pos["value"]

                # Vega PnL
                vega_pnl = g.get("vega", 0) * vol_shock * 100 * pos["value"]

                total_pnl += delta_pnl + gamma_pnl + vega_pnl

        return {
            "scenario_name": scenario.get("description", "Custom"),
            "spot_shock": spot_shock,
            "vol_shock": vol_shock,
            "estimated_pnl": total_pnl,
            "pct_of_portfolio": total_pnl / positions["value"].sum() * 100,
        }

    def run_all_scenarios(self, positions: pd.DataFrame, greeks: pd.DataFrame) -> pd.DataFrame:
        """Run all predefined stress scenarios."""
        results = []

        for scenario_name in self.SCENARIOS:
            result = self.run_stress_test(positions, greeks, scenario_name)
            result["scenario"] = scenario_name
            results.append(result)

        return pd.DataFrame(results)
