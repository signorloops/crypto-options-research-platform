"""Value at Risk (VaR) and Expected Shortfall (CVaR) calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

OPTION_PRICING_EXCEPTIONS = (
    ValueError,
    TypeError,
    ArithmeticError,
    FloatingPointError,
    OverflowError,
    ZeroDivisionError,
)


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


@dataclass(frozen=True)
class _OptionRevaluationInputs:
    """Validated option inputs needed for full revaluation PnL simulation."""

    option_type: str
    underlying_idx: int
    spot_0: float
    strike: float
    time_to_expiry: float
    implied_vol: float
    risk_free_rate: float
    vol_of_vol: float


def _option_schema_available(aligned_positions: pd.DataFrame) -> bool:
    """Check whether positions include full option-revaluation schema."""
    option_fields = {"spot", "strike", "time_to_expiry", "option_type", "implied_vol"}
    return option_fields.issubset(set(aligned_positions.columns))


def _compute_monte_carlo_pnl(
    *,
    calculator: "VaRCalculator",
    aligned_positions: pd.DataFrame,
    aligned_columns: list[str],
    simulated_returns: np.ndarray,
    weights: np.ndarray,
    total_value: float,
    greeks: pd.DataFrame | None,
    n_simulations: int,
    holding_period: int,
    leverage_correlation: float,
    rng: Any,
) -> np.ndarray:
    """Dispatch to option-schema, Greeks approximation, or linear PnL path."""
    if _option_schema_available(aligned_positions):
        return calculator._option_schema_pnl(
            aligned_positions=aligned_positions,
            aligned_columns=aligned_columns,
            simulated_returns=simulated_returns,
            greeks=greeks,
            n_simulations=n_simulations,
            holding_period=holding_period,
            leverage_correlation=leverage_correlation,
            rng=rng,
        )
    if greeks is not None:
        return calculator._greeks_approximation_pnl(
            aligned_positions=aligned_positions,
            simulated_returns=simulated_returns,
            greeks=greeks,
            n_simulations=n_simulations,
            rng=rng,
        )
    return simulated_returns @ weights * total_value


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
        """Calculate parametric VaR/CVaR under a normal return assumption."""
        if holding_period <= 0:
            raise ValueError("holding_period must be positive")
        total_value, weights_arr, aligned_returns = self._prepare_portfolio_inputs(
            positions, returns
        )
        mean_return = aligned_returns.mean().values @ weights_arr
        cov_matrix = aligned_returns.cov().values
        portfolio_std = np.sqrt(weights_arr @ cov_matrix @ weights_arr)
        adjusted_std = portfolio_std * np.sqrt(holding_period)
        z_95 = stats.norm.ppf(0.95)
        z_99 = stats.norm.ppf(0.99)
        var_95 = total_value * (-mean_return * holding_period + z_95 * adjusted_std)
        var_99 = total_value * (-mean_return * holding_period + z_99 * adjusted_std)
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
        """Cornish-Fisher VaR with skew/kurtosis-adjusted quantiles."""
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
            return (
                z
                + (z**2 - 1) * skew / 6.0
                + (z**3 - 3 * z) * (kurt - 3.0) / 24.0
                - (2 * z**3 - 5 * z) * (skew**2) / 36.0
            )
        z95 = cf_quantile(0.95)
        z99 = cf_quantile(0.99)
        var_95 = total_value * max(0.0, -mu + z95 * sigma)
        var_99 = total_value * max(0.0, -mu + z99 * sigma)
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
        """Calculate historical VaR/CVaR from empirical portfolio-return quantiles."""
        if holding_period <= 0:
            raise ValueError("holding_period must be positive")
        total_value, weights_arr, aligned_returns = self._prepare_portfolio_inputs(
            positions, returns
        )
        portfolio_returns = pd.Series(
            aligned_returns.to_numpy() @ weights_arr, index=aligned_returns.index
        )
        if holding_period > 1:
            n_periods = len(portfolio_returns) // holding_period
            if n_periods > 0:
                reshaped = portfolio_returns.iloc[: n_periods * holding_period].values.reshape(
                    n_periods, holding_period
                )
                portfolio_returns = pd.Series(reshaped.sum(axis=1))
            else:
                portfolio_returns = portfolio_returns * np.sqrt(holding_period)
        var_95 = -np.percentile(portfolio_returns, 5) * total_value
        var_99 = -np.percentile(portfolio_returns, 1) * total_value
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
        random_seed: int | None = None,
    ) -> VaRResult:
        """Filtered Historical Simulation (EWMA volatility + residual resampling)."""
        if holding_period <= 0:
            raise ValueError("holding_period must be positive")
        total_value, weights_arr, aligned_returns = self._prepare_portfolio_inputs(
            positions, returns
        )
        portfolio_returns = self._portfolio_return_series(aligned_returns, weights_arr)
        if len(portfolio_returns) < 30:
            return self.historical_var(positions, returns, holding_period)
        cond_vol = self._ewma_conditional_volatility(
            portfolio_returns=portfolio_returns, lambda_param=lambda_param
        )
        standardized = portfolio_returns / cond_vol
        pnl = self._simulate_fhs_pnl(
            standardized_returns=standardized,
            latest_vol=float(cond_vol[-1]),
            holding_period=holding_period,
            total_value=total_value,
            random_seed=random_seed,
        )
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

    @staticmethod
    def _ewma_conditional_volatility(
        *, portfolio_returns: np.ndarray, lambda_param: float, eps: float = 1e-12
    ) -> np.ndarray:
        ewma_var = np.zeros_like(portfolio_returns, dtype=float)
        ewma_var[0] = np.var(portfolio_returns)
        for i in range(1, len(portfolio_returns)):
            ewma_var[i] = (
                lambda_param * ewma_var[i - 1] + (1 - lambda_param) * portfolio_returns[i - 1] ** 2
            )
        return np.sqrt(np.maximum(ewma_var, eps))

    @staticmethod
    def _simulate_fhs_pnl(
        *,
        standardized_returns: np.ndarray,
        latest_vol: float,
        holding_period: int,
        total_value: float,
        random_seed: int | None,
    ) -> np.ndarray:
        rng = np.random.default_rng(random_seed)
        n_sim = max(5000, len(standardized_returns) * 20)
        sampled_resid = rng.choice(standardized_returns, size=n_sim, replace=True)
        simulated_returns = sampled_resid * latest_vol * np.sqrt(holding_period)
        return simulated_returns * total_value

    def evt_var(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        holding_period: int = 1,
        threshold_quantile: float = 0.9,
    ) -> VaRResult:
        """Peaks-over-threshold EVT VaR using a GPD fit on tail losses."""
        if holding_period <= 0:
            raise ValueError("holding_period must be positive")
        total_value, weights_arr, aligned_returns = self._prepare_portfolio_inputs(
            positions, returns
        )
        portfolio_returns = self._portfolio_return_series(aligned_returns, weights_arr) * np.sqrt(holding_period)
        losses = -portfolio_returns
        tail_fit = self._fit_evt_tail(losses=losses, threshold_quantile=threshold_quantile)
        if tail_fit is None:
            return self.historical_var(positions, returns, holding_period)
        threshold, shape, scale, pu = tail_fit
        q95 = self._evt_quantile(alpha=0.95, threshold=threshold, shape=shape, scale=scale, pu=pu)
        q99 = self._evt_quantile(alpha=0.99, threshold=threshold, shape=shape, scale=scale, pu=pu)
        var_95 = q95 * total_value
        var_99 = q99 * total_value
        cvar_95 = self._evt_expected_shortfall(
            q_alpha=q95, threshold=threshold, shape=shape, scale=scale, total_value=total_value
        )
        cvar_99 = self._evt_expected_shortfall(
            q_alpha=q99, threshold=threshold, shape=shape, scale=scale, total_value=total_value
        )
        return VaRResult(
            var_95=float(var_95),
            var_99=float(var_99),
            cvar_95=float(cvar_95),
            cvar_99=float(cvar_99),
            method="evt",
        )

    @staticmethod
    def _fit_evt_tail(
        *, losses: np.ndarray, threshold_quantile: float
    ) -> tuple[float, float, float, float] | None:
        if len(losses) < 50:
            return None
        threshold = float(np.quantile(losses, threshold_quantile))
        tail_losses = losses[losses > threshold]
        if len(tail_losses) < 20:
            return None
        excess = tail_losses - threshold
        shape, _loc, scale = stats.genpareto.fit(excess, floc=0.0)
        pu = len(tail_losses) / len(losses)
        return threshold, float(shape), float(scale), float(pu)

    @staticmethod
    def _evt_quantile(
        *, alpha: float, threshold: float, shape: float, scale: float, pu: float
    ) -> float:
        tail_prob = (1 - alpha) / max(pu, 1e-12)
        tail_prob = max(tail_prob, 1e-12)
        if abs(shape) < 1e-8:
            q = threshold + scale * np.log(1.0 / tail_prob)
        else:
            q = threshold + scale / shape * (tail_prob ** (-shape) - 1.0)
        return float(max(q, 0.0))

    @staticmethod
    def _evt_expected_shortfall(
        *, q_alpha: float, threshold: float, shape: float, scale: float, total_value: float
    ) -> float:
        if shape >= 1:
            return float(q_alpha * total_value)
        es_loss = q_alpha + (scale - shape * threshold) / (1 - shape)
        return float(max(es_loss, q_alpha) * total_value)

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

    @staticmethod
    def _parse_option_revaluation_inputs(
        *,
        row: pd.Series,
        option_type: str,
        default_asset_idx: int,
        column_index: dict[object, int],
    ) -> _OptionRevaluationInputs | None:
        underlying_asset = row.get("underlying_asset", row.name)
        underlying_idx = column_index.get(underlying_asset, default_asset_idx)
        try:
            spot_0 = float(row["spot"])
            strike = float(row["strike"])
            time_to_expiry = float(row["time_to_expiry"])
            implied_vol = float(row["implied_vol"])
            risk_free_rate = float(row.get("risk_free_rate", 0.0))
            vol_of_vol = float(row.get("vol_of_vol", 0.20))
        except (TypeError, ValueError):
            return None
        if spot_0 <= 0 or strike <= 0 or implied_vol <= 0:
            return None
        return _OptionRevaluationInputs(
            option_type=option_type,
            underlying_idx=underlying_idx,
            spot_0=spot_0,
            strike=strike,
            time_to_expiry=time_to_expiry,
            implied_vol=implied_vol,
            risk_free_rate=risk_free_rate,
            vol_of_vol=vol_of_vol,
        )

    @staticmethod
    def _simulate_revaluation_price_paths(
        *,
        inputs: _OptionRevaluationInputs,
        underlying_returns: np.ndarray,
        n_simulations: int,
        holding_period: int,
        leverage_correlation: float,
        rng: Any,
    ) -> tuple[float, np.ndarray] | None:
        shocked_spot, shocked_vol, shocked_tte = VaRCalculator._build_revaluation_shocks(
            inputs=inputs,
            underlying_returns=underlying_returns,
            n_simulations=n_simulations,
            holding_period=holding_period,
            leverage_correlation=leverage_correlation,
            rng=rng,
        )
        priced = VaRCalculator._price_revalued_inverse_option_series(
            inputs=inputs,
            shocked_spot=shocked_spot,
            shocked_vol=shocked_vol,
            shocked_tte=shocked_tte,
        )
        if priced is None:
            return None
        base_price_btc, revalued_price_btc = priced

        if not np.isfinite(base_price_btc):
            return None
        base_price_usd = base_price_btc * inputs.spot_0
        if base_price_usd <= 1e-12:
            return None

        revalued_price_usd = revalued_price_btc * shocked_spot
        if not np.isfinite(revalued_price_usd).all():
            return None
        return base_price_usd, revalued_price_usd

    @staticmethod
    def _build_revaluation_shocks(
        *,
        inputs: _OptionRevaluationInputs,
        underlying_returns: np.ndarray,
        n_simulations: int,
        holding_period: int,
        leverage_correlation: float,
        rng: Any,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        shocked_spot = np.clip(inputs.spot_0 * np.exp(underlying_returns), 1e-8, None)
        vol_shock_scale = max(inputs.vol_of_vol, 1e-6) * np.sqrt(holding_period / 365.25)
        vol_shock = rng.normal(0.0, vol_shock_scale, n_simulations)
        if abs(leverage_correlation) > 1e-8:
            vol_shock += -leverage_correlation * underlying_returns
        shocked_vol = np.clip(inputs.implied_vol * (1.0 + vol_shock), 0.01, 5.0)
        shocked_tte = max(1e-8, inputs.time_to_expiry - holding_period / 365.25)
        return shocked_spot, shocked_vol, shocked_tte

    @staticmethod
    def _price_revalued_inverse_option_series(
        *,
        inputs: _OptionRevaluationInputs,
        shocked_spot: np.ndarray,
        shocked_vol: np.ndarray,
        shocked_tte: float,
    ) -> tuple[float, np.ndarray] | None:
        from research.pricing.inverse_options import InverseOptionPricer

        try:
            base_price_btc = InverseOptionPricer.calculate_price(
                S=inputs.spot_0,
                K=inputs.strike,
                T=max(inputs.time_to_expiry, 1e-8),
                r=inputs.risk_free_rate,
                sigma=inputs.implied_vol,
                option_type=inputs.option_type,
            )
            revalued_price_btc = np.array(
                [
                    InverseOptionPricer.calculate_price(
                        S=float(s),
                        K=inputs.strike,
                        T=shocked_tte,
                        r=inputs.risk_free_rate,
                        sigma=float(v),
                        option_type=inputs.option_type,
                    )
                    for s, v in zip(shocked_spot, shocked_vol)
                ],
                dtype=float,
            )
        except OPTION_PRICING_EXCEPTIONS:
            return None
        return base_price_btc, revalued_price_btc

    def _non_option_position_pnl(
        self,
        *,
        idx: object,
        position_value: float,
        linear_component: np.ndarray,
        shocks: np.ndarray,
        greeks: pd.DataFrame | None,
        n_simulations: int,
        rng: Any,
    ) -> np.ndarray:
        if greeks is not None and idx in greeks.index:
            return self._greeks_component_pnl(
                greek_row=greeks.loc[idx],
                position_value=position_value,
                shocks=shocks,
                n_simulations=n_simulations,
                rng=rng,
            )
        return linear_component

    @staticmethod
    def _position_quantity_from_value(position_value: float, base_price_usd: float) -> float | None:
        quantity = position_value / base_price_usd
        if not np.isfinite(quantity):
            return None
        return float(quantity)

    def _position_revaluation_pnl_or_linear(
        self,
        *,
        position_value: float,
        linear_component: np.ndarray,
        revalued: tuple[float, np.ndarray] | None,
    ) -> np.ndarray:
        if revalued is None:
            return linear_component
        base_price_usd, revalued_price_usd = revalued
        quantity = self._position_quantity_from_value(position_value, base_price_usd)
        if quantity is None:
            return linear_component
        return quantity * (revalued_price_usd - base_price_usd)

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
        """Simulate one-position PnL with option revaluation and linear/Greeks fallbacks."""
        position_value = float(row["value"])
        linear_component = simulated_returns[:, default_asset_idx] * position_value
        option_type = self._normalize_option_type(row.get("option_type"))
        if option_type is None:
            return self._non_option_position_pnl(
                idx=idx,
                position_value=position_value,
                linear_component=linear_component,
                shocks=simulated_returns[:, default_asset_idx],
                greeks=greeks,
                n_simulations=n_simulations,
                rng=rng,
            )
        inputs = self._parse_option_revaluation_inputs(row=row, option_type=option_type, default_asset_idx=default_asset_idx, column_index=column_index)
        if inputs is None: return linear_component
        underlying_returns = simulated_returns[:, inputs.underlying_idx]
        revalued = self._simulate_revaluation_price_paths(
            inputs=inputs,
            underlying_returns=underlying_returns,
            n_simulations=n_simulations,
            holding_period=holding_period,
            leverage_correlation=leverage_correlation,
            rng=rng,
        )
        return self._position_revaluation_pnl_or_linear(
            position_value=position_value,
            linear_component=linear_component,
            revalued=revalued,
        )

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

    def _prepare_monte_carlo_path_inputs(
        self,
        *,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        n_simulations: int,
        holding_period: int,
        leverage_correlation: float,
        random_seed: int | None,
    ) -> tuple[float, np.ndarray, pd.DataFrame, pd.DataFrame, np.ndarray, Any]:
        total_value, weights, aligned_returns = self._prepare_portfolio_inputs(positions, returns)
        aligned_positions = positions.loc[aligned_returns.columns]
        mean = aligned_returns.mean().values
        cov = self._regularize_covariance(aligned_returns.cov().to_numpy(copy=True))
        rng = np.random.default_rng(random_seed)
        simulated_returns = self._simulate_correlated_returns(
            mean=mean,
            cov=cov,
            weights=weights,
            n_simulations=n_simulations,
            holding_period=holding_period,
            leverage_correlation=leverage_correlation,
            rng=rng,
        )
        return total_value, weights, aligned_returns, aligned_positions, simulated_returns, rng

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
        """Calculate Monte Carlo VaR with optional Greeks-aware option revaluation."""
        if n_simulations <= 0:
            raise ValueError("n_simulations must be positive")
        if holding_period <= 0:
            raise ValueError("holding_period must be positive")

        total_value, weights, aligned_returns, aligned_positions, simulated_returns, rng = (
            self._prepare_monte_carlo_path_inputs(
                positions=positions,
                returns=returns,
                n_simulations=n_simulations,
                holding_period=holding_period,
                leverage_correlation=leverage_correlation,
                random_seed=random_seed,
            )
        )
        pnl = _compute_monte_carlo_pnl(
            calculator=self,
            aligned_positions=aligned_positions,
            aligned_columns=list(aligned_returns.columns),
            simulated_returns=simulated_returns,
            weights=weights,
            total_value=total_value,
            greeks=greeks,
            n_simulations=n_simulations,
            holding_period=holding_period,
            leverage_correlation=leverage_correlation,
            rng=rng,
        )

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
        """Run stress test for a predefined or custom scenario."""
        if isinstance(scenario, str):
            scenario = self.SCENARIOS.get(scenario, {})
        spot_shock = scenario.get("spot_shock", 0)
        vol_shock = scenario.get("vol_shock", 0)
        total_pnl = 0
        for idx, pos in positions.iterrows():
            if idx in greeks.index:
                g = greeks.loc[idx]
                delta_pnl = g["delta"] * spot_shock * pos["value"]
                gamma_pnl = 0.5 * g.get("gamma", 0) * (spot_shock**2) * pos["value"]
                vega_pnl = g.get("vega", 0) * vol_shock * 100 * pos["value"]
                total_pnl += delta_pnl + gamma_pnl + vega_pnl
        gross_exposure = float(np.abs(positions["value"]).sum()) if "value" in positions else 0.0
        pct_of_portfolio = (total_pnl / gross_exposure * 100) if gross_exposure > 1e-12 else 0.0
        return {
            "scenario_name": scenario.get("description", "Custom"),
            "spot_shock": spot_shock,
            "vol_shock": vol_shock,
            "estimated_pnl": total_pnl,
            "pct_of_portfolio": pct_of_portfolio,
        }

    def run_all_scenarios(self, positions: pd.DataFrame, greeks: pd.DataFrame) -> pd.DataFrame:
        """Run all predefined stress scenarios."""
        results = []

        for scenario_name in self.SCENARIOS:
            result = self.run_stress_test(positions, greeks, scenario_name)
            result["scenario"] = scenario_name
            results.append(result)

        return pd.DataFrame(results)
