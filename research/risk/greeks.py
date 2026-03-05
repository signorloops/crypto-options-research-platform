"""
Greeks calculation and portfolio risk analysis for options.
"""
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.types import Greeks, OptionContract, Position


@dataclass
class PortfolioGreeks:
    """Aggregate Greeks for a portfolio."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

    # Second-order and cross Greeks
    vanna: float
    charm: float
    veta: float

    def __add__(self, other: 'PortfolioGreeks') -> 'PortfolioGreeks':
        """Add two PortfolioGreeks together."""
        return PortfolioGreeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            theta=self.theta + other.theta,
            vega=self.vega + other.vega,
            rho=self.rho + other.rho,
            vanna=self.vanna + other.vanna,
            charm=self.charm + other.charm,
            veta=self.veta + other.veta
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho,
            'vanna': self.vanna,
            'charm': self.charm,
            'veta': self.veta
        }


class BlackScholesGreeks:
    """
    Calculate Greeks using Black-Scholes model.
    """

    @staticmethod
    def calculate(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str
    ) -> Greeks:
        """Calculate all Greeks for an option."""
        if S <= 0:
            raise ValueError(f"Spot price S must be positive, got {S}")
        if K <= 0:
            raise ValueError(f"Strike price K must be positive, got {K}")
        if sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {sigma}")
        if T <= 0:
            return Greeks(delta=0, gamma=0, theta=0, vega=0, rho=0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        n_prime_d1 = norm.pdf(d1)
        if option_type == 'call':
            delta = nd1
            rho = K * T * np.exp(-r * T) * nd2 / 100  # Per 1% rate change
        else:  # put
            delta = nd1 - 1
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        gamma = n_prime_d1 / (S * sigma * np.sqrt(T))
        theta_core = -S * n_prime_d1 * sigma / (2 * np.sqrt(T))
        theta = (theta_core - r * K * np.exp(-r * T) * nd2) / 365 if option_type == 'call' else (theta_core + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        vega = S * n_prime_d1 * np.sqrt(T) / 100
        vanna = -n_prime_d1 * d2 / sigma
        charm = -n_prime_d1 * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
        return Greeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            vanna=vanna,
            charm=charm
        )


class GreeksRiskAnalyzer:
    """
    Analyze Greeks exposure for options portfolios.
    """

    def __init__(self, risk_free_rate: float = None):
        if risk_free_rate is None:
            risk_free_rate = float(os.getenv("RISK_FREE_RATE", "0.05"))
        self.risk_free_rate = risk_free_rate
        self.calculator = BlackScholesGreeks()

    @staticmethod
    def _group_positions_by_currency(
        positions: List[Tuple[Position, OptionContract, float, float]]
    ) -> Dict[str, List[Tuple[Position, OptionContract, float, float]]]:
        from collections import defaultdict

        by_currency: Dict[str, List[Tuple[Position, OptionContract, float, float]]] = defaultdict(list)
        for position, contract, spot, iv in positions:
            currency = contract.underlying.split("-")[0]
            by_currency[currency].append((position, contract, spot, iv))
        return by_currency

    def _convert_position_greeks_to_portfolio(
        self,
        *,
        position_greeks: Greeks,
        contract: OptionContract,
        spot: float,
        iv: float,
        as_of: datetime,
        fx_rate: float,
    ) -> PortfolioGreeks:
        """Convert a single-position Greeks vector into portfolio USD terms."""
        if contract.inverse:
            spot_safe = max(spot, 1e-6)
            iv_safe = float(iv)
            if not np.isfinite(iv_safe) or iv_safe <= 0:
                iv_safe = 1e-6
            from research.pricing.inverse_options import InverseOptionPricer
            T = contract.time_to_expiry(as_of)
            option_type = "call" if contract.option_type.value == "C" else "put"
            price_btc = InverseOptionPricer.calculate_price(
                spot_safe, contract.strike, T, self.risk_free_rate, iv_safe, option_type
            )
            delta_usd_btc = price_btc + spot_safe * position_greeks.delta
            delta_usd = delta_usd_btc * spot_safe
            gamma_usd = position_greeks.gamma * (spot_safe ** 3)
            return PortfolioGreeks(
                delta=delta_usd,
                gamma=gamma_usd,
                theta=position_greeks.theta * fx_rate,
                vega=position_greeks.vega * fx_rate,
                rho=(position_greeks.rho or 0) * fx_rate,
                vanna=(position_greeks.vanna or 0) * fx_rate,
                charm=(position_greeks.charm or 0) * fx_rate,
                veta=0,
            )
        spot_fx = spot * fx_rate
        return PortfolioGreeks(
            delta=position_greeks.delta * spot_fx,
            gamma=position_greeks.gamma * spot_fx * spot,
            theta=position_greeks.theta * fx_rate,
            vega=position_greeks.vega * fx_rate,
            rho=(position_greeks.rho or 0) * fx_rate,
            vanna=(position_greeks.vanna or 0) * fx_rate,
            charm=(position_greeks.charm or 0) * fx_rate,
            veta=0,
        )

    def _aggregate_currency_greeks(
        self,
        *,
        currency: str,
        currency_positions: List[Tuple[Position, OptionContract, float, float]],
        as_of: datetime,
        fx_rates: Optional[Dict[str, float]],
    ) -> PortfolioGreeks:
        total = PortfolioGreeks(
            delta=0, gamma=0, theta=0, vega=0, rho=0, vanna=0, charm=0, veta=0
        )
        fx_rate = fx_rates.get(currency, 1.0) if fx_rates else 1.0

        for position, contract, spot, iv in currency_positions:
            _, position_greeks = self.analyze_position(position, contract, spot, iv, as_of)
            total = total + self._convert_position_greeks_to_portfolio(
                position_greeks=position_greeks,
                contract=contract,
                spot=spot,
                iv=iv,
                as_of=as_of,
                fx_rate=fx_rate,
            )
        return total

    def _contract_greeks(
        self,
        *,
        contract: OptionContract,
        spot: float,
        implied_vol: float,
        as_of: datetime,
        option_type: str,
    ) -> Greeks:
        if contract.is_coin_margined:
            from research.pricing.inverse_options import InverseOptionPricer

            inv_greeks = InverseOptionPricer.calculate_greeks(
                S=spot,
                K=contract.strike,
                T=contract.time_to_expiry(as_of),
                r=self.risk_free_rate,
                sigma=implied_vol,
                option_type=option_type,
            )
            return Greeks(
                delta=inv_greeks.delta,
                gamma=inv_greeks.gamma,
                theta=inv_greeks.theta,
                vega=inv_greeks.vega,
                rho=inv_greeks.rho,
                vanna=inv_greeks.vanna,
                charm=inv_greeks.charm,
            )
        return self.calculator.calculate(
            S=spot,
            K=contract.strike,
            T=contract.time_to_expiry(as_of),
            r=self.risk_free_rate,
            sigma=implied_vol,
            option_type=option_type,
        )

    def analyze_position(
        self,
        position: Position,
        contract: OptionContract,
        spot: float,
        implied_vol: float,
        as_of: datetime
    ) -> Tuple[Greeks, Greeks]:
        """Calculate per-contract and position-scaled Greeks for a single option position."""
        option_type = 'call' if contract.option_type.value == 'C' else 'put'
        greeks = self._contract_greeks(
            contract=contract, spot=spot, implied_vol=implied_vol, as_of=as_of, option_type=option_type
        )
        position_greeks = Greeks(
            delta=greeks.delta * position.size,
            gamma=greeks.gamma * position.size,
            theta=greeks.theta * position.size,
            vega=greeks.vega * position.size,
            rho=greeks.rho * position.size if greeks.rho else 0,
            vanna=greeks.vanna * position.size if greeks.vanna else 0,
            charm=greeks.charm * position.size if greeks.charm else 0
        )

        return greeks, position_greeks

    def analyze_portfolio(
        self,
        positions: List[Tuple[Position, OptionContract, float, float]],
        as_of: datetime,
        fx_rates: Optional[Dict[str, float]] = None
    ) -> Dict[str, PortfolioGreeks]:
        """
        Calculate aggregate Greeks for a portfolio, grouped by underlying currency.

        For cross-currency portfolios, Greeks are reported per currency rather than
        summed, as adding BTC Delta to ETH Delta is not meaningful.

        Args:
            positions: List of (Position, OptionContract, spot, implied_vol)
            as_of: Current time
            fx_rates: Optional dict mapping underlying currency to FX rate.
                     e.g., {"BTC": 50000.0, "ETH": 3000.0} for USD conversion.
                     If provided, Greeks are converted to USD terms.

        Returns:
            Dict mapping currency (e.g., "BTC", "ETH") to PortfolioGreeks.
            Special key "total_usd" contains aggregated USD exposure if fx_rates provided.
        """
        by_currency = self._group_positions_by_currency(positions)

        # Calculate Greeks per currency
        result: Dict[str, PortfolioGreeks] = {}
        for currency, currency_positions in by_currency.items():
            result[currency] = self._aggregate_currency_greeks(
                currency=currency,
                currency_positions=currency_positions,
                as_of=as_of,
                fx_rates=fx_rates,
            )

        return result

    def calculate_greeks_scenarios(
        self,
        positions: List[Tuple[Position, OptionContract, float, float]],
        as_of: datetime,
        spot_shocks: List[float] = None,
        vol_shocks: List[float] = None
    ) -> pd.DataFrame:
        """
        Calculate PnL under various spot and vol scenarios.

        Args:
            spot_shocks: List of spot price multipliers (e.g., [0.95, 1.0, 1.05])
            vol_shocks: List of vol changes (e.g., [-0.1, 0, 0.1])

        Returns:
            DataFrame with PnL for each scenario
        """
        spot_shocks = spot_shocks or self._default_spot_shocks()
        vol_shocks = vol_shocks or self._default_vol_shocks()
        results = []
        for spot_mult in spot_shocks:
            for vol_change in vol_shocks:
                pnl = sum(
                    self._scenario_position_pnl(
                        position=position,
                        contract=contract,
                        spot=spot,
                        iv=iv,
                        as_of=as_of,
                        spot_mult=spot_mult,
                        vol_change=vol_change,
                    )
                    for position, contract, spot, iv in positions
                )
                results.append(
                    {"spot_shock": f"{spot_mult:.0%}", "vol_shock": f"{vol_change:+.0%}", "pnl": pnl}
                )
        return pd.DataFrame(results)

    @staticmethod
    def _default_spot_shocks() -> List[float]:
        return [0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1]

    @staticmethod
    def _default_vol_shocks() -> List[float]:
        return [-0.2, -0.1, 0, 0.1, 0.2]

    def _scenario_position_pnl(
        self,
        *,
        position: Position,
        contract: OptionContract,
        spot: float,
        iv: float,
        as_of: datetime,
        spot_mult: float,
        vol_change: float,
    ) -> float:
        base_greeks, _ = self.analyze_position(position, contract, spot, iv, as_of)
        shocked_spot = spot * spot_mult
        shocked_iv = max(0.01, iv + vol_change)
        _shocked_greeks, _ = self.analyze_position(position, contract, shocked_spot, shocked_iv, as_of)
        delta_pnl = base_greeks.delta * (shocked_spot - spot) * position.size
        vega_pnl = base_greeks.vega * vol_change * 100 * position.size
        gamma_pnl = 0.5 * base_greeks.gamma * (shocked_spot - spot) ** 2 * position.size
        return float(delta_pnl + vega_pnl + gamma_pnl)

    def find_hedge_ratio(
        self,
        portfolio_greeks: PortfolioGreeks,
        hedge_instrument: str = 'spot'
    ) -> float:
        """
        Calculate hedge ratio to neutralize Delta.

        Args:
            portfolio_greeks: Current portfolio Greeks
            hedge_instrument: Type of hedge ('spot', 'future', 'option')

        Returns:
            Hedge quantity (positive = buy, negative = sell)
        """
        if hedge_instrument == 'spot':
            # Simple spot hedge
            return -portfolio_greeks.delta

        # Could extend with other hedge instruments
        return -portfolio_greeks.delta
