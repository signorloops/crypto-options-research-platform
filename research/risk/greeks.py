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
        """
        Calculate all Greeks for an option.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        """
        # Input validation
        if S <= 0:
            raise ValueError(f"Spot price S must be positive, got {S}")
        if K <= 0:
            raise ValueError(f"Strike price K must be positive, got {K}")
        if sigma <= 0:
            raise ValueError(f"Volatility sigma must be positive, got {sigma}")
        if T <= 0:
            # At expiry, Greeks are degenerate
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

        # Gamma (same for calls and puts)
        gamma = n_prime_d1 / (S * sigma * np.sqrt(T))

        # Theta (daily)
        if option_type == 'call':
            theta = (-S * n_prime_d1 * sigma / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * nd2) / 365
        else:
            theta = (-S * n_prime_d1 * sigma / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

        # Vega (per 1% change in vol)
        vega = S * n_prime_d1 * np.sqrt(T) / 100

        # Higher-order Greeks
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

    def analyze_position(
        self,
        position: Position,
        contract: OptionContract,
        spot: float,
        implied_vol: float,
        as_of: datetime
    ) -> Tuple[Greeks, Greeks]:
        """
        Calculate Greeks for a position.

        Automatically detects coin-margined (inverse) options and uses
        the appropriate pricing model.

        Returns:
            (per_contract_greeks, position_greeks)
        """
        T = contract.time_to_expiry(as_of)
        option_type = 'call' if contract.option_type.value == 'C' else 'put'

        if contract.is_coin_margined:
            # Use inverse option pricer for coin-margined options
            from research.pricing.inverse_options import InverseOptionPricer
            inv_greeks = InverseOptionPricer.calculate_greeks(
                S=spot,
                K=contract.strike,
                T=T,
                r=self.risk_free_rate,
                sigma=implied_vol,
                option_type=option_type
            )
            # Convert InverseGreeks to Greeks
            greeks = Greeks(
                delta=inv_greeks.delta,
                gamma=inv_greeks.gamma,
                theta=inv_greeks.theta,
                vega=inv_greeks.vega,
                rho=inv_greeks.rho,
                vanna=inv_greeks.vanna,
                charm=inv_greeks.charm
            )
        else:
            # Use standard Black-Scholes for U-margined options
            greeks = self.calculator.calculate(
                S=spot,
                K=contract.strike,
                T=T,
                r=self.risk_free_rate,
                sigma=implied_vol,
                option_type=option_type
            )

        # Scale by position size
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
        from collections import defaultdict

        # Group positions by currency
        by_currency: Dict[str, List[Tuple]] = defaultdict(list)
        for position, contract, spot, iv in positions:
            currency = contract.underlying.split('-')[0]
            by_currency[currency].append((position, contract, spot, iv))

        # Calculate Greeks per currency
        result: Dict[str, PortfolioGreeks] = {}
        for currency, currency_positions in by_currency.items():
            total = PortfolioGreeks(
                delta=0, gamma=0, theta=0, vega=0, rho=0,
                vanna=0, charm=0, veta=0
            )

            for position, contract, spot, iv in currency_positions:
                _, position_greeks = self.analyze_position(
                    position, contract, spot, iv, as_of
                )

                # Get FX rate for USD conversion
                fx_rate = 1.0
                if fx_rates:
                    fx_rate = fx_rates.get(currency, 1.0)

                # Convert Greeks to USD terms with proper dimensionality
                # For coin-margined (inverse) options:
                #   Price is in BTC, Spot is in USD/BTC
                #   Delta = dV/dS = BTC / (USD/BTC) = BTC^2/USD
                #   To get USD notional Delta: Delta_USD = Delta_BTC * (USD/BTC)^2 = Delta_BTC * spot^2
                #   Gamma = d^2V/dS^2 = BTC^3/USD^2
                #   To get USD Gamma: Gamma_USD = Gamma_BTC * (USD/BTC)^3 = Gamma_BTC * spot^3
                # For standard (USD-margined) options:
                #   Delta is dimensionless (USD/USD)
                #   To get USD notional: Delta_USD = Delta * spot * fx_rate
                #   Gamma is 1/USD, Gamma_USD = Gamma * spot^2 * fx_rate
                # Vega/Theta/Rho: already in USD terms -> multiply by fx_rate only
                if contract.inverse:
                    # Coin-margined (inverse) conversion
                    # Correct USD Delta: d(V_BTC * S)/dS = V_BTC + S * Delta_BTC
                    # Where V_BTC is the option price in BTC, Delta_BTC is dV/dS in BTC^2/USD
                    # Gamma_USD conversion: Gamma_BTC * S^3 (verified by dimensional analysis)
                    spot_safe = max(spot, 1e-6)

                    # Calculate option price for correct Delta conversion
                    from research.pricing.inverse_options import InverseOptionPricer
                    T = contract.time_to_expiry(as_of)
                    option_type = 'call' if contract.option_type.value == 'C' else 'put'
                    # Get IV from the tuple (position, contract, spot, iv)
                    # Find matching position data to get IV
                    iv = 0.6  # default value
                    for pos, contr, s, implied in currency_positions:
                        if contr == contract and pos == position:
                            iv = implied
                            break
                    price_btc = InverseOptionPricer.calculate_price(
                        spot_safe, contract.strike, T, self.risk_free_rate, iv, option_type
                    )

                    # Correct USD Delta: V_BTC + S * Delta_BTC (in BTC, then convert to USD)
                    # For inverse options, BTC -> USD conversion is just * spot (not * spot * fx_rate,
                    # since fx_rate IS spot for the base currency)
                    delta_usd_btc = price_btc + spot_safe * position_greeks.delta  # in BTC
                    delta_usd = delta_usd_btc * spot_safe  # BTC -> USD

                    # Gamma conversion: Gamma_BTC * S^3 (dimension: BTC^3/USD^2 * USD^3/BTC^3 = USD)
                    gamma_usd = position_greeks.gamma * (spot_safe ** 3)

                    total = total + PortfolioGreeks(
                        delta=delta_usd,
                        gamma=gamma_usd,
                        theta=position_greeks.theta * fx_rate,
                        vega=position_greeks.vega * fx_rate,
                        rho=(position_greeks.rho or 0) * fx_rate,
                        vanna=(position_greeks.vanna or 0) * fx_rate,
                        charm=(position_greeks.charm or 0) * fx_rate,
                        veta=0
                    )
                else:
                    # Standard (USD-margined) conversion
                    spot_fx = spot * fx_rate
                    total = total + PortfolioGreeks(
                        delta=position_greeks.delta * spot_fx,
                        gamma=position_greeks.gamma * spot_fx * spot,
                        theta=position_greeks.theta * fx_rate,
                        vega=position_greeks.vega * fx_rate,
                        rho=(position_greeks.rho or 0) * fx_rate,
                        vanna=(position_greeks.vanna or 0) * fx_rate,
                        charm=(position_greeks.charm or 0) * fx_rate,
                        veta=0
                    )

            result[currency] = total

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
        if spot_shocks is None:
            spot_shocks = [0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1]
        if vol_shocks is None:
            vol_shocks = [-0.2, -0.1, 0, 0.1, 0.2]

        results = []

        for spot_mult in spot_shocks:
            for vol_change in vol_shocks:
                pnl = 0

                for position, contract, spot, iv in positions:
                    base_greeks, _ = self.analyze_position(
                        position, contract, spot, iv, as_of
                    )

                    shocked_spot = spot * spot_mult
                    shocked_iv = max(0.01, iv + vol_change)

                    shocked_greeks, _ = self.analyze_position(
                        position, contract, shocked_spot, shocked_iv, as_of
                    )

                    # Simplified PnL from Greeks
                    delta_pnl = base_greeks.delta * (shocked_spot - spot) * position.size
                    vega_pnl = base_greeks.vega * vol_change * 100 * position.size
                    gamma_pnl = 0.5 * base_greeks.gamma * (shocked_spot - spot)**2 * position.size

                    pnl += delta_pnl + vega_pnl + gamma_pnl

                results.append({
                    'spot_shock': f"{spot_mult:.0%}",
                    'vol_shock': f"{vol_change:+.0%}",
                    'pnl': pnl
                })

        return pd.DataFrame(results)

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
