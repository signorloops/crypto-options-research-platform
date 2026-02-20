"""
Synthetic market data generators for backtesting.
Includes geometric Brownian motion, jump diffusion, and microstructure models.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.types import (
    Greeks,
    OptionContract,
    OptionType,
    OrderBook,
    OrderBookLevel,
    OrderSide,
)


@dataclass
class PriceModelParams:
    """Parameters for price process simulation."""
    S0: float = 50000.0           # Initial price
    mu: float = 0.1               # Annual drift
    sigma: float = 0.5            # Annual volatility
    dt: float = 1 / 365 / 24      # Time step (1 hour)

    # Jump diffusion parameters
    jump_intensity: float = 0.05  # Jumps per year
    jump_mean: float = 0.0        # Mean jump size
    jump_std: float = 0.05        # Jump size std


class GBMPriceGenerator:
    """
    Geometric Brownian Motion price generator.
    dS/S = mu*dt + sigma*dW
    """

    def __init__(self, params: PriceModelParams, seed: Optional[int] = None):
        self.params = params
        self.seed = seed

    def generate(
        self,
        T: float,  # Time horizon in years
        start_time: datetime = None
    ) -> pd.DataFrame:
        """Generate price path."""
        # Reset seed for reproducibility on each call
        if self.seed:
            np.random.seed(self.seed)

        n_steps = int(T / self.params.dt) + 1  # +1 to include start point

        # Generate random walk (n_steps - 1 random returns, first return is 0)
        dW = np.random.normal(0, np.sqrt(self.params.dt), n_steps - 1)
        drift = (self.params.mu - 0.5 * self.params.sigma**2) * self.params.dt
        log_returns = np.concatenate([[0], drift + self.params.sigma * dW])

        # Cumulative - first price is S0, subsequent prices based on returns
        log_prices = np.cumsum(log_returns)
        prices = self.params.S0 * np.exp(log_prices)

        # Create timestamps
        if start_time is None:
            start_time = datetime(2024, 1, 1)
        timestamps = [start_time + timedelta(hours=i) for i in range(n_steps)]

        return pd.DataFrame({
            "timestamp": timestamps,
            "price": prices,
            "returns": np.concatenate([[0], np.diff(log_prices)]),
            "volatility": self.params.sigma
        })


class MertonJumpDiffusion(GBMPriceGenerator):
    """
    Merton Jump Diffusion model.
    Adds Poisson jumps to GBM for more realistic fat tails.
    """

    def generate(self, T: float, start_time: datetime = None) -> pd.DataFrame:
        """Generate price path with jumps."""
        # Reset seed for reproducibility on each call
        if self.seed:
            np.random.seed(self.seed)

        n_steps = int(T / self.params.dt) + 1  # +1 to include start point

        # GBM component (n_steps - 1 random returns, first return is 0)
        dW = np.random.normal(0, np.sqrt(self.params.dt), n_steps - 1)
        drift = (self.params.mu - 0.5 * self.params.sigma**2) * self.params.dt
        log_returns = np.concatenate([[0], drift + self.params.sigma * dW])

        # Jump component (Poisson) - no jumps at start
        jumps = np.random.poisson(
            self.params.jump_intensity * self.params.dt,
            n_steps - 1
        )
        jump_sizes = np.random.normal(
            self.params.jump_mean,
            self.params.jump_std,
            n_steps - 1
        )
        log_returns[1:] += jumps * jump_sizes

        # Cumulative - first price is S0
        log_prices = np.cumsum(log_returns)
        prices = self.params.S0 * np.exp(log_prices)

        if start_time is None:
            start_time = datetime(2024, 1, 1)
        timestamps = [start_time + timedelta(hours=i) for i in range(n_steps)]

        # Pad jumps array to match length (first element is 0 - no jump at start)
        jumps_padded = np.concatenate([[0], jumps])

        return pd.DataFrame({
            "timestamp": timestamps,
            "price": prices,
            "returns": log_returns,
            "jump_count": jumps_padded,
            "volatility": self.params.sigma
        })


class OrderBookSimulator:
    """
    Simulates realistic order book dynamics.
    Based on microstructure research on order arrival and cancellation.
    """

    def __init__(
        self,
        base_spread_bps: float = 10.0,
        depth_levels: int = 10,
        tick_size: float = 0.5
    ):
        self.base_spread_bps = base_spread_bps
        self.depth_levels = depth_levels
        self.tick_size = tick_size

    def generate_snapshot(
        self,
        mid_price: float,
        volatility_regime: float = 1.0,  # 1.0 = normal, >1 = high vol
        spread_multiplier: float = 1.0
    ) -> OrderBook:
        """Generate order book around mid price."""
        # Adjust spread based on volatility
        effective_spread = self.base_spread_bps * volatility_regime * spread_multiplier
        half_spread = mid_price * effective_spread / 10000 / 2

        bids = []
        asks = []

        for i in range(self.depth_levels):
            # Price levels
            bid_price = mid_price - half_spread - i * self.tick_size
            ask_price = mid_price + half_spread + i * self.tick_size

            # Size with exponential decay by depth
            # Higher volatility = thinner book
            decay_rate = 0.3 * volatility_regime
            base_size = 10 * np.exp(-decay_rate * i)

            # Add noise
            bid_size = max(0.1, base_size * np.random.lognormal(0, 0.5))
            ask_size = max(0.1, base_size * np.random.lognormal(0, 0.5))

            bids.append(OrderBookLevel(
                price=round(bid_price, 2),
                size=round(bid_size, 4),
                num_orders=np.random.poisson(3)
            ))
            asks.append(OrderBookLevel(
                price=round(ask_price, 2),
                size=round(ask_size, 4),
                num_orders=np.random.poisson(3)
            ))

        return OrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument="SYNTHETIC",
            bids=bids,
            asks=asks
        )

    def generate_time_series(
        self,
        price_path: pd.DataFrame,
        volatility_column: str = "volatility"
    ) -> List[OrderBook]:
        """Generate order book series following price path."""
        snapshots = []

        for _, row in price_path.iterrows():
            # Volatility regime from 0.5 to 2.0
            regime = 0.5 + 1.5 * (row[volatility_column] / price_path[volatility_column].mean())

            ob = self.generate_snapshot(
                mid_price=row["price"],
                volatility_regime=regime
            )
            ob.timestamp = row["timestamp"]
            snapshots.append(ob)

        return snapshots


class TradeFlowSimulator:
    """
    Simulates realistic trade flow with:
    - Autoregressive trade arrival
    - Size distribution (power law for large trades)
    - Directional bias based on recent price moves
    """

    def __init__(
        self,
        base_arrival_rate: float = 5.0,  # Trades per second
        size_alpha: float = 2.0,  # Pareto shape for size distribution
        informed_trade_prob: float = 0.1  # Probability of informed trade
    ):
        self.base_arrival_rate = base_arrival_rate
        self.size_alpha = size_alpha
        self.informed_trade_prob = informed_trade_prob

    def generate(
        self,
        price_path: pd.DataFrame,
        order_books: Optional[List[OrderBook]] = None
    ) -> pd.DataFrame:
        """Generate trade flow matching price dynamics."""
        trades = []

        for i, row in price_path.iterrows():
            # Arrival rate varies with volatility
            hour_vol = abs(row["returns"]) if not np.isnan(row["returns"]) else 0
            arrival_rate = self.base_arrival_rate * (1 + 5 * hour_vol * np.sqrt(365 * 24))

            # Number of trades this hour (Poisson)
            n_trades = np.random.poisson(arrival_rate * 3600)

            for _ in range(n_trades):
                # Trade timing
                offset_ms = np.random.randint(0, 3600 * 1000)
                timestamp = row["timestamp"] + timedelta(milliseconds=offset_ms)

                # Is this an informed trade?
                is_informed = np.random.random() < self.informed_trade_prob

                # Direction: informed trades follow current momentum (no look-ahead)
                if is_informed and i > 0:
                    current_return = row["returns"]
                    side = OrderSide.BUY if current_return > 0 else OrderSide.SELL
                else:
                    # Uninformed: slight momentum following
                    momentum_bias = 0.5 + 0.3 * np.sign(row["returns"]) if row["returns"] != 0 else 0.5
                    side = OrderSide.BUY if np.random.random() < momentum_bias else OrderSide.SELL

                # Size: Pareto distribution (many small, few large)
                base_size = np.random.pareto(self.size_alpha) * 0.1
                if is_informed:
                    base_size *= 3  # Informed trades are larger
                size = max(0.001, base_size)

                # Price: within spread or outside for market orders
                if order_books:
                    ob = order_books[i]
                    if side == OrderSide.BUY:
                        price = ob.best_ask * (1 + np.random.exponential(0.0001))
                    else:
                        price = ob.best_bid * (1 - np.random.exponential(0.0001))
                else:
                    price = row["price"] * (1 + np.random.normal(0, 0.001))

                trades.append({
                    "timestamp": timestamp,
                    "price": price,
                    "size": round(size, 4),
                    "side": side.value,
                    "is_informed": is_informed
                })

        df = pd.DataFrame(trades)
        if not df.empty:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df


class OptionMarketSimulator:
    """
    Simulates complete option market with multiple strikes and expiries.
    """

    def __init__(
        self,
        underlying: str = "BTC",
        risk_free_rate: float = 0.05
    ):
        self.underlying = underlying
        self.risk_free_rate = risk_free_rate

    def generate_option_chain(
        self,
        spot: float,
        expiry_dates: List[datetime],
        moneyness_range: Tuple[float, float] = (0.8, 1.2),
        n_strikes: int = 11
    ) -> List[OptionContract]:
        """Generate option chain around current spot."""
        chain = []

        for expiry in expiry_dates:
            # Strikes around spot
            min_strike = spot * moneyness_range[0]
            max_strike = spot * moneyness_range[1]
            strikes = np.linspace(min_strike, max_strike, n_strikes)

            for strike in strikes:
                # Both calls and puts
                for opt_type in [OptionType.CALL, OptionType.PUT]:
                    chain.append(OptionContract(
                        underlying=self.underlying,
                        strike=round(strike, -int(np.floor(np.log10(strike))) + 2),
                        expiry=expiry,
                        option_type=opt_type
                    ))

        return chain

    def calculate_greeks(
        self,
        contract: OptionContract,
        spot: float,
        volatility: float,
        as_of: datetime
    ) -> Greeks:
        """Calculate option Greeks using Black-Scholes."""
        from scipy.stats import norm

        T = contract.time_to_expiry(as_of)
        if T <= 0:
            return Greeks(delta=0, gamma=0, theta=0, vega=0)

        S = spot
        K = contract.strike
        r = self.risk_free_rate
        sigma = volatility

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if contract.option_type == OptionType.CALL:
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Theta (annualized)
        if contract.option_type == OptionType.CALL:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                     + r * K * np.exp(-r * T) * norm.cdf(-d2))

        # Vega (per 1% change in vol)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100

        return Greeks(
            delta=delta,
            gamma=gamma,
            theta=theta / 365,  # Convert to daily
            vega=vega
        )

    def implied_volatility_smile(
        self,
        moneyness: float,
        time_to_expiry: float,
        base_vol: float = 0.5
    ) -> float:
        """
        Generate realistic volatility smile.

        moneyness = strike / spot
        """
        # ATM vol
        atm_vol = base_vol

        # Skew (puts more expensive than calls)
        skew = -0.2 * (moneyness - 1)

        # Smile (wings lift)
        smile = 0.15 * (moneyness - 1)**2

        # Term structure (shorter dated = higher vol)
        term = 0.1 * np.sqrt(30 / max(time_to_expiry * 365, 1))

        return atm_vol + skew + smile + term


class CompleteMarketSimulator:
    """
    High-level simulator that generates complete market data:
    - Spot price path
    - Order book snapshots
    - Trade flow
    - Option chain with Greeks
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

        self.price_gen = MertonJumpDiffusion(PriceModelParams(), seed)
        self.ob_sim = OrderBookSimulator()
        self.trade_sim = TradeFlowSimulator()
        self.option_sim = OptionMarketSimulator()

    def generate(
        self,
        days: int = 30,
        hours: Optional[int] = None,
        include_options: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate complete synthetic market dataset.

        Args:
            days: Number of days to simulate (default: 30)
            hours: Alternative to days, simulate this many hours (overrides days if set)
            include_options: Whether to include option chain data

        Returns:
            Dictionary with keys: spot, order_book, trades, options
        """
        # Reset random seed for reproducibility
        if self.seed:
            np.random.seed(self.seed)

        # Support hours parameter for shorter simulations (useful in tests)
        if hours is not None:
            T = hours / 24 / 365
        else:
            T = days / 365

        # 1. Generate spot price
        spot = self.price_gen.generate(T)

        # 2. Generate order books
        obs = self.ob_sim.generate_time_series(spot)

        # 3. Generate trades
        trades = self.trade_sim.generate(spot, obs)

        result = {
            "spot": spot,
            "order_book": self._obs_to_df(obs),
            "trades": trades
        }

        # 4. Generate option data if requested
        if include_options:
            option_data = []

            # Sample timestamps (every hour)
            for _, row in spot.iloc[::1].iterrows():  # Hourly
                # Generate option chain
                expiries = [
                    row["timestamp"] + timedelta(days=d)
                    for d in [7, 14, 30, 60, 90]
                ]
                chain = self.option_sim.generate_option_chain(
                    spot=row["price"],
                    expiry_dates=expiries
                )

                for contract in chain:
                    T = contract.time_to_expiry(row["timestamp"])
                    if T <= 0:
                        continue

                    moneyness = contract.strike / row["price"]
                    iv = self.option_sim.implied_volatility_smile(moneyness, T)
                    greeks = self.option_sim.calculate_greeks(
                        contract, row["price"], iv, row["timestamp"]
                    )

                    # Generate bid/ask around theoretical price
                    spread_bps = 50 + 100 * iv  # Higher vol = wider spread

                    option_data.append({
                        "timestamp": row["timestamp"],
                        "instrument": contract.instrument_name,
                        "underlying": contract.underlying,
                        "strike": contract.strike,
                        "expiry": contract.expiry,
                        "option_type": contract.option_type.value,
                        "spot": row["price"],
                        "implied_vol": iv,
                        "delta": greeks.delta,
                        "gamma": greeks.gamma,
                        "theta": greeks.theta,
                        "vega": greeks.vega,
                        "bid_iv": iv * (1 - spread_bps / 20000),
                        "ask_iv": iv * (1 + spread_bps / 20000),
                    })

            result["options"] = pd.DataFrame(option_data)

        return result

    def _obs_to_df(self, obs: List[OrderBook]) -> pd.DataFrame:
        """Convert order book list to DataFrame."""
        records = []
        for ob in obs:
            records.append({
                "timestamp": ob.timestamp,
                "best_bid": ob.best_bid,
                "best_ask": ob.best_ask,
                "mid": ob.mid_price,
                "spread": ob.spread,
                "spread_bps": (ob.spread / ob.mid_price * 10000) if ob.mid_price else None,
                "imbalance": ob.imbalance(),
                "bid_volume_5": sum(lvl.size for lvl in ob.bids[:5]),
                "ask_volume_5": sum(lvl.size for lvl in ob.asks[:5])
            })
        return pd.DataFrame(records)
