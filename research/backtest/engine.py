"""Event-driven backtest engine for coin-margined market making strategies."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.types import (
    Fill,
    MarketState,
    OrderBook,
    OrderSide,
    Position,
    QuoteAction,
    Trade,
)
from research.backtest.fill_model import FillSimulatorConfig, RealisticFillSimulator
from research.pricing.inverse_options import InverseOptionPricer
from strategies.base import MarketMakingStrategy

__all__ = [
    "FillSimulatorConfig",
    "RealisticFillSimulator",
    "BacktestResult",
    "BacktestEngine",
]


def _build_market_state_snapshot(
    timestamp: datetime, price: float, order_book: OrderBook
) -> MarketState:
    """Create a market state snapshot for current tick."""
    return MarketState(
        timestamp=timestamp,
        instrument="SYNTHETIC",
        spot_price=price,
        order_book=order_book,
        recent_trades=[],
    )


def _history_to_series(history: List[Tuple[datetime, float]]) -> pd.Series:
    """Convert internal (timestamp, value) history to Series."""
    if not history:
        return pd.Series(dtype=float)
    return pd.Series([x[1] for x in history], index=[x[0] for x in history], dtype=float)


def _calculate_max_drawdown(pnl_series: pd.Series) -> float:
    """Calculate max drawdown using running peak denominator."""
    if len(pnl_series) == 0:
        return 0.0

    running_max = pnl_series.expanding().max()
    running_max_safe = running_max.replace(0, np.nan)
    drawdown = (running_max_safe - pnl_series) / running_max_safe
    max_dd = drawdown.max()
    if np.isnan(max_dd):
        return 0.0
    return float(max_dd)


def _trade_side_counts(trades: List[Fill]) -> Tuple[int, int]:
    """Count buy/sell fills."""
    buys = len([t for t in trades if t.side == OrderSide.BUY])
    sells = len([t for t in trades if t.side == OrderSide.SELL])
    return buys, sells


@dataclass
class BacktestResult:
    """Results from a backtest run (coin-margined)."""

    strategy_name: str

    # PnL metrics (in crypto units for coin-margined)
    total_pnl_crypto: float  # PnL in cryptocurrency units (BTC, ETH)
    total_pnl_usd: float  # PnL converted to USD at current price
    realized_pnl: float
    unrealized_pnl: float
    inventory_pnl: float  # PnL from inventory changes

    # Risk metrics
    sharpe_ratio: float
    deflated_sharpe_ratio: float
    max_drawdown: float
    volatility: float
    sharpe_ci_95: Tuple[float, float]
    drawdown_ci_95: Tuple[float, float]

    # Trade statistics
    trade_count: int
    buy_count: int
    sell_count: int
    avg_trade_size: float
    avg_trade_pnl_crypto: float

    # Spread capture
    total_spread_captured: float
    avg_spread_captured_bps: float

    # Costs
    inventory_cost: float  # Cost of holding inventory
    adverse_selection_cost: float  # Losses to informed traders

    # Crypto balance tracking (coin-margined specific)
    crypto_balance: float  # Balance in cryptocurrency units
    crypto_balance_series: pd.Series = field(default_factory=lambda: pd.Series())

    # Time series
    pnl_series: pd.Series = field(default_factory=lambda: pd.Series())
    inventory_series: pd.Series = field(default_factory=lambda: pd.Series())

    def summary(self) -> str:
        """Generate text summary of results."""
        return f"""
Backtest Results: {self.strategy_name} (Coin-Margined)
{'='*50}
Total PnL (Crypto): {self.total_pnl_crypto:.8f}
Total PnL (USD):    ${self.total_pnl_usd:,.2f}
Crypto Balance:     {self.crypto_balance:.8f}
Sharpe Ratio:       {self.sharpe_ratio:.2f}
Deflated Sharpe:    {self.deflated_sharpe_ratio:.2f}
Max Drawdown:       {self.max_drawdown:.2%}

Trade Statistics:
  Total Trades:     {self.trade_count}
  Buys:             {self.buy_count}
  Sells:            {self.sell_count}
  Avg Trade PnL:    {self.avg_trade_pnl_crypto:.8f} crypto

PnL Breakdown:
  Spread Capture:   ${self.total_spread_captured:,.2f}
  Inventory Cost:   ${self.inventory_cost:,.2f}
  Adverse Select:   ${self.adverse_selection_cost:,.2f}
"""


class BacktestEngine:
    """Event-driven backtest engine for coin-margined strategies."""

    def __init__(
        self,
        strategy: MarketMakingStrategy,
        fill_simulator: Optional[RealisticFillSimulator] = None,
        initial_crypto_balance: float = 1.0,
        base_currency: str = "BTC",
        random_seed: Optional[int] = None,
        transaction_cost_bps: float = 2.0,
    ):
        self.strategy = strategy
        self.rng = np.random.default_rng(random_seed)
        self.fill_simulator = fill_simulator or RealisticFillSimulator(rng=self.rng)
        self.fill_simulator.rng = self.rng
        self.initial_crypto_balance = initial_crypto_balance
        self.base_currency = base_currency
        self.random_seed = random_seed
        self.transaction_cost_bps = transaction_cost_bps

        self.crypto_balance = initial_crypto_balance
        self.positions: Dict[str, Position] = {}
        self.trades: List[Fill] = []
        self.quotes: List[QuoteAction] = []

        self._pnl_history: List[tuple] = []
        self._inventory_history: List[tuple] = []
        self._crypto_balance_history: List[tuple] = []

        self._history_sampling_interval: int = 10
        self._max_history_points: int = 1_000_000
        self._tick_counter: int = 0

    def _reset_run_state(self) -> None:
        """Reset runtime state before each backtest run."""
        self.crypto_balance = self.initial_crypto_balance
        self.positions = {}
        self.trades = []
        self.quotes = []
        self._pnl_history = []
        self._inventory_history = []
        self._crypto_balance_history = []
        self._tick_counter = 0
        self.strategy.reset()

        self.rng = np.random.default_rng(self.random_seed)
        if self.fill_simulator is not None:
            self.fill_simulator.rng = self.rng
            self.fill_simulator.reset_metrics()

    def _maybe_process_previous_quote(
        self,
        previous_quote: Optional[QuoteAction],
        market_state: MarketState,
        position: Position,
        event_volume: float,
        current_price: float,
    ) -> Position:
        """Try to fill previous quote against synthetic trades and update position."""
        if previous_quote is None or self.fill_simulator is None:
            return position

        synthetic_trades = self._generate_synthetic_trades(
            market_state,
            volume=event_volume,
        )
        if not synthetic_trades:
            return position

        fill = self.fill_simulator.simulate_fill(
            previous_quote,
            market_state,
            synthetic_trades,
            transaction_cost_bps=self.transaction_cost_bps,
        )
        if fill is None:
            return position

        self._process_fill(fill, position, current_price)
        updated_position = self.positions.get("SYNTHETIC", position)
        self.strategy.on_fill(fill, updated_position)
        return updated_position

    def _quote_with_lagged_price(
        self,
        prices: np.ndarray,
        idx: int,
        market_state: MarketState,
        position: Position,
    ) -> QuoteAction:
        """Query strategy quote on lagged price to avoid look-ahead bias."""
        current_price = float(prices[idx])
        lagged_price = float(prices[idx - 1]) if idx > 0 else current_price
        lagged_market_state = MarketState(
            timestamp=market_state.timestamp,
            instrument=market_state.instrument,
            spot_price=lagged_price,
            order_book=market_state.order_book,
            recent_trades=market_state.recent_trades,
        )
        return self.strategy.quote(lagged_market_state, position)

    def run(
        self,
        market_data: pd.DataFrame,
        price_column: str = "price",
        timestamp_column: str = "timestamp",
    ) -> BacktestResult:
        """Run backtest on historical market data."""
        self._reset_run_state()

        prices = market_data[price_column].to_numpy(dtype=np.float64)
        timestamps_arr = market_data[timestamp_column].to_numpy()
        n_events = len(prices)

        if n_events == 0:
            return self._compute_result(current_price=0.0)

        event_volumes = self._prepare_event_volumes(market_data)

        current_ob = self._create_dummy_order_book(prices[0])
        previous_quote: Optional[QuoteAction] = None

        for i in range(n_events):
            price = float(prices[i])
            timestamp = timestamps_arr[i]

            current_ob = self._update_order_book(current_ob, price)
            market_state = _build_market_state_snapshot(
                timestamp=timestamp, price=price, order_book=current_ob
            )

            position = self.positions.get("SYNTHETIC", Position("SYNTHETIC", 0, 0))
            position = self._maybe_process_previous_quote(
                previous_quote=previous_quote,
                market_state=market_state,
                position=position,
                event_volume=float(event_volumes[i]),
                current_price=price,
            )
            new_quote = self._quote_with_lagged_price(
                prices=prices, idx=i, market_state=market_state, position=position
            )
            self.quotes.append(new_quote)
            previous_quote = new_quote

            self._record_state(timestamp, price)

        return self._compute_result(current_price=float(prices[-1]))

    def _prepare_event_volumes(self, market_data: pd.DataFrame) -> np.ndarray:
        """Prepare per-event activity volume without materializing row dictionaries."""
        if "volume" not in market_data.columns:
            return np.ones(len(market_data), dtype=np.float64)

        values = market_data["volume"].to_numpy(dtype=np.float64, copy=False)
        if np.isnan(values).any():
            values = np.nan_to_num(values, nan=1.0, posinf=1.0, neginf=0.0)
        return np.maximum(values, 0.0)

    def _create_dummy_order_book(self, price: float) -> OrderBook:
        """Create a simple order book around given price."""
        from core.types import OrderBookLevel

        spread = price * 0.001  # 10 bps spread

        return OrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument="SYNTHETIC",
            bids=[OrderBookLevel(price=price - spread / 2, size=1.0)],
            asks=[OrderBookLevel(price=price + spread / 2, size=1.0)],
        )

    def _update_order_book(self, ob: OrderBook, new_price: float) -> OrderBook:
        """Update order book with new price."""
        spread = ob.spread or new_price * 0.001
        from core.types import OrderBookLevel

        return OrderBook(
            timestamp=ob.timestamp,
            instrument=ob.instrument,
            bids=[OrderBookLevel(price=new_price - spread / 2, size=1.0)],
            asks=[OrderBookLevel(price=new_price + spread / 2, size=1.0)],
        )

    def _generate_synthetic_trades(
        self, market_state: MarketState, volume: float = 1.0
    ) -> List[Trade]:
        """Generate synthetic trades from current event activity."""
        trades = []
        timestamp = market_state.timestamp

        num_trades = min(3, max(0, int(volume * self.rng.random())))

        for _ in range(num_trades):
            imbalance = (
                market_state.order_book.imbalance()
                if hasattr(market_state.order_book, "imbalance")
                else 0
            )
            side_bias = 0.5 + imbalance * 0.2
            side = OrderSide.BUY if self.rng.random() < side_bias else OrderSide.SELL

            mid = market_state.spot_price
            price_offset = self.rng.normal(0, mid * 0.0001)
            trade_price = mid + price_offset

            trade_size = volume * self.rng.random() * 0.3

            trades.append(
                Trade(
                    timestamp=timestamp,
                    instrument=market_state.instrument,
                    price=abs(trade_price),
                    size=abs(trade_size),
                    side=side,
                )
            )

        return trades

    def _process_fill(self, fill: Fill, current_position: Position, current_price: float) -> None:
        """Process a fill and update position plus crypto balance."""
        self.trades.append(fill)

        new_position = current_position.apply_fill(fill)
        self.positions[fill.instrument] = new_position

        premium_crypto = (fill.price * fill.size) / current_price

        if fill.side == OrderSide.BUY:
            self.crypto_balance -= premium_crypto
        else:
            self.crypto_balance += premium_crypto

    def _record_state(self, timestamp: datetime, current_price: float) -> None:
        """Record sampled state for analysis."""
        self._tick_counter += 1

        if self._tick_counter % self._history_sampling_interval != 0 and self._tick_counter > 1:
            return

        position = self.positions.get("SYNTHETIC", Position("SYNTHETIC", 0, 0))
        self._inventory_history.append((timestamp, position.size))

        crypto_pnl = self._calculate_crypto_pnl(current_price)
        self._pnl_history.append((timestamp, crypto_pnl))
        self._crypto_balance_history.append((timestamp, self.crypto_balance))

        if len(self._pnl_history) > self._max_history_points:
            remove_count = self._max_history_points // 5
            self._pnl_history = self._pnl_history[remove_count:]
            self._inventory_history = self._inventory_history[remove_count:]
            self._crypto_balance_history = self._crypto_balance_history[remove_count:]

    def _calculate_crypto_pnl(self, current_price: Optional[float] = None) -> float:
        """Calculate realized + unrealized coin PnL."""
        realized_pnl = self.crypto_balance - self.initial_crypto_balance

        unrealized_pnl = 0.0
        if current_price is not None and current_price > 0:
            for _, position in self.positions.items():
                if position.size != 0 and position.avg_entry_price > 0:
                    unrealized_pnl += InverseOptionPricer.calculate_pnl(
                        entry_price=position.avg_entry_price,
                        exit_price=current_price,
                        size=position.size,
                        inverse=True,
                    )

        return realized_pnl + unrealized_pnl

    def _bootstrap_risk_ci(
        self, returns: pd.Series, n_bootstrap: int = 500, alpha: float = 0.05
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Bootstrap confidence intervals for Sharpe and drawdown."""
        if len(returns) < 10:
            return (0.0, 0.0), (0.0, 0.0)

        sharpe_samples: List[float] = []
        dd_samples: List[float] = []
        annualization = np.sqrt(365.0)
        values = returns.to_numpy()

        for _ in range(n_bootstrap):
            sample = self.rng.choice(values, size=len(values), replace=True)
            std = np.std(sample, ddof=1)
            if std > 1e-12:
                sharpe_samples.append(float(np.mean(sample) / std * annualization))
            else:
                sharpe_samples.append(0.0)

            cum = np.cumsum(sample)
            running_max = np.maximum.accumulate(cum)
            denom = np.where(np.abs(running_max) > 1e-12, np.abs(running_max), 1.0)
            dd = (running_max - cum) / denom
            dd_samples.append(float(np.max(dd)))

        lo = alpha / 2
        hi = 1 - alpha / 2
        sharpe_ci = (
            float(np.quantile(sharpe_samples, lo)),
            float(np.quantile(sharpe_samples, hi)),
        )
        drawdown_ci = (
            float(np.quantile(dd_samples, lo)),
            float(np.quantile(dd_samples, hi)),
        )
        return sharpe_ci, drawdown_ci

    @staticmethod
    def _deflated_sharpe_ratio(sharpe: float, n_obs: int, n_trials: int = 1) -> float:
        """Simplified deflated Sharpe ratio approximation."""
        if n_obs < 5:
            return 0.0
        expected_max_sr = norm.ppf(1.0 - 1.0 / max(n_trials, 2)) / np.sqrt(max(n_obs - 1, 1))
        denom = max(1e-12, np.sqrt(1.0 / max(n_obs - 1, 1)))
        z = (sharpe - expected_max_sr) / denom
        return float(norm.cdf(z))

    def _calculate_sharpe_ratio(self, pnl_series: pd.Series) -> float:
        """Calculate annualized Sharpe ratio from coin PnL series."""
        if len(pnl_series) <= 1:
            return 0.0

        pnl_changes = pnl_series.diff().dropna()
        returns = pnl_changes / self.initial_crypto_balance
        if returns.std() <= 0:
            return 0.0

        if len(returns) > 1:
            time_span = (pnl_series.index[-1] - pnl_series.index[0]).total_seconds()
            periods_per_year = len(returns) * (365.0 * 24 * 3600) / max(time_span, 1)
            annualization = np.sqrt(periods_per_year)
        else:
            annualization = np.sqrt(365.0 * 24)

        risk_free_rate = 0.0
        excess_returns = returns - risk_free_rate / (365.0 * 24)
        return float((excess_returns.mean() / returns.std()) * annualization)

    def _execution_costs(self) -> Tuple[float, float]:
        """Return execution and adverse selection costs from fill simulator."""
        if self.fill_simulator is None:
            return 0.0, 0.0
        execution_cost = float(
            self.fill_simulator.transaction_cost_paid + self.fill_simulator.slippage_cost
        )
        adverse_selection_cost = float(self.fill_simulator.adverse_selection_cost)
        return execution_cost, adverse_selection_cost

    def _compute_result(self, current_price: Optional[float] = None) -> BacktestResult:
        """Compute final backtest metrics (coin-margined)."""
        pnl_series = _history_to_series(self._pnl_history)
        inventory_series = _history_to_series(self._inventory_history)
        crypto_balance_series = _history_to_series(self._crypto_balance_history)

        total_pnl_crypto = float(pnl_series.iloc[-1]) if len(pnl_series) > 0 else 0.0
        total_pnl_usd = total_pnl_crypto * current_price

        sharpe = self._calculate_sharpe_ratio(pnl_series)
        max_dd = _calculate_max_drawdown(pnl_series)

        buys, sells = _trade_side_counts(self.trades)
        avg_size = np.mean([t.size for t in self.trades]) if self.trades else 0
        returns_for_ci = (
            pnl_series.diff().dropna() / max(self.initial_crypto_balance, 1e-12)
            if len(pnl_series) > 1
            else pd.Series(dtype=float)
        )
        sharpe_ci, drawdown_ci = self._bootstrap_risk_ci(returns_for_ci)
        deflated_sharpe = self._deflated_sharpe_ratio(
            float(sharpe), n_obs=len(returns_for_ci), n_trials=1
        )
        execution_cost, adverse_selection_cost = self._execution_costs()

        return BacktestResult(
            strategy_name=self.strategy.name,
            total_pnl_crypto=total_pnl_crypto,
            total_pnl_usd=total_pnl_usd,
            realized_pnl=total_pnl_crypto,
            unrealized_pnl=0,
            inventory_pnl=0,
            sharpe_ratio=sharpe,
            deflated_sharpe_ratio=deflated_sharpe,
            max_drawdown=max_dd,
            volatility=pnl_series.std() if len(pnl_series) > 1 else 0,
            sharpe_ci_95=sharpe_ci,
            drawdown_ci_95=drawdown_ci,
            trade_count=len(self.trades),
            buy_count=buys,
            sell_count=sells,
            avg_trade_size=avg_size,
            avg_trade_pnl_crypto=total_pnl_crypto / len(self.trades) if self.trades else 0,
            total_spread_captured=0,
            avg_spread_captured_bps=0,
            inventory_cost=execution_cost,
            adverse_selection_cost=adverse_selection_cost,
            crypto_balance=self.crypto_balance,
            crypto_balance_series=crypto_balance_series,
            pnl_series=pnl_series,
            inventory_series=inventory_series,
        )
