"""
Order book feature extraction for market microstructure analysis.
Extracts features that predict short-term price movements and toxicity.
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class OrderBookFeatures:
    """Complete set of order book features."""
    timestamp: pd.Timestamp

    # Basic features
    mid_price: float
    spread: float
    spread_bps: float
    best_bid: float
    best_ask: float

    # Depth features
    bid_depth_5: float
    ask_depth_5: float
    bid_depth_10: float
    ask_depth_10: float
    depth_imbalance: float

    # Weighted prices
    vwap_bid: float
    vwap_ask: float
    vwap_mid: float

    # Microprice (volume-weighted mid)
    microprice: float
    microprice_bias: float

    # Order book slope
    bid_slope: float
    ask_slope: float

    # Queue position features
    bid_queue_ratio: float
    ask_queue_ratio: float

    # Volatility features
    realized_vol_1min: Optional[float]
    realized_vol_5min: Optional[float]

    # Flow features
    trade_flow_imbalance: Optional[float]
    volume_order_imbalance: Optional[float]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'mid_price': self.mid_price,
            'spread': self.spread,
            'spread_bps': self.spread_bps,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'bid_depth_5': self.bid_depth_5,
            'ask_depth_5': self.ask_depth_5,
            'bid_depth_10': self.bid_depth_10,
            'ask_depth_10': self.ask_depth_10,
            'depth_imbalance': self.depth_imbalance,
            'vwap_bid': self.vwap_bid,
            'vwap_ask': self.vwap_ask,
            'vwap_mid': self.vwap_mid,
            'microprice': self.microprice,
            'microprice_bias': self.microprice_bias,
            'bid_slope': self.bid_slope,
            'ask_slope': self.ask_slope,
            'bid_queue_ratio': self.bid_queue_ratio,
            'ask_queue_ratio': self.ask_queue_ratio,
            'realized_vol_1min': self.realized_vol_1min,
            'realized_vol_5min': self.realized_vol_5min,
            'trade_flow_imbalance': self.trade_flow_imbalance,
            'volume_order_imbalance': self.volume_order_imbalance
        }


class OrderBookFeatureExtractor:
    """
    Extract features from order book data.
    """

    def __init__(self):
        self._max_history = 300  # 5 minutes of 1-second data
        self._history: deque = deque(maxlen=self._max_history)

    @staticmethod
    def _vwap(levels: List, depth: int) -> float:
        selected = levels[:depth]
        total = sum(lvl.size for lvl in selected)
        if total == 0:
            return 0.0
        return float(sum(lvl.price * lvl.size for lvl in selected) / total)

    @staticmethod
    def _calc_slope(levels: List, max_depth: int = 5) -> float:
        if len(levels) < 2:
            return 0.0
        depths = np.arange(1, min(len(levels), max_depth) + 1)
        prices = [lvl.price for lvl in levels[:max_depth]]
        if len(depths) != len(prices):
            return 0.0
        slope, _, _, _, _ = stats.linregress(depths, prices)
        return float(slope)

    def _compute_realized_volatility(self) -> tuple[Optional[float], Optional[float]]:
        realized_vol_1min = None
        realized_vol_5min = None

        if len(self._history) >= 60:
            recent = list(self._history)[-60:]
            prices = [h["mid_price"] for h in recent if h["mid_price"] > 0]
            if len(prices) > 1:
                returns = np.diff(np.log(prices))
                realized_vol_1min = float(np.std(returns) * np.sqrt(365 * 24 * 60))

        if len(self._history) >= 300:
            recent = list(self._history)[-300:]
            prices = [h["mid_price"] for h in recent if h["mid_price"] > 0]
            if len(prices) > 1:
                returns = np.diff(np.log(prices))
                realized_vol_5min = float(np.std(returns) * np.sqrt(365 * 24 * 12))

        return realized_vol_1min, realized_vol_5min

    @staticmethod
    def _compute_trade_flow_features(
        recent_trades: Optional[List["Trade"]],
        depth_imbalance: float,
    ) -> tuple[Optional[float], Optional[float]]:
        trade_flow_imbalance = None
        volume_order_imbalance = None
        if not recent_trades:
            return trade_flow_imbalance, volume_order_imbalance

        buy_volume = sum(t.size for t in recent_trades if t.side.value == "buy")
        sell_volume = sum(t.size for t in recent_trades if t.side.value == "sell")
        total = buy_volume + sell_volume
        if total > 0:
            trade_flow_imbalance = (buy_volume - sell_volume) / total
        if len(recent_trades) > 0:
            volume_order_imbalance = trade_flow_imbalance * depth_imbalance

        return trade_flow_imbalance, volume_order_imbalance

    @staticmethod
    def _extract_basic_features(order_book: "OrderBook") -> Dict[str, float]:
        """Extract basic mid/spread and top-of-book features."""
        mid = order_book.mid_price or 0
        spread = order_book.spread or 0
        best_bid = order_book.best_bid or 0
        best_ask = order_book.best_ask or 0
        spread_bps = (spread / mid * 10000) if mid > 0 else 0
        return {
            "mid": float(mid),
            "spread": float(spread),
            "best_bid": float(best_bid),
            "best_ask": float(best_ask),
            "spread_bps": float(spread_bps),
        }

    @staticmethod
    def _extract_depth_features(order_book: "OrderBook") -> Dict[str, float]:
        """Extract near-book depth aggregates and imbalance."""
        bid_depth_5 = float(sum(lvl.size for lvl in order_book.bids[:5]))
        ask_depth_5 = float(sum(lvl.size for lvl in order_book.asks[:5]))
        bid_depth_10 = float(sum(lvl.size for lvl in order_book.bids[:10]))
        ask_depth_10 = float(sum(lvl.size for lvl in order_book.asks[:10]))
        total_depth = bid_depth_10 + ask_depth_10
        depth_imbalance = (bid_depth_5 - ask_depth_5) / total_depth if total_depth > 0 else 0.0
        return {
            "bid_depth_5": bid_depth_5,
            "ask_depth_5": ask_depth_5,
            "bid_depth_10": bid_depth_10,
            "ask_depth_10": ask_depth_10,
            "depth_imbalance": float(depth_imbalance),
        }

    @staticmethod
    def _compute_microprice_features(
        order_book: "OrderBook",
        *,
        best_bid: float,
        best_ask: float,
        mid: float,
    ) -> tuple[float, float]:
        """Compute microprice and basis-point bias vs mid."""
        bid_vol = order_book.bids[0].size if order_book.bids else 0
        ask_vol = order_book.asks[0].size if order_book.asks else 0
        total_vol = bid_vol + ask_vol
        if total_vol > 0:
            microprice = (best_ask * bid_vol + best_bid * ask_vol) / total_vol
            microprice_bias = (microprice - mid) / mid * 10000 if mid > 0 else 0
        else:
            microprice = mid
            microprice_bias = 0
        return float(microprice), float(microprice_bias)

    def _append_history(self, timestamp: pd.Timestamp, mid: float, spread: float) -> None:
        """Append latest top-of-book snapshot to rolling history."""
        self._history.append(
            {"timestamp": timestamp, "mid_price": mid, "spread": spread}
        )

    @staticmethod
    def _compute_queue_ratios(
        order_book: "OrderBook", bid_depth_5: float, ask_depth_5: float
    ) -> tuple[float, float]:
        bid_front = order_book.bids[0].size if order_book.bids else 0.0
        ask_front = order_book.asks[0].size if order_book.asks else 0.0
        bid_queue_ratio = bid_front / bid_depth_5 if bid_depth_5 > 0 else 0.0
        ask_queue_ratio = ask_front / ask_depth_5 if ask_depth_5 > 0 else 0.0
        return float(bid_queue_ratio), float(ask_queue_ratio)

    @staticmethod
    def _build_orderbook_features(
        *,
        timestamp: pd.Timestamp,
        basic: Dict[str, float],
        depth: Dict[str, float],
        derived: Dict[str, float | Optional[float]],
    ) -> OrderBookFeatures:
        return OrderBookFeatures(
            timestamp=timestamp,
            mid_price=basic["mid"],
            spread=basic["spread"],
            spread_bps=basic["spread_bps"],
            best_bid=basic["best_bid"],
            best_ask=basic["best_ask"],
            bid_depth_5=depth["bid_depth_5"],
            ask_depth_5=depth["ask_depth_5"],
            bid_depth_10=depth["bid_depth_10"],
            ask_depth_10=depth["ask_depth_10"],
            depth_imbalance=depth["depth_imbalance"],
            vwap_bid=float(derived["vwap_bid"]),
            vwap_ask=float(derived["vwap_ask"]),
            vwap_mid=float(derived["vwap_mid"]),
            microprice=float(derived["microprice"]),
            microprice_bias=float(derived["microprice_bias"]),
            bid_slope=float(derived["bid_slope"]),
            ask_slope=float(derived["ask_slope"]),
            bid_queue_ratio=float(derived["bid_queue_ratio"]),
            ask_queue_ratio=float(derived["ask_queue_ratio"]),
            realized_vol_1min=derived["realized_vol_1min"],
            realized_vol_5min=derived["realized_vol_5min"],
            trade_flow_imbalance=derived["trade_flow_imbalance"],
            volume_order_imbalance=derived["volume_order_imbalance"],
        )

    def extract(
        self,
        order_book: 'OrderBook',
        recent_trades: Optional[List['Trade']] = None
    ) -> OrderBookFeatures:
        """Extract features from order book and recent trades."""
        timestamp = order_book.timestamp
        basic = self._extract_basic_features(order_book)
        depth = self._extract_depth_features(order_book)
        vwap_bid, vwap_ask = self._vwap(order_book.bids, 5), self._vwap(order_book.asks, 5)
        vwap_mid = (vwap_bid + vwap_ask) / 2
        microprice, microprice_bias = self._compute_microprice_features(order_book, best_bid=basic["best_bid"], best_ask=basic["best_ask"], mid=basic["mid"])
        bid_slope = self._calc_slope(order_book.bids)
        ask_slope = self._calc_slope(order_book.asks)
        bid_queue_ratio, ask_queue_ratio = self._compute_queue_ratios(order_book, depth["bid_depth_5"], depth["ask_depth_5"])
        self._append_history(timestamp, basic["mid"], basic["spread"])
        realized_vol_1min, realized_vol_5min = self._compute_realized_volatility()
        trade_flow_imbalance, volume_order_imbalance = self._compute_trade_flow_features(recent_trades, depth["depth_imbalance"])
        derived: Dict[str, float | Optional[float]] = {
            "vwap_bid": vwap_bid,
            "vwap_ask": vwap_ask,
            "vwap_mid": vwap_mid,
            "microprice": microprice,
            "microprice_bias": microprice_bias,
            "bid_slope": bid_slope,
            "ask_slope": ask_slope,
            "bid_queue_ratio": bid_queue_ratio,
            "ask_queue_ratio": ask_queue_ratio,
            "realized_vol_1min": realized_vol_1min,
            "realized_vol_5min": realized_vol_5min,
            "trade_flow_imbalance": trade_flow_imbalance,
            "volume_order_imbalance": volume_order_imbalance,
        }
        return self._build_orderbook_features(
            timestamp=timestamp,
            basic=basic,
            depth=depth,
            derived=derived,
        )

    def reset(self) -> None:
        """Clear history."""
        self._history.clear()


class FeaturePipeline:
    """
    Pipeline for processing order book data into features.
    """

    def __init__(self):
        self.extractor = OrderBookFeatureExtractor()
        self._features: List[OrderBookFeatures] = []

    def process_stream(
        self,
        order_books: List['OrderBook'],
        trades: Optional[List['Trade']] = None
    ) -> pd.DataFrame:
        """
        Process a stream of order books into feature DataFrame.
        """
        features_list = []

        for i, ob in enumerate(order_books):
            # Get relevant trades for this order book
            recent_trades = None
            if trades:
                ob_time = ob.timestamp
                recent_trades = [
                    t for t in trades
                    if abs((t.timestamp - ob_time).total_seconds()) < 60
                ]

            features = self.extractor.extract(ob, recent_trades)
            features_list.append(features.to_dict())

        return pd.DataFrame(features_list)

    def get_predictive_features(self, df: pd.DataFrame, lookahead: int = 10) -> pd.DataFrame:
        """
        Create features for predicting future price movements.

        Args:
            df: Feature DataFrame
            lookahead: Number of periods to look ahead for target

        Returns:
            DataFrame with additional lag and rolling features
        """
        result = df.copy()

        # Lag features
        for col in ['mid_price', 'spread', 'depth_imbalance', 'microprice_bias']:
            if col in result.columns:
                for lag in [1, 5, 10]:
                    result[f'{col}_lag_{lag}'] = result[col].shift(lag)

        # Rolling statistics
        for window in [10, 30, 60]:
            result[f'spread_ma_{window}'] = result['spread'].rolling(window).mean()
            result[f'imbalance_std_{window}'] = result['depth_imbalance'].rolling(window).std()
            result[f'volume_imbalance_ma_{window}'] = result['depth_imbalance'].rolling(window).mean()

        # Price momentum
        result['returns'] = result['mid_price'].pct_change()
        result['momentum_10'] = result['returns'].rolling(10).sum()
        result['momentum_30'] = result['returns'].rolling(30).sum()

        # Target: future return
        result['target_return'] = result['mid_price'].shift(-lookahead) / result['mid_price'] - 1

        return result.dropna()
