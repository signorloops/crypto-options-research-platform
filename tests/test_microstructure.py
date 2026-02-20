"""
Tests for microstructure analysis modules.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from core.types import OrderBook, OrderBookLevel, OrderSide, Trade
from research.microstructure.orderbook_features import (
    FeaturePipeline,
    OrderBookFeatureExtractor,
    OrderBookFeatures,
)
from research.microstructure.vpin import (
    OrderFlowToxicityAnalyzer,
    VPINCalculator,
    VPINResult,
)


def _legacy_create_volume_buckets(df: pd.DataFrame, bucket_size: float):
    """Reference implementation matching the previous iterative bucket logic."""
    if df.empty:
        return [], [], [], []

    buckets = []
    current_bucket = {
        "buy_volume": 0.0,
        "sell_volume": 0.0,
        "total_volume": 0.0,
        "timestamp": df["timestamp"].iloc[0],
    }

    prices = df["price"].to_numpy(dtype=float)
    sizes = df["size"].to_numpy(dtype=float)
    sides = df["side"].to_numpy()
    timestamps = df["timestamp"].to_numpy()

    for idx in range(len(df)):
        volume = max(sizes[idx] * prices[idx], 0.0)
        is_buy = sides[idx] == "buy"
        trade_ts = timestamps[idx]

        remaining_volume = volume
        while remaining_volume > 0:
            space_left = bucket_size - current_bucket["total_volume"]
            volume_to_add = min(remaining_volume, space_left)
            if is_buy:
                current_bucket["buy_volume"] += volume_to_add
            else:
                current_bucket["sell_volume"] += volume_to_add
            current_bucket["total_volume"] += volume_to_add
            current_bucket["timestamp"] = trade_ts
            remaining_volume -= volume_to_add

            if current_bucket["total_volume"] >= bucket_size:
                buckets.append(current_bucket.copy())
                current_bucket = {
                    "buy_volume": 0.0,
                    "sell_volume": 0.0,
                    "total_volume": 0.0,
                    "timestamp": trade_ts,
                }

    if current_bucket["total_volume"] >= bucket_size * 0.5:
        buckets.append(current_bucket.copy())

    ts = [b["timestamp"] for b in buckets]
    buy = [b["buy_volume"] for b in buckets]
    sell = [b["sell_volume"] for b in buckets]
    total = [b["total_volume"] for b in buckets]
    return ts, buy, sell, total


class TestVPINCalculator:
    """Test VPIN (Volume-Synchronized Probability of Informed Trading) calculator."""

    def test_initialization(self):
        """Test VPIN calculator initialization."""
        calc = VPINCalculator(volume_bucket_size=1000, num_buckets=50)
        assert calc.volume_bucket_size == 1000
        assert calc.num_buckets == 50

    def test_calculate_with_synthetic_data(self):
        """Test VPIN calculation with synthetic trade data."""
        calc = VPINCalculator(volume_bucket_size=100, num_buckets=10)

        # Create synthetic trade data
        np.random.seed(42)
        n_trades = 1000
        trades = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_trades, freq='1min'),
            'price': np.cumsum(np.random.randn(n_trades) * 0.1) + 50000,
            'size': np.random.uniform(0.1, 2.0, n_trades),
            'side': np.random.choice(['buy', 'sell'], n_trades)
        })

        result = calc.calculate(trades)

        assert isinstance(result, VPINResult)
        assert len(result.vpin_values) > 0
        # VPIN should be between 0 and 1
        assert result.vpin_values.min() >= 0
        assert result.vpin_values.max() <= 1

    def test_vpin_toxicity_detection(self):
        """Test VPIN detects toxic flow."""
        calc = VPINCalculator(volume_bucket_size=100, num_buckets=20)

        # Create data with informed trading pattern
        # Large orders in one direction with price impact
        n = 500
        trades = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1s'),
            'price': 50000 + np.concatenate([np.zeros(n//2), np.ones(n//2) * 100]),
            'size': np.ones(n) * 10,
            'side': ['buy'] * (n//2) + ['sell'] * (n//2)
        })

        result = calc.calculate(trades)

        # VPIN should be elevated during informed trading
        assert len(result.vpin_values) > 0
        # Some VPIN values should be > 0
        assert result.vpin_values.max() > 0

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        calc = VPINCalculator(volume_bucket_size=1000, num_buckets=50)

        # Small dataset
        trades = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'price': np.ones(10) * 50000,
            'size': np.ones(10),
            'side': ['buy'] * 10
        })

        # Should handle gracefully
        result = calc.calculate(trades)
        assert isinstance(result, VPINResult)
        # VPIN is calculated per bucket, not per trade
        # Each trade has volume 50000, bucket_size=100, so each trade creates ~500 buckets
        assert len(result.vpin_values) > 0
        assert result.vpin_values.max() <= 1.0
        assert result.vpin_values.min() >= 0.0

    def test_vpin_result_get_high_toxicity_periods(self):
        """Test VPINResult.get_high_toxicity_periods method."""
        calc = VPINCalculator(volume_bucket_size=100, num_buckets=10)

        # Create trade data
        n_trades = 500
        trades = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_trades, freq='1min'),
            'price': np.ones(n_trades) * 50000,
            'size': np.ones(n_trades),
            'side': np.random.choice(['buy', 'sell'], n_trades)
        })

        result = calc.calculate(trades)
        periods = result.get_high_toxicity_periods(threshold=0.6)

        assert isinstance(periods, list)
        # Each period should be a tuple of (start, end) timestamps
        for period in periods:
            assert isinstance(period, tuple)
            assert len(period) == 2

    def test_bucket_vectorization_matches_reference(self):
        """Vectorized bucket builder should match legacy split semantics."""
        calc = VPINCalculator(volume_bucket_size=100, num_buckets=5)
        trades = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=7, freq="1min"),
                "price": [10, 12, 11, 13, 9, 10, 8],
                "size": [3.0, 20.0, 2.0, 15.0, 1.0, 7.0, 9.0],
                "side": ["buy", "sell", "buy", "sell", "buy", "sell", "buy"],
            }
        )

        ref_ts, ref_buy, ref_sell, ref_total = _legacy_create_volume_buckets(trades, 100)
        ts, buy, sell, total = calc._create_volume_buckets(trades)

        np.testing.assert_array_equal(
            np.array(ts, dtype="datetime64[ns]"),
            np.array(ref_ts, dtype="datetime64[ns]"),
        )
        np.testing.assert_allclose(buy, np.array(ref_buy))
        np.testing.assert_allclose(sell, np.array(ref_sell))
        np.testing.assert_allclose(total, np.array(ref_total))


class TestOrderFlowToxicityAnalyzer:
    """Test OrderFlowToxicityAnalyzer."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = OrderFlowToxicityAnalyzer()
        assert analyzer.vpin_calc is not None

    def test_analyze(self):
        """Test full toxicity analysis."""
        analyzer = OrderFlowToxicityAnalyzer()

        # Create trade data
        n = 200
        trades = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1s'),
            'price': 50000 + np.random.randn(n) * 10,
            'size': np.random.uniform(0.1, 1.0, n),
            'side': np.random.choice(['buy', 'sell'], n)
        })

        result = analyzer.analyze(trades)

        assert isinstance(result, pd.DataFrame)
        assert 'vpin' in result.columns
        assert 'volume_imbalance' in result.columns
        assert 'large_trade_ratio' in result.columns

    def test_detect_anomalies(self):
        """Test anomaly detection."""
        analyzer = OrderFlowToxicityAnalyzer()

        # Create trade data
        n = 200
        trades = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='1s'),
            'price': 50000 + np.random.randn(n) * 10,
            'size': np.random.uniform(0.1, 1.0, n),
            'side': np.random.choice(['buy', 'sell'], n)
        })

        analysis_df = analyzer.analyze(trades)
        anomalies = analyzer.detect_anomalies(analysis_df, zscore_threshold=2.0)

        assert isinstance(anomalies, pd.DataFrame)
        assert 'is_anomaly' in anomalies.columns
        assert 'vpin_zscore' in anomalies.columns


class TestOrderBookFeatureExtractor:
    """Test order book feature extraction."""

    def test_initialization(self):
        """Test extractor initialization."""
        extractor = OrderBookFeatureExtractor()
        assert len(extractor._history) == 0
        assert extractor._max_history == 300

    def test_extract_basic_features(self):
        """Test basic feature extraction."""
        extractor = OrderBookFeatureExtractor()

        # Create order book using proper types
        timestamp = datetime.now()
        order_book = OrderBook(
            timestamp=timestamp,
            instrument='BTC-USD',
            bids=[
                OrderBookLevel(price=49900, size=1.0),
                OrderBookLevel(price=49800, size=2.0),
            ],
            asks=[
                OrderBookLevel(price=50100, size=1.5),
                OrderBookLevel(price=50200, size=2.5),
            ]
        )

        features = extractor.extract(order_book, recent_trades=None)

        assert isinstance(features, OrderBookFeatures)
        assert features.mid_price == pytest.approx(50000.0)
        assert features.spread == pytest.approx(200.0)
        assert features.best_bid == pytest.approx(49900.0)
        assert features.best_ask == pytest.approx(50100.0)

    def test_extract_with_trades(self):
        """Test extraction with trades."""
        extractor = OrderBookFeatureExtractor()

        timestamp = datetime.now()
        order_book = OrderBook(
            timestamp=timestamp,
            instrument='BTC-USD',
            bids=[
                OrderBookLevel(price=49900, size=1.0),
                OrderBookLevel(price=49800, size=2.0),
            ],
            asks=[
                OrderBookLevel(price=50100, size=1.5),
                OrderBookLevel(price=50200, size=2.5),
            ]
        )

        trades = [
            Trade(
                timestamp=timestamp,
                instrument='BTC-USD',
                price=50000,
                size=1.0,
                side=OrderSide.BUY
            )
        ]

        features = extractor.extract(order_book, recent_trades=trades)

        assert isinstance(features, OrderBookFeatures)
        assert features.trade_flow_imbalance is not None
        assert features.volume_order_imbalance is not None

    def test_extract_with_empty_order_book(self):
        """Test extraction with empty order book."""
        extractor = OrderBookFeatureExtractor()

        timestamp = datetime.now()
        order_book = OrderBook(
            timestamp=timestamp,
            instrument='BTC-USD',
            bids=[],
            asks=[]
        )

        features = extractor.extract(order_book, recent_trades=None)

        assert isinstance(features, OrderBookFeatures)
        assert features.mid_price is None or features.mid_price == 0

    def test_reset(self):
        """Test reset clears history."""
        extractor = OrderBookFeatureExtractor()

        timestamp = datetime.now()
        order_book = OrderBook(
            timestamp=timestamp,
            instrument='BTC-USD',
            bids=[OrderBookLevel(price=49900, size=1.0)],
            asks=[OrderBookLevel(price=50100, size=1.5)]
        )

        # Extract to populate history
        extractor.extract(order_book, recent_trades=None)
        assert len(extractor._history) > 0

        # Reset should clear history
        extractor.reset()
        assert len(extractor._history) == 0

    def test_features_to_dict(self):
        """Test OrderBookFeatures.to_dict method."""
        timestamp = datetime.now()
        features = OrderBookFeatures(
            timestamp=timestamp,
            mid_price=50000.0,
            spread=200.0,
            spread_bps=4.0,
            best_bid=49900.0,
            best_ask=50100.0,
            bid_depth_5=10.0,
            ask_depth_5=15.0,
            bid_depth_10=20.0,
            ask_depth_10=25.0,
            depth_imbalance=0.2,
            vwap_bid=49850.0,
            vwap_ask=50150.0,
            vwap_mid=50000.0,
            microprice=50010.0,
            microprice_bias=0.2,
            bid_slope=-100.0,
            ask_slope=100.0,
            bid_queue_ratio=0.5,
            ask_queue_ratio=0.4,
            realized_vol_1min=None,
            realized_vol_5min=None,
            trade_flow_imbalance=None,
            volume_order_imbalance=None
        )

        result = features.to_dict()
        assert isinstance(result, dict)
        assert result['mid_price'] == pytest.approx(50000.0)
        assert result['spread'] == pytest.approx(200.0)


class TestFeaturePipeline:
    """Test FeaturePipeline."""

    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = FeaturePipeline()
        assert pipeline.extractor is not None
        assert pipeline._features == []

    def test_process_stream(self):
        """Test processing a stream of order books."""
        pipeline = FeaturePipeline()

        timestamp = datetime.now()
        order_books = [
            OrderBook(
                timestamp=timestamp + timedelta(seconds=i),
                instrument='BTC-USD',
                bids=[OrderBookLevel(price=49900 - i, size=1.0 + i*0.1)],
                asks=[OrderBookLevel(price=50100 + i, size=1.5 + i*0.1)]
            )
            for i in range(10)
        ]

        result = pipeline.process_stream(order_books, trades=None)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert 'mid_price' in result.columns
        assert 'spread' in result.columns

    def test_get_predictive_features(self):
        """Test creating predictive features."""
        pipeline = FeaturePipeline()

        timestamp = datetime.now()
        order_books = [
            OrderBook(
                timestamp=timestamp + timedelta(seconds=i),
                instrument='BTC-USD',
                bids=[OrderBookLevel(price=49900 - i, size=1.0)],
                asks=[OrderBookLevel(price=50100 + i, size=1.5)]
            )
            for i in range(100)
        ]

        features_df = pipeline.process_stream(order_books, trades=None)
        predictive_df = pipeline.get_predictive_features(features_df, lookahead=10)

        assert isinstance(predictive_df, pd.DataFrame)
        assert 'target_return' in predictive_df.columns
        assert 'returns' in predictive_df.columns
        assert 'momentum_10' in predictive_df.columns
