"""
测试订单簿重建算法
"""
import pytest
import numpy as np
from datetime import datetime, timezone

from data.orderbook_reconstructor import (
    OrderBookReconstructor, MultiInstrumentReconstructor,
    OrderBookDelta, ReconstructionState
)
from core.types import OrderBook, OrderBookLevel


class TestOrderBookReconstructor:
    """测试订单簿重建器"""

    @pytest.fixture
    def reconstructor(self):
        return OrderBookReconstructor("BTC-PERPETUAL")

    def test_initialize_snapshot(self, reconstructor):
        """测试快照初始化"""
        ob = OrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument="BTC-PERPETUAL",
            bids=[OrderBookLevel(50000, 1.0), OrderBookLevel(49999, 2.0)],
            asks=[OrderBookLevel(50001, 1.5), OrderBookLevel(50002, 2.5)]
        )

        reconstructor.initialize_snapshot(ob, sequence=100)

        assert reconstructor.state.is_synchronized
        assert reconstructor.state.last_sequence == 100
        assert len(reconstructor.state.bids) == 2
        assert len(reconstructor.state.asks) == 2

    def test_apply_delta_add(self, reconstructor):
        """测试添加增量更新"""
        # 初始化
        ob = OrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument="BTC-PERPETUAL",
            bids=[OrderBookLevel(50000, 1.0)],
            asks=[OrderBookLevel(50001, 1.5)]
        )
        reconstructor.initialize_snapshot(ob, sequence=100)

        # 添加新的bid
        delta = OrderBookDelta(
            price=49999,
            size=2.0,
            side='bid',
            sequence=101,
            timestamp=datetime.now(timezone.utc)
        )

        success = reconstructor.apply_delta(delta)
        assert success
        assert reconstructor.state.last_sequence == 101
        assert reconstructor.state.bids[49999] == 2.0

    def test_apply_delta_remove(self, reconstructor):
        """测试删除增量更新"""
        ob = OrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument="BTC-PERPETUAL",
            bids=[OrderBookLevel(50000, 1.0), OrderBookLevel(49999, 2.0)],
            asks=[OrderBookLevel(50001, 1.5)]
        )
        reconstructor.initialize_snapshot(ob, sequence=100)

        # 删除49999
        delta = OrderBookDelta(
            price=49999,
            size=0,  # size=0表示删除
            side='bid',
            sequence=101,
            timestamp=datetime.now(timezone.utc)
        )

        success = reconstructor.apply_delta(delta)
        assert success
        assert 49999 not in reconstructor.state.bids

    def test_gap_detection(self, reconstructor):
        """测试丢包检测"""
        ob = OrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument="BTC-PERPETUAL",
            bids=[OrderBookLevel(50000, 1.0)],
            asks=[OrderBookLevel(50001, 1.5)]
        )
        reconstructor.initialize_snapshot(ob, sequence=100)

        # 跳过序列号101，直接发送102
        delta = OrderBookDelta(
            price=49999,
            size=2.0,
            side='bid',
            sequence=102,  # 应该是101
            timestamp=datetime.now(timezone.utc)
        )

        success = reconstructor.apply_delta(delta)
        assert not success  # 应该失败
        assert reconstructor.state.gap_detected
        assert not reconstructor.state.is_synchronized

    def test_gap_callback(self, reconstructor):
        """测试丢包回调"""
        gap_detected = []

        def on_gap(prev_seq, curr_seq):
            gap_detected.append((prev_seq, curr_seq))

        reconstructor.add_gap_callback(on_gap)

        ob = OrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument="BTC-PERPETUAL",
            bids=[OrderBookLevel(50000, 1.0)],
            asks=[OrderBookLevel(50001, 1.5)]
        )
        reconstructor.initialize_snapshot(ob, sequence=100)

        # 制造gap
        delta = OrderBookDelta(
            price=49999,
            size=2.0,
            side='bid',
            sequence=105,  # 跳过了101-104
            timestamp=datetime.now(timezone.utc)
        )

        reconstructor.apply_delta(delta)

        assert len(gap_detected) == 1
        assert gap_detected[0] == (100, 105)

    def test_get_order_book(self, reconstructor):
        """测试获取订单簿"""
        ob = OrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument="BTC-PERPETUAL",
            bids=[
                OrderBookLevel(50000, 1.0),
                OrderBookLevel(49999, 2.0),
                OrderBookLevel(49998, 3.0)
            ],
            asks=[OrderBookLevel(50001, 1.5)]
        )
        reconstructor.initialize_snapshot(ob, sequence=100)

        result = reconstructor.get_order_book()

        assert result.instrument == "BTC-PERPETUAL"
        # bids应该按价格降序排列
        assert result.bids[0].price == 50000
        assert result.bids[1].price == 49999
        assert result.bids[2].price == 49998

    def test_trim_price_levels(self, reconstructor):
        """测试价格级别数量限制"""
        reconstructor.max_price_levels = 3

        bids = [OrderBookLevel(50000 - i, float(i)) for i in range(5)]
        ob = OrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument="BTC-PERPETUAL",
            bids=bids,
            asks=[]
        )
        reconstructor.initialize_snapshot(ob, sequence=100)

        # 只保留最优的3个
        assert len(reconstructor.state.bids) == 3
        assert 50000 in reconstructor.state.bids  # 最优价格应该保留

    def test_reset(self, reconstructor):
        """测试重置"""
        ob = OrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument="BTC-PERPETUAL",
            bids=[OrderBookLevel(50000, 1.0)],
            asks=[OrderBookLevel(50001, 1.5)]
        )
        reconstructor.initialize_snapshot(ob, sequence=100)

        reconstructor.reset()

        assert not reconstructor.state.is_synchronized
        assert reconstructor.state.last_sequence is None
        assert len(reconstructor.state.bids) == 0
        assert len(reconstructor.state.asks) == 0


class TestMultiInstrumentReconstructor:
    """测试多合约重建器"""

    def test_get_or_create(self):
        """测试获取或创建重建器"""
        multi = MultiInstrumentReconstructor()

        recon1 = multi.get_or_create("BTC-PERPETUAL")
        recon2 = multi.get_or_create("BTC-PERPETUAL")
        recon3 = multi.get_or_create("ETH-PERPETUAL")

        assert recon1 is recon2  # 同一个合约应该返回同一个实例
        assert recon1 is not recon3  # 不同合约应该返回不同实例

    def test_remove(self):
        """测试移除重建器"""
        multi = MultiInstrumentReconstructor()

        recon = multi.get_or_create("BTC-PERPETUAL")
        multi.remove("BTC-PERPETUAL")

        recon2 = multi.get_or_create("BTC-PERPETUAL")
        assert recon is not recon2  # 应该创建新的实例

    def test_get_all_health(self):
        """测试获取所有健康状态"""
        multi = MultiInstrumentReconstructor()

        # 创建两个重建器
        btc_recon = multi.get_or_create("BTC-PERPETUAL")
        eth_recon = multi.get_or_create("ETH-PERPETUAL")

        # 初始化BTC
        ob = OrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument="BTC-PERPETUAL",
            bids=[OrderBookLevel(50000, 1.0)],
            asks=[OrderBookLevel(50001, 1.5)]
        )
        btc_recon.initialize_snapshot(ob, sequence=100)

        health = multi.get_all_health()

        assert "BTC-PERPETUAL" in health
        assert "ETH-PERPETUAL" in health
        assert health["BTC-PERPETUAL"]["is_synchronized"] is True
        assert health["ETH-PERPETUAL"]["is_synchronized"] is False

    def test_reset_all(self):
        """测试重置所有"""
        multi = MultiInstrumentReconstructor()

        btc_recon = multi.get_or_create("BTC-PERPETUAL")
        ob = OrderBook(
            timestamp=datetime.now(timezone.utc),
            instrument="BTC-PERPETUAL",
            bids=[OrderBookLevel(50000, 1.0)],
            asks=[OrderBookLevel(50001, 1.5)]
        )
        btc_recon.initialize_snapshot(ob, sequence=100)

        multi.reset_all()

        assert not btc_recon.state.is_synchronized


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
