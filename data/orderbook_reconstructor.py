"""
订单簿重建（Order Book Reconstruction）算法

处理增量更新，维护本地订单簿状态，检测丢包并触发重新同步。
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict

from core.types import OrderBook, OrderBookLevel

logger = logging.getLogger(__name__)


@dataclass
class OrderBookDelta:
    """订单簿增量更新"""
    price: float
    size: float  # 0表示删除
    side: str  # 'bid' 或 'ask'
    sequence: int
    timestamp: datetime


@dataclass
class ReconstructionState:
    """重建状态"""
    bids: Dict[float, float] = field(default_factory=dict)
    asks: Dict[float, float] = field(default_factory=dict)
    last_sequence: Optional[int] = None
    last_timestamp: Optional[datetime] = None
    is_synchronized: bool = False
    gap_detected: bool = False


class OrderBookReconstructor:
    """
    订单簿重建器

    维护本地订单簿副本，处理增量更新，验证序列号连续性。
    """

    def __init__(self, instrument: str, max_price_levels: int = 100):
        self.instrument = instrument
        self.max_price_levels = max_price_levels
        self.state = ReconstructionState()
        self._on_gap_callbacks: List[Callable[[int, int], None]] = []
        self._snapshot_received = False

    def add_gap_callback(self, callback: Callable[[int, int], None]) -> None:
        """添加丢包检测回调 (prev_seq, current_seq)"""
        self._on_gap_callbacks.append(callback)

    def initialize_snapshot(self, order_book: OrderBook, sequence: int) -> None:
        """使用快照初始化订单簿"""
        self.state.bids = {level.price: level.size for level in order_book.bids}
        self.state.asks = {level.price: level.size for level in order_book.asks}
        self.state.last_sequence = sequence
        self.state.last_timestamp = order_book.timestamp
        self.state.is_synchronized = True
        self.state.gap_detected = False
        self._snapshot_received = True
        # 修剪价格级别
        self._trim_price_levels()
        logger.info(f"{self.instrument}: 订单簿快照初始化完成，序列号 {sequence}")

    def apply_delta(self, delta: OrderBookDelta) -> bool:
        """
        应用增量更新

        Args:
            delta: 增量更新数据

        Returns:
            bool: 是否成功应用
        """
        # 检查序列号连续性
        if self.state.last_sequence is not None:
            expected_seq = self.state.last_sequence + 1
            if delta.sequence != expected_seq:
                logger.error(
                    f"{self.instrument}: 序列号间隙 detected! "
                    f"期望: {expected_seq}, 实际: {delta.sequence}"
                )
                self.state.gap_detected = True
                self.state.is_synchronized = False

                # 触发回调
                for callback in self._on_gap_callbacks:
                    try:
                        callback(self.state.last_sequence, delta.sequence)
                    except Exception as e:
                        logger.error(f"Gap callback error: {e}")

                return False

        # 应用更新
        target_side = self.state.bids if delta.side == 'bid' else self.state.asks

        if delta.size <= 0:
            # 删除价格级别
            target_side.pop(delta.price, None)
        else:
            # 添加或更新价格级别
            target_side[delta.price] = delta.size

        # 限制价格级别数量
        self._trim_price_levels()

        self.state.last_sequence = delta.sequence
        self.state.last_timestamp = delta.timestamp

        return True

    def apply_deltas(self, deltas: List[OrderBookDelta]) -> Tuple[int, int]:
        """
        批量应用增量更新

        Returns:
            (成功数量, 失败数量)
        """
        success_count = 0
        fail_count = 0

        for delta in deltas:
            if self.apply_delta(delta):
                success_count += 1
            else:
                fail_count += 1

        return success_count, fail_count

    def _trim_price_levels(self) -> None:
        """限制价格级别数量，保留最优价格"""
        if len(self.state.bids) > self.max_price_levels:
            # 保留最高bid价格
            sorted_bids = sorted(self.state.bids.items(), key=lambda x: x[0], reverse=True)
            self.state.bids = dict(sorted_bids[:self.max_price_levels])

        if len(self.state.asks) > self.max_price_levels:
            # 保留最低ask价格
            sorted_asks = sorted(self.state.asks.items(), key=lambda x: x[0])
            self.state.asks = dict(sorted_asks[:self.max_price_levels])

    def get_order_book(self) -> OrderBook:
        """获取当前订单簿快照"""
        bids = [
            OrderBookLevel(price=price, size=size)
            for price, size in sorted(self.state.bids.items(), key=lambda x: x[0], reverse=True)
        ]
        asks = [
            OrderBookLevel(price=price, size=size)
            for price, size in sorted(self.state.asks.items(), key=lambda x: x[0])
        ]

        return OrderBook(
            timestamp=self.state.last_timestamp or datetime.now(),
            instrument=self.instrument,
            bids=bids,
            asks=asks
        )

    def check_health(self) -> Dict[str, any]:
        """检查重建器健康状态"""
        return {
            'instrument': self.instrument,
            'is_synchronized': self.state.is_synchronized,
            'gap_detected': self.state.gap_detected,
            'last_sequence': self.state.last_sequence,
            'bid_levels': len(self.state.bids),
            'ask_levels': len(self.state.asks),
            'snapshot_received': self._snapshot_received
        }

    def reset(self) -> None:
        """重置状态"""
        self.state = ReconstructionState()
        self._snapshot_received = False
        logger.info(f"{self.instrument}: 订单簿重建器已重置")


class MultiInstrumentReconstructor:
    """管理多个合约的订单簿重建"""

    def __init__(self, max_price_levels: int = 100):
        self.reconstructors: Dict[str, OrderBookReconstructor] = {}
        self.max_price_levels = max_price_levels

    def get_or_create(self, instrument: str) -> OrderBookReconstructor:
        """获取或创建重建器"""
        if instrument not in self.reconstructors:
            self.reconstructors[instrument] = OrderBookReconstructor(
                instrument, self.max_price_levels
            )
        return self.reconstructors[instrument]

    def remove(self, instrument: str) -> None:
        """移除重建器"""
        if instrument in self.reconstructors:
            del self.reconstructors[instrument]

    def get_all_health(self) -> Dict[str, Dict]:
        """获取所有重建器健康状态"""
        return {
            instrument: recon.check_health()
            for instrument, recon in self.reconstructors.items()
        }

    def reset_all(self) -> None:
        """重置所有重建器"""
        for recon in self.reconstructors.values():
            recon.reset()
