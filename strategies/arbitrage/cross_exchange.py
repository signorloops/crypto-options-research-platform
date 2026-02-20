"""
跨交易所套利策略。

监控同一资产在不同交易所的价格差异，当价差超过阈值时执行套利。

套利公式:
    利润 = |Price_A - Price_B| - 交易成本_A - 交易成本_B - 滑点

适用场景:
    - 同一加密货币在不同交易所的价格差异
    - 需要考虑提币时间、手续费、流动性
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional


@dataclass
class ArbitrageOpportunity:
    """套利机会数据结构。"""
    buy_exchange: str
    sell_exchange: str
    instrument: str
    buy_price: float
    sell_price: float
    spread_bps: float  # 价差 (基点)
    profit_pct: float  # 预期利润率
    timestamp: datetime


@dataclass
class TriangularOpportunity:
    """三角套利机会。"""
    exchange: str
    path: List[str]
    start_amount: float
    end_amount: float
    profit_pct: float
    timestamp: datetime


@dataclass
class ExchangeFees:
    """交易所费率结构。"""
    maker_fee: float  # 挂单费率
    taker_fee: float  # 吃单费率
    withdrawal_fee: float = 0.0  # 提币费
    deposit_fee: float = 0.0  # 充值费


class CrossExchangeArbitrage:
    """
    跨交易所套利策略。

    监控多个交易所的同一交易对，当价差超过阈值时发出信号。
    """

    def __init__(
        self,
        min_spread_bps: float = 50.0,  # 最小价差阈值 (基点)
        min_profit_pct: float = 0.1,  # 最小利润率 (%)
        max_position_size: float = 1.0,  # 最大头寸
        slippage_coeff: float = 0.00005,  # 滑点系数 (每单位仓位)
    ):
        self.min_spread_bps = min_spread_bps
        self.min_profit_pct = min_profit_pct
        self.max_position_size = max_position_size
        self.slippage_coeff = slippage_coeff

        self.exchange_fees: Dict[str, ExchangeFees] = {}
        self.price_cache: Dict[str, Dict[str, float]] = {}
        self.opportunity_callback: Optional[Callable] = None
        self._opportunity_history: List[ArbitrageOpportunity] = []

    def reset(self) -> None:
        """重置策略状态（用于测试）。"""
        self.exchange_fees.clear()
        self.price_cache.clear()
        self._opportunity_history.clear()
        self.opportunity_callback = None

    def set_exchange_fees(self, exchange: str, fees: ExchangeFees) -> None:
        """设置交易所费率。"""
        self.exchange_fees[exchange] = fees

    def on_opportunity(self, callback: Callable[[ArbitrageOpportunity], None]) -> None:
        """设置套利机会回调函数。"""
        self.opportunity_callback = callback

    def update_price(
        self,
        exchange: str,
        instrument: str,
        price: float,
        timestamp: Optional[datetime] = None,
        max_age_ms: float = 100.0
    ) -> None:
        """
        更新价格缓存。

        Args:
            exchange: 交易所名称
            instrument: 交易对 (如 "BTC-USD")
            price: 当前价格
            timestamp: 价格时间戳（交易所时间）
            max_age_ms: 最大允许价格延迟（毫秒）
        """
        from dataclasses import dataclass

        @dataclass
        class PriceEntry:
            price: float
            timestamp: datetime
            received_at: datetime

        now = datetime.now(timezone.utc)

        if instrument not in self.price_cache:
            self.price_cache[instrument] = {}

        entry = PriceEntry(
            price=price,
            timestamp=timestamp or now,
            received_at=now
        )

        # 检查数据新鲜度
        if timestamp:
            age_ms = (now - timestamp).total_seconds() * 1000
            if age_ms > max_age_ms:
                # 忽略过期数据，但保留之前的有效价格
                return

        self.price_cache[instrument][exchange] = entry

        # 检查套利机会
        self._check_arbitrage(instrument)

    def _check_arbitrage(self, instrument: str) -> None:
        """检查特定交易对的套利机会。"""
        price_entries = self.price_cache.get(instrument, {})

        if len(price_entries) < 2:
            return

        # 获取价格和检查时间同步
        now = datetime.now(timezone.utc)
        prices = {}
        for ex, entry in price_entries.items():
            # 检查是否仍然是新鲜数据
            if hasattr(entry, 'received_at'):
                age_ms = (now - entry.received_at).total_seconds() * 1000
                if age_ms > 200:  # 超过200ms视为过期
                    continue
                prices[ex] = entry.price
            else:
                prices[ex] = entry

        if len(prices) < 2:
            return

        # 优化: 排序后只检查最大价差的一对 (O(n log n) 替代 O(n²))
        sorted_prices = sorted(prices.items(), key=lambda x: x[1])
        min_ex, min_price = sorted_prices[0]
        max_ex, max_price = sorted_prices[-1]

        # 只在有套利空间时继续
        if max_price <= min_price:
            return

        buy_ex, buy_price = min_ex, min_price
        sell_ex, sell_price = max_ex, max_price

        # 计算价差 (基点)
        spread_bps = (sell_price - buy_price) / buy_price * 10000

        # 计算费率
        buy_fees = self.exchange_fees.get(buy_ex, ExchangeFees(0.001, 0.001))
        sell_fees = self.exchange_fees.get(sell_ex, ExchangeFees(0.001, 0.001))

        total_fee = buy_fees.taker_fee + sell_fees.taker_fee
        slippage = self._estimate_slippage_pct(self.max_position_size)

        # 计算净利润
        profit_pct = (sell_price - buy_price) / buy_price - total_fee - slippage

        # 检查阈值
        if spread_bps >= self.min_spread_bps and profit_pct >= self.min_profit_pct:
            opportunity = ArbitrageOpportunity(
                buy_exchange=buy_ex,
                sell_exchange=sell_ex,
                instrument=instrument,
                buy_price=buy_price,
                sell_price=sell_price,
                spread_bps=spread_bps,
                profit_pct=profit_pct * 100,
                timestamp=datetime.now(timezone.utc)
            )

            if self.opportunity_callback:
                self.opportunity_callback(opportunity)

    def get_best_opportunities(self, top_n: int = 10) -> List[ArbitrageOpportunity]:
        """
        获取当前最佳套利机会。

        注意: 此方法需要维护机会历史，当前实现为简化版。
        """
        opportunities = []

        for instrument, price_entries in self.price_cache.items():
            # Extract prices from PriceEntry objects
            prices = {}
            for ex, entry in price_entries.items():
                if hasattr(entry, 'price'):
                    prices[ex] = entry.price
                else:
                    prices[ex] = entry

            if len(prices) < 2:
                continue

            # 优化: 排序后只检查最大价差的一对 (O(n log n) 替代 O(n²))
            sorted_prices = sorted(prices.items(), key=lambda x: x[1])
            buy_ex, buy_price = sorted_prices[0]
            sell_ex, sell_price = sorted_prices[-1]

            if sell_price <= buy_price:
                continue

            spread_bps = (sell_price - buy_price) / buy_price * 10000

            if spread_bps >= self.min_spread_bps:
                buy_fees = self.exchange_fees.get(buy_ex, ExchangeFees(0.001, 0.001))
                sell_fees = self.exchange_fees.get(sell_ex, ExchangeFees(0.001, 0.001))
                total_fee = buy_fees.taker_fee + sell_fees.taker_fee + self._estimate_slippage_pct(self.max_position_size)
                profit_pct = (sell_price - buy_price) / buy_price - total_fee
                opportunities.append(ArbitrageOpportunity(
                    buy_exchange=buy_ex,
                    sell_exchange=sell_ex,
                    instrument=instrument,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    spread_bps=spread_bps,
                    profit_pct=profit_pct * 100,
                    timestamp=datetime.now(timezone.utc)
                ))

        # 按价差排序
        opportunities.sort(key=lambda x: x.spread_bps, reverse=True)
        return opportunities[:top_n]

    def calculate_required_capital(
        self,
        opportunity: ArbitrageOpportunity,
        position_size: float
    ) -> Dict[str, float]:
        """
        计算执行套利所需资本。

        Returns:
            {
                'buy_capital': 买入所需资金,
                'sell_collateral': 卖出保证金,
                'total_required': 总需求,
                'expected_profit': 预期利润
            }
        """
        buy_capital = position_size * opportunity.buy_price
        sell_collateral = position_size * opportunity.sell_price * 0.5  # 假设 50% 保证金

        # 扣除费用后的利润
        buy_fees = self.exchange_fees.get(opportunity.buy_exchange, ExchangeFees(0.001, 0.001))
        sell_fees = self.exchange_fees.get(opportunity.sell_exchange, ExchangeFees(0.001, 0.001))

        gross_profit = position_size * (opportunity.sell_price - opportunity.buy_price)
        total_fees = position_size * opportunity.buy_price * buy_fees.taker_fee + \
                     position_size * opportunity.sell_price * sell_fees.taker_fee

        expected_profit = gross_profit - total_fees

        return {
            'buy_capital': buy_capital,
            'sell_collateral': sell_collateral,
            'total_required': buy_capital + sell_collateral,
            'expected_profit': expected_profit,
            'roi_pct': expected_profit / (buy_capital + sell_collateral) * 100
        }

    def estimate_execution_time(self, exchange_a: str, exchange_b: str) -> float:
        """
        估计跨交易所套利执行时间。

        包括:
        - 交易执行时间
        - 提币时间 (如果需要)
        - 资金转移时间

        Returns:
            估计时间 (分钟)
        """
        # 简化估计
        base_execution = 0.5  # 30秒执行交易
        withdrawal_time = 30.0  # 30分钟提币 (BTC典型)

        return base_execution + withdrawal_time

    def _estimate_slippage_pct(self, position_size: float) -> float:
        """Estimate one-way slippage as percentage of notional."""
        return min(0.002, self.slippage_coeff * max(position_size, 0.0))

    def simulate_execution(
        self,
        opportunity: ArbitrageOpportunity,
        position_size: Optional[float] = None,
        latency_ms: float = 80.0
    ) -> Dict[str, float]:
        """
        执行仿真: 估算滑点、延迟冲击和实际利润。
        """
        size = min(position_size or self.max_position_size, self.max_position_size)
        slippage_pct = self._estimate_slippage_pct(size)
        latency_penalty_pct = min(0.001, latency_ms / 1_000_000)  # 简化延迟损耗

        buy_price_eff = opportunity.buy_price * (1 + slippage_pct + latency_penalty_pct)
        sell_price_eff = opportunity.sell_price * (1 - slippage_pct - latency_penalty_pct)

        buy_fees = self.exchange_fees.get(opportunity.buy_exchange, ExchangeFees(0.001, 0.001))
        sell_fees = self.exchange_fees.get(opportunity.sell_exchange, ExchangeFees(0.001, 0.001))
        gross = size * (sell_price_eff - buy_price_eff)
        fees = size * (buy_price_eff * buy_fees.taker_fee + sell_price_eff * sell_fees.taker_fee)
        net = gross - fees
        invested = size * buy_price_eff
        roi_pct = (net / invested * 100) if invested > 0 else 0.0

        return {
            "position_size": float(size),
            "buy_price_effective": float(buy_price_eff),
            "sell_price_effective": float(sell_price_eff),
            "net_profit": float(net),
            "roi_pct": float(roi_pct),
            "latency_ms": float(latency_ms),
            "slippage_pct": float(slippage_pct),
        }

    def check_triangular_arbitrage(
        self,
        exchange: str,
        pair_prices: Dict[str, float],
        start_currency: str = "USDT",
        start_amount: float = 1.0,
        min_profit_pct: float = 0.001
    ) -> Optional[TriangularOpportunity]:
        """
        检查简化三角套利（3条边）:
        path: A/B, B/C, C/A
        pair_prices 使用 "BASE/QUOTE": price 约定。
        """
        currencies = set()
        for pair in pair_prices:
            base, quote = pair.split("/")
            currencies.add(base)
            currencies.add(quote)

        currencies = sorted(currencies)
        for b in currencies:
            for c in currencies:
                if b == c or b == start_currency or c == start_currency:
                    continue
                path = [f"{start_currency}/{b}", f"{b}/{c}", f"{c}/{start_currency}"]
                if not all(p in pair_prices and pair_prices[p] > 0 for p in path):
                    continue

                amount_b = start_amount / pair_prices[path[0]]
                amount_c = amount_b / pair_prices[path[1]]
                end_amount = amount_c * pair_prices[path[2]]
                profit_pct = (end_amount - start_amount) / start_amount

                if profit_pct >= min_profit_pct:
                    return TriangularOpportunity(
                        exchange=exchange,
                        path=path,
                        start_amount=float(start_amount),
                        end_amount=float(end_amount),
                        profit_pct=float(profit_pct * 100),
                        timestamp=datetime.now(timezone.utc)
                    )

        return None
