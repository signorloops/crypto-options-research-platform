"""
期权盒式套利 (Box Spread Arbitrage)。

盒式套利利用不同执行价格的看涨和看跌期权构建合成远期合约，
当合成远期价格偏离理论值时产生套利机会。

盒式套利组合:
    Long Box: 买入低执行价合成远期 + 卖出高执行价合成远期
    Short Box: 卖出低执行价合成远期 + 买入高执行价合成远期

无风险收益:
    收益 = K_high - K_low (到期时确定)
    成本 = 期权权利金差额
    利润 = (K_high - K_low) - 成本

适用场景:
    - 利率套利 (盒式价格反映隐含利率)
    - 边界情况套利
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np


def _time_to_expiry_years(expiry: datetime) -> float:
    """Return time-to-expiry in years using full timestamp precision."""
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max(0.0, (expiry - now).total_seconds() / (365.0 * 24 * 3600))


@dataclass
class BoxSpread:
    """盒式套利组合。"""
    low_strike: float
    high_strike: float
    expiry: datetime

    # 低执行价合成远期: 买入看涨 + 卖出看跌
    long_call_low: float   # 买入低执行价看涨
    short_put_low: float   # 卖出低执行价看跌

    # 高执行价合成远期: 卖出看涨 + 买入看跌
    short_call_high: float  # 卖出高执行价看涨
    long_put_high: float    # 买入高执行价看跌

    @property
    def net_premium(self) -> float:
        """净权利金支出 (成本)。"""
        return (self.long_call_low - self.short_put_low -
                self.short_call_high + self.long_put_high)

    @property
    def box_value_at_expiry(self) -> float:
        """到期时盒式组合价值 (确定值)。"""
        return self.high_strike - self.low_strike

    @property
    def profit(self) -> float:
        """套利利润。"""
        return self.box_value_at_expiry - self.net_premium

    @property
    def implied_rate(self) -> float:
        """隐含的无风险利率。"""
        spread_width = self.high_strike - self.low_strike
        if spread_width <= 0:
            return 0.0

        if self.net_premium <= 0:
            return 0.0

        T = _time_to_expiry_years(self.expiry)
        if T <= 0:
            return 0.0

        # Box no-arbitrage relation: premium = (K_high - K_low) * exp(-rT)
        ratio = self.net_premium / spread_width
        if ratio <= 0:
            return 0.0

        r = -np.log(ratio) / T
        return float(r)


@dataclass
class BoxOpportunity:
    """盒式套利机会。"""
    underlying: str
    expiry: datetime
    low_strike: float
    high_strike: float
    net_cost: float
    payoff: float
    profit: float
    implied_rate: float
    annualized_return: float
    box_type: str = "long_box"
    liquidity_score: float = 1.0


class OptionBoxArbitrage:
    """
    期权盒式套利策略。

    识别盒式价格偏离理论值的套利机会。
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        min_annualized_return: float = 0.02,  # 2% 最小年化
        transaction_cost_per_leg: float = 0.001,  # 每条腿成本
        min_leg_liquidity: float = 1.0,
    ):
        self.risk_free_rate = risk_free_rate
        self.min_annualized_return = min_annualized_return
        self.transaction_cost = transaction_cost_per_leg
        self.min_leg_liquidity = min_leg_liquidity

    def build_box(
        self,
        low_strike: float,
        high_strike: float,
        expiry: datetime,
        option_prices: Dict[str, float]
    ) -> Optional[BoxSpread]:
        """
        构建盒式套利组合。

        Args:
            low_strike: 低执行价
            high_strike: 高执行价
            expiry: 到期日
            option_prices: {
                'call_low': 低执行价看涨价格,
                'put_low': 低执行价看跌价格,
                'call_high': 高执行价看涨价格,
                'put_high': 高执行价看跌价格
            }

        Returns:
            BoxSpread 对象或 None
        """
        required = ['call_low', 'put_low', 'call_high', 'put_high']
        if not all(k in option_prices for k in required):
            return None

        if low_strike <= 0 or high_strike <= 0 or low_strike >= high_strike:
            return None

        if any(not np.isfinite(option_prices[k]) or option_prices[k] < 0 for k in required):
            return None

        return BoxSpread(
            low_strike=low_strike,
            high_strike=high_strike,
            expiry=expiry,
            long_call_low=option_prices['call_low'],
            short_put_low=option_prices['put_low'],
            short_call_high=option_prices['call_high'],
            long_put_high=option_prices['put_high']
        )

    def find_arbitrage(
        self,
        box: BoxSpread,
        underlying: str = "",
        liquidity_score: float = 1.0
    ) -> Optional[BoxOpportunity]:
        """
        检查盒式组合是否存在套利机会。

        Args:
            box: BoxSpread 对象
            underlying: 标的资产名称

        Returns:
            BoxOpportunity 或 None
        """
        fees = 4 * self.transaction_cost
        payoff = box.box_value_at_expiry

        # Long box: pay net_premium, receive payoff
        long_cost = box.net_premium + fees
        long_profit = payoff - long_cost

        # Short box: receive net_premium, pay payoff at expiry
        short_profit = (box.net_premium - fees) - payoff

        if long_profit <= 0 and short_profit <= 0:
            return None

        if short_profit > long_profit:
            box_type = "short_box"
            profit = short_profit
            net_cost = -box.net_premium + fees  # margin-like capital usage approximation
            capital_base = max(payoff, 1e-12)
        else:
            box_type = "long_box"
            profit = long_profit
            net_cost = long_cost
            capital_base = max(long_cost, 1e-12)

        # 计算年化收益率
        T = _time_to_expiry_years(box.expiry)
        if T <= 0:
            return None

        if net_cost <= 0:
            annualized_return = float("inf")
        else:
            annualized_return = profit / capital_base / T

        if annualized_return < self.min_annualized_return:
            return None

        return BoxOpportunity(
            underlying=underlying,
            expiry=box.expiry,
            low_strike=box.low_strike,
            high_strike=box.high_strike,
            net_cost=net_cost,
            payoff=payoff,
            profit=profit,
            implied_rate=box.implied_rate,
            annualized_return=annualized_return,
            box_type=box_type,
            liquidity_score=liquidity_score
        )

    def _liquidity_score(self, option_chain: Dict[float, Dict[str, float]], low_k: float, high_k: float) -> float:
        """Score leg liquidity using provided call/put sizes."""
        low = option_chain.get(low_k, {})
        high = option_chain.get(high_k, {})
        legs = [
            float(low.get("call_size", low.get("size", 0.0))),
            float(low.get("put_size", low.get("size", 0.0))),
            float(high.get("call_size", high.get("size", 0.0))),
            float(high.get("put_size", high.get("size", 0.0))),
        ]
        if any(v <= 0 for v in legs):
            return 0.0
        return float(min(1.0, min(legs) / max(self.min_leg_liquidity, 1e-12)))

    def scan_strikes(
        self,
        strikes: List[float],
        expiry: datetime,
        option_chain: Dict[float, Dict[str, float]],
        underlying: str = ""
    ) -> List[BoxOpportunity]:
        """
        扫描所有执行价组合寻找套利机会。

        Args:
            strikes: 可用执行价列表
            expiry: 到期日
            option_chain: {strike: {'call': price, 'put': price}}
            underlying: 标的资产

        Returns:
            套利机会列表
        """
        opportunities = []

        # 遍历所有执行价对
        for i, low_k in enumerate(strikes):
            for high_k in strikes[i+1:]:
                low_chain = option_chain.get(low_k, {})
                high_chain = option_chain.get(high_k, {})

                prices = {
                    'call_low': low_chain.get('call', 0),
                    'put_low': low_chain.get('put', 0),
                    'call_high': high_chain.get('call', 0),
                    'put_high': high_chain.get('put', 0)
                }

                if any(p == 0 for p in prices.values()):
                    continue

                box = self.build_box(low_k, high_k, expiry, prices)
                if box:
                    liquidity_score = self._liquidity_score(option_chain, low_k, high_k)
                    if liquidity_score <= 0:
                        continue
                    opp = self.find_arbitrage(box, underlying, liquidity_score=liquidity_score)
                    if opp:
                        opportunities.append(opp)

        # 按年化收益率排序
        opportunities.sort(key=lambda x: x.annualized_return, reverse=True)
        return opportunities

    def calculate_boundaries(self, box: BoxSpread) -> Dict[str, float]:
        """
        计算盒式套利的边界条件。

        根据无套利原理:
        - 盒式价格应在 [0, K_high - K_low] 之间
        - 盒式价格反映隐含利率
        """
        max_value = box.high_strike - box.low_strike
        min_value = 0

        # 考虑利率的边界
        T = _time_to_expiry_years(box.expiry)

        # 理论价格 (无套利)
        theoretical_price = max_value * np.exp(-self.risk_free_rate * T)

        return {
            'min_price': min_value,
            'max_price': max_value,
            'theoretical_price': theoretical_price,
            'actual_price': box.net_premium,
            'deviation': box.net_premium - theoretical_price,
            'arbitrage_free_range': (
                theoretical_price * 0.99,
                theoretical_price * 1.01
            )
        }
