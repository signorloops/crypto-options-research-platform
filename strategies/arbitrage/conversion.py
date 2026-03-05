"""Conversion and reversal arbitrage based on put-call parity."""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Literal, Optional

import numpy as np


def _time_to_expiry_years(expiry: datetime) -> float:
    """Return time-to-expiry in years using full timestamp precision."""
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max(0.0, (expiry - now).total_seconds() / (365.0 * 24 * 3600))


def _resolve_opportunity_inputs(
    *,
    strategy: "ConversionArbitrage",
    underlying: str,
    call_price: Optional[float],
    put_price: Optional[float],
    spot_price: Optional[float],
    strike: Optional[float],
    expiry: Optional[datetime],
) -> Optional[tuple[float, float, float, float, datetime]]:
    if any(v is None for v in [call_price, put_price, spot_price, strike, expiry]):
        snapshot = strategy.get_latest_snapshot(underlying)
        if snapshot is None:
            return None
        call_price = snapshot["call_price"] if call_price is None else call_price
        put_price = snapshot["put_price"] if put_price is None else put_price
        spot_price = snapshot["spot_price"] if spot_price is None else spot_price
        strike = snapshot["strike"] if strike is None else strike
        expiry = snapshot["expiry"] if expiry is None else expiry
    return float(call_price), float(put_price), float(spot_price), float(strike), expiry


def _strategy_and_profit_from_deviation(
    *,
    deviation: float,
    total_cost: float,
) -> Optional[tuple[Literal["conversion", "reversal"], float]]:
    if deviation > total_cost:
        return "conversion", float(deviation - total_cost)
    if deviation < -total_cost:
        return "reversal", float(-deviation - total_cost)
    return None


@dataclass
class ConversionOpportunity:
    """转换/反转套利机会。"""
    underlying: str
    strike: float
    expiry: datetime
    call_price: float
    put_price: float
    spot_price: float

    strategy: Literal["conversion", "reversal"]
    synthetic_forward: float  # C - P
    actual_forward: float     # S - K*exp(-rT)
    deviation: float          # 偏离程度
    profit: float             # 套利利润
    annualized_return: float  # 年化收益率


def _build_conversion_opportunity(
    *,
    underlying: str,
    strike: float,
    expiry: datetime,
    call_price: float,
    put_price: float,
    spot_price: float,
    strategy: Literal["conversion", "reversal"],
    parity: Dict[str, float],
    deviation: float,
    profit: float,
    annualized_return: float,
) -> ConversionOpportunity:
    return ConversionOpportunity(
        underlying=underlying,
        strike=float(strike),
        expiry=expiry,
        call_price=float(call_price),
        put_price=float(put_price),
        spot_price=float(spot_price),
        strategy=strategy,
        synthetic_forward=parity["synthetic_forward"],
        actual_forward=parity["theoretical_forward"],
        deviation=deviation,
        profit=profit,
        annualized_return=annualized_return,
    )


class ConversionArbitrage:
    """Conversion/reversal arbitrage strategy."""

    def __init__(
        self,
        risk_free_rate: float = 0.05,
        min_profit_threshold: float = 0.001,  # 0.1% 最小利润
        transaction_cost: float = 0.002,       # 单边交易成本
        staking_yield: float = 0.0,            # ETH staking yield 等持有收益
    ):
        self.risk_free_rate = risk_free_rate
        self.min_profit = min_profit_threshold
        self.transaction_cost = transaction_cost
        self.staking_yield = staking_yield
        self._market_snapshot: Dict[str, Dict[str, float]] = {}

    def update_market_snapshot(
        self,
        underlying: str,
        call_price: float,
        put_price: float,
        spot_price: float,
        strike: float,
        expiry: datetime
    ) -> None:
        """Streaming更新: 维护最新期权/现货报价快照。"""
        self._market_snapshot[underlying] = {
            "call_price": float(call_price),
            "put_price": float(put_price),
            "spot_price": float(spot_price),
            "strike": float(strike),
            "expiry": expiry,
        }

    def get_latest_snapshot(self, underlying: str) -> Optional[Dict[str, float]]:
        return self._market_snapshot.get(underlying)

    def calculate_parity_deviation(
        self,
        call_price: float,
        put_price: float,
        spot_price: float,
        strike: float,
        expiry: datetime,
        carry_yield: Optional[float] = None
    ) -> Dict[str, float]:
        """计算期权平价偏离。"""
        # 合成远期价格
        synthetic_fwd = call_price - put_price

        # 理论远期价格
        T = _time_to_expiry_years(expiry)
        q = self.staking_yield if carry_yield is None else carry_yield
        theoretical_fwd = spot_price * np.exp(-q * T) - strike * np.exp(-self.risk_free_rate * T)

        # 偏离
        deviation = synthetic_fwd - theoretical_fwd
        deviation_pct = deviation / spot_price if spot_price > 0 else 0

        return {
            'synthetic_forward': synthetic_fwd,
            'theoretical_forward': theoretical_fwd,
            'deviation': deviation,
            'deviation_pct': deviation_pct,
            'time_to_expiry': T
        }

    def check_opportunity(
        self,
        underlying: str,
        call_price: Optional[float] = None,
        put_price: Optional[float] = None,
        spot_price: Optional[float] = None,
        strike: Optional[float] = None,
        expiry: Optional[datetime] = None
    ) -> Optional[ConversionOpportunity]:
        """检查是否存在转换/反转套利机会。"""
        resolved = _resolve_opportunity_inputs(strategy=self, underlying=underlying, call_price=call_price, put_price=put_price, spot_price=spot_price, strike=strike, expiry=expiry)
        if resolved is None:
            return None
        call_price, put_price, spot_price, strike, expiry = resolved
        parity = self.calculate_parity_deviation(call_price, put_price, spot_price, strike, expiry)
        deviation = parity['deviation']
        T = parity['time_to_expiry']
        if spot_price <= 0 or T <= 0:
            return None
        total_cost = 3 * self.transaction_cost * spot_price
        strategy_profit = _strategy_and_profit_from_deviation(deviation=deviation, total_cost=total_cost)
        if strategy_profit is None:
            return None
        strategy, profit = strategy_profit
        if profit / spot_price < self.min_profit:
            return None
        annualized_return = (profit / spot_price) / T
        return _build_conversion_opportunity(
            underlying=underlying,
            strike=strike,
            expiry=expiry,
            call_price=call_price,
            put_price=put_price,
            spot_price=spot_price,
            strategy=strategy,
            parity=parity,
            deviation=deviation,
            profit=profit,
            annualized_return=annualized_return,
        )

    def get_hedge_position(self, opp: ConversionOpportunity) -> Dict[str, float]:
        """
        获取对冲头寸。

        Returns:
            {
                'call': 看涨头寸 (+买入, -卖出),
                'put': 看跌头寸,
                'underlying': 标的头寸
            }
        """
        if opp.strategy == "conversion":
            return {
                'call': -1,      # 卖出看涨
                'put': 1,        # 买入看跌
                'underlying': 1  # 买入标的
            }
        else:  # reversal
            return {
                'call': 1,       # 买入看涨
                'put': -1,       # 卖出看跌
                'underlying': -1 # 卖出标的
            }

    def calculate_margin_requirement(
        self,
        opp: ConversionOpportunity
    ) -> float:
        """
        计算保证金需求。

        Conversion:
        - 卖出看涨: 需要标的作为备兑 (Covered Call)
        - 买入看跌: 支付权利金
        - 买入标的: 支付全款

        Reversal:
        - 卖出看跌: 需要保证金
        - 其他: 标准支付
        """
        if opp.strategy == "conversion":
            # 备兑看涨 + 买入看跌
            # 最大风险有限 (put strike 以下)
            return opp.strike * 0.2  # 约 20% 保证金
        else:
            # 卖出看跌需要较高保证金
            naked_put_margin = opp.strike * 0.2
            return naked_put_margin + opp.call_price + opp.put_price

    def calculate_pnl_scenarios(
        self,
        opp: ConversionOpportunity,
        spot_at_expiry: float
    ) -> float:
        """
        计算到期时在不同现货价格下的 P&L。

        Args:
            spot_at_expiry: 到期现货价格

        Returns:
            净利润/亏损
        """
        if opp.strategy == "conversion":
            # 卖出看涨 (义务): -max(S-K, 0)
            call_pnl = -max(spot_at_expiry - opp.strike, 0) + opp.call_price
            # 买入看跌 (权利): max(K-S, 0)
            put_pnl = max(opp.strike - spot_at_expiry, 0) - opp.put_price
            # 买入标的
            underlying_pnl = spot_at_expiry - opp.spot_price
        else:
            # 买入看涨
            call_pnl = max(spot_at_expiry - opp.strike, 0) - opp.call_price
            # 卖出看跌
            put_pnl = -max(opp.strike - spot_at_expiry, 0) + opp.put_price
            # 卖出标的
            underlying_pnl = opp.spot_price - spot_at_expiry

        total_pnl = call_pnl + put_pnl + underlying_pnl

        # 扣除交易成本
        total_pnl -= 3 * self.transaction_cost * opp.spot_price

        return total_pnl

    def verify_arbitrage_bounds(
        self,
        call_price: float,
        put_price: float,
        spot_price: float,
        strike: float,
        expiry: datetime
    ) -> Dict[str, bool]:
        """
        验证期权价格是否满足无套利边界。

        基本边界条件:
        1. C >= max(S - K*exp(-rT), 0)
        2. P >= max(K*exp(-rT) - S, 0)
        3. C - P = S - K*exp(-rT) (平价公式)
        """
        T = _time_to_expiry_years(expiry)
        pv_strike = strike * np.exp(-self.risk_free_rate * T)

        # 看涨期权下界
        call_lower_bound = max(spot_price - pv_strike, 0)
        call_bound_ok = call_price >= call_lower_bound

        # 看跌期权下界
        put_lower_bound = max(pv_strike - spot_price, 0)
        put_bound_ok = put_price >= put_lower_bound

        # 平价偏离
        parity_deviation = abs(call_price - put_price - spot_price + pv_strike)
        parity_ok = parity_deviation < 3 * self.transaction_cost * spot_price

        return {
            'call_lower_bound_satisfied': call_bound_ok,
            'put_lower_bound_satisfied': put_bound_ok,
            'parity_satisfied': parity_ok,
            'all_bounds_satisfied': call_bound_ok and put_bound_ok and parity_ok
        }
