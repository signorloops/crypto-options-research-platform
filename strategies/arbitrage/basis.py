"""
期现套利策略 (Basis Trading / Cash-and-Carry Arbitrage)。

利用现货和期货之间的价差进行套利。当基差超过持有成本时，
可以锁定无风险利润。

套利公式:
    理论期货价格 = 现货价格 * exp((r - q) * T)
    基差 = 期货价格 - 理论期货价格

其中:
    r = 无风险利率
    q = 分红率/资金费率
    T = 到期时间 (年化)

策略类型:
1. 正基差套利 (期货溢价): 卖期货 + 买现货
2. 负基差套利 (期货折价): 买期货 + 卖现货
"""
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


@dataclass
class BasisOpportunity:
    """期现套利机会。"""
    instrument: str
    spot_price: float
    futures_price: float
    expiry: datetime
    basis: float  # 基差 (期货 - 现货)
    basis_pct: float  # 基差百分比
    annualized_return: float  # 年化收益率
    strategy: Literal["long_basis", "short_basis"]  # 策略方向
    required_capital: float
    expected_profit: float


class BasisArbitrage:
    """
    期现套利策略。

    监控现货和期货的价差，当基差超过阈值时生成交易信号。
    """

    def __init__(
        self,
        risk_free_rate: float = 0.05,  # 无风险利率 5%
        min_annualized_return: float = 0.10,  # 最小年化收益率 10%
        funding_cost: float = 0.0001,  # 资金费率 (每小时)
        transaction_cost: float = 0.001,  # 交易成本 0.1%
        margin_requirement_ratio: float = 0.1,
        liquidation_buffer_pct: float = 0.15,
    ):
        self.risk_free_rate = risk_free_rate
        self.min_annualized_return = min_annualized_return
        self.funding_cost = funding_cost
        self.transaction_cost = transaction_cost
        self.margin_requirement_ratio = margin_requirement_ratio
        self.liquidation_buffer_pct = liquidation_buffer_pct

        self.spot_prices: Dict[str, float] = {}
        self.futures_prices: Dict[str, float] = {}
        self.futures_expiry: Dict[str, datetime] = {}
        self.funding_rate_history: Dict[str, list] = {}

    def update_spot_price(self, instrument: str, price: float) -> None:
        """更新现货价格。"""
        if price <= 0 or not np.isfinite(price):
            raise ValueError(f"Invalid spot price for {instrument}: {price}")
        self.spot_prices[instrument] = price

    def update_futures_price(
        self,
        instrument: str,
        price: float,
        expiry: datetime
    ) -> None:
        """更新期货价格。"""
        if price <= 0 or not np.isfinite(price):
            raise ValueError(f"Invalid futures price for {instrument}: {price}")
        self.futures_prices[instrument] = price
        self.futures_expiry[instrument] = expiry

    def calculate_fair_value(
        self,
        spot: float,
        expiry: datetime,
        funding_rate: float = 0.0
    ) -> float:
        """
        计算期货理论价格 (持有成本模型)。

        F = S * exp((r - q) * T)

        Args:
            spot: 现货价格
            expiry: 到期时间
            funding_rate: 资金费率 (年化)

        Returns:
            理论期货价格
        """
        T = _time_to_expiry_years(expiry)

        if T == 0:
            return spot

        cost_of_carry = self.risk_free_rate - funding_rate
        fair_value = spot * np.exp(cost_of_carry * T)
        return float(fair_value)

    def update_funding_rate(self, instrument: str, funding_rate: float) -> None:
        """Update funding rate observation for dynamic carry modelling."""
        history = self.funding_rate_history.setdefault(instrument, [])
        history.append(float(funding_rate))
        if len(history) > 200:
            del history[:-200]

    def get_dynamic_funding_rate(self, instrument: str, default_rate: float = 0.0) -> float:
        """
        获取动态 funding rate (EWMA)。
        """
        history = self.funding_rate_history.get(instrument, [])
        if not history:
            return default_rate

        weights = np.array([0.97 ** (len(history) - 1 - i) for i in range(len(history))], dtype=float)
        weights /= weights.sum()
        return float(np.dot(weights, np.array(history)))

    def calculate_basis(
        self,
        instrument: str,
        funding_rate: float = 0.0
    ) -> Optional[Dict]:
        """
        计算基差。

        Returns:
            {
                'spot': 现货价格,
                'futures': 期货价格,
                'fair_value': 理论价格,
                'basis': 基差,
                'basis_pct': 基差百分比,
                'time_to_expiry': 剩余时间(年化)
            }
        """
        spot = self.spot_prices.get(instrument)
        futures = self.futures_prices.get(instrument)
        expiry = self.futures_expiry.get(instrument)

        if spot is None or futures is None or expiry is None:
            return None

        if spot <= 0:
            return None

        fair_value = self.calculate_fair_value(spot, expiry, funding_rate)
        basis = futures - fair_value
        basis_pct = basis / spot

        T = _time_to_expiry_years(expiry)

        return {
            'spot': spot,
            'futures': futures,
            'fair_value': fair_value,
            'basis': basis,
            'basis_pct': basis_pct,
            'time_to_expiry': T
        }

    def check_opportunity(
        self,
        instrument: str,
        funding_rate: Optional[float] = None
    ) -> Optional[BasisOpportunity]:
        """
        检查是否存在套利机会。

        Args:
            instrument: 交易对
            funding_rate: 当前资金费率

        Returns:
            BasisOpportunity 或 None
        """
        dynamic_funding = funding_rate if funding_rate is not None else self.get_dynamic_funding_rate(instrument, self.funding_cost)
        basis_info = self.calculate_basis(instrument, dynamic_funding)
        if basis_info is None:
            return None

        spot = basis_info['spot']
        futures = basis_info['futures']
        basis = basis_info['basis']
        basis_pct = basis_info['basis_pct']
        T = basis_info['time_to_expiry']

        if T <= 0:
            return None

        # 计算年化收益率
        annualized_return = basis_pct / T

        # 扣除交易成本
        funding_drag = abs(dynamic_funding)
        net_return = annualized_return - 2 * self.transaction_cost / T - funding_drag

        # 判断策略方向
        strategy: Literal["long_basis", "short_basis"]
        if basis > 0 and net_return >= self.min_annualized_return:
            # 正基差: 卖期货买现货
            strategy = "short_basis"
            expected_profit = basis - 2 * spot * self.transaction_cost
        elif basis < 0 and -net_return >= self.min_annualized_return:
            # 负基差: 买期货卖现货
            strategy = "long_basis"
            expected_profit = -basis - 2 * spot * self.transaction_cost
        else:
            return None

        # 计算所需资金（动态保证金 + 安全缓冲）
        required_capital = spot * (1 + self.margin_requirement_ratio + self.liquidation_buffer_pct)

        return BasisOpportunity(
            instrument=instrument,
            spot_price=spot,
            futures_price=futures,
            expiry=self.futures_expiry[instrument],
            basis=basis,
            basis_pct=basis_pct * 100,
            annualized_return=annualized_return * 100,
            strategy=strategy,
            required_capital=required_capital,
            expected_profit=expected_profit
        )

    def get_hedge_ratio(
        self,
        instrument: str,
        inverse_contract: bool = False,
        contract_multiplier: float = 1.0
    ) -> float:
        """
        计算 Delta 中性对冲比率。

        期现套利通常 1:1 对冲，但需要考虑合约乘数。
        """
        spot = self.spot_prices.get(instrument, 0.0)
        futures = self.futures_prices.get(instrument, spot if spot > 0 else 1.0)
        if spot <= 0:
            return 1.0
        if inverse_contract:
            # Inverse futures notional scales as 1/price.
            return float((spot / max(futures, 1e-12)) * contract_multiplier)
        return float(contract_multiplier)

    def calculate_pnl(
        self,
        entry_spot: float,
        entry_futures: float,
        exit_spot: float,
        exit_futures: float,
        position_size: float = 1.0
    ) -> Dict[str, float]:
        """
        计算平仓后的 P&L。

        Args:
            entry_spot: 入场现货价格
            entry_futures: 入场期货价格
            exit_spot: 平仓现货价格
            exit_futures: 平仓期货价格
            position_size: 头寸大小

        Returns:
            P&L 明细
        """
        # 假设 short_basis 策略 (卖期货买现货)
        spot_pnl = position_size * (exit_spot - entry_spot)
        futures_pnl = position_size * (entry_futures - exit_futures)

        gross_pnl = spot_pnl + futures_pnl
        transaction_costs = 2 * position_size * entry_spot * self.transaction_cost

        return {
            'spot_pnl': spot_pnl,
            'futures_pnl': futures_pnl,
            'gross_pnl': gross_pnl,
            'transaction_costs': transaction_costs,
            'net_pnl': gross_pnl - transaction_costs
        }

    def estimate_carry_cost(
        self,
        position_value: float,
        days: int
    ) -> float:
        """
        估算持仓成本。

        包括:
        - 资金成本
        - 资金费率
        """
        daily_cost = position_value * (self.risk_free_rate / 365 + self.funding_cost)
        return daily_cost * days

    def assess_margin_liquidation_risk(
        self,
        instrument: str,
        position_size: float
    ) -> Dict[str, float]:
        """
        简化保证金/清算风险评估。
        """
        spot = self.spot_prices.get(instrument, 0.0)
        futures = self.futures_prices.get(instrument, spot if spot > 0 else 0.0)
        notional = abs(position_size) * max(spot, 0.0)
        initial_margin = notional * self.margin_requirement_ratio
        liquidation_distance_pct = self.liquidation_buffer_pct
        liquidation_price = futures * (1 - liquidation_distance_pct) if position_size > 0 else futures * (1 + liquidation_distance_pct)
        return {
            "notional": float(notional),
            "initial_margin": float(initial_margin),
            "liquidation_price": float(liquidation_price),
            "liquidation_distance_pct": float(liquidation_distance_pct * 100),
        }
