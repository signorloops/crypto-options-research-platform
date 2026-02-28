"""
币本位期权定价模型 (Inverse Option Pricing Model)。

币本位期权以加密货币计价和结算，其定价与U本位期权不同。
关键区别：
1. 标的以USD报价，但以加密货币结算
2. 盈亏是非线性的: PnL = (1/entry - 1/exit) * size
3. 需要调整后的Black-Scholes公式

数学原理：
- 设 S 为标的资产价格 (USD per BTC)
- 币本位期权的 payoff 以 BTC 计价
- Inverse Call payoff: max(0, 1/K - 1/S) BTC
- Inverse Put payoff: max(0, 1/S - 1/K) BTC

利用Y = 1/S的测度变换，可以将币本位期权定价转化为标准Black-Scholes框架。

参考：
- OKX documentation on inverse options
- Black-Scholes with numeraire change for inverse contracts
"""
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

import numpy as np
from scipy.stats import norm

from core.exceptions import ValidationError

import logging
logger = logging.getLogger(__name__)


@dataclass
class InverseGreeks:
    """Greeks for coin-margined (inverse) options.

    注意：币本位期权的希腊字母与U本位不同：
    - Delta: 对标的价变化的敏感度（单位：BTC per USD）
    - Gamma: Delta对标的价变化的二阶导
    - 其他希腊字母相应调整
    """
    delta: float          # Delta (per USD)
    gamma: float          # Gamma (per USD^2)
    theta: float          # Theta (daily time decay)
    vega: float           # Vega (1% vol change)
    rho: float            # Rho (1% rate change)
    vanna: float = 0.0    # Vanna (dDelta/dVol)
    charm: float = 0.0    # Charm (dDelta/dTime)
    veta: float = 0.0     # Veta (dVega/dTime)


class InverseOptionPricer:
    """
    币本位期权定价模型。

    使用调整后的Black-Scholes公式，考虑币本位的非线性盈亏特性。

    核心数学：
    - 通过测度变换将inverse option转化为标准BS公式
    - 设 Y = 1/S，则币本位call = 标准put on Y（经调整）
    - 币本位put = 标准call on Y（经调整）

    波动率说明：
    - 输入的sigma是标的S的波动率（不是Y的波动率）
    - 由于d(1/S) ≈ -sigma*(1/S)*dW，Y和S的波动率相同
    """

    EPSILON = 1e-10  # 数值计算的小量
    MAX_REASONABLE_VOLATILITY = 10.0  # 1000%波动率作为合理上限
    MAX_REASONABLE_RATE = 1.0  # 100%利率作为合理上限
    THETA_DAYS_PER_YEAR = 365.0  # 年化Theta的分母
    VEGA_SCALING = 0.01  # Vega从每单位波动率转换为每1%波动率的缩放因子
    RHO_SCALING = 100.0  # Rho从每单位利率转换为每1%利率的缩放因子

    @staticmethod
    def _validate_option_type(option_type: str) -> None:
        """Validate option type."""
        if option_type not in {"call", "put"}:
            raise ValueError(
                f"Invalid option_type={option_type}. Expected 'call' or 'put'."
            )

    @staticmethod
    def _validate_inputs(S: float, K: float, T: float, r: float, sigma: float) -> None:
        """验证输入参数的有效性。"""
        # 检查NaN和Inf
        if not (np.isfinite(S) and np.isfinite(K) and np.isfinite(T)
                and np.isfinite(r) and np.isfinite(sigma)):
            raise ValidationError(
                "NaN or Inf detected in inputs",
                field="inputs",
                value=(S, K, T, r, sigma),
            )

        if S <= 0:
            raise ValidationError("Spot price must be positive", field="S", value=S)
        if K <= 0:
            raise ValidationError("Strike price must be positive", field="K", value=K)
        if sigma <= 0:
            raise ValidationError("Volatility must be positive", field="sigma", value=sigma)
        if sigma > InverseOptionPricer.MAX_REASONABLE_VOLATILITY:
            raise ValidationError(
                f"Volatility exceeds reasonable maximum ({InverseOptionPricer.MAX_REASONABLE_VOLATILITY})",
                field="sigma",
                value=sigma,
            )
        if T < 0:
            raise ValidationError("Time to expiry must be non-negative", field="T", value=T)
        if abs(r) > InverseOptionPricer.MAX_REASONABLE_RATE:
            raise ValidationError(
                f"Risk-free rate exceeds reasonable range (-{InverseOptionPricer.MAX_REASONABLE_RATE}, {InverseOptionPricer.MAX_REASONABLE_RATE})",
                field="r",
                value=r,
            )

    @staticmethod
    def _calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """
        计算 d1 和 d2（币本位期权专用）。

        对于币本位期权，使用调整后的公式：
        d1 = [ln(K/S) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)

        注意：与标准BS相比，S和K的位置互换（因为payoff在1/S空间）。
        """
        if T < InverseOptionPricer.EPSILON:
            # T接近0时的极限情况
            # d1 = [ln(K/S) + ...] / (sigma*sqrt(T))
            # 当T->0，分子趋近于ln(K/S)
            # S>K时，ln(K/S)<0，所以d1->-inf
            # S<K时，ln(K/S)>0，所以d1->+inf
            if S > K:
                return float('-inf'), float('-inf')
            elif S < K:
                return float('inf'), float('inf')
            else:
                return 0.0, 0.0

        # 币本位期权：使用K/S而不是S/K
        d1 = (np.log(K / S) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @staticmethod
    def calculate_price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"]
    ) -> float:
        """
        计算币本位期权价格（以加密货币计价）。

        数学公式：
        - Inverse Call = e^(-rT) * (1/K) * N(-d2) - (1/S) * N(-d1)
        - Inverse Put = (1/S) * N(d1) - e^(-rT) * (1/K) * N(d2)

        Args:
            S: 标的当前价格 (USD per BTC)
            K: 行权价 (USD per BTC)
            T: 到期时间（年）
            r: 无风险利率（连续复利）
            sigma: 波动率
            option_type: "call" 或 "put"

        Returns:
            期权价格（以加密货币为单位，如 BTC）
        """
        InverseOptionPricer._validate_option_type(option_type)
        InverseOptionPricer._validate_inputs(S, K, T, r, sigma)

        # 到期边界条件
        if T < InverseOptionPricer.EPSILON:
            if option_type == "call":
                # Payoff = max(0, 1/K - 1/S)
                return max(0.0, 1.0/K - 1.0/S)
            else:
                # Payoff = max(0, 1/S - 1/K)
                return max(0.0, 1.0/S - 1.0/K)

        d1, d2 = InverseOptionPricer._calculate_d1_d2(S, K, T, r, sigma)
        inv_S = 1.0 / S
        inv_K = 1.0 / K
        discount = np.exp(-r * T)

        if option_type == "call":
            # 币本位看涨 = e^(-rT)*(1/K)*N(-d2) - (1/S)*N(-d1)
            price = float(discount * inv_K * norm.cdf(-d2) - inv_S * norm.cdf(-d1))
        else:
            # 币本位看跌 = (1/S)*N(d1) - e^(-rT)*(1/K)*N(d2)
            price = float(inv_S * norm.cdf(d1) - discount * inv_K * norm.cdf(d2))

        # 确保非负（数值误差可能导致微小负值）
        return max(0.0, price)

    @staticmethod
    def calculate_price_and_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"]
    ) -> Tuple[float, InverseGreeks]:
        """
        同时计算价格和希腊字母（避免重复计算）。

        Args:
            S, K, T, r, sigma, option_type: 标准参数

        Returns:
            (价格, 希腊字母) 元组
        """
        InverseOptionPricer._validate_option_type(option_type)
        InverseOptionPricer._validate_inputs(S, K, T, r, sigma)

        # 到期边界条件
        if T < InverseOptionPricer.EPSILON:
            if option_type == "call":
                price = max(0.0, 1.0/K - 1.0/S) if S > K else 0.0
            else:
                price = max(0.0, 1.0/S - 1.0/K) if K > S else 0.0
            return price, InverseGreeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

        # 统一计算 d1, d2
        d1, d2 = InverseOptionPricer._calculate_d1_d2(S, K, T, r, sigma)
        inv_S = 1.0 / S
        inv_K = 1.0 / K
        discount = np.exp(-r * T)
        sqrt_T = np.sqrt(T)
        n_d1 = norm.pdf(d1)

        # 计算价格
        if option_type == "call":
            price = discount * inv_K * norm.cdf(-d2) - inv_S * norm.cdf(-d1)
        else:
            price = inv_S * norm.cdf(d1) - discount * inv_K * norm.cdf(d2)
        price = max(0.0, price)

        # 计算希腊字母
        greeks = InverseOptionPricer._calculate_greeks_from_d(
            d1, d2, S, K, T, r, sigma, option_type, inv_S, inv_K, discount, sqrt_T, n_d1
        )

        return price, greeks

    @staticmethod
    def _calculate_greeks_from_d(
        d1: float,
        d2: float,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        inv_S: float,
        inv_K: float,
        discount: float,
        sqrt_T: float,
        n_d1: float
    ) -> InverseGreeks:
        """从已计算的 d1, d2 计算希腊字母（内部辅助函数）。"""

        # Delta (对S的偏导)
        # 币本位期权需要完整的Delta推导，包含n(d1)项
        # d(d1)/dS = -1/(S * sigma * sqrt(T))
        d1_dS = -1.0 / (S * sigma * sqrt_T)

        if option_type == "call":
            # 完整Delta = (1/S^2)*N(-d1) + (1/S)*n(d1)*d(d1)/dS - discount*(1/K)*n(-d2)*d(d2)/dS
            # 由于d2 = d1 - sigma*sqrt(T)，所以d(d2)/dS = d(d1)/dS
            delta = ((inv_S ** 2) * norm.cdf(-d1)
                     + inv_S * n_d1 * d1_dS
                     - discount * inv_K * norm.pdf(-d2) * d1_dS)
        else:
            # 币本位put: delta = -(1/S^2)*N(d1) + (1/S)*n(d1)*d(d1)/dS - discount*(1/K)*n(d2)*d(d2)/dS
            delta = (-(inv_S ** 2) * norm.cdf(d1)
                     + inv_S * n_d1 * d1_dS
                     - discount * inv_K * norm.pdf(d2) * d1_dS)

        # Gamma (delta对S的二阶导)
        # 币本位期权Gamma推导:
        # Call: Gamma = -2/S³ * N(-d1) + n(d1)/(S³*σ*√T)
        # Put:  Gamma = 2/S³ * N(d1) + n(d1)/(S³*σ*√T)
        # 注意: 币本位期权的Call和Put Gamma不相等
        gamma_term2 = n_d1 / (S ** 3 * sigma * sqrt_T)
        if option_type == "call":
            gamma = -2 * (inv_S ** 3) * norm.cdf(-d1) + gamma_term2
        else:
            gamma = 2 * (inv_S ** 3) * norm.cdf(d1) + gamma_term2

        # Theta (时间衰减，每日）
        # 币本位期权 Theta = -dV/dT / 365 (负号表示时间流逝导致价值减少)
        # 从 Black-76 币本位公式推导
        # dV/dT = -r*e^(-rT)*(1/K)*N(-d2) - e^(-rT)*(1/K)*n(d2)*dd2/dT + (1/S)*n(d1)*dd1/dT
        d_d1_dT = ((r + 0.5 * sigma ** 2) * T - np.log(K / S)) / (2 * sigma * T ** 1.5)
        d_d2_dT = d_d1_dT - sigma / (2 * sqrt_T)

        if option_type == "call":
            dV_dT = (-r * discount * inv_K * norm.cdf(-d2)
                     - discount * inv_K * norm.pdf(d2) * d_d2_dT
                     + inv_S * n_d1 * d_d1_dT)
            theta = -dV_dT / InverseOptionPricer.THETA_DAYS_PER_YEAR
        else:
            dV_dT = (r * discount * inv_K * norm.cdf(d2)
                     + discount * inv_K * norm.pdf(d2) * d_d2_dT
                     - inv_S * n_d1 * d_d1_dT)
            theta = -dV_dT / InverseOptionPricer.THETA_DAYS_PER_YEAR

        # Vega (波动率变化1%时的价格变化）
        # 完整的Vega推导需要考虑所有项对sigma的导数
        # d(d1)/dsigma = (ln(S/K) + (r - 0.5*sigma^2)*T) / (sigma^2 * sqrt(T)) = -d2/sigma
        d1_dsigma = -d2 / sigma if abs(sigma) > 1e-10 else 0.0
        d2_dsigma = d1_dsigma - sqrt_T

        if option_type == "call":
            # Vega = dV/dsigma = discount*(1/K)*n(-d2)*(-d2_dsigma) - (1/S)*n(-d1)*(-d1_dsigma)
            # 由于n(-x) = n(x)（正态分布对称性）
            vega_per_unit = (discount * inv_K * norm.pdf(d2) * (-d2_dsigma)
                            - inv_S * n_d1 * (-d1_dsigma))
        else:
            # Vega = (1/S)*n(d1)*d1_dsigma - discount*(1/K)*n(d2)*d2_dsigma
            vega_per_unit = (inv_S * n_d1 * d1_dsigma
                            - discount * inv_K * norm.pdf(d2) * d2_dsigma)

        vega = vega_per_unit * InverseOptionPricer.VEGA_SCALING

        # Rho (利率变化1%时的价格变化）
        if option_type == "call":
            rho = -T * discount * inv_K * norm.cdf(-d2) / InverseOptionPricer.RHO_SCALING
        else:
            rho = T * discount * inv_K * norm.cdf(d2) / InverseOptionPricer.RHO_SCALING

        return InverseGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho
        )

    @staticmethod
    def calculate_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"]
    ) -> InverseGreeks:
        """
        计算币本位期权的希腊字母。

        Args:
            S: 标的当前价格 (USD)
            K: 行权价 (USD)
            T: 到期时间（年）
            r: 无风险利率
            sigma: 波动率
            option_type: "call" 或 "put"

        Returns:
            InverseGreeks对象
        """
        InverseOptionPricer._validate_option_type(option_type)
        InverseOptionPricer._validate_inputs(S, K, T, r, sigma)

        if T < InverseOptionPricer.EPSILON:
            return InverseGreeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)

        # 统一计算 d1, d2
        d1, d2 = InverseOptionPricer._calculate_d1_d2(S, K, T, r, sigma)
        inv_S = 1.0 / S
        inv_K = 1.0 / K
        discount = np.exp(-r * T)
        sqrt_T = np.sqrt(T)
        n_d1 = norm.pdf(d1)

        # 使用辅助函数计算希腊字母
        return InverseOptionPricer._calculate_greeks_from_d(
            d1, d2, S, K, T, r, sigma, option_type, inv_S, inv_K, discount, sqrt_T, n_d1
        )

    @staticmethod
    def calculate_implied_volatility(
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: Literal["call", "put"],
        tol: float = 1e-6,
        max_iter: int = 50,
        stabilize_short_maturity: bool = False,
        short_maturity_threshold: float = 14.0 / 365.0,
        anchor_sigma: Optional[float] = None,
        max_anchor_deviation: float = 0.50,
    ) -> float:
        """
        通过牛顿-拉夫逊法计算隐含波动率。

        注意：这里的vega是每1%波动率变化的价格变化，所以在牛顿迭代中
        需要乘以0.01来得到每单位波动率的价格变化。

        Args:
            price: 观测到的期权价格（加密货币单位）
            S, K, T, r: 标准参数
            option_type: "call" 或 "put"
            tol: 收敛容差（默认更严格）
            max_iter: 最大迭代次数

        Returns:
            隐含波动率
        """
        InverseOptionPricer._validate_option_type(option_type)
        if price <= 0:
            return 0.0

        # 检查价格合理性（不能超过理论上限）
        if option_type == "call":
            max_price = np.exp(-r * T) / K  # 当S->inf时的极限
        else:
            max_price = 1.0 / S  # 当S->0时的极限

        if price >= max_price:
            return 0.0

        # 初始猜测：使用ATM近似
        moneyness = S / K
        sigma = float(0.5 + 0.1 * abs(np.log(moneyness)))
        sigma = max(0.1, min(2.0, sigma))

        # 使用相对容差，适应不同价格尺度
        price_scale = max(abs(price), InverseOptionPricer.EPSILON)
        rel_tol = tol

        def _stabilize_iv(raw_sigma: float) -> float:
            if not stabilize_short_maturity:
                return float(raw_sigma)
            if T <= 0 or T > short_maturity_threshold:
                return float(raw_sigma)
            if anchor_sigma is None or not np.isfinite(anchor_sigma) or anchor_sigma <= 0:
                return float(raw_sigma)

            anchor = float(np.clip(anchor_sigma, 0.001, 5.0))
            deviation = float(max(max_anchor_deviation, 1e-6))
            bounded = float(np.clip(raw_sigma, anchor - deviation, anchor + deviation))

            maturity_ratio = float(np.clip(T / max(short_maturity_threshold, InverseOptionPricer.EPSILON), 0.0, 1.0))
            weight = 0.60 * (1.0 - maturity_ratio)
            stabilized = (1.0 - weight) * bounded + weight * anchor
            return float(np.clip(stabilized, 0.001, 5.0))

        for i in range(max_iter):
            try:
                model_price = InverseOptionPricer.calculate_price(S, K, T, r, sigma, option_type)
                diff = model_price - price

                # 使用相对误差检查收敛
                if abs(diff) < rel_tol * price_scale:
                    return _stabilize_iv(sigma)

                # Vega作为导数（每单位波动率，不是每1%）
                greeks = InverseOptionPricer.calculate_greeks(S, K, T, r, sigma, option_type)
                # greeks.vega是每1%的变化，转换为每单位波动率需要除以0.01（乘以100）
                vega = greeks.vega / InverseOptionPricer.VEGA_SCALING

                if abs(vega) < 1e-14:
                    # Vega太小，使用二分法
                    break

                # 牛顿迭代，带阻尼
                step = diff / vega
                # 限制步长防止震荡
                step = max(-0.5, min(0.5, step))
                sigma_new = sigma - step

                # 限制范围
                sigma_new = max(0.001, min(5.0, sigma_new))

                if abs(sigma_new - sigma) < 1e-6:
                    return _stabilize_iv(sigma_new)

                sigma = sigma_new

            except (ValueError, FloatingPointError, RuntimeError):
                # Numerical errors in iteration: fall back to bisection method
                break

        # 如果牛顿法失败，使用二分法作为fallback
        raw_sigma = InverseOptionPricer._iv_bisection(price, S, K, T, r, option_type, tol, max_iter)
        return _stabilize_iv(raw_sigma)

    @staticmethod
    def _iv_bisection(
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: Literal["call", "put"],
        tol: float = 1e-9,
        max_iter: int = 100
    ) -> float:
        """使用二分法计算隐含波动率（作为fallback）。"""
        sigma_low, sigma_high = 0.001, 5.0

        price_low = InverseOptionPricer.calculate_price(S, K, T, r, sigma_low, option_type)
        price_high = InverseOptionPricer.calculate_price(S, K, T, r, sigma_high, option_type)

        # 检查价格是否在合理范围内
        if price < price_low or price > price_high:
            # 价格超出范围，可能是输入有误
            logger.warning(
                f"IV calculation: price {price} outside valid range [{price_low:.2e}, {price_high:.2e}] "
                f"for {option_type} S={S}, K={K}, T={T}"
            )
            return sigma_low if price < price_low else sigma_high

        for _ in range(max_iter):
            sigma_mid = (sigma_low + sigma_high) / 2
            price_mid = InverseOptionPricer.calculate_price(S, K, T, r, sigma_mid, option_type)

            if abs(price_mid - price) < tol:
                return sigma_mid

            if price_mid < price:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid

            if sigma_high - sigma_low < tol:
                return (sigma_low + sigma_high) / 2

        return (sigma_low + sigma_high) / 2

    @staticmethod
    def calculate_pnl(
        entry_price: float,
        exit_price: float,
        size: float,
        inverse: bool = True
    ) -> float:
        """
        计算盈亏。

        Args:
            entry_price: 入场价格（USD）
            exit_price: 出场价格（USD）
            size: 合约数量
            inverse: 是否币本位（默认True）

        Returns:
            PnL（加密货币单位）
        """
        if not inverse:
            # U本位线性盈亏
            return size * (exit_price - entry_price)

        # 币本位非线性盈亏
        if entry_price < InverseOptionPricer.EPSILON or exit_price < InverseOptionPricer.EPSILON:
            return 0.0

        # PnL = size * (1/entry - 1/exit)
        return size * (1.0 / entry_price - 1.0 / exit_price)


def inverse_option_parity(
    call_price: float,
    put_price: float,
    S: float,
    K: float,
    T: float,
    r: float
) -> float:
    """
    币本位期权的Put-Call Parity验证。

    币本位的Put-Call Parity（以加密货币计价）：
    C - P = (1/K) * e^(-rT) - 1/S

    推导：
    - 币本位call + 币本位put = 持有现金的put-call parity
    - 在BTC numeraire下，考虑现金的现值

    Args:
        call_price: 看涨期权价格（加密货币单位）
        put_price: 看跌期权价格（加密货币单位）
        S: 标的现价
        K: 行权价
        T: 到期时间
        r: 无风险利率

    Returns:
        Parity偏差（越接近0越好）
    """
    if S <= 0 or K <= 0 or T < 0:
        return float('inf')

    if T < InverseOptionPricer.EPSILON:
        # 到期时
        lhs = call_price - put_price
        rhs = max(0, 1/K - 1/S) - max(0, 1/S - 1/K)
    else:
        lhs = call_price - put_price
        rhs = (1.0 / K) * np.exp(-r * T) - 1.0 / S

    return lhs - rhs


def calculate_position_value(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    size: float,
    option_type: Literal["call", "put"],
    avg_entry_price_usd: float,
    inverse: bool = True
) -> Tuple[float, float, float]:
    """
    计算期权头寸的当前价值和盈亏。

    Args:
        S: 当前标的价 (USD)
        K: 行权价 (USD)
        T: 剩余时间（年）
        r: 利率
        sigma: 波动率
        size: 头寸大小（正数=多头，负数=空头）
        option_type: "call" 或 "put"
        avg_entry_price_usd: 平均入场价（期权价格，单位BTC）
        inverse: 是否币本位

    Returns:
        (当前期权价值BTC, 未实现盈亏BTC, 市值BTC)
    """
    InverseOptionPricer._validate_option_type(option_type)
    InverseOptionPricer._validate_inputs(S, K, T, r, sigma)

    if inverse:
        # 币本位计算
        current_option_value = InverseOptionPricer.calculate_price(S, K, T, r, sigma, option_type)
        # 未实现盈亏 = 头寸大小 * (当前期权价值 - 入场期权价值)
        unrealized_pnl = size * (current_option_value - avg_entry_price_usd)
        market_value = size * current_option_value
    else:
        # U本位计算
        from scipy.stats import norm
        if T <= InverseOptionPricer.EPSILON:
            # 到期边界：回落到内在价值，避免 sqrt(T) 除零。
            if option_type == "call":
                current_option_value = max(0.0, S - K)
            else:
                current_option_value = max(0.0, K - S)
        else:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            if option_type == "call":
                current_option_value = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                current_option_value = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        unrealized_pnl = size * (current_option_value - avg_entry_price_usd)
        market_value = size * current_option_value

    return current_option_value, unrealized_pnl, market_value
