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
from typing import Callable, Literal, Tuple, Optional

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


@dataclass(frozen=True)
class _GreeksComputationContext:
    """Pre-computed scalar terms reused across inverse Greeks formulas."""

    inv_S: float
    inv_K: float
    discount: float
    sqrt_T: float
    n_d1: float


def _raise_validation_error(
    *,
    condition: bool,
    message: str,
    field: str,
    value,
) -> None:
    if condition:
        raise ValidationError(message, field=field, value=value)


def _run_inverse_input_checks(checks: list[tuple[bool, str, str, float]]) -> None:
    for condition, message, field, value in checks:
        _raise_validation_error(
            condition=condition,
            message=message,
            field=field,
            value=value,
        )


def _inverse_inputs_finite(*values: float) -> bool:
    return all(np.isfinite(value) for value in values)


def _inverse_validation_checks(
    *, S: float, K: float, T: float, r: float, sigma: float
) -> list[tuple[bool, str, str, float]]:
    max_vol = InverseOptionPricer.MAX_REASONABLE_VOLATILITY
    max_rate = InverseOptionPricer.MAX_REASONABLE_RATE
    return [
        (S <= 0, "Spot price must be positive", "S", S),
        (K <= 0, "Strike price must be positive", "K", K),
        (sigma <= 0, "Volatility must be positive", "sigma", sigma),
        (
            sigma > max_vol,
            f"Volatility exceeds reasonable maximum ({max_vol})",
            "sigma",
            sigma,
        ),
        (T < 0, "Time to expiry must be non-negative", "T", T),
        (
            abs(r) > max_rate,
            f"Risk-free rate exceeds reasonable range (-{max_rate}, {max_rate})",
            "r",
            r,
        ),
    ]


def _inverse_iv_bracket_prices(
    *,
    price_fn: Callable[[float], float],
) -> Tuple[float, float, float, float]:
    sigma_low, sigma_high = 0.001, 5.0
    price_low = float(price_fn(sigma_low))
    price_high = float(price_fn(sigma_high))
    return sigma_low, sigma_high, price_low, price_high


def _inverse_iv_bisection_search(
    *,
    price_fn: Callable[[float], float],
    target_price: float,
    sigma_low: float,
    sigma_high: float,
    tol: float,
    max_iter: int,
) -> float:
    for _ in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        price_mid = float(price_fn(sigma_mid))
        if abs(price_mid - target_price) < tol:
            return sigma_mid
        if price_mid < target_price:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
        if sigma_high - sigma_low < tol:
            return (sigma_low + sigma_high) / 2
    return (sigma_low + sigma_high) / 2


def _iterate_newton_iv_solve(
    *,
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"],
    sigma: float,
    rel_tol: float,
    price_scale: float,
    max_iter: int,
    update_fn: Callable[..., tuple[Optional[float], bool, bool]],
) -> Optional[float]:
    base_kwargs = {"price": price, "S": S, "K": K, "T": T, "r": r, "option_type": option_type, "rel_tol": rel_tol, "price_scale": price_scale}
    for _ in range(max_iter):
        sigma_new, converged, use_bisection = update_fn(sigma=sigma, **base_kwargs)
        if converged and sigma_new is not None:
            return float(sigma_new)
        if use_bisection or sigma_new is None:
            return None
        sigma = sigma_new
    return None


def _inverse_intrinsic_price(S: float, K: float, option_type: Literal["call", "put"]) -> float:
    if option_type == "call":
        return max(0.0, 1.0 / K - 1.0 / S)
    return max(0.0, 1.0 / S - 1.0 / K)


def _inverse_price_from_d(
    *,
    option_type: Literal["call", "put"],
    inv_S: float,
    inv_K: float,
    discount: float,
    d1: float,
    d2: float,
) -> float:
    if option_type == "call":
        return float(discount * inv_K * norm.cdf(-d2) - inv_S * norm.cdf(-d1))
    return float(inv_S * norm.cdf(d1) - discount * inv_K * norm.cdf(d2))


def _zero_inverse_greeks() -> InverseGreeks:
    return InverseGreeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)


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
        if not _inverse_inputs_finite(S, K, T, r, sigma):
            raise ValidationError(
                "NaN or Inf detected in inputs",
                field="inputs",
                value=(S, K, T, r, sigma),
            )
        _run_inverse_input_checks(
            _inverse_validation_checks(S=S, K=K, T=T, r=r, sigma=sigma)
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
        """计算币本位期权价格（以加密货币计价）。"""
        InverseOptionPricer._validate_option_type(option_type)
        InverseOptionPricer._validate_inputs(S, K, T, r, sigma)
        if T < InverseOptionPricer.EPSILON:
            return _inverse_intrinsic_price(S, K, option_type)
        d1, d2 = InverseOptionPricer._calculate_d1_d2(S, K, T, r, sigma)
        inv_S, inv_K = 1.0 / S, 1.0 / K
        discount = np.exp(-r * T)
        price = _inverse_price_from_d(
            option_type=option_type,
            inv_S=inv_S,
            inv_K=inv_K,
            discount=discount,
            d1=d1,
            d2=d2,
        )
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
        """Calculate inverse-option price and Greeks in one pass."""
        InverseOptionPricer._validate_option_type(option_type)
        InverseOptionPricer._validate_inputs(S, K, T, r, sigma)
        if T < InverseOptionPricer.EPSILON:
            return _inverse_intrinsic_price(S, K, option_type), _zero_inverse_greeks()
        d1, d2 = InverseOptionPricer._calculate_d1_d2(S, K, T, r, sigma)
        inv_S, inv_K = 1.0 / S, 1.0 / K
        discount = np.exp(-r * T)
        sqrt_T = np.sqrt(T)
        n_d1 = norm.pdf(d1)
        price = max(
            0.0,
            _inverse_price_from_d(
                option_type=option_type,
                inv_S=inv_S,
                inv_K=inv_K,
                discount=discount,
                d1=d1,
                d2=d2,
            ),
        )
        context = _GreeksComputationContext(inv_S=inv_S, inv_K=inv_K, discount=discount, sqrt_T=sqrt_T, n_d1=n_d1)
        greeks = InverseOptionPricer._calculate_greeks_from_d(d1, d2, S, K, T, r, sigma, option_type, context)
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
        context: _GreeksComputationContext,
    ) -> InverseGreeks:
        """从已计算的 d1, d2 计算希腊字母（内部辅助函数）。"""
        delta = InverseOptionPricer._calculate_delta_from_d(d1=d1, d2=d2, S=S, sigma=sigma, option_type=option_type, context=context)
        gamma = InverseOptionPricer._calculate_gamma_from_d(d1=d1, S=S, sigma=sigma, option_type=option_type, context=context)
        theta = InverseOptionPricer._calculate_theta_from_d(d1=d1, d2=d2, S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, context=context)
        vega = InverseOptionPricer._calculate_vega_from_d(d1=d1, d2=d2, sigma=sigma, option_type=option_type, context=context)
        rho = InverseOptionPricer._calculate_rho_from_d(d2=d2, T=T, option_type=option_type, context=context)
        return InverseGreeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)

    @staticmethod
    def _calculate_delta_from_d(
        *,
        d1: float,
        d2: float,
        S: float,
        sigma: float,
        option_type: Literal["call", "put"],
        context: _GreeksComputationContext,
    ) -> float:
        """Compute inverse-option delta from precomputed d1/d2 terms."""
        inv_S = context.inv_S
        inv_K = context.inv_K
        discount = context.discount
        n_d1 = context.n_d1
        sqrt_T = context.sqrt_T
        d1_dS = -1.0 / (S * sigma * sqrt_T)
        if option_type == "call":
            return float(
                (inv_S ** 2) * norm.cdf(-d1)
                + inv_S * n_d1 * d1_dS
                - discount * inv_K * norm.pdf(-d2) * d1_dS
            )
        return float(
            -(inv_S ** 2) * norm.cdf(d1)
            + inv_S * n_d1 * d1_dS
            - discount * inv_K * norm.pdf(d2) * d1_dS
        )

    @staticmethod
    def _calculate_gamma_from_d(
        *,
        d1: float,
        S: float,
        sigma: float,
        option_type: Literal["call", "put"],
        context: _GreeksComputationContext,
    ) -> float:
        """Compute inverse-option gamma from precomputed d1 terms."""
        inv_S = context.inv_S
        n_d1 = context.n_d1
        sqrt_T = context.sqrt_T
        gamma_term2 = n_d1 / (S ** 3 * sigma * sqrt_T)
        if option_type == "call":
            return float(-2 * (inv_S ** 3) * norm.cdf(-d1) + gamma_term2)
        return float(2 * (inv_S ** 3) * norm.cdf(d1) + gamma_term2)

    @staticmethod
    def _calculate_theta_from_d(
        *,
        d1: float,
        d2: float,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"],
        context: _GreeksComputationContext,
    ) -> float:
        """Compute daily theta from precomputed d terms."""
        inv_S = context.inv_S
        inv_K = context.inv_K
        discount = context.discount
        n_d1 = context.n_d1
        sqrt_T = context.sqrt_T
        d_d1_dT = ((r + 0.5 * sigma ** 2) * T - np.log(K / S)) / (2 * sigma * T ** 1.5)
        d_d2_dT = d_d1_dT - sigma / (2 * sqrt_T)
        if option_type == "call":
            dV_dT = (
                -r * discount * inv_K * norm.cdf(-d2)
                - discount * inv_K * norm.pdf(d2) * d_d2_dT
                + inv_S * n_d1 * d_d1_dT
            )
        else:
            dV_dT = (
                r * discount * inv_K * norm.cdf(d2)
                + discount * inv_K * norm.pdf(d2) * d_d2_dT
                - inv_S * n_d1 * d_d1_dT
            )
        return float(-dV_dT / InverseOptionPricer.THETA_DAYS_PER_YEAR)

    @staticmethod
    def _calculate_vega_from_d(
        *,
        d1: float,
        d2: float,
        sigma: float,
        option_type: Literal["call", "put"],
        context: _GreeksComputationContext,
    ) -> float:
        """Compute vega scaled to 1% volatility changes."""
        del d1  # d1 density is provided in context
        inv_S = context.inv_S
        inv_K = context.inv_K
        discount = context.discount
        n_d1 = context.n_d1
        sqrt_T = context.sqrt_T
        d1_dsigma = -d2 / sigma if abs(sigma) > 1e-10 else 0.0
        d2_dsigma = d1_dsigma - sqrt_T
        if option_type == "call":
            vega_per_unit = (
                discount * inv_K * norm.pdf(d2) * (-d2_dsigma)
                - inv_S * n_d1 * (-d1_dsigma)
            )
        else:
            vega_per_unit = (
                inv_S * n_d1 * d1_dsigma
                - discount * inv_K * norm.pdf(d2) * d2_dsigma
            )
        return float(vega_per_unit * InverseOptionPricer.VEGA_SCALING)

    @staticmethod
    def _calculate_rho_from_d(
        *,
        d2: float,
        T: float,
        option_type: Literal["call", "put"],
        context: _GreeksComputationContext,
    ) -> float:
        """Compute rho scaled to 1% rate changes."""
        inv_K = context.inv_K
        discount = context.discount
        if option_type == "call":
            rho = -T * discount * inv_K * norm.cdf(-d2)
        else:
            rho = T * discount * inv_K * norm.cdf(d2)
        return float(rho / InverseOptionPricer.RHO_SCALING)

    @staticmethod
    def _stabilize_iv_estimate(
        raw_sigma: float,
        T: float,
        *,
        stabilize_short_maturity: bool,
        short_maturity_threshold: float,
        anchor_sigma: Optional[float],
        max_anchor_deviation: float,
    ) -> float:
        if not stabilize_short_maturity:
            return float(raw_sigma)
        if T <= 0 or T > short_maturity_threshold:
            return float(raw_sigma)
        if anchor_sigma is None or not np.isfinite(anchor_sigma) or anchor_sigma <= 0:
            return float(raw_sigma)

        anchor = float(np.clip(anchor_sigma, 0.001, 5.0))
        deviation = float(max(max_anchor_deviation, 1e-6))
        bounded = float(np.clip(raw_sigma, anchor - deviation, anchor + deviation))
        maturity_ratio = float(
            np.clip(T / max(short_maturity_threshold, InverseOptionPricer.EPSILON), 0.0, 1.0)
        )
        weight = 0.60 * (1.0 - maturity_ratio)
        stabilized = (1.0 - weight) * bounded + weight * anchor
        return float(np.clip(stabilized, 0.001, 5.0))

    @staticmethod
    def _newton_iv_update(
        *,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: Literal["call", "put"],
        sigma: float,
        rel_tol: float,
        price_scale: float,
    ) -> tuple[Optional[float], bool, bool]:
        try:
            model_price = InverseOptionPricer.calculate_price(S, K, T, r, sigma, option_type)
            diff = model_price - price
            if abs(diff) < rel_tol * price_scale:
                return sigma, True, False

            greeks = InverseOptionPricer.calculate_greeks(S, K, T, r, sigma, option_type)
            vega = greeks.vega / InverseOptionPricer.VEGA_SCALING
            if abs(vega) < 1e-14:
                return None, False, True

            step = np.clip(diff / vega, -0.5, 0.5)
            sigma_new = float(np.clip(sigma - step, 0.001, 5.0))
            if abs(sigma_new - sigma) < 1e-6:
                return sigma_new, True, False
            return sigma_new, False, False
        except (ValueError, FloatingPointError, RuntimeError):
            return None, False, True

    @staticmethod
    def _iv_price_upper_bound(
        *,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: Literal["call", "put"],
    ) -> float:
        """Theoretical upper bound for inverse-option prices."""
        if option_type == "call":
            return float(np.exp(-r * T) / K)
        return float(1.0 / S)

    @staticmethod
    def _initial_iv_guess(S: float, K: float) -> float:
        """ATM-style initial guess clipped to stable range."""
        moneyness = S / K
        sigma = float(0.5 + 0.1 * abs(np.log(moneyness)))
        return float(np.clip(sigma, 0.1, 2.0))

    @staticmethod
    def _solve_iv_newton(
        *,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: Literal["call", "put"],
        tol: float,
        max_iter: int,
    ) -> Optional[float]:
        """Attempt Newton IV solve, returning None when fallback is required."""
        sigma = InverseOptionPricer._initial_iv_guess(S, K)
        price_scale = max(abs(price), InverseOptionPricer.EPSILON)
        return _iterate_newton_iv_solve(
            price=price,
            S=S,
            K=K,
            T=T,
            r=r,
            option_type=option_type,
            sigma=sigma,
            rel_tol=tol,
            price_scale=price_scale,
            max_iter=max_iter,
            update_fn=InverseOptionPricer._newton_iv_update,
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
        """计算币本位期权的希腊字母。"""
        InverseOptionPricer._validate_option_type(option_type)
        InverseOptionPricer._validate_inputs(S, K, T, r, sigma)
        if T < InverseOptionPricer.EPSILON:
            return InverseGreeks(delta=0.0, gamma=0.0, theta=0.0, vega=0.0, rho=0.0)
        d1, d2 = InverseOptionPricer._calculate_d1_d2(S, K, T, r, sigma)
        inv_S = 1.0 / S
        inv_K = 1.0 / K
        discount = np.exp(-r * T)
        sqrt_T = np.sqrt(T)
        n_d1 = norm.pdf(d1)
        context = _GreeksComputationContext(
            inv_S=inv_S,
            inv_K=inv_K,
            discount=discount,
            sqrt_T=sqrt_T,
            n_d1=n_d1,
        )
        return InverseOptionPricer._calculate_greeks_from_d(
            d1, d2, S, K, T, r, sigma, option_type, context
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
        """通过牛顿迭代（失败回退二分法）计算隐含波动率。"""
        InverseOptionPricer._validate_option_type(option_type)
        if price <= 0: return 0.0
        if price >= InverseOptionPricer._iv_price_upper_bound(S=S, K=K, T=T, r=r, option_type=option_type): return 0.0
        if (raw_sigma := InverseOptionPricer._solve_iv_newton(
            price=price,
            S=S,
            K=K,
            T=T,
            r=r,
            option_type=option_type,
            tol=tol,
            max_iter=max_iter,
        )) is None:
            raw_sigma = InverseOptionPricer._iv_bisection(price, S, K, T, r, option_type, tol, max_iter)
        return InverseOptionPricer._stabilize_iv_estimate(raw_sigma, T, stabilize_short_maturity=stabilize_short_maturity, short_maturity_threshold=short_maturity_threshold, anchor_sigma=anchor_sigma, max_anchor_deviation=max_anchor_deviation)

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
        price_fn = lambda sigma: InverseOptionPricer.calculate_price(
            S, K, T, r, sigma, option_type
        )
        sigma_low, sigma_high, price_low, price_high = _inverse_iv_bracket_prices(
            price_fn=price_fn
        )
        # 检查价格是否在合理范围内
        if price < price_low or price > price_high:
            # 价格超出范围，可能是输入有误
            logger.warning(
                f"IV calculation: price {price} outside valid range [{price_low:.2e}, {price_high:.2e}] "
                f"for {option_type} S={S}, K={K}, T={T}"
            )
            return sigma_low if price < price_low else sigma_high
        return _inverse_iv_bisection_search(
            price_fn=price_fn,
            target_price=price,
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            tol=tol,
            max_iter=max_iter,
        )

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
    """币本位期权 Put-Call Parity 偏差，越接近 0 说明一致性越好。"""
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


def _vanilla_option_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"],
) -> float:
    if T <= InverseOptionPricer.EPSILON:
        return max(0.0, S - K) if option_type == "call" else max(0.0, K - S)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


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
    """Compute current option value, unrealized PnL, and market value for a position."""
    InverseOptionPricer._validate_option_type(option_type)
    InverseOptionPricer._validate_inputs(S, K, T, r, sigma)
    current_option_value = (
        InverseOptionPricer.calculate_price(S, K, T, r, sigma, option_type)
        if inverse
        else _vanilla_option_price(S, K, T, r, sigma, option_type)
    )
    market_value = size * current_option_value
    unrealized_pnl = market_value - size * avg_entry_price_usd
    return current_option_value, unrealized_pnl, market_value
