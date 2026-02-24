"""
隐含波动率计算与波动率曲面构建。

已实现:
- 期权定价 (Black-Scholes)
- 隐含波动率数值求解 (二分法、Newton-Raphson)
- 波动率曲面插值与可视化

参考:
- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
try:
    from scipy.optimize import minimize
    HAS_SCIPY_OPT = True
except ImportError:
    HAS_SCIPY_OPT = False
try:
    from py_vollib.black_scholes.implied_volatility import implied_volatility as _jaeckel_iv
    HAS_JAECKEL_SOLVER = True
except ImportError:
    HAS_JAECKEL_SOLVER = False

from utils.logging_config import get_logger, log_extra

logger = get_logger(__name__)


@dataclass
class VolatilityPoint:
    """波动率曲面上的一个点。"""
    strike: float
    expiry: float  # 年化到期时间
    volatility: float
    underlying_price: float
    is_call: bool = True

    @property
    def moneyness(self) -> float:
        """计算价内外程度 (moneyness)。"""
        return self.strike / self.underlying_price

    @property
    def log_moneyness(self) -> float:
        """计算对数价内外程度。"""
        return np.log(self.strike / self.underlying_price)


@dataclass
class SVIParams:
    """
    SVI 参数化 (raw SVI): w(k) = a + b * [rho*(k-m) + sqrt((k-m)^2 + sigma^2)].

    其中 w(k) 是总方差 (sigma_impl^2 * T), k 是 log-moneyness.
    """
    a: float
    b: float
    rho: float
    m: float
    sigma: float


@dataclass
class SSVIParams:
    """Global SSVI parameters."""
    rho: float
    eta: float
    lam: float


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float,
                       is_call: bool = True) -> float:
    """
    Black-Scholes 期权定价公式。

    Args:
        S: 标的资产价格
        K: 执行价格
        T: 到期时间 (年化)
        r: 无风险利率
        sigma: 波动率
        is_call: True=看涨, False=看跌

    Returns:
        期权理论价格
    """
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
        return intrinsic

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if is_call:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return float(price)


def black_scholes_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes Vega (价格对波动率的敏感度)。

    Args:
        S: 标的资产价格
        K: 执行价格
        T: 到期时间
        r: 无风险利率
        sigma: 波动率

    Returns:
        Vega 值
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1)

    return float(vega)


def _option_price_bounds(S: float, K: float, T: float, r: float, is_call: bool) -> Tuple[float, float]:
    """No-arbitrage bounds for vanilla option prices."""
    discount = np.exp(-r * T)
    if is_call:
        return max(0.0, S - K * discount), S
    return max(0.0, K * discount - S), K * discount


def implied_volatility_bisection(market_price: float, S: float, K: float, T: float,
                                 r: float = None, is_call: bool = True,
                                 tol: float = 1e-5, max_iter: int = 100) -> float:
    if r is None:
        r = float(os.getenv("RISK_FREE_RATE", "0.05"))
    """
    使用二分法求解隐含波动率。

    稳健但收敛较慢，适合作为初始估计。

    Args:
        market_price: 市场期权价格
        S: 标的资产价格
        K: 执行价格
        T: 到期时间
        r: 无风险利率
        is_call: 是否看涨期权
        tol: 收敛容差
        max_iter: 最大迭代次数

    Returns:
        隐含波动率

    Raises:
        ValueError: 如果无法收敛
    """
    if market_price <= 0:
        return 0.0

    lower_bound, upper_bound = _option_price_bounds(S, K, T, r, is_call)
    if market_price < lower_bound - 1e-10 or market_price > upper_bound + 1e-10:
        raise ValueError(
            f"Option price {market_price} violates no-arbitrage bounds "
            f"[{lower_bound}, {upper_bound}]"
        )

    # 二分法搜索范围
    sigma_low, sigma_high = 0.001, 5.0

    price_low = black_scholes_price(S, K, T, r, sigma_low, is_call)
    price_high = black_scholes_price(S, K, T, r, sigma_high, is_call)

    # 检查单调性
    if market_price < price_low:
        return sigma_low
    if market_price > price_high:
        return sigma_high

    for i in range(max_iter):
        sigma_mid = (sigma_low + sigma_high) / 2
        price_mid = black_scholes_price(S, K, T, r, sigma_mid, is_call)

        if abs(price_mid - market_price) < tol:
            return float(sigma_mid)

        if price_mid < market_price:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid

    # 返回最佳估计
    return float((sigma_low + sigma_high) / 2)


def implied_volatility_newton(market_price: float, S: float, K: float, T: float,
                             r: float = None, is_call: bool = True,
                             initial_sigma: float = 0.3,
                             tol: float = 1e-5, max_iter: int = 100) -> float:
    """
    使用 Newton-Raphson 方法求解隐含波动率。

    收敛快但需要好的初始值，Vega 接近 0 时可能不稳定。

    Args:
        market_price: 市场期权价格
        S: 标的资产价格
        K: 执行价格
        T: 到期时间
        r: 无风险利率
        is_call: 是否看涨期权
        initial_sigma: 初始波动率猜测
        tol: 收敛容差
        max_iter: 最大迭代次数

    Returns:
        隐含波动率
    """
    if r is None:
        r = float(os.getenv("RISK_FREE_RATE", "0.05"))
    sigma = initial_sigma

    for i in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, is_call)
        diff = market_price - price

        if abs(diff) < tol:
            return float(sigma)

        vega = black_scholes_vega(S, K, T, r, sigma)

        if vega < 1e-10:  # Vega 太小，退化为二分法
            return implied_volatility_bisection(
                market_price, S, K, T, r, is_call, tol, max_iter
            )

        # Newton 更新
        sigma_new = sigma + diff / vega

        # 保持正值和合理范围
        sigma_new = max(0.001, min(5.0, sigma_new))

        if abs(sigma_new - sigma) < tol:
            return float(sigma_new)

        sigma = sigma_new

    return float(sigma)


def _implied_volatility_lbr_fallback(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = None,
    is_call: bool = True,
    tol: float = 1e-8,
    max_iter: int = 20
) -> float:
    """
    近似 Let's-Be-Rational 风格 IV 回退求解器。

    实现思路:
    1) 价格边界检查
    2) Brenner-Subrahmanyam 初值
    3) Halley 迭代 (较 Newton 更稳健)
    4) 失败回退到二分法
    """
    if r is None:
        r = float(os.getenv("RISK_FREE_RATE", "0.05"))

    if market_price <= 0 or T <= 0:
        return 0.0

    lower_bound, upper_bound = _option_price_bounds(S, K, T, r, is_call)
    if market_price < lower_bound - 1e-10 or market_price > upper_bound + 1e-10:
        raise ValueError(
            f"Option price {market_price} violates no-arbitrage bounds "
            f"[{lower_bound}, {upper_bound}]"
        )

    # Brenner-Subrahmanyam ATM initial guess (clipped to stable range).
    sigma = np.sqrt(2 * np.pi / max(T, 1e-12)) * (market_price / max(S, 1e-12))
    sigma = float(np.clip(sigma, 0.01, 3.0))

    for _ in range(max_iter):
        price = black_scholes_price(S, K, T, r, sigma, is_call)
        diff = price - market_price
        if abs(diff) < tol:
            return float(sigma)

        vega = black_scholes_vega(S, K, T, r, sigma)
        if vega < 1e-12:
            break

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        volga = vega * d1 * d2 / max(sigma, 1e-12)

        # Halley update; fallback to Newton if denominator degenerates.
        denom = 2.0 * vega * vega - diff * volga
        if abs(denom) > 1e-14:
            step = (2.0 * diff * vega) / denom
        else:
            step = diff / vega

        sigma_new = sigma - step
        sigma_new = float(np.clip(sigma_new, 0.001, 5.0))
        if abs(sigma_new - sigma) < tol:
            return sigma_new
        sigma = sigma_new

    # Robust fallback
    return implied_volatility_bisection(
        market_price=market_price,
        S=S,
        K=K,
        T=T,
        r=r,
        is_call=is_call,
        tol=1e-8,
        max_iter=200
    )


def implied_volatility_jaeckel(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = None,
    is_call: bool = True,
    tol: float = 1e-8,
    max_iter: int = 20
) -> float:
    """
    Jaeckel Let's Be Rational IV 求解器（优先）+ 稳健回退。

    当环境中安装了 py_vollib（内部使用 Let's Be Rational）时，优先使用其解析近似；
    否则回退到本地 Halley+bisection 稳健实现。
    """
    if r is None:
        r = float(os.getenv("RISK_FREE_RATE", "0.05"))

    if market_price <= 0 or T <= 0:
        return 0.0

    lower_bound, upper_bound = _option_price_bounds(S, K, T, r, is_call)
    if market_price < lower_bound - 1e-10 or market_price > upper_bound + 1e-10:
        raise ValueError(
            f"Option price {market_price} violates no-arbitrage bounds "
            f"[{lower_bound}, {upper_bound}]"
        )

    if HAS_JAECKEL_SOLVER:
        flag = "c" if is_call else "p"
        try:
            iv = _jaeckel_iv(market_price, S, K, T, r, flag)
            if np.isfinite(iv):
                return float(np.clip(iv, 0.0, 5.0))
        except Exception as exc:  # pragma: no cover - fallback path
            logger.debug(
                "Jaeckel solver failed, fallback to local LBR",
                extra=log_extra(error=str(exc), strike=K, expiry=T)
            )

    return _implied_volatility_lbr_fallback(
        market_price=market_price,
        S=S,
        K=K,
        T=T,
        r=r,
        is_call=is_call,
        tol=tol,
        max_iter=max_iter,
    )


def implied_volatility_lbr(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = None,
    is_call: bool = True,
    tol: float = 1e-8,
    max_iter: int = 20
) -> float:
    """兼容入口：对外保持 lbr 命名，内部优先走 Jaeckel LBR。"""
    return implied_volatility_jaeckel(
        market_price=market_price,
        S=S,
        K=K,
        T=T,
        r=r,
        is_call=is_call,
        tol=tol,
        max_iter=max_iter,
    )


def implied_volatility(market_price: float, S: float, K: float, T: float,
                      r: float = None, is_call: bool = True,
                      method: str = "hybrid") -> float:
    """
    求解隐含波动率 (组合方法)。

    Args:
        market_price: 市场期权价格
        S: 标的资产价格
        K: 执行价格
        T: 到期时间
        r: 无风险利率
        is_call: 是否看涨期权
        method: "bisection", "newton", "lbr", "jaeckel" 或 "hybrid"

    Returns:
        隐含波动率
    """
    if r is None:
        r = float(os.getenv("RISK_FREE_RATE", "0.05"))
    method = method.lower()
    if method == "bisection":
        return implied_volatility_bisection(market_price, S, K, T, r, is_call)
    elif method == "newton":
        return implied_volatility_newton(market_price, S, K, T, r, is_call)
    elif method in {"lbr", "jaeckel"}:
        return implied_volatility_jaeckel(market_price, S, K, T, r, is_call)
    else:  # hybrid
        return implied_volatility_jaeckel(market_price, S, K, T, r, is_call)


class VolatilitySurface:
    """
    波动率曲面模型。

    存储和执行价/到期日的隐含波动率数据，提供插值和外推功能。
    """

    def __init__(self):
        self.points: List[VolatilityPoint] = []
        self._grid: Optional[Dict] = None
        self._svi_params: Dict[float, SVIParams] = {}
        self._ssvi_params: Optional[SSVIParams] = None
        self._ssvi_atm_expiries: Optional[np.ndarray] = None
        self._ssvi_atm_total_vars: Optional[np.ndarray] = None

    def add_point(self, point: VolatilityPoint) -> None:
        """添加波动率曲面上的一个点。"""
        self.points.append(point)
        self._grid = None  # 清空缓存
        self._svi_params = {}
        self._ssvi_params = None
        self._ssvi_atm_expiries = None
        self._ssvi_atm_total_vars = None

    def add_from_market_data(self, strikes: List[float], expiries: List[float],
                            market_prices: List[float], underlying: float,
                            r: float = None, is_calls: Optional[List[bool]] = None) -> None:
        """
        从市场数据构建波动率曲面。

        Args:
            strikes: 执行价格列表
            expiries: 到期时间列表 (年化)
            market_prices: 市场价格列表
            underlying: 标的资产价格
            r: 无风险利率
            is_calls: 期权类型列表 (默认全部为看涨)
        """
        if r is None:
            r = float(os.getenv("RISK_FREE_RATE", "0.05"))
        if is_calls is None:
            is_calls = [True] * len(strikes)

        for K, T, price, is_call in zip(strikes, expiries, market_prices, is_calls):
            try:
                iv = implied_volatility(price, underlying, K, T, r, is_call)
                point = VolatilityPoint(
                    strike=K, expiry=T, volatility=iv,
                    underlying_price=underlying, is_call=is_call
                )
                self.add_point(point)
            except ValueError as e:
                logger.warning("Could not compute IV", extra=log_extra(strike=K, expiry=T, error=str(e)))

    @staticmethod
    def _svi_total_variance(k: np.ndarray, params: SVIParams) -> np.ndarray:
        return params.a + params.b * (
            params.rho * (k - params.m) + np.sqrt((k - params.m) ** 2 + params.sigma ** 2)
        )

    @staticmethod
    def _ssvi_total_variance(k: np.ndarray, theta: np.ndarray, params: SSVIParams) -> np.ndarray:
        """SSVI total variance surface."""
        theta_safe = np.maximum(theta, 1e-10)
        phi = params.eta * np.power(theta_safe, -params.lam)
        term = phi * k + params.rho
        inner = np.maximum(term * term + 1.0 - params.rho ** 2, 1e-12)
        return 0.5 * theta_safe * (1.0 + params.rho * phi * k + np.sqrt(inner))

    def fit_ssvi(self, expiry_tol: float = 0.01) -> Optional[SSVIParams]:
        """Fit global SSVI parameters and ATM total variance term structure."""
        if len(self.points) < 8:
            return None

        expiries = sorted({float(p.expiry) for p in self.points})
        atm_expiries: List[float] = []
        atm_total_vars: List[float] = []

        for expiry in expiries:
            expiry_points = [p for p in self.points if abs(p.expiry - expiry) <= expiry_tol]
            if len(expiry_points) < 3:
                continue

            k = np.array([p.log_moneyness for p in expiry_points], dtype=float)
            w = np.array([p.volatility ** 2 * p.expiry for p in expiry_points], dtype=float)
            w = np.maximum(w, 1e-10)
            weights = np.exp(-8.0 * np.abs(k))
            theta_atm = float(np.average(w, weights=weights))
            atm_expiries.append(float(expiry))
            atm_total_vars.append(theta_atm)

        if len(atm_expiries) < 2:
            return None

        x_exp = np.array(atm_expiries, dtype=float)
        y_theta = np.maximum(np.array(atm_total_vars, dtype=float), 1e-10)
        # Enforce no-calendar-arbitrage ATM structure.
        y_theta = np.maximum.accumulate(y_theta)

        def theta_of_t(t: np.ndarray) -> np.ndarray:
            return np.interp(t, x_exp, y_theta, left=y_theta[0], right=y_theta[-1])

        if HAS_SCIPY_OPT:
            k_obs = np.array([p.log_moneyness for p in self.points], dtype=float)
            t_obs = np.array([float(p.expiry) for p in self.points], dtype=float)
            w_obs = np.array([p.volatility ** 2 * p.expiry for p in self.points], dtype=float)
            w_obs = np.maximum(w_obs, 1e-10)

            def objective(x: np.ndarray) -> float:
                params = SSVIParams(rho=float(x[0]), eta=float(x[1]), lam=float(x[2]))
                theta = theta_of_t(t_obs)
                w_model = self._ssvi_total_variance(k_obs, theta, params)
                if not np.all(np.isfinite(w_model)):
                    return 1e9

                # Stability guard: keep eta in a conservative region.
                eta_upper = 2.0 / (np.sqrt(np.max(theta)) * (1.0 + abs(params.rho) + 1e-8))
                penalty = 0.0
                if params.eta > eta_upper:
                    penalty += 1e5 * (params.eta - eta_upper) ** 2
                return float(np.mean((w_model - w_obs) ** 2) + penalty)

            init = np.array([-0.2, 1.0, 0.2], dtype=float)
            bounds = [(-0.999, 0.999), (1e-4, 10.0), (0.0, 0.5)]
            result = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
            x = result.x if result.success else init
            rho, eta, lam = float(x[0]), float(x[1]), float(x[2])
        else:
            rho, eta, lam = -0.2, 1.0, 0.2

        # Final eta clipping for numerical stability.
        eta_upper = 2.0 / (np.sqrt(np.max(y_theta)) * (1.0 + abs(rho) + 1e-8))
        eta = float(np.clip(eta, 1e-4, eta_upper))

        self._ssvi_params = SSVIParams(rho=rho, eta=eta, lam=lam)
        self._ssvi_atm_expiries = x_exp
        self._ssvi_atm_total_vars = y_theta
        return self._ssvi_params

    def fit_svi(self, expiry: float, expiry_tol: float = 0.01) -> Optional[SVIParams]:
        """拟合单一期限的 SVI 参数。"""
        expiry_points = [p for p in self.points if abs(p.expiry - expiry) <= expiry_tol]
        if len(expiry_points) < 5:
            return None

        k = np.array([p.log_moneyness for p in expiry_points], dtype=float)
        w = np.array([p.volatility ** 2 * p.expiry for p in expiry_points], dtype=float)
        w = np.maximum(w, 1e-8)

        if not HAS_SCIPY_OPT:
            # 无 scipy 时使用启发式参数，保证可用性
            params = SVIParams(
                a=float(np.min(w) * 0.8),
                b=float(max(1e-4, np.std(w))),
                rho=0.0,
                m=float(np.median(k)),
                sigma=0.1
            )
            self._svi_params[float(expiry)] = params
            return params

        def objective(x: np.ndarray) -> float:
            p = SVIParams(a=x[0], b=x[1], rho=x[2], m=x[3], sigma=x[4])
            model_w = self._svi_total_variance(k, p)
            if np.any(model_w <= 0):
                return 1e9
            return float(np.mean((model_w - w) ** 2))

        init = np.array([
            float(np.min(w) * 0.8),
            float(max(1e-4, np.std(w))),
            0.0,
            float(np.median(k)),
            0.1
        ])
        bounds = [
            (-5.0, 5.0),      # a
            (1e-8, 10.0),     # b
            (-0.999, 0.999),  # rho
            (-5.0, 5.0),      # m
            (1e-6, 5.0),      # sigma
        ]

        result = minimize(objective, init, method="L-BFGS-B", bounds=bounds)
        x = result.x if result.success else init
        params = SVIParams(a=float(x[0]), b=float(x[1]), rho=float(x[2]), m=float(x[3]), sigma=float(x[4]))
        self._svi_params[float(expiry)] = params
        return params

    def fit_all_svi(self) -> Dict[float, SVIParams]:
        """拟合所有期限的 SVI 参数。"""
        self._svi_params = {}
        for expiry in sorted({float(p.expiry) for p in self.points}):
            self.fit_svi(expiry)
        return self._svi_params

    def _vol_from_ssvi(self, strike: float, expiry: float) -> Optional[float]:
        """Try SSVI-based volatility lookup."""
        if self._ssvi_params is None or self._ssvi_atm_expiries is None or self._ssvi_atm_total_vars is None:
            self.fit_ssvi()

        if self._ssvi_params is None or self._ssvi_atm_expiries is None or self._ssvi_atm_total_vars is None:
            return None

        theta = np.interp(
            float(expiry),
            self._ssvi_atm_expiries,
            self._ssvi_atm_total_vars,
            left=float(self._ssvi_atm_total_vars[0]),
            right=float(self._ssvi_atm_total_vars[-1]),
        )
        theta = float(max(theta, 1e-10))
        k = np.log(strike / self.points[0].underlying_price)
        total_var = self._ssvi_total_variance(
            np.array([k], dtype=float),
            np.array([theta], dtype=float),
            self._ssvi_params,
        )[0]
        vol = np.sqrt(max(float(total_var), 1e-10) / max(float(expiry), 1e-8))
        return float(np.clip(vol, 0.01, 2.0))

    def _vol_from_svi(self, strike: float, expiry: float) -> Optional[float]:
        """Try SVI-based volatility lookup."""
        if not self._svi_params:
            self.fit_all_svi()

        if not self._svi_params:
            return None

        nearest_expiry = min(self._svi_params.keys(), key=lambda x: abs(x - expiry))
        params = self._svi_params[nearest_expiry]
        k = np.log(strike / self.points[0].underlying_price)
        total_var = self._svi_total_variance(np.array([k]), params)[0]
        t_eff = max(expiry, 1e-8)
        vol = np.sqrt(max(total_var, 1e-10) / t_eff)
        return float(np.clip(vol, 0.01, 2.0))

    def _vol_from_idw(self, strike: float, expiry: float) -> float:
        """Distance-weighted fallback volatility interpolation."""
        points_array = np.array([
            [p.log_moneyness, p.expiry, p.volatility]
            for p in self.points
        ])

        x_target = np.log(strike / self.points[0].underlying_price)
        y_target = expiry
        distances = np.sqrt(
            (points_array[:, 0] - x_target) ** 2 +
            (points_array[:, 1] - y_target) ** 2
        )
        distances = np.maximum(distances, 1e-10)

        weights = 1.0 / distances
        weights /= weights.sum()
        vol = np.sum(weights * points_array[:, 2])
        return float(np.clip(vol, 0.01, 2.0))

    def get_volatility(self, strike: float, expiry: float,
                      method: str = "linear") -> float:
        """
        插值获取任意点的隐含波动率。

        Args:
            strike: 执行价格
            expiry: 到期时间
            method: 插值方法 ("linear", "cubic", "nearest")

        Returns:
            插值后的隐含波动率
        """
        if len(self.points) == 0:
            return 0.2  # 默认值

        if len(self.points) == 1:
            return self.points[0].volatility

        if method == "ssvi":
            ssvi_vol = self._vol_from_ssvi(strike, expiry)
            if ssvi_vol is not None:
                return ssvi_vol

        if method == "svi":
            svi_vol = self._vol_from_svi(strike, expiry)
            if svi_vol is not None:
                return svi_vol

        return self._vol_from_idw(strike, expiry)

    def check_butterfly_arbitrage(
        self,
        expiry: float,
        n_strikes: int = 25,
        tol: float = 1e-6
    ) -> Dict[str, object]:
        """
        检查 butterfly 无套利: call(K) 对 K 应凸 (离散二阶差分 >= 0)。
        """
        if not self.points:
            return {"has_arbitrage": False, "violations": []}

        S = self.points[0].underlying_price
        r = float(os.getenv("RISK_FREE_RATE", "0.05"))
        strikes = np.linspace(0.6 * S, 1.4 * S, n_strikes)
        vols = np.array([self.get_volatility(float(K), expiry, method="svi") for K in strikes])
        calls = np.array([black_scholes_price(S, float(K), expiry, r, float(v), True) for K, v in zip(strikes, vols)])

        violations: List[Tuple[float, float]] = []
        for i in range(1, len(strikes) - 1):
            h1 = strikes[i] - strikes[i - 1]
            h2 = strikes[i + 1] - strikes[i]
            second_diff = (calls[i + 1] - calls[i]) / h2 - (calls[i] - calls[i - 1]) / h1
            if second_diff < -tol:
                violations.append((float(strikes[i]), float(second_diff)))

        return {
            "has_arbitrage": len(violations) > 0,
            "violations": violations
        }

    def check_calendar_arbitrage(
        self,
        moneyness_grid: Optional[List[float]] = None,
        tol: float = 1e-6
    ) -> Dict[str, object]:
        """
        检查 calendar 无套利: 固定 k 下，总方差 w(T)=sigma(T,k)^2*T 非递减。
        """
        if not self.points:
            return {"has_arbitrage": False, "violations": []}

        expiries = sorted({float(p.expiry) for p in self.points})
        if len(expiries) < 2:
            return {"has_arbitrage": False, "violations": []}

        S = self.points[0].underlying_price
        if moneyness_grid is None:
            moneyness_grid = [0.8, 0.9, 1.0, 1.1, 1.2]

        violations: List[Tuple[float, float, float]] = []
        for m in moneyness_grid:
            strike = S * m
            total_vars = []
            for T in expiries:
                vol = self.get_volatility(strike, T, method="svi")
                total_vars.append(vol * vol * T)
            for i in range(1, len(total_vars)):
                if total_vars[i] + tol < total_vars[i - 1]:
                    violations.append((float(m), float(expiries[i - 1]), float(expiries[i])))

        return {
            "has_arbitrage": len(violations) > 0,
            "violations": violations
        }

    def validate_no_arbitrage(self) -> Dict[str, object]:
        """综合检查 butterfly + calendar 无套利条件。"""
        expiries = sorted({float(p.expiry) for p in self.points})
        butterfly = {
            str(T): self.check_butterfly_arbitrage(T)
            for T in expiries
        }
        calendar = self.check_calendar_arbitrage()
        has_bfly = any(v["has_arbitrage"] for v in butterfly.values())
        return {
            "butterfly": butterfly,
            "calendar": calendar,
            "no_arbitrage": (not has_bfly) and (not calendar["has_arbitrage"])
        }

    def detect_arbitrage_opportunities(self) -> Dict[str, object]:
        """
        Return a flattened arbitrage diagnostics report for execution/risk modules.
        """
        checks = self.validate_no_arbitrage()
        findings: List[Dict[str, object]] = []

        for expiry, detail in checks.get("butterfly", {}).items():
            if not detail.get("has_arbitrage", False):
                continue
            for strike, second_diff in detail.get("violations", []):
                findings.append(
                    {
                        "type": "butterfly",
                        "expiry": float(expiry),
                        "strike": float(strike),
                        "severity": float(abs(second_diff)),
                        "detail": float(second_diff),
                    }
                )

        calendar = checks.get("calendar", {})
        if calendar.get("has_arbitrage", False):
            for moneyness, t_prev, t_next in calendar.get("violations", []):
                findings.append(
                    {
                        "type": "calendar",
                        "moneyness": float(moneyness),
                        "expiry_prev": float(t_prev),
                        "expiry_next": float(t_next),
                        "severity": float(abs(t_next - t_prev)),
                    }
                )

        findings_sorted = sorted(findings, key=lambda x: float(x.get("severity", 0.0)), reverse=True)
        return {
            "has_arbitrage": len(findings_sorted) > 0,
            "n_findings": len(findings_sorted),
            "findings": findings_sorted,
            "summary": checks,
        }

    def get_skew(
        self,
        expiry: float,
        stabilize_short_maturity: bool = False,
        short_maturity_threshold: float = 14.0 / 365.0,
        atm_anchor_window: float = 0.10,
        max_adjacent_jump: float = 0.20,
    ) -> List[Tuple[float, float]]:
        """
        获取特定到期日的波动率偏斜 (volatility skew/smile)。

        Returns:
            [(moneyness, volatility), ...] 列表
        """
        points = [p for p in self.points if abs(p.expiry - expiry) < 0.01]

        if not points:
            return []

        ordered_points = sorted(points, key=lambda x: x.strike)
        skew = [(p.moneyness, p.volatility) for p in ordered_points]

        if (
            not stabilize_short_maturity
            or expiry > short_maturity_threshold
            or len(skew) < 3
        ):
            return skew

        moneyness = np.array([m for m, _ in skew], dtype=float)
        vols = np.array([v for _, v in skew], dtype=float)
        log_m = np.log(np.maximum(moneyness, 1e-12))

        # ATM-prior shrinkage: short-dated wings keep more freedom, ATM gets stronger anchoring.
        window = max(atm_anchor_window, 1e-6)
        atm_weights = np.exp(-np.abs(log_m) / window)
        atm_anchor = float(np.average(vols, weights=atm_weights))
        smoothed = (1.0 - atm_weights) * vols + atm_weights * atm_anchor

        jump_cap = max(max_adjacent_jump, 1e-6)
        for i in range(1, len(smoothed)):
            smoothed[i] = float(np.clip(smoothed[i], smoothed[i - 1] - jump_cap, smoothed[i - 1] + jump_cap))
        for i in range(len(smoothed) - 2, -1, -1):
            smoothed[i] = float(np.clip(smoothed[i], smoothed[i + 1] - jump_cap, smoothed[i + 1] + jump_cap))

        return [(float(m), float(v)) for m, v in zip(moneyness, smoothed)]

    def get_term_structure(self, moneyness: float = 1.0) -> List[Tuple[float, float]]:
        """
        获取波动率期限结构。

        Returns:
            [(expiry, volatility), ...] 列表
        """
        points = [p for p in self.points if abs(p.moneyness - moneyness) < 0.05]

        if not points:
            return []

        return [(p.expiry, p.volatility) for p in sorted(points, key=lambda x: x.expiry)]

    def atm_volatility(self, expiry: float) -> float:
        """获取平值 (ATM) 隐含波动率。"""
        return self.get_volatility(self.points[0].underlying_price, expiry)

    def summary(self) -> Dict:
        """返回波动率曲面统计摘要。"""
        if not self.points:
            return {}

        vols = [p.volatility for p in self.points]

        return {
            "n_points": len(self.points),
            "min_vol": min(vols),
            "max_vol": max(vols),
            "mean_vol": np.mean(vols),
            "atm_vol": self.atm_volatility(self.points[0].expiry),
        }


def plot_volatility_surface(surface: VolatilitySurface,
                           strike_range: Tuple[float, float] = (0.8, 1.2),
                           expiry_range: Tuple[float, float] = (0.1, 1.0)):
    """
    绘制波动率曲面 (需要 matplotlib)。

    Args:
        surface: VolatilitySurface 对象
        strike_range: 价内外范围 (相对 ATM)
        expiry_range: 到期时间范围 (年化)
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.warning("matplotlib required for plotting")
        return

    if not surface.points:
        logger.warning("Empty surface, nothing to plot")
        return

    S = surface.points[0].underlying_price

    # 创建网格
    strikes = np.linspace(strike_range[0] * S, strike_range[1] * S, 50)
    expiries = np.linspace(expiry_range[0], expiry_range[1], 50)

    K_grid, T_grid = np.meshgrid(strikes, expiries)
    V_grid = np.zeros_like(K_grid)

    for i in range(len(strikes)):
        for j in range(len(expiries)):
            V_grid[j, i] = surface.get_volatility(strikes[i], expiries[j])

    fig = plt.figure(figsize=(12, 5))

    # 3D 曲面
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(K_grid / S, T_grid, V_grid, cmap='viridis')
    ax1.set_xlabel('Moneyness (K/S)')
    ax1.set_ylabel('Time to Expiry')
    ax1.set_zlabel('Implied Volatility')
    ax1.set_title('Volatility Surface')

    # 波动率微笑
    ax2 = fig.add_subplot(122)
    for T in np.linspace(0.1, 1.0, 5):
        skew = surface.get_skew(T)
        if skew:
            moneyness, vols = zip(*skew)
            ax2.plot(moneyness, vols, label=f'T={T:.2f}')

    ax2.set_xlabel('Moneyness')
    ax2.set_ylabel('Implied Volatility')
    ax2.set_title('Volatility Skew by Maturity')
    ax2.legend()

    plt.tight_layout()
    plt.show()
