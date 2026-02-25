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

from research.volatility.iv_solvers import (
    black_scholes_price,
    black_scholes_vega,
    implied_volatility,
    implied_volatility_bisection,
    implied_volatility_jaeckel,
    implied_volatility_lbr,
    implied_volatility_newton,
)
from research.volatility.surface_fit import (
    fit_all_svi as _fit_all_svi,
    fit_ssvi as _fit_ssvi,
    fit_svi as _fit_svi,
    ssvi_total_variance as _ssvi_total_variance,
    svi_total_variance as _svi_total_variance,
)
from research.volatility.surface_query import (
    atm_volatility as _atm_volatility,
    check_butterfly_arbitrage as _check_butterfly_arbitrage,
    check_calendar_arbitrage as _check_calendar_arbitrage,
    detect_arbitrage_opportunities as _detect_arbitrage_opportunities,
    get_skew as _get_skew,
    get_term_structure as _get_term_structure,
    get_volatility as _get_volatility,
    summary as _summary,
    validate_no_arbitrage as _validate_no_arbitrage,
    vol_from_idw as _vol_from_idw,
    vol_from_ssvi as _vol_from_ssvi,
    vol_from_svi as _vol_from_svi,
)
from utils.logging_config import get_logger, log_extra

logger = get_logger(__name__)

__all__ = [
    "VolatilityPoint",
    "SVIParams",
    "SSVIParams",
    "black_scholes_price",
    "black_scholes_vega",
    "implied_volatility_bisection",
    "implied_volatility_newton",
    "implied_volatility_jaeckel",
    "implied_volatility_lbr",
    "implied_volatility",
    "VolatilitySurface",
    "plot_volatility_surface",
]


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

    def add_from_market_data(
        self,
        strikes: List[float],
        expiries: List[float],
        market_prices: List[float],
        underlying: float,
        r: float = None,
        is_calls: Optional[List[bool]] = None,
    ) -> None:
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
                    strike=K, expiry=T, volatility=iv, underlying_price=underlying, is_call=is_call
                )
                self.add_point(point)
            except ValueError as e:
                logger.warning(
                    "Could not compute IV", extra=log_extra(strike=K, expiry=T, error=str(e))
                )

    @staticmethod
    def _svi_total_variance(k: np.ndarray, params: SVIParams) -> np.ndarray:
        return _svi_total_variance(k, params)

    @staticmethod
    def _ssvi_total_variance(k: np.ndarray, theta: np.ndarray, params: SSVIParams) -> np.ndarray:
        return _ssvi_total_variance(k, theta, params)

    def fit_ssvi(self, expiry_tol: float = 0.01) -> Optional[SSVIParams]:
        return _fit_ssvi(self, ssvi_params_cls=SSVIParams, expiry_tol=expiry_tol)

    def fit_svi(self, expiry: float, expiry_tol: float = 0.01) -> Optional[SVIParams]:
        return _fit_svi(self, expiry=expiry, svi_params_cls=SVIParams, expiry_tol=expiry_tol)

    def fit_all_svi(self) -> Dict[float, SVIParams]:
        return _fit_all_svi(self)

    def _vol_from_ssvi(self, strike: float, expiry: float) -> Optional[float]:
        return _vol_from_ssvi(self, strike, expiry, self._ssvi_total_variance)

    def _vol_from_svi(self, strike: float, expiry: float) -> Optional[float]:
        return _vol_from_svi(self, strike, expiry, self._svi_total_variance)

    def _vol_from_idw(self, strike: float, expiry: float) -> float:
        return _vol_from_idw(self, strike, expiry)

    def get_volatility(self, strike: float, expiry: float, method: str = "linear") -> float:
        return _get_volatility(self, strike, expiry, method=method)

    def check_butterfly_arbitrage(
        self, expiry: float, n_strikes: int = 25, tol: float = 1e-6
    ) -> Dict[str, object]:
        return _check_butterfly_arbitrage(self, expiry=expiry, n_strikes=n_strikes, tol=tol)

    def check_calendar_arbitrage(
        self, moneyness_grid: Optional[List[float]] = None, tol: float = 1e-6
    ) -> Dict[str, object]:
        return _check_calendar_arbitrage(self, moneyness_grid=moneyness_grid, tol=tol)

    def validate_no_arbitrage(self) -> Dict[str, object]:
        return _validate_no_arbitrage(self)

    def detect_arbitrage_opportunities(self) -> Dict[str, object]:
        return _detect_arbitrage_opportunities(self)

    def get_skew(
        self,
        expiry: float,
        stabilize_short_maturity: bool = False,
        short_maturity_threshold: float = 14.0 / 365.0,
        atm_anchor_window: float = 0.10,
        max_adjacent_jump: float = 0.20,
    ) -> List[Tuple[float, float]]:
        return _get_skew(
            self,
            expiry=expiry,
            stabilize_short_maturity=stabilize_short_maturity,
            short_maturity_threshold=short_maturity_threshold,
            atm_anchor_window=atm_anchor_window,
            max_adjacent_jump=max_adjacent_jump,
        )

    def get_term_structure(self, moneyness: float = 1.0) -> List[Tuple[float, float]]:
        return _get_term_structure(self, moneyness=moneyness)

    def atm_volatility(self, expiry: float) -> float:
        return _atm_volatility(self, expiry=expiry)

    def summary(self) -> Dict:
        return _summary(self)


def plot_volatility_surface(
    surface: VolatilitySurface,
    strike_range: Tuple[float, float] = (0.8, 1.2),
    expiry_range: Tuple[float, float] = (0.1, 1.0),
):
    """
    绘制波动率曲面 (需要 matplotlib)。

    Args:
        surface: VolatilitySurface 对象
        strike_range: 价内外范围 (相对 ATM)
        expiry_range: 到期时间范围 (年化)
    """
    try:
        import matplotlib.pyplot as plt
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
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(K_grid / S, T_grid, V_grid, cmap="viridis")
    ax1.set_xlabel("Moneyness (K/S)")
    ax1.set_ylabel("Time to Expiry")
    ax1.set_zlabel("Implied Volatility")
    ax1.set_title("Volatility Surface")

    # 波动率微笑
    ax2 = fig.add_subplot(122)
    for T in np.linspace(0.1, 1.0, 5):
        skew = surface.get_skew(T)
        if skew:
            moneyness, vols = zip(*skew)
            ax2.plot(moneyness, vols, label=f"T={T:.2f}")

    ax2.set_xlabel("Moneyness")
    ax2.set_ylabel("Implied Volatility")
    ax2.set_title("Volatility Skew by Maturity")
    ax2.legend()

    plt.tight_layout()
    plt.show()
