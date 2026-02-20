"""
波动率预测模型。

已实现:
- EWMA (Exponentially Weighted Moving Average)
- GARCH(1,1) (Generalized Autoregressive Conditional Heteroskedasticity)
- HAR-RV (Heterogeneous Autoregressive for Realized Volatility)

参考:
- RiskMetrics (1996). Technical Document.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity.
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility.
"""
from typing import Tuple, Optional, Dict

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Optional scipy import for constrained optimization
try:
    from scipy.optimize import minimize, nnls
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _garch_terminal_variance_python(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    init_variance: float
) -> float:
    """Pure Python fallback for GARCH recursion."""
    variance = float(init_variance)
    for r in returns:
        variance = omega + alpha * (r * r) + beta * variance
    return variance


def _garch_log_likelihood_python(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    init_variance: float
) -> float:
    """Pure Python fallback for GARCH log-likelihood."""
    variance = float(init_variance)
    ll = 0.0
    for r in returns:
        variance = omega + alpha * (r * r) + beta * variance
        if variance <= 0.0:
            return float(-np.inf)
        ll += -0.5 * (np.log(2.0 * np.pi * variance) + (r * r) / variance)
    return float(ll)


if HAS_NUMBA:
    @njit(cache=True)
    def _garch_terminal_variance_numba(
        returns: np.ndarray,
        omega: float,
        alpha: float,
        beta: float,
        init_variance: float
    ) -> float:
        variance = init_variance
        for i in range(returns.shape[0]):
            r = returns[i]
            variance = omega + alpha * (r * r) + beta * variance
        return variance

    @njit(cache=True)
    def _garch_log_likelihood_numba(
        returns: np.ndarray,
        omega: float,
        alpha: float,
        beta: float,
        init_variance: float
    ) -> float:
        variance = init_variance
        ll = 0.0
        for i in range(returns.shape[0]):
            r = returns[i]
            variance = omega + alpha * (r * r) + beta * variance
            if variance <= 0.0:
                return -np.inf
            ll += -0.5 * (np.log(2.0 * np.pi * variance) + (r * r) / variance)
        return ll
else:
    _garch_terminal_variance_numba = None
    _garch_log_likelihood_numba = None


def _garch_terminal_variance_fast(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    init_variance: float
) -> float:
    if HAS_NUMBA and _garch_terminal_variance_numba is not None:
        return float(_garch_terminal_variance_numba(returns, omega, alpha, beta, init_variance))
    return _garch_terminal_variance_python(returns, omega, alpha, beta, init_variance)


def _garch_log_likelihood_fast(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    init_variance: float
) -> float:
    if HAS_NUMBA and _garch_log_likelihood_numba is not None:
        return float(_garch_log_likelihood_numba(returns, omega, alpha, beta, init_variance))
    return _garch_log_likelihood_python(returns, omega, alpha, beta, init_variance)


def ewma_volatility(returns: np.ndarray, lambda_param: float = 0.94,
                   annualize: bool = True, periods: int = 365) -> float:
    """
    EWMA (Exponentially Weighted Moving Average) 波动率。

    标准 RiskMetrics 模型: sigma_t^2 = lambda * sigma_{t-1}^2 + (1-lambda) * r_{t-1}^2

    Args:
        returns: 收益率序列
        lambda_param: 衰减因子 (RiskMetrics 推荐 0.94 用于日数据, 0.97 用于月数据)
        annualize: 是否年化
        periods: 年化周期数 (默认365，加密货币24/7交易)

    Returns:
        当前 EWMA 波动率估计

    Example:
        >>> vol = ewma_volatility(returns, lambda_param=0.94)
    """
    if len(returns) < 2:
        return 0.0

    vols = ewma_series(returns, lambda_param=lambda_param, annualize=annualize, periods=periods)
    return float(vols[-1]) if len(vols) > 0 else 0.0


def ewma_series(returns: np.ndarray, lambda_param: float = 0.94,
               annualize: bool = True, periods: int = 365) -> np.ndarray:
    """
    计算 EWMA 波动率序列（使用 scipy.signal.lfilter 向量化）。

    Args:
        returns: 收益率序列
        lambda_param: 衰减因子
        annualize: 是否年化
        periods: 年化周期数

    Returns:
        波动率序列 (与 returns 等长)
    """
    if len(returns) == 0:
        return np.array([])

    alpha = 1 - lambda_param
    r2 = returns ** 2

    try:
        from scipy.signal import lfilter
        init_var = np.var(returns) if len(returns) > 1 else r2[0]
        variances = lfilter([alpha], [1, -lambda_param], r2,
                            zi=[init_var * lambda_param])[0]
    except ImportError:
        variances = np.zeros(len(returns))
        variance = np.var(returns) if len(returns) > 1 else r2[0]
        for i in range(len(returns)):
            variance = lambda_param * variance + alpha * r2[i]
            variances[i] = variance

    vols = np.sqrt(np.maximum(variances, 0))

    if annualize:
        vols *= np.sqrt(periods)

    return vols


def garch_volatility(returns: np.ndarray, omega: float = 0.000001,
                    alpha: float = 0.1, beta: float = 0.85,
                    annualize: bool = True, periods: int = 365) -> float:
    """
    GARCH(1,1) 波动率预测。

    GARCH(1,1) 模型: sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

    典型参数:
    - omega: 长期平均方差成分
    - alpha: 对新闻的反应程度 (0.05-0.15)
    - beta: 持续性 (0.8-0.95)

    约束条件: alpha + beta < 1 (平稳性)

    Args:
        returns: 收益率序列
        omega: 常数项
        alpha: ARCH 参数
        beta: GARCH 参数
        annualize: 是否年化
        periods: 年化周期数

    Returns:
        下一期波动率预测

    Example:
        >>> vol = garch_volatility(returns, omega=1e-6, alpha=0.1, beta=0.85)
    """
    if len(returns) < 2:
        return 0.0

    # Enhanced parameter validation
    if omega < 0:
        raise ValueError(f"omega must be non-negative, got {omega}")
    if alpha < 0 or alpha > 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if beta < 0 or beta > 1:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if alpha + beta >= 1:
        raise ValueError(f"Non-stationary parameters: alpha + beta = {alpha + beta} >= 1")

    values = np.asarray(returns, dtype=np.float64)
    variance = _garch_terminal_variance_fast(values, omega, alpha, beta, float(np.var(values)))

    vol = np.sqrt(variance)

    if annualize:
        vol *= np.sqrt(periods)

    return float(vol)


def estimate_garch_params(returns: np.ndarray) -> Tuple[float, float, float]:
    """
    使用最大似然估计 GARCH(1,1) 参数。

    使用 scipy.optimize 进行约束优化，确保参数满足平稳性条件。

    Args:
        returns: 收益率序列

    Returns:
        (omega, alpha, beta) 参数元组
    """
    values = np.asarray(returns, dtype=np.float64)
    if len(values) < 30:
        # 样本量不足，返回默认参数
        return (float(np.var(values) * 0.01), 0.1, 0.85)

    # 如果 scipy 不可用，回退到固定参数
    if not HAS_SCIPY:
        return (float(np.var(values) * 0.01), 0.1, 0.85)

    # 初始参数估计
    var_returns = float(np.var(values))
    omega_init = var_returns * 0.01
    alpha_init = 0.1
    beta_init = 0.85

    def neg_log_likelihood(params):
        """负对数似然函数（用于最小化）。"""
        omega, alpha, beta = params
        # 参数约束检查
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10

        ll = _garch_log_likelihood_fast(values, omega, alpha, beta, var_returns)
        if not np.isfinite(ll):
            return 1e10
        return float(-ll)

    # 约束优化
    # 约束条件: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
    bounds = [(1e-8, None), (0, 0.5), (0, 0.99)]
    constraints = {'type': 'ineq', 'fun': lambda x: 0.999 - x[1] - x[2]}

    result = minimize(
        neg_log_likelihood,
        [omega_init, alpha_init, beta_init],
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )

    if result.success:
        omega, alpha, beta = result.x
        # 确保数值稳定性
        alpha = max(0.0, min(alpha, 0.5))
        beta = max(0.0, min(beta, 0.99))
        if alpha + beta >= 1.0:
            beta = 0.99 - alpha
        return (float(omega), float(alpha), float(beta))
    else:
        # 优化失败，返回初始值
        return (omega_init, alpha_init, beta_init)


def _garch_log_likelihood(returns: np.ndarray, omega: float,
                         alpha: float, beta: float) -> float:
    """计算 GARCH(1,1) 对数似然。"""
    values = np.asarray(returns, dtype=np.float64)
    init_variance = float(np.var(values))
    return _garch_log_likelihood_fast(values, omega, alpha, beta, init_variance)


def estimate_har_params(rv_daily: np.ndarray, periods: Tuple[int, int, int] = (1, 5, 22)) -> Tuple[float, float, float, float]:
    """
    使用 OLS 估计 HAR-RV 模型参数。

    HAR-RV 模型: RV_t = beta_0 + beta_d * RV_{t-1} + beta_w * RV_{t-5:t-1} + beta_m * RV_{t-22:t-1}

    Args:
        rv_daily: 日已实现波动率序列
        periods: (日, 周, 月) 周期

    Returns:
        (beta_0, beta_d, beta_w, beta_m) 参数元组
    """
    d, w, m = periods

    if len(rv_daily) < m + 2:
        # 样本不足，返回固定系数
        return (0.0, 0.4, 0.3, 0.2)

    # 构建特征矩阵
    n = len(rv_daily) - m
    X = np.zeros((n, 4))  # [1, RV_d, RV_w, RV_m]
    y = rv_daily[m:]

    for i in range(n):
        idx = i + m
        X[i, 0] = 1.0  # 截距
        X[i, 1] = rv_daily[idx - 1]  # 日
        X[i, 2] = np.mean(rv_daily[idx - w:idx])  # 周
        X[i, 3] = np.mean(rv_daily[idx - m:idx])  # 月

    # 估计参数，使用非负最小二乘(NNLS)确保经济学合理性
    try:
        if HAS_SCIPY:
            # 使用NNLS进行带约束的优化，保持OLS的最优性
            beta, _ = nnls(X, y)
        else:
            # 回退到OLS，然后强制截断（次优但可用）
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            beta = np.maximum(beta, 0)
        return (float(beta[0]), float(beta[1]), float(beta[2]), float(beta[3]))
    except (np.linalg.LinAlgError, ValueError):
        # 回退到固定系数
        return (0.0, 0.4, 0.3, 0.2)


def har_volatility(rv_daily: np.ndarray, periods: Tuple[int, int, int] = (1, 5, 22),
                   beta: Optional[Tuple[float, float, float, float]] = None) -> float:
    """
    HAR-RV (Heterogeneous Autoregressive Realized Volatility) 模型。

    HAR-RV 模型将 RV 回归到不同频率的成分上:
    RV_t = beta_0 + beta_d * RV_{t-1} + beta_w * RV_{t-5:t-1} + beta_m * RV_{t-22:t-1}

    这是一个"近似长记忆"模型，能捕捉波动率聚集和粗糙波动率特征。

    Args:
        rv_daily: 日已实现波动率序列
        periods: (日, 周, 月) 周期 (默认 1, 5, 22 天)
                 加密货币24/7交易，所以是连续日历日
        beta: 可选的预估计参数 (beta_0, beta_d, beta_w, beta_m)
              如果为 None，则自动从历史数据估计

    Returns:
        下一期 RV 预测

    Example:
        >>> rv_pred = har_volatility(rv_series)  # rv_series 是日 RV 序列

    Reference:
        Corsi, F. (2009). "A simple approximate long-memory model of realized volatility"
    """
    if len(rv_daily) < max(periods) + 1:
        return float(np.mean(rv_daily)) if len(rv_daily) > 0 else 0.0

    # 如果没有提供参数，从历史数据估计
    if beta is None:
        beta = estimate_har_params(rv_daily, periods)

    beta_0, beta_d, beta_w, beta_m = beta

    # 构建特征: 日、周、月 RV
    rv_d = rv_daily[-1]  # 昨日
    rv_w = np.mean(rv_daily[-5:])  # 本周 (过去5天)
    rv_m = np.mean(rv_daily[-22:])  # 本月 (过去22天)

    rv_pred = beta_0 + beta_d * rv_d + beta_w * rv_w + beta_m * rv_m

    return float(max(rv_pred, 0))


def har_forecast(rv_series: np.ndarray, h: int = 1,
                periods: Tuple[int, int, int] = (1, 5, 22)) -> float:
    """
    HAR-RV h 期向前预测。

    Args:
        rv_series: 日已实现波动率序列
        h: 预测期数 (1=明天, 5=一周后, 22=一月后)
        periods: HAR 周期

    Returns:
        h 期向前 RV 预测
    """
    # 迭代预测
    forecast = rv_series.copy()

    for _ in range(h):
        next_rv = har_volatility(forecast, periods)
        forecast = np.append(forecast, next_rv)

    return float(forecast[-1])


def rough_volatility_signature(log_prices: np.ndarray, sampling: str = "daily") -> float:
    """
    估计波动率的粗糙指数 (Roughness Index)。

    粗糙指数 H 表征波动率路径的平滑程度:
    - H ≈ 0.1: 典型的粗糙波动率 (rough volatility)
    - H = 0.5: 布朗运动 (随机游走)
    - H > 0.5: 长记忆过程

    现代研究 (Gatheral et al., 2018) 发现实际波动率 H ≈ 0.1，称为"粗糙波动率"。

    Args:
        log_prices: 对数价格序列
        sampling: 采样频率 ("daily", "hourly", "minute")

    Returns:
        Hurst 指数 H (粗糙指数)

    Reference:
        Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). "Volatility is rough"
    """
    if len(log_prices) < 100:
        return 0.1  # 数据不足时返回典型值

    # 计算对数收益率
    returns = np.diff(log_prices)

    # 对于不同时间尺度 delta，计算非重叠 block 累加增量的 q-阶变差
    def mqd(delta: int, q: float = 2.0) -> float:
        """计算 delta 时间尺度的 q-阶矩（非重叠 block 聚合）。"""
        n_blocks = len(returns) // delta
        if n_blocks == 0:
            return np.nan
        blocked = returns[:n_blocks * delta].reshape(n_blocks, delta).sum(axis=1)
        return np.mean(np.abs(blocked) ** q)

    # 不同时间尺度
    scales = np.array([1, 2, 4, 8, 16])
    mq_values = np.array([mqd(int(s)) for s in scales])

    # 对数回归: log(M_q(delta)) ≈ q*H*log(delta) + const
    valid = mq_values > 0
    if np.sum(valid) < 2:
        return 0.1

    log_scales = np.log(scales[valid])
    log_mq = np.log(mq_values[valid])

    # 线性回归估计斜率
    slope = np.polyfit(log_scales, log_mq, 1)[0]
    H = slope / 2.0  # q=2 时斜率 = 2*H

    return float(np.clip(H, 0.01, 0.5))


def bipower_variation(
    returns: np.ndarray,
    annualize: bool = True,
    periods: int = 365
) -> float:
    """
    Bipower variation, 对跳跃更稳健的连续方差估计。
    """
    n = len(returns)
    if n < 2:
        return 0.0
    mu1 = np.sqrt(2.0 / np.pi)
    bv = (1.0 / (mu1 ** 2)) * np.sum(np.abs(returns[1:]) * np.abs(returns[:-1])) / (n - 1)
    vol = np.sqrt(max(bv, 0.0))
    if annualize:
        vol *= np.sqrt(periods)
    return float(vol)


def medrv_volatility(
    returns: np.ndarray,
    annualize: bool = True,
    periods: int = 365
) -> float:
    """
    Median Realized Volatility (MedRV), 对跳跃和异常点鲁棒。
    """
    n = len(returns)
    if n < 3:
        return float(np.std(returns) * np.sqrt(periods) if annualize and n > 0 else np.std(returns))

    med = np.median(
        np.vstack([
            np.abs(returns[:-2]),
            np.abs(returns[1:-1]),
            np.abs(returns[2:])
        ]),
        axis=0
    )
    c = np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi)
    medrv = c * (n / (n - 2.0)) * np.mean(med ** 2)
    vol = np.sqrt(max(medrv, 0.0))
    if annualize:
        vol *= np.sqrt(periods)
    return float(vol)


def two_scale_realized_volatility(
    returns: np.ndarray,
    k: int = 5,
    annualize: bool = True,
    periods: int = 365
) -> float:
    """
    Two-Scale Realized Volatility (TSRV) 简化实现, 用于缓解微结构噪声。
    """
    n = len(returns)
    if n < max(5, k + 1):
        return 0.0

    rv_fast = np.sum(returns ** 2)
    rv_sparse = 0.0
    for offset in range(k):
        subs = returns[offset::k]
        if len(subs) > 0:
            rv_sparse += np.sum(subs ** 2)
    rv_sparse /= k

    tsrv = rv_sparse - (k / n) * rv_fast
    tsrv = max(float(tsrv / max(1, n // k)), 0.0)
    vol = np.sqrt(tsrv)
    if annualize:
        vol *= np.sqrt(periods)
    return float(vol)


def realized_kernel_volatility(
    returns: np.ndarray,
    bandwidth: int = 5,
    annualize: bool = True,
    periods: int = 365
) -> float:
    """
    Realized Kernel volatility (Bartlett kernel)。
    """
    n = len(returns)
    if n < 2:
        return 0.0

    rv = np.sum(returns ** 2)
    h_max = min(bandwidth, n - 1)
    for h in range(1, h_max + 1):
        weight = 1.0 - h / (h_max + 1.0)
        gamma_h = np.sum(returns[h:] * returns[:-h])
        rv += 2.0 * weight * gamma_h

    var = max(rv / n, 0.0)
    vol = np.sqrt(var)
    if annualize:
        vol *= np.sqrt(periods)
    return float(vol)


def egarch_volatility(
    returns: np.ndarray,
    omega: float = -0.1,
    alpha: float = 0.1,
    beta: float = 0.95,
    gamma: float = -0.1,
    annualize: bool = True,
    periods: int = 365
) -> float:
    """
    EGARCH(1,1): log(sigma_t^2) 建模, 能捕捉杠杆效应。
    """
    if len(returns) < 2:
        return 0.0

    log_var = np.log(np.var(returns) + 1e-12)
    e_abs = np.sqrt(2.0 / np.pi)

    for r in returns:
        sigma = np.sqrt(np.exp(log_var))
        z = r / max(sigma, 1e-12)
        log_var = omega + beta * log_var + alpha * (abs(z) - e_abs) + gamma * z

    vol = np.sqrt(np.exp(log_var))
    if annualize:
        vol *= np.sqrt(periods)
    return float(vol)


def gjr_garch_volatility(
    returns: np.ndarray,
    omega: float = 1e-6,
    alpha: float = 0.08,
    beta: float = 0.88,
    gamma: float = 0.08,
    annualize: bool = True,
    periods: int = 365
) -> float:
    """
    GJR-GARCH(1,1): 对负收益引入额外冲击项。
    """
    if len(returns) < 2:
        return 0.0

    var = np.var(returns)
    for r in returns:
        indicator = 1.0 if r < 0 else 0.0
        var = omega + (alpha + gamma * indicator) * (r ** 2) + beta * var
        var = max(var, 1e-12)

    vol = np.sqrt(var)
    if annualize:
        vol *= np.sqrt(periods)
    return float(vol)


def hamilton_filter_regime_switching(
    returns: np.ndarray,
    n_iter: int = 20
) -> Dict:
    """
    2-state Hamilton filter (EM + transition matrix).
    """
    x = np.asarray(returns, dtype=float)
    if len(x) < 20:
        vol = float(np.std(x) + 1e-6) if len(x) else 0.01
        mean = float(np.mean(x)) if len(x) else 0.0
        return {
            "current_high_vol_probability": 0.5,
            "transition_matrix": [[0.95, 0.05], [0.05, 0.95]],
            "state_means": [mean, mean],
            "state_vols": [vol, vol],
            "low_vol_state": {"mean": mean, "vol": vol},
            "high_vol_state": {"mean": mean, "vol": vol},
            "threshold": float(np.median(np.abs(x))) if len(x) else 0.0,
            "smoothed_probabilities": []
        }

    # Initialization from quantiles
    q = np.quantile(np.abs(x), [0.4, 0.8])
    s0 = np.abs(x) <= q[0]
    s1 = np.abs(x) >= q[1]
    mu = np.array([
        np.mean(x[s0]) if np.any(s0) else np.mean(x),
        np.mean(x[s1]) if np.any(s1) else np.mean(x),
    ])
    sigma = np.array([
        max(np.std(x[s0]) if np.any(s0) else np.std(x), 1e-4),
        max(np.std(x[s1]) if np.any(s1) else np.std(x), 1e-4),
    ])
    P = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)
    pi = np.array([0.5, 0.5], dtype=float)

    def emission(val: float) -> np.ndarray:
        coef = 1.0 / (np.sqrt(2 * np.pi) * sigma)
        expo = np.exp(-0.5 * ((val - mu) / sigma) ** 2)
        return np.maximum(coef * expo, 1e-14)

    T = len(x)
    for _ in range(n_iter):
        # Forward
        alpha = np.zeros((T, 2), dtype=float)
        c = np.zeros(T, dtype=float)
        alpha[0] = pi * emission(x[0])
        c[0] = max(alpha[0].sum(), 1e-14)
        alpha[0] /= c[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ P) * emission(x[t])
            c[t] = max(alpha[t].sum(), 1e-14)
            alpha[t] /= c[t]

        # Backward
        beta = np.zeros((T, 2), dtype=float)
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (P @ (emission(x[t + 1]) * beta[t + 1])) / max(c[t + 1], 1e-14)

        gamma_prob = alpha * beta
        gamma_prob /= np.maximum(gamma_prob.sum(axis=1, keepdims=True), 1e-14)

        xi_sum = np.zeros((2, 2), dtype=float)
        for t in range(T - 1):
            num = np.outer(alpha[t], emission(x[t + 1]) * beta[t + 1]) * P
            denom = max(num.sum(), 1e-14)
            xi_sum += num / denom

        pi = gamma_prob[0]
        P = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-14)
        mu = (gamma_prob.T @ x) / np.maximum(gamma_prob.sum(axis=0), 1e-14)
        for j in range(2):
            var_j = np.sum(gamma_prob[:, j] * (x - mu[j]) ** 2) / max(np.sum(gamma_prob[:, j]), 1e-14)
            sigma[j] = max(np.sqrt(var_j), 1e-4)

    high_idx = int(np.argmax(sigma))
    low_idx = 1 - high_idx
    return {
        "current_high_vol_probability": float(gamma_prob[-1, high_idx]),
        "transition_matrix": P.tolist(),
        "state_means": [float(mu[low_idx]), float(mu[high_idx])],
        "state_vols": [float(sigma[low_idx]), float(sigma[high_idx])],
        "low_vol_state": {"mean": float(mu[low_idx]), "vol": float(sigma[low_idx])},
        "high_vol_state": {"mean": float(mu[high_idx]), "vol": float(sigma[high_idx])},
        "threshold": float(np.median(np.abs(x))),
        "smoothed_probabilities": gamma_prob.tolist(),
    }


def volatility_regime_switching(returns: np.ndarray, n_states: int = 2, method: str = "hamilton") -> dict:
    """
    简单的波动率状态转换模型 (Hamilton Filter 简化版)。

    识别高波动率和低波动率两种状态。

    Args:
        returns: 收益率序列
        n_states: 状态数 (2=高低波动率)

    Returns:
        包含状态概率和参数的字典
    """
    if n_states != 2:
        raise NotImplementedError("Only 2-state model implemented")

    if method == "hamilton":
        return hamilton_filter_regime_switching(returns)

    # 简单实现: 使用收益率绝对值的阈值划分
    abs_returns = np.abs(returns)

    # K-means 风格的分割
    threshold = np.median(abs_returns)

    low_vol_mask = abs_returns <= threshold
    high_vol_mask = ~low_vol_mask

    mu_low = np.mean(returns[low_vol_mask]) if np.any(low_vol_mask) else 0
    sigma_low = np.std(returns[low_vol_mask]) if np.any(low_vol_mask) else 0.01

    mu_high = np.mean(returns[high_vol_mask]) if np.any(high_vol_mask) else 0
    sigma_high = np.std(returns[high_vol_mask]) if np.any(high_vol_mask) else 0.05

    # 计算当前状态概率 (最后 N 个观测)
    recent = abs_returns[-min(20, len(abs_returns)):]
    p_high = np.mean(recent > threshold)

    return {
        "current_high_vol_probability": float(p_high),
        "low_vol_state": {"mean": float(mu_low), "vol": float(sigma_low)},
        "high_vol_state": {"mean": float(mu_high), "vol": float(sigma_high)},
        "threshold": float(threshold),
    }
