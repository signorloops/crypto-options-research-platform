"""
历史波动率计算方法。

已实现:
- Realized Variance/Volatility (已实现方差/波动率)
- Parkinson Volatility (Parkinson 波动率，使用高低价)
- Garman-Klass Volatility (Garman-Klass 波动率)
- Rogers-Satchell Volatility (Rogers-Satchell 波动率)
- Yang-Zhang Volatility (Yang-Zhang 波动率)

参考:
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return.
- Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities.
- Rogers, L. C., & Satchell, S. E. (1991). Estimating variance from high, low and closing prices.
- Yang, D., & Zhang, Q. (2000). Drift-independent volatility estimation.
"""

import numpy as np
import pandas as pd


def realized_variance(returns: np.ndarray) -> float:
    """
    计算已实现方差 (Realized Variance)。

    RV = sum(r^2)

    Args:
        returns: 收益率序列

    Returns:
        已实现方差
    """
    return float(np.sum(returns ** 2))


def realized_volatility(returns: np.ndarray, annualize: bool = True, periods: int = 365) -> float:
    """
    计算已实现波动率 (Realized Volatility)。

    RV = sum(r^2), daily_var = RV / N, annualized_vol = sqrt(daily_var * periods)

    Args:
        returns: 收益率序列
        annualize: 是否年化
        periods: 年化周期数 (加密货币日数据=365, 小时=365*24)

    Returns:
        已实现波动率
    """
    n = len(returns)
    if n == 0:
        return 0.0

    rv = realized_variance(returns)
    vol = np.sqrt(rv / n)

    if annualize:
        vol *= np.sqrt(periods)

    return float(vol)


def parkinson_volatility(high: np.ndarray, low: np.ndarray,
                        annualize: bool = True, periods: int = 365,
                        unbiased: bool = True) -> float:
    """
    计算 Parkinson Volatility。

    使用日内最高价和最低价估计波动率，比收盘价估计更有效。

    sigma_p = sqrt( (1/(4N*ln2)) * sum(ln(hi/li))^2 )

    小样本修正: 当 unbiased=True 时，应用修正因子 n/(n-0.83) 来减少偏差。
    参考: Garman & Klass (1980) 的模拟研究表明 Parkinson 估计在小样本下有系统性低估。

    Args:
        high: 最高价序列
        low: 最低价序列
        annualize: 是否年化
        periods: 年化周期数
        unbiased: 是否应用小样本无偏修正 (默认 True)

    Returns:
        Parkinson 波动率估计
    """
    log_hl = np.log(high / low)
    n = len(log_hl)

    if n == 0:
        return 0.0

    # 4 * ln(2) ≈ 2.7726
    var = np.sum(log_hl ** 2) / (4 * n * np.log(2))

    # 小样本无偏修正 (参考 Garman & Klass, 1980)
    if unbiased and n > 1:
        # 修正因子: n / (n - 0.83) 来自极值理论的模拟结果
        # 也可以使用更简单的 n / (n - 1)，但 0.83 对 Parkinson 估计更精确
        adjustment = n / (n - 0.83)
        var *= adjustment

    vol = np.sqrt(var)

    if annualize:
        vol *= np.sqrt(periods)

    return float(vol)


def garman_klass_volatility(open: np.ndarray, high: np.ndarray,
                           low: np.ndarray, close: np.ndarray,
                           annualize: bool = True, periods: int = 365) -> float:
    """
    计算 Garman-Klass Volatility。

    使用 OHLC 数据，比 Parkinson 估计更有效。

    sigma_gk = sqrt( (1/N) * sum(0.5*ln(hi/li)^2 - (2*ln2-1)*ln(ci/oi)^2) )

    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        annualize: 是否年化
        periods: 年化周期数

    Returns:
        Garman-Klass 波动率估计
    """
    log_hl = np.log(high / low)
    log_oc = np.log(close / open)
    n = len(log_hl)

    if n == 0:
        return 0.0

    var = np.sum(0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_oc ** 2) / n

    # 处理可能的数值误差
    var = max(var, 0)
    vol = np.sqrt(var)

    if annualize:
        vol *= np.sqrt(periods)

    return float(vol)


def rogers_satchell_volatility(open: np.ndarray, high: np.ndarray,
                               low: np.ndarray, close: np.ndarray,
                               annualize: bool = True, periods: int = 365) -> float:
    """
    计算 Rogers-Satchell Volatility。

    对漂移稳健 (drift-independent) 的波动率估计。

    sigma_rs = sqrt( (1/N) * sum(ln(hi/ci)*ln(hi/oi) + ln(li/ci)*ln(li/oi)) )

    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        annualize: 是否年化
        periods: 年化周期数

    Returns:
        Rogers-Satchell 波动率估计
    """
    log_hc = np.log(high / close)
    log_ho = np.log(high / open)
    log_lc = np.log(low / close)
    log_lo = np.log(low / open)

    n = len(log_hc)
    if n == 0:
        return 0.0

    var = np.sum(log_hc * log_ho + log_lc * log_lo) / n
    var = max(var, 0)
    vol = np.sqrt(var)

    if annualize:
        vol *= np.sqrt(periods)

    return float(vol)


def yang_zhang_volatility(open: np.ndarray, high: np.ndarray,
                         low: np.ndarray, close: np.ndarray,
                         annualize: bool = True, periods: int = 365) -> float:
    """
    计算 Yang-Zhang Volatility。

    结合隔夜跳空 (overnight) 和日内波动，是最有效的波动率估计之一。

    sigma_yz = sqrt(sigma_o^2 + k*sigma_c^2 + (1-k)*sigma_rs^2)

    其中:
    - sigma_o: 隔夜波动率 (开盘对数收益率)
    - sigma_c: 收盘价波动率 (开收对数收益率)
    - sigma_rs: Rogers-Satchell 波动率
    - k = 0.34 / (1.34 + (n+1)/(n-1))

    Args:
        open: 开盘价序列
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        annualize: 是否年化
        periods: 年化周期数

    Returns:
        Yang-Zhang 波动率估计
    """
    n = len(open)
    if n < 2:
        return 0.0

    # 隔夜波动率 (overnight)
    log_oc_prev = np.log(open[1:] / close[:-1])
    # 使用 n-2 自由度进行无偏估计 (去均值消耗1个自由度)
    var_o = np.sum((log_oc_prev - np.mean(log_oc_prev)) ** 2) / (n - 2) if n > 2 else 0.0

    # 开盘-收盘波动率 (n-1 自由度无偏估计)
    log_oc = np.log(close / open)
    var_c = np.sum((log_oc - np.mean(log_oc)) ** 2) / (n - 1)

    # Rogers-Satchell 分量
    var_rs = rogers_satchell_volatility(open, high, low, close, annualize=False, periods=periods) ** 2

    # 权重 k
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    var_yz = var_o + k * var_c + (1 - k) * var_rs
    var_yz = max(var_yz, 0)
    vol = np.sqrt(var_yz)

    if annualize:
        vol *= np.sqrt(periods)

    return float(vol)


def calculate_volatility_from_ohlc(df: pd.DataFrame, method: str = "yang_zhang",
                                   annualize: bool = True, periods: int = 365) -> float:
    """
    从 OHLC DataFrame 计算波动率。

    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close']
        method: 计算方法 ("realized", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang")
        annualize: 是否年化
        periods: 年化周期数

    Returns:
        波动率估计值

    Example:
        >>> vol = calculate_volatility_from_ohlc(df, method="yang_zhang")
    """
    method = method.lower()

    if method == "realized":
        if 'close' not in df.columns:
            raise ValueError("realized method requires 'close' column")
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        return realized_volatility(returns.values, annualize, periods)

    elif method == "parkinson":
        if 'high' not in df.columns or 'low' not in df.columns:
            raise ValueError("parkinson method requires 'high' and 'low' columns")
        return parkinson_volatility(
            df['high'].values, df['low'].values, annualize, periods
        )

    elif method == "garman_klass":
        required = ['open', 'high', 'low', 'close']
        if not all(c in df.columns for c in required):
            raise ValueError(f"garman_klass method requires {required}")
        return garman_klass_volatility(
            df['open'].values, df['high'].values,
            df['low'].values, df['close'].values, annualize, periods
        )

    elif method == "rogers_satchell":
        required = ['open', 'high', 'low', 'close']
        if not all(c in df.columns for c in required):
            raise ValueError(f"rogers_satchell method requires {required}")
        return rogers_satchell_volatility(
            df['open'].values, df['high'].values,
            df['low'].values, df['close'].values, annualize, periods
        )

    elif method == "yang_zhang":
        required = ['open', 'high', 'low', 'close']
        if not all(c in df.columns for c in required):
            raise ValueError(f"yang_zhang method requires {required}")
        return yang_zhang_volatility(
            df['open'].values, df['high'].values,
            df['low'].values, df['close'].values, annualize, periods
        )

    else:
        raise ValueError(f"Unknown method: {method}")
