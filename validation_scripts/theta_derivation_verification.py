"""
币本位期权Theta公式的深度数学推导验证

重点验证:
1. dV_dT 的每一项符号
2. Theta符号错误的根源
"""

import numpy as np
from scipy.stats import norm


def derive_inverse_call_theta():
    """
    币本位看涨期权的Theta推导

    币本位看涨期权价格:
    V_call = e^(-rT) * (1/K) * N(-d2) - (1/S) * N(-d1)

    其中:
    d1 = [ln(K/S) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    需要计算 dV/dT
    """
    print("=" * 70)
    print("币本位看涨期权 Theta 推导")
    print("=" * 70)

    print("\n1. 价格公式:")
    print("   V_call = e^(-rT) * (1/K) * N(-d2) - (1/S) * N(-d1)")

    print("\n2. 对T求导 (dV/dT):")
    print("   第一项: d/dT[e^(-rT) * (1/K) * N(-d2)]")
    print("         = -r * e^(-rT) * (1/K) * N(-d2) + e^(-rT) * (1/K) * n(-d2) * (-dd2/dT)")
    print("         = -r * e^(-rT) * (1/K) * N(-d2) - e^(-rT) * (1/K) * n(d2) * dd2/dT")
    print("         (因为 n(-x) = n(x))")

    print("\n   第二项: d/dT[-(1/S) * N(-d1)]")
    print("         = -(1/S) * n(-d1) * (-dd1/dT)")
    print("         = (1/S) * n(d1) * dd1/dT")
    print("         (因为 n(-x) = n(x))")

    print("\n3. 合并:")
    print("   dV/dT = -r * e^(-rT) * (1/K) * N(-d2)")
    print("         - e^(-rT) * (1/K) * n(d2) * dd2/dT")
    print("         + (1/S) * n(d1) * dd1/dT")

    print("\n4. 关键问题：dd1/dT 和 dd2/dT 的符号")
    print("   d1 = [ln(K/S) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))")
    print("   令 A = ln(K/S) + (r + 0.5*sigma^2)*T")
    print("   则 d1 = A / (sigma*sqrt(T))")
    print("   dd1/dT = [A' * sigma*sqrt(T) - A * sigma/(2*sqrt(T))] / (sigma^2 * T)")
    print("          = [(r + 0.5*sigma^2) * sigma*sqrt(T) - A * sigma/(2*sqrt(T))] / (sigma^2 * T)")
    print("          = [(r + 0.5*sigma^2) * T - 0.5 * A] / (sigma * T^1.5)")
    print("          = [(r + 0.5*sigma^2) * T - 0.5*ln(K/S) - 0.5*(r + 0.5*sigma^2)*T] / (sigma * T^1.5)")
    print("          = [0.5*(r + 0.5*sigma^2)*T - 0.5*ln(K/S)] / (sigma * T^1.5)")
    print("          = [(r + 0.5*sigma^2)*T - ln(K/S)] / (2 * sigma * T^1.5)")

    print("\n   d2 = d1 - sigma*sqrt(T)")
    print("   dd2/dT = dd1/dT - sigma/(2*sqrt(T))")

    print("\n5. 代码实现对比:")
    print("   代码第264-265行:")
    print("   d_d1_dT = ((r + 0.5 * sigma ** 2) * T - np.log(K / S)) / (2 * sigma * T ** 1.5)")
    print("   d_d2_dT = d_d1_dT - sigma / (2 * sqrt_T)")
    print("   -> 这与推导一致 ✓")

    print("\n6. 代码第267-271行 (Call Theta):")
    print("   dV_dT = (-r * discount * inv_K * norm.cdf(-d2)")
    print("            - discount * inv_K * norm.pdf(d2) * d_d2_dT")
    print("            + inv_S * n_d1 * d_d1_dT)")

    print("\n   问题分析:")
    print("   - 第一项: -r * discount * inv_K * N(-d2) < 0 (正确，时间价值衰减)")
    print("   - 第二项: -discount * inv_K * n(d2) * dd2/dT")
    print("     符号取决于 dd2/dT")
    print("   - 第三项: +inv_S * n(d1) * dd1/dT")
    print("     符号取决于 dd1/dT")


def derive_inverse_put_theta():
    """
    币本位看跌期权的Theta推导

    币本位看跌期权价格:
    V_put = (1/S) * N(d1) - e^(-rT) * (1/K) * N(d2)
    """
    print("\n" + "=" * 70)
    print("币本位看跌期权 Theta 推导")
    print("=" * 70)

    print("\n1. 价格公式:")
    print("   V_put = (1/S) * N(d1) - e^(-rT) * (1/K) * N(d2)")

    print("\n2. 对T求导 (dV/dT):")
    print("   第一项: d/dT[(1/S) * N(d1)]")
    print("         = (1/S) * n(d1) * dd1/dT")

    print("\n   第二项: d/dT[-e^(-rT) * (1/K) * N(d2)]")
    print("         = r * e^(-rT) * (1/K) * N(d2) - e^(-rT) * (1/K) * n(d2) * dd2/dT")

    print("\n3. 合并:")
    print("   dV/dT = (1/S) * n(d1) * dd1/dT")
    print("         + r * e^(-rT) * (1/K) * N(d2)")
    print("         - e^(-rT) * (1/K) * n(d2) * dd2/dT")

    print("\n4. 代码第273-276行 (Put Theta):")
    print("   dV_dT = (r * discount * inv_K * norm.cdf(d2)")
    print("            + discount * inv_K * norm.pdf(d2) * d_d2_dT")
    print("            - inv_S * n_d1 * d_d1_dT)")

    print("\n   问题分析:")
    print("   - 第一项: +r * discount * inv_K * N(d2) > 0")
    print("     这与标准看跌期权Theta通常为负矛盾！")
    print("   - 但是币本位期权的特殊性：当利率r>0时，持有现金有收益")


def numerical_verification():
    """数值验证Theta计算"""
    print("\n" + "=" * 70)
    print("数值验证")
    print("=" * 70)

    # 测试参数
    S = 50000.0  # BTC价格
    K = 50000.0  # ATM
    T = 30 / 365.0  # 30天
    r = 0.05  # 5%利率
    sigma = 0.6  # 60%波动率

    inv_S = 1.0 / S
    inv_K = 1.0 / K
    discount = np.exp(-r * T)
    sqrt_T = np.sqrt(T)

    # 计算d1, d2
    d1 = (np.log(K / S) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    n_d1 = norm.pdf(d1)

    # 计算dd1/dT, dd2/dT
    d_d1_dT = ((r + 0.5 * sigma ** 2) * T - np.log(K / S)) / (2 * sigma * T ** 1.5)
    d_d2_dT = d_d1_dT - sigma / (2 * sqrt_T)

    print(f"\n参数: S={S}, K={K}, T={T:.4f}, r={r}, sigma={sigma}")
    print(f"d1={d1:.4f}, d2={d2:.4f}")
    print(f"dd1/dT={d_d1_dT:.6f}, dd2/dT={d_d2_dT:.6f}")

    # Call Theta各项
    print("\n--- 看涨期权 Theta 各项 ---")
    term1_call = -r * discount * inv_K * norm.cdf(-d2)
    term2_call = -discount * inv_K * norm.pdf(d2) * d_d2_dT
    term3_call = inv_S * n_d1 * d_d1_dT
    theta_call = (term1_call + term2_call + term3_call) / 365.0

    print(f"Term1 (-r*e^(-rT)*(1/K)*N(-d2)): {term1_call:.10f}")
    print(f"Term2 (-e^(-rT)*(1/K)*n(d2)*dd2/dT): {term2_call:.10f}")
    print(f"Term3 (+(1/S)*n(d1)*dd1/dT): {term3_call:.10f}")
    print(f"总和 (年化): {term1_call + term2_call + term3_call:.10f}")
    print(f"Theta (每日): {theta_call:.10f}")

    # Put Theta各项
    print("\n--- 看跌期权 Theta 各项 ---")
    term1_put = r * discount * inv_K * norm.cdf(d2)
    term2_put = discount * inv_K * norm.pdf(d2) * d_d2_dT
    term3_put = -inv_S * n_d1 * d_d1_dT
    theta_put = (term1_put + term2_put + term3_put) / 365.0

    print(f"Term1 (+r*e^(-rT)*(1/K)*N(d2)): {term1_put:.10f}")
    print(f"Term2 (+e^(-rT)*(1/K)*n(d2)*dd2/dT): {term2_put:.10f}")
    print(f"Term3 (-(1/S)*n(d1)*dd1/dT): {term3_put:.10f}")
    print(f"总和 (年化): {term1_put + term2_put + term3_put:.10f}")
    print(f"Theta (每日): {theta_put:.10f}")

    # 验证：使用有限差分
    print("\n--- 有限差分验证 ---")
    dt = 1e-6

    from research.pricing.inverse_options import InverseOptionPricer

    price_call_T = InverseOptionPricer.calculate_price(S, K, T, r, sigma, "call")
    price_call_T_dt = InverseOptionPricer.calculate_price(S, K, T - dt, r, sigma, "call")
    theta_fd_call = (price_call_T_dt - price_call_T) / dt / 365.0

    price_put_T = InverseOptionPricer.calculate_price(S, K, T, r, sigma, "put")
    price_put_T_dt = InverseOptionPricer.calculate_price(S, K, T - dt, r, sigma, "put")
    theta_fd_put = (price_put_T_dt - price_put_T) / dt / 365.0

    print(f"Call Theta (公式): {theta_call:.10f}")
    print(f"Call Theta (差分): {theta_fd_call:.10f}")
    print(f"相对误差: {abs(theta_call - theta_fd_call) / abs(theta_fd_call) * 100:.4f}%")

    print(f"\nPut Theta (公式): {theta_put:.10f}")
    print(f"Put Theta (差分): {theta_fd_put:.10f}")
    print(f"相对误差: {abs(theta_put - theta_fd_put) / abs(theta_fd_put) * 100:.4f}%")


if __name__ == "__main__":
    derive_inverse_call_theta()
    derive_inverse_put_theta()
    numerical_verification()
