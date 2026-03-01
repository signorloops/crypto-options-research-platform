"""
币本位期权Gamma公式推导分析

币本位期权定价：
- Call: C = e^(-rT)*(1/K)*N(-d2) - (1/S)*N(-d1)
- Put: P = (1/S)*N(d1) - e^(-rT)*(1/K)*N(d2)

其中：
- d1 = [ln(K/S) + (r + 0.5*σ²)*T] / (σ*√T)
- d2 = d1 - σ*√T

注意：ln(K/S) = -ln(S/K)，所以币本位的d1与标准BS的d1符号相反
"""

import numpy as np
from scipy.stats import norm


def calculate_d1_d2(S, K, T, r, sigma):
    """计算d1和d2（币本位版本）"""
    d1 = (np.log(K / S) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def gamma_derivation_step_by_step(S, K, T, r, sigma):
    """
    逐步推导Gamma公式

    币本位Call价格：C = e^(-rT)*(1/K)*N(-d2) - (1/S)*N(-d1)

    计算dC/dS：
    - 第一项：e^(-rT)*(1/K)*n(-d2)*(-dd2/dS)
    - 第二项：-(-1/S²)*N(-d1) - (1/S)*n(-d1)*(-dd1/dS)
           = (1/S²)*N(-d1) + (1/S)*n(-d1)*(dd1/dS)

    计算dd1/dS和dd2/dS：
    d1 = [ln(K) - ln(S) + (r + 0.5*σ²)*T] / (σ*√T)
    dd1/dS = -1/(S*σ*√T)

    d2 = d1 - σ*√T
    dd2/dS = dd1/dS = -1/(S*σ*√T)

    所以：
    dC/dS = e^(-rT)*(1/K)*n(-d2)*(1/(S*σ*√T)) + (1/S²)*N(-d1) + (1/S)*n(-d1)*(1/(S*σ*√T))

    注意：n(-d2) = n(d2)（正态分布对称性）
          n(-d1) = n(d1)

    并且：d2 = d1 - σ*√T，所以 n(d2) = n(d1) * exp(d1*σ*√T - 0.5*σ²*T) ... 这个关系比较复杂

    实际上，根据Black-Scholes的恒等式：
    S*n(d1) = K*e^(-rT)*n(d2) 对于标准BS

    但对于币本位，我们需要重新推导...
    """
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    inv_S = 1.0 / S
    inv_K = 1.0 / K
    discount = np.exp(-r * T)
    sqrt_T = np.sqrt(T)
    n_d1 = norm.pdf(d1)
    n_d2 = norm.pdf(d2)

    print(f"参数: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"d1 = {d1:.6f}, d2 = {d2:.6f}")
    print(f"n(d1) = {n_d1:.10e}, n(d2) = {n_d2:.10e}")
    print(f"N(-d1) = {norm.cdf(-d1):.6f}, N(-d2) = {norm.cdf(-d2):.6f}")
    print(f"N(d1) = {norm.cdf(d1):.6f}, N(d2) = {norm.cdf(d2):.6f}")

    # 币本位Call的Delta
    delta_call = (inv_S ** 2) * norm.cdf(-d1)
    print(f"\n币本位Call Delta = (1/S²)*N(-d1) = {delta_call:.10e}")

    # 币本位Put的Delta
    delta_put = -(inv_S ** 2) * norm.cdf(d1)
    print(f"币本位Put Delta = -(1/S²)*N(d1) = {delta_put:.10e}")

    # 现在计算Gamma（Delta对S的导数）
    # d/dS[(1/S²)*N(-d1)] = d/dS[S^(-2)*N(-d1)]
    # = -2*S^(-3)*N(-d1) + S^(-2)*n(-d1)*(-dd1/dS)
    # = -2/S³*N(-d1) - S^(-2)*n(d1)*(-1/(S*σ*√T))
    # = -2/S³*N(-d1) + n(d1)/(S³*σ*√T)

    dd1_dS = -1.0 / (S * sigma * sqrt_T)
    print(f"\ndd1/dS = {dd1_dS:.10e}")

    # Call Gamma推导
    gamma_call_term1 = -2 * (inv_S ** 3) * norm.cdf(-d1)
    gamma_call_term2 = n_d1 / (S ** 3 * sigma * sqrt_T)  # 注意这里是正号
    gamma_call = gamma_call_term1 + gamma_call_term2

    print(f"\n=== Call Gamma推导 ===")
    print(f"Term1 = -2/S³ * N(-d1) = {gamma_call_term1:.10e}")
    print(f"Term2 = n(d1)/(S³*σ*√T) = {gamma_call_term2:.10e}")
    print(f"Gamma_Call = {gamma_call:.10e}")

    # Put Gamma推导
    # d/dS[-(1/S²)*N(d1)] = d/dS[-S^(-2)*N(d1)]
    # = 2*S^(-3)*N(d1) - S^(-2)*n(d1)*(dd1/dS)
    # = 2/S³*N(d1) - S^(-2)*n(d1)*(-1/(S*σ*√T))
    # = 2/S³*N(d1) + n(d1)/(S³*σ*√T)

    gamma_put_term1 = 2 * (inv_S ** 3) * norm.cdf(d1)
    gamma_put_term2 = n_d1 / (S ** 3 * sigma * sqrt_T)  # 同样是正号
    gamma_put = gamma_put_term1 + gamma_put_term2

    print(f"\n=== Put Gamma推导 ===")
    print(f"Term1 = 2/S³ * N(d1) = {gamma_put_term1:.10e}")
    print(f"Term2 = n(d1)/(S³*σ*√T) = {gamma_put_term2:.10e}")
    print(f"Gamma_Put = {gamma_put:.10e}")

    # 关键发现：在ATM情况下，N(-d1) = N(d1) = 0.5（近似），所以
    # Gamma_Call ≈ -2/S³ * 0.5 + n(d1)/(S³*σ*√T) = -1/S³ + n(d1)/(S³*σ*√T)
    # Gamma_Put ≈ 2/S³ * 0.5 + n(d1)/(S³*σ*√T) = 1/S³ + n(d1)/(S³*σ*√T)

    # 这意味着Put Gamma是正的，而Call Gamma可能是负的（取决于哪项占主导）

    print(f"\n=== 关键发现 ===")
    print(f"在ATM附近，N(-d1) ≈ N(d1) ≈ 0.5")
    print(f"Gamma_Call ≈ -1/S³ + n(d1)/(S³*σ*√T)")
    print(f"Gamma_Put ≈ +1/S³ + n(d1)/(S³*σ*√T)")
    print(f"")
    print(f"这意味着：")
    print(f"- Put Gamma 始终为正（两项都为正）")
    print(f"- Call Gamma 可能为负（如果第一项占主导）")

    return gamma_call, gamma_put


def verify_gamma_numerically(S, K, T, r, sigma):
    """数值验证Gamma"""
    from research.pricing.inverse_options import InverseOptionPricer

    print("\n" + "=" * 60)
    print("数值验证Gamma")
    print("=" * 60)

    h = 1.0

    # Call的数值Gamma
    price_S_plus_h = InverseOptionPricer.calculate_price(S + h, K, T, r, sigma, "call")
    price_S = InverseOptionPricer.calculate_price(S, K, T, r, sigma, "call")
    price_S_minus_h = InverseOptionPricer.calculate_price(S - h, K, T, r, sigma, "call")

    delta_plus = (price_S_plus_h - price_S) / h
    delta_minus = (price_S - price_S_minus_h) / h
    gamma_call_numerical = (delta_plus - delta_minus) / h

    print(f"Call数值Gamma = {gamma_call_numerical:.10e}")

    # Put的数值Gamma
    price_S_plus_h = InverseOptionPricer.calculate_price(S + h, K, T, r, sigma, "put")
    price_S = InverseOptionPricer.calculate_price(S, K, T, r, sigma, "put")
    price_S_minus_h = InverseOptionPricer.calculate_price(S - h, K, T, r, sigma, "put")

    delta_plus = (price_S_plus_h - price_S) / h
    delta_minus = (price_S - price_S_minus_h) / h
    gamma_put_numerical = (delta_plus - delta_minus) / h

    print(f"Put数值Gamma = {gamma_put_numerical:.10e}")

    return gamma_call_numerical, gamma_put_numerical


def main():
    from research.pricing.inverse_options import InverseOptionPricer

    print("=" * 60)
    print("币本位期权Gamma公式推导分析")
    print("=" * 60)

    # ATM测试
    S, K, T, r, sigma = 50000, 50000, 0.25, 0.05, 0.60

    gamma_call, gamma_put = gamma_derivation_step_by_step(S, K, T, r, sigma)

    # 获取代码中的Gamma
    greeks_call = InverseOptionPricer.calculate_greeks(S, K, T, r, sigma, "call")
    greeks_put = InverseOptionPricer.calculate_greeks(S, K, T, r, sigma, "put")

    print(f"\n代码中的Gamma_Call = {greeks_call.gamma:.10e}")
    print(f"代码中的Gamma_Put = {greeks_put.gamma:.10e}")

    # 数值验证
    gamma_call_num, gamma_put_num = verify_gamma_numerically(S, K, T, r, sigma)

    # 结论
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)

    print("""
根据数学推导和数值验证：

1. 币本位期权的Call和Put Gamma不相等（与标准BS不同）
2. Call Gamma为负（代码正确）
3. Put Gamma为正（代码中的符号有误）

代码中的问题：
- 第243行：gamma_term2 = n_d1 / (S ** 3 * sigma * sqrt_T) 应该是正号
- 但这导致Put Gamma为正，与数值验证不符

实际上，数值验证显示Put Gamma应该也是负的！
这意味着推导过程中有误...
""")


if __name__ == "__main__":
    main()
