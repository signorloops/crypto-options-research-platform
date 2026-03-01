"""
币本位期权Gamma公式的深度数学验证

验证:
1. Gamma = term1 + term2 的正确性
2. Put和Call的Gamma符号问题
3. 量纲分析
"""

import numpy as np
from scipy.stats import norm


def derive_gamma_from_first_principles():
    """
    从第一性原理推导币本位期权的Gamma

    Gamma = d²V/dS² = d(Delta)/dS

    币本位看涨期权:
    V_call = e^(-rT) * (1/K) * N(-d2) - (1/S) * N(-d1)

    Delta_call = dV/dS = (1/S²) * N(-d1)

    币本位看跌期权:
    V_put = (1/S) * N(d1) - e^(-rT) * (1/K) * N(d2)

    Delta_put = dV/dS = -(1/S²) * N(d1)
    """
    print("=" * 70)
    print("币本位期权 Gamma 推导 (从第一性原理)")
    print("=" * 70)

    print("\n1. 币本位看涨期权:")
    print("   V_call = e^(-rT) * (1/K) * N(-d2) - (1/S) * N(-d1)")
    print("\n   Delta_call = dV/dS")
    print("   第一项: d/dS[e^(-rT)*(1/K)*N(-d2)]")
    print("         = e^(-rT)*(1/K)*n(-d2)*(-dd2/dS)")
    print("         = -e^(-rT)*(1/K)*n(d2)*dd2/dS")

    print("\n   第二项: d/dS[-(1/S)*N(-d1)]")
    print("         = (1/S²)*N(-d1) - (1/S)*n(-d1)*(-dd1/dS)")
    print("         = (1/S²)*N(-d1) + (1/S)*n(d1)*dd1/dS")

    print("\n   关键：计算 dd1/dS 和 dd2/dS")
    print("   d1 = [ln(K/S) + (r + 0.5*sigma²)*T] / (sigma*sqrt(T))")
    print("      = [ln(K) - ln(S) + (r + 0.5*sigma²)*T] / (sigma*sqrt(T))")
    print("   dd1/dS = -1/(S * sigma * sqrt(T))")

    print("\n   d2 = d1 - sigma*sqrt(T)")
    print("   dd2/dS = dd1/dS = -1/(S * sigma * sqrt(T))")

    print("\n   代回Delta表达式:")
    print("   Delta_call = -e^(-rT)*(1/K)*n(d2)*(-1/(S*sigma*sqrt(T)))")
    print("              + (1/S²)*N(-d1) + (1/S)*n(d1)*(-1/(S*sigma*sqrt(T)))")
    print("            = e^(-rT)*(1/K)*n(d2)/(S*sigma*sqrt(T))")
    print("              + (1/S²)*N(-d1) - n(d1)/(S²*sigma*sqrt(T))")

    print("\n   注意：根据d2的定义，n(d2) = n(d1) * exp(d1*sigma*sqrt(T) - 0.5*sigma²*T)")
    print("   这个关系使得第一项和第三项部分抵消")

    print("\n   简化后的Delta (代码实现):")
    print("   Delta_call = (1/S²) * N(-d1)")

    print("\n2. 计算 Gamma = d(Delta)/dS:")
    print("   Gamma_call = d/dS[(1/S²) * N(-d1)]")
    print("              = -2/S³ * N(-d1) + (1/S²) * n(-d1) * (-dd1/dS)")
    print("              = -2/S³ * N(-d1) + (1/S²) * n(d1) * (1/(S*sigma*sqrt(T)))")
    print("              = -2/S³ * N(-d1) + n(d1)/(S³*sigma*sqrt(T))")

    print("\n   这与代码第252-258行一致:")
    print("   gamma_term1 = -2 * (inv_S ** 3) * norm.cdf(-d1)")
    print("   gamma_term2 = n_d1 / (S ** 3 * sigma * sqrt_T)")
    print("   gamma = gamma_term1 + gamma_term2")


def derive_put_gamma():
    """推导看跌期权的Gamma"""
    print("\n" + "=" * 70)
    print("币本位看跌期权 Gamma 推导")
    print("=" * 70)

    print("\n1. 币本位看跌期权:")
    print("   V_put = (1/S) * N(d1) - e^(-rT) * (1/K) * N(d2)")

    print("\n2. Delta_put = dV/dS")
    print("   第一项: d/dS[(1/S)*N(d1)]")
    print("         = -(1/S²)*N(d1) + (1/S)*n(d1)*dd1/dS")
    print("         = -(1/S²)*N(d1) - n(d1)/(S²*sigma*sqrt(T))")

    print("\n   第二项: d/dS[-e^(-rT)*(1/K)*N(d2)]")
    print("         = -e^(-rT)*(1/K)*n(d2)*dd2/dS")
    print("         = e^(-rT)*(1/K)*n(d2)/(S*sigma*sqrt(T))")

    print("\n   简化后的Delta:")
    print("   Delta_put = -(1/S²) * N(d1)")

    print("\n3. Gamma_put = d(Delta_put)/dS")
    print("             = d/dS[-(1/S²) * N(d1)]")
    print("             = 2/S³ * N(d1) - (1/S²) * n(d1) * dd1/dS")
    print("             = 2/S³ * N(d1) + n(d1)/(S³*sigma*sqrt(T))")

    print("\n   这与代码一致:")
    print("   gamma_term1 = 2 * (inv_S ** 3) * norm.cdf(d1)  <- 注意符号为正")
    print("   gamma_term2 = n_d1 / (S ** 3 * sigma * sqrt_T)")


def verify_gamma_symmetry():
    """
    验证Call和Put的Gamma关系

    根据Put-Call Parity，Call和Put的Gamma应该相同
    """
    print("\n" + "=" * 70)
    print("Gamma 对称性验证")
    print("=" * 70)

    print("\n理论分析:")
    print("Gamma_call = -2/S³ * N(-d1) + n(d1)/(S³*sigma*sqrt(T))")
    print("Gamma_put = 2/S³ * N(d1) + n(d1)/(S³*sigma*sqrt(T))")

    print("\n利用 N(d1) + N(-d1) = 1:")
    print("Gamma_call = -2/S³ * (1 - N(d1)) + n(d1)/(S³*sigma*sqrt(T))")
    print("           = -2/S³ + 2/S³ * N(d1) + n(d1)/(S³*sigma*sqrt(T))")
    print("           = -2/S³ + Gamma_put")

    print("\n⚠️  发现不一致！Call和Put的Gamma应该相等，但这里相差 -2/S³")
    print("\n这表明代码中的Gamma公式可能有误。")

    print("\n正确的推导应该是:")
    print("从完整的Delta表达式推导，而不是简化后的 (1/S²)*N(-d1)")


def numerical_gamma_verification():
    """数值验证Gamma计算"""
    print("\n" + "=" * 70)
    print("数值验证")
    print("=" * 70)

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from research.pricing.inverse_options import InverseOptionPricer

    # 测试参数
    S = 50000.0
    K = 50000.0
    T = 30 / 365.0
    r = 0.05
    sigma = 0.6

    print(f"\n参数: S={S}, K={K}, T={T:.4f}, r={r}, sigma={sigma}")

    # 计算Gamma
    greeks_call = InverseOptionPricer.calculate_greeks(S, K, T, r, sigma, "call")
    greeks_put = InverseOptionPricer.calculate_greeks(S, K, T, r, sigma, "put")

    print(f"\n代码计算的Gamma:")
    print(f"Call Gamma: {greeks_call.gamma:.10e}")
    print(f"Put Gamma:  {greeks_put.gamma:.10e}")
    print(f"差异: {abs(greeks_call.gamma - greeks_put.gamma):.10e}")

    # 有限差分验证
    print("\n--- 有限差分验证 ---")
    dS = S * 0.0001  # 0.01%的扰动

    # Call Gamma
    delta_up_call = InverseOptionPricer.calculate_greeks(S + dS, K, T, r, sigma, "call").delta
    delta_down_call = InverseOptionPricer.calculate_greeks(S - dS, K, T, r, sigma, "call").delta
    gamma_fd_call = (delta_up_call - delta_down_call) / (2 * dS)

    # Put Gamma
    delta_up_put = InverseOptionPricer.calculate_greeks(S + dS, K, T, r, sigma, "put").delta
    delta_down_put = InverseOptionPricer.calculate_greeks(S - dS, K, T, r, sigma, "put").delta
    gamma_fd_put = (delta_up_put - delta_down_put) / (2 * dS)

    print(f"\nCall Gamma (代码):  {greeks_call.gamma:.10e}")
    print(f"Call Gamma (差分):  {gamma_fd_call:.10e}")
    print(f"相对误差: {abs(greeks_call.gamma - gamma_fd_call) / abs(gamma_fd_call) * 100:.2f}%")

    print(f"\nPut Gamma (代码):   {greeks_put.gamma:.10e}")
    print(f"Put Gamma (差分):   {gamma_fd_put:.10e}")
    print(f"相对误差: {abs(greeks_put.gamma - gamma_fd_put) / abs(gamma_fd_put) * 100:.2f}%")

    # 验证Gamma是否为正
    print(f"\n--- Gamma符号验证 ---")
    print(f"Call Gamma > 0: {greeks_call.gamma > 0}")
    print(f"Put Gamma > 0:  {greeks_put.gamma > 0}")
    print("理论上，期权的Gamma应该始终为正（无论Call还是Put）")


def dimension_analysis():
    """Gamma的量纲分析"""
    print("\n" + "=" * 70)
    print("Gamma量纲分析")
    print("=" * 70)

    print("\n币本位期权的量纲:")
    print("- S: [USD/BTC] (标的价)")
    print("- V: [BTC] (期权价格)")
    print("- Delta = dV/dS: [BTC] / [USD/BTC] = [BTC²/USD]")
    print("- Gamma = d²V/dS²: [BTC²/USD] / [USD/BTC] = [BTC³/USD²]")

    print("\n代码中的Gamma:")
    print("gamma_term1 = -2 * (1/S)³ * N(-d1)")
    print("            = [1/USD³/BTC³] = [BTC³/USD³]")
    print("\n⚠️  量纲不匹配！代码计算的是 [BTC³/USD³]，但应该是 [BTC³/USD²]")

    print("\n正确的Gamma应该是:")
    print("Gamma = (1/S²) * [标准BS Gamma]")
    print("      = (1/S²) * [1/USD]")
    print("      = [BTC²/USD²] * [1/USD]")
    print("      = [BTC²/USD³]")

    print("\n等等，让我重新计算...")
    print("标准BS Gamma = n(d1) / (S * sigma * sqrt(T))")
    print("             = 1 / [USD/BTC]")
    print("             = [BTC/USD]")

    print("\n币本位Gamma = d/dS [Delta]")
    print("Delta = (1/S²) * N(-d1) = [BTC²/USD²]")
    print("Gamma = d/dS [BTC²/USD²]")
    print("      = [BTC²/USD²] / [USD/BTC]")
    print("      = [BTC³/USD³]")

    print("\n实际上代码的量纲是正确的！")


if __name__ == "__main__":
    derive_gamma_from_first_principles()
    derive_put_gamma()
    verify_gamma_symmetry()
    numerical_gamma_verification()
    dimension_analysis()
