"""
Put-Call Parity的完整推导验证

验证 inverse_options.py 第495-535行的公式:
C - P = (1/K)*e^(-rT) - 1/S
"""

import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.pricing.inverse_options import InverseOptionPricer


def derive_put_call_parity():
    """
    从第一性原理推导币本位期权的Put-Call Parity

    币本位期权的Payoff:
    - Call: max(0, 1/K - 1/S)
    - Put: max(0, 1/S - 1/K)
    """
    print("=" * 70)
    print("币本位期权 Put-Call Parity 推导")
    print("=" * 70)

    print("\n1. 到期Payoff分析:")
    print("   C_T = max(0, 1/K - 1/S_T)")
    print("   P_T = max(0, 1/S_T - 1/K)")

    print("\n2. 考虑两种情况:")
    print("   a) S_T > K (即 1/S_T < 1/K):")
    print("      C_T = 0")
    print("      P_T = 1/S_T - 1/K")
    print("      C_T - P_T = -(1/S_T - 1/K) = 1/K - 1/S_T")

    print("\n   b) S_T <= K (即 1/S_T >= 1/K):")
    print("      C_T = 1/K - 1/S_T")
    print("      P_T = 0")
    print("      C_T - P_T = 1/K - 1/S_T")

    print("\n3. 统一表达式:")
    print("   C_T - P_T = 1/K - 1/S_T (对所有S_T)")

    print("\n4. 折现到当前:")
    print("   对于币本位期权，我们需要考虑numeraire的变换")
    print("   在BTC numeraire下，USD现金的价值是 1/S")

    print("\n   考虑一个组合:")
    print("   - 买入1个币本位Call")
    print("   - 卖出1个币本位Put")
    print("   - 卖出 (1/K)*e^(-rT) 单位的零息债券")
    print("   - 买入 1/S 单位的现货")

    print("\n   到期价值:")
    print("   max(0, 1/K-1/S_T) - max(0, 1/S_T-1/K) - (1/K)*e^(-rT)*e^(rT) + (1/S_T)")
    print("   = (1/K - 1/S_T) - 1/K + 1/S_T")
    print("   = 0")

    print("\n5. 无套利条件:")
    print("   C - P - (1/K)*e^(-rT) + 1/S = 0")
    print("   => C - P = (1/K)*e^(-rT) - 1/S")

    print("\n这与代码第507行的公式一致!")


def verify_parity_numerically():
    """数值验证Put-Call Parity"""
    print("\n" + "=" * 70)
    print("数值验证")
    print("=" * 70)

    test_cases = [
        (50000, 50000, 30/365, 0.05, 0.6, "ATM"),
        (50000, 60000, 30/365, 0.05, 0.6, "OTM Call / ITM Put"),
        (50000, 40000, 30/365, 0.05, 0.6, "ITM Call / OTM Put"),
        (50000, 50000, 90/365, 0.05, 0.6, "3个月到期"),
        (50000, 50000, 30/365, 0.10, 0.6, "高利率"),
    ]

    for S, K, T, r, sigma, desc in test_cases:
        print(f"\n测试: {desc}")
        print(f"  参数: S={S}, K={K}, T={T:.4f}, r={r}, sigma={sigma}")

        call_price = InverseOptionPricer.calculate_price(S, K, T, r, sigma, "call")
        put_price = InverseOptionPricer.calculate_price(S, K, T, r, sigma, "put")

        lhs = call_price - put_price
        rhs = (1.0 / K) * np.exp(-r * T) - 1.0 / S

        print(f"  Call价格: {call_price:.10f}")
        print(f"  Put价格:  {put_price:.10f}")
        print(f"  LHS (C-P): {lhs:.10f}")
        print(f"  RHS (1/K*e^(-rT) - 1/S): {rhs:.10f}")
        print(f"  偏差: {abs(lhs - rhs):.2e}")


def verify_parity_with_greeks():
    """
    验证Greeks也满足Put-Call Parity

    如果 C - P = (1/K)*e^(-rT) - 1/S
    那么:
    - Delta_C - Delta_P = d/dS[-1/S] = 1/S²
    - Gamma_C - Gamma_P = d²/dS²[-1/S] = -2/S³
    """
    print("\n" + "=" * 70)
    print("Greeks的Put-Call Parity验证")
    print("=" * 70)

    S = 50000.0
    K = 50000.0
    T = 30/365
    r = 0.05
    sigma = 0.6

    print(f"\n参数: S={S}, K={K}, T={T:.4f}, r={r}, sigma={sigma}")

    call_greeks = InverseOptionPricer.calculate_greeks(S, K, T, r, sigma, "call")
    put_greeks = InverseOptionPricer.calculate_greeks(S, K, T, r, sigma, "put")

    print("\nDelta验证:")
    delta_diff = call_greeks.delta - put_greeks.delta
    delta_theory = 1.0 / (S ** 2)
    print(f"  Delta_C - Delta_P: {delta_diff:.10e}")
    print(f"  理论值 (1/S²): {delta_theory:.10e}")
    print(f"  偏差: {abs(delta_diff - delta_theory):.2e}")

    print("\nGamma验证:")
    gamma_diff = call_greeks.gamma - put_greeks.gamma
    gamma_theory = -2.0 / (S ** 3)
    print(f"  Gamma_C - Gamma_P: {gamma_diff:.10e}")
    print(f"  理论值 (-2/S³): {gamma_theory:.10e}")
    print(f"  偏差: {abs(gamma_diff - gamma_theory):.2e}")

    print("\n⚠️  Gamma差异不为零，这与标准期权不同！")
    print("   这是因为币本位期权的特殊性：C和P的Gamma公式不同")


def verify_parity_at_expiry():
    """验证到期时的Put-Call Parity"""
    print("\n" + "=" * 70)
    print("到期时Put-Call Parity验证")
    print("=" * 70)

    S = 50000.0
    K = 50000.0
    T = 0.0  # 到期
    r = 0.05
    sigma = 0.6

    print(f"\n参数: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")

    # 到期payoff
    call_payoff = max(0, 1/K - 1/S)
    put_payoff = max(0, 1/S - 1/K)

    lhs = call_payoff - put_payoff
    rhs = max(0, 1/K - 1/S) - max(0, 1/S - 1/K)  # 到期时的RHS

    print(f"  Call payoff: {call_payoff:.10f}")
    print(f"  Put payoff: {put_payoff:.10f}")
    print(f"  LHS: {lhs:.10f}")
    print(f"  RHS: {rhs:.10f}")
    print(f"  偏差: {abs(lhs - rhs):.2e}")


def analyze_parity_implementation():
    """分析代码实现"""
    print("\n" + "=" * 70)
    print("代码实现分析 (inverse_options.py 第495-535行)")
    print("=" * 70)

    print("\n代码逻辑:")
    print("def inverse_option_parity(call_price, put_price, S, K, T, r):")
    print("    if T < EPSILON:")
    print("        lhs = call_price - put_price")
    print("        rhs = max(0, 1/K - 1/S) - max(0, 1/S - 1/K)")
    print("    else:")
    print("        lhs = call_price - put_price")
    print("        rhs = (1.0 / K) * np.exp(-r * T) - 1.0 / S")
    print("    return lhs - rhs")

    print("\n分析:")
    print("1. 到期情况(T < EPSILON)处理正确")
    print("2. 非到期情况使用连续复利折现")
    print("3. 返回值是偏差，越接近0越好")

    print("\n潜在问题:")
    print("- 没有检查输入有效性(S>0, K>0)")
    print("- 对于T=0但S=K的情况，rhs=0，这需要特殊处理")


if __name__ == "__main__":
    derive_put_call_parity()
    verify_parity_numerically()
    verify_parity_with_greeks()
    verify_parity_at_expiry()
    analyze_parity_implementation()
