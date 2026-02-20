"""
分析币本位期权Gamma的符号变化

Gamma公式：
- Call Gamma = -2/S³ * N(-d1) + n(d1)/(S³*σ*√T)
- Put Gamma = 2/S³ * N(d1) + n(d1)/(S³*σ*√T)

Gamma的符号取决于两项的相对大小
"""

import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.pricing.inverse_options import InverseOptionPricer


def analyze_gamma_sign(S, K, T, r, sigma):
    """分析Gamma的符号"""
    d1, d2 = InverseOptionPricer._calculate_d1_d2(S, K, T, r, sigma)
    inv_S = 1.0 / S
    n_d1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T)

    # Call Gamma分解
    term1_call = -2 * (inv_S ** 3) * norm.cdf(-d1)
    term2_call = n_d1 / (S ** 3 * sigma * sqrt_T)
    gamma_call = term1_call + term2_call

    # Put Gamma分解
    term1_put = 2 * (inv_S ** 3) * norm.cdf(d1)
    term2_put = n_d1 / (S ** 3 * sigma * sqrt_T)
    gamma_put = term1_put + term2_put

    return {
        'S': S, 'K': K, 'd1': d1,
        'call_term1': term1_call, 'call_term2': term2_call, 'gamma_call': gamma_call,
        'put_term1': term1_put, 'put_term2': term2_put, 'gamma_put': gamma_put
    }


def main():
    T, r, sigma = 0.25, 0.05, 0.60
    K = 50000

    print("币本位期权Gamma符号分析")
    print("=" * 80)
    print(f"参数: K={K}, T={T}, r={r}, sigma={sigma}")
    print()

    test_prices = [30000, 40000, 45000, 49000, 50000, 51000, 55000, 60000, 70000]

    print(f"{'S':>10} {'d1':>10} {'Call T1':>15} {'Call T2':>15} {'Call Gamma':>15} {'Put Gamma':>15}")
    print("-" * 95)

    for S in test_prices:
        result = analyze_gamma_sign(S, K, T, r, sigma)
        print(f"{S:>10} {result['d1']:>10.4f} {result['call_term1']:>15.6e} {result['call_term2']:>15.6e} "
              f"{result['gamma_call']:>15.6e} {result['gamma_put']:>15.6e}")

    print()
    print("关键发现：")
    print("- Call Gamma在ITM时（S > K）可能为负，当N(-d1)较大时")
    print("- Call Gamma在OTM时（S < K）通常为正，因为N(-d1)接近1")
    print("- Put Gamma始终为正，因为两项都为正")


if __name__ == "__main__":
    main()
