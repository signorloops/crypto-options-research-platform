"""
隐含波动率计算的收敛性分析

验证:
1. 牛顿法的收敛条件
2. 步长限制 (第407行 step = max(-0.5, min(0.5, step))) 的数学依据
3. 二分法fallback的触发条件
"""

import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.pricing.inverse_options import InverseOptionPricer


def analyze_newton_convergence():
    """
    分析牛顿法的收敛条件

    牛顿迭代: sigma_{n+1} = sigma_n - f(sigma)/f'(sigma)
    其中 f(sigma) = Price(sigma) - MarketPrice
    """
    print("=" * 70)
    print("牛顿法收敛性分析")
    print("=" * 70)

    print("\n1. 牛顿法的收敛条件:")
    print("   - 初始猜测足够接近真实解")
    print("   - 函数在解附近二阶可导")
    print("   - f'(sigma) ≠ 0 (Vega不能为零)")

    print("\n2. 币本位期权Vega的特性:")
    print("   Vega = (1/S) * n(d1) * sqrt(T) * 0.01")
    print("   - 当 S → ∞, Vega → 0")
    print("   - 当 T → 0, Vega → 0")
    print("   - 当 sigma → 0 或 sigma → ∞, Vega → 0")

    print("\n3. 收敛问题:")
    print("   - 在极端参数下，Vega接近0，牛顿法会失效")
    print("   - 需要步长限制防止震荡")


def analyze_step_size_limit():
    """
    分析步长限制的数学依据

    代码第407行: step = max(-0.5, min(0.5, step))
    """
    print("\n" + "=" * 70)
    print("步长限制分析 (第407行)")
    print("=" * 70)

    print("\n1. 代码中的步长限制:")
    print("   step = diff / vega")
    print("   step = max(-0.5, min(0.5, step))")

    print("\n2. 数学依据:")
    print("   - 限制步长在 [-0.5, 0.5] 范围内")
    print("   - 这意味着每次迭代sigma变化不超过50%")
    print("   - 这是典型的阻尼牛顿法 (Damped Newton's Method)")

    print("\n3. 为什么需要限制:")
    print("   a) 防止震荡: 当Vega很小时，step会很大")
    print("   b) 保证单调性: 限制步长可以防止越过真实解")
    print("   c) 全局收敛: 纯牛顿法只有局部收敛性")

    print("\n4. 限制值0.5的合理性:")
    print("   - 对于波动率(通常在0.1到2.0之间)，0.5是合理的")
    print("   - 如果真实sigma=0.2，步长0.5会将其变为0.7或-0.3")
    print("   - 这可能导致发散或进入无效区域(负波动率)")

    print("\n5. 潜在问题:")
    print("   - 0.5的限制可能过于激进，导致收敛变慢")
    print("   - 更好的方法是使用自适应步长")


def analyze_bisection_fallback():
    """分析二分法fallback"""
    print("\n" + "=" * 70)
    print("二分法Fallback分析")
    print("=" * 70)

    print("\n1. 触发条件 (第400-420行):")
    print("   - Vega < 1e-14 (太小)")
    print("   - 数值错误 (ValueError, FloatingPointError, RuntimeError)")
    print("   - 牛顿法达到最大迭代次数仍未收敛")

    print("\n2. 二分法实现 (第425-462行):")
    print("   sigma_low, sigma_high = 0.001, 5.0")
    print("   检查价格是否在 [price_low, price_high] 范围内")
    print("   标准二分法迭代")

    print("\n3. 问题分析:")
    print("   a) 初始范围 [0.001, 5.0] 是否合理?")
    print("      - 对于加密货币，5.0 (500%) 是合理的上限")
    print("      - 0.001 (0.1%) 是合理的下限")

    print("\n   b) 价格范围检查 (第443-445行):")
    print("      if price < price_low or price > price_high:")
    print("          return sigma_low if price < price_low else sigma_high")
    print("      问题: 直接返回边界值，没有警告或错误!")

    print("\n   c) 收敛条件:")
    print("      - abs(price_mid - price) < tol (绝对误差)")
    print("      - sigma_high - sigma_low < tol (区间足够小)")
    print("      没有相对误差检查，对于不同价格尺度可能不公平")


def test_convergence_cases():
    """测试各种收敛情况"""
    print("\n" + "=" * 70)
    print("收敛性测试")
    print("=" * 70)

    test_cases = [
        # (S, K, T, r, sigma, option_type, description)
        (50000, 50000, 30/365, 0.05, 0.6, "call", "ATM正常情况"),
        (50000, 60000, 30/365, 0.05, 0.6, "call", "OTM Call"),
        (50000, 40000, 30/365, 0.05, 0.6, "call", "ITM Call"),
        (50000, 50000, 1/365, 0.05, 0.6, "call", "临近到期"),
        (50000, 50000, 365/365, 0.05, 0.6, "call", "长期到期"),
        (50000, 50000, 30/365, 0.05, 0.1, "call", "低波动率"),
        (50000, 50000, 30/365, 0.05, 2.0, "call", "高波动率"),
    ]

    for S, K, T, r, sigma, option_type, desc in test_cases:
        print(f"\n测试: {desc}")
        print(f"  参数: S={S}, K={K}, T={T:.4f}, r={r}, sigma={sigma}")

        try:
            # 计算理论价格
            price = InverseOptionPricer.calculate_price(S, K, T, r, sigma, option_type)
            print(f"  理论价格: {price:.8f}")

            # 反解IV
            iv = InverseOptionPricer.calculate_implied_volatility(
                price, S, K, T, r, option_type
            )
            print(f"  反解IV: {iv:.6f}")
            print(f"  误差: {abs(iv - sigma):.8f}")

            # 检查Vega
            greeks = InverseOptionPricer.calculate_greeks(S, K, T, r, sigma, option_type)
            print(f"  Vega: {greeks.vega:.10e}")

        except Exception as e:
            print(f"  错误: {e}")


def analyze_edge_cases():
    """分析边界情况"""
    print("\n" + "=" * 70)
    print("边界情况分析")
    print("=" * 70)

    edge_cases = [
        # (S, K, T, r, price, option_type, description)
        (50000, 50000, 0.001, 0.05, 0.0001, "call", "极短到期时间"),
        (50000, 50000, 30/365, 0.05, 1e-10, "call", "极小价格"),
        (50000, 50000, 30/365, 0.05, 0.00002, "call", "接近最大价格"),
    ]

    for S, K, T, r, price, option_type, desc in edge_cases:
        print(f"\n测试: {desc}")
        print(f"  价格: {price}")

        try:
            iv = InverseOptionPricer.calculate_implied_volatility(
                price, S, K, T, r, option_type
            )
            print(f"  反解IV: {iv:.6f}")
        except Exception as e:
            print(f"  错误: {e}")


def suggest_improvements():
    """提出改进建议"""
    print("\n" + "=" * 70)
    print("改进建议")
    print("=" * 70)

    print("\n1. 步长限制改进:")
    print("   当前: step = max(-0.5, min(0.5, step))")
    print("   建议: 使用自适应步长，如:")
    print("         step = diff / vega")
    print("         damping = min(1.0, 0.5 / (abs(step) + 0.001))")
    print("         step = step * damping")

    print("\n2. 收敛条件改进:")
    print("   当前: abs(diff) < rel_tol * price_scale")
    print("   建议: 同时使用绝对和相对收敛条件:")
    print("         converged = (abs(diff) < abs_tol) or (abs(diff) < rel_tol * price)")

    print("\n3. 二分法fallback改进:")
    print("   当前: 价格超出范围时直接返回边界值")
    print("   建议: 返回None或抛出异常，并记录警告")

    print("\n4. 初始猜测改进:")
    print("   当前: sigma = 0.5 + 0.1 * abs(log(S/K))")
    print("   建议: 使用更精确的近似，如Brenner-Subrahmanyam公式")


if __name__ == "__main__":
    analyze_newton_convergence()
    analyze_step_size_limit()
    analyze_bisection_fallback()
    test_convergence_cases()
    analyze_edge_cases()
    suggest_improvements()
