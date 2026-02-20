"""
Greeks跨币种转换公式的量纲分析

验证 greeks.py 第270-296行的转换公式:
- 币本位(inverse)期权的转换
- USD本位期权的转换
- spot^2 * fx_rate 和 spot^3 * fx_rate 的正确性
"""

import numpy as np


def analyze_inverse_greeks_dimensions():
    """
    分析币本位期权的Greeks量纲

    币本位期权:
    - 价格 V: [BTC] (以BTC计价)
    - 标的价 S: [USD/BTC]
    - 行权价 K: [USD/BTC]
    """
    print("=" * 70)
    print("币本位期权 Greeks 量纲分析")
    print("=" * 70)

    print("\n1. 基本量纲:")
    print("   S: [USD/BTC] (标的价)")
    print("   V: [BTC] (期权价格)")
    print("   K: [USD/BTC] (行权价)")

    print("\n2. Delta 量纲:")
    print("   Delta = dV/dS")
    print("         = [BTC] / [USD/BTC]")
    print("         = [BTC²/USD]")

    print("\n3. Gamma 量纲:")
    print("   Gamma = d²V/dS² = d(Delta)/dS")
    print("         = [BTC²/USD] / [USD/BTC]")
    print("         = [BTC³/USD²]")

    print("\n4. 转换为USD计价的Delta:")
    print("   我们希望得到 USD per (USD/BTC) 的敏感度")
    print("   或者说，当S变化1 USD/BTC时，头寸价值变化多少USD")

    print("\n   币本位Delta: [BTC²/USD]")
    print("   乘以 S²: [BTC²/USD] * [USD²/BTC²] = [USD]")
    print("   这表示当S变化1单位时，价值变化多少USD")

    print("\n   等等，这不对。让我重新分析...")


def analyze_conversion_logic():
    """分析转换逻辑"""
    print("\n" + "=" * 70)
    print("Greeks转换逻辑深度分析")
    print("=" * 70)

    print("\n场景：持有1个币本位看涨期权")
    print("S = 50000 USD/BTC")
    print("Delta = 8e-9 BTC²/USD (来自代码的实际值)")

    print("\n问题：如果S上涨1 USD (从50000到50001)，头寸价值变化多少？")

    print("\n币本位价值变化:")
    print("  dV = Delta * dS")
    print("     = 8e-9 BTC²/USD * 1 USD/BTC")
    print("     = 8e-9 BTC")

    print("\n转换为USD:")
    print("  dV_USD = dV * S (因为 1 BTC = S USD)")
    print("         = 8e-9 BTC * 50000 USD/BTC")
    print("         = 4e-4 USD")

    print("\n使用代码的转换方法:")
    print("  Delta_USD = Delta * S²")
    print("            = 8e-9 * (50000)²")
    print("            = 8e-9 * 2.5e9")
    print("            = 20 USD/(USD/BTC)")

    print("\n这意味着当S变化1 USD/BTC时，价值变化20 USD")
    print("但我们计算出的是 4e-4 USD，相差了 S² 倍！")

    print("\n⚠️  发现转换公式的问题！")


def verify_code_conversion():
    """验证代码中的转换公式"""
    print("\n" + "=" * 70)
    print("代码转换公式验证 (greeks.py 第270-296行)")
    print("=" * 70)

    print("\n代码逻辑:")
    print("if contract.inverse:")
    print("    delta_usd = position_greeks.delta * (spot_safe ** 2) * fx_rate")
    print("    gamma_usd = position_greeks.gamma * (spot_safe ** 3) * fx_rate")

    print("\n问题分析:")
    print("1. Delta的量纲是 [BTC²/USD]")
    print("2. spot² 的量纲是 [USD²/BTC²]")
    print("3. Delta * spot² 的量纲是 [BTC²/USD] * [USD²/BTC²] = [USD]")
    print("4. 这不是Delta的量纲，而是头寸价值的量纲！")

    print("\n正确的转换应该是:")
    print("  Delta_USD = Delta * S")
    print("            = [BTC²/USD] * [USD/BTC]")
    print("            = [BTC]")
    print("  然后转换为USD: Delta_USD_USD = Delta_USD * S = Delta * S²")

    print("\n或者从经济意义上理解:")
    print("  Delta_USD = dV/dS 的单位是 [BTC²/USD]")
    print("  我们想要求 d(V*S)/dS = V + S*dV/dS")
    print("  这是价值对S的敏感度，单位是 [BTC]")


def derive_correct_conversion():
    """推导正确的转换公式"""
    print("\n" + "=" * 70)
    print("正确转换公式的推导")
    print("=" * 70)

    print("\n假设:")
    print("  V_BTC: 币本位期权价格 [BTC]")
    print("  V_USD = V_BTC * S: USD价值 [USD]")
    print("  S: 标的价 [USD/BTC]")

    print("\n币本位Delta:")
    print("  Delta_BTC = dV_BTC/dS [BTC²/USD]")

    print("\nUSD Delta (对USD价值的敏感度):")
    print("  Delta_USD = d(V_BTC * S)/dS")
    print("            = V_BTC + S * dV_BTC/dS")
    print("            = V_BTC + S * Delta_BTC")

    print("\n但代码中的转换是:")
    print("  Delta_converted = Delta_BTC * S²")

    print("\n这两者相等吗？")
    print("  V_BTC + S * Delta_BTC =? Delta_BTC * S²")
    print("  V_BTC =? Delta_BTC * S² - S * Delta_BTC")
    print("  V_BTC =? Delta_BTC * S * (S - 1)")

    print("\n显然不相等！代码的转换公式是错误的。")

    print("\n正确的转换应该是:")
    print("  如果目标是 USD Notional Delta:")
    print("    Notional_Delta = Delta_BTC * S")
    print("    (单位: [BTC²/USD] * [USD/BTC] = [BTC])")
    print("    这表示需要对冲的BTC数量")

    print("\n  如果目标是 USD价值变化 per 1 USD spot change:")
    print("    USD_Sensitivity = Delta_BTC * S²")
    print("    (单位: [BTC²/USD] * [USD²/BTC²] = [USD/(USD/BTC)])")


def numerical_verification():
    """数值验证"""
    print("\n" + "=" * 70)
    print("数值验证")
    print("=" * 70)

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from research.pricing.inverse_options import InverseOptionPricer

    S = 50000.0
    K = 50000.0
    T = 30 / 365.0
    r = 0.05
    sigma = 0.6

    greeks = InverseOptionPricer.calculate_greeks(S, K, T, r, sigma, "call")

    print(f"\n参数: S={S}, K={K}, T={T:.4f}, r={r}, sigma={sigma}")
    print(f"\n币本位 Greeks:")
    print(f"  Delta: {greeks.delta:.10e} BTC²/USD")
    print(f"  Gamma: {greeks.gamma:.10e} BTC³/USD²")

    # 代码的转换
    delta_code = greeks.delta * S**2
    gamma_code = greeks.gamma * S**3

    print(f"\n代码转换后的值:")
    print(f"  Delta * S²: {delta_code:.6f}")
    print(f"  Gamma * S³: {gamma_code:.6f}")

    # 有限差分验证
    dS = 1.0
    price_up = InverseOptionPricer.calculate_price(S + dS, K, T, r, sigma, "call")
    price_down = InverseOptionPricer.calculate_price(S - dS, K, T, r, sigma, "call")
    price = InverseOptionPricer.calculate_price(S, K, T, r, sigma, "call")

    # USD价值
    value_usd = price * S
    value_usd_up = price_up * (S + dS)
    value_usd_down = price_down * (S - dS)

    # USD Delta (每1 USD spot change)
    delta_usd_fd = (value_usd_up - value_usd_down) / (2 * dS)

    print(f"\n有限差分验证:")
    print(f"  当前USD价值: {value_usd:.6f} USD")
    print(f"  USD Delta (差分): {delta_usd_fd:.6f} USD/(USD/BTC)")
    print(f"  代码转换值: {delta_code:.6f}")
    print(f"  差异: {abs(delta_usd_fd - delta_code):.6f}")


if __name__ == "__main__":
    analyze_inverse_greeks_dimensions()
    analyze_conversion_logic()
    verify_code_conversion()
    derive_correct_conversion()
    numerical_verification()
