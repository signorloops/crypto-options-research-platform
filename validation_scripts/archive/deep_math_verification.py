"""
深度数学逻辑审查 - 第二次审查
重点验证Gamma和Theta公式的数学正确性
"""

import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.pricing.inverse_options import InverseOptionPricer


def numerical_derivative(f, x, h=1e-6):
    """中心差分数值导数"""
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_second_derivative(f, x, h=1e-5):
    """数值二阶导数"""
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)


def verify_gamma_formula():
    """
    Gamma公式深度验证

    币本位看涨期权价格:
    C = e^(-rT) * (1/K) * N(-d2) - (1/S) * N(-d1)

    对S求一阶导:
    dC/dS = (1/S^2) * N(-d1)  [这是Delta]

    对S求二阶导 (Gamma):
    d²C/dS² = d/dS[(1/S^2) * N(-d1)]
            = -2/S^3 * N(-d1) + (1/S^2) * d/dS[N(-d1)]
            = -2/S^3 * N(-d1) + (1/S^2) * n(-d1) * (-1) * dd1/dS

    其中 dd1/dS = -1/(S * sigma * sqrt(T))

    所以:
    d²C/dS² = -2/S^3 * N(-d1) + (1/S^2) * n(d1) * 1/(S * sigma * sqrt(T))
            = -2/S^3 * N(-d1) + n(d1)/(S^3 * sigma * sqrt(T))

    注意: n(-d1) = n(d1) 因为正态分布PDF是偶函数

    关键问题: 文档说"Gamma为负"，但第二项为正！
    实际上Gamma的符号取决于两项的相对大小。
    """
    print("=" * 70)
    print("Gamma公式深度验证")
    print("=" * 70)

    S, K, T, r, sigma = 50000, 50000, 0.25, 0.05, 0.60

    d1, d2 = InverseOptionPricer._calculate_d1_d2(S, K, T, r, sigma)
    inv_S = 1.0 / S
    sqrt_T = np.sqrt(T)
    n_d1 = norm.pdf(d1)

    print(f"\n测试参数: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"d1={d1:.6f}, d2={d2:.6f}")
    print(f"N(-d1)={norm.cdf(-d1):.6f}, n(d1)={n_d1:.6e}")

    # 手动计算Gamma的两项
    gamma_term1_call = -2 * (inv_S ** 3) * norm.cdf(-d1)
    gamma_term2 = n_d1 / (S ** 3 * sigma * sqrt_T)
    manual_gamma_call = gamma_term1_call + gamma_term2

    gamma_term1_put = 2 * (inv_S ** 3) * norm.cdf(d1)
    manual_gamma_put = gamma_term1_put + gamma_term2

    print(f"\n--- Call Gamma分解 ---")
    print(f"第一项 (-2/S^3 * N(-d1)): {gamma_term1_call:.10e}")
    print(f"第二项 (n(d1)/(S^3*sigma*sqrt(T))): {gamma_term2:.10e}")
    print(f"手动计算Gamma: {manual_gamma_call:.10e}")

    print(f"\n--- Put Gamma分解 ---")
    print(f"第一项 (2/S^3 * N(d1)): {gamma_term1_put:.10e}")
    print(f"第二项 (n(d1)/(S^3*sigma*sqrt(T))): {gamma_term2:.10e}")
    print(f"手动计算Gamma: {manual_gamma_put:.10e}")

    # 代码计算的Gamma
    greeks_call = InverseOptionPricer.calculate_greeks(S, K, T, r, sigma, "call")
    greeks_put = InverseOptionPricer.calculate_greeks(S, K, T, r, sigma, "put")

    print(f"\n--- 代码计算结果 ---")
    print(f"Call Gamma (代码): {greeks_call.gamma:.10e}")
    print(f"Put Gamma (代码): {greeks_put.gamma:.10e}")

    # 数值验证
    def price_func(s):
        return InverseOptionPricer.calculate_price(s, K, T, r, sigma, "call")

    numerical_gamma = numerical_second_derivative(price_func, S, h=10.0)
    print(f"\n数值二阶导数: {numerical_gamma:.10e}")

    # 验证
    print(f"\n--- 验证结果 ---")
    diff_call = abs(greeks_call.gamma - manual_gamma_call)
    diff_put = abs(greeks_put.gamma - manual_gamma_put)
    diff_numerical = abs(greeks_call.gamma - numerical_gamma)

    print(f"Call Gamma手动vs代码: {'PASS' if diff_call < 1e-10 else 'FAIL'} (diff={diff_call:.2e})")
    print(f"Put Gamma手动vs代码: {'PASS' if diff_put < 1e-10 else 'FAIL'} (diff={diff_put:.2e})")
    print(f"Gamma代码vs数值: {'PASS' if diff_numerical < 0.05 * abs(greeks_call.gamma) else 'FAIL'} (diff={diff_numerical:.2e})")

    # 关键发现：Gamma的符号
    print(f"\n--- Gamma符号分析 ---")
    print(f"对于ATM期权，Gamma第一项: {gamma_term1_call:.10e}")
    print(f"对于ATM期权，Gamma第二项: {gamma_term2:.10e}")
    print(f"总和: {manual_gamma_call:.10e}")

    if manual_gamma_call < 0:
        print("结论: ATM Call Gamma为负")
    else:
        print("结论: ATM Call Gamma为正 (这与文档'Gamma为负'的说法矛盾)")

    return greeks_call.gamma, greeks_put.gamma


def verify_theta_formula():
    """
    Theta公式深度验证

    币本位看涨期权:
    C = e^(-rT) * (1/K) * N(-d2) - (1/S) * N(-d1)

    dC/dT = -r*e^(-rT)*(1/K)*N(-d2) + e^(-rT)*(1/K)*n(-d2)*(-1)*dd2/dT
            + (1/S)*n(-d1)*(-1)*dd1/dT

          = -r*e^(-rT)*(1/K)*N(-d2) - e^(-rT)*(1/K)*n(d2)*dd2/dT
            - (1/S)*n(d1)*dd1/dT

    注意: n(-x) = n(x) (PDF是偶函数)

    但代码中使用的是:
    theta = (-r * discount * inv_K * norm.cdf(-d2)
             - discount * inv_K * norm.pdf(d2) * d_d2_dT
             + inv_S * n_d1 * d_d1_dT) / 365.0

    问题:
    1. 第三项符号是+，但推导是-
    2. 代码使用n(d2)而不是n(-d2)，但由于PDF是偶函数，这没问题

    对于Put:
    P = (1/S) * N(d1) - e^(-rT) * (1/K) * N(d2)

    dP/dT = -(1/S)*n(d1)*dd1/dT + r*e^(-rT)*(1/K)*N(d2) + e^(-rT)*(1/K)*n(d2)*dd2/dT

    代码中使用:
    theta = (r * discount * inv_K * norm.cdf(d2)
             + discount * inv_K * norm.pdf(-d2) * d_d2_dT
             - inv_S * n_d1 * (-d_d1_dT)) / 365.0

    问题:
    1. 代码使用 norm.pdf(-d2) = n(-d2) = n(d2)，正确
    2. 第三项是 -inv_S * n_d1 * (-d_d1_dT) = +inv_S * n_d1 * d_d1_dT
       但推导应该是 -inv_S * n_d1 * d_d1_dT

    所以Put的Theta第三项符号可能有错误！
    """
    print("\n" + "=" * 70)
    print("Theta公式深度验证")
    print("=" * 70)

    S, K, T, r, sigma = 50000, 50000, 0.25, 0.05, 0.60

    d1, d2 = InverseOptionPricer._calculate_d1_d2(S, K, T, r, sigma)
    inv_S = 1.0 / S
    inv_K = 1.0 / K
    discount = np.exp(-r * T)
    sqrt_T = np.sqrt(T)
    n_d1 = norm.pdf(d1)
    n_d2 = norm.pdf(d2)

    # dd1/dT 和 dd2/dT 的计算
    # d1 = [ln(K/S) + (r + 0.5*sigma^2)*T] / (sigma*sqrt(T))
    # 令 A = ln(K/S), B = (r + 0.5*sigma^2)
    # d1 = (A + B*T) / (sigma * sqrt(T))
    # dd1/dT = [B * sigma * sqrt(T) - (A + B*T) * sigma * (1/2) * T^(-1/2)] / (sigma^2 * T)
    #        = [B * sqrt(T) - (A + B*T) / (2*sqrt(T))] / (sigma * T)
    #        = [2*B*T - (A + B*T)] / (2 * sigma * T^1.5)
    #        = (B*T - A) / (2 * sigma * T^1.5)

    A = np.log(K / S)
    B = r + 0.5 * sigma ** 2

    d_d1_dT_formula = (B * T - A) / (2 * sigma * T ** 1.5)
    d_d2_dT_formula = d_d1_dT_formula - sigma / (2 * sqrt_T)

    # 代码中的计算
    d_d1_dT_code = ((r + 0.5 * sigma ** 2) * T - np.log(K / S)) / (2 * sigma * T ** 1.5)
    d_d2_dT_code = d_d1_dT_code - sigma / (2 * sqrt_T)

    print(f"\n测试参数: S={S}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"\ndd1/dT验证:")
    print(f"  公式推导: {d_d1_dT_formula:.10e}")
    print(f"  代码实现: {d_d1_dT_code:.10e}")
    print(f"  匹配: {'PASS' if abs(d_d1_dT_formula - d_d1_dT_code) < 1e-10 else 'FAIL'}")

    print(f"\ndd2/dT验证:")
    print(f"  公式推导: {d_d2_dT_formula:.10e}")
    print(f"  代码实现: {d_d2_dT_code:.10e}")
    print(f"  匹配: {'PASS' if abs(d_d2_dT_formula - d_d2_dT_code) < 1e-10 else 'FAIL'}")

    # 手动推导Call Theta
    # dC/dT = -r*discount*inv_K*N(-d2) - discount*inv_K*n(d2)*dd2/dT - inv_S*n(d1)*dd1/dT
    manual_theta_call = (-r * discount * inv_K * norm.cdf(-d2)
                         - discount * inv_K * n_d2 * d_d2_dT_formula
                         - inv_S * n_d1 * d_d1_dT_formula) / 365.0

    # 代码中的Call Theta
    code_theta_call = (-r * discount * inv_K * norm.cdf(-d2)
                       - discount * inv_K * norm.pdf(d2) * d_d2_dT_code
                       + inv_S * n_d1 * d_d1_dT_code) / 365.0

    print(f"\n--- Call Theta ---")
    print(f"手动推导: {manual_theta_call:.10e}")
    print(f"代码实现: {code_theta_call:.10e}")

    # 差异分析
    diff_call = manual_theta_call - code_theta_call
    print(f"差异: {diff_call:.10e}")

    # 第三项符号问题
    term3_manual = -inv_S * n_d1 * d_d1_dT_formula / 365.0
    term3_code = +inv_S * n_d1 * d_d1_dT_code / 365.0
    print(f"\n第三项符号分析:")
    print(f"  手动推导第三项: {term3_manual:.10e}")
    print(f"  代码实现第三项: {term3_code:.10e}")
    print(f"  符号相反: {'是' if abs(term3_manual + term3_code) < 1e-15 else '否'}")

    # 手动推导Put Theta
    # dP/dT = -inv_S*n(d1)*dd1/dT + r*discount*inv_K*N(d2) + discount*inv_K*n(d2)*dd2/dT
    manual_theta_put = (-inv_S * n_d1 * d_d1_dT_formula
                        + r * discount * inv_K * norm.cdf(d2)
                        + discount * inv_K * n_d2 * d_d2_dT_formula) / 365.0

    # 代码中的Put Theta
    code_theta_put = (r * discount * inv_K * norm.cdf(d2)
                      + discount * inv_K * norm.pdf(-d2) * d_d2_dT_code
                      - inv_S * n_d1 * (-d_d1_dT_code)) / 365.0

    print(f"\n--- Put Theta ---")
    print(f"手动推导: {manual_theta_put:.10e}")
    print(f"代码实现: {code_theta_put:.10e}")

    diff_put = manual_theta_put - code_theta_put
    print(f"差异: {diff_put:.10e}")

    # 数值验证
    def price_func_call(t):
        if t <= 0:
            return max(0, 1/K - 1/S)
        return InverseOptionPricer.calculate_price(S, K, t, r, sigma, "call")

    def price_func_put(t):
        if t <= 0:
            return max(0, 1/S - 1/K)
        return InverseOptionPricer.calculate_price(S, K, t, r, sigma, "put")

    h = 1e-6
    numerical_theta_call = (price_func_call(T + h) - price_func_call(T - h)) / (2 * h) / 365.0
    numerical_theta_put = (price_func_put(T + h) - price_func_put(T - h)) / (2 * h) / 365.0

    print(f"\n--- 数值验证 ---")
    print(f"Call Theta数值: {numerical_theta_call:.10e}")
    print(f"Put Theta数值: {numerical_theta_put:.10e}")

    greeks_call = InverseOptionPricer.calculate_greeks(S, K, T, r, sigma, "call")
    greeks_put = InverseOptionPricer.calculate_greeks(S, K, T, r, sigma, "put")

    print(f"\n代码Call Theta vs 数值: {greeks_call.theta:.10e} vs {numerical_theta_call:.10e}")
    print(f"代码Put Theta vs 数值: {greeks_put.theta:.10e} vs {numerical_theta_put:.10e}")

    return greeks_call.theta, greeks_put.theta


def verify_portfolio_conversion():
    """
    验证跨币种Greeks转换逻辑

    在greeks.py的analyze_portfolio中:
    spot_fx = spot * fx_rate
    total = total + PortfolioGreeks(
        delta=position_greeks.delta * spot_fx,
        gamma=position_greeks.gamma * spot_fx * spot,
        ...
    )

    问题:
    1. Delta转换: delta * spot * fx_rate
       如果delta是币本位期权的delta (单位 BTC per USD)
       那么 delta * spot (USD) = BTC
       再 * fx_rate (USD per BTC) = USD
       这看起来是正确的

    2. Gamma转换: gamma * spot_fx * spot = gamma * spot^2 * fx_rate
       币本位gamma单位是 BTC per USD^2
       gamma * spot^2 (USD^2) = BTC
       再 * fx_rate = USD
       这也看起来正确

    但需要验证实际数值
    """
    print("\n" + "=" * 70)
    print("跨币种Greeks转换验证")
    print("=" * 70)

    from research.risk.greeks import GreeksRiskAnalyzer
    from core.types import OptionContract, OptionType, Position
    from datetime import datetime

    analyzer = GreeksRiskAnalyzer(risk_free_rate=0.05)

    as_of = datetime(2024, 1, 1)
    expiry = datetime(2024, 4, 1)

    # 创建BTC期权头寸
    btc_contract = OptionContract(
        underlying="BTC-USD",
        strike=50000,
        expiry=expiry,
        option_type=OptionType.CALL
    )
    btc_position = Position(instrument="BTC-USD", size=1.0, avg_entry_price=0.0001)

    spot = 50000
    iv = 0.60

    per_contract, position_greeks = analyzer.analyze_position(
        btc_position, btc_contract, spot, iv, as_of
    )

    print(f"\n--- 原始Greeks (标准BS) ---")
    print(f"Delta: {per_contract.delta:.6f}")
    print(f"Gamma: {per_contract.gamma:.10e}")

    # 分析组合
    positions = [(btc_position, btc_contract, spot, iv)]
    fx_rates = {"BTC": 50000}

    portfolio_result = analyzer.analyze_portfolio(positions, as_of, fx_rates)
    btc_portfolio = portfolio_result.get("BTC")

    print(f"\n--- 转换后Greeks (USD名义价值) ---")
    print(f"Delta: {btc_portfolio.delta:.2f} USD")
    print(f"Gamma: {btc_portfolio.gamma:.6f} USD")

    # 验证Delta转换
    # 标准BS delta是 per 1 USD spot change
    # 对于1个BTC的期权，delta * spot 应该是 USD名义价值
    expected_delta_usd = per_contract.delta * spot  # 这是USD价值
    print(f"\n--- Delta转换验证 ---")
    print(f"原始Delta: {per_contract.delta:.6f} (per USD)")
    print(f"乘以spot ({spot}): {per_contract.delta * spot:.2f} USD")
    print(f"代码转换后: {btc_portfolio.delta:.2f} USD")
    print(f"注意: 代码还乘以了fx_rate，但BTC的fx_rate就是spot本身")

    # 实际上代码中是: delta * spot * fx_rate
    # 对于BTC，fx_rate = 50000, spot = 50000
    # 所以是 delta * 50000 * 50000 = delta * 2.5e9
    # 这看起来太大了！

    print(f"\n--- 量纲分析 ---")
    print(f"标准BS Delta量纲: [USD per USD] = 无单位 (或理解为期权价值变化/标的变化)")
    print(f"对于看涨期权，Delta ~ 0.5")
    print(f"乘以 spot ({spot}) 和 fx_rate ({fx_rates['BTC']}):")
    print(f"  结果: {per_contract.delta * spot * fx_rates['BTC']:.2f}")
    print(f"  这代表了名义价值，但单位需要仔细考虑")

    return btc_portfolio


def main():
    print("\n" + "=" * 70)
    print("币本位期权定价模型 - 第二次深度数学审查")
    print("=" * 70)

    # 1. Gamma公式验证
    gamma_call, gamma_put = verify_gamma_formula()

    # 2. Theta公式验证
    theta_call, theta_put = verify_theta_formula()

    # 3. 跨币种转换验证
    portfolio = verify_portfolio_conversion()

    print("\n" + "=" * 70)
    print("审查总结")
    print("=" * 70)

    print("""
关键发现:

1. Gamma公式:
   - 代码实现: Gamma = -2/S^3 * N(-d1) + n(d1)/(S^3 * sigma * sqrt(T)) [Call]
   - 文档说"Gamma为负"，但对于ATM期权，第二项可能超过第一项
   - 需要进一步验证数值结果

2. Theta公式:
   - Call Theta第三项符号可能有错误
   - 手动推导: -inv_S * n_d1 * d_d1_dT
   - 代码实现: +inv_S * n_d1 * d_d1_dT
   - Put Theta的第三项经过双重负号，实际为 +

3. 跨币种转换:
   - Delta转换使用 delta * spot * fx_rate
   - 对于币本位期权，这个公式可能需要调整
   - 因为币本位期权的delta量纲与标准BS不同
""")


if __name__ == "__main__":
    main()
