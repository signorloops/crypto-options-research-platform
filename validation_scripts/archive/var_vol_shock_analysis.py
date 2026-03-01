"""
VaR计算中的波动率冲击假设分析

验证 var.py 第205行 np.random.normal(0, 0.05) 的合理性
以及蒙特卡洛模拟的样本量是否足够
"""

import numpy as np
from scipy import stats


def analyze_vega_pnl_calculation():
    """
    分析Vega PnL的计算

    代码第205行:
    vega_pnl = g.get('vega', 0) * np.random.normal(0, 0.05, n_simulations) * row['value']
    """
    print("=" * 70)
    print("Vega PnL计算分析")
    print("=" * 70)

    print("\n1. 代码逻辑:")
    print("   vega_pnl = vega * np.random.normal(0, 0.05, n_simulations) * position_value")

    print("\n2. 各项含义:")
    print("   - vega: 每1%波动率变化的价格变化")
    print("   - np.random.normal(0, 0.05): 波动率变化的随机样本")
    print("     均值为0，标准差为0.05 (即5%)")
    print("   - position_value: 头寸价值")

    print("\n3. 问题分析:")
    print("   a) 为什么使用0.05的标准差?")
    print("      - 这假设波动率在持有期内变化约5%")
    print("      - 对于1天的持有期，这可能过高")
    print("      - 对于1个月的持有期，这可能过低")

    print("\n   b) 波动率变化的分布:")
    print("      - 代码假设波动率变化服从正态分布")
    print("      - 实际上波动率变化可能有肥尾")
    print("      - 波动率本身是非负的，但正态分布会产生负值")

    print("\n   c) 与spot return的独立性:")
    print("      - 代码假设波动率变化与价格收益独立")
    print("      - 实际上，波动率与价格通常有负相关(杠杆效应)")


def analyze_vol_shock_assumption():
    """分析波动率冲击假设的合理性"""
    print("\n" + "=" * 70)
    print("波动率冲击假设的合理性分析")
    print("=" * 70)

    print("\n1. 0.05 (5%) 标准差的来源:")
    print("   - 可能是基于历史数据的估计")
    print("   - 或者是行业惯例")
    print("   - 但没有在代码中明确说明")

    print("\n2. 不同持有期下的合理性:")
    holding_periods = [1, 5, 10, 21, 63]  # 1天, 1周, 2周, 1月, 1季度
    daily_vol_vol = 0.05  # 假设的日波动率的波动率

    for hp in holding_periods:
        # 波动率的波动率随时间平方根缩放
        vol_vol_scaled = daily_vol_vol * np.sqrt(hp)
        print(f"   持有期 {hp} 天: 波动率变化标准差 = {vol_vol_scaled:.3f} ({vol_vol_scaled*100:.1f}%)")

    print("\n3. 对比历史数据:")
    print("   - VIX的日均变化约为2-3%")
    print("   - VIX的年化波动率约为100-150%")
    print("   - 0.05的日标准差对应年化约79% (0.05 * sqrt(252))")
    print("   - 这与VIX的实际波动率相当")

    print("\n4. 潜在问题:")
    print("   - 0.05是硬编码的，没有根据市场环境调整")
    print("   - 在危机时期，波动率变化可能远大于5%")
    print("   - 在低波动时期，5%可能过于保守")


def analyze_monte_carlo_sample_size():
    """分析蒙特卡洛样本量的充足性"""
    print("\n" + "=" * 70)
    print("蒙特卡洛样本量分析")
    print("=" * 70)

    print("\n1. 当前设置:")
    print("   n_simulations = 10000")

    print("\n2. 样本量对VaR估计的影响:")
    confidence_levels = [0.95, 0.99, 0.999]
    n_samples = 10000

    for cl in confidence_levels:
        # 分位数的标准误差近似
        # SE ≈ sqrt(p*(1-p)/n) / f(F^-1(p))
        # 对于正态分布，简化估计
        p = 1 - cl
        z_score = stats.norm.ppf(cl)

        # 分位数估计的标准误差 (近似)
        se_quantile = np.sqrt(p * (1-p) / n_samples) / stats.norm.pdf(z_score)

        print(f"   {cl*100:.1f}% VaR的分位数标准误差: {se_quantile:.4f}")

    print("\n3. 样本量建议:")
    sample_sizes = [1000, 10000, 100000, 1000000]

    for n in sample_sizes:
        # 95% VaR的估计误差
        p = 0.05
        se = np.sqrt(p * (1-p) / n)
        print(f"   n={n:7d}: 相对标准误差 = {se*100:.3f}%")

    print("\n4. 结论:")
    print("   - 10000样本对于95% VaR是足够的")
    print("   - 但对于99% VaR，估计可能不够稳定")
    print("   - 对于99.9% VaR，需要更多样本")


def analyze_correlation_structure():
    """分析相关性结构"""
    print("\n" + "=" * 70)
    print("相关性结构分析")
    print("=" * 70)

    print("\n1. 代码中的相关性处理:")
    print("   - 使用历史收益的相关性矩阵")
    print("   - 通过multivariate_normal生成相关随机数")

    print("\n2. 问题:")
    print("   - 历史相关性可能无法预测未来")
    print("   - 在危机时期，相关性通常会上升")
    print("   - 代码没有考虑相关性的时变性")

    print("\n3. 改进建议:")
    print("   - 使用GARCH或EWMA估计时变相关性")
    print("   - 或者使用Copula模型捕捉非线性相关")
    print("   - 进行压力测试，假设相关性上升")


def analyze_greeks_approximation():
    """分析Greeks近似的准确性"""
    print("\n" + "=" * 70)
    print("Greeks近似准确性分析")
    print("=" * 70)

    print("\n1. 代码中的PnL近似:")
    print("   delta_pnl = delta * spot_return * value")
    print("   gamma_pnl = 0.5 * gamma * spot_return^2 * value")
    print("   vega_pnl = vega * vol_shock * value")

    print("\n2. 近似的局限性:")
    print("   - Delta-Gamma近似只在小的价格变化下准确")
    print("   - 对于大的spot shock，高阶项可能很重要")
    print("   - 没有考虑Vanna和Volga等高阶 Greeks")

    print("\n3. 误差分析:")
    spot_returns = np.array([0.01, 0.05, 0.10, 0.20])
    for sr in spot_returns:
        # Taylor展开的误差 (三阶项)
        error_estimate = abs(sr)**3 / 6
        print(f"   Spot return = {sr*100:.0f}%: 估计三阶误差 = {error_estimate*100:.4f}%")


def suggest_improvements():
    """提出改进建议"""
    print("\n" + "=" * 70)
    print("改进建议")
    print("=" * 70)

    print("\n1. 波动率冲击模型:")
    print("   当前: np.random.normal(0, 0.05)")
    print("   建议:")
    print("     a) 根据持有期动态调整标准差")
    print("     b) 使用GARCH模型预测波动率的波动率")
    print("     c) 考虑波动率与价格的负相关性")

    print("\n2. 样本量:")
    print("   当前: n_simulations = 10000")
    print("   建议:")
    print("     a) 对于99% VaR，使用至少50000样本")
    print("     b) 使用重要性采样减少方差")
    print("     c) 使用拟蒙特卡洛(如Sobol序列)")

    print("\n3.  Greeks近似:")
    print("   建议:")
    print("     a) 对于大shock，使用全重估(full revaluation)")
    print("     b) 包含Vanna和Volga等高阶Greeks")
    print("     c) 定期用全重估校准近似误差")

    print("\n4. 相关性:")
    print("   建议:")
    print("     a) 使用时变相关性模型")
    print("     b) 进行相关性压力测试")
    print("     c) 考虑极端情况下的相关性上升")


def numerical_example():
    """数值示例"""
    print("\n" + "=" * 70)
    print("数值示例")
    print("=" * 70)

    # 模拟参数
    n_simulations = 10000
    vega = 0.01  # 每1%波动率变化，价格变化1%
    position_value = 1000000

    print(f"\n假设:")
    print(f"  Vega = {vega} (每1% vol change)")
    print(f"  头寸价值 = ${position_value:,.0f}")

    # 不同的波动率冲击假设
    vol_shock_scenarios = [
        (0.03, "3% std (低波动市场)"),
        (0.05, "5% std (正常市场)"),
        (0.10, "10% std (高波动市场)"),
        (0.20, "20% std (危机市场)"),
    ]

    for vol_std, desc in vol_shock_scenarios:
        vol_shocks = np.random.normal(0, vol_std, n_simulations)
        vega_pnls = vega * vol_shocks * position_value

        var_95 = np.percentile(vega_pnls, 5)
        var_99 = np.percentile(vega_pnls, 1)

        print(f"\n{desc}:")
        print(f"  95% VaR: ${-var_95:,.0f}")
        print(f"  99% VaR: ${-var_99:,.0f}")


if __name__ == "__main__":
    analyze_vega_pnl_calculation()
    analyze_vol_shock_assumption()
    analyze_monte_carlo_sample_size()
    analyze_correlation_structure()
    analyze_greeks_approximation()
    suggest_improvements()
    numerical_example()
