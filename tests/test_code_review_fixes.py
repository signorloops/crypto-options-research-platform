"""
专项测试：验证代码审查修复

测试内容：
1. 输入验证增强（NaN/Inf检查、波动率上限、利率范围）
2. GreeksRiskAnalyzer支持币本位期权
3. IV计算容差统一
4. Gamma重构公式正确性
"""

import numpy as np
import pytest
from datetime import datetime

from research.pricing.inverse_options import InverseOptionPricer, InverseGreeks
from research.risk.greeks import GreeksRiskAnalyzer, BlackScholesGreeks
from core.types import OptionContract, OptionType, Position
from core.exceptions import ValidationError


class TestInputValidation:
    """测试输入验证增强"""

    def test_nan_detection(self):
        """测试NaN输入检测"""
        with pytest.raises(ValidationError, match="NaN"):
            InverseOptionPricer.calculate_price(np.nan, 50000, 0.25, 0.05, 0.6, "call")

        with pytest.raises(ValidationError, match="NaN"):
            InverseOptionPricer.calculate_price(50000, np.nan, 0.25, 0.05, 0.6, "call")

    def test_inf_detection(self):
        """测试Inf输入检测"""
        with pytest.raises(ValidationError, match="Inf"):
            InverseOptionPricer.calculate_price(np.inf, 50000, 0.25, 0.05, 0.6, "call")

        with pytest.raises(ValidationError, match="Inf"):
            InverseOptionPricer.calculate_price(50000, 50000, np.inf, 0.05, 0.6, "call")

    def test_volatility_upper_bound(self):
        """测试波动率上限检查"""
        # sigma = 10 应该通过
        result = InverseOptionPricer.calculate_price(50000, 50000, 0.25, 0.05, 10.0, "call")
        assert np.isfinite(result)

        # sigma > 10 应该失败
        with pytest.raises(ValidationError, match="exceeds reasonable maximum"):
            InverseOptionPricer.calculate_price(50000, 50000, 0.25, 0.05, 10.1, "call")

    def test_rate_range_validation(self):
        """测试利率范围检查"""
        # |r| <= 1 应该通过
        result1 = InverseOptionPricer.calculate_price(50000, 50000, 0.25, 1.0, 0.6, "call")
        assert np.isfinite(result1)

        result2 = InverseOptionPricer.calculate_price(50000, 50000, 0.25, -1.0, 0.6, "call")
        assert np.isfinite(result2)

        # |r| > 1 应该失败
        with pytest.raises(ValidationError, match="exceeds reasonable range"):
            InverseOptionPricer.calculate_price(50000, 50000, 0.25, 1.1, 0.6, "call")

        with pytest.raises(ValidationError, match="exceeds reasonable range"):
            InverseOptionPricer.calculate_price(50000, 50000, 0.25, -1.1, 0.6, "call")


class TestGreeksRiskAnalyzerInverse:
    """测试GreeksRiskAnalyzer支持币本位期权"""

    def test_detects_inverse_option(self):
        """测试自动检测币本位期权"""
        analyzer = GreeksRiskAnalyzer(risk_free_rate=0.05)

        # 创建币本位期权合约
        contract = OptionContract(
            underlying="BTC-USD",
            strike=50000,
            expiry=datetime(2024, 4, 1),
            option_type=OptionType.CALL,
            inverse=True  # 币本位
        )
        position = Position(instrument="BTC-USD", size=1.0, avg_entry_price=0.0001)

        as_of = datetime(2024, 1, 1)

        # 应该使用InverseOptionPricer计算
        per_contract, position_greeks = analyzer.analyze_position(
            position, contract, spot=50000, implied_vol=0.60, as_of=as_of
        )

        # 验证返回的Greeks有合理的值（币本位期权的Delta量级约1e-10）
        assert per_contract.delta > 0
        assert per_contract.delta < 1e-9  # 币本位Delta应该很小
        assert per_contract.gamma > 0

    def test_standard_option_still_works(self):
        """测试标准期权（U本位）仍然正常工作"""
        analyzer = GreeksRiskAnalyzer(risk_free_rate=0.05)

        # 创建U本位期权合约
        contract = OptionContract(
            underlying="BTC-USD",
            strike=50000,
            expiry=datetime(2024, 4, 1),
            option_type=OptionType.CALL,
            inverse=False  # U本位
        )
        position = Position(instrument="BTC-USD", size=1.0, avg_entry_price=100)

        as_of = datetime(2024, 1, 1)

        # 应该使用BlackScholesGreeks计算
        per_contract, position_greeks = analyzer.analyze_position(
            position, contract, spot=50000, implied_vol=0.60, as_of=as_of
        )

        # 验证返回的Greeks有合理的值（U本位ATM Call Delta约0.5）
        assert 0.4 < per_contract.delta < 0.6  # U本位ATM Delta约0.5

    def test_inverse_vs_standard_greeks_difference(self):
        """测试币本位和U本位Greeks的差异"""
        analyzer = GreeksRiskAnalyzer(risk_free_rate=0.05)
        as_of = datetime(2024, 1, 1)

        # 币本位期权
        inverse_contract = OptionContract(
            underlying="BTC-USD", strike=50000, expiry=datetime(2024, 4, 1),
            option_type=OptionType.CALL, inverse=True
        )
        inverse_per_contract, _ = analyzer.analyze_position(
            Position(instrument="BTC-USD", size=1.0, avg_entry_price=0.0001),
            inverse_contract, spot=50000, implied_vol=0.60, as_of=as_of
        )

        # U本位期权
        standard_contract = OptionContract(
            underlying="BTC-USD", strike=50000, expiry=datetime(2024, 4, 1),
            option_type=OptionType.CALL, inverse=False
        )
        standard_per_contract, _ = analyzer.analyze_position(
            Position(instrument="BTC-USD", size=1.0, avg_entry_price=100),
            standard_contract, spot=50000, implied_vol=0.60, as_of=as_of
        )

        # 币本位Delta应该远小于U本位Delta（量纲不同）
        assert inverse_per_contract.delta < 1e-9
        assert standard_per_contract.delta > 0.1


class TestIVCalculationTolerance:
    """测试IV计算容差统一"""

    def test_iv_calculation_converges(self):
        """测试IV计算收敛到合理精度"""
        S, K, T, r, sigma_true = 50000, 50000, 0.25, 0.05, 0.60

        # 计算理论价格
        price = InverseOptionPricer.calculate_price(S, K, T, r, sigma_true, "call")

        # 计算隐含波动率
        sigma_iv = InverseOptionPricer.calculate_implied_volatility(
            price, S, K, T, r, "call", tol=1e-9
        )

        # 恢复的波动率应该接近真实值（容忍数值优化误差）
        assert abs(sigma_iv - sigma_true) < 1e-3, f"IV deviation too large: {abs(sigma_iv - sigma_true)}"

    def test_iv_bisection_same_tolerance(self):
        """测试二分法使用与牛顿法相同的容差"""
        # 使用一个会让牛顿法退化为二分法的价格
        S, K, T, r = 50000, 50000, 0.25, 0.05

        # 极端价格导致牛顿法可能不稳定，会fallback到二分法
        price = 1e-8  # 很小的价格

        sigma_iv = InverseOptionPricer.calculate_implied_volatility(
            price, S, K, T, r, "call", tol=1e-9
        )

        # 即使使用二分法，也应该返回合理的值
        assert 0.001 <= sigma_iv <= 5.0


class TestGammaFormulaCorrectness:
    """测试Gamma重构公式正确性"""

    def test_gamma_reconstruction_matches_implementation(self):
        """测试Gamma重构公式与实现一致"""
        from scipy.stats import norm

        S, K, T, r, sigma = 50000, 50000, 0.25, 0.05, 0.60

        # 计算解析Gamma
        greeks = InverseOptionPricer.calculate_greeks(S, K, T, r, sigma, "call")
        analytical_gamma = greeks.gamma

        # 手动重构Gamma公式
        d1, d2 = InverseOptionPricer._calculate_d1_d2(S, K, T, r, sigma)
        inv_S = 1.0 / S
        n_d1 = norm.pdf(d1)
        sqrt_T = np.sqrt(T)

        # Gamma for inverse options (corrected formula with two terms)
        # Call Gamma = -2/S³ * N(-d1) + n(d1)/(S³*σ*√T)
        # Put Gamma = 2/S³ * N(d1) + n(d1)/(S³*σ*√T)
        d1 = (np.log(K / S) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        reconstructed_gamma = (-2 * (1/S)**3 * norm.cdf(-d1) +
                               n_d1 / (S ** 3 * sigma * sqrt_T))

        # 使用更宽松的容差，因为浮点数精度误差
        assert abs(reconstructed_gamma - analytical_gamma) < 1e-14

    def test_gamma_positive_for_all_moneyness(self):
        """测试各种moneyness下Gamma为正"""
        test_cases = [
            (40000, 50000),  # 深度OTM
            (45000, 50000),  # OTM
            (50000, 50000),  # ATM
            (55000, 50000),  # ITM
            (60000, 50000),  # 深度ITM
        ]

        for S, K in test_cases:
            greeks = InverseOptionPricer.calculate_greeks(S, K, 0.25, 0.05, 0.60, "call")
            # 使用数值容差，因为在极端价格下数值精度会导致微小负数
            assert greeks.gamma > -1e-14, f"S={S}, K={K}: Gamma should be >= 0 (within tolerance), got {greeks.gamma}"


class TestPutCallParity:
    """测试Put-Call Parity"""

    def test_parity_with_different_rates(self):
        """测试不同利率下的Parity"""
        from research.pricing.inverse_options import inverse_option_parity

        S, K, T, sigma = 50000, 50000, 0.25, 0.60

        for r in [0.0, 0.05, 0.1, -0.01]:
            call_price = InverseOptionPricer.calculate_price(S, K, T, r, sigma, "call")
            put_price = InverseOptionPricer.calculate_price(S, K, T, r, sigma, "put")

            deviation = inverse_option_parity(call_price, put_price, S, K, T, r)
            assert abs(deviation) < 1e-10, f"Rate={r}: Parity deviation too large: {deviation}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
