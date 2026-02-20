"""
Tests for risk management modules.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.types import Greeks, OptionContract, OptionType, Position
from research.risk.greeks import BlackScholesGreeks, GreeksRiskAnalyzer, PortfolioGreeks
from research.risk.var import StressTest, VaRCalculator, VaRResult


class TestBlackScholesGreeks:
    """Test Black-Scholes Greeks calculation."""

    def test_calculate_call_greeks(self):
        """Test calculating call option Greeks."""
        greeks = BlackScholesGreeks.calculate(
            S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='call'
        )

        assert isinstance(greeks, Greeks)
        assert 0.4 < greeks.delta < 0.6  # ATM call delta ~ 0.5
        assert greeks.gamma > 0
        assert greeks.vega > 0
        assert greeks.theta < 0  # Theta is typically negative for long options
        assert greeks.rho > 0  # Call rho is positive

    def test_calculate_put_greeks(self):
        """Test calculating put option Greeks."""
        greeks = BlackScholesGreeks.calculate(
            S=100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='put'
        )

        assert isinstance(greeks, Greeks)
        assert -0.6 < greeks.delta < -0.4  # ATM put delta ~ -0.5
        assert greeks.gamma > 0
        assert greeks.vega > 0
        assert greeks.rho < 0  # Put rho is negative

    def test_itm_call_delta(self):
        """Test ITM call has higher delta."""
        greeks = BlackScholesGreeks.calculate(
            S=110, K=100, T=0.25, r=0.05, sigma=0.2, option_type='call'
        )
        assert greeks.delta > 0.6

    def test_otm_call_delta(self):
        """Test OTM call has lower delta."""
        greeks = BlackScholesGreeks.calculate(
            S=90, K=100, T=0.25, r=0.05, sigma=0.2, option_type='call'
        )
        assert greeks.delta < 0.4

    def test_input_validation(self):
        """Test input validation."""
        with pytest.raises(ValueError, match="Spot price S must be positive"):
            BlackScholesGreeks.calculate(
                S=-100, K=100, T=0.25, r=0.05, sigma=0.2, option_type='call'
            )

        with pytest.raises(ValueError, match="Strike price K must be positive"):
            BlackScholesGreeks.calculate(
                S=100, K=-100, T=0.25, r=0.05, sigma=0.2, option_type='call'
            )

        with pytest.raises(ValueError, match="Volatility sigma must be positive"):
            BlackScholesGreeks.calculate(
                S=100, K=100, T=0.25, r=0.05, sigma=-0.2, option_type='call'
            )

    def test_expired_option(self):
        """Test expired option returns zero Greeks."""
        greeks = BlackScholesGreeks.calculate(
            S=100, K=100, T=0, r=0.05, sigma=0.2, option_type='call'
        )
        assert greeks.delta == 0
        assert greeks.gamma == 0
        assert greeks.theta == 0
        assert greeks.vega == 0
        assert greeks.rho == 0


class TestPortfolioGreeks:
    """Test PortfolioGreeks dataclass."""

    def test_initialization(self):
        """Test PortfolioGreeks initialization."""
        pg = PortfolioGreeks(
            delta=0.5, gamma=0.01, theta=-0.1,
            vega=0.2, rho=0.05, vanna=0.01, charm=-0.01, veta=0.0
        )
        assert pg.delta == 0.5
        assert pg.gamma == 0.01

    def test_addition(self):
        """Test adding two PortfolioGreeks."""
        pg1 = PortfolioGreeks(
            delta=0.5, gamma=0.01, theta=-0.1,
            vega=0.2, rho=0.05, vanna=0.01, charm=-0.01, veta=0.0
        )
        pg2 = PortfolioGreeks(
            delta=-0.3, gamma=0.02, theta=-0.05,
            vega=0.1, rho=-0.02, vanna=0.005, charm=-0.005, veta=0.0
        )

        result = pg1 + pg2
        assert result.delta == pytest.approx(0.2)
        assert result.gamma == pytest.approx(0.03)
        assert result.theta == pytest.approx(-0.15)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        pg = PortfolioGreeks(
            delta=0.5, gamma=0.01, theta=-0.1,
            vega=0.2, rho=0.05, vanna=0.01, charm=-0.01, veta=0.0
        )
        d = pg.to_dict()
        assert d['delta'] == 0.5
        assert d['gamma'] == 0.01
        assert 'vanna' in d


class TestGreeksRiskAnalyzer:
    """Test GreeksRiskAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = GreeksRiskAnalyzer(risk_free_rate=0.05)
        assert analyzer.risk_free_rate == 0.05

    def test_analyze_position(self):
        """Test analyzing a single position."""
        analyzer = GreeksRiskAnalyzer(risk_free_rate=0.05)

        position = Position(instrument='BTC-CALL-100', size=2.0, avg_entry_price=5.0)
        contract = OptionContract(
            underlying='BTC',
            strike=100.0,
            expiry=datetime.now() + timedelta(days=30),
            option_type=OptionType.CALL
        )

        per_contract, position_greeks = analyzer.analyze_position(
            position=position,
            contract=contract,
            spot=100.0,
            implied_vol=0.5,
            as_of=datetime.now()
        )

        assert isinstance(per_contract, Greeks)
        assert isinstance(position_greeks, Greeks)
        # Position Greeks should be scaled by position size
        assert position_greeks.delta == pytest.approx(per_contract.delta * 2.0)

    def test_analyze_portfolio(self):
        """Test analyzing a portfolio of positions."""
        analyzer = GreeksRiskAnalyzer(risk_free_rate=0.05)

        as_of = datetime.now()
        positions = [
            (
                Position(instrument='BTC-CALL-100', size=2.0, avg_entry_price=5.0),
                OptionContract(
                    underlying='BTC', strike=100.0,
                    expiry=as_of + timedelta(days=30),
                    option_type=OptionType.CALL
                ),
                100.0,  # spot
                0.5     # implied vol
            ),
            (
                Position(instrument='BTC-PUT-100', size=-1.0, avg_entry_price=5.0),
                OptionContract(
                    underlying='BTC', strike=100.0,
                    expiry=as_of + timedelta(days=30),
                    option_type=OptionType.PUT
                ),
                100.0,  # spot
                0.5     # implied vol
            )
        ]

        portfolio_greeks = analyzer.analyze_portfolio(positions, as_of)
        assert isinstance(portfolio_greeks, dict)
        # Should have BTC key
        assert 'BTC' in portfolio_greeks
        assert isinstance(portfolio_greeks['BTC'], PortfolioGreeks)
        # Portfolio should have net Greeks from both positions
        assert portfolio_greeks['BTC'].delta != 0

    def test_find_hedge_ratio(self):
        """Test hedge ratio calculation."""
        analyzer = GreeksRiskAnalyzer(risk_free_rate=0.05)

        portfolio_greeks = PortfolioGreeks(
            delta=5.0, gamma=0.1, theta=-0.5,
            vega=1.0, rho=0.1, vanna=0.01, charm=-0.01, veta=0.0
        )

        hedge = analyzer.find_hedge_ratio(portfolio_greeks, hedge_instrument='spot')
        assert hedge == -5.0  # Should be negative of delta

    def test_calculate_greeks_scenarios(self):
        """Test scenario analysis."""
        analyzer = GreeksRiskAnalyzer(risk_free_rate=0.05)

        as_of = datetime.now()
        positions = [
            (
                Position(instrument='BTC-CALL-100', size=1.0, avg_entry_price=5.0),
                OptionContract(
                    underlying='BTC', strike=100.0,
                    expiry=as_of + timedelta(days=30),
                    option_type=OptionType.CALL
                ),
                100.0,  # spot
                0.5     # implied vol
            )
        ]

        scenarios = analyzer.calculate_greeks_scenarios(
            positions, as_of,
            spot_shocks=[0.95, 1.0, 1.05],
            vol_shocks=[-0.1, 0, 0.1]
        )

        assert isinstance(scenarios, pd.DataFrame)
        assert 'spot_shock' in scenarios.columns
        assert 'vol_shock' in scenarios.columns
        assert 'pnl' in scenarios.columns
        assert len(scenarios) == 9  # 3 spot shocks * 3 vol shocks


class TestVaRCalculator:
    """Test Value at Risk calculator."""

    def test_initialization(self):
        """Test calculator initialization."""
        calc = VaRCalculator(confidence_level=0.95)
        assert calc.confidence_level == 0.95

    def test_parametric_var(self):
        """Test parametric VaR with DataFrames."""
        calc = VaRCalculator(confidence_level=0.95)

        # Create positions DataFrame
        positions = pd.DataFrame({
            'value': [50000, 50000]
        }, index=['BTC', 'ETH'])

        # Create returns DataFrame
        np.random.seed(42)
        returns = pd.DataFrame({
            'BTC': np.random.normal(0, 0.02, 1000),
            'ETH': np.random.normal(0, 0.025, 1000)
        })

        result = calc.parametric_var(positions, returns)

        assert isinstance(result, VaRResult)
        assert result.method == 'parametric'
        assert result.var_95 > 0  # VaR is positive (loss amount)
        assert result.var_99 > result.var_95  # 99% VaR should be larger
        assert result.cvar_95 > result.var_95  # CVaR should be larger than VaR

    def test_historical_var(self):
        """Test historical VaR with DataFrames."""
        calc = VaRCalculator(confidence_level=0.95)

        positions = pd.DataFrame({
            'value': [50000, 50000]
        }, index=['BTC', 'ETH'])

        np.random.seed(42)
        returns = pd.DataFrame({
            'BTC': np.random.normal(0, 0.02, 1000),
            'ETH': np.random.normal(0, 0.025, 1000)
        })

        result = calc.historical_var(positions, returns)

        assert isinstance(result, VaRResult)
        assert result.method == 'historical'
        assert result.var_95 > 0
        assert result.var_99 > result.var_95

    def test_monte_carlo_var(self):
        """Test Monte Carlo VaR with DataFrames."""
        calc = VaRCalculator(confidence_level=0.95)

        positions = pd.DataFrame({
            'value': [50000, 50000]
        }, index=['BTC', 'ETH'])

        np.random.seed(42)
        returns = pd.DataFrame({
            'BTC': np.random.normal(0, 0.02, 1000),
            'ETH': np.random.normal(0, 0.025, 1000)
        })

        result = calc.monte_carlo_var(positions, returns, n_simulations=5000)

        assert isinstance(result, VaRResult)
        assert result.method == 'monte_carlo'
        assert result.var_95 > 0
        assert result.var_99 > result.var_95

    def test_monte_carlo_var_full_revaluation_for_options(self):
        """Option contract fields should trigger non-linear full revaluation path."""
        calc = VaRCalculator(confidence_level=0.95)

        positions = pd.DataFrame(
            {
                "value": [25000.0],
                "spot": [50000.0],
                "strike": [50000.0],
                "time_to_expiry": [0.25],  # years
                "option_type": ["call"],
                "implied_vol": [0.75],
                "risk_free_rate": [0.01],
                "vol_of_vol": [0.30],
            },
            index=["BTC-OPT"],
        )

        # Zero return variance means linear path would collapse to ~zero VaR.
        returns = pd.DataFrame({"BTC-OPT": np.zeros(365)})
        np.random.seed(42)
        result = calc.monte_carlo_var(positions, returns, greeks=None, n_simulations=4000)

        assert result.method == "monte_carlo"
        assert np.isfinite(result.var_95)
        assert np.isfinite(result.var_99)
        assert result.var_95 > 0

    def test_monte_carlo_full_revaluation_depends_on_strike(self):
        """With full revaluation, different strikes should produce different VaR."""
        calc = VaRCalculator(confidence_level=0.95)

        base_positions = pd.DataFrame(
            {
                "value": [25000.0],
                "spot": [50000.0],
                "time_to_expiry": [0.35],
                "option_type": ["call"],
                "implied_vol": [0.70],
                "risk_free_rate": [0.01],
                "vol_of_vol": [0.20],
            },
            index=["BTC-OPT"],
        )
        positions_low_strike = base_positions.assign(strike=[42000.0])
        positions_high_strike = base_positions.assign(strike=[62000.0])

        returns = pd.DataFrame({"BTC-OPT": np.random.normal(0, 0.01, 400)})

        np.random.seed(123)
        low = calc.monte_carlo_var(positions_low_strike, returns, greeks=None, n_simulations=3000)
        np.random.seed(123)
        high = calc.monte_carlo_var(positions_high_strike, returns, greeks=None, n_simulations=3000)

        assert low.method == "monte_carlo"
        assert high.method == "monte_carlo"
        assert abs(low.var_95 - high.var_95) > 1e-6

    def test_var_result_to_dict(self):
        """Test VaRResult conversion to dict."""
        result = VaRResult(
            var_95=1000.0,
            var_99=2000.0,
            cvar_95=1500.0,
            cvar_99=2500.0,
            method='parametric'
        )
        d = result.to_dict()
        assert d['var_95'] == 1000.0
        assert d['var_99'] == 2000.0
        assert d['method'] == 'parametric'

    def test_parametric_var_aligns_positions_with_return_columns(self):
        """VaR should align weights to return columns by asset name, not raw array order."""
        calc = VaRCalculator(confidence_level=0.95)

        positions = pd.DataFrame(
            {"value": [90000.0, 10000.0]},
            index=["BTC", "ETH"],
        )

        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            {
                # Intentionally reversed column order vs positions index
                "ETH": rng.normal(0, 0.005, 2000),
                "BTC": rng.normal(0, 0.040, 2000),
            }
        )

        result = calc.parametric_var(positions, returns)

        aligned_returns = returns[["BTC", "ETH"]]
        total_value = positions["value"].sum()
        weights = (positions["value"] / total_value).values
        mean_return = aligned_returns.mean().values @ weights
        portfolio_std = np.sqrt(weights @ aligned_returns.cov().values @ weights)

        z_95 = stats.norm.ppf(0.95)
        expected_var_95 = total_value * (-mean_return + z_95 * portfolio_std)

        assert result.var_95 == pytest.approx(expected_var_95, rel=1e-12)

    def test_parametric_var_raises_when_returns_missing_asset(self):
        """Missing returns for a held asset should raise, not silently miscompute."""
        calc = VaRCalculator(confidence_level=0.95)

        positions = pd.DataFrame(
            {"value": [90000.0, 10000.0]},
            index=["BTC", "SOL"],
        )
        returns = pd.DataFrame({"BTC": np.random.normal(0, 0.02, 1000)})

        with pytest.raises(ValueError, match="missing return series"):
            calc.parametric_var(positions, returns)

    def test_parametric_var_raises_on_zero_portfolio_value(self):
        """Zero-value portfolio should raise instead of dividing by zero."""
        calc = VaRCalculator(confidence_level=0.95)

        positions = pd.DataFrame({"value": [0.0, 0.0]}, index=["BTC", "ETH"])
        returns = pd.DataFrame(
            {
                "BTC": np.random.normal(0, 0.02, 1000),
                "ETH": np.random.normal(0, 0.02, 1000),
            }
        )

        with pytest.raises(ValueError, match="total portfolio value must be non-zero"):
            calc.parametric_var(positions, returns)

    def test_parametric_var_supports_dollar_neutral_long_short(self):
        """Net-zero long/short books should still produce finite risk metrics."""
        calc = VaRCalculator(confidence_level=0.95)

        positions = pd.DataFrame(
            {"value": [100000.0, -100000.0]},
            index=["BTC", "ETH"],
        )

        rng = np.random.default_rng(123)
        returns = pd.DataFrame(
            {
                "BTC": rng.normal(0, 0.02, 2000),
                "ETH": rng.normal(0, 0.025, 2000),
            }
        )

        result = calc.parametric_var(positions, returns)

        assert np.isfinite(result.var_95)
        assert np.isfinite(result.var_99)
        assert result.var_99 >= result.var_95 >= 0


class TestStressTest:
    """Test StressTest class."""

    def test_initialization(self):
        """Test StressTest initialization."""
        st = StressTest()
        assert hasattr(st, 'SCENARIOS')
        assert 'market_crash' in st.SCENARIOS
        assert 'vol_spike' in st.SCENARIOS

    def test_run_stress_test(self):
        """Test running a stress test scenario."""
        st = StressTest()

        positions = pd.DataFrame({
            'value': [100000, 50000]
        }, index=['BTC-CALL', 'BTC-PUT'])

        greeks = pd.DataFrame({
            'delta': [0.5, -0.3],
            'gamma': [0.01, 0.01],
            'vega': [0.2, 0.15]
        }, index=['BTC-CALL', 'BTC-PUT'])

        result = st.run_stress_test(positions, greeks, 'market_crash')

        assert 'scenario_name' in result
        assert 'spot_shock' in result
        assert 'vol_shock' in result
        assert 'estimated_pnl' in result
        assert 'pct_of_portfolio' in result
        assert result['spot_shock'] == -0.20  # Market crash scenario

    def test_run_all_scenarios(self):
        """Test running all predefined scenarios."""
        st = StressTest()

        positions = pd.DataFrame({
            'value': [100000, 50000]
        }, index=['BTC-CALL', 'BTC-PUT'])

        greeks = pd.DataFrame({
            'delta': [0.5, -0.3],
            'gamma': [0.01, 0.01],
            'vega': [0.2, 0.15]
        }, index=['BTC-CALL', 'BTC-PUT'])

        results = st.run_all_scenarios(positions, greeks)

        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(st.SCENARIOS)
        assert 'scenario' in results.columns
        assert 'estimated_pnl' in results.columns

    def test_custom_scenario(self):
        """Test running a custom scenario."""
        st = StressTest()

        positions = pd.DataFrame({
            'value': [100000]
        }, index=['BTC-CALL'])

        greeks = pd.DataFrame({
            'delta': [0.5],
            'gamma': [0.01],
            'vega': [0.2]
        }, index=['BTC-CALL'])

        custom_scenario = {
            'description': 'Custom test scenario',
            'spot_shock': -0.15,
            'vol_shock': 0.30
        }

        result = st.run_stress_test(positions, greeks, custom_scenario)

        assert result['scenario_name'] == 'Custom test scenario'
        assert result['spot_shock'] == -0.15
        assert result['vol_shock'] == 0.30
