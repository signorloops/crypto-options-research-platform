"""
Tests for arbitrage strategies.
"""
import math
from datetime import datetime, timedelta, timezone

import pytest

from strategies.arbitrage import (
    BasisArbitrage,
    ConversionArbitrage,
    CrossExchangeArbitrage,
    OptionBoxArbitrage,
)
from strategies.arbitrage.conversion import ConversionOpportunity
from strategies.arbitrage.cross_exchange import (
    ArbitrageOpportunity,
    ExchangeFees,
)
from strategies.arbitrage.option_box import BoxSpread


class TestCrossExchangeArbitrage:
    """Test cross-exchange arbitrage strategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = CrossExchangeArbitrage(
            min_spread_bps=50.0,
            min_profit_pct=0.1
        )
        assert strategy.min_spread_bps == 50.0
        assert strategy.min_profit_pct == 0.1

    def test_set_exchange_fees(self):
        """Test setting exchange fees."""
        strategy = CrossExchangeArbitrage()
        fees = ExchangeFees(maker_fee=0.001, taker_fee=0.002)
        strategy.set_exchange_fees("binance", fees)

        assert "binance" in strategy.exchange_fees
        assert strategy.exchange_fees["binance"].taker_fee == 0.002

    def test_update_price(self):
        """Test price update."""
        strategy = CrossExchangeArbitrage()
        strategy.update_price("binance", "BTC-USD", 50000.0)

        assert "BTC-USD" in strategy.price_cache
        # PriceEntry object stores price in .price attribute
        assert strategy.price_cache["BTC-USD"]["binance"].price == 50000.0

    def test_arbitrage_detection(self):
        """Test arbitrage opportunity detection."""
        opportunities = []

        def callback(opp):
            opportunities.append(opp)

        strategy = CrossExchangeArbitrage(
            min_spread_bps=50.0,   # 0.5% threshold
            min_profit_pct=0.001   # 0.1% profit threshold
        )
        strategy.on_opportunity(callback)

        # Set fees
        strategy.set_exchange_fees("exchange_a", ExchangeFees(0.001, 0.002))
        strategy.set_exchange_fees("exchange_b", ExchangeFees(0.001, 0.002))

        # Update prices with 2% spread
        strategy.update_price("exchange_a", "BTC-USD", 50000.0)
        strategy.update_price("exchange_b", "BTC-USD", 51000.0)

        assert len(opportunities) == 1
        assert opportunities[0].spread_bps == 200.0  # (51000-50000)/50000*10000

    def test_no_arbitrage_when_spread_too_small(self):
        """Test that small spreads don't trigger arbitrage."""
        opportunities = []

        strategy = CrossExchangeArbitrage(min_spread_bps=200.0)
        strategy.on_opportunity(lambda x: opportunities.append(x))

        strategy.update_price("a", "BTC", 50000.0)
        strategy.update_price("b", "BTC", 50100.0)  # 0.2% spread

        assert len(opportunities) == 0

    def test_get_best_opportunities(self):
        """Test getting top opportunities."""
        strategy = CrossExchangeArbitrage(min_spread_bps=50.0)

        # Add multiple price pairs
        strategy.update_price("a", "BTC", 50000.0)
        strategy.update_price("b", "BTC", 50500.0)  # 1% spread
        strategy.update_price("c", "BTC", 51000.0)  # 2% spread

        opps = strategy.get_best_opportunities(top_n=2)

        assert len(opps) <= 2
        if len(opps) == 2:
            assert opps[0].spread_bps >= opps[1].spread_bps

    def test_calculate_required_capital(self):
        """Test capital calculation."""
        strategy = CrossExchangeArbitrage()
        strategy.set_exchange_fees("a", ExchangeFees(0.001, 0.002))
        strategy.set_exchange_fees("b", ExchangeFees(0.001, 0.002))

        opp = ArbitrageOpportunity(
            buy_exchange="a",
            sell_exchange="b",
            instrument="BTC",
            buy_price=50000.0,
            sell_price=51000.0,
            spread_bps=200.0,
            profit_pct=1.8,
            timestamp=datetime.now(timezone.utc)
        )

        capital = strategy.calculate_required_capital(opp, position_size=1.0)

        assert "buy_capital" in capital
        assert "sell_collateral" in capital
        assert "expected_profit" in capital
        assert capital["buy_capital"] == 50000.0

    def test_simulate_execution(self):
        """Execution simulator should return realized net metrics."""
        strategy = CrossExchangeArbitrage()
        opp = ArbitrageOpportunity(
            buy_exchange="a",
            sell_exchange="b",
            instrument="BTC",
            buy_price=50000.0,
            sell_price=50500.0,
            spread_bps=100.0,
            profit_pct=0.8,
            timestamp=datetime.now(timezone.utc),
        )
        out = strategy.simulate_execution(opp, position_size=0.5, latency_ms=50)
        assert "net_profit" in out
        assert "roi_pct" in out

    def test_triangular_arbitrage_detection(self):
        """Triangular arbitrage detector should find profitable cycle."""
        strategy = CrossExchangeArbitrage()
        pair_prices = {
            "USDT/BTC": 0.00002,   # 1 USDT -> 0.00002 BTC
            "BTC/ETH": 14.0,       # 1 BTC -> 14 ETH
            "ETH/USDT": 3700.0,    # 1 ETH -> 3700 USDT
        }
        tri = strategy.check_triangular_arbitrage("binance", pair_prices, start_currency="USDT", start_amount=1.0)
        assert tri is not None
        assert tri.profit_pct > 0


class TestBasisArbitrage:
    """Test basis/cash-and-carry arbitrage."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = BasisArbitrage(
            risk_free_rate=0.05,
            min_annualized_return=0.10
        )
        assert strategy.risk_free_rate == 0.05
        assert strategy.min_annualized_return == 0.10

    def test_update_prices(self):
        """Test price updates."""
        strategy = BasisArbitrage()
        strategy.update_spot_price("BTC", 50000.0)
        strategy.update_futures_price("BTC", 52000.0, datetime.now(timezone.utc) + timedelta(days=90))

        assert strategy.spot_prices["BTC"] == 50000.0
        assert strategy.futures_prices["BTC"] == 52000.0

    def test_calculate_fair_value(self):
        """Test fair value calculation."""
        strategy = BasisArbitrage(risk_free_rate=0.05)
        expiry = datetime.now(timezone.utc) + timedelta(days=365)

        fair_value = strategy.calculate_fair_value(
            spot=50000.0,
            expiry=expiry
        )

        # F = S * exp(rT) = 50000 * exp(0.05) ≈ 52564
        assert fair_value > 50000.0
        assert fair_value < 53000.0

    def test_fair_value_uses_subday_precision(self):
        """Sub-day expiry should still accrue carry cost."""
        strategy = BasisArbitrage(risk_free_rate=1.0)
        expiry = datetime.now(timezone.utc) + timedelta(hours=12)
        fair_value = strategy.calculate_fair_value(100.0, expiry)

        assert fair_value > 100.05

    def test_update_spot_price_rejects_nonpositive(self):
        """Spot prices must be positive to avoid invalid basis ratios."""
        strategy = BasisArbitrage()
        with pytest.raises(ValueError):
            strategy.update_spot_price("BTC", 0.0)

    def test_calculate_basis(self):
        """Test basis calculation."""
        strategy = BasisArbitrage()
        strategy.update_spot_price("BTC", 50000.0)
        strategy.update_futures_price(
            "BTC", 52000.0,
            datetime.now(timezone.utc) + timedelta(days=90)
        )

        basis_info = strategy.calculate_basis("BTC")

        assert basis_info is not None
        assert "basis" in basis_info
        assert "basis_pct" in basis_info
        assert basis_info["spot"] == 50000.0
        assert basis_info["futures"] == 52000.0

    def test_opportunity_detection_positive_basis(self):
        """Test positive basis opportunity."""
        strategy = BasisArbitrage(
            risk_free_rate=0.05,
            min_annualized_return=0.0,  # No minimum for test
            transaction_cost=0.0001  # Low cost
        )

        strategy.update_spot_price("BTC", 50000.0)
        strategy.update_futures_price(
            "BTC", 54000.0,  # 8% premium
            datetime.now(timezone.utc) + timedelta(days=90)
        )

        opp = strategy.check_opportunity("BTC")

        assert opp is not None
        assert opp.strategy == "short_basis"
        assert opp.basis > 0

    def test_opportunity_detection_negative_basis(self):
        """Test negative basis opportunity."""
        strategy = BasisArbitrage(
            risk_free_rate=0.05,
            min_annualized_return=0.0,
            transaction_cost=0.0001
        )

        strategy.update_spot_price("BTC", 50000.0)
        strategy.update_futures_price(
            "BTC", 48000.0,  # 4% discount
            datetime.now(timezone.utc) + timedelta(days=30)
        )

        opp = strategy.check_opportunity("BTC")

        assert opp is not None
        assert opp.strategy == "long_basis"
        assert opp.basis < 0

    def test_no_opportunity_when_basis_small(self):
        """Test no opportunity when basis is within bounds."""
        strategy = BasisArbitrage(min_annualized_return=0.20)  # 20% threshold

        strategy.update_spot_price("BTC", 50000.0)
        strategy.update_futures_price(
            "BTC", 50500.0,  # 1% premium
            datetime.now(timezone.utc) + timedelta(days=90)
        )

        opp = strategy.check_opportunity("BTC")
        assert opp is None

    def test_calculate_pnl(self):
        """Test P&L calculation."""
        strategy = BasisArbitrage(transaction_cost=0.001)

        pnl = strategy.calculate_pnl(
            entry_spot=50000.0,
            entry_futures=52000.0,
            exit_spot=51000.0,
            exit_futures=51000.0,
            position_size=1.0
        )

        assert "spot_pnl" in pnl
        assert "futures_pnl" in pnl
        assert "net_pnl" in pnl

    def test_dynamic_funding_and_inverse_hedge_ratio(self):
        """Dynamic funding rate and inverse hedge ratio should be supported."""
        strategy = BasisArbitrage()
        strategy.update_funding_rate("BTC", 0.0001)
        strategy.update_funding_rate("BTC", 0.0002)
        fr = strategy.get_dynamic_funding_rate("BTC")
        assert fr > 0
        ratio = strategy.get_hedge_ratio("BTC", inverse_contract=True, contract_multiplier=1.0)
        assert ratio > 0


class TestOptionBoxArbitrage:
    """Test box spread arbitrage."""

    def test_box_spread_creation(self):
        """Test box spread construction."""
        box = BoxSpread(
            low_strike=100.0,
            high_strike=110.0,
            expiry=datetime.now(timezone.utc) + timedelta(days=30),
            long_call_low=5.0,
            short_put_low=3.0,
            short_call_high=2.0,
            long_put_high=7.0
        )

        assert box.low_strike == 100.0
        assert box.high_strike == 110.0

    def test_box_net_premium(self):
        """Test box premium calculation."""
        box = BoxSpread(
            low_strike=100.0,
            high_strike=110.0,
            expiry=datetime.now(timezone.utc) + timedelta(days=30),
            long_call_low=5.0,   # -5 (pay)
            short_put_low=3.0,   # +3 (receive)
            short_call_high=2.0, # +2 (receive)
            long_put_high=7.0    # -7 (pay)
        )

        # Net = 5 - 3 - 2 + 7 = 7 (cost to enter the box)
        assert box.net_premium == 7.0

    def test_box_value_at_expiry(self):
        """Test box expiry value."""
        box = BoxSpread(
            low_strike=100.0,
            high_strike=110.0,
            expiry=datetime.now(timezone.utc),
            long_call_low=0, short_put_low=0,
            short_call_high=0, long_put_high=0
        )

        assert box.box_value_at_expiry == 10.0  # 110 - 100

    def test_implied_rate_uses_box_spread_discount_relation(self):
        """Implied rate should be derived from box PV relation."""
        expected_rate = 0.05
        premium = 10.0 * math.exp(-expected_rate)
        box = BoxSpread(
            low_strike=100.0,
            high_strike=110.0,
            expiry=datetime.now(timezone.utc) + timedelta(days=365),
            long_call_low=premium,
            short_put_low=0.0,
            short_call_high=0.0,
            long_put_high=0.0
        )

        assert abs(box.implied_rate - expected_rate) < 0.005

    def test_box_building(self):
        """Test building box from option prices."""
        strategy = OptionBoxArbitrage()

        prices = {
            'call_low': 5.0,
            'put_low': 3.0,
            'call_high': 2.0,
            'put_high': 7.0
        }

        box = strategy.build_box(100.0, 110.0, datetime.now(timezone.utc), prices)

        assert box is not None
        assert box.low_strike == 100.0
        assert box.high_strike == 110.0

    def test_box_building_rejects_invalid_strikes(self):
        """Low strike must be strictly below high strike."""
        strategy = OptionBoxArbitrage()
        prices = {
            'call_low': 5.0,
            'put_low': 3.0,
            'call_high': 2.0,
            'put_high': 7.0
        }
        box = strategy.build_box(110.0, 100.0, datetime.now(timezone.utc), prices)
        assert box is None

    def test_arbitrage_detection(self):
        """Test box arbitrage detection."""
        strategy = OptionBoxArbitrage(
            min_annualized_return=0.0,  # No minimum
            transaction_cost_per_leg=0.0  # No cost
        )

        # Box with cost 7, payoff 10 = 3 profit
        box = BoxSpread(
            low_strike=100.0,
            high_strike=110.0,
            expiry=datetime.now(timezone.utc) + timedelta(days=30),
            long_call_low=5.0,
            short_put_low=3.0,
            short_call_high=2.0,
            long_put_high=7.0
        )

        opp = strategy.find_arbitrage(box, "BTC")

        assert opp is not None
        assert opp.profit > 0
        assert opp.payoff == 10.0

    def test_arbitrage_detection_with_net_credit_box(self):
        """Net credit boxes should be treated as strong arbitrage opportunities."""
        strategy = OptionBoxArbitrage(
            min_annualized_return=0.0,
            transaction_cost_per_leg=0.0
        )
        box = BoxSpread(
            low_strike=100.0,
            high_strike=110.0,
            expiry=datetime.now(timezone.utc) + timedelta(days=30),
            long_call_low=0.0,
            short_put_low=1.0,
            short_call_high=0.0,
            long_put_high=0.0
        )

        opp = strategy.find_arbitrage(box, "BTC")

        assert opp is not None
        assert opp.net_cost < 0
        assert opp.annualized_return == float("inf")

    def test_no_arbitrage_when_expensive(self):
        """Test no arb when box is too expensive."""
        strategy = OptionBoxArbitrage()

        # Box costs 11, payoff is 10 = loss
        box = BoxSpread(
            low_strike=100.0,
            high_strike=110.0,
            expiry=datetime.now(timezone.utc),
            long_call_low=6.0,
            short_put_low=2.0,
            short_call_high=1.0,
            long_put_high=8.0
        )

        opp = strategy.find_arbitrage(box)
        assert opp is None

    def test_short_box_detection(self):
        """Overpriced box should be identified as short-box opportunity."""
        strategy = OptionBoxArbitrage(min_annualized_return=0.0, transaction_cost_per_leg=0.0)
        # net_premium=14, payoff=10 => short box profit 4
        box = BoxSpread(
            low_strike=100.0,
            high_strike=110.0,
            expiry=datetime.now(timezone.utc) + timedelta(days=30),
            long_call_low=10.0,
            short_put_low=1.0,
            short_call_high=1.0,
            long_put_high=6.0,
        )
        opp = strategy.find_arbitrage(box, "BTC")
        assert opp is not None
        assert opp.box_type == "short_box"


class TestConversionArbitrage:
    """Test conversion/reversal arbitrage."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = ConversionArbitrage(
            risk_free_rate=0.05,
            min_profit_threshold=0.001
        )
        assert strategy.risk_free_rate == 0.05

    def test_parity_deviation_calculation(self):
        """Test parity deviation calculation."""
        strategy = ConversionArbitrage(risk_free_rate=0.0)

        expiry = datetime.now(timezone.utc) + timedelta(days=365)
        parity = strategy.calculate_parity_deviation(
            call_price=10.0,
            put_price=5.0,
            spot_price=100.0,
            strike=100.0,
            expiry=expiry
        )

        # C - P = 5, S - K*exp(-rT) = 0 (at the money, r=0)
        # Deviation = 5 - 0 = 5
        assert parity['synthetic_forward'] == 5.0
        assert parity['deviation'] == 5.0

    def test_parity_time_to_expiry_uses_subday_precision(self):
        """1-hour expiry should not be rounded up to a full floor year fraction."""
        strategy = ConversionArbitrage(risk_free_rate=1.0)
        expiry = datetime.now(timezone.utc) + timedelta(hours=1)
        parity = strategy.calculate_parity_deviation(
            call_price=10.0,
            put_price=10.0,
            spot_price=100.0,
            strike=100.0,
            expiry=expiry
        )

        assert 0 < parity['time_to_expiry'] < 0.0002

    def test_check_opportunity_returns_none_for_zero_spot(self):
        """Zero spot price should be rejected safely."""
        strategy = ConversionArbitrage(min_profit_threshold=0.0)
        expiry = datetime.now(timezone.utc) + timedelta(days=30)

        opp = strategy.check_opportunity(
            underlying="BTC",
            call_price=20.0,
            put_price=5.0,
            spot_price=0.0,
            strike=100.0,
            expiry=expiry
        )

        assert opp is None

    def test_conversion_opportunity(self):
        """Test conversion arbitrage detection."""
        strategy = ConversionArbitrage(
            risk_free_rate=0.05,
            min_profit_threshold=0.0,
            transaction_cost=0.0
        )

        # Synthetic forward is too expensive
        # C - P = 15, S - K*exp(-rT) ≈ 0
        expiry = datetime.now(timezone.utc) + timedelta(days=90)
        opp = strategy.check_opportunity(
            underlying="BTC",
            call_price=20.0,
            put_price=5.0,  # C - P = 15
            spot_price=100.0,
            strike=100.0,
            expiry=expiry
        )

        assert opp is not None
        assert opp.strategy == "conversion"
        assert opp.deviation > 0

    def test_reversal_opportunity(self):
        """Test reversal arbitrage detection."""
        strategy = ConversionArbitrage(
            risk_free_rate=0.05,
            min_profit_threshold=0.0,
            transaction_cost=0.0
        )

        # Synthetic forward is too cheap
        expiry = datetime.now(timezone.utc) + timedelta(days=90)
        opp = strategy.check_opportunity(
            underlying="BTC",
            call_price=5.0,
            put_price=20.0,  # C - P = -15
            spot_price=100.0,
            strike=100.0,
            expiry=expiry
        )

        assert opp is not None
        assert opp.strategy == "reversal"
        assert opp.deviation < 0

    def test_no_opportunity_at_parity(self):
        """Test no arb when at parity."""
        strategy = ConversionArbitrage(min_profit_threshold=0.01)

        expiry = datetime.now(timezone.utc) + timedelta(days=90)
        opp = strategy.check_opportunity(
            underlying="BTC",
            call_price=10.0,
            put_price=10.0,  # C - P = 0
            spot_price=100.0,
            strike=100.0,
            expiry=expiry
        )

        assert opp is None

    def test_hedge_position(self):
        """Test hedge position calculation."""
        strategy = ConversionArbitrage()

        expiry = datetime.now(timezone.utc) + timedelta(days=30)
        opp = ConversionOpportunity(
            underlying="BTC",
            strike=100.0,
            expiry=expiry,
            call_price=10.0,
            put_price=5.0,
            spot_price=100.0,
            strategy="conversion",
            synthetic_forward=5.0,
            actual_forward=0.0,
            deviation=5.0,
            profit=5.0,
            annualized_return=0.5
        )

        hedge = strategy.get_hedge_position(opp)

        assert hedge['call'] == -1  # Sell call
        assert hedge['put'] == 1    # Buy put
        assert hedge['underlying'] == 1  # Buy underlying

    def test_pnl_calculation(self):
        """Test P&L at expiry."""
        strategy = ConversionArbitrage(transaction_cost=0.0)

        expiry = datetime.now(timezone.utc) + timedelta(days=30)
        opp = ConversionOpportunity(
            underlying="BTC",
            strike=100.0,
            expiry=expiry,
            call_price=12.0,
            put_price=2.0,
            spot_price=100.0,
            strategy="conversion",
            synthetic_forward=10.0,
            actual_forward=0.0,
            deviation=10.0,
            profit=10.0,
            annualized_return=1.0
        )

        # At expiry, spot = strike = 100
        pnl = strategy.calculate_pnl_scenarios(opp, spot_at_expiry=100.0)

        # Conversion: sell call (expires worthless), buy put (expires worthless), buy spot
        # PnL = 12 - 2 - 0 + 0 = 10 (roughly)
        assert pnl > 0

    def test_boundary_verification(self):
        """Test no-arbitrage boundary verification."""
        strategy = ConversionArbitrage(risk_free_rate=0.0)

        expiry = datetime.now(timezone.utc) + timedelta(days=90)
        bounds = strategy.verify_arbitrage_bounds(
            call_price=10.0,
            put_price=10.0,
            spot_price=100.0,
            strike=100.0,
            expiry=expiry
        )

        assert 'call_lower_bound_satisfied' in bounds
        assert 'put_lower_bound_satisfied' in bounds
        assert 'parity_satisfied' in bounds

    def test_streaming_snapshot_and_staking_yield(self):
        """Streaming snapshot should support opportunity check with staking-yield carry."""
        strategy = ConversionArbitrage(
            risk_free_rate=0.05,
            staking_yield=0.03,
            min_profit_threshold=0.0,
            transaction_cost=0.0
        )
        expiry = datetime.now(timezone.utc) + timedelta(days=45)
        strategy.update_market_snapshot(
            underlying="ETH",
            call_price=20.0,
            put_price=5.0,
            spot_price=100.0,
            strike=100.0,
            expiry=expiry
        )
        opp = strategy.check_opportunity("ETH")
        assert opp is not None
