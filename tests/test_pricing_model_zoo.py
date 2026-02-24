"""
Tests for crypto option pricing model zoo.
"""
import numpy as np

from research.pricing.model_zoo import CryptoOptionModelZoo, OptionQuote
from research.volatility.implied import black_scholes_price


class TestCryptoOptionModelZoo:
    """Test model zoo pricing and benchmarking."""

    def test_all_models_produce_finite_non_negative_prices(self):
        """Each model should produce a valid option price."""
        zoo = CryptoOptionModelZoo()
        quote = OptionQuote(
            spot=50000.0,
            strike=52000.0,
            maturity=30.0 / 365.0,
            rate=0.03,
            market_price=1200.0,
            is_call=True,
        )

        for model in zoo.available_models:
            price = zoo.price_option(
                model=model,
                spot=quote.spot,
                strike=quote.strike,
                maturity=quote.maturity,
                rate=quote.rate,
                sigma=0.65,
                is_call=quote.is_call,
            )
            assert np.isfinite(price)
            assert price >= 0.0

    def test_benchmark_returns_sorted_results(self):
        """Benchmark should return rmse-sorted model ranking."""
        zoo = CryptoOptionModelZoo()
        spot = 50000.0
        rate = 0.02
        sigma = 0.6
        maturities = [7.0 / 365.0, 30.0 / 365.0, 90.0 / 365.0]
        strikes = [45000.0, 50000.0, 55000.0]

        quotes = []
        for t in maturities:
            for k in strikes:
                px = black_scholes_price(spot, k, t, rate, sigma, is_call=True)
                quotes.append(
                    OptionQuote(
                        spot=spot,
                        strike=k,
                        maturity=t,
                        rate=rate,
                        market_price=px,
                        is_call=True,
                    )
                )

        table = zoo.benchmark(quotes, sigma=0.6)
        assert not table.empty
        assert "model" in table.columns
        assert "rmse" in table.columns
        assert table["rmse"].iloc[0] <= table["rmse"].iloc[-1]

    def test_select_best_model_returns_known_model(self):
        """Best model selector should return one of available model ids."""
        zoo = CryptoOptionModelZoo()
        quotes = [
            OptionQuote(
                spot=50000.0,
                strike=50000.0,
                maturity=14.0 / 365.0,
                rate=0.01,
                market_price=black_scholes_price(50000.0, 50000.0, 14.0 / 365.0, 0.01, 0.55, True),
                is_call=True,
            ),
            OptionQuote(
                spot=50000.0,
                strike=53000.0,
                maturity=60.0 / 365.0,
                rate=0.01,
                market_price=black_scholes_price(50000.0, 53000.0, 60.0 / 365.0, 0.01, 0.55, True),
                is_call=True,
            ),
        ]

        best = zoo.select_best_model(quotes, sigma=0.55)
        assert best in zoo.available_models
