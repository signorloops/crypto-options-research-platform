"""
Crypto option pricing model zoo for benchmark experiments.

This module provides a unified interface for several commonly-used models:
- Black-Scholes
- Merton Jump Diffusion
- Kou-style asymmetric jump approximation
- Heston-style stochastic volatility approximation
- Bates-style SV + jumps approximation
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.volatility.implied import black_scholes_price, implied_volatility

IV_INVERSION_EXCEPTIONS = (
    ValueError,
    TypeError,
    ArithmeticError,
    FloatingPointError,
    OverflowError,
    RuntimeError,
)

MODEL_PRICER_DISPATCH: dict[str, tuple[str, bool]] = {
    "black_scholes": ("_price_black_scholes", False),
    "merton_jump_diffusion": ("_price_merton_jump_diffusion", True),
    "kou_jump": ("_price_kou_approx", True),
    "heston": ("_price_heston_approx", True),
    "bates": ("_price_bates_approx", True),
}


@dataclass
class OptionQuote:
    """Single option quote used by benchmark."""

    spot: float
    strike: float
    maturity: float
    rate: float
    market_price: float
    is_call: bool = True


def _safe_iv_abs_error(quote: OptionQuote, predicted_price: float) -> Optional[float]:
    try:
        market_iv = implied_volatility(
            quote.market_price,
            quote.spot,
            quote.strike,
            quote.maturity,
            quote.rate,
            is_call=quote.is_call,
            method="hybrid",
        )
        pred_iv = implied_volatility(
            predicted_price,
            quote.spot,
            quote.strike,
            quote.maturity,
            quote.rate,
            is_call=quote.is_call,
            method="hybrid",
        )
        if np.isfinite(market_iv) and np.isfinite(pred_iv):
            return abs(float(pred_iv - market_iv))
    except IV_INVERSION_EXCEPTIONS:
        return None
    return None


def _benchmark_row(
    *,
    zoo: "CryptoOptionModelZoo",
    model: str,
    quotes: Sequence[OptionQuote],
    sigma: float,
    model_params: Optional[Dict[str, float]],
) -> Dict[str, float]:
    abs_err, sq_err, iv_abs_err = [], [], []
    for quote in quotes:
        err, iv_err = _benchmark_quote_errors(
            zoo=zoo,
            model=model,
            quote=quote,
            sigma=sigma,
            model_params=model_params,
        )
        abs_err.append(abs(err))
        sq_err.append(err * err)
        if iv_err is not None:
            iv_abs_err.append(iv_err)
    return _benchmark_summary(
        model=model,
        abs_err=abs_err,
        sq_err=sq_err,
        iv_abs_err=iv_abs_err,
    )


def _benchmark_quote_errors(
    *,
    zoo: "CryptoOptionModelZoo",
    model: str,
    quote: OptionQuote,
    sigma: float,
    model_params: Optional[Dict[str, float]],
) -> tuple[float, Optional[float]]:
    predicted = zoo.price_option(
        model=model,
        spot=quote.spot,
        strike=quote.strike,
        maturity=quote.maturity,
        rate=quote.rate,
        sigma=sigma,
        is_call=quote.is_call,
        model_params=model_params,
    )
    err = float(predicted - quote.market_price)
    iv_err = _safe_iv_abs_error(quote, predicted)
    return err, iv_err


def _benchmark_summary(
    *,
    model: str,
    abs_err: list[float],
    sq_err: list[float],
    iv_abs_err: list[float],
) -> Dict[str, float]:
    return {
        "model": model,
        "rmse": float(np.sqrt(np.mean(sq_err))) if sq_err else np.nan,
        "mae": float(np.mean(abs_err)) if abs_err else np.nan,
        "mean_abs_iv_error": float(np.mean(iv_abs_err)) if iv_abs_err else np.nan,
        "n_quotes": float(len(abs_err)),
    }


class CryptoOptionModelZoo:
    """Unified model-pricing and benchmark interface."""

    available_models = (
        "black_scholes",
        "merton_jump_diffusion",
        "kou_jump",
        "heston",
        "bates",
    )

    @staticmethod
    def _safe_price(value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(max(0.0, value))

    @staticmethod
    def _price_black_scholes(
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        sigma: float,
        is_call: bool,
    ) -> float:
        return black_scholes_price(spot, strike, maturity, rate, sigma, is_call=is_call)

    @staticmethod
    def _price_merton_jump_diffusion(
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        sigma: float,
        is_call: bool,
        jump_intensity: float = 1.0,
        jump_mean: float = -0.05,
        jump_std: float = 0.25,
        n_terms: int = 30,
    ) -> float:
        t = max(float(maturity), 1e-8)
        lam = max(float(jump_intensity), 0.0)
        mu_j = float(jump_mean)
        sig_j = max(float(jump_std), 1e-8)

        kappa = np.exp(mu_j + 0.5 * sig_j * sig_j) - 1.0
        lam_t = lam * t

        price = 0.0
        poisson_w = np.exp(-lam_t)
        for n in range(max(int(n_terms), 1)):
            if n > 0:
                poisson_w *= lam_t / n
            r_n = rate - lam * kappa + n * mu_j / t
            sigma_n = np.sqrt(max(sigma * sigma + n * sig_j * sig_j / t, 1e-10))
            price += poisson_w * black_scholes_price(
                spot, strike, t, r_n, sigma_n, is_call=is_call
            )
        return float(price)

    @staticmethod
    def _heston_effective_sigma(
        sigma: float,
        maturity: float,
        spot: float,
        strike: float,
        kappa: float = 1.5,
        theta: float = 0.35,
        v0: float = 0.40,
        rho: float = -0.5,
    ) -> float:
        t = max(float(maturity), 1e-8)
        kappa = max(float(kappa), 1e-6)
        theta = max(float(theta), 1e-8)
        v0 = max(float(v0), 1e-8)
        rho = float(np.clip(rho, -0.999, 0.999))

        avg_var = theta + (v0 - theta) * (1.0 - np.exp(-kappa * t)) / (kappa * t)
        sigma_eff = np.sqrt(max(avg_var, 1e-10))
        # Lightweight skew correction: negative rho steepens downside vol.
        skew_adj = 1.0 + 0.15 * rho * np.log(max(strike / max(spot, 1e-12), 1e-12))
        sigma_eff *= float(np.clip(skew_adj, 0.3, 2.5))
        sigma_eff = 0.5 * sigma_eff + 0.5 * max(float(sigma), 1e-4)
        return float(np.clip(sigma_eff, 0.01, 5.0))

    @staticmethod
    def _price_heston_approx(
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        sigma: float,
        is_call: bool,
        **params: float,
    ) -> float:
        sigma_eff = CryptoOptionModelZoo._heston_effective_sigma(
            sigma=sigma,
            maturity=maturity,
            spot=spot,
            strike=strike,
            kappa=float(params.get("kappa", 1.5)),
            theta=float(params.get("theta", 0.35)),
            v0=float(params.get("v0", 0.40)),
            rho=float(params.get("rho", -0.5)),
        )
        return black_scholes_price(spot, strike, maturity, rate, sigma_eff, is_call=is_call)

    @staticmethod
    def _price_kou_approx(
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        sigma: float,
        is_call: bool,
        jump_intensity: float = 1.0,
        p_up: float = 0.35,
        eta1: float = 12.0,
        eta2: float = 8.0,
        n_terms: int = 30,
    ) -> float:
        p = float(np.clip(p_up, 1e-4, 1 - 1e-4))
        eta1 = max(float(eta1), 1e-4)
        eta2 = max(float(eta2), 1e-4)

        jump_mean = p / eta1 - (1.0 - p) / eta2
        jump_var = 2.0 * p / (eta1 * eta1) + 2.0 * (1.0 - p) / (eta2 * eta2) - jump_mean * jump_mean
        jump_std = np.sqrt(max(jump_var, 1e-8))

        return CryptoOptionModelZoo._price_merton_jump_diffusion(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            sigma=sigma,
            is_call=is_call,
            jump_intensity=float(jump_intensity),
            jump_mean=float(jump_mean),
            jump_std=float(jump_std),
            n_terms=n_terms,
        )

    @staticmethod
    def _price_bates_approx(
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        sigma: float,
        is_call: bool,
        **params: float,
    ) -> float:
        sigma_eff = CryptoOptionModelZoo._heston_effective_sigma(
            sigma=sigma,
            maturity=maturity,
            spot=spot,
            strike=strike,
            kappa=float(params.get("kappa", 1.5)),
            theta=float(params.get("theta", 0.35)),
            v0=float(params.get("v0", 0.40)),
            rho=float(params.get("rho", -0.5)),
        )
        return CryptoOptionModelZoo._price_merton_jump_diffusion(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            sigma=sigma_eff,
            is_call=is_call,
            jump_intensity=float(params.get("jump_intensity", 1.0)),
            jump_mean=float(params.get("jump_mean", -0.05)),
            jump_std=float(params.get("jump_std", 0.25)),
            n_terms=int(params.get("n_terms", 30)),
        )

    def price_option(
        self,
        model: str,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        sigma: float,
        is_call: bool = True,
        model_params: Optional[Dict[str, float]] = None,
    ) -> float:
        """Unified pricing API for all supported models."""
        model_id = str(model).lower()
        params = model_params or {}
        sigma = float(np.clip(sigma, 0.001, 5.0))
        dispatch = MODEL_PRICER_DISPATCH.get(model_id)
        if dispatch is None:
            raise ValueError(f"Unknown model '{model}'")
        method_name, accepts_params = dispatch
        pricer = getattr(self, method_name)
        if accepts_params:
            px = pricer(spot, strike, maturity, rate, sigma, is_call, **params)
        else:
            px = pricer(spot, strike, maturity, rate, sigma, is_call)
        return self._safe_price(px)

    def benchmark(
        self,
        quotes: Sequence[OptionQuote],
        sigma: float,
        model_params_by_model: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> pd.DataFrame:
        """
        Evaluate all models on option quote set and return rmse-sorted ranking.
        """
        if not quotes:
            return pd.DataFrame(
                columns=["model", "rmse", "mae", "mean_abs_iv_error", "n_quotes"]
            )

        model_params_by_model = model_params_by_model or {}
        rows = [
            _benchmark_row(
                zoo=self,
                model=model,
                quotes=quotes,
                sigma=sigma,
                model_params=model_params_by_model.get(model),
            )
            for model in self.available_models
        ]
        table = pd.DataFrame(rows)
        return table.sort_values("rmse", ascending=True, na_position="last").reset_index(drop=True)

    def select_best_model(
        self,
        quotes: Sequence[OptionQuote],
        sigma: float,
        model_params_by_model: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> str:
        """Return best model id by benchmark RMSE."""
        table = self.benchmark(
            quotes=quotes,
            sigma=sigma,
            model_params_by_model=model_params_by_model,
        )
        if table.empty:
            return "black_scholes"
        return str(table.iloc[0]["model"])
