"""Hedging helpers for quanto-inverse option exposures."""
from dataclasses import dataclass

from research.pricing.quanto_inverse import QuantoInverseGreeks


@dataclass
class QuantoInverseHedgePlan:
    """Recommended hedge quantities for spot and FX legs."""

    spot_hedge_units: float
    fx_hedge_units: float
    spot_notional_usd: float
    fx_notional_settlement: float
    residual_spot_delta: float
    residual_fx_delta: float


class QuantoInverseHedger:
    """Translate quanto Greeks into executable hedge quantities."""

    @staticmethod
    def _round_to_lot(quantity: float, lot_size: float) -> float:
        if lot_size <= 0:
            return quantity
        return round(quantity / lot_size) * lot_size

    @staticmethod
    def build_hedge_plan(
        greeks: QuantoInverseGreeks,
        position_size: float,
        spot_price: float,
        fx_rate: float,
        spot_lot_size: float = 1e-4,
        fx_lot_size: float = 1e-4,
    ) -> QuantoInverseHedgePlan:
        """Build hedge plan for position-level spot and FX deltas.

        Args:
            greeks: Per-contract quanto greeks.
            position_size: Number of option contracts (signed).
            spot_price: Underlying spot for notional conversion.
            fx_rate: Settlement FX conversion rate.
            spot_lot_size: Execution lot size for spot hedge.
            fx_lot_size: Execution lot size for FX hedge.
        """
        if fx_rate <= 0:
            raise ValueError("fx_rate must be positive")
        if spot_price <= 0:
            raise ValueError("spot_price must be positive")

        net_spot_delta = greeks.delta * position_size
        net_fx_delta = greeks.fx_delta * position_size

        raw_spot_hedge = -net_spot_delta
        raw_fx_hedge = -net_fx_delta

        spot_hedge = QuantoInverseHedger._round_to_lot(raw_spot_hedge, spot_lot_size)
        fx_hedge = QuantoInverseHedger._round_to_lot(raw_fx_hedge, fx_lot_size)

        residual_spot = net_spot_delta + spot_hedge
        residual_fx = net_fx_delta + fx_hedge

        return QuantoInverseHedgePlan(
            spot_hedge_units=float(spot_hedge),
            fx_hedge_units=float(fx_hedge),
            spot_notional_usd=float(abs(spot_hedge) * spot_price),
            fx_notional_settlement=float(abs(fx_hedge) * fx_rate),
            residual_spot_delta=float(residual_spot),
            residual_fx_delta=float(residual_fx),
        )
