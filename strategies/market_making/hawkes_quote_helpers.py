from __future__ import annotations
from typing import Dict, Optional

def timestamp_seconds(timestamp: object) -> float:
    return timestamp.timestamp() if hasattr(timestamp, "timestamp") else 0.0

def price_change_ratio(mid: float, last_price: Optional[float]) -> float:
    if last_price is None or last_price <= 0:
        return 0.0
    return (mid - last_price) / last_price

def select_spread_bps(
    *,
    adverse_selection: bool,
    max_spread_bps: float,
    dynamic_spread_bps: float,
) -> float:
    if adverse_selection:
        return float(max_spread_bps * 0.8)
    return float(dynamic_spread_bps)

def build_control_signals(
    *,
    intensity: float,
    buy_intensity: float,
    sell_intensity: float,
    flow_imbalance: float,
    adverse_selection: bool,
    spread_bps: float,
    skew: float,
) -> Dict[str, float]:
    return {
        "intensity": float(intensity),
        "buy_intensity": float(buy_intensity),
        "sell_intensity": float(sell_intensity),
        "flow_imbalance": float(flow_imbalance),
        "adverse_selection": bool(adverse_selection),
        "spread_bps": float(spread_bps),
        "skew_signal": float(skew),
    }

def build_quote_metadata(
    *,
    strategy_name: str,
    control_signals: Dict[str, float],
    mu: float,
    alpha: float,
    beta: float,
) -> Dict[str, object]:
    return {
        "strategy": strategy_name,
        "control_signals": control_signals,
        "hawkes_params": {
            "mu": float(mu),
            "alpha": float(alpha),
            "beta": float(beta),
        },
    }
