"""
Naive market making strategy - baseline for comparison.
Simple symmetric quotes around mid price with fixed spread.
"""
import os
from dataclasses import dataclass, field
from typing import Dict

from core.types import MarketState, Position, QuoteAction
from strategies.base import MarketMakingStrategy


@dataclass
class NaiveMMConfig:
    """Configuration for naive market maker."""
    spread_bps: float = field(default_factory=lambda: float(os.getenv("MM_SPREAD_BPS", "20.0")))
    quote_size: float = field(default_factory=lambda: float(os.getenv("MM_QUOTE_SIZE", "1.0")))
    inventory_limit: float = field(default_factory=lambda: float(os.getenv("MM_INVENTORY_LIMIT", "10.0")))


class NaiveMarketMaker(MarketMakingStrategy):
    """
    Naive market maker with fixed spread and no inventory management.
    Serves as a baseline for comparing more sophisticated strategies.
    """

    def __init__(self, config: NaiveMMConfig = None):
        self.config = config or NaiveMMConfig()
        self.name = "NaiveMM"

    def quote(self, state: MarketState, position: Position) -> QuoteAction:
        """
        Generate symmetric quote around mid price.

        Args:
            state: Current market state
            position: Current position

        Returns:
            Quote action with bid/ask prices and sizes
        """
        mid = state.order_book.mid_price
        if mid is None:
            raise ValueError("Cannot quote without valid order book")

        # Fixed spread
        half_spread = mid * self.config.spread_bps / 10000 / 2

        bid_price = mid - half_spread
        ask_price = mid + half_spread

        # Check inventory limits (account for quote size to avoid exceeding limit)
        bid_size = self.config.quote_size
        ask_size = self.config.quote_size

        if position.size + self.config.quote_size > self.config.inventory_limit:
            # Would exceed long limit
            bid_size = 0
        if position.size - self.config.quote_size < -self.config.inventory_limit:
            # Would exceed short limit
            ask_size = 0

        return QuoteAction(
            bid_price=bid_price,
            bid_size=bid_size,
            ask_price=ask_price,
            ask_size=ask_size,
            metadata={
                "strategy": self.name,
                "mid_price": mid,
                "half_spread": half_spread,
                "inventory": position.size,
                "inventory_limit_hit": abs(position.size) >= self.config.inventory_limit
            }
        )

    def get_internal_state(self) -> Dict:
        """Return current configuration."""
        return {
            "spread_bps": self.config.spread_bps,
            "quote_size": self.config.quote_size,
            "inventory_limit": self.config.inventory_limit
        }
