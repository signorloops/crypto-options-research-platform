"""
Base strategy interface for market making and other trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd

from core.types import MarketState, Position, QuoteAction


class MarketMakingStrategy(ABC):
    """
    Abstract base class for market making strategies.

    All market making strategies must implement:
    - quote(): Generate bid/ask quotes based on market state and position
    - get_internal_state(): Return interpretable state for debugging
    """

    def __init__(self):
        self.name = "BaseStrategy"
        self._training_data: Optional[pd.DataFrame] = None

    @abstractmethod
    def quote(self, state: MarketState, position: Position) -> QuoteAction:
        """
        Generate quotes based on current market state and position.

        Args:
            state: Current market state including order book, trades, etc.
            position: Current position in the instrument

        Returns:
            QuoteAction with bid/ask prices and sizes
        """
        pass

    @abstractmethod
    def get_internal_state(self) -> Dict:
        """
        Return current internal state for debugging and logging.

        Returns:
            Dictionary of interpretable state variables
        """
        pass

    def train(self, historical_data: pd.DataFrame) -> None:
        """
        Train strategy on historical data (optional).
        Base implementation does nothing. ML/RL strategies override.

        Args:
            historical_data: DataFrame with market data for training
        """
        self._training_data = historical_data
        pass

    def on_fill(self, fill, position: Position) -> None:
        """
        Callback when an order is filled.
        Strategies can use this to update internal state.

        Args:
            fill: Fill details
            position: Updated position after fill
        """
        pass

    def reset(self) -> None:
        """Reset strategy state for new episode/backtest."""
        pass
