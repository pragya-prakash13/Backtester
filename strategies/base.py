"""
Base Strategy Class
All trading strategies inherit from this class
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict
from engine.portfolio import Portfolio


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Every strategy must implement:
    - initialize(): Called once with full price data
    - generate_signals(): Called each day to produce trading signals
    """
    
    def __init__(self, **params):
        """Store strategy parameters"""
        self.params = params
        self.name = self.__class__.__name__
        
    def initialize(self, price_data: pd.DataFrame):
        """
        Called once before backtest starts.
        Override to precompute indicators, set up state, etc.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            Full historical price data (index=dates, columns=tickers)
        """
        self.price_data = price_data
        self.tickers = list(price_data.columns)
        
    @abstractmethod
    def generate_signals(
        self,
        date: pd.Timestamp,
        current_prices: Dict[str, float],
        historical_data: pd.DataFrame,
        portfolio: Portfolio
    ) -> List[dict]:
        """
        Generate trading signals for current day.
        
        Parameters:
        -----------
        date : pd.Timestamp
            Current date
        current_prices : dict
            Today's prices {ticker: price}
        historical_data : pd.DataFrame
            Price data from start up to and including today
            (never includes future data)
        portfolio : Portfolio
            Current portfolio state
            
        Returns:
        --------
        List of signal dicts, each containing:
        {
            'ticker': str,
            'action': 'BUY' or 'SELL',
            'shares' or 'amount' or 'pct_portfolio': sizing info
        }
        """
        pass
