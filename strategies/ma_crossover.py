"""
Moving Average Crossover Strategy

Classic trend-following strategy:
- BUY when fast MA crosses above slow MA (golden cross)
- SELL when fast MA crosses below slow MA (death cross)
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from strategies.base import BaseStrategy
from engine.portfolio import Portfolio


class MACrossoverStrategy(BaseStrategy):
    """
    50/200 Day Moving Average Crossover Strategy
    
    Entry: Fast MA crosses above slow MA
    Exit: Fast MA crosses below slow MA
    
    Position sizing: Equal weight across selected stocks
    """
    
    def __init__(
        self,
        fast_period: int = 50,
        slow_period: int = 200,
        max_positions: int = 10,
        position_size_pct: float = 0.10  # 10% per position
    ):
        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            max_positions=max_positions,
            position_size_pct=position_size_pct
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        
        # Track crossover state
        self.prev_positions = {}  # ticker -> True if fast > slow yesterday
        
    def initialize(self, price_data: pd.DataFrame):
        """Precompute moving averages"""
        super().initialize(price_data)
        
        # Calculate MAs for all stocks
        self.fast_ma = price_data.rolling(window=self.fast_period).mean()
        self.slow_ma = price_data.rolling(window=self.slow_period).mean()
        
    def generate_signals(
        self,
        date: pd.Timestamp,
        current_prices: Dict[str, float],
        historical_data: pd.DataFrame,
        portfolio: Portfolio
    ) -> List[dict]:
        """Generate buy/sell signals based on MA crossovers"""
        signals = []
        
        # Need enough data for slow MA
        if len(historical_data) < self.slow_period:
            return signals
            
        # Get today's MA values
        try:
            fast_today = self.fast_ma.loc[date]
            slow_today = self.slow_ma.loc[date]
        except KeyError:
            return signals
        
        # Get yesterday's MA values for crossover detection
        if len(historical_data) < 2:
            return signals
            
        prev_date = historical_data.index[-2]
        try:
            fast_prev = self.fast_ma.loc[prev_date]
            slow_prev = self.slow_ma.loc[prev_date]
        except KeyError:
            return signals
        
        current_positions = set(portfolio.positions.keys())
        
        for ticker in self.tickers:
            if ticker not in current_prices:
                continue
                
            # Skip if MA values are NaN
            if pd.isna(fast_today[ticker]) or pd.isna(slow_today[ticker]):
                continue
            if pd.isna(fast_prev[ticker]) or pd.isna(slow_prev[ticker]):
                continue
            
            # Detect crossover
            was_above = fast_prev[ticker] > slow_prev[ticker]
            is_above = fast_today[ticker] > slow_today[ticker]
            
            # Golden cross - BUY signal
            if is_above and not was_above:
                if ticker not in current_positions:
                    if len(current_positions) < self.max_positions:
                        signals.append({
                            "ticker": ticker,
                            "action": "BUY",
                            "pct_portfolio": self.position_size_pct
                        })
                        current_positions.add(ticker)
            
            # Death cross - SELL signal
            elif was_above and not is_above:
                if ticker in current_positions:
                    shares = portfolio.positions.get(ticker, 0)
                    if shares > 0:
                        signals.append({
                            "ticker": ticker,
                            "action": "SELL",
                            "shares": shares
                        })
                        current_positions.discard(ticker)
        
        return signals
