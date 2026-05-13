"""
12-1 Month Momentum Strategy

Classic momentum factor strategy used in academic research:
- Rank stocks by 12-month return, skipping the most recent month
- Buy top performers, sell bottom performers
- Skip last month to avoid short-term reversal
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from strategies.base import BaseStrategy
from engine.portfolio import Portfolio


class MomentumStrategy(BaseStrategy):
    """
    12-1 Month Momentum Strategy
    
    Logic:
    - Calculate 12-month return for each stock
    - Skip the most recent month (avoids short-term reversal)
    - Buy top N stocks by momentum
    - Rebalance monthly
    
    This is the classic "winners keep winning" factor strategy.
    """
    
    def __init__(
        self,
        lookback_months: int = 12,
        skip_months: int = 1,
        top_n: int = 10,
        rebalance_days: int = 21,  # ~1 month of trading days
        position_size_pct: float = 0.10
    ):
        super().__init__(
            lookback_months=lookback_months,
            skip_months=skip_months,
            top_n=top_n,
            rebalance_days=rebalance_days,
            position_size_pct=position_size_pct
        )
        self.lookback_days = lookback_months * 21  # ~21 trading days per month
        self.skip_days = skip_months * 21
        self.top_n = top_n
        self.rebalance_days = rebalance_days
        self.position_size_pct = position_size_pct
        
        self.days_since_rebalance = 0
        self.current_holdings = set()
        
    def initialize(self, price_data: pd.DataFrame):
        """Initialize strategy"""
        super().initialize(price_data)
        self.days_since_rebalance = self.rebalance_days  # Force rebalance on first day
        
    def calculate_momentum(self, historical_data: pd.DataFrame) -> pd.Series:
        """
        Calculate 12-1 momentum score for each stock.
        
        Formula: (Price_t-skip / Price_t-lookback) - 1
        
        This gives the return over the lookback period, excluding recent months.
        """
        if len(historical_data) < self.lookback_days:
            return pd.Series()
            
        # Price today (minus skip period)
        end_idx = -self.skip_days if self.skip_days > 0 else len(historical_data)
        if end_idx <= 0:
            end_idx = len(historical_data)
            
        price_end = historical_data.iloc[min(end_idx, len(historical_data)-1)]
        
        # Price at start of lookback
        start_idx = end_idx - self.lookback_days
        if start_idx < 0:
            return pd.Series()
            
        price_start = historical_data.iloc[start_idx]
        
        # Calculate momentum (return)
        momentum = (price_end / price_start) - 1
        
        return momentum.dropna()
        
    def generate_signals(
        self,
        date: pd.Timestamp,
        current_prices: Dict[str, float],
        historical_data: pd.DataFrame,
        portfolio: Portfolio
    ) -> List[dict]:
        """Generate signals - rebalance monthly"""
        signals = []
        
        self.days_since_rebalance += 1
        
        # Only rebalance on schedule
        if self.days_since_rebalance < self.rebalance_days:
            return signals
            
        self.days_since_rebalance = 0
        
        # Need enough history
        if len(historical_data) < self.lookback_days:
            return signals
        
        # Calculate momentum scores
        momentum = self.calculate_momentum(historical_data)
        
        if len(momentum) == 0:
            return signals
        
        # Rank and get top N
        momentum_ranked = momentum.sort_values(ascending=False)
        
        # Filter to stocks we have price data for
        momentum_ranked = momentum_ranked[
            momentum_ranked.index.isin(current_prices.keys())
        ]
        
        new_holdings = set(momentum_ranked.head(self.top_n).index)
        current_positions = set(portfolio.positions.keys())
        
        # Sell positions no longer in top N
        for ticker in current_positions:
            if ticker not in new_holdings:
                shares = portfolio.positions.get(ticker, 0)
                if shares > 0:
                    signals.append({
                        "ticker": ticker,
                        "action": "SELL",
                        "shares": shares
                    })
        
        # Buy new positions
        for ticker in new_holdings:
            if ticker not in current_positions:
                signals.append({
                    "ticker": ticker,
                    "action": "BUY",
                    "pct_portfolio": self.position_size_pct
                })
        
        self.current_holdings = new_holdings
        return signals
