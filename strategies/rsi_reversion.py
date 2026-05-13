"""
RSI Mean Reversion Strategy

Contrarian strategy based on Relative Strength Index:
- BUY when RSI < 30 (oversold)
- SELL when RSI > 70 (overbought)
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from strategies.base import BaseStrategy
from engine.portfolio import Portfolio


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    
    RSI ranges from 0 to 100:
    - RSI < 30: Oversold (potential buy)
    - RSI > 70: Overbought (potential sell)
    """
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)
    
    # Calculate average gain/loss using exponential moving average
    avg_gain = gains.ewm(span=period, adjust=False).mean()
    avg_loss = losses.ewm(span=period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


class RSIReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy
    
    Entry: RSI < oversold_threshold (default 30)
    Exit: RSI > overbought_threshold (default 70)
    
    This is a contrarian strategy - buys weakness, sells strength
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30,
        overbought_threshold: float = 70,
        max_positions: int = 10,
        position_size_pct: float = 0.10
    ):
        super().__init__(
            rsi_period=rsi_period,
            oversold_threshold=oversold_threshold,
            overbought_threshold=overbought_threshold,
            max_positions=max_positions,
            position_size_pct=position_size_pct
        )
        self.rsi_period = rsi_period
        self.oversold = oversold_threshold
        self.overbought = overbought_threshold
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        
    def initialize(self, price_data: pd.DataFrame):
        """Precompute RSI for all stocks"""
        super().initialize(price_data)
        
        # Calculate RSI for each stock
        self.rsi = pd.DataFrame(index=price_data.index)
        for ticker in price_data.columns:
            self.rsi[ticker] = calculate_rsi(price_data[ticker], self.rsi_period)
            
    def generate_signals(
        self,
        date: pd.Timestamp,
        current_prices: Dict[str, float],
        historical_data: pd.DataFrame,
        portfolio: Portfolio
    ) -> List[dict]:
        """Generate signals based on RSI levels"""
        signals = []
        
        # Need enough data for RSI calculation
        if len(historical_data) < self.rsi_period + 1:
            return signals
            
        # Get today's RSI values
        try:
            rsi_today = self.rsi.loc[date]
        except KeyError:
            return signals
            
        current_positions = set(portfolio.positions.keys())
        
        # Score stocks by how oversold they are
        buy_candidates = []
        
        for ticker in self.tickers:
            if ticker not in current_prices:
                continue
                
            rsi_value = rsi_today.get(ticker)
            if pd.isna(rsi_value):
                continue
            
            # Check for sell signals first (overbought positions)
            if ticker in current_positions:
                if rsi_value > self.overbought:
                    shares = portfolio.positions.get(ticker, 0)
                    if shares > 0:
                        signals.append({
                            "ticker": ticker,
                            "action": "SELL",
                            "shares": shares
                        })
                        current_positions.discard(ticker)
            
            # Check for buy candidates (oversold stocks)
            else:
                if rsi_value < self.oversold:
                    buy_candidates.append((ticker, rsi_value))
        
        # Sort by most oversold (lowest RSI) and limit positions
        buy_candidates.sort(key=lambda x: x[1])
        
        available_slots = self.max_positions - len(current_positions)
        for ticker, _ in buy_candidates[:available_slots]:
            signals.append({
                "ticker": ticker,
                "action": "BUY",
                "pct_portfolio": self.position_size_pct
            })
        
        return signals
