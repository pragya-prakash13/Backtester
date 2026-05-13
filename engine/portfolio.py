"""
Portfolio Engine
Tracks positions, cash, P&L, and equity curve throughout the backtest
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional


class Portfolio:
    """
    Core portfolio tracking class.
    
    Maintains:
    - Cash balance
    - Stock positions (ticker -> shares)
    - Daily equity values
    - Trade history
    """
    
    def __init__(self, initial_cash: float = 1_000_000):
        """
        Initialize portfolio with starting capital.
        
        Parameters:
        -----------
        initial_cash : float
            Starting capital in INR (default: 10 lakh)
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, int] = {}  # ticker -> number of shares
        self.position_prices: Dict[str, float] = {}  # ticker -> avg entry price
        
        # Tracking
        self.equity_curve = []  # List of (date, equity_value)
        self.trade_history = []  # List of trade dicts
        self.daily_returns = []
        
    def get_position_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate total value of all stock positions.
        
        Parameters:
        -----------
        prices : dict
            Current prices {ticker: price}
        """
        value = 0.0
        for ticker, shares in self.positions.items():
            if ticker in prices and shares > 0:
                value += shares * prices[ticker]
        return value
    
    def get_total_equity(self, prices: Dict[str, float]) -> float:
        """Total portfolio value = cash + positions"""
        return self.cash + self.get_position_value(prices)
    
    def update_equity(self, date: datetime, prices: Dict[str, float]):
        """Record daily equity value"""
        equity = self.get_total_equity(prices)
        self.equity_curve.append({
            "date": date,
            "equity": equity,
            "cash": self.cash,
            "positions_value": self.get_position_value(prices)
        })
        
        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]["equity"]
            daily_return = (equity - prev_equity) / prev_equity
            self.daily_returns.append(daily_return)
    
    def can_buy(self, ticker: str, shares: int, price: float, commission: float) -> bool:
        """Check if we have enough cash for this purchase"""
        total_cost = (shares * price) + commission
        return self.cash >= total_cost
    
    def buy(
        self, 
        ticker: str, 
        shares: int, 
        price: float, 
        date: datetime,
        commission: float = 0
    ) -> bool:
        """
        Execute a buy order.
        
        Returns True if successful, False if insufficient cash.
        """
        total_cost = (shares * price) + commission
        
        if self.cash < total_cost:
            return False
        
        # Update cash
        self.cash -= total_cost
        
        # Update position with average price
        if ticker in self.positions:
            # Calculate new average price
            existing_shares = self.positions[ticker]
            existing_value = existing_shares * self.position_prices[ticker]
            new_value = shares * price
            total_shares = existing_shares + shares
            self.position_prices[ticker] = (existing_value + new_value) / total_shares
            self.positions[ticker] = total_shares
        else:
            self.positions[ticker] = shares
            self.position_prices[ticker] = price
        
        # Record trade
        self.trade_history.append({
            "date": date,
            "ticker": ticker,
            "action": "BUY",
            "shares": shares,
            "price": price,
            "commission": commission,
            "total_cost": total_cost
        })
        
        return True
    
    def sell(
        self,
        ticker: str,
        shares: int,
        price: float,
        date: datetime,
        commission: float = 0
    ) -> bool:
        """
        Execute a sell order.
        
        Returns True if successful, False if insufficient shares.
        """
        if ticker not in self.positions or self.positions[ticker] < shares:
            return False
        
        # Calculate proceeds
        proceeds = (shares * price) - commission
        
        # Update cash and position
        self.cash += proceeds
        self.positions[ticker] -= shares
        
        # Clean up empty positions
        if self.positions[ticker] == 0:
            del self.positions[ticker]
            del self.position_prices[ticker]
        
        # Record trade
        self.trade_history.append({
            "date": date,
            "ticker": ticker,
            "action": "SELL",
            "shares": shares,
            "price": price,
            "commission": commission,
            "proceeds": proceeds
        })
        
        return True
    
    def get_equity_series(self) -> pd.Series:
        """Convert equity curve to pandas Series"""
        if not self.equity_curve:
            return pd.Series()
        
        df = pd.DataFrame(self.equity_curve)
        return df.set_index("date")["equity"]
    
    def get_returns_series(self) -> pd.Series:
        """Get daily returns as pandas Series"""
        equity = self.get_equity_series()
        if len(equity) < 2:
            return pd.Series()
        return equity.pct_change().dropna()
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        if not self.trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_history)
    
    def get_summary(self) -> dict:
        """Get portfolio summary statistics"""
        equity_series = self.get_equity_series()
        
        if len(equity_series) == 0:
            return {}
        
        final_equity = equity_series.iloc[-1]
        total_return = (final_equity - self.initial_cash) / self.initial_cash
        
        return {
            "initial_cash": self.initial_cash,
            "final_equity": final_equity,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "num_trades": len(self.trade_history),
            "current_cash": self.cash,
            "current_positions": dict(self.positions)
        }
    
    def reset(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_cash
        self.positions = {}
        self.position_prices = {}
        self.equity_curve = []
        self.trade_history = []
        self.daily_returns = []
