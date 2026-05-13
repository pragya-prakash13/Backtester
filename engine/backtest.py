"""
Core Backtesting Engine
The heart of the system - processes data day by day, generates signals,
executes trades, and tracks performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Type

from engine.portfolio import Portfolio
from engine.orders import OrderExecutor, create_order


class BacktestEngine:
    """
    Event-driven backtesting engine.
    
    Processes historical data day-by-day to avoid look-ahead bias.
    Each day: generate signals -> validate -> execute -> update equity
    """
    
    def __init__(
        self,
        initial_cash: float = 1_000_000,
        slippage_pct: float = 0.0005,
        commission: float = 20.0
    ):
        self.initial_cash = initial_cash
        self.portfolio = Portfolio(initial_cash)
        self.executor = OrderExecutor(slippage_pct, commission)
        self.strategy = None
        self.results = None
        
    def set_strategy(self, strategy):
        """Set the trading strategy to backtest"""
        self.strategy = strategy
        
    def run(
        self,
        price_data: pd.DataFrame,
        start_date: str = None,
        end_date: str = None
    ) -> dict:
        """
        Run the backtest.
        
        Parameters:
        -----------
        price_data : pd.DataFrame
            DataFrame with dates as index, tickers as columns
            Values should be adjusted close prices
        start_date : str
            Start date for backtest (YYYY-MM-DD)
        end_date : str
            End date for backtest (YYYY-MM-DD)
            
        Returns:
        --------
        dict with backtest results
        """
        if self.strategy is None:
            raise ValueError("No strategy set. Call set_strategy() first.")
        
        # Reset portfolio
        self.portfolio.reset()
        
        # Filter date range
        if start_date:
            price_data = price_data[price_data.index >= start_date]
        if end_date:
            price_data = price_data[price_data.index <= end_date]
            
        if len(price_data) == 0:
            raise ValueError("No data in specified date range")
        
        # Initialize strategy with data
        self.strategy.initialize(price_data)
        
        print(f"Running backtest from {price_data.index[0].date()} to {price_data.index[-1].date()}")
        print(f"Total trading days: {len(price_data)}")
        
        # Main event loop - process day by day
        for i, (date, row) in enumerate(price_data.iterrows()):
            # Current prices as dict
            current_prices = row.dropna().to_dict()
            
            if not current_prices:
                continue
                
            # Get historical data up to (and including) current day
            # This prevents look-ahead bias
            historical_data = price_data.iloc[:i+1]
            
            # Generate signals from strategy
            signals = self.strategy.generate_signals(
                date=date,
                current_prices=current_prices,
                historical_data=historical_data,
                portfolio=self.portfolio
            )
            
            # Execute each signal
            for signal in signals:
                self._execute_signal(signal, current_prices, date)
            
            # Update portfolio equity at end of day
            self.portfolio.update_equity(date, current_prices)
        
        # Compile results
        self.results = self._compile_results(price_data)
        return self.results
    
    def _execute_signal(
        self,
        signal: dict,
        prices: Dict[str, float],
        date: datetime
    ):
        """
        Execute a single trading signal.
        
        Signal format:
        {
            'ticker': str,
            'action': 'BUY' or 'SELL',
            'shares': int (optional, for fixed share orders)
            'amount': float (optional, for amount-based orders)
            'pct_portfolio': float (optional, for portfolio % orders)
        }
        """
        ticker = signal.get("ticker")
        action = signal.get("action")
        
        if ticker not in prices:
            return  # Can't trade - no price data
            
        price = prices[ticker]
        
        # Determine number of shares
        if "shares" in signal:
            shares = signal["shares"]
        elif "amount" in signal:
            shares = self.executor.calculate_shares_for_amount(
                signal["amount"], price, action
            )
        elif "pct_portfolio" in signal:
            equity = self.portfolio.get_total_equity(prices)
            amount = equity * signal["pct_portfolio"]
            shares = self.executor.calculate_shares_for_amount(
                amount, price, action
            )
        else:
            return  # No valid size specification
        
        if shares <= 0:
            return
            
        # Create and execute order
        order = create_order(ticker, shares, action, date)
        if order:
            self.executor.execute_order(order, self.portfolio, price)
    
    def _compile_results(self, price_data: pd.DataFrame) -> dict:
        """Compile backtest results into a summary dict"""
        equity_series = self.portfolio.get_equity_series()
        
        return {
            "equity_curve": equity_series,
            "trade_history": self.portfolio.get_trade_history(),
            "summary": self.portfolio.get_summary(),
            "start_date": price_data.index[0],
            "end_date": price_data.index[-1],
            "strategy_name": self.strategy.__class__.__name__,
            "initial_cash": self.initial_cash
        }
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve from last backtest run"""
        return self.portfolio.get_equity_series()
    
    def get_trades(self) -> pd.DataFrame:
        """Get trade history from last backtest run"""
        return self.portfolio.get_trade_history()
