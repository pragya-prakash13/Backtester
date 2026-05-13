"""
Order Execution Engine
Handles order processing with realistic slippage and commission models
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from engine.portfolio import Portfolio


@dataclass
class Order:
    """Represents a trading order"""
    ticker: str
    shares: int
    order_type: str  # 'BUY' or 'SELL'
    timestamp: datetime
    
    
class OrderExecutor:
    """
    Executes orders with realistic market friction.
    
    Models:
    - Slippage: 0.05% of price (bid-ask spread proxy)
    - Commission: ₹20 flat per trade (Zerodha-like)
    """
    
    def __init__(
        self,
        slippage_pct: float = 0.0005,  # 0.05%
        commission_flat: float = 20.0   # ₹20 per trade
    ):
        """
        Parameters:
        -----------
        slippage_pct : float
            Percentage of price added/subtracted as slippage
            0.05% is conservative estimate for liquid NSE stocks
        commission_flat : float
            Flat commission per trade in INR
        """
        self.slippage_pct = slippage_pct
        self.commission_flat = commission_flat
        
    def calculate_execution_price(
        self, 
        base_price: float, 
        order_type: str
    ) -> float:
        """
        Calculate actual execution price including slippage.
        
        Buy orders execute slightly higher (you pay the ask).
        Sell orders execute slightly lower (you receive the bid).
        """
        if order_type == "BUY":
            # Pay more when buying (slippage works against you)
            return base_price * (1 + self.slippage_pct)
        else:
            # Receive less when selling
            return base_price * (1 - self.slippage_pct)
    
    def execute_order(
        self,
        order: Order,
        portfolio: Portfolio,
        market_price: float
    ) -> dict:
        """
        Execute an order against the portfolio.
        
        Parameters:
        -----------
        order : Order
            The order to execute
        portfolio : Portfolio
            Portfolio to update
        market_price : float
            Current market price (close price from data)
            
        Returns:
        --------
        dict with execution details
        """
        # Calculate execution price with slippage
        exec_price = self.calculate_execution_price(market_price, order.order_type)
        
        # Execute based on order type
        if order.order_type == "BUY":
            success = portfolio.buy(
                ticker=order.ticker,
                shares=order.shares,
                price=exec_price,
                date=order.timestamp,
                commission=self.commission_flat
            )
        else:  # SELL
            success = portfolio.sell(
                ticker=order.ticker,
                shares=order.shares,
                price=exec_price,
                date=order.timestamp,
                commission=self.commission_flat
            )
        
        return {
            "success": success,
            "ticker": order.ticker,
            "order_type": order.order_type,
            "shares": order.shares,
            "market_price": market_price,
            "execution_price": exec_price,
            "slippage_cost": abs(exec_price - market_price) * order.shares,
            "commission": self.commission_flat if success else 0,
            "timestamp": order.timestamp
        }
    
    def calculate_shares_for_amount(
        self,
        amount: float,
        price: float,
        order_type: str
    ) -> int:
        """
        Calculate how many shares can be bought/sold for a given amount.
        Accounts for slippage and commission.
        """
        exec_price = self.calculate_execution_price(price, order_type)
        
        if order_type == "BUY":
            # Reserve commission from amount
            available = amount - self.commission_flat
            if available <= 0:
                return 0
            return int(available / exec_price)
        else:
            # For sells, just divide by price
            return int(amount / exec_price)


def create_order(
    ticker: str,
    shares: int,
    order_type: str,
    timestamp: datetime
) -> Optional[Order]:
    """Factory function to create valid orders"""
    if shares <= 0:
        return None
    if order_type not in ["BUY", "SELL"]:
        return None
    
    return Order(
        ticker=ticker,
        shares=shares,
        order_type=order_type,
        timestamp=timestamp
    )
