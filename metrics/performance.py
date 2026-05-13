"""
Performance Metrics
Calculates professional-grade risk and return metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.065,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe Ratio.
    
    Sharpe = (Mean Excess Return) / (Std Dev of Returns) * sqrt(252)
    
    Measures risk-adjusted return. Higher is better.
    - < 1.0: Suboptimal
    - 1.0-2.0: Good
    - 2.0-3.0: Very good
    - > 3.0: Excellent (verify it's real)
    
    Parameters:
    -----------
    returns : pd.Series
        Daily returns
    risk_free_rate : float
        Annual risk-free rate (default 6.5% for India)
    periods_per_year : int
        Trading days per year (252)
    """
    if len(returns) < 2:
        return 0.0
        
    # Daily risk-free rate
    daily_rf = risk_free_rate / periods_per_year
    
    # Excess returns
    excess_returns = returns - daily_rf
    
    # Annualized Sharpe
    if excess_returns.std() == 0:
        return 0.0
        
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)
    
    return sharpe


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.065,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino Ratio.
    
    Like Sharpe, but only penalizes downside volatility.
    Uses only negative returns in denominator.
    
    Sortino = (Mean Excess Return) / (Downside Std Dev) * sqrt(252)
    """
    if len(returns) < 2:
        return 0.0
        
    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf
    
    # Only consider negative returns for downside deviation
    negative_returns = returns[returns < 0]
    
    if len(negative_returns) == 0 or negative_returns.std() == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
        
    downside_std = negative_returns.std()
    sortino = (excess_returns.mean() / downside_std) * np.sqrt(periods_per_year)
    
    return sortino


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate Maximum Drawdown.
    
    Max Drawdown = (Trough - Peak) / Peak
    
    Measures worst peak-to-trough decline.
    Returns negative number (e.g., -0.20 = 20% drawdown).
    
    Key risk metric - how much could you lose from peak?
    """
    if len(equity_curve) < 2:
        return 0.0
        
    # Running maximum
    rolling_max = equity_curve.cummax()
    
    # Drawdown at each point
    drawdown = (equity_curve - rolling_max) / rolling_max
    
    return drawdown.min()


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Calculate drawdown at each point in time"""
    rolling_max = equity_curve.cummax()
    return (equity_curve - rolling_max) / rolling_max


def cagr(
    equity_curve: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    CAGR = (Final / Initial)^(1/years) - 1
    
    Annualized return that accounts for compounding.
    """
    if len(equity_curve) < 2:
        return 0.0
        
    initial = equity_curve.iloc[0]
    final = equity_curve.iloc[-1]
    
    if initial <= 0:
        return 0.0
    
    # Calculate years
    num_periods = len(equity_curve)
    years = num_periods / periods_per_year
    
    if years <= 0:
        return 0.0
    
    # CAGR formula
    cagr_value = (final / initial) ** (1 / years) - 1
    
    return cagr_value


def calmar_ratio(
    equity_curve: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar Ratio.
    
    Calmar = CAGR / |Max Drawdown|
    
    Risk-adjusted return using drawdown as risk measure.
    Higher is better - want high returns relative to worst loss.
    """
    cagr_val = cagr(equity_curve, periods_per_year)
    max_dd = max_drawdown(equity_curve)
    
    if max_dd == 0:
        return float('inf') if cagr_val > 0 else 0.0
        
    return cagr_val / abs(max_dd)


def win_rate(trades: pd.DataFrame) -> float:
    """
    Calculate win rate from trade history.
    
    Win Rate = Winning Trades / Total Trades
    
    Note: Requires paired buy/sell trades to calculate P&L.
    """
    if trades.empty or len(trades) < 2:
        return 0.0
    
    # Group trades by ticker to find completed round trips
    wins = 0
    total = 0
    
    for ticker in trades['ticker'].unique():
        ticker_trades = trades[trades['ticker'] == ticker].sort_values('date')
        
        buy_price = None
        for _, trade in ticker_trades.iterrows():
            if trade['action'] == 'BUY':
                buy_price = trade['price']
            elif trade['action'] == 'SELL' and buy_price is not None:
                if trade['price'] > buy_price:
                    wins += 1
                total += 1
                buy_price = None
    
    return wins / total if total > 0 else 0.0


def avg_win_loss_ratio(trades: pd.DataFrame) -> float:
    """
    Calculate average win / average loss ratio.
    
    Also called reward-to-risk ratio.
    > 1.0 means wins are bigger than losses on average.
    """
    if trades.empty:
        return 0.0
    
    wins = []
    losses = []
    
    for ticker in trades['ticker'].unique():
        ticker_trades = trades[trades['ticker'] == ticker].sort_values('date')
        
        buy_price = None
        for _, trade in ticker_trades.iterrows():
            if trade['action'] == 'BUY':
                buy_price = trade['price']
            elif trade['action'] == 'SELL' and buy_price is not None:
                pnl_pct = (trade['price'] - buy_price) / buy_price
                if pnl_pct > 0:
                    wins.append(pnl_pct)
                else:
                    losses.append(abs(pnl_pct))
                buy_price = None
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 1
    
    if avg_loss == 0:
        return float('inf') if avg_win > 0 else 0.0
        
    return avg_win / avg_loss


def calculate_all_metrics(
    equity_curve: pd.Series,
    trades: pd.DataFrame = None,
    risk_free_rate: float = 0.065,
    benchmark_curve: pd.Series = None
) -> Dict:
    """
    Calculate all performance metrics.
    
    Returns dict with all metrics for display.
    """
    returns = equity_curve.pct_change().dropna()
    
    metrics = {
        "CAGR": f"{cagr(equity_curve) * 100:.2f}%",
        "Total Return": f"{((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1) * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio(returns, risk_free_rate):.2f}",
        "Sortino Ratio": f"{sortino_ratio(returns, risk_free_rate):.2f}",
        "Max Drawdown": f"{max_drawdown(equity_curve) * 100:.2f}%",
        "Calmar Ratio": f"{calmar_ratio(equity_curve):.2f}",
        "Volatility (Ann.)": f"{returns.std() * np.sqrt(252) * 100:.2f}%",
    }
    
    if trades is not None and not trades.empty:
        metrics["Total Trades"] = len(trades)
        metrics["Win Rate"] = f"{win_rate(trades) * 100:.1f}%"
        metrics["Avg Win/Loss"] = f"{avg_win_loss_ratio(trades):.2f}"
    
    if benchmark_curve is not None and len(benchmark_curve) > 0:
        bench_returns = benchmark_curve.pct_change().dropna()
        
        # Align returns
        common_idx = returns.index.intersection(bench_returns.index)
        if len(common_idx) > 0:
            aligned_returns = returns.loc[common_idx]
            aligned_bench = bench_returns.loc[common_idx]
            
            # Alpha and Beta
            covariance = aligned_returns.cov(aligned_bench)
            bench_variance = aligned_bench.var()
            
            if bench_variance > 0:
                beta = covariance / bench_variance
                alpha = (aligned_returns.mean() - beta * aligned_bench.mean()) * 252
                metrics["Beta"] = f"{beta:.2f}"
                metrics["Alpha (Ann.)"] = f"{alpha * 100:.2f}%"
            
            metrics["Benchmark CAGR"] = f"{cagr(benchmark_curve) * 100:.2f}%"
    
    return metrics
