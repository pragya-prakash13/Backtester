"""
Streamlit Dashboard for Backtesting Engine
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from data.fetcher import DataFetcher, NIFTY50_SYMBOLS
from engine.backtest import BacktestEngine
from strategies.ma_crossover import MACrossoverStrategy
from strategies.rsi_reversion import RSIReversionStrategy
from strategies.momentum import MomentumStrategy
from metrics.performance import calculate_all_metrics, drawdown_series


# Page config
st.set_page_config(
    page_title="Algorithmic Trading Backtester",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Algorithmic Trading Backtester")
st.markdown("*Built from scratch - no Backtrader*")

# Sidebar - Strategy Selection and Parameters
st.sidebar.header("⚙️ Configuration")

# Strategy selector
strategy_name = st.sidebar.selectbox(
    "Select Strategy",
    ["MA Crossover (50/200)", "RSI Mean Reversion", "12-1 Momentum"]
)

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=3*365),
        min_value=datetime(2015, 1, 1),
        max_value=datetime.now()
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        min_value=datetime(2015, 1, 1),
        max_value=datetime.now()
    )

# Capital
initial_capital = st.sidebar.number_input(
    "Initial Capital (₹)",
    min_value=100000,
    max_value=100000000,
    value=1000000,
    step=100000
)

# Strategy-specific parameters
st.sidebar.subheader("Strategy Parameters")

if strategy_name == "MA Crossover (50/200)":
    fast_ma = st.sidebar.slider("Fast MA Period", 10, 100, 50)
    slow_ma = st.sidebar.slider("Slow MA Period", 50, 300, 200)
    max_pos = st.sidebar.slider("Max Positions", 1, 20, 10)
    
elif strategy_name == "RSI Mean Reversion":
    rsi_period = st.sidebar.slider("RSI Period", 7, 28, 14)
    oversold = st.sidebar.slider("Oversold Threshold", 10, 40, 30)
    overbought = st.sidebar.slider("Overbought Threshold", 60, 90, 70)
    max_pos = st.sidebar.slider("Max Positions", 1, 20, 10)
    
else:  # Momentum
    lookback = st.sidebar.slider("Lookback Months", 6, 18, 12)
    skip = st.sidebar.slider("Skip Months", 0, 3, 1)
    top_n = st.sidebar.slider("Top N Stocks", 5, 20, 10)

# Run button
run_backtest = st.sidebar.button("🚀 Run Backtest", type="primary")


@st.cache_data(ttl=3600)
def load_data(start_str, end_str):
    """Load and cache price data"""
    fetcher = DataFetcher()
    
    # Fetch stock data
    stock_data = fetcher.fetch_multiple_stocks(
        symbols=NIFTY50_SYMBOLS[:30],  # Use 30 stocks for speed
        start_date=start_str,
        end_date=end_str,
        use_cache=True
    )
    
    # Get adjusted prices
    prices = fetcher.get_adjusted_prices(stock_data)
    
    # Fetch benchmark
    benchmark = fetcher.fetch_benchmark(
        start_date=start_str,
        end_date=end_str,
        use_cache=True
    )
    
    if not benchmark.empty:
        benchmark_prices = benchmark["Adj Close"]
    else:
        benchmark_prices = pd.Series()
    
    return prices, benchmark_prices


if run_backtest:
    with st.spinner("Loading data..."):
        prices, benchmark = load_data(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
    
    if prices.empty:
        st.error("No price data available for selected date range")
    else:
        # Initialize strategy
        if strategy_name == "MA Crossover (50/200)":
            strategy = MACrossoverStrategy(
                fast_period=fast_ma,
                slow_period=slow_ma,
                max_positions=max_pos
            )
        elif strategy_name == "RSI Mean Reversion":
            strategy = RSIReversionStrategy(
                rsi_period=rsi_period,
                oversold_threshold=oversold,
                overbought_threshold=overbought,
                max_positions=max_pos
            )
        else:
            strategy = MomentumStrategy(
                lookback_months=lookback,
                skip_months=skip,
                top_n=top_n
            )
        
        # Run backtest
        with st.spinner(f"Running {strategy_name} backtest..."):
            engine = BacktestEngine(initial_cash=initial_capital)
            engine.set_strategy(strategy)
            results = engine.run(prices)
        
        equity = results["equity_curve"]
        trades = results["trade_history"]
        
        # Normalize benchmark to same starting value
        if not benchmark.empty:
            benchmark_aligned = benchmark.loc[equity.index[0]:equity.index[-1]]
            if len(benchmark_aligned) > 0:
                benchmark_normalized = benchmark_aligned / benchmark_aligned.iloc[0] * initial_capital
            else:
                benchmark_normalized = pd.Series()
        else:
            benchmark_normalized = pd.Series()
        
        # Layout: 2 columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Equity Curve Chart
            st.subheader("📊 Equity Curve vs Benchmark")
            
            fig = go.Figure()
            
            # Strategy equity
            fig.add_trace(go.Scatter(
                x=equity.index,
                y=equity.values,
                name=strategy_name,
                line=dict(color='#2E86AB', width=2)
            ))
            
            # Benchmark
            if not benchmark_normalized.empty:
                fig.add_trace(go.Scatter(
                    x=benchmark_normalized.index,
                    y=benchmark_normalized.values,
                    name='Nifty 50',
                    line=dict(color='#E94F37', width=2, dash='dash')
                ))
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Portfolio Value (₹)",
                hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown Chart
            st.subheader("📉 Drawdown")
            
            dd = drawdown_series(equity) * 100
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd.index,
                y=dd.values,
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.3)',
                line=dict(color='#E74C3C', width=1),
                name='Drawdown'
            ))
            
            fig_dd.update_layout(
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_dd, use_container_width=True)
        
        with col2:
            # Metrics Table
            st.subheader("📋 Performance Metrics")
            
            metrics = calculate_all_metrics(
                equity,
                trades,
                benchmark_curve=benchmark_normalized if not benchmark_normalized.empty else None
            )
            
            # Display as formatted table
            for metric, value in metrics.items():
                st.metric(metric, value)
            
            # Trade History
            st.subheader("📜 Recent Trades")
            
            if not trades.empty:
                display_trades = trades.tail(10)[["date", "ticker", "action", "shares", "price"]]
                display_trades["price"] = display_trades["price"].round(2)
                st.dataframe(display_trades, hide_index=True)
            else:
                st.info("No trades executed")

else:
    # Instructions
    st.info("👈 Configure your strategy and click **Run Backtest** to start")
    
    st.markdown("""
    ### Available Strategies
    
    1. **MA Crossover (50/200)**: Classic trend-following. Buys on golden cross, sells on death cross.
    
    2. **RSI Mean Reversion**: Contrarian approach. Buys oversold stocks (RSI < 30), sells overbought (RSI > 70).
    
    3. **12-1 Momentum**: Academic factor strategy. Buys past winners, avoids recent month to skip reversal.
    
    ---
    
    ### Features
    - Real NSE data via yfinance
    - Realistic slippage (0.05%) and commission (₹20/trade)
    - Event-driven backtest (no look-ahead bias)
    - Professional risk metrics
    - Benchmark comparison vs Nifty 50
    """)
