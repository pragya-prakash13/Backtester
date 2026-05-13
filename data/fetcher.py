"""
NSE Data Fetcher using yfinance
Handles data download, caching, and corporate action adjustments
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import os


# Nifty 50 stocks with NSE suffix for yfinance
NIFTY50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "WIPRO.NS",
    "NESTLEIND.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS", "TATAMOTORS.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "ONGC.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "COALINDIA.NS", "BAJAJFINSV.NS", "TECHM.NS", "LTIM.NS", "INDUSINDBK.NS",
    "HINDALCO.NS", "GRASIM.NS", "CIPLA.NS", "APOLLOHOSP.NS", "EICHERMOT.NS",
    "DRREDDY.NS", "DIVISLAB.NS", "BPCL.NS", "BRITANNIA.NS", "SBILIFE.NS",
    "HDFCLIFE.NS", "HEROMOTOCO.NS", "TATACONSUM.NS", "BAJAJ-AUTO.NS", "SHRIRAMFIN.NS"
]

# Nifty 50 index for benchmark
NIFTY_INDEX = "^NSEI"


class DataFetcher:
    """
    Fetches and caches NSE stock data from yfinance.
    Uses adjusted close prices to handle splits and dividends automatically.
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_cache_path(self, symbol: str) -> Path:
        """Generate cache file path for a symbol"""
        clean_symbol = symbol.replace(".", "_").replace("^", "")
        return self.cache_dir / f"{clean_symbol}.csv"
    
    def _is_cache_valid(self, cache_path: Path, max_age_days: int = 1) -> bool:
        """Check if cached data is fresh enough"""
        if not cache_path.exists():
            return False
        
        modified_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - modified_time
        return age < timedelta(days=max_age_days)
    
    def fetch_single_stock(
        self, 
        symbol: str, 
        start_date: str = None,
        end_date: str = None,
        years: int = 5,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single stock.
        
        Parameters:
        -----------
        symbol : str
            Stock ticker with exchange suffix (e.g., 'RELIANCE.NS')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format  
        years : int
            If start_date not provided, fetch this many years of data
        use_cache : bool
            Whether to use cached data if available
            
        Returns:
        --------
        pd.DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
        """
        cache_path = self._get_cache_path(symbol)
        
        # Try cache first
        if use_cache and self._is_cache_valid(cache_path):
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            print(f"Loaded {symbol} from cache")
            return df
        
        # Set date range
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
        
        # Fetch from yfinance
        print(f"Fetching {symbol} from yfinance...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
            
            if df.empty:
                print(f"Warning: No data returned for {symbol}")
                return pd.DataFrame()
            
            # Clean up columns - keep only what we need
            df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
            
            # Save to cache
            df.to_csv(cache_path)
            print(f"Cached {symbol}: {len(df)} rows")
            
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_stocks(
        self,
        symbols: list = None,
        start_date: str = None,
        end_date: str = None,
        years: int = 5,
        use_cache: bool = True
    ) -> dict:
        """
        Fetch data for multiple stocks.
        
        Returns:
        --------
        dict: {symbol: DataFrame}
        """
        if symbols is None:
            symbols = NIFTY50_SYMBOLS
            
        data = {}
        for symbol in symbols:
            df = self.fetch_single_stock(
                symbol, start_date, end_date, years, use_cache
            )
            if not df.empty:
                data[symbol] = df
                
        return data
    
    def fetch_benchmark(
        self,
        start_date: str = None,
        end_date: str = None,
        years: int = 5,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Fetch Nifty 50 index data as benchmark"""
        return self.fetch_single_stock(
            NIFTY_INDEX, start_date, end_date, years, use_cache
        )
    
    def get_adjusted_prices(self, data: dict) -> pd.DataFrame:
        """
        Create a DataFrame of adjusted close prices for all stocks.
        Adjusted prices handle splits and dividends automatically.
        
        Returns:
        --------
        pd.DataFrame with dates as index, symbols as columns
        """
        adj_close = pd.DataFrame()
        
        for symbol, df in data.items():
            if "Adj Close" in df.columns:
                adj_close[symbol] = df["Adj Close"]
                
        return adj_close.dropna(how="all")


def download_all_data():
    """Utility function to pre-download all Nifty50 data"""
    fetcher = DataFetcher()
    
    print("Downloading Nifty 50 stocks...")
    fetcher.fetch_multiple_stocks(use_cache=False)
    
    print("\nDownloading Nifty 50 index (benchmark)...")
    fetcher.fetch_benchmark(use_cache=False)
    
    print("\nDownload complete!")


if __name__ == "__main__":
    download_all_data()
