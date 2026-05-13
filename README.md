# Algorithmic Trading Backtesting Engine

A from-scratch backtesting framework for NSE stocks with professional-grade metrics.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Pre-download data (optional but recommended)
python data/fetcher.py

# Run the dashboard
streamlit run app.py
