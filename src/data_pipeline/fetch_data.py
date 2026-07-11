import os
import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical daily data for a given stock ticker from Yahoo Finance.
    """
    print(f"🚀 Fetching data for {ticker} from {start_date} to {end_date}...")
    try:
        # For Indian stocks on Yahoo Finance, append '.NS' for NSE
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"⚠️ No data found for {ticker}. Check the ticker symbol.")
            return None
            
        print(f"✅ Successfully fetched {len(df)} rows.")
        return df
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def save_data(df: pd.DataFrame, filename: str):
    """Saves the dataframe to the data/ directory."""
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    df.to_csv(filepath)
    print(f"💾 Data saved securely to {filepath}")

if __name__ == "__main__":
    # Let's start with a prominent NSE stock: Tata Steel
    TARGET_TICKER = "TATASTEEL.NS" 
    START = "2023-01-01"
    END = datetime.today().strftime('%Y-%m-%d')
    
    data = fetch_stock_data(TARGET_TICKER, START, END)
    if data is not None:
        save_data(data, f"{TARGET_TICKER}_historical.csv")