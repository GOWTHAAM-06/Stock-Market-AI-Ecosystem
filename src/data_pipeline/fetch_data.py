import os
import yaml
import yfinance as yf
import pandas as pd
from datetime import datetime

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Loads the YAML configuration file dynamically."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Configuration file not found at {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical daily data for a given stock ticker from Yahoo Finance."""
    print(f"🚀 Fetching data for {ticker} from {start_date} to {end_date}...")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"⚠️ No data found for {ticker}. Check the ticker symbol.")
            return None
            
        print(f"✅ Successfully fetched {len(df)} rows for {ticker}.")
        return df
    except Exception as e:
        print(f"❌ Error fetching data for {ticker}: {e}")
        return None

def save_data(df: pd.DataFrame, directory: str, filename: str):
    """Saves the dataframe to the designated data directory."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath)
    print(f"💾 Data saved securely to {filepath}\n")

if __name__ == "__main__":
    # 1. Load configuration dynamically
    config = load_config()
    
    # 2. Extract settings from config
    pipeline_settings = config["pipeline"]
    tickers = pipeline_settings["tickers"]
    start_date = pipeline_settings["start_date"]
    data_dir = pipeline_settings["data_directory"]
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    # 3. Iterate through all tickers configured in config.yaml
    for ticker in tickers:
        data = fetch_stock_data(ticker, start_date, end_date)
        if data is not None:
            save_data(data, data_dir, f"{ticker}_historical.csv")