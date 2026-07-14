import os
import yaml
import yfinance as yf
import pandas as pd
from datetime import datetime
from src.utils.logger import get_logger

from src.data_pipeline.validate_data import validate_dataset

logger = get_logger("DataIngestion")

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Loads the YAML configuration file dynamically."""
    if not os.path.exists(config_path):
        logger.error(f"Configuration file missing at {config_path}")
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    logger.info("Configuration parameters loaded successfully.")
    return config

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical daily data for a given stock ticker from Yahoo Finance."""
    logger.info(f"Initiating ingestion for {ticker} from {start_date} to {end_date}")
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            logger.warning(f"No data records returned for ticker: {ticker}")
            return None
            
        logger.info(f"Successfully retrieved {len(df)} data rows for {ticker}.")
        return df
    except Exception as e:
        logger.error(f"Critical exception occurred during {ticker} ingestion: {str(e)}")
        return None

def save_data(df: pd.DataFrame, directory: str, filename: str):
    """Saves the dataframe to the designated data directory."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    df.to_csv(filepath)
    logger.info(f"Data layer committed to storage path: {filepath}")

if __name__ == "__main__":
    logger.info("Starting Stock Market AI Ingestion Pipeline...")
    try:
        config = load_config()
        pipeline_settings = config["pipeline"]
        tickers = pipeline_settings["tickers"]
        start_date = pipeline_settings["start_date"]
        data_dir = pipeline_settings["data_directory"]
        end_date = datetime.today().strftime('%Y-%m-%d')
        
        for ticker in tickers:
            filename = f"{ticker}_historical.csv"
            data = fetch_stock_data(ticker, start_date, end_date)
            if data is not None:
                save_data(data, data_dir, filename)
                
                
                full_path = os.path.join(data_dir, filename)
                validate_dataset(full_path)
                
        logger.info("Pipeline processing batch completed successfully.\n")
    except Exception as e:
        logger.critical(f"Pipeline crashed during execution: {str(e)}\n")