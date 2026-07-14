import os
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("FeatureEngineering")

def generate_technical_indicators(filepath: str) -> pd.DataFrame:
    """
    Loads validated historical data and generates quantitative 
    technical indicators for predictive modeling.
    """
    if not os.path.exists(filepath):
        logger.error(f"Feature engineering target missing: {filepath}")
        return None

    logger.info(f"📊 Initializing feature calculations for {filepath}...")
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    
    # Core mathematical placeholders for Day 4 calculations
    # 1. Rolling Moving Averages (Trend)
    df['MA_14'] = df['Close'].rolling(window=14).mean()
    
    # 2. Daily Returns (Volatility)
    df['Daily_Return'] = df['Close'].pct_change()
    
    logger.info(f"✨ Feature matrix generated successfully. Data dimensions: {df.shape}")
    return df

if __name__ == "__main__":
    test_file = "data/TATASTEEL.NS_historical.csv"
    if os.path.exists(test_file):
        generate_technical_indicators(test_file)
    else:
        logger.warning("Target test file not found. Skipping local indicator dry-run.")