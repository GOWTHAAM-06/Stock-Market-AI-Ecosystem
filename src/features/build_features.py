import os
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("FeatureEngineering")

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI) using exponential rolling averages."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    
    rs = gain / (loss + 1e-9)  
    return 100 - (100 / (1 + rs))

def generate_technical_indicators(input_filepath: str, output_filepath: str) -> pd.DataFrame:
   
    if not os.path.exists(input_filepath):
        logger.error(f"Feature target missing: {input_filepath}")
        return None

    logger.info(f"📊 Extrapolating math matrices for {input_filepath}...")
    df = pd.read_csv(input_filepath, parse_dates=['Date'], index_col='Date')
    
    
    df['MA_14'] = df['Close'].rolling(window=14).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
   
    df['RSI_14'] = calculate_rsi(df['Close'], period=14)
    
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
   
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Rolling_Volatility'] = df['Log_Return'].rolling(window=14).std()
    
  
    df.dropna(inplace=True)
    
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath)
    logger.info(f"✨ Feature matrix successfully saved to {output_filepath}. Final Dimensions: {df.shape}")
    
    return df

if __name__ == "__main__":

    stocks = ["TATASTEEL.NS", "RELIANCE.NS"]
    
    for stock in stocks:
        raw_path = f"data/{stock}_historical.csv"
        features_path = f"data/{stock}_features.csv"
        generate_technical_indicators(raw_path, features_path)