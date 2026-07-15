import os
import sys
import pandas as pd
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.logger import get_logger

logger = get_logger("FeatureEngineering")

def add_advanced_signals(filepath: str, stock_name: str):
    if not os.path.exists(filepath):
        logger.error(f"Target missing: {filepath}")
        return

    logger.info(f"🧪 Engineering advanced cross-features for {stock_name}...")
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    
    
    df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['20STD'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA_20'] + (df['20STD'] * 2)
    df['BB_Lower'] = df['MA_20'] - (df['20STD'] * 2)
    
    df['BB_Position'] = np.where(
        (df['BB_Upper'] - df['BB_Lower']) != 0,
        (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']),
        0.5
    )
    
    
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Vol_MA20']
    
    
    df['VPM_10'] = df['ROC_10'] * df['Volume_Ratio']
    
    df.dropna(inplace=True)
    df.to_csv(filepath)
    logger.info(f"✅ Volume-Price Multipliers integrated into {filepath}")

if __name__ == "__main__":
    stocks = ["TATASTEEL.NS", "RELIANCE.NS"]
    for stock in stocks:
        feature_file = f"data/{stock}_features.csv"
        add_advanced_signals(feature_file, stock)