import os
import sys
import numpy as np
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.logger import get_logger
from src.models.train_model import train_momentum_model
from src.models.preprocess import prepare_model_data

logger = get_logger("BacktestEngine")

def run_portfolio_backtest(feature_filepath: str, stock_name: str):
    
    logger.info(f"📈 Initiating historical algorithmic backtest for {stock_name}...")
    
    
    df = pd.read_csv(feature_filepath, parse_dates=['Date'], index_col='Date')
    df['Target'] = df['Log_Return'].shift(-1)
    df.dropna(inplace=True)
    
    split_index = int(len(df) * 0.8)
    test_df = df.iloc[split_index:].copy()
    
    model = train_momentum_model(feature_filepath, stock_name)
    _, X_test, _, _, _ = prepare_model_data(feature_filepath, target_col='Log_Return')
    test_predictions = model.predict(X_test)
    
   
    trading_signals = np.where(test_predictions == 1, 1, 0)
    
   
    test_df['Market_Return'] = test_df['Log_Return']
    test_df['Strategy_Return'] = test_df['Market_Return'] * trading_signals
    
    
    test_df['Cumulative_Market'] = (1 + test_df['Market_Return']).cumprod() - 1
    test_df['Cumulative_Strategy'] = (1 + test_df['Strategy_Return']).cumprod() - 1
    
    final_market_perf = test_df['Cumulative_Market'].iloc[-1] * 100
    final_strategy_perf = test_df['Cumulative_Strategy'].iloc[-1] * 100
    
    logger.info(f"📊 --- BACKTEST RESULTS FOR {stock_name} ---")
    logger.info(f"🏆 Baseline Buy & Hold Return: {final_market_perf:.2f}%")
    logger.info(f"🤖 AI Algorithmic Return: {final_strategy_perf:.2f}%")
   
    os.makedirs("data/results", exist_ok=True)
    test_df.to_csv(f"data/results/{stock_name}_backtest.csv")
    logger.info(f"💾 Backtest log matrix saved to data/results/{stock_name}_backtest.csv\n")

if __name__ == "__main__":
    stocks = ["TATASTEEL.NS", "RELIANCE.NS"]
    for stock in stocks:
        feature_file = f"data/{stock}_features.csv"
        if os.path.exists(feature_file):
            run_portfolio_backtest(feature_file, stock)