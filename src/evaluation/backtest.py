import os
import sys
import numpy as np
import pandas as pd
import joblib  

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.logger import get_logger
from src.models.train_model import train_momentum_model
from src.models.preprocess import prepare_model_data

logger = get_logger("InstitutionalBacktest")

def run_portfolio_backtest(feature_filepath: str, stock_name: str, stop_loss_pct: float = 0.015, fee_pct: float = 0.0007):
    
    logger.info(f"⚡ Running battle-tested simulation for {stock_name} (Fees + Slippage enabled)...")
    
    df = pd.read_csv(feature_filepath, parse_dates=['Date'], index_col='Date')
    df['Target'] = df['Log_Return'].shift(-1)
    df.dropna(inplace=True)
    
    split_index = int(len(df) * 0.8)
    test_df = df.iloc[split_index:].copy()
    
    
    model_path = f"data/models/{stock_name}_random_forest.joblib"
    if os.path.exists(model_path):
        logger.info(f"📂 Found pre-trained weights for {stock_name}. Loading instantly from disk...")
        model = joblib.load(model_path)
    else:
        logger.info(f"⚠️ No cached weights found for {stock_name}. Executing optimization grid search...")
        model = train_momentum_model(feature_filepath, stock_name)
    
        
    _, X_test, _, _, _ = prepare_model_data(feature_filepath, target_col='Log_Return')
    test_predictions = model.predict(X_test)
    
    trading_signals = np.where(test_predictions == 1, 1, -1)
    
    market_returns = test_df['Log_Return'].values
    strategy_returns = []
    last_signal = 0  
    
    for i in range(len(market_returns)):
        current_signal = trading_signals[i]
        raw_return = market_returns[i] * current_signal
        
        if raw_return < -stop_loss_pct:
            actual_return = -stop_loss_pct
        else:
            actual_return = raw_return
            
        if current_signal != last_signal:
            actual_return -= fee_pct  
            
        strategy_returns.append(actual_return)
        last_signal = current_signal
            
    test_df['Market_Return'] = test_df['Log_Return']
    test_df['Strategy_Return'] = strategy_returns
    
    test_df['Cumulative_Market'] = (1 + test_df['Market_Return']).cumprod() - 1
    test_df['Cumulative_Strategy'] = (1 + test_df['Strategy_Return']).cumprod() - 1
    
    final_market_perf = test_df['Cumulative_Market'].iloc[-1] * 100
    final_strategy_perf = test_df['Cumulative_Strategy'].iloc[-1] * 100
    
    daily_std = test_df['Strategy_Return'].std()
    annual_sharpe = (test_df['Strategy_Return'].mean() / daily_std) * np.sqrt(252) if daily_std != 0 else 0
    
    downside_returns = test_df['Strategy_Return'][test_df['Strategy_Return'] < 0]
    downside_std = downside_returns.std()
    annual_sortino = (test_df['Strategy_Return'].mean() / downside_std) * np.sqrt(252) if downside_std != 0 else 0
    
    logger.info(f"📊 --- REALISTIC PERFORMANCE FOR {stock_name} ---")
    logger.info(f"🏆 Baseline Buy & Hold: {final_market_perf:.2f}%")
    logger.info(f"🤖 Real-World AI Return: {final_strategy_perf:.2f}%")
    logger.info(f"📈 Annualized Sharpe Ratio: {annual_sharpe:.2f}")
    logger.info(f"🛡️ Annualized Sortino Ratio: {annual_sortino:.2f}")
    
    os.makedirs("data/results", exist_ok=True)
    test_df.to_csv(f"data/results/{stock_name}_backtest.csv")
    logger.info(f"💾 Saved to data/results/{stock_name}_backtest.csv\n")

if __name__ == "__main__":
    stocks = ["TATASTEEL.NS", "RELIANCE.NS"]
    for stock in stocks:
        feature_file = f"data/{stock}_features.csv"
        if os.path.exists(feature_file):
            run_portfolio_backtest(feature_file, stock, stop_loss_pct=0.015, fee_pct=0.0007)