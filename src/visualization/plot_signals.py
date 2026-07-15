import os
import sys
import numpy as np
import pandas as pd
import mplfinance as mpf


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.logger import get_logger
from src.models.train_model import train_momentum_model

logger = get_logger("SignalVisualizer")

def generate_prediction_chart(feature_filepath: str, stock_name: str):
   
    logger.info(f"🔮 Generating AI Predictive Candlestick Chart for {stock_name}...")
    
   
    df = pd.read_csv(feature_filepath, parse_dates=['Date'], index_col='Date')
    df['Target'] = df['Log_Return'].shift(-1)
    df.dropna(inplace=True)
    

    split_index = int(len(df) * 0.8)
    test_df = df.iloc[split_index:].copy()
    
   
    model = train_momentum_model(feature_filepath, stock_name)
    
    
    from src.models.preprocess import prepare_model_data
    _, X_test, _, _, _ = prepare_model_data(feature_filepath, target_col='Log_Return')
    
    
    test_predictions = model.predict(X_test)
    

    buy_signals = np.full(len(test_df), np.nan)
    sell_signals = np.full(len(test_df), np.nan)
    
    
    for i in range(len(test_predictions)):
        if test_predictions[i] == 1:
            buy_signals[i] = test_df['Low'].iloc[i] * 0.98
        else:
            sell_signals[i] = test_df['High'].iloc[i] * 1.02
            
    
    apds = [
        mpf.make_addplot(test_df['MA_14'], color='#ff7f0e', width=1.2),
        mpf.make_addplot(test_df['MA_50'], color='#2ca02c', width=1.2),
        mpf.make_addplot(buy_signals, type='scatter', marker='^', markersize=60, color='#00ff00'),
        mpf.make_addplot(sell_signals, type='scatter', marker='v', markersize=60, color='#ff0000')
    ]
    
  
    style = mpf.make_mpf_style(base_mpf_style='charles', gridcolor='#2a2a2a', facecolor='#0d1117')
    
    output_path = f"data/plots/{stock_name}_ai_signals.png"
    
    mpf.plot(
        test_df,
        type='candle',
        addplot=apds,
        volume=True,
        style=style,
        title=f"\n{stock_name} - AI Predictive Directional Core",
        savefig=dict(fname=output_path, dpi=300, bbox_inches='tight'),
        volume_panel=1,
        figscale=1.5
    )
    
    logger.info(f"🚀 AI Signal Candlestick chart successfully exported to: {output_path}\n")

if __name__ == "__main__":
    stocks = ["TATASTEEL.NS", "RELIANCE.NS"]
    for stock in stocks:
        feature_file = f"data/{stock}_features.csv"
        if os.path.exists(feature_file):
            generate_prediction_chart(feature_file, stock)