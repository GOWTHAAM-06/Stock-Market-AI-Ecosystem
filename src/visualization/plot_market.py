import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
import mplfinance as mpf
from src.utils.logger import get_logger

logger = get_logger("VisualizationEngine")

def generate_candlestick_chart(feature_filepath: str, stock_name: str, lookback_days: int = 60):
    
    if not os.path.exists(feature_filepath):
        logger.error(f"Visualization target dataset missing: {feature_filepath}")
        return

    logger.info(f"🎨 Generating candlestick matrix visuals for {stock_name}...")
 
    df = pd.read_csv(feature_filepath, parse_dates=['Date'], index_col='Date')
    
    
    df_slice = df.tail(lookback_days).copy()
    
    
    apds = [
        mpf.make_addplot(df_slice['MA_14'], color='#ff7f0e', width=1.5, label='MA 14'),
        mpf.make_addplot(df_slice['MA_50'], color='#2ca02c', width=1.5, label='MA 50')
    ]
    
   
    style = mpf.make_mpf_style(base_mpf_style='charles', gridcolor='#2a2a2a', facecolor='#0d1117')
    
    os.makedirs("data/plots", exist_ok=True)
    output_path = f"data/plots/{stock_name}_candlesticks.png"
    
   
    mpf.plot(
        df_slice,
        type='candle',
        addplot=apds,
        volume=True,
        style=style,
        title=f"\n{stock_name} - Advanced Financial Analytics Engine",
        savefig=dict(fname=output_path, dpi=300, bbox_inches='tight'),
        volume_panel=1,
        figscale=1.5
    )
    
    logger.info(f"📉 High-fidelity candlestick chart saved to: {output_path}")

if __name__ == "__main__":
    stocks = ["TATASTEEL.NS", "RELIANCE.NS"]
    for stock in stocks:
        feature_file = f"data/{stock}_features.csv"
        generate_candlestick_chart(feature_file, stock)