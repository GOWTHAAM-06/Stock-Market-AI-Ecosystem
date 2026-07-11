import os
import pandas as pd
import numpy as np

def clean_historical_data(input_filepath: str, output_filepath: str):
    
    if not os.path.exists(input_filepath):
        print(f"❌ Error: Raw data file not found at {input_filepath}")
        return

    print(f"🧹 Loading raw data from {input_filepath}...")
    df = pd.read_csv(input_filepath)
    
    initial_rows = len(df)
    

    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df.set_index('Date', inplace=True)
    
    
    df = df.drop_duplicates()
    
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0).astype(int)
        
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill() 

    
    columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[[col for col in columns_to_keep if col in df.columns]]
    
    final_rows = len(df)
    print(f"✅ Data cleaning complete. (Dropped {initial_rows - final_rows} duplicate/corrupt rows).")
    

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath)
    print(f"💾 Cleaned data successfully saved to {output_filepath}\n")

if __name__ == "__main__":

    RAW_PATH = "data/TATASTEEL.NS_historical.csv"
    CLEANED_PATH = "data/TATASTEEL.NS_cleaned.csv"
    
    clean_historical_data(RAW_PATH, CLEANED_PATH)