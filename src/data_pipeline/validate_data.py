import os
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("DataValidation")

def validate_dataset(filepath: str) -> bool:
    """
    Performs critical integrity checks on the processed financial data.
    Returns True if data passes all production checks, False otherwise.
    """
    if not os.path.exists(filepath):
        logger.error(f"Validation target missing: {filepath}")
        return False

    logger.info(f"🛡️ Running integrity checks on {filepath}...")
    df = pd.read_csv(filepath)
    
    is_valid = True
    
    # Check 1: Empty Data Check
    if df.empty:
        logger.error(f"❌ Validation Failed: Dataset is completely empty.")
        return False
        
    # Check 2: Missing Essential Columns Check
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"❌ Validation Failed: Missing columns {missing_cols}")
        is_valid = False

    # Check 3: Mathematical Multi-Point Checks (Only run if columns exist)
    if is_valid:
        # High must always be greater than or equal to Low
        anomaly_high_low = df[df['High'] < df['Low']]
        if not anomaly_high_low.empty:
            logger.warning(f"⚠️ Data Anomaly: Found {len(anomaly_high_low)} rows where High < Low!")
            is_valid = False
            
        # Prices cannot be negative or zero
        for col in ['Open', 'High', 'Low', 'Close']:
            negative_prices = df[df[col] <= 0]
            if not negative_prices.empty:
                logger.error(f"❌ Validation Failed: Negative or zero prices detected in {col} column.")
                is_valid = False

    if is_valid:
        logger.info(f"✅ Validation Passed: {filepath} meets all data standards. Total rows: {len(df)}")
    else:
        logger.error(f"🚨 Validation Failed: {filepath} contains corrupted records.")
        
    return is_valid

if __name__ == "__main__":
    # Test our checks on one of the historical files we generated tonight
    test_file = "data/TATASTEEL.NS_historical.csv"
    validate_dataset(test_file)