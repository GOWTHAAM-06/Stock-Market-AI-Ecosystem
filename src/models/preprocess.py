import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils.logger import get_logger

logger = get_logger("DataPreprocessing")

def prepare_model_data(feature_filepath: str, target_col: str = 'Log_Return', test_size: float = 0.2):
    
    if not os.path.exists(feature_filepath):
        logger.error(f"Preprocessing target missing: {feature_filepath}")
        return None

    logger.info(f"⚙️ Initializing preprocessing pipeline for {feature_filepath}...")
    df = pd.read_csv(feature_filepath, parse_dates=['Date'], index_col='Date')
    
    
    df['Target'] = df[target_col].shift(-1)
    df.dropna(inplace=True)
    
    X = df.drop(columns=['Target'])
    y = df['Target']
    
    
    feature_names = X.columns.tolist()
    
    
    split_index = int(len(df) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    logger.info(f"📊 Data split chronologically: Train size = {len(X_train)}, Test size = {len(X_test)}")
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("✅ Feature matrix scaling completed using standard normalization.")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names