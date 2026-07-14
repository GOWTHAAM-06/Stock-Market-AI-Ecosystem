import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils.logger import get_logger
from src.models.preprocess import prepare_model_data

logger = get_logger("ModelTraining")

def train_momentum_model(feature_filepath: str):
    
    
    logger.info(f"🤖 Initializing Machine Learning pipeline for {feature_filepath}...")
    
   
    # We target 'Log_Return' to determine price direction
    X_train, X_test, y_train, y_test = prepare_model_data(feature_filepath, target_col='Log_Return')
    
   
    y_train_binary = np.where(y_train > 0, 1, 0)
    y_test_binary = np.where(y_test > 0, 1, 0)
    
   
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    logger.info("🏋️ Training Random Forest Classifier on historical matrix...")
    model.fit(X_train, y_train_binary)
    logger.info("✅ Model training complete.")
    
   
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test_binary, predictions)
    
    logger.info(f"🎯 Operational Metrics Matrix Summary:")
    logger.info(f"📊 Test Set Directional Accuracy: {accuracy * 100:.2f}%")
    
   
    report = classification_report(y_test_binary, predictions, target_names=['DOWN', 'UP'])
    print("\n--- CLASSIFICATION REPORT MATRIX ---")
    print(report)
    print("------------------------------------\n")
    
    return model

if __name__ == "__main__":
    test_feature_file = "data/TATASTEEL.NS_features.csv"
    if os.path.exists(test_feature_file):
        train_momentum_model(test_feature_file)