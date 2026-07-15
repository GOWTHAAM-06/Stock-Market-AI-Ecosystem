import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from src.utils.logger import get_logger
from src.models.preprocess import prepare_model_data

logger = get_logger("ModelTraining")

def train_momentum_model(feature_filepath: str, stock_name: str):
    
    logger.info(f"🤖 Initializing Hyperparameter Tuning Pipeline for {stock_name}...")
    
    X_train, X_test, y_train, y_test, feature_names = prepare_model_data(feature_filepath, target_col='Log_Return')
    
    y_train_binary = np.where(y_train > 0, 1, 0)
    y_test_binary = np.where(y_test > 0, 1, 0)
    
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train_binary)
    
    best_model = grid_search.best_estimator_
    logger.info(f"⚙️ Best Parameters Found: {grid_search.best_params_}")
    
   
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test_binary, predictions)
    logger.info(f"📊 {stock_name} Optimized Directional Accuracy: {accuracy * 100:.2f}%")
    
   
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Optimized Feature Importance Matrix - {stock_name}")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center", color="#1f77b4")
    plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    
    os.makedirs("data/plots", exist_ok=True)
    plot_path = f"data/plots/{stock_name}_feature_importance.png"
    plt.savefig(plot_path)
    plt.close()
    
    return best_model

if __name__ == "__main__":
    stocks = ["TATASTEEL.NS", "RELIANCE.NS"]
    for stock in stocks:
        feature_file = f"data/{stock}_features.csv"
        if os.path.exists(feature_file):
            train_momentum_model(feature_file, stock)