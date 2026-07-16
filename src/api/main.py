import os
import sys
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

# Force Python to recognize the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.logger import get_logger
from src.models.train_model import train_momentum_model

logger = get_logger("BullstrikeAPI")

# Cache to store active model weights in memory
model_registry = {}

# Modern Lifespan Event Handler (Replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("⚡ BULLSTRIKE COBALT: Quant Engine booting up...")
    tata_features = "data/TATASTEEL.NS_features.csv"
    if os.path.exists(tata_features):
        logger.info("💎 Pre-loading Tata Steel predictive weight matrices...")
        model_registry["TATASTEEL.NS"] = train_momentum_model(tata_features, "TATASTEEL.NS")
    yield
    logger.info("🔌 BULLSTRIKE COBALT: Shutting down safely...")
    model_registry.clear()

app = FastAPI(
    title="BULLSTRIKE COBALT API",
    description="Institutional-Grade Algorithmic API Core",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {
        "brand": "BULLSTRIKE COBALT",
        "status": "Operational",
        "active_networks": list(model_registry.keys())
    }

@app.get("/predict/{stock_name}")
def get_prediction(stock_name: str):
    stock_upper = stock_name.upper()
    feature_file = f"data/{stock_upper}_features.csv"
    
    if not os.path.exists(feature_file):
        raise HTTPException(status_code=404, detail=f"Asset features for {stock_upper} not found.")
        
    if stock_upper not in model_registry:
        logger.info(f"💾 Dynamically caching weights for {stock_upper}...")
        model_registry[stock_upper] = train_momentum_model(feature_file, stock_upper)
        
    df = pd.read_csv(feature_file, parse_dates=['Date'], index_col='Date')
    df.dropna(inplace=True)
    
    if df.empty:
        raise HTTPException(status_code=400, detail="Empty dataset matrix.")
        
    latest_features = df.iloc[-1].drop(['Close', 'Volume', 'Log_Return'], errors='ignore').values.reshape(1, -1)
    model = model_registry[stock_upper]
    prediction = int(model.predict(latest_features)[0])
    
    action = "LONG" if prediction == 1 else "SHORT"
    confidence = max(model.predict_proba(latest_features)[0]) * 100
    
    return {
        "ticker": stock_upper,
        "date_processed": str(df.index[-1].date()),
        "signal": action,
        "confidence": f"{confidence:.2f}%",
        "vector_value": 1 if action == "LONG" else -1
    }

# NEW: Dynamic HTML Dashboard Route hosted directly inside the API!
@app.get("/dashboard", response_class=HTMLResponse)
def get_interactive_dashboard():
    results_dir = "data/results"
    if not os.path.exists(results_dir):
        return "<h1>No backtest metrics found. Please execute src/evaluation/backtest.py first.</h1>"
        
    cards_html = ""
    for file in os.listdir(results_dir):
        if file.endswith("_backtest.csv"):
            stock_name = file.replace("_backtest.csv", "")
            df = pd.read_csv(os.path.join(results_dir, file))
            
            market_ret = df['Cumulative_Market'].iloc[-1] * 100
            strategy_ret = df['Cumulative_Strategy'].iloc[-1] * 100
            
            daily_std = df['Strategy_Return'].std()
            sharpe = (df['Strategy_Return'].mean() / daily_std) * (252**0.5) if daily_std != 0 else 0
            
            performance_color = "text-green-400" if strategy_ret > market_ret else "text-red-400"
            
            cards_html += f"""
            <div class="bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-700">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-2xl font-bold text-white tracking-wider">{stock_name}</h2>
                    <span class="px-3 py-1 text-xs font-semibold rounded-full bg-gray-900 text-cyan-400 border border-cyan-800">COBALT RF</span>
                </div>
                <div class="grid grid-cols-2 gap-4 mb-6">
                    <div class="bg-gray-900 p-3 rounded-lg">
                        <p class="text-xs text-gray-500 font-semibold uppercase">Baseline Buy & Hold</p>
                        <p class="text-lg font-bold text-gray-300">{market_ret:.2f}%</p>
                    </div>
                    <div class="bg-gray-900 p-3 rounded-lg">
                        <p class="text-xs text-gray-500 font-semibold uppercase">Cobalt AI Return</p>
                        <p class="text-lg font-bold {performance_color}">{strategy_ret:.2f}%</p>
                    </div>
                </div>
                <div class="border-t border-gray-700 pt-4 text-sm text-gray-400 flex justify-between">
                    <div>📊 Sharpe: <span class="font-bold text-white">{sharpe:.2f}</span></div>
                    <div class="text-xs text-cyan-500 uppercase tracking-widest font-semibold flex items-center">
                        <span class="w-2 h-2 rounded-full bg-cyan-400 mr-2 animate-pulse"></span>Active Signal
                    </div>
                </div>
            </div>
            """

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BULLSTRIKE COBALT Terminal</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-950 text-gray-100 min-h-screen flex flex-col font-sans">
        <header class="border-b border-gray-800 py-6 px-8 shadow-md bg-gray-900/50 backdrop-blur-md">
            <div class="max-w-7xl mx-auto flex justify-between items-center">
                <div class="flex items-center space-x-3">
                    <span class="text-3xl">⚡</span>
                    <h1 class="text-2xl font-black bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent tracking-widest">BULLSTRIKE COBALT</h1>
                </div>
                <div class="text-xs font-semibold text-cyan-400 tracking-widest border border-cyan-800 px-3 py-1 rounded-md bg-cyan-950/30">LIFESPAN CORE ACTIVE</div>
            </div>
        </header>

        <main class="flex-grow max-w-7xl w-full mx-auto p-8">
            <div class="mb-8 flex justify-between items-end">
                <div>
                    <h2 class="text-xl font-medium text-gray-400">Algorithmic Execution Node</h2>
                    <p class="text-sm text-gray-600">Dynamic model predictions served with zero latency.</p>
                </div>
                <a href="/docs" class="text-xs bg-cyan-500 hover:bg-cyan-600 text-black px-4 py-2 rounded-lg font-bold transition duration-150">GO TO API SWAGGER</a>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                {cards_html}
            </div>
        </main>

        <footer class="border-t border-gray-900 py-4 text-center text-xs text-gray-600 bg-gray-950">
            BULLSTRIKE COBALT • Built for Absolute Efficiency
        </footer>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)