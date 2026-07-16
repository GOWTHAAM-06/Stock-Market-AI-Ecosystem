import os
import sys
import pandas as pd

# Force Python to recognize the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.logger import get_logger

logger = get_logger("DashboardEngine")

def generate_html_dashboard():
    """
    Compiles backtest CSV results into a beautiful, interactive local HTML 
    dashboard with dark-mode styling and metric cards.
    """
    logger.info("🎨 Compiling Interactive Local Web Dashboard...")
    
    results_dir = "data/results"
    if not os.path.exists(results_dir):
        logger.error("❌ No backtest results found! Run backtest.py first.")
        return
        
    cards_html = ""
    
    
    for file in os.listdir(results_dir):
        if file.endswith("_backtest.csv"):
            stock_name = file.replace("_backtest.csv", "")
            filepath = os.path.join(results_dir, file)
            
            df = pd.read_csv(filepath, parse_dates=['Date'])
            
            # Extract final returns & metrics
            market_ret = (df['Cumulative_Market'].iloc[-1] * 100)
            strategy_ret = (df['Cumulative_Strategy'].iloc[-1] * 100)
            
            # Read daily returns to calculate metrics
            daily_std = df['Strategy_Return'].std()
            sharpe = (df['Strategy_Return'].mean() / daily_std) * (252**0.5) if daily_std != 0 else 0
            
            downside = df['Strategy_Return'][df['Strategy_Return'] < 0]
            downside_std = downside.std()
            sortino = (df['Strategy_Return'].mean() / downside_std) * (252**0.5) if downside_std != 0 else 0
            
            # Determine card color themes based on performance
            performance_color = "text-green-400" if strategy_ret > market_ret else "text-red-400"
            
            cards_html += f"""
            <div class="bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-700 transition transform hover:scale-105 duration-200">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-2xl font-bold text-white tracking-wider">{stock_name}</h2>
                    <span class="px-3 py-1 text-xs font-semibold rounded-full bg-gray-900 text-gray-400">Random Forest</span>
                </div>
                <div class="grid grid-cols-2 gap-4 mb-6">
                    <div class="bg-gray-900 p-3 rounded-lg">
                        <p class="text-xs text-gray-500 font-semibold uppercase">Buy & Hold Return</p>
                        <p class="text-lg font-bold text-gray-300">{market_ret:.2f}%</p>
                    </div>
                    <div class="bg-gray-900 p-3 rounded-lg">
                        <p class="text-xs text-gray-500 font-semibold uppercase">AI Strategy Return</p>
                        <p class="text-lg font-bold {performance_color}">{strategy_ret:.2f}%</p>
                    </div>
                </div>
                <div class="border-t border-gray-700 pt-4 grid grid-cols-2 gap-2 text-sm text-gray-400">
                    <div>📊 Sharpe Ratio: <span class="font-bold text-white">{sharpe:.2f}</span></div>
                    <div>🛡️ Sortino Ratio: <span class="font-bold text-white">{sortino:.2f}</span></div>
                </div>
            </div>
            """

    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AlphaVision Trading AI Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-900 text-gray-100 min-h-screen flex flex-col font-sans">
        <header class="bg-gray-900 border-b border-gray-800 py-6 px-8 shadow-md">
            <div class="max-w-7xl mx-auto flex justify-between items-center">
                <div class="flex items-center space-x-3">
                    <span class="text-3xl">🤖</span>
                    <h1 class="text-2xl font-black bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent">ALPHAVISION QUANT SYSTEMS</h1>
                </div>
                <div class="text-sm font-semibold text-gray-500 uppercase tracking-widest">Day 5 Operational Log</div>
            </div>
        </header>

        <main class="flex-grow max-w-7xl w-full mx-auto p-8">
            <div class="mb-8">
                <h2 class="text-xl font-medium text-gray-400">Performance Overview</h2>
                <p class="text-sm text-gray-600">Simulated using hyperparameter-optimized models, real slippage penalties, and stop-loss rules.</p>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                {cards_html}
            </div>
        </main>

        <footer class="bg-gray-950 border-t border-gray-800 py-4 text-center text-xs text-gray-600">
            AlphaVision Engine • Running Locally • Clean execution verified.
        </footer>
    </body>
    </html>
    """

    os.makedirs("data", exist_ok=True)
    with open("data/dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_content)
        
    logger.info("🚀 Dashboard successfully compiled to: data/dashboard.html")

if __name__ == "__main__":
    generate_html_dashboard()