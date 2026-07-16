# ⚡ BULLSTRIKE COBALT 📈

> **An institutional-grade, end-to-end quantitative data science and algorithmic trading ecosystem built to ingest, preprocess, and predict predictive momentum trends on National Stock Exchange (NSE) equities.**

---

## 🏗️ Project Architecture

```text
Stock-Market-AI-Ecosystem/
├── data/                          # Feature stores, CSV datasets, & backtest evaluations (git-ignored)
├── src/
│   ├── api/
│   │   └── main.py                # Fast API Lifespan core & dynamic Web Terminal dashboard
│   ├── data_pipeline/
│   │   ├── fetch_data.py          # Automated NSE stock ingestion pipeline
│   │   └── clean_data.py          # Timezone alignment, anomalies filter, forward-fills
│   ├── features/
│   │   └── feature_engineering.py # Technical indicators and feature engineering scripts
│   ├── models/
│   │   ├── preprocess.py          # Scaling, alignment, and dataset partitioners
│   │   └── train_model.py         # Hyperparameter tuning, grid searches, and RF models
│   ├── evaluation/
│   │   └── backtest.py            # Real-world performance simulator (Slippage, Fees, Sharpe, Sortino)
│   └── utils/
│       └── logger.py              # Centralized platform logging

📅 Development Log
Day 1: Ingestion & Data Hygiene
Goal: Establish repository core structure and stabilize data gathering.

Accomplishments:

Initialized modular directory structure.

Built fetch_data.py to automate historical NSE stock ingestion using Python and Pandas.

Built clean_data.py to handle timezone normalization, drop market anomalies, and execute forward-fills on empty pricing ticks.

Status: ✅ Complete.

Day 2 & 3: Quant Feature Engineering & Model Building
Goal: Extract alpha vectors and design predictive classifiers.

Accomplishments:

Engineered 24-dimensional feature matrices (Moving Averages, RSI, Bollinger Bands, and MACD).

Integrated customized StandardScaler pipelines to sanitize inputs.

Developed automated hyperparameter-tuning via grid-search Random Forests to forecast next-day momentum.

Status: ✅ Complete.

Day 4: Performance Attribution & Risk Backtesting
Goal: Assess performance under realistic market conditions.

Accomplishments:

Coded an institutional performance simulator featuring custom transaction costs and slippage penalties.

Implemented key risk attribution metrics including Sharpe Ratio and Sortino Ratio (measuring downside deviation).

Status: ✅ Complete.

Day 5: Rebranding & Production API Core
Goal: Rebrand the architecture to BULLSTRIKE COBALT and establish a unified deployment core.

Accomplishments:

Built a high-performance FastAPI core with Python's modern lifespan design to cache model weights on startup with zero warnings.

Developed a dynamic, responsive HTML/Tailwind CSS dashboard served live at /dashboard.

Solved 24-feature preprocessing alignment anomalies to serve live /predict model inferences with zero latency.

Status: ✅ Complete.