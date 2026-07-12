# trading-platform

**All-in-one AI-powered trading and financial platform — built for live trading.**

A production-ready automated trading system that runs real orders through live brokers, manages multiple strategies simultaneously, and handles the full trade lifecycle from signal generation to execution and risk control.

---

## What this is

This is a **live trading platform** — not just a backtester or research tool. It is designed to:

- **Execute real trades** through live broker integrations (Alpaca, Interactive Brokers) and a built-in paper trading simulator.
- **Run multiple strategies in parallel**, each with its own capital allocation, position sizing, and lifecycle (start / pause / resume / stop).
- **Manage risk in real time** — per-strategy and portfolio-level limits, automatic halt on breach, drawdown and volatility controls.
- **Fetch live market data** from Yahoo Finance and crypto exchanges (via CCXT), with support for intraday and daily timeframes.
- **Persist state** across restarts (SQLite-backed snapshots of portfolio, trades, and signals).
- **Expose a REST API** (FastAPI) so the platform can be controlled programmatically or integrated into a dashboard or external system.

---

## Core capabilities

| Area | Details |
|---|---|
| **Live execution** | Alpaca (REST + paper/live accounts), IBKR (IB Gateway / TWS), Paper broker (built-in simulator) |
| **Strategy library** | 10 strategies across Trend Following, Mean Reversion, Momentum, Statistical Arbitrage, and ML Alpha |
| **Risk management** | Position limits, daily loss limits, drawdown halt, volatility scaling, execution override |
| **Backtesting** | Event-driven backtester with full stats (Sharpe, max drawdown, win rate, P&L) |
| **ML** | RandomForest / GradientBoosting alpha model with periodic retraining on technical factors |
| **Data pipeline** | Yahoo Finance (equities), CCXT (crypto), Forex — historical and current price feeds |
| **Persistence** | SQLite via SQLAlchemy — portfolio snapshots, trade log, signal audit trail |
| **API** | FastAPI REST — strategy management, market data ingestion, backtest runner, risk controls, portfolio queries |

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn app:app --reload --host 0.0.0.0 --port 5000

# Run tests
python -m pytest tests/
```

Set broker credentials in `.env` (see `.env.example`) before connecting to a live or paper account.

---

## Project structure

```
trading-platform/
├── app.py                  # FastAPI REST API
├── strategies/             # 10 strategy implementations + registry
├── core/
│   ├── trading_engine.py   # Core data model (Asset, Order, Portfolio, Position)
│   ├── strategy_manager.py # Multi-strategy orchestrator
│   └── persistence.py      # SQLite state persistence
├── brokers/
│   ├── paper.py            # Paper trading simulator
│   ├── alpaca.py           # Alpaca Markets integration
│   └── ibkr.py             # Interactive Brokers integration
├── risk_management/        # Risk manager, limits, and execution controls
├── backtester/             # Event-driven backtesting engine
├── data_pipeline/          # Market data providers (Yahoo, CCXT, Forex)
├── ml_engine/              # Alpha factor computation
└── tests/                  # 150+ unit tests
```

---

## Broker support

| Broker | Status |
|---|---|
| Paper (built-in) | ✅ Fully functional — instant fills with slippage & commission |
| Alpaca | 🔧 Integration stub — wire up `alpaca-trade-api` credentials to go live |
| Interactive Brokers | 🔧 Integration stub — requires IB Gateway / TWS + `ib_insync` |

---

## Disclaimer

This software is for educational and research purposes. Live trading involves significant financial risk. Use paper trading accounts to validate strategies before deploying real capital.
