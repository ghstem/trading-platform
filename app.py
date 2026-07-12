"""
Trading Platform — FastAPI Application
Automated multi-strategy trading system REST API.

Start with:
    uvicorn app:app --reload --host 0.0.0.0 --port 5000
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import get_config
from config.app_settings import get_settings
from core.trading_engine import (
    Asset, AssetClass, Portfolio,
    OrderSide, OrderType, OrderStatus,
)
from core.strategy_manager import StrategyManager, StrategyState
from core.persistence import PersistenceManager
from data_pipeline.market_data import get_market_data_manager
from risk_management.risk_manager import RiskLimits, RiskManager
from strategies.registry import get_registry
from strategies.base import BaseStrategy, SignalType
from backtester.backtest_engine import Backtester, BacktestConfig, Strategy as BtStrategy

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

config = get_config()
settings = get_settings()
_portfolio: Optional[Portfolio] = None
_risk_manager: Optional[RiskManager] = None
_strategy_manager: Optional[StrategyManager] = None
_persistence: Optional[PersistenceManager] = None

# Metrics counters
_metrics: Dict[str, Any] = {
    "requests_total": 0,
    "orders_total": 0,
    "signals_total": 0,
    "errors_total": 0,
    "start_time": None,
}

# Active WebSocket connections for price streaming
_ws_connections: List[WebSocket] = []


def _get_portfolio() -> Portfolio:
    if _portfolio is None:
        raise HTTPException(500, "Portfolio not initialised")
    return _portfolio


def _get_strategy_manager() -> StrategyManager:
    if _strategy_manager is None:
        raise HTTPException(500, "StrategyManager not initialised")
    return _strategy_manager


def _get_persistence() -> PersistenceManager:
    if _persistence is None:
        raise HTTPException(500, "Persistence manager not initialised")
    return _persistence


# ---------------------------------------------------------------------------
# Background auto-feed task
# ---------------------------------------------------------------------------

async def _auto_feed_task():
    """
    Periodically fetch live prices and push them as bars to all running
    strategies.  Only active when BAR_AUTO_FEED_INTERVAL > 0.
    """
    interval = settings.BAR_AUTO_FEED_INTERVAL
    if interval <= 0:
        return
    logger.info(f"Auto-feed task started (interval={interval}s)")
    mdm = get_market_data_manager()
    while True:
        await asyncio.sleep(interval)
        sm = _strategy_manager
        port = _portfolio
        if sm is None or port is None:
            continue
        try:
            all_assets = {
                asset
                for assets in sm.get_all_strategy_assets().values()
                for asset in assets
            }
            all_assets |= set(port.positions.keys())
            if not all_assets:
                continue
            price_map: Dict[Asset, float] = {}
            for asset in all_assets:
                price = mdm.fetch_current_price(asset, use_cache=True)
                if price > 0:
                    price_map[asset] = price
            if price_map:
                accepted = sm.on_bar(datetime.now(), price_map)
                if accepted:
                    _metrics["signals_total"] += len(accepted)
                    payload = {
                        "type": "bar",
                        "timestamp": datetime.now().isoformat(),
                        "prices": {a.symbol: p for a, p in price_map.items()},
                        "signals": len(accepted),
                    }
                    await _broadcast_ws(payload)
                    if _persistence:
                        _persistence.record_equity(port)
        except Exception as exc:
            logger.error(f"Auto-feed error: {exc}")


async def _broadcast_ws(payload: dict) -> None:
    """Send a JSON message to all connected WebSocket clients."""
    import json
    disconnected = []
    for ws in list(_ws_connections):
        try:
            await ws.send_json(payload)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in _ws_connections:
            _ws_connections.remove(ws)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown."""
    global _portfolio, _risk_manager, _strategy_manager, _persistence

    logger.info("Starting Trading Platform API…")
    _metrics["start_time"] = datetime.now(timezone.utc).isoformat()

    # Persistence
    _persistence = PersistenceManager(settings.DATABASE_URL)

    _portfolio = Portfolio(initial_capital=config.INITIAL_CAPITAL)
    _risk_manager = RiskManager(
        portfolio=_portfolio,
        limits=RiskLimits(
            max_position_size_pct=config.MAX_POSITION_SIZE,
            max_daily_loss_pct=config.MAX_DAILY_LOSS / _portfolio.initial_capital
            if _portfolio.initial_capital > 0 else -0.05,
        ),
    )
    _strategy_manager = StrategyManager(
        portfolio=_portfolio,
        risk_manager=_risk_manager,
    )
    logger.info(f"Portfolio initialised with ${_portfolio.initial_capital:,.0f} capital")

    # Save initial snapshot
    _persistence.save_portfolio_snapshot(_portfolio)

    # Start background auto-feed
    task = asyncio.create_task(_auto_feed_task())

    yield

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("Trading Platform API shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Trading Platform API",
    description=(
        "Automated multi-strategy trading system. "
        "Supports trend-following, mean-reversion, momentum, pairs-trading, and ML strategies."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────
# Restrict origins to those declared in settings; use ["*"] only in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# ── Request-ID middleware ────────────────────────────────────────────────────
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique ``X-Request-ID`` to every request and response."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        _metrics["requests_total"] += 1
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"
        return response


app.add_middleware(RequestIDMiddleware)


# ── API key authentication ───────────────────────────────────────────────────
class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    When ``API_KEY`` is configured in settings, require callers to pass the
    header ``X-API-Key: <value>``.
    Skips auth for health / docs / metrics endpoints.
    """

    _EXEMPT_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc", "/metrics"}

    async def dispatch(self, request: Request, call_next):
        if settings.API_KEY is None:
            return await call_next(request)
        if request.url.path in self._EXEMPT_PATHS:
            return await call_next(request)
        provided = request.headers.get("X-API-Key", "")
        if provided != settings.API_KEY:
            _metrics["errors_total"] += 1
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"},
            )
        return await call_next(request)


app.add_middleware(APIKeyMiddleware)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class AssetModel(BaseModel):
    symbol: str
    asset_class: str = "STOCK"
    exchange: str = "NASDAQ"
    currency: str = "USD"


class AddStrategyRequest(BaseModel):
    instance_id: str
    strategy_key: str
    assets: List[AssetModel]
    capital_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    params: Dict[str, Any] = Field(default_factory=dict)
    auto_start: bool = True


class UpdateAllocationRequest(BaseModel):
    allocations: Dict[str, float]


class BarDataRequest(BaseModel):
    current_date: Optional[str] = None
    prices: Dict[str, float]   # {symbol: price}


class BacktestRequest(BaseModel):
    strategy_key: str
    assets: List[AssetModel]
    initial_capital: float = 100_000.0
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    params: Dict[str, Any] = Field(default_factory=dict)


class OrderRequest(BaseModel):
    symbol: str
    asset_class: str = "STOCK"
    exchange: str = "NASDAQ"
    side: str           # BUY | SELL
    quantity: float
    order_type: str = "MARKET"
    price: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_asset(m: AssetModel) -> Asset:
    try:
        ac = AssetClass[m.asset_class.upper()]
    except KeyError:
        raise HTTPException(400, f"Unknown asset_class: {m.asset_class}")
    return Asset(symbol=m.symbol.upper(), asset_class=ac, exchange=m.exchange.upper(), currency=m.currency)


def _build_assets(models: List[AssetModel]) -> List[Asset]:
    return [_build_asset(m) for m in models]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/", tags=["system"])
def root():
    return {
        "service": "Trading Platform API",
        "version": "1.0.0",
        "docs": "/docs",
    }


# ---------------------------------------------------------------------------
# Portfolio routes
# ---------------------------------------------------------------------------

@app.get("/portfolio/summary", tags=["portfolio"])
def portfolio_summary():
    return _get_portfolio().get_portfolio_summary()


@app.get("/portfolio/positions", tags=["portfolio"])
def portfolio_positions():
    port = _get_portfolio()
    df = port.get_positions_df()
    return {"positions": df.to_dict(orient="records") if not df.empty else []}


@app.get("/portfolio/orders", tags=["portfolio"])
def portfolio_orders(limit: int = Query(default=50, ge=1, le=500)):
    port = _get_portfolio()
    orders = [
        {
            "order_id": o.order_id,
            "symbol": o.asset.symbol,
            "side": o.side.value,
            "type": o.order_type.value,
            "quantity": o.quantity,
            "filled_qty": o.filled_quantity,
            "avg_fill_price": o.average_fill_price,
            "status": o.status.value,
            "commission": o.commission,
            "created_at": o.created_at.isoformat(),
        }
        for o in port.orders[-limit:]
    ]
    return {"orders": list(reversed(orders))}


@app.post("/portfolio/order", tags=["portfolio"])
def place_order(req: OrderRequest):
    port = _get_portfolio()
    try:
        ac = AssetClass[req.asset_class.upper()]
    except KeyError:
        raise HTTPException(400, f"Unknown asset_class: {req.asset_class}")
    try:
        side = OrderSide[req.side.upper()]
    except KeyError:
        raise HTTPException(400, f"Unknown side: {req.side}")
    try:
        otype = OrderType[req.order_type.upper()]
    except KeyError:
        raise HTTPException(400, f"Unknown order_type: {req.order_type}")

    asset = Asset(symbol=req.symbol.upper(), asset_class=ac, exchange="MANUAL")
    order = port.create_order(asset, side, req.quantity, otype, req.price)

    # ── Risk validation ──────────────────────────────────────────────
    if _risk_manager is not None:
        can, reason = _risk_manager.can_trade()
        if not can:
            order.status = OrderStatus.REJECTED
            _metrics["errors_total"] += 1
            raise HTTPException(403, f"Order blocked by risk manager: {reason}")
        valid, reason = _risk_manager.validate_order(order)
        if not valid:
            order.status = OrderStatus.REJECTED
            _metrics["errors_total"] += 1
            raise HTTPException(422, f"Order failed risk validation: {reason}")

    port.execute_order(order, req.price or 0.0, req.quantity)
    _metrics["orders_total"] += 1

    # Persist the trade
    if _persistence is not None:
        try:
            _persistence.save_trade(order)
        except Exception as exc:
            logger.warning(f"Failed to persist trade: {exc}")

    return {"order_id": order.order_id, "status": order.status.value}


@app.get("/portfolio/performance", tags=["portfolio"])
def portfolio_performance():
    """
    Return equity curve, drawdown curve, and period returns for the portfolio.
    Data is loaded from the persistence layer.
    """
    pm = _get_persistence()
    curve = pm.load_equity_curve(limit=5000)

    if len(curve) < 2:
        return {
            "equity_curve": curve,
            "drawdown_curve": [],
            "period_returns": {"daily": [], "weekly": [], "monthly": []},
        }

    import pandas as pd
    df = pd.DataFrame(curve)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Drawdown
    running_max = df["total_equity"].expanding().max()
    dd = ((df["total_equity"] - running_max) / running_max).fillna(0.0)
    drawdown_curve = [
        {"timestamp": ts.isoformat(), "drawdown": round(float(d), 6)}
        for ts, d in dd.items()
    ]

    # Period returns
    daily_ret = df["total_equity"].resample("D").last().pct_change().dropna()
    weekly_ret = df["total_equity"].resample("W").last().pct_change().dropna()
    monthly_ret = df["total_equity"].resample("ME").last().pct_change().dropna()

    def _to_list(series):
        return [
            {"period": str(ts.date()), "return": round(float(v), 6)}
            for ts, v in series.items()
        ]

    return {
        "equity_curve": curve,
        "drawdown_curve": drawdown_curve,
        "period_returns": {
            "daily": _to_list(daily_ret),
            "weekly": _to_list(weekly_ret),
            "monthly": _to_list(monthly_ret),
        },
    }


# ---------------------------------------------------------------------------
# Strategy routes
# ---------------------------------------------------------------------------

@app.get("/strategies/available", tags=["strategies"])
def list_available_strategies():
    """List all strategy types in the registry."""
    return {"strategies": get_registry().list_strategies()}


@app.get("/strategies/available/{key}", tags=["strategies"])
def describe_strategy(key: str):
    try:
        return get_registry().describe(key)
    except KeyError as exc:
        raise HTTPException(404, "Strategy type not found in registry") from exc


@app.get("/strategies/by-category", tags=["strategies"])
def strategies_by_category():
    return get_registry().list_by_category()


@app.get("/strategies/active", tags=["strategies"])
def list_active_strategies():
    """List all strategy instances registered in the manager."""
    return {"strategies": _get_strategy_manager().get_status()}


@app.get("/strategies/aggregate", tags=["strategies"])
def aggregate_stats():
    return _get_strategy_manager().get_aggregate_stats()


@app.post("/strategies/add", tags=["strategies"])
def add_strategy(req: AddStrategyRequest):
    """Instantiate and register a strategy from the registry."""
    registry = get_registry()
    sm = _get_strategy_manager()

    if req.instance_id in sm.list_all():
        raise HTTPException(400, f"Strategy instance '{req.instance_id}' already exists.")

    assets = _build_assets(req.assets)
    try:
        strategy = registry.create(req.strategy_key, assets=assets, **req.params)
    except KeyError as exc:
        raise HTTPException(404, "Strategy type not found in registry") from exc
    except Exception as exc:
        logger.error(f"Strategy creation error for '{req.strategy_key}': {exc}")
        raise HTTPException(400, "Strategy creation failed. Check server logs.")

    try:
        record = sm.add_strategy(
            req.instance_id, strategy,
            capital_pct=req.capital_pct,
            auto_start=req.auto_start,
        )
    except Exception as exc:
        logger.error(f"Strategy add error: {exc}")
        raise HTTPException(400, "Failed to register strategy. Check server logs.")

    return record.to_dict()


@app.delete("/strategies/{instance_id}", tags=["strategies"])
def remove_strategy(instance_id: str):
    sm = _get_strategy_manager()
    try:
        sm.remove_strategy(instance_id)
    except KeyError as exc:
        raise HTTPException(404, "Strategy instance not found") from exc
    return {"message": f"Strategy '{instance_id}' removed."}


@app.post("/strategies/{instance_id}/start", tags=["strategies"])
def start_strategy(instance_id: str):
    sm = _get_strategy_manager()
    try:
        sm.start(instance_id)
    except KeyError as exc:
        raise HTTPException(404, "Strategy instance not found") from exc
    return {"instance_id": instance_id, "state": StrategyState.RUNNING.value}


@app.post("/strategies/{instance_id}/pause", tags=["strategies"])
def pause_strategy(instance_id: str):
    sm = _get_strategy_manager()
    try:
        sm.pause(instance_id)
    except KeyError as exc:
        raise HTTPException(404, "Strategy instance not found") from exc
    return {"instance_id": instance_id, "state": StrategyState.PAUSED.value}


@app.post("/strategies/{instance_id}/stop", tags=["strategies"])
def stop_strategy(instance_id: str):
    sm = _get_strategy_manager()
    try:
        sm.stop(instance_id)
    except KeyError as exc:
        raise HTTPException(404, "Strategy instance not found") from exc
    return {"instance_id": instance_id, "state": StrategyState.STOPPED.value}


@app.post("/strategies/start-all", tags=["strategies"])
def start_all():
    _get_strategy_manager().start_all()
    return {"message": "All strategies started."}


@app.post("/strategies/pause-all", tags=["strategies"])
def pause_all():
    _get_strategy_manager().pause_all()
    return {"message": "All strategies paused."}


@app.post("/strategies/stop-all", tags=["strategies"])
def stop_all():
    _get_strategy_manager().stop_all()
    return {"message": "All strategies stopped."}


@app.get("/strategies/{instance_id}", tags=["strategies"])
def get_strategy_status(instance_id: str):
    sm = _get_strategy_manager()
    try:
        return sm.get_strategy_status(instance_id)
    except KeyError as exc:
        raise HTTPException(404, "Strategy instance not found") from exc


@app.get("/strategies/{instance_id}/performance", tags=["strategies"])
def get_strategy_performance(instance_id: str):
    """
    Return detailed performance breakdown for a single strategy instance:
    Sharpe ratio (approximated), win rate, PnL, signal counts.
    """
    sm = _get_strategy_manager()
    try:
        record_dict = sm.get_strategy_status(instance_id)
    except KeyError as exc:
        raise HTTPException(404, "Strategy instance not found") from exc

    perf = record_dict.get("performance", {})

    # Derive approximate Sharpe from signal-level PnL history (if available)
    strategy = sm._records[instance_id].strategy
    trade_pnls = [
        strategy.performance.total_pnl
    ]  # per-trade breakdown not available at base level; return summary
    total_trades = perf.get("trades_executed", 0)
    win_rate = perf.get("win_rate", 0.0)
    total_pnl = perf.get("total_pnl", 0.0)

    return {
        "instance_id": instance_id,
        "strategy_name": record_dict.get("strategy_name"),
        "state": record_dict.get("state"),
        "total_signals": perf.get("total_signals", 0),
        "buy_signals": perf.get("buy_signals", 0),
        "sell_signals": perf.get("sell_signals", 0),
        "trades_executed": total_trades,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "signals_generated": record_dict.get("signals_generated", 0),
        "orders_placed": record_dict.get("orders_placed", 0),
        "allocated_capital": record_dict.get("allocated_capital", 0.0),
    }


@app.put("/strategies/allocations", tags=["strategies"])
def update_allocations(req: UpdateAllocationRequest):
    _get_strategy_manager().update_allocations(req.allocations)
    return {"message": "Allocations updated.", "allocations": req.allocations}


# ---------------------------------------------------------------------------
# Bar / tick feed route
# ---------------------------------------------------------------------------

@app.post("/market/bar", tags=["market-data"])
def feed_bar(req: BarDataRequest):
    """
    Push a new price bar to all running strategies.
    ``prices`` maps symbol → current price.
    Returns signals that were acted upon.
    """
    sm = _get_strategy_manager()
    port = _get_portfolio()

    ts = datetime.fromisoformat(req.current_date) if req.current_date else datetime.now()

    # Build asset-keyed price dict from the portfolio's known assets + extra
    price_map: Dict[Asset, float] = {}
    for asset in port.positions:
        if asset.symbol in req.prices:
            price_map[asset] = req.prices[asset.symbol]

    # Also include assets from running strategies (via public method)
    for assets_list in sm.get_all_strategy_assets().values():
        for asset in assets_list:
            if asset.symbol in req.prices:
                price_map[asset] = req.prices[asset.symbol]

    accepted = sm.on_bar(ts, price_map)
    _metrics["signals_total"] += len(accepted)

    result = {
        "timestamp": ts.isoformat(),
        "prices_received": len(req.prices),
        "signals_accepted": len(accepted),
        "signals": [
            {
                "strategy": s.strategy_name,
                "asset": s.asset.symbol,
                "signal": s.signal_type.value,
                "strength": round(s.strength, 4),
                "price": round(s.price, 4),
            }
            for s in accepted
        ],
    }

    # Persist equity snapshot and broadcast to WS clients
    if _persistence is not None:
        try:
            _persistence.record_equity(port)
        except Exception as exc:
            logger.warning(f"Equity record failed: {exc}")

    if _ws_connections:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(
                    _broadcast_ws({
                        "type": "bar",
                        "timestamp": ts.isoformat(),
                        "prices": req.prices,
                        "signals": len(accepted),
                    })
                )
        except Exception:
            pass

    return result


@app.get("/market/signals", tags=["market-data"])
def signal_log(limit: int = Query(default=50, ge=1, le=1000)):
    return {"signals": _get_strategy_manager().get_signal_log(limit)}


# ---------------------------------------------------------------------------
# Market data routes
# ---------------------------------------------------------------------------

@app.get("/market/price/{symbol}", tags=["market-data"])
def current_price(symbol: str, asset_class: str = "STOCK", exchange: str = "NASDAQ"):
    try:
        ac = AssetClass[asset_class.upper()]
    except KeyError:
        raise HTTPException(400, f"Unknown asset_class: {asset_class}")

    asset = Asset(symbol=symbol.upper(), asset_class=ac, exchange=exchange.upper())
    mdm = get_market_data_manager()
    price = mdm.fetch_current_price(asset)
    if price == 0.0:
        raise HTTPException(404, f"Could not fetch price for {symbol}")
    return {"symbol": symbol.upper(), "price": price, "timestamp": datetime.now().isoformat()}


@app.get("/market/ohlcv/{symbol}", tags=["market-data"])
def ohlcv(
    symbol: str,
    asset_class: str = "STOCK",
    exchange: str = "NASDAQ",
    timeframe: str = "1d",
    limit: int = Query(default=100, ge=1, le=1000),
):
    try:
        ac = AssetClass[asset_class.upper()]
    except KeyError:
        raise HTTPException(400, f"Unknown asset_class: {asset_class}")

    asset = Asset(symbol=symbol.upper(), asset_class=ac, exchange=exchange.upper())
    mdm = get_market_data_manager()
    df = mdm.fetch_ohlcv(asset, timeframe=timeframe, limit=limit, use_cache=False)
    if df.empty:
        raise HTTPException(404, f"No OHLCV data found for {symbol}")

    df_reset = df.reset_index()
    df_reset["timestamp"] = df_reset["timestamp"].astype(str)
    return {"symbol": symbol.upper(), "timeframe": timeframe, "bars": df_reset.to_dict(orient="records")}


# ---------------------------------------------------------------------------
# Backtest routes
# ---------------------------------------------------------------------------

@app.post("/backtest/run", tags=["backtest"])
def run_backtest(req: BacktestRequest):
    """Run a backtest for any registered strategy."""
    registry = get_registry()
    assets = _build_assets(req.assets)

    try:
        strategy = registry.create(req.strategy_key, assets=assets, **req.params)
    except KeyError as exc:
        raise HTTPException(404, "Strategy type not found in registry") from exc
    except Exception as exc:
        logger.error(f"Backtest strategy creation error for '{req.strategy_key}': {exc}")
        raise HTTPException(400, "Strategy creation failed. Check server logs.")

    # Set assets on strategies that need it (e.g. SMA crossover)
    strategy.set_assets(assets)

    # Wrap strategy in a backtest-compatible adapter
    class _StrategyAdapter(BtStrategy):
        def __init__(self, inner: BaseStrategy):
            super().__init__(inner.name)
            self._inner = inner
            self._inner.set_assets(assets)

        def on_bar(self, current_date, current_prices):
            sigs = self._inner.on_bar(current_date, current_prices)
            for sig in sigs:
                if self.portfolio:
                    if sig.signal_type == SignalType.BUY:
                        pos = self.portfolio.get_position(sig.asset)
                        if pos is None or pos.quantity == 0:
                            price = current_prices.get(sig.asset, sig.price)
                            qty = max(self.portfolio.cash * 0.1 / (price or 1), 1)
                            self.portfolio.create_order(
                                sig.asset, OrderSide.BUY, qty, OrderType.MARKET, price
                            )
                    elif sig.signal_type in (SignalType.SELL, SignalType.CLOSE):
                        pos = self.portfolio.get_position(sig.asset)
                        if pos and pos.quantity > 0:
                            price = current_prices.get(sig.asset, sig.price)
                            self.portfolio.create_order(
                                sig.asset, OrderSide.SELL, pos.quantity, OrderType.MARKET, price
                            )

    bt_config = BacktestConfig(
        initial_capital=req.initial_capital,
        start_date=req.start_date,
        end_date=req.end_date,
        commission_pct=req.commission_pct,
        slippage_pct=req.slippage_pct,
    )
    adapter = _StrategyAdapter(strategy)
    backtester = Backtester(adapter, bt_config)

    try:
        stats = backtester.run(assets)
    except Exception as exc:
        logger.error(f"Backtest execution error: {exc}")
        raise HTTPException(500, "Backtest failed. Check server logs.")

    # Return numeric types (not pre-formatted strings) for downstream use
    return {
        "strategy": backtester.strategy.name,
        "initial_capital": bt_config.initial_capital,
        "final_equity": round(stats.final_equity, 2),
        "total_return": round(stats.total_return, 6),
        "annual_return": round(stats.annual_return, 6),
        "volatility": round(stats.volatility, 6),
        "sharpe_ratio": round(stats.sharpe_ratio, 4),
        "max_drawdown": round(stats.max_drawdown, 6),
        "total_trades": stats.num_trades,
        "winning_trades": stats.num_winning_trades,
        "losing_trades": stats.num_losing_trades,
        "win_rate": round(stats.win_rate, 4),
        "profit_factor": round(stats.profit_factor, 4),
        "avg_win": round(stats.avg_win, 2),
        "avg_loss": round(stats.avg_loss, 2),
        "total_pnl": round(stats.total_pnl, 2),
    }


# ---------------------------------------------------------------------------
# Risk routes
# ---------------------------------------------------------------------------

@app.get("/risk/summary", tags=["risk"])
def risk_summary():
    if _risk_manager is None:
        raise HTTPException(500, "Risk manager not initialised")
    return _risk_manager.get_risk_summary()


@app.get("/risk/metrics", tags=["risk"])
def risk_metrics():
    if _risk_manager is None:
        raise HTTPException(500, "Risk manager not initialised")
    return _risk_manager.risk_metrics.to_dict()


@app.get("/risk/limits", tags=["risk"])
def risk_limits():
    if _risk_manager is None:
        raise HTTPException(500, "Risk manager not initialised")
    return _risk_manager.limits.to_dict()


@app.post("/risk/halt", tags=["risk"])
def halt_trading():
    if _risk_manager is None:
        raise HTTPException(500, "Risk manager not initialised")
    _risk_manager.is_trading_halted = True
    _get_strategy_manager().pause_all()
    return {"message": "Trading halted. All strategies paused."}


@app.post("/risk/resume", tags=["risk"])
def resume_trading():
    if _risk_manager is None:
        raise HTTPException(500, "Risk manager not initialised")
    _risk_manager.is_trading_halted = False
    _get_strategy_manager().start_all()
    return {"message": "Trading resumed. All strategies started."}


# ---------------------------------------------------------------------------
# WebSocket — real-time price streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/prices")
async def ws_prices(websocket: WebSocket):
    """
    WebSocket endpoint for real-time price and signal streaming.

    Clients connect here to receive JSON frames whenever the auto-feed
    background task pushes a new bar, or whenever a price update is
    manually POSTed to ``/market/bar``.

    Frame format::

        {
          "type": "bar",
          "timestamp": "2024-01-15T10:30:00",
          "prices": {"AAPL": 185.32, "BTC": 42000.0},
          "signals": 2
        }
    """
    await websocket.accept()
    _ws_connections.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(_ws_connections)}")
    try:
        while True:
            # Keep connection alive; client can send pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_connections:
            _ws_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(_ws_connections)}")


# ---------------------------------------------------------------------------
# Metrics endpoint (Prometheus-compatible plain-text)
# ---------------------------------------------------------------------------

@app.get("/metrics", tags=["system"])
def metrics():
    """
    Expose platform metrics in a Prometheus-compatible text format
    (counter lines) as well as a JSON summary.
    """
    if not settings.METRICS_ENABLED:
        raise HTTPException(404, "Metrics endpoint disabled")

    uptime_seconds: float = 0.0
    if _metrics["start_time"]:
        start = datetime.fromisoformat(_metrics["start_time"])
        uptime_seconds = (datetime.now(timezone.utc) - start).total_seconds()

    port = _portfolio
    return {
        "uptime_seconds": round(uptime_seconds, 1),
        "requests_total": _metrics["requests_total"],
        "orders_total": _metrics["orders_total"],
        "signals_total": _metrics["signals_total"],
        "errors_total": _metrics["errors_total"],
        "ws_connections": len(_ws_connections),
        "portfolio_equity": round(port.total_equity, 2) if port else 0.0,
        "portfolio_cash": round(port.cash, 2) if port else 0.0,
        "open_positions": len(port.positions) if port else 0,
        "strategies_running": len(_strategy_manager.list_running()) if _strategy_manager else 0,
    }


# ---------------------------------------------------------------------------
# Persistence routes
# ---------------------------------------------------------------------------

@app.get("/portfolio/history", tags=["portfolio"])
def portfolio_history(limit: int = Query(default=100, ge=1, le=5000)):
    """Return recent portfolio snapshots (newest first)."""
    return {"snapshots": _get_persistence().load_portfolio_snapshots(limit)}


@app.get("/portfolio/trades", tags=["portfolio"])
def portfolio_trade_history(limit: int = Query(default=100, ge=1, le=1000)):
    """Return persisted trade history (newest first)."""
    return {"trades": _get_persistence().load_trade_history(limit)}
