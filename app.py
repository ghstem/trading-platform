"""
Trading Platform — FastAPI Application
Automated multi-strategy trading system REST API.

Start with:
    uvicorn app:app --reload --host 0.0.0.0 --port 5000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from config.settings import get_config
from core.trading_engine import (
    Asset, AssetClass, Portfolio,
    OrderSide, OrderType,
)
from core.strategy_manager import StrategyManager, StrategyState
from data_pipeline.market_data import get_market_data_manager
from risk_management.risk_manager import RiskLimits, RiskManager
from strategies.registry import get_registry
from strategies.base import BaseStrategy, SignalType
from backtester.backtest_engine import Backtester, BacktestConfig, Strategy as BtStrategy

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

config = get_config()
_portfolio: Optional[Portfolio] = None
_risk_manager: Optional[RiskManager] = None
_strategy_manager: Optional[StrategyManager] = None


def _get_portfolio() -> Portfolio:
    if _portfolio is None:
        raise HTTPException(500, "Portfolio not initialised")
    return _portfolio


def _get_strategy_manager() -> StrategyManager:
    if _strategy_manager is None:
        raise HTTPException(500, "StrategyManager not initialised")
    return _strategy_manager


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown."""
    global _portfolio, _risk_manager, _strategy_manager

    logger.info("Starting Trading Platform API…")
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
    yield
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    port.execute_order(order, req.price or 0.0, req.quantity)
    return {"order_id": order.order_id, "status": order.status.value}


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
        raise HTTPException(404, "Not found") from exc


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
        raise HTTPException(404, "Not found") from exc
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
        raise HTTPException(404, "Not found") from exc
    return {"message": f"Strategy '{instance_id}' removed."}


@app.post("/strategies/{instance_id}/start", tags=["strategies"])
def start_strategy(instance_id: str):
    sm = _get_strategy_manager()
    try:
        sm.start(instance_id)
    except KeyError as exc:
        raise HTTPException(404, "Not found") from exc
    return {"instance_id": instance_id, "state": StrategyState.RUNNING.value}


@app.post("/strategies/{instance_id}/pause", tags=["strategies"])
def pause_strategy(instance_id: str):
    sm = _get_strategy_manager()
    try:
        sm.pause(instance_id)
    except KeyError as exc:
        raise HTTPException(404, "Not found") from exc
    return {"instance_id": instance_id, "state": StrategyState.PAUSED.value}


@app.post("/strategies/{instance_id}/stop", tags=["strategies"])
def stop_strategy(instance_id: str):
    sm = _get_strategy_manager()
    try:
        sm.stop(instance_id)
    except KeyError as exc:
        raise HTTPException(404, "Not found") from exc
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
        raise HTTPException(404, "Not found") from exc


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

    return {
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
        raise HTTPException(404, "Not found") from exc
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

    return backtester.get_stats_summary()


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
