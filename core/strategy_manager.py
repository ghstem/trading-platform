"""
Strategy Manager — Automated Multi-Strategy Orchestrator
=========================================================
Central hub that:
  * Manages the full lifecycle of any number of strategies
    (register → start → pause → resume → stop)
  * Routes incoming price bars to every active strategy
  * Translates signals into portfolio orders, validated by RiskManager
  * Allocates capital across strategies proportionally
  * Records per-strategy and aggregate performance statistics

Typical usage
-------------
    manager = StrategyManager(portfolio, risk_manager)
    manager.add_strategy("macd_aapl", macd_strategy, capital_pct=0.3)
    manager.add_strategy("bb_btc",   bb_strategy,   capital_pct=0.2)
    manager.start_all()

    # Each market tick / bar:
    signals = manager.on_bar(current_date, current_prices)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from core.trading_engine import (
    Asset, Order, OrderSide, OrderStatus, OrderType, Portfolio,
)
from risk_management.risk_manager import RiskManager
from strategies.base import BaseStrategy, Signal, SignalType


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

class StrategyState(Enum):
    REGISTERED = "REGISTERED"
    RUNNING    = "RUNNING"
    PAUSED     = "PAUSED"
    STOPPED    = "STOPPED"
    ERROR      = "ERROR"


@dataclass
class StrategyRecord:
    """Internal record kept for each registered strategy."""
    instance_id: str
    strategy: BaseStrategy
    state: StrategyState = StrategyState.REGISTERED
    capital_pct: float = 0.0            # fraction of total portfolio capital
    allocated_capital: float = 0.0      # absolute dollar value
    signals_generated: int = 0
    orders_placed: int = 0
    total_pnl: float = 0.0
    added_at: datetime = field(default_factory=datetime.now)
    last_active_at: Optional[datetime] = None
    error_message: str = ""

    def to_dict(self) -> Dict:
        return {
            "instance_id": self.instance_id,
            "strategy_name": self.strategy.name,
            "state": self.state.value,
            "capital_pct": round(self.capital_pct, 4),
            "allocated_capital": round(self.allocated_capital, 2),
            "signals_generated": self.signals_generated,
            "orders_placed": self.orders_placed,
            "total_pnl": round(self.total_pnl, 2),
            "added_at": self.added_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None,
            "error_message": self.error_message,
            "performance": self.strategy.get_performance(),
        }


# ---------------------------------------------------------------------------
# StrategyManager
# ---------------------------------------------------------------------------

class StrategyManager:
    """
    Automated multi-strategy trading system assistant.

    Manages all strategy types (trend-following, mean-reversion, momentum,
    pairs-trading, ML) through a unified lifecycle and execution model.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        risk_manager: Optional[RiskManager] = None,
        default_order_quantity: float = 10.0,
        auto_size_by_signal_strength: bool = True,
    ):
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.default_order_quantity = default_order_quantity
        self.auto_size_by_signal_strength = auto_size_by_signal_strength

        self._records: Dict[str, StrategyRecord] = {}  # keyed by instance_id
        self._bar_count: int = 0
        self._all_signals: List[Tuple[datetime, Signal]] = []   # full audit log
        self._all_orders: List[Order] = []
        self._stop_levels: Dict[Asset, Dict] = {}  # asset → {stop_loss, take_profit, instance_id}

        logger.info("StrategyManager initialised")

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def add_strategy(
        self,
        instance_id: str,
        strategy: BaseStrategy,
        capital_pct: float = 0.0,
        auto_start: bool = True,
    ) -> StrategyRecord:
        """
        Register a strategy with the manager.

        Parameters
        ----------
        instance_id  : unique identifier for this strategy instance
        strategy     : an initialised BaseStrategy subclass
        capital_pct  : fraction of total portfolio to allocate (0 = no limit)
        auto_start   : immediately transition to RUNNING state
        """
        if instance_id in self._records:
            raise ValueError(f"Strategy '{instance_id}' already registered.")

        record = StrategyRecord(
            instance_id=instance_id,
            strategy=strategy,
            capital_pct=capital_pct,
        )
        self._records[instance_id] = record

        # Initialise strategy with portfolio
        try:
            strategy.initialize(self.portfolio)
            if auto_start:
                record.state = StrategyState.RUNNING
                record.last_active_at = datetime.now()
                logger.info(f"Strategy '{instance_id}' added and started.")
            else:
                logger.info(f"Strategy '{instance_id}' added (not yet started).")
        except Exception as exc:
            record.state = StrategyState.ERROR
            record.error_message = "Initialisation error (see server logs)"
            logger.error(f"Strategy '{instance_id}' failed to initialise: {exc}")

        self._rebalance_allocations()
        return record

    def remove_strategy(self, instance_id: str) -> None:
        """Stop and remove a strategy."""
        record = self._get_record(instance_id)
        self.stop(instance_id)
        del self._records[instance_id]
        self._rebalance_allocations()
        logger.info(f"Strategy '{instance_id}' removed.")

    def start(self, instance_id: str) -> None:
        record = self._get_record(instance_id)
        if record.state in (StrategyState.PAUSED, StrategyState.REGISTERED):
            record.state = StrategyState.RUNNING
            record.last_active_at = datetime.now()
            logger.info(f"Strategy '{instance_id}' started.")

    def pause(self, instance_id: str) -> None:
        record = self._get_record(instance_id)
        if record.state == StrategyState.RUNNING:
            record.state = StrategyState.PAUSED
            logger.info(f"Strategy '{instance_id}' paused.")

    def resume(self, instance_id: str) -> None:
        self.start(instance_id)

    def stop(self, instance_id: str) -> None:
        record = self._get_record(instance_id)
        if record.state != StrategyState.STOPPED:
            try:
                record.strategy.finalize()
            except Exception as exc:
                logger.warning(f"Error finalising '{instance_id}': {exc}")
            record.state = StrategyState.STOPPED
            logger.info(f"Strategy '{instance_id}' stopped.")

    def start_all(self) -> None:
        for iid in list(self._records):
            self.start(iid)

    def pause_all(self) -> None:
        for iid in list(self._records):
            self.pause(iid)

    def stop_all(self) -> None:
        for iid in list(self._records):
            self.stop(iid)

    # ------------------------------------------------------------------
    # Bar processing — the main event loop entry point
    # ------------------------------------------------------------------

    def on_bar(
        self,
        current_date: datetime,
        current_prices: Dict[Asset, float],
    ) -> List[Signal]:
        """
        Feed a new bar to all running strategies, collect signals,
        run risk validation, and submit orders.

        Returns the list of signals that passed risk validation.
        """
        self._bar_count += 1
        accepted_signals: List[Signal] = []

        # Update risk metrics with latest prices
        if self.risk_manager:
            self.risk_manager.update_metrics(current_prices)

        # Check stop-loss / take-profit levels before processing new signals
        self._check_stop_levels(current_prices)

        for iid, record in self._records.items():
            if record.state != StrategyState.RUNNING:
                continue

            try:
                signals = record.strategy.on_bar(current_date, current_prices)
            except Exception as exc:
                logger.error(f"Strategy '{iid}' raised on_bar error: {exc}")
                record.state = StrategyState.ERROR
                record.error_message = str(exc)
                continue

            record.signals_generated += len(signals)
            record.last_active_at = datetime.now()

            for signal in signals:
                self._all_signals.append((current_date, signal))
                order = self._process_signal(signal, record, current_prices)
                if order:
                    accepted_signals.append(signal)
                    record.orders_placed += 1

        return accepted_signals

    # ------------------------------------------------------------------
    # Signal → Order translation
    # ------------------------------------------------------------------

    def _process_signal(
        self,
        signal: Signal,
        record: StrategyRecord,
        current_prices: Dict[Asset, float],
    ) -> Optional[Order]:
        """
        Convert a signal to a portfolio order after risk validation.
        Returns the Order if submitted, else None.
        """
        # --- Risk gate ---
        if self.risk_manager:
            can_trade, reason = self.risk_manager.can_trade()
            if not can_trade:
                logger.warning(f"[{record.instance_id}] Signal blocked by risk: {reason}")
                return None

        asset = signal.asset
        price = current_prices.get(asset, signal.price) or signal.price

        if signal.signal_type == SignalType.BUY:
            side = OrderSide.BUY
            quantity = self._compute_quantity(signal, record, price)
            if quantity <= 0:
                return None

        elif signal.signal_type in (SignalType.SELL, SignalType.CLOSE):
            side = OrderSide.SELL
            position = self.portfolio.get_position(asset)
            if position is None or position.quantity <= 0:
                return None
            quantity = abs(position.quantity)
            # Clear any tracked SL/TP for this asset when strategy closes the position
            self._stop_levels.pop(asset, None)

        elif signal.signal_type == SignalType.HOLD:
            return None

        else:
            return None

        # --- Create & execute order ---
        try:
            order = self.portfolio.create_order(
                asset=asset,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET,
                price=price,
                stop_loss=signal.stop_loss if side == OrderSide.BUY else None,
                take_profit=signal.take_profit if side == OrderSide.BUY else None,
                broker="strategy_manager",
            )
            # Validate against risk limits
            if self.risk_manager:
                valid, reason = self.risk_manager.validate_order(order)
                if not valid:
                    logger.warning(f"[{record.instance_id}] Order rejected: {reason}")
                    order.status = OrderStatus.CANCELLED
                    return None

            self.portfolio.execute_order(order, price, quantity)
            self._all_orders.append(order)

            # Register SL/TP tracking after a successful BUY
            if side == OrderSide.BUY and (signal.stop_loss or signal.take_profit):
                self._stop_levels[asset] = {
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                    "instance_id": record.instance_id,
                }
                logger.info(
                    f"[{record.instance_id}] SL/TP registered for {asset.symbol}: "
                    f"SL={signal.stop_loss}, TP={signal.take_profit}"
                )

            logger.info(
                f"[{record.instance_id}] Executed {side.value} {quantity:.4f} "
                f"{asset.symbol} @ ${price:.4f}"
            )
            return order

        except Exception as exc:
            logger.error(f"[{record.instance_id}] Order execution error: {exc}")
            return None

    def _compute_quantity(
        self,
        signal: Signal,
        record: StrategyRecord,
        price: float,
    ) -> float:
        """
        Determine order quantity based on capital allocation,
        signal strength, and risk-manager position sizing.
        """
        if price <= 0:
            return 0.0

        # Use risk manager's position-size calculation if available
        if self.risk_manager and signal.stop_loss:
            qty = self.risk_manager.calculate_position_size(
                signal.asset, price, signal.stop_loss
            )
            if qty > 0:
                return qty

        # Allocation-based sizing
        if record.allocated_capital > 0:
            max_qty = record.allocated_capital / price
        else:
            max_qty = self.portfolio.cash / price

        if self.auto_size_by_signal_strength:
            qty = max_qty * signal.strength
        else:
            qty = self.default_order_quantity

        return max(round(qty, 8), 0.0)

    def _check_stop_levels(self, current_prices: Dict[Asset, float]) -> List[Order]:
        """
        Evaluate all tracked stop-loss / take-profit levels against current prices.
        Automatically closes any position whose SL or TP has been breached.
        Returns the list of close orders that were placed.
        """
        triggered_orders: List[Order] = []

        for asset, levels in list(self._stop_levels.items()):
            price = current_prices.get(asset)
            if price is None:
                continue

            position = self.portfolio.get_position(asset)
            if position is None or position.quantity <= 0:
                # Position already gone — remove stale tracking entry
                self._stop_levels.pop(asset, None)
                continue

            sl = levels.get("stop_loss")
            tp = levels.get("take_profit")
            trigger_reason: Optional[str] = None

            if sl is not None and price <= sl:
                trigger_reason = f"stop-loss hit ({price:.4f} <= {sl:.4f})"
            elif tp is not None and price >= tp:
                trigger_reason = f"take-profit hit ({price:.4f} >= {tp:.4f})"

            if trigger_reason:
                instance_id = levels.get("instance_id", "stop_tracker")
                logger.info(
                    f"[{instance_id}] {asset.symbol} — {trigger_reason}. "
                    f"Closing {position.quantity:.4f} @ {price:.4f}."
                )
                try:
                    order = self.portfolio.create_order(
                        asset=asset,
                        side=OrderSide.SELL,
                        quantity=abs(position.quantity),
                        order_type=OrderType.MARKET,
                        price=price,
                        broker="stop_tracker",
                    )
                    self.portfolio.execute_order(order, price, abs(position.quantity))
                    self._all_orders.append(order)
                    self._stop_levels.pop(asset, None)
                    triggered_orders.append(order)
                except Exception as exc:
                    logger.error(
                        f"[stop_tracker] Failed to close {asset.symbol} on {trigger_reason}: {exc}"
                    )

        return triggered_orders

    # ------------------------------------------------------------------
    # Capital allocation
    # ------------------------------------------------------------------

    def _rebalance_allocations(self) -> None:
        """
        Re-compute the absolute capital allocated to each strategy based on
        its ``capital_pct``.  Strategies with capital_pct == 0 are uncapped.
        """
        total_equity = self.portfolio.total_equity
        for record in self._records.values():
            record.allocated_capital = (
                total_equity * record.capital_pct if record.capital_pct > 0 else 0.0
            )

    def update_allocations(self, allocations: Dict[str, float]) -> None:
        """
        Manually update capital percentages and re-balance.

        Parameters
        ----------
        allocations : {instance_id: capital_pct, ...}
        """
        for iid, pct in allocations.items():
            if iid in self._records:
                self._records[iid].capital_pct = pct
        self._rebalance_allocations()
        logger.info(f"Capital allocations updated: {allocations}")

    # ------------------------------------------------------------------
    # Introspection & reporting
    # ------------------------------------------------------------------

    def get_status(self) -> List[Dict]:
        return [r.to_dict() for r in self._records.values()]

    def get_strategy_status(self, instance_id: str) -> Dict:
        return self._get_record(instance_id).to_dict()

    def list_running(self) -> List[str]:
        return [iid for iid, r in self._records.items() if r.state == StrategyState.RUNNING]

    def list_all(self) -> List[str]:
        return list(self._records.keys())

    def get_aggregate_stats(self) -> Dict:
        """Summary statistics across all strategies."""
        total_signals = sum(r.signals_generated for r in self._records.values())
        total_orders = sum(r.orders_placed for r in self._records.values())
        total_pnl = self.portfolio.total_pnl
        return {
            "num_strategies": len(self._records),
            "running": len(self.list_running()),
            "total_bars_processed": self._bar_count,
            "total_signals": total_signals,
            "total_orders": total_orders,
            "portfolio_total_pnl": round(total_pnl, 2),
            "portfolio_total_equity": round(self.portfolio.total_equity, 2),
            "portfolio_cash": round(self.portfolio.cash, 2),
        }

    def get_signal_log(self, limit: int = 100) -> List[Dict]:
        """Return the most recent signals (newest first)."""
        return [
            {
                "timestamp": ts.isoformat(),
                "strategy": sig.strategy_name,
                "asset": sig.asset.symbol,
                "signal": sig.signal_type.value,
                "strength": round(sig.strength, 4),
                "price": round(sig.price, 4),
                "stop_loss": sig.stop_loss,
                "take_profit": sig.take_profit,
                "metadata": sig.metadata,
            }
            for ts, sig in reversed(self._all_signals[-limit:])
        ]

    def get_all_strategy_assets(self) -> Dict[str, List[Asset]]:
        """Return a mapping of instance_id → assets for all registered strategies."""
        return {
            iid: list(record.strategy.assets)
            for iid, record in self._records.items()
        }

    def get_stop_levels(self) -> List[Dict]:
        """Return all currently active stop-loss / take-profit levels."""
        return [
            {
                "symbol": asset.symbol,
                "stop_loss": levels.get("stop_loss"),
                "take_profit": levels.get("take_profit"),
                "instance_id": levels.get("instance_id"),
            }
            for asset, levels in self._stop_levels.items()
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_record(self, instance_id: str) -> StrategyRecord:
        record = self._records.get(instance_id)
        if record is None:
            raise KeyError(f"No strategy registered with id '{instance_id}'")
        return record
