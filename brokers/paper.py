"""
Paper Trading Broker
====================
In-memory simulated broker that executes orders instantly at the
requested price with configurable commission and slippage.
Useful for development, testing, and live paper-trading.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from core.trading_engine import Asset, Order, OrderSide, OrderStatus, OrderType
from brokers.base import BrokerAdapter, BrokerOrderResult, BrokerPosition


class PaperBroker(BrokerAdapter):
    """
    Simulated paper-trading broker.

    Parameters
    ----------
    commission_pct : float
        Commission as a fraction of notional (default 0.1 %).
    slippage_pct : float
        One-way slippage as a fraction of price (default 0.05 %).
    """

    def __init__(
        self,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
    ):
        super().__init__("Paper Trading")
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct

        self._orders: Dict[str, BrokerOrderResult] = {}   # broker_order_id → result
        self._positions: Dict[str, BrokerPosition] = {}   # symbol → position
        self._equity: float = 0.0                          # tracks cash equivalent

    # ------------------------------------------------------------------
    # BrokerAdapter implementation
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> BrokerOrderResult:
        """Instantly fill the order at price ± slippage with commission."""
        broker_order_id = str(uuid.uuid4())
        price = order.price or 0.0

        if price <= 0:
            result = BrokerOrderResult(
                broker_order_id=broker_order_id,
                local_order_id=order.order_id,
                status=OrderStatus.REJECTED,
                error_message="Price must be positive",
            )
            self._orders[broker_order_id] = result
            logger.warning(f"[PaperBroker] Order {order.order_id} rejected: price <= 0")
            return result

        # Apply slippage
        if order.side == OrderSide.BUY:
            fill_price = price * (1 + self.slippage_pct)
        else:
            fill_price = price * (1 - self.slippage_pct)

        commission = order.quantity * fill_price * self.commission_pct
        notional = order.quantity * fill_price

        # Update internal position
        self._update_position(order, fill_price)

        # Adjust equity (simplified — mirrors cash flow)
        if order.side == OrderSide.BUY:
            self._equity -= notional + commission
        else:
            self._equity += notional - commission

        result = BrokerOrderResult(
            broker_order_id=broker_order_id,
            local_order_id=order.order_id,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            average_fill_price=fill_price,
            commission=commission,
            submitted_at=datetime.now(),
            filled_at=datetime.now(),
        )
        self._orders[broker_order_id] = result
        logger.info(
            f"[PaperBroker] Filled {order.side.value} {order.quantity} "
            f"{order.asset.symbol} @ {fill_price:.4f} (comm={commission:.2f})"
        )
        return result

    def cancel_order(self, order_id: str) -> bool:
        """Paper broker fills instantly — nothing to cancel."""
        logger.info(f"[PaperBroker] cancel_order called for {order_id} (no-op)")
        return False

    def get_order_status(self, order_id: str) -> Optional[BrokerOrderResult]:
        return self._orders.get(order_id)

    def get_positions(self) -> List[BrokerPosition]:
        return [p for p in self._positions.values() if p.quantity != 0]

    def get_account_equity(self) -> float:
        return self._equity

    # ------------------------------------------------------------------
    # Setup helper (call once to set initial cash)
    # ------------------------------------------------------------------

    def set_initial_equity(self, equity: float) -> None:
        """Seed the paper-trading account with initial cash."""
        self._equity = equity

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_position(self, order: Order, fill_price: float) -> None:
        symbol = order.asset.symbol
        if symbol not in self._positions:
            self._positions[symbol] = BrokerPosition(
                symbol=symbol,
                quantity=0.0,
                average_entry_price=0.0,
                current_price=fill_price,
                unrealized_pnl=0.0,
            )
        pos = self._positions[symbol]
        qty = order.quantity

        if order.side == OrderSide.BUY:
            total_cost = pos.quantity * pos.average_entry_price + qty * fill_price
            pos.quantity += qty
            pos.average_entry_price = total_cost / pos.quantity if pos.quantity else 0.0
        else:
            pos.quantity -= qty
            if pos.quantity < 0:
                pos.quantity = 0.0
                pos.average_entry_price = 0.0

        pos.current_price = fill_price
        pos.unrealized_pnl = (fill_price - pos.average_entry_price) * pos.quantity
