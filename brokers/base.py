"""
Broker Adapter — Abstract Interface
====================================
All broker integrations (paper trading, Alpaca, IBKR, …) implement this
contract so the rest of the platform can be broker-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from core.trading_engine import Asset, Order, OrderSide, OrderStatus, OrderType


@dataclass
class BrokerOrderResult:
    """Normalised result returned by the broker after submitting an order."""
    broker_order_id: str
    local_order_id: str            # maps back to Order.order_id
    status: OrderStatus
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    error_message: str = ""
    submitted_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None

    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    @property
    def is_rejected(self) -> bool:
        return self.status == OrderStatus.REJECTED


@dataclass
class BrokerPosition:
    """Normalised position as reported by the broker."""
    symbol: str
    quantity: float
    average_entry_price: float
    current_price: float
    unrealized_pnl: float
    side: str = "LONG"   # "LONG" or "SHORT"


class BrokerAdapter(ABC):
    """
    Abstract base class for all broker integrations.

    Concrete sub-classes must implement:
      - ``submit_order``       — place a new order
      - ``cancel_order``       — cancel an open order
      - ``get_order_status``   — query live order status
      - ``get_positions``      — retrieve open positions
      - ``get_account_equity`` — retrieve account equity / cash

    Optional:
      - ``connect``   — authenticate / open connection (default: no-op)
      - ``disconnect``— close connection (default: no-op)
    """

    def __init__(self, name: str):
        self.name = name
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Authenticate and open connection to the broker."""
        self._connected = True

    def disconnect(self) -> None:
        """Close the broker connection cleanly."""
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Abstract trading operations
    # ------------------------------------------------------------------

    @abstractmethod
    def submit_order(self, order: Order) -> BrokerOrderResult:
        """
        Submit an order to the broker.

        Parameters
        ----------
        order : Order
            The platform order to route.

        Returns
        -------
        BrokerOrderResult
            Normalised result including fill details.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Returns True if successfully cancelled, False otherwise.
        """

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[BrokerOrderResult]:
        """Return the latest status of a previously submitted order."""

    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]:
        """Return all currently open positions at the broker."""

    @abstractmethod
    def get_account_equity(self) -> float:
        """Return total account equity (cash + unrealised P&L)."""
