"""
Alpaca Broker Stub
==================
Stub implementation for Alpaca Markets (https://alpaca.markets).
Raises ``NotImplementedError`` for every method to make clear what needs
to be wired up before going live.  Connection parameters are read from the
application config so no secrets appear in code.
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger

from core.trading_engine import Order, OrderStatus
from brokers.base import BrokerAdapter, BrokerOrderResult, BrokerPosition


class AlpacaBroker(BrokerAdapter):
    """
    Alpaca Markets broker stub.

    Replace ``NotImplementedError`` bodies with calls to the
    ``alpaca-trade-api`` (already listed in requirements.txt) once
    you have a live/paper Alpaca account.

    Example wiring::

        from brokers.alpaca import AlpacaBroker
        broker = AlpacaBroker(api_key="...", secret_key="...", base_url="...")
        broker.connect()
        result = broker.submit_order(order)
    """

    def __init__(self, api_key: str = "", secret_key: str = "", base_url: str = ""):
        super().__init__("Alpaca")
        self._api_key = api_key
        self._secret_key = secret_key
        self._base_url = base_url or "https://paper-api.alpaca.markets"
        self._api = None   # alpaca_trade_api.REST instance

    def connect(self) -> None:
        """
        Authenticate with Alpaca.

        Uncomment when alpaca-trade-api is in scope::

            import alpaca_trade_api as tradeapi
            self._api = tradeapi.REST(
                self._api_key, self._secret_key, self._base_url
            )
        """
        logger.warning("[AlpacaBroker] connect() is a stub — not connected to Alpaca.")
        self._connected = False

    def submit_order(self, order: Order) -> BrokerOrderResult:
        raise NotImplementedError(
            "AlpacaBroker.submit_order is a stub. "
            "Implement using alpaca_trade_api.REST.submit_order()."
        )

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("AlpacaBroker.cancel_order is a stub.")

    def get_order_status(self, order_id: str) -> Optional[BrokerOrderResult]:
        raise NotImplementedError("AlpacaBroker.get_order_status is a stub.")

    def get_positions(self) -> List[BrokerPosition]:
        raise NotImplementedError("AlpacaBroker.get_positions is a stub.")

    def get_account_equity(self) -> float:
        raise NotImplementedError("AlpacaBroker.get_account_equity is a stub.")
