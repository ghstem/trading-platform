"""
Interactive Brokers (IBKR) Broker Stub
=======================================
Stub implementation for Interactive Brokers via the IB Gateway / TWS API.
Raises ``NotImplementedError`` for every method until the ``ib_insync``
or ``ibapi`` library is integrated.
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger

from core.trading_engine import Order
from brokers.base import BrokerAdapter, BrokerOrderResult, BrokerPosition


class IBKRBroker(BrokerAdapter):
    """
    Interactive Brokers (IBKR) broker stub.

    Replace ``NotImplementedError`` bodies with ``ib_insync`` calls once
    IB Gateway or TWS is running and the ``ib_insync`` package is installed.

    Example wiring::

        from brokers.ibkr import IBKRBroker
        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker.connect()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        super().__init__("IBKR")
        self._host = host
        self._port = port
        self._client_id = client_id
        self._ib = None   # ib_insync.IB instance

    def connect(self) -> None:
        """
        Connect to IB Gateway / TWS.

        Uncomment when ib_insync is available::

            from ib_insync import IB
            self._ib = IB()
            self._ib.connect(self._host, self._port, clientId=self._client_id)
        """
        logger.warning("[IBKRBroker] connect() is a stub — not connected to IB Gateway.")
        self._connected = False

    def submit_order(self, order) -> BrokerOrderResult:
        raise NotImplementedError(
            "IBKRBroker.submit_order is a stub. "
            "Implement using ib_insync.IB.placeOrder()."
        )

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("IBKRBroker.cancel_order is a stub.")

    def get_order_status(self, order_id: str) -> Optional[BrokerOrderResult]:
        raise NotImplementedError("IBKRBroker.get_order_status is a stub.")

    def get_positions(self) -> List[BrokerPosition]:
        raise NotImplementedError("IBKRBroker.get_positions is a stub.")

    def get_account_equity(self) -> float:
        raise NotImplementedError("IBKRBroker.get_account_equity is a stub.")
