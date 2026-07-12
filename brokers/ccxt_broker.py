"""
CCXT Broker Adapter
===================
Generic broker adapter backed by the `ccxt` library, which supports 200+
crypto exchanges including Binance, Kraken, Bybit, OKX, Coinbase Advanced
Trade, KuCoin, Gate.io, and many more.

Set ``exchange_id`` to any exchange ID listed at https://github.com/ccxt/ccxt.

Example usage::

    from brokers.ccxt_broker import CCXTBroker

    # Binance
    broker = CCXTBroker(
        exchange_id="binance",
        api_key="YOUR_API_KEY",
        secret="YOUR_SECRET",
    )
    broker.connect()

    # Bybit
    broker = CCXTBroker(
        exchange_id="bybit",
        api_key="YOUR_API_KEY",
        secret="YOUR_SECRET",
        testnet=True,   # use testnet for paper trading
    )
    broker.connect()

Supported exchanges include (but are not limited to):
  binance, bybit, okx, kraken, coinbase, kucoin, gateio, bitfinex,
  huobi, bitmex, deribit, phemex, mexc, bitget, and 200+ more.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger

from core.trading_engine import Order, OrderSide, OrderStatus, OrderType
from brokers.base import BrokerAdapter, BrokerOrderResult, BrokerPosition


class CCXTBroker(BrokerAdapter):
    """
    CCXT-based broker adapter.

    Parameters
    ----------
    exchange_id : str
        The ccxt exchange identifier (e.g. ``"binance"``, ``"bybit"``,
        ``"okx"``, ``"kraken"``).
    api_key : str
        Exchange API key.
    secret : str
        Exchange API secret.
    password : str, optional
        API passphrase required by some exchanges (e.g. OKX, KuCoin).
    testnet : bool
        If ``True``, enables the exchange's sandbox / testnet environment
        for paper trading (supported by Binance, Bybit, OKX, and others).
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: str = "",
        secret: str = "",
        password: str = "",
        testnet: bool = False,
    ):
        super().__init__(f"CCXT:{exchange_id}")
        self._exchange_id = exchange_id
        self._api_key = api_key
        self._secret = secret
        self._password = password
        self._testnet = testnet
        self._exchange = None   # ccxt.Exchange instance

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Instantiate and authenticate the CCXT exchange object."""
        try:
            import ccxt

            exchange_class = getattr(ccxt, self._exchange_id, None)
            if exchange_class is None:
                logger.error(
                    f"[CCXTBroker] Unknown exchange '{self._exchange_id}'. "
                    "Check https://github.com/ccxt/ccxt for valid IDs."
                )
                return

            params: Dict = {
                "apiKey": self._api_key,
                "secret": self._secret,
            }
            if self._password:
                params["password"] = self._password

            self._exchange = exchange_class(params)

            if self._testnet:
                if self._exchange.has.get("sandbox"):
                    self._exchange.set_sandbox_mode(True)
                    logger.info(f"[CCXTBroker] Sandbox/testnet enabled for {self._exchange_id}.")
                else:
                    logger.warning(
                        f"[CCXTBroker] {self._exchange_id} does not support sandbox mode."
                    )

            self._exchange.load_markets()
            self._connected = True
            logger.info(f"[CCXTBroker] Connected to {self._exchange_id}.")
        except Exception as exc:
            logger.error(f"[CCXTBroker] Failed to connect to {self._exchange_id}: {exc}")
            self._connected = False

    def disconnect(self) -> None:
        self._exchange = None
        self._connected = False
        logger.info(f"[CCXTBroker] Disconnected from {self._exchange_id}.")

    # ------------------------------------------------------------------
    # BrokerAdapter implementation
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> BrokerOrderResult:
        """Submit a market or limit order via CCXT."""
        self._require_connected()

        symbol = self._to_ccxt_symbol(order.asset.symbol)
        side = "buy" if order.side == OrderSide.BUY else "sell"
        order_type = "market" if order.order_type == OrderType.MARKET else "limit"
        price = order.price if order.order_type == OrderType.LIMIT else None

        try:
            response = self._exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=order.quantity,
                price=price,
            )
            broker_id = str(response.get("id", ""))
            filled_qty = float(response.get("filled", 0.0))
            fill_price = float(response.get("average") or response.get("price") or 0.0)
            status_raw = response.get("status", "open")
            status = self._map_status(status_raw)

            logger.info(
                f"[CCXTBroker:{self._exchange_id}] Order {broker_id} — "
                f"{side} {order.quantity} {symbol} @ {fill_price} [{status_raw}]"
            )
            return BrokerOrderResult(
                broker_order_id=broker_id,
                local_order_id=order.order_id,
                status=status,
                filled_quantity=filled_qty,
                average_fill_price=fill_price,
                submitted_at=datetime.now(),
                filled_at=datetime.now() if status == OrderStatus.FILLED else None,
            )
        except Exception as exc:
            logger.error(f"[CCXTBroker:{self._exchange_id}] submit_order failed: {exc}")
            return BrokerOrderResult(
                broker_order_id="",
                local_order_id=order.order_id,
                status=OrderStatus.REJECTED,
                error_message=str(exc),
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by broker order ID."""
        self._require_connected()
        try:
            self._exchange.cancel_order(order_id)
            logger.info(f"[CCXTBroker:{self._exchange_id}] Cancelled order {order_id}.")
            return True
        except Exception as exc:
            logger.error(f"[CCXTBroker:{self._exchange_id}] cancel_order failed: {exc}")
            return False

    def get_order_status(self, order_id: str) -> Optional[BrokerOrderResult]:
        """Fetch latest status for a previously submitted order."""
        self._require_connected()
        try:
            response = self._exchange.fetch_order(order_id)
            return BrokerOrderResult(
                broker_order_id=str(response["id"]),
                local_order_id=order_id,
                status=self._map_status(response.get("status", "open")),
                filled_quantity=float(response.get("filled", 0.0)),
                average_fill_price=float(response.get("average") or 0.0),
            )
        except Exception as exc:
            logger.error(f"[CCXTBroker:{self._exchange_id}] get_order_status failed: {exc}")
            return None

    def get_positions(self) -> List[BrokerPosition]:
        """Return open positions (futures/margin) or non-zero spot balances."""
        self._require_connected()
        positions: List[BrokerPosition] = []
        try:
            if self._exchange.has.get("fetchPositions"):
                raw = self._exchange.fetch_positions()
                for p in raw:
                    qty = float(p.get("contracts") or p.get("size") or 0.0)
                    if qty == 0:
                        continue
                    positions.append(
                        BrokerPosition(
                            symbol=p.get("symbol", ""),
                            quantity=qty,
                            average_entry_price=float(p.get("entryPrice") or 0.0),
                            current_price=float(p.get("markPrice") or 0.0),
                            unrealized_pnl=float(p.get("unrealizedPnl") or 0.0),
                            side="LONG" if p.get("side") == "long" else "SHORT",
                        )
                    )
            else:
                # Spot: treat non-zero balance as a position
                balance = self._exchange.fetch_balance()
                for currency, info in balance.get("total", {}).items():
                    qty = float(info or 0.0)
                    if qty > 0 and currency not in ("USDT", "USD", "BUSD", "USDC"):
                        positions.append(
                            BrokerPosition(
                                symbol=currency,
                                quantity=qty,
                                average_entry_price=0.0,
                                current_price=0.0,
                                unrealized_pnl=0.0,
                            )
                        )
        except Exception as exc:
            logger.error(f"[CCXTBroker:{self._exchange_id}] get_positions failed: {exc}")
        return positions

    def get_account_equity(self) -> float:
        """Return total USD-equivalent account balance."""
        self._require_connected()
        try:
            balance = self._exchange.fetch_balance()
            usd_total = 0.0
            for currency in ("USDT", "USD", "BUSD", "USDC"):
                usd_total += float(balance.get("total", {}).get(currency, 0.0))
            return usd_total
        except Exception as exc:
            logger.error(f"[CCXTBroker:{self._exchange_id}] get_account_equity failed: {exc}")
            return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected or self._exchange is None:
            raise RuntimeError(
                f"CCXTBroker ({self._exchange_id}) is not connected. Call connect() first."
            )

    @staticmethod
    def _to_ccxt_symbol(symbol: str) -> str:
        """Convert a raw symbol like 'BTCUSDT' to ccxt format 'BTC/USDT' if needed."""
        if "/" not in symbol and len(symbol) > 4:
            for quote in ("USDT", "USD", "BTC", "ETH", "BNB", "USDC"):
                if symbol.endswith(quote):
                    base = symbol[: -len(quote)]
                    return f"{base}/{quote}"
        return symbol

    @staticmethod
    def _map_status(raw: str) -> OrderStatus:
        return {
            "closed": OrderStatus.FILLED,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.CANCELLED,
        }.get(raw, OrderStatus.PENDING)
