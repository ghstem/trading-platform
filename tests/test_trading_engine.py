"""
Unit tests for core trading engine
"""
import pytest
import numpy as np

from core.trading_engine import (
    Asset, AssetClass, Order, OrderSide, OrderStatus, OrderType,
    Position, Portfolio,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def apple():
    return Asset(symbol="AAPL", asset_class=AssetClass.STOCK, exchange="NASDAQ")


@pytest.fixture
def bitcoin():
    return Asset(symbol="BTC", asset_class=AssetClass.CRYPTO, exchange="COINBASE")


@pytest.fixture
def portfolio():
    return Portfolio(initial_capital=100_000.0)


# ---------------------------------------------------------------------------
# Asset
# ---------------------------------------------------------------------------

class TestAsset:
    def test_hash_equality(self, apple):
        apple2 = Asset(symbol="AAPL", asset_class=AssetClass.STOCK, exchange="NASDAQ")
        assert apple == apple2
        assert hash(apple) == hash(apple2)

    def test_different_exchange_not_equal(self, apple):
        nyse = Asset(symbol="AAPL", asset_class=AssetClass.STOCK, exchange="NYSE")
        assert apple != nyse


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

class TestOrder:
    def test_create_order_requires_asset_and_side(self):
        with pytest.raises(ValueError):
            Order(asset=None, side=OrderSide.BUY)
        with pytest.raises(ValueError):
            asset = Asset("X", AssetClass.STOCK, "NYSE")
            Order(asset=asset, side=None)

    def test_update_fill_partial(self, apple):
        order = Order(asset=apple, side=OrderSide.BUY, quantity=10, price=150.0)
        order.update_fill(5, 150.0)
        assert order.filled_quantity == 5
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert abs(order.average_fill_price - 150.0) < 1e-6

    def test_update_fill_complete(self, apple):
        order = Order(asset=apple, side=OrderSide.BUY, quantity=10, price=150.0)
        order.update_fill(10, 150.0, commission=5.0)
        assert order.status == OrderStatus.FILLED
        assert order.is_filled
        assert order.commission == 5.0

    def test_total_cost(self, apple):
        order = Order(asset=apple, side=OrderSide.BUY, quantity=10, price=100.0)
        order.update_fill(10, 100.0, commission=2.0)
        assert order.total_cost == 1002.0

    def test_average_fill_price_across_partial_fills(self, apple):
        order = Order(asset=apple, side=OrderSide.BUY, quantity=10, price=100.0)
        order.update_fill(5, 100.0)
        order.update_fill(5, 110.0)
        assert order.status == OrderStatus.FILLED
        assert abs(order.average_fill_price - 105.0) < 1e-6


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class TestPosition:
    def test_market_value(self, apple):
        pos = Position(asset=apple, quantity=10, current_price=200.0)
        assert pos.market_value == 2000.0

    def test_pnl_percent(self, apple):
        pos = Position(asset=apple, quantity=10, average_entry_price=100.0, current_price=100.0)
        pos.update_price(110.0)
        assert abs(pos.unrealized_pnl - 100.0) < 1e-6
        assert abs(pos.pnl_percent - 10.0) < 1e-4

    def test_add_to_position_updates_avg_price(self, apple):
        pos = Position(asset=apple, quantity=0, average_entry_price=0.0)
        pos.add_to_position(10, 100.0)
        pos.add_to_position(10, 120.0)
        assert abs(pos.average_entry_price - 110.0) < 1e-6

    def test_reduce_position_realizes_pnl(self, apple):
        pos = Position(asset=apple, quantity=10, average_entry_price=100.0)
        pnl = pos.reduce_position(10, 110.0)
        assert abs(pnl - 100.0) < 1e-6
        assert pos.quantity == 0


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_initial_state(self, portfolio):
        assert portfolio.cash == 100_000.0
        assert portfolio.total_equity == 100_000.0
        assert portfolio.total_pnl == 0.0

    def test_buy_reduces_cash(self, portfolio, apple):
        order = portfolio.create_order(apple, OrderSide.BUY, 10, price=100.0)
        portfolio.execute_order(order, fill_price=100.0, filled_qty=10, commission=0.0)
        assert portfolio.cash == 99_000.0
        assert apple in portfolio.positions

    def test_sell_removes_position(self, portfolio, apple):
        buy = portfolio.create_order(apple, OrderSide.BUY, 10, price=100.0)
        portfolio.execute_order(buy, 100.0, 10)
        sell = portfolio.create_order(apple, OrderSide.SELL, 10, price=110.0)
        portfolio.execute_order(sell, 110.0, 10)
        assert apple not in portfolio.positions
        assert abs(portfolio.cash - 100_100.0) < 1e-6   # 100k - 1000 + 1100

    def test_update_prices(self, portfolio, apple):
        order = portfolio.create_order(apple, OrderSide.BUY, 10, price=100.0)
        portfolio.execute_order(order, 100.0, 10)
        portfolio.update_prices({apple: 120.0})
        pos = portfolio.get_position(apple)
        assert abs(pos.unrealized_pnl - 200.0) < 1e-6

    def test_portfolio_summary_keys(self, portfolio):
        summary = portfolio.get_portfolio_summary()
        assert "total_equity" in summary
        assert "cash" in summary
        assert "total_pnl" in summary

    def test_close_position(self, portfolio, apple):
        order = portfolio.create_order(apple, OrderSide.BUY, 5, price=200.0)
        portfolio.execute_order(order, 200.0, 5)
        close_order = portfolio.close_position(apple, close_price=210.0)
        assert close_order is not None
        assert close_order.is_filled
        assert apple not in portfolio.positions

    def test_quantity_zero_raises(self, portfolio, apple):
        with pytest.raises(ValueError):
            portfolio.create_order(apple, OrderSide.BUY, quantity=0)
