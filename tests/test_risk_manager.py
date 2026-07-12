"""
Unit tests for risk_management/risk_manager.py
Covers RiskLimits, RiskMetrics, DailyRiskTracker, ExecutionOverride,
RiskManager, and ExecutionOverrideManager.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from core.trading_engine import Asset, AssetClass, Portfolio, Order, OrderSide, OrderType
from risk_management.risk_manager import (
    OverrideType,
    OverrideStatus,
    RiskLevel,
    RiskLimits,
    RiskMetrics,
    DailyRiskTracker,
    ExecutionOverride,
    RiskManager,
    ExecutionOverrideManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asset(symbol: str = "AAPL") -> Asset:
    return Asset(symbol=symbol, asset_class=AssetClass.STOCK, exchange="NASDAQ")


def _make_portfolio(capital: float = 100_000.0) -> Portfolio:
    return Portfolio(initial_capital=capital)


def _buy_into_portfolio(portfolio: Portfolio, asset: Asset, qty: float, price: float):
    order = portfolio.create_order(asset, OrderSide.BUY, qty, price=price)
    portfolio.execute_order(order, price, qty)
    return order


# ---------------------------------------------------------------------------
# RiskLimits
# ---------------------------------------------------------------------------

class TestRiskLimits:
    def test_defaults(self):
        limits = RiskLimits()
        assert limits.max_position_size_pct == pytest.approx(0.1)
        assert limits.max_daily_loss_pct == pytest.approx(-0.05)
        assert limits.max_drawdown_pct == pytest.approx(-0.20)
        assert limits.max_positions == 20

    def test_custom_values(self):
        limits = RiskLimits(max_position_size_pct=0.05, max_positions=5)
        assert limits.max_position_size_pct == pytest.approx(0.05)
        assert limits.max_positions == 5

    def test_to_dict_has_all_keys(self):
        limits = RiskLimits()
        d = limits.to_dict()
        for key in (
            "max_position_size_pct",
            "max_daily_loss_pct",
            "max_drawdown_pct",
            "max_leverage",
            "max_positions",
            "min_cash_buffer_pct",
            "risk_per_trade_pct",
        ):
            assert key in d


# ---------------------------------------------------------------------------
# RiskMetrics
# ---------------------------------------------------------------------------

class TestRiskMetrics:
    def test_defaults(self):
        m = RiskMetrics()
        assert m.current_drawdown_pct == 0.0
        assert m.risk_level == RiskLevel.LOW

    def test_to_dict_stringified(self):
        m = RiskMetrics(current_drawdown_pct=-0.05, num_open_positions=3)
        d = m.to_dict()
        assert "current_drawdown_pct" in d
        assert "num_open_positions" in d
        assert d["num_open_positions"] == 3


# ---------------------------------------------------------------------------
# DailyRiskTracker
# ---------------------------------------------------------------------------

class TestDailyRiskTracker:
    def test_daily_return_pct_zero_when_flat(self):
        tracker = DailyRiskTracker(starting_equity=100_000, current_equity=100_000)
        assert tracker.daily_return_pct == pytest.approx(0.0)

    def test_daily_return_pct_positive(self):
        tracker = DailyRiskTracker(starting_equity=100_000, current_equity=105_000)
        assert tracker.daily_return_pct == pytest.approx(0.05)

    def test_daily_return_pct_negative(self):
        tracker = DailyRiskTracker(starting_equity=100_000, current_equity=95_000)
        assert tracker.daily_return_pct == pytest.approx(-0.05)

    def test_zero_starting_equity(self):
        tracker = DailyRiskTracker(starting_equity=0, current_equity=0)
        assert tracker.daily_return_pct == 0.0

    def test_record_trade_result(self):
        tracker = DailyRiskTracker(starting_equity=100_000)
        tracker.record_trade_result(500.0, is_win=True)
        tracker.record_trade_result(-200.0, is_win=False)
        assert tracker.trades_executed == 2
        assert tracker.winning_trades == 1
        assert tracker.losing_trades == 1
        assert tracker.largest_win == pytest.approx(500.0)
        assert tracker.largest_loss == pytest.approx(-200.0)
        assert tracker.daily_pnl == pytest.approx(300.0)

    def test_win_rate_zero_when_no_trades(self):
        tracker = DailyRiskTracker()
        assert tracker.win_rate == 0.0

    def test_win_rate_calculation(self):
        tracker = DailyRiskTracker()
        tracker.record_trade_result(100, is_win=True)
        tracker.record_trade_result(100, is_win=True)
        tracker.record_trade_result(-50, is_win=False)
        assert tracker.win_rate == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# ExecutionOverride
# ---------------------------------------------------------------------------

class TestExecutionOverride:
    def test_approve_sets_status(self):
        override = ExecutionOverride(override_type=OverrideType.EMERGENCY_STOP)
        override.approve("admin")
        assert override.status == OverrideStatus.APPROVED
        assert override.approver == "admin"
        assert override.approved_at is not None

    def test_deny_sets_status(self):
        override = ExecutionOverride(override_type=OverrideType.MANUAL_CLOSE)
        override.deny("supervisor", reason="Not authorised")
        assert override.status == OverrideStatus.DENIED
        assert override.approver == "supervisor"

    def test_execute_sets_status(self):
        override = ExecutionOverride(override_type=OverrideType.DAILY_LOSS_OVERRIDE)
        override.execute()
        assert override.status == OverrideStatus.EXECUTED
        assert override.executed_at is not None


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class TestRiskManager:
    def test_initialises_correctly(self):
        port = _make_portfolio()
        rm = RiskManager(port)
        assert rm.is_trading_halted is False
        assert rm.max_equity == pytest.approx(100_000.0)

    def test_can_trade_allowed_by_default(self):
        port = _make_portfolio()
        rm = RiskManager(port)
        rm.update_metrics()
        allowed, reason = rm.can_trade()
        assert allowed is True

    def test_can_trade_blocked_when_halted(self):
        port = _make_portfolio()
        rm = RiskManager(port)
        rm.is_trading_halted = True
        allowed, reason = rm.can_trade()
        assert allowed is False
        assert "halted" in reason.lower()

    def test_can_trade_blocked_on_max_positions(self):
        port = _make_portfolio(500_000)
        limits = RiskLimits(max_positions=2)
        rm = RiskManager(port, limits=limits)
        asset1 = _make_asset("AAPL")
        asset2 = _make_asset("GOOG")
        _buy_into_portfolio(port, asset1, 10, 100.0)
        _buy_into_portfolio(port, asset2, 10, 200.0)
        rm.update_metrics()
        allowed, reason = rm.can_trade()
        assert allowed is False
        assert "positions" in reason.lower()

    def test_validate_order_passes_small_order(self):
        port = _make_portfolio(100_000)
        rm = RiskManager(port)
        rm.update_metrics()
        asset = _make_asset()
        order = Order(asset=asset, side=OrderSide.BUY, quantity=1, price=100.0)
        valid, reason = rm.validate_order(order)
        assert valid is True

    def test_validate_order_rejects_oversized_position(self):
        port = _make_portfolio(100_000)
        limits = RiskLimits(max_position_size_pct=0.05)  # 5% max
        rm = RiskManager(port, limits=limits)
        rm.update_metrics()
        asset = _make_asset()
        # Order value = 10_000 = 10% of equity → exceeds 5%
        order = Order(asset=asset, side=OrderSide.BUY, quantity=100, price=100.0)
        valid, reason = rm.validate_order(order)
        assert valid is False
        assert "position size" in reason.lower()

    def test_validate_order_rejects_insufficient_cash_buffer(self):
        port = _make_portfolio(10_000)
        # min_cash_buffer_pct = 0.9 means 90% must stay cash → almost no buying allowed
        limits = RiskLimits(max_position_size_pct=1.0, min_cash_buffer_pct=0.9)
        rm = RiskManager(port, limits=limits)
        rm.update_metrics()
        asset = _make_asset()
        # Order that would consume 50% of cash, leaving less than 90%
        order = Order(asset=asset, side=OrderSide.BUY, quantity=50, price=100.0)
        valid, reason = rm.validate_order(order)
        assert valid is False
        assert "cash" in reason.lower()

    def test_update_metrics_tracks_drawdown(self):
        port = _make_portfolio(100_000)
        rm = RiskManager(port)
        rm.update_metrics()
        assert rm.risk_metrics.current_drawdown_pct == pytest.approx(0.0)

        # Simulate equity drop by reducing cash
        port.cash = 80_000.0
        rm.update_metrics()
        # drawdown should now be negative
        assert rm.risk_metrics.current_drawdown_pct < 0.0

    def test_update_metrics_sets_risk_level(self):
        port = _make_portfolio(100_000)
        rm = RiskManager(port)
        rm.update_metrics()
        assert rm.risk_metrics.risk_level in list(RiskLevel)

    def test_calculate_position_size(self):
        port = _make_portfolio(100_000)
        rm = RiskManager(port, limits=RiskLimits(risk_per_trade_pct=0.02, max_position_size_pct=0.1))
        # risk_amount=2000, points_at_risk=10 → uncapped=200; max_qty=10000/100=100 → capped to 100
        size = rm.calculate_position_size(_make_asset(), entry_price=100.0, stop_loss_price=90.0)
        assert size == pytest.approx(100.0)

    def test_calculate_position_size_clamps_to_max(self):
        port = _make_portfolio(100_000)
        limits = RiskLimits(risk_per_trade_pct=0.5, max_position_size_pct=0.1)
        rm = RiskManager(port, limits=limits)
        # Without capping: risk_amount=50k, points=10 → 5000 shares @ 100 = $500k > 10% cap
        size = rm.calculate_position_size(_make_asset(), entry_price=100.0, stop_loss_price=90.0)
        max_expected = 100_000 * 0.1 / 100.0   # 100 shares
        assert size <= max_expected

    def test_calculate_position_size_invalid_stop(self):
        port = _make_portfolio()
        rm = RiskManager(port)
        size = rm.calculate_position_size(_make_asset(), entry_price=90.0, stop_loss_price=100.0)
        assert size == 0.0

    def test_halt_triggered_on_max_drawdown(self):
        port = _make_portfolio(100_000)
        limits = RiskLimits(max_drawdown_pct=-0.10)
        rm = RiskManager(port, limits=limits)
        rm.max_equity = 100_000.0
        # Simulate > 10% drawdown
        port.cash = 85_000.0
        rm.update_metrics()
        assert rm.is_trading_halted is True

    def test_get_risk_summary_keys(self):
        port = _make_portfolio()
        rm = RiskManager(port)
        summary = rm.get_risk_summary()
        assert "Risk Level" in summary
        assert "Trading Halted" in summary

    def test_price_history_tracked_in_correlation(self):
        """update_metrics with prices should not raise even with one position."""
        port = _make_portfolio(100_000)
        rm = RiskManager(port)
        asset = _make_asset()
        _buy_into_portfolio(port, asset, 10, 100.0)
        for price in [100.0, 101.0, 102.0, 103.0]:
            rm.update_metrics({asset: price})
        # single asset → correlation stays 0
        assert rm.risk_metrics.portfolio_correlation == 0.0


# ---------------------------------------------------------------------------
# ExecutionOverrideManager
# ---------------------------------------------------------------------------

class TestExecutionOverrideManager:
    def _make_manager(self):
        port = _make_portfolio()
        rm = RiskManager(port)
        return ExecutionOverrideManager(port, rm, require_approval=True)

    def test_request_override_creates_pending_override(self):
        mgr = self._make_manager()
        override = mgr.request_override(
            OverrideType.EMERGENCY_STOP,
            requester="trader1",
            reason="Urgent halt",
        )
        assert override.status == OverrideStatus.PENDING
        assert override.override_type == OverrideType.EMERGENCY_STOP

    def test_approve_override_requires_approved_user(self):
        mgr = self._make_manager()
        override = mgr.request_override(
            OverrideType.MANUAL_CLOSE,
            requester="trader1",
            reason="Close all",
        )
        mgr.approve_override(override.override_id, "admin")
        assert override.status == OverrideStatus.APPROVED

    def test_approve_by_unknown_user_not_approved(self):
        mgr = self._make_manager()
        override = mgr.request_override(
            OverrideType.MANUAL_CLOSE,
            requester="trader1",
            reason="test",
        )
        result = mgr.approve_override(override.override_id, "random_user")
        assert result is False
        # Status stays PENDING because the unauthorised user can't approve
        assert override.status == OverrideStatus.PENDING

    def test_get_pending_overrides(self):
        mgr = self._make_manager()
        mgr.request_override(OverrideType.EMERGENCY_STOP, "t1", "r1")
        mgr.request_override(OverrideType.MANUAL_CLOSE, "t2", "r2")
        pending = mgr.get_pending_overrides()
        assert len(pending) == 2
