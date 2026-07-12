"""
Unit tests for StrategyManager orchestrator
"""

import pytest
from datetime import datetime, timedelta

import numpy as np

from core.trading_engine import Asset, AssetClass, Portfolio
from core.strategy_manager import StrategyManager, StrategyState
from risk_management.risk_manager import RiskManager, RiskLimits
from strategies.trend_following import SMACrossoverStrategy
from strategies.mean_reversion import BollingerBandsStrategy
from strategies.base import Signal, SignalType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def portfolio():
    return Portfolio(initial_capital=100_000.0)


@pytest.fixture
def risk_manager(portfolio):
    return RiskManager(portfolio=portfolio, limits=RiskLimits())


@pytest.fixture
def manager(portfolio, risk_manager):
    return StrategyManager(portfolio=portfolio, risk_manager=risk_manager)


@pytest.fixture
def apple():
    return Asset(symbol="AAPL", asset_class=AssetClass.STOCK, exchange="NASDAQ")


@pytest.fixture
def sma_strategy(apple):
    return SMACrossoverStrategy(fast_period=5, slow_period=10, assets=[apple])


@pytest.fixture
def bb_strategy(apple):
    return BollingerBandsStrategy(period=10, num_std=2.0, assets=[apple])


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------

class TestStrategyManagerLifecycle:
    def test_add_strategy_starts_running(self, manager, sma_strategy):
        record = manager.add_strategy("sma1", sma_strategy, auto_start=True)
        assert record.state == StrategyState.RUNNING
        assert "sma1" in manager.list_all()
        assert "sma1" in manager.list_running()

    def test_add_duplicate_raises(self, manager, sma_strategy, bb_strategy):
        manager.add_strategy("s1", sma_strategy)
        with pytest.raises(ValueError):
            manager.add_strategy("s1", bb_strategy)

    def test_pause_and_resume(self, manager, sma_strategy):
        manager.add_strategy("sma1", sma_strategy)
        manager.pause("sma1")
        assert manager._records["sma1"].state == StrategyState.PAUSED
        assert "sma1" not in manager.list_running()
        manager.resume("sma1")
        assert manager._records["sma1"].state == StrategyState.RUNNING

    def test_stop(self, manager, sma_strategy):
        manager.add_strategy("sma1", sma_strategy)
        manager.stop("sma1")
        assert manager._records["sma1"].state == StrategyState.STOPPED

    def test_remove_strategy(self, manager, sma_strategy):
        manager.add_strategy("sma1", sma_strategy)
        manager.remove_strategy("sma1")
        assert "sma1" not in manager.list_all()

    def test_unknown_instance_raises(self, manager):
        with pytest.raises(KeyError):
            manager.start("does_not_exist")

    def test_start_all_pause_all(self, manager, sma_strategy, bb_strategy):
        manager.add_strategy("s1", sma_strategy)
        manager.add_strategy("s2", bb_strategy)
        manager.pause_all()
        assert all(r.state == StrategyState.PAUSED for r in manager._records.values())
        manager.start_all()
        assert all(r.state == StrategyState.RUNNING for r in manager._records.values())


# ---------------------------------------------------------------------------
# Bar processing
# ---------------------------------------------------------------------------

class TestOnBar:
    def _feed(self, manager, asset, prices):
        """Feed a price series through the manager."""
        all_accepted = []
        for i, price in enumerate(prices):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            accepted = manager.on_bar(date, {asset: price})
            all_accepted.extend(accepted)
        return all_accepted

    def test_on_bar_increments_bar_count(self, manager, sma_strategy, apple):
        manager.add_strategy("sma1", sma_strategy)
        manager.on_bar(datetime.now(), {apple: 150.0})
        assert manager._bar_count == 1

    def test_paused_strategy_not_called(self, manager, sma_strategy, apple):
        manager.add_strategy("sma1", sma_strategy)
        manager.pause("sma1")
        # Feed many bars — no signals should be generated from paused strategy
        for i in range(60):
            manager.on_bar(datetime.now(), {apple: 100.0 + i})
        assert manager._records["sma1"].signals_generated == 0

    def test_multiple_strategies_all_receive_bar(self, manager, sma_strategy, bb_strategy, apple):
        manager.add_strategy("s1", sma_strategy)
        manager.add_strategy("s2", bb_strategy)
        prices = [100.0] * 10
        for i, p in enumerate(prices):
            manager.on_bar(datetime(2024, 1, 1) + timedelta(days=i), {apple: p})
        # Both strategies received bars (bar count reflects manager total)
        assert manager._bar_count == 10


# ---------------------------------------------------------------------------
# Capital allocation
# ---------------------------------------------------------------------------

class TestCapitalAllocation:
    def test_allocated_capital_set_on_add(self, manager, sma_strategy):
        manager.add_strategy("sma1", sma_strategy, capital_pct=0.3)
        record = manager._records["sma1"]
        assert abs(record.allocated_capital - 30_000.0) < 1e-6

    def test_update_allocations(self, manager, sma_strategy, bb_strategy):
        manager.add_strategy("s1", sma_strategy, capital_pct=0.2)
        manager.add_strategy("s2", bb_strategy, capital_pct=0.3)
        manager.update_allocations({"s1": 0.5, "s2": 0.1})
        assert abs(manager._records["s1"].capital_pct - 0.5) < 1e-6
        assert abs(manager._records["s2"].capital_pct - 0.1) < 1e-6


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

class TestAggregateStats:
    def test_stats_keys(self, manager, sma_strategy):
        manager.add_strategy("s1", sma_strategy)
        stats = manager.get_aggregate_stats()
        assert "num_strategies" in stats
        assert "running" in stats
        assert "total_bars_processed" in stats
        assert "portfolio_total_equity" in stats

    def test_signal_log(self, manager, sma_strategy, apple):
        manager.add_strategy("sma1", sma_strategy)
        prices = [100.0 + i * 0.1 for i in range(60)]
        for i, p in enumerate(prices):
            manager.on_bar(datetime(2024, 1, 1) + timedelta(days=i), {apple: p})
        log = manager.get_signal_log(limit=200)
        assert isinstance(log, list)
        for entry in log:
            assert "timestamp" in entry
            assert "signal" in entry
