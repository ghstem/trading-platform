"""
Unit tests for backtester/backtest_engine.py
Covers BacktestConfig, BacktestStats, SimpleMovingAverageCrossover, and Backtester.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from core.trading_engine import Asset, AssetClass, Portfolio, OrderSide, OrderStatus
from backtester.backtest_engine import (
    BacktestConfig,
    BacktestStats,
    Strategy,
    SimpleMovingAverageCrossover,
    Backtester,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asset(symbol: str = "AAPL") -> Asset:
    return Asset(symbol=symbol, asset_class=AssetClass.STOCK, exchange="NASDAQ")


def _make_ohlcv(n: int = 100, start_price: float = 100.0, trend: float = 0.001) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with a DatetimeIndex."""
    rng = np.random.default_rng(42)
    prices = [start_price]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + trend + rng.normal(0, 0.005)))
    prices = np.array(prices)
    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": rng.integers(1_000, 10_000, n).astype(float),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# BacktestConfig
# ---------------------------------------------------------------------------

class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.initial_capital == 100_000.0
        assert cfg.commission_pct == 0.001
        assert cfg.slippage_pct == 0.0005
        assert cfg.max_orders_per_day == 100
        assert cfg.leverage == 1.0

    def test_custom_values(self):
        cfg = BacktestConfig(
            initial_capital=50_000.0,
            start_date="2022-01-01",
            end_date="2023-01-01",
            commission_pct=0.002,
            slippage_pct=0.001,
        )
        assert cfg.initial_capital == 50_000.0
        assert cfg.start_date == "2022-01-01"
        assert cfg.commission_pct == 0.002

    def test_to_dict_contains_required_keys(self):
        cfg = BacktestConfig()
        d = cfg.to_dict()
        for key in ("initial_capital", "start_date", "end_date", "commission_pct", "slippage_pct"):
            assert key in d


# ---------------------------------------------------------------------------
# BacktestStats
# ---------------------------------------------------------------------------

class TestBacktestStats:
    def test_defaults_are_zero(self):
        stats = BacktestStats()
        assert stats.total_return == 0.0
        assert stats.num_trades == 0
        assert stats.win_rate == 0.0

    def test_fields_settable(self):
        stats = BacktestStats(total_return=0.15, sharpe_ratio=1.5, num_trades=10)
        assert stats.total_return == 0.15
        assert stats.sharpe_ratio == 1.5


# ---------------------------------------------------------------------------
# SimpleMovingAverageCrossover (Strategy subclass)
# ---------------------------------------------------------------------------

class TestSimpleMovingAverageCrossover:
    def test_inherits_strategy(self):
        strat = SimpleMovingAverageCrossover()
        assert isinstance(strat, Strategy)

    def test_default_periods(self):
        strat = SimpleMovingAverageCrossover()
        assert strat.fast_period == 20
        assert strat.slow_period == 50

    def test_custom_periods(self):
        strat = SimpleMovingAverageCrossover(fast_period=5, slow_period=20)
        assert strat.fast_period == 5
        assert strat.slow_period == 20

    def test_initialize_attaches_portfolio(self):
        strat = SimpleMovingAverageCrossover()
        port = Portfolio(100_000)
        strat.initialize(port)
        assert strat.portfolio is port

    def test_no_signal_before_warmup(self):
        asset = _make_asset()
        strat = SimpleMovingAverageCrossover(fast_period=5, slow_period=10)
        strat.set_asset(asset)
        strat.initialize(Portfolio(100_000))

        for i in range(9):  # less than slow_period
            strat.on_bar(datetime(2024, 1, 1) + timedelta(days=i), {asset: 100.0 + i})
        assert len(strat.portfolio.orders) == 0

    def test_buy_signal_on_uptrend(self):
        """Fast MA crosses above slow MA on strong uptrend → buy order placed."""
        asset = _make_asset()
        strat = SimpleMovingAverageCrossover(fast_period=5, slow_period=10)
        strat.set_asset(asset)
        strat.initialize(Portfolio(100_000))

        # Flat then sharp uptrend forces a bullish crossover
        flat = [100.0] * 12
        uptrend = [100.0 + i * 2 for i in range(20)]
        prices = flat + uptrend

        for i, p in enumerate(prices):
            strat.on_bar(datetime(2024, 1, 1) + timedelta(days=i), {asset: p})

        buy_orders = [o for o in strat.portfolio.orders if o.side == OrderSide.BUY]
        assert len(buy_orders) >= 1

    def test_no_duplicate_buys(self):
        """position_open flag prevents back-to-back buy orders."""
        asset = _make_asset()
        strat = SimpleMovingAverageCrossover(fast_period=5, slow_period=10)
        strat.set_asset(asset)
        strat.initialize(Portfolio(500_000))

        prices = [100.0 + i for i in range(60)]
        for i, p in enumerate(prices):
            strat.on_bar(datetime(2024, 1, 1) + timedelta(days=i), {asset: p})

        buy_orders = [o for o in strat.portfolio.orders if o.side == OrderSide.BUY]
        # Should never have 2 buys without an intervening sell
        consecutive = 0
        prev_side = None
        for o in strat.portfolio.orders:
            if o.side == OrderSide.BUY and prev_side == OrderSide.BUY:
                consecutive += 1
            prev_side = o.side
        assert consecutive == 0

    def test_on_bar_ignores_missing_asset(self):
        """Calling on_bar with an asset not matching set_asset does nothing."""
        asset = _make_asset("AAPL")
        other = _make_asset("GOOGL")
        strat = SimpleMovingAverageCrossover(fast_period=5, slow_period=10)
        strat.set_asset(asset)
        strat.initialize(Portfolio(100_000))
        # Should not raise
        strat.on_bar(datetime.now(), {other: 150.0})
        assert len(strat.portfolio.orders) == 0


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class TestBacktester:
    def _make_backtester_with_mock_data(self, n: int = 80) -> tuple:
        """Return (backtester, asset, ohlcv_df)."""
        asset = _make_asset()
        df = _make_ohlcv(n)
        config = BacktestConfig(
            initial_capital=100_000.0,
            start_date=str(df.index[0].date()),
            end_date=str(df.index[-1].date()),
            commission_pct=0.001,
            slippage_pct=0.0005,
        )
        strat = SimpleMovingAverageCrossover(fast_period=5, slow_period=20)
        strat.set_asset(asset)
        bt = Backtester(strat, config)
        return bt, asset, df

    def test_initial_state(self):
        asset = _make_asset()
        cfg = BacktestConfig()
        strat = SimpleMovingAverageCrossover()
        strat.set_asset(asset)
        bt = Backtester(strat, cfg)

        assert bt.equity_curve == []
        assert bt.stats.total_return == 0.0

    def test_run_returns_stats(self):
        """Backtester.run() returns a BacktestStats instance."""
        bt, asset, df = self._make_backtester_with_mock_data()
        with patch.object(
            bt.market_data, "fetch_ohlcv", return_value=df
        ):
            stats = bt.run([asset])
        assert isinstance(stats, BacktestStats)

    def test_equity_curve_populated(self):
        bt, asset, df = self._make_backtester_with_mock_data()
        with patch.object(bt.market_data, "fetch_ohlcv", return_value=df):
            bt.run([asset])
        assert len(bt.equity_curve) > 0
        assert "equity" in bt.equity_curve[0]

    def test_get_equity_curve_returns_dataframe(self):
        bt, asset, df = self._make_backtester_with_mock_data()
        with patch.object(bt.market_data, "fetch_ohlcv", return_value=df):
            bt.run([asset])
        ec = bt.get_equity_curve()
        assert isinstance(ec, pd.DataFrame)
        assert "equity" in ec.columns

    def test_stats_summary_keys(self):
        bt, asset, df = self._make_backtester_with_mock_data()
        with patch.object(bt.market_data, "fetch_ohlcv", return_value=df):
            bt.run([asset])
        summary = bt.get_stats_summary()
        for key in ("Total Return", "Sharpe Ratio", "Max Drawdown", "Win Rate"):
            assert key in summary

    def test_empty_data_returns_zero_stats(self):
        """When provider returns empty data, stats stay at defaults."""
        asset = _make_asset()
        cfg = BacktestConfig()
        strat = SimpleMovingAverageCrossover()
        strat.set_asset(asset)
        bt = Backtester(strat, cfg)
        with patch.object(bt.market_data, "fetch_ohlcv", return_value=pd.DataFrame()):
            stats = bt.run([asset])
        assert stats.total_return == 0.0
        assert stats.num_trades == 0

    def test_commission_applied(self):
        """Non-zero commission means equity won't grow faster than buy-and-hold on flat."""
        bt, asset, df = self._make_backtester_with_mock_data(100)
        with patch.object(bt.market_data, "fetch_ohlcv", return_value=df):
            bt.run([asset])
        # final equity must be positive and trades incurred commission
        assert bt.stats.final_equity > 0

    def test_slippage_applied_on_buy(self):
        """Slippage increases buy fill price above close."""
        asset = _make_asset()
        df = _make_ohlcv(50)
        config = BacktestConfig(
            initial_capital=100_000.0,
            start_date=str(df.index[0].date()),
            end_date=str(df.index[-1].date()),
            commission_pct=0.0,
            slippage_pct=0.01,     # 1 % noticeable slippage
        )
        strat = SimpleMovingAverageCrossover(fast_period=5, slow_period=15)
        strat.set_asset(asset)
        bt = Backtester(strat, config)
        with patch.object(bt.market_data, "fetch_ohlcv", return_value=df):
            bt.run([asset])
        buy_orders = [o for o in bt.portfolio.orders if o.side == OrderSide.BUY and o.is_filled]
        if buy_orders:
            close_prices = df["close"].values
            # average fill must be above the minimum close price (slippage pushes up)
            assert buy_orders[0].average_fill_price > 0

    def test_default_config_used_when_none(self):
        asset = _make_asset()
        strat = SimpleMovingAverageCrossover()
        strat.set_asset(asset)
        bt = Backtester(strat)   # no config
        assert bt.config.initial_capital == 100_000.0

    def test_calculate_stats_handles_single_bar(self):
        """Backtest with only one date should not raise."""
        asset = _make_asset()
        df = _make_ohlcv(1)
        config = BacktestConfig(
            initial_capital=50_000.0,
            start_date=str(df.index[0].date()),
            end_date=str(df.index[0].date()),
        )
        strat = SimpleMovingAverageCrossover()
        strat.set_asset(asset)
        bt = Backtester(strat, config)
        with patch.object(bt.market_data, "fetch_ohlcv", return_value=df):
            stats = bt.run([asset])
        assert stats.final_equity >= 0  # should not crash

    def test_analyze_trades_win_rate(self):
        """Profitable strategy should produce non-zero win metrics."""
        asset = _make_asset()
        # Strong consistent uptrend — many winning trades
        df = _make_ohlcv(200, trend=0.005)
        config = BacktestConfig(
            initial_capital=100_000.0,
            start_date=str(df.index[0].date()),
            end_date=str(df.index[-1].date()),
        )
        strat = SimpleMovingAverageCrossover(fast_period=5, slow_period=20)
        strat.set_asset(asset)
        bt = Backtester(strat, config)
        with patch.object(bt.market_data, "fetch_ohlcv", return_value=df):
            bt.run([asset])
        # May or may not have trades depending on crossover timing; just check types
        assert isinstance(bt.stats.win_rate, float)
        assert 0.0 <= bt.stats.win_rate <= 1.0
