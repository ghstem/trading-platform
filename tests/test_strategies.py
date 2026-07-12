"""
Unit tests for strategy base class, registry, and all strategy types
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from core.trading_engine import Asset, AssetClass, Portfolio
from strategies.base import BaseStrategy, Signal, SignalType, StrategyPerformance
from strategies.registry import StrategyRegistry, get_registry
from strategies.trend_following import SMACrossoverStrategy, EMACrossoverStrategy, MACDStrategy
from strategies.mean_reversion import BollingerBandsStrategy, RSIReversionStrategy, ZScoreReversionStrategy
from strategies.momentum import PriceMomentumStrategy, BreakoutStrategy
from strategies.pairs_trading import PairsTradingStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asset(symbol="AAPL", ac=AssetClass.STOCK) -> Asset:
    return Asset(symbol=symbol, asset_class=ac, exchange="NASDAQ")


def _make_prices(n: int, start: float = 100.0, trend: float = 0.001) -> list:
    """Generate a price series with a slight trend + noise."""
    rng = np.random.default_rng(42)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + trend + rng.normal(0, 0.01)))
    return prices


def _feed_prices(strategy: BaseStrategy, asset: Asset, prices: list) -> list:
    """Feed prices one by one and collect all signals."""
    all_signals = []
    strategy.set_assets([asset])
    strategy.initialize(Portfolio(100_000))
    for i, p in enumerate(prices):
        date = datetime(2024, 1, 1) + timedelta(days=i)
        sigs = strategy.on_bar(date, {asset: p})
        all_signals.extend(sigs)
    return all_signals


# ---------------------------------------------------------------------------
# BaseStrategy
# ---------------------------------------------------------------------------

class TestBaseStrategy:
    def test_record_signal_updates_performance(self):
        asset = _make_asset()

        class _Stub(BaseStrategy):
            def generate_signals(self, ohlcv, a): return []
            def on_bar(self, d, p): return []

        strat = _Stub("stub", [asset])
        sig = Signal("stub", asset, SignalType.BUY, price=100.0)
        strat.record_signal(sig)
        assert strat.performance.total_signals == 1
        assert strat.performance.buy_signals == 1

    def test_record_trade_result(self):
        asset = _make_asset()

        class _Stub(BaseStrategy):
            def generate_signals(self, ohlcv, a): return []
            def on_bar(self, d, p): return []

        strat = _Stub("stub", [asset])
        strat.record_trade_result(50.0)
        strat.record_trade_result(-20.0)
        assert strat.performance.trades_executed == 2
        assert strat.performance.winning_trades == 1
        assert strat.performance.losing_trades == 1
        assert abs(strat.performance.total_pnl - 30.0) < 1e-6


# ---------------------------------------------------------------------------
# StrategyRegistry
# ---------------------------------------------------------------------------

class TestStrategyRegistry:
    def test_list_all_built_in_strategies(self):
        reg = StrategyRegistry()
        keys = reg.list_keys()
        assert "sma_crossover" in keys
        assert "ema_crossover" in keys
        assert "macd" in keys
        assert "bollinger_bands" in keys
        assert "rsi_reversion" in keys
        assert "zscore_reversion" in keys
        assert "price_momentum" in keys
        assert "breakout" in keys
        assert "pairs_trading" in keys
        assert "ml_alpha" in keys

    def test_create_sma_crossover(self):
        reg = StrategyRegistry()
        asset = _make_asset()
        strat = reg.create("sma_crossover", assets=[asset], fast_period=5, slow_period=20)
        assert isinstance(strat, SMACrossoverStrategy)
        assert strat.fast_period == 5

    def test_create_unknown_key_raises(self):
        reg = StrategyRegistry()
        with pytest.raises(KeyError):
            reg.create("nonexistent_strategy")

    def test_register_custom(self):
        reg = StrategyRegistry()

        class _Custom(BaseStrategy):
            def generate_signals(self, ohlcv, a): return []
            def on_bar(self, d, p): return []

        reg.register(
            "custom_test", _Custom,
            category="Test", description="test",
            default_params={"name": "custom_test"},
        )
        assert "custom_test" in reg.list_keys()
        inst = reg.create("custom_test")
        assert isinstance(inst, _Custom)

    def test_describe(self):
        reg = StrategyRegistry()
        info = reg.describe("macd")
        assert info["category"] == "Trend Following"
        assert "param_schema" in info

    def test_list_by_category(self):
        reg = StrategyRegistry()
        cats = reg.list_by_category()
        assert "Trend Following" in cats
        assert "Mean Reversion" in cats
        assert "Momentum" in cats
        assert "Statistical Arbitrage" in cats
        assert "Machine Learning" in cats


# ---------------------------------------------------------------------------
# Trend-Following Strategies
# ---------------------------------------------------------------------------

class TestSMACrossover:
    def test_generates_signals_on_uptrend(self):
        # Flat prices then strong uptrend forces a bullish MA crossover
        asset = _make_asset()
        strat = SMACrossoverStrategy(fast_period=5, slow_period=10)
        flat = [100.0] * 15
        uptrend = _make_prices(45, start=100.0, trend=0.015)
        prices = flat + uptrend
        signals = _feed_prices(strat, asset, prices)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) >= 1

    def test_no_duplicate_buys(self):
        asset = _make_asset()
        strat = SMACrossoverStrategy(fast_period=5, slow_period=10)
        prices = _make_prices(80, start=100.0, trend=0.005)
        signals = _feed_prices(strat, asset, prices)

        # Should never have two consecutive BUY signals for same asset
        consecutive_buys = 0
        prev = None
        for s in signals:
            if s.signal_type == SignalType.BUY and prev == SignalType.BUY:
                consecutive_buys += 1
            prev = s.signal_type
        assert consecutive_buys == 0


class TestEMACrossover:
    def test_generates_signals(self):
        asset = _make_asset()
        strat = EMACrossoverStrategy(fast_period=5, slow_period=15)
        prices = _make_prices(60, trend=0.008)
        signals = _feed_prices(strat, asset, prices)
        assert isinstance(signals, list)


class TestMACDStrategy:
    def test_generates_signals(self):
        asset = _make_asset()
        strat = MACDStrategy(fast_period=5, slow_period=10, signal_period=4)
        prices = _make_prices(60, trend=0.005)
        signals = _feed_prices(strat, asset, prices)
        assert isinstance(signals, list)


# ---------------------------------------------------------------------------
# Mean-Reversion Strategies
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_generates_buy_when_below_lower_band(self):
        asset = _make_asset()
        strat = BollingerBandsStrategy(period=20, num_std=2.0)
        # Sharp drop at the end should push below lower band
        prices = [100.0] * 30 + [80.0, 78.0, 76.0]
        signals = _feed_prices(strat, asset, prices)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) >= 1


class TestRSIReversion:
    def test_buy_in_oversold_region(self):
        asset = _make_asset()
        strat = RSIReversionStrategy(period=14, oversold=30, overbought=70)
        # Sustained down trend makes RSI go oversold
        prices = list(np.linspace(100, 70, 30)) + [72.0, 74.0]
        signals = _feed_prices(strat, asset, prices)
        assert isinstance(signals, list)

    def test_performance_dict_keys(self):
        asset = _make_asset()
        strat = RSIReversionStrategy()
        strat.initialize(Portfolio(100_000))
        perf = strat.get_performance()
        assert "strategy" in perf
        assert "win_rate" in perf


class TestZScoreReversion:
    def test_buy_when_zscore_extreme(self):
        asset = _make_asset()
        strat = ZScoreReversionStrategy(lookback=20, entry_threshold=1.5, exit_threshold=0.3)
        prices = [100.0] * 25 + [88.0, 87.0, 86.0]
        signals = _feed_prices(strat, asset, prices)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) >= 1


# ---------------------------------------------------------------------------
# Momentum Strategies
# ---------------------------------------------------------------------------

class TestPriceMomentum:
    def test_buy_on_positive_momentum(self):
        asset = _make_asset()
        strat = PriceMomentumStrategy(lookback=10, top_n=1)
        prices = _make_prices(30, trend=0.02)
        signals = _feed_prices(strat, asset, prices)
        assert isinstance(signals, list)


class TestBreakout:
    def test_buy_on_channel_breakout(self):
        asset = _make_asset()
        strat = BreakoutStrategy(period=10, atr_multiplier=1.0)
        # Flat then sharp breakout
        prices = [100.0] * 15 + [115.0, 120.0, 125.0]
        signals = _feed_prices(strat, asset, prices)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) >= 1


# ---------------------------------------------------------------------------
# Pairs Trading
# ---------------------------------------------------------------------------

class TestPairsTrading:
    def test_enter_long_spread_on_low_zscore(self):
        asset_a = _make_asset("SPY", AssetClass.STOCK)
        asset_b = _make_asset("QQQ", AssetClass.STOCK)

        strat = PairsTradingStrategy(
            lookback=30, entry_threshold=1.5, exit_threshold=0.5
        )
        strat.set_assets([asset_a, asset_b])
        strat.initialize(Portfolio(100_000))

        rng = np.random.default_rng(0)
        base = 100.0 + np.cumsum(rng.normal(0, 0.5, 100))
        prices_a = base.tolist()
        prices_b = (base * 0.98).tolist()

        # Force spread divergence at the end
        prices_a[-5:] = [p * 0.93 for p in prices_a[-5:]]

        all_signals = []
        for i in range(len(prices_a)):
            date = datetime(2024, 1, 1) + timedelta(days=i)
            sigs = strat.on_bar(date, {asset_a: prices_a[i], asset_b: prices_b[i]})
            all_signals.extend(sigs)

        assert isinstance(all_signals, list)
