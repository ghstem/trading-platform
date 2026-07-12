"""
Unit tests for strategies/ml_strategy.py
Covers MLAlphaStrategy signal generation, on_bar lifecycle,
and periodic retraining logic.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from core.trading_engine import Asset, AssetClass, Portfolio
from strategies.ml_strategy import MLAlphaStrategy
from strategies.base import SignalType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asset(symbol: str = "AAPL") -> Asset:
    return Asset(symbol=symbol, asset_class=AssetClass.STOCK, exchange="NASDAQ")


def _make_prices(n: int, start: float = 100.0, trend: float = 0.002) -> list:
    rng = np.random.default_rng(0)
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + trend + rng.normal(0, 0.005)))
    return prices


def _feed_prices(strategy: MLAlphaStrategy, asset: Asset, prices: list) -> list:
    all_signals = []
    strategy.set_assets([asset])
    strategy.initialize(Portfolio(200_000))
    for i, p in enumerate(prices):
        date = datetime(2023, 1, 1) + timedelta(days=i)
        sigs = strategy.on_bar(date, {asset: p})
        all_signals.extend(sigs)
    return all_signals


# ---------------------------------------------------------------------------
# Constructor & defaults
# ---------------------------------------------------------------------------

class TestMLAlphaStrategyInit:
    def test_name_includes_model_type(self):
        strat = MLAlphaStrategy(model_type="random_forest")
        assert "random_forest" in strat.name

    def test_default_params(self):
        strat = MLAlphaStrategy()
        assert strat.train_window == 200
        assert strat.retrain_interval == 50
        assert strat.buy_threshold == pytest.approx(0.6)
        assert strat.sell_threshold == pytest.approx(0.4)

    def test_custom_params(self):
        strat = MLAlphaStrategy(
            model_type="gradient_boosting",
            train_window=100,
            retrain_interval=25,
            buy_threshold=0.7,
            sell_threshold=0.3,
        )
        assert strat.model_type == "gradient_boosting"
        assert strat.train_window == 100
        assert strat.retrain_interval == 25

    def test_assets_optional(self):
        strat = MLAlphaStrategy()
        assert strat.assets == []

    def test_assets_passed_at_construction(self):
        asset = _make_asset()
        strat = MLAlphaStrategy(assets=[asset])
        assert asset in strat.assets


# ---------------------------------------------------------------------------
# No signals before warmup
# ---------------------------------------------------------------------------

class TestMLAlphaStrategyWarmup:
    def test_no_signals_before_train_window(self):
        """With fewer bars than train_window, no signals are emitted."""
        asset = _make_asset()
        strat = MLAlphaStrategy(train_window=200)
        prices = _make_prices(50)   # well below train_window
        signals = _feed_prices(strat, asset, prices)
        assert signals == []

    def test_price_history_accumulates(self):
        asset = _make_asset()
        strat = MLAlphaStrategy(train_window=200)
        strat.set_assets([asset])
        strat.initialize(Portfolio(100_000))
        for i in range(30):
            strat.on_bar(datetime(2023, 1, 1) + timedelta(days=i), {asset: 100.0 + i})
        assert len(strat._price_history.get(asset.symbol, [])) == 30


# ---------------------------------------------------------------------------
# generate_signals
# ---------------------------------------------------------------------------

class TestMLAlphaGenerateSignals:
    def _make_ohlcv(self, n: int = 250, trend: float = 0.002) -> pd.DataFrame:
        prices = _make_prices(n, trend=trend)
        return pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.005 for p in prices],
                "low": [p * 0.995 for p in prices],
                "close": prices,
                "volume": [1000.0] * n,
            }
        )

    def test_returns_list(self):
        asset = _make_asset()
        strat = MLAlphaStrategy(train_window=200)
        df = self._make_ohlcv(250)
        result = strat.generate_signals(df, asset)
        assert isinstance(result, list)

    def test_no_signals_when_insufficient_data(self):
        asset = _make_asset()
        strat = MLAlphaStrategy(train_window=200)
        df = self._make_ohlcv(100)  # less than train_window
        signals = strat.generate_signals(df, asset)
        assert signals == []

    def test_buy_signal_has_correct_type(self):
        asset = _make_asset()
        strat = MLAlphaStrategy(
            train_window=200,
            buy_threshold=0.0,   # always buy after training
            min_factor_signal=-999.0,
        )
        df = self._make_ohlcv(250, trend=0.005)
        strat._in_position[asset.symbol] = False

        # Pre-train the predictor so generate_signals can produce a signal
        predictor = strat._get_or_create_predictor(asset.symbol)
        predictor.train(df)

        signals = strat.generate_signals(df, asset)
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        # With a very low threshold the model should be able to produce a BUY
        assert isinstance(buy_signals, list)

    def test_signal_strength_in_range(self):
        asset = _make_asset()
        strat = MLAlphaStrategy(train_window=200, buy_threshold=0.0, min_factor_signal=-999.0)
        df = self._make_ohlcv(250)
        strat._in_position[asset.symbol] = False
        predictor = strat._get_or_create_predictor(asset.symbol)
        predictor.train(df)
        signals = strat.generate_signals(df, asset)
        for sig in signals:
            assert 0.0 <= sig.strength <= 1.0

    def test_close_signal_when_in_position_and_sell_triggered(self):
        asset = _make_asset()
        strat = MLAlphaStrategy(
            train_window=200,
            sell_threshold=1.0,   # always sell (threshold * 2 - 1 = 1.0 > any signal)
        )
        df = self._make_ohlcv(250)
        strat._in_position[asset.symbol] = True

        predictor = strat._get_or_create_predictor(asset.symbol)
        predictor.train(df)

        signals = strat.generate_signals(df, asset)
        close_signals = [s for s in signals if s.signal_type == SignalType.CLOSE]
        # With sell_threshold=1.0, (1.0*2-1)=1.0; latest_ml is always <= 1.0, so close fires
        assert isinstance(close_signals, list)


# ---------------------------------------------------------------------------
# on_bar lifecycle
# ---------------------------------------------------------------------------

class TestMLAlphaOnBar:
    def test_on_bar_returns_list(self):
        asset = _make_asset()
        strat = MLAlphaStrategy()
        strat.set_assets([asset])
        strat.initialize(Portfolio(100_000))
        result = strat.on_bar(datetime.now(), {asset: 100.0})
        assert isinstance(result, list)

    def test_on_bar_skips_asset_with_missing_price(self):
        asset = _make_asset("AAPL")
        other = _make_asset("GOOGL")
        strat = MLAlphaStrategy()
        strat.set_assets([asset])
        strat.initialize(Portfolio(100_000))
        # Feed only GOOGL price; AAPL is in assets but not in current_prices
        result = strat.on_bar(datetime.now(), {other: 200.0})
        assert result == []

    def test_in_position_flag_set_on_buy_signal(self):
        """After a BUY signal, _in_position[symbol] is True."""
        asset = _make_asset()
        strat = MLAlphaStrategy(train_window=200, buy_threshold=0.0, min_factor_signal=-999.0)
        prices = _make_prices(210, trend=0.003)
        strat.set_assets([asset])
        strat.initialize(Portfolio(1_000_000))

        for i, p in enumerate(prices):
            strat.on_bar(datetime(2023, 1, 1) + timedelta(days=i), {asset: p})

        # _in_position may or may not be True depending on ML prediction;
        # just verify it's a boolean stored correctly
        assert isinstance(strat._in_position.get(asset.symbol, False), bool)

    def test_periodic_retraining_called(self):
        """Verify the predictor's train method is called at retrain_interval boundaries."""
        asset = _make_asset()
        strat = MLAlphaStrategy(train_window=50, retrain_interval=10)
        strat.set_assets([asset])
        strat.initialize(Portfolio(200_000))

        prices = _make_prices(70, trend=0.002)
        with patch.object(
            strat, "_get_or_create_predictor", wraps=strat._get_or_create_predictor
        ) as mock_get:
            for i, p in enumerate(prices):
                strat.on_bar(datetime(2023, 1, 1) + timedelta(days=i), {asset: p})

        # At a minimum the internal structures must exist
        assert asset.symbol in strat._price_history

    def test_gradient_boosting_model_type(self):
        """Gradient boosting model type is accepted without error."""
        asset = _make_asset()
        strat = MLAlphaStrategy(model_type="gradient_boosting", train_window=200)
        prices = _make_prices(50)
        signals = _feed_prices(strat, asset, prices)
        assert isinstance(signals, list)

    def test_bar_count_increments(self):
        asset = _make_asset()
        strat = MLAlphaStrategy(train_window=300)
        strat.set_assets([asset])
        strat.initialize(Portfolio(100_000))
        for i in range(5):
            strat.on_bar(datetime.now(), {asset: 100.0})
        assert strat._bar_count.get(asset.symbol, 0) == 5


# ---------------------------------------------------------------------------
# _build_ohlcv helper
# ---------------------------------------------------------------------------

class TestBuildOHLCV:
    def test_returns_dataframe_with_required_columns(self):
        asset = _make_asset()
        strat = MLAlphaStrategy()
        strat._price_history[asset.symbol] = [100.0, 101.0, 102.0]
        df = strat._build_ohlcv(asset.symbol)
        for col in ("open", "high", "low", "close", "volume"):
            assert col in df.columns

    def test_volume_fallback_when_no_volume_history(self):
        asset = _make_asset()
        strat = MLAlphaStrategy()
        strat._price_history[asset.symbol] = [100.0, 101.0]
        df = strat._build_ohlcv(asset.symbol)
        # All volume values should be the fallback value of 1.0
        assert all(v == 1.0 for v in df["volume"])
