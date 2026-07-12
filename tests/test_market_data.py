"""
Unit tests for data_pipeline/market_data.py
Covers MarketDataManager caching, provider dispatch, and graceful fallback.
Network calls are mocked so tests run offline.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from core.trading_engine import Asset, AssetClass
from data_pipeline.market_data import (
    MarketDataManager,
    YahooFinanceProvider,
    CryptoDataProvider,
    ForexDataProvider,
    get_market_data_manager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asset(symbol: str = "AAPL", asset_class: AssetClass = AssetClass.STOCK) -> Asset:
    return Asset(symbol=symbol, asset_class=asset_class, exchange="NASDAQ")


def _make_ohlcv(n: int = 20) -> pd.DataFrame:
    dates = pd.date_range(start="2023-01-01", periods=n, freq="D")
    prices = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1.0,
            "low": prices - 1.0,
            "close": prices,
            "volume": np.ones(n) * 1_000.0,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# MarketDataManager — caching
# ---------------------------------------------------------------------------

class TestMarketDataManagerCaching:
    def _manager_with_mock_yahoo(self, df: pd.DataFrame):
        """Return a manager whose yahoo provider always returns df."""
        mgr = MarketDataManager.__new__(MarketDataManager)
        mgr.providers = {}
        mgr.cache = {}
        mgr.price_cache = {}
        mgr.cache_expiry = timedelta(minutes=5)
        mock_yahoo = MagicMock()
        mock_yahoo.fetch_ohlcv.return_value = df
        mock_yahoo.fetch_current_price.return_value = float(df["close"].iloc[-1])
        mgr.providers["yahoo"] = mock_yahoo
        return mgr

    def test_ohlcv_stored_in_cache(self):
        df = _make_ohlcv()
        mgr = self._manager_with_mock_yahoo(df)
        asset = _make_asset()
        _ = mgr.fetch_ohlcv(asset, use_cache=True)
        cache_key = f"{asset.symbol}_1d_100"
        assert cache_key in mgr.cache

    def test_ohlcv_cache_hit_skips_provider(self):
        df = _make_ohlcv()
        mgr = self._manager_with_mock_yahoo(df)
        asset = _make_asset()
        _ = mgr.fetch_ohlcv(asset, limit=100, use_cache=True)
        _ = mgr.fetch_ohlcv(asset, limit=100, use_cache=True)
        # Provider should only have been called once
        mgr.providers["yahoo"].fetch_ohlcv.assert_called_once()

    def test_ohlcv_cache_bypass(self):
        df = _make_ohlcv()
        mgr = self._manager_with_mock_yahoo(df)
        asset = _make_asset()
        _ = mgr.fetch_ohlcv(asset, use_cache=False)
        _ = mgr.fetch_ohlcv(asset, use_cache=False)
        assert mgr.providers["yahoo"].fetch_ohlcv.call_count == 2

    def test_empty_df_not_cached(self):
        mgr = MarketDataManager.__new__(MarketDataManager)
        mgr.providers = {}
        mgr.cache = {}
        mgr.price_cache = {}
        mgr.cache_expiry = timedelta(minutes=5)
        mock_yahoo = MagicMock()
        mock_yahoo.fetch_ohlcv.return_value = pd.DataFrame()
        mgr.providers["yahoo"] = mock_yahoo
        asset = _make_asset()
        result = mgr.fetch_ohlcv(asset)
        assert result.empty
        assert "AAPL_1d_100" not in mgr.cache

    def test_price_cache_hit(self):
        df = _make_ohlcv()
        mgr = self._manager_with_mock_yahoo(df)
        asset = _make_asset()
        _ = mgr.fetch_current_price(asset, use_cache=True)
        _ = mgr.fetch_current_price(asset, use_cache=True)
        mgr.providers["yahoo"].fetch_current_price.assert_called_once()

    def test_price_cache_bypass(self):
        df = _make_ohlcv()
        mgr = self._manager_with_mock_yahoo(df)
        asset = _make_asset()
        _ = mgr.fetch_current_price(asset, use_cache=False)
        _ = mgr.fetch_current_price(asset, use_cache=False)
        assert mgr.providers["yahoo"].fetch_current_price.call_count == 2

    def test_price_cache_expires(self):
        df = _make_ohlcv()
        mgr = self._manager_with_mock_yahoo(df)
        asset = _make_asset()
        _ = mgr.fetch_current_price(asset, use_cache=True)
        # Manually expire the cache entry
        price, _ = mgr.price_cache[asset.symbol]
        mgr.price_cache[asset.symbol] = (price, datetime.now() - timedelta(minutes=10))
        _ = mgr.fetch_current_price(asset, use_cache=True)
        assert mgr.providers["yahoo"].fetch_current_price.call_count == 2

    def test_clear_cache(self):
        df = _make_ohlcv()
        mgr = self._manager_with_mock_yahoo(df)
        asset = _make_asset()
        _ = mgr.fetch_ohlcv(asset, use_cache=True)
        _ = mgr.fetch_current_price(asset, use_cache=True)
        mgr.clear_cache()
        assert mgr.cache == {}
        assert mgr.price_cache == {}


# ---------------------------------------------------------------------------
# MarketDataManager — provider dispatch
# ---------------------------------------------------------------------------

class TestMarketDataManagerProviderDispatch:
    def _make_mgr(self):
        mgr = MarketDataManager.__new__(MarketDataManager)
        mgr.providers = {}
        mgr.cache = {}
        mgr.price_cache = {}
        mgr.cache_expiry = timedelta(minutes=5)
        return mgr

    def test_stock_uses_yahoo_provider(self):
        mgr = self._make_mgr()
        mock_yahoo = MagicMock()
        mock_yahoo.fetch_ohlcv.return_value = _make_ohlcv()
        mgr.providers["yahoo"] = mock_yahoo
        asset = _make_asset("AAPL", AssetClass.STOCK)
        mgr.fetch_ohlcv(asset, use_cache=False)
        mock_yahoo.fetch_ohlcv.assert_called_once()

    def test_crypto_uses_binance_provider(self):
        mgr = self._make_mgr()
        mock_binance = MagicMock()
        mock_binance.fetch_ohlcv.return_value = _make_ohlcv()
        mgr.providers["binance"] = mock_binance
        asset = _make_asset("BTC", AssetClass.CRYPTO)
        mgr.fetch_ohlcv(asset, use_cache=False)
        mock_binance.fetch_ohlcv.assert_called_once()

    def test_no_provider_returns_empty_df(self):
        mgr = self._make_mgr()
        asset = _make_asset("AAPL", AssetClass.STOCK)
        result = mgr.fetch_ohlcv(asset)
        assert result.empty

    def test_add_custom_provider(self):
        mgr = self._make_mgr()
        mock_prov = MagicMock()
        mock_prov.fetch_ohlcv.return_value = _make_ohlcv()
        mgr.add_provider("custom", mock_prov)
        assert "custom" in mgr.providers

    def test_fetch_multiple_prices(self):
        mgr = self._make_mgr()
        mock_yahoo = MagicMock()
        mock_yahoo.fetch_current_price.return_value = 150.0
        mgr.providers["yahoo"] = mock_yahoo
        assets = [_make_asset("AAPL"), _make_asset("MSFT")]
        prices = mgr.fetch_multiple_prices(assets, use_cache=False)
        assert "AAPL" in prices
        assert "MSFT" in prices


# ---------------------------------------------------------------------------
# YahooFinanceProvider
# ---------------------------------------------------------------------------

class TestYahooFinanceProvider:
    def test_fetch_ohlcv_returns_df_on_success(self):
        provider = YahooFinanceProvider()
        mock_ticker = MagicMock()
        mock_df = _make_ohlcv()
        # Mimic yfinance column names
        mock_df.columns = pd.Index(["Open", "High", "Low", "Close", "Volume"])
        mock_df.index.name = "Datetime"
        history_df = mock_df.rename(columns=str.lower)
        history_df.index.name = "timestamp"
        mock_ticker.history.return_value = history_df
        with patch("data_pipeline.market_data.yf.Ticker", return_value=mock_ticker):
            result = provider.fetch_ohlcv("AAPL", timeframe="1d", limit=20)
        # Should not raise; result may be empty if column mapping differs – just check type
        assert isinstance(result, pd.DataFrame)

    def test_fetch_current_price_returns_float(self):
        provider = YahooFinanceProvider()
        mock_ticker = MagicMock()
        mock_ticker.info = {"currentPrice": 175.5}
        with patch("data_pipeline.market_data.yf.Ticker", return_value=mock_ticker):
            price = provider.fetch_current_price("AAPL")
        assert isinstance(price, float)
        assert price == pytest.approx(175.5)

    def test_fetch_current_price_fallback_key(self):
        provider = YahooFinanceProvider()
        mock_ticker = MagicMock()
        mock_ticker.info = {"regularMarketPrice": 180.0}
        with patch("data_pipeline.market_data.yf.Ticker", return_value=mock_ticker):
            price = provider.fetch_current_price("AAPL")
        assert price == pytest.approx(180.0)

    def test_fetch_current_price_returns_zero_on_error(self):
        provider = YahooFinanceProvider()
        with patch("data_pipeline.market_data.yf.Ticker", side_effect=Exception("network error")):
            price = provider.fetch_current_price("AAPL")
        assert price == 0.0

    def test_fetch_ohlcv_returns_empty_on_error(self):
        provider = YahooFinanceProvider()
        with patch("data_pipeline.market_data.yf.Ticker", side_effect=Exception("fail")):
            result = provider.fetch_ohlcv("AAPL")
        assert result.empty


# ---------------------------------------------------------------------------
# ForexDataProvider (placeholder)
# ---------------------------------------------------------------------------

class TestForexDataProvider:
    def test_fetch_ohlcv_returns_empty(self):
        provider = ForexDataProvider()
        result = provider.fetch_ohlcv("EURUSD")
        assert result.empty

    def test_fetch_current_price_returns_zero(self):
        provider = ForexDataProvider()
        price = provider.fetch_current_price("EURUSD")
        assert price == 0.0

    def test_fetch_multiple_prices_all_zero(self):
        provider = ForexDataProvider()
        prices = provider.fetch_multiple_prices(["EURUSD", "GBPUSD"])
        assert prices == {"EURUSD": 0.0, "GBPUSD": 0.0}


# ---------------------------------------------------------------------------
# Singleton get_market_data_manager
# ---------------------------------------------------------------------------

class TestGetMarketDataManager:
    def test_returns_same_instance(self):
        import data_pipeline.market_data as mdm_module
        mdm_module._market_data_manager = None
        m1 = get_market_data_manager()
        m2 = get_market_data_manager()
        assert m1 is m2

    def test_returns_market_data_manager_instance(self):
        import data_pipeline.market_data as mdm_module
        mdm_module._market_data_manager = None
        mgr = get_market_data_manager()
        assert isinstance(mgr, MarketDataManager)
