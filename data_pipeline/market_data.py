"""
Data Pipeline - Market Data Fetcher
Fetches and normalizes market data from multiple sources
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
import yfinance as yf
import ccxt
import requests

from core.trading_engine import Asset, AssetClass


class MarketDataProvider(ABC):
    """Abstract base class for market data providers"""
    
    def __init__(self, name: str):
        self.name = name
        self.last_update = None
    
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data for a symbol"""
        pass
    
    @abstractmethod
    def fetch_current_price(self, symbol: str) -> float:
        """Fetch current price for a symbol"""
        pass
    
    @abstractmethod
    def fetch_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch current prices for multiple symbols"""
        pass


class YahooFinanceProvider(MarketDataProvider):
    """Yahoo Finance data provider for stocks"""
    
    def __init__(self):
        super().__init__("Yahoo Finance")
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance"""
        try:
            # Map timeframe to yfinance interval
            interval_map = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "1h": "1h",
                "1d": "1d",
                "1wk": "1wk",
                "1mo": "1mo",
            }
            interval = interval_map.get(timeframe, "1d")
            
            # Calculate period based on limit and interval
            if interval == "1d":
                period = f"{limit}d"
            elif interval == "1wk":
                period = f"{limit * 7}d"
            elif interval == "1mo":
                period = f"{limit * 30}d"
            else:
                period = "1d"  # For intraday, default to 1 day
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            # Normalize column names
            df.columns = df.columns.str.lower()
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.index.name = 'timestamp'
            
            logger.info(f"Fetched {len(df)} candles for {symbol} from Yahoo Finance")
            self.last_update = datetime.now()
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_current_price(self, symbol: str) -> float:
        """Fetch current price from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            price = ticker.info.get('currentPrice') or ticker.info.get('regularMarketPrice', 0)
            logger.info(f"Fetched current price for {symbol}: ${price}")
            return float(price)
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {str(e)}")
            return 0.0
    
    def fetch_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch current prices for multiple symbols"""
        prices = {}
        for symbol in symbols:
            prices[symbol] = self.fetch_current_price(symbol)
        return prices


class CryptoDataProvider(MarketDataProvider):
    """Cryptocurrency data provider using CCXT"""
    
    def __init__(self, exchange: str = "binance"):
        super().__init__(f"CCXT - {exchange}")
        self.exchange_name = exchange
        self.exchange = self._initialize_exchange(exchange)
    
    def _initialize_exchange(self, exchange_name: str):
        """Initialize CCXT exchange"""
        try:
            exchange_class = getattr(ccxt, exchange_name)
            return exchange_class()
        except Exception as e:
            logger.error(f"Error initializing {exchange_name}: {str(e)}")
            return None
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV data from cryptocurrency exchange"""
        if self.exchange is None:
            logger.error(f"Exchange {self.exchange_name} not initialized")
            return pd.DataFrame()
        
        try:
            # CCXT expects symbol like 'BTC/USDT'
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            # Fetch OHLCV data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Fetched {len(df)} candles for {symbol} from {self.exchange_name}")
            self.last_update = datetime.now()
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_current_price(self, symbol: str) -> float:
        """Fetch current price from cryptocurrency exchange"""
        if self.exchange is None:
            return 0.0
        
        try:
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker['last']
            logger.info(f"Fetched current price for {symbol}: ${price}")
            return float(price)
        except Exception as e:
            logger.error(f"Error fetching crypto price for {symbol}: {str(e)}")
            return 0.0
    
    def fetch_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch current prices for multiple symbols"""
        prices = {}
        for symbol in symbols:
            prices[symbol] = self.fetch_current_price(symbol)
        return prices


class ForexDataProvider(MarketDataProvider):
    """Forex data provider (placeholder for future API integration)"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Forex Data")
        self.api_key = api_key
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV forex data"""
        # TODO: Integrate with OANDA, FXCM, or other forex APIs
        logger.warning(f"Forex data fetching not yet implemented for {symbol}")
        return pd.DataFrame()
    
    def fetch_current_price(self, symbol: str) -> float:
        """Fetch current forex price"""
        # TODO: Implement forex price fetching
        logger.warning(f"Forex price fetching not yet implemented for {symbol}")
        return 0.0
    
    def fetch_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch multiple forex prices"""
        prices = {}
        for symbol in symbols:
            prices[symbol] = self.fetch_current_price(symbol)
        return prices


class MarketDataManager:
    """Central manager for all market data operations"""
    
    def __init__(self):
        self.providers: Dict[str, MarketDataProvider] = {}
        self.cache: Dict[str, pd.DataFrame] = {}
        self.price_cache: Dict[str, Tuple[float, datetime]] = {}
        self.cache_expiry = timedelta(minutes=5)  # Cache prices for 5 minutes
        
        # Initialize default providers
        self._initialize_default_providers()
    
    def _initialize_default_providers(self):
        """Initialize default market data providers"""
        self.providers['yahoo'] = YahooFinanceProvider()
        
        # Initialize cryptocurrency providers
        try:
            self.providers['binance'] = CryptoDataProvider('binance')
        except Exception as e:
            logger.warning(f"Could not initialize Binance provider: {str(e)}")
        
        try:
            self.providers['coinbase'] = CryptoDataProvider('coinbase')
        except Exception as e:
            logger.warning(f"Could not initialize Coinbase provider: {str(e)}")
        
        self.providers['forex'] = ForexDataProvider()
    
    def add_provider(self, name: str, provider: MarketDataProvider):
        """Add a custom market data provider"""
        self.providers[name] = provider
        logger.info(f"Added market data provider: {name}")
    
    def get_provider(self, asset_class: AssetClass) -> Optional[MarketDataProvider]:
        """Get appropriate provider for asset class"""
        if asset_class == AssetClass.STOCK:
            return self.providers.get('yahoo')
        elif asset_class == AssetClass.CRYPTO:
            return self.providers.get('binance') or self.providers.get('coinbase')
        elif asset_class == AssetClass.FOREX:
            return self.providers.get('forex')
        else:
            return self.providers.get('yahoo')  # Default to Yahoo
    
    def fetch_ohlcv(self, asset: Asset, timeframe: str = "1d", limit: int = 100, 
                   use_cache: bool = True) -> pd.DataFrame:
        """Fetch OHLCV data for an asset"""
        cache_key = f"{asset.symbol}_{timeframe}_{limit}"
        
        # Check cache
        if use_cache and cache_key in self.cache:
            logger.info(f"Using cached data for {asset.symbol}")
            return self.cache[cache_key]
        
        # Get appropriate provider
        provider = self.get_provider(asset.asset_class)
        if provider is None:
            logger.error(f"No provider found for {asset.asset_class}")
            return pd.DataFrame()
        
        # Fetch data
        df = provider.fetch_ohlcv(asset.symbol, timeframe, limit)
        
        # Cache result
        if not df.empty:
            self.cache[cache_key] = df
        
        return df
    
    def fetch_current_price(self, asset: Asset, use_cache: bool = True) -> float:
        """Fetch current price for an asset"""
        # Check cache
        if use_cache and asset.symbol in self.price_cache:
            cached_price, timestamp = self.price_cache[asset.symbol]
            if datetime.now() - timestamp < self.cache_expiry:
                logger.info(f"Using cached price for {asset.symbol}: ${cached_price}")
                return cached_price
        
        # Get appropriate provider
        provider = self.get_provider(asset.asset_class)
        if provider is None:
            logger.error(f"No provider found for {asset.asset_class}")
            return 0.0
        
        # Fetch price
        price = provider.fetch_current_price(asset.symbol)
        
        # Cache result
        if price > 0:
            self.price_cache[asset.symbol] = (price, datetime.now())
        
        return price
    
    def fetch_multiple_prices(self, assets: List[Asset], use_cache: bool = True) -> Dict[str, float]:
        """Fetch current prices for multiple assets"""
        prices = {}
        for asset in assets:
            prices[asset.symbol] = self.fetch_current_price(asset, use_cache)
        return prices
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.price_cache.clear()
        logger.info("Market data cache cleared")


# Global instance
_market_data_manager = None


def get_market_data_manager() -> MarketDataManager:
    """Get or create global market data manager instance"""
    global _market_data_manager
    if _market_data_manager is None:
        _market_data_manager = MarketDataManager()
    return _market_data_manager


# Example usage
if __name__ == "__main__":
    manager = get_market_data_manager()
    
    # Fetch stock data
    apple = Asset(symbol="AAPL", asset_class=AssetClass.STOCK, exchange="NASDAQ")
    df = manager.fetch_ohlcv(apple, timeframe="1d", limit=30)
    print("Stock Data:")
    print(df.head())
    
    # Fetch crypto data
    bitcoin = Asset(symbol="BTC", asset_class=AssetClass.CRYPTO, exchange="BINANCE")
    df = manager.fetch_ohlcv(bitcoin, timeframe="1d", limit=30)
    print("\nCrypto Data:")
    print(df.head())
    
    # Fetch current prices
    price = manager.fetch_current_price(apple)
    print(f"\nApple current price: ${price}")
    
    price = manager.fetch_current_price(bitcoin)
    print(f"Bitcoin current price: ${price}")
