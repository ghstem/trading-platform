"""
Configuration Management System
Loads and validates environment variables and settings
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

class Config:
    """Base configuration class"""
    
    # ==================== ENVIRONMENT ====================
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    DEBUG = os.getenv('DEBUG', 'true').lower() == 'true'
    
    # ==================== API & SERVER ====================
    FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 5000))
    
    # ==================== DATABASE ====================
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', 5432))
    DB_NAME = os.getenv('DB_NAME', 'trading_platform')
    DB_USER = os.getenv('DB_USER', 'trading_user')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
    
    # SQLAlchemy Database URL
    SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # MongoDB
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/trading_platform')
    
    # Redis
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    
    # ==================== BROKER APIs ====================
    # Interactive Brokers
    IB_ACCOUNT_ID = os.getenv('IB_ACCOUNT_ID', '')
    IB_HOST = os.getenv('IB_HOST', '127.0.0.1')
    IB_PORT = int(os.getenv('IB_PORT', 7497))
    
    # OANDA
    OANDA_API_KEY = os.getenv('OANDA_API_KEY', '')
    OANDA_ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID', '')
    OANDA_ENVIRONMENT = os.getenv('OANDA_ENVIRONMENT', 'practice')
    
    # Alpaca
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # ==================== CRYPTOCURRENCY EXCHANGES ====================
    # Coinbase
    COINBASE_API_KEY = os.getenv('COINBASE_API_KEY', '')
    COINBASE_SECRET_KEY = os.getenv('COINBASE_SECRET_KEY', '')
    COINBASE_PASSPHRASE = os.getenv('COINBASE_PASSPHRASE', '')
    
    # Binance
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    
    # Kraken
    KRAKEN_API_KEY = os.getenv('KRAKEN_API_KEY', '')
    KRAKEN_SECRET_KEY = os.getenv('KRAKEN_SECRET_KEY', '')
    
    # ==================== FINANCIAL DATA PROVIDERS ====================
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    QUANDL_API_KEY = os.getenv('QUANDL_API_KEY', '')
    YAHOO_FINANCE_ENABLED = os.getenv('YAHOO_FINANCE_ENABLED', 'true').lower() == 'true'
    
    # ==================== SPORTS BETTING ====================
    SPORTSRADAR_API_KEY = os.getenv('SPORTSRADAR_API_KEY', '')
    ODDS_API_KEY = os.getenv('ODDS_API_KEY', '')
    BETFAIR_USERNAME = os.getenv('BETFAIR_USERNAME', '')
    BETFAIR_PASSWORD = os.getenv('BETFAIR_PASSWORD', '')
    BETFAIR_APP_KEY = os.getenv('BETFAIR_APP_KEY', '')
    
    # ==================== LOGGING ====================
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/trading_platform.log')
    
    # ==================== TRADING PARAMETERS ====================
    DEFAULT_RISK_PERCENTAGE = float(os.getenv('DEFAULT_RISK_PERCENTAGE', 2.0))
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 0.1))
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', -1000))
    
    # ==================== BACKTESTING ====================
    BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2023-01-01')
    BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '2024-01-01')
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 100000))
    
    # ==================== MARKET DATA ====================
    MARKET_DATA_UPDATE_INTERVAL = int(os.getenv('MARKET_DATA_UPDATE_INTERVAL', 60))
    INTRADAY_DATA_UPDATE_INTERVAL = int(os.getenv('INTRADAY_DATA_UPDATE_INTERVAL', 5))
    ENABLED_ASSET_CLASSES = os.getenv('ENABLED_ASSET_CLASSES', 'stocks,crypto,forex,options,futures').split(',')
    
    # ==================== MACHINE LEARNING ====================
    RL_LEARNING_RATE = float(os.getenv('RL_LEARNING_RATE', 0.001))
    RL_BATCH_SIZE = int(os.getenv('RL_BATCH_SIZE', 32))
    RL_EPISODES = int(os.getenv('RL_EPISODES', 1000))
    RL_EPSILON_DECAY = float(os.getenv('RL_EPSILON_DECAY', 0.995))
    
    ALPHA_WINDOW = int(os.getenv('ALPHA_WINDOW', 20))
    ALPHA_MIN_CORRELATION = float(os.getenv('ALPHA_MIN_CORRELATION', 0.3))
    
    # ==================== NOTIFICATIONS ====================
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    SMTP_USERNAME = os.getenv('SMTP_USERNAME', '')
    SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '')
    ALERT_EMAIL = os.getenv('ALERT_EMAIL', '')
    
    DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
    
    # ==================== SECURITY ====================
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key-change-in-production')
    JWT_EXPIRATION_HOURS = int(os.getenv('JWT_EXPIRATION_HOURS', 24))
    
    RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
    RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', 100))
    RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 3600))
    
    # ==================== BLOCKCHAIN ====================
    WEB3_PROVIDER_URL = os.getenv('WEB3_PROVIDER_URL', '')
    PRIVATE_KEY = os.getenv('PRIVATE_KEY', '')
    WALLET_ADDRESS = os.getenv('WALLET_ADDRESS', '')
    
    # ==================== FEATURE FLAGS ====================
    ENABLE_LIVE_TRADING = os.getenv('ENABLE_LIVE_TRADING', 'false').lower() == 'true'
    ENABLE_SPORTS_BETTING = os.getenv('ENABLE_SPORTS_BETTING', 'true').lower() == 'true'
    ENABLE_DEFI_INTEGRATION = os.getenv('ENABLE_DEFI_INTEGRATION', 'false').lower() == 'true'
    
    @classmethod
    def get_broker_config(cls, broker_name: str) -> Dict[str, Any]:
        """Get configuration for a specific broker"""
        broker_configs = {
            'interactive_brokers': {
                'account_id': cls.IB_ACCOUNT_ID,
                'host': cls.IB_HOST,
                'port': cls.IB_PORT,
            },
            'oanda': {
                'api_key': cls.OANDA_API_KEY,
                'account_id': cls.OANDA_ACCOUNT_ID,
                'environment': cls.OANDA_ENVIRONMENT,
            },
            'alpaca': {
                'api_key': cls.ALPACA_API_KEY,
                'secret_key': cls.ALPACA_SECRET_KEY,
                'base_url': cls.ALPACA_BASE_URL,
            },
            'coinbase': {
                'api_key': cls.COINBASE_API_KEY,
                'secret_key': cls.COINBASE_SECRET_KEY,
                'passphrase': cls.COINBASE_PASSPHRASE,
            },
            'binance': {
                'api_key': cls.BINANCE_API_KEY,
                'secret_key': cls.BINANCE_SECRET_KEY,
            },
            'kraken': {
                'api_key': cls.KRAKEN_API_KEY,
                'secret_key': cls.KRAKEN_SECRET_KEY,
            },
        }
        return broker_configs.get(broker_name.lower(), {})
    
    @classmethod
    def validate_required_configs(cls, required_keys: list) -> bool:
        """Validate that required configuration keys are set"""
        missing_keys = []
        for key in required_keys:
            if not getattr(cls, key, None):
                missing_keys.append(key)
        
        if missing_keys:
            logger.warning(f"Missing required configuration keys: {missing_keys}")
            return False
        return True


class DevelopmentConfig(Config):
    """Development configuration"""
    ENVIRONMENT = 'development'
    DEBUG = True
    ENABLE_LIVE_TRADING = False


class ProductionConfig(Config):
    """Production configuration"""
    ENVIRONMENT = 'production'
    DEBUG = False
    ENABLE_LIVE_TRADING = True
    # In production, ensure all critical configs are set
    
    @classmethod
    def validate(cls):
        """Validate production config has required settings"""
        required = ['DB_PASSWORD', 'FLASK_SECRET_KEY', 'JWT_SECRET_KEY']
        return cls.validate_required_configs(required)


class TestingConfig(Config):
    """Testing configuration"""
    ENVIRONMENT = 'testing'
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    ENABLE_LIVE_TRADING = False


# Configuration factory
def get_config() -> Config:
    """Get appropriate configuration based on environment"""
    environment = os.getenv('ENVIRONMENT', 'development').lower()
    
    config_map = {
        'development': DevelopmentConfig,
        'production': ProductionConfig,
        'testing': TestingConfig,
    }
    
    config_class = config_map.get(environment, DevelopmentConfig)
    logger.info(f"Loaded {config_class.__name__}")
    
    return config_class()


# Initialize logger
def setup_logging():
    """Setup logging configuration"""
    config = get_config()
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    logger.remove()  # Remove default handler
    logger.add(
        config.LOG_FILE,
        level=config.LOG_LEVEL,
        rotation="500 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    logger.add(
        lambda msg: print(msg, end=''),
        level=config.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    )


if __name__ == '__main__':
    # Test configuration loading
    config = get_config()
    print(f"Environment: {config.ENVIRONMENT}")
    print(f"Debug: {config.DEBUG}")
    print(f"Database: {config.DB_NAME}")
    print(f"API Port: {config.API_PORT}")
