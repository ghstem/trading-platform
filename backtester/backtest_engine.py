"""
Backtesting Framework
Event-driven backtesting system for strategy validation
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass, field

from core.trading_engine import (
    Asset, AssetClass, Order, OrderSide, OrderType, 
    OrderStatus, Position, Portfolio
)
from data_pipeline.market_data import get_market_data_manager


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000.0
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"
    commission_pct: float = 0.001  # 0.1% commission
    slippage_pct: float = 0.0005   # 0.05% slippage
    max_orders_per_day: int = 100
    leverage: float = 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'initial_capital': self.initial_capital,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'commission_pct': self.commission_pct,
            'slippage_pct': self.slippage_pct,
            'max_orders_per_day': self.max_orders_per_day,
            'leverage': self.leverage,
        }


@dataclass
class BacktestStats:
    """Backtesting statistics"""
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0
    num_winning_trades: int = 0
    num_losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    final_equity: float = 0.0
    total_pnl: float = 0.0


class Strategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.portfolio = None
        self.market_data = get_market_data_manager()
    
    @abstractmethod
    def on_bar(self, current_date: datetime, current_prices: Dict[Asset, float]):
        """Called on each bar - implement trading logic here"""
        pass
    
    def initialize(self, portfolio: Portfolio):
        """Called once at start of backtest"""
        self.portfolio = portfolio
    
    def finalize(self):
        """Called at end of backtest"""
        pass


class SimpleMovingAverageCrossover(Strategy):
    """Example strategy: SMA crossover"""
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__("SMA Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.asset = None
        self.price_history = []
        self.position_open = False
    
    def set_asset(self, asset: Asset):
        """Set the asset to trade"""
        self.asset = asset
    
    def on_bar(self, current_date: datetime, current_prices: Dict[Asset, float]):
        """Execute strategy logic"""
        if self.asset not in current_prices:
            return
        
        current_price = current_prices[self.asset]
        self.price_history.append(current_price)
        
        # Need enough data for both MAs
        if len(self.price_history) < self.slow_period:
            return
        
        # Calculate moving averages
        fast_ma = np.mean(self.price_history[-self.fast_period:])
        slow_ma = np.mean(self.price_history[-self.slow_period:])
        
        # Trading logic
        position = self.portfolio.get_position(self.asset)
        
        # Buy signal: fast MA crosses above slow MA
        if fast_ma > slow_ma and not self.position_open:
            order = self.portfolio.create_order(
                asset=self.asset,
                side=OrderSide.BUY,
                quantity=10,
                order_type=OrderType.MARKET,
                price=current_price
            )
            self.position_open = True
            logger.info(f"{self.name} - BUY signal at {current_price}")
        
        # Sell signal: fast MA crosses below slow MA
        elif fast_ma < slow_ma and self.position_open:
            if position and position.quantity > 0:
                order = self.portfolio.create_order(
                    asset=self.asset,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    order_type=OrderType.MARKET,
                    price=current_price
                )
                self.position_open = False
                logger.info(f"{self.name} - SELL signal at {current_price}")


class Backtester:
    """Event-driven backtester"""
    
    def __init__(self, strategy: Strategy, config: BacktestConfig = None):
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.portfolio = None
        self.equity_curve = []
        self.drawdown_curve = []
        self.trade_log = []
        self.stats = BacktestStats()
        self.market_data = get_market_data_manager()
    
    def run(self, assets: List[Asset]) -> BacktestStats:
        """Run the backtest"""
        logger.info(f"Starting backtest: {self.strategy.name}")
        logger.info(f"Configuration: {self.config.to_dict()}")
        
        # Initialize portfolio
        self.portfolio = Portfolio(initial_capital=self.config.initial_capital)
        self.strategy.initialize(self.portfolio)
        
        # Prepare data
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        
        # Fetch historical data for all assets
        historical_data = {}
        for asset in assets:
            df = self.market_data.fetch_ohlcv(asset, timeframe="1d", limit=500)
            if not df.empty:
                historical_data[asset] = df
                logger.info(f"Loaded {len(df)} bars for {asset.symbol}")
        
        if not historical_data:
            logger.error("No historical data available")
            return self.stats
        
        # Get common date range
        all_dates = []
        for df in historical_data.values():
            all_dates.extend(df.index.tolist())
        
        all_dates = sorted(set(all_dates))
        all_dates = [d for d in all_dates if start_date <= d <= end_date]
        
        # Backtest loop
        for date_idx, current_date in enumerate(all_dates):
            # Get current prices
            current_prices = {}
            for asset, df in historical_data.items():
                if current_date in df.index:
                    current_prices[asset] = df.loc[current_date, 'close']
            
            if not current_prices:
                continue
            
            # Update portfolio prices
            self.portfolio.update_prices(current_prices)
            
            # Execute strategy
            self.strategy.on_bar(current_date, current_prices)
            
            # Execute pending orders
            for order in self.portfolio.orders:
                if order.status == order.OrderStatus.PENDING and order.asset in current_prices:
                    fill_price = current_prices[order.asset]
                    
                    # Apply slippage
                    if order.side == OrderSide.BUY:
                        fill_price *= (1 + self.config.slippage_pct)
                    else:
                        fill_price *= (1 - self.config.slippage_pct)
                    
                    # Calculate commission
                    commission = order.quantity * fill_price * self.config.commission_pct
                    
                    # Execute order
                    self.portfolio.execute_order(order, fill_price, order.quantity, commission)
            
            # Record equity
            self.equity_curve.append({
                'date': current_date,
                'equity': self.portfolio.total_equity,
                'cash': self.portfolio.cash,
                'pnl': self.portfolio.total_pnl,
            })
        
        # Finalize strategy
        self.strategy.finalize()
        
        # Calculate statistics
        self._calculate_stats()
        
        logger.info(f"Backtest completed: {self.strategy.name}")
        logger.info(f"Final equity: ${self.stats.final_equity:,.2f}")
        logger.info(f"Total return: {self.stats.total_return * 100:.2f}%")
        logger.info(f"Sharpe ratio: {self.stats.sharpe_ratio:.2f}")
        logger.info(f"Max drawdown: {self.stats.max_drawdown * 100:.2f}%")
        
        return self.stats
    
    def _calculate_stats(self):
        """Calculate backtest statistics"""
        if not self.equity_curve:
            return
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)
        
        # Total return
        initial_equity = self.config.initial_capital
        final_equity = self.portfolio.total_equity
        self.stats.final_equity = final_equity
        self.stats.total_pnl = final_equity - initial_equity
        self.stats.total_return = (final_equity - initial_equity) / initial_equity
        
        # Annual return
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365.0
        if years > 0:
            self.stats.annual_return = (self.stats.total_return / years)
        
        # Volatility (annualized)
        daily_returns = equity_df['equity'].pct_change()
        self.stats.volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        if self.stats.volatility > 0:
            self.stats.sharpe_ratio = (self.stats.annual_return / self.stats.volatility)
        
        # Max drawdown
        running_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - running_max) / running_max
        self.stats.max_drawdown = drawdown.min()
        
        # Trade analysis
        self._analyze_trades()
    
    def _analyze_trades(self):
        """Analyze completed trades"""
        trades = []
        buy_price = None
        
        for order in self.portfolio.orders:
            if order.status == OrderStatus.FILLED:
                if order.side == OrderSide.BUY:
                    buy_price = order.average_fill_price
                else:
                    if buy_price:
                        pnl = (order.average_fill_price - buy_price) * order.quantity
                        trades.append({
                            'entry_price': buy_price,
                            'exit_price': order.average_fill_price,
                            'pnl': pnl,
                            'return': (order.average_fill_price - buy_price) / buy_price,
                        })
                        buy_price = None
        
        if trades:
            self.stats.num_trades = len(trades)
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            self.stats.num_winning_trades = len(winning_trades)
            self.stats.num_losing_trades = len(losing_trades)
            self.stats.win_rate = len(winning_trades) / len(trades) if trades else 0
            
            if winning_trades:
                self.stats.avg_win = np.mean([t['pnl'] for t in winning_trades])
            
            if losing_trades:
                self.stats.avg_loss = np.mean([t['pnl'] for t in losing_trades])
            
            total_wins = sum([t['pnl'] for t in winning_trades])
            total_losses = abs(sum([t['pnl'] for t in losing_trades]))
            
            if total_losses > 0:
                self.stats.profit_factor = total_wins / total_losses
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame(self.equity_curve)
    
    def plot_equity_curve(self):
        """Plot equity curve (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            df = self.get_equity_curve()
            plt.figure(figsize=(12, 6))
            plt.plot(df['date'], df['equity'])
            plt.title(f"{self.strategy.name} - Equity Curve")
            plt.xlabel("Date")
            plt.ylabel("Equity ($)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except ImportError:
            logger.warning("matplotlib not installed, cannot plot")
    
    def get_stats_summary(self) -> Dict:
        """Get statistics summary"""
        return {
            'Strategy': self.strategy.name,
            'Initial Capital': f"${self.config.initial_capital:,.2f}",
            'Final Equity': f"${self.stats.final_equity:,.2f}",
            'Total Return': f"{self.stats.total_return * 100:.2f}%",
            'Annual Return': f"{self.stats.annual_return * 100:.2f}%",
            'Volatility': f"{self.stats.volatility * 100:.2f}%",
            'Sharpe Ratio': f"{self.stats.sharpe_ratio:.2f}",
            'Max Drawdown': f"{self.stats.max_drawdown * 100:.2f}%",
            'Total Trades': self.stats.num_trades,
            'Winning Trades': self.stats.num_winning_trades,
            'Losing Trades': self.stats.num_losing_trades,
            'Win Rate': f"{self.stats.win_rate * 100:.2f}%",
            'Profit Factor': f"{self.stats.profit_factor:.2f}",
            'Avg Win': f"${self.stats.avg_win:,.2f}",
            'Avg Loss': f"${self.stats.avg_loss:,.2f}",
        }


# Example usage
if __name__ == "__main__":
    # Create backtest configuration
    config = BacktestConfig(
        initial_capital=100000.0,
        start_date="2023-01-01",
        end_date="2024-01-01",
        commission_pct=0.001,
    )
    
    # Create strategy
    strategy = SimpleMovingAverageCrossover(fast_period=20, slow_period=50)
    
    # Create asset
    apple = Asset(symbol="AAPL", asset_class=AssetClass.STOCK, exchange="NASDAQ")
    strategy.set_asset(apple)
    
    # Run backtest
    backtester = Backtester(strategy, config)
    stats = backtester.run([apple])
    
    # Print results
    print(pd.DataFrame([backtester.get_stats_summary()]).T)
