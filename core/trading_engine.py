"""
Core Trading Engine
Central hub for managing trades, positions, and portfolio across multiple markets
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import uuid
from loguru import logger
import pandas as pd
import numpy as np


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order side: Buy or Sell"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class AssetClass(Enum):
    """Asset classes supported by the platform"""
    STOCK = "STOCK"
    CRYPTO = "CRYPTO"
    FOREX = "FOREX"
    OPTION = "OPTION"
    FUTURE = "FUTURE"
    COMMODITY = "COMMODITY"


@dataclass
class Asset:
    """Represents a tradable asset"""
    symbol: str
    asset_class: AssetClass
    exchange: str
    currency: str = "USD"
    multiplier: float = 1.0
    
    def __hash__(self):
        return hash((self.symbol, self.asset_class, self.exchange))
    
    def __eq__(self, other):
        if not isinstance(other, Asset):
            return False
        return (self.symbol == other.symbol and 
                self.asset_class == other.asset_class and 
                self.exchange == other.exchange)


@dataclass
class Order:
    """Represents a trading order"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    asset: Asset = None
    side: OrderSide = None
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: float = 0.0
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    broker: str = ""
    
    def __post_init__(self):
        if self.asset is None:
            raise ValueError("Asset is required for an order")
        if self.side is None:
            raise ValueError("Side (BUY/SELL) is required for an order")
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_open(self) -> bool:
        """Check if order is still open"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def total_cost(self) -> float:
        """Total cost of the order including commission"""
        return (self.filled_quantity * self.average_fill_price) + self.commission
    
    def update_fill(self, filled_qty: float, fill_price: float, commission: float = 0.0):
        """Update order with fill information"""
        total_cost_before = self.filled_quantity * self.average_fill_price
        self.filled_quantity += filled_qty
        
        # Recalculate average fill price
        if self.filled_quantity > 0:
            self.average_fill_price = (total_cost_before + (filled_qty * fill_price)) / self.filled_quantity
        
        self.commission += commission
        
        # Update status
        if abs(self.filled_quantity - self.quantity) < 1e-8:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.now()
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        logger.info(f"Order {self.order_id} updated: {self.filled_quantity}/{self.quantity} filled at ${fill_price}")


@dataclass
class Position:
    """Represents a position in an asset"""
    asset: Asset
    quantity: float = 0.0
    average_entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __hash__(self):
        return hash((self.asset.symbol, self.asset.asset_class, self.asset.exchange))
    
    @property
    def market_value(self) -> float:
        """Current market value of position"""
        return self.quantity * self.current_price * self.asset.multiplier
    
    @property
    def entry_cost(self) -> float:
        """Cost basis of the position"""
        return self.quantity * self.average_entry_price * self.asset.multiplier
    
    @property
    def pnl(self) -> float:
        """Total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def pnl_percent(self) -> float:
        """P&L as percentage"""
        if self.entry_cost == 0:
            return 0.0
        return (self.pnl / abs(self.entry_cost)) * 100
    
    def update_price(self, current_price: float):
        """Update current price and calculate unrealized P&L"""
        self.current_price = current_price
        
        if self.quantity != 0:
            self.unrealized_pnl = (current_price - self.average_entry_price) * self.quantity * self.asset.multiplier
        else:
            self.unrealized_pnl = 0.0
        
        self.last_updated = datetime.now()
    
    def add_to_position(self, quantity: float, price: float):
        """Add to existing position"""
        total_cost = (self.quantity * self.average_entry_price) + (quantity * price)
        self.quantity += quantity
        
        if self.quantity != 0:
            self.average_entry_price = total_cost / self.quantity
    
    def reduce_position(self, quantity: float, sell_price: float) -> float:
        """Reduce position and realize P&L"""
        quantity_to_reduce = min(quantity, self.quantity)
        pnl = (sell_price - self.average_entry_price) * quantity_to_reduce * self.asset.multiplier
        
        self.realized_pnl += pnl
        self.quantity -= quantity_to_reduce
        
        return pnl


class Portfolio:
    """Manages the overall portfolio across all positions"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[Asset, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Tuple[Order, Order]] = []  # Entry/exit order pairs
        self.created_at = datetime.now()
    
    @property
    def total_equity(self) -> float:
        """Total equity (cash + positions market value)"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def portfolio_value(self) -> float:
        """Alias for total equity"""
        return self.total_equity
    
    @property
    def total_pnl(self) -> float:
        """Total realized + unrealized P&L"""
        return sum(pos.pnl for pos in self.positions.values())
    
    @property
    def total_pnl_percent(self) -> float:
        """Total P&L as percentage"""
        if self.initial_capital == 0:
            return 0.0
        return (self.total_pnl / self.initial_capital) * 100
    
    @property
    def margin_used(self) -> float:
        """Margin used by open positions"""
        return sum(abs(pos.entry_cost) for pos in self.positions.values() if pos.quantity != 0)
    
    @property
    def cash_utilization(self) -> float:
        """Percentage of capital deployed"""
        if self.initial_capital == 0:
            return 0.0
        return (self.margin_used / self.initial_capital) * 100
    
    def get_position(self, asset: Asset) -> Optional[Position]:
        """Get position for an asset"""
        return self.positions.get(asset)
    
    def create_order(self, asset: Asset, side: OrderSide, quantity: float,
                    order_type: OrderType = OrderType.MARKET, price: float = 0.0,
                    stop_price: Optional[float] = None, broker: str = "default") -> Order:
        """Create a new order"""
        
        # Validate order
        if quantity <= 0:
            raise ValueError("Order quantity must be positive")
        
        # Check if sufficient capital for buy orders
        if side == OrderSide.BUY and order_type == OrderType.MARKET:
            estimated_cost = quantity * price
            if estimated_cost > self.cash:
                logger.warning(f"Insufficient cash. Required: ${estimated_cost}, Available: ${self.cash}")
        
        order = Order(
            asset=asset,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            broker=broker
        )
        
        self.orders.append(order)
        logger.info(f"Created order: {order.order_id} - {side.value} {quantity} {asset.symbol}")
        
        return order
    
    def execute_order(self, order: Order, fill_price: float, filled_qty: float, commission: float = 0.0):
        """Execute an order"""
        
        # Update order
        order.update_fill(filled_qty, fill_price, commission)
        
        # Update position
        asset = order.asset
        if asset not in self.positions:
            self.positions[asset] = Position(asset=asset, current_price=fill_price)
        
        position = self.positions[asset]
        
        if order.side == OrderSide.BUY:
            position.add_to_position(filled_qty, fill_price)
            self.cash -= (filled_qty * fill_price) + commission
        else:  # SELL
            pnl = position.reduce_position(filled_qty, fill_price)
            self.cash += (filled_qty * fill_price) - commission
            
            if position.quantity == 0:
                del self.positions[asset]
        
        logger.info(f"Order executed: {filled_qty} shares of {asset.symbol} at ${fill_price}")
    
    def update_prices(self, price_data: Dict[Asset, float]):
        """Update prices for all positions"""
        for asset, price in price_data.items():
            if asset in self.positions:
                self.positions[asset].update_price(price)
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary statistics"""
        return {
            'total_equity': self.total_equity,
            'cash': self.cash,
            'positions_value': sum(pos.market_value for pos in self.positions.values()),
            'total_pnl': self.total_pnl,
            'total_pnl_percent': self.total_pnl_percent,
            'num_positions': len(self.positions),
            'cash_utilization': self.cash_utilization,
            'margin_used': self.margin_used,
        }
    
    def close_position(self, asset: Asset, close_price: float, broker: str = "default") -> Optional[Order]:
        """Close a position"""
        position = self.positions.get(asset)
        if position is None or position.quantity == 0:
            logger.warning(f"No position to close for {asset.symbol}")
            return None
        
        # Create sell order
        sell_order = self.create_order(
            asset=asset,
            side=OrderSide.SELL,
            quantity=abs(position.quantity),
            order_type=OrderType.MARKET,
            price=close_price,
            broker=broker
        )
        
        # Execute order
        self.execute_order(sell_order, close_price, abs(position.quantity))
        
        return sell_order
    
    def get_positions_df(self) -> pd.DataFrame:
        """Get positions as a pandas DataFrame"""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for asset, position in self.positions.items():
            data.append({
                'Symbol': asset.symbol,
                'Asset Class': asset.asset_class.value,
                'Quantity': position.quantity,
                'Entry Price': position.average_entry_price,
                'Current Price': position.current_price,
                'Market Value': position.market_value,
                'Unrealized P&L': position.unrealized_pnl,
                'Realized P&L': position.realized_pnl,
                'Total P&L': position.pnl,
                'P&L %': position.pnl_percent,
            })
        
        return pd.DataFrame(data)


# Example usage
if __name__ == "__main__":
    # Create a portfolio
    portfolio = Portfolio(initial_capital=100000.0)
    
    # Create assets
    apple = Asset(symbol="AAPL", asset_class=AssetClass.STOCK, exchange="NASDAQ", currency="USD")
    bitcoin = Asset(symbol="BTC", asset_class=AssetClass.CRYPTO, exchange="COINBASE", currency="USD")
    
    # Create and execute orders
    buy_order = portfolio.create_order(apple, OrderSide.BUY, 10, OrderType.MARKET, price=150.0)
    portfolio.execute_order(buy_order, fill_price=150.0, filled_qty=10, commission=10.0)
    
    # Update prices
    portfolio.update_prices({apple: 155.0})
    
    # Get summary
    print(portfolio.get_portfolio_summary())
    print(portfolio.get_positions_df())
