"""
Base Strategy Interface
All trading strategies implement this contract, enabling the StrategyManager
to treat them uniformly regardless of underlying logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from core.trading_engine import Asset, Order, Portfolio


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


@dataclass
class Signal:
    """A trading signal emitted by a strategy"""
    strategy_name: str
    asset: Asset
    signal_type: SignalType
    strength: float = 1.0          # 0.0 – 1.0, used for position sizing
    price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return (
            f"Signal({self.signal_type.value} {self.asset.symbol} "
            f"@ ${self.price:.2f} strength={self.strength:.2f})"
        )


@dataclass
class StrategyPerformance:
    """Running performance statistics for a strategy"""
    total_signals: int = 0
    buy_signals: int = 0
    sell_signals: int = 0
    trades_executed: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    started_at: Optional[datetime] = None
    last_signal_at: Optional[datetime] = None

    @property
    def win_rate(self) -> float:
        total = self.winning_trades + self.losing_trades
        return self.winning_trades / total if total > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "total_signals": self.total_signals,
            "buy_signals": self.buy_signals,
            "sell_signals": self.sell_signals,
            "trades_executed": self.trades_executed,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_signal_at": self.last_signal_at.isoformat() if self.last_signal_at else None,
        }


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Subclasses must implement:
      - ``generate_signals(ohlcv_data, asset)`` — pure signal logic, no side-effects
      - ``on_bar(current_date, current_prices)`` — called each bar during live/backtest run

    Optional hooks:
      - ``initialize(portfolio)`` — called once before the first bar
      - ``finalize()`` — called after the last bar
    """

    def __init__(self, name: str, assets: Optional[List[Asset]] = None):
        self.name = name
        self.assets: List[Asset] = assets or []
        self.portfolio: Optional[Portfolio] = None
        self.performance = StrategyPerformance()
        self._signal_history: List[Signal] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signals(self, ohlcv_data: pd.DataFrame, asset: Asset) -> List[Signal]:
        """
        Pure signal generation from OHLCV data.
        Must not modify portfolio state directly.
        """

    @abstractmethod
    def on_bar(self, current_date: datetime, current_prices: Dict[Asset, float]) -> List[Signal]:
        """
        Called on each new bar (live or backtest).
        Returns a list of signals for the StrategyManager to act on.
        """

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional overrides)
    # ------------------------------------------------------------------

    def initialize(self, portfolio: Portfolio) -> None:
        """Called once before the first bar."""
        self.portfolio = portfolio
        self.performance.started_at = datetime.now()
        logger.info(f"Strategy '{self.name}' initialized")

    def finalize(self) -> None:
        """Called after the last bar."""
        logger.info(
            f"Strategy '{self.name}' finalized — "
            f"signals={self.performance.total_signals}, "
            f"pnl={self.performance.total_pnl:.2f}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_assets(self, assets: List[Asset]) -> None:
        self.assets = assets

    def record_signal(self, signal: Signal) -> None:
        """Track signal for performance stats."""
        self._signal_history.append(signal)
        self.performance.total_signals += 1
        self.performance.last_signal_at = signal.generated_at
        if signal.signal_type == SignalType.BUY:
            self.performance.buy_signals += 1
        elif signal.signal_type in (SignalType.SELL, SignalType.CLOSE):
            self.performance.sell_signals += 1

    def record_trade_result(self, pnl: float) -> None:
        """Update performance after a trade closes."""
        self.performance.trades_executed += 1
        self.performance.total_pnl += pnl
        if pnl >= 0:
            self.performance.winning_trades += 1
        else:
            self.performance.losing_trades += 1

    def get_signal_history(self) -> List[Signal]:
        return list(self._signal_history)

    def get_performance(self) -> Dict:
        return {"strategy": self.name, **self.performance.to_dict()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', assets={[a.symbol for a in self.assets]})"
