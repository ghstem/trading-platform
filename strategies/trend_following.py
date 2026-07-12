"""
Trend-Following Strategies
Covers: SMA Crossover, EMA Crossover, MACD Crossover
"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.trading_engine import Asset, OrderSide, OrderType, Portfolio
from strategies.base import BaseStrategy, Signal, SignalType


class SMACrossoverStrategy(BaseStrategy):
    """
    Simple Moving Average crossover.
    BUY when fast MA crosses above slow MA.
    SELL when fast MA crosses below slow MA.
    """

    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        assets: Optional[List[Asset]] = None,
    ):
        super().__init__(f"SMA_Crossover_{fast_period}_{slow_period}", assets)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._price_history: Dict[str, List[float]] = {}
        self._in_position: Dict[str, bool] = {}

    def generate_signals(self, ohlcv_data: pd.DataFrame, asset: Asset) -> List[Signal]:
        signals: List[Signal] = []
        if len(ohlcv_data) < self.slow_period:
            return signals

        close = ohlcv_data["close"]
        fast_ma = close.rolling(self.fast_period).mean()
        slow_ma = close.rolling(self.slow_period).mean()

        prev_fast, curr_fast = fast_ma.iloc[-2], fast_ma.iloc[-1]
        prev_slow, curr_slow = slow_ma.iloc[-2], slow_ma.iloc[-1]

        in_pos = self._in_position.get(asset.symbol, False)
        current_price = float(close.iloc[-1])

        if prev_fast <= prev_slow and curr_fast > curr_slow and not in_pos:
            sig = Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.BUY,
                strength=min(abs(curr_fast - curr_slow) / curr_slow, 1.0),
                price=current_price,
                metadata={"fast_ma": round(curr_fast, 4), "slow_ma": round(curr_slow, 4)},
            )
            signals.append(sig)
        elif prev_fast >= prev_slow and curr_fast < curr_slow and in_pos:
            sig = Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.SELL,
                strength=1.0,
                price=current_price,
                metadata={"fast_ma": round(curr_fast, 4), "slow_ma": round(curr_slow, 4)},
            )
            signals.append(sig)

        return signals

    def on_bar(self, current_date: datetime, current_prices: Dict[Asset, float]) -> List[Signal]:
        all_signals: List[Signal] = []
        for asset in self.assets:
            price = current_prices.get(asset)
            if price is None:
                continue
            hist = self._price_history.setdefault(asset.symbol, [])
            hist.append(price)
            if len(hist) < self.slow_period + 1:
                continue

            df = pd.DataFrame({"close": hist})
            sigs = self.generate_signals(df, asset)
            for sig in sigs:
                if sig.signal_type == SignalType.BUY:
                    self._in_position[asset.symbol] = True
                elif sig.signal_type == SignalType.SELL:
                    self._in_position[asset.symbol] = False
                self.record_signal(sig)
            all_signals.extend(sigs)

        return all_signals


class EMACrossoverStrategy(BaseStrategy):
    """
    Exponential Moving Average crossover.
    Uses EMA instead of SMA for faster reaction to recent prices.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        assets: Optional[List[Asset]] = None,
    ):
        super().__init__(f"EMA_Crossover_{fast_period}_{slow_period}", assets)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._price_history: Dict[str, List[float]] = {}
        self._in_position: Dict[str, bool] = {}

    def generate_signals(self, ohlcv_data: pd.DataFrame, asset: Asset) -> List[Signal]:
        signals: List[Signal] = []
        if len(ohlcv_data) < self.slow_period + 1:
            return signals

        close = ohlcv_data["close"]
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        prev_fast, curr_fast = fast_ema.iloc[-2], fast_ema.iloc[-1]
        prev_slow, curr_slow = slow_ema.iloc[-2], slow_ema.iloc[-1]

        in_pos = self._in_position.get(asset.symbol, False)
        current_price = float(close.iloc[-1])

        if prev_fast <= prev_slow and curr_fast > curr_slow and not in_pos:
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.BUY,
                strength=min(abs(curr_fast - curr_slow) / curr_slow, 1.0),
                price=current_price,
                metadata={"fast_ema": round(curr_fast, 4), "slow_ema": round(curr_slow, 4)},
            ))
        elif prev_fast >= prev_slow and curr_fast < curr_slow and in_pos:
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.SELL,
                strength=1.0,
                price=current_price,
            ))

        return signals

    def on_bar(self, current_date: datetime, current_prices: Dict[Asset, float]) -> List[Signal]:
        all_signals: List[Signal] = []
        for asset in self.assets:
            price = current_prices.get(asset)
            if price is None:
                continue
            hist = self._price_history.setdefault(asset.symbol, [])
            hist.append(price)
            if len(hist) < self.slow_period + 1:
                continue

            df = pd.DataFrame({"close": hist})
            sigs = self.generate_signals(df, asset)
            for sig in sigs:
                if sig.signal_type == SignalType.BUY:
                    self._in_position[asset.symbol] = True
                elif sig.signal_type == SignalType.SELL:
                    self._in_position[asset.symbol] = False
                self.record_signal(sig)
            all_signals.extend(sigs)

        return all_signals


class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence/Divergence) strategy.
    BUY when MACD line crosses above signal line.
    SELL when MACD line crosses below signal line.
    """

    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        assets: Optional[List[Asset]] = None,
    ):
        super().__init__(f"MACD_{fast_period}_{slow_period}_{signal_period}", assets)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self._price_history: Dict[str, List[float]] = {}
        self._in_position: Dict[str, bool] = {}

    def generate_signals(self, ohlcv_data: pd.DataFrame, asset: Asset) -> List[Signal]:
        signals: List[Signal] = []
        min_bars = self.slow_period + self.signal_period
        if len(ohlcv_data) < min_bars + 1:
            return signals

        close = ohlcv_data["close"]
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        prev_macd, curr_macd = macd_line.iloc[-2], macd_line.iloc[-1]
        prev_sig, curr_sig = signal_line.iloc[-2], signal_line.iloc[-1]

        in_pos = self._in_position.get(asset.symbol, False)
        current_price = float(close.iloc[-1])

        if prev_macd <= prev_sig and curr_macd > curr_sig and not in_pos:
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.BUY,
                strength=min(abs(curr_macd - curr_sig) / (abs(curr_sig) + 1e-8), 1.0),
                price=current_price,
                metadata={"macd": round(curr_macd, 6), "signal": round(curr_sig, 6)},
            ))
        elif prev_macd >= prev_sig and curr_macd < curr_sig and in_pos:
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.SELL,
                strength=1.0,
                price=current_price,
                metadata={"macd": round(curr_macd, 6), "signal": round(curr_sig, 6)},
            ))

        return signals

    def on_bar(self, current_date: datetime, current_prices: Dict[Asset, float]) -> List[Signal]:
        all_signals: List[Signal] = []
        for asset in self.assets:
            price = current_prices.get(asset)
            if price is None:
                continue
            hist = self._price_history.setdefault(asset.symbol, [])
            hist.append(price)
            min_bars = self.slow_period + self.signal_period
            if len(hist) < min_bars + 1:
                continue

            df = pd.DataFrame({"close": hist})
            sigs = self.generate_signals(df, asset)
            for sig in sigs:
                if sig.signal_type == SignalType.BUY:
                    self._in_position[asset.symbol] = True
                elif sig.signal_type == SignalType.SELL:
                    self._in_position[asset.symbol] = False
                self.record_signal(sig)
            all_signals.extend(sigs)

        return all_signals
