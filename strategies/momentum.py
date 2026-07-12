"""
Momentum Strategies
Covers: Price Momentum (cross-sectional), Breakout
"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.trading_engine import Asset
from strategies.base import BaseStrategy, Signal, SignalType


class PriceMomentumStrategy(BaseStrategy):
    """
    Price momentum strategy.
    Ranks assets by their N-period return and generates BUY signals for the
    top performers, SELL signals for the bottom performers.

    Works best when ``assets`` contains multiple instruments so relative
    ranking is meaningful.
    """

    def __init__(
        self,
        lookback: int = 20,
        top_n: int = 1,
        assets: Optional[List[Asset]] = None,
    ):
        super().__init__(f"Price_Momentum_{lookback}", assets)
        self.lookback = lookback
        self.top_n = top_n
        self._price_history: Dict[str, List[float]] = {}
        self._in_position: Dict[str, bool] = {}

    def generate_signals(self, ohlcv_data: pd.DataFrame, asset: Asset) -> List[Signal]:
        """
        Single-asset signal: BUY when momentum (N-period return) is positive
        and above the median of its own history; SELL/CLOSE otherwise.
        """
        signals: List[Signal] = []
        if len(ohlcv_data) < self.lookback + 1:
            return signals

        close = ohlcv_data["close"]
        momentum = float(close.iloc[-1] / close.iloc[-1 - self.lookback] - 1)
        current_price = float(close.iloc[-1])
        in_pos = self._in_position.get(asset.symbol, False)

        if momentum > 0 and not in_pos:
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.BUY,
                strength=min(abs(momentum), 1.0),
                price=current_price,
                metadata={"momentum_pct": round(momentum * 100, 4)},
            ))
        elif momentum < 0 and in_pos:
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.SELL,
                strength=1.0,
                price=current_price,
                metadata={"momentum_pct": round(momentum * 100, 4)},
            ))

        return signals

    def on_bar(self, current_date: datetime, current_prices: Dict[Asset, float]) -> List[Signal]:
        # Build price histories
        for asset in self.assets:
            price = current_prices.get(asset)
            if price is None:
                continue
            hist = self._price_history.setdefault(asset.symbol, [])
            hist.append(price)

        # Rank all assets by momentum
        momentums: Dict[str, float] = {}
        for asset in self.assets:
            hist = self._price_history.get(asset.symbol, [])
            if len(hist) >= self.lookback + 1:
                momentums[asset.symbol] = hist[-1] / hist[-1 - self.lookback] - 1

        if not momentums:
            return []

        ranked = sorted(momentums, key=momentums.get, reverse=True)
        top_symbols = set(ranked[: self.top_n])
        bottom_symbols = set(ranked[-self.top_n :]) if len(ranked) >= self.top_n else set()

        all_signals: List[Signal] = []
        for asset in self.assets:
            price = current_prices.get(asset)
            if price is None or asset.symbol not in momentums:
                continue

            in_pos = self._in_position.get(asset.symbol, False)
            mom = momentums[asset.symbol]

            if asset.symbol in top_symbols and not in_pos:
                sig = Signal(
                    strategy_name=self.name,
                    asset=asset,
                    signal_type=SignalType.BUY,
                    strength=min(abs(mom), 1.0),
                    price=price,
                    metadata={"momentum_pct": round(mom * 100, 4), "rank": ranked.index(asset.symbol) + 1},
                )
                self._in_position[asset.symbol] = True
                self.record_signal(sig)
                all_signals.append(sig)
            elif asset.symbol in bottom_symbols and in_pos:
                sig = Signal(
                    strategy_name=self.name,
                    asset=asset,
                    signal_type=SignalType.SELL,
                    strength=1.0,
                    price=price,
                    metadata={"momentum_pct": round(mom * 100, 4)},
                )
                self._in_position[asset.symbol] = False
                self.record_signal(sig)
                all_signals.append(sig)

        return all_signals


class BreakoutStrategy(BaseStrategy):
    """
    Price breakout / channel breakout strategy.
    BUY when price closes above the N-period high (breakout up).
    SELL when price closes below the N-period low (breakout down / stop).
    """

    def __init__(
        self,
        period: int = 20,
        atr_multiplier: float = 1.5,
        assets: Optional[List[Asset]] = None,
    ):
        super().__init__(f"Breakout_{period}", assets)
        self.period = period
        self.atr_multiplier = atr_multiplier
        self._price_history: Dict[str, List[float]] = {}
        self._in_position: Dict[str, bool] = {}

    def _compute_atr(self, close: pd.Series, period: int) -> pd.Series:
        """Approximated ATR using close-to-close ranges."""
        tr = close.diff().abs()
        return tr.rolling(period).mean()

    def generate_signals(self, ohlcv_data: pd.DataFrame, asset: Asset) -> List[Signal]:
        signals: List[Signal] = []
        if len(ohlcv_data) < self.period + 1:
            return signals

        close = ohlcv_data["close"]
        period_high = close.rolling(self.period).max()
        period_low = close.rolling(self.period).min()
        atr = self._compute_atr(close, self.period)

        current_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        curr_high = float(period_high.iloc[-2])   # use yesterday's channel to avoid look-ahead
        curr_low = float(period_low.iloc[-2])
        curr_atr = float(atr.iloc[-1])

        in_pos = self._in_position.get(asset.symbol, False)

        if prev_price <= curr_high and current_price > curr_high and not in_pos:
            stop = current_price - self.atr_multiplier * curr_atr
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.BUY,
                strength=min((current_price - curr_high) / (curr_atr + 1e-8), 1.0),
                price=current_price,
                stop_loss=stop,
                take_profit=current_price + 2 * self.atr_multiplier * curr_atr,
                metadata={"channel_high": round(curr_high, 4), "atr": round(curr_atr, 4)},
            ))

        elif current_price < curr_low and in_pos:
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.CLOSE,
                strength=1.0,
                price=current_price,
                metadata={"channel_low": round(curr_low, 4)},
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
            if len(hist) < self.period + 1:
                continue

            df = pd.DataFrame({"close": hist})
            sigs = self.generate_signals(df, asset)
            for sig in sigs:
                if sig.signal_type == SignalType.BUY:
                    self._in_position[asset.symbol] = True
                elif sig.signal_type in (SignalType.SELL, SignalType.CLOSE):
                    self._in_position[asset.symbol] = False
                self.record_signal(sig)
            all_signals.extend(sigs)

        return all_signals
