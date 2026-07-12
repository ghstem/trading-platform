"""
Mean-Reversion Strategies
Covers: Bollinger Bands, RSI Extremes, Z-Score Reversion
"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.trading_engine import Asset
from strategies.base import BaseStrategy, Signal, SignalType


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands mean-reversion strategy.
    BUY when price touches / crosses below the lower band (oversold).
    SELL / CLOSE when price touches / crosses above the upper band (overbought).
    """

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        assets: Optional[List[Asset]] = None,
    ):
        super().__init__(f"BollingerBands_{period}_{num_std}", assets)
        self.period = period
        self.num_std = num_std
        self._price_history: Dict[str, List[float]] = {}
        self._in_position: Dict[str, bool] = {}

    def generate_signals(self, ohlcv_data: pd.DataFrame, asset: Asset) -> List[Signal]:
        signals: List[Signal] = []
        if len(ohlcv_data) < self.period + 1:
            return signals

        close = ohlcv_data["close"]
        sma = close.rolling(self.period).mean()
        std = close.rolling(self.period).std()
        upper = sma + self.num_std * std
        lower = sma - self.num_std * std

        current_price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        curr_lower = float(lower.iloc[-1])
        prev_lower = float(lower.iloc[-2])
        curr_upper = float(upper.iloc[-1])
        prev_upper = float(upper.iloc[-2])

        in_pos = self._in_position.get(asset.symbol, False)

        # Price crosses below lower band → BUY (mean reversion up expected)
        if prev_price >= prev_lower and current_price < curr_lower and not in_pos:
            band_width = curr_upper - curr_lower
            strength = min((curr_lower - current_price) / (band_width + 1e-8) + 0.5, 1.0)
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.BUY,
                strength=strength,
                price=current_price,
                stop_loss=current_price * 0.98,
                take_profit=float(sma.iloc[-1]),
                metadata={
                    "upper_band": round(curr_upper, 4),
                    "lower_band": round(curr_lower, 4),
                    "sma": round(float(sma.iloc[-1]), 4),
                },
            ))

        # Price crosses above upper band → CLOSE / SELL
        elif prev_price <= prev_upper and current_price > curr_upper and in_pos:
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.CLOSE,
                strength=1.0,
                price=current_price,
                metadata={"upper_band": round(curr_upper, 4)},
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


class RSIReversionStrategy(BaseStrategy):
    """
    RSI-based mean-reversion strategy.
    BUY when RSI drops below oversold threshold.
    SELL when RSI rises above overbought threshold.
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        assets: Optional[List[Asset]] = None,
    ):
        super().__init__(f"RSI_Reversion_{period}", assets)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self._price_history: Dict[str, List[float]] = {}
        self._in_position: Dict[str, bool] = {}

    def _compute_rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.period).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def generate_signals(self, ohlcv_data: pd.DataFrame, asset: Asset) -> List[Signal]:
        signals: List[Signal] = []
        if len(ohlcv_data) < self.period + 1:
            return signals

        close = ohlcv_data["close"]
        rsi = self._compute_rsi(close)
        curr_rsi = float(rsi.iloc[-1])
        prev_rsi = float(rsi.iloc[-2])
        current_price = float(close.iloc[-1])

        in_pos = self._in_position.get(asset.symbol, False)

        # RSI crosses up through oversold → BUY
        if prev_rsi < self.oversold and curr_rsi >= self.oversold and not in_pos:
            strength = (self.oversold - prev_rsi) / self.oversold
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.BUY,
                strength=min(strength, 1.0),
                price=current_price,
                metadata={"rsi": round(curr_rsi, 2)},
            ))

        # RSI crosses down through overbought → SELL
        elif prev_rsi > self.overbought and curr_rsi <= self.overbought and in_pos:
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.SELL,
                strength=1.0,
                price=current_price,
                metadata={"rsi": round(curr_rsi, 2)},
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


class ZScoreReversionStrategy(BaseStrategy):
    """
    Z-Score mean-reversion strategy.
    Computes how many standard deviations the current price is from its rolling mean.
    BUY when z-score < -entry_threshold (price far below mean).
    CLOSE when z-score returns above exit_threshold.
    """

    def __init__(
        self,
        lookback: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        assets: Optional[List[Asset]] = None,
    ):
        super().__init__(f"ZScore_Reversion_{lookback}", assets)
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self._price_history: Dict[str, List[float]] = {}
        self._in_position: Dict[str, bool] = {}

    def generate_signals(self, ohlcv_data: pd.DataFrame, asset: Asset) -> List[Signal]:
        signals: List[Signal] = []
        if len(ohlcv_data) < self.lookback + 1:
            return signals

        close = ohlcv_data["close"]
        mean = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std()
        zscore = (close - mean) / (std + 1e-8)

        curr_z = float(zscore.iloc[-1])
        current_price = float(close.iloc[-1])
        in_pos = self._in_position.get(asset.symbol, False)

        if curr_z < -self.entry_threshold and not in_pos:
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.BUY,
                strength=min(abs(curr_z) / self.entry_threshold - 1, 1.0),
                price=current_price,
                metadata={"zscore": round(curr_z, 4)},
            ))
        elif abs(curr_z) < self.exit_threshold and in_pos:
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.CLOSE,
                strength=1.0,
                price=current_price,
                metadata={"zscore": round(curr_z, 4)},
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
            if len(hist) < self.lookback + 1:
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
