"""
Pairs Trading (Statistical Arbitrage) Strategy
Trades the spread between two correlated assets.

Entry: spread deviates beyond Z-score entry threshold (mean-reversion bet)
Exit:  spread reverts toward zero
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from core.trading_engine import Asset
from strategies.base import BaseStrategy, Signal, SignalType


class PairsTradingStrategy(BaseStrategy):
    """
    Statistical pairs trading strategy.

    Requires exactly two assets.  The spread is defined as:
        spread = log(price_A) - hedge_ratio * log(price_B)

    A BUY signal means: buy asset A / sell asset B (spread expected to widen).
    A SELL signal means: sell asset A / buy asset B (spread expected to narrow).

    Parameters
    ----------
    lookback         : rolling window for computing spread statistics
    entry_threshold  : z-score magnitude to enter a trade
    exit_threshold   : z-score magnitude (near 0) to exit
    hedge_ratio      : fixed hedge ratio; if None it is estimated via OLS
    """

    def __init__(
        self,
        lookback: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        hedge_ratio: Optional[float] = None,
        assets: Optional[List[Asset]] = None,
    ):
        super().__init__("Pairs_Trading", assets)
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.fixed_hedge_ratio = hedge_ratio
        self._price_history: Dict[str, List[float]] = {}
        self._position_direction: int = 0   # +1 long spread, -1 short spread, 0 flat

    # ------------------------------------------------------------------

    @property
    def asset_a(self) -> Optional[Asset]:
        return self.assets[0] if len(self.assets) >= 1 else None

    @property
    def asset_b(self) -> Optional[Asset]:
        return self.assets[1] if len(self.assets) >= 2 else None

    def _estimate_hedge_ratio(self, prices_a: np.ndarray, prices_b: np.ndarray) -> float:
        """OLS hedge ratio: regress log(A) on log(B)."""
        log_a = np.log(prices_a)
        log_b = np.log(prices_b)
        X = np.column_stack([np.ones(len(log_b)), log_b])
        coef, *_ = np.linalg.lstsq(X, log_a, rcond=None)
        return float(coef[1])

    def _compute_spread_zscore(
        self, prices_a: List[float], prices_b: List[float]
    ) -> Tuple[float, float]:
        """
        Returns (current_zscore, hedge_ratio) using the last ``lookback`` bars.
        """
        n = min(len(prices_a), len(prices_b), self.lookback)
        arr_a = np.array(prices_a[-n:], dtype=float)
        arr_b = np.array(prices_b[-n:], dtype=float)

        hr = self.fixed_hedge_ratio
        if hr is None:
            hr = self._estimate_hedge_ratio(arr_a, arr_b)

        spread = np.log(arr_a) - hr * np.log(arr_b)
        mean = spread.mean()
        std = spread.std() + 1e-8
        zscore = (spread[-1] - mean) / std
        return float(zscore), float(hr)

    # ------------------------------------------------------------------

    def generate_signals(self, ohlcv_data: pd.DataFrame, asset: Asset) -> List[Signal]:
        """Not used directly for pairs (we need two assets); implemented to satisfy ABC."""
        return []

    def on_bar(self, current_date: datetime, current_prices: Dict[Asset, float]) -> List[Signal]:
        if self.asset_a is None or self.asset_b is None:
            logger.warning(f"{self.name}: requires exactly 2 assets")
            return []

        price_a = current_prices.get(self.asset_a)
        price_b = current_prices.get(self.asset_b)
        if price_a is None or price_b is None:
            return []

        hist_a = self._price_history.setdefault(self.asset_a.symbol, [])
        hist_b = self._price_history.setdefault(self.asset_b.symbol, [])
        hist_a.append(price_a)
        hist_b.append(price_b)

        min_bars = self.lookback
        if len(hist_a) < min_bars or len(hist_b) < min_bars:
            return []

        zscore, hedge_ratio = self._compute_spread_zscore(hist_a, hist_b)
        signals: List[Signal] = []

        metadata = {
            "zscore": round(zscore, 4),
            "hedge_ratio": round(hedge_ratio, 4),
            f"price_{self.asset_a.symbol}": price_a,
            f"price_{self.asset_b.symbol}": price_b,
        }

        # Spread too low → long A / short B
        if zscore < -self.entry_threshold and self._position_direction == 0:
            signals.append(Signal(
                strategy_name=self.name,
                asset=self.asset_a,
                signal_type=SignalType.BUY,
                strength=min(abs(zscore) / self.entry_threshold - 1, 1.0),
                price=price_a,
                metadata={**metadata, "leg": "long"},
            ))
            signals.append(Signal(
                strategy_name=self.name,
                asset=self.asset_b,
                signal_type=SignalType.SELL,
                strength=min(abs(zscore) / self.entry_threshold - 1, 1.0),
                price=price_b,
                metadata={**metadata, "leg": "short"},
            ))
            self._position_direction = 1
            logger.info(f"{self.name}: Enter LONG spread (z={zscore:.2f})")

        # Spread too high → short A / long B
        elif zscore > self.entry_threshold and self._position_direction == 0:
            signals.append(Signal(
                strategy_name=self.name,
                asset=self.asset_a,
                signal_type=SignalType.SELL,
                strength=min(abs(zscore) / self.entry_threshold - 1, 1.0),
                price=price_a,
                metadata={**metadata, "leg": "short"},
            ))
            signals.append(Signal(
                strategy_name=self.name,
                asset=self.asset_b,
                signal_type=SignalType.BUY,
                strength=min(abs(zscore) / self.entry_threshold - 1, 1.0),
                price=price_b,
                metadata={**metadata, "leg": "long"},
            ))
            self._position_direction = -1
            logger.info(f"{self.name}: Enter SHORT spread (z={zscore:.2f})")

        # Spread reverted → exit
        elif abs(zscore) < self.exit_threshold and self._position_direction != 0:
            signals.append(Signal(
                strategy_name=self.name,
                asset=self.asset_a,
                signal_type=SignalType.CLOSE,
                strength=1.0,
                price=price_a,
                metadata={**metadata, "leg": "close_a"},
            ))
            signals.append(Signal(
                strategy_name=self.name,
                asset=self.asset_b,
                signal_type=SignalType.CLOSE,
                strength=1.0,
                price=price_b,
                metadata={**metadata, "leg": "close_b"},
            ))
            logger.info(f"{self.name}: Exit spread (z={zscore:.2f})")
            self._position_direction = 0

        for sig in signals:
            self.record_signal(sig)

        return signals
