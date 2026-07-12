"""
ML-Based Strategy
Uses the existing MLSignalPredictor and FactorCombiner from ml_engine
to generate buy/sell signals from trained alpha factors.
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from core.trading_engine import Asset
from ml_engine.alpha_factors import (
    FactorCombiner,
    MLSignalPredictor,
    MomentumFactor,
    VolatilityFactor,
    MeanReversionFactor,
    RSIFactor,
    VolumeWeightedPriceFactor,
    SignalGenerator,
)
from strategies.base import BaseStrategy, Signal, SignalType


class MLAlphaStrategy(BaseStrategy):
    """
    Machine-learning alpha factor strategy.

    Trains a ``RandomForestClassifier`` (or ``GradientBoostingClassifier``) on
    historical OHLCV data and generates signals based on the predicted
    probability that next-bar price will be higher.

    Composite alpha factors (momentum, volatility, mean-reversion, RSI, VWAP)
    are also used to confirm the ML signal.

    Parameters
    ----------
    model_type          : "random_forest" or "gradient_boosting"
    train_window        : number of bars to use for (re-)training
    retrain_interval    : retrain every N bars (0 = train once)
    buy_threshold       : ML signal probability above which we go long
    sell_threshold      : ML signal probability below which we go short/close
    min_factor_signal   : minimum composite factor value to confirm a BUY
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        train_window: int = 200,
        retrain_interval: int = 50,
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.4,
        min_factor_signal: float = 0.2,
        assets: Optional[List[Asset]] = None,
    ):
        super().__init__(f"ML_Alpha_{model_type}", assets)
        self.model_type = model_type
        self.train_window = train_window
        self.retrain_interval = retrain_interval
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.min_factor_signal = min_factor_signal

        self._price_history: Dict[str, List[float]] = {}
        self._volume_history: Dict[str, List[float]] = {}
        self._bar_count: Dict[str, int] = {}
        self._in_position: Dict[str, bool] = {}
        self._ml_predictors: Dict[str, MLSignalPredictor] = {}
        self._combiners: Dict[str, FactorCombiner] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_predictor(self, symbol: str) -> MLSignalPredictor:
        if symbol not in self._ml_predictors:
            self._ml_predictors[symbol] = MLSignalPredictor(model_type=self.model_type)
        return self._ml_predictors[symbol]

    def _get_or_create_combiner(self, symbol: str) -> FactorCombiner:
        if symbol not in self._combiners:
            combiner = FactorCombiner()
            combiner.add_factor(MomentumFactor(20), weight=1.0)
            combiner.add_factor(VolatilityFactor(20), weight=0.5)
            combiner.add_factor(MeanReversionFactor(20), weight=0.8)
            combiner.add_factor(RSIFactor(14), weight=1.0)
            combiner.add_factor(VolumeWeightedPriceFactor(20), weight=0.7)
            self._combiners[symbol] = combiner
        return self._combiners[symbol]

    def _build_ohlcv(self, symbol: str) -> pd.DataFrame:
        prices = self._price_history.get(symbol, [])
        volumes = self._volume_history.get(symbol, [prices[-1]] * len(prices))  # fallback
        n = min(len(prices), len(volumes))
        return pd.DataFrame({
            "open": prices[-n:],
            "high": prices[-n:],
            "low": prices[-n:],
            "close": prices[-n:],
            "volume": volumes[-n:],
        })

    # ------------------------------------------------------------------

    def generate_signals(self, ohlcv_data: pd.DataFrame, asset: Asset) -> List[Signal]:
        """Single-asset signal generation from supplied OHLCV DataFrame."""
        signals: List[Signal] = []
        if len(ohlcv_data) < self.train_window:
            return signals

        symbol = asset.symbol
        predictor = self._get_or_create_predictor(symbol)
        combiner = self._get_or_create_combiner(symbol)

        # Train if not yet trained
        if not predictor.is_trained:
            logger.info(f"{self.name}: Training ML model for {symbol}...")
            success = predictor.train(ohlcv_data)
            if not success:
                return signals

        # ML signal
        ml_signals = predictor.predict_signals(ohlcv_data)
        if ml_signals.empty:
            return signals
        latest_ml = float(ml_signals.iloc[-1])     # -1 to 1

        # Factor composite signal (confirmation)
        composite = combiner.compute_composite_signal(ohlcv_data)
        latest_factor = float(composite.iloc[-1]) if not composite.empty else 0.0

        current_price = float(ohlcv_data["close"].iloc[-1])
        in_pos = self._in_position.get(symbol, False)

        # BUY: ML predicts up AND factor confirms
        if latest_ml > (self.buy_threshold * 2 - 1) and latest_factor > self.min_factor_signal and not in_pos:
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.BUY,
                strength=min((latest_ml + 1) / 2, 1.0),
                price=current_price,
                metadata={
                    "ml_signal": round(latest_ml, 4),
                    "factor_signal": round(latest_factor, 4),
                },
            ))

        # SELL / CLOSE: ML predicts down
        elif latest_ml < (self.sell_threshold * 2 - 1) and in_pos:
            signals.append(Signal(
                strategy_name=self.name,
                asset=asset,
                signal_type=SignalType.CLOSE,
                strength=1.0,
                price=current_price,
                metadata={
                    "ml_signal": round(latest_ml, 4),
                    "factor_signal": round(latest_factor, 4),
                },
            ))

        return signals

    def on_bar(self, current_date: datetime, current_prices: Dict[Asset, float]) -> List[Signal]:
        all_signals: List[Signal] = []

        for asset in self.assets:
            price = current_prices.get(asset)
            if price is None:
                continue

            symbol = asset.symbol
            self._price_history.setdefault(symbol, []).append(price)
            self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1

            n = len(self._price_history[symbol])
            if n < self.train_window:
                continue

            predictor = self._get_or_create_predictor(symbol)

            # Periodic retraining
            if self.retrain_interval > 0 and self._bar_count[symbol] % self.retrain_interval == 0:
                logger.info(f"{self.name}: Re-training model for {symbol} at bar {self._bar_count[symbol]}")
                ohlcv = self._build_ohlcv(symbol)
                predictor.train(ohlcv)

            ohlcv = self._build_ohlcv(symbol)
            sigs = self.generate_signals(ohlcv, asset)
            for sig in sigs:
                if sig.signal_type == SignalType.BUY:
                    self._in_position[symbol] = True
                elif sig.signal_type in (SignalType.SELL, SignalType.CLOSE):
                    self._in_position[symbol] = False
                self.record_signal(sig)
            all_signals.extend(sigs)

        return all_signals
