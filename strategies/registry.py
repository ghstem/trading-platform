"""
Strategy Registry
Central catalogue of all available strategy types.
Supports instantiation by name and listing available strategies with metadata.
"""

from typing import Any, Dict, List, Optional, Type

from loguru import logger

from core.trading_engine import Asset
from strategies.base import BaseStrategy
from strategies.trend_following import SMACrossoverStrategy, EMACrossoverStrategy, MACDStrategy
from strategies.mean_reversion import BollingerBandsStrategy, RSIReversionStrategy, ZScoreReversionStrategy
from strategies.momentum import PriceMomentumStrategy, BreakoutStrategy
from strategies.pairs_trading import PairsTradingStrategy
from strategies.ml_strategy import MLAlphaStrategy


# ---------------------------------------------------------------------------
# Registry definition
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Trend Following ──────────────────────────────────────────────────
    "sma_crossover": {
        "cls": SMACrossoverStrategy,
        "category": "Trend Following",
        "description": "Golden/death cross using simple moving averages (fast vs slow).",
        "default_params": {"fast_period": 20, "slow_period": 50},
        "param_schema": {
            "fast_period": {"type": "int", "min": 2, "max": 200, "description": "Fast SMA period"},
            "slow_period": {"type": "int", "min": 5, "max": 500, "description": "Slow SMA period"},
        },
    },
    "ema_crossover": {
        "cls": EMACrossoverStrategy,
        "category": "Trend Following",
        "description": "Crossover using exponential moving averages for faster signal response.",
        "default_params": {"fast_period": 12, "slow_period": 26},
        "param_schema": {
            "fast_period": {"type": "int", "min": 2, "max": 200},
            "slow_period": {"type": "int", "min": 5, "max": 500},
        },
    },
    "macd": {
        "cls": MACDStrategy,
        "category": "Trend Following",
        "description": "MACD line / signal line crossover strategy.",
        "default_params": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "param_schema": {
            "fast_period": {"type": "int", "min": 2, "max": 200},
            "slow_period": {"type": "int", "min": 5, "max": 500},
            "signal_period": {"type": "int", "min": 2, "max": 50},
        },
    },
    # ── Mean Reversion ───────────────────────────────────────────────────
    "bollinger_bands": {
        "cls": BollingerBandsStrategy,
        "category": "Mean Reversion",
        "description": "Buy on lower-band touch, exit on upper-band touch.",
        "default_params": {"period": 20, "num_std": 2.0},
        "param_schema": {
            "period": {"type": "int", "min": 5, "max": 200},
            "num_std": {"type": "float", "min": 0.5, "max": 5.0},
        },
    },
    "rsi_reversion": {
        "cls": RSIReversionStrategy,
        "category": "Mean Reversion",
        "description": "Buy when RSI exits oversold; sell when RSI exits overbought.",
        "default_params": {"period": 14, "oversold": 30.0, "overbought": 70.0},
        "param_schema": {
            "period": {"type": "int", "min": 2, "max": 50},
            "oversold": {"type": "float", "min": 5.0, "max": 45.0},
            "overbought": {"type": "float", "min": 55.0, "max": 95.0},
        },
    },
    "zscore_reversion": {
        "cls": ZScoreReversionStrategy,
        "category": "Mean Reversion",
        "description": "Enter when price z-score is extreme; exit when it reverts to mean.",
        "default_params": {"lookback": 20, "entry_threshold": 2.0, "exit_threshold": 0.5},
        "param_schema": {
            "lookback": {"type": "int", "min": 5, "max": 500},
            "entry_threshold": {"type": "float", "min": 0.5, "max": 5.0},
            "exit_threshold": {"type": "float", "min": 0.0, "max": 2.0},
        },
    },
    # ── Momentum ─────────────────────────────────────────────────────────
    "price_momentum": {
        "cls": PriceMomentumStrategy,
        "category": "Momentum",
        "description": "Buy assets with positive N-period return; sell laggards.",
        "default_params": {"lookback": 20, "top_n": 1},
        "param_schema": {
            "lookback": {"type": "int", "min": 5, "max": 252},
            "top_n": {"type": "int", "min": 1, "max": 10},
        },
    },
    "breakout": {
        "cls": BreakoutStrategy,
        "category": "Momentum",
        "description": "Enter long on N-period high breakout; exit on N-period low break.",
        "default_params": {"period": 20, "atr_multiplier": 1.5},
        "param_schema": {
            "period": {"type": "int", "min": 5, "max": 252},
            "atr_multiplier": {"type": "float", "min": 0.5, "max": 5.0},
        },
    },
    # ── Statistical Arbitrage ────────────────────────────────────────────
    "pairs_trading": {
        "cls": PairsTradingStrategy,
        "category": "Statistical Arbitrage",
        "description": "Pairs trade: long/short spread between two correlated assets.",
        "default_params": {"lookback": 60, "entry_threshold": 2.0, "exit_threshold": 0.5},
        "param_schema": {
            "lookback": {"type": "int", "min": 20, "max": 500},
            "entry_threshold": {"type": "float", "min": 0.5, "max": 5.0},
            "exit_threshold": {"type": "float", "min": 0.0, "max": 2.0},
            "hedge_ratio": {"type": "float", "min": 0.01, "max": 10.0, "optional": True},
        },
    },
    # ── Machine Learning ─────────────────────────────────────────────────
    "ml_alpha": {
        "cls": MLAlphaStrategy,
        "category": "Machine Learning",
        "description": "Random-forest / gradient-boosting model on technical alpha factors.",
        "default_params": {
            "model_type": "random_forest",
            "train_window": 200,
            "retrain_interval": 50,
            "buy_threshold": 0.6,
            "sell_threshold": 0.4,
        },
        "param_schema": {
            "model_type": {"type": "str", "choices": ["random_forest", "gradient_boosting"]},
            "train_window": {"type": "int", "min": 100, "max": 2000},
            "retrain_interval": {"type": "int", "min": 0, "max": 500},
            "buy_threshold": {"type": "float", "min": 0.5, "max": 1.0},
            "sell_threshold": {"type": "float", "min": 0.0, "max": 0.5},
        },
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class StrategyRegistry:
    """
    Global registry for trading strategies.

    Usage
    -----
    registry = StrategyRegistry()

    # List all strategies
    registry.list_strategies()

    # Build an instance with custom params
    strategy = registry.create("macd", fast_period=8, slow_period=21, assets=[aapl])
    """

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, Any]] = dict(_REGISTRY)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        key: str,
        cls: Type[BaseStrategy],
        category: str = "Custom",
        description: str = "",
        default_params: Optional[Dict] = None,
        param_schema: Optional[Dict] = None,
    ) -> None:
        """Register a custom strategy class."""
        if key in self._entries:
            logger.warning(f"Strategy '{key}' already registered; overwriting.")
        self._entries[key] = {
            "cls": cls,
            "category": category,
            "description": description,
            "default_params": default_params or {},
            "param_schema": param_schema or {},
        }
        logger.info(f"Registered strategy: {key} ({category})")

    # ------------------------------------------------------------------
    # Lookup / instantiation
    # ------------------------------------------------------------------

    def get_class(self, key: str) -> Type[BaseStrategy]:
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(f"Unknown strategy: '{key}'. Available: {self.list_keys()}")
        return entry["cls"]

    def create(
        self,
        key: str,
        assets: Optional[List[Asset]] = None,
        **params: Any,
    ) -> BaseStrategy:
        """
        Instantiate a strategy by registry key.

        Merges ``default_params`` with any caller-supplied ``params``.
        """
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(f"Unknown strategy: '{key}'")

        kwargs = {**entry["default_params"], **params}
        if assets is not None:
            kwargs["assets"] = assets

        strategy = entry["cls"](**kwargs)
        logger.info(f"Created strategy '{key}': {strategy.name}")
        return strategy

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_keys(self) -> List[str]:
        return sorted(self._entries.keys())

    def list_strategies(self) -> List[Dict[str, Any]]:
        """Return a list of strategy metadata dicts (safe for JSON serialisation)."""
        result = []
        for key, entry in sorted(self._entries.items()):
            result.append({
                "key": key,
                "name": entry["cls"].__name__,
                "category": entry["category"],
                "description": entry["description"],
                "default_params": entry["default_params"],
                "param_schema": entry["param_schema"],
            })
        return result

    def list_by_category(self) -> Dict[str, List[str]]:
        """Return {category: [key, ...]} mapping."""
        cats: Dict[str, List[str]] = {}
        for key, entry in self._entries.items():
            cats.setdefault(entry["category"], []).append(key)
        return cats

    def describe(self, key: str) -> Dict[str, Any]:
        """Return full metadata for a single strategy."""
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(f"Unknown strategy: '{key}'")
        return {
            "key": key,
            "name": entry["cls"].__name__,
            "category": entry["category"],
            "description": entry["description"],
            "default_params": entry["default_params"],
            "param_schema": entry["param_schema"],
        }


# Singleton instance
_registry: Optional[StrategyRegistry] = None


def get_registry() -> StrategyRegistry:
    """Return the global StrategyRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
    return _registry
