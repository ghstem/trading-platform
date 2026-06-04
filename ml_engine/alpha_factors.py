"""
Alpha Factor Engine
Machine Learning-based signal generation for trading
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from loguru import logger
from datetime import datetime
import pickle
import os

from core.trading_engine import Asset, AssetClass


class AlphaFactor(ABC):
    """Abstract base class for alpha factors"""
    
    def __init__(self, name: str, lookback_period: int = 20):
        self.name = name
        self.lookback_period = lookback_period
    
    @abstractmethod
    def compute(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """Compute factor values from OHLCV data"""
        pass


class MomentumFactor(AlphaFactor):
    """Momentum factor based on price changes"""
    
    def __init__(self, lookback_period: int = 20):
        super().__init__("Momentum", lookback_period)
    
    def compute(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """Calculate momentum factor"""
        if len(ohlcv_data) < self.lookback_period:
            return pd.Series(0.0, index=ohlcv_data.index)
        
        returns = ohlcv_data['close'].pct_change(self.lookback_period)
        return returns


class VolatilityFactor(AlphaFactor):
    """Volatility factor"""
    
    def __init__(self, lookback_period: int = 20):
        super().__init__("Volatility", lookback_period)
    
    def compute(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """Calculate volatility factor"""
        if len(ohlcv_data) < self.lookback_period:
            return pd.Series(0.0, index=ohlcv_data.index)
        
        returns = ohlcv_data['close'].pct_change()
        volatility = returns.rolling(self.lookback_period).std()
        return volatility


class MeanReversionFactor(AlphaFactor):
    """Mean reversion factor based on Bollinger Bands"""
    
    def __init__(self, lookback_period: int = 20, num_std: float = 2.0):
        super().__init__("MeanReversion", lookback_period)
        self.num_std = num_std
    
    def compute(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """Calculate mean reversion factor"""
        if len(ohlcv_data) < self.lookback_period:
            return pd.Series(0.0, index=ohlcv_data.index)
        
        sma = ohlcv_data['close'].rolling(self.lookback_period).mean()
        std = ohlcv_data['close'].rolling(self.lookback_period).std()
        upper_band = sma + (self.num_std * std)
        lower_band = sma - (self.num_std * std)
        
        # Factor: how far from mean in terms of std
        factor = (ohlcv_data['close'] - sma) / (std + 1e-8)
        return factor


class VolumeWeightedPriceFactor(AlphaFactor):
    """Volume-weighted price trend factor"""
    
    def __init__(self, lookback_period: int = 20):
        super().__init__("VolumeWeightedPrice", lookback_period)
    
    def compute(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """Calculate volume-weighted price factor"""
        if len(ohlcv_data) < self.lookback_period:
            return pd.Series(0.0, index=ohlcv_data.index)
        
        vwap = (ohlcv_data['close'] * ohlcv_data['volume']).rolling(self.lookback_period).sum() / \
               ohlcv_data['volume'].rolling(self.lookback_period).sum()
        
        factor = (ohlcv_data['close'] - vwap) / (vwap + 1e-8)
        return factor


class RSIFactor(AlphaFactor):
    """Relative Strength Index factor"""
    
    def __init__(self, lookback_period: int = 14):
        super().__init__("RSI", lookback_period)
    
    def compute(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """Calculate RSI factor"""
        if len(ohlcv_data) < self.lookback_period:
            return pd.Series(0.0, index=ohlcv_data.index)
        
        delta = ohlcv_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.lookback_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.lookback_period).mean()
        
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # Normalize to -1, 1 range
        rsi_normalized = (rsi - 50) / 50
        return rsi_normalized


class FactorCombiner:
    """Combines multiple alpha factors"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.factors: Dict[str, AlphaFactor] = {}
        self.weights = weights or {}
        self.scaler = StandardScaler()
        self.normalized_data = {}
    
    def add_factor(self, factor: AlphaFactor, weight: float = 1.0):
        """Add a factor to the combiner"""
        self.factors[factor.name] = factor
        self.weights[factor.name] = weight
        logger.info(f"Added factor: {factor.name} with weight {weight}")
    
    def compute_all_factors(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Compute all factors"""
        factor_data = {}
        
        for name, factor in self.factors.items():
            try:
                factor_values = factor.compute(ohlcv_data)
                factor_data[name] = factor_values
                logger.info(f"Computed {name} factor")
            except Exception as e:
                logger.error(f"Error computing {name} factor: {str(e)}")
                factor_data[name] = pd.Series(0.0, index=ohlcv_data.index)
        
        return pd.DataFrame(factor_data)
    
    def compute_composite_signal(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """Compute weighted composite signal"""
        factor_df = self.compute_all_factors(ohlcv_data)
        
        # Normalize each factor
        normalized_factors = {}
        for col in factor_df.columns:
            values = factor_df[col].dropna().values
            if len(values) > 1:
                scaler = StandardScaler()
                normalized = pd.Series(
                    scaler.fit_transform(values.reshape(-1, 1)).flatten(),
                    index=factor_df[col].dropna().index
                )
                normalized_factors[col] = normalized
        
        # Combine with weights
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            total_weight = len(self.weights)
        
        composite = pd.Series(0.0, index=ohlcv_data.index)
        for factor_name, weight in self.weights.items():
            if factor_name in normalized_factors:
                composite += (normalized_factors[factor_name] * weight / total_weight)
        
        return composite


class SignalGenerator:
    """Generates trading signals from alpha factors"""
    
    def __init__(self, threshold_buy: float = 0.5, threshold_sell: float = -0.5):
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.signal_history = []
    
    def generate_signals(self, composite_signal: pd.Series) -> pd.Series:
        """Generate buy/sell signals from composite signal"""
        signals = pd.Series(0, index=composite_signal.index)
        
        signals[composite_signal > self.threshold_buy] = 1   # BUY
        signals[composite_signal < self.threshold_sell] = -1 # SELL
        
        return signals
    
    def get_signal_strength(self, composite_signal: pd.Series) -> pd.Series:
        """Get signal strength (-1 to 1)"""
        strength = composite_signal.copy()
        strength = strength.clip(-1, 1)
        return strength


class MLSignalPredictor:
    """Machine learning-based signal predictor"""
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
    
    def _prepare_features(self, ohlcv_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training"""
        
        # Create features from technical indicators
        features = []
        
        # Momentum
        momentum = ohlcv_data['close'].pct_change(10)
        features.append(momentum.values)
        
        # Volatility
        returns = ohlcv_data['close'].pct_change()
        volatility = returns.rolling(20).std()
        features.append(volatility.values)
        
        # RSI
        delta = ohlcv_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.values)
        
        # Moving average ratio
        sma_20 = ohlcv_data['close'].rolling(20).mean()
        sma_50 = ohlcv_data['close'].rolling(50).mean()
        ma_ratio = (sma_20 - sma_50) / (sma_50 + 1e-8)
        features.append(ma_ratio.values)
        
        # Volume change
        volume_change = ohlcv_data['volume'].pct_change()
        features.append(volume_change.values)
        
        X = np.column_stack(features)
        
        # Create labels: 1 if price goes up in next period, 0 otherwise
        future_returns = ohlcv_data['close'].shift(-1).pct_change()
        y = (future_returns > 0).astype(int).values
        
        # Remove NaN values
        valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_idx]
        y = y[valid_idx]
        
        self.feature_names = ['Momentum', 'Volatility', 'RSI', 'MA_Ratio', 'Volume_Change']
        
        return X, y
    
    def train(self, ohlcv_data: pd.DataFrame):
        """Train the ML model"""
        try:
            X, y = self._prepare_features(ohlcv_data)
            
            if len(X) < 100:
                logger.warning("Insufficient data for ML training")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create and train model
            if self.model_type == "random_forest":
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif self.model_type == "gradient_boosting":
                self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            else:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            logger.info(f"ML model trained successfully: {self.model_type}")
            
            # Get feature importance
            importances = self.model.feature_importances_
            for name, importance in zip(self.feature_names, importances):
                logger.info(f"Feature importance - {name}: {importance:.4f}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error training ML model: {str(e)}")
            return False
    
    def predict_signals(self, ohlcv_data: pd.DataFrame) -> pd.Series:
        """Predict buy/sell signals"""
        if not self.is_trained:
            logger.warning("Model not trained yet")
            return pd.Series(0, index=ohlcv_data.index)
        
        try:
            X, _ = self._prepare_features(ohlcv_data)
            X_scaled = self.scaler.transform(X)
            
            # Get probability predictions
            predictions = self.model.predict_proba(X_scaled)[:, 1]
            
            # Convert probabilities to signals (-1 to 1)
            signals = pd.Series((predictions - 0.5) * 2, index=ohlcv_data.index[-len(predictions):])
            
            return signals
        
        except Exception as e:
            logger.error(f"Error predicting signals: {str(e)}")
            return pd.Series(0, index=ohlcv_data.index)
    
    def save_model(self, path: str):
        """Save trained model to disk"""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                }, f)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, path: str):
        """Load trained model from disk"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.scaler = data['scaler']
                self.feature_names = data['feature_names']
                self.is_trained = True
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Create factor combiner
    combiner = FactorCombiner()
    combiner.add_factor(MomentumFactor(20), weight=1.0)
    combiner.add_factor(VolatilityFactor(20), weight=0.5)
    combiner.add_factor(MeanReversionFactor(20), weight=0.8)
    combiner.add_factor(RSIFactor(14), weight=1.0)
    
    # Create signal generator
    signal_gen = SignalGenerator(threshold_buy=0.5, threshold_sell=-0.5)
    
    # Create ML predictor
    ml_predictor = MLSignalPredictor(model_type="random_forest")
    
    print("Alpha Factor Engine initialized successfully")
