"""
Risk Management & Execution Override System
Automated risk controls and manual intervention layer
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger
from abc import ABC, abstractmethod

from core.trading_engine import (
    Asset, Order, OrderSide, OrderType, Position, Portfolio
)


class OverrideType(Enum):
    """Types of execution overrides"""
    EMERGENCY_STOP = "EMERGENCY_STOP"
    MANUAL_CLOSE = "MANUAL_CLOSE"
    POSITION_LIMIT_OVERRIDE = "POSITION_LIMIT_OVERRIDE"
    DAILY_LOSS_OVERRIDE = "DAILY_LOSS_OVERRIDE"
    LEVERAGE_OVERRIDE = "LEVERAGE_OVERRIDE"


class OverrideStatus(Enum):
    """Status of override requests"""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"


class RiskLevel(Enum):
    """Risk level classification"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class RiskLimits:
    """Risk management limits"""
    max_position_size_pct: float = 0.1      # Max position as % of portfolio
    max_daily_loss_pct: float = -0.05       # -5% daily loss limit
    max_drawdown_pct: float = -0.20         # -20% max drawdown
    max_leverage: float = 1.0               # No leverage by default
    max_correlation_threshold: float = 0.8 # Max correlation between positions
    max_positions: int = 20                 # Max number of open positions
    min_cash_buffer_pct: float = 0.05       # Keep 5% cash minimum
    risk_per_trade_pct: float = 0.02        # Risk 2% per trade
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'max_position_size_pct': self.max_position_size_pct,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'max_drawdown_pct': self.max_drawdown_pct,
            'max_leverage': self.max_leverage,
            'max_correlation_threshold': self.max_correlation_threshold,
            'max_positions': self.max_positions,
            'min_cash_buffer_pct': self.min_cash_buffer_pct,
            'risk_per_trade_pct': self.risk_per_trade_pct,
        }


@dataclass
class RiskMetrics:
    """Current risk metrics for the portfolio"""
    current_drawdown_pct: float = 0.0
    current_daily_loss_pct: float = 0.0
    cash_utilization_pct: float = 0.0
    portfolio_correlation: float = 0.0
    num_open_positions: int = 0
    largest_position_pct: float = 0.0
    total_leverage: float = 1.0
    risk_level: RiskLevel = RiskLevel.LOW
    estimated_position_size: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'current_drawdown_pct': f"{self.current_drawdown_pct * 100:.2f}%",
            'current_daily_loss_pct': f"{self.current_daily_loss_pct * 100:.2f}%",
            'cash_utilization_pct': f"{self.cash_utilization_pct * 100:.2f}%",
            'portfolio_correlation': f"{self.portfolio_correlation:.3f}",
            'num_open_positions': self.num_open_positions,
            'largest_position_pct': f"{self.largest_position_pct * 100:.2f}%",
            'total_leverage': f"{self.total_leverage:.2f}x",
            'risk_level': self.risk_level.value,
            'estimated_position_size': f"${self.estimated_position_size:,.2f}",
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class ExecutionOverride:
    """Represents a manual execution override"""
    override_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    override_type: OverrideType = None
    status: OverrideStatus = OverrideStatus.PENDING
    requester: str = ""
    reason: str = ""
    affected_assets: List[Asset] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    approved_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    approver: str = ""
    execution_result: str = ""
    
    def approve(self, approver: str):
        """Approve the override"""
        self.status = OverrideStatus.APPROVED
        self.approver = approver
        self.approved_at = datetime.now()
        logger.info(f"Override {self.override_id} approved by {approver}")
    
    def deny(self, approver: str, reason: str = ""):
        """Deny the override"""
        self.status = OverrideStatus.DENIED
        self.approver = approver
        self.execution_result = reason
        logger.warning(f"Override {self.override_id} denied by {approver}")
    
    def execute(self):
        """Mark override as executed"""
        self.status = OverrideStatus.EXECUTED
        self.executed_at = datetime.now()
        logger.info(f"Override {self.override_id} executed")


@dataclass
class DailyRiskTracker:
    """Tracks daily trading statistics for risk management"""
    date: datetime = field(default_factory=lambda: datetime.now().date())
    starting_equity: float = 0.0
    current_equity: float = 0.0
    trades_executed: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_intraday_drawdown: float = 0.0
    daily_pnl: float = 0.0
    
    @property
    def daily_return_pct(self) -> float:
        """Daily return as percentage"""
        if self.starting_equity == 0:
            return 0.0
        return (self.current_equity - self.starting_equity) / self.starting_equity
    
    @property
    def win_rate(self) -> float:
        """Win rate percentage"""
        total_trades = self.winning_trades + self.losing_trades
        if total_trades == 0:
            return 0.0
        return self.winning_trades / total_trades
    
    def record_trade_result(self, pnl: float, is_win: bool):
        """Record a trade result"""
        self.trades_executed += 1
        if is_win:
            self.winning_trades += 1
            self.largest_win = max(self.largest_win, pnl)
        else:
            self.losing_trades += 1
            self.largest_loss = min(self.largest_loss, pnl)
        self.daily_pnl += pnl


class RiskManager:
    """Central risk management system"""
    
    def __init__(self, portfolio: Portfolio, limits: Optional[RiskLimits] = None):
        self.portfolio = portfolio
        self.limits = limits or RiskLimits()
        self.risk_metrics = RiskMetrics()
        self.daily_tracker = DailyRiskTracker(starting_equity=portfolio.total_equity)
        self.execution_overrides: List[ExecutionOverride] = []
        self.price_history: Dict[Asset, List[float]] = {}
        self.equity_history: List[Tuple[datetime, float]] = []
        self.max_equity = portfolio.total_equity
        self.is_trading_halted = False
        
        logger.info("Risk Manager initialized")
    
    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed"""
        if self.is_trading_halted:
            return False, "Trading halted by risk manager"
        
        # Check daily loss limit
        if self.risk_metrics.current_daily_loss_pct <= self.limits.max_daily_loss_pct:
            return False, f"Daily loss limit exceeded: {self.risk_metrics.current_daily_loss_pct * 100:.2f}%"
        
        # Check drawdown limit
        if self.risk_metrics.current_drawdown_pct <= self.limits.max_drawdown_pct:
            return False, f"Max drawdown exceeded: {self.risk_metrics.current_drawdown_pct * 100:.2f}%"
        
        # Check number of open positions
        if self.risk_metrics.num_open_positions >= self.limits.max_positions:
            return False, f"Max positions limit reached: {self.risk_metrics.num_open_positions}"
        
        return True, "Trading allowed"
    
    def validate_order(self, order: Order) -> Tuple[bool, str]:
        """Validate an order against risk parameters"""
        
        # Check if trading is allowed
        can_trade, reason = self.can_trade()
        if not can_trade:
            return False, reason
        
        # Calculate position size
        estimated_cost = order.quantity * order.price
        position_size_pct = estimated_cost / self.portfolio.total_equity
        
        # Check position size limit
        if position_size_pct > self.limits.max_position_size_pct:
            return False, f"Position size {position_size_pct * 100:.2f}% exceeds limit {self.limits.max_position_size_pct * 100:.2f}%"
        
        # Check cash buffer
        post_trade_cash = self.portfolio.cash - estimated_cost
        min_cash = self.portfolio.total_equity * self.limits.min_cash_buffer_pct
        
        if post_trade_cash < min_cash:
            return False, f"Insufficient cash buffer. Required: ${min_cash:,.2f}, Available: ${post_trade_cash:,.2f}"
        
        # Check leverage
        if order.side == OrderSide.BUY:
            post_trade_margin = self.portfolio.margin_used + estimated_cost
            total_leverage = post_trade_margin / self.portfolio.total_equity
            
            if total_leverage > self.limits.max_leverage:
                return False, f"Leverage {total_leverage:.2f}x exceeds limit {self.limits.max_leverage:.2f}x"
        
        return True, "Order validated"
    
    def calculate_position_size(self, asset: Asset, entry_price: float, stop_loss_price: float) -> float:
        """Calculate optimal position size based on risk per trade"""
        if entry_price <= stop_loss_price:
            logger.warning("Entry price must be above stop loss")
            return 0.0
        
        # Calculate risk per trade in dollars
        risk_amount = self.portfolio.total_equity * self.limits.risk_per_trade_pct
        
        # Calculate points at risk
        points_at_risk = entry_price - stop_loss_price
        
        # Calculate optimal quantity
        position_size = risk_amount / points_at_risk
        
        # Apply position size limit
        max_position_value = self.portfolio.total_equity * self.limits.max_position_size_pct
        max_quantity = max_position_value / entry_price
        
        position_size = min(position_size, max_quantity)
        
        return position_size
    
    def update_metrics(self, current_prices: Dict[Asset, float] = None):
        """Update current risk metrics"""
        self.risk_metrics.timestamp = datetime.now()
        
        # Update equity
        self.portfolio.update_prices(current_prices or {})
        self.daily_tracker.current_equity = self.portfolio.total_equity
        
        # Track equity history
        self.equity_history.append((datetime.now(), self.portfolio.total_equity))
        
        # Update max equity for drawdown calculation
        self.max_equity = max(self.max_equity, self.portfolio.total_equity)
        
        # Calculate drawdown
        current_drawdown = (self.portfolio.total_equity - self.max_equity) / self.max_equity
        self.risk_metrics.current_drawdown_pct = current_drawdown
        
        # Calculate daily loss
        daily_loss = (self.portfolio.total_equity - self.daily_tracker.starting_equity) / self.daily_tracker.starting_equity
        self.risk_metrics.current_daily_loss_pct = daily_loss
        
        # Update cash utilization
        self.risk_metrics.cash_utilization_pct = (self.portfolio.margin_used / self.portfolio.total_equity) if self.portfolio.total_equity > 0 else 0
        
        # Update position count
        self.risk_metrics.num_open_positions = len(self.portfolio.positions)
        
        # Calculate largest position percentage
        if self.portfolio.positions:
            position_values = [pos.market_value for pos in self.portfolio.positions.values()]
            self.risk_metrics.largest_position_pct = max(position_values) / self.portfolio.total_equity if position_values else 0
        
        # Update portfolio correlation
        self._update_portfolio_correlation()
        
        # Determine risk level
        self._determine_risk_level()
        
        # Calculate estimated position size
        self.risk_metrics.estimated_position_size = self.portfolio.total_equity * self.limits.risk_per_trade_pct
        
        # Check if trading should be halted
        self._check_halt_conditions()
    
    def _update_portfolio_correlation(self):
        """Calculate average correlation between positions"""
        if len(self.portfolio.positions) < 2:
            self.risk_metrics.portfolio_correlation = 0.0
            return
        
        assets = list(self.portfolio.positions.keys())
        if len(assets) < 2:
            self.risk_metrics.portfolio_correlation = 0.0
            return
        
        # Calculate correlations (simplified - would need more data in production)
        correlations = []
        for i, asset1 in enumerate(assets):
            for asset2 in assets[i+1:]:
                if asset1 in self.price_history and asset2 in self.price_history:
                    hist1 = self.price_history[asset1]
                    hist2 = self.price_history[asset2]
                    
                    if len(hist1) > 1 and len(hist2) > 1:
                        corr = np.corrcoef(hist1[-20:], hist2[-20:])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
        
        if correlations:
            self.risk_metrics.portfolio_correlation = np.mean(correlations)
    
    def _determine_risk_level(self):
        """Determine overall portfolio risk level"""
        if self.risk_metrics.current_drawdown_pct < -0.15:
            self.risk_metrics.risk_level = RiskLevel.CRITICAL
        elif self.risk_metrics.current_drawdown_pct < -0.10:
            self.risk_metrics.risk_level = RiskLevel.HIGH
        elif self.risk_metrics.cash_utilization_pct > 0.80:
            self.risk_metrics.risk_level = RiskLevel.HIGH
        elif self.risk_metrics.cash_utilization_pct > 0.50:
            self.risk_metrics.risk_level = RiskLevel.MEDIUM
        else:
            self.risk_metrics.risk_level = RiskLevel.LOW
    
    def _check_halt_conditions(self):
        """Check if trading should be halted"""
        if self.risk_metrics.current_drawdown_pct <= self.limits.max_drawdown_pct:
            logger.critical(f"Trading halted: Max drawdown exceeded {self.risk_metrics.current_drawdown_pct * 100:.2f}%")
            self.is_trading_halted = True
        
        if self.risk_metrics.current_daily_loss_pct <= self.limits.max_daily_loss_pct:
            logger.critical(f"Trading halted: Daily loss limit exceeded {self.risk_metrics.current_daily_loss_pct * 100:.2f}%")
            self.is_trading_halted = True
    
    def get_risk_summary(self) -> Dict:
        """Get risk summary"""
        return {
            'Risk Level': self.risk_metrics.risk_level.value,
            'Current Drawdown': f"{self.risk_metrics.current_drawdown_pct * 100:.2f}%",
            'Daily Loss': f"{self.risk_metrics.current_daily_loss_pct * 100:.2f}%",
            'Cash Utilization': f"{self.risk_metrics.cash_utilization_pct * 100:.2f}%",
            'Open Positions': self.risk_metrics.num_open_positions,
            'Largest Position': f"{self.risk_metrics.largest_position_pct * 100:.2f}%",
            'Trading Halted': self.is_trading_halted,
        }


class ExecutionOverrideManager:
    """Manages execution overrides with approval workflow"""
    
    def __init__(self, portfolio: Portfolio, risk_manager: RiskManager, 
                 require_approval: bool = True):
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.require_approval = require_approval
        self.overrides: List[ExecutionOverride] = []
        self.approved_users: List[str] = ["admin", "supervisor"]
        
        logger.info("Execution Override Manager initialized")
    
    def request_override(self, override_type: OverrideType, requester: str,
                        reason: str, affected_assets: List[Asset] = None) -> ExecutionOverride:
        """Request an execution override"""
        
        override = ExecutionOverride(
            override_type=override_type,
            requester=requester,
            reason=reason,
            affected_assets=affected_assets or []
        )
        
        self.overrides.append(override)
        logger.info(f"Override requested: {override.override_id} ({override_type.value}) by {requester}")
        
        if not self.require_approval:
            self.approve_override(override.override_id, "auto")
        
        return override
    
    def approve_override(self, override_id: str, approver: str) -> bool:
        """Approve an override"""
        override = self._find_override(override_id)
        if override is None:
            logger.error(f"Override {override_id} not found")
            return False
        
        if approver not in self.approved_users:
            logger.warning(f"User {approver} not authorized to approve overrides")
            return False
        
        override.approve(approver)
        return True
    
    def deny_override(self, override_id: str, approver: str, reason: str = "") -> bool:
        """Deny an override"""
        override = self._find_override(override_id)
        if override is None:
            logger.error(f"Override {override_id} not found")
            return False
        
        override.deny(approver, reason)
        return True
    
    def execute_override(self, override_id: str) -> bool:
        """Execute an approved override"""
        override = self._find_override(override_id)
        if override is None:
            logger.error(f"Override {override_id} not found")
            return False
        
        if override.status != OverrideStatus.APPROVED:
            logger.warning(f"Override {override_id} is not approved")
            return False
        
        try:
            if override.override_type == OverrideType.EMERGENCY_STOP:
                self._execute_emergency_stop(override)
            
            elif override.override_type == OverrideType.MANUAL_CLOSE:
                self._execute_manual_close(override)
            
            elif override.override_type == OverrideType.POSITION_LIMIT_OVERRIDE:
                self._execute_position_limit_override(override)
            
            elif override.override_type == OverrideType.DAILY_LOSS_OVERRIDE:
                self._execute_daily_loss_override(override)
            
            override.execute()
            override.execution_result = "Executed successfully"
            logger.info(f"Override {override_id} executed successfully")
            return True
        
        except Exception as e:
            override.execution_result = f"Error: {str(e)}"
            logger.error(f"Error executing override {override_id}: {str(e)}")
            return False
    
    def _execute_emergency_stop(self, override: ExecutionOverride):
        """Execute emergency stop - close all positions"""
        logger.critical("EMERGENCY STOP: Closing all positions")
        
        for asset in list(self.portfolio.positions.keys()):
            position = self.portfolio.positions[asset]
            if position.quantity > 0:
                # Create sell order
                sell_order = self.portfolio.create_order(
                    asset=asset,
                    side=OrderSide.SELL,
                    quantity=position.quantity,
                    order_type=OrderType.MARKET,
                    price=position.current_price
                )
                # Execute immediately
                self.portfolio.execute_order(
                    sell_order,
                    position.current_price,
                    position.quantity
                )
        
        self.risk_manager.is_trading_halted = True
    
    def _execute_manual_close(self, override: ExecutionOverride):
        """Execute manual close for specific positions"""
        logger.warning(f"Manual close requested for {len(override.affected_assets)} assets")
        
        for asset in override.affected_assets:
            position = self.portfolio.get_position(asset)
            if position and position.quantity > 0:
                self.portfolio.close_position(asset, position.current_price)
    
    def _execute_position_limit_override(self, override: ExecutionOverride):
        """Override position limit restrictions"""
        logger.warning("Position limit override executed")
        self.risk_manager.limits.max_position_size_pct *= 1.5
    
    def _execute_daily_loss_override(self, override: ExecutionOverride):
        """Override daily loss limit"""
        logger.warning("Daily loss limit override executed")
        self.risk_manager.limits.max_daily_loss_pct *= 1.5
    
    def _find_override(self, override_id: str) -> Optional[ExecutionOverride]:
        """Find override by ID"""
        for override in self.overrides:
            if override.override_id == override_id:
                return override
        return None
    
    def get_pending_overrides(self) -> List[ExecutionOverride]:
        """Get all pending overrides"""
        return [o for o in self.overrides if o.status == OverrideStatus.PENDING]
    
    def get_override_history(self) -> List[Dict]:
        """Get override history"""
        history = []
        for override in self.overrides:
            history.append({
                'id': override.override_id,
                'type': override.override_type.value,
                'status': override.status.value,
                'requester': override.requester,
                'reason': override.reason,
                'created_at': override.created_at.isoformat(),
                'executed_at': override.executed_at.isoformat() if override.executed_at else None,
                'result': override.execution_result,
            })
        return history


# Example usage
if __name__ == "__main__":
    from core.trading_engine import Portfolio
    
    # Create portfolio
    portfolio = Portfolio(initial_capital=100000.0)
    
    # Create risk manager
    limits = RiskLimits(
        max_position_size_pct=0.1,
        max_daily_loss_pct=-0.05,
        max_drawdown_pct=-0.20,
    )
    risk_manager = RiskManager(portfolio, limits)
    
    # Create override manager
    override_manager = ExecutionOverrideManager(portfolio, risk_manager, require_approval=True)
    
    print("Risk Management System initialized")
    print(risk_manager.get_risk_summary())
