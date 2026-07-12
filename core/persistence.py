"""
Persistence Layer — SQLite-backed state snapshots
===================================================
Persists portfolio snapshots, trade history, and signal history to a local
SQLite database so that state survives server restarts.

Usage
-----
    from core.persistence import PersistenceManager

    pm = PersistenceManager("trading_platform.db")
    pm.save_portfolio_snapshot(portfolio)
    pm.save_signal(signal, timestamp)

    # On startup — restore:
    snapshot = pm.load_latest_portfolio_snapshot()
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from core.trading_engine import Portfolio


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class PortfolioSnapshot(Base):
    """A point-in-time snapshot of the portfolio's key metrics."""

    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    total_equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    total_pnl = Column(Float, nullable=False)
    num_positions = Column(Integer, nullable=False)
    # JSON blob of the full positions dict for richer restore
    positions_json = Column(Text, nullable=False, default="{}")


class TradeRecord(Base):
    """A completed order / fill stored for audit purposes."""

    __tablename__ = "trade_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(64), nullable=False, index=True)
    symbol = Column(String(32), nullable=False)
    side = Column(String(8), nullable=False)
    quantity = Column(Float, nullable=False)
    fill_price = Column(Float, nullable=False)
    commission = Column(Float, nullable=False, default=0.0)
    status = Column(String(32), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    filled_at = Column(DateTime, nullable=True)


class SignalRecord(Base):
    """A trading signal emitted by a strategy, for audit and analytics."""

    __tablename__ = "signal_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    strategy_name = Column(String(128), nullable=False)
    symbol = Column(String(32), nullable=False)
    signal_type = Column(String(16), nullable=False)
    strength = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    metadata_json = Column(Text, nullable=False, default="{}")


class EquityCurvePoint(Base):
    """One data-point on the equity curve (recorded periodically)."""

    __tablename__ = "equity_curve"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    total_equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    total_pnl = Column(Float, nullable=False)


# ---------------------------------------------------------------------------
# PersistenceManager
# ---------------------------------------------------------------------------

class PersistenceManager:
    """
    Thin SQLAlchemy wrapper that provides save/load helpers for the platform's
    runtime state.

    Parameters
    ----------
    db_url : str
        SQLAlchemy database URL.  Defaults to a local SQLite file.
        Use ``"sqlite:///:memory:"`` in tests.
    """

    def __init__(self, db_url: str = "sqlite:///trading_platform.db"):
        self._engine = create_engine(db_url, echo=False, future=True)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)
        logger.info(f"PersistenceManager initialised ({db_url})")

    # ------------------------------------------------------------------
    # Context manager helper
    # ------------------------------------------------------------------

    @contextmanager
    def _session(self):
        session: Session = self._Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Portfolio snapshots
    # ------------------------------------------------------------------

    def save_portfolio_snapshot(self, portfolio: Portfolio) -> None:
        """Persist a lightweight snapshot of the portfolio."""
        summary = portfolio.get_portfolio_summary()
        positions: Dict[str, Any] = {}
        for asset, pos in portfolio.positions.items():
            positions[asset.symbol] = {
                "quantity": pos.quantity,
                "avg_entry": pos.average_entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
            }

        row = PortfolioSnapshot(
            timestamp=datetime.utcnow(),
            total_equity=summary["total_equity"],
            cash=summary["cash"],
            positions_value=summary["positions_value"],
            total_pnl=summary["total_pnl"],
            num_positions=summary["num_positions"],
            positions_json=json.dumps(positions),
        )
        with self._session() as s:
            s.add(row)
        logger.debug("Portfolio snapshot saved")

    def load_latest_portfolio_snapshot(self) -> Optional[Dict[str, Any]]:
        """Load the most recent portfolio snapshot, or None if not found."""
        with self._session() as s:
            row: Optional[PortfolioSnapshot] = (
                s.query(PortfolioSnapshot)
                .order_by(PortfolioSnapshot.timestamp.desc())
                .first()
            )
        if row is None:
            return None
        return {
            "timestamp": row.timestamp.isoformat(),
            "total_equity": row.total_equity,
            "cash": row.cash,
            "positions_value": row.positions_value,
            "total_pnl": row.total_pnl,
            "num_positions": row.num_positions,
            "positions": json.loads(row.positions_json),
        }

    def load_portfolio_snapshots(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Load the most recent N portfolio snapshots (newest first)."""
        with self._session() as s:
            rows = (
                s.query(PortfolioSnapshot)
                .order_by(PortfolioSnapshot.timestamp.desc())
                .limit(limit)
                .all()
            )
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "total_equity": r.total_equity,
                "cash": r.cash,
                "total_pnl": r.total_pnl,
                "num_positions": r.num_positions,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Trade records
    # ------------------------------------------------------------------

    def save_trade(self, order) -> None:
        """Persist a filled/cancelled order."""
        row = TradeRecord(
            order_id=order.order_id,
            symbol=order.asset.symbol,
            side=order.side.value,
            quantity=order.quantity,
            fill_price=order.average_fill_price,
            commission=order.commission,
            status=order.status.value,
            created_at=order.created_at,
            filled_at=order.filled_at,
        )
        with self._session() as s:
            s.add(row)

    def load_trade_history(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Load the most recent N trade records (newest first)."""
        with self._session() as s:
            rows = (
                s.query(TradeRecord)
                .order_by(TradeRecord.created_at.desc())
                .limit(limit)
                .all()
            )
        return [
            {
                "order_id": r.order_id,
                "symbol": r.symbol,
                "side": r.side,
                "quantity": r.quantity,
                "fill_price": r.fill_price,
                "commission": r.commission,
                "status": r.status,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "filled_at": r.filled_at.isoformat() if r.filled_at else None,
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Signal records
    # ------------------------------------------------------------------

    def save_signal(self, signal, timestamp: Optional[datetime] = None) -> None:
        """Persist a strategy signal."""
        row = SignalRecord(
            timestamp=timestamp or datetime.utcnow(),
            strategy_name=signal.strategy_name,
            symbol=signal.asset.symbol,
            signal_type=signal.signal_type.value,
            strength=signal.strength,
            price=signal.price,
            metadata_json=json.dumps(signal.metadata),
        )
        with self._session() as s:
            s.add(row)

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    def record_equity(self, portfolio: Portfolio) -> None:
        """Append a data-point to the equity curve."""
        row = EquityCurvePoint(
            timestamp=datetime.utcnow(),
            total_equity=portfolio.total_equity,
            cash=portfolio.cash,
            total_pnl=portfolio.total_pnl,
        )
        with self._session() as s:
            s.add(row)

    def load_equity_curve(self, limit: int = 5000) -> List[Dict[str, Any]]:
        """Load the equity curve (oldest first) for charting."""
        with self._session() as s:
            rows = (
                s.query(EquityCurvePoint)
                .order_by(EquityCurvePoint.timestamp.asc())
                .limit(limit)
                .all()
            )
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "total_equity": r.total_equity,
                "cash": r.cash,
                "total_pnl": r.total_pnl,
            }
            for r in rows
        ]
