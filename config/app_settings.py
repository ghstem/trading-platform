"""
Application Settings — Pydantic BaseSettings
=============================================
All runtime configuration is declared here as a typed, validated model.
Values are loaded (in order of precedence):
  1. Environment variables
  2. .env file (loaded by python-dotenv automatically)
  3. Declared default values

Usage
-----
    from config.app_settings import get_settings
    settings = get_settings()
    print(settings.INITIAL_CAPITAL)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """
    Validated application settings.

    Pydantic raises ``ValidationError`` at import time when a required
    value is missing or has the wrong type, giving clear error messages
    before the server starts.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Environment ──────────────────────────────────────────────────
    ENVIRONMENT: str = Field(default="development", description="development | production | testing")
    DEBUG: bool = Field(default=True)

    # ── API / Server ─────────────────────────────────────────────────
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=5000, ge=1, le=65535)
    API_KEY: Optional[str] = Field(
        default=None,
        description=(
            "When set, all API requests must include the header "
            "'X-API-Key: <value>'.  Leave unset to disable auth (dev only)."
        ),
    )

    # ── CORS ─────────────────────────────────────────────────────────
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins. Use ['*'] only in development.",
    )

    # ── Security ─────────────────────────────────────────────────────
    SECRET_KEY: str = Field(
        default="dev-secret-key-CHANGE-IN-PRODUCTION",
        description="Used for signing tokens; must be changed in production.",
    )
    JWT_SECRET_KEY: str = Field(default="jwt-secret-key-CHANGE-IN-PRODUCTION")
    JWT_EXPIRATION_HOURS: int = Field(default=24, ge=1)

    # ── Database ─────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="sqlite:///trading_platform.db",
        description="SQLAlchemy database URL. Defaults to local SQLite.",
    )

    # ── Broker APIs ──────────────────────────────────────────────────
    ALPACA_API_KEY: Optional[str] = Field(default=None)
    ALPACA_SECRET_KEY: Optional[str] = Field(default=None)
    ALPACA_BASE_URL: str = Field(default="https://paper-api.alpaca.markets")

    IB_HOST: str = Field(default="127.0.0.1")
    IB_PORT: int = Field(default=7497)

    # ── Data Providers ───────────────────────────────────────────────
    YAHOO_FINANCE_ENABLED: bool = Field(default=True)
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(default=None)

    # ── Trading Parameters ───────────────────────────────────────────
    INITIAL_CAPITAL: float = Field(default=100_000.0, gt=0)
    MAX_POSITION_SIZE: float = Field(default=0.1, gt=0, le=1.0)
    MAX_DAILY_LOSS: float = Field(default=-1_000.0, le=0)

    # ── Market Data ──────────────────────────────────────────────────
    MARKET_DATA_UPDATE_INTERVAL: int = Field(default=60, ge=1, description="Seconds")
    BAR_AUTO_FEED_INTERVAL: int = Field(
        default=0,
        ge=0,
        description="Seconds between automatic bar feeds. 0 = disabled.",
    )

    # ── Logging ──────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: str = Field(default="logs/trading_platform.log")

    # ── Observability ────────────────────────────────────────────────
    METRICS_ENABLED: bool = Field(default=True)

    # ── Backtesting ──────────────────────────────────────────────────
    BACKTEST_START_DATE: str = Field(default="2023-01-01")
    BACKTEST_END_DATE: str = Field(default="2024-01-01")

    # ──────────────────────────────────────────────────────────────────
    # Validators
    # ──────────────────────────────────────────────────────────────────

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "production", "testing"}
        if v.lower() not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}, got '{v}'")
        return upper

    @model_validator(mode="after")
    def warn_insecure_defaults_in_production(self) -> "AppSettings":
        if self.ENVIRONMENT == "production":
            insecure = []
            if "CHANGE-IN-PRODUCTION" in self.SECRET_KEY:
                insecure.append("SECRET_KEY")
            if "CHANGE-IN-PRODUCTION" in self.JWT_SECRET_KEY:
                insecure.append("JWT_SECRET_KEY")
            if insecure:
                raise ValueError(
                    f"The following settings use insecure default values in production: "
                    f"{insecure}. Set them via environment variables."
                )
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the validated application settings singleton."""
    return AppSettings()
