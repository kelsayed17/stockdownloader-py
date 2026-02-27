"""Configuration for GME options analysis platform."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from stockdownloader.core.config import PROJECT_ROOT


def _default_data_dir() -> Path:
    return PROJECT_ROOT / "data" / "GME" / "options"


@dataclass(frozen=True, slots=True)
class GMEOptionsConfig:
    """Immutable configuration for the GME options pipeline."""

    # Symbol
    symbol: str = "GME"

    # Date range
    start_date: date = date(2023, 1, 1)
    end_date: date = date(2026, 2, 26)

    # API
    polygon_api_key: str = ""
    rate_limit_delay: float = 0.2

    # Storage
    data_dir: Path = field(default_factory=_default_data_dir)

    # Backtest
    initial_capital: float = 100_000.0
    commission_per_contract: float = 0.65
    risk_per_trade_pct: float = 1.0
    sl_atr_mult: float = 1.5
    rr_ratio: float = 1.5

    # Risk management
    max_trades_per_day: int = 4
    min_bars_between: int = 3
    circuit_breaker_losses: int = 3
    daily_loss_limit_pct: float = 3.0

    # OI proxy
    oi_decay_rate: float = 0.03

    # Walk-forward
    train_years: int = 2
    test_years: int = 1
    roll_months: int = 3

    @classmethod
    def from_env(cls, **overrides) -> GMEOptionsConfig:
        """Create config with env-var fallbacks."""
        defaults = {
            "polygon_api_key": os.environ.get("POLYGON_API_KEY", ""),
        }
        defaults.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**defaults)
