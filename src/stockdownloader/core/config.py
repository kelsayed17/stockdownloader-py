"""Project-wide constants and JSON-based configuration loader.

Single source of truth for default values used across the codebase.
These can be overridden at runtime via CLI arguments or config files,
but these values serve as sensible defaults.

Also provides a lightweight, generic config system that replaces hardcoded
defaults throughout the codebase.  Configs are plain JSON files loaded
from disk (or a dict passed in-memory for testing).

Design principles:

- **Generic** — works for any subsystem (backtest, ML pipeline, strategy, etc.)
- **Layered** — base defaults → JSON preset → CLI overrides
- **Simple** — just JSON, no YAML/TOML dependencies
- **Optional** — everything works without config files (built-in defaults apply)

Usage::

    from stockdownloader.core.config import load_config

    # Load a named preset from the config directory
    cfg = load_config("backtest/spy_daily.json")

    # Or load with explicit path
    cfg = load_config("/path/to/config.json")

    # Access nested values with dot notation
    capital = cfg.get("backtest.initial_capital", 100000.0)

    # Merge CLI overrides on top
    cfg = load_config("ml/standard.json", overrides={"quick": True})

Config file search order:

1. Absolute path (if path starts with ``/``)
2. ``$PROJECT_ROOT/config/<path>``
3. ``$PROJECT_ROOT/<path>``

Example JSON config::

    {
        "backtest": {
            "initial_capital": 100000.0,
            "risk_per_trade": 0.01,
            "commission": 0.0
        },
        "training": {
            "forward_periods": [5, 10, 20],
            "profit_thresholds": [0.003, 0.005, 0.01],
            "model_types": ["gradient_boosting", "logistic_regression"],
            "n_estimators": 200,
            "max_depth": 4
        }
    }
"""

from __future__ import annotations

import bisect
import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

# ======================================================================
# Capital & risk
# ======================================================================

INITIAL_CAPITAL = Decimal("100000.00")
RISK_PER_TRADE = Decimal("0.01")  # 1 % of capital per trade
OPTIONS_COMMISSION = Decimal("0.65")  # Per-contract options commission

# ======================================================================
# Project paths (relative to project root)
# ======================================================================

#: Project root (3 levels up from this file: util → stockdownloader → src → root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

#: Default path to the SPY 5-minute bar CSV.
DEFAULT_DATA_FILE: Path = PROJECT_ROOT / "data" / "SPY" / "5m_bars.csv"

#: Default output directory for logs and reports.
DEFAULT_OUTPUT_DIR: Path = PROJECT_ROOT / "output"

#: Default models directory.
DEFAULT_MODELS_DIR: Path = DEFAULT_OUTPUT_DIR / "models"

#: Default PineScript output directory.
DEFAULT_PINESCRIPT_DIR: Path = DEFAULT_OUTPUT_DIR / "pinescript"

#: Default ML pipeline output directory.
DEFAULT_ML_PIPELINE_DIR: Path = DEFAULT_OUTPUT_DIR / "ml_pipeline"

#: Default alert history file.
DEFAULT_ALERT_HISTORY: Path = DEFAULT_OUTPUT_DIR / "alert_history.json"

#: Default pattern catalog directory.
DEFAULT_PATTERNS_DIR: Path = DEFAULT_OUTPUT_DIR / "patterns"

# ======================================================================
# Walk-forward defaults
# ======================================================================

WF_WINDOWS = 5
WF_IS_RATIO = 0.7  # 70 % in-sample, 30 % out-of-sample

logger = logging.getLogger(__name__)

#: Default config directory (project root / config /).
CONFIG_DIR: Path = PROJECT_ROOT / "config"


class Config:
    """Lightweight config wrapper with dot-notation access.

    Supports nested key access via ``get("a.b.c", default)`` and
    dict-style ``cfg["key"]`` access.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value using dot-notation (e.g. ``"backtest.capital"``).

        Parameters
        ----------
        key:
            Dot-separated path to the value.
        default:
            Fallback if the key is missing at any level.
        """
        parts = key.split(".")
        current: Any = self._data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    @property
    def data(self) -> dict[str, Any]:
        """Raw config dictionary."""
        return self._data

    def section(self, key: str) -> Config:
        """Return a sub-config for a nested section.

        >>> cfg = Config({"backtest": {"capital": 100000}})
        >>> cfg.section("backtest").get("capital")
        100000
        """
        val = self.get(key, {})
        if isinstance(val, dict):
            return Config(val)
        return Config({})

    def merge(self, overrides: dict[str, Any]) -> Config:
        """Return a new Config with *overrides* merged on top.

        Performs a shallow merge at each nesting level.
        """
        return Config(_deep_merge(self._data, overrides))

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> Config:
    """Load a JSON config file and optionally merge overrides.

    Parameters
    ----------
    path:
        Path to a JSON file.  Resolved against ``config/`` then project root.
        If ``None``, returns an empty config (or just *overrides*).
    overrides:
        Dict to merge on top of the loaded config (e.g. CLI args).
    data:
        Direct dict to use instead of reading from disk (for testing).

    Returns
    -------
    Config
        Loaded and merged config.
    """
    if data is not None:
        cfg = Config(data)
    elif path is not None:
        resolved = _resolve_path(path)
        if resolved is None:
            logger.warning("Config file not found: %s", path)
            cfg = Config({})
        else:
            logger.debug("Loading config from %s", resolved)
            with open(resolved) as f:
                cfg = Config(json.load(f))
    else:
        cfg = Config({})

    if overrides:
        cfg = cfg.merge(overrides)

    return cfg


def _resolve_path(path: str | Path) -> Path | None:
    """Resolve a config path against known directories."""
    p = Path(path)

    # Absolute path
    if p.is_absolute():
        return p if p.exists() else None

    # Try config/ directory first
    candidate = CONFIG_DIR / p
    if candidate.exists():
        return candidate

    # Try project root
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return candidate

    return None


def list_configs(subdir: str = "") -> list[str]:
    """List available config files under ``config/<subdir>/``.

    Returns relative paths (e.g. ``["backtest/spy.json", "ml/standard.json"]``).
    """
    search = CONFIG_DIR / subdir
    if not search.is_dir():
        return []
    return sorted(
        str(p.relative_to(CONFIG_DIR))
        for p in search.rglob("*.json")
    )


# ======================================================================
# Application configuration (env-var based)
# ======================================================================


@dataclass(frozen=True)
class AppConfig:
    """Application-wide configuration read from environment.

    All fields default to empty strings so callers can check
    ``if config.polygon_api_key:`` without worrying about None.

    Preferred construction is via :meth:`from_env`, which reads
    environment variables once and stores the results immutably.
    """

    # -- Polygon.io ----------------------------------------------------------
    polygon_api_key: str = ""

    # -- FINRA (short interest, dark pool, reg-SHO) --------------------------
    finra_client_id: str = ""
    finra_client_secret: str = ""

    # -- Tradier (options chain data) ----------------------------------------
    tradier_api_token: str = ""
    tradier_sandbox_token: str = ""

    @classmethod
    def from_env(cls, **overrides: str) -> AppConfig:
        """Load configuration from environment variables.

        Any keyword argument overrides the corresponding env-var value,
        making it easy to merge CLI flags::

            config = AppConfig.from_env(polygon_api_key=args.polygon_key)

        Parameters
        ----------
        **overrides:
            Field-name / value pairs that take priority over the
            environment.  Empty-string or ``None`` overrides are ignored
            so that ``args.polygon_key or ""`` never clobbers a real
            env-var value.
        """

        def _get(field: str, env_var: str) -> str:
            override = overrides.get(field)
            if override:               # non-empty string wins
                return override
            return os.environ.get(env_var, "")

        return cls(
            polygon_api_key=_get("polygon_api_key", "POLYGON_API_KEY"),
            finra_client_id=_get("finra_client_id", "FINRA_CLIENT_ID"),
            finra_client_secret=_get("finra_client_secret", "FINRA_CLIENT_SECRET"),
            tradier_api_token=_get("tradier_api_token", "TRADIER_API_TOKEN"),
            tradier_sandbox_token=_get("tradier_sandbox_token", "TRADIER_SANDBOX_TOKEN"),
        )


# =========================================================================
# Event calendar (formerly event_calendar.py)
# =========================================================================

# ── FOMC meeting dates (announcement day) ─────────────────────────────
# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

FOMC_DATES: tuple[str, ...] = (
    # 2024
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
    # 2025
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-17",
    # 2026
    "2026-01-28",
    "2026-03-18",
    "2026-05-06",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-10-28",
    "2026-12-16",
)

_ANCHOR_TABLES: dict[str, tuple[str, ...]] = {
    "fomc": FOMC_DATES,
}


def get_anchor_date(
    trading_date: str,
    anchor_type: str = "fomc",
) -> str | None:
    """Return the most recent anchor date on or before *trading_date*.

    Parameters
    ----------
    trading_date:
        ``"YYYY-MM-DD"`` date string.
    anchor_type:
        Anchor event type.  Currently only ``"fomc"`` is supported.

    Returns
    -------
    The anchor date string, or ``None`` if *trading_date* precedes all
    known anchors.
    """
    dates = _ANCHOR_TABLES.get(anchor_type)
    if dates is None:
        return None
    # bisect_right finds insert point *after* any equal element
    idx = bisect.bisect_right(dates, trading_date)
    if idx == 0:
        return None
    return dates[idx - 1]


def is_anchor_date(
    trading_date: str,
    anchor_type: str = "fomc",
) -> bool:
    """Return ``True`` if *trading_date* is an anchor date."""
    dates = _ANCHOR_TABLES.get(anchor_type)
    if dates is None:
        return False
    idx = bisect.bisect_left(dates, trading_date)
    return idx < len(dates) and dates[idx] == trading_date


def days_since_anchor(
    trading_date: str,
    anchor_type: str = "fomc",
) -> int | None:
    """Return the number of calendar days since the most recent anchor.

    Returns ``None`` if *trading_date* precedes all known anchors.
    Returns ``0`` on the anchor day itself.
    """
    anchor = get_anchor_date(trading_date, anchor_type)
    if anchor is None:
        return None
    td = _parse_date(trading_date)
    ad = _parse_date(anchor)
    return (td - ad).days


def _parse_date(date_str: str) -> date:
    """Parse ``YYYY-MM-DD`` string to :class:`datetime.date`."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()
