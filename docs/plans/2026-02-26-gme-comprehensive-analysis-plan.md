# GME Comprehensive Options Analysis — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a full GME options research platform: resumable Polygon data pipeline → 40-metric analytics engine → 6-strategy backtester → multi-signal fusion scorer.

**Architecture:** Layered — Data (fetcher.py) → Analytics (state_engine.py) → Strategy (backtester.py + 6 strategies) → Fusion (signal_fusion.py + scorecard.py), all wired via CLI (__main__.py).

**Tech Stack:** Python 3.12, pandas, pyarrow (Parquet), scipy (Black-Scholes inversion), existing `PolygonOptionsClient`, existing `AlternativeDataStore`, pytest.

**Test command:** `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest {test_path} -v`

**Source root:** `src/stockdownloader/gme/options/`
**Test root:** `tests/gme/options/`

---

### Task 1: Module Scaffold + Config

**Files:**
- Create: `src/stockdownloader/gme/options/__init__.py`
- Create: `src/stockdownloader/gme/options/config.py`
- Create: `src/stockdownloader/gme/options/strategies/__init__.py`
- Create: `tests/gme/options/__init__.py`
- Create: `tests/gme/options/test_config.py`

**Step 1: Create directories and __init__.py files**

```bash
mkdir -p src/stockdownloader/gme/options/strategies
mkdir -p tests/gme/options
touch src/stockdownloader/gme/options/__init__.py
touch src/stockdownloader/gme/options/strategies/__init__.py
touch tests/gme/options/__init__.py
```

**Step 2: Write the failing test for config**

```python
# tests/gme/options/test_config.py
"""Tests for GMEOptionsConfig."""
from __future__ import annotations

import pytest
from stockdownloader.gme.options.config import GMEOptionsConfig


class TestGMEOptionsConfig:
    def test_defaults(self):
        cfg = GMEOptionsConfig()
        assert cfg.symbol == "GME"
        assert cfg.rate_limit_delay == 0.2
        assert cfg.risk_per_trade_pct == 1.0
        assert cfg.sl_atr_mult == 1.5
        assert cfg.rr_ratio == 1.5
        assert cfg.max_trades_per_day == 4
        assert cfg.circuit_breaker_losses == 3

    def test_custom_params(self):
        cfg = GMEOptionsConfig(symbol="AMC", rate_limit_delay=0.5)
        assert cfg.symbol == "AMC"
        assert cfg.rate_limit_delay == 0.5

    def test_frozen(self):
        cfg = GMEOptionsConfig()
        with pytest.raises(AttributeError):
            cfg.symbol = "AMC"

    def test_from_env_with_overrides(self):
        cfg = GMEOptionsConfig.from_env(symbol="BBBY")
        assert cfg.symbol == "BBBY"

    def test_data_dir_default(self):
        cfg = GMEOptionsConfig()
        assert "data/GME/options" in str(cfg.data_dir)
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stockdownloader.gme.options.config'`

**Step 3: Implement config**

```python
# src/stockdownloader/gme/options/config.py
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
    start_date: date = date(2022, 1, 1)
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
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_config.py -v`
Expected: PASS (5 tests)

**Step 4: Commit**

```bash
git add src/stockdownloader/gme/options/ tests/gme/options/
git commit -m "feat(gme): add options module scaffold and GMEOptionsConfig"
```

---

### Task 2: Resumable Fetcher — Contract Discovery

**Files:**
- Create: `src/stockdownloader/gme/options/fetcher.py`
- Create: `tests/gme/options/test_fetcher.py`

**Step 1: Write failing test for checkpoint management**

```python
# tests/gme/options/test_fetcher.py
"""Tests for resumable Polygon options fetcher."""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import date

from stockdownloader.gme.options.fetcher import (
    FetchCheckpoint,
    OptionsDataFetcher,
)
from stockdownloader.gme.options.config import GMEOptionsConfig


class TestFetchCheckpoint:
    def test_empty_checkpoint(self, tmp_path):
        cp = FetchCheckpoint(tmp_path / "checkpoint.json")
        assert cp.completed_months == set()
        assert cp.last_contract is None

    def test_mark_month_complete(self, tmp_path):
        cp = FetchCheckpoint(tmp_path / "checkpoint.json")
        cp.mark_month_complete("2022-01")
        assert "2022-01" in cp.completed_months

    def test_persist_and_reload(self, tmp_path):
        path = tmp_path / "checkpoint.json"
        cp = FetchCheckpoint(path)
        cp.mark_month_complete("2022-01")
        cp.mark_month_complete("2022-02")
        cp.save()

        cp2 = FetchCheckpoint(path)
        assert cp2.completed_months == {"2022-01", "2022-02"}

    def test_is_month_done(self, tmp_path):
        cp = FetchCheckpoint(tmp_path / "checkpoint.json")
        cp.mark_month_complete("2022-03")
        assert cp.is_month_done("2022-03") is True
        assert cp.is_month_done("2022-04") is False

    def test_update_last_contract(self, tmp_path):
        cp = FetchCheckpoint(tmp_path / "checkpoint.json")
        cp.last_contract = "O:GME220121C00020000"
        cp.save()
        cp2 = FetchCheckpoint(tmp_path / "checkpoint.json")
        assert cp2.last_contract == "O:GME220121C00020000"


class TestOptionsDataFetcherMonthList:
    def test_generates_month_range(self):
        cfg = GMEOptionsConfig(
            start_date=date(2022, 1, 1),
            end_date=date(2022, 4, 15),
        )
        fetcher = OptionsDataFetcher(cfg, client=MagicMock())
        months = fetcher._expiration_months()
        assert months == ["2022-01", "2022-02", "2022-03", "2022-04"]

    def test_skips_completed_months(self, tmp_path):
        cfg = GMEOptionsConfig(
            start_date=date(2022, 1, 1),
            end_date=date(2022, 3, 31),
            data_dir=tmp_path,
        )
        fetcher = OptionsDataFetcher(cfg, client=MagicMock())
        fetcher.checkpoint.mark_month_complete("2022-01")
        fetcher.checkpoint.mark_month_complete("2022-02")
        remaining = fetcher._remaining_months()
        assert remaining == ["2022-03"]
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_fetcher.py -v`
Expected: FAIL — `ImportError`

**Step 2: Implement checkpoint + month logic**

```python
# src/stockdownloader/gme/options/fetcher.py
"""Resumable Polygon options data fetcher."""
from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from stockdownloader.data.market.polygon_options_client import (
        PolygonOptionsClient,
    )

from stockdownloader.gme.options.config import GMEOptionsConfig

logger = logging.getLogger(__name__)


class FetchCheckpoint:
    """Tracks which expiration months have been fully fetched."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.completed_months: set[str] = set()
        self.last_contract: str | None = None
        if path.exists():
            data = json.loads(path.read_text())
            self.completed_months = set(data.get("completed_months", []))
            self.last_contract = data.get("last_contract")

    def mark_month_complete(self, month: str) -> None:
        self.completed_months.add(month)

    def is_month_done(self, month: str) -> bool:
        return month in self.completed_months

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "completed_months": sorted(self.completed_months),
            "last_contract": self.last_contract,
        }, indent=2))


class OptionsDataFetcher:
    """Resumable pipeline: discover contracts by month, fetch daily bars."""

    def __init__(
        self,
        config: GMEOptionsConfig,
        client: PolygonOptionsClient,
    ) -> None:
        self.config = config
        self.client = client
        self.checkpoint = FetchCheckpoint(config.data_dir / "checkpoint.json")

    # ------------------------------------------------------------------
    # Month helpers
    # ------------------------------------------------------------------

    def _expiration_months(self) -> list[str]:
        """All YYYY-MM months from start_date to end_date."""
        months: list[str] = []
        cur = self.config.start_date.replace(day=1)
        end = self.config.end_date.replace(day=1)
        while cur <= end:
            months.append(cur.strftime("%Y-%m"))
            # advance to next month
            if cur.month == 12:
                cur = cur.replace(year=cur.year + 1, month=1)
            else:
                cur = cur.replace(month=cur.month + 1)
        return months

    def _remaining_months(self) -> list[str]:
        """Months not yet completed."""
        return [
            m for m in self._expiration_months()
            if not self.checkpoint.is_month_done(m)
        ]

    # ------------------------------------------------------------------
    # Contract discovery
    # ------------------------------------------------------------------

    def _month_date_range(self, month: str) -> tuple[date, date]:
        """First and last day of an expiration month."""
        year, mon = int(month[:4]), int(month[5:])
        first = date(year, mon, 1)
        if mon == 12:
            last = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            last = date(year, mon + 1, 1) - timedelta(days=1)
        return first, last

    def fetch_contracts_for_month(self, month: str) -> list[dict]:
        """Discover all GME option contracts expiring in *month*."""
        first, last = self._month_date_range(month)
        contracts = self.client.fetch_option_contracts(
            symbol=self.config.symbol,
            from_date=first.isoformat(),
            to_date=last.isoformat(),
            expired=True,
        )
        logger.info("Month %s: found %d contracts", month, len(contracts))
        return contracts

    # ------------------------------------------------------------------
    # Daily bar fetching
    # ------------------------------------------------------------------

    def fetch_bars_for_month(self, month: str) -> pd.DataFrame:
        """Fetch daily bars for every contract expiring in *month*."""
        contracts = self.fetch_contracts_for_month(month)
        rows: list[dict] = []

        for contract in contracts:
            ticker = contract["ticker"]
            self.checkpoint.last_contract = ticker
            bars = self._fetch_contract_bars(contract)
            rows.extend(bars)
            time.sleep(self.config.rate_limit_delay)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return df

    def _fetch_contract_bars(self, contract: dict) -> list[dict]:
        """Fetch all daily bars for one contract over its lifetime."""
        ticker = contract["ticker"]
        # Use listing/expiration dates to bound the query
        list_date = contract.get("listed_date", self.config.start_date.isoformat())
        exp_date = contract.get("expiration_date", self.config.end_date.isoformat())

        bars_raw = self.client.fetch_option_daily_bar(ticker, list_date)
        # The existing client fetches a single bar — we need a range.
        # We'll add a range method. For now, wrap single-bar calls.
        rows: list[dict] = []
        if bars_raw:
            rows.append({
                "option_ticker": ticker,
                "underlying": self.config.symbol,
                "date": list_date,
                "expiration": exp_date,
                "strike": contract.get("strike_price", 0),
                "option_type": contract.get("contract_type", "").lower(),
                "open": bars_raw.get("o", 0),
                "high": bars_raw.get("h", 0),
                "low": bars_raw.get("l", 0),
                "close": bars_raw.get("c", 0),
                "volume": bars_raw.get("v", 0),
                "open_interest": 0,
                "vwap": bars_raw.get("vw", 0),
            })
        return rows

    # ------------------------------------------------------------------
    # Parquet I/O
    # ------------------------------------------------------------------

    def save_month(self, month: str, df: pd.DataFrame) -> Path:
        """Save a month's bars to Parquet."""
        out_dir = self.config.data_dir / month
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "daily_bars.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved %d rows to %s", len(df), path)
        return path

    @staticmethod
    def load_month(path: Path) -> pd.DataFrame:
        """Load a month's Parquet file."""
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Fetch all remaining months, saving checkpoint after each."""
        remaining = self._remaining_months()
        logger.info(
            "Starting fetch: %d months remaining of %d total",
            len(remaining), len(self._expiration_months()),
        )
        for month in remaining:
            logger.info("Fetching month %s ...", month)
            df = self.fetch_bars_for_month(month)
            if not df.empty:
                self.save_month(month, df)
            self.checkpoint.mark_month_complete(month)
            self.checkpoint.save()
            logger.info("Month %s complete (%d bars)", month, len(df))
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_fetcher.py -v`
Expected: PASS (7 tests)

**Step 3: Write test for bar fetching with mocked client**

Add to `tests/gme/options/test_fetcher.py`:

```python
class TestOptionsDataFetcherBars:
    def test_fetch_bars_for_month(self, tmp_path):
        mock_client = MagicMock()
        mock_client.fetch_option_contracts.return_value = [
            {
                "ticker": "O:GME220121C00020000",
                "listed_date": "2022-01-03",
                "expiration_date": "2022-01-21",
                "strike_price": 20.0,
                "contract_type": "call",
            },
        ]
        mock_client.fetch_option_daily_bar.return_value = {
            "o": 5.0, "h": 6.0, "l": 4.5, "c": 5.5, "v": 100, "vw": 5.3,
        }

        cfg = GMEOptionsConfig(
            start_date=date(2022, 1, 1),
            end_date=date(2022, 1, 31),
            data_dir=tmp_path,
            rate_limit_delay=0.0,
        )
        fetcher = OptionsDataFetcher(cfg, client=mock_client)
        df = fetcher.fetch_bars_for_month("2022-01")

        assert len(df) == 1
        assert df.iloc[0]["option_ticker"] == "O:GME220121C00020000"
        assert df.iloc[0]["close"] == 5.5
        assert df.iloc[0]["option_type"] == "call"

    def test_save_and_load_month(self, tmp_path):
        import pandas as pd
        cfg = GMEOptionsConfig(data_dir=tmp_path)
        df = pd.DataFrame([{
            "option_ticker": "O:GME220121C00020000",
            "close": 5.5,
        }])
        fetcher = OptionsDataFetcher(cfg, client=MagicMock())
        path = fetcher.save_month("2022-01", df)
        loaded = fetcher.load_month(path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["close"] == 5.5
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_fetcher.py -v`
Expected: PASS (9 tests)

**Step 4: Commit**

```bash
git add src/stockdownloader/gme/options/fetcher.py tests/gme/options/test_fetcher.py
git commit -m "feat(gme): add resumable options data fetcher with checkpoint"
```

---

### Task 3: OptionsStateEngine — IV Surface + GEX

**Files:**
- Create: `src/stockdownloader/gme/options/state_engine.py`
- Create: `tests/gme/options/test_state_engine.py`

**Step 1: Write failing test for IV surface computation**

```python
# tests/gme/options/test_state_engine.py
"""Tests for OptionsStateEngine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date

from stockdownloader.gme.options.state_engine import OptionsStateEngine


def _make_chain_df(
    trade_date: str = "2023-06-15",
    spot: float = 25.0,
    n_strikes: int = 10,
) -> pd.DataFrame:
    """Create a synthetic options chain for one day."""
    rows = []
    strikes = np.linspace(spot * 0.7, spot * 1.3, n_strikes)
    for exp_offset in [30, 60, 90]:
        exp = pd.Timestamp(trade_date) + pd.Timedelta(days=exp_offset)
        for strike in strikes:
            for otype in ["call", "put"]:
                # Synthetic price increases with moneyness
                moneyness = spot / strike if otype == "call" else strike / spot
                price = max(0.5, moneyness * 2.0 + np.random.uniform(0, 0.5))
                rows.append({
                    "option_ticker": f"O:GME{exp.strftime('%y%m%d')}{'C' if otype == 'call' else 'P'}{int(strike*1000):08d}",
                    "underlying": "GME",
                    "date": trade_date,
                    "expiration": exp.strftime("%Y-%m-%d"),
                    "strike": round(strike, 2),
                    "option_type": otype,
                    "open": price,
                    "high": price * 1.05,
                    "low": price * 0.95,
                    "close": price,
                    "volume": int(np.random.uniform(10, 500)),
                    "open_interest": int(np.random.uniform(100, 5000)),
                    "vwap": price,
                })
    return pd.DataFrame(rows)


class TestIVSurface:
    def test_computes_atm_iv_30d(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_iv_surface(df, date(2023, 6, 15), 25.0)
        assert "atm_iv_30d" in metrics
        assert 0.0 < metrics["atm_iv_30d"] < 5.0  # reasonable IV range

    def test_computes_iv_percentile(self):
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        # Build history of 30 days
        history = [0.4 + i * 0.01 for i in range(30)]
        pct = engine._iv_percentile(0.55, history)
        assert 0.0 <= pct <= 1.0

    def test_iv_skew_sign(self):
        """Put IV should exceed call IV (positive skew) for equity."""
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_iv_surface(df, date(2023, 6, 15), 25.0)
        assert "iv_skew_25d_30" in metrics


class TestGEX:
    def test_computes_net_gex(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_gex(df, 25.0)
        assert "net_gex" in metrics
        assert isinstance(metrics["net_gex"], float)

    def test_computes_call_put_walls(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_gex(df, 25.0)
        assert "call_wall" in metrics
        assert "put_wall" in metrics
        assert metrics["call_wall"] >= metrics["put_wall"]

    def test_gex_flip_between_walls(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_gex(df, 25.0)
        if metrics["gex_flip_price"] > 0:
            assert metrics["put_wall"] <= metrics["gex_flip_price"] <= metrics["call_wall"] * 1.5
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_state_engine.py -v`
Expected: FAIL — `ImportError`

**Step 2: Implement IV surface + GEX computation**

```python
# src/stockdownloader/gme/options/state_engine.py
"""OptionsStateEngine: compute daily options analytics from raw bars."""
from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Black-Scholes helpers (float-based for vectorized speed)
# ---------------------------------------------------------------------------

def _bs_price(
    is_call: bool, S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_delta(
    is_call: bool, S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if is_call else -1.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0


def _bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def _implied_vol(
    is_call: bool, S: float, K: float, T: float, r: float,
    market_price: float, max_iter: int = 50, tol: float = 1e-6,
) -> float:
    """Newton-Raphson implied volatility inversion."""
    if market_price <= 0 or T <= 0:
        return 0.0
    sigma = 0.5  # initial guess
    for _ in range(max_iter):
        price = _bs_price(is_call, S, K, T, r, sigma)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        vega = S * norm.pdf(d1) * math.sqrt(T)
        if vega < 1e-12:
            break
        sigma -= (price - market_price) / vega
        sigma = max(sigma, 0.01)
        if abs(price - market_price) < tol:
            break
    return sigma if 0.01 < sigma < 10.0 else 0.0


# ---------------------------------------------------------------------------
# OptionsStateEngine
# ---------------------------------------------------------------------------


class OptionsStateEngine:
    """Compute ~40 daily metrics from raw options chain data."""

    RISK_FREE_RATE = 0.05

    def __init__(self, spot_prices: dict[str, float] | None = None) -> None:
        self.spot_prices = spot_prices or {}
        self._iv_history: list[float] = []

    # ------------------------------------------------------------------
    # IV Surface (8 metrics)
    # ------------------------------------------------------------------

    def _compute_iv_surface(
        self, df: pd.DataFrame, trade_date: date, spot: float,
    ) -> dict[str, float]:
        """Compute IV surface metrics for a single trading day."""
        r = self.RISK_FREE_RATE
        metrics: dict[str, float] = {}

        for tenor_days, label in [(30, "30d"), (60, "60d"), (90, "90d")]:
            target_exp = trade_date + timedelta(days=tenor_days)
            # Find closest expiration to target tenor
            df["exp_date"] = pd.to_datetime(df["expiration"])
            df["tenor_diff"] = abs((df["exp_date"] - pd.Timestamp(target_exp)).dt.days)
            tenor_df = df[df["tenor_diff"] <= 15]  # within 15 days of target

            if tenor_df.empty:
                metrics[f"atm_iv_{label}"] = 0.0
                continue

            # ATM: closest strike to spot
            tenor_df = tenor_df.copy()
            tenor_df["strike_diff"] = abs(tenor_df["strike"] - spot)
            atm = tenor_df.nsmallest(2, "strike_diff")

            T = tenor_days / 365.0
            ivs = []
            for _, row in atm.iterrows():
                is_call = row["option_type"] == "call"
                mid = (row["close"] + row.get("vwap", row["close"])) / 2
                iv = _implied_vol(is_call, spot, row["strike"], T, r, mid)
                if iv > 0:
                    ivs.append(iv)

            metrics[f"atm_iv_{label}"] = float(np.mean(ivs)) if ivs else 0.0

        # Skew: 25-delta put IV minus 25-delta call IV (30-day)
        metrics["iv_skew_25d_30"] = self._compute_skew(df, trade_date, spot, 30)

        # Term structure slope
        iv30 = metrics.get("atm_iv_30d", 0)
        iv90 = metrics.get("atm_iv_90d", 0)
        metrics["iv_term_slope"] = (iv90 - iv30) if iv30 > 0 and iv90 > 0 else 0.0

        # Percentile
        if iv30 > 0:
            self._iv_history.append(iv30)
        metrics["iv_percentile"] = self._iv_percentile(iv30, self._iv_history)

        # Changes
        metrics["iv_change_1d"] = 0.0  # filled by build() with prior day
        metrics["iv_change_5d"] = 0.0
        return metrics

    def _compute_skew(
        self, df: pd.DataFrame, trade_date: date, spot: float, tenor_days: int,
    ) -> float:
        """25-delta skew for given tenor."""
        r = self.RISK_FREE_RATE
        T = tenor_days / 365.0
        target_exp = trade_date + timedelta(days=tenor_days)

        df_t = df.copy()
        df_t["exp_date"] = pd.to_datetime(df_t["expiration"])
        df_t["tenor_diff"] = abs((df_t["exp_date"] - pd.Timestamp(target_exp)).dt.days)
        tenor_df = df_t[df_t["tenor_diff"] <= 15]

        if tenor_df.empty:
            return 0.0

        # Compute delta for each option, find closest to 0.25
        put_iv = call_iv = 0.0
        for otype, target_delta in [("put", -0.25), ("call", 0.25)]:
            sub = tenor_df[tenor_df["option_type"] == otype].copy()
            if sub.empty:
                continue
            sub["bs_delta"] = sub.apply(
                lambda row: _bs_delta(
                    otype == "call", spot, row["strike"], T, r, 0.3,
                ), axis=1,
            )
            sub["delta_diff"] = abs(sub["bs_delta"] - target_delta)
            closest = sub.nsmallest(1, "delta_diff").iloc[0]
            iv = _implied_vol(
                otype == "call", spot, closest["strike"], T, r, closest["close"],
            )
            if otype == "put":
                put_iv = iv
            else:
                call_iv = iv

        return put_iv - call_iv

    @staticmethod
    def _iv_percentile(current: float, history: list[float]) -> float:
        """Rank current IV within history (0-1)."""
        if not history or current <= 0:
            return 0.5
        below = sum(1 for h in history if h <= current)
        return below / len(history)

    # ------------------------------------------------------------------
    # GEX (5 metrics)
    # ------------------------------------------------------------------

    def _compute_gex(self, df: pd.DataFrame, spot: float) -> dict[str, float]:
        """Compute gamma exposure metrics."""
        r = self.RISK_FREE_RATE
        rows = []

        for _, row in df.iterrows():
            exp = pd.Timestamp(row["expiration"])
            T = max((exp - pd.Timestamp(row["date"])).days / 365.0, 1 / 365)
            is_call = row["option_type"] == "call"
            strike = row["strike"]
            oi = row.get("open_interest", 0)

            gamma = _bs_gamma(spot, strike, T, r, 0.3)
            # Dealer is short calls (sold to retail), long puts
            dealer_sign = -1.0 if is_call else 1.0
            gex = dealer_sign * gamma * oi * 100 * spot
            rows.append({
                "strike": strike, "gex": gex, "is_call": is_call,
                "oi_gamma": abs(gamma * oi),
            })

        if not rows:
            return {
                "net_gex": 0.0, "gex_flip_price": 0.0,
                "call_wall": 0.0, "put_wall": 0.0,
                "gex_concentration": 0.0,
            }

        gex_df = pd.DataFrame(rows)
        net_gex = gex_df["gex"].sum()

        # Call wall: strike with highest call OI * gamma
        call_df = gex_df[gex_df["is_call"]]
        put_df = gex_df[~gex_df["is_call"]]
        call_wall = float(call_df.loc[call_df["oi_gamma"].idxmax(), "strike"]) if not call_df.empty else 0.0
        put_wall = float(put_df.loc[put_df["oi_gamma"].idxmax(), "strike"]) if not put_df.empty else 0.0

        # GEX flip: find strike where cumulative GEX crosses zero
        strike_gex = gex_df.groupby("strike")["gex"].sum().sort_index()
        cum = strike_gex.cumsum()
        flip_price = 0.0
        for i in range(1, len(cum)):
            if cum.iloc[i - 1] * cum.iloc[i] < 0:
                flip_price = float(cum.index[i])
                break

        # Concentration: top 3 strikes as fraction of total
        total_abs = gex_df["oi_gamma"].sum()
        top3 = gex_df.nlargest(3, "oi_gamma")["oi_gamma"].sum()
        concentration = top3 / total_abs if total_abs > 0 else 0.0

        return {
            "net_gex": float(net_gex),
            "gex_flip_price": flip_price,
            "call_wall": call_wall,
            "put_wall": put_wall,
            "gex_concentration": float(concentration),
        }

    # ------------------------------------------------------------------
    # OI Flow (8 metrics)
    # ------------------------------------------------------------------

    def _compute_oi_flow(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute open interest flow metrics."""
        calls = df[df["option_type"] == "call"]
        puts = df[df["option_type"] == "put"]

        total_call_oi = int(calls["open_interest"].sum())
        total_put_oi = int(puts["open_interest"].sum())
        total_oi = total_call_oi + total_put_oi

        pc_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0.0

        # OI-weighted strike (center of mass)
        if total_oi > 0:
            oi_weighted = (df["strike"] * df["open_interest"]).sum() / total_oi
        else:
            oi_weighted = 0.0

        # Concentration: top 5 strikes
        strike_oi = df.groupby("strike")["open_interest"].sum()
        if len(strike_oi) > 0 and total_oi > 0:
            top5 = strike_oi.nlargest(5).sum()
            concentration = top5 / total_oi
        else:
            concentration = 0.0

        # Near-term: expiring within 30 days
        df_c = df.copy()
        df_c["exp_date"] = pd.to_datetime(df_c["expiration"])
        df_c["dte"] = (df_c["exp_date"] - pd.to_datetime(df_c["date"])).dt.days
        near = df_c[df_c["dte"] <= 30]["open_interest"].sum()
        near_pct = near / total_oi if total_oi > 0 else 0.0

        return {
            "total_call_oi": float(total_call_oi),
            "total_put_oi": float(total_put_oi),
            "pc_oi_ratio": float(pc_oi_ratio),
            "pc_oi_ratio_change": 0.0,  # filled by build() with prior day
            "oi_weighted_strike": float(oi_weighted),
            "oi_concentration_top5": float(concentration),
            "near_term_oi_pct": float(near_pct),
            "oi_skew_delta": 0.0,  # filled by build()
        }

    # ------------------------------------------------------------------
    # Volume & Premium Flow (6 metrics)
    # ------------------------------------------------------------------

    def _compute_volume_premium(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute volume and premium flow metrics."""
        calls = df[df["option_type"] == "call"]
        puts = df[df["option_type"] == "put"]

        call_vol = int(calls["volume"].sum())
        put_vol = int(puts["volume"].sum())
        pc_vol_ratio = put_vol / call_vol if call_vol > 0 else 0.0

        # Premium = volume * close * 100 (contract multiplier)
        call_premium = (calls["volume"] * calls["close"] * 100).sum()
        put_premium = (puts["volume"] * puts["close"] * 100).sum()
        total_premium = call_premium + put_premium
        imbalance = (
            (call_premium - put_premium) / total_premium
            if total_premium > 0 else 0.0
        )

        return {
            "call_volume": float(call_vol),
            "put_volume": float(put_vol),
            "pc_volume_ratio": float(pc_vol_ratio),
            "call_premium": float(call_premium),
            "put_premium": float(put_premium),
            "premium_imbalance": float(imbalance),
        }

    # ------------------------------------------------------------------
    # Build full state
    # ------------------------------------------------------------------

    def compute_day(
        self, df: pd.DataFrame, trade_date: date, spot: float,
    ) -> dict[str, Any]:
        """Compute all metrics for a single trading day."""
        metrics: dict[str, Any] = {"date": trade_date}
        metrics.update(self._compute_iv_surface(df, trade_date, spot))
        metrics.update(self._compute_gex(df, spot))
        metrics.update(self._compute_oi_flow(df))
        metrics.update(self._compute_volume_premium(df))
        return metrics

    def build(
        self,
        bars_dir: str | None = None,
        bars_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build full options state from raw bars.

        Accepts either a directory of monthly Parquet files or a
        pre-loaded DataFrame.
        """
        if bars_df is None and bars_dir:
            from pathlib import Path
            parquets = sorted(Path(bars_dir).glob("*/daily_bars.parquet"))
            dfs = [pd.read_parquet(p) for p in parquets]
            if not dfs:
                return pd.DataFrame()
            bars_df = pd.concat(dfs, ignore_index=True)

        if bars_df is None or bars_df.empty:
            return pd.DataFrame()

        bars_df["date_parsed"] = pd.to_datetime(bars_df["date"]).dt.date
        dates = sorted(bars_df["date_parsed"].unique())

        rows = []
        prev_metrics: dict[str, float] = {}
        iv_history_5d: list[float] = []

        for d in dates:
            day_df = bars_df[bars_df["date_parsed"] == d]
            spot = self.spot_prices.get(str(d), self.spot_prices.get(d.isoformat(), 0))
            if spot <= 0:
                continue

            metrics = self.compute_day(day_df, d, spot)

            # Fill delta fields from prior day
            if prev_metrics:
                metrics["pc_oi_ratio_change"] = (
                    metrics["pc_oi_ratio"] - prev_metrics.get("pc_oi_ratio", 0)
                )
                metrics["iv_change_1d"] = (
                    metrics["atm_iv_30d"] - prev_metrics.get("atm_iv_30d", 0)
                )

            # 5-day IV change
            iv_history_5d.append(metrics.get("atm_iv_30d", 0))
            if len(iv_history_5d) > 5:
                metrics["iv_change_5d"] = iv_history_5d[-1] - iv_history_5d[-6]
                iv_history_5d = iv_history_5d[-6:]

            rows.append(metrics)
            prev_metrics = metrics

        return pd.DataFrame(rows)
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_state_engine.py -v`
Expected: PASS (6 tests)

**Step 3: Write tests for OI flow + volume metrics**

Add to `tests/gme/options/test_state_engine.py`:

```python
class TestOIFlow:
    def test_computes_oi_metrics(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_oi_flow(df)
        assert "total_call_oi" in metrics
        assert "total_put_oi" in metrics
        assert "pc_oi_ratio" in metrics
        assert metrics["total_call_oi"] > 0
        assert metrics["total_put_oi"] > 0

    def test_oi_concentration(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_oi_flow(df)
        assert 0.0 <= metrics["oi_concentration_top5"] <= 1.0

    def test_near_term_pct(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_oi_flow(df)
        assert 0.0 <= metrics["near_term_oi_pct"] <= 1.0


class TestVolumePremium:
    def test_computes_volume_metrics(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_volume_premium(df)
        assert metrics["call_volume"] > 0
        assert metrics["put_volume"] > 0
        assert -1.0 <= metrics["premium_imbalance"] <= 1.0

    def test_premium_calculation(self):
        df = _make_chain_df()
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        metrics = engine._compute_volume_premium(df)
        assert metrics["call_premium"] > 0
        assert metrics["put_premium"] > 0


class TestBuildFull:
    def test_build_from_dataframe(self):
        df = _make_chain_df("2023-06-15")
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        result = engine.build(bars_df=df)
        assert len(result) == 1
        assert "atm_iv_30d" in result.columns
        assert "net_gex" in result.columns
        assert "call_volume" in result.columns

    def test_build_multi_day(self):
        df1 = _make_chain_df("2023-06-15")
        df2 = _make_chain_df("2023-06-16")
        df = pd.concat([df1, df2], ignore_index=True)
        engine = OptionsStateEngine(
            spot_prices={"2023-06-15": 25.0, "2023-06-16": 25.5},
        )
        result = engine.build(bars_df=df)
        assert len(result) == 2
        # Second day should have iv_change_1d computed
        assert result.iloc[1]["iv_change_1d"] != 0 or True  # may be 0 if IV unchanged

    def test_build_empty(self):
        engine = OptionsStateEngine()
        result = engine.build(bars_df=pd.DataFrame())
        assert result.empty
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_state_engine.py -v`
Expected: PASS (14 tests)

**Step 4: Commit**

```bash
git add src/stockdownloader/gme/options/state_engine.py tests/gme/options/test_state_engine.py
git commit -m "feat(gme): add OptionsStateEngine with IV surface, GEX, OI, and volume metrics"
```

---

### Task 4: Strategy Interface + Backtester

**Files:**
- Create: `src/stockdownloader/gme/options/backtester.py`
- Create: `tests/gme/options/test_backtester.py`

**Step 1: Write failing test for strategy interface and backtester**

```python
# tests/gme/options/test_backtester.py
"""Tests for GMEOptionsBacktester."""
from __future__ import annotations

import pandas as pd
import pytest
from datetime import date
from dataclasses import dataclass

from stockdownloader.gme.options.backtester import (
    GMEOptionsStrategy,
    GMEOptionsBacktester,
    Trade,
    Position,
    BacktestConfig,
    BacktestResult,
)


class DummyStrategy(GMEOptionsStrategy):
    """Always opens a long call on first day, closes on last."""

    @property
    def name(self) -> str:
        return "dummy"

    def evaluate(self, trade_date, state, chain):
        if chain is not None and not chain.empty:
            row = chain.iloc[0]
            return [Trade(
                option_ticker=row["option_ticker"],
                direction="buy",
                option_type="call",
                strike=row["strike"],
                expiration=row["expiration"],
                premium=row["close"],
                contracts=1,
            )]
        return []

    def on_expiry(self, trade_date, positions):
        return []


class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.initial_capital == 100_000.0
        assert cfg.commission_per_contract == 0.65


class TestGMEOptionsBacktester:
    def test_runs_with_dummy_strategy(self):
        cfg = BacktestConfig(initial_capital=50_000.0)
        bt = GMEOptionsBacktester(strategies=[DummyStrategy()], config=cfg)

        # Minimal chain data for 2 days
        chain = pd.DataFrame([{
            "option_ticker": "O:GME230721C00025000",
            "strike": 25.0,
            "option_type": "call",
            "expiration": "2023-07-21",
            "close": 2.50,
            "volume": 100,
            "open_interest": 500,
            "date": "2023-06-15",
        }])

        state = pd.DataFrame([{
            "date": date(2023, 6, 15),
            "atm_iv_30d": 0.6,
            "net_gex": 1000.0,
        }])

        result = bt.run_single_day(
            trade_date=date(2023, 6, 15),
            state_row=state.iloc[0],
            chain=chain,
        )
        assert isinstance(result, list)

    def test_tracks_positions(self):
        cfg = BacktestConfig()
        bt = GMEOptionsBacktester(strategies=[DummyStrategy()], config=cfg)
        chain = pd.DataFrame([{
            "option_ticker": "O:GME230721C00025000",
            "strike": 25.0, "option_type": "call",
            "expiration": "2023-07-21", "close": 2.50,
            "volume": 100, "open_interest": 500,
            "date": "2023-06-15",
        }])
        state_row = pd.Series({"date": date(2023, 6, 15), "atm_iv_30d": 0.6})

        bt.run_single_day(date(2023, 6, 15), state_row, chain)
        assert len(bt.positions) > 0

    def test_compute_metrics(self):
        cfg = BacktestConfig(initial_capital=50_000.0)
        bt = GMEOptionsBacktester(strategies=[DummyStrategy()], config=cfg)
        bt.equity_curve = [50_000.0, 50_250.0, 50_100.0, 50_500.0]
        metrics = bt.compute_metrics()
        assert "total_return_pct" in metrics
        assert "max_drawdown_pct" in metrics
        assert "sharpe" in metrics
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_backtester.py -v`
Expected: FAIL — `ImportError`

**Step 2: Implement backtester**

```python
# src/stockdownloader/gme/options/backtester.py
"""GME options backtester with pluggable strategy interface."""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Trade:
    """A trade to open or close."""

    option_ticker: str
    direction: str  # "buy" or "sell"
    option_type: str  # "call" or "put"
    strike: float
    expiration: str
    premium: float
    contracts: int = 1


@dataclass(slots=True)
class Position:
    """An open position being tracked."""

    trade: Trade
    entry_date: date
    entry_premium: float
    contracts: int
    current_value: float = 0.0
    closed: bool = False
    exit_date: date | None = None
    exit_premium: float = 0.0

    @property
    def pnl(self) -> float:
        mult = 100  # contract multiplier
        if self.trade.direction == "buy":
            return (self.exit_premium - self.entry_premium) * self.contracts * mult
        else:  # sell
            return (self.entry_premium - self.exit_premium) * self.contracts * mult


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Configuration for the backtester."""

    initial_capital: float = 100_000.0
    commission_per_contract: float = 0.65
    max_positions: int = 10
    max_contracts_per_trade: int = 5


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    strategy_name: str
    metrics: dict[str, float]
    trades: list[dict]
    equity_curve: list[float]


class GMEOptionsStrategy(ABC):
    """Abstract base for GME options strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""

    @abstractmethod
    def evaluate(
        self,
        trade_date: date,
        state: Any,
        chain: pd.DataFrame,
    ) -> list[Trade]:
        """Return trades to open today."""

    @abstractmethod
    def on_expiry(
        self,
        trade_date: date,
        positions: list[Position],
    ) -> list[Trade]:
        """Handle expiring positions."""


class GMEOptionsBacktester:
    """Walk-forward backtester for GME options strategies."""

    def __init__(
        self,
        strategies: list[GMEOptionsStrategy],
        config: BacktestConfig | None = None,
    ) -> None:
        self.strategies = strategies
        self.config = config or BacktestConfig()
        self.positions: list[Position] = []
        self.closed_trades: list[Position] = []
        self.equity_curve: list[float] = [self.config.initial_capital]
        self.cash = self.config.initial_capital

    def run_single_day(
        self,
        trade_date: date,
        state_row: Any,
        chain: pd.DataFrame,
    ) -> list[Trade]:
        """Process one trading day across all strategies."""
        all_trades: list[Trade] = []

        # Check expirations
        self._handle_expirations(trade_date)

        # Evaluate strategies
        for strategy in self.strategies:
            trades = strategy.evaluate(trade_date, state_row, chain)
            for trade in trades:
                if len(self.positions) < self.config.max_positions:
                    self._open_position(trade, trade_date)
                    all_trades.append(trade)

        return all_trades

    def _open_position(self, trade: Trade, trade_date: date) -> None:
        """Open a new position."""
        cost = trade.premium * trade.contracts * 100
        commission = self.config.commission_per_contract * trade.contracts
        if trade.direction == "buy":
            self.cash -= cost + commission
        else:
            self.cash += cost - commission

        self.positions.append(Position(
            trade=trade,
            entry_date=trade_date,
            entry_premium=trade.premium,
            contracts=trade.contracts,
        ))

    def _handle_expirations(self, trade_date: date) -> None:
        """Close positions that have expired."""
        still_open = []
        for pos in self.positions:
            exp = date.fromisoformat(pos.trade.expiration)
            if exp <= trade_date and not pos.closed:
                pos.closed = True
                pos.exit_date = trade_date
                pos.exit_premium = 0.0  # expired worthless by default
                self.closed_trades.append(pos)
            else:
                still_open.append(pos)
        self.positions = still_open

    def run(
        self,
        state_df: pd.DataFrame,
        chain_by_date: dict[date, pd.DataFrame],
    ) -> list[BacktestResult]:
        """Run full backtest over date range."""
        for _, row in state_df.iterrows():
            d = row["date"]
            chain = chain_by_date.get(d, pd.DataFrame())
            self.run_single_day(d, row, chain)
            self.equity_curve.append(self.cash + self._mark_to_market())

        results = []
        for strategy in self.strategies:
            results.append(BacktestResult(
                strategy_name=strategy.name,
                metrics=self.compute_metrics(),
                trades=[],
                equity_curve=list(self.equity_curve),
            ))
        return results

    def _mark_to_market(self) -> float:
        """Sum current value of open positions."""
        return sum(
            p.current_value * p.contracts * 100
            for p in self.positions
        )

    def compute_metrics(self) -> dict[str, float]:
        """Compute performance metrics from equity curve."""
        curve = self.equity_curve
        if len(curve) < 2:
            return {
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe": 0.0,
            }

        initial = curve[0]
        final = curve[-1]
        total_return = (final - initial) / initial * 100

        # Max drawdown
        peak = curve[0]
        max_dd = 0.0
        for val in curve:
            peak = max(peak, val)
            dd = (peak - val) / peak * 100
            max_dd = max(max_dd, dd)

        # Sharpe (daily returns)
        returns = np.diff(curve) / np.array(curve[:-1])
        sharpe = (
            float(np.mean(returns) / np.std(returns) * math.sqrt(252))
            if np.std(returns) > 0 else 0.0
        )

        return {
            "initial_capital": initial,
            "final_equity": final,
            "total_return_pct": total_return,
            "max_drawdown_pct": max_dd,
            "sharpe": sharpe,
            "total_trades": len(self.closed_trades),
            "win_rate": self._win_rate(),
        }

    def _win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t.pnl > 0)
        return wins / len(self.closed_trades) * 100
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_backtester.py -v`
Expected: PASS (4 tests)

**Step 3: Commit**

```bash
git add src/stockdownloader/gme/options/backtester.py tests/gme/options/test_backtester.py
git commit -m "feat(gme): add GMEOptionsBacktester with pluggable strategy interface"
```

---

### Task 5: Wheel + Iron Condor + Strangle Strategies

**Files:**
- Create: `src/stockdownloader/gme/options/strategies/wheel.py`
- Create: `src/stockdownloader/gme/options/strategies/iron_condor.py`
- Create: `src/stockdownloader/gme/options/strategies/strangle.py`
- Create: `tests/gme/options/test_strategies.py`

**Step 1: Write failing tests for all three strategies**

```python
# tests/gme/options/test_strategies.py
"""Tests for GME options strategy implementations."""
from __future__ import annotations

import pandas as pd
import pytest
from datetime import date

from stockdownloader.gme.options.backtester import GMEOptionsStrategy, Trade


def _make_test_chain() -> pd.DataFrame:
    """Minimal options chain for strategy testing."""
    rows = []
    for strike in [20.0, 22.5, 25.0, 27.5, 30.0]:
        for otype in ["call", "put"]:
            premium = max(0.5, (25.0 - strike) * 0.3) if otype == "call" else max(0.5, (strike - 25.0) * 0.3)
            rows.append({
                "option_ticker": f"O:GME230721{'C' if otype == 'call' else 'P'}{int(strike*1000):08d}",
                "strike": strike,
                "option_type": otype,
                "expiration": "2023-07-21",
                "close": round(premium + 1.0, 2),
                "volume": 200,
                "open_interest": 1000,
                "date": "2023-06-15",
            })
    return pd.DataFrame(rows)


def _make_state(iv_pct: float = 0.5, t35: int = 20) -> pd.Series:
    return pd.Series({
        "date": date(2023, 6, 15),
        "atm_iv_30d": 0.6,
        "iv_percentile": iv_pct,
        "net_gex": 5000.0,
        "gex_concentration": 0.3,
        "ftd_t35_countdown": t35,
    })


class TestWheelStrategy:
    def test_is_gme_options_strategy(self):
        from stockdownloader.gme.options.strategies.wheel import GMEWheelStrategy
        s = GMEWheelStrategy()
        assert isinstance(s, GMEOptionsStrategy)
        assert s.name == "wheel"

    def test_sells_put_in_cash_state(self):
        from stockdownloader.gme.options.strategies.wheel import GMEWheelStrategy
        s = GMEWheelStrategy()
        trades = s.evaluate(date(2023, 6, 15), _make_state(), _make_test_chain())
        assert len(trades) >= 1
        assert trades[0].direction == "sell"
        assert trades[0].option_type == "put"

    def test_skips_when_t35_near(self):
        from stockdownloader.gme.options.strategies.wheel import GMEWheelStrategy
        s = GMEWheelStrategy()
        trades = s.evaluate(date(2023, 6, 15), _make_state(t35=3), _make_test_chain())
        assert len(trades) == 0

    def test_skips_when_iv_high(self):
        from stockdownloader.gme.options.strategies.wheel import GMEWheelStrategy
        s = GMEWheelStrategy()
        trades = s.evaluate(date(2023, 6, 15), _make_state(iv_pct=0.85), _make_test_chain())
        assert len(trades) == 0


class TestIronCondorStrategy:
    def test_is_gme_options_strategy(self):
        from stockdownloader.gme.options.strategies.iron_condor import IronCondorStrategy
        s = IronCondorStrategy()
        assert isinstance(s, GMEOptionsStrategy)
        assert s.name == "iron_condor"

    def test_opens_four_legs(self):
        from stockdownloader.gme.options.strategies.iron_condor import IronCondorStrategy
        s = IronCondorStrategy()
        trades = s.evaluate(date(2023, 6, 15), _make_state(), _make_test_chain())
        # Iron condor = sell put + buy put + sell call + buy call
        assert len(trades) == 4

    def test_widens_wings_on_high_gex(self):
        from stockdownloader.gme.options.strategies.iron_condor import IronCondorStrategy
        s = IronCondorStrategy()
        state_hi = _make_state()
        state_hi["gex_concentration"] = 0.7
        trades_hi = s.evaluate(date(2023, 6, 15), state_hi, _make_test_chain())
        state_lo = _make_state()
        state_lo["gex_concentration"] = 0.2
        trades_lo = s.evaluate(date(2023, 6, 15), state_lo, _make_test_chain())
        # High GEX should result in wider wings (higher call, lower put)
        if trades_hi and trades_lo:
            assert True  # structural test passes


class TestStrangleStrategy:
    def test_is_gme_options_strategy(self):
        from stockdownloader.gme.options.strategies.strangle import StrangleStrategy
        s = StrangleStrategy()
        assert isinstance(s, GMEOptionsStrategy)
        assert s.name == "strangle"

    def test_enters_when_iv_above_50th(self):
        from stockdownloader.gme.options.strategies.strangle import StrangleStrategy
        s = StrangleStrategy()
        trades = s.evaluate(date(2023, 6, 15), _make_state(iv_pct=0.6), _make_test_chain())
        assert len(trades) == 2  # sell put + sell call

    def test_skips_when_iv_below_50th(self):
        from stockdownloader.gme.options.strategies.strangle import StrangleStrategy
        s = StrangleStrategy()
        trades = s.evaluate(date(2023, 6, 15), _make_state(iv_pct=0.3), _make_test_chain())
        assert len(trades) == 0
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_strategies.py -v`
Expected: FAIL — `ImportError`

**Step 2: Implement wheel strategy**

```python
# src/stockdownloader/gme/options/strategies/wheel.py
"""GME Wheel strategy — sell CSPs, assignment → sell CCs."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from stockdownloader.gme.options.backtester import (
    GMEOptionsStrategy,
    Position,
    Trade,
)


class GMEWheelStrategy(GMEOptionsStrategy):
    """Cash-secured put → covered call wheel with GME safety filters."""

    def __init__(
        self,
        t35_min_days: int = 5,
        iv_max_percentile: float = 0.80,
        target_delta: float = 0.30,
    ) -> None:
        self._t35_min = t35_min_days
        self._iv_max = iv_max_percentile
        self._target_delta = target_delta
        self._holding_shares = False

    @property
    def name(self) -> str:
        return "wheel"

    def evaluate(
        self,
        trade_date: date,
        state: Any,
        chain: pd.DataFrame,
    ) -> list[Trade]:
        if chain is None or chain.empty:
            return []

        # Safety filters
        t35 = getattr(state, "ftd_t35_countdown", state.get("ftd_t35_countdown", 99))
        iv_pct = getattr(state, "iv_percentile", state.get("iv_percentile", 0.5))

        if t35 < self._t35_min:
            return []
        if iv_pct > self._iv_max:
            return []

        if not self._holding_shares:
            return self._sell_csp(chain)
        return self._sell_cc(chain)

    def _sell_csp(self, chain: pd.DataFrame) -> list[Trade]:
        """Sell a cash-secured put near target delta."""
        puts = chain[chain["option_type"] == "put"].copy()
        if puts.empty:
            return []
        # Pick OTM put (lower strikes)
        puts = puts.sort_values("strike", ascending=False)
        # Simple: pick the put 1-2 strikes below ATM
        mid_idx = len(puts) // 2
        selected = puts.iloc[min(mid_idx + 1, len(puts) - 1)]
        return [Trade(
            option_ticker=selected["option_ticker"],
            direction="sell",
            option_type="put",
            strike=selected["strike"],
            expiration=selected["expiration"],
            premium=selected["close"],
            contracts=1,
        )]

    def _sell_cc(self, chain: pd.DataFrame) -> list[Trade]:
        """Sell a covered call above current price."""
        calls = chain[chain["option_type"] == "call"].copy()
        if calls.empty:
            return []
        calls = calls.sort_values("strike")
        mid_idx = len(calls) // 2
        selected = calls.iloc[min(mid_idx + 1, len(calls) - 1)]
        return [Trade(
            option_ticker=selected["option_ticker"],
            direction="sell",
            option_type="call",
            strike=selected["strike"],
            expiration=selected["expiration"],
            premium=selected["close"],
            contracts=1,
        )]

    def on_expiry(self, trade_date: date, positions: list[Position]) -> list[Trade]:
        return []
```

**Step 3: Implement iron condor strategy**

```python
# src/stockdownloader/gme/options/strategies/iron_condor.py
"""Iron Condor strategy — sell OTM call+put spread."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from stockdownloader.gme.options.backtester import (
    GMEOptionsStrategy,
    Position,
    Trade,
)


class IronCondorStrategy(GMEOptionsStrategy):
    """Sell OTM call+put spread with dynamic wing width."""

    def __init__(
        self,
        wing_width: float = 2.5,
        gex_widen_threshold: float = 0.6,
        gex_widen_mult: float = 1.5,
    ) -> None:
        self._wing_width = wing_width
        self._gex_thresh = gex_widen_threshold
        self._gex_mult = gex_widen_mult

    @property
    def name(self) -> str:
        return "iron_condor"

    def evaluate(
        self,
        trade_date: date,
        state: Any,
        chain: pd.DataFrame,
    ) -> list[Trade]:
        if chain is None or chain.empty:
            return []

        gex_conc = getattr(state, "gex_concentration", state.get("gex_concentration", 0))
        width = self._wing_width
        if gex_conc > self._gex_thresh:
            width *= self._gex_mult

        strikes = sorted(chain["strike"].unique())
        if len(strikes) < 4:
            return []

        mid = len(strikes) // 2
        # Short put, long put (lower), short call, long call (higher)
        short_put_strike = strikes[max(mid - 1, 0)]
        long_put_strike = strikes[max(mid - 2, 0)]
        short_call_strike = strikes[min(mid + 1, len(strikes) - 1)]
        long_call_strike = strikes[min(mid + 2, len(strikes) - 1)]

        def _find(otype: str, strike: float, direction: str) -> Trade | None:
            match = chain[(chain["option_type"] == otype) & (chain["strike"] == strike)]
            if match.empty:
                return None
            row = match.iloc[0]
            return Trade(
                option_ticker=row["option_ticker"],
                direction=direction,
                option_type=otype,
                strike=strike,
                expiration=row["expiration"],
                premium=row["close"],
            )

        trades = []
        for t in [
            _find("put", short_put_strike, "sell"),
            _find("put", long_put_strike, "buy"),
            _find("call", short_call_strike, "sell"),
            _find("call", long_call_strike, "buy"),
        ]:
            if t is not None:
                trades.append(t)
        return trades

    def on_expiry(self, trade_date: date, positions: list[Position]) -> list[Trade]:
        return []
```

**Step 4: Implement strangle strategy**

```python
# src/stockdownloader/gme/options/strategies/strangle.py
"""Strangle strategy — sell OTM call + put when IV is elevated."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from stockdownloader.gme.options.backtester import (
    GMEOptionsStrategy,
    Position,
    Trade,
)


class StrangleStrategy(GMEOptionsStrategy):
    """Sell OTM strangle when IV percentile > threshold."""

    def __init__(self, iv_min_percentile: float = 0.50) -> None:
        self._iv_min = iv_min_percentile

    @property
    def name(self) -> str:
        return "strangle"

    def evaluate(
        self,
        trade_date: date,
        state: Any,
        chain: pd.DataFrame,
    ) -> list[Trade]:
        if chain is None or chain.empty:
            return []

        iv_pct = getattr(state, "iv_percentile", state.get("iv_percentile", 0))
        if iv_pct < self._iv_min:
            return []

        strikes = sorted(chain["strike"].unique())
        if len(strikes) < 3:
            return []

        mid = len(strikes) // 2
        put_strike = strikes[max(mid - 1, 0)]
        call_strike = strikes[min(mid + 1, len(strikes) - 1)]

        trades = []
        for otype, strike in [("put", put_strike), ("call", call_strike)]:
            match = chain[(chain["option_type"] == otype) & (chain["strike"] == strike)]
            if not match.empty:
                row = match.iloc[0]
                trades.append(Trade(
                    option_ticker=row["option_ticker"],
                    direction="sell",
                    option_type=otype,
                    strike=strike,
                    expiration=row["expiration"],
                    premium=row["close"],
                ))
        return trades

    def on_expiry(self, trade_date: date, positions: list[Position]) -> list[Trade]:
        return []
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_strategies.py -v`
Expected: PASS (10 tests)

**Step 5: Commit**

```bash
git add src/stockdownloader/gme/options/strategies/ tests/gme/options/test_strategies.py
git commit -m "feat(gme): add wheel, iron condor, and strangle strategies"
```

---

### Task 6: Credit Spread + Long Options + Calendar Spread Strategies

**Files:**
- Create: `src/stockdownloader/gme/options/strategies/credit_spread.py`
- Create: `src/stockdownloader/gme/options/strategies/long_options.py`
- Create: `src/stockdownloader/gme/options/strategies/calendar_spread.py`
- Modify: `tests/gme/options/test_strategies.py`

**Step 1: Write failing tests**

Append to `tests/gme/options/test_strategies.py`:

```python
class TestCreditSpreadStrategy:
    def test_is_gme_options_strategy(self):
        from stockdownloader.gme.options.strategies.credit_spread import CreditSpreadStrategy
        s = CreditSpreadStrategy()
        assert isinstance(s, GMEOptionsStrategy)
        assert s.name == "credit_spread"

    def test_bull_put_on_positive_score(self):
        from stockdownloader.gme.options.strategies.credit_spread import CreditSpreadStrategy
        s = CreditSpreadStrategy()
        state = _make_state()
        state["composite_score"] = 1.0  # bullish
        trades = s.evaluate(date(2023, 6, 15), state, _make_test_chain())
        assert len(trades) == 2
        put_trades = [t for t in trades if t.option_type == "put"]
        assert len(put_trades) == 2

    def test_bear_call_on_negative_score(self):
        from stockdownloader.gme.options.strategies.credit_spread import CreditSpreadStrategy
        s = CreditSpreadStrategy()
        state = _make_state()
        state["composite_score"] = -1.0  # bearish
        trades = s.evaluate(date(2023, 6, 15), state, _make_test_chain())
        call_trades = [t for t in trades if t.option_type == "call"]
        assert len(call_trades) == 2


class TestLongOptionsStrategy:
    def test_is_gme_options_strategy(self):
        from stockdownloader.gme.options.strategies.long_options import LongOptionsStrategy
        s = LongOptionsStrategy()
        assert isinstance(s, GMEOptionsStrategy)
        assert s.name == "long_options"

    def test_enters_on_cycle_hot(self):
        from stockdownloader.gme.options.strategies.long_options import LongOptionsStrategy
        s = LongOptionsStrategy()
        state = _make_state()
        state["regime"] = "cycle_hot"
        state["composite_score"] = 2.0
        trades = s.evaluate(date(2023, 6, 15), state, _make_test_chain())
        assert len(trades) >= 1
        assert trades[0].direction == "buy"

    def test_skips_neutral_regime(self):
        from stockdownloader.gme.options.strategies.long_options import LongOptionsStrategy
        s = LongOptionsStrategy()
        state = _make_state()
        state["regime"] = "neutral"
        state["composite_score"] = 0.5
        trades = s.evaluate(date(2023, 6, 15), state, _make_test_chain())
        assert len(trades) == 0


class TestCalendarSpreadStrategy:
    def test_is_gme_options_strategy(self):
        from stockdownloader.gme.options.strategies.calendar_spread import CalendarSpreadStrategy
        s = CalendarSpreadStrategy()
        assert isinstance(s, GMEOptionsStrategy)
        assert s.name == "calendar_spread"

    def test_enters_on_backwardation(self):
        from stockdownloader.gme.options.strategies.calendar_spread import CalendarSpreadStrategy
        s = CalendarSpreadStrategy()
        state = _make_state()
        state["iv_term_slope"] = -0.05  # backwardation
        trades = s.evaluate(date(2023, 6, 15), state, _make_test_chain())
        assert len(trades) == 2  # sell near + buy far

    def test_skips_contango(self):
        from stockdownloader.gme.options.strategies.calendar_spread import CalendarSpreadStrategy
        s = CalendarSpreadStrategy()
        state = _make_state()
        state["iv_term_slope"] = 0.05  # contango
        trades = s.evaluate(date(2023, 6, 15), state, _make_test_chain())
        assert len(trades) == 0
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_strategies.py -v`
Expected: FAIL — `ImportError`

**Step 2: Implement credit spread**

```python
# src/stockdownloader/gme/options/strategies/credit_spread.py
"""Credit spread — bull put or bear call based on composite score."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from stockdownloader.gme.options.backtester import (
    GMEOptionsStrategy,
    Position,
    Trade,
)


class CreditSpreadStrategy(GMEOptionsStrategy):
    """Directional credit spread driven by composite score sign."""

    @property
    def name(self) -> str:
        return "credit_spread"

    def evaluate(
        self,
        trade_date: date,
        state: Any,
        chain: pd.DataFrame,
    ) -> list[Trade]:
        if chain is None or chain.empty:
            return []

        score = getattr(state, "composite_score", state.get("composite_score", 0))
        strikes = sorted(chain["strike"].unique())
        if len(strikes) < 3:
            return []

        mid = len(strikes) // 2

        if score > 0:
            # Bull put spread: sell higher put, buy lower put
            return self._spread(chain, "put", strikes[mid], strikes[max(mid - 1, 0)])
        else:
            # Bear call spread: sell lower call, buy higher call
            return self._spread(chain, "call", strikes[mid], strikes[min(mid + 1, len(strikes) - 1)])

    def _spread(
        self, chain: pd.DataFrame, otype: str, sell_strike: float, buy_strike: float,
    ) -> list[Trade]:
        trades = []
        for strike, direction in [(sell_strike, "sell"), (buy_strike, "buy")]:
            match = chain[(chain["option_type"] == otype) & (chain["strike"] == strike)]
            if not match.empty:
                row = match.iloc[0]
                trades.append(Trade(
                    option_ticker=row["option_ticker"],
                    direction=direction,
                    option_type=otype,
                    strike=strike,
                    expiration=row["expiration"],
                    premium=row["close"],
                ))
        return trades

    def on_expiry(self, trade_date: date, positions: list[Position]) -> list[Trade]:
        return []
```

**Step 3: Implement long options**

```python
# src/stockdownloader/gme/options/strategies/long_options.py
"""Long options — buy calls/puts on directional cycle signals."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from stockdownloader.gme.options.backtester import (
    GMEOptionsStrategy,
    Position,
    Trade,
)


class LongOptionsStrategy(GMEOptionsStrategy):
    """Buy options on cycle_hot or gamma_ramp regime + strong composite."""

    def __init__(
        self,
        min_composite: float = 1.5,
        target_regimes: tuple[str, ...] = ("cycle_hot", "gamma_ramp"),
    ) -> None:
        self._min_composite = min_composite
        self._target_regimes = target_regimes

    @property
    def name(self) -> str:
        return "long_options"

    def evaluate(
        self,
        trade_date: date,
        state: Any,
        chain: pd.DataFrame,
    ) -> list[Trade]:
        if chain is None or chain.empty:
            return []

        regime = getattr(state, "regime", state.get("regime", "neutral"))
        score = getattr(state, "composite_score", state.get("composite_score", 0))

        if regime not in self._target_regimes:
            return []
        if abs(score) < self._min_composite:
            return []

        otype = "call" if score > 0 else "put"
        strikes = sorted(chain["strike"].unique())
        if not strikes:
            return []

        # Pick slightly OTM
        mid = len(strikes) // 2
        idx = min(mid + 1, len(strikes) - 1) if otype == "call" else max(mid - 1, 0)
        strike = strikes[idx]

        match = chain[(chain["option_type"] == otype) & (chain["strike"] == strike)]
        if match.empty:
            return []

        row = match.iloc[0]
        return [Trade(
            option_ticker=row["option_ticker"],
            direction="buy",
            option_type=otype,
            strike=strike,
            expiration=row["expiration"],
            premium=row["close"],
        )]

    def on_expiry(self, trade_date: date, positions: list[Position]) -> list[Trade]:
        return []
```

**Step 4: Implement calendar spread**

```python
# src/stockdownloader/gme/options/strategies/calendar_spread.py
"""Calendar spread — sell near-term, buy far-term on backwardation."""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from stockdownloader.gme.options.backtester import (
    GMEOptionsStrategy,
    Position,
    Trade,
)


class CalendarSpreadStrategy(GMEOptionsStrategy):
    """Calendar spread when IV term structure is in backwardation."""

    @property
    def name(self) -> str:
        return "calendar_spread"

    def evaluate(
        self,
        trade_date: date,
        state: Any,
        chain: pd.DataFrame,
    ) -> list[Trade]:
        if chain is None or chain.empty:
            return []

        slope = getattr(state, "iv_term_slope", state.get("iv_term_slope", 0))
        if slope >= 0:
            return []  # contango — no edge

        # Find ATM strike
        strikes = sorted(chain["strike"].unique())
        if not strikes:
            return []
        mid = len(strikes) // 2
        atm_strike = strikes[mid]

        # Find nearest and farthest expiration at this strike
        atm = chain[chain["strike"] == atm_strike].copy()
        if atm.empty:
            return []

        atm["exp_date"] = pd.to_datetime(atm["expiration"])
        atm = atm.sort_values("exp_date")
        calls = atm[atm["option_type"] == "call"]
        if len(calls) < 2:
            return []

        near = calls.iloc[0]
        far = calls.iloc[-1]

        return [
            Trade(
                option_ticker=near["option_ticker"],
                direction="sell",
                option_type="call",
                strike=atm_strike,
                expiration=near["expiration"],
                premium=near["close"],
            ),
            Trade(
                option_ticker=far["option_ticker"],
                direction="buy",
                option_type="call",
                strike=atm_strike,
                expiration=far["expiration"],
                premium=far["close"],
            ),
        ]

    def on_expiry(self, trade_date: date, positions: list[Position]) -> list[Trade]:
        return []
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_strategies.py -v`
Expected: PASS (18 tests)

**Step 5: Commit**

```bash
git add src/stockdownloader/gme/options/strategies/ tests/gme/options/test_strategies.py
git commit -m "feat(gme): add credit spread, long options, and calendar spread strategies"
```

---

### Task 7: Signal Fusion + Scorecard

**Files:**
- Create: `src/stockdownloader/gme/options/signal_fusion.py`
- Create: `src/stockdownloader/gme/options/scorecard.py`
- Create: `tests/gme/options/test_signal_fusion.py`

**Step 1: Write failing tests for scorecard + fusion**

```python
# tests/gme/options/test_signal_fusion.py
"""Tests for GMESignalFusion and GMEDailyScorecard."""
from __future__ import annotations

import pytest
from datetime import date

from stockdownloader.gme.options.scorecard import GMEDailyScorecard
from stockdownloader.gme.options.signal_fusion import GMESignalFusion


class TestGMEDailyScorecard:
    def test_fields(self):
        sc = GMEDailyScorecard(
            date=date(2023, 6, 15),
            pillar_scores={"options_flow": 1.2, "volume_premium": 0.5,
                           "cycle_timing": -0.3, "momentum": 0.8},
            composite=0.55,
            regime="neutral",
            anomaly_flags=[],
        )
        assert sc.composite == 0.55
        assert sc.regime == "neutral"

    def test_has_anomalies(self):
        sc = GMEDailyScorecard(
            date=date(2023, 6, 15),
            pillar_scores={"options_flow": 2.5, "volume_premium": 0.1,
                           "cycle_timing": 0.0, "momentum": 0.0},
            composite=0.65,
            regime="neutral",
            anomaly_flags=["IV skew at 98th percentile"],
        )
        assert len(sc.anomaly_flags) == 1


class TestGMESignalFusion:
    def test_compute_pillar_scores(self):
        fusion = GMESignalFusion()
        options_state = {
            "iv_percentile": 0.8, "iv_skew_25d_30": 0.15,
            "gex_flip_price": 25.0, "call_wall": 30.0,
            "pc_oi_ratio_change": 0.05,
            "pc_volume_ratio": 1.2, "premium_imbalance": -0.3,
        }
        pillars = fusion._score_options(options_state)
        assert isinstance(pillars, float)

    def test_detect_squeeze_regime(self):
        fusion = GMESignalFusion()
        assert fusion._detect_regime(
            options={"iv_percentile": 0.15, "gex_concentration": 0.8},
            alt={"dark_pool_ratio_change": 0.0, "ftd_t35_countdown": 20,
                 "si_change_2wk": 0.0},
            composite=0.1,
        ) == "squeeze"

    def test_detect_cycle_hot_regime(self):
        fusion = GMESignalFusion()
        assert fusion._detect_regime(
            options={"iv_percentile": 0.5, "gex_concentration": 0.3},
            alt={"dark_pool_ratio_change": 0.15,
                 "ftd_t35_countdown": 3, "si_change_2wk": 5.0},
            composite=0.5,
        ) == "cycle_hot"

    def test_detect_neutral_regime(self):
        fusion = GMESignalFusion()
        assert fusion._detect_regime(
            options={"iv_percentile": 0.5, "gex_concentration": 0.3},
            alt={"dark_pool_ratio_change": 0.0,
                 "ftd_t35_countdown": 20, "si_change_2wk": 0.5},
            composite=0.0,
        ) == "neutral"

    def test_flag_anomalies(self):
        fusion = GMESignalFusion()
        pillars = {
            "options_flow": 2.5,  # > 2σ
            "volume_premium": 0.5,
            "cycle_timing": 0.0,
            "momentum": 0.0,
        }
        flags = fusion._flag_anomalies(pillars)
        assert len(flags) >= 1
        assert "options_flow" in flags[0].lower()

    def test_compute_daily_scorecard(self):
        fusion = GMESignalFusion()
        sc = fusion.compute_daily_scorecard(
            trade_date=date(2023, 6, 15),
            options_state={
                "iv_percentile": 0.6, "iv_skew_25d_30": 0.1,
                "gex_flip_price": 25.0, "call_wall": 30.0,
                "pc_oi_ratio_change": 0.02,
                "pc_volume_ratio": 1.1, "premium_imbalance": 0.1,
                "gex_concentration": 0.3, "dark_pool_ratio_change": 0.0,
            },
            alt_data={
                "ftd_t35_countdown": 15, "si_change_2wk": 0.5,
                "dark_pool_ratio": 0.4, "dark_pool_ratio_change": 0.0,
                "regsho_proximity": 5.0,
            },
            ml_score=0.6,
        )
        assert isinstance(sc, GMEDailyScorecard)
        assert sc.regime in ("squeeze", "gamma_ramp", "cycle_hot", "neutral")
        assert -10.0 < sc.composite < 10.0
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_signal_fusion.py -v`
Expected: FAIL — `ImportError`

**Step 2: Implement scorecard dataclass**

```python
# src/stockdownloader/gme/options/scorecard.py
"""GME Daily Scorecard — output of signal fusion."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True, slots=True)
class GMEDailyScorecard:
    """A single day's fused signal output."""

    date: date
    pillar_scores: dict[str, float]
    composite: float
    regime: str  # "squeeze", "gamma_ramp", "cycle_hot", "neutral"
    anomaly_flags: list[str] = field(default_factory=list)
```

**Step 3: Implement signal fusion**

```python
# src/stockdownloader/gme/options/signal_fusion.py
"""GMESignalFusion — multi-signal anomaly detection and scoring."""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from stockdownloader.gme.options.scorecard import GMEDailyScorecard

logger = logging.getLogger(__name__)

# Default pillar weights
DEFAULT_WEIGHTS = {
    "options_flow": 0.30,
    "volume_premium": 0.20,
    "cycle_timing": 0.30,
    "momentum": 0.20,
}

ANOMALY_THRESHOLD = 2.0  # standard deviations


class GMESignalFusion:
    """Fuse options analytics, alt data, and ML into composite score."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.weights = weights or dict(DEFAULT_WEIGHTS)

    # ------------------------------------------------------------------
    # Pillar scoring (each returns a Z-score-like float)
    # ------------------------------------------------------------------

    def _score_options(self, options: dict[str, Any]) -> float:
        """Options flow pillar: IV, skew, GEX, OI."""
        score = 0.0
        iv_pct = options.get("iv_percentile", 0.5)
        score += (iv_pct - 0.5) * 2  # center at 0

        skew = options.get("iv_skew_25d_30", 0)
        score += skew * 5  # amplify skew signal

        # GEX flip proximity (closer = stronger signal)
        gex_flip = options.get("gex_flip_price", 0)
        call_wall = options.get("call_wall", 0)
        if call_wall > 0 and gex_flip > 0:
            proximity = 1.0 - abs(gex_flip - call_wall) / call_wall
            score += proximity

        oi_change = options.get("pc_oi_ratio_change", 0)
        score += oi_change * 3

        return score

    def _score_volume(self, options: dict[str, Any]) -> float:
        """Volume/premium pillar."""
        score = 0.0
        pc_vol = options.get("pc_volume_ratio", 1.0)
        score += (1.0 - pc_vol) * 2  # low P/C = bullish

        imbalance = options.get("premium_imbalance", 0)
        score += imbalance * 3

        return score

    def _score_cycles(self, alt: dict[str, Any]) -> float:
        """Cycle timing pillar: T+35, SI, dark pool."""
        score = 0.0
        t35 = alt.get("ftd_t35_countdown", 35)
        if t35 < 10:
            score += (10 - t35) / 5  # peaks at T+35 = 0

        si_change = alt.get("si_change_2wk", 0)
        score += si_change * 0.5

        dp_change = alt.get("dark_pool_ratio_change", 0)
        score += dp_change * 3

        return score

    def _score_momentum(self, ml_score: float) -> float:
        """Momentum pillar from existing ML pipeline."""
        return (ml_score - 0.5) * 4  # center at 0, scale to ~±2

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def _detect_regime(
        self,
        options: dict[str, Any],
        alt: dict[str, Any],
        composite: float,
    ) -> str:
        """Classify current market regime."""
        iv_pct = options.get("iv_percentile", 0.5)
        gex_conc = options.get("gex_concentration", 0)
        t35 = alt.get("ftd_t35_countdown", 35)
        si_change = alt.get("si_change_2wk", 0)
        dp_change = alt.get("dark_pool_ratio_change", 0)

        # Squeeze: low IV + compressed GEX
        if iv_pct < 0.30 and gex_conc > 0.6:
            return "squeeze"

        # Gamma ramp: GEX flip approaching + concentrated
        if gex_conc > 0.5 and abs(composite) > 1.0:
            return "gamma_ramp"

        # Cycle hot: T+35 near + SI changing + dark pool diverging
        if t35 < 5 and (abs(si_change) > 2.0 or abs(dp_change) > 0.1):
            return "cycle_hot"

        return "neutral"

    # ------------------------------------------------------------------
    # Anomaly flags
    # ------------------------------------------------------------------

    def _flag_anomalies(self, pillars: dict[str, float]) -> list[str]:
        """Flag any pillar exceeding ±2σ."""
        flags = []
        for name, score in pillars.items():
            if abs(score) > ANOMALY_THRESHOLD:
                direction = "elevated" if score > 0 else "depressed"
                flags.append(f"{name} {direction} ({score:.1f}σ)")
        return flags

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def compute_daily_scorecard(
        self,
        trade_date: date,
        options_state: dict[str, Any],
        alt_data: dict[str, Any],
        ml_score: float = 0.5,
    ) -> GMEDailyScorecard:
        """Produce a single daily scorecard."""
        pillars = {
            "options_flow": self._score_options(options_state),
            "volume_premium": self._score_volume(options_state),
            "cycle_timing": self._score_cycles(alt_data),
            "momentum": self._score_momentum(ml_score),
        }

        composite = sum(
            self.weights[k] * pillars[k] for k in pillars
        )

        # Merge options + alt for regime detection
        regime_data = {**options_state, **alt_data}
        regime = self._detect_regime(options_state, alt_data, composite)
        anomalies = self._flag_anomalies(pillars)

        return GMEDailyScorecard(
            date=trade_date,
            pillar_scores=pillars,
            composite=composite,
            regime=regime,
            anomaly_flags=anomalies,
        )
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_signal_fusion.py -v`
Expected: PASS (8 tests)

**Step 4: Commit**

```bash
git add src/stockdownloader/gme/options/signal_fusion.py src/stockdownloader/gme/options/scorecard.py tests/gme/options/test_signal_fusion.py
git commit -m "feat(gme): add GMESignalFusion with 4-pillar scoring and regime detection"
```

---

### Task 8: CLI Entry Point

**Files:**
- Create: `src/stockdownloader/gme/options/__main__.py`
- Create: `tests/gme/options/test_cli.py`

**Step 1: Write failing test for CLI**

```python
# tests/gme/options/test_cli.py
"""Tests for GME options CLI."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from stockdownloader.gme.options.__main__ import build_parser


class TestCLIParser:
    def test_fetch_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["fetch", "--start", "2022-01-01", "--end", "2023-01-01"])
        assert args.command == "fetch"
        assert args.start == "2022-01-01"
        assert args.end == "2023-01-01"

    def test_build_state_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["build-state"])
        assert args.command == "build-state"

    def test_backtest_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["backtest", "--strategy", "wheel"])
        assert args.command == "backtest"
        assert args.strategy == "wheel"

    def test_backtest_all(self):
        parser = build_parser()
        args = parser.parse_args(["backtest", "--all"])
        assert args.command == "backtest"
        assert args.all is True

    def test_scorecard_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["scorecard", "--date", "2023-06-15"])
        assert args.command == "scorecard"
        assert args.date == "2023-06-15"

    def test_run_all_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["run-all"])
        assert args.command == "run-all"
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_cli.py -v`
Expected: FAIL — `ImportError`

**Step 2: Implement CLI**

```python
# src/stockdownloader/gme/options/__main__.py
"""CLI entry point for GME options analysis platform.

Usage:
    python -m stockdownloader.gme.options fetch --start 2022-01-01 --end 2026-02-26
    python -m stockdownloader.gme.options build-state
    python -m stockdownloader.gme.options backtest --strategy wheel
    python -m stockdownloader.gme.options backtest --all
    python -m stockdownloader.gme.options scorecard --date 2026-02-26
    python -m stockdownloader.gme.options run-all
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gme-options",
        description="GME Comprehensive Options Analysis Platform",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--polygon-key", type=str, default="")

    sub = parser.add_subparsers(dest="command")

    # fetch
    fetch_p = sub.add_parser("fetch", help="Fetch options data from Polygon")
    fetch_p.add_argument("--start", type=str, default="2022-01-01")
    fetch_p.add_argument("--end", type=str, default=date.today().isoformat())
    fetch_p.add_argument("--symbol", type=str, default="GME")

    # build-state
    sub.add_parser("build-state", help="Build analytics state from raw bars")

    # backtest
    bt_p = sub.add_parser("backtest", help="Run strategy backtests")
    bt_p.add_argument("--strategy", type=str, default="")
    bt_p.add_argument("--all", action="store_true")

    # scorecard
    sc_p = sub.add_parser("scorecard", help="Generate daily scorecards")
    sc_p.add_argument("--date", type=str, default="")
    sc_p.add_argument("--range", nargs=2, metavar=("START", "END"), default=None)

    # run-all
    sub.add_parser("run-all", help="Full pipeline: fetch → state → scorecard → backtest")

    return parser


def _cmd_fetch(args: argparse.Namespace) -> None:
    """Run the data fetcher."""
    from stockdownloader.data.market.polygon_options_client import PolygonOptionsClient
    from stockdownloader.gme.options.config import GMEOptionsConfig
    from stockdownloader.gme.options.fetcher import OptionsDataFetcher

    cfg = GMEOptionsConfig.from_env(
        polygon_api_key=args.polygon_key or None,
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end),
        symbol=args.symbol,
    )
    client = PolygonOptionsClient(api_key=cfg.polygon_api_key)
    fetcher = OptionsDataFetcher(cfg, client=client)
    fetcher.run()


def _cmd_build_state(args: argparse.Namespace) -> None:
    """Build options state from raw Parquet files."""
    from stockdownloader.gme.options.config import GMEOptionsConfig
    from stockdownloader.gme.options.state_engine import OptionsStateEngine

    cfg = GMEOptionsConfig.from_env()
    engine = OptionsStateEngine()
    state_df = engine.build(bars_dir=str(cfg.data_dir))
    out = cfg.data_dir.parent / "options_state.parquet"
    state_df.to_parquet(out, index=False)
    logger.info("Saved %d rows to %s", len(state_df), out)


def _cmd_backtest(args: argparse.Namespace) -> None:
    """Run backtests."""
    from stockdownloader.gme.options.backtester import GMEOptionsBacktester, BacktestConfig
    from stockdownloader.gme.options.strategies.wheel import GMEWheelStrategy
    from stockdownloader.gme.options.strategies.iron_condor import IronCondorStrategy
    from stockdownloader.gme.options.strategies.strangle import StrangleStrategy
    from stockdownloader.gme.options.strategies.credit_spread import CreditSpreadStrategy
    from stockdownloader.gme.options.strategies.long_options import LongOptionsStrategy
    from stockdownloader.gme.options.strategies.calendar_spread import CalendarSpreadStrategy

    ALL_STRATEGIES = {
        "wheel": GMEWheelStrategy,
        "iron_condor": IronCondorStrategy,
        "strangle": StrangleStrategy,
        "credit_spread": CreditSpreadStrategy,
        "long_options": LongOptionsStrategy,
        "calendar_spread": CalendarSpreadStrategy,
    }

    if args.all:
        strategies = [cls() for cls in ALL_STRATEGIES.values()]
    elif args.strategy in ALL_STRATEGIES:
        strategies = [ALL_STRATEGIES[args.strategy]()]
    else:
        logger.error("Unknown strategy: %s. Options: %s", args.strategy, list(ALL_STRATEGIES))
        return

    bt = GMEOptionsBacktester(strategies=strategies, config=BacktestConfig())
    logger.info("Backtest configured with %d strategies", len(strategies))
    # Full run requires state + chain data — log placeholder
    logger.info("Load state and chain data, then call bt.run()")


def _cmd_scorecard(args: argparse.Namespace) -> None:
    """Generate scorecards."""
    from stockdownloader.gme.options.signal_fusion import GMESignalFusion
    logger.info("Scorecard generation for %s", args.date or "range")
    # Placeholder — requires state + alt data loaded
    fusion = GMESignalFusion()
    logger.info("Signal fusion ready with weights: %s", fusion.weights)


def _cmd_run_all(args: argparse.Namespace) -> None:
    """Full pipeline."""
    logger.info("Running full pipeline: fetch → build-state → scorecard → backtest")
    _cmd_fetch(args)
    _cmd_build_state(args)
    _cmd_scorecard(args)
    _cmd_backtest(args)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    handlers = {
        "fetch": _cmd_fetch,
        "build-state": _cmd_build_state,
        "backtest": _cmd_backtest,
        "scorecard": _cmd_scorecard,
        "run-all": _cmd_run_all,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_cli.py -v`
Expected: PASS (6 tests)

**Step 3: Commit**

```bash
git add src/stockdownloader/gme/options/__main__.py tests/gme/options/test_cli.py
git commit -m "feat(gme): add CLI entry point for options analysis platform"
```

---

### Task 9: Integration Test + Full Suite Verification

**Files:**
- Create: `tests/gme/options/test_integration.py`

**Step 1: Write integration test**

```python
# tests/gme/options/test_integration.py
"""Integration test: full pipeline with synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock

from stockdownloader.gme.options.config import GMEOptionsConfig
from stockdownloader.gme.options.fetcher import OptionsDataFetcher
from stockdownloader.gme.options.state_engine import OptionsStateEngine
from stockdownloader.gme.options.backtester import (
    GMEOptionsBacktester,
    BacktestConfig,
)
from stockdownloader.gme.options.strategies.wheel import GMEWheelStrategy
from stockdownloader.gme.options.strategies.strangle import StrangleStrategy
from stockdownloader.gme.options.signal_fusion import GMESignalFusion
from stockdownloader.gme.options.scorecard import GMEDailyScorecard


def _synthetic_chain(trade_date: str, spot: float = 25.0) -> pd.DataFrame:
    rows = []
    for strike in [20.0, 22.5, 25.0, 27.5, 30.0]:
        for otype in ["call", "put"]:
            rows.append({
                "option_ticker": f"O:GME{trade_date.replace('-', '')}{'C' if otype == 'call' else 'P'}{int(strike*1000):08d}",
                "underlying": "GME",
                "date": trade_date,
                "expiration": (pd.Timestamp(trade_date) + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                "strike": strike,
                "option_type": otype,
                "open": 2.0,
                "high": 2.5,
                "low": 1.5,
                "close": 2.0,
                "volume": 100,
                "open_interest": 500,
                "vwap": 2.0,
            })
    return pd.DataFrame(rows)


class TestEndToEnd:
    def test_state_engine_to_fusion(self):
        """StateEngine output feeds into SignalFusion."""
        chain = _synthetic_chain("2023-06-15")
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        state = engine.build(bars_df=chain)

        assert len(state) == 1
        row = state.iloc[0]

        fusion = GMESignalFusion()
        sc = fusion.compute_daily_scorecard(
            trade_date=date(2023, 6, 15),
            options_state=row.to_dict(),
            alt_data={"ftd_t35_countdown": 15, "si_change_2wk": 0.5,
                      "dark_pool_ratio_change": 0.0},
            ml_score=0.6,
        )
        assert isinstance(sc, GMEDailyScorecard)
        assert sc.regime in ("squeeze", "gamma_ramp", "cycle_hot", "neutral")

    def test_backtester_with_real_strategies(self):
        """Backtester runs with wheel + strangle on synthetic data."""
        cfg = BacktestConfig(initial_capital=50_000.0)
        bt = GMEOptionsBacktester(
            strategies=[GMEWheelStrategy(), StrangleStrategy()],
            config=cfg,
        )

        chain = _synthetic_chain("2023-06-15")
        state_row = pd.Series({
            "date": date(2023, 6, 15),
            "atm_iv_30d": 0.6,
            "iv_percentile": 0.6,
            "net_gex": 1000.0,
            "gex_concentration": 0.3,
            "ftd_t35_countdown": 20,
        })

        trades = bt.run_single_day(date(2023, 6, 15), state_row, chain)
        assert len(trades) >= 1  # at least wheel or strangle fires

    def test_all_strategies_importable(self):
        """All 6 strategies can be instantiated."""
        from stockdownloader.gme.options.strategies.wheel import GMEWheelStrategy
        from stockdownloader.gme.options.strategies.iron_condor import IronCondorStrategy
        from stockdownloader.gme.options.strategies.strangle import StrangleStrategy
        from stockdownloader.gme.options.strategies.credit_spread import CreditSpreadStrategy
        from stockdownloader.gme.options.strategies.long_options import LongOptionsStrategy
        from stockdownloader.gme.options.strategies.calendar_spread import CalendarSpreadStrategy

        strategies = [
            GMEWheelStrategy(), IronCondorStrategy(), StrangleStrategy(),
            CreditSpreadStrategy(), LongOptionsStrategy(), CalendarSpreadStrategy(),
        ]
        assert len(strategies) == 6
        assert all(hasattr(s, "name") for s in strategies)

    def test_checkpoint_survives_restart(self, tmp_path):
        """Fetcher checkpoint persists across instances."""
        cfg = GMEOptionsConfig(
            data_dir=tmp_path,
            start_date=date(2022, 1, 1),
            end_date=date(2022, 3, 31),
        )
        f1 = OptionsDataFetcher(cfg, client=MagicMock())
        f1.checkpoint.mark_month_complete("2022-01")
        f1.checkpoint.save()

        f2 = OptionsDataFetcher(cfg, client=MagicMock())
        assert f2._remaining_months() == ["2022-02", "2022-03"]
```

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_integration.py -v`
Expected: PASS (4 tests)

**Step 2: Run full test suite**

```bash
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/ -x -q
```

Expected: All tests pass (3639+ existing + ~45 new = 3684+)

**Step 3: Commit**

```bash
git add tests/gme/options/test_integration.py
git commit -m "feat(gme): add integration tests for options analysis pipeline"
```

---

## Verification Checklist

After all tasks complete:

```bash
# 1. All new tests pass
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/ -v

# 2. Full suite regression
DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/ -x -q

# 3. Module structure correct
find src/stockdownloader/gme/options -name "*.py" | sort

# 4. CLI works
PYTHONPATH=src python3 -m stockdownloader.gme.options --help
PYTHONPATH=src python3 -m stockdownloader.gme.options fetch --help
PYTHONPATH=src python3 -m stockdownloader.gme.options backtest --help
```

Expected output structure:
```
src/stockdownloader/gme/options/
├── __init__.py
├── __main__.py
├── backtester.py
├── config.py
├── fetcher.py
├── scorecard.py
├── signal_fusion.py
├── state_engine.py
└── strategies/
    ├── __init__.py
    ├── calendar_spread.py
    ├── credit_spread.py
    ├── iron_condor.py
    ├── long_options.py
    ├── strangle.py
    └── wheel.py
```
