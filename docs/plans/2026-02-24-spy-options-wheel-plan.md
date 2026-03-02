# SPY ML-Guided Weekly Wheel Strategy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an ML-filtered weekly wheel strategy on SPY that sells cash-secured puts and covered calls using real Polygon option prices, with the existing walk-forward ensemble as a sell/skip filter.

**Architecture:** Standalone pipeline (`spy-options-wheel`) with three new modules: a Polygon options data client with disk caching, a wheel backtest state machine, and a CLI app that chains ML signal generation with the wheel backtest. Reuses the existing walk-forward ensemble for ML predictions.

**Tech Stack:** Python 3.11, requests, scipy (Black-Scholes delta), numpy, existing ML pipeline (scikit-learn, xgboost, lightgbm)

**Environment:** `DYLD_LIBRARY_PATH=/Users/kelsayed/lib` required for XGBoost/LightGBM on this macOS system.

---

### Task 1: Polygon Options Client — Contract Discovery

**Files:**
- Create: `src/stockdownloader/data/market/polygon_options_client.py`
- Create: `tests/data/market/test_polygon_options_client.py`

**Step 1: Write failing tests for contract fetching**

```python
# tests/data/market/test_polygon_options_client.py
"""Tests for the Polygon options data client."""
from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from stockdownloader.data.market.polygon_options_client import PolygonOptionsClient


# ── Fixtures ──────────────────────────────────────────────────────────

_CONTRACTS_RESPONSE = {
    "status": "OK",
    "request_id": "test",
    "results": [
        {
            "ticker": "O:SPY250207C00590000",
            "underlying_ticker": "SPY",
            "contract_type": "call",
            "exercise_style": "american",
            "expiration_date": "2025-02-07",
            "strike_price": 590.0,
            "shares_per_contract": 100,
        },
        {
            "ticker": "O:SPY250207C00600000",
            "underlying_ticker": "SPY",
            "contract_type": "call",
            "exercise_style": "american",
            "expiration_date": "2025-02-07",
            "strike_price": 600.0,
            "shares_per_contract": 100,
        },
        {
            "ticker": "O:SPY250207P00590000",
            "underlying_ticker": "SPY",
            "contract_type": "put",
            "exercise_style": "american",
            "expiration_date": "2025-02-07",
            "strike_price": 590.0,
            "shares_per_contract": 100,
        },
        {
            "ticker": "O:SPY250207P00580000",
            "underlying_ticker": "SPY",
            "contract_type": "put",
            "exercise_style": "american",
            "expiration_date": "2025-02-07",
            "strike_price": 580.0,
            "shares_per_contract": 100,
        },
    ],
    "next_url": None,
}


class TestFetchOptionContracts:
    """Test contract discovery from Polygon reference API."""

    def test_returns_contracts_for_expiration(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _CONTRACTS_RESPONSE
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        client = PolygonOptionsClient(api_key="test_key")
        client._session = session

        contracts = client.fetch_option_contracts(
            "SPY",
            from_date=date(2025, 2, 1),
            to_date=date(2025, 2, 28),
        )
        assert len(contracts) == 4
        assert contracts[0]["ticker"] == "O:SPY250207C00590000"
        assert contracts[0]["strike_price"] == 590.0
        assert contracts[0]["contract_type"] == "call"

    def test_handles_pagination(self):
        page1 = {
            "status": "OK",
            "results": _CONTRACTS_RESPONSE["results"][:2],
            "next_url": "https://api.polygon.io/v3/reference/options/contracts?cursor=abc",
        }
        page2 = {
            "status": "OK",
            "results": _CONTRACTS_RESPONSE["results"][2:],
            "next_url": None,
        }

        session = MagicMock()
        resp1 = MagicMock()
        resp1.json.return_value = page1
        resp1.raise_for_status = MagicMock()
        resp2 = MagicMock()
        resp2.json.return_value = page2
        resp2.raise_for_status = MagicMock()
        session.get.side_effect = [resp1, resp2]

        client = PolygonOptionsClient(api_key="test_key")
        client._session = session

        contracts = client.fetch_option_contracts(
            "SPY",
            from_date=date(2025, 2, 1),
            to_date=date(2025, 2, 28),
        )
        assert len(contracts) == 4
        assert session.get.call_count == 2

    def test_empty_results(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"status": "OK", "results": [], "next_url": None}
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        client = PolygonOptionsClient(api_key="test_key")
        client._session = session

        contracts = client.fetch_option_contracts(
            "SPY", from_date=date(2025, 2, 1), to_date=date(2025, 2, 28),
        )
        assert contracts == []

    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                PolygonOptionsClient(api_key="")

    def test_filters_to_friday_expirations(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _CONTRACTS_RESPONSE
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        client = PolygonOptionsClient(api_key="test_key")
        client._session = session

        contracts = client.fetch_option_contracts(
            "SPY",
            from_date=date(2025, 2, 1),
            to_date=date(2025, 2, 28),
            fridays_only=True,
        )
        # 2025-02-07 is a Friday — all 4 contracts should pass
        assert len(contracts) == 4
```

**Step 2: Run tests to verify they fail**

Run: `DYLD_LIBRARY_PATH=/Users/kelsayed/lib PYTHONPATH=src python -m pytest tests/data/market/test_polygon_options_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stockdownloader.data.market.polygon_options_client'`

**Step 3: Write minimal implementation**

```python
# src/stockdownloader/data/market/polygon_options_client.py
"""Fetches historical options data from Polygon.io.

Downloads option contract metadata and daily bars for backtesting
options strategies with real market prices.

Usage::

    client = PolygonOptionsClient(api_key="YOUR_KEY")
    contracts = client.fetch_option_contracts("SPY", from_date, to_date)
    bar = client.fetch_option_daily_bar("O:SPY250207P00580000", trade_date)

Set the API key via:
  - Constructor parameter: ``PolygonOptionsClient(api_key="...")``
  - Environment variable: ``POLYGON_API_KEY``
"""
from __future__ import annotations

import json
import logging
import os
import time as _time
from datetime import date, datetime, timedelta
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests

logger = logging.getLogger(__name__)

_CONTRACTS_URL = "https://api.polygon.io/v3/reference/options/contracts"
_AGGS_URL = (
    "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
)

# Upgraded tier: unlimited calls, but be respectful
_DEFAULT_DELAY = 0.2


class PolygonOptionsClient:
    """Fetches option contract metadata and prices from Polygon.io."""

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit_delay: float = _DEFAULT_DELAY,
    ) -> None:
        self._api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Polygon API key required. Pass api_key= or set POLYGON_API_KEY env var."
            )
        self._delay = rate_limit_delay
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {self._api_key}"}
        )

    def fetch_option_contracts(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        *,
        expired: bool = True,
        fridays_only: bool = False,
        contract_type: str | None = None,
    ) -> list[dict]:
        """Fetch option contract metadata from Polygon reference API.

        Parameters
        ----------
        symbol:
            Underlying ticker (e.g. ``"SPY"``).
        from_date:
            Include contracts expiring on or after this date.
        to_date:
            Include contracts expiring on or before this date.
        expired:
            If ``True``, include expired contracts (needed for backtesting).
        fridays_only:
            If ``True``, only return contracts expiring on a Friday.
        contract_type:
            Filter to ``"call"`` or ``"put"`` only.  ``None`` returns both.

        Returns
        -------
        list[dict]
            Raw contract dicts from Polygon, each with keys like
            ``ticker``, ``strike_price``, ``expiration_date``,
            ``contract_type``, etc.
        """
        params: dict = {
            "underlying_ticker": symbol.upper(),
            "expiration_date.gte": str(from_date),
            "expiration_date.lte": str(to_date),
            "expired": str(expired).lower(),
            "limit": 1000,
            "order": "asc",
            "sort": "expiration_date",
        }
        if contract_type:
            params["contract_type"] = contract_type

        all_contracts: list[dict] = []

        try:
            resp = self._session.get(_CONTRACTS_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            all_contracts.extend(results)

            # Handle pagination
            next_url = data.get("next_url")
            while next_url:
                _time.sleep(self._delay)
                parsed = urlparse(next_url)
                qs = parse_qs(parsed.query, keep_blank_values=True)
                qs.pop("apiKey", None)
                clean_query = urlencode(qs, doseq=True)
                clean_url = urlunparse(parsed._replace(query=clean_query))
                resp = self._session.get(clean_url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
                all_contracts.extend(results)
                next_url = data.get("next_url")

        except requests.RequestException as exc:
            logger.warning("Polygon options contracts request failed: %s", exc)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Polygon options contracts parse error: %s", exc)

        # Filter to Fridays if requested
        if fridays_only:
            all_contracts = [
                c for c in all_contracts
                if _is_friday(c.get("expiration_date", ""))
            ]

        logger.info(
            "Fetched %d option contracts for %s (%s to %s)",
            len(all_contracts), symbol, from_date, to_date,
        )
        return all_contracts

    def fetch_option_daily_bar(
        self,
        option_ticker: str,
        trade_date: date,
    ) -> dict | None:
        """Fetch a single daily bar for an options contract.

        Parameters
        ----------
        option_ticker:
            Polygon options ticker (e.g. ``"O:SPY250207P00580000"``).
        trade_date:
            The date to fetch the bar for.

        Returns
        -------
        dict | None
            Bar dict with keys ``o``, ``h``, ``l``, ``c``, ``v``, ``t``,
            or ``None`` if no data is available.
        """
        url = _AGGS_URL.format(
            ticker=option_ticker,
            from_date=str(trade_date),
            to_date=str(trade_date),
        )
        params = {"adjusted": "true", "sort": "asc", "limit": 1}

        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if results:
                return results[0]
        except requests.RequestException as exc:
            logger.warning(
                "Polygon option bar request failed for %s: %s",
                option_ticker, exc,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "Polygon option bar parse error for %s: %s",
                option_ticker, exc,
            )

        return None


def _is_friday(date_str: str) -> bool:
    """Return True if the date string (YYYY-MM-DD) falls on a Friday."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").weekday() == 4
    except (ValueError, TypeError):
        return False
```

**Step 4: Run tests to verify they pass**

Run: `DYLD_LIBRARY_PATH=/Users/kelsayed/lib PYTHONPATH=src python -m pytest tests/data/market/test_polygon_options_client.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/stockdownloader/data/market/polygon_options_client.py tests/data/market/test_polygon_options_client.py
git commit -m "feat: add Polygon options client for contract discovery"
```

---

### Task 2: Polygon Options Client — Daily Bar Fetching & Cache

**Files:**
- Modify: `src/stockdownloader/data/market/polygon_options_client.py`
- Modify: `tests/data/market/test_polygon_options_client.py`

**Step 1: Write failing tests for daily bar fetching and caching**

```python
# Add to tests/data/market/test_polygon_options_client.py

_BAR_RESPONSE = {
    "status": "OK",
    "ticker": "O:SPY250207P00580000",
    "resultsCount": 1,
    "results": [
        {"o": 3.50, "h": 3.80, "l": 3.20, "c": 3.55, "v": 1500, "t": 1738886400000}
    ],
}


class TestFetchOptionDailyBar:
    """Test daily bar fetching for specific option contracts."""

    def test_returns_bar_data(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = _BAR_RESPONSE
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        client = PolygonOptionsClient(api_key="test_key")
        client._session = session

        bar = client.fetch_option_daily_bar(
            "O:SPY250207P00580000", date(2025, 2, 3),
        )
        assert bar is not None
        assert bar["c"] == 3.55
        assert bar["v"] == 1500

    def test_returns_none_for_no_data(self):
        session = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"status": "OK", "results": []}
        resp.raise_for_status = MagicMock()
        session.get.return_value = resp

        client = PolygonOptionsClient(api_key="test_key")
        client._session = session

        bar = client.fetch_option_daily_bar(
            "O:SPY250207P00580000", date(2025, 2, 3),
        )
        assert bar is None


class TestChainCache:
    """Test disk caching for option chain data."""

    def test_save_and_load_cache(self, tmp_path):
        from stockdownloader.data.market.polygon_options_client import (
            save_chain_cache,
            load_chain_cache,
        )

        cache_data = {
            "expiration_date": "2025-02-07",
            "contracts": _CONTRACTS_RESPONSE["results"],
            "bars": {"O:SPY250207P00580000": _BAR_RESPONSE["results"][0]},
        }
        save_chain_cache(tmp_path, "2025-02-07", cache_data)
        loaded = load_chain_cache(tmp_path, "2025-02-07")

        assert loaded is not None
        assert loaded["expiration_date"] == "2025-02-07"
        assert len(loaded["contracts"]) == 4

    def test_load_returns_none_for_missing(self, tmp_path):
        from stockdownloader.data.market.polygon_options_client import (
            load_chain_cache,
        )

        loaded = load_chain_cache(tmp_path, "2025-02-07")
        assert loaded is None
```

**Step 2: Run tests to verify new tests fail**

Run: `DYLD_LIBRARY_PATH=/Users/kelsayed/lib PYTHONPATH=src python -m pytest tests/data/market/test_polygon_options_client.py -v`
Expected: New cache tests FAIL — `cannot import name 'save_chain_cache'`

**Step 3: Add cache functions to implementation**

```python
# Add to polygon_options_client.py — at module level, after the class

from pathlib import Path


def save_chain_cache(cache_dir: Path, expiration_date: str, data: dict) -> None:
    """Save option chain data to disk cache.

    Parameters
    ----------
    cache_dir:
        Root cache directory (e.g. ``data/options_cache/SPY``).
    expiration_date:
        Expiration date string (``YYYY-MM-DD``), used as filename.
    data:
        Dict with keys ``expiration_date``, ``contracts``, ``bars``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{expiration_date}.json"
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug("Saved cache: %s", cache_file)


def load_chain_cache(cache_dir: Path, expiration_date: str) -> dict | None:
    """Load option chain data from disk cache.

    Returns
    -------
    dict | None
        Cached data dict, or ``None`` if cache file does not exist.
    """
    cache_file = cache_dir / f"{expiration_date}.json"
    if not cache_file.exists():
        return None
    with open(cache_file) as f:
        return json.load(f)
```

**Step 4: Run tests to verify they pass**

Run: `DYLD_LIBRARY_PATH=/Users/kelsayed/lib PYTHONPATH=src python -m pytest tests/data/market/test_polygon_options_client.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add src/stockdownloader/data/market/polygon_options_client.py tests/data/market/test_polygon_options_client.py
git commit -m "feat: add daily bar fetching and disk cache for options client"
```

---

### Task 3: Strike Selection via Black-Scholes Delta

**Files:**
- Modify: `src/stockdownloader/data/market/polygon_options_client.py`
- Modify: `tests/data/market/test_polygon_options_client.py`

**Step 1: Write failing tests for strike selection**

```python
# Add to tests/data/market/test_polygon_options_client.py

class TestSelectStrikeByDelta:
    """Test BS-delta-based strike selection."""

    def test_selects_put_near_030_delta(self):
        from stockdownloader.data.market.polygon_options_client import (
            select_strike_by_delta,
        )

        contracts = [
            {"ticker": "O:SPY250207P00570000", "strike_price": 570.0, "contract_type": "put"},
            {"ticker": "O:SPY250207P00580000", "strike_price": 580.0, "contract_type": "put"},
            {"ticker": "O:SPY250207P00590000", "strike_price": 590.0, "contract_type": "put"},
            {"ticker": "O:SPY250207P00600000", "strike_price": 600.0, "contract_type": "put"},
        ]
        # SPY at 600, 5 DTE, ~20% vol => 0.30 delta put is ~580-590 strike
        result = select_strike_by_delta(
            contracts,
            contract_type="put",
            spot=600.0,
            target_delta=0.30,
            days_to_expiry=5,
            volatility=0.20,
        )
        assert result is not None
        assert result["contract_type"] == "put"
        # The selected strike should be OTM (below spot for puts)
        assert result["strike_price"] < 600.0

    def test_selects_call_near_030_delta(self):
        from stockdownloader.data.market.polygon_options_client import (
            select_strike_by_delta,
        )

        contracts = [
            {"ticker": "O:SPY250207C00600000", "strike_price": 600.0, "contract_type": "call"},
            {"ticker": "O:SPY250207C00610000", "strike_price": 610.0, "contract_type": "call"},
            {"ticker": "O:SPY250207C00620000", "strike_price": 620.0, "contract_type": "call"},
            {"ticker": "O:SPY250207C00630000", "strike_price": 630.0, "contract_type": "call"},
        ]
        result = select_strike_by_delta(
            contracts,
            contract_type="call",
            spot=600.0,
            target_delta=0.30,
            days_to_expiry=5,
            volatility=0.20,
        )
        assert result is not None
        assert result["contract_type"] == "call"
        assert result["strike_price"] > 600.0

    def test_returns_none_for_empty_contracts(self):
        from stockdownloader.data.market.polygon_options_client import (
            select_strike_by_delta,
        )

        result = select_strike_by_delta(
            [], contract_type="put", spot=600.0,
            target_delta=0.30, days_to_expiry=5, volatility=0.20,
        )
        assert result is None
```

**Step 2: Run tests to verify new tests fail**

Run: `DYLD_LIBRARY_PATH=/Users/kelsayed/lib PYTHONPATH=src python -m pytest tests/data/market/test_polygon_options_client.py::TestSelectStrikeByDelta -v`
Expected: FAIL — `cannot import name 'select_strike_by_delta'`

**Step 3: Implement strike selection**

```python
# Add to polygon_options_client.py at module level

from decimal import Decimal

from stockdownloader.analysis.options.pricing import delta as bs_delta
from stockdownloader.core.models.options import OptionType


_DEFAULT_RISK_FREE_RATE = 0.05


def select_strike_by_delta(
    contracts: list[dict],
    *,
    contract_type: str,
    spot: float,
    target_delta: float,
    days_to_expiry: int,
    volatility: float,
    risk_free_rate: float = _DEFAULT_RISK_FREE_RATE,
) -> dict | None:
    """Select the contract whose BS delta is closest to the target.

    Parameters
    ----------
    contracts:
        List of contract dicts (must have ``strike_price`` and ``contract_type``).
    contract_type:
        ``"call"`` or ``"put"``.
    spot:
        Current underlying price.
    target_delta:
        Target absolute delta (e.g. 0.30).
    days_to_expiry:
        Days to expiration.
    volatility:
        Annualized volatility estimate.
    risk_free_rate:
        Annual risk-free rate.

    Returns
    -------
    dict | None
        The contract dict closest to the target delta, or ``None`` if
        *contracts* is empty.
    """
    filtered = [c for c in contracts if c.get("contract_type") == contract_type]
    if not filtered:
        return None

    opt_type = OptionType.CALL if contract_type == "call" else OptionType.PUT
    time_to_expiry = Decimal(str(days_to_expiry)) / Decimal("365")
    spot_d = Decimal(str(spot))
    vol_d = Decimal(str(volatility))
    rfr_d = Decimal(str(risk_free_rate))

    best_contract = None
    best_diff = float("inf")

    for c in filtered:
        strike_d = Decimal(str(c["strike_price"]))
        d = bs_delta(opt_type, spot_d, strike_d, time_to_expiry, rfr_d, vol_d)
        diff = abs(abs(float(d)) - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_contract = c

    return best_contract
```

**Step 4: Run tests to verify they pass**

Run: `DYLD_LIBRARY_PATH=/Users/kelsayed/lib PYTHONPATH=src python -m pytest tests/data/market/test_polygon_options_client.py -v`
Expected: All 12 tests PASS

**Step 5: Commit**

```bash
git add src/stockdownloader/data/market/polygon_options_client.py tests/data/market/test_polygon_options_client.py
git commit -m "feat: add BS-delta strike selection for options contracts"
```

---

### Task 4: Wheel Backtest Engine — State Machine

**Files:**
- Create: `src/stockdownloader/backtesting/engines/wheel.py`
- Create: `tests/backtesting/engines/test_wheel.py`

**Step 1: Write failing tests for wheel state transitions**

```python
# tests/backtesting/engines/test_wheel.py
"""Tests for the weekly wheel backtest engine."""
from __future__ import annotations

import pytest

from stockdownloader.backtesting.engines.wheel import (
    WheelBacktestEngine,
    WheelState,
    WeekRecord,
)


def _make_week(
    week_num: int,
    spy_open: float,
    spy_close: float,
    put_premium: float = 2.0,
    call_premium: float = 2.0,
    put_strike: float = 580.0,
    call_strike: float = 620.0,
    ml_prob: float = 0.50,
) -> WeekRecord:
    """Create a WeekRecord for testing."""
    return WeekRecord(
        week_num=week_num,
        expiration_date=f"2025-02-{7 + week_num * 7:02d}",
        entry_date=f"2025-02-{3 + week_num * 7:02d}",
        spy_price_at_entry=spy_open,
        spy_price_at_expiry=spy_close,
        put_strike=put_strike,
        call_strike=call_strike,
        put_premium=put_premium,
        call_premium=call_premium,
        ml_prob=ml_prob,
    )


class TestWheelStateTransitions:
    """Test the wheel lifecycle state machine."""

    def test_initial_state_is_cash(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1)
        assert engine.state == WheelState.CASH

    def test_first_week_sells_put(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1)
        week = _make_week(0, spy_open=600.0, spy_close=605.0)
        engine.process_week(week)
        # Put expired OTM (605 > 580 strike), should stay in PUT_PHASE
        # Premium collected = 2.0 * 100 = $200
        assert engine.state == WheelState.PUT_PHASE
        assert engine.total_premium_collected > 0

    def test_put_assignment_transitions_to_call_phase(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1)
        # Week 0: sell put, SPY drops below put strike => assigned
        week = _make_week(
            0, spy_open=600.0, spy_close=570.0,
            put_strike=580.0, put_premium=3.0,
        )
        engine.process_week(week)
        # Assigned: bought 100 shares at $580 strike
        assert engine.state in (WheelState.HOLDING, WheelState.CALL_PHASE)
        assert engine.shares_held == 100

    def test_call_exercise_transitions_to_put_phase(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1)
        # Manually set to CALL_PHASE with shares
        engine._state = WheelState.CALL_PHASE
        engine._shares = 100
        engine._share_cost_basis = 580.0

        # SPY rallies above call strike => called away
        week = _make_week(
            0, spy_open=600.0, spy_close=625.0,
            call_strike=620.0, call_premium=2.0,
        )
        engine.process_week(week)
        assert engine.state == WheelState.PUT_PHASE
        assert engine.shares_held == 0

    def test_otm_call_keeps_premium(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1)
        engine._state = WheelState.CALL_PHASE
        engine._shares = 100
        engine._share_cost_basis = 580.0

        # SPY stays below call strike => OTM, keep premium
        week = _make_week(
            0, spy_open=600.0, spy_close=610.0,
            call_strike=620.0, call_premium=2.0,
        )
        engine.process_week(week)
        assert engine.state == WheelState.CALL_PHASE
        assert engine.shares_held == 100

    def test_ml_filter_skips_put(self):
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            skip_put_thresh=0.35,
        )
        # ML prob < 0.35 => skip selling put
        week = _make_week(0, spy_open=600.0, spy_close=605.0, ml_prob=0.30)
        engine.process_week(week, use_ml_filter=True)
        assert engine.state == WheelState.CASH
        assert engine.total_premium_collected == 0

    def test_ml_filter_skips_call(self):
        engine = WheelBacktestEngine(
            initial_capital=100_000.0, contracts=1,
            skip_call_thresh=0.65,
        )
        engine._state = WheelState.CALL_PHASE
        engine._shares = 100
        engine._share_cost_basis = 580.0

        # ML prob > 0.65 => skip selling call (rally expected)
        week = _make_week(0, spy_open=600.0, spy_close=605.0, ml_prob=0.70)
        engine.process_week(week, use_ml_filter=True)
        # Should still hold shares but not have sold a call
        assert engine.shares_held == 100


class TestWheelMetrics:
    """Test metrics computation."""

    def test_returns_metrics_dict(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1)
        # Run a few weeks
        weeks = [
            _make_week(i, spy_open=600.0, spy_close=605.0)
            for i in range(4)
        ]
        for w in weeks:
            engine.process_week(w)

        metrics = engine.compute_metrics()
        assert "total_return_pct" in metrics
        assert "total_premium_collected" in metrics
        assert "n_assignments" in metrics
        assert "n_calls_exercised" in metrics
        assert "sharpe" in metrics
        assert "max_drawdown_pct" in metrics

    def test_equity_curve_tracks_weekly(self):
        engine = WheelBacktestEngine(initial_capital=100_000.0, contracts=1)
        weeks = [
            _make_week(i, spy_open=600.0, spy_close=605.0)
            for i in range(4)
        ]
        for w in weeks:
            engine.process_week(w)

        assert len(engine.equity_curve) == 4
```

**Step 2: Run tests to verify they fail**

Run: `DYLD_LIBRARY_PATH=/Users/kelsayed/lib PYTHONPATH=src python -m pytest tests/backtesting/engines/test_wheel.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement the wheel engine**

```python
# src/stockdownloader/backtesting/engines/wheel.py
"""Weekly wheel backtest engine.

Simulates the wheel strategy (cash-secured puts -> covered calls)
on SPY with optional ML-based sell/skip filter.

The wheel lifecycle:
    CASH -> PUT_PHASE (sell CSP)
    PUT_PHASE -> HOLDING (assigned when put is ITM)
    HOLDING -> CALL_PHASE (sell CC)
    CALL_PHASE -> PUT_PHASE (called away when call is ITM)
    PUT/CALL OTM expiry -> stay in same phase, collect premium
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class WheelState(Enum):
    """Current phase of the wheel strategy."""

    CASH = "CASH"
    PUT_PHASE = "PUT_PHASE"
    HOLDING = "HOLDING"
    CALL_PHASE = "CALL_PHASE"


@dataclass(frozen=True, slots=True)
class WeekRecord:
    """Data for one trading week in the wheel backtest."""

    week_num: int
    expiration_date: str
    entry_date: str
    spy_price_at_entry: float
    spy_price_at_expiry: float
    put_strike: float
    call_strike: float
    put_premium: float
    call_premium: float
    ml_prob: float


class WheelBacktestEngine:
    """Runs the weekly wheel backtest.

    Parameters
    ----------
    initial_capital:
        Starting cash in dollars.
    contracts:
        Number of option contracts per trade (1 contract = 100 shares).
    skip_put_thresh:
        Skip selling puts when ML prob < this (crash danger).
    skip_call_thresh:
        Skip selling calls when ML prob > this (rally expected).
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        contracts: int = 1,
        skip_put_thresh: float = 0.35,
        skip_call_thresh: float = 0.65,
    ) -> None:
        self._initial_capital = initial_capital
        self._cash = initial_capital
        self._contracts = contracts
        self._skip_put_thresh = skip_put_thresh
        self._skip_call_thresh = skip_call_thresh

        # State
        self._state = WheelState.CASH
        self._shares: int = 0
        self._share_cost_basis: float = 0.0

        # Tracking
        self._total_premium: float = 0.0
        self._n_assignments: int = 0
        self._n_calls_exercised: int = 0
        self._n_puts_sold: int = 0
        self._n_calls_sold: int = 0
        self._n_puts_skipped: int = 0
        self._n_calls_skipped: int = 0
        self._equity_curve: list[float] = []
        self._weeks_processed: int = 0

    # ── Properties ────────────────────────────────────────────────

    @property
    def state(self) -> WheelState:
        return self._state

    @property
    def shares_held(self) -> int:
        return self._shares

    @property
    def total_premium_collected(self) -> float:
        return self._total_premium

    @property
    def equity_curve(self) -> list[float]:
        return list(self._equity_curve)

    # ── Core logic ────────────────────────────────────────────────

    def process_week(
        self,
        week: WeekRecord,
        *,
        use_ml_filter: bool = False,
    ) -> None:
        """Process one week of the wheel strategy.

        Parameters
        ----------
        week:
            Data for this trading week.
        use_ml_filter:
            If ``True``, consult ML probability to decide whether to sell.
        """
        multiplier = self._contracts * 100

        if self._state in (WheelState.CASH, WheelState.PUT_PHASE):
            # ── PUT PHASE ──
            if use_ml_filter and week.ml_prob < self._skip_put_thresh:
                self._n_puts_skipped += 1
                # Don't sell, stay in CASH
            else:
                # Sell cash-secured put
                premium = week.put_premium * multiplier
                self._cash += premium
                self._total_premium += premium
                self._n_puts_sold += 1
                self._state = WheelState.PUT_PHASE

                # Check expiration
                if week.spy_price_at_expiry < week.put_strike:
                    # ITM: assigned — buy shares at strike
                    cost = week.put_strike * multiplier
                    self._cash -= cost
                    self._shares = multiplier
                    self._share_cost_basis = week.put_strike
                    self._n_assignments += 1
                    self._state = WheelState.CALL_PHASE
                # else: OTM, keep premium, stay in PUT_PHASE

        elif self._state in (WheelState.HOLDING, WheelState.CALL_PHASE):
            # ── CALL PHASE ──
            if use_ml_filter and week.ml_prob > self._skip_call_thresh:
                self._n_calls_skipped += 1
                # Don't sell call, just hold shares
            else:
                # Sell covered call
                premium = week.call_premium * multiplier
                self._cash += premium
                self._total_premium += premium
                self._n_calls_sold += 1
                self._state = WheelState.CALL_PHASE

                # Check expiration
                if week.spy_price_at_expiry > week.call_strike:
                    # ITM: called away — sell shares at strike
                    proceeds = week.call_strike * multiplier
                    self._cash += proceeds
                    self._shares = 0
                    self._share_cost_basis = 0.0
                    self._n_calls_exercised += 1
                    self._state = WheelState.PUT_PHASE
                # else: OTM, keep premium + shares, stay in CALL_PHASE

        # ── Update equity curve ──
        equity = self._cash + self._shares * week.spy_price_at_expiry
        self._equity_curve.append(equity)
        self._weeks_processed += 1

    def compute_metrics(self) -> dict[str, float]:
        """Compute summary metrics for the backtest.

        Returns
        -------
        dict[str, float]
            Keys: initial_capital, final_equity, total_return_pct,
            total_return_dollar, total_premium_collected, n_assignments,
            n_calls_exercised, n_puts_sold, n_calls_sold,
            n_puts_skipped, n_calls_skipped, weeks, sharpe,
            max_drawdown_pct, max_drawdown_dollar,
            annualized_return_pct.
        """
        final_equity = self._equity_curve[-1] if self._equity_curve else self._initial_capital
        total_return = final_equity - self._initial_capital
        total_return_pct = (total_return / self._initial_capital) * 100

        # Sharpe (weekly returns, annualized)
        sharpe = 0.0
        if len(self._equity_curve) >= 2:
            returns = []
            prev = self._initial_capital
            for eq in self._equity_curve:
                returns.append((eq - prev) / prev if prev > 0 else 0.0)
                prev = eq
            if returns:
                import statistics
                mean_r = statistics.mean(returns)
                std_r = statistics.pstdev(returns)
                if std_r > 0:
                    sharpe = (mean_r / std_r) * math.sqrt(52)  # annualize weekly

        # Max drawdown
        max_dd_pct = 0.0
        max_dd_dollar = 0.0
        peak = self._initial_capital
        for eq in self._equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq)
            dd_pct = dd / peak * 100 if peak > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd_dollar = dd

        # Annualized return
        years = self._weeks_processed / 52.0 if self._weeks_processed > 0 else 1.0
        annualized = 0.0
        if years > 0 and final_equity > 0 and self._initial_capital > 0:
            annualized = ((final_equity / self._initial_capital) ** (1.0 / years) - 1.0) * 100

        return {
            "initial_capital": self._initial_capital,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "total_return_dollar": total_return,
            "total_premium_collected": self._total_premium,
            "n_assignments": float(self._n_assignments),
            "n_calls_exercised": float(self._n_calls_exercised),
            "n_puts_sold": float(self._n_puts_sold),
            "n_calls_sold": float(self._n_calls_sold),
            "n_puts_skipped": float(self._n_puts_skipped),
            "n_calls_skipped": float(self._n_calls_skipped),
            "weeks": float(self._weeks_processed),
            "sharpe": sharpe,
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_dollar": max_dd_dollar,
            "annualized_return_pct": annualized,
        }
```

**Step 4: Run tests to verify they pass**

Run: `DYLD_LIBRARY_PATH=/Users/kelsayed/lib PYTHONPATH=src python -m pytest tests/backtesting/engines/test_wheel.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add src/stockdownloader/backtesting/engines/wheel.py tests/backtesting/engines/test_wheel.py
git commit -m "feat: add wheel backtest engine with state machine and ML filter"
```

---

### Task 5: CLI App — Pipeline Orchestration

**Files:**
- Create: `src/stockdownloader/app/spy_options_wheel.py`
- Create: `tests/app/test_spy_options_wheel.py`
- Modify: `pyproject.toml` (add console_scripts entry)

**Step 1: Write failing tests for parser and pipeline structure**

```python
# tests/app/test_spy_options_wheel.py
"""Tests for the spy-options-wheel CLI pipeline."""
from __future__ import annotations

import pytest

from stockdownloader.app.spy_options_wheel import _build_parser


class TestWheelParser:
    """Test CLI argument parsing."""

    def test_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.delta == 0.30
        assert args.initial_capital == 100_000.0
        assert args.contracts == 1
        assert args.skip_put_thresh == 0.35
        assert args.skip_call_thresh == 0.65
        assert args.no_ml_filter is False

    def test_custom_delta(self):
        parser = _build_parser()
        args = parser.parse_args(["--delta", "0.20"])
        assert args.delta == 0.20

    def test_no_ml_filter_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--no-ml-filter"])
        assert args.no_ml_filter is True

    def test_custom_thresholds(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--skip-put-thresh", "0.30",
            "--skip-call-thresh", "0.70",
        ])
        assert args.skip_put_thresh == 0.30
        assert args.skip_call_thresh == 0.70

    def test_date_range(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--from-date", "2022-01-01",
            "--to-date", "2026-01-01",
        ])
        assert args.from_date == "2022-01-01"
        assert args.to_date == "2026-01-01"

    def test_contracts_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["--contracts", "2"])
        assert args.contracts == 2

    def test_walk_forward_windows(self):
        parser = _build_parser()
        args = parser.parse_args(["--walk-forward-windows", "8"])
        assert args.walk_forward_windows == 8
```

**Step 2: Run tests to verify they fail**

Run: `DYLD_LIBRARY_PATH=/Users/kelsayed/lib PYTHONPATH=src python -m pytest tests/app/test_spy_options_wheel.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement the CLI app**

```python
# src/stockdownloader/app/spy_options_wheel.py
"""SPY ML-Guided Weekly Wheel Strategy pipeline.

End-to-end pipeline that:
  [1/5] Downloads SPY daily data
  [2/5] Runs ML walk-forward ensemble for probability predictions
  [3/5] Downloads option chain data from Polygon (with caching)
  [4/5] Runs wheel backtest (ML-filtered vs mechanical)
  [5/5] Prints comparison results

Usage::

    spy-options-wheel                        # Full pipeline with ML filter
    spy-options-wheel --no-ml-filter         # Pure mechanical wheel
    spy-options-wheel --delta 0.20           # Conservative strike selection
    spy-options-wheel --contracts 2          # 2 contracts per trade
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from stockdownloader.app.ml_helpers import add_common_ml_args, init_ml_env
from stockdownloader.core.config import DEFAULT_ML_PIPELINE_DIR


# ── Model grid (reuse from spy_ml_ensemble) ──────────────────────────

_FULL_GRID = [
    ("gradient_boosting", False),
    ("random_forest", False),
    ("logistic_regression", False),
    ("gradient_boosting", True),
    ("random_forest", True),
]

_QUICK_GRID = [
    ("gradient_boosting", False),
    ("random_forest", False),
    ("logistic_regression", False),
]


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for ``spy-options-wheel``."""
    parser = argparse.ArgumentParser(
        prog="spy-options-wheel",
        description=(
            "SPY ML-Guided Weekly Wheel: download SPY data -> "
            "ML walk-forward predictions -> Polygon option prices -> "
            "wheel backtest with ML filter."
        ),
    )

    # Common ML args
    add_common_ml_args(parser)

    # Wheel-specific args
    parser.add_argument(
        "--delta", type=float, default=0.30,
        help="Target delta for strike selection (default: 0.30)",
    )
    parser.add_argument(
        "--contracts", type=int, default=1,
        help="Number of option contracts per trade (default: 1)",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=100_000.0,
        help="Initial capital in dollars (default: 100000)",
    )
    parser.add_argument(
        "--from-date", type=str, default="2022-01-01",
        help="Backtest start date YYYY-MM-DD (default: 2022-01-01)",
    )
    parser.add_argument(
        "--to-date", type=str, default=None,
        help="Backtest end date YYYY-MM-DD (default: today)",
    )

    # ML filter args
    parser.add_argument(
        "--no-ml-filter", action="store_true",
        help="Run pure mechanical wheel without ML filter",
    )
    parser.add_argument(
        "--skip-put-thresh", type=float, default=0.35,
        help="Skip selling puts when ML prob < this (default: 0.35)",
    )
    parser.add_argument(
        "--skip-call-thresh", type=float, default=0.65,
        help="Skip selling calls when ML prob > this (default: 0.65)",
    )

    # Walk-forward settings
    parser.add_argument(
        "--walk-forward-windows", type=int, default=5,
        help="Number of expanding windows for walk-forward (default: 5)",
    )
    parser.add_argument(
        "--top-models", type=int, default=3,
        help="Number of top models in ensemble (default: 3)",
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for artifacts",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory for Polygon options cache (default: data/options_cache/SPY)",
    )

    return parser


def _get_fridays_in_range(from_date: date, to_date: date) -> list[date]:
    """Return all Fridays between from_date and to_date inclusive."""
    fridays: list[date] = []
    current = from_date
    # Advance to first Friday
    while current.weekday() != 4:
        current += timedelta(days=1)
    while current <= to_date:
        fridays.append(current)
        current += timedelta(days=7)
    return fridays


def _get_previous_trading_day(d: date) -> date:
    """Return the Monday before a Friday (entry day for weekly options)."""
    # Friday - 4 = Monday
    monday = d - timedelta(days=4)
    return monday


def _print_results_table(
    ml_metrics: dict[str, float] | None,
    mech_metrics: dict[str, float],
    bh_return_pct: float,
) -> None:
    """Print side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("SPY WEEKLY WHEEL BACKTEST RESULTS")
    print("=" * 70)

    headers = ["Metric"]
    if ml_metrics:
        headers.append("ML Wheel")
    headers.extend(["Mechanical", "Buy & Hold"])

    def _row(label: str, ml_val: str, mech_val: str, bh_val: str) -> None:
        parts = [f"  {label:<24}"]
        if ml_metrics:
            parts.append(f"{ml_val:>14}")
        parts.extend([f"{mech_val:>14}", f"{bh_val:>14}"])
        print("".join(parts))

    header_line = f"  {'':24}"
    if ml_metrics:
        header_line += f"{'ML Wheel':>14}"
    header_line += f"{'Mechanical':>14}{'Buy & Hold':>14}"
    print(header_line)
    print("  " + "-" * (66 if ml_metrics else 52))

    ml = ml_metrics or {}

    _row(
        "Total Return",
        f"+{ml.get('total_return_pct', 0):.1f}%" if ml else "",
        f"+{mech_metrics['total_return_pct']:.1f}%",
        f"+{bh_return_pct:.1f}%",
    )
    _row(
        "Annualized Return",
        f"+{ml.get('annualized_return_pct', 0):.1f}%" if ml else "",
        f"+{mech_metrics['annualized_return_pct']:.1f}%",
        "N/A",
    )
    _row(
        "Premium Collected",
        f"${ml.get('total_premium_collected', 0):,.0f}" if ml else "",
        f"${mech_metrics['total_premium_collected']:,.0f}",
        "N/A",
    )
    _row(
        "Assignments",
        f"{int(ml.get('n_assignments', 0))}" if ml else "",
        f"{int(mech_metrics['n_assignments'])}",
        "N/A",
    )
    _row(
        "Calls Exercised",
        f"{int(ml.get('n_calls_exercised', 0))}" if ml else "",
        f"{int(mech_metrics['n_calls_exercised'])}",
        "N/A",
    )
    if ml:
        _row(
            "Weeks Skipped (ML)",
            f"{int(ml.get('n_puts_skipped', 0) + ml.get('n_calls_skipped', 0))}",
            "N/A",
            "N/A",
        )
    _row(
        "Sharpe",
        f"{ml.get('sharpe', 0):.2f}" if ml else "",
        f"{mech_metrics['sharpe']:.2f}",
        "N/A",
    )
    _row(
        "Max Drawdown",
        f"-{ml.get('max_drawdown_pct', 0):.1f}%" if ml else "",
        f"-{mech_metrics['max_drawdown_pct']:.1f}%",
        "N/A",
    )
    print("=" * 70)


def main(argv: list[str] | None = None) -> None:
    """Run the SPY Weekly Wheel pipeline."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    init_ml_env(args.verbose)

    import numpy as np

    from stockdownloader.analysis.options.pricing import (
        estimate_volatility,
    )
    from stockdownloader.backtesting.engines.wheel import (
        WeekRecord,
        WheelBacktestEngine,
    )
    from stockdownloader.data.market.polygon_options_client import (
        PolygonOptionsClient,
        load_chain_cache,
        save_chain_cache,
        select_strike_by_delta,
    )
    from stockdownloader.data.market.yahoo_data_client import YahooDataClient
    from stockdownloader.ml.dataset_builder import DatasetBuilder, LabelConfig
    from stockdownloader.ml.feature_extractor import FeatureExtractor

    t0 = time.time()

    grid = _QUICK_GRID if args.quick else _FULL_GRID
    to_date_str = args.to_date or str(date.today())

    output_dir = Path(args.output_dir) if args.output_dir else (
        DEFAULT_ML_PIPELINE_DIR / "spy_wheel"
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir else (
        Path("data/options_cache/SPY")
    )

    print("=" * 60)
    print("SPY ML-GUIDED WEEKLY WHEEL PIPELINE")
    print("=" * 60)
    print(f"  Delta:         {args.delta}")
    print(f"  Contracts:     {args.contracts}")
    print(f"  Capital:       ${args.initial_capital:,.0f}")
    print(f"  Period:        {args.from_date} to {to_date_str}")
    print(f"  ML Filter:     {'OFF' if args.no_ml_filter else 'ON'}")
    if not args.no_ml_filter:
        print(f"  Skip put <     {args.skip_put_thresh}")
        print(f"  Skip call >    {args.skip_call_thresh}")
    print(f"  Cache:         {cache_dir}")
    print()

    # ------------------------------------------------------------------
    # [1/5] Download SPY daily data
    # ------------------------------------------------------------------
    print("[1/5] Downloading SPY daily data (10y)...")
    t1 = time.time()

    yahoo = YahooDataClient()
    daily_data = yahoo.fetch_price_data("SPY", range_="10y")

    if len(daily_data) < 500:
        print(f"ERROR: Insufficient data ({len(daily_data)} bars)", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(daily_data)} bars ({daily_data[0].date} to {daily_data[-1].date})")
    print(f"  [{time.time() - t1:.1f}s]")

    # Build date-to-close map
    date_close: dict[str, float] = {}
    for bar in daily_data:
        date_close[bar.date] = float(bar.close)

    # ------------------------------------------------------------------
    # [2/5] ML walk-forward predictions
    # ------------------------------------------------------------------
    ml_probs: dict[str, float] = {}  # date -> probability

    if not args.no_ml_filter:
        print("\n[2/5] Running ML walk-forward ensemble...")
        t2 = time.time()

        from stockdownloader.app.spy_ml_ensemble import (
            _generate_walk_forward_predictions,
        )

        label_config = LabelConfig(
            forward_period=int(args.forward_periods.split(",")[0])
            if hasattr(args, "forward_periods") else 10,
            profit_threshold=float(args.profit_thresholds.split(",")[0])
            if hasattr(args, "profit_thresholds") else 0.005,
        )
        extractor = FeatureExtractor()
        builder = DatasetBuilder(extractor, label_config)
        dataset = builder.build(daily_data)

        oos_preds, oos_dates, n_windows = _generate_walk_forward_predictions(
            dataset, daily_data, grid,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            top_models=args.top_models,
            use_direct_ensemble=True,
            n_windows=args.walk_forward_windows,
        )

        for dt, prob in zip(oos_dates, oos_preds):
            ml_probs[dt] = float(prob)

        print(f"  {len(ml_probs)} daily predictions generated")
        print(f"  [{time.time() - t2:.1f}s]")
    else:
        print("\n[2/5] Skipped (--no-ml-filter)")

    # ------------------------------------------------------------------
    # [3/5] Download option chain data from Polygon
    # ------------------------------------------------------------------
    print("\n[3/5] Downloading option chain data from Polygon...")
    t3 = time.time()

    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    to_dt = datetime.strptime(to_date_str, "%Y-%m-%d").date()
    fridays = _get_fridays_in_range(from_dt, to_dt)

    polygon = PolygonOptionsClient()

    # Pre-fetch all contracts for the date range (paginated, cached)
    all_contracts = polygon.fetch_option_contracts(
        "SPY", from_date=from_dt, to_date=to_dt, expired=True,
    )
    print(f"  {len(all_contracts)} total contracts discovered")

    # Group contracts by expiration
    contracts_by_exp: dict[str, list[dict]] = {}
    for c in all_contracts:
        exp = c.get("expiration_date", "")
        contracts_by_exp.setdefault(exp, []).append(c)

    print(f"  {len(fridays)} Fridays in range")
    print(f"  [{time.time() - t3:.1f}s]")

    # ------------------------------------------------------------------
    # [4/5] Build weekly schedule and run wheel backtest
    # ------------------------------------------------------------------
    print("\n[4/5] Running wheel backtest...")
    t4 = time.time()

    # Estimate historical volatility
    close_prices = [Decimal(str(bar.close)) for bar in daily_data]
    hist_vol = float(estimate_volatility(close_prices, 20))

    week_records: list[WeekRecord] = []
    n_cached = 0
    n_fetched = 0

    for week_num, friday in enumerate(fridays):
        exp_str = str(friday)
        monday = _get_previous_trading_day(friday)
        mon_str = str(monday)

        spy_at_entry = date_close.get(mon_str)
        spy_at_expiry = date_close.get(exp_str)

        if spy_at_entry is None or spy_at_expiry is None:
            continue  # Skip weeks where we don't have SPY price data

        # Get contracts for this expiration
        week_contracts = contracts_by_exp.get(exp_str, [])
        if not week_contracts:
            continue

        # Select put and call at target delta
        put_contract = select_strike_by_delta(
            week_contracts, contract_type="put", spot=spy_at_entry,
            target_delta=args.delta, days_to_expiry=5, volatility=hist_vol,
        )
        call_contract = select_strike_by_delta(
            week_contracts, contract_type="call", spot=spy_at_entry,
            target_delta=args.delta, days_to_expiry=5, volatility=hist_vol,
        )

        if put_contract is None or call_contract is None:
            continue

        # Get real option prices from Polygon (with cache)
        cached = load_chain_cache(cache_dir, exp_str)
        if cached and put_contract["ticker"] in cached.get("bars", {}):
            put_bar = cached["bars"].get(put_contract["ticker"])
            call_bar = cached["bars"].get(call_contract["ticker"])
            n_cached += 1
        else:
            put_bar = polygon.fetch_option_daily_bar(
                put_contract["ticker"], monday,
            )
            call_bar = polygon.fetch_option_daily_bar(
                call_contract["ticker"], monday,
            )
            # Save to cache
            bars_data = {}
            if put_bar:
                bars_data[put_contract["ticker"]] = put_bar
            if call_bar:
                bars_data[call_contract["ticker"]] = call_bar
            cache_data = {
                "expiration_date": exp_str,
                "contracts": week_contracts,
                "bars": bars_data,
            }
            save_chain_cache(cache_dir, exp_str, cache_data)
            n_fetched += 1

        put_premium = put_bar["c"] if put_bar else 0.0
        call_premium = call_bar["c"] if call_bar else 0.0

        if put_premium <= 0 and call_premium <= 0:
            continue

        # Get ML probability (use Friday's prediction for Monday's decision)
        # Find the most recent prediction on or before Monday
        ml_prob = 0.50  # neutral default
        if ml_probs:
            for lookback in range(5):
                check_date = str(monday - timedelta(days=lookback))
                if check_date in ml_probs:
                    ml_prob = ml_probs[check_date]
                    break

        week_records.append(WeekRecord(
            week_num=week_num,
            expiration_date=exp_str,
            entry_date=mon_str,
            spy_price_at_entry=spy_at_entry,
            spy_price_at_expiry=spy_at_expiry,
            put_strike=put_contract["strike_price"],
            call_strike=call_contract["strike_price"],
            put_premium=put_premium,
            call_premium=call_premium,
            ml_prob=ml_prob,
        ))

    print(f"  {len(week_records)} tradeable weeks built")
    print(f"  Cache hits: {n_cached}, API fetches: {n_fetched}")

    # Run mechanical wheel
    mech_engine = WheelBacktestEngine(
        initial_capital=args.initial_capital,
        contracts=args.contracts,
    )
    for w in week_records:
        mech_engine.process_week(w, use_ml_filter=False)
    mech_metrics = mech_engine.compute_metrics()

    # Run ML-filtered wheel (if enabled)
    ml_metrics = None
    if not args.no_ml_filter and ml_probs:
        ml_engine = WheelBacktestEngine(
            initial_capital=args.initial_capital,
            contracts=args.contracts,
            skip_put_thresh=args.skip_put_thresh,
            skip_call_thresh=args.skip_call_thresh,
        )
        for w in week_records:
            ml_engine.process_week(w, use_ml_filter=True)
        ml_metrics = ml_engine.compute_metrics()

    print(f"  [{time.time() - t4:.1f}s]")

    # ------------------------------------------------------------------
    # [5/5] Print results
    # ------------------------------------------------------------------
    # Buy-and-hold return
    bh_start = date_close.get(str(fridays[0])) if fridays else None
    bh_end = date_close.get(str(fridays[-1])) if fridays else None
    bh_return_pct = 0.0
    if bh_start and bh_end and bh_start > 0:
        bh_return_pct = ((bh_end - bh_start) / bh_start) * 100

    _print_results_table(ml_metrics, mech_metrics, bh_return_pct)

    print(f"\nTotal pipeline time: {time.time() - t0:.1f}s")
```

**Step 4: Run tests to verify they pass**

Run: `DYLD_LIBRARY_PATH=/Users/kelsayed/lib PYTHONPATH=src python -m pytest tests/app/test_spy_options_wheel.py -v`
Expected: All 7 parser tests PASS

**Step 5: Add console_scripts entry to pyproject.toml**

Add to the `[project.scripts]` section:
```
spy-options-wheel = "stockdownloader.app.spy_options_wheel:main"
```

**Step 6: Commit**

```bash
git add src/stockdownloader/app/spy_options_wheel.py tests/app/test_spy_options_wheel.py pyproject.toml
git commit -m "feat: add spy-options-wheel CLI pipeline with ML-filtered wheel strategy"
```

---

### Task 6: Full Test Suite Verification & Integration Smoke Test

**Files:**
- Modify: `tests/app/test_spy_options_wheel.py` (add integration test)

**Step 1: Add integration smoke test**

```python
# Add to tests/app/test_spy_options_wheel.py

from unittest.mock import MagicMock, patch
from stockdownloader.backtesting.engines.wheel import WeekRecord, WheelBacktestEngine


class TestWheelIntegration:
    """Integration smoke test for the wheel pipeline."""

    def test_full_wheel_lifecycle(self):
        """Run a complete wheel lifecycle: CSP -> assigned -> CC -> called away."""
        engine = WheelBacktestEngine(
            initial_capital=100_000.0,
            contracts=1,
            skip_put_thresh=0.35,
            skip_call_thresh=0.65,
        )

        weeks = [
            # Week 0: Sell put at 580, SPY closes at 605 (OTM) -> keep premium
            WeekRecord(0, "2025-02-07", "2025-02-03", 600.0, 605.0, 580.0, 620.0, 3.0, 2.0, 0.50),
            # Week 1: Sell put at 580, SPY drops to 570 (ITM) -> ASSIGNED
            WeekRecord(1, "2025-02-14", "2025-02-10", 600.0, 570.0, 580.0, 620.0, 4.0, 2.0, 0.50),
            # Week 2: Now in CALL_PHASE, sell call at 590, SPY at 575 (OTM) -> keep premium
            WeekRecord(2, "2025-02-21", "2025-02-17", 575.0, 575.0, 570.0, 590.0, 3.0, 2.5, 0.50),
            # Week 3: Sell call at 590, SPY rallies to 600 (ITM) -> CALLED AWAY
            WeekRecord(3, "2025-02-28", "2025-02-24", 580.0, 600.0, 570.0, 590.0, 3.0, 2.0, 0.50),
            # Week 4: Back to PUT_PHASE, sell put, SPY at 605 (OTM)
            WeekRecord(4, "2025-03-07", "2025-03-03", 600.0, 605.0, 580.0, 620.0, 3.0, 2.0, 0.50),
        ]

        for w in weeks:
            engine.process_week(w, use_ml_filter=True)

        metrics = engine.compute_metrics()

        # Verify lifecycle completed correctly
        assert metrics["n_assignments"] == 1  # Week 1
        assert metrics["n_calls_exercised"] == 1  # Week 3
        assert metrics["total_premium_collected"] > 0
        assert metrics["weeks"] == 5
        assert len(engine.equity_curve) == 5
        # Should have made money from premiums
        assert metrics["final_equity"] > metrics["initial_capital"]
```

**Step 2: Run the full test suite**

Run: `DYLD_LIBRARY_PATH=/Users/kelsayed/lib PYTHONPATH=src python -m pytest tests/ -x -q`
Expected: All tests PASS (existing 3498 + new ~25 = ~3523 total)

**Step 3: Commit**

```bash
git add tests/app/test_spy_options_wheel.py
git commit -m "test: add integration smoke test for wheel lifecycle"
```

---

### Task 7: Run Live Backtest

**Step 1: Run the wheel backtest with real data**

```bash
DYLD_LIBRARY_PATH=/Users/kelsayed/lib PYTHONPATH=src python -m stockdownloader.app.spy_options_wheel --quick --from-date 2022-01-01
```

**Step 2: Verify output shows the comparison table**

Expected: Side-by-side results for ML Wheel vs Mechanical vs Buy & Hold.

**Step 3: Commit any fixes needed**

If adjustments are needed based on real data, fix and commit.

---

## Summary of Tasks

| Task | Description | New Files | Tests |
|------|-------------|-----------|-------|
| 1 | Polygon options client — contract discovery | `polygon_options_client.py`, `test_polygon_options_client.py` | 5 |
| 2 | Daily bar fetching + disk cache | (modify above) | +4 |
| 3 | BS-delta strike selection | (modify above) | +3 |
| 4 | Wheel backtest engine — state machine | `wheel.py`, `test_wheel.py` | 9 |
| 5 | CLI pipeline + pyproject.toml | `spy_options_wheel.py`, `test_spy_options_wheel.py` | 7 |
| 6 | Integration smoke test + full suite | (modify tests) | +1 |
| 7 | Live backtest with real data | — | — |
| **Total** | | **6 new files** | **~29 tests** |
