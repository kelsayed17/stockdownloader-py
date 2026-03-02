# Snapshot OI Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add OI data to the GME options pipeline via Polygon snapshots + volume-based proxy backfill, unlocking 11 zeroed-out state metrics (GEX, OI flow, regime detection).

**Architecture:** New `SnapshotCollector` fetches current OI/greeks from Polygon's bulk snapshot endpoint. New `OIProxyEstimator` estimates historical OI using cumulative-volume decay. `OIEnricher` merges real + proxy OI into existing monthly Parquets. No changes to StateEngine, strategies, or fusion — they already consume `open_interest`.

**Tech Stack:** Python 3.11, pandas, requests, pytest, Parquet

**Test command:** `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/ -v --tb=short`

---

### Task 1: Add `fetch_options_chain_snapshot` to PolygonOptionsClient

**Files:**
- Modify: `src/stockdownloader/data/market/polygon_options_client.py`
- Test: `tests/gme/options/test_snapshot.py` (create)

**Context:** The client already has paginated fetching in `fetch_option_contracts` (lines 62-146). The snapshot endpoint `GET /v3/snapshot/options/{underlyingAsset}?limit=250` uses the same pagination pattern (`next_url`). Each result dict has `details`, `greeks`, `open_interest`, `implied_volatility`, `day`, and `underlying_asset` sub-objects. The `next_url` embeds `apiKey` which must be stripped since we use session-based Bearer auth.

**Step 1: Write the failing test**

Create `tests/gme/options/test_snapshot.py`:

```python
"""Tests for Polygon snapshot collector and OI enrichment."""
from __future__ import annotations

import pytest
from datetime import date
from unittest.mock import MagicMock, patch, PropertyMock

from stockdownloader.data.market.polygon_options_client import PolygonOptionsClient


def _make_snapshot_result(
    ticker: str = "O:GME260227C00025000",
    strike: float = 25.0,
    contract_type: str = "call",
    expiration: str = "2026-02-27",
    close: float = 0.04,
    volume: int = 9493,
    open_interest: int = 8921,
    iv: float = 0.57,
    delta: float = 0.11,
    gamma: float = 0.24,
    theta: float = -0.06,
    vega: float = 0.003,
    underlying_price: float = 23.975,
) -> dict:
    """Build a snapshot result dict matching Polygon's response shape."""
    return {
        "details": {
            "ticker": ticker,
            "strike_price": strike,
            "contract_type": contract_type,
            "expiration_date": expiration,
        },
        "day": {"close": close, "volume": volume},
        "open_interest": open_interest,
        "implied_volatility": iv,
        "greeks": {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
        },
        "underlying_asset": {"price": underlying_price},
    }


class TestFetchOptionsChainSnapshot:
    def test_returns_all_results_single_page(self):
        """Single page of results with no next_url."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                _make_snapshot_result("O:GME260227C00025000", open_interest=100),
                _make_snapshot_result("O:GME260227P00025000", contract_type="put", open_interest=200),
            ],
            "status": "OK",
        }
        mock_resp.raise_for_status = MagicMock()

        client = PolygonOptionsClient.__new__(PolygonOptionsClient)
        client._delay = 0.0
        client._session = MagicMock()
        client._session.get.return_value = mock_resp

        results = client.fetch_options_chain_snapshot("GME")
        assert len(results) == 2
        assert results[0]["open_interest"] == 100
        assert results[1]["open_interest"] == 200

    def test_paginates_through_next_url(self):
        """Follows next_url to collect all pages."""
        page1_resp = MagicMock()
        page1_resp.json.return_value = {
            "results": [_make_snapshot_result(open_interest=100)],
            "next_url": "https://api.polygon.io/v3/snapshot/options/GME?cursor=abc&apiKey=SECRET",
            "status": "OK",
        }
        page1_resp.raise_for_status = MagicMock()

        page2_resp = MagicMock()
        page2_resp.json.return_value = {
            "results": [_make_snapshot_result(open_interest=200)],
            "status": "OK",
        }
        page2_resp.raise_for_status = MagicMock()

        client = PolygonOptionsClient.__new__(PolygonOptionsClient)
        client._delay = 0.0
        client._session = MagicMock()
        client._session.get.side_effect = [page1_resp, page2_resp]

        results = client.fetch_options_chain_snapshot("GME")
        assert len(results) == 2
        assert results[0]["open_interest"] == 100
        assert results[1]["open_interest"] == 200

        # Verify apiKey was stripped from the next_url
        second_call_url = client._session.get.call_args_list[1][0][0]
        assert "apiKey" not in second_call_url
        assert "SECRET" not in second_call_url

    def test_handles_request_failure_gracefully(self):
        """Returns empty list on network failure."""
        import requests

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")

        client = PolygonOptionsClient.__new__(PolygonOptionsClient)
        client._delay = 0.0
        client._session = MagicMock()
        client._session.get.return_value = mock_resp

        results = client.fetch_options_chain_snapshot("GME")
        assert results == []

    def test_handles_missing_greeks(self):
        """Contracts with empty greeks are still included."""
        result = _make_snapshot_result()
        result["greeks"] = {}
        result.pop("implied_volatility", None)

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": [result], "status": "OK"}
        mock_resp.raise_for_status = MagicMock()

        client = PolygonOptionsClient.__new__(PolygonOptionsClient)
        client._delay = 0.0
        client._session = MagicMock()
        client._session.get.return_value = mock_resp

        results = client.fetch_options_chain_snapshot("GME")
        assert len(results) == 1
        assert results[0]["open_interest"] == 8921
```

**Step 2: Run test to verify it fails**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_snapshot.py -v --tb=short`

Expected: FAIL with `AttributeError: type object 'PolygonOptionsClient' has no attribute 'fetch_options_chain_snapshot'`

**Step 3: Write the implementation**

Add to `src/stockdownloader/data/market/polygon_options_client.py`, inside the `PolygonOptionsClient` class, after `fetch_option_daily_bars_range`:

```python
_SNAPSHOT_URL = "https://api.polygon.io/v3/snapshot/options/{underlying_asset}"
```

Add this constant after `_AGGS_URL` at the module level (line ~37).

Then add this method to the class:

```python
    def fetch_options_chain_snapshot(
        self,
        symbol: str,
        *,
        limit: int = 250,
    ) -> list[dict]:
        """Fetch snapshot data for all active option contracts.

        Returns the full chain with OI, greeks, and IV for each
        contract via paginated calls to the Polygon snapshot endpoint.

        Parameters
        ----------
        symbol:
            Underlying ticker (e.g. ``"GME"``).
        limit:
            Results per page (max 250).

        Returns
        -------
        list[dict]
            Raw snapshot result dicts from Polygon.
        """
        url = _SNAPSHOT_URL.format(underlying_asset=symbol.upper())
        params: dict = {"limit": limit}
        all_results: list[dict] = []

        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            all_results.extend(data.get("results", []))

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
                all_results.extend(data.get("results", []))
                next_url = data.get("next_url")

        except requests.RequestException as exc:
            logger.warning(
                "Polygon options snapshot request failed for %s: %s",
                symbol, exc,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning(
                "Polygon options snapshot parse error for %s: %s",
                symbol, exc,
            )

        logger.info(
            "Fetched %d option snapshots for %s",
            len(all_results), symbol,
        )
        return all_results
```

**Step 4: Run test to verify it passes**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_snapshot.py -v --tb=short`

Expected: 4 passed

**Step 5: Run full test suite**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/ -v --tb=short`

Expected: All pass (77 existing + 4 new = 81)

**Step 6: Commit**

```bash
git add src/stockdownloader/data/market/polygon_options_client.py tests/gme/options/test_snapshot.py
git commit -m "feat(options): add fetch_options_chain_snapshot to PolygonOptionsClient

Paginated bulk snapshot endpoint for OI, greeks, and IV.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Create SnapshotCollector

**Files:**
- Create: `src/stockdownloader/gme/options/snapshot.py`
- Test: `tests/gme/options/test_snapshot.py` (append)

**Context:** `SnapshotCollector` transforms raw Polygon snapshot dicts into a flat Parquet file. It uses the client method from Task 1. The output schema must match the bar schema from `fetcher.py` for downstream compatibility: `date`, `option_ticker`, `strike`, `option_type`, `expiration`, `close`, `volume`, `open_interest`. Plus snapshot-specific columns: `implied_volatility`, `delta`, `gamma`, `theta`, `vega`, `underlying_price`.

**Step 1: Write the failing tests**

Append to `tests/gme/options/test_snapshot.py`:

```python
from stockdownloader.gme.options.snapshot import SnapshotCollector
from stockdownloader.gme.options.config import GMEOptionsConfig


class TestSnapshotCollector:
    def test_collect_produces_dataframe(self, tmp_path):
        """collect() returns a DataFrame with correct schema."""
        mock_client = MagicMock()
        mock_client.fetch_options_chain_snapshot.return_value = [
            _make_snapshot_result("O:GME260227C00025000", open_interest=100, volume=500),
            _make_snapshot_result("O:GME260227P00020000", contract_type="put",
                                  strike=20.0, open_interest=200, volume=300),
        ]

        cfg = GMEOptionsConfig(data_dir=tmp_path)
        collector = SnapshotCollector(cfg, mock_client)
        df = collector.collect()

        assert len(df) == 2
        required_cols = {
            "date", "option_ticker", "strike", "option_type", "expiration",
            "close", "volume", "open_interest", "implied_volatility",
            "delta", "gamma", "theta", "vega", "underlying_price",
        }
        assert required_cols.issubset(set(df.columns))
        assert df.iloc[0]["open_interest"] == 100
        assert df.iloc[1]["option_type"] == "put"

    def test_collect_handles_missing_greeks(self, tmp_path):
        """Contracts with empty greeks get NaN for greek columns."""
        import math

        result = _make_snapshot_result()
        result["greeks"] = {}
        result.pop("implied_volatility", None)

        mock_client = MagicMock()
        mock_client.fetch_options_chain_snapshot.return_value = [result]

        cfg = GMEOptionsConfig(data_dir=tmp_path)
        collector = SnapshotCollector(cfg, mock_client)
        df = collector.collect()

        assert len(df) == 1
        assert math.isnan(df.iloc[0]["delta"])
        assert math.isnan(df.iloc[0]["implied_volatility"])

    def test_save_and_load(self, tmp_path):
        """save() writes Parquet, load() reads it back."""
        import pandas as pd

        mock_client = MagicMock()
        mock_client.fetch_options_chain_snapshot.return_value = [
            _make_snapshot_result(open_interest=42),
        ]

        cfg = GMEOptionsConfig(data_dir=tmp_path)
        collector = SnapshotCollector(cfg, mock_client)
        df = collector.collect()

        path = collector.save(df, date(2026, 2, 26))
        assert path.exists()
        assert "2026-02-26" in path.name

        loaded = collector.load(date(2026, 2, 26))
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded.iloc[0]["open_interest"] == 42

    def test_load_returns_none_when_missing(self, tmp_path):
        """load() returns None for dates with no snapshot."""
        cfg = GMEOptionsConfig(data_dir=tmp_path)
        collector = SnapshotCollector(cfg, MagicMock())
        assert collector.load(date(1999, 1, 1)) is None

    def test_run_collects_and_saves(self, tmp_path):
        """run() collects and saves in one call."""
        mock_client = MagicMock()
        mock_client.fetch_options_chain_snapshot.return_value = [
            _make_snapshot_result(open_interest=77),
        ]

        cfg = GMEOptionsConfig(data_dir=tmp_path)
        collector = SnapshotCollector(cfg, mock_client)
        df = collector.run()

        assert len(df) == 1
        # Verify file was saved
        snapshots_dir = tmp_path / "snapshots"
        assert any(snapshots_dir.glob("*.parquet"))
```

**Step 2: Run test to verify it fails**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_snapshot.py::TestSnapshotCollector -v --tb=short`

Expected: FAIL with `ModuleNotFoundError: No module named 'stockdownloader.gme.options.snapshot'`

**Step 3: Write the implementation**

Create `src/stockdownloader/gme/options/snapshot.py`:

```python
"""Snapshot-based options chain collector with OI, greeks, and IV.

Fetches the current-day options chain snapshot from Polygon's bulk
endpoint and saves per-contract data as daily Parquet files.

Usage::

    from stockdownloader.gme.options.snapshot import SnapshotCollector

    collector = SnapshotCollector(config, client)
    df = collector.run()  # collect + save
"""
from __future__ import annotations

import logging
import math
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from stockdownloader.data.market.polygon_options_client import (
        PolygonOptionsClient,
    )
    from stockdownloader.gme.options.config import GMEOptionsConfig

logger = logging.getLogger(__name__)


class SnapshotCollector:
    """Fetches and stores daily options chain snapshots.

    Parameters
    ----------
    config:
        Pipeline configuration (data directory).
    client:
        Polygon API client with ``fetch_options_chain_snapshot``.
    """

    def __init__(
        self,
        config: GMEOptionsConfig,
        client: PolygonOptionsClient,
    ) -> None:
        self.config = config
        self.client = client

    @property
    def _snapshots_dir(self) -> Path:
        return self.config.data_dir / "snapshots"

    def collect(self, collection_date: date | None = None) -> pd.DataFrame:
        """Fetch the full options chain snapshot and return as DataFrame.

        Parameters
        ----------
        collection_date:
            Date to stamp on each row.  Defaults to today.

        Returns
        -------
        DataFrame with one row per contract.
        """
        import pandas as pd

        if collection_date is None:
            collection_date = date.today()

        raw = self.client.fetch_options_chain_snapshot(self.config.symbol)

        rows: list[dict] = []
        for snap in raw:
            details = snap.get("details", {})
            day = snap.get("day", {})
            greeks = snap.get("greeks", {})
            underlying = snap.get("underlying_asset", {})

            rows.append({
                "date": str(collection_date),
                "option_ticker": details.get("ticker", ""),
                "strike": details.get("strike_price", 0.0),
                "option_type": details.get("contract_type", ""),
                "expiration": details.get("expiration_date", ""),
                "close": day.get("close", 0.0),
                "volume": day.get("volume", 0),
                "open_interest": snap.get("open_interest", 0),
                "implied_volatility": snap.get("implied_volatility", float("nan")),
                "delta": greeks.get("delta", float("nan")),
                "gamma": greeks.get("gamma", float("nan")),
                "theta": greeks.get("theta", float("nan")),
                "vega": greeks.get("vega", float("nan")),
                "underlying_price": underlying.get("price", 0.0),
            })

        df = pd.DataFrame(rows)
        logger.info(
            "Collected snapshot: %d contracts for %s on %s",
            len(df), self.config.symbol, collection_date,
        )
        return df

    def save(self, df: pd.DataFrame, collection_date: date) -> Path:
        """Save a snapshot DataFrame as a dated Parquet file.

        Returns the path to the written file.
        """
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        path = self._snapshots_dir / f"{collection_date}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved snapshot (%d rows) to %s", len(df), path)
        return path

    def load(self, collection_date: date) -> pd.DataFrame | None:
        """Load a snapshot Parquet file for a given date.

        Returns ``None`` if no snapshot exists for that date.
        """
        import pandas as pd

        path = self._snapshots_dir / f"{collection_date}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def run(self, collection_date: date | None = None) -> pd.DataFrame:
        """Collect the snapshot and save to disk in one call.

        Returns the collected DataFrame.
        """
        if collection_date is None:
            collection_date = date.today()

        df = self.collect(collection_date)
        if not df.empty:
            self.save(df, collection_date)
        return df
```

**Step 4: Run test to verify it passes**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_snapshot.py -v --tb=short`

Expected: 9 passed (4 client + 5 collector)

**Step 5: Commit**

```bash
git add src/stockdownloader/gme/options/snapshot.py tests/gme/options/test_snapshot.py
git commit -m "feat(options): add SnapshotCollector for daily OI/greeks

Collects full options chain snapshot from Polygon, saves as
dated Parquet with OI, IV, and greeks per contract.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: Create OIProxyEstimator

**Files:**
- Create: `src/stockdownloader/gme/options/oi_proxy.py`
- Create: `tests/gme/options/test_oi_proxy.py`

**Context:** The proxy estimates OI from volume using: `estimated_oi = cumulative_volume × exp(-λ × days_since_first_bar) × calibration_ratio`. The calibration ratio is learned from real snapshot data: `ratio = mean(real_oi / cumulative_volume)` for contracts where both are available. Default ratio = 0.15 when no calibration data exists. The decay rate λ defaults to 0.03/day (configurable via `config.oi_decay_rate`).

**Step 1: Write the failing tests**

Create `tests/gme/options/test_oi_proxy.py`:

```python
"""Tests for OI proxy estimation and enrichment."""
from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
import pytest
from datetime import date
from pathlib import Path

from stockdownloader.gme.options.oi_proxy import OIProxyEstimator, OIEnricher
from stockdownloader.gme.options.config import GMEOptionsConfig


class TestOIProxyEstimator:
    def test_calibrate_computes_ratio(self):
        """calibrate() learns OI/cumulative-volume ratio from snapshot."""
        snapshot_df = pd.DataFrame([
            {"option_ticker": "O:GME260227C00025000", "open_interest": 100, "volume": 50},
            {"option_ticker": "O:GME260227P00025000", "open_interest": 200, "volume": 80},
        ])
        # Simulate bars with cumulative volume
        bars_df = pd.DataFrame([
            {"option_ticker": "O:GME260227C00025000", "volume": 200, "date": "2026-02-26"},
            {"option_ticker": "O:GME260227P00025000", "volume": 400, "date": "2026-02-26"},
        ])

        estimator = OIProxyEstimator()
        estimator.calibrate(snapshot_df, bars_df)

        # ratio = mean(100/200, 200/400) = mean(0.5, 0.5) = 0.5
        assert abs(estimator.calibration_ratio - 0.5) < 0.01

    def test_calibrate_uses_default_when_no_data(self):
        """calibrate() uses default ratio when no matching data."""
        estimator = OIProxyEstimator()
        estimator.calibrate(pd.DataFrame(), pd.DataFrame())
        assert estimator.calibration_ratio == estimator.DEFAULT_RATIO

    def test_estimate_produces_nonzero_oi(self):
        """estimate() fills open_interest with proxy values."""
        bars_df = pd.DataFrame([
            {"option_ticker": "A", "date": "2023-01-03", "volume": 100, "open_interest": 0},
            {"option_ticker": "A", "date": "2023-01-04", "volume": 150, "open_interest": 0},
            {"option_ticker": "A", "date": "2023-01-05", "volume": 200, "open_interest": 0},
        ])
        estimator = OIProxyEstimator(calibration_ratio=0.5, decay_rate=0.0)
        result = estimator.estimate(bars_df)

        # With decay=0, OI = cumulative_volume * ratio
        # Day 1: cumvol=100, oi=50
        # Day 2: cumvol=250, oi=125
        # Day 3: cumvol=450, oi=225
        assert result.iloc[0]["open_interest"] == 50
        assert result.iloc[1]["open_interest"] == 125
        assert result.iloc[2]["open_interest"] == 225

    def test_estimate_applies_decay(self):
        """Exponential decay reduces OI for older volume."""
        bars_df = pd.DataFrame([
            {"option_ticker": "A", "date": "2023-01-03", "volume": 1000, "open_interest": 0},
            {"option_ticker": "A", "date": "2023-01-04", "volume": 0, "open_interest": 0},
            {"option_ticker": "A", "date": "2023-01-05", "volume": 0, "open_interest": 0},
        ])
        estimator = OIProxyEstimator(calibration_ratio=1.0, decay_rate=0.1)
        result = estimator.estimate(bars_df)

        oi_day1 = result.iloc[0]["open_interest"]
        oi_day3 = result.iloc[2]["open_interest"]

        # OI should decrease over time due to decay
        assert oi_day3 < oi_day1
        assert oi_day3 > 0

    def test_estimate_handles_multiple_contracts(self):
        """estimate() computes OI independently per contract."""
        bars_df = pd.DataFrame([
            {"option_ticker": "A", "date": "2023-01-03", "volume": 100, "open_interest": 0},
            {"option_ticker": "B", "date": "2023-01-03", "volume": 500, "open_interest": 0},
        ])
        estimator = OIProxyEstimator(calibration_ratio=0.5, decay_rate=0.0)
        result = estimator.estimate(bars_df)

        a_oi = result[result["option_ticker"] == "A"].iloc[0]["open_interest"]
        b_oi = result[result["option_ticker"] == "B"].iloc[0]["open_interest"]
        assert a_oi == 50
        assert b_oi == 250

    def test_save_and_load_calibration(self, tmp_path):
        """Calibration persists to JSON and reloads."""
        estimator = OIProxyEstimator(calibration_ratio=0.42)
        estimator.save_calibration(tmp_path / "cal.json")

        loaded = OIProxyEstimator.load_calibration(tmp_path / "cal.json")
        assert abs(loaded.calibration_ratio - 0.42) < 0.001
```

**Step 2: Run test to verify it fails**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_oi_proxy.py::TestOIProxyEstimator -v --tb=short`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/stockdownloader/gme/options/oi_proxy.py`:

```python
"""OI proxy estimation and enrichment for historical options data.

Estimates open interest from volume data using exponential-decay
cumulative volume, calibrated against real snapshot OI where available.

Usage::

    estimator = OIProxyEstimator(calibration_ratio=0.5, decay_rate=0.03)
    enriched = estimator.estimate(bars_df)
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class OIProxyEstimator:
    """Estimates open interest from volume using exponential-decay proxy.

    Parameters
    ----------
    calibration_ratio:
        Multiplier applied to cumulative volume to estimate OI.
        Learned from real snapshot data via :meth:`calibrate`.
    decay_rate:
        Exponential decay rate per day (lambda).  Higher values
        mean OI decays faster as volume ages.
    """

    DEFAULT_RATIO: float = 0.15

    def __init__(
        self,
        calibration_ratio: float | None = None,
        decay_rate: float = 0.03,
    ) -> None:
        self.calibration_ratio = calibration_ratio or self.DEFAULT_RATIO
        self.decay_rate = decay_rate

    def calibrate(
        self,
        snapshot_df: pd.DataFrame,
        bars_df: pd.DataFrame,
    ) -> None:
        """Learn calibration ratio from real snapshot OI vs cumulative volume.

        Parameters
        ----------
        snapshot_df:
            Snapshot data with ``option_ticker``, ``open_interest``.
        bars_df:
            Bar data with ``option_ticker``, ``volume``, ``date``.
        """
        if snapshot_df.empty or bars_df.empty:
            logger.info(
                "No calibration data available, using default ratio %.3f",
                self.DEFAULT_RATIO,
            )
            self.calibration_ratio = self.DEFAULT_RATIO
            return

        # Compute cumulative volume per contract from bars
        cum_vol = (
            bars_df.groupby("option_ticker")["volume"]
            .sum()
            .rename("cum_volume")
        )

        # Join with snapshot OI
        snap_oi = snapshot_df.set_index("option_ticker")["open_interest"]
        joined = pd.concat([snap_oi, cum_vol], axis=1).dropna()
        joined = joined[joined["cum_volume"] > 0]

        if joined.empty:
            self.calibration_ratio = self.DEFAULT_RATIO
            return

        ratios = joined["open_interest"] / joined["cum_volume"]
        self.calibration_ratio = float(ratios.mean())
        logger.info("Calibrated OI ratio: %.4f from %d contracts", self.calibration_ratio, len(joined))

    def estimate(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """Add estimated open_interest to bars DataFrame.

        Computes: ``oi = cumsum(volume * exp(-decay * days_since)) * ratio``

        Parameters
        ----------
        bars_df:
            Must have ``option_ticker``, ``date``, ``volume``, ``open_interest``.

        Returns
        -------
        Copy of bars_df with ``open_interest`` replaced by proxy values.
        """
        df = bars_df.copy()

        if df.empty:
            return df

        results: list[pd.DataFrame] = []

        for ticker, group in df.groupby("option_ticker"):
            g = group.sort_values("date").copy()
            dates = pd.to_datetime(g["date"])
            first_date = dates.iloc[0]
            days_elapsed = (dates - first_date).dt.days.values.astype(float)

            volumes = g["volume"].values.astype(float)

            # Compute decayed cumulative volume
            n = len(volumes)
            oi_values = np.zeros(n)

            for i in range(n):
                # Sum all prior volume with decay from their age
                decayed_sum = 0.0
                for j in range(i + 1):
                    age = days_elapsed[i] - days_elapsed[j]
                    decayed_sum += volumes[j] * math.exp(-self.decay_rate * age)
                oi_values[i] = decayed_sum * self.calibration_ratio

            g["open_interest"] = oi_values.astype(int)
            results.append(g)

        return pd.concat(results, ignore_index=True)

    def save_calibration(self, path: Path) -> None:
        """Save calibration parameters to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "calibration_ratio": self.calibration_ratio,
                    "decay_rate": self.decay_rate,
                },
                f,
                indent=2,
            )

    @classmethod
    def load_calibration(cls, path: Path) -> OIProxyEstimator:
        """Load calibration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            calibration_ratio=data["calibration_ratio"],
            decay_rate=data.get("decay_rate", 0.03),
        )


class OIEnricher:
    """Merges real and proxy OI into monthly bar Parquet files.

    Priority cascade:
    1. Snapshot OI (from ``snapshots/`` directory)
    2. Proxy OI (from ``OIProxyEstimator``)

    Parameters
    ----------
    config:
        Pipeline configuration (data directories).
    estimator:
        Calibrated OI proxy estimator.
    """

    def __init__(
        self,
        data_dir: Path,
        estimator: OIProxyEstimator,
    ) -> None:
        self.data_dir = data_dir
        self.estimator = estimator

    def enrich_month(self, month_path: Path) -> pd.DataFrame:
        """Enrich a single monthly Parquet with OI data.

        Parameters
        ----------
        month_path:
            Path to a monthly Parquet file (e.g. ``monthly/2023-01.parquet``).

        Returns
        -------
        Enriched DataFrame with ``open_interest`` and ``oi_source`` columns.
        """
        df = pd.read_parquet(month_path)

        if df.empty:
            df["oi_source"] = pd.Series(dtype=str)
            return df

        # Step 1: Apply proxy OI to all rows
        df = self.estimator.estimate(df)
        df["oi_source"] = "proxy"

        # Step 2: Overlay real snapshot OI where available
        snapshots_dir = self.data_dir / "snapshots"
        if snapshots_dir.exists():
            for snap_file in snapshots_dir.glob("*.parquet"):
                snap_date = snap_file.stem  # YYYY-MM-DD
                snap_df = pd.read_parquet(snap_file)

                if snap_df.empty:
                    continue

                # Build lookup: (date, option_ticker) → open_interest
                snap_lookup = {}
                for _, row in snap_df.iterrows():
                    key = (str(row.get("date", snap_date))[:10], row["option_ticker"])
                    snap_lookup[key] = row["open_interest"]

                # Overlay
                for idx, row in df.iterrows():
                    key = (str(row["date"])[:10], row["option_ticker"])
                    if key in snap_lookup:
                        df.at[idx, "open_interest"] = snap_lookup[key]
                        df.at[idx, "oi_source"] = "snapshot"

        return df

    def enrich_all(self) -> int:
        """Enrich all monthly Parquet files in place.

        Returns the number of files enriched.
        """
        monthly_dir = self.data_dir / "monthly"
        if not monthly_dir.exists():
            logger.warning("No monthly directory at %s", monthly_dir)
            return 0

        count = 0
        for pf in sorted(monthly_dir.glob("*.parquet")):
            logger.info("Enriching %s", pf.name)
            enriched = self.enrich_month(pf)
            enriched.to_parquet(pf, index=False)
            count += 1

            proxy_count = (enriched["oi_source"] == "proxy").sum()
            snap_count = (enriched["oi_source"] == "snapshot").sum()
            logger.info(
                "  %s: %d rows (proxy=%d, snapshot=%d)",
                pf.name, len(enriched), proxy_count, snap_count,
            )

        return count
```

**Step 4: Run test to verify it passes**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_oi_proxy.py::TestOIProxyEstimator -v --tb=short`

Expected: 6 passed

**Step 5: Commit**

```bash
git add src/stockdownloader/gme/options/oi_proxy.py tests/gme/options/test_oi_proxy.py
git commit -m "feat(options): add OIProxyEstimator and OIEnricher

Volume-based OI proxy with exponential decay, calibrated from
snapshot data. OIEnricher merges real + proxy OI into monthly
Parquets with oi_source tracking.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Add OIEnricher Tests

**Files:**
- Modify: `tests/gme/options/test_oi_proxy.py` (append)

**Step 1: Write the failing tests**

Append to `tests/gme/options/test_oi_proxy.py`:

```python
class TestOIEnricher:
    def test_enrich_month_adds_proxy_oi(self, tmp_path):
        """enrich_month() populates OI with proxy values."""
        monthly_dir = tmp_path / "monthly"
        monthly_dir.mkdir()

        bars = pd.DataFrame([
            {"option_ticker": "A", "date": "2023-01-03", "volume": 100,
             "open_interest": 0, "strike": 25.0, "option_type": "call",
             "expiration": "2023-01-20", "open": 2.0, "high": 2.5,
             "low": 1.5, "close": 2.0, "vwap": 2.0},
        ])
        bars.to_parquet(monthly_dir / "2023-01.parquet", index=False)

        estimator = OIProxyEstimator(calibration_ratio=0.5, decay_rate=0.0)
        enricher = OIEnricher(data_dir=tmp_path, estimator=estimator)
        result = enricher.enrich_month(monthly_dir / "2023-01.parquet")

        assert result.iloc[0]["open_interest"] == 50
        assert result.iloc[0]["oi_source"] == "proxy"

    def test_snapshot_oi_overrides_proxy(self, tmp_path):
        """Snapshot OI takes priority over proxy OI."""
        monthly_dir = tmp_path / "monthly"
        monthly_dir.mkdir()
        snapshots_dir = tmp_path / "snapshots"
        snapshots_dir.mkdir()

        bars = pd.DataFrame([
            {"option_ticker": "O:GME230120C00025000", "date": "2023-01-20",
             "volume": 100, "open_interest": 0, "strike": 25.0,
             "option_type": "call", "expiration": "2023-01-20",
             "open": 2.0, "high": 2.5, "low": 1.5, "close": 2.0, "vwap": 2.0},
        ])
        bars.to_parquet(monthly_dir / "2023-01.parquet", index=False)

        # Snapshot for that exact date + ticker
        snap = pd.DataFrame([{
            "date": "2023-01-20",
            "option_ticker": "O:GME230120C00025000",
            "open_interest": 9999,
            "strike": 25.0,
            "option_type": "call",
        }])
        snap.to_parquet(snapshots_dir / "2023-01-20.parquet", index=False)

        estimator = OIProxyEstimator(calibration_ratio=0.5, decay_rate=0.0)
        enricher = OIEnricher(data_dir=tmp_path, estimator=estimator)
        result = enricher.enrich_month(monthly_dir / "2023-01.parquet")

        assert result.iloc[0]["open_interest"] == 9999
        assert result.iloc[0]["oi_source"] == "snapshot"

    def test_enrich_all_processes_multiple_files(self, tmp_path):
        """enrich_all() processes all monthly Parquet files."""
        monthly_dir = tmp_path / "monthly"
        monthly_dir.mkdir()

        for month in ["2023-01", "2023-02"]:
            bars = pd.DataFrame([
                {"option_ticker": "A", "date": f"{month}-15", "volume": 100,
                 "open_interest": 0, "strike": 25.0, "option_type": "call",
                 "expiration": f"{month}-20", "open": 2.0, "high": 2.5,
                 "low": 1.5, "close": 2.0, "vwap": 2.0},
            ])
            bars.to_parquet(monthly_dir / f"{month}.parquet", index=False)

        estimator = OIProxyEstimator(calibration_ratio=0.5, decay_rate=0.0)
        enricher = OIEnricher(data_dir=tmp_path, estimator=estimator)
        count = enricher.enrich_all()

        assert count == 2

        # Verify files were updated in place
        for month in ["2023-01", "2023-02"]:
            df = pd.read_parquet(monthly_dir / f"{month}.parquet")
            assert "oi_source" in df.columns
            assert df.iloc[0]["open_interest"] > 0

    def test_oi_source_column_values(self, tmp_path):
        """oi_source is exactly 'proxy' or 'snapshot'."""
        monthly_dir = tmp_path / "monthly"
        monthly_dir.mkdir()

        bars = pd.DataFrame([
            {"option_ticker": "A", "date": "2023-01-03", "volume": 100,
             "open_interest": 0, "strike": 25.0, "option_type": "call",
             "expiration": "2023-01-20", "open": 2.0, "high": 2.5,
             "low": 1.5, "close": 2.0, "vwap": 2.0},
        ])
        bars.to_parquet(monthly_dir / "2023-01.parquet", index=False)

        estimator = OIProxyEstimator(calibration_ratio=0.5, decay_rate=0.0)
        enricher = OIEnricher(data_dir=tmp_path, estimator=estimator)
        result = enricher.enrich_month(monthly_dir / "2023-01.parquet")

        valid_sources = {"proxy", "snapshot"}
        assert set(result["oi_source"].unique()).issubset(valid_sources)
```

**Step 2: Run test to verify it passes**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_oi_proxy.py -v --tb=short`

Expected: 10 passed (6 estimator + 4 enricher)

**Step 3: Commit**

```bash
git add tests/gme/options/test_oi_proxy.py
git commit -m "test(options): add OIEnricher test coverage

Tests priority cascade (snapshot > proxy), multi-file enrichment,
and oi_source column correctness.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Add Config Field + CLI Subcommands

**Files:**
- Modify: `src/stockdownloader/gme/options/config.py`
- Modify: `src/stockdownloader/gme/options/__main__.py`
- Modify: `tests/gme/options/test_cli.py`

**Context:** Add `oi_decay_rate` to config. Add `snapshot` and `enrich` subcommands to the CLI. Update `run-all` to include them.

**Step 1: Add config field**

In `src/stockdownloader/gme/options/config.py`, after line 45 (`daily_loss_limit_pct`), add:

```python
    # OI proxy
    oi_decay_rate: float = 0.03
```

**Step 2: Add CLI subcommands**

In `src/stockdownloader/gme/options/__main__.py`:

After the `fetch_parser` block (line ~108), add:

```python
    # -- snapshot --
    subparsers.add_parser(
        "snapshot",
        help="Fetch current options chain snapshot (OI, greeks, IV).",
    )

    # -- enrich --
    subparsers.add_parser(
        "enrich",
        help="Enrich monthly bars with OI data (snapshot + proxy).",
    )
```

Add handler functions after `_cmd_fetch`:

```python
def _cmd_snapshot(args: argparse.Namespace) -> None:
    """Fetch options chain snapshot from Polygon."""
    from stockdownloader.gme.options.config import GMEOptionsConfig
    from stockdownloader.data.market.polygon_options_client import PolygonOptionsClient
    from stockdownloader.gme.options.snapshot import SnapshotCollector

    config = GMEOptionsConfig.from_env(polygon_api_key=args.polygon_key)
    client = PolygonOptionsClient(api_key=config.polygon_api_key)
    collector = SnapshotCollector(config, client)

    logger.info("Collecting options snapshot for %s", config.symbol)
    df = collector.run()
    logger.info("Snapshot complete: %d contracts", len(df))


def _cmd_enrich(args: argparse.Namespace) -> None:
    """Enrich monthly bars with OI data."""
    from stockdownloader.gme.options.config import GMEOptionsConfig
    from stockdownloader.gme.options.oi_proxy import OIProxyEstimator, OIEnricher
    from stockdownloader.gme.options.snapshot import SnapshotCollector

    config = GMEOptionsConfig.from_env(polygon_api_key=args.polygon_key)
    cal_path = config.data_dir / "oi_calibration.json"

    if cal_path.exists():
        estimator = OIProxyEstimator.load_calibration(cal_path)
        logger.info("Loaded calibration: ratio=%.4f", estimator.calibration_ratio)
    else:
        estimator = OIProxyEstimator(decay_rate=config.oi_decay_rate)
        # Try to calibrate from latest snapshot
        collector = SnapshotCollector(config, None)
        from datetime import date as _date
        snap_df = collector.load(_date.today())
        if snap_df is not None:
            import pandas as pd
            monthly_dir = config.data_dir / "monthly"
            if monthly_dir.exists():
                bars = pd.concat(
                    [pd.read_parquet(f) for f in monthly_dir.glob("*.parquet")],
                    ignore_index=True,
                )
                estimator.calibrate(snap_df, bars)
                estimator.save_calibration(cal_path)
        logger.info("Using calibration ratio: %.4f", estimator.calibration_ratio)

    enricher = OIEnricher(data_dir=config.data_dir, estimator=estimator)
    count = enricher.enrich_all()
    logger.info("Enriched %d monthly files", count)
```

Update `_COMMANDS` dict to include:

```python
    "snapshot": _cmd_snapshot,
    "enrich": _cmd_enrich,
```

Update `_cmd_run_all` to call snapshot and enrich between fetch and build-state:

```python
def _cmd_run_all(args: argparse.Namespace) -> None:
    """Run the full pipeline: fetch -> snapshot -> enrich -> build-state -> backtest -> scorecard."""
    logger.info("Running full pipeline")

    _cmd_fetch(args)
    _cmd_snapshot(args)
    _cmd_enrich(args)
    _cmd_build_state(args)

    args.all = True
    args.strategy = None
    _cmd_backtest(args)

    args.date = None
    args.range = None
    _cmd_scorecard(args)

    logger.info("Full pipeline complete")
```

**Step 3: Update CLI tests**

Append to `tests/gme/options/test_cli.py`:

```python
    def test_snapshot_subcommand(self):
        args = build_parser().parse_args(["snapshot"])
        assert args.command == "snapshot"

    def test_enrich_subcommand(self):
        args = build_parser().parse_args(["enrich"])
        assert args.command == "enrich"
```

**Step 4: Run tests**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/test_cli.py tests/gme/options/test_config.py -v --tb=short`

Expected: All pass

**Step 5: Commit**

```bash
git add src/stockdownloader/gme/options/config.py src/stockdownloader/gme/options/__main__.py tests/gme/options/test_cli.py
git commit -m "feat(options): add snapshot and enrich CLI subcommands

New subcommands for daily snapshot collection and OI enrichment.
Updated run-all to include snapshot → enrich in pipeline.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 6: Integration Test + Full Suite

**Files:**
- Modify: `tests/gme/options/test_integration.py`

**Step 1: Add integration test**

Append to `tests/gme/options/test_integration.py`:

```python
    def test_enriched_state_has_nonzero_gex(self):
        """StateEngine computes non-zero GEX with enriched OI data."""
        chain = _synthetic_chain("2023-06-15")
        # Verify OI is already non-zero in synthetic data
        assert chain["open_interest"].sum() > 0

        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        state = engine.build(bars_df=chain)

        assert len(state) == 1
        # With OI > 0, GEX metrics should be non-zero
        assert state.iloc[0]["total_call_oi"] > 0
        assert state.iloc[0]["total_put_oi"] > 0
        assert state.iloc[0]["pc_oi_ratio"] > 0

    def test_oi_enricher_to_state_engine(self):
        """Full pipeline: bars → enrich → state engine produces GEX."""
        from stockdownloader.gme.options.oi_proxy import OIProxyEstimator, OIEnricher

        chain = _synthetic_chain("2023-06-15")
        # Zero out OI to simulate raw fetcher output
        chain["open_interest"] = 0
        assert chain["open_interest"].sum() == 0

        # Apply proxy
        estimator = OIProxyEstimator(calibration_ratio=5.0, decay_rate=0.0)
        enriched = estimator.estimate(chain)
        assert enriched["open_interest"].sum() > 0

        # Feed to state engine
        engine = OptionsStateEngine(spot_prices={"2023-06-15": 25.0})
        state = engine.build(bars_df=enriched)

        assert len(state) == 1
        assert state.iloc[0]["total_call_oi"] > 0
        assert state.iloc[0]["net_gex"] != 0
```

**Step 2: Run full test suite**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -m pytest tests/gme/options/ -v --tb=short`

Expected: All pass (~95 total tests)

**Step 3: Commit**

```bash
git add tests/gme/options/test_integration.py
git commit -m "test(options): add OI enrichment integration tests

Verifies enriched OI flows through StateEngine to produce
non-zero GEX and OI metrics end-to-end.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 7: Run Live Pipeline (Snapshot + Enrich + Rebuild State)

**Files:**
- No code changes — this is execution + verification

**Step 1: Fetch today's snapshot**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -c "
from stockdownloader.gme.options.config import GMEOptionsConfig
from stockdownloader.data.market.polygon_options_client import PolygonOptionsClient
from stockdownloader.gme.options.snapshot import SnapshotCollector

cfg = GMEOptionsConfig(polygon_api_key='jtFFyq1sO7sEznPZChj2vMZEeoY9GQSu')
client = PolygonOptionsClient(api_key=cfg.polygon_api_key)
collector = SnapshotCollector(cfg, client)
df = collector.run()
print(f'Collected {len(df)} contracts')
print(f'OI > 0: {(df[\"open_interest\"] > 0).sum()}/{len(df)}')
print(f'Has greeks: {df[\"delta\"].notna().sum()}/{len(df)}')
"`

Expected: ~1400 contracts collected, ~85% with OI > 0, ~75% with greeks

**Step 2: Calibrate and enrich all monthly data**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -c "
from stockdownloader.gme.options.config import GMEOptionsConfig
from stockdownloader.gme.options.oi_proxy import OIProxyEstimator, OIEnricher
from stockdownloader.gme.options.snapshot import SnapshotCollector
from datetime import date
import pandas as pd, logging
logging.basicConfig(level=logging.INFO)

cfg = GMEOptionsConfig(polygon_api_key='jtFFyq1sO7sEznPZChj2vMZEeoY9GQSu')
collector = SnapshotCollector(cfg, None)
snap_df = collector.load(date.today())

# Calibrate from snapshot
estimator = OIProxyEstimator(decay_rate=cfg.oi_decay_rate)
if snap_df is not None:
    monthly = cfg.data_dir / 'monthly'
    bars = pd.concat([pd.read_parquet(f) for f in monthly.glob('*.parquet')], ignore_index=True)
    estimator.calibrate(snap_df, bars)
    estimator.save_calibration(cfg.data_dir / 'oi_calibration.json')

# Enrich all months
enricher = OIEnricher(data_dir=cfg.data_dir, estimator=estimator)
count = enricher.enrich_all()
print(f'Enriched {count} monthly files')
"`

Expected: 38 monthly files enriched with proxy OI

**Step 3: Rebuild state with enriched data**

Run: `DYLD_LIBRARY_PATH=~/lib PYTHONPATH=src python3 -c "
import pandas as pd, logging
from pathlib import Path
from stockdownloader.gme.options.state_engine import OptionsStateEngine
logging.basicConfig(level=logging.INFO)

bars_csv = pd.read_csv('data/GME/daily_bars.csv')
spot_prices = dict(zip(bars_csv['date'], bars_csv['close']))

engine = OptionsStateEngine(spot_prices=spot_prices)
state = engine.build(bars_dir=Path('data/GME/options/monthly'))
state.to_parquet('data/GME/options/options_state.parquet', index=False)

print(f'State: {len(state)} days × {state.shape[1]} cols')
oi_cols = ['net_gex', 'total_call_oi', 'total_put_oi', 'pc_oi_ratio', 'gex_concentration']
for col in oi_cols:
    nz = (state[col] != 0).sum()
    print(f'  {col}: {nz}/{len(state)} non-zero')
"`

Expected: GEX and OI metrics now have non-zero values across most trading days

**Step 4: Regenerate scorecards with enriched state**

Rerun the scorecard generation from `scripts/run_gme_options_pipeline.py` step 4.

Expected: Regime distribution should now include squeeze/gamma_ramp/cycle_hot, not just "neutral"

**Step 5: Commit data artifacts (optional)**

```bash
git add data/GME/options/oi_calibration.json
git commit -m "data: add OI calibration from live snapshot

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
