# Snapshot OI Integration Design

**Date:** 2026-02-26
**Status:** Approved
**Goal:** Integrate Polygon's Option Contract Snapshot endpoint to populate open_interest data, unlocking GEX, OI flow, and regime detection metrics across the GME options pipeline.

## Problem

The existing pipeline fetches 411K daily option bars via Polygon's aggregates API (`/v2/aggs/...`), which only returns OHLCV data. The `open_interest` column is hardcoded to 0, leaving 11 of 28 state metrics zeroed out (net_gex, gex_flip_price, gex_concentration, total_call_oi, total_put_oi, pc_oi_ratio, oi_weighted_strike, oi_concentration_top5, near_term_oi_pct, oi_skew_delta, and pc_oi_ratio_change). Signal fusion regime detection always returns "neutral".

## Approach: Snapshot Collector + OCC Backfill + Volume Proxy

Two-pronged strategy:
- **Forward:** Daily snapshot collection via `/v3/snapshot/options/GME` provides real OI, greeks, and IV for all ~1,400 active contracts in 6 paginated API calls.
- **Backfill:** Enrich existing 38 months of historical bars using a priority cascade: snapshot OI > OCC aggregate OI > volume-based proxy estimation.

## Section 1: Snapshot Collector

**New class:** `SnapshotCollector` in `snapshot.py`.

Fetches the bulk snapshot endpoint `GET /v3/snapshot/options/GME?limit=250`, paginating through all active contracts. Saves one Parquet file per day to `data/GME/options/snapshots/YYYY-MM-DD.parquet`.

**Schema per row:**

| Column | Source |
|--------|--------|
| `date` | Collection date |
| `option_ticker` | `details.ticker` |
| `strike` | `details.strike_price` |
| `option_type` | `details.contract_type` |
| `expiration` | `details.expiration_date` |
| `close` | `day.close` |
| `volume` | `day.volume` |
| `open_interest` | `results.open_interest` |
| `implied_volatility` | `results.implied_volatility` |
| `delta` | `greeks.delta` |
| `gamma` | `greeks.gamma` |
| `theta` | `greeks.theta` |
| `vega` | `greeks.vega` |
| `underlying_price` | `underlying_asset.price` |

**API cost:** 6 paginated calls per collection (~1.2s with rate limiting).

**Client method:** `PolygonOptionsClient.fetch_options_chain_snapshot(symbol)` handles pagination internally.

**CLI:** New `snapshot` subcommand: `python -m stockdownloader.gme.options snapshot`.

## Section 2: Historical OI Backfill via Volume Proxy

**Volume-Based OI Proxy Algorithm:**

For each contract on each day:
1. Compute cumulative volume from first bar to current day
2. Apply exponential decay (configurable λ, default 0.03/day) — positions close over time
3. Scale by OI-to-volume calibration ratio learned from real snapshot data

**New classes in `oi_proxy.py`:**

- `OIProxyEstimator`:
  - `calibrate(snapshot_df)` — learns OI/cumulative-volume ratio from real snapshot data
  - `estimate(bars_df) → DataFrame` — adds `open_interest` column using proxy
  - Saves calibration to `data/GME/options/oi_calibration.json`

- `OIEnricher`:
  - Merges real OI (snapshots) with proxy OI (historical) using priority cascade
  - Priority: snapshot OI > OCC aggregate OI > proxy OI
  - Writes enriched `open_interest` + `oi_source` columns back to monthly Parquets
  - `oi_source` values: `"snapshot"`, `"occ"`, `"proxy"`

**Expected accuracy:** Volume-based OI proxies achieve ~50-70% rank correlation with actual OI. Sufficient for GEX direction and OI concentration signals.

## Section 3: Pipeline Integration

**Updated pipeline step ordering:**

```
fetch → snapshot → enrich → build-state → backtest → scorecard
         ↑ new      ↑ new
```

1. `fetch` — existing monthly bar fetcher (already done, 38 months)
2. `snapshot` — SnapshotCollector fetches today's full chain (6 API calls)
3. `enrich` — OIEnricher merges real OI + proxy OI into monthly Parquets
4. `build-state` — existing StateEngine, now with non-zero OI
5. `backtest` / `scorecard` — unchanged

**StateEngine:** No changes needed. Already reads `open_interest` from DataFrame.

**`run-all` subcommand** updated to include snapshot + enrich steps.

## Section 4: Files & Testing

**New files:**

| File | Purpose |
|------|---------|
| `src/stockdownloader/gme/options/snapshot.py` | `SnapshotCollector` class |
| `src/stockdownloader/gme/options/oi_proxy.py` | `OIProxyEstimator` + `OIEnricher` |
| `tests/gme/options/test_snapshot.py` | Snapshot collector tests |
| `tests/gme/options/test_oi_proxy.py` | OI proxy + enricher tests |

**Modified files:**

| File | Change |
|------|--------|
| `polygon_options_client.py` | Add `fetch_options_chain_snapshot(symbol)` |
| `__main__.py` | Add `snapshot` and `enrich` subcommands |
| `config.py` | Add `oi_decay_rate: float = 0.03` |

**No changes to:** `state_engine.py`, `backtester.py`, `signal_fusion.py`, `scorecard.py`, strategy files.

**Test coverage:**
- Snapshot: paginated parsing, Parquet schema, underlying price extraction
- OI proxy: calibration ratio, decay curve shape, enricher priority cascade, `oi_source` labeling
