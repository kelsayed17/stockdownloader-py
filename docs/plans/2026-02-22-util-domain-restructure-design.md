# Util Domain Restructure — Design

**Date:** 2026-02-22
**Scope:** Full overhaul of `src/stockdownloader/util/` — eliminate duplication, unify batch + streaming indicators, reorganize by domain.

## Problem

The util package (26 files, ~9,000 lines) has accumulated concrete issues:

1. **`_quantize()` duplicated 4 times** — identical logic in `technical/__init__`, `intraday_indicators`, `streaming/__init__`, and `big_decimal_math`
2. **`true_range()` duplicated** — identical in `technical/__init__` and `streaming/__init__`
3. **Session VWAP implemented 3 times** — batch in `technical/trend`, streaming in `streaming/volume`, extended in `intraday_indicators`
4. **CVD implemented twice** — batch in `intraday_indicators`, streaming in `streaming/volume`
5. **Decimal constants scattered** — `_HALF`, `_THREE` defined locally instead of centralized
6. **Dead code** — `execute()`, `execute_with_result()` exported but never used
7. **Misplaced files** — `pinescript_models.py`, `pinescript_modes.py`, `pinescript_ml_strategy.py` sit in util root but belong in `pinescript/`
8. **Batch vs streaming as organizational principle** — RSI batch and RSI streaming live in separate directory trees (`technical/momentum` vs `streaming/oscillators`) when they belong together

## Approach: Domain Packages

Reorganize by **what the code does** rather than how it computes. Batch and streaming implementations of the same indicator live in the same file.

## New Layout

```
util/
├── math.py                         ← big_decimal_math (renamed)
│                                      + canonical _quantize(value, scale=10)
│                                      + HALF, THREE, TEN constants
│
├── config.py                       ← config_loader + event_calendar (merged)
│                                      Calendar section: FOMC dates + anchor functions
│
├── io.py                           ← io_helpers + parsers (merged)
│                                      Remove dead code: execute(), execute_with_result()
│
├── timeframe.py                    ← timeframe_aggregator (renamed, unchanged)
│
├── indicators/                     ← NEW unified package
│   ├── __init__.py                 ← public re-exports
│   ├── _core.py                    ← shared internals: true_range, _find_session_start,
│   │                                  _compute_session_vwap_core
│   ├── momentum.py                 ← batch: RSI, MACD, Stoch, MFI, CCI, Williams%R, ROC, OBV
│   │                                  streaming: StreamingRSI, StreamingMACD, StreamingOBV
│   ├── trend.py                    ← batch: ADX, SAR, Ichimoku, Fibonacci, S/R
│   │                                  streaming: StreamingADX, StreamingSAR
│   ├── volatility.py               ← batch: Bollinger, ATR, std_dev
│   │                                  streaming: StreamingEMA, StreamingATR
│   ├── volume.py                   ← batch: VWAP, session_vwap, CVD, OBV
│   │                                  streaming: StreamingSessionVWAP, StreamingAnchoredVWAP, StreamingCVD
│   │                                  intraday: ExtendedSessionVWAP, AnchoredVWAPBands
│   ├── intraday.py                 ← remaining intraday-specific: candle_strength,
│   │                                  time_of_day_rvol, linear_regression_slope, etc.
│   ├── smc.py                      ← smc_indicators + streaming_structure (merged)
│   │                                  Primitives + StreamingStructureTracker
│   ├── hub.py                      ← indicator_hub (moved from root)
│   └── htf.py                      ← StreamingHTFResample (from streaming/accumulators)
│
├── options/
│   └── black_scholes.py            ← black_scholes_calculator (renamed)
│
└── pinescript/                     ← absorb loose pinescript_* files
    ├── __init__.py
    ├── generator.py                ← core generation logic (~600 lines)
    ├── renderer.py                 ← rendering/template methods (~600 lines)
    ├── models.py                   ← pinescript_models (moved from root)
    ├── modes.py                    ← pinescript_modes (moved from root)
    └── ml_export.py                ← pinescript_ml_strategy (moved from root)
```

## Duplication Elimination

### `_quantize()` — 4 copies → 1

**Canonical source:** `math.py`

```python
def _quantize(value: Decimal, scale: int = DEFAULT_SCALE) -> Decimal:
    return value.quantize(Decimal(10) ** -scale, rounding=ROUND_HALF_UP)
```

All indicator modules import from `math.py`.

### `true_range()` — 2 copies → 1

**Canonical source:** `indicators/_core.py`

```python
def true_range(data: Sequence[PriceData], index: int) -> Decimal:
    ...
```

Re-exported via `indicators/__init__.py`. StreamingATR imports from `_core`.

### Session VWAP — 3 implementations → 1 file

**Canonical source:** `indicators/volume.py`

- `_compute_session_vwap_core()` — shared batch logic
- `session_vwap()`, `session_vwap_bands()` — batch API
- `extended_session_vwap()`, `ExtendedSessionVWAP` — intraday bands
- `StreamingSessionVWAP` — O(1) incremental version
- `StreamingAnchoredVWAP`, `AnchoredVWAPBands` — anchored variant

### CVD — 2 implementations → 1 file

**Canonical source:** `indicators/volume.py`

- `cvd_session()` — batch
- `StreamingCVD` — streaming

### Decimal constants — scattered → centralized

**`math.py` provides:** `ZERO`, `ONE`, `TWO`, `HALF`, `THREE`, `TEN`, `HUNDRED`

No more local `_HALF = Decimal("0.5")` in indicator files.

### Dead code removal

- `execute()` and `execute_with_result()` in `io_helpers.py` — exported but zero callers outside the file. Removed.
- `DateHelper` — removed from `__init__.py` re-exports (kept in `io.py` for potential future use, just not re-exported).

## File Merges

| New File | Sources | Rationale |
|----------|---------|-----------|
| `math.py` | `big_decimal_math.py` | Shorter name, + shared constants |
| `config.py` | `config_loader.py` + `event_calendar.py` | event_calendar is 110 lines of config data |
| `io.py` | `io_helpers.py` + `parsers.py` | Both are I/O utilities, small enough to merge |
| `indicators/momentum.py` | `technical/momentum.py` + `streaming/oscillators.py` | Same indicators, batch + streaming together |
| `indicators/trend.py` | `technical/trend.py` (non-VWAP) + `streaming/trend.py` | ADX, SAR batch + streaming together |
| `indicators/volatility.py` | `technical/volatility.py` + parts of `streaming/accumulators.py` | Bollinger, ATR + StreamingEMA, StreamingATR |
| `indicators/volume.py` | `technical/trend.py` (VWAP parts) + `streaming/volume.py` + `intraday_indicators.py` (VWAP/CVD parts) | All volume-based indicators unified |
| `indicators/smc.py` | `smc_indicators.py` + `streaming_structure.py` | Primitives + stateful tracker |
| `pinescript/models.py` | `pinescript_models.py` (moved into subpackage) | Was misplaced in root |
| `pinescript/modes.py` | `pinescript_modes.py` (moved into subpackage) | Was misplaced in root |
| `pinescript/ml_export.py` | `pinescript_ml_strategy.py` (moved into subpackage) | Was misplaced in root |

## Migration Strategy

### Phase 1: Create new structure with backward-compatible shims

Create all new files. Old files become thin re-export stubs:

```python
# old: src/stockdownloader/util/big_decimal_math.py (becomes shim)
from stockdownloader.util.math import *  # noqa: F401,F403
```

Zero breakage — all 79 external files keep working.

### Phase 2: Update callers

Batch-update all external files to use new import paths. Systematic, file-by-file. Run tests after each batch.

### Phase 3: Remove shims

Delete old files (now empty stubs). Update `util/__init__.py`.

### Phase 4: Split pinescript/generator.py

Split the 1,211-line file into `generator.py` (~600 lines, core logic) + `renderer.py` (~600 lines, template methods).

## What Does NOT Change

- **IndicatorHub public API** — callers import `IndicatorHub` and call methods identically
- **All indicator function signatures** — `rsi(data, period)` etc. unchanged
- **All streaming class APIs** — `StreamingRSI.update()` etc. unchanged
- **Config constants** — `INITIAL_CAPITAL`, `PROJECT_ROOT`, etc. keep names
- **CsvParser API** — unchanged
- **Test assertions** — only test import paths change

## Impact

- ~79 files need import updates (mechanical find-replace)
- 26 old files → 20 new files
- ~200 lines of duplicate code eliminated
- ~200 lines of dead code removed
- Batch + streaming implementations co-located per indicator category
