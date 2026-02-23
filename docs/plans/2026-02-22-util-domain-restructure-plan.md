# Util Domain Restructure — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure `src/stockdownloader/util/` from 26 files to ~20 domain-focused files, eliminating all code duplication and unifying batch + streaming indicators per category.

**Architecture:** Four-phase migration: (1) Create new files with consolidated code, (2) Add backward-compatible shims to old files, (3) Update all callers to new import paths, (4) Delete shims and old files. Each phase ends with all 3,198 tests passing.

**Tech Stack:** Python 3.12, pytest, no new dependencies.

---

## Phase 1: Foundation Layer (math, config, io)

### Task 1: Create `util/math.py` from `big_decimal_math.py`

**Files:**
- Create: `src/stockdownloader/util/math.py`
- Modify: `src/stockdownloader/util/big_decimal_math.py` (becomes shim)

**Step 1: Create `math.py`**

Copy `big_decimal_math.py` to `math.py`. Add these new constants after line 18:

```python
HALF = Decimal("0.5")
THREE = Decimal("3")
TEN = Decimal("10")
```

Make `_quantize` public with default `scale` parameter:

```python
DEFAULT_SCALE = 10

def quantize(value: Decimal, scale: int = DEFAULT_SCALE) -> Decimal:
    """Quantize *value* to the given number of decimal places."""
    return value.quantize(Decimal(10) ** -scale, rounding=ROUND_HALF_UP)
```

Keep `_quantize` as an alias: `_quantize = quantize` for internal use.

Add `__all__` with all public names: `ZERO, ONE, TWO, HALF, THREE, TEN, HUNDRED, DEFAULT_SCALE, divide, quantize_decimal, scale2, average, percent_change, quantize`.

**Step 2: Convert `big_decimal_math.py` to shim**

Replace contents with:
```python
"""Backward-compatible shim — import from stockdownloader.util.math instead."""
from stockdownloader.util.math import *  # noqa: F401,F403
from stockdownloader.util.math import _quantize  # noqa: F401
```

**Step 3: Run tests**

Run: `pytest tests/util/test_big_decimal_math.py -v`
Expected: All pass (shim re-exports everything)

**Step 4: Commit**

```
feat: create util/math.py with canonical quantize and decimal constants
```

---

### Task 2: Create `util/config.py` from `config_loader.py` + `event_calendar.py`

**Files:**
- Create: `src/stockdownloader/util/config.py`
- Modify: `src/stockdownloader/util/config_loader.py` (becomes shim)
- Modify: `src/stockdownloader/util/event_calendar.py` (becomes shim)

**Step 1: Create `config.py`**

Copy `config_loader.py` (331 lines) to `config.py`. Append the full contents of `event_calendar.py` (111 lines) below a section header:

```python
# =========================================================================
# Event calendar (formerly event_calendar.py)
# =========================================================================
```

All imports from both files merged at the top. Remove any duplicates.

**Step 2: Convert old files to shims**

`config_loader.py`:
```python
"""Backward-compatible shim — import from stockdownloader.util.config instead."""
from stockdownloader.util.config import *  # noqa: F401,F403
from stockdownloader.util.config import _deep_merge  # noqa: F401
```

`event_calendar.py`:
```python
"""Backward-compatible shim — import from stockdownloader.util.config instead."""
from stockdownloader.util.config import (  # noqa: F401
    FOMC_DATES,
    get_anchor_date,
    is_anchor_date,
    days_since_anchor,
)
```

**Step 3: Run tests**

Run: `pytest tests/util/test_config_loader.py tests/util/test_event_calendar.py -v`
Expected: All pass

**Step 4: Commit**

```
feat: create util/config.py merging config_loader + event_calendar
```

---

### Task 3: Create `util/io.py` from `io_helpers.py` + `parsers.py`

**Files:**
- Create: `src/stockdownloader/util/io.py`
- Modify: `src/stockdownloader/util/io_helpers.py` (becomes shim)
- Modify: `src/stockdownloader/util/parsers.py` (becomes shim)

**Step 1: Create `io.py`**

Merge `io_helpers.py` and `parsers.py` into one file. **Remove** `execute()` and `execute_with_result()` (dead code — only tested, never called from production). Keep `TeeWriter`.

Sections:
1. File I/O helpers (from io_helpers)
2. TeeWriter (from io_helpers)
3. Date formats and DateHelper (from parsers)
4. CsvParser (from parsers)

**Step 2: Convert old files to shims**

`io_helpers.py`:
```python
"""Backward-compatible shim — import from stockdownloader.util.io instead."""
from stockdownloader.util.io import *  # noqa: F401,F403
```

`parsers.py`:
```python
"""Backward-compatible shim — import from stockdownloader.util.io instead."""
from stockdownloader.util.io import *  # noqa: F401,F403
from stockdownloader.util.io import (  # noqa: F401
    _adjust_to_next_market_day,
    _subtract_months,
)
```

**Step 3: Delete `tests/util/test_retry_executor.py`** (tests dead code we removed)

**Step 4: Run tests**

Run: `pytest tests/ -v -k "not test_retry_executor" --co | head -20` then full `pytest tests/ -v`
Expected: All pass (minus removed retry tests)

**Step 5: Commit**

```
feat: create util/io.py merging io_helpers + parsers, remove dead retry code
```

---

### Task 4: Rename `timeframe_aggregator.py` → `timeframe.py`

**Files:**
- Create: `src/stockdownloader/util/timeframe.py`
- Modify: `src/stockdownloader/util/timeframe_aggregator.py` (becomes shim)

**Step 1: Copy file, update any self-referencing imports**

Copy `timeframe_aggregator.py` to `timeframe.py`. No content changes.

**Step 2: Convert old file to shim**

```python
"""Backward-compatible shim — import from stockdownloader.util.timeframe instead."""
from stockdownloader.util.timeframe import *  # noqa: F401,F403
```

**Step 3: Run tests**

Run: `pytest tests/util/test_timeframe_aggregator.py -v`
Expected: All pass

**Step 4: Commit**

```
refactor: rename timeframe_aggregator.py to timeframe.py with shim
```

---

## Phase 2: Unified Indicators Package

### Task 5: Create `indicators/_core.py` with shared helpers

**Files:**
- Create: `src/stockdownloader/util/indicators/__init__.py`
- Create: `src/stockdownloader/util/indicators/_core.py`

**Step 1: Create `indicators/` package directory and `__init__.py`**

`__init__.py` starts empty — will be populated as indicator modules are created:
```python
"""Unified technical and streaming indicators."""
```

**Step 2: Create `_core.py` with shared helpers**

Move from `technical/__init__.py`:
- `true_range()` (lines 99-112)
- `sma()` (lines 41-61)
- `ema()` (lines 63-97)
- `standard_deviation()` (lines 115-135)
- `_period_midpoint()` (lines 138-152)
- `_deduplicate_levels()` (lines 154-176)
- `_find_session_start()` (from `technical/trend.py` lines 410-417)
- `_compute_session_vwap_core()` (from `technical/trend.py` lines 420-465)
- Crossover functions: `crossed_above`, `crossed_below`, `crossed_above_series`, `crossed_below_series` (lines 178-228)

All import `quantize` from `stockdownloader.util.math` (the new canonical source) instead of defining local `_quantize`.

Import `ZERO` from `stockdownloader.util.math`.

**Step 3: Run tests**

Run: `pytest tests/util/test_moving_average_calculator.py tests/util/test_crossover.py -v`
Expected: FAIL (tests still import from old paths, but _core.py should at least be importable)

Run: `python -c "from stockdownloader.util.indicators._core import true_range, sma, ema"`
Expected: Success

**Step 4: Commit**

```
feat: create indicators/_core.py with shared helpers (true_range, sma, ema, crossovers)
```

---

### Task 6: Create `indicators/momentum.py` (batch + streaming)

**Files:**
- Create: `src/stockdownloader/util/indicators/momentum.py`

**Step 1: Create file**

Merge from `technical/momentum.py` (414 lines) + `streaming/oscillators.py` (201 lines):

Section 1 — Batch functions (from technical/momentum.py):
- `Stochastic` dataclass
- `stochastic()`, `rsi()`, `macd_line()`, `macd_signal()`, `_macd_signal_from_lines()`, `macd_histogram()`, `roc()`, `mfi()`, `williams_r()`, `cci()`, `obv()`, `is_obv_rising()`, `average_volume()`

Section 2 — Streaming classes (from streaming/oscillators.py + streaming/accumulators.py):
- `StreamingRSI`
- `StreamingMACD`
- `StreamingOBV` (from accumulators.py)

All import `quantize` from `stockdownloader.util.math` instead of local `_quantize`.
Import `sma`, `ema` from `stockdownloader.util.indicators._core`.

**Step 2: Verify importable**

Run: `python -c "from stockdownloader.util.indicators.momentum import rsi, StreamingRSI, StreamingMACD"`
Expected: Success

**Step 3: Commit**

```
feat: create indicators/momentum.py unifying batch + streaming
```

---

### Task 7: Create `indicators/volatility.py` (batch + streaming)

**Files:**
- Create: `src/stockdownloader/util/indicators/volatility.py`

**Step 1: Create file**

Merge from `technical/volatility.py` (140 lines) + parts of `streaming/accumulators.py`:

Section 1 — Batch:
- `BollingerBands` dataclass
- `bollinger_bands()`, `bollinger_percent_b()`, `_bollinger_percent_b_from_bands()`, `atr()`

Section 2 — Streaming:
- `StreamingEMA` (from accumulators.py)
- `StreamingATR` (from accumulators.py)

Import `true_range`, `sma`, `standard_deviation` from `_core`. Import `quantize, ZERO` from `math`.

**Step 2: Verify importable**

Run: `python -c "from stockdownloader.util.indicators.volatility import bollinger_bands, atr, StreamingEMA, StreamingATR"`

**Step 3: Commit**

```
feat: create indicators/volatility.py unifying batch + streaming
```

---

### Task 8: Create `indicators/trend.py` (batch + streaming)

**Files:**
- Create: `src/stockdownloader/util/indicators/trend.py`

**Step 1: Create file**

Merge from `technical/trend.py` (non-VWAP parts, ~340 lines) + `streaming/trend.py` (261 lines):

Section 1 — Batch:
- `ADXResult`, `IchimokuCloud`, `FibonacciLevels`, `SupportResistance` dataclasses
- `adx()`, `parabolic_sar()`, `is_sar_bullish()`, `vwap()` (lookback-based), `fibonacci_retracement()`, `ichimoku()`, `support_resistance()`

Section 2 — Streaming:
- `ADXState` dataclass
- `StreamingADX`
- `StreamingSAR`

Import `true_range`, `sma`, `ema`, `_period_midpoint`, `_deduplicate_levels` from `_core`.

**Step 2: Verify importable**

**Step 3: Commit**

```
feat: create indicators/trend.py unifying batch + streaming
```

---

### Task 9: Create `indicators/volume.py` (all VWAP + CVD + OBV unified)

**Files:**
- Create: `src/stockdownloader/util/indicators/volume.py`

**Step 1: Create file**

This is the most complex merge. Consolidates from 3 files:
- `technical/trend.py` VWAP section (lines 399-503)
- `streaming/volume.py` (283 lines)
- `intraday_indicators.py` VWAP/CVD section (lines 44-250)

Section 1 — Data structures:
- `SessionVWAP` dataclass (from technical/trend)
- `ExtendedSessionVWAP` dataclass (from intraday_indicators)
- `AnchoredVWAPBands` dataclass (from intraday_indicators)

Section 2 — Batch functions:
- `_find_session_start()` — import from `_core`
- `_compute_session_vwap_core()` — import from `_core`
- `session_vwap()`, `session_vwap_bands()`
- `extended_session_vwap_bands()`
- `cvd_session()`, `cvd_normalized()`

Section 3 — Streaming classes:
- `StreamingSessionVWAP`
- `StreamingAnchoredVWAP`
- `StreamingCVD`

All import `quantize, ZERO, HALF` from `stockdownloader.util.math`.

**Step 2: Verify importable**

**Step 3: Commit**

```
feat: create indicators/volume.py unifying VWAP, CVD, and OBV
```

---

### Task 10: Create `indicators/intraday.py` (remaining intraday functions)

**Files:**
- Create: `src/stockdownloader/util/indicators/intraday.py`

**Step 1: Create file**

Move from `intraday_indicators.py` everything NOT already in `volume.py`:
- `tod_rvol()`, `linear_regression_slope()`, `lrs_normalized()`
- `vwap_slope()`, `vwap_acceleration()`, `_session_vwap_at()`
- `aggregate_to_daily()`, `daily_atr_prior()`, `resample_to_htf()`, `htf_ema_trend()`
- `CandleStrength` dataclass, `candle_strength()`, `is_hammer()`, `is_inv_hammer()`
- `is_bull_engulfing()`, `is_bear_engulfing()`, `near_level()`, `compute_sr_score()`
- `_round_to_5()`, `rel_vol()`

Import VWAP functions from `indicators.volume` and `indicators._core`.

**Step 2: Verify importable**

**Step 3: Commit**

```
feat: create indicators/intraday.py with non-VWAP intraday functions
```

---

### Task 11: Create `indicators/smc.py` (merged primitives + tracker)

**Files:**
- Create: `src/stockdownloader/util/indicators/smc.py`

**Step 1: Create file**

Merge `smc_indicators.py` (262 lines) + `streaming_structure.py` (339 lines):

Section 1 — Data structures:
- `SwingPoint`, `StructureState`, `_EMPTY_STRUCTURE`

Section 2 — Primitives:
- `is_swing_high()`, `is_swing_low()`, `is_liquidity_sweep_high()`, `is_liquidity_sweep_low()`, `impulse_strength()`, `zone_from_impulse_origin()`

Section 3 — Stateful tracker:
- `StreamingStructureTracker` class

The tracker no longer needs cross-file imports since primitives are in the same file.

**Step 2: Verify importable**

**Step 3: Commit**

```
feat: create indicators/smc.py merging primitives + streaming tracker
```

---

### Task 12: Create `indicators/htf.py` (HTF resampling)

**Files:**
- Create: `src/stockdownloader/util/indicators/htf.py`

**Step 1: Create file**

Move `StreamingHTFResample` from `streaming/accumulators.py` (lines 203-285).

**Step 2: Commit**

```
feat: create indicators/htf.py with StreamingHTFResample
```

---

### Task 13: Move `indicator_hub.py` → `indicators/hub.py`

**Files:**
- Create: `src/stockdownloader/util/indicators/hub.py`
- Modify: `src/stockdownloader/util/indicator_hub.py` (becomes shim)

**Step 1: Copy indicator_hub.py to indicators/hub.py**

Update internal imports to use new indicator module paths:
- `from stockdownloader.util import technical as ti` → `from stockdownloader.util.indicators import momentum, trend, volatility, volume` etc.
- `from stockdownloader.util import intraday_indicators as ii` → `from stockdownloader.util.indicators import intraday as ii`
- `from stockdownloader.util.streaming import ...` → `from stockdownloader.util.indicators.momentum import StreamingRSI, ...` etc.
- `from stockdownloader.util.streaming_structure import StreamingStructureTracker` → `from stockdownloader.util.indicators.smc import StreamingStructureTracker`

**Step 2: Convert old file to shim**

```python
"""Backward-compatible shim — import from stockdownloader.util.indicators.hub instead."""
from stockdownloader.util.indicators.hub import *  # noqa: F401,F403
```

**Step 3: Populate `indicators/__init__.py`**

Add comprehensive re-exports for all public names from all indicator modules (momentum, trend, volatility, volume, intraday, smc, htf, hub, _core).

**Step 4: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All 3,198 pass (shims ensure backward compatibility)

**Step 5: Commit**

```
feat: move indicator_hub to indicators/hub.py, populate indicators __init__
```

---

## Phase 3: PineScript and Options Reorganization

### Task 14: Move loose pinescript files into `pinescript/` subpackage

**Files:**
- Move: `src/stockdownloader/util/pinescript_models.py` → `src/stockdownloader/util/pinescript/models.py`
- Move: `src/stockdownloader/util/pinescript_modes.py` → `src/stockdownloader/util/pinescript/modes.py`
- Move: `src/stockdownloader/util/pinescript_ml_strategy.py` → `src/stockdownloader/util/pinescript/ml_export.py`
- Leave shims at old locations

**Step 1: Copy files to new locations**

For each file, copy to new path. Update internal imports if any reference each other.

`pinescript/models.py` ← copy of `pinescript_models.py`
`pinescript/modes.py` ← copy of `pinescript_modes.py`, update import: `from stockdownloader.util.pinescript.models import ...`
`pinescript/ml_export.py` ← copy of `pinescript_ml_strategy.py`, update import if needed

**Step 2: Update `pinescript/__init__.py`**

```python
"""PineScript generation utilities."""

from stockdownloader.util.pinescript.generator import (
    PineScriptGenerator,
    mode_to_strategy,
    strategy_to_mode,
)
from stockdownloader.util.pinescript.models import (
    CompositeStrategyDefinition,
    Condition,
    Indicator,
    Input,
    InputType,
    ModeDefinition,
    SharedInfrastructure,
    StrategyDefinition,
)

__all__ = [...]
```

**Step 3: Update `generator.py` import**

Change: `from stockdownloader.util.pinescript_models import ...`
To: `from stockdownloader.util.pinescript.models import ...`

**Step 4: Create shims at old locations**

`pinescript_models.py`:
```python
"""Backward-compatible shim."""
from stockdownloader.util.pinescript.models import *  # noqa: F401,F403
```

Same pattern for `pinescript_modes.py` and `pinescript_ml_strategy.py`.

**Step 5: Run pinescript tests**

Run: `pytest tests/util/test_pinescript*.py -v`
Expected: All pass

**Step 6: Commit**

```
refactor: move pinescript_models/modes/ml_strategy into pinescript/ subpackage
```

---

### Task 15: Create `options/black_scholes.py`

**Files:**
- Create: `src/stockdownloader/util/options/__init__.py`
- Create: `src/stockdownloader/util/options/black_scholes.py`
- Modify: `src/stockdownloader/util/black_scholes_calculator.py` (becomes shim)

**Step 1: Create options package, copy file**

Copy `black_scholes_calculator.py` to `options/black_scholes.py`. No content changes.

`options/__init__.py`:
```python
"""Options pricing utilities."""
from stockdownloader.util.options.black_scholes import (
    delta, estimate_volatility, intrinsic_value, price, theta,
)
__all__ = ["delta", "estimate_volatility", "intrinsic_value", "price", "theta"]
```

**Step 2: Shim old file**

**Step 3: Run tests**

Run: `pytest tests/util/test_black_scholes_calculator.py -v`

**Step 4: Commit**

```
refactor: move black_scholes_calculator into options/ subpackage
```

---

## Phase 4: Caller Migration and Cleanup

### Task 16: Update `util/__init__.py` to use new paths

**Files:**
- Modify: `src/stockdownloader/util/__init__.py`

**Step 1: Rewrite imports**

Replace all old import paths with new ones:
- `from stockdownloader.util.big_decimal_math import ...` → `from stockdownloader.util.math import ...`
- `from stockdownloader.util.technical import ...` → `from stockdownloader.util.indicators import ...`
- `from stockdownloader.util.black_scholes_calculator import ...` → `from stockdownloader.util.options import ...`
- `from stockdownloader.util.parsers import ...` → `from stockdownloader.util.io import ...`
- `from stockdownloader.util.io_helpers import ...` → `from stockdownloader.util.io import ...`

Remove `execute` and `execute_with_result` from exports (dead code).

**Step 2: Run full tests**

Run: `pytest tests/ -x -q`
Expected: All pass

**Step 3: Commit**

```
refactor: update util/__init__.py to use new module paths
```

---

### Task 17: Update all external callers (src/ files)

**Files:**
- Modify: ~50 files across `src/stockdownloader/` (strategy, app, backtest, analysis, ml, data)

**Step 1: Batch update imports**

For each old import pattern, find-and-replace across the codebase:

| Old Import | New Import |
|-----------|-----------|
| `from stockdownloader.util.big_decimal_math import` | `from stockdownloader.util.math import` |
| `from stockdownloader.util.config_loader import` | `from stockdownloader.util.config import` |
| `from stockdownloader.util.io_helpers import` | `from stockdownloader.util.io import` |
| `from stockdownloader.util.parsers import` | `from stockdownloader.util.io import` |
| `from stockdownloader.util.indicator_hub import` | `from stockdownloader.util.indicators.hub import` |
| `from stockdownloader.util.intraday_indicators import` | `from stockdownloader.util.indicators.intraday import` (+ `.volume` for VWAP/CVD) |
| `from stockdownloader.util.timeframe_aggregator import` | `from stockdownloader.util.timeframe import` |
| `from stockdownloader.util.smc_indicators import` | `from stockdownloader.util.indicators.smc import` |
| `from stockdownloader.util.streaming_structure import` | `from stockdownloader.util.indicators.smc import` |
| `from stockdownloader.util.event_calendar import` | `from stockdownloader.util.config import` |
| `from stockdownloader.util.black_scholes_calculator import` | `from stockdownloader.util.options.black_scholes import` |
| `from stockdownloader.util.pinescript_models import` | `from stockdownloader.util.pinescript.models import` |
| `from stockdownloader.util.pinescript_modes import` | `from stockdownloader.util.pinescript.modes import` |
| `from stockdownloader.util.pinescript_ml_strategy import` | `from stockdownloader.util.pinescript.ml_export import` |
| `from stockdownloader.util import technical as ti` | `from stockdownloader.util import indicators as ti` (or specific module) |
| `from stockdownloader.util.streaming import` | `from stockdownloader.util.indicators.<specific> import` |
| `from stockdownloader.util.technical import` (external) | `from stockdownloader.util.indicators import` |

Process directory by directory:
1. `strategy/` (most files)
2. `app/`
3. `backtest/`
4. `analysis/`
5. `ml/`
6. `data/`

**Step 2: Run tests after each directory batch**

Run: `pytest tests/ -x -q` after each directory
Expected: All pass at each checkpoint

**Step 3: Commit per directory**

```
refactor: update strategy/ imports to new util paths
refactor: update app/ imports to new util paths
refactor: update backtest/ imports to new util paths
refactor: update analysis/ + ml/ + data/ imports to new util paths
```

---

### Task 18: Update all test file imports

**Files:**
- Modify: 22 test files in `tests/util/`
- Possibly rename test files to match new module names

**Step 1: Update test imports**

| Old Test File | New Imports From |
|--------------|-----------------|
| `test_big_decimal_math.py` | `stockdownloader.util.math` |
| `test_config_loader.py` | `stockdownloader.util.config` |
| `test_event_calendar.py` | `stockdownloader.util.config` |
| `test_technical_indicators.py` | `stockdownloader.util.indicators` |
| `test_moving_average_calculator.py` | `stockdownloader.util.indicators._core` |
| `test_crossover.py` | `stockdownloader.util.indicators._core` |
| `test_session_vwap.py` | `stockdownloader.util.indicators.volume` |
| `test_intraday_indicators.py` | `stockdownloader.util.indicators.intraday` + `.volume` |
| `test_incremental_indicators.py` | `stockdownloader.util.indicators.momentum` etc. |
| `test_streaming_avwap.py` | `stockdownloader.util.indicators.volume` |
| `test_streaming_structure.py` | `stockdownloader.util.indicators.smc` |
| `test_smc_indicators.py` | `stockdownloader.util.indicators.smc` |
| `test_indicator_hub.py` | `stockdownloader.util.indicators.hub` |
| `test_timeframe_aggregator.py` | `stockdownloader.util.timeframe` |
| `test_black_scholes_calculator.py` | `stockdownloader.util.options.black_scholes` |
| `test_pinescript_*.py` (6 files) | `stockdownloader.util.pinescript.models` / `.modes` / `.ml_export` |

**Step 2: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All 3,198 pass (or 3,196 if retry tests removed)

**Step 3: Commit**

```
refactor: update test imports to new util module paths
```

---

### Task 19: Delete all shim files and old directories

**Files:**
- Delete: `src/stockdownloader/util/big_decimal_math.py` (shim)
- Delete: `src/stockdownloader/util/config_loader.py` (shim)
- Delete: `src/stockdownloader/util/event_calendar.py` (shim)
- Delete: `src/stockdownloader/util/io_helpers.py` (shim)
- Delete: `src/stockdownloader/util/parsers.py` (shim)
- Delete: `src/stockdownloader/util/timeframe_aggregator.py` (shim)
- Delete: `src/stockdownloader/util/indicator_hub.py` (shim)
- Delete: `src/stockdownloader/util/intraday_indicators.py` (shim)
- Delete: `src/stockdownloader/util/smc_indicators.py` (shim)
- Delete: `src/stockdownloader/util/streaming_structure.py` (shim)
- Delete: `src/stockdownloader/util/black_scholes_calculator.py` (shim)
- Delete: `src/stockdownloader/util/pinescript_models.py` (shim)
- Delete: `src/stockdownloader/util/pinescript_modes.py` (shim)
- Delete: `src/stockdownloader/util/pinescript_ml_strategy.py` (shim)
- Delete: `src/stockdownloader/util/technical/` (entire directory)
- Delete: `src/stockdownloader/util/streaming/` (entire directory)
- Delete: `tests/util/test_retry_executor.py` (if not already removed)

**Step 1: Verify no remaining imports reference old paths**

Search for any remaining old import patterns across the entire codebase:
```bash
grep -r "from stockdownloader.util.big_decimal_math\|from stockdownloader.util.config_loader\|from stockdownloader.util.io_helpers\|from stockdownloader.util.parsers\|from stockdownloader.util.technical\|from stockdownloader.util.streaming\|from stockdownloader.util.indicator_hub\|from stockdownloader.util.intraday_indicators\|from stockdownloader.util.smc_indicators\|from stockdownloader.util.streaming_structure\|from stockdownloader.util.black_scholes_calculator\|from stockdownloader.util.pinescript_models\|from stockdownloader.util.pinescript_modes\|from stockdownloader.util.pinescript_ml_strategy\|from stockdownloader.util.timeframe_aggregator\|from stockdownloader.util.event_calendar" src/ tests/
```

Expected: Only hits in shim files (about to be deleted). If any non-shim file still references old paths, fix it first.

**Step 2: Delete all shim files and old directories**

```bash
rm src/stockdownloader/util/big_decimal_math.py
rm src/stockdownloader/util/config_loader.py
rm src/stockdownloader/util/event_calendar.py
rm src/stockdownloader/util/io_helpers.py
rm src/stockdownloader/util/parsers.py
rm src/stockdownloader/util/timeframe_aggregator.py
rm src/stockdownloader/util/indicator_hub.py
rm src/stockdownloader/util/intraday_indicators.py
rm src/stockdownloader/util/smc_indicators.py
rm src/stockdownloader/util/streaming_structure.py
rm src/stockdownloader/util/black_scholes_calculator.py
rm src/stockdownloader/util/pinescript_models.py
rm src/stockdownloader/util/pinescript_modes.py
rm src/stockdownloader/util/pinescript_ml_strategy.py
rm -rf src/stockdownloader/util/technical/
rm -rf src/stockdownloader/util/streaming/
```

**Step 3: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All pass — this is the critical verification

**Step 4: Commit**

```
refactor: delete shim files and old technical/streaming directories
```

---

### Task 20: Final verification and cleanup

**Step 1: Verify final directory structure**

```bash
find src/stockdownloader/util -name "*.py" | sort
```

Expected:
```
src/stockdownloader/util/__init__.py
src/stockdownloader/util/config.py
src/stockdownloader/util/indicators/__init__.py
src/stockdownloader/util/indicators/_core.py
src/stockdownloader/util/indicators/htf.py
src/stockdownloader/util/indicators/hub.py
src/stockdownloader/util/indicators/intraday.py
src/stockdownloader/util/indicators/momentum.py
src/stockdownloader/util/indicators/smc.py
src/stockdownloader/util/indicators/trend.py
src/stockdownloader/util/indicators/volatility.py
src/stockdownloader/util/indicators/volume.py
src/stockdownloader/util/io.py
src/stockdownloader/util/math.py
src/stockdownloader/util/options/__init__.py
src/stockdownloader/util/options/black_scholes.py
src/stockdownloader/util/pinescript/__init__.py
src/stockdownloader/util/pinescript/generator.py
src/stockdownloader/util/pinescript/ml_export.py
src/stockdownloader/util/pinescript/models.py
src/stockdownloader/util/pinescript/modes.py
src/stockdownloader/util/timeframe.py
```

**Step 2: Run full test suite one final time**

Run: `pytest tests/ -v`
Expected: ~3,196 pass, 0 fail

**Step 3: Commit**

```
refactor: util domain restructure complete — 26 files → 22 domain-focused modules
```
