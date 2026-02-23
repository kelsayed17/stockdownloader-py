# App Package Restructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split `pipeline_app.py` (949 lines) into a `pipeline/` subpackage, split `tournament/stages.py` (866 lines) into core + advanced, and consolidate duplicated helpers.

**Architecture:** Follow the existing `tournament/` subpackage pattern for `pipeline/`. The slim `pipeline_app.py` remains as a backward-compat re-export shim (console scripts + tests import from it). Shared helpers (`status_label`, `box_title`) move to `app_helpers.py` to eliminate 3-way and 2-way duplication.

**Tech Stack:** Python 3.12, dataclasses, argparse, multiprocessing, pytest

---

### Task 1: Consolidate Duplicated Helpers into `app_helpers.py`

**Files:**
- Modify: `src/stockdownloader/app/app_helpers.py`
- Modify: `src/stockdownloader/app/tournament/helpers.py`
- Modify: `src/stockdownloader/app/optimize_app.py`
- Modify: `src/stockdownloader/app/pattern_discovery_app.py`
- Modify: `tests/app/test_app_helpers.py` (or create if not exists)

**Step 1: Read all files involved**

Read these files to understand current state:
- `src/stockdownloader/app/app_helpers.py`
- `src/stockdownloader/app/tournament/helpers.py`
- `src/stockdownloader/app/optimize_app.py` (around line 313)
- `src/stockdownloader/app/pattern_discovery_app.py` (around line 83)
- `tests/app/test_app_helpers.py` (if exists)

**Step 2: Add `status_label()` and `box_title()` to `app_helpers.py`**

Append these two functions to `src/stockdownloader/app/app_helpers.py` (after the existing functions, before EOF):

```python
# ======================================================================
# Output formatting helpers
# ======================================================================


def status_label(degradation: float) -> str:
    """Map walk-forward degradation ratio to a human-readable label.

    Parameters
    ----------
    degradation:
        Walk-forward degradation ratio (0.0 to 1.0+).
        >= 0.8 is ROBUST, >= 0.5 is ACCEPTABLE, below is OVERFIT.
    """
    if degradation >= 0.8:
        return "ROBUST"
    if degradation >= 0.5:
        return "ACCEPTABLE"
    return "OVERFIT"


def box_title(title: str, width: int = 100) -> str:
    """Render a Unicode box around a title string.

    Parameters
    ----------
    title:
        Text to display centered inside the box.
    width:
        Interior width in characters (default 100).
    """
    lines = [
        "\u2554" + "\u2550" * width + "\u2557",
        "\u2551" + title.center(width) + "\u2551",
        "\u255a" + "\u2550" * width + "\u255d",
    ]
    return "\n".join(lines)
```

**Step 3: Write tests for the new helpers**

Add tests to the existing `tests/app/test_app_helpers.py` (or create it). Add:

```python
class TestStatusLabel:
    def test_robust(self):
        assert status_label(0.8) == "ROBUST"
        assert status_label(1.0) == "ROBUST"
        assert status_label(0.95) == "ROBUST"

    def test_acceptable(self):
        assert status_label(0.5) == "ACCEPTABLE"
        assert status_label(0.7) == "ACCEPTABLE"

    def test_overfit(self):
        assert status_label(0.49) == "OVERFIT"
        assert status_label(0.0) == "OVERFIT"
        assert status_label(-0.1) == "OVERFIT"


class TestBoxTitle:
    def test_basic(self):
        result = box_title("HELLO", width=20)
        lines = result.split("\n")
        assert len(lines) == 3
        assert "HELLO" in lines[1]
        assert len(lines[0]) == 22  # width + 2 border chars

    def test_default_width(self):
        result = box_title("TEST")
        lines = result.split("\n")
        assert len(lines[0]) == 102  # 100 + 2 border chars

    def test_unicode_borders(self):
        result = box_title("X", width=10)
        lines = result.split("\n")
        assert lines[0].startswith("\u2554")
        assert lines[0].endswith("\u2557")
        assert lines[2].startswith("\u255a")
        assert lines[2].endswith("\u255d")
```

**Step 4: Update `tournament/helpers.py`**

In `src/stockdownloader/app/tournament/helpers.py`:
- Remove the `_box_title` function definition (lines 18-24)
- Remove the `_status_label` function definition (lines 27-32)
- Add import: `from stockdownloader.app.app_helpers import box_title as _box_title, status_label as _status_label`

This preserves the private naming convention. All internal call sites in `tournament/stages.py` and `tournament/cli.py` that import `_box_title` and `_status_label` from `tournament/helpers` continue to work unchanged.

**Step 5: Update `optimize_app.py`**

In `src/stockdownloader/app/optimize_app.py`:
- Remove the `_status_label` function definition (lines 313-319)
- Add import at top: `from stockdownloader.app.app_helpers import status_label as _status_label`

**Step 6: Update `pattern_discovery_app.py`**

In `src/stockdownloader/app/pattern_discovery_app.py`:
- Remove the `_box_title` function definition (lines 83-89)
- Add import at top: `from stockdownloader.app.app_helpers import box_title as _box_title`

**Step 7: Run tests and commit**

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass (3220+)

```bash
git add src/stockdownloader/app/app_helpers.py \
        src/stockdownloader/app/tournament/helpers.py \
        src/stockdownloader/app/optimize_app.py \
        src/stockdownloader/app/pattern_discovery_app.py \
        tests/app/test_app_helpers.py
git commit -m "refactor: consolidate status_label/box_title into app_helpers, eliminate duplication"
```

---

### Task 2: Split `tournament/stages.py` into Core + Advanced

**Files:**
- Create: `src/stockdownloader/app/tournament/stages_advanced.py`
- Modify: `src/stockdownloader/app/tournament/stages.py`
- Modify: `src/stockdownloader/app/tournament/cli.py`

**Step 1: Read current state**

Read:
- `src/stockdownloader/app/tournament/stages.py` (full file)
- `src/stockdownloader/app/tournament/cli.py` (full file)

**Step 2: Create `tournament/stages_advanced.py`**

Create `src/stockdownloader/app/tournament/stages_advanced.py` containing the last 4 stage functions extracted verbatim from `stages.py`:

- `_run_regime_analysis()` — lines 430-568
- `_run_monte_carlo_stage()` — lines 576-679
- `_run_bracket()` — lines 687-751
- `_run_portfolio()` — lines 759-866

The new file needs these imports (adapted from what these functions use):
```python
from __future__ import annotations

from stockdownloader.app.tournament.helpers import (
    _box_title,
    _status_label,
)
from stockdownloader.backtest.tournament_engine import (
    INITIAL_CAPITAL,
    ComboKey,
    ComboResult,
    MatchResult,
    run_elimination_bracket,
    run_monte_carlo,
    run_regime_analysis,
)
from stockdownloader.backtest.portfolio_analyzer import (
    correlation_matrix,
    portfolio_equity_curve,
    portfolio_metrics,
    select_portfolio,
)
from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.strategy.regime.regime_detector import MarketRegime
from stockdownloader.strategy.regime.regime_strategy_map import RegimeStrategyMapper
```

Add module docstring:
```python
"""Advanced tournament stages — regime analysis, Monte Carlo, bracket, portfolio."""
```

**Step 3: Slim `tournament/stages.py`**

Remove:
- `_run_regime_analysis()` function and its section header
- `_run_monte_carlo_stage()` function and its section header
- `_run_bracket()` function and its section header
- `_run_portfolio()` function and its section header

Remove now-unused imports from stages.py:
- `MatchResult` — only used by `_run_bracket`
- `run_elimination_bracket` — only used by `_run_bracket`
- `run_monte_carlo` — only used by `_run_monte_carlo_stage`
- `run_regime_analysis` — only used by `_run_regime_analysis`
- `from stockdownloader.backtest.portfolio_analyzer import ...` — only used by `_run_portfolio`
- `from stockdownloader.strategy.regime.regime_detector import MarketRegime` — only used by regime analysis
- `from stockdownloader.strategy.regime.regime_strategy_map import RegimeStrategyMapper` — only used by regime analysis
- `INITIAL_CAPITAL` — only used by `_run_portfolio`

Keep:
- `multiprocessing`, `os`, `concurrent.futures` — used by core stages
- `ComboKey`, `ComboResult` — used by core stages
- `classify_timeframe_bars` — used by `_run_regime_analysis` but actually it's in advanced... CHECK. Yes, `classify_timeframe_bars` is called in `_run_regime_analysis` which moves to advanced. Remove from stages.py.
- `run_combo_backtest`, `run_combo_optimize`, `run_combo_rebacktest`, `run_combo_walkforward` — used by core stages
- `apply_cross_timeframe_bonus` — used at end of `_run_round_robin`. KEEP.
- `_box_title`, `_build_skip_set`, `_status_label` — used by core stages
- `IntradayPriceData` — used by core stages
- `ensure_registered`, `StrategyRegistry` — used by `_run_round_robin`

**Step 4: Update `tournament/cli.py`**

In `src/stockdownloader/app/tournament/cli.py`, update the import:

Change:
```python
from stockdownloader.app.tournament.stages import (
    _run_bracket,
    _run_monte_carlo_stage,
    _run_optimization,
    _run_portfolio,
    _run_regime_analysis,
    _run_round_robin,
    _run_walkforward,
)
```

To:
```python
from stockdownloader.app.tournament.stages import (
    _run_optimization,
    _run_round_robin,
    _run_walkforward,
)
from stockdownloader.app.tournament.stages_advanced import (
    _run_bracket,
    _run_monte_carlo_stage,
    _run_portfolio,
    _run_regime_analysis,
)
```

**Step 5: Run tests and commit**

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass

```bash
git add src/stockdownloader/app/tournament/stages_advanced.py \
        src/stockdownloader/app/tournament/stages.py \
        src/stockdownloader/app/tournament/cli.py
git commit -m "refactor: split tournament/stages.py into core + stages_advanced.py"
```

---

### Task 3: Create `pipeline/` Subpackage — Models + Helpers

**Files:**
- Create: `src/stockdownloader/app/pipeline/__init__.py`
- Create: `src/stockdownloader/app/pipeline/models.py`
- Create: `src/stockdownloader/app/pipeline/helpers.py`

**Step 1: Read `pipeline_app.py`**

Read the full `src/stockdownloader/app/pipeline_app.py`.

**Step 2: Create `pipeline/__init__.py`**

```python
"""Unified strategy pipeline — backtest, optimize, re-backtest, validate."""

from stockdownloader.app.pipeline.models import SlotResult
from stockdownloader.app.pipeline.cli import main

__all__ = ["main", "SlotResult"]
```

Note: `cli.py` doesn't exist yet — this import will fail until Task 5. Create a temporary placeholder if needed, or create `__init__.py` last in Task 5.

Actually — better approach: create `__init__.py` as a minimal placeholder now, then update it in Task 5 after cli.py exists:

```python
"""Unified strategy pipeline — backtest, optimize, re-backtest, validate."""
```

**Step 3: Create `pipeline/models.py`**

Extract `SlotResult` dataclass (lines 69-122) from `pipeline_app.py`:

```python
"""Data model for the strategy pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stockdownloader.backtest.backtest_result import (
    BacktestResult,
    BaseBacktestResult,
)
from stockdownloader.backtest.optimizer_scoring import score_v2
from stockdownloader.backtest.walk_forward import WalkForwardResult


@dataclass
class SlotResult:
    """One strategy's results across all pipeline stages."""

    name: str
    display_name: str
    category: str
    baseline: BaseBacktestResult | None = None
    optimized_kwargs: dict[str, Any] | None = None
    optimized: BaseBacktestResult | None = None
    wf_result: WalkForwardResult | None = None

    @property
    def best_result(self) -> BaseBacktestResult | None:
        """Return optimized result if available, else baseline."""
        return self.optimized or self.baseline

    def ranking_score(self, trading_days: int) -> float:
        """Unified score for cross-category ranking."""
        result = self.best_result
        if result is None:
            return -999.0

        if isinstance(result, BacktestResult):
            base = score_v2(result, trading_days)
        else:
            base = float(result.total_return)

        if self.wf_result is not None:
            deg = self.wf_result.degradation_ratio
            if deg >= 0.8:
                factor = 1.0
            elif deg >= 0.5:
                factor = 0.8
            else:
                factor = 0.5

            if base >= 0:
                return base * factor
            else:
                return base / factor
        return base
```

**Step 4: Create `pipeline/helpers.py`**

Extract utility functions (lines 130-165) from `pipeline_app.py`:

```python
"""Pipeline helper utilities — output, data loading."""
from __future__ import annotations

from stockdownloader.data.intraday_csv import IntradayCsvLoader
from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.util.io import TeeWriter


def make_print_fn(tee: TeeWriter | None):
    """Return a print function that writes to TeeWriter or stdout."""
    def _print(msg: str = "") -> None:
        if tee:
            tee.write(msg + "\n")
        else:
            print(msg)
    return _print


def load_intraday_data(csv_path: str, out) -> list[IntradayPriceData]:
    """Load 5-min bars from CSV."""
    out(f"Loading intraday data from {csv_path}...")
    data = IntradayCsvLoader.load_from_file(csv_path)
    if not data:
        out(f"ERROR: No data loaded from {csv_path}")
        return []
    trading_days = len({bar.date[:10] for bar in data})
    out(f"Loaded {len(data):,} bars across {trading_days} trading days")
    out(f"Date range: {data[0].date[:10]} to {data[-1].date[:10]}")
    return data


def load_daily_data(intraday_data: list[IntradayPriceData], out):
    """Aggregate 5-min bars to daily PriceData for options backtesting."""
    from stockdownloader.util.timeframe import TimeframeAggregator, Timeframe
    out("Aggregating 5-min bars to daily for options strategies...")
    agg = TimeframeAggregator(intraday_data)
    daily = agg.as_price_data(Timeframe.DAILY)
    out(f"  {len(daily)} daily bars")
    return daily


def unique_days(data: list[IntradayPriceData]) -> int:
    """Count unique trading days in intraday data."""
    return len({d.date[:10] for d in data})
```

Note: Functions are public (no underscore prefix) since they're now module-level exports.

**Step 5: Run tests and commit**

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass (new files exist but aren't imported by anything yet — no breakage)

```bash
git add src/stockdownloader/app/pipeline/__init__.py \
        src/stockdownloader/app/pipeline/models.py \
        src/stockdownloader/app/pipeline/helpers.py
git commit -m "feat: create pipeline/ subpackage with models.py and helpers.py"
```

---

### Task 4: Create `pipeline/stages.py` + `pipeline/report.py`

**Files:**
- Create: `src/stockdownloader/app/pipeline/stages.py`
- Create: `src/stockdownloader/app/pipeline/report.py`

**Step 1: Create `pipeline/stages.py`**

Extract the 4 stage functions + their workers (lines 173-713) from `pipeline_app.py`. This is the largest file (~540 lines).

The file needs these imports:
```python
from __future__ import annotations

import dataclasses
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from decimal import Decimal
from operator import attrgetter

from stockdownloader.app.app_helpers import status_label
from stockdownloader.app.pipeline.models import SlotResult
from stockdownloader.backtest.backtest_result import (
    BacktestResult,
    BaseBacktestResult,
    OptionsBacktestResult,
)
from stockdownloader.backtest.intraday_backtest_engine import IntradayBacktestEngine
from stockdownloader.backtest.optimizer_scoring import score_v2
from stockdownloader.backtest.walk_forward import WalkForwardResult, WalkForwardValidator
from stockdownloader.model.price_data import IntradayPriceData
from stockdownloader.strategy.intraday.daily_to_intraday_adapter import DailyToIntradayAdapter
from stockdownloader.strategy.registration_loader import ensure_registered
from stockdownloader.strategy.base_registry import StrategyRegistry
from stockdownloader.util.config import INITIAL_CAPITAL, OPTIONS_COMMISSION, RISK_PER_TRADE

_MP_CTX = multiprocessing.get_context("fork")
```

Copy the following functions EXACTLY from `pipeline_app.py`:
- `_baseline_one_slot()` + `_run_baseline()` (Stage 1, lines 173-257)
- `_run_optimize()` + `_optimize_one_strategy()` + `_run_optimize_wf()` + `_run_optimize_full()` (Stage 2, lines 264-489)
- `_rebacktest_one_slot()` + `_run_rebacktest()` (Stage 3, lines 497-587)
- `_validate_one_slot()` + `_run_walkforward()` (Stage 4, lines 594-713)

**IMPORTANT**: Replace the inline `_status_label` call (originally at line 706) with `status_label` (imported from `app_helpers`). The original function `_status_label` at line 706-712 of pipeline_app.py should NOT be copied — use the shared one.

Also use helper functions from `pipeline.helpers`:
- Replace `_unique_days(data)` calls with `from stockdownloader.app.pipeline.helpers import unique_days` and call `unique_days(data)`

**Step 2: Create `pipeline/report.py`**

Extract `_print_holistic_report()` (lines 720-816) from `pipeline_app.py`:

```python
"""Holistic pipeline report formatter."""
from __future__ import annotations

from decimal import Decimal
from operator import attrgetter

from stockdownloader.app.app_helpers import status_label
from stockdownloader.app.pipeline.helpers import unique_days
from stockdownloader.app.pipeline.models import SlotResult
from stockdownloader.backtest.backtest_result import BacktestResult
from stockdownloader.model.price_data import IntradayPriceData
```

Copy `_print_holistic_report()` verbatim. Replace `_status_label(...)` calls with `status_label(...)`. Replace `_unique_days(...)` calls with `unique_days(...)`.

**Step 3: Run tests and commit**

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass

```bash
git add src/stockdownloader/app/pipeline/stages.py \
        src/stockdownloader/app/pipeline/report.py
git commit -m "feat: create pipeline/stages.py and pipeline/report.py"
```

---

### Task 5: Create `pipeline/cli.py` + Slim `pipeline_app.py`

**Files:**
- Create: `src/stockdownloader/app/pipeline/cli.py`
- Modify: `src/stockdownloader/app/pipeline/__init__.py`
- Modify: `src/stockdownloader/app/pipeline_app.py`

**Step 1: Create `pipeline/cli.py`**

Extract `main()`, `_run_pipeline()`, and constants (lines 54-61, 823-949) from `pipeline_app.py`:

```python
"""Pipeline CLI — argument parsing and entry point."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import TextIO

from stockdownloader.app.app_helpers import add_intraday_csv_arg, add_log_arg
from stockdownloader.app.pipeline.helpers import (
    load_daily_data,
    load_intraday_data,
    make_print_fn,
)
from stockdownloader.app.pipeline.models import SlotResult
from stockdownloader.app.pipeline.report import print_holistic_report
from stockdownloader.app.pipeline.stages import (
    _run_baseline,
    _run_optimize,
    _run_rebacktest,
    _run_walkforward,
)
from stockdownloader.strategy.registration_loader import ensure_registered
from stockdownloader.strategy.base_registry import StrategyRegistry
from stockdownloader.util.io import TeeWriter

_CATEGORIES = ("intraday", "daily", "options")
_STAGES = ("backtest", "optimize", "validate", "all")
_OPTIMIZE_MODES = ("wf", "full", "skip")
```

Copy `main()` and `_run_pipeline()` from `pipeline_app.py` lines 823-949.

Update internal references:
- `_print_fn(tee)` → `make_print_fn(tee)`
- `_load_intraday_data(...)` → `load_intraday_data(...)`
- `_load_daily_data(...)` → `load_daily_data(...)`
- `_print_holistic_report(...)` → `print_holistic_report(...)`

Note: The report function should be exported as `print_holistic_report` (public name) from `pipeline/report.py`.

**Step 2: Update `pipeline/__init__.py`**

Replace contents:
```python
"""Unified strategy pipeline — backtest, optimize, re-backtest, validate."""

from stockdownloader.app.pipeline.cli import main, _OPTIMIZE_MODES
from stockdownloader.app.pipeline.models import SlotResult

__all__ = ["main", "SlotResult", "_OPTIMIZE_MODES"]
```

**Step 3: Slim `pipeline_app.py`**

Replace the ENTIRE contents of `src/stockdownloader/app/pipeline_app.py` with:

```python
"""Backward-compatibility re-exports for the pipeline subpackage.

The pipeline has moved to :mod:`stockdownloader.app.pipeline`.
This module re-exports the public API so that existing console
scripts and imports continue to work.
"""
from stockdownloader.app.pipeline.cli import main, _OPTIMIZE_MODES  # noqa: F401
from stockdownloader.app.pipeline.models import SlotResult  # noqa: F401
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/app/test_pipeline_app.py -v`
Expected: All PASS (tests import `main`, `SlotResult`, `_OPTIMIZE_MODES` from `pipeline_app` which re-exports)

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass (3220+)

**Step 5: Commit**

```bash
git add src/stockdownloader/app/pipeline/cli.py \
        src/stockdownloader/app/pipeline/__init__.py \
        src/stockdownloader/app/pipeline_app.py
git commit -m "refactor: create pipeline/cli.py, slim pipeline_app.py to re-export shim"
```

---

### Task 6: Verify Final State + Line Counts

**Step 1: Verify line counts**

Run:
```bash
wc -l src/stockdownloader/app/pipeline/models.py \
      src/stockdownloader/app/pipeline/helpers.py \
      src/stockdownloader/app/pipeline/stages.py \
      src/stockdownloader/app/pipeline/report.py \
      src/stockdownloader/app/pipeline/cli.py \
      src/stockdownloader/app/pipeline/__init__.py \
      src/stockdownloader/app/pipeline_app.py \
      src/stockdownloader/app/tournament/stages.py \
      src/stockdownloader/app/tournament/stages_advanced.py
```

Expected approximate:
- `pipeline/models.py`: ~60 lines
- `pipeline/helpers.py`: ~55 lines
- `pipeline/stages.py`: ~540 lines
- `pipeline/report.py`: ~100 lines
- `pipeline/cli.py`: ~130 lines
- `pipeline/__init__.py`: ~7 lines
- `pipeline_app.py`: ~10 lines (was 949)
- `tournament/stages.py`: ~330 lines (was 866)
- `tournament/stages_advanced.py`: ~540 lines

**Step 2: Run full test suite one final time**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass (3220+)

**Step 3: Verify no broken imports**

Run:
```bash
python3 -c "from stockdownloader.app.pipeline_app import main, SlotResult, _OPTIMIZE_MODES; print('pipeline_app imports OK')"
python3 -c "from stockdownloader.app.pipeline import main, SlotResult; print('pipeline imports OK')"
python3 -c "from stockdownloader.app.tournament.stages import _run_round_robin, _run_optimization, _run_walkforward; print('tournament stages OK')"
python3 -c "from stockdownloader.app.tournament.stages_advanced import _run_regime_analysis, _run_monte_carlo_stage, _run_bracket, _run_portfolio; print('tournament stages_advanced OK')"
python3 -c "from stockdownloader.app.app_helpers import status_label, box_title; print('app_helpers OK')"
```
