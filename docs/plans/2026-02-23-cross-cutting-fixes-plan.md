# Cross-Cutting Fixes Implementation Plan (Round 5)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Final cleanup pass — fix import inconsistencies, resolve naming confusion, add regime detection documentation.

**Architecture:** Three independent fixes touching different parts of the codebase. No interdependencies between tasks.

**Tech Stack:** Python 3.12, pytest

---

### Task 1: Standardize Constant Imports in `pattern_discovery_app.py`

**Files:**
- Modify: `src/stockdownloader/app/pattern_discovery_app.py`

**Step 1: Read the file**

Read `src/stockdownloader/app/pattern_discovery_app.py` lines 39-44 to see current imports.

**Step 2: Split the `tournament_engine` import**

In `src/stockdownloader/app/pattern_discovery_app.py`, change lines 39-44 from:

```python
from stockdownloader.backtest.tournament_engine import (
    INITIAL_CAPITAL,
    RISK_PER_TRADE,
    run_monte_carlo,
)
from stockdownloader.backtest.tournament_engine import classify_timeframe_bars
```

To:

```python
from stockdownloader.backtest.tournament_engine import (
    classify_timeframe_bars,
    run_monte_carlo,
)
from stockdownloader.util.config import INITIAL_CAPITAL, RISK_PER_TRADE
```

This imports constants from their canonical source (`util.config`) while keeping `run_monte_carlo` and `classify_timeframe_bars` from `tournament_engine` where they genuinely belong.

**Step 3: Run tests and commit**

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass (3226)

```bash
git add src/stockdownloader/app/pattern_discovery_app.py
git commit -m "fix: import INITIAL_CAPITAL/RISK_PER_TRADE from canonical util.config"
```

---

### Task 2: Rename `analysis/signal_generator.py` → `alert_generator.py`

**Files:**
- Rename: `src/stockdownloader/analysis/signal_generator.py` → `src/stockdownloader/analysis/alert_generator.py`
- Modify: `src/stockdownloader/analysis/__init__.py`
- Modify: `src/stockdownloader/analysis/signal_advisor.py`
- Modify: `src/stockdownloader/app/gme_analysis_app.py`
- Modify: `src/stockdownloader/app/symbol_analysis_app.py`

**Step 1: Read all files to confirm current imports**

Read:
- `src/stockdownloader/analysis/__init__.py` (line 6)
- `src/stockdownloader/analysis/signal_advisor.py` (line 22)
- `src/stockdownloader/app/gme_analysis_app.py` (line 49)
- `src/stockdownloader/app/symbol_analysis_app.py` (line 28)

All four import `generate_alert` from `stockdownloader.analysis.signal_generator`.

**Step 2: Rename the file**

```bash
git mv src/stockdownloader/analysis/signal_generator.py src/stockdownloader/analysis/alert_generator.py
```

**Step 3: Update `analysis/__init__.py`**

Change line 6:

```python
# From:
from stockdownloader.analysis.signal_generator import generate_alert
# To:
from stockdownloader.analysis.alert_generator import generate_alert
```

**Step 4: Update `analysis/signal_advisor.py`**

Change line 22:

```python
# From:
from stockdownloader.analysis.signal_generator import generate_alert
# To:
from stockdownloader.analysis.alert_generator import generate_alert
```

**Step 5: Update `app/gme_analysis_app.py`**

Change line 49:

```python
# From:
from stockdownloader.analysis.signal_generator import generate_alert
# To:
from stockdownloader.analysis.alert_generator import generate_alert
```

**Step 6: Update `app/symbol_analysis_app.py`**

Change line 28:

```python
# From:
from stockdownloader.analysis.signal_generator import generate_alert
# To:
from stockdownloader.analysis.alert_generator import generate_alert
```

**Step 7: Run tests and commit**

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass (3226)

```bash
git add src/stockdownloader/analysis/alert_generator.py \
        src/stockdownloader/analysis/__init__.py \
        src/stockdownloader/analysis/signal_advisor.py \
        src/stockdownloader/app/gme_analysis_app.py \
        src/stockdownloader/app/symbol_analysis_app.py
git commit -m "refactor: rename analysis/signal_generator.py to alert_generator.py for clarity"
```

---

### Task 3: Add Regime Detection Design Document

**Files:**
- Create: `docs/design/regime-detection.md`

**Step 1: Read the regime detection modules**

Read:
- `src/stockdownloader/strategy/regime/regime_detector.py` (first 100 lines for the classify method and regime definitions)
- `src/stockdownloader/strategy/regime/regime_strategy_map.py` (first 50 lines for the mapper)
- `src/stockdownloader/strategy/regime/__init__.py` (already read — has good docstring)
- `src/stockdownloader/backtest/tournament_analysis.py` — grep for `regime` to see how it's used

**Step 2: Create the design document**

Create `docs/design/regime-detection.md` documenting:

1. **Overview** — What regime detection does and why
2. **The 5 Market Regimes** — `STRONG_TREND_UP`, `STRONG_TREND_DOWN`, `WEAK_TREND`, `MEAN_REVERTING`, `HIGH_VOLATILITY`
3. **Detection Methodology** — ADX, Bollinger Band width, linear regression slope, SMA(200) position
4. **Integration Points**:
   - `strategy/regime/ensemble_strategy.py` — adaptive strategy selection
   - `backtest/tournament_analysis.py` — regime-aware trade statistics
   - `app/tournament/stages_advanced.py` — tournament regime analysis stage
5. **Module Dependency Map**

The document should be concise (under 150 lines) and factual. Read the actual source code to extract the detection thresholds and methodology — don't guess.

**Step 3: Commit**

```bash
mkdir -p docs/design
git add docs/design/regime-detection.md
git commit -m "docs: add regime detection design document"
```

---

### Task 4: Verify Final State

**Step 1: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass (3226+)

**Step 2: Verify import consistency**

```bash
python3 -c "from stockdownloader.analysis.alert_generator import generate_alert; print('alert_generator OK')"
python3 -c "from stockdownloader.analysis import generate_alert; print('analysis re-export OK')"
python3 -c "from stockdownloader.app.pattern_discovery_app import main; print('pattern_discovery OK')"
```

**Step 3: Report line counts of all changed files**

```bash
wc -l src/stockdownloader/app/pattern_discovery_app.py \
      src/stockdownloader/analysis/alert_generator.py \
      src/stockdownloader/analysis/__init__.py
```
