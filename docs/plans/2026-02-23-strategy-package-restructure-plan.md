# Strategy Package Restructure Implementation Plan (Round 6)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split `intraday/infra.py` (540 lines) and expand root `strategy/__init__.py` (3 → ~20 exports).

**Architecture:** Mechanical extraction of `DayTracker` into its own file, then comprehensive `__init__.py` expansion following the pattern established in backtest/ and data/ rounds.

**Tech Stack:** Python 3.12, dataclasses, pytest

---

### Task 1: Extract `DayTracker` into `day_tracker.py`

**Files:**
- Create: `src/stockdownloader/strategy/intraday/day_tracker.py`
- Modify: `src/stockdownloader/strategy/intraday/infra.py`
- Modify: `src/stockdownloader/strategy/intraday/__init__.py`

**Step 1: Read `infra.py` fully**

Read `src/stockdownloader/strategy/intraday/infra.py` to identify:
- Where `DayTracker` class starts and ends
- What imports `DayTracker` needs
- What imports `IntradayInfra` uses from `DayTracker`

**Step 2: Create `day_tracker.py`**

Create `src/stockdownloader/strategy/intraday/day_tracker.py` containing:
- Module docstring: `"""Day-boundary state transitions — daily bars, opening range, trend tracking."""`
- The `DayTracker` class extracted verbatim from `infra.py`
- Only the imports that `DayTracker` needs (trim unused imports from the original `infra.py` import block)

**Step 3: Update `infra.py`**

In `src/stockdownloader/strategy/intraday/infra.py`:
- Remove the `DayTracker` class definition entirely
- Add import: `from stockdownloader.strategy.intraday.day_tracker import DayTracker`
- Remove any imports that were ONLY used by `DayTracker` (not by `IntradayInfra`)

**Step 4: Update `intraday/__init__.py`**

Change line 34:
```python
# From:
from stockdownloader.strategy.intraday.infra import DayTracker
# To:
from stockdownloader.strategy.intraday.day_tracker import DayTracker
```

Keep the existing `IntradayInfra` import from `infra` on line 44.

**Step 5: Run tests and commit**

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass (3226)

```bash
git add src/stockdownloader/strategy/intraday/day_tracker.py \
        src/stockdownloader/strategy/intraday/infra.py \
        src/stockdownloader/strategy/intraday/__init__.py
git commit -m "refactor: extract DayTracker from infra.py into day_tracker.py"
```

---

### Task 2: Expand Root `strategy/__init__.py`

**Files:**
- Modify: `src/stockdownloader/strategy/__init__.py`

**Step 1: Read the current `__init__.py`**

Read `src/stockdownloader/strategy/__init__.py` (21 lines, 3 exports).

Also read these subpackage `__init__.py` files to understand available exports:
- `src/stockdownloader/strategy/signals/__init__.py`
- `src/stockdownloader/strategy/regime/__init__.py`
- `src/stockdownloader/strategy/daily/__init__.py`

**Step 2: Replace `__init__.py` contents**

Replace the entire file with:

```python
"""Trading strategy implementations.

Subpackages
-----------
daily : Daily equity strategies (SMA, RSI, MACD, etc.)
options : Options strategies (covered call, protective put)
intraday : Standalone intraday strategies and shared infrastructure
regime : Market regime detection and adaptive ensemble meta-strategy
exit_mechanisms : Exit mechanism implementations for tournaments
signals : Signal filtering and routing
"""

# Base interfaces
from stockdownloader.strategy.trading_strategy import (
    IntradayTradingStrategy,
    Signal,
    TradingStrategy,
)
from stockdownloader.strategy.exit_mechanisms.trailing_exit_base import ExitMechanism

# Registries
from stockdownloader.strategy.base_registry import (
    RegistryEntry,
    SignalGeneratorRegistry,
    StrategyRegistry,
)
from stockdownloader.strategy.registration_loader import ensure_registered

# Intraday infrastructure
from stockdownloader.strategy.intraday.base_strategy import BaseIntradayStrategy
from stockdownloader.strategy.intraday.infra import IntradayInfra
from stockdownloader.strategy.intraday.daily_to_intraday_adapter import (
    DailyToIntradayAdapter,
)

# Signal infrastructure
from stockdownloader.strategy.signals.signal_generator import AtomicSignalGenerator
from stockdownloader.strategy.signals.stacked_signal_engine import StackedSignalEngine
from stockdownloader.strategy.signals.stacked_daily_strategy import StackedDailyStrategy
from stockdownloader.strategy.signals.stacked_intraday_strategy import (
    StackedIntradayStrategy,
)

# Ensemble
from stockdownloader.strategy.regime.ensemble_strategy import EnsembleIntradayStrategy

# Subpackages (for strategy.daily, strategy.intraday, etc.)
from stockdownloader.strategy import (  # noqa: E402
    daily,
    exit_mechanisms,
    intraday,
    options,
    regime,
    signals,
)

__all__ = [
    # Base interfaces
    "ExitMechanism",
    "IntradayTradingStrategy",
    "Signal",
    "TradingStrategy",
    # Registries
    "RegistryEntry",
    "SignalGeneratorRegistry",
    "StrategyRegistry",
    "ensure_registered",
    # Intraday infrastructure
    "BaseIntradayStrategy",
    "DailyToIntradayAdapter",
    "IntradayInfra",
    # Signal infrastructure
    "AtomicSignalGenerator",
    "StackedDailyStrategy",
    "StackedIntradayStrategy",
    "StackedSignalEngine",
    # Ensemble
    "EnsembleIntradayStrategy",
    # Subpackages
    "daily",
    "exit_mechanisms",
    "intraday",
    "options",
    "regime",
    "signals",
]
```

IMPORTANT: Read the actual source files to verify these import paths are correct before writing. The imports above are based on exploration but must be verified.

Also verify that the self-import `from stockdownloader.strategy import daily, ...` doesn't cause circular imports. If it does, remove it and just list the subpackages in the docstring.

**Step 3: Run tests and commit**

Run: `python3 -m pytest tests/ -x -q`
Expected: All pass (3226)

```bash
git add src/stockdownloader/strategy/__init__.py
git commit -m "refactor: expand strategy/__init__.py with comprehensive exports"
```

---

### Task 3: Verify Final State

**Step 1: Verify line counts**

```bash
wc -l src/stockdownloader/strategy/intraday/day_tracker.py \
      src/stockdownloader/strategy/intraday/infra.py \
      src/stockdownloader/strategy/__init__.py
```

Expected:
- `day_tracker.py`: ~175 lines
- `infra.py`: ~365 lines (was 540)
- `__init__.py`: ~65 lines (was 21)

**Step 2: Run full test suite**

Run: `python3 -m pytest tests/ -x -q`
Expected: All tests pass (3226+)

**Step 3: Verify imports**

```bash
python3 -c "from stockdownloader.strategy import TradingStrategy, Signal, StrategyRegistry, ensure_registered; print('base OK')"
python3 -c "from stockdownloader.strategy import BaseIntradayStrategy, IntradayInfra, DailyToIntradayAdapter; print('intraday OK')"
python3 -c "from stockdownloader.strategy import AtomicSignalGenerator, StackedSignalEngine; print('signals OK')"
python3 -c "from stockdownloader.strategy import EnsembleIntradayStrategy; print('ensemble OK')"
python3 -c "from stockdownloader.strategy.intraday.day_tracker import DayTracker; print('day_tracker OK')"
python3 -c "from stockdownloader.strategy.intraday import DayTracker; print('intraday re-export OK')"
```
