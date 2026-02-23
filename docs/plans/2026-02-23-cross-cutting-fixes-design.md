# Cross-Cutting Fixes Design (Round 5)

**Goal:** Final cleanup pass after 4 rounds of package restructuring. Fix import inconsistencies, resolve naming confusion, and add missing documentation.

---

## Fix 1: Standardize Constant Imports

**Problem:** `pattern_discovery_app.py` imports `INITIAL_CAPITAL` and `RISK_PER_TRADE` from `backtest.tournament_engine` instead of their canonical source `util.config`. This creates a fragile transitive dependency.

**Solution:** Change the import to `from stockdownloader.util.config import INITIAL_CAPITAL, RISK_PER_TRADE`. Keep the `tournament_engine` import for `run_monte_carlo` and `classify_timeframe_bars` which genuinely belong there.

**Impact:** 1 file, 1 import block change.

---

## Fix 2: Rename `analysis/signal_generator.py` → `alert_generator.py`

**Problem:** Two unrelated modules share the name `signal_generator`:
- `analysis/signal_generator.py` (567 lines) — confluence-based alert generation with options recommendations
- `strategy/signals/signal_generator.py` — abstract base for atomic signal generators (`AtomicSignalGenerator`)

The `analysis/` module generates *alerts* (buy/sell with recommendations), not *signals* (atomic indicator outputs). The naming collision confuses developers.

**Solution:** Rename `analysis/signal_generator.py` → `analysis/alert_generator.py`. Update the 4 importers:
1. `analysis/__init__.py`
2. `analysis/signal_advisor.py`
3. `app/gme_analysis_app.py`
4. `app/symbol_analysis_app.py`

All import `generate_alert` — the function name stays the same; only the module name changes. Add a backward-compat re-export in `analysis/__init__.py` (which already re-exports `generate_alert`).

**Impact:** 1 rename, 4 import updates. Zero API change since `generate_alert` is already re-exported via `analysis/__init__.py`.

---

## Fix 3: Regime Detection Documentation

**Problem:** The regime detection system spans multiple modules without centralized documentation:
- `strategy/regime/regime_detector.py` — detection logic (5 regimes)
- `strategy/regime/regime_strategy_map.py` — regime-to-strategy mapping
- `backtest/tournament_analysis.py` — regime-aware statistics
- `app/tournament/stages_advanced.py` — tournament regime analysis stage

**Solution:** Add a `docs/design/regime-detection.md` design document covering:
- The 5 market regimes and their detection methodology
- How regime detection integrates with the tournament pipeline
- Cross-module dependency map
