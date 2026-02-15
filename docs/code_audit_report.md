# Code Audit Report

## Project: stockdownloader-py (charming-shamir branch)

**Audit Date:** February 2026
**Scope:** Full codebase -- ~120 source files across 8 packages
**Test Suite:** 1,254 tests passing, 48 deselected (live tests)

---

## Summary

| Severity | Count | Fixed | Remaining |
|----------|-------|-------|-----------|
| Critical | 3 | 1 | 2 |
| High | 6 | 0 | 6 |
| Medium | 11 | 0 | 11 |
| Low | 8 | 0 | 8 |

**Overall Assessment:** The codebase is well-structured with clean architecture (strategy registry, ABC-based strategies, incremental caching). The most impactful issues are the mutable class-level registry dict (C1), silent exception swallowing in data parsers (H1), and Decimal construction inconsistency (C2).

---

## CRITICAL

### C1. Mutable class-level dict on StrategyRegistry [OPEN]
**File:** `src/stockdownloader/strategy/registry.py:60`

The `_entries` dict is a mutable class variable shared globally. Strategies self-register at import time, creating test pollution risks and thread-safety hazards for parallel backtest optimization.

**Recommendation:** Add a threading lock around `register()` and `clear()`, or use a `defaultdict` pattern with per-test isolation via pytest fixtures.

### C2. `Decimal(0)` used instead of `Decimal("0")` in data parsers [OPEN]
**Files:** `yahoo_data_client.py`, `yahoo_options_client.py`, `yahoo_finance_client.py`, `yahoo_historical_client.py` (10 occurrences)

While `Decimal(0)` is technically safe (integer argument), it creates inconsistency with the rest of the codebase which uses `Decimal("0")`. A latent bug if any occurrence changes to `Decimal(0.1)` (float).

**Recommendation:** Global find-replace `Decimal(0)` to `Decimal("0")` in data modules.

### C3. `avg_trade_duration_bars` was a hardcoded placeholder [FIXED]
**File:** `src/stockdownloader/backtest/backtest_result.py:262-281`

Both branches returned `1`, making the metric useless. **Fixed** to parse ISO datetime strings and compute actual duration in 5-minute bar equivalents.

---

## HIGH

### H1. Silent exception swallowing in data parsers [OPEN]
**Files:** `yahoo_data_client.py:309-310`, `yahoo_options_client.py:220`, `yahoo_finance_client.py:127` (6 locations)

Bare `except Exception: pass` catches network errors, JSON failures, and programming bugs silently. Data quality issues become invisible.

**Recommendation:** Log at `DEBUG` level minimum; re-raise on non-network exceptions.

### H2. `type: ignore[attr-defined]` monkey-patching on BacktestResult [OPEN]
**File:** `src/stockdownloader/backtest/intraday_backtest_engine.py:144`

`result.trade_modes = trade_modes` dynamically sets an attribute not defined on `BacktestResult`. Will cause `AttributeError` if accessed on non-intraday results.

**Recommendation:** Add `trade_modes: list[str]` as an optional field on `BacktestResult`.

### H3. Options volatility estimation uses wrong window at bar 0 [OPEN]
**File:** `src/stockdownloader/backtest/options_backtest_engine.py:96`

For bar 0, `min(i + 1, 20)` gives period=1, producing degenerate volatility. No guard against sub-minimum lookback.

**Recommendation:** Skip options trades until sufficient volatility history exists.

### H4. Hardcoded volatility lookback constant [OPEN]
**File:** `src/stockdownloader/backtest/options_backtest_engine.py:32`

`_VOLATILITY_LOOKBACK = 20` is fixed. Different data timeframes need different lookbacks.

**Recommendation:** Make configurable via constructor parameter.

### H5. Empty TYPE_CHECKING block and unused import [OPEN]
**File:** `src/stockdownloader/backtest/options_backtest_engine.py:19-22`

Dead code: `if TYPE_CHECKING: pass` and unused `ROUND_HALF_UP` import.

**Recommendation:** Remove dead code.

### H6. Hardcoded contracts cap of 10 [OPEN]
**File:** `src/stockdownloader/backtest/options_backtest_engine.py:185`

`min(contracts, 10)` silently limits position sizes without logging or configuration.

**Recommendation:** Make configurable; add debug logging when cap is hit.

---

## MEDIUM

### M1-M2. Unused imports [OPEN]
- `strategy/daily/__init__.py:6` -- unused `Decimal` import
- `model/pattern_result.py:7` -- unused `FrozenSet` from typing

### M3. Unusual `Decimal(str(float('inf')))` pattern [OPEN]
**File:** `technical_indicators.py:181,468,570,691`

Creates `Decimal('inf')` via float roundtrip. Use `Decimal('Infinity')` for clarity.

### M4. Float-to-Decimal roundtrip in std-dev calculation [OPEN]
**File:** `technical_indicators.py:1010,1076`

Breaks Decimal precision chain for `math.sqrt()`. Acceptable for statistics but inconsistent.

### M5. IndicatorHub cache keyed by `id(data)` [OPEN]
**File:** `indicator_hub.py:96-102`

If data list is recreated with same contents, cache invalidates unnecessarily. If `id()` is reused after GC, stale values could be served. Pragmatic but should be documented.

### M6. `print()` in library code [OPEN]
**File:** `data/polygon_data_client.py:238-256`

Library module uses `print()` instead of `logger.info()`.

### M7. Sortino denominator uses full count `n` [OPEN]
**File:** `backtest_result.py:214`

Divides by total periods, not just downside periods. Both approaches are valid but should be documented.

### M8. `session_start_equity` always zero unless set externally [OPEN]
**File:** `strategy/vwap_strategy/session_state.py:137`

Daily loss limit can never trigger because equity is zero. Guarded by `> _ZERO` check.

### M9-M10. Missing `__init__.py` exports [OPEN]
- `data/__init__.py` missing: `intraday_csv_writer`, `intraday_data_accumulator`, `polygon_data_client`
- `backtest/__init__.py` missing: `combinatorial_tester`, `daily_strategy_optimizer`, `optimizer_scoring`, `strategy_optimizer`, `walk_forward`

### M11. ExitMechanismSummary float roundtrip [OPEN]
**File:** `model/exit_mechanism_result.py:174-179`

Same Decimal -> float -> Decimal pattern as M4.

---

## LOW

### L1-L8 Summary
- `AlertResult.confluence_score` typed `float` while rest uses `Decimal`
- Duplicate `Direction` enum in `trade.py` and `alert_result.py`
- ~20% of public methods lack docstrings
- Deprecated `typing.FrozenSet` import alongside PEP 585 `frozenset`
- Empty `analysis/__init__.py` and `app/__init__.py` stubs
- `__all__` in `strategy/__init__.py` doesn't include backward-compat classes
- `_get_decimal_at` doesn't check negative indices
- EMA seed window varies with position

---

## Test Coverage Gaps

| Module | Status |
|--------|--------|
| `backtest/backtest_report_formatter.py` | No test |
| `backtest/options_backtest_report_formatter.py` | No test |
| `backtest/exit_tournament_report_formatter.py` | No test |
| `backtest/intraday_backtest_report_formatter.py` | No test |
| `data/tradingview_trade_loader.py` | No test |
| `util/file_helper.py` | No test |
| `model/alert_result.py` | No test |
| `model/indicator_values.py` | No test |
| All `app/*.py` modules | No unit tests |

Report formatters are pure formatting (low risk). The data loader and utility gaps are more concerning for regression safety.

---

## Performance Notes

The streaming indicator migration (Phase 1) addressed the biggest performance bottleneck: `IndicatorHub` now uses O(1) streaming accumulators for RSI, EMA, ATR, ADX, MACD, and SAR. The remaining `session_vwap` is O(k^2) per session but with k=78 bars this is negligible.

---

## Architecture Strengths

1. **Clean ABC hierarchy**: `TradingStrategy` (daily) and `IntradayTradingStrategy` (intraday) with well-defined contracts
2. **Strategy registry**: Self-registration pattern enables CLI resolution and optimizer discovery
3. **Indicator caching**: `IndicatorHub` with `id(data)` binding prevents redundant computation
4. **Streaming accumulators**: O(1) per-bar updates for hot-path indicators
5. **All-Decimal arithmetic**: Consistent use of `Decimal` for financial precision
6. **Frozen dataclasses**: Immutable models (`PriceData`, `ADXResult`, `SessionVWAP`, etc.)
7. **Walk-forward validation**: Anti-overfitting framework with degradation ratio
