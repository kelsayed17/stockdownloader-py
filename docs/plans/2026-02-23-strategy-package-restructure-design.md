# Strategy Package Restructure Design (Round 6)

**Goal:** Split the oversized `intraday/infra.py` (540 lines) and expand the minimal root `__init__.py` (3 exports → ~20).

---

## Current State

The strategy/ package spans 10,692 lines across 38 modules in 7 subpackages. The architecture is clean — composition-based design, no circular dependencies, good separation between subpackages (signals/ and intraday/ are completely independent).

**Issues identified:**

1. **`intraday/infra.py` (540 lines)** — Two distinct classes (`DayTracker` 175 LOC, `IntradayInfra` 340 LOC) crammed into one file. `DayTracker` handles day-boundary state transitions while `IntradayInfra` orchestrates indicators, risk, exits, and bar context construction. These are separate responsibilities.

2. **Root `__init__.py` (21 lines, 3 exports)** — Only exports `ExitMechanism`, `IntradayTradingStrategy`, `StrategyRegistry`. Missing: `TradingStrategy`, `Signal`, `SignalGeneratorRegistry`, `ensure_registered`, `BaseIntradayStrategy`, `DailyToIntradayAdapter`, signal abstractions, and subpackage re-exports.

## What We're NOT Doing (YAGNI)

- **Config refactoring** — Splitting `InfraExitConfig` (28 fields) into sub-configs would touch all 9 strategy files for marginal benefit. The config pattern works.
- **Signal generator registry refactoring** — The auto-registration in `generators/__init__.py` works well. No need for a separate `signal_registry.py`.
- **Daily base template** — 6 daily strategies are small and focused. Adding a template adds abstraction without clear benefit.
- **Feature additions** — Time-weighted aggregation, stateful generators, etc. are new features, not cleanup.

## Fix 1: Split `intraday/infra.py`

Extract `DayTracker` into its own file:

- **Create `intraday/day_tracker.py`** (~175 lines) — Contains `DayTracker` class with day-boundary transitions, OR tracking, trend tracking, NR7/gap detection
- **Slim `intraday/infra.py`** (~365 lines) — Contains only `IntradayInfra`, imports `DayTracker` from `day_tracker`
- **Update `intraday/__init__.py`** — Import `DayTracker` from its new location (already exported, just change source)

This is a mechanical extraction following the exact same pattern used in Rounds 3-4.

## Fix 2: Expand Root `strategy/__init__.py`

Add comprehensive exports organized by category:

- **Base interfaces**: `TradingStrategy`, `IntradayTradingStrategy`, `Signal`, `ExitMechanism`
- **Registries**: `StrategyRegistry`, `SignalGeneratorRegistry`, `RegistryEntry`, `ensure_registered`
- **Intraday infrastructure**: `BaseIntradayStrategy`, `IntradayInfra`, `DailyToIntradayAdapter`
- **Signal infrastructure**: `AtomicSignalGenerator`, `StackedSignalEngine`, `StackedDailyStrategy`, `StackedIntradayStrategy`
- **Ensemble**: `EnsembleIntradayStrategy`
- **Subpackage modules**: `daily`, `intraday`, `signals`, `regime`, `exit_mechanisms`, `options`
