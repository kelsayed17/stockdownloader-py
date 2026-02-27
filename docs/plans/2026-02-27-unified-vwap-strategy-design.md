# Unified Multi-Mode VWAP Strategy Design

**Date**: 2026-02-27
**Branch**: `claude/vigorous-easley`
**Status**: Approved

## Problem

PineScript v11.2 runs all 5 VWAP modes (PS, ORB, ORR, PB, REV) inside a single
strategy with shared state: one `max_day` budget, one `spacing` counter, and
strict priority dispatch (PS > ORB > ORR > PB > REV). Python runs each mode as
a standalone strategy with independent state. This causes:

1. **No priority gating** — PB fires on bars where PS/ORB would be better
2. **No shared day-trade budget** — each mode gets its own `max_day` allocation
3. **No shared spacing** — modes can fire on adjacent bars
4. **REV not suppressed by PB** — 14 REV trades fire that PB would have blocked

After fixing volume scoring, PS gates, and REV configs, the combined standalone
backtest produces +$1,500 (72 trades) vs TV's +$10,892 (107 trades). The
remaining gap is primarily architectural.

## Design

### Approach: Composite Evaluator (Delegate to Existing)

Create `UnifiedVWAPStrategy` that holds a **single** `IntradayInfra` instance
and delegates entry evaluation to existing standalone strategy `_evaluate_entry()`
methods. No existing strategy code is modified.

### Architecture

```
UnifiedVWAPStrategy (extends BaseIntradayStrategy)
├── _infra: IntradayInfra (single shared instance)
│   ├── SessionState (shared day_trades, spacing, risk controls)
│   ├── IndicatorHub (computed once per bar)
│   └── IntradayExitManager (trail swapped per entry)
├── _modes: ordered list of (strategy_instance, TrailClass)
│   ├── (PatternScalpStrategy, BreakevenTrail)    # priority 1
│   ├── (ORBreakoutStrategy, AtrChandelierTrail)  # priority 2
│   ├── (PullbackStrategy, VwapRatchetTrail)      # priority 3
│   └── (ReversalStrategy, BreakevenTrail)        # priority 4
└── _c: UnifiedVWAPConfig
```

ORR (OR Reversal) is disabled in TV v11.2 and omitted from initial implementation.

### Per-Bar Dispatch

```python
def _evaluate_entry(self, ctx: BarContext) -> IntradaySignal | None:
    # Shared risk gates (applied once across all modes)
    s = ctx.state
    if s.day_trades >= self._c.max_day:
        return None
    if s.last_entry_bar > 0 and (ctx.bar_of_day - s.last_entry_bar) < self._c.spacing:
        return None

    # Priority dispatch: first non-None signal wins
    for strategy, trail_cls in self._modes:
        signal = strategy._evaluate_entry(ctx)
        if signal is not None:
            self._infra._exit_manager.set_active_trail(trail_cls())
            return signal
    return None
```

### Config

`UnifiedVWAPConfig` is a frozen dataclass extending `InfraExitConfig`. It holds:

- Shared infra/risk fields: `max_day`, `spacing`, `circuit`, `day_loss`, etc.
- Per-mode enable flags: `pb_enable`, `ps_enable`, `orb_enable`, `rev_enable`

Each standalone strategy is constructed with its own config object (e.g.,
`PullbackStrategyConfig`) containing mode-specific entry parameters. The
standalone instances' `_infra` is unused — only `_evaluate_entry()` is called.

### Trail Swapping

Each mode maps to a trail class:
- PB → `VwapRatchetTrail` (VWAP ratchet, never retreats)
- PS → `BreakevenTrail` (step-wise profit locking)
- ORB → `AtrChandelierTrail` (chandelier from MFE)
- REV → `BreakevenTrail`

When a mode's evaluator returns a signal, `set_active_trail()` injects the
correct trail before the infra records the entry.

### Shared State Benefits

Because all modes share one `SessionState`:
- `day_trades` is incremented once per entry regardless of mode
- `spacing` is checked against the last entry bar across all modes
- `tripped` (circuit breaker) halts all modes simultaneously
- `fired_today` can gate PS (fire-once per day)

### What Changes

| Item | Description |
|------|-------------|
| **New**: `src/.../intraday/unified_vwap.py` | `UnifiedVWAPStrategy` + `UnifiedVWAPConfig` |
| **New**: `tests/.../intraday/test_unified_vwap.py` | Unit tests for priority dispatch, trail swap, shared state |
| **Modified**: `scripts/tv_parity_backtest.py` | Add unified strategy run + comparison |
| **Modified**: `src/.../intraday/__init__.py` | Export new strategy |
| **Unchanged**: All existing strategy files | Entry logic reused via delegation |

### Standalone Strategies

The existing standalone strategies (PullbackStrategy, PatternScalpStrategy, etc.)
remain fully functional for independent use. The unified strategy composes them
but does not replace them.

### Success Criteria

1. Unified strategy produces fewer total trades than sum of standalone modes
   (priority gating suppresses lower-priority signals)
2. Trade distribution more closely matches TV's mode breakdown
   (PB ~77, PS ~18, ORB ~7, REV ~5)
3. Combined PnL improves toward TV's $10,892

### TV-Parity Config for Unified

The unified strategy will use the same TV-matched configs from the tv_parity
backtest, with shared fields:
- `max_day=2`, `spacing=3` (shared across all modes)
- `be_trigger=0.5`, `trail_buf=0.15`, `trail_keep_tp=True`
- `close_eod=True`, `circuit=3`, `day_loss=3.0`
