# Round 12: Split Monoliths — util/ Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split the two largest source files in the codebase — `pinescript/generator.py` (1,212 lines) and `indicators/hub.py` (894 lines) — into focused modules.

**Architecture:** Extract strategy-mode rendering (~450 lines) and composite-mode rendering (~230 lines) from `generator.py` into separate modules with standalone functions, imported by the main generator class. Extract session-scoped, anchored-VWAP, and intraday indicator groups (~280 lines) from `hub.py` into a `hub_intraday.py` mixin, keeping core hub under 620 lines.

**Tech Stack:** Python, no new dependencies

---

### Task 1: Extract strategy-mode rendering from generator.py

**Files:**
- Create: `src/stockdownloader/util/pinescript/strategy_renderer.py`
- Modify: `src/stockdownloader/util/pinescript/generator.py`

**What to extract** (lines 529-983 of generator.py → new file):

Move these methods to module-level functions in `strategy_renderer.py`:
- `render_strategy_header(name, short_name, desc_block, s)` — was `_render_strategy_header` (lines 533-553)
- `emit_strategy_logic(s, use_session_filter)` — was `_emit_strategy_logic` (lines 555-916, **362 lines**)
- `render_strategy_background(s)` — was `_render_strategy_background` (lines 918-931)
- `emit_state_machine(use_session_filter)` — was `_emit_state_machine` (lines 933-983)

These methods only use `self` to call static helpers (`_section_header`, `_section`, `_infer_type`). Convert them to standalone functions that import the static helpers.

**In generator.py**: Replace the method bodies with calls to the new functions:
```python
from stockdownloader.util.pinescript.strategy_renderer import (
    render_strategy_header,
    emit_strategy_logic,
    render_strategy_background,
    emit_state_machine,
)

# In PineScriptGenerator class:
def _render_strategy_header(self, name, short_name, desc_block, s):
    return render_strategy_header(name, short_name, desc_block, s)

def _emit_strategy_logic(self, s, use_session_filter):
    return emit_strategy_logic(s, use_session_filter)

def _render_strategy_background(self, s):
    return render_strategy_background(s)

def _emit_state_machine(self, use_session_filter):
    return emit_state_machine(use_session_filter)
```

**Tests:**
1. Run: `python3 -m pytest tests/util/test_pinescript_generator.py tests/util/test_pinescript_strategy_mode.py -x -q`
2. Run: `python3 -m pytest tests/util/test_pinescript_composite.py tests/util/test_pinescript_spy_strategies.py tests/util/test_pinescript_gme_prediction.py tests/util/test_pinescript_ml_strategy.py -x -q`
3. Full regression: `python3 -m pytest tests/ -x -q`

**Commit:** `refactor: extract strategy-mode renderer from PineScriptGenerator`

### Task 2: Extract composite-mode rendering from generator.py

**Files:**
- Create: `src/stockdownloader/util/pinescript/composite_renderer.py`
- Modify: `src/stockdownloader/util/pinescript/generator.py`

**What to extract** (lines 257-492 of generator.py → new file):

Move these methods to module-level functions in `composite_renderer.py`:
- `composite_header(comp)` — was `_composite_header` (lines 287-297)
- `composite_inputs(comp)` — was `_composite_inputs` (lines 299-329)
- `composite_session(comp)` — was `_composite_session` (lines 331-334)
- `composite_shared_indicators(comp)` — was `_composite_shared_indicators` (lines 336-342)
- `composite_mode_section(mode, comp)` — was `_composite_mode_section` (lines 344-375)
- `composite_aggregation(comp)` — was `_composite_aggregation` (lines 377-428)
- `composite_labels(comp)` — was `_composite_labels` (lines 430-451)
- `composite_background(comp)` — was `_composite_background` (lines 453-459)
- `composite_alerts(comp)` — was `_composite_alerts` (lines 461-492)

These methods reference `self` for helpers like `_input_line`, `_render_indicators`, `_emit_label`, `_emit_alert`, `_section_header`, `_render_session`. Two approaches:

**Option A (preferred):** Extract as functions, pass `gen: PineScriptGenerator` as first parameter to the ones that need helper access. Then `gen._input_line(...)`, etc.

**Option B:** Extract helper methods themselves to render_utils and pass as standalone functions.

Use Option A — it's the simplest refactoring with minimal risk.

**In generator.py**: Replace `generate_composite()` body to delegate to the new functions:
```python
from stockdownloader.util.pinescript.composite_renderer import (
    composite_header, composite_inputs, composite_session,
    composite_shared_indicators, composite_mode_section,
    composite_aggregation, composite_labels, composite_background,
    composite_alerts,
)
```

**Tests:**
1. Run: `python3 -m pytest tests/util/test_pinescript_composite.py -x -q`
2. Full regression: `python3 -m pytest tests/ -x -q`

**Commit:** `refactor: extract composite-mode renderer from PineScriptGenerator`

### Task 3: Extract intraday/session/anchored-VWAP methods from hub.py

**Files:**
- Create: `src/stockdownloader/util/indicators/hub_intraday.py`
- Modify: `src/stockdownloader/util/indicators/hub.py`

**What to extract** (lines 550-893 of hub.py → new mixin class):

Create `IntraDayHubMixin` class in `hub_intraday.py` containing:

**Session-scoped (lines 554-633):**
- `_vwap_core()` (lines 554-563)
- `session_vwap()` (lines 565-576)
- `session_vwap_bands()` (lines 578-600)
- `extended_session_vwap_bands()` (lines 602-633)

**Anchored VWAP (lines 639-694):**
- `_avwap_core()` (lines 639-649)
- `anchored_vwap_bands()` (lines 651-694)

**Intraday indicators (lines 700-813):**
- `tod_rvol()` (lines 700-711)
- `cvd_session()` (lines 713-725)
- `cvd_normalized()` (lines 727-750)
- `linear_regression_slope()` (lines 752-762)
- `lrs_normalized()` (lines 764-779)
- `vwap_slope()` (lines 781-795)
- `vwap_acceleration()` (lines 797-813)

**Higher-timeframe (lines 815-846):**
- `htf_ema_trend()` (lines 815-846)

**Relative volume (lines 848-858):**
- `rel_vol()` (lines 848-858)

**Market structure (lines 864-893):**
- `structure_state()` (lines 864-893)

**In hub.py**: Change `IndicatorHub` to inherit from both `IntraDayHubMixin` and keep its own methods:
```python
from stockdownloader.util.indicators.hub_intraday import IntraDayHubMixin

class IndicatorHub(IntraDayHubMixin):
    # ... core + momentum + volatility + trend + volume methods
```

The mixin uses `self._cache`, `self._ensure_bound()`, `self._get()`, and streaming accumulators (`self._s_vwap`, `self._s_avwap`, etc.) from the main hub. Use `TYPE_CHECKING` and annotations to handle this cleanly — the mixin just assumes these attributes exist at runtime.

**Tests:**
1. Run: `python3 -m pytest tests/util/test_indicator_hub.py -x -q`
2. Full regression: `python3 -m pytest tests/ -x -q`

**Commit:** `refactor: extract intraday/session indicators from IndicatorHub to mixin`

### Task 4: Update exports and verify line counts

**Files:**
- Check: `src/stockdownloader/util/pinescript/__init__.py` (may need no changes)
- Check: `src/stockdownloader/util/indicators/__init__.py` (may need no changes)
- Run: full test suite

**Steps:**

1. Check if `pinescript/__init__.py` needs updates — it currently imports from `generator`. If converters (`mode_to_strategy`, `strategy_to_mode`) are still in generator.py (they should be), no changes needed.

2. Check if `indicators/__init__.py` needs updates — it imports `IndicatorHub` from `hub`. Since the mixin is internal, no changes needed.

3. Run: `python3 -m pytest tests/ -x -q`

4. Verify line counts:
```bash
wc -l src/stockdownloader/util/pinescript/generator.py \
      src/stockdownloader/util/pinescript/strategy_renderer.py \
      src/stockdownloader/util/pinescript/composite_renderer.py \
      src/stockdownloader/util/indicators/hub.py \
      src/stockdownloader/util/indicators/hub_intraday.py
```

**Expected results:**
| File | Before | After |
|------|--------|-------|
| `generator.py` | 1,212 | ~530 |
| `strategy_renderer.py` | (new) | ~450 |
| `composite_renderer.py` | (new) | ~230 |
| `hub.py` | 894 | ~560 |
| `hub_intraday.py` | (new) | ~340 |

5. Commit if any export changes needed.

---

## Verification

```bash
python3 -m pytest tests/ -x -q
wc -l src/stockdownloader/util/pinescript/*.py
wc -l src/stockdownloader/util/indicators/hub*.py
```
