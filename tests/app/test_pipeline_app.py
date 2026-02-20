"""Smoke tests for the unified strategy pipeline CLI."""

from stockdownloader.app.pipeline_app import main, SlotResult, _OPTIMIZE_MODES


def test_main_callable():
    """main() is importable and callable."""
    assert callable(main)


def test_slot_result_defaults():
    """SlotResult has sensible defaults."""
    slot = SlotResult(name="test", display_name="Test", category="intraday")
    assert slot.baseline is None
    assert slot.optimized is None
    assert slot.optimized_kwargs is None
    assert slot.wf_result is None
    assert slot.best_result is None
    assert slot.ranking_score(200) == -999.0


def test_slot_result_best_result_prefers_optimized():
    """best_result returns optimized if available."""
    from unittest.mock import MagicMock
    slot = SlotResult(name="test", display_name="Test", category="intraday")
    baseline = MagicMock()
    optimized = MagicMock()
    slot.baseline = baseline
    slot.optimized = optimized
    assert slot.best_result is optimized


def test_slot_result_best_result_falls_back_to_baseline():
    """best_result falls back to baseline when no optimized result."""
    from unittest.mock import MagicMock
    slot = SlotResult(name="test", display_name="Test", category="intraday")
    baseline = MagicMock()
    slot.baseline = baseline
    assert slot.best_result is baseline


def test_optimize_modes_available():
    """All three optimize modes are registered."""
    assert "wf" in _OPTIMIZE_MODES
    assert "full" in _OPTIMIZE_MODES
    assert "skip" in _OPTIMIZE_MODES
