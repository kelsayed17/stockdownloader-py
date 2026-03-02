"""Smoke tests for the walk-forward validation CLI."""

from stockdownloader.app.optimize import main_walk_forward


def test_main_callable():
    """main_walk_forward() is importable and callable."""
    assert callable(main_walk_forward)
