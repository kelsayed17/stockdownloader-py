"""Smoke tests for the walk-forward validation CLI."""

from stockdownloader.app.walk_forward_app import main


def test_main_callable():
    """main() is importable and callable."""
    assert callable(main)
